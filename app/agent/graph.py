"""LangGraph agent: query, then respond (no search) or retrieve -> ground -> answer.

Ported from `03_evaluation_groq.ipynb`'s `answer()`/`search()` control flow, restructured
as LangGraph nodes so app/'s FastAPI layer can invoke one compiled graph per guest turn, and
(via `checkpointer`) resume it with prior-turn memory -- see app/agent/checkpointer.py and
app/agent/memory.py. The multi-turn history mechanism is new, additive code, not part of the
verified single-turn notebook pipeline; see app/agent/memory.py's own docstring.

Design invariant (Section 3 of the app/deployment plan): this graph has exactly the five
nodes below and no write-capable tools -- it cannot place orders, modify data, or call
anything beyond read-only retrieval, so there is no "high-risk agent action" surface that
would need human-in-the-loop approval. If a future feature ever adds a write-capable tool
(order placement, reservation booking), re-read that whole section before shipping it.
"""

import logging
import operator
from typing import Annotated, Any, Literal, Required, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agent.browse import DIRECT_INTENTS, direct_answer, is_known_pick
from app.agent.cards import Choices, CitedItem, cards_for_answer
from app.agent.generation import (
    GENERATION_REASONING_EFFORT,
    MALFORMED_REPLY,
    SAFE_FALLBACK_REPLY,
    AnswerStreamDecoder,
    LeakHoldback,
    build_context,
    build_user_prompt,
    citable_slugs,
    contains_system_prompt_leak,
    parse_generation_reply,
    salvage_cited_slugs,
    temperature_for,
    tone_for,
)
from app.agent.llm import GroqClient, Usage, zero_usage
from app.agent.memory import HistoryTurn, build_history_context
from app.agent.prompts import (
    CITATION_OUTPUT_INSTRUCTIONS,
    GENERATION_SYSTEM_PROMPT,
    SCOPE_AND_SAFETY,
)
from app.agent.understanding import (
    CategoryIndex,
    UnderstandingResult,
    picked_browse,
    understand_query,
)
from app.retrieval import RetrievalTool, SearchResult
from app.timing import timed

logger = logging.getLogger("app.agent.graph")


class CardPick(TypedDict):
    """A clicked group or category card (rule R-15); `category` is None for a group's card."""

    group: str
    category: str | None


class AgentState(TypedDict, total=False):
    """total=False since graph.invoke()'s input only ever supplies `question` -- everything
    else is populated progressively by earlier nodes. Fields read via state["key"] (rather
    than state.get(...)) are marked Required: by the time each node runs, the graph's order
    (query -> retrieve -> ground -> answer, or query -> respond) guarantees the node before it
    has already set that key, even though the schema as a whole can't require it up front.
    A turn that took the `respond` route never sets search_result, tone, prompts or slugs.
    """

    question: Required[str]
    # The group/category card the guest clicked this turn ({"group", "category"}), or None for
    # typed text (rule R-15). Written with every turn's input, so a click never carries over.
    browse: CardPick | None
    # Reduced with operator.add (list concatenation) across checkpointed turns: each
    # answer_node call contributes a one-item list, appended to what's already persisted.
    history: Annotated[list[HistoryTurn], operator.add]
    understanding: Required[UnderstandingResult]
    search_result: Required[SearchResult]
    tone: str
    temperature: Required[float]
    system_prompt: Required[str]
    user_prompt: Required[str]
    citable_slugs: Required[list[str]]
    answer: str
    cited_slugs: list[str]
    # The item cards for this turn's answer. Written by whichever node ends the turn (`respond` or
    # `answer`), every turn, an empty list when there are none -- the state is saved per session, so
    # a node that skipped it would leave the previous turn's cards behind for the next answer.
    cited_items: list[CitedItem]
    # The picture cards of a reply that lists groups or categories (rule C-23), or None. Written by
    # both turn-ending nodes every turn, for the same reason as `cited_items`.
    choices: Choices | None
    usage: dict[str, Any]


def _generate_answer(
    groq_client: GroqClient,
    system_prompt: str,
    user_prompt: str,
    *,
    model: str,
    temperature: float,
    candidate_slugs: list[str],
) -> tuple[str, list[str], Usage]:
    """Run the generation call as a stream; returns (reply, cited_slugs, usage).

    Every call streams, whether or not anyone is watching: each chunk of guest-visible text is
    handed to LangGraph's custom stream writer (a no-op unless the caller asked for
    stream_mode="custom", as /chat/stream does), so /chat and /chat/stream run one and the same
    generation path. The returned reply is always the authoritative one, parsed from the
    complete stream -- what was streamed is only a live preview of it.

    Output-side check (Section 3): defense-in-depth behind the system prompt's own "never reveal
    yourself" instruction, not a replacement for it. Checked against SCOPE_AND_SAFETY
    specifically, not the full system_prompt -- GENERATION_RULES (the other half of
    GENERATION_SYSTEM_PROMPT) deliberately instructs content that's supposed to reach the guest
    almost verbatim (e.g. the demo/limited-data decline wording), so scanning the reply against
    it produces false positives. Streaming would otherwise show a leak before this check could
    run, so LeakHoldback releases text a few words behind the model and stops the stream the
    moment a leak is flagged.
    """
    write = get_stream_writer()
    json_mode = bool(candidate_slugs)
    decoder = AnswerStreamDecoder(json_mode=json_mode)
    guard = LeakHoldback(SCOPE_AND_SAFETY)
    raw: list[str] = []

    with timed("generate"):
        # No response_format on purpose: Groq only streams tokens when none is set (see
        # parse_generation_reply()); the reply shape comes from the system prompt instead.
        stream = groq_client.stream(
            system_prompt,
            user_prompt,
            model=model,
            temperature=temperature,
            reasoning_effort=GENERATION_REASONING_EFFORT,
        )
        try:
            for piece in stream:
                raw.append(piece)
                visible = guard.push(decoder.feed(piece))
                if visible:
                    write({"delta": visible})
                if guard.leaked:
                    break
        finally:
            stream.close()

    if guard.leaked:
        return SAFE_FALLBACK_REPLY, [], stream.usage

    text = "".join(raw).strip()
    cited_slugs: list[str] = []
    if json_mode:
        parsed = parse_generation_reply(text, candidate_slugs)
        if parsed is not None:
            reply, cited_slugs = parsed
        else:
            # The model didn't produce the requested JSON. Keep whatever is still usable, and
            # log it -- a rising rate of these means the prompt-only format has stopped holding.
            logger.warning("Generation reply was not the requested JSON object")
            cited_slugs = salvage_cited_slugs(text, candidate_slugs)
            if decoder.text:
                reply = decoder.text
            elif text.startswith(("{", "`")):
                reply = MALFORMED_REPLY
            else:
                reply = text
    else:
        reply = text
    if contains_system_prompt_leak(SCOPE_AND_SAFETY, reply):
        return SAFE_FALLBACK_REPLY, [], stream.usage

    tail = guard.flush()
    if tail:
        write({"delta": tail})
    return reply, cited_slugs, stream.usage


def build_graph(
    *,
    retrieval_tool: RetrievalTool,
    category_index: CategoryIndex,
    groq_client: GroqClient | None,
    understand_model: str,
    generation_model: str,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """Compile the agent graph: query, then either respond (a greeting, an off-topic message,
    or menu browsing -- answered from a fixed reply or the catalog with no search and no second
    model call) or retrieve -> ground -> answer.

    groq_client=None mirrors the notebooks' own no-key fallback -- the graph still runs end
    to end (deterministic understanding, a stub answer) so retrieval/prompt logic can be
    exercised without a live key.

    checkpointer=None (the default) makes each .invoke() call fully stateless, e.g. for
    testing. Pass a RedisSaver (app/agent/checkpointer.py) and a `thread_id` in .invoke()'s
    config to persist `history` across turns for one guest session.
    """

    def query_node(state: AgentState) -> dict[str, Any]:
        pick = state.get("browse")
        if pick and is_known_pick(category_index.catalog, pick["group"], pick["category"]):
            # A card click names its group/category exactly: nothing to understand (R-15).
            understanding = picked_browse(state["question"], pick["group"], pick["category"])
            return {"understanding": understanding}
        history_context = build_history_context(state.get("history", []))
        with timed("understand"):
            understanding = understand_query(
                state["question"],
                category_index=category_index,
                groq_client=groq_client,
                model=understand_model,
                context=history_context,
            )
        return {"understanding": understanding}

    def route_after_query(state: AgentState) -> Literal["respond", "retrieve"]:
        return "respond" if state["understanding"]["intent"] in DIRECT_INTENTS else "retrieve"

    def respond_node(state: AgentState) -> dict[str, Any]:
        understanding = state["understanding"]
        direct = direct_answer(understanding, category_index.catalog)
        reply = direct.text
        # The whole reply at once: there is no generation stream to forward. The caller still
        # gets it as a delta, so /chat/stream and /chat behave the same on every route. A list of
        # groups or categories streams only the sentence before its cards, which the final
        # message keeps as its start, so the bullet list is never shown and then replaced.
        get_stream_writer()({"delta": direct.choices["intro"] if direct.choices else reply})
        understand_usage = understanding["usage"]
        return {
            "answer": reply,
            "cited_slugs": [],
            "cited_items": direct.cards,
            "choices": direct.choices,
            "usage": {
                "understand": understand_usage,
                "generate": zero_usage(),
                "total_tokens": understand_usage["total_tokens"],
            },
            "history": [{"question": state["question"], "answer": reply}],
        }

    def retrieve_node(state: AgentState) -> dict[str, Any]:
        search_result = retrieval_tool.search(state["understanding"])
        return {"search_result": search_result}

    def ground_node(state: AgentState) -> dict[str, Any]:
        understanding = state["understanding"]
        search_result = state["search_result"]
        intent = understanding["intent"]
        tone = tone_for(intent)
        temperature = temperature_for(intent)
        ranked = search_result["ranked"] if search_result["answerable"] else []
        candidate_slugs = citable_slugs(ranked)
        system_prompt = f"{GENERATION_SYSTEM_PROMPT}\n\n{tone}"
        if candidate_slugs:
            system_prompt += f"\n\n{CITATION_OUTPUT_INSTRUCTIONS}"
        context = build_context(ranked) if search_result["answerable"] else "(no confident match)"
        user_prompt = build_user_prompt(
            state["question"],
            context,
            search_result["relaxed_fields"],
            search_result["excluded_top_match"],
            resolved_question=understanding["resolved_question"],
        )
        return {
            "tone": tone,
            "temperature": temperature,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "citable_slugs": candidate_slugs,
        }

    def answer_node(state: AgentState) -> dict[str, Any]:
        understand_usage = state["understanding"]["usage"]
        candidate_slugs = state["citable_slugs"]
        if groq_client is None:
            reply = "[LLM not configured -- skipping live call]"
            cited_slugs: list[str] = []
            gen_usage = zero_usage()
        else:
            reply, cited_slugs, gen_usage = _generate_answer(
                groq_client,
                state["system_prompt"],
                state["user_prompt"],
                model=generation_model,
                temperature=state["temperature"],
                candidate_slugs=candidate_slugs,
            )
        usage = {
            "understand": understand_usage,
            "generate": gen_usage,
            "total_tokens": understand_usage["total_tokens"] + gen_usage["total_tokens"],
        }
        search_result = state["search_result"]
        ranked = search_result["ranked"] if search_result["answerable"] else []
        screened = search_result["excluded_top_match"]
        return {
            "answer": reply,
            "cited_slugs": cited_slugs,
            "choices": None,
            "cited_items": cards_for_answer(
                reply,
                ranked=ranked,
                cited_slugs=cited_slugs,
                screened_out=screened["name"] if screened else None,
            ),
            "usage": usage,
            "history": [{"question": state["question"], "answer": reply}],
        }

    graph = StateGraph(AgentState)
    graph.add_node("query", query_node)
    graph.add_node("respond", respond_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("ground", ground_node)
    graph.add_node("answer", answer_node)
    graph.set_entry_point("query")
    graph.add_conditional_edges("query", route_after_query, ["respond", "retrieve"])
    graph.add_edge("respond", END)
    graph.add_edge("retrieve", "ground")
    graph.add_edge("ground", "answer")
    graph.add_edge("answer", END)
    return graph.compile(checkpointer=checkpointer)
