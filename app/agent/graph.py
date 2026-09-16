"""LangGraph agent: query -> retrieve -> ground -> answer.

Ported from `03_evaluation_groq.ipynb`'s `answer()`/`search()` control flow, restructured
as LangGraph nodes so app/'s FastAPI layer can invoke one compiled graph per guest turn, and
(via `checkpointer`) resume it with prior-turn memory -- see app/agent/checkpointer.py and
app/agent/memory.py. The multi-turn history mechanism is new, additive code, not part of the
verified single-turn notebook pipeline; see app/agent/memory.py's own docstring.

Design invariant (Section 3 of the app/deployment plan): this graph has exactly the four
nodes below and no write-capable tools -- it cannot place orders, modify data, or call
anything beyond read-only retrieval, so there is no "high-risk agent action" surface that
would need human-in-the-loop approval. If a future feature ever adds a write-capable tool
(order placement, reservation booking), re-read that whole section before shipping it.
"""

import operator
from typing import Annotated, Any, Required, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agent.generation import (
    GENERATION_SYSTEM_PROMPT,
    SAFE_FALLBACK_REPLY,
    build_context,
    build_user_prompt,
    contains_system_prompt_leak,
    temperature_for,
    tone_for,
)
from app.agent.llm import GroqClient, zero_usage
from app.agent.memory import HistoryTurn, build_history_context
from app.agent.understanding import CategoryIndex, UnderstandingResult, understand_query
from app.retrieval import RetrievalTool, SearchResult


class AgentState(TypedDict, total=False):
    """total=False since graph.invoke()'s input only ever supplies `question` -- everything
    else is populated progressively by earlier nodes. Fields read via state["key"] (rather
    than state.get(...)) are marked Required: by the time each node runs, the graph's fixed
    linear order (query -> retrieve -> ground -> answer) guarantees the node before it has
    already set that key, even though the schema as a whole can't require it up front.
    """

    question: Required[str]
    # Reduced with operator.add (list concatenation) across checkpointed turns: each
    # answer_node call contributes a one-item list, appended to what's already persisted.
    history: Annotated[list[HistoryTurn], operator.add]
    understanding: Required[UnderstandingResult]
    search_result: Required[SearchResult]
    tone: str
    temperature: Required[float]
    system_prompt: Required[str]
    user_prompt: Required[str]
    answer: str
    usage: dict[str, Any]


def build_graph(
    *,
    retrieval_tool: RetrievalTool,
    category_index: CategoryIndex,
    groq_client: GroqClient | None,
    understand_model: str,
    generation_model: str,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """Compile the query -> retrieve -> ground -> answer graph.

    groq_client=None mirrors the notebooks' own no-key fallback -- the graph still runs end
    to end (deterministic understanding, a stub answer) so retrieval/prompt logic can be
    exercised without a live key.

    checkpointer=None (the default) makes each .invoke() call fully stateless, e.g. for
    testing. Pass a RedisSaver (app/agent/checkpointer.py) and a `thread_id` in .invoke()'s
    config to persist `history` across turns for one guest session.
    """

    def query_node(state: AgentState) -> dict[str, Any]:
        history_context = build_history_context(state.get("history", []))
        understanding = understand_query(
            state["question"],
            category_index=category_index,
            groq_client=groq_client,
            model=understand_model,
            context=history_context,
        )
        return {"understanding": understanding}

    def retrieve_node(state: AgentState) -> dict[str, Any]:
        search_result = retrieval_tool.search(state["understanding"])
        return {"search_result": search_result}

    def ground_node(state: AgentState) -> dict[str, Any]:
        understanding = state["understanding"]
        search_result = state["search_result"]
        intent = understanding["intent"]
        tone = tone_for(intent)
        temperature = temperature_for(intent)
        system_prompt = f"{GENERATION_SYSTEM_PROMPT}\n\n{tone}"
        context = (
            build_context(search_result["ranked"])
            if search_result["answerable"]
            else "(no confident match)"
        )
        user_prompt = build_user_prompt(
            state["question"],
            context,
            search_result["relaxed_fields"],
            search_result["excluded_top_match"],
        )
        return {
            "tone": tone,
            "temperature": temperature,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        }

    def answer_node(state: AgentState) -> dict[str, Any]:
        understand_usage = state["understanding"]["usage"]
        if groq_client is None:
            reply = "[LLM not configured -- skipping live call]"
            gen_usage = zero_usage()
        else:
            gen = groq_client.call(
                state["system_prompt"],
                state["user_prompt"],
                model=generation_model,
                temperature=state["temperature"],
            )
            reply = gen["text"]
            gen_usage = gen["usage"]
            # Output-side check (Section 3): defense-in-depth behind the system prompt's own
            # "never reveal yourself" instruction, not a replacement for it.
            if contains_system_prompt_leak(state["system_prompt"], reply):
                reply = SAFE_FALLBACK_REPLY
        usage = {
            "understand": understand_usage,
            "generate": gen_usage,
            "total_tokens": understand_usage["total_tokens"] + gen_usage["total_tokens"],
        }
        return {
            "answer": reply,
            "usage": usage,
            "history": [{"question": state["question"], "answer": reply}],
        }

    graph = StateGraph(AgentState)
    graph.add_node("query", query_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("ground", ground_node)
    graph.add_node("answer", answer_node)
    graph.set_entry_point("query")
    graph.add_edge("query", "retrieve")
    graph.add_edge("retrieve", "ground")
    graph.add_edge("ground", "answer")
    graph.add_edge("answer", END)
    return graph.compile(checkpointer=checkpointer)
