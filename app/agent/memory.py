"""Recent-conversation context for query understanding.

NOT part of the ported/verified notebook pipeline -- `03_evaluation_groq.ipynb` and its
gold-set evaluation are single-turn only, so nothing here has been run through that eval
harness. Kept intentionally minimal and additive: with no history, understand_query() sees
the bare question exactly as it did before this module existed, so the single-turn behavior
the eval harness actually verified is unchanged. Revisit/evaluate multi-turn follow-up
resolution accuracy specifically (see docs/APP_AND_DEPLOYMENT_PLAN.md) before relying on it.
"""

from typing import NotRequired, TypedDict

MAX_HISTORY_TURNS = 3


class HistoryTurn(TypedDict):
    question: str
    answer: str
    # True for a group or category card click answered from the catalog with no model call
    # (rule R-15): such a turn is free, so it does not count towards the conversation-turn cap
    # (app/main.py's _precheck_reply()). Absent on every other turn.
    card_click: NotRequired[bool]


def build_history_context(history: list[HistoryTurn]) -> str:
    """Recent turns rendered as context text for the query-understanding prompt, or ""
    when there's no history yet (the single-turn case).

    Kept separate from the guest's actual new message (see understand_query()'s `context`
    parameter) so a fallback ever needing "the original question" always gets the bare
    current message, never this history text.
    """
    if not history:
        return ""
    recent = history[-MAX_HISTORY_TURNS:]
    lines = [f"Guest: {turn['question']}\nAssistant: {turn['answer']}" for turn in recent]
    return "Recent conversation so far (most recent last):\n" + "\n".join(lines)
