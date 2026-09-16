from app.agent.memory import MAX_HISTORY_TURNS, HistoryTurn, build_history_context


def test_build_history_context_empty_for_no_history() -> None:
    assert build_history_context([]) == ""


def test_build_history_context_renders_turns() -> None:
    history: list[HistoryTurn] = [
        {"question": "do you have vegan options", "answer": "Yes, several dishes."}
    ]
    context = build_history_context(history)
    assert context.startswith("Recent conversation so far")
    assert "Guest: do you have vegan options" in context
    assert "Assistant: Yes, several dishes." in context


def test_build_history_context_keeps_only_most_recent_turns() -> None:
    history: list[HistoryTurn] = [
        {"question": f"question {i}", "answer": f"answer {i}"} for i in range(MAX_HISTORY_TURNS + 2)
    ]
    context = build_history_context(history)
    assert f"question {MAX_HISTORY_TURNS + 1}" in context  # most recent kept
    assert "question 0" not in context  # oldest dropped
    assert "question 1" not in context
