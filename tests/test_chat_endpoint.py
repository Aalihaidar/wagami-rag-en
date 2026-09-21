import datetime as dt
import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.config import Settings
from app.cost_control import CAPACITY_REPLY, CONVERSATION_LIMIT_REPLY, _daily_key
from app.main import app, get_checkpointer, get_graph, get_redis_client


@pytest.fixture(autouse=True)
def _unconfigured_settings(monkeypatch: Any) -> None:
    """Force lifespan's skip-real-connections branch regardless of this dev machine's own
    .env (which has real Weaviate/Cohere/Redis credentials for local manual testing) --
    every test here exercises only the HTTP layer, via dependency_overrides for the graph/
    checkpointer/redis client, and must never make a real outbound call."""
    monkeypatch.setattr(
        main_module,
        "settings",
        Settings(weaviate_url="", weaviate_read_api_key="", image_base_url=""),
    )
    # The module-level `limiter` was already constructed at import time against whatever
    # REDIS_URL was ambient then -- disabling it (slowapi's own documented mechanism) means
    # its check short-circuits before ever touching storage, so no test needs a real Redis
    # just to reach the rate limiter.
    monkeypatch.setattr(main_module.limiter, "enabled", False)


class FakeRedis:
    """Minimal stand-in for redis.Redis -- just enough for app.cost_control's counter
    reads/writes, with no real Redis needed."""

    def __init__(self) -> None:
        self.store: dict[str, int] = {}

    def get(self, key: str) -> int | None:
        return self.store.get(key)

    def pipeline(self) -> FakeRedis:
        return self

    def incrby(self, key: str, amount: int) -> FakeRedis:
        self.store[key] = self.store.get(key, 0) + amount
        return self

    def expire(self, key: str, ttl: int) -> FakeRedis:
        return self

    def execute(self) -> None:
        pass


def test_lifespan_skips_real_connections_when_weaviate_unconfigured() -> None:
    with TestClient(app):
        assert app.state.graph is None
        assert app.state.checkpointer is None
        assert app.state.redis_client is None


def test_healthz_unaffected_by_lifespan() -> None:
    with TestClient(app) as client:
        response = client.get("/healthz")
        assert response.status_code == 200


def test_chat_returns_503_when_not_configured() -> None:
    with TestClient(app) as client:
        response = client.post("/chat", json={"session_id": "s1", "message": "hi"})
        assert response.status_code == 503


def test_delete_session_returns_503_when_not_configured() -> None:
    with TestClient(app) as client:
        response = client.delete("/session/s1")
        assert response.status_code == 503


def test_create_session_returns_a_session_id() -> None:
    with TestClient(app) as client:
        response = client.post("/session")
        assert response.status_code == 200
        session_id = response.json()["session_id"]
        assert len(session_id) > 0


def test_chat_request_rejects_unknown_fields() -> None:
    # Override get_graph/get_redis_client so a 503 (unconfigured chat) can't mask the 422
    # this test is actually checking for -- FastAPI's dependency solving can raise before
    # body validation runs, and a real deployment would have both dependencies succeed here.
    app.dependency_overrides[get_graph] = lambda: object()
    app.dependency_overrides[get_redis_client] = lambda: FakeRedis()
    try:
        with TestClient(app) as client:
            response = client.post(
                "/chat", json={"session_id": "s1", "message": "hi", "model": "gpt-5"}
            )
        assert response.status_code == 422
        # Section E: one {"error": "..."} JSON shape everywhere, not FastAPI's default
        # {"detail": [...]} -- matching the shape slowapi's own RateLimitExceeded handler uses.
        body = response.json()
        assert body["error"] == "Invalid request."
        assert isinstance(body["details"], list)
    finally:
        app.dependency_overrides.pop(get_graph, None)
        app.dependency_overrides.pop(get_redis_client, None)


def test_chat_returns_503_with_consistent_error_shape_when_kill_switch_is_off() -> None:
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(main_module.settings, "chat_enabled", False)
    app.dependency_overrides[get_graph] = lambda: object()
    app.dependency_overrides[get_redis_client] = lambda: FakeRedis()
    try:
        with TestClient(app) as client:
            response = client.post("/chat", json={"session_id": "s1", "message": "hi"})
        assert response.status_code == 503
        assert response.json() == {"error": "Chat is temporarily disabled."}
    finally:
        monkeypatch.undo()
        app.dependency_overrides.pop(get_graph, None)
        app.dependency_overrides.pop(get_redis_client, None)


def test_images_route_404s_for_a_missing_file() -> None:
    with TestClient(app) as client:
        response = client.get("/images/does-not-exist.png")
        assert response.status_code == 404


def test_chat_request_rejects_an_overlong_message() -> None:
    app.dependency_overrides[get_graph] = lambda: object()
    app.dependency_overrides[get_redis_client] = lambda: FakeRedis()
    try:
        with TestClient(app) as client:
            response = client.post("/chat", json={"session_id": "s1", "message": "x" * 501})
        assert response.status_code == 422
    finally:
        app.dependency_overrides.pop(get_graph, None)
        app.dependency_overrides.pop(get_redis_client, None)


def test_chat_request_rejects_an_empty_message() -> None:
    app.dependency_overrides[get_graph] = lambda: object()
    app.dependency_overrides[get_redis_client] = lambda: FakeRedis()
    try:
        with TestClient(app) as client:
            response = client.post("/chat", json={"session_id": "s1", "message": ""})
        assert response.status_code == 422
    finally:
        app.dependency_overrides.pop(get_graph, None)
        app.dependency_overrides.pop(get_redis_client, None)


class FakeStateSnapshot:
    def __init__(self, values: dict[str, Any]) -> None:
        self.values = values


class FakeGraph:
    def __init__(self, final_state: dict[str, Any], *, history: list[Any] | None = None) -> None:
        self._final_state = final_state
        self._history = history or []
        self.invoke_calls: list[dict[str, Any]] = []

    def get_state(self, config: dict[str, Any]) -> FakeStateSnapshot:
        return FakeStateSnapshot({"history": self._history})

    def invoke(self, input: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        self.invoke_calls.append({"input": input, "config": config})
        return self._final_state


def test_chat_returns_answer_and_cited_items_with_overridden_graph() -> None:
    final_state = {
        "answer": "Our vegan ramen is £9.50.",
        "cited_slugs": ["vegan-ramen"],
        "usage": {"total_tokens": 42},
        "search_result": {
            "ranked": [
                {
                    "row": {
                        "uuid": "abc-123",
                        "score": 0.9,
                        "properties": {
                            "item_type": "menu_item",
                            "slug": "vegan-ramen",
                            "name": "Vegan Ramen",
                            "description": "Rich miso broth with tofu and greens.",
                            "ingredients": ["tofu", "miso", "soya"],
                            "price_gbp": 9.5,
                            "image": "vegan-ramen.png",
                        },
                    },
                    "rerank": 0.9,
                    "hybrid": 0.9,
                }
            ]
        },
    }
    fake_graph = FakeGraph(final_state)
    fake_redis = FakeRedis()
    app.dependency_overrides[get_graph] = lambda: fake_graph
    app.dependency_overrides[get_redis_client] = lambda: fake_redis
    try:
        with TestClient(app) as client:
            response = client.post("/chat", json={"session_id": "s1", "message": "vegan ramen"})
        assert response.status_code == 200
        body = response.json()
        assert body["answer"] == "Our vegan ramen is £9.50."
        assert body["session_id"] == "s1"
        assert body["cited_items"] == [
            {
                "id": "abc-123",
                "slug": "vegan-ramen",
                "name": "Vegan Ramen",
                "description": "Rich miso broth with tofu and greens.",
                "ingredients": ["tofu", "miso", "soya"],
                "price_gbp": 9.5,
                "image": "vegan-ramen.png",
            }
        ]
        assert fake_graph.invoke_calls[0]["config"] == {"configurable": {"thread_id": "s1"}}
        assert fake_redis.store  # token usage was recorded
    finally:
        app.dependency_overrides.pop(get_graph, None)
        app.dependency_overrides.pop(get_redis_client, None)


def test_chat_logs_one_per_stage_timing_line_per_turn(caplog: pytest.LogCaptureFixture) -> None:
    fake_graph = FakeGraph(
        {
            "answer": "hi",
            "cited_slugs": [],
            "usage": {"total_tokens": 1},
            "search_result": {"ranked": []},
        }
    )
    app.dependency_overrides[get_graph] = lambda: fake_graph
    app.dependency_overrides[get_redis_client] = lambda: FakeRedis()
    try:
        with caplog.at_level(logging.INFO, logger="app.main"), TestClient(app) as client:
            response = client.post("/chat", json={"session_id": "s1", "message": "hello"})
        assert response.status_code == 200
        records = [r for r in caplog.records if r.getMessage() == "chat turn timings"]
        assert len(records) == 1
        timings = records[0].timings_ms  # type: ignore[attr-defined]
        assert {"turn", "state_read", "cost_check", "graph", "cost_record"} <= set(timings)
        assert all(ms >= 0 for ms in timings.values())
    finally:
        app.dependency_overrides.pop(get_graph, None)
        app.dependency_overrides.pop(get_redis_client, None)


def test_chat_returns_a_fixed_reply_once_conversation_turn_cap_is_hit() -> None:
    long_history = [{"question": f"q{i}", "answer": f"a{i}"} for i in range(50)]
    fake_graph = FakeGraph({"answer": "unused", "usage": {"total_tokens": 0}}, history=long_history)
    fake_redis = FakeRedis()
    app.dependency_overrides[get_graph] = lambda: fake_graph
    app.dependency_overrides[get_redis_client] = lambda: fake_redis
    try:
        with TestClient(app) as client:
            response = client.post("/chat", json={"session_id": "s1", "message": "hi again"})
        assert response.status_code == 200
        assert response.json()["answer"] == CONVERSATION_LIMIT_REPLY
        assert fake_graph.invoke_calls == []  # never reached the LLM
    finally:
        app.dependency_overrides.pop(get_graph, None)
        app.dependency_overrides.pop(get_redis_client, None)


def test_chat_returns_capacity_reply_once_daily_spend_limit_is_hit() -> None:
    fake_graph = FakeGraph({"answer": "unused", "usage": {"total_tokens": 0}})
    fake_redis = FakeRedis()
    fake_redis.store[_daily_key(dt.datetime.now(dt.UTC))] = 10_000_000  # far over any default
    app.dependency_overrides[get_graph] = lambda: fake_graph
    app.dependency_overrides[get_redis_client] = lambda: fake_redis
    try:
        with TestClient(app) as client:
            response = client.post("/chat", json={"session_id": "s1", "message": "hi"})
        assert response.status_code == 200
        assert response.json()["answer"] == CAPACITY_REPLY
        assert fake_graph.invoke_calls == []
    finally:
        app.dependency_overrides.pop(get_graph, None)
        app.dependency_overrides.pop(get_redis_client, None)


class FakeOutageGraph:
    """A graph whose invoke() raises a transient outbound error -- e.g. Weaviate/Groq timing
    out, or the Redis-backed checkpointer being unreachable."""

    def get_state(self, config: dict[str, Any]) -> FakeStateSnapshot:
        return FakeStateSnapshot({"history": []})

    def invoke(self, input: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        raise TimeoutError("simulated Weaviate/Groq timeout")


def test_chat_degrades_to_a_fallback_reply_on_transient_outbound_failure() -> None:
    fake_redis = FakeRedis()
    app.dependency_overrides[get_graph] = lambda: FakeOutageGraph()
    app.dependency_overrides[get_redis_client] = lambda: fake_redis
    try:
        with TestClient(app) as client:
            response = client.post("/chat", json={"session_id": "s1", "message": "hi"})
        assert response.status_code == 200
        assert response.json()["answer"] == main_module.OUTBOUND_ERROR_REPLY
    finally:
        app.dependency_overrides.pop(get_graph, None)
        app.dependency_overrides.pop(get_redis_client, None)


class FakeCheckpointer:
    def __init__(self) -> None:
        self.deleted_threads: list[str] = []

    def delete_thread(self, thread_id: str) -> None:
        self.deleted_threads.append(thread_id)


def test_delete_session_discards_history_with_overridden_checkpointer() -> None:
    fake_checkpointer = FakeCheckpointer()
    app.dependency_overrides[get_checkpointer] = lambda: fake_checkpointer
    try:
        with TestClient(app) as client:
            response = client.delete("/session/s1")
        assert response.status_code == 204
        assert fake_checkpointer.deleted_threads == ["s1"]
    finally:
        app.dependency_overrides.pop(get_checkpointer, None)


# --- /chat/stream (Server-Sent Events) -------------------------------------------------------

RANKED_ESPRESSO = {
    "row": {
        "uuid": "abc-123",
        "score": 0.9,
        "properties": {
            "item_type": "menu_item",
            "slug": "double-espresso",
            "name": "Double Espresso",
            "description": "Two shots.",
            "ingredients": ["coffee"],
            "price_gbp": 2.5,
            "image": "espresso.png",
        },
    },
    "rerank": 0.9,
    "hybrid": 0.9,
}


class FakeStreamingGraph(FakeGraph):
    """A graph whose stream() yields what LangGraph's stream_mode=["custom", "values"] does:
    ("custom", {"delta": ...}) events, then the final state as a ("values", ...) pair."""

    def __init__(
        self,
        deltas: list[str],
        *,
        cited_slugs: list[str] | None = None,
        fail_after_deltas: Exception | None = None,
        history: list[Any] | None = None,
    ) -> None:
        final_state = {
            "answer": "".join(deltas),
            "cited_slugs": cited_slugs or [],
            "usage": {"total_tokens": 42},
            "search_result": {"ranked": [RANKED_ESPRESSO]},
        }
        super().__init__(final_state, history=history)
        self._deltas = deltas
        self._fail_after_deltas = fail_after_deltas
        self.stream_calls: list[dict[str, Any]] = []

    def stream(
        self, input: dict[str, Any], config: dict[str, Any], stream_mode: list[str]
    ) -> Iterator[tuple[str, Any]]:
        self.stream_calls.append({"input": input, "config": config, "stream_mode": stream_mode})
        for delta in self._deltas:
            yield "custom", {"delta": delta}
        if self._fail_after_deltas is not None:
            raise self._fail_after_deltas
        yield "values", {"answer": "intermediate"}
        yield "values", self._final_state


@contextmanager
def _client_with(graph: Any, redis: Any = None) -> Iterator[TestClient]:
    app.dependency_overrides[get_graph] = lambda: graph
    app.dependency_overrides[get_redis_client] = lambda: redis if redis is not None else FakeRedis()
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.pop(get_graph, None)
        app.dependency_overrides.pop(get_redis_client, None)


def _events(body: str) -> list[tuple[str, Any]]:
    """Parse an SSE body into (event, decoded JSON data) pairs."""
    events = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def _post_stream(client: TestClient, message: str = "espresso price?") -> Any:
    return client.post("/chat/stream", json={"session_id": "s1", "message": message})


def test_chat_stream_sends_deltas_then_a_done_event_with_the_full_response() -> None:
    graph = FakeStreamingGraph(
        ["Our double ", "espresso is ", "£2.50."], cited_slugs=["double-espresso"]
    )
    redis = FakeRedis()
    with _client_with(graph, redis) as client:
        response = _post_stream(client)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    events = _events(response.text)
    assert [name for name, _ in events] == ["delta", "delta", "delta", "done"]
    assert [data["text"] for name, data in events if name == "delta"] == [
        "Our double ",
        "espresso is ",
        "£2.50.",
    ]
    done = events[-1][1]
    assert done["session_id"] == "s1"
    assert done["answer"] == "Our double espresso is £2.50."
    assert done["cited_items"][0]["slug"] == "double-espresso"
    assert graph.stream_calls[0]["config"] == {"configurable": {"thread_id": "s1"}}
    assert graph.stream_calls[0]["stream_mode"] == ["custom", "values"]
    assert redis.store  # token usage was recorded, once the stream completed


def test_chat_stream_is_blocked_by_the_conversation_cap_before_reaching_the_llm() -> None:
    history = [{"question": f"q{i}", "answer": f"a{i}"} for i in range(50)]
    graph = FakeStreamingGraph(["never sent"], history=history)
    with _client_with(graph) as client:
        response = _post_stream(client)

    assert _events(response.text) == [
        ("done", {"session_id": "s1", "answer": CONVERSATION_LIMIT_REPLY, "cited_items": []})
    ]
    assert graph.stream_calls == []


def test_chat_stream_is_blocked_by_the_spend_cap_before_reaching_the_llm() -> None:
    graph = FakeStreamingGraph(["never sent"])
    redis = FakeRedis()
    redis.store[_daily_key(dt.datetime.now(dt.UTC))] = 10_000_000
    with _client_with(graph, redis) as client:
        response = _post_stream(client)

    assert _events(response.text)[0][1]["answer"] == CAPACITY_REPLY
    assert graph.stream_calls == []


def test_chat_stream_degrades_to_the_fallback_reply_on_an_outbound_failure_mid_stream() -> None:
    graph = FakeStreamingGraph(["Our double "], fail_after_deltas=TimeoutError("groq stalled"))
    with _client_with(graph) as client:
        response = _post_stream(client)

    events = _events(response.text)
    assert [name for name, _ in events] == ["delta", "done"]
    # The done event's answer replaces the partial text the guest already saw.
    assert events[-1][1]["answer"] == main_module.OUTBOUND_ERROR_REPLY
    assert events[-1][1]["cited_items"] == []


def test_chat_stream_reports_a_genuine_bug_as_an_error_event() -> None:
    graph = FakeStreamingGraph([], fail_after_deltas=RuntimeError("real bug"))
    with _client_with(graph) as client:
        response = _post_stream(client)

    assert response.status_code == 200  # the status line was already sent when it happened
    assert _events(response.text) == [("error", {"error": main_module.INTERNAL_ERROR_MESSAGE})]


def test_chat_stream_gives_its_concurrency_slot_back_after_every_outcome() -> None:
    assert main_module._chat_slots_in_use == 0
    outcomes = [
        FakeStreamingGraph(["ok"]),
        FakeStreamingGraph(["x"], fail_after_deltas=TimeoutError()),
        FakeStreamingGraph([], fail_after_deltas=RuntimeError("bug")),
    ]
    for graph in outcomes:
        with _client_with(graph) as client:
            _post_stream(client)
        assert main_module._chat_slots_in_use == 0


def test_chat_stream_returns_503_when_the_kill_switch_is_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main_module.settings, "chat_enabled", False)
    with _client_with(FakeStreamingGraph(["x"])) as client:
        response = _post_stream(client)

    assert response.status_code == 503
    assert response.json() == {"error": "Chat is temporarily disabled."}
    assert main_module._chat_slots_in_use == 0


def test_chat_stream_rejects_an_invalid_request_like_chat_does() -> None:
    with _client_with(FakeStreamingGraph(["x"])) as client:
        response = client.post("/chat/stream", json={"session_id": "s1", "message": ""})

    assert response.status_code == 422


def test_chat_stream_logs_time_to_first_delta_alongside_the_stage_timings(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with (
        caplog.at_level(logging.INFO, logger="app.main"),
        _client_with(FakeStreamingGraph(["a", "b"])) as client,
    ):
        _post_stream(client)

    records = [r for r in caplog.records if r.getMessage() == "chat turn timings"]
    assert len(records) == 1
    timings = records[0].timings_ms  # type: ignore[attr-defined]
    assert {"turn", "state_read", "cost_check", "graph", "cost_record", "first_delta"} <= set(
        timings
    )
    assert timings["first_delta"] <= timings["turn"]


# --- a turn answered without a search (greeting, off-topic, menu browsing) -----------------------


def test_chat_handles_a_turn_with_no_search_result() -> None:
    """A direct-reply turn never runs retrieval, so its final state has no `search_result`."""
    final_state = {"answer": "Hello!", "cited_slugs": [], "usage": {"total_tokens": 15}}
    with _client_with(FakeGraph(final_state)) as client:
        response = client.post("/chat", json={"session_id": "s1", "message": "hi"})

    assert response.status_code == 200
    assert response.json()["answer"] == "Hello!"
    assert response.json()["cited_items"] == []


def test_chat_stream_handles_a_turn_with_no_search_result() -> None:
    class DirectReplyGraph(FakeStreamingGraph):
        def __init__(self) -> None:
            super().__init__(["Hello, welcome!"])
            self._final_state = {"answer": "Hello, welcome!", "usage": {"total_tokens": 15}}

    with _client_with(DirectReplyGraph()) as client:
        response = _post_stream(client, "hi")

    events = _events(response.text)
    assert [name for name, _ in events] == ["delta", "done"]
    assert events[-1][1]["answer"] == "Hello, welcome!"
    assert events[-1][1]["cited_items"] == []


LISTED_CARD = {
    "id": "id-1",
    "slug": "lychee-sangria",
    "name": "Lychee Sangria",
    "description": "fruity",
    "ingredients": ["lychee"],
    "price_gbp": 8.0,
    "image": "lychee.png",
}


def test_chat_returns_the_cards_a_direct_reply_supplies_itself() -> None:
    """A listing of a category's items brings its own cards, with no search_result behind them."""
    final_state = {
        "answer": "Here is everything in cocktails (drinks):\n- Lychee Sangria: fruity.",
        "cited_slugs": [],
        "cited_items": [LISTED_CARD],
        "usage": {"total_tokens": 15},
    }
    with _client_with(FakeGraph(final_state)) as client:
        response = client.post("/chat", json={"session_id": "s1", "message": "cocktails"})

    assert response.status_code == 200
    assert response.json()["cited_items"] == [LISTED_CARD]


def test_chat_stream_done_event_carries_the_cards_a_direct_reply_supplies() -> None:
    class ListingGraph(FakeStreamingGraph):
        def __init__(self) -> None:
            super().__init__(["Here is everything in cocktails."])
            self._final_state = {
                "answer": "Here is everything in cocktails.",
                "cited_items": [LISTED_CARD],
                "usage": {"total_tokens": 15},
            }

    with _client_with(ListingGraph()) as client:
        response = _post_stream(client, "cocktails")

    done = _events(response.text)[-1]
    assert done[0] == "done"
    assert done[1]["cited_items"] == [LISTED_CARD]
