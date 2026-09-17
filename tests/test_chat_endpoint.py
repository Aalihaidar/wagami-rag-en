import datetime as dt
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
