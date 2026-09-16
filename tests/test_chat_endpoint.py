from typing import Any

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.config import Settings
from app.main import app, get_checkpointer, get_graph


@pytest.fixture(autouse=True)
def _unconfigured_settings(monkeypatch: Any) -> None:
    """Force lifespan's skip-real-connections branch regardless of this dev machine's own
    .env (which has real Weaviate/Cohere/Redis credentials for local manual testing) --
    every test here exercises only the HTTP layer, via dependency_overrides for the graph/
    checkpointer, and must never make a real outbound call."""
    monkeypatch.setattr(
        main_module,
        "settings",
        Settings(weaviate_url="", weaviate_read_api_key="", image_base_url=""),
    )


def test_lifespan_skips_real_connections_when_weaviate_unconfigured() -> None:
    with TestClient(app):
        assert app.state.graph is None
        assert app.state.checkpointer is None


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
    # Override get_graph so a 503 (unconfigured chat) can't mask the 422 this test is
    # actually checking for -- FastAPI's dependency solving can raise before body
    # validation runs, and a real deployment would have get_graph succeed here.
    app.dependency_overrides[get_graph] = lambda: object()
    try:
        with TestClient(app) as client:
            response = client.post(
                "/chat", json={"session_id": "s1", "message": "hi", "model": "gpt-5"}
            )
        assert response.status_code == 422
    finally:
        app.dependency_overrides.pop(get_graph, None)


def test_images_route_404s_for_a_missing_file() -> None:
    with TestClient(app) as client:
        response = client.get("/images/does-not-exist.png")
        assert response.status_code == 404


def test_chat_request_rejects_an_overlong_message() -> None:
    # Same get_graph override reasoning as test_chat_request_rejects_unknown_fields above --
    # a 503 (unconfigured chat) can otherwise preempt the 422 this test checks for.
    app.dependency_overrides[get_graph] = lambda: object()
    try:
        with TestClient(app) as client:
            response = client.post("/chat", json={"session_id": "s1", "message": "x" * 501})
        assert response.status_code == 422
    finally:
        app.dependency_overrides.pop(get_graph, None)


def test_chat_request_rejects_an_empty_message() -> None:
    app.dependency_overrides[get_graph] = lambda: object()
    try:
        with TestClient(app) as client:
            response = client.post("/chat", json={"session_id": "s1", "message": ""})
        assert response.status_code == 422
    finally:
        app.dependency_overrides.pop(get_graph, None)


class FakeGraph:
    def __init__(self, final_state: dict[str, Any]) -> None:
        self._final_state = final_state
        self.invoke_calls: list[dict[str, Any]] = []

    def invoke(self, input: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        self.invoke_calls.append({"input": input, "config": config})
        return self._final_state


def test_chat_returns_answer_and_cited_items_with_overridden_graph() -> None:
    final_state = {
        "answer": "Our vegan ramen is £9.50.",
        "search_result": {
            "ranked": [
                {
                    "row": {
                        "uuid": "abc-123",
                        "score": 0.9,
                        "properties": {
                            "item_type": "menu_item",
                            "slug": "vegan-ramen",
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
    app.dependency_overrides[get_graph] = lambda: fake_graph
    try:
        with TestClient(app) as client:
            response = client.post("/chat", json={"session_id": "s1", "message": "vegan ramen"})
        assert response.status_code == 200
        body = response.json()
        assert body["answer"] == "Our vegan ramen is £9.50."
        assert body["session_id"] == "s1"
        assert body["cited_items"] == [
            {"id": "abc-123", "slug": "vegan-ramen", "image": "vegan-ramen.png"}
        ]
        assert fake_graph.invoke_calls[0]["config"] == {"configurable": {"thread_id": "s1"}}
    finally:
        app.dependency_overrides.pop(get_graph, None)


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
