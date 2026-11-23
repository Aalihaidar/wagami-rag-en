"""Startup with a backend that is configured but unreachable: the app still starts, chat answers
503, and /readyz names what is down -- instead of the whole service crashing."""

from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.config import Settings
from app.main import app

HOST_IN_ERROR = "db.private-host.example:12345"


class FakeWeaviate:
    def __init__(self) -> None:
        self.closed = False
        self.collections = SimpleNamespace(get=lambda name: object())

    def close(self) -> None:
        self.closed = True


class FakeTool:
    def __init__(self, kb: Any, cohere_api_key: str) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakeRedisClient:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class Backends:
    """What the fakes were asked to do, so a test can check everything was closed."""

    def __init__(self) -> None:
        self.weaviate = FakeWeaviate()
        self.redis = FakeRedisClient()
        self.tools: list[FakeTool] = []
        self.checkpointer_exited = False


@pytest.fixture
def backends(monkeypatch: pytest.MonkeyPatch) -> Backends:
    """Configured settings and healthy fakes; a test breaks one backend by overriding it."""
    fakes = Backends()
    monkeypatch.setattr(
        main_module,
        "settings",
        Settings(
            weaviate_url="https://cluster.example",
            weaviate_read_api_key="key",
            embedding_api_key="key",
            image_base_url="",
            telegram_bot_token="",
        ),
    )
    monkeypatch.setattr(main_module.limiter, "enabled", False)
    monkeypatch.setattr(main_module, "connect", lambda settings: fakes.weaviate)

    def make_tool(kb: Any, cohere_api_key: str) -> FakeTool:
        tool = FakeTool(kb, cohere_api_key)
        fakes.tools.append(tool)
        return tool

    monkeypatch.setattr(main_module, "RetrievalTool", make_tool)
    monkeypatch.setattr(main_module, "load_category_index", lambda kb: {"main": ["a"]})
    monkeypatch.setattr(
        main_module, "Redis", SimpleNamespace(from_url=lambda url, **kwargs: fakes.redis)
    )

    @contextmanager
    def healthy_checkpointer(url: str) -> Iterator[object]:
        try:
            yield object()
        finally:
            fakes.checkpointer_exited = True

    monkeypatch.setattr(main_module, "build_checkpointer", healthy_checkpointer)
    monkeypatch.setattr(main_module, "load_groq_key_pool", lambda key: [])
    monkeypatch.setattr(main_module, "build_graph", lambda **kwargs: object())
    return fakes


def test_everything_reachable_is_ready(backends: Backends) -> None:
    with TestClient(app) as client:
        response = client.get("/readyz")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "weaviate": "ok",
        "redis": "ok",
        "agent": "ok",
    }


def test_an_unreachable_weaviate_does_not_stop_the_app_and_is_named(
    backends: Backends, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    class UnexpectedStatusCodeError(Exception):
        pass

    def broken_connect(settings: Settings) -> FakeWeaviate:
        raise UnexpectedStatusCodeError(
            f"Meta endpoint! Unexpected status code: 301 {HOST_IN_ERROR}"
        )

    monkeypatch.setattr(main_module, "connect", broken_connect)

    with TestClient(app) as client:  # starting is the point: it used to raise here
        assert client.get("/healthz").status_code == 200
        readyz = client.get("/readyz")
        chat = client.post("/chat", json={"session_id": "s1", "message": "hi"})

    assert readyz.status_code == 503
    assert readyz.json() == {
        "status": "degraded",
        "weaviate": "error: UnexpectedStatusCodeError",
        "redis": "ok",
        "agent": "not started",
    }
    assert chat.status_code == 503
    # the host name is in the service log only, never on the public page
    assert HOST_IN_ERROR not in readyz.text
    assert HOST_IN_ERROR in caplog.text
    assert backends.checkpointer_exited and backends.redis.closed


def test_an_unreachable_redis_does_not_stop_the_app_and_is_named(
    backends: Backends, monkeypatch: pytest.MonkeyPatch
) -> None:
    @contextmanager
    def broken_checkpointer(url: str) -> Iterator[object]:
        raise ConnectionError(f"Error -2 connecting to {HOST_IN_ERROR}. Name or service not known.")
        yield  # pragma: no cover

    monkeypatch.setattr(main_module, "build_checkpointer", broken_checkpointer)

    with TestClient(app) as client:
        assert client.get("/healthz").status_code == 200
        readyz = client.get("/readyz")
        chat = client.post("/chat", json={"session_id": "s1", "message": "hi"})

    assert readyz.status_code == 503
    assert readyz.json() == {
        "status": "degraded",
        "weaviate": "ok",
        "redis": "error: ConnectionError",
        "agent": "not started",
    }
    assert HOST_IN_ERROR not in readyz.text
    assert chat.status_code == 503
    # what did open is still closed on shutdown
    assert backends.weaviate.closed and backends.redis.closed
    assert all(tool.closed for tool in backends.tools)


def test_both_down_are_both_reported(backends: Backends, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken_connect(settings: Settings) -> FakeWeaviate:
        raise TimeoutError("timed out")

    @contextmanager
    def broken_checkpointer(url: str) -> Iterator[object]:
        raise ConnectionError("refused")
        yield  # pragma: no cover

    monkeypatch.setattr(main_module, "connect", broken_connect)
    monkeypatch.setattr(main_module, "build_checkpointer", broken_checkpointer)

    with TestClient(app) as client:
        readyz = client.get("/readyz")

    assert readyz.json()["weaviate"] == "error: TimeoutError"
    assert readyz.json()["redis"] == "error: ConnectionError"
    assert readyz.status_code == 503


def test_a_graph_that_cannot_be_built_is_reported(
    backends: Backends, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken_graph(**kwargs: Any) -> object:
        raise ValueError("bad graph")

    monkeypatch.setattr(main_module, "build_graph", broken_graph)

    with TestClient(app) as client:
        readyz = client.get("/readyz")

    assert readyz.status_code == 503
    assert readyz.json()["agent"] == "error: ValueError"


def test_an_unconfigured_app_reports_not_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        main_module,
        "settings",
        Settings(weaviate_url="", weaviate_read_api_key="", image_base_url=""),
    )

    with TestClient(app) as client:
        readyz = client.get("/readyz")

    assert readyz.status_code == 503
    assert readyz.json() == {
        "status": "degraded",
        "weaviate": "not configured",
        "redis": "not configured",
        "agent": "not started",
    }
