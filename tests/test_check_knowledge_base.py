import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import redis


def load_script() -> ModuleType:
    """scripts/ is not a package, so load the script the way python would run it."""
    path = Path(__file__).resolve().parent.parent / "scripts" / "check_knowledge_base.py"
    spec = importlib.util.spec_from_file_location("check_knowledge_base", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = load_script()


class FakeRedis:
    """Just the calls check_redis() makes; no network."""

    def __init__(
        self,
        *,
        has_search: bool = True,
        set_error: Exception | None = None,
        returns_wrong_value: bool = False,
    ) -> None:
        self.store: dict[str, str] = {}
        self.ttls: dict[str, int] = {}
        self.has_search = has_search
        self.set_error = set_error
        self.returns_wrong_value = returns_wrong_value
        self.closed = False

    def set(self, key: str, value: str, ex: int | None = None) -> None:
        if self.set_error:
            raise self.set_error
        self.store[key] = value
        if ex is not None:
            self.ttls[key] = ex

    def get(self, key: str) -> str | None:
        return "something else" if self.returns_wrong_value else self.store.get(key)

    def execute_command(self, *args: Any) -> list[str]:
        if args == ("FT._LIST",) and self.has_search:
            return []
        raise redis.ResponseError("unknown command 'FT._LIST'")

    def close(self) -> None:
        self.closed = True


def use_fake(monkeypatch: pytest.MonkeyPatch, fake: FakeRedis) -> None:
    monkeypatch.setattr(redis.Redis, "from_url", lambda url, **kwargs: fake)


def test_an_unset_redis_url_is_reported_not_guessed() -> None:
    assert script.check_redis("") == ["missing settings: REDIS_URL"]


def test_a_healthy_redis_writes_a_short_lived_key_and_reports_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeRedis()
    use_fake(monkeypatch, fake)

    assert script.check_redis("redis://default:secret@example.test:6379") == []
    assert script.KEEPALIVE_KEY in fake.store
    # the key expires on its own, so it never lingers in a 30 MB database
    assert fake.ttls[script.KEEPALIVE_KEY] == script.KEEPALIVE_TTL_SECONDS
    assert fake.closed


def test_a_redis_without_the_search_module_fails_with_a_clear_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_fake(monkeypatch, FakeRedis(has_search=False))

    failures = script.check_redis("redis://example.test:6379")

    assert len(failures) == 1
    assert "Search" in failures[0]


def test_a_write_that_does_not_read_back_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    use_fake(monkeypatch, FakeRedis(returns_wrong_value=True))

    failures = script.check_redis("redis://example.test:6379")

    assert any("did not return it" in reason for reason in failures)


def test_a_connection_error_fails_and_closes_the_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    fake = FakeRedis(set_error=redis.ConnectionError("Error -2 connecting to db.example.test:1"))
    use_fake(monkeypatch, fake)

    failures = script.check_redis("redis://example.test:6379")

    assert len(failures) == 1
    assert "ConnectionError" in failures[0]
    assert "db.example.test" in failures[0]  # run locally, the full message helps
    assert fake.closed


def test_on_github_actions_the_log_never_names_the_redis_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # the repository is public, and so are its Actions logs
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    fake = FakeRedis(set_error=redis.ConnectionError("Error -2 connecting to db.example.test:1"))
    use_fake(monkeypatch, fake)

    failures = script.check_redis("redis://example.test:6379")

    assert failures == ["cannot use Redis: ConnectionError"]
    assert "db.example.test" not in " ".join(failures)
