import io
import json
import urllib.error
from unittest.mock import patch

import pytest

from app.agent.llm import GroqClient, load_groq_key_pool


class FakeResponse:
    """Minimal stand-in for urlopen()'s context-managed response -- just enough for
    json.load(resp) (which calls resp.read()) to work."""

    def __init__(self, payload: dict) -> None:
        self._data = json.dumps(payload).encode()

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False

    def read(self) -> bytes:
        return self._data


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url="https://api.groq.com/openai/v1/chat/completions",
        code=code,
        msg="error",
        hdrs={},  # type: ignore[arg-type]
        fp=io.BytesIO(b'{"error": {"message": "boom"}}'),
    )


def _success(text: str = "hello") -> FakeResponse:
    return FakeResponse(
        {
            "choices": [{"message": {"content": text}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        }
    )


@pytest.fixture(autouse=True)
def _clear_ambient_key_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    """This dev machine's own .env carries a real 21-key rotation pool (CLAUDE.md's
    Environment & tooling section) -- without this, load_groq_key_pool() picks up real
    ambient keys instead of the ones each test sets, the same "unconfigured isn't actually
    true on every machine" gotcha test_chat_endpoint.py's _unconfigured_settings hit first."""
    for i in range(1, 22):
        monkeypatch.delenv(f"GROQ_API_KEY_{i}", raising=False)


def test_load_groq_key_pool_reads_base_and_numbered_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY_1", "key-one")
    monkeypatch.setenv("GROQ_API_KEY_2", "key-two")

    pool = load_groq_key_pool("base-key")

    assert pool == ["base-key", "key-one", "key-two"]


def test_load_groq_key_pool_skips_unset_indices(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY_2", "key-two")

    pool = load_groq_key_pool("")

    assert pool == ["key-two"]


def test_load_groq_key_pool_empty_when_nothing_set() -> None:
    assert load_groq_key_pool("") == []


def test_single_key_client_calls_groq_and_parses_response() -> None:
    client = GroqClient("only-key")
    with patch("app.agent.llm.urllib.request.urlopen", return_value=_success("hi there")) as mock:
        result = client.call("system", "user", model="test-model")

    assert result["text"] == "hi there"
    assert result["usage"] == {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
    mock.assert_called_once()


def test_pool_rotates_to_next_key_after_a_failure() -> None:
    client = GroqClient("key-1", key_pool=["key-1", "key-2", "key-3"])
    calls: list[str] = []

    def fake_urlopen(req: object, timeout: float = 60) -> FakeResponse:
        auth = req.get_header("Authorization")  # type: ignore[attr-defined]
        calls.append(auth)
        if len(calls) == 1:
            raise _http_error(429)
        return _success("recovered")

    with patch("app.agent.llm.urllib.request.urlopen", side_effect=fake_urlopen):
        result = client.call("system", "user", model="test-model")

    assert result["text"] == "recovered"
    # First attempt used key-1, second attempt (after rotation) used key-2.
    assert len(calls) == 2
    assert calls[0] != calls[1]
    # The pool pointer stays advanced for the next call, rather than resetting.
    assert client._pool_index == 1


def test_pool_falls_back_to_full_retry_after_every_key_fails_once() -> None:
    client = GroqClient("key-1", key_pool=["key-1", "key-2"])
    call_count = 0

    def fake_urlopen(req: object, timeout: float = 60) -> FakeResponse:
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            raise _http_error(500)
        return _success("finally")

    with patch("app.agent.llm.urllib.request.urlopen", side_effect=fake_urlopen):
        result = client.call("system", "user", model="test-model")

    assert result["text"] == "finally"
    # One fast attempt per pool key (2), then a third call as the full-retry fallback.
    assert call_count == 3


def test_pool_of_one_behaves_like_plain_single_key_client() -> None:
    client = GroqClient("only-key", key_pool=["only-key"])
    with patch("app.agent.llm.urllib.request.urlopen", return_value=_success("solo")):
        result = client.call("system", "user", model="test-model")

    assert result["text"] == "solo"
