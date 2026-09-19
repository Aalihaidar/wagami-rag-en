import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from app.agent.llm import (
    GROQ_RPM_LIMIT,
    AllKeysRateLimitedError,
    GroqClient,
    LLMStreamError,
    _KeyRateLimiter,
    load_groq_key_pool,
)

Handler = Callable[[httpx.Request], httpx.Response]


def _client(handler: Handler) -> httpx.Client:
    """An httpx client whose every request goes to `handler` instead of the network."""
    return httpx.Client(transport=httpx.MockTransport(handler))


def _groq(handler: Handler, *keys: str) -> GroqClient:
    keys = keys or ("only-key",)
    return GroqClient(keys[0], key_pool=list(keys), http_client=_client(handler))


def _success(text: str = "hello") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": text}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        },
    )


def _error(code: int, **headers: str) -> httpx.Response:
    return httpx.Response(code, json={"error": {"message": "boom"}}, headers=headers)


def _sse(*chunks: dict[str, Any] | str) -> httpx.Response:
    body = "".join(f"data: {c if isinstance(c, str) else json.dumps(c)}\n\n" for c in chunks)
    return httpx.Response(200, content=body.encode(), headers={"content-type": "text/event-stream"})


def _delta(text: str) -> dict[str, Any]:
    return {"choices": [{"delta": {"content": text}}]}


@pytest.fixture(autouse=True)
def _clear_ambient_key_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    """This dev machine's own .env carries a real 21-key rotation pool -- without this,
    load_groq_key_pool() picks up real ambient keys instead of the ones each test sets, the
    same "unconfigured isn't actually true on every machine" gotcha test_chat_endpoint.py's
    _unconfigured_settings hit first."""
    for i in range(1, 22):
        monkeypatch.delenv(f"GROQ_API_KEY_{i}", raising=False)


@pytest.fixture(autouse=True)
def _no_real_sleeping(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Retry backoff and pacing must never actually wait in a test; records what was asked."""
    waits: list[float] = []
    monkeypatch.setattr("app.agent.llm.time.sleep", waits.append)
    return waits


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
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _success("hi there")

    result = _groq(handler).call("system", "user", model="test-model")

    assert result["text"] == "hi there"
    assert result["usage"] == {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
    assert len(seen) == 1
    assert seen[0].headers["Authorization"] == "Bearer only-key"
    body = json.loads(seen[0].content)
    assert body["model"] == "test-model"
    assert "stream" not in body


def test_pool_rotates_to_next_key_after_a_failure() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.headers["Authorization"])
        return _error(429) if len(calls) == 1 else _success("recovered")

    client = _groq(handler, "key-1", "key-2", "key-3")
    result = client.call("system", "user", model="test-model")

    assert result["text"] == "recovered"
    # First attempt used key-1, second attempt (after rotation) used key-2.
    assert calls == ["Bearer key-1", "Bearer key-2"]
    # The pool pointer stays advanced for the next call, rather than resetting.
    assert client._pool_index == 1


def test_pool_falls_back_to_full_retry_after_every_key_fails_once() -> None:
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return _error(500) if call_count <= 2 else _success("finally")

    result = _groq(handler, "key-1", "key-2").call("system", "user", model="test-model")

    assert result["text"] == "finally"
    # One fast attempt per pool key (2), then a third call as the full-retry fallback.
    assert call_count == 3


def test_pool_of_one_behaves_like_plain_single_key_client() -> None:
    result = _groq(lambda r: _success("solo"), "only-key").call("system", "user", model="m")

    assert result["text"] == "solo"


def test_single_key_raises_without_calling_when_already_at_rpm_limit() -> None:
    calls: list[httpx.Request] = []
    client = _groq(lambda r: calls.append(r) or _success())  # type: ignore[func-returns-value]
    for _ in range(GROQ_RPM_LIMIT):
        client._rate_limiter.record("only-key")

    with pytest.raises(AllKeysRateLimitedError):
        client.call("system", "user", model="test-model")
    assert calls == []


def test_pool_skips_a_rate_limited_key_and_calls_an_available_one() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.headers["Authorization"])
        return _success("from key-2")

    client = _groq(handler, "key-1", "key-2")
    for _ in range(GROQ_RPM_LIMIT):
        client._rate_limiter.record("key-1")

    result = client.call("system", "user", model="test-model")

    assert result["text"] == "from key-2"
    # Only one call was made at all -- key-1 was skipped locally, never sent an HTTP request.
    assert calls == ["Bearer key-2"]


def test_pool_raises_without_calling_when_every_key_is_at_its_limit() -> None:
    calls: list[httpx.Request] = []
    client = _groq(lambda r: calls.append(r) or _success(), "key-1", "key-2", "key-3")  # type: ignore[func-returns-value]
    for key in client._key_pool:
        for _ in range(GROQ_RPM_LIMIT):
            client._rate_limiter.record(key)

    with pytest.raises(AllKeysRateLimitedError):
        client.call("system", "user", model="test-model")
    assert calls == []


def test_a_non_retryable_status_raises_immediately_without_retrying() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return _error(400)

    with pytest.raises(httpx.HTTPStatusError) as exc:
        _groq(handler).call("system", "user", model="test-model")

    assert exc.value.response.status_code == 400
    assert calls == 1


def test_a_retryable_status_is_retried_and_honors_retry_after(
    _no_real_sleeping: list[float],
) -> None:
    responses = [_error(429, **{"Retry-After": "7"}), _success("after wait")]

    result = _groq(lambda r: responses.pop(0)).call("system", "user", model="test-model")

    assert result["text"] == "after wait"
    assert 7.0 in _no_real_sleeping


def test_a_connection_error_is_retried() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.ConnectError("boom", request=request)
        return _success("connected")

    result = _groq(handler).call("system", "user", model="test-model")

    assert result["text"] == "connected"
    assert attempts == 2


def test_a_timeout_is_not_retried() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(httpx.ReadTimeout):
        _groq(handler).call("system", "user", model="test-model")
    assert attempts == 1


def test_stream_yields_text_deltas_and_reads_usage_from_the_final_chunk() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _sse(
            {"choices": [{"delta": {"role": "assistant"}}]},
            _delta("Hel"),
            {"choices": [{"delta": {"reasoning": "thinking"}}]},
            _delta("lo"),
            {
                "choices": [],
                "x_groq": {
                    "usage": {"prompt_tokens": 4, "completion_tokens": 5, "total_tokens": 9}
                },
            },
            "[DONE]",
        )

    stream = _groq(handler).stream("system", "user", model="test-model")
    assert stream.usage["total_tokens"] == 0  # nothing is known until it's been consumed
    assert list(stream) == ["Hel", "lo"]

    assert stream.usage == {"prompt_tokens": 4, "completion_tokens": 5, "total_tokens": 9}
    body = json.loads(seen[0].content)
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}


def test_stream_rotates_keys_on_an_upfront_failure() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.headers["Authorization"])
        return _error(429) if len(calls) == 1 else _sse(_delta("ok"), "[DONE]")

    stream = _groq(handler, "key-1", "key-2").stream("system", "user", model="test-model")

    assert list(stream) == ["ok"]
    assert calls == ["Bearer key-1", "Bearer key-2"]


def test_stream_raises_without_calling_when_every_key_is_at_its_limit() -> None:
    calls: list[httpx.Request] = []
    client = _groq(lambda r: calls.append(r) or _sse("[DONE]"))  # type: ignore[func-returns-value]
    for _ in range(GROQ_RPM_LIMIT):
        client._rate_limiter.record("only-key")

    with pytest.raises(AllKeysRateLimitedError):
        client.stream("system", "user", model="test-model")
    assert calls == []


def test_stream_surfaces_an_error_chunk_from_inside_the_stream() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _sse(_delta("partial"), {"error": {"message": "overloaded"}})

    stream = _groq(handler).stream("system", "user", model="test-model")
    received: list[str] = []

    with pytest.raises(LLMStreamError, match="overloaded"):
        for text in stream:
            received.append(text)
    assert received == ["partial"]


def test_stream_closes_the_connection_when_the_consumer_stops_early() -> None:
    response = _sse(_delta("a"), _delta("b"), _delta("c"), "[DONE]")
    stream = _groq(lambda r: response).stream("system", "user", model="test-model")

    for text in stream:
        assert text == "a"
        break
    stream.close()

    assert response.is_closed


def test_rate_limiter_available_resets_after_the_minute_window_rolls_over(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    limiter = _KeyRateLimiter(rpm_limit=GROQ_RPM_LIMIT, rpd_limit=1_000_000)
    fake_now = [0.0]
    monkeypatch.setattr("app.agent.llm.time.monotonic", lambda: fake_now[0])
    monkeypatch.setattr("app.agent.llm.time.time", lambda: fake_now[0])

    for _ in range(GROQ_RPM_LIMIT):
        limiter.record("k")
    assert limiter.available("k") is False

    fake_now[0] += 61  # past the 60s window boundary
    assert limiter.available("k") is True


def test_rate_limiter_enforces_rpd_independently_of_rpm(monkeypatch: pytest.MonkeyPatch) -> None:
    limiter = _KeyRateLimiter(rpm_limit=1_000_000, rpd_limit=2)
    fake_now = [0.0]
    monkeypatch.setattr("app.agent.llm.time.monotonic", lambda: fake_now[0])
    monkeypatch.setattr("app.agent.llm.time.time", lambda: fake_now[0])

    limiter.record("k")
    fake_now[0] += 90  # a new minute window, same day -- RPD should still accumulate
    limiter.record("k")
    assert limiter.available("k") is False  # hit the RPD cap even though RPM reset
