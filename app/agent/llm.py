"""Groq chat-completions client -- the LLM backend for app/agent/'s query
understanding and answer generation.

Ported from the verified source, `03_evaluation_groq.ipynb`'s `call_llm()`
helper (see docs/APP_AND_DEPLOYMENT_PLAN.md's LLM-provider note for why Groq,
not Gemini, and why that specific notebook).
"""

import json
import os
import random
import threading
import time
from collections.abc import Callable, Iterator
from typing import TypedDict

import httpx

from app.timing import timed

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
RETRYABLE_HTTP_CODES = {408, 429, 500, 502, 503, 504}
MAX_LLM_RETRIES = 5
BASE_DELAY_S = 1.0
MAX_DELAY_S = 20.0
# Edit to match your key's actual quota -- check its own x-ratelimit-limit-requests response
# header rather than trusting a provider docs page, which can read lower/stale for your tier.
LLM_MAX_RPM = 1000.0

# Groq free-tier limits, confirmed 2026-09-18 -- a *local, proactive* guard, separate from
# LLM_MAX_RPM's reactive pacing above. GroqClient tracks each pool key's own request count in
# fixed windows and skips a key already at its confirmed limit rather than spending a real
# request just to find out via a 429. Edit if your tier differs.
GROQ_RPM_LIMIT = 30
GROQ_RPD_LIMIT = 1000

# One shared, keep-alive connection pool per GroqClient instead of a fresh TCP+TLS handshake
# per request (~125ms to api.groq.com, measured) -- httpx.Client is thread-safe, which the
# threadpool-shared GroqClient needs. The read timeout also bounds the gap between streamed
# chunks, not just a whole response.
LLM_HTTP_TIMEOUT = httpx.Timeout(60.0, connect=10.0)

# Cloudflare (in front of api.groq.com) 403s a bare library-default User-Agent with body
# "error code: 1010" -- easy to mistake for an auth failure. A normal-looking User-Agent
# avoids it.
_GROQ_HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0 (compatible; wagami-rag-app/1.0)",
}


class Usage(TypedDict):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class LLMResponse(TypedDict):
    text: str
    usage: Usage


def zero_usage() -> Usage:
    """A fresh {prompt,completion,total: 0} usage dict, for calls that never hit the API."""
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def load_groq_key_pool(base_key: str) -> list[str]:
    """base_key plus any GROQ_API_KEY_1..GROQ_API_KEY_21 set in the environment, in order.

    Same pool/convention the notebooks already use -- read directly from os.environ rather
    than app/config.py's Settings, since these are a variable-length,
    deliberately-undocumented-in-.env.example personal-account pool, not a fixed set of
    fields worth modeling on Settings.
    """
    pool = [base_key.strip()] if base_key.strip() else []
    for i in range(1, 22):
        key = os.environ.get(f"GROQ_API_KEY_{i}", "").strip()
        if key:
            pool.append(key)
    return pool


class AllKeysRateLimitedError(Exception):
    """Every key in the pool is at its local RPM/RPD limit -- call() raises this *without*
    making any HTTP request, rather than sending a call the provider would just reject."""


class _KeyRateLimiter:
    """Per-key fixed-window request counter (RPM + RPD), shared across the whole pool.

    Fixed windows, not a rolling one -- simpler, and "approximately N requests per
    minute/day" is the actual goal (a client-side safety margin, not exact provider-side
    parity). The minute window uses time.monotonic() (immune to system clock changes); the
    day window uses time.time() since "day" is inherently calendar-relative. Both are process-
    local: with more than one worker process, each has its own view of a key's usage, so the
    real limit is enforced approximately, not exactly -- acceptable for a proactive guard
    backed by the provider's own reactive 429 as the real backstop (see call()'s docstring).
    """

    def __init__(self, rpm_limit: int, rpd_limit: int) -> None:
        self.rpm_limit = rpm_limit
        self.rpd_limit = rpd_limit
        self._lock = threading.Lock()
        self._minute_window: dict[str, int] = {}
        self._minute_count: dict[str, int] = {}
        self._day_window: dict[str, int] = {}
        self._day_count: dict[str, int] = {}

    def available(self, key: str) -> bool:
        with self._lock:
            minute = int(time.monotonic() // 60)
            day = int(time.time() // 86400)
            in_minute = self._minute_window.get(key) == minute
            in_day = self._day_window.get(key) == day
            minute_count = self._minute_count.get(key, 0) if in_minute else 0
            day_count = self._day_count.get(key, 0) if in_day else 0
            return minute_count < self.rpm_limit and day_count < self.rpd_limit

    def record(self, key: str) -> None:
        with self._lock:
            minute = int(time.monotonic() // 60)
            day = int(time.time() // 86400)
            if self._minute_window.get(key) != minute:
                self._minute_window[key] = minute
                self._minute_count[key] = 0
            self._minute_count[key] += 1
            if self._day_window.get(key) != day:
                self._day_window[key] = day
                self._day_count[key] = 0
            self._day_count[key] += 1


def _http_error_detail(response: httpx.Response) -> str:
    """Truncated response body of an already-read error response.

    A 429's body is where the actual reason lives -- Groq's error responses are shaped like
    {"error": {"message", "type", "code"}} (e.g. code "rate_limit_exceeded"), which the bare
    status code and reason phrase never show.
    """
    return response.text[:400].replace("\n", " ")


class LLMStreamError(Exception):
    """Groq reported an error inside an already-open (HTTP 200) event stream."""


class LLMStream:
    """One live streamed chat completion: iterate it for text deltas as they arrive.

    `usage` is only complete once iteration has finished; it stays zero if the stream is
    abandoned early. Always closes the underlying connection when iteration ends -- normally,
    on error, or when the consumer stops early (close() / generator finalization).
    """

    def __init__(self, response: httpx.Response) -> None:
        self._response = response
        self.usage: Usage = zero_usage()

    def __iter__(self) -> Iterator[str]:
        try:
            for line in self._response.iter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if "error" in chunk:
                    raise LLMStreamError(str(chunk["error"])[:400])
                # Groq reports usage on the final chunk under its own `x_groq` key; the
                # OpenAI-style top-level `usage` is read too in case that changes.
                meta = chunk.get("usage") or chunk.get("x_groq", {}).get("usage")
                if meta:
                    self.usage = {
                        "prompt_tokens": meta.get("prompt_tokens", 0),
                        "completion_tokens": meta.get("completion_tokens", 0),
                        "total_tokens": meta.get("total_tokens", 0),
                    }
                for choice in chunk.get("choices", []):
                    text = choice.get("delta", {}).get("content")
                    if text:
                        yield text
        finally:
            self._response.close()

    def close(self) -> None:
        self._response.close()


class GroqClient:
    """Paced, retrying client for Groq's OpenAI-compatible chat/completions endpoint.

    Holds its own call-pacing state and HTTP connection pool, so it's instantiated once
    (FastAPI lifespan) and reused across requests rather than recreated per call -- the same
    pattern as app/retrieval.py's RetrievalTool for Cohere rerank pacing.

    Optionally rotates across a pool of keys (key_pool) -- the same GROQ_API_KEY /
    GROQ_API_KEY_1.._21 convention the notebooks use (see load_groq_key_pool()), so a key
    that's hit its rate/day limit doesn't take the whole app down with it. Pacing
    (LLM_MAX_RPM) stays a single shared budget across the whole pool rather than per-key --
    deliberately conservative; the pool exists for failover, not for maximizing aggregate
    throughput.
    """

    def __init__(
        self,
        api_key: str,
        key_pool: list[str] | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._key_pool = key_pool if key_pool else [api_key]
        self._pool_index = 0
        self._pool_lock = threading.Lock()
        self._last_call_at = 0.0
        self._rate_limiter = _KeyRateLimiter(GROQ_RPM_LIMIT, GROQ_RPD_LIMIT)
        self._http = http_client or httpx.Client(timeout=LLM_HTTP_TIMEOUT, headers=_GROQ_HEADERS)

    def close(self) -> None:
        self._http.close()

    def _pace(self) -> None:
        wait = (60.0 / LLM_MAX_RPM) - (time.monotonic() - self._last_call_at)
        if wait > 0:
            time.sleep(wait)
        self._last_call_at = time.monotonic()

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        model: str,
        response_schema: dict | None = None,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
    ) -> LLMResponse:
        """POST to Groq's chat/completions endpoint; retry transient failures with backoff.

        Pass response_schema to force strict structured JSON output (used by
        understand_query()); omit it for a free-text reply. Pass reasoning_effort
        ("low"/"medium"/"high", gpt-oss models only) to control how many reasoning tokens the
        model spends before answering. Key selection, local rate-limit skipping and rotation
        are described on _dispatch().
        """
        body = _build_body(
            system_prompt,
            user_prompt,
            model=model,
            response_schema=response_schema,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            stream=False,
        )
        return self._dispatch(
            lambda key, retries: _parse_completion(
                self._request_with_retries(key, body, stream=False, max_retries=retries)
            )
        )

    def stream(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        model: str,
        response_schema: dict | None = None,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
    ) -> LLMStream:
        """Like call(), but returns an open LLMStream to iterate for text deltas.

        Everything that can fail up front -- a 429, a bad key, every key rate-limited -- is
        handled (or raised) here, exactly as in call(), before any text has been produced.
        Once this returns, the response is a live HTTP 200: a failure mid-stream can no
        longer be retried or rotated (part of the answer may already be with the guest), so
        it surfaces as an exception from iterating the stream instead.
        """
        body = _build_body(
            system_prompt,
            user_prompt,
            model=model,
            response_schema=response_schema,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            stream=True,
        )
        return self._dispatch(
            lambda key, retries: LLMStream(
                self._request_with_retries(key, body, stream=True, max_retries=retries)
            )
        )

    def _dispatch[T](self, attempt: Callable[[str, int], T]) -> T:
        """Run `attempt(key, max_retries)` against the pool, rotating keys on failure.

        Before ever calling out, each candidate key is checked against its own local RPM/RPD
        budget (GROQ_RPM_LIMIT/GROQ_RPD_LIMIT) -- a key already at its limit is skipped with no
        HTTP request made, not tried and left to 429. If every pool key is at its limit,
        raises AllKeysRateLimitedError without sending anything (see that class's docstring).
        This is a proactive local guard on top of, not instead of, the provider's own reactive
        429 -- the local counters are an approximation (see _KeyRateLimiter), so a genuine 429
        from a key this method thought was available is still handled the normal way below.

        With more than one pool key, tries each once (a single fast attempt, no backoff)
        before falling to the next -- honoring a failing key's full retry/backoff first would
        make rotation too slow to be worth it (same reasoning as the notebooks' own
        `_call_llm_with_rotation`). Once every key has failed once or is rate-limited, falls
        through to one full-retry/backoff call on whichever key still has budget, in case the
        failure was transient rather than the whole pool being genuinely exhausted. Unlike the
        notebooks (single-threaded, a shared global "current key"), the key used per attempt is
        kept local to this call rather than mutating shared state beyond the rotation pointer
        itself -- this class is shared across concurrent request-handling threads (FastAPI's
        threadpool), so a "currently active key" attribute would race.
        """
        total_keys = len(self._key_pool)
        if total_keys <= 1:
            key = self._key_pool[0]
            if not self._rate_limiter.available(key):
                raise AllKeysRateLimitedError(
                    f"The only configured key is at its local rate limit "
                    f"({GROQ_RPM_LIMIT} RPM / {GROQ_RPD_LIMIT} RPD) -- not sending this request."
                )
            self._rate_limiter.record(key)
            return attempt(key, MAX_LLM_RETRIES)

        for position in range(total_keys):
            with self._pool_lock:
                idx = self._pool_index
            key = self._key_pool[idx]
            if not self._rate_limiter.available(key):
                with self._pool_lock:
                    if self._pool_index == idx:
                        self._pool_index = (idx + 1) % total_keys
                print(
                    f"   [Groq key #{idx + 1}/{total_keys} at its local rate limit -- "
                    "skipping, no call made]"
                )
                continue
            self._rate_limiter.record(key)
            try:
                return attempt(key, 1)
            except httpx.HTTPStatusError as e:
                with self._pool_lock:
                    if self._pool_index == idx:
                        self._pool_index = (idx + 1) % total_keys
                if position < total_keys - 1:
                    print(
                        f"   [Groq key #{idx + 1}/{total_keys} failed fast "
                        f"(HTTP {e.response.status_code}); "
                        f"rotating to key #{self._pool_index + 1}/{total_keys}]"
                    )

        with self._pool_lock:
            idx = self._pool_index
        candidate_key = self._key_pool[idx]
        fallback_key = (
            candidate_key
            if self._rate_limiter.available(candidate_key)
            else next((k for k in self._key_pool if self._rate_limiter.available(k)), None)
        )
        if fallback_key is None:
            raise AllKeysRateLimitedError(
                f"All {total_keys} pool keys are at their local rate limit "
                f"({GROQ_RPM_LIMIT} RPM / {GROQ_RPD_LIMIT} RPD) -- not sending this request."
            )
        self._rate_limiter.record(fallback_key)
        print(
            "   [every pool key failed once or was rate-limited -- falling back to full "
            "retry/backoff]"
        )
        return attempt(fallback_key, MAX_LLM_RETRIES)

    def _request_with_retries(
        self, api_key: str, body: bytes, *, stream: bool, max_retries: int
    ) -> httpx.Response:
        """POST `body`, retrying retryable failures with jittered exponential backoff.

        Returns the first successful response -- fully read for stream=False, headers-only
        (body still unread, caller must close) for stream=True. Raises httpx.HTTPStatusError
        for a non-retryable status or once retries run out. A timeout is never retried: a
        60s read timeout retried five times would hold a guest for minutes.
        """
        delay_cap = BASE_DELAY_S
        for attempt in range(1, max_retries + 1):
            self._pace()
            request = self._http.build_request(
                "POST",
                GROQ_URL,
                content=body,
                headers={"Authorization": f"Bearer {api_key.strip()}"},
            )
            retry_after: str | None = None
            detail = ""
            try:
                response = self._http.send(request, stream=stream)
            except httpx.TimeoutException:
                raise
            except httpx.TransportError as e:
                label = f"connection error ({e})"
                if attempt == max_retries:
                    raise
            else:
                if response.is_success:
                    return response
                response.read()
                response.close()
                retryable = response.status_code in RETRYABLE_HTTP_CODES
                if not retryable or attempt == max_retries:
                    response.raise_for_status()
                retry_after = response.headers.get("Retry-After")
                label = f"HTTP {response.status_code} {response.reason_phrase}"
                detail = _http_error_detail(response)

            if retry_after:
                try:
                    wait = float(retry_after)
                except ValueError:
                    wait = random.uniform(0, delay_cap)
            else:
                wait = random.uniform(0, delay_cap)
            suffix = f"  {detail}" if detail else ""
            print(f"   [LLM {label}; retrying in {wait:.1f}s ({attempt}/{max_retries})]{suffix}")
            with timed("llm_backoff"):
                time.sleep(wait)
            delay_cap = min(delay_cap * 2, MAX_DELAY_S)

        raise RuntimeError("unreachable")  # the loop above always returns or raises


def _build_body(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str,
    response_schema: dict | None,
    temperature: float | None,
    reasoning_effort: str | None,
    stream: bool,
) -> bytes:
    payload: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if reasoning_effort is not None:
        payload["reasoning_effort"] = reasoning_effort
    if response_schema is not None:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "response", "strict": True, "schema": response_schema},
        }
    if stream:
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
    return json.dumps(payload).encode()


def _parse_completion(response: httpx.Response) -> LLMResponse:
    data = response.json()
    text = data["choices"][0]["message"]["content"].strip()
    meta = data.get("usage", {})
    usage: Usage = {
        "prompt_tokens": meta.get("prompt_tokens", 0),
        "completion_tokens": meta.get("completion_tokens", 0),
        "total_tokens": meta.get("total_tokens", 0),
    }
    return {"text": text, "usage": usage}
