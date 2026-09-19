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
import urllib.error
import urllib.request
from typing import TypedDict

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

# Cloudflare (in front of api.groq.com) 403s Python's default "Python-urllib/x.y" User-Agent
# with body "error code: 1010" -- easy to mistake for an auth failure. A normal-looking
# User-Agent avoids it.
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


def _http_error_detail(e: urllib.error.HTTPError) -> str:
    """Read and return a truncated response body from an HTTPError, then close it.

    A 429's body is where the actual reason lives -- Groq's error responses are shaped like
    {"error": {"message", "type", "code"}} (e.g. code "rate_limit_exceeded"), which the bare
    status code and reason phrase never show.
    """
    try:
        raw = e.read().decode("utf-8", errors="replace")
    except OSError:
        raw = ""
    e.close()
    return raw[:400].replace("\n", " ")


class GroqClient:
    """Paced, retrying client for Groq's OpenAI-compatible chat/completions endpoint.

    Holds its own call-pacing state, so it's instantiated once (FastAPI lifespan) and reused
    across requests rather than recreated per call -- the same pattern as
    app/retrieval.py's RetrievalTool for Cohere rerank pacing.

    Optionally rotates across a pool of keys (key_pool) -- the same GROQ_API_KEY /
    GROQ_API_KEY_1.._21 convention the notebooks use (see load_groq_key_pool()), so a key
    that's hit its rate/day limit doesn't take the whole app down with it. Pacing
    (LLM_MAX_RPM) stays a single shared budget across the whole pool rather than per-key --
    deliberately conservative; the pool exists for failover, not for maximizing aggregate
    throughput.
    """

    def __init__(self, api_key: str, key_pool: list[str] | None = None) -> None:
        self._key_pool = key_pool if key_pool else [api_key]
        self._pool_index = 0
        self._pool_lock = threading.Lock()
        self._last_call_at = 0.0
        self._rate_limiter = _KeyRateLimiter(GROQ_RPM_LIMIT, GROQ_RPD_LIMIT)

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
        understand_query()); omit it for a free-text reply (used by answer generation). Pass
        reasoning_effort ("low"/"medium"/"high", gpt-oss models only) to control how many
        reasoning tokens the model spends before answering.

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
            return self._call_single_key(
                key,
                system_prompt,
                user_prompt,
                model=model,
                response_schema=response_schema,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                max_retries=MAX_LLM_RETRIES,
            )

        for attempt in range(total_keys):
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
                return self._call_single_key(
                    key,
                    system_prompt,
                    user_prompt,
                    model=model,
                    response_schema=response_schema,
                    temperature=temperature,
                    reasoning_effort=reasoning_effort,
                    max_retries=1,
                )
            except urllib.error.HTTPError as e:
                with self._pool_lock:
                    if self._pool_index == idx:
                        self._pool_index = (idx + 1) % total_keys
                if attempt < total_keys - 1:
                    print(
                        f"   [Groq key #{idx + 1}/{total_keys} failed fast (HTTP {e.code}); "
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
        return self._call_single_key(
            fallback_key,
            system_prompt,
            user_prompt,
            model=model,
            response_schema=response_schema,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            max_retries=MAX_LLM_RETRIES,
        )

    def _call_single_key(
        self,
        api_key: str,
        system_prompt: str,
        user_prompt: str,
        *,
        model: str,
        response_schema: dict | None,
        temperature: float | None,
        reasoning_effort: str | None,
        max_retries: int,
    ) -> LLMResponse:
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
        body = json.dumps(payload).encode()

        delay_cap = BASE_DELAY_S
        for attempt in range(1, max_retries + 1):
            self._pace()
            req = urllib.request.Request(GROQ_URL, data=body, headers=_GROQ_HEADERS, method="POST")
            req.add_header("Authorization", f"Bearer {api_key.strip()}")
            retry_after: str | None = None
            detail = ""
            label = ""
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = json.load(resp)
                text = data["choices"][0]["message"]["content"].strip()
                meta = data.get("usage", {})
                usage: Usage = {
                    "prompt_tokens": meta.get("prompt_tokens", 0),
                    "completion_tokens": meta.get("completion_tokens", 0),
                    "total_tokens": meta.get("total_tokens", 0),
                }
                return {"text": text, "usage": usage}
            except urllib.error.HTTPError as e:
                retryable = e.code in RETRYABLE_HTTP_CODES
                retry_after = e.headers.get("Retry-After") if retryable else None
                label = f"HTTP {e.code} {e.reason}"
                detail = _http_error_detail(e)
                if not retryable or attempt == max_retries:
                    raise
            except urllib.error.URLError as e:
                label = f"connection error ({e.reason})"
                if attempt == max_retries:
                    raise

            if retry_after:
                try:
                    wait = float(retry_after)
                except ValueError:
                    wait = random.uniform(0, delay_cap)
            else:
                wait = random.uniform(0, delay_cap)
            suffix = f"  {detail}" if detail else ""
            print(f"   [LLM {label}; retrying in {wait:.1f}s ({attempt}/{max_retries})]{suffix}")
            time.sleep(wait)
            delay_cap = min(delay_cap * 2, MAX_DELAY_S)

        raise RuntimeError("unreachable")  # the loop above always returns or raises
