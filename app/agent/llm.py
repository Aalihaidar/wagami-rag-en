"""Groq chat-completions client -- the LLM backend for app/agent/'s query
understanding and answer generation.

Ported from the verified source, `03_evaluation_groq.ipynb`'s `call_llm()`
helper (see docs/APP_AND_DEPLOYMENT_PLAN.md's LLM-provider note for why Groq,
not Gemini, and why that specific notebook).
"""

import json
import random
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
    """

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._last_call_at = 0.0

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
        """
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
        for attempt in range(1, MAX_LLM_RETRIES + 1):
            self._pace()
            req = urllib.request.Request(GROQ_URL, data=body, headers=_GROQ_HEADERS, method="POST")
            req.add_header("Authorization", f"Bearer {self._api_key.strip()}")
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
                if not retryable or attempt == MAX_LLM_RETRIES:
                    raise
            except urllib.error.URLError as e:
                label = f"connection error ({e.reason})"
                if attempt == MAX_LLM_RETRIES:
                    raise

            if retry_after:
                try:
                    wait = float(retry_after)
                except ValueError:
                    wait = random.uniform(0, delay_cap)
            else:
                wait = random.uniform(0, delay_cap)
            suffix = f"  {detail}" if detail else ""
            print(
                f"   [LLM {label}; retrying in {wait:.1f}s ({attempt}/{MAX_LLM_RETRIES})]{suffix}"
            )
            time.sleep(wait)
            delay_cap = min(delay_cap * 2, MAX_DELAY_S)

        raise RuntimeError("unreachable")  # the loop above always returns or raises
