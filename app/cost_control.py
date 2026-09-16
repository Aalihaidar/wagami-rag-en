"""Section D (rate limiting & cost control) helpers that aren't specifically about the
Weaviate/LLM pipeline itself: a daily/monthly spend circuit breaker backed by Redis.

Token counters are a proxy for spend, not real dollar-cost tracking -- token-to-dollar
conversion varies by provider/model. Good enough to stop a traffic spike from hitting
provider bills unbounded; still check the actual Cohere/Groq/Weaviate usage dashboards
regularly (Section 3) -- this is a backstop, not a substitute.
"""

import datetime as dt
from typing import Any

CONVERSATION_LIMIT_REPLY = (
    "This conversation has gotten pretty long! Please start a new chat so I can keep giving "
    'you my full attention -- look for the "New chat" button.'
)

CAPACITY_REPLY = (
    "This demo is temporarily at capacity for today -- please check back later. Sorry for "
    "the inconvenience!"
)

DAILY_TOKEN_LIMIT_DEFAULT = 200_000
MONTHLY_TOKEN_LIMIT_DEFAULT = 2_000_000
MAX_CONVERSATION_TURNS_DEFAULT = 20

# TTLs are longer than the natural period as a safety net (e.g. clock skew), not because
# the counters are meant to survive past their period -- each period's key is fresh.
_DAILY_KEY_TTL_SECONDS = 2 * 24 * 3600
_MONTHLY_KEY_TTL_SECONDS = 35 * 24 * 3600


def _daily_key(now: dt.datetime) -> str:
    return f"usage:tokens:daily:{now:%Y-%m-%d}"


def _monthly_key(now: dt.datetime) -> str:
    return f"usage:tokens:monthly:{now:%Y-%m}"


def is_over_spend_limit(
    redis_client: Any,
    *,
    daily_limit: int,
    monthly_limit: int,
    now: dt.datetime | None = None,
) -> bool:
    """True if either the daily or monthly token counter is already at/over its limit.

    Checked before graph.invoke() -- this can only block calls made *after* the one that
    pushed the counter over the limit, not that call itself, which is the standard shape of
    this kind of soft circuit breaker.
    """
    now = now or dt.datetime.now(dt.UTC)
    daily = int(redis_client.get(_daily_key(now)) or 0)
    monthly = int(redis_client.get(_monthly_key(now)) or 0)
    return daily >= daily_limit or monthly >= monthly_limit


def record_token_usage(
    redis_client: Any, total_tokens: int, *, now: dt.datetime | None = None
) -> None:
    """Add total_tokens to today's/this month's running counters."""
    if total_tokens <= 0:
        return
    now = now or dt.datetime.now(dt.UTC)
    daily_key = _daily_key(now)
    monthly_key = _monthly_key(now)
    pipe = redis_client.pipeline()
    pipe.incrby(daily_key, total_tokens)
    pipe.expire(daily_key, _DAILY_KEY_TTL_SECONDS)
    pipe.incrby(monthly_key, total_tokens)
    pipe.expire(monthly_key, _MONTHLY_KEY_TTL_SECONDS)
    pipe.execute()
