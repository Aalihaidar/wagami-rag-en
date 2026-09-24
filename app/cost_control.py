"""Section D (rate limiting & cost control) helpers that aren't specifically about the
Weaviate/LLM pipeline itself: a daily/monthly spend circuit breaker backed by Redis.

Token counters are a proxy for spend, not real dollar-cost tracking -- token-to-dollar
conversion varies by provider/model. Good enough to stop a traffic spike from hitting
provider bills unbounded; still check the actual Cohere/Groq/Weaviate usage dashboards
regularly (Section 3) -- this is a backstop, not a substitute.
"""

import datetime as dt
import logging
from typing import Any

import redis.exceptions

logger = logging.getLogger("app.cost_control")

CONVERSATION_LIMIT_REPLY = (
    "This conversation has gotten pretty long! Please start a new chat so I can keep giving "
    'you my full attention. Look for the "New chat" button.'
)

CAPACITY_REPLY = (
    "This demo is temporarily at capacity for today. Please check back later. Sorry for "
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

    Fails open (returns False) if Redis itself is unreachable (Section E's graceful-
    degradation requirement) -- the per-IP rate limit and concurrency cap still bound cost
    without this backstop, so refusing every guest because the spend counter can't be read
    would be a worse outcome than temporarily losing the backstop itself.
    """
    now = now or dt.datetime.now(dt.UTC)
    try:
        daily = int(redis_client.get(_daily_key(now)) or 0)
        monthly = int(redis_client.get(_monthly_key(now)) or 0)
    except redis.exceptions.RedisError:
        logger.warning(
            "Redis unreachable while checking spend limit -- failing open", exc_info=True
        )
        return False
    return daily >= daily_limit or monthly >= monthly_limit


def record_token_usage(
    redis_client: Any, total_tokens: int, *, now: dt.datetime | None = None
) -> None:
    """Add total_tokens to today's/this month's running counters.

    Fails open (swallows the error) if Redis is unreachable -- same reasoning as
    is_over_spend_limit above: this turn's usage silently goes uncounted rather than the
    guest's already-answered turn turning into a 500.
    """
    if total_tokens <= 0:
        return
    now = now or dt.datetime.now(dt.UTC)
    daily_key = _daily_key(now)
    monthly_key = _monthly_key(now)
    try:
        pipe = redis_client.pipeline()
        pipe.incrby(daily_key, total_tokens)
        pipe.expire(daily_key, _DAILY_KEY_TTL_SECONDS)
        pipe.incrby(monthly_key, total_tokens)
        pipe.expire(monthly_key, _MONTHLY_KEY_TTL_SECONDS)
        pipe.execute()
    except redis.exceptions.RedisError:
        logger.warning(
            "Redis unreachable while recording token usage -- usage not counted", exc_info=True
        )
