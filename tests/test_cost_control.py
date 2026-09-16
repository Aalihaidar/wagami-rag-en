import datetime as dt
from typing import Any

import redis.exceptions

from app.cost_control import is_over_spend_limit, record_token_usage


class FakeRedis:
    def __init__(self) -> None:
        self.store: dict[str, int] = {}

    def get(self, key: str) -> int | None:
        return self.store.get(key)

    def pipeline(self) -> FakeRedis:
        return self

    def incrby(self, key: str, amount: int) -> FakeRedis:
        self.store[key] = self.store.get(key, 0) + amount
        return self

    def expire(self, key: str, ttl: int) -> FakeRedis:
        return self

    def execute(self) -> None:
        pass


def test_is_over_spend_limit_false_when_no_usage_recorded_yet() -> None:
    redis_client = FakeRedis()
    assert is_over_spend_limit(redis_client, daily_limit=1000, monthly_limit=10000) is False


def test_record_token_usage_then_is_over_spend_limit_true_once_daily_limit_hit() -> None:
    redis_client = FakeRedis()
    now = dt.datetime(2026, 9, 16, tzinfo=dt.UTC)
    record_token_usage(redis_client, 600, now=now)
    assert (
        is_over_spend_limit(redis_client, daily_limit=1000, monthly_limit=10000, now=now) is False
    )
    record_token_usage(redis_client, 500, now=now)
    assert is_over_spend_limit(redis_client, daily_limit=1000, monthly_limit=10000, now=now) is True


def test_record_token_usage_also_accumulates_the_monthly_counter() -> None:
    redis_client = FakeRedis()
    now = dt.datetime(2026, 9, 16, tzinfo=dt.UTC)
    record_token_usage(redis_client, 100, now=now)
    later_same_month = dt.datetime(2026, 9, 30, tzinfo=dt.UTC)
    record_token_usage(redis_client, 100, now=later_same_month)
    # Daily limit is generous, but the monthly total (200) should trip a low monthly limit.
    assert (
        is_over_spend_limit(redis_client, daily_limit=1000, monthly_limit=150, now=later_same_month)
        is True
    )


def test_record_token_usage_ignores_zero_or_negative() -> None:
    redis_client = FakeRedis()
    record_token_usage(redis_client, 0)
    record_token_usage(redis_client, -5)
    assert redis_client.store == {}


def test_daily_counter_does_not_leak_into_a_different_day() -> None:
    redis_client = FakeRedis()
    day_one = dt.datetime(2026, 9, 16, tzinfo=dt.UTC)
    day_two = dt.datetime(2026, 9, 17, tzinfo=dt.UTC)
    record_token_usage(redis_client, 900, now=day_one)
    # Day two starts fresh even though day one was close to a tight daily limit.
    over_limit = is_over_spend_limit(
        redis_client, daily_limit=1000, monthly_limit=10000, now=day_two
    )
    assert over_limit is False


class UnreachableRedis:
    """Simulates Redis being down -- every call raises, like a real redis-py ConnectionError."""

    def get(self, key: str) -> int | None:
        raise redis.exceptions.ConnectionError("simulated Redis outage")

    def pipeline(self) -> UnreachableRedis:
        return self

    def incrby(self, key: str, amount: int) -> UnreachableRedis:
        raise redis.exceptions.ConnectionError("simulated Redis outage")

    def expire(self, key: str, ttl: int) -> UnreachableRedis:
        return self

    def execute(self) -> None:
        raise redis.exceptions.ConnectionError("simulated Redis outage")


def test_is_over_spend_limit_fails_open_when_redis_is_unreachable() -> None:
    """Section E's graceful-degradation requirement: an unreachable Redis must not turn the
    cost-control backstop into an outage of its own -- guests keep getting answered, bounded
    only by the per-IP rate limit and concurrency cap until Redis comes back."""
    assert is_over_spend_limit(UnreachableRedis(), daily_limit=1000, monthly_limit=10000) is False


def test_record_token_usage_does_not_raise_when_redis_is_unreachable() -> None:
    record_token_usage(UnreachableRedis(), 500)  # must not raise


def test_rate_limiting_actually_enforces_the_configured_limit() -> None:
    """Proves the slowapi wiring pattern app/main.py uses (Limiter + @limiter.limit(...) +
    a `request: Request` parameter + the RateLimitExceeded exception handler) genuinely
    enforces a limit -- using an isolated app/Limiter with in-memory storage, not the real
    Redis-backed module-level limiter, so this needs no live Redis."""
    from fastapi import FastAPI, Request
    from fastapi.testclient import TestClient
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from slowapi.util import get_remote_address

    limiter = Limiter(key_func=get_remote_address, storage_uri="memory://")
    app = FastAPI()
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]

    @app.get("/ping")
    @limiter.limit("2/minute")
    async def ping(request: Request) -> dict[str, Any]:
        return {"ok": True}

    with TestClient(app) as client:
        assert client.get("/ping").status_code == 200
        assert client.get("/ping").status_code == 200
        third = client.get("/ping")
        assert third.status_code == 429
