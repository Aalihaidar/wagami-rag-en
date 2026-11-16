#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Weekly health check for the two hosted services the chat depends on (run by
.github/workflows/kb-health.yml):

Weaviate Cloud (the knowledge base), with the same read-only key and connection code the app uses:
  1. the cluster is reachable and ready (a deleted or paused cluster fails here),
  2. the `KnowledgeBase` collection exists and still holds the corpus, and
  3. one fixed question ("What are your opening hours?") comes back through a real hybrid
     search with an FAQ row among the top hits -- which also proves the embedding key still
     vectorizes queries.

Redis (the conversation memory):
  4. the database answers, a small key can be written and read back, and Redis Search is present
     (the chat memory builds search indexes at startup). The write is deliberate: free hosted
     Redis databases are removed after about two weeks without a read or write command, and a
     weekly write keeps this one active.

No LLM is called. Both checks always run, so one failing service does not hide the other.
Exits 0 when healthy, 1 with a one-line reason per failed check.

Env vars (same names as the app): WEAVIATE_URL, WEAVIATE_READ_API_KEY, EMBEDDING_API_KEY, REDIS_URL.
Locally: `uv run python scripts/check_knowledge_base.py` (`.env` is loaded first).
"""

import os
import sys
from datetime import UTC, datetime

import redis
from dotenv import load_dotenv

from app.config import Settings
from app.retrieval import connect

COLLECTION = "KnowledgeBase"
# The corpus holds 197 rows; this floor only has to catch an emptied or truncated collection,
# so it stays well below the real count and needs no edit when a few rows are added or removed.
MIN_OBJECTS = 150
QUESTION = "What are your opening hours?"
TOP = 3

KEEPALIVE_KEY = "healthcheck:keepalive"
# Long enough to read back during the run, short enough that the key never lingers in the 30 MB
# free database for more than a few days.
KEEPALIVE_TTL_SECONDS = 3 * 24 * 3600


def _detail(exc: Exception) -> str:
    """The error text for the log -- the type only on GitHub Actions.

    This repository is public, so its Actions logs are too, and a Redis connection error names the
    host it could not reach. Run the script locally to see the full message.
    """
    if os.environ.get("GITHUB_ACTIONS") == "true":
        return type(exc).__name__
    return f"{type(exc).__name__}: {exc}"


def check_weaviate(settings: Settings) -> list[str]:
    """Return one reason per failed Weaviate check (an empty list means healthy)."""
    missing = [
        name
        for name, value in (
            ("WEAVIATE_URL", settings.weaviate_url),
            ("WEAVIATE_READ_API_KEY", settings.weaviate_read_api_key),
            ("EMBEDDING_API_KEY", settings.embedding_api_key),
        )
        if not value
    ]
    if missing:
        return [f"missing settings: {', '.join(missing)}"]

    try:
        client = connect(settings)
    except Exception as exc:  # noqa: BLE001 - any connect error means the cluster is not usable
        return [f"cannot connect to the cluster: {type(exc).__name__}: {exc}"]

    failures: list[str] = []
    try:
        if not client.is_ready():
            failures.append("cluster connected but is not ready")
        elif not client.collections.exists(COLLECTION):
            failures.append(f"collection {COLLECTION!r} does not exist")
        else:
            collection = client.collections.get(COLLECTION)
            total = collection.aggregate.over_all(total_count=True).total_count or 0
            print(f"ok    {COLLECTION} holds {total} objects")
            if total < MIN_OBJECTS:
                failures.append(f"only {total} objects, expected at least {MIN_OBJECTS}")

            result = collection.query.hybrid(query=QUESTION, limit=TOP)
            hits = [
                (o.properties.get("item_type"), o.properties.get("name")) for o in result.objects
            ]
            print(f"ok    {QUESTION!r} returned {len(hits)} hits: {hits}")
            if not any(item_type == "faq" for item_type, _ in hits):
                failures.append(f"no FAQ row among the top {TOP} hits for {QUESTION!r}")
    except Exception as exc:  # noqa: BLE001 - report any query/aggregate error as a failed check
        failures.append(f"query failed: {type(exc).__name__}: {exc}")
    finally:
        client.close()
    return failures


def check_redis(redis_url: str) -> list[str]:
    """Return one reason per failed Redis check (an empty list means healthy).

    Writes and reads back one short-lived key, which also counts as the activity that keeps a
    free hosted database from being removed, then confirms Redis Search is available.
    """
    if not redis_url:
        return ["missing settings: REDIS_URL"]

    try:
        client = redis.Redis.from_url(
            redis_url, decode_responses=True, socket_connect_timeout=10, socket_timeout=10
        )
    except Exception as exc:  # noqa: BLE001 - a malformed URL
        return [f"REDIS_URL is not usable: {_detail(exc)}"]

    failures: list[str] = []
    try:
        stamp = datetime.now(UTC).isoformat()
        client.set(KEEPALIVE_KEY, stamp, ex=KEEPALIVE_TTL_SECONDS)
        if client.get(KEEPALIVE_KEY) != stamp:
            failures.append("Redis accepted a write but did not return it")
        else:
            print("ok    Redis answered, and a keep-alive key was written and read back")
        try:
            client.execute_command("FT._LIST")
            print("ok    Redis Search is available")
        except redis.ResponseError:
            failures.append("Redis is up but has no Search module (the chat memory needs it)")
    except Exception as exc:  # noqa: BLE001 - any connection or command error means Redis is down
        failures.append(f"cannot use Redis: {_detail(exc)}")
    finally:
        client.close()
    return failures


def main() -> int:
    load_dotenv()
    settings = Settings()
    failures = [
        *(f"Weaviate: {reason}" for reason in check_weaviate(settings)),
        *(f"Redis: {reason}" for reason in check_redis(settings.redis_url)),
    ]
    for reason in failures:
        print(f"FAIL  {reason}")
    if not failures:
        print("healthy")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
