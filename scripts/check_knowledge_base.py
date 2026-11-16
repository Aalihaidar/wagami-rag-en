#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Weekly health check for the Weaviate Cloud knowledge base (run by .github/workflows/kb-health.yml).

Confirms, with the same read-only key and connection code the app uses, that:
  1. the cluster is reachable and ready (a deleted or paused cluster fails here),
  2. the `KnowledgeBase` collection exists and still holds the corpus, and
  3. one fixed question ("What are your opening hours?") comes back through a real hybrid
     search with an FAQ row among the top hits -- which also proves the embedding key still
     vectorizes queries.

No LLM is called: this only touches Weaviate (and Cohere, through Weaviate's vectorizer).
Exits 0 when healthy, 1 with a one-line reason per failed check.

Env vars (same names as the app): WEAVIATE_URL, WEAVIATE_READ_API_KEY, EMBEDDING_API_KEY.
Locally: `uv run python scripts/check_knowledge_base.py` (`.env` is loaded first).
"""

import sys

from dotenv import load_dotenv

from app.config import Settings
from app.retrieval import connect

COLLECTION = "KnowledgeBase"
# The corpus holds 197 rows; this floor only has to catch an emptied or truncated collection,
# so it stays well below the real count and needs no edit when a few rows are added or removed.
MIN_OBJECTS = 150
QUESTION = "What are your opening hours?"
TOP = 3


def main() -> int:
    load_dotenv()
    settings = Settings()
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
        print(f"FAIL  missing settings: {', '.join(missing)}")
        return 1

    failures: list[str] = []
    try:
        client = connect(settings)
    except Exception as exc:  # noqa: BLE001 - any connect error means the cluster is not usable
        print(f"FAIL  cannot connect to the cluster: {type(exc).__name__}: {exc}")
        return 1

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

    for reason in failures:
        print(f"FAIL  {reason}")
    if not failures:
        print("healthy")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
