#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Deletes the KnowledgeBase collection's data on Weaviate Cloud. Companion to
load_knowledge_base.py -- same env vars, same directory (scripts/).

Setup: `uv sync` (weaviate-client is a project dependency). `.env` is loaded
first via python-dotenv.
Env vars (same as load_knowledge_base.py):
  WEAVIATE_URL, WEAVIATE_API_KEY
  EMBEDDING_PROVIDER, EMBEDDING_API_KEY  -- only needed with --keep-schema,
    since that path recreates the collection (and therefore its vectorizer
    config) via load_knowledge_base.create_collection

Two modes:
  (default)       Drop the whole collection -- schema AND data. Frees up
                   the free-tier's 1-collection slot. Re-run
                   load_knowledge_base.py afterwards to recreate it.
  --keep-schema   Drop the collection, then immediately recreate the same
                   (empty) schema/vectorizer config, so you can re-import
                   fresh data without redefining properties.

Usage:
  python delete_knowledge_base.py                # full delete, asks to confirm
  python delete_knowledge_base.py --keep-schema   # empty it, keep schema
  python delete_knowledge_base.py --yes           # skip the confirmation prompt
"""

import argparse
import os
import sys
from pathlib import Path

import weaviate
from dotenv import load_dotenv
from weaviate import WeaviateClient
from weaviate.classes.init import Auth

COLLECTION_NAME = "KnowledgeBase"


def get_object_count(client: WeaviateClient) -> int | None:
    """Best-effort count of existing objects, for the confirmation prompt."""
    if not client.collections.exists(COLLECTION_NAME):
        return None
    collection = client.collections.get(COLLECTION_NAME)
    result = collection.aggregate.over_all(total_count=True)
    return result.total_count


def confirm(count: int, keep_schema: bool) -> bool:
    action = "empty (keep schema)" if keep_schema else "permanently delete"
    print(f"About to {action} collection '{COLLECTION_NAME}' ({count} objects).")
    reply = input("Type 'yes' to continue: ").strip().lower()
    return reply == "yes"


def delete_collection(client: WeaviateClient) -> None:
    client.collections.delete(COLLECTION_NAME)
    print(f"Deleted collection '{COLLECTION_NAME}'.")


def recreate_empty_schema(client: WeaviateClient) -> None:
    # Reuse the exact schema/vectorizer definition from load_knowledge_base.py
    # so the two scripts can never drift out of sync with each other.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from load_knowledge_base import create_collection

    create_collection(client)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--keep-schema",
        action="store_true",
        help="Recreate the empty collection schema after deleting (default: leave it deleted).",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the interactive confirmation prompt (e.g. for CI use).",
    )
    args = parser.parse_args()
    load_dotenv()

    header_name = (
        "X-OpenAI-Api-Key"
        if os.environ.get("EMBEDDING_PROVIDER", "cohere").lower() == "openai"
        else "X-Cohere-Api-Key"
    )
    headers = {}
    embedding_api_key = os.environ.get("EMBEDDING_API_KEY")
    if embedding_api_key:
        headers[header_name] = embedding_api_key
    elif args.keep_schema:
        print(
            "WARNING: --keep-schema recreates the vectorizer config but "
            "EMBEDDING_API_KEY is not set. Import will fail later without it.",
            file=sys.stderr,
        )

    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=os.environ["WEAVIATE_URL"],
        auth_credentials=Auth.api_key(os.environ["WEAVIATE_API_KEY"]),
        headers=headers,
    )
    try:
        count = get_object_count(client)
        if count is None:
            print(f"Collection '{COLLECTION_NAME}' does not exist -- nothing to do.")
            return
        if not args.yes and not confirm(count, args.keep_schema):
            print("Aborted -- no changes made.")
            return

        delete_collection(client)
        if args.keep_schema:
            recreate_empty_schema(client)
    finally:
        client.close()


if __name__ == "__main__":
    main()
