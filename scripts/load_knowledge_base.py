#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Creates the KnowledgeBase collection (free-tier: 1 collection limit) and
batch-imports knowledge_base.json.

English-only menu + FAQ corpus -- no `language` / `group_id` pairing fields.
The vectorizer is Cohere's embed-english-v3.0, its English-specialized
embedding model. Every row carries an `item_type` of "menu_item" or "faq"
on one shared schema.

Uses the current (weaviate-client >=4.16) `vector_config` / `Configure.Vectors`
API, not the deprecated `vectorizer_config` / `Configure.Vectorizer` path --
the client emits a deprecation warning if the old kwarg is used at all, and
imports written today should target where the API already is, not where it
used to be. Same reasoning applies to `Auth.api_key(...)` over the older
`weaviate.auth.AuthApiKey(...)`.

Setup: `uv sync` (weaviate-client is a project dependency).
Env vars are read from the process environment, with `.env` loaded first via
python-dotenv. Same names as the project's .env / .env.example:
  WEAVIATE_URL, WEAVIATE_API_KEY, EMBEDDING_PROVIDER, EMBEDDING_API_KEY
EMBEDDING_PROVIDER selects the Weaviate vectorizer module: "cohere" (default,
embed-english-v3.0) or "openai". EMBEDDING_API_KEY is the matching Cohere or
OpenAI key.
"""

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import weaviate
import weaviate.classes.config as wvc
from dotenv import load_dotenv
from weaviate import WeaviateClient
from weaviate.classes.init import Auth

# Resolve relative to this script, not the caller's CWD.
# Lives in data/ (repo root), one level up from this script's scripts/ folder.
DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "knowledge_base.json"

EMBEDDING_PROVIDER = os.environ.get("EMBEDDING_PROVIDER", "cohere").lower()

# Only this field is ever embedded -- every other property is retrieval
# metadata (filtering / rendering), never vector input. Declaring it in one
# place and reusing it for both `source_properties` (belt) and each other
# property's `skip_vectorization=True` (suspenders) means a stray typo can't
# silently vectorize the wrong field: without an explicit `source_properties`,
# Weaviate vectorizes *all* text properties, so one undeclared property would
# poison every vector in the collection.
VECTORIZED_PROPERTY = "embedding_text"


def create_collection(client: WeaviateClient) -> None:
    if client.collections.exists("KnowledgeBase"):
        print("Collection 'KnowledgeBase' already exists, skipping create.")
        return

    if EMBEDDING_PROVIDER == "openai":
        vector_config = wvc.Configure.Vectors.text2vec_openai(
            model="text-embedding-3-small",
            source_properties=[VECTORIZED_PROPERTY],
            vectorize_collection_name=False,  # don't fold "KnowledgeBase" into every vector
        )
    else:
        vector_config = wvc.Configure.Vectors.text2vec_cohere(
            model="embed-english-v3.0",
            source_properties=[VECTORIZED_PROPERTY],
            vectorize_collection_name=False,
        )

    client.collections.create(
        name="KnowledgeBase",
        vector_config=vector_config,
        properties=[
            wvc.Property(
                name="item_type",
                data_type=wvc.DataType.TEXT,
                tokenization=wvc.Tokenization.FIELD,
                skip_vectorization=True,
            ),
            wvc.Property(name=VECTORIZED_PROPERTY, data_type=wvc.DataType.TEXT),
            wvc.Property(name="name", data_type=wvc.DataType.TEXT, skip_vectorization=True),
            wvc.Property(name="slug", data_type=wvc.DataType.TEXT, skip_vectorization=True),
            wvc.Property(name="description", data_type=wvc.DataType.TEXT, skip_vectorization=True),
            wvc.Property(name="category", data_type=wvc.DataType.TEXT, skip_vectorization=True),
            wvc.Property(
                name="category_slug",
                data_type=wvc.DataType.TEXT,
                tokenization=wvc.Tokenization.FIELD,
                skip_vectorization=True,
            ),
            wvc.Property(
                name="category_path",
                data_type=wvc.DataType.TEXT_ARRAY,
                skip_vectorization=True,
            ),
            # --- price: a single per-item GBP figure. Some rows were filled at
            # transform time with a category-median estimate rather than a real
            # scraped price; that per-row provenance is no longer tracked in the
            # schema -- all prices are now treated the same.
            wvc.Property(name="price_gbp", data_type=wvc.DataType.NUMBER, skip_vectorization=True),
            # --- nutrition: per-serving values only. Wagami's own data also
            # carries per-100g and %GDA for each of these, kept in the source
            # JSON but not promoted here -- these flat fields are for guest-
            # facing answers and filtering, not a full nutrition-label render.
            wvc.Property(name="kcal", data_type=wvc.DataType.NUMBER, skip_vectorization=True),
            wvc.Property(name="protein_g", data_type=wvc.DataType.NUMBER, skip_vectorization=True),
            wvc.Property(name="fat_g", data_type=wvc.DataType.NUMBER, skip_vectorization=True),
            wvc.Property(name="carbs_g", data_type=wvc.DataType.NUMBER, skip_vectorization=True),
            wvc.Property(name="sugars_g", data_type=wvc.DataType.NUMBER, skip_vectorization=True),
            wvc.Property(name="sat_fat_g", data_type=wvc.DataType.NUMBER, skip_vectorization=True),
            wvc.Property(name="sodium_g", data_type=wvc.DataType.NUMBER, skip_vectorization=True),
            wvc.Property(name="salt_g", data_type=wvc.DataType.NUMBER, skip_vectorization=True),
            wvc.Property(name="fibre_g", data_type=wvc.DataType.NUMBER, skip_vectorization=True),
            # --- allergens: Wagami's own contains / may-contain lists,
            # already resolved against their 14-EU-allergen master table.
            wvc.Property(
                name="allergens_contains",
                data_type=wvc.DataType.TEXT_ARRAY,
                skip_vectorization=True,
            ),
            wvc.Property(
                name="allergens_may_contain",
                data_type=wvc.DataType.TEXT_ARRAY,
                skip_vectorization=True,
            ),
            wvc.Property(
                name="dietary_tags", data_type=wvc.DataType.TEXT_ARRAY, skip_vectorization=True
            ),
            wvc.Property(
                name="is_gluten_free_listed", data_type=wvc.DataType.BOOL, skip_vectorization=True
            ),
            wvc.Property(
                name="portion_value", data_type=wvc.DataType.NUMBER, skip_vectorization=True
            ),
            wvc.Property(name="portion_unit", data_type=wvc.DataType.TEXT, skip_vectorization=True),
            wvc.Property(name="servings", data_type=wvc.DataType.TEXT, skip_vectorization=True),
            wvc.Property(
                name="abv_percent", data_type=wvc.DataType.NUMBER, skip_vectorization=True
            ),
            wvc.Property(name="image", data_type=wvc.DataType.TEXT, skip_vectorization=True),
            # --- provenance: how stale a row may be ---
            wvc.Property(name="last_updated", data_type=wvc.DataType.DATE, skip_vectorization=True),
        ],
    )
    print(f"Created collection 'KnowledgeBase' (vectorizer: {EMBEDDING_PROVIDER}).")


def load_records() -> list[dict[str, Any]]:
    with DATA_FILE.open(encoding="utf-8") as f:
        records: list[dict[str, Any]] = json.load(f)

    dupes = [uid for uid, count in Counter(r["id"] for r in records).items() if count > 1]
    if dupes:
        print(f"WARNING: {len(dupes)} duplicate id(s) in {DATA_FILE.name}: {dupes}")

    return records


def import_data(client: WeaviateClient) -> None:
    records = load_records()
    collection = client.collections.get("KnowledgeBase")

    with collection.batch.fixed_size(batch_size=100) as batch:
        for record in records:
            properties = {k: v for k, v in record.items() if k != "id"}
            batch.add_object(properties=properties, uuid=record["id"])

    failed = collection.batch.failed_objects
    print(f"Imported {len(records) - len(failed)}/{len(records)} objects.")
    if failed:
        print(f"{len(failed)} objects failed. First error: {failed[0].message}")


def main() -> None:
    load_dotenv()
    header_name = "X-OpenAI-Api-Key" if EMBEDDING_PROVIDER == "openai" else "X-Cohere-Api-Key"
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=os.environ["WEAVIATE_URL"],
        auth_credentials=Auth.api_key(os.environ["WEAVIATE_API_KEY"]),
        headers={header_name: os.environ["EMBEDDING_API_KEY"]},
    )
    try:
        create_collection(client)
        import_data(client)
    finally:
        client.close()


if __name__ == "__main__":
    main()
