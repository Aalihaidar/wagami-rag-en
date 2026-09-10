# wagami-rag-en

An agentic **Retrieval-Augmented Generation** backend that answers guest
questions about a restaurant menu — dish details, prices, per-serving
nutrition, and allergens — grounded entirely in a curated knowledge base
rather than a language model's own knowledge. English-only.

![CI](https://github.com/Aalihaidar/wagami-rag-en/actions/workflows/ci.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)

> The knowledge-base schema and Weaviate ingestion tooling are implemented
> today; the HTTP API and the agent are on the [roadmap](#roadmap).

## What it does

A guest asks something like *"which vegan ramen has the fewest calories?"* or
*"does the katsu curry contain gluten?"*. A LangGraph agent turns the question
into a hybrid (vector + keyword) search over a Weaviate Cloud collection,
retrieves the matching menu items, and an LLM composes the answer from **only
those rows**. When the knowledge base doesn't hold the answer, the agent says
so rather than inventing one.

## Architecture

| Layer | Component |
|---|---|
| API | FastAPI service exposing the chat endpoint |
| Agent | LangGraph state machine: *query → retrieve → ground → answer* |
| Retrieval | Weaviate Cloud hybrid search over one `KnowledgeBase` collection, one object per menu item |
| Embeddings | Cohere `embed-english-v3.0` via Weaviate's `text2vec-cohere` module — only the composed `embedding_text` field is vectorized; everything else is structured metadata for filtering and display |
| LLM | Called via API (Google Gemini by default; provider configurable) |
| Session memory | Optional Redis-backed checkpointing for multi-turn conversations |

## Knowledge base

`data/knowledge_base.json` is a flat array — one row per menu item, plus a
set of `faq` rows (`item_type` tells them apart). All 29 keys are present on
every row (`null` where a field doesn't apply — FAQ rows null out the menu-
specific fields):

| Group | Fields |
|---|---|
| Identity | `id`, `name`, `slug`, `category` / `category_slug` / `category_path`, `description` |
| Retrieval | `embedding_text` — the only vectorized field |
| Price | `price_gbp` |
| Nutrition (per serving) | `kcal`, `protein_g`, `fat_g`, `carbs_g`, `sugars_g`, `sat_fat_g`, `sodium_g`, `salt_g`, `fibre_g` |
| Allergens & diet | `allergens_contains`, `allergens_may_contain`, `dietary_tags`, `is_gluten_free_listed` |
| Serving | `portion_value` / `portion_unit`, `servings`, `abv_percent` |
| Media & meta | `image`, `last_updated` |

The corpus is a demo dataset and is **not included in this repository** —
provide your own `data/knowledge_base.json` in the same shape.

## Getting started

Requires **Python 3.14** and [uv](https://docs.astral.sh/uv/). A VS Code dev
container (Docker Compose + Redis + debugpy) is included.

```bash
uv sync                 # install dependencies
cp .env.example .env     # then fill in the values below
```

### Configuration

All configuration is via environment variables / `.env` (see `.env.example`):

| Variable | Purpose |
|---|---|
| `WEAVIATE_URL`, `WEAVIATE_API_KEY` | Weaviate Cloud cluster |
| `EMBEDDING_PROVIDER`, `EMBEDDING_API_KEY` | vectorizer module — `cohere` (`embed-english-v3.0`) or `openai` |
| `LLM_PROVIDER`, `LLM_API_KEY`, `LLM_MODEL` | answer-generation model |
| `IMAGE_BASE_URL` | public base URL prepended to each row's `image` filename |
| `REDIS_URL` | session memory (optional) |

### Loading the knowledge base

```bash
uv run python scripts/load_knowledge_base.py                    # create the collection + import
uv run python scripts/delete_knowledge_base.py --keep-schema     # empty it, keep the schema
uv run python scripts/delete_knowledge_base.py                   # drop the collection entirely
```

Both scripts write to live Weaviate Cloud. The schema lives in
`load_knowledge_base.py` and is imported by the delete script so the two can't
drift.

## Development

```bash
uv run ruff check .     # lint
uv run ruff format .    # format
uv run mypy scripts     # type-check
uv run pytest           # tests
```

Pre-commit hooks, GitHub Actions CI (lint · type-check · test · dependency
audit), and Dependabot are configured. See
[.github/CONTRIBUTING.md](.github/CONTRIBUTING.md) for the branching model,
branch-protection rulesets, and release flow.

## Roadmap

1. **Now** — knowledge-base schema and Weaviate ingestion (`scripts/`).
2. Retrieval-prototyping notebook to tune hybrid search against the corpus.
3. `app/` — FastAPI service, LangGraph agent, and retrieval tool.
4. Production — `docker/Dockerfile.prod`, image build, and deployment.

## License

**Code** — [MIT](LICENSE) © 2026 Ali Haidar. Covers everything in this
repository: the scripts, schema, configuration, and tooling.

**Data** — the `data/` corpus is a separate demo dataset. It is not included
in this repository and is **not** licensed here; supply your own
`data/knowledge_base.json`.
