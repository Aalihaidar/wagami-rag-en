# wagami-rag-en

An agentic **Retrieval-Augmented Generation** chatbot for a restaurant menu.
Guests ask about dishes, prices, per-serving nutrition and allergens, and get
answers grounded entirely in a curated knowledge base rather than a language
model's own knowledge. English-only. The repo holds the FastAPI service, the
LangGraph agent, the guest-facing chat page, a Telegram bot that talks to the same
agent, and the tooling that loads the knowledge base into Weaviate Cloud.

![CI](https://github.com/Aalihaidar/wagami-rag-en/actions/workflows/ci.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)

## Try it

This is a demo restaurant, not a real one, running on the free tiers of every service it uses.

| | Link |
| --- | --- |
| **Website** (chat page) | <https://wagami-rag-en-staging.onrender.com> |
| **Telegram bot** | [@wagami_restaurant_bot](https://t.me/wagami_restaurant_bot) |
| **Telegram Mini App** (the website, opened inside Telegram) | <https://t.me/wagami_restaurant_bot/ask> |

All three are the staging deployment, built from `develop`. Render's free plan puts the
service to sleep when it is idle, so the first request after a quiet spell can take a while
while it wakes up.

## What it does

A guest asks something like *"which vegan ramen has the fewest calories?"* or
*"does the katsu curry contain gluten?"*. The agent turns the question into a
hybrid (vector + keyword) search over a Weaviate Cloud collection, reranks the
hits, and an LLM composes the answer from **only those rows**. Every dish the answer
contains is shown as a card with its photo, and only those dishes — never a card left
over from an earlier answer. When the knowledge base doesn't
hold the answer, the agent says so rather than inventing one.

Not every message needs a search. One LLM call reads the message and decides how
it is handled:

| The guest says | Handled as | LLM calls |
| --- | --- | --- |
| "hello" | fixed greeting | 1 |
| something unrelated to the restaurant | fixed, polite redirect | 1 |
| "what's on the menu?", "show me the drinks", "sides" | the menu's groups → categories → items, straight from the corpus: groups and categories as picture cards (the name and a dedicated picture for it), items each described with a card and its own dish photo — no search | 1 |
| a dish, ingredient, price, allergy, or a filtered question ("vegan starters under £6") | hybrid search → rerank → grounded answer | 2 |
| hours, bookings, delivery, payments | the same search path over the FAQ rows | 2 |

A "browse" that states a diet, allergy, price, calorie or protein requirement is
always turned into a real search in code, so allergen exclusion is never skipped.

The same agent answers on two surfaces: the **web chat page** and the **Telegram bot**
(see [Telegram bot](#telegram-bot)). Both run the same chat turn, under the same token-budget,
conversation-length and concurrency caps.

## How a message flows

```mermaid
flowchart TD
    M["Guest message"] --> U["Understand (one LLM call)"]
    U -->|greeting| G["Fixed greeting"]
    U -->|off topic| O["Fixed redirect"]
    U -->|menu browse| B["List groups, categories or items<br/>from the corpus catalog"]
    U -->|menu / faq| R["Hybrid search + rerank"]
    R --> Q["Ground: build the context and its notes"]
    Q --> A["Generate the answer (streamed)"]
    G --> S["Reply"]
    O --> S
    B --> S
    A --> S
```

## Architecture

| Layer | Component |
| --- | --- |
| API & UI | FastAPI service: `POST /chat`, streaming `POST /chat/stream` (SSE), session endpoints, `/healthz`, and a server-rendered chat page at `/` (plain JS + CSS, no build step) |
| Telegram | `POST /telegram/webhook` → the same chat turn as `/chat`, sent back through the Bot API; menu lists and dishes as inline buttons and photos; the chat page itself as a Mini App |
| Agent | LangGraph state graph: *query → respond \| retrieve → ground → answer* |
| Retrieval | Weaviate Cloud hybrid search over one `KnowledgeBase` collection, then Cohere rerank |
| Embeddings | Cohere `embed-english-v3.0` via Weaviate's `text2vec-cohere` module — only the composed `embedding_text` field is vectorized; everything else is structured metadata for filtering and display |
| LLM | Groq (OpenAI-compatible API); the query-understanding and answer-generation models are set separately |
| Memory | Redis Stack: per-session conversation history (LangGraph checkpointer), rate-limit counters, and token-budget counters |
| Prompts | Every system prompt lives in [`app/agent/prompts.py`](app/agent/prompts.py); the app and the notebooks import the same objects |

## Safety, limits and cost control

- **Grounded only.** The generation prompt answers from the retrieved rows and
  declines otherwise; a scope and prompt-injection guard is part of every call,
  and an output-side check blocks replies that leak the instructions.
- **Rate limit:** 10 chat requests a minute per client IP (Redis-backed); on Telegram, 10
  messages a minute per chat, since every update arrives from Telegram's own servers.
- **Input cap:** messages over 500 characters are rejected before any LLM call.
- **Token budgets:** a daily and a monthly cap, and a maximum number of turns per
  conversation; when one is reached the guest gets a polite "at capacity" reply.
- **Kill switch:** `CHAT_ENABLED=false` takes `/chat` offline (503) on the next restart.
- **Privacy:** access logs are structured and never include the guest's message.
- **Hardening:** security headers on every response (CSP, `X-Frame-Options`,
  `nosniff`, no-referrer, HSTS in production); `/docs` and `/redoc` are off in production.
  The page refuses to be framed, except that with `TELEGRAM_WEB_APP_URL` set only Telegram's
  own domains may frame it (for the Mini App).

## API

| Endpoint | Purpose |
| --- | --- |
| `GET /` | the guest chat page |
| `POST /session` | start a conversation → `{"session_id": ...}` |
| `POST /chat` | `{"session_id", "message", "browse"?}` → `{"session_id", "answer", "cited_items", "choices"}` |
| `POST /chat/stream` | same request; server-sent events `delta` (text as it is written), `done` (final answer + cited items), `error` |
| `DELETE /session/{session_id}` | end a conversation and drop its history |
| `GET /healthz` | liveness probe |
| `POST /telegram/webhook` | where Telegram posts the bot's updates; `404` while the bot is off, `403` without the secret Telegram echoes back |

Each entry of `cited_items` has `id`, `slug`, `name`, `description`, `ingredients`,
`price_gbp` and `image`. `choices` is `null` except for a reply that lists menu groups or
categories, where it holds `intro`, `outro` and `cards` (`name`, `image`, `group`, `category`) so
the page can show a picture card per name between the two sentences instead of a bullet list
(`answer` still has the whole text, list included). `cited_items` belongs to that answer alone:
the dishes it names (or a listing shows), and empty when it names none.

Every card on the chat page can be clicked instead of typing (the magnifier on its picture zooms
it instead), and the reply simply appears -- the question the click sends is not shown as a
guest message, though it is saved to the conversation history. A group's card lists its
categories and a category's card lists its items: the page sends a readable `message` plus `"browse": {"group", "category"}` (from the card, `category` is
`null` for a group), and the server answers that browse from the menu catalog with no model call.
A dish's card sends `Tell me about <name>` as an ordinary message.

## Menu images

`IMAGE_BASE_URL` points at a `menu/` folder (`data/images/menu/` locally, served at
`/images/menu`) laid out like the menu itself, with folder names lower-cased and anything that
isn't a letter or digit collapsed to one hyphen (`app/agent/choice_images.py`):

- a dish photo at `<group>/<category>/<image>`, where `<image>` is the row's own `image` filename
  and the folders come from its `category_path` — or straight in `<group>/` when the group's only
  category has the group's own name (`extras`, `lunch time`, `desserts + sweet treats`);
- a `cover.png` in every group folder, shown on that group's card in the menu overview
  (`group_image_filename()`), and in every category folder, shown on that category's card in its
  group's list (`category_image_filename()`). A category's folder sits inside its group's, so
  `drinks` (a group) and `kids/drinks` (a category), or `kids/ramen` and `the-main-event/ramen`,
  each have their own cover.

A picture that is missing is not an error: the card just shows its name.
Run `uv run python scripts/list_choice_images.py` for every path the currently loaded
`data/knowledge_base.json` needs, and which of them are missing from `data/images/menu/` — a
hand-written list drifts out of sync the moment the corpus changes. It also flags a category that, with today's menu shape, a guest can never actually
browse to (a group with only that one category goes straight to its items, no category card;
see the script's own output for which ones and why) — those are safe to skip.

## Knowledge base

`data/knowledge_base.json` is a flat array of 197 rows: 162 `menu_item` rows (one
per dish) and 35 `faq` rows, told apart by `item_type`. All 30 keys are present on
every row (`null` where a field doesn't apply — FAQ rows null out the menu-specific
fields):

| Group | Fields |
| --- | --- |
| Identity | `id`, `item_type`, `name`, `slug`, `description`, `ingredients`, `category` / `category_slug` / `category_path` |
| Retrieval | `embedding_text` — the only vectorized field |
| Price | `price_gbp` |
| Nutrition (per serving) | `kcal`, `protein_g`, `fat_g`, `carbs_g`, `sugars_g`, `sat_fat_g`, `sodium_g`, `salt_g`, `fibre_g` |
| Allergens & diet | `allergens_contains`, `allergens_may_contain`, `dietary_tags`, `is_gluten_free_listed` |
| Serving | `portion_value` / `portion_unit`, `servings`, `abv_percent` |
| Media & meta | `image`, `last_updated` |

The menu is organised as **group → category → items** (for example *drinks →
cocktails → the dishes*), taken from `category_path` and `category`. At startup the
service profiles the corpus into an in-memory catalog that drives menu browsing
and is also rendered into the query-understanding prompt.

The corpus and the dish photos are a demo dataset and are **not included in this
repository** — provide your own `data/knowledge_base.json` in the same shape, and
host the images wherever `IMAGE_BASE_URL` points. The cards for menu groups and categories show the
`cover.png` in that group's or category's own image folder — see **Menu images** below for the
layout.

## Getting started

Requires **Python 3.14** and [uv](https://docs.astral.sh/uv/). The supported
setup is the VS Code dev container (Docker Compose: the app, Redis Stack, debugpy).

```bash
cp .env.example .env     # then fill in the values below
```

Open the folder in the dev container. **The server starts by itself with the
container** on <http://localhost:8000> and **reloads on its own** whenever you save
a `.py`, `.css` or `.js` file under `app/` — there is nothing to start or restart.

```bash
tail -f /tmp/dev-server.log            # server logs (inside the container)
pkill -f scripts/dev_server.py         # restart it by hand; it comes back in ~10 s
```

If a start fails (Weaviate unreachable, a syntax error), the container stays up and
the server tries again the next time a file under `app/` is saved.

Without the dev container, run `uv sync`, start Redis Stack on `:6379`, and use
`uv run python scripts/dev_server.py`.

### Configuration

All configuration is via environment variables / `.env`:

| Variable | Purpose |
| --- | --- |
| `APP_ENV`, `PORT`, `LOG_LEVEL` | `development` or `production` (production turns off `/docs` and adds HSTS), listen port, log level |
| `CHAT_ENABLED` | set `false` to take `/chat` offline |
| `WEAVIATE_URL` | Weaviate Cloud cluster |
| `WEAVIATE_API_KEY` | admin key — used only by the scripts that create, empty and load the collection |
| `WEAVIATE_READ_API_KEY` | read-only key — used by the running service |
| `EMBEDDING_PROVIDER`, `EMBEDDING_API_KEY` | `cohere` (`embed-english-v3.0`) or `openai`; with Cohere the same key also drives rerank |
| `GROQ_API_KEY` | LLM access |
| `UNDERSTAND_MODEL`, `GENERATION_MODEL` | models for query understanding and answer generation |
| `REDIS_URL` | Redis **Stack** (not plain Redis): session memory, rate limits, token budgets |
| `IMAGE_BASE_URL` | public base URL of the `menu/` image folder (see **Menu images**) |
| `TELEGRAM_BOT_TOKEN`, `TELEGRAM_WEBHOOK_SECRET` | the Telegram bot's token (from @BotFather) and the random secret passed to `setWebhook`; leave the token empty to keep the bot off |
| `TELEGRAM_WEB_APP_URL` | the chat page's public https URL: adds an "Open Wagami assistant" button to `/start` and lets Telegram frame the page (the Mini App) |
| `DAILY_TOKEN_LIMIT`, `MONTHLY_TOKEN_LIMIT`, `MAX_CONVERSATION_TURNS` | cost-control limits |

### Telegram bot

The bot is optional and off until `TELEGRAM_BOT_TOKEN` is set; the web chat never depends on it.

What a guest gets in the bot chat:

- Any question is answered by the same agent, with its own conversation per Telegram chat.
- A list of menu groups or categories arrives as **one picture** (each cover with its name under
  it) with a button per name; tapping one browses into it with no model call, like a card click on
  the page (`app/telegram_grid.py`).
- One dish comes with its photo, price, diet tags, calories and allergens; a list of dishes comes
  as photos with a "Tell me about …" button each.
- `/start` sends the welcome, with an **Open Wagami assistant** button when `TELEGRAM_WEB_APP_URL`
  is set. `/reset` forgets the conversation, deletes the chat's latest messages (up to 1000, and
  only those Telegram still allows: nothing older than 48 hours) and sends the welcome again.
- The **Mini App** opens the web chat page itself inside Telegram, so its cards, grids and
  single-dish view look exactly as on the website. It keeps its own conversation, separate from
  the bot chat's.

Only private chats are answered. Redelivered updates are ignored, and the bot token is never
logged.

Setting it up:

1. In [@BotFather](https://t.me/BotFather): `/newbot`, then set a description, about text, and
   `/setcommands` (`start`, `reset`).
2. Set `TELEGRAM_BOT_TOKEN`, `TELEGRAM_WEBHOOK_SECRET` (any long random string) and, for the
   Mini App, `TELEGRAM_WEB_APP_URL` on the service. A bot has one webhook, so staging and
   production each need their own bot and token.
3. Register the webhook once the service is deployed:

   ```bash
   curl "https://api.telegram.org/bot<TOKEN>/setWebhook" \
     -d url=https://<your-service>/telegram/webhook \
     -d secret_token=<TELEGRAM_WEBHOOK_SECRET> \
     -d 'allowed_updates=["message","callback_query"]'
   ```

4. Optional, for the Mini App: `/newapp` in BotFather (its short name gives a direct link,
   `t.me/<bot>/<short name>`, which also works as a QR code), and the menu button:

   ```bash
   curl "https://api.telegram.org/bot<TOKEN>/setChatMenuButton" \
     -H 'Content-Type: application/json' \
     -d '{"menu_button":{"type":"web_app","text":"Open menu","web_app":{"url":"https://<your-service>"}}}'
   ```

Telegram cannot open a Mini App without a tap: the direct link, the menu button, the `/start`
button and the bot profile's launch button are the ways in.

### Loading the knowledge base

```bash
uv run python scripts/load_knowledge_base.py                    # create the collection + import
uv run python scripts/delete_knowledge_base.py --keep-schema     # empty it, keep the schema
uv run python scripts/delete_knowledge_base.py                   # drop the collection entirely
```

Both scripts write to live Weaviate Cloud. The schema lives in
`load_knowledge_base.py` and is imported by the delete script so the two can't drift.

## Development

```bash
uv run ruff check .                        # lint
uv run ruff format .                       # format
uv run mypy scripts app                    # type-check
uv run pytest --cov=app --cov-report=term  # tests — fake LLM and Weaviate, no live calls
```

- `scripts/latency_check.py` sends a few questions to a running server and reports
  time to first word and total time per turn, optionally with the server's own
  per-stage breakdown.
- `scripts/adversarial_check.py` sends jailbreak, prompt-injection and off-topic
  probes to a running app and flags any reply that complies or leaks the prompt.
- `notebooks/` — `01` retrieval checks, `02` generation checks, `03` an evaluation
  harness (gold questions, deterministic safety checks, an LLM judge, latency
  percentiles). `02` and `03` import the app's own pipeline code (prompts,
  understanding, retrieval, prompt assembly, cards), so they exercise what actually ships. Re-run `03` after any prompt change.

Pre-commit hooks, GitHub Actions CI (lint · type-check · test · dependency audit ·
Dockerfile scan), and Dependabot are configured. See
[.github/CONTRIBUTING.md](.github/CONTRIBUTING.md) for the branching model,
branch-protection rulesets, and release flow.

## Deployment

```text
feature/*  ─PR─▶  develop  ─PR─▶  main
                     │              │
                  staging       production
```

Merging to `develop` or `main` (both PR-only) runs CI; when it is green,
[`docker.yml`](.github/workflows/docker.yml) builds the production image
([`docker/Dockerfile.prod`](docker/Dockerfile.prod): multi-stage `uv` build,
non-root, gunicorn with uvicorn workers), scans it, pushes it to GHCR
(`staging` from `develop`, `latest` from `main`) and calls the matching Render
deploy hook. [`render.yaml`](render.yaml) is the Blueprint for the two Render
services. Staging is live at <https://wagami-rag-en-staging.onrender.com>. The Telegram webhook
and the bot's variables are set per service (see [Telegram bot](#telegram-bot)).

## Project layout

```text
app/
  main.py            FastAPI app: routes, middleware, chat turns
  config.py          settings (env / .env)
  cost_control.py    token budgets and conversation limits
  retrieval.py       Weaviate hybrid search + Cohere rerank
  telegram_bot.py    the Telegram bot: updates, replies, buttons, /reset
  telegram_grid.py   the picture of menu covers with names sent for a list
  agent/
    graph.py         the LangGraph state graph
    understanding.py query understanding: routing, filters, safety nets
    catalog.py       corpus profile: groups, categories, items
    browse.py        replies that need no search
    generation.py    grounded answer prompt and output checks
    prompts.py       every system prompt
    llm.py, memory.py, checkpointer.py
  templates/, static/  the chat page (static/js/telegram.js: Mini App support)
scripts/             knowledge-base load/delete, dev server, latency and adversarial checks
notebooks/           retrieval, generation and evaluation notebooks
tests/               pytest suite
docker/              dev and production images
```

## License

**Code** — [MIT](LICENSE) © 2026 Ali Haidar. Covers everything in this
repository: the scripts, schema, configuration, and tooling.

**Data** — the `data/` corpus is a separate demo dataset. It is not included
in this repository and is **not** licensed here; supply your own
`data/knowledge_base.json`.
