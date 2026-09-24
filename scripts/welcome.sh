#!/usr/bin/env bash
# Runs on every container start via postStartCommand.
# Activates the venv if present and prints what to run next.

cd /workspace || exit 0

if [ -f /workspace/.venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source /workspace/.venv/bin/activate
fi

echo ""
echo "wagami-rag-en dev container"
echo "=============================="
echo "Dev server: http://localhost:8000  (starts with the container, reloads when app/ changes)"
echo "  logs:    tail -f /tmp/dev-server.log"
echo "  restart: pkill -f scripts/dev_server.py     # dev-start.sh brings it back in ~10s"
echo ""

if [ ! -f /workspace/.venv/bin/python ]; then
  echo "👋 First run — no .venv yet."
  echo ""
  echo "  1) uv sync                 # installs deps from pyproject.toml, creates uv.lock"
  echo "  2) cp .env.example .env    # fill in WEAVIATE_URL / WEAVIATE_API_KEY / EMBEDDING_API_KEY"
  echo "  3) uv run python scripts/load_knowledge_base.py     # create schema + import data/knowledge_base.json"
  echo ""
else
  echo "✅ Environment ready."
  echo ""
  echo "  uv run python scripts/load_knowledge_base.py                    # (re)create schema + import"
  echo "  uv run python scripts/delete_knowledge_base.py --keep-schema    # empty the collection, keep schema"
  echo "  uv run ruff check --fix .                                       # lint"
  echo ""
fi
