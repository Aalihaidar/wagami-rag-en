#!/usr/bin/env bash
# Main process of the dev container (docker-compose.yml `command`): starts the dev server as soon
# as the container starts and keeps the container alive whatever the server does, so a broken
# start can be fixed from inside the container instead of leaving it stopped.
#
# Logs go to the container's output (`docker compose logs -f app`) and to /tmp/dev-server.log
# (`tail -f /tmp/dev-server.log` from a terminal in the container).
# Restart the server by hand:  pkill -f scripts/dev_server.py   (this script starts it again).

cd /workspace || exit 1

LOG=/tmp/dev-server.log
RETRY_SECONDS=10
: > "$LOG"

child=""
stop() {
  [ -n "$child" ] && kill "$child" 2>/dev/null
  exit 0
}
trap stop TERM INT

while true; do
  # `uv run` syncs .venv with uv.lock first, so a fresh checkout needs no separate `uv sync`.
  uv run python scripts/dev_server.py > >(tee -a "$LOG") 2>&1 &
  child=$!
  wait "$child"
  echo "dev server exited (status $?); starting it again in ${RETRY_SECONDS}s" | tee -a "$LOG"
  sleep "$RETRY_SECONDS" &
  wait $!
done
