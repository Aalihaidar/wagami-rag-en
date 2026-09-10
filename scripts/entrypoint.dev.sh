#!/usr/bin/env bash

set -euo pipefail

if [ -d /root/.ssh ]; then
  chmod 700 /root/.ssh || true
  find /root/.ssh -type f -name "id_*" ! -name "*.pub" -exec chmod 600 {} \; 2>/dev/null || true
  find /root/.ssh -type f -name "*.pub" -exec chmod 644 {} \; 2>/dev/null || true
  [ -f /root/.ssh/config ] && chmod 600 /root/.ssh/config || true
  [ -f /root/.ssh/known_hosts ] && chmod 600 /root/.ssh/known_hosts || true
fi

exec "$@"
