#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Development server: hot reload, reachable from the host through the devcontainer's ports.

Started by scripts/dev-start.sh when the dev container starts; run it by hand only after stopping
that one (both want the same port).

* Reload: edits under app/ restart the server on their own (Python, plus CSS/JS so the
  cache-busting `?v=` on the static files changes). Tests, notebooks and scripts are not watched.
* Sockets: docker-compose publishes the port through Docker, which connects to the container's
  network address, so a loopback-only server is unreachable from the host browser (it shows
  ERR_EMPTY_RESPONSE). VS Code's own port forward can land on the IPv6 loopback instead. uvicorn's
  --host takes a single address and a `::` socket is IPv6-only under asyncio, so this opens one
  socket on 0.0.0.0 and one on ::1 and hands both to uvicorn's reloader.

A start that fails (Weaviate unreachable, a syntax error) leaves the reloader waiting: fix the
cause and save a file under app/ and it tries again.
"""

import logging
import os
import socket

import uvicorn
from uvicorn.supervisors import ChangeReload

logger = logging.getLogger("uvicorn.error")

PORT = int(os.environ.get("PORT", "8000"))
WATCHED_DIRS = ["app"]
WATCHED_PATTERNS = ["*.py", "*.css", "*.js"]


def listening_socket(family: int, address: str) -> socket.socket:
    """A bound, listening socket on `address` (IPv6 sockets are IPv6-only, so 0.0.0.0 and ::1
    can be held side by side)."""
    sock = socket.socket(family, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    if family == socket.AF_INET6:
        sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
    sock.bind((address, PORT))
    sock.listen(2048)
    sock.set_inheritable(True)
    return sock


def open_sockets() -> list[socket.socket]:
    sockets = [listening_socket(socket.AF_INET, "0.0.0.0")]
    try:
        sockets.append(listening_socket(socket.AF_INET6, "::1"))
    except OSError as exc:  # a host with IPv6 switched off: IPv4 alone still works
        logger.warning("Not listening on [::1]:%d (%s)", PORT, exc)
    return sockets


def main() -> None:
    config = uvicorn.Config(
        "app.main:app",
        reload=True,
        reload_dirs=WATCHED_DIRS,
        reload_includes=WATCHED_PATTERNS,
        log_level=os.environ.get("LOG_LEVEL", "info").lower(),
    )
    server = uvicorn.Server(config)
    ChangeReload(config, target=server.run, sockets=open_sockets()).run()


if __name__ == "__main__":
    main()
