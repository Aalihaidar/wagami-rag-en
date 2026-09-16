"""Structured (JSON-lines) logging for Section E of the app/deployment plan.

Stdlib-only -- no new dependency -- since the need is narrow: one JSON object per log line
(timestamp, level, logger, message, plus whatever `extra=` fields a call site adds), so any
log aggregator the eventual host provides can parse it without custom rules. Governs the
app's own loggers (`app.*`) via the root logger; uvicorn/gunicorn's own access/error logs keep
their own handlers and are untouched here.
"""

import json
import logging
from typing import Any

_RESERVED_RECORD_KEYS = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
    "message",
    "asctime",
}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extras = {k: v for k, v in record.__dict__.items() if k not in _RESERVED_RECORD_KEYS}
        if extras:
            payload.update(extras)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(log_level: str) -> None:
    """Call once at import time (app/main.py) -- swaps the root logger's handler for a single
    JSON-lines stream handler at the configured level."""
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(log_level.upper())
