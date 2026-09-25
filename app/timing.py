"""Per-stage latency accounting for one /chat turn.

`track_timings()` opens a per-turn dict in a ContextVar; `timed(stage)` adds the elapsed
milliseconds of its block to that dict under `stage`. Outside a `track_timings()` block
`timed()` is a no-op, so instrumented code (graph nodes, retrieval, the LLM client) needs no
plumbing and behaves identically in tests and scripts.

The dict is shared by reference, so it survives the hops a turn makes -- the event loop to the
threadpool (`run_in_threadpool`) and LangGraph's own node execution both copy the context but
not the dict it points at. Repeated calls to the same stage (e.g. a rerank retried after
constraint relaxation) accumulate rather than overwrite.

Stages nest on purpose: `rerank` includes `rerank_pace` (the client-side wait that keeps
Cohere under its request budget), and `understand`/`generate` include any `llm_backoff` (retry
sleeps after a 429/5xx). Subtract the inner one to get the pure network+compute time.
"""

import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_active: ContextVar[dict[str, float] | None] = ContextVar("chat_turn_timings", default=None)


@contextmanager
def track_timings() -> Iterator[dict[str, float]]:
    """Collect every `timed()` block run inside this one, keyed by stage, in milliseconds."""
    timings: dict[str, float] = {}
    token = _active.set(timings)
    try:
        yield timings
    finally:
        try:
            _active.reset(token)
        except ValueError:
            # An abandoned async generator (a client that disconnected mid-stream) is finalized
            # by the event loop in a different context than the one that set the token.
            _active.set(None)


@contextmanager
def timed(stage: str) -> Iterator[None]:
    timings = _active.get()
    if timings is None:
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        timings[stage] = timings.get(stage, 0.0) + (time.perf_counter() - start) * 1000
