"""Redis-backed LangGraph checkpointer -- per-session conversation memory (Section 4 of
docs/APP_AND_DEPLOYMENT_PLAN.md).

Requires Redis Stack (RediSearch), not plain Redis -- see docker-compose.yml's `redis`
service comment. `langgraph-checkpoint-redis` (via redisvl) issues `FT.CREATE` to build its
checkpoint indices at `setup()` time, which plain `redis:*-alpine` doesn't support.
"""

from collections.abc import Iterator
from contextlib import contextmanager

from langgraph.checkpoint.redis import RedisSaver


@contextmanager
def build_checkpointer(redis_url: str) -> Iterator[RedisSaver]:
    """Yield a ready-to-use RedisSaver, with its indices already created.

    Intended for FastAPI's `lifespan`: enter this context once at startup, pass the yielded
    saver into app.agent.graph.build_graph(), and let the context manager close the Redis
    connection on shutdown.
    """
    with RedisSaver.from_conn_string(redis_url) as saver:
        saver.setup()
        yield saver
