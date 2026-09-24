import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph
from redis import Redis
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool
from starlette.types import Receive, Scope, Send

from app.agent.cards import Choices as GeneratedChoices
from app.agent.cards import CitedItem as GeneratedCitedItem
from app.agent.checkpointer import build_checkpointer
from app.agent.graph import build_graph
from app.agent.llm import AllKeysRateLimitedError, GroqClient, load_groq_key_pool
from app.agent.understanding import load_category_index
from app.config import get_settings
from app.cost_control import (
    CAPACITY_REPLY,
    CONVERSATION_LIMIT_REPLY,
    is_over_spend_limit,
    record_token_usage,
)
from app.errors import (
    INTERNAL_ERROR_MESSAGE,
    OUTBOUND_ERROR_REPLY,
    TRANSIENT_OUTBOUND_ERRORS,
    register_exception_handlers,
)
from app.logging_config import configure_logging
from app.retrieval import RetrievalTool, connect
from app.schemas import (
    ChatRequest,
    ChatResponse,
    ChoiceCard,
    Choices,
    CitedItem,
    SessionResponse,
)
from app.timing import timed, track_timings

settings = get_settings()
configure_logging(settings.log_level)
logger = logging.getLogger("app.main")
access_logger = logging.getLogger("app.access")

# Resolved relative to this file, not the caller's cwd -- same reasoning as
# scripts/load_knowledge_base.py's own DATA_FILE.
IMAGES_DIR = Path(__file__).resolve().parent.parent / "data" / "images"
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
STATIC_DIR = Path(__file__).resolve().parent / "static"
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Cache-busts /static/* on every process restart (a Render redeploy gets a fresh process) --
# simplest possible fix for browsers caching a stale chat.js/chat.css after a deploy, with no
# build step to compute a real content hash from.
STATIC_VERSION = str(int(time.time()))

# When IMAGE_BASE_URL points at external storage (e.g. a Cloudflare R2 bucket) rather than
# this service's own /images route, the CSP's img-src must allow that origin explicitly --
# derived from the setting itself so the two can never drift out of sync.
_image_base_origin = urlparse(settings.image_base_url)
CSP_IMG_SRC = (
    f"img-src 'self' data: {_image_base_origin.scheme}://{_image_base_origin.netloc}"
    if _image_base_origin.scheme in ("http", "https")
    else "img-src 'self' data:"
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Connect Weaviate/Redis once at startup and build the compiled agent graph, stored on
    app.state -- never reconnected per request (Section A of the app/deployment plan).

    Skips real connections entirely when Weaviate isn't configured (empty WEAVIATE_URL /
    WEAVIATE_READ_API_KEY -- e.g. CI, or a checkout with no .env secrets yet): /healthz still
    works, and /chat + /session 503 instead of the app failing to start.
    """
    app.state.graph = None
    app.state.checkpointer = None
    app.state.redis_client = None
    if not (settings.weaviate_url and settings.weaviate_read_api_key):
        yield
        return

    weaviate_client = connect(settings)
    redis_client = Redis.from_url(settings.redis_url)
    retrieval_tool: RetrievalTool | None = None
    groq_client: GroqClient | None = None
    try:
        kb = weaviate_client.collections.get("KnowledgeBase")
        retrieval_tool = RetrievalTool(kb=kb, cohere_api_key=settings.embedding_api_key)
        category_index = load_category_index(kb)
        groq_key_pool = load_groq_key_pool(settings.groq_api_key)
        groq_client = (
            GroqClient(settings.groq_api_key, key_pool=groq_key_pool) if groq_key_pool else None
        )
        app.state.redis_client = redis_client
        with build_checkpointer(settings.redis_url) as checkpointer:
            app.state.checkpointer = checkpointer
            app.state.graph = build_graph(
                retrieval_tool=retrieval_tool,
                category_index=category_index,
                groq_client=groq_client,
                understand_model=settings.understand_model,
                generation_model=settings.generation_model,
                checkpointer=checkpointer,
            )
            yield
    finally:
        # The keep-alive HTTP pools behind Cohere rerank and Groq.
        if retrieval_tool is not None:
            retrieval_tool.close()
        if groq_client is not None:
            groq_client.close()
        weaviate_client.close()
        redis_client.close()


app = FastAPI(
    title="Restaurant Chatbot API",
    docs_url=None if settings.app_env == "production" else "/docs",
    redoc_url=None if settings.app_env == "production" else "/redoc",
    lifespan=lifespan,
)

# Section E: one {"error": "..."} JSON shape for every failure -- HTTPException, request
# validation, and anything unhandled -- instead of an ad hoc shape per route.
register_exception_handlers(app)


@app.middleware("http")
async def security_headers(request: Request, call_next):  # type: ignore[no-untyped-def]
    """Section E's standard security response headers, applied to every route.

    CSP is intentionally tight (no inline script or style) -- Section B's chat frontend loads
    its JS/CSS from /static as separate files and never sets an inline `style="..."` attribute,
    so nothing here needed loosening once that landed. img-src additionally allows
    IMAGE_BASE_URL's own origin (see CSP_IMG_SRC above) when it points at external storage
    instead of this service's own /images route. Revisit only if a future change actually
    needs an inline script/style or another cross-origin resource.
    """
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Content-Security-Policy"] = (
        f"default-src 'self'; frame-ancestors 'none'; {CSP_IMG_SRC}; style-src 'self'"
    )
    if settings.app_env == "production":
        # Only meaningful over HTTPS, which is what production actually runs behind (Render);
        # sending it in dev over plain HTTP would just be inert noise.
        response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains"
    return response


@app.middleware("http")
async def log_requests(request: Request, call_next):  # type: ignore[no-untyped-def]
    """Structured, PII-free access logging (Section E) -- method/path/status/duration/client
    IP only, never the request body (so a guest's chat message is never logged)."""
    start = time.monotonic()
    response = await call_next(request)
    duration_ms = (time.monotonic() - start) * 1000
    access_logger.info(
        "%s %s -> %d",
        request.method,
        request.url.path,
        response.status_code,
        extra={
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": round(duration_ms, 1),
            "client_ip": request.client.host if request.client else None,
        },
    )
    return response


# Section D: per-IP request rate limit. storage_uri points at the same Redis so the limit
# holds across multiple workers (a single in-process Limiter wouldn't); in_memory_fallback
# keeps /chat self-protected (imperfectly, per-process) rather than fully unprotected if
# Redis has a hiccup, per slowapi's own documented fallback mechanism.
CHAT_RATE_LIMIT = "10/minute"  # per guest IP -- tune to actual guest traffic once observed

limiter = Limiter(
    key_func=get_remote_address,
    storage_uri=settings.redis_url,
    in_memory_fallback_enabled=True,
    in_memory_fallback=[CHAT_RATE_LIMIT],
)
app.state.limiter = limiter
# slowapi's handler is typed for RateLimitExceeded specifically, narrower than Starlette's
# generic Exception handler signature -- safe at runtime (Starlette dispatches by the
# registered exception class), just not variance-compatible for mypy.
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]

# Section D: a cap on total concurrent /chat work in flight, sized to a small free-tier
# instance's CPU/RAM -- the per-IP rate limit above bounds one guest's volume, this bounds
# total load from every guest combined. Rejects immediately (503) rather than queuing
# guests indefinitely behind a slow instance.
#
# Not asyncio.Semaphore: asyncio.wait_for(semaphore.acquire(), timeout=0) is unreliable for
# a non-blocking "try acquire, else reject" check -- a zero timeout can fire before the
# acquire's fast path (plenty of capacity, no real wait) gets a chance to run, so it can
# spuriously report "busy" even with free slots (confirmed directly: 3/3 spurious timeouts
# against a fresh Semaphore(4) with nothing else running). A plain counter + lock has no
# such race.
MAX_CONCURRENT_CHAT_REQUESTS = 4
_chat_slots_in_use = 0
_chat_slots_lock = asyncio.Lock()


async def _try_acquire_chat_slot() -> bool:
    global _chat_slots_in_use
    async with _chat_slots_lock:
        if _chat_slots_in_use >= MAX_CONCURRENT_CHAT_REQUESTS:
            return False
        _chat_slots_in_use += 1
        return True


async def _release_chat_slot() -> None:
    global _chat_slots_in_use
    async with _chat_slots_lock:
        _chat_slots_in_use -= 1


# Section 4's image-hosting choice: serve data/images/ from this same service rather than
# standing up separate object storage. StaticFiles re-checks the directory on every
# request (not just at mount time), so check_dir=False alone doesn't survive a checkout
# without the (gitignored, local-only) data/ corpus -- it defers the crash from startup to
# the first image request instead of preventing it. Guarantee the directory actually exists
# (possibly empty) instead: missing files then 404 normally, the corpus's own README/data
# provenance still governs whether real images ever land here.
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def chat_page(request: Request) -> HTMLResponse:
    """Section B: the guest-facing chat page itself -- one server-rendered Jinja2 template,
    all interactivity (session lifecycle, sending messages, images) driven by /static/js/chat.js
    calling this same service's own /session and /chat endpoints."""
    return templates.TemplateResponse(
        request,
        "chat.html",
        {
            "title": "Menu & FAQ Assistant",
            "meta_description": (
                "Ask a demo restaurant chatbot about menu items, prices, nutrition, "
                "allergens, or FAQs."
            ),
            "static_version": STATIC_VERSION,
        },
    )


def get_graph(request: Request) -> CompiledStateGraph:
    graph = request.app.state.graph
    if graph is None:
        raise HTTPException(status_code=503, detail="Chat is not configured")
    return graph


def get_checkpointer(request: Request) -> BaseCheckpointSaver:
    checkpointer = request.app.state.checkpointer
    if checkpointer is None:
        raise HTTPException(status_code=503, detail="Chat is not configured")
    return checkpointer


def get_redis_client(request: Request) -> Redis:
    redis_client = request.app.state.redis_client
    if redis_client is None:
        raise HTTPException(status_code=503, detail="Chat is not configured")
    return redis_client


def _image_url(filename: str) -> str:
    if not settings.image_base_url:
        return filename
    return f"{settings.image_base_url.rstrip('/')}/{filename}"


def _log_turn_timings(timings: dict[str, float]) -> None:
    """One PII-free line per /chat turn breaking its latency down by stage, in ms.

    `turn` is the whole threadpool hop (so it also includes any wait for a free worker thread);
    `graph` is graph.invoke() alone, whose remainder after understand + weaviate + rerank +
    generate is LangGraph overhead plus the checkpointer's Redis write. `rerank_pace` and
    `llm_backoff` are waits nested inside `rerank` / `understand` / `generate` -- see
    app/timing.py. Stages a turn never reached (e.g. rerank on a capped conversation) are
    simply absent.

    For /chat/stream, `graph` spans the whole stream and `first_delta` is the time until the
    guest first sees text.
    """
    logger.info(
        "chat turn timings",
        extra={"timings_ms": {stage: round(ms, 1) for stage, ms in timings.items()}},
    )


@dataclass(frozen=True)
class TurnReply:
    """What one /chat turn hands back: the text, the item cards under it, and, for a list of
    groups or categories, that reply cut around its list with a card per name."""

    answer: str
    cited: list[GeneratedCitedItem] = field(default_factory=list)
    choices: GeneratedChoices | None = None


def _cards_for_turn(final_state: dict[str, Any]) -> list[GeneratedCitedItem]:
    """The item cards to show with a turn's reply: exactly the ones the graph decided on for this
    turn's answer (guarantee C-21). Nothing is derived here from other state fields, which the
    per-session checkpoint carries over from earlier turns."""
    return list(final_state.get("cited_items", []))


def _choices_for_turn(final_state: dict[str, Any]) -> GeneratedChoices | None:
    """The picture cards of this turn's reply, if it lists groups or categories: exactly what the
    graph decided for this turn, for the same reason as _cards_for_turn()."""
    return final_state.get("choices")


def _precheck_reply(
    graph: CompiledStateGraph, redis_client: Redis, config: RunnableConfig
) -> str | None:
    """A fixed guest-facing reply if this turn must not reach the LLM at all -- the
    conversation-turn cap or the spend cap is already hit -- else None."""
    with timed("state_read"):
        snapshot = graph.get_state(config)
    history = (snapshot.values or {}).get("history", [])
    if len(history) >= settings.max_conversation_turns:
        return CONVERSATION_LIMIT_REPLY

    with timed("cost_check"):
        over_spend_limit = is_over_spend_limit(
            redis_client,
            daily_limit=settings.daily_token_limit,
            monthly_limit=settings.monthly_token_limit,
        )
    return CAPACITY_REPLY if over_spend_limit else None


def _run_chat_turn(
    graph: CompiledStateGraph,
    redis_client: Redis,
    session_id: str,
    message: str,
) -> TurnReply:
    """The blocking part of a /chat turn: conversation-cap and spend-cap checks, the graph
    invocation itself, and recording token usage -- run in one threadpool hop (Section A)
    so none of it blocks the event loop.

    Both caps are checked *before* graph.invoke() so a guest who's already hit one never
    reaches the LLM again for that turn (Section D's cost-control point).

    Section E: graph.get_state()/graph.invoke() are the only calls that reach Weaviate, Cohere,
    Groq, or the Redis-backed checkpointer -- a transient failure in any of them (timeout,
    connection error, Redis unreachable) is caught here and degrades to one clear fallback
    reply instead of a raw 500. A non-transient exception (a real bug) still propagates, to be
    caught by app/errors.py's generic handler and logged as an actual error.
    """
    config: RunnableConfig = {"configurable": {"thread_id": session_id}}

    try:
        fixed_reply = _precheck_reply(graph, redis_client, config)
        if fixed_reply is not None:
            return TurnReply(fixed_reply)

        with timed("graph"):
            final_state = graph.invoke({"question": message}, config=config)
    except AllKeysRateLimitedError:
        # Every Groq pool key is at its own local RPM/RPD budget -- nothing was actually sent
        # to Groq for this turn. CAPACITY_REPLY (Section D's spend-cap message) fits this
        # exactly: a temporary capacity situation, not a broken/unreachable backend, so it
        # deliberately isn't OUTBOUND_ERROR_REPLY.
        logger.warning("All Groq pool keys at local rate limit -- turn skipped, no LLM call made")
        return TurnReply(CAPACITY_REPLY)
    except TRANSIENT_OUTBOUND_ERRORS:
        logger.exception("Outbound service failure during a /chat turn")
        return TurnReply(OUTBOUND_ERROR_REPLY)

    with timed("cost_record"):
        record_token_usage(redis_client, final_state["usage"]["total_tokens"])
    return TurnReply(
        final_state["answer"], _cards_for_turn(final_state), _choices_for_turn(final_state)
    )


@app.post("/chat", response_model=ChatResponse)
@limiter.limit(CHAT_RATE_LIMIT)
async def chat(
    request: Request,
    payload: ChatRequest,
    graph: CompiledStateGraph = Depends(get_graph),
    redis_client: Redis = Depends(get_redis_client),
) -> ChatResponse:
    if not settings.chat_enabled:
        raise HTTPException(status_code=503, detail="Chat is temporarily disabled.")
    if not await _try_acquire_chat_slot():
        raise HTTPException(status_code=503, detail="Server busy, please try again shortly.")
    with track_timings() as timings:
        try:
            with timed("turn"):
                reply = await run_in_threadpool(
                    _run_chat_turn, graph, redis_client, payload.session_id, payload.message
                )
        finally:
            await _release_chat_slot()
    _log_turn_timings(timings)
    return _build_chat_response(payload.session_id, reply)


def _build_chat_response(session_id: str, reply: TurnReply) -> ChatResponse:
    choices = reply.choices
    return ChatResponse(
        session_id=session_id,
        answer=reply.answer,
        cited_items=[
            CitedItem(
                id=item["id"],
                slug=item["slug"],
                name=item["name"],
                description=item["description"],
                ingredients=item["ingredients"],
                price_gbp=item["price_gbp"],
                image=_image_url(item["image"]),
            )
            for item in reply.cited
        ],
        choices=Choices(
            intro=choices["intro"],
            outro=choices["outro"],
            cards=[
                ChoiceCard(name=card["name"], image=_image_url(card["image"]))
                for card in choices["cards"]
            ],
        )
        if choices
        else None,
    )


def _sse(event: str, data: Any) -> str:
    """One Server-Sent Event. json.dumps keeps `data` on a single line (newlines in the text
    are escaped), which is all the SSE framing needs."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _stream_chat_turn(
    graph: CompiledStateGraph,
    redis_client: Redis,
    session_id: str,
    message: str,
) -> Iterator[tuple[str, Any]]:
    """The blocking body of a /chat/stream turn, as a generator of ("delta", text) events
    followed by exactly one ("done", TurnReply).

    Same guards and the same degrade-to-a-fallback-reply handling as _run_chat_turn() -- a
    failure before or during the stream still ends in a "done" carrying the fallback text, which
    replaces whatever partial text the guest was already shown. Only a genuine bug escapes, to be
    reported by the caller as an "error" event.
    """
    config: RunnableConfig = {"configurable": {"thread_id": session_id}}
    final_state: dict[str, Any] | None = None

    try:
        fixed_reply = _precheck_reply(graph, redis_client, config)
        if fixed_reply is not None:
            yield "done", TurnReply(fixed_reply)
            return

        with timed("graph"):
            for item in graph.stream(
                {"question": message}, config=config, stream_mode=["custom", "values"]
            ):
                # With a list of modes LangGraph yields (mode, chunk) pairs; its stubs only
                # describe the single-mode (bare chunk) shape.
                mode, chunk = cast(tuple[str, Any], item)
                if mode == "custom":
                    yield "delta", chunk["delta"]
                else:
                    final_state = chunk
    except AllKeysRateLimitedError:
        logger.warning("All Groq pool keys at local rate limit -- turn skipped, no LLM call made")
        yield "done", TurnReply(CAPACITY_REPLY)
        return
    except TRANSIENT_OUTBOUND_ERRORS:
        logger.exception("Outbound service failure during a /chat/stream turn")
        yield "done", TurnReply(OUTBOUND_ERROR_REPLY)
        return

    assert final_state is not None, "graph.stream() ended without a final state"
    with timed("cost_record"):
        record_token_usage(redis_client, final_state["usage"]["total_tokens"])
    yield (
        "done",
        TurnReply(
            final_state["answer"], _cards_for_turn(final_state), _choices_for_turn(final_state)
        ),
    )


async def _sse_events(
    graph: CompiledStateGraph, redis_client: Redis, payload: ChatRequest
) -> AsyncIterator[str]:
    """Drive _stream_chat_turn() on the threadpool and encode its events for the wire.

    `first_delta` (ms from the request reaching this handler to the first visible text) is the
    latency a guest actually feels, and is logged alongside the per-stage timings.
    """
    started = time.perf_counter()
    with track_timings() as timings:
        try:
            async for event, data in iterate_in_threadpool(
                _stream_chat_turn(graph, redis_client, payload.session_id, payload.message)
            ):
                if event == "delta":
                    timings.setdefault("first_delta", (time.perf_counter() - started) * 1000)
                    yield _sse("delta", {"text": data})
                else:
                    response = _build_chat_response(payload.session_id, data)
                    yield _sse("done", response.model_dump())
        except Exception:
            # Headers (HTTP 200) are long gone by now, so the generic-500 handler can't run:
            # report the same generic message as an event instead.
            logger.exception("Unhandled exception on POST /chat/stream")
            yield _sse("error", {"error": INTERNAL_ERROR_MESSAGE})
        timings["turn"] = (time.perf_counter() - started) * 1000
    _log_turn_timings(timings)


class _ChatSlotResponse(StreamingResponse):
    """A StreamingResponse that gives its /chat concurrency slot back when it's finished.

    Done here, not in the body generator's `finally`: Starlette only ever runs a response's
    __call__, so this fires on a normal finish, a client disconnect and an error alike -- even
    if the generator was never started, which a `finally` inside it can't cover.
    """

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            await _release_chat_slot()


@app.post("/chat/stream")
@limiter.limit(CHAT_RATE_LIMIT)
async def chat_stream(
    request: Request,
    payload: ChatRequest,
    graph: CompiledStateGraph = Depends(get_graph),
    redis_client: Redis = Depends(get_redis_client),
) -> StreamingResponse:
    """Server-Sent Events version of /chat: `delta` events carrying the answer text as it is
    generated, then one `done` event with the same JSON body /chat returns (its `answer` is
    authoritative -- clients should replace the streamed text with it), or an `error` event."""
    if not settings.chat_enabled:
        raise HTTPException(status_code=503, detail="Chat is temporarily disabled.")
    if not await _try_acquire_chat_slot():
        raise HTTPException(status_code=503, detail="Server busy, please try again shortly.")
    return _ChatSlotResponse(
        _sse_events(graph, redis_client, payload),
        media_type="text/event-stream",
        # no-cache/X-Accel-Buffering: a proxy that buffers the body would hold every delta back
        # until the stream ends, defeating the point.
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/session", response_model=SessionResponse)
def create_session() -> SessionResponse:
    return SessionResponse(session_id=str(uuid.uuid4()))


@app.delete("/session/{session_id}", status_code=204)
async def delete_session(
    session_id: str, checkpointer: BaseCheckpointSaver = Depends(get_checkpointer)
) -> None:
    await run_in_threadpool(checkpointer.delete_thread, session_id)
