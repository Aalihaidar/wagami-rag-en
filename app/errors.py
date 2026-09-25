"""Section E (security & ops hygiene): one consistent JSON error shape across every endpoint,
and the set of outbound-call failures app/main.py degrades to a guest-facing fallback message
for instead of a raw 500.

The shape is `{"error": "..."}` everywhere -- chosen to match slowapi's own
`_rate_limit_exceeded_handler` (already wired in app/main.py for 429s), rather than inventing
a second shape alongside it.
"""

import logging

import httpx
import redis.exceptions
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from weaviate.exceptions import WeaviateBaseError

from app.agent.llm import LLMStreamError

logger = logging.getLogger("app.errors")

INTERNAL_ERROR_MESSAGE = "Something went wrong on our end. Please try again."

OUTBOUND_ERROR_REPLY = (
    "Sorry, I'm having trouble reaching one of my backend services right now. Please try "
    "again in a moment."
)

# Transient failures from the app's three outbound dependencies (Weaviate, the Cohere/Groq
# httpx calls, Redis) -- app/main.py's _run_chat_turn() catches exactly this tuple around
# graph.invoke() and returns OUTBOUND_ERROR_REPLY instead of letting it fall through to the
# generic 500 handler below. Deliberately not `Exception` itself: a genuine programming bug
# should still surface as a 500 (and get logged as one), not be silently smoothed over as "the
# backend is busy."
TRANSIENT_OUTBOUND_ERRORS: tuple[type[Exception], ...] = (
    WeaviateBaseError,
    httpx.HTTPError,  # status errors and transport errors/timeouts (Cohere rerank / Groq calls)
    LLMStreamError,  # Groq reporting an error inside an already-open answer stream
    redis.exceptions.RedisError,
    TimeoutError,
)


def register_exception_handlers(app: FastAPI) -> None:
    """Register the three handlers that make every error response `{"error": "..."}`."""

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse({"error": exc.detail}, status_code=exc.status_code, headers=exc.headers)

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # Field-level details are safe to return here -- they describe what's wrong with the
        # guest's own request, not any server-side internals.
        return JSONResponse({"error": "Invalid request.", "details": exc.errors()}, status_code=422)

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        # Full traceback goes server-side only, in prod and dev alike -- the client never sees
        # more than the generic message, satisfying Section E's "generic error responses in
        # production" without special-casing APP_ENV here.
        logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
        return JSONResponse({"error": INTERNAL_ERROR_MESSAGE}, status_code=500)
