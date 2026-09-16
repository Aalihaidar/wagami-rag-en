import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph
from starlette.concurrency import run_in_threadpool

from app.agent.checkpointer import build_checkpointer
from app.agent.generation import cited_items_from_ranked
from app.agent.graph import build_graph
from app.agent.llm import GroqClient
from app.agent.understanding import load_category_index
from app.config import get_settings
from app.retrieval import RetrievalTool, connect
from app.schemas import ChatRequest, ChatResponse, CitedItem, SessionResponse

settings = get_settings()

# Resolved relative to this file, not the caller's cwd -- same reasoning as
# scripts/load_knowledge_base.py's own DATA_FILE.
IMAGES_DIR = Path(__file__).resolve().parent.parent / "data" / "images"


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
    if not (settings.weaviate_url and settings.weaviate_read_api_key):
        yield
        return

    weaviate_client = connect(settings)
    try:
        kb = weaviate_client.collections.get("KnowledgeBase")
        retrieval_tool = RetrievalTool(kb=kb, cohere_api_key=settings.embedding_api_key)
        category_index = load_category_index(kb)
        groq_client = GroqClient(settings.groq_api_key) if settings.groq_api_key else None
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
        weaviate_client.close()


app = FastAPI(
    title="Restaurant Chatbot API",
    docs_url=None if settings.app_env == "production" else "/docs",
    redoc_url=None if settings.app_env == "production" else "/redoc",
    lifespan=lifespan,
)

# Section 4's image-hosting choice: serve data/images/ from this same service rather than
# standing up separate object storage. check_dir=False so a checkout without the
# (gitignored, local-only) data/ corpus still starts -- image requests just 404 instead.
app.mount("/images", StaticFiles(directory=IMAGES_DIR, check_dir=False), name="images")


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


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


def _image_url(filename: str) -> str:
    if not settings.image_base_url:
        return filename
    return f"{settings.image_base_url.rstrip('/')}/{filename}"


@app.post("/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest, graph: CompiledStateGraph = Depends(get_graph)
) -> ChatResponse:
    config: RunnableConfig = {"configurable": {"thread_id": payload.session_id}}
    # graph.invoke() makes blocking Weaviate/Cohere/Groq HTTP calls -- run it off the event
    # loop so one slow guest turn can't stall every other request on this single instance.
    final_state = await run_in_threadpool(
        graph.invoke, {"question": payload.message}, config=config
    )
    cited = cited_items_from_ranked(final_state["search_result"]["ranked"])
    return ChatResponse(
        session_id=payload.session_id,
        answer=final_state["answer"],
        cited_items=[
            CitedItem(id=item["id"], slug=item["slug"], image=_image_url(item["image"]))
            for item in cited
        ],
    )


@app.post("/session", response_model=SessionResponse)
def create_session() -> SessionResponse:
    return SessionResponse(session_id=str(uuid.uuid4()))


@app.delete("/session/{session_id}", status_code=204)
async def delete_session(
    session_id: str, checkpointer: BaseCheckpointSaver = Depends(get_checkpointer)
) -> None:
    await run_in_threadpool(checkpointer.delete_thread, session_id)
