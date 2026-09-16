from fastapi import FastAPI

from app.config import get_settings

settings = get_settings()

app = FastAPI(
    title="Restaurant Chatbot API",
    docs_url=None if settings.app_env == "production" else "/docs",
    redoc_url=None if settings.app_env == "production" else "/redoc",
)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}
