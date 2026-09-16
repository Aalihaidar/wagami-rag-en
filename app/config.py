from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: Literal["development", "production"] = "development"
    port: int = 8000
    log_level: str = "info"

    weaviate_url: str = ""
    # Read-only key for the app's retrieval path -- never the admin key
    # scripts/ use to create/drop the collection and write objects.
    weaviate_read_api_key: str = ""
    embedding_provider: Literal["cohere", "openai"] = "cohere"
    # Also used directly as the Cohere rerank API key (app/retrieval.py) when
    # embedding_provider is "cohere", matching the notebooks' dual use.
    embedding_api_key: str = ""

    # LLM provider for app/agent/ -- Groq for now, not Gemini (see
    # docs/APP_AND_DEPLOYMENT_PLAN.md's LLM-provider note: every one of the
    # four candidate notebooks was independently missing at least one of
    # three verified retrieval bugfixes; only the Groq eval notebook has all
    # three). Same env var names the notebooks already use.
    groq_api_key: str = ""
    understand_model: str = "openai/gpt-oss-120b"
    generation_model: str = "openai/gpt-oss-120b"

    # Backs app/agent/'s LangGraph checkpointer (per-session conversation memory) -- needs
    # Redis Stack, not plain Redis (see docker-compose.yml's redis service comment).
    redis_url: str = "redis://redis:6379/0"

    # Base URL a knowledge_base row's `image` filename is appended to when building a
    # /chat response's cited items. Empty (default) leaves cited images as bare filenames.
    image_base_url: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
