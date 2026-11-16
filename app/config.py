from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

from app.cost_control import (
    DAILY_TOKEN_LIMIT_DEFAULT,
    MAX_CONVERSATION_TURNS_DEFAULT,
    MONTHLY_TOKEN_LIMIT_DEFAULT,
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: Literal["development", "production"] = "development"
    port: int = 8000
    log_level: str = "info"

    # Section E's kill switch: flip to false via the platform's env store + a restart to take
    # /chat offline immediately (a 503) without pushing/rebuilding a new deploy.
    chat_enabled: bool = True

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

    # Telegram bot (a second chat surface next to the web page, same agent). Empty token (the
    # default) leaves the bot off and its webhook route answering 404. The secret is the value
    # Telegram sends back in X-Telegram-Bot-Api-Secret-Token on every update, so only Telegram
    # can drive the bot.
    telegram_bot_token: str = ""
    telegram_webhook_secret: str = ""
    # The chat page's own public https URL: turns it into a Telegram Mini App -- /start gets an
    # "open" button for it, and the page may be framed by Telegram's own domains (see
    # app/main.py's security_headers). Empty leaves /start as plain text.
    telegram_web_app_url: str = ""

    # Section D (rate limiting & cost control) -- tune to your actual budget/provider limits.
    daily_token_limit: int = DAILY_TOKEN_LIMIT_DEFAULT
    monthly_token_limit: int = MONTHLY_TOKEN_LIMIT_DEFAULT
    max_conversation_turns: int = MAX_CONVERSATION_TURNS_DEFAULT


@lru_cache
def get_settings() -> Settings:
    return Settings()
