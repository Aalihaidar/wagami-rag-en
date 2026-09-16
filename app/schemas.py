"""Pydantic request/response models for app/main.py's HTTP endpoints."""

from pydantic import BaseModel, ConfigDict, Field

# Section 3's cheap pre-filter for obviously off-topic/abusive input: pydantic rejects an
# over-length message before it ever reaches understand_query()/generation, no LLM call spent.
CHAT_MESSAGE_MAX_LENGTH = 500


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    message: str = Field(min_length=1, max_length=CHAT_MESSAGE_MAX_LENGTH)


class CitedItem(BaseModel):
    id: str
    slug: str
    image: str


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    cited_items: list[CitedItem]


class SessionResponse(BaseModel):
    session_id: str
