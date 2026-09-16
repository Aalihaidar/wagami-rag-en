"""Pydantic request/response models for app/main.py's HTTP endpoints."""

from pydantic import BaseModel, ConfigDict


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    message: str


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
