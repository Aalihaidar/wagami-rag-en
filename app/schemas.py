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
    name: str
    description: str | None = None
    ingredients: list[str] = Field(default_factory=list)
    price_gbp: float | None = None
    image: str


class ChoiceCard(BaseModel):
    name: str
    image: str


class Choices(BaseModel):
    """A reply that lists groups or categories, cut around its list so the chat page can lay out
    the opening sentence, a picture card per name, then the closing question."""

    intro: str
    outro: str
    cards: list[ChoiceCard]


class ChatResponse(BaseModel):
    session_id: str
    # The whole reply as text. For a list of groups or categories that includes the bullet list;
    # a client that shows `choices` instead lays out intro, cards and outro in its place.
    answer: str
    cited_items: list[CitedItem]
    choices: Choices | None = None


class SessionResponse(BaseModel):
    session_id: str
