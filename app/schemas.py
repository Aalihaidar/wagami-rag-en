"""Pydantic request/response models for app/main.py's HTTP endpoints."""

from pydantic import BaseModel, ConfigDict, Field

# Section 3's cheap pre-filter for obviously off-topic/abusive input: pydantic rejects an
# over-length message before it ever reaches understand_query()/generation, no LLM call spent.
CHAT_MESSAGE_MAX_LENGTH = 500


class BrowsePick(BaseModel):
    """The group (and category) of a picture card the guest clicked (rule R-15): answered from
    the catalog as that browse, with no understanding call. `category` is None for a group."""

    model_config = ConfigDict(extra="forbid")

    group: str = Field(min_length=1, max_length=100)
    category: str | None = Field(default=None, min_length=1, max_length=100)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    message: str = Field(min_length=1, max_length=CHAT_MESSAGE_MAX_LENGTH)
    # Set only by a click on a group or category card; `message` is still the guest's visible
    # wording of it, and what is saved to history.
    browse: BrowsePick | None = None


class CitedItem(BaseModel):
    id: str
    slug: str
    name: str
    description: str | None = None
    ingredients: list[str] = Field(default_factory=list)
    price_gbp: float | None = None
    image: str
    # Only shown by the chat page in the single-dish detail view (rule C-26); a listing's or a
    # multi-dish answer's gallery card ignores these, same as before.
    dietary_tags: list[str] = Field(default_factory=list)
    allergens_contains: list[str] = Field(default_factory=list)
    allergens_may_contain: list[str] = Field(default_factory=list)
    # The rest of the dish's row, also only shown in the single-dish detail view (C-26).
    category: str | None = None
    category_path: list[str] = Field(default_factory=list)
    # Per serving, keyed by field name (kcal, protein_g, carbs_g, ...).
    nutrition: dict[str, float | None] = Field(default_factory=dict)
    is_gluten_free_listed: bool = False
    portion_value: float | None = None
    portion_unit: str | None = None
    servings: str | None = None
    abv_percent: float | None = None


class ChoiceCard(BaseModel):
    name: str
    image: str
    # What a click on the card browses to (rule R-15): the group, and the category for a
    # category's card (None for a group's).
    group: str
    category: str | None = None


class Choices(BaseModel):
    """A reply that lists groups or categories, cut around its list so the chat page can lay out
    the opening sentence, a picture card per name, then the closing question."""

    intro: str
    outro: str
    cards: list[ChoiceCard]


class ChatResponse(BaseModel):
    session_id: str
    # The whole reply as text. For a list of groups or categories, or of a category's items, that
    # includes the bullet list; a client that shows `choices` (groups/categories) or `intro`/
    # `outro` (a category's items, alongside `cited_items`) instead lays those out in its place.
    answer: str
    cited_items: list[CitedItem]
    choices: Choices | None = None
    # The reply cut around a category's item listing (rule C-25): the opening sentence and the
    # closing question, with `cited_items` shown as cards between them instead of a bullet list.
    # None when the reply isn't a full item listing (an item with no image keeps the bullet list).
    intro: str | None = None
    outro: str | None = None


class SessionResponse(BaseModel):
    session_id: str
