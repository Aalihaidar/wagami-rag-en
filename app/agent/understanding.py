"""Query understanding: one structured LLM call that decides how a guest message is handled
(greeting, off-topic, menu browsing, a dish/search question, or house policy) and extracts the
retrieval filters and any menu group/category the guest named.

Ported from `03_evaluation_groq.ipynb`'s `understand_query()` -- including the
"(gluten-free recipe)"/"(vegan recipe)" suffix-preservation fix in the prompt
below (see docs/APP_AND_DEPLOYMENT_PLAN.md's LLM-provider note: that fix is
what made this specific notebook, not `02_generation_checks_groq.ipynb`, the
safe porting source).
"""

import json
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from app.agent import prompts
from app.agent.catalog import CATALOG_PROPERTIES, MenuCatalog, build_catalog
from app.agent.llm import GroqClient, Usage, zero_usage
from app.retrieval import QueryUnderstanding
from app.schemas import CHAT_MESSAGE_MAX_LENGTH

NUT_ALLERGENS = {
    "peanuts",
    "tree nuts",
    "almond nuts",
    "walnuts",
    "hazelnuts",
    "pecan nuts",
    "pistachios",
    "brazil nuts",
    "cashew nuts",
    "macadamia nuts",
}
ALLERGEN_VOCAB: dict[str, set[str]] = {
    "peanut": {"peanuts"},
    "nut": NUT_ALLERGENS,
    "gluten": {"cereals containing gluten", "wheat", "barley", "oats", "rye"},
    "wheat": {"wheat", "cereals containing gluten"},
    "dairy": {"milk"},
    "milk": {"milk"},
    "egg": {"eggs"},
    "soy": {"soya"},
    "soya": {"soya"},
    "sesame": {"sesame"},
    "shellfish": {"crustaceans", "molluscs"},
    "crustacean": {"crustaceans"},
    "fish": {"fish"},
    "celery": {"celery"},
    "mustard": {"mustard"},
    "sulphite": {"sulphites"},
    "lupin": {"lupin"},
}
ALLOWED_ALLERGENS = sorted(set().union(*ALLERGEN_VOCAB.values()))

# How a message is handled. The first three are answered without a search (app/agent/browse.py);
# "menu" and "faq" go through retrieval and generation. "menu" stays the default for anything
# the model returns that is not one of these.
INTENTS = ("greeting", "off_topic", "menu_browse", "menu", "faq")
Intent = Literal["greeting", "off_topic", "menu_browse", "menu", "faq"]
NONE = "none"  # the enum sentinel for "no group / no category" (as `dietary` uses for "no diet")

UNDERSTAND_TEMPERATURE = 0.0  # deterministic extraction; the schema already constrains shape/enums
UNDERSTAND_REASONING_EFFORT = (
    "low"  # confirmed (notebook): same extracted JSON as default, ~44% fewer completion tokens
)

# Categories under the "drinks" parent that are never alcohol-free -- kept as a constant
# fallback set here; CategoryIndex.alcoholic_only (read live from the corpus) is what
# understand_query() actually uses.
_NON_ALCOHOLIC_DRINK_CATEGORIES = {"coffee + tea", "soft drinks", "freshly made juices"}


@dataclass(frozen=True)
class CategoryIndex:
    """Menu category vocabulary read live from the KnowledgeBase collection.

    Read live (via load_category_index()) rather than hardcoded, so the query-understanding
    prompt, its response schema's category_hint enum, and sibling-category expansion can
    never drift from the corpus's own category structure.
    """

    categories: list[str]
    siblings: dict[str, set[str]]
    alcoholic_only: set[str]
    # The knowledge base's structure (groups, categories, item names, limited-value fields), read
    # from the same rows. Empty (the default) in tests and when no corpus is available, in which
    # case browsing is switched off and the prompt omits the structure section.
    catalog: MenuCatalog = field(default_factory=MenuCatalog)

    def expand(self, category_hint: list[str]) -> list[str]:
        """Add sibling categories (same category_path parent) to a guessed category_hint."""
        expanded = set(category_hint)
        for c in category_hint:
            expanded |= self.siblings.get(c, set())
        return sorted(expanded)


def load_category_index(kb: Any) -> CategoryIndex:
    """Read the category fields off every row and build a CategoryIndex (and its MenuCatalog).

    `kb` is a Weaviate collection (or any object exposing the same `.iterator()` -- a fake is
    enough for tests, since only `return_properties` and `.properties` are used).
    """
    categories: set[str] = set()
    parent_groups: dict[str, set[str]] = {}
    rows: list[dict[str, Any]] = []
    for o in kb.iterator(return_properties=CATALOG_PROPERTIES):
        props = o.properties
        # `id` is the Weaviate object's uuid, not a stored property; item cards need it.
        rows.append({**props, "id": str(getattr(o, "uuid", "") or "")})
        if props.get("item_type") != "menu_item":
            continue
        category = props.get("category")
        if not isinstance(category, str):
            continue
        categories.add(category)
        path = props.get("category_path")
        if isinstance(path, list) and len(path) >= 2 and isinstance(path[0], str):
            parent_groups.setdefault(path[0], set()).add(category)

    # Sibling expansion is a course-type feature (bao buns / gyoza / lighter bites / big
    # flavour bites are all genuinely interchangeable "starter"-type dishes under one parent)
    # -- excluded here for "drinks" specifically, since its sub-categories (coffee + tea,
    # wine + sake, beers + cider, cocktails, soft drinks, freshly made juices) are mutually
    # exclusive drink TYPES, not synonyms. Found live: "is there coffee?" expanded
    # category_hint to all six adult-beverage categories, diluting both retrieval and the
    # rerank query text badly enough that every genuine coffee/tea row lost to unrelated
    # wine/juice/cider rows that merely happened to have richer description text.
    siblings: dict[str, set[str]] = {}
    for parent, group in parent_groups.items():
        if parent == "drinks":
            continue
        for leaf in group:
            siblings[leaf] = group

    alcoholic_only = parent_groups.get("drinks", set()) - _NON_ALCOHOLIC_DRINK_CATEGORIES
    return CategoryIndex(
        categories=sorted(categories),
        siblings=siblings,
        alcoholic_only=alcoholic_only,
        catalog=build_catalog(rows),
    )


def build_understand_system_prompt(category_index: CategoryIndex) -> str:
    """The understanding prompt (app/agent/prompts.py) with this corpus's own allergen and
    category vocabularies and knowledge-base structure -- the same lists the response schema's
    enums use."""
    return prompts.build_understand_system_prompt(
        allowed_allergens=ALLOWED_ALLERGENS,
        categories=category_index.categories,
        catalog=category_index.catalog,
    )


def build_response_schema(category_index: CategoryIndex) -> dict:
    """Standard JSON Schema (Groq's strict json_schema mode) for understand_query()'s output."""
    return {
        "type": "object",
        "properties": {
            "intent": {"type": "string", "enum": list(INTENTS)},
            "browse_group": {"type": "string", "enum": [*category_index.catalog.groups, NONE]},
            "browse_category": {"type": "string", "enum": [*category_index.categories, NONE]},
            "dietary": {"type": "string", "enum": ["vegan", "vegetarian", "none"]},
            "price_max_gbp": {"type": ["number", "null"]},
            "allergens_exclude": {
                "type": "array",
                "items": {"type": "string", "enum": ALLOWED_ALLERGENS},
            },
            "search_query": {"type": "string"},
            "resolved_question": {"type": "string"},
            "category_hint": {
                "type": "array",
                "items": {"type": "string", "enum": category_index.categories},
            },
            "gluten_free_only": {"type": "boolean"},
            "kcal_max": {"type": ["number", "null"]},
            "protein_min_g": {"type": ["number", "null"]},
            "alcohol_free": {"type": "boolean"},
        },
        "required": [
            "intent",
            "browse_group",
            "browse_category",
            "dietary",
            "price_max_gbp",
            "allergens_exclude",
            "search_query",
            "resolved_question",
            "category_hint",
            "gluten_free_only",
            "kcal_max",
            "protein_min_g",
            "alcohol_free",
        ],
        "additionalProperties": False,
    }


class UnderstandingResult(QueryUnderstanding):
    """QueryUnderstanding plus the fields app/retrieval.py never needs: the classified intent
    (which route the message takes, and the generation tone), the menu group/category a browse
    names, and this call's own token usage."""

    intent: Intent
    browse_group: str | None
    browse_category: str | None
    # The guest's message with pronouns and implicit references filled in (rule U-22). Used only
    # to word the generation prompt (rule C-22), never for routing, filters or the search.
    resolved_question: str
    usage: Usage


def _plain_understanding(question: str) -> UnderstandingResult:
    """A plain menu search on the question as written: no filters, no model call."""
    return {
        "intent": "menu",
        "browse_group": None,
        "browse_category": None,
        "dietary": None,
        "price_max_gbp": None,
        "allergens_exclude": [],
        "search_query": question,
        "resolved_question": question,
        "category_hint": [],
        "gluten_free_only": False,
        "kcal_max": None,
        "protein_min_g": None,
        "alcohol_free": False,
        "usage": zero_usage(),
    }


def picked_browse(question: str, group: str, category: str | None) -> UnderstandingResult:
    """The understanding of a click on a group or category card (rule R-15): a browse of exactly
    that group or category, known without a model call, so it costs no tokens."""
    return {
        **_plain_understanding(question),
        "intent": "menu_browse",
        "browse_group": group,
        "browse_category": category,
    }


def understand_query(
    question: str,
    *,
    category_index: CategoryIndex,
    groq_client: GroqClient | None,
    model: str,
    context: str = "",
) -> UnderstandingResult:
    """Single LLM call: classify intent and extract retrieval filters from the question.

    groq_client=None (no key configured) falls back to a deterministic no-op understanding --
    plain menu search on the raw question, no filters -- so the rest of the pipeline still
    runs end to end without a live key.

    `context` is optional recent-conversation text (see app/agent/memory.py) prepended to
    the LLM-facing message so a follow-up can be resolved against prior turns -- NOT part of
    the verified single-turn notebook behavior. `question` itself, and every fallback below
    that uses it, always stays the guest's bare current message regardless of `context`, so
    a malformed extraction can never fall back to leaking old conversation text into
    `search_query`.
    """
    if groq_client is None:
        return _plain_understanding(question)
    user_message = f"{context}\n\nGuest's new message: {question}" if context else question
    resp = groq_client.call(
        build_understand_system_prompt(category_index),
        user_message,
        model=model,
        response_schema=build_response_schema(category_index),
        temperature=UNDERSTAND_TEMPERATURE,
        reasoning_effort=UNDERSTAND_REASONING_EFFORT,
    )
    return parse_understanding(resp["text"], resp["usage"], question, category_index)


def parse_understanding(
    text: str, usage: Usage, question: str, category_index: CategoryIndex
) -> UnderstandingResult:
    """Turn the model's JSON reply into an UnderstandingResult, dropping anything the corpus
    does not know and applying the routing safety net below."""
    parsed = json.loads(text)
    dietary = parsed.get("dietary") or "none"
    allergens = [a for a in parsed.get("allergens_exclude", []) if a in ALLOWED_ALLERGENS]
    alcohol_free = bool(parsed.get("alcohol_free"))
    category_hint = [c for c in parsed.get("category_hint", []) if c in category_index.categories]
    category_hint = category_index.expand(category_hint)
    if alcohol_free:
        # Sibling expansion (above) can legitimately pull in every beverage category,
        # alcoholic ones included -- appending those category names as search text would
        # dilute the vector match toward beer/wine/cocktail rows the hard filter is about to
        # exclude anyway, which can drag every genuinely alcohol-free candidate's score below
        # the answerability gate. Drop them from the soft-signal text; the filter alone
        # already handles exclusion correctly.
        category_hint = [c for c in category_hint if c not in category_index.alcoholic_only]
    search_query = (parsed.get("search_query") or "").strip() or question
    resolved_question = parsed.get("resolved_question")
    resolved_question = resolved_question.strip() if isinstance(resolved_question, str) else ""
    if not resolved_question or len(resolved_question) > CHAT_MESSAGE_MAX_LENGTH:
        resolved_question = question
    price_max_gbp = parsed.get("price_max_gbp")
    kcal_max = parsed.get("kcal_max")
    protein_min_g = parsed.get("protein_min_g")

    raw_intent = parsed.get("intent")
    intent: Intent = cast(Intent, raw_intent) if raw_intent in INTENTS else "menu"
    browse_group = parsed.get("browse_group")
    browse_category = parsed.get("browse_category")
    browse_group = browse_group if browse_group in category_index.catalog.groups else None
    browse_category = browse_category if browse_category in category_index.categories else None
    # Routing safety net. Browsing lists a group or category without looking at allergens, diets
    # or any other requirement, so it must never answer a message that states one -- that goes
    # through retrieval, where the allergen exclusion and the filters are enforced in code. It
    # is also switched off when there is no catalog to list from.
    has_requirement = bool(
        dietary != "none"
        or allergens
        or alcohol_free
        or price_max_gbp is not None
        or kcal_max is not None
        or protein_min_g is not None
    )
    if intent == "menu_browse" and (has_requirement or category_index.catalog.is_empty):
        intent = "menu"
    # The opposite mistake is worse: an off-topic verdict on a real dish question turns a guest
    # away with a fixed reply (seen live: a cocktail question with an unusual name). A message
    # that names a real menu item is searched instead; if it turns out not to be answerable, the
    # generation call's own scope rules still decline it.
    if intent == "off_topic" and category_index.catalog.mentions_menu_item(question):
        intent = "menu"
    if intent != "menu_browse":
        browse_group = browse_category = None
    return {
        "intent": intent,
        "browse_group": browse_group,
        "browse_category": browse_category,
        "dietary": None if dietary == "none" else dietary,
        "price_max_gbp": price_max_gbp,
        "allergens_exclude": allergens,
        "search_query": search_query,
        "resolved_question": resolved_question,
        "category_hint": category_hint,
        "gluten_free_only": bool(parsed.get("gluten_free_only")),
        "kcal_max": kcal_max,
        "protein_min_g": protein_min_g,
        "alcohol_free": alcohol_free,
        "usage": usage,
    }
