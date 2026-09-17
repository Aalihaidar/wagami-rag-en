"""Query understanding: one structured LLM call that classifies intent and extracts
retrieval filters from a guest's raw question.

Ported from `03_evaluation_groq.ipynb`'s `understand_query()` -- including the
"(gluten-free recipe)"/"(vegan recipe)" suffix-preservation fix in the prompt
below (see docs/APP_AND_DEPLOYMENT_PLAN.md's LLM-provider note: that fix is
what made this specific notebook, not `02_generation_checks_groq.ipynb`, the
safe porting source).
"""

import json
from dataclasses import dataclass
from typing import Any, Literal

from app.agent.llm import GroqClient, Usage, zero_usage
from app.retrieval import QueryUnderstanding

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

    def expand(self, category_hint: list[str]) -> list[str]:
        """Add sibling categories (same category_path parent) to a guessed category_hint."""
        expanded = set(category_hint)
        for c in category_hint:
            expanded |= self.siblings.get(c, set())
        return sorted(expanded)


def load_category_index(kb: Any) -> CategoryIndex:
    """Read category/category_path off every menu_item row and build a CategoryIndex.

    `kb` is a Weaviate collection (or any object exposing the same `.iterator()` -- a fake is
    enough for tests, since only `return_properties` and `.properties` are used).
    """
    categories: set[str] = set()
    parent_groups: dict[str, set[str]] = {}
    for o in kb.iterator(return_properties=["category", "category_path", "item_type"]):
        props = o.properties
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
        categories=sorted(categories), siblings=siblings, alcoholic_only=alcoholic_only
    )


def build_understand_system_prompt(category_index: CategoryIndex) -> str:
    return f"""
You turn one guest question for a restaurant chatbot into a structured query-understanding
result used to search a knowledge base. Return only the JSON described by the response
schema -- no extra text.

Fields:
- intent: "menu" if the guest is asking about a dish, ingredient, price, or nutrition value --
  this includes comparing two or more named dishes to each other (e.g. "what's the difference
  between the yasai cha han and the vegan recipe version" is still a menu question, not FAQ,
  even though it doesn't ask about just one dish), and includes a general browse/availability
  question about what's on the menu (e.g. "do you have vegan options", "what desserts do you
  have") even when no specific dish is named. "faq" if they're asking about restaurant policy
  itself -- hours, bookings, delivery, payments, gift cards, or where to find allergen
  information -- not about menu content.
- dietary: "vegan" or "vegetarian" ONLY when the guest wants dishes filtered to that
  restriction (e.g. "a vegan curry", "vegetarian mains"). Use "none" for a general
  availability question like "do you have vegan options" -- that should still search
  everything rather than be filtered down, since FAQ rows about dietary options carry no
  dietary_tags of their own and a hard filter would hide them.
- price_max_gbp: a number ONLY when the guest gives a firm ceiling ("under £8", "less than
  £10"). null for vague wording like "affordable" or "cheap".
- allergens_exclude: canonical allergen names (from the list below) the guest wants excluded,
  ONLY when they state an allergy, intolerance, or something to avoid (e.g. "I have a nut
  allergy", "dairy-free options", "no shellfish"). Map colloquial terms to every matching
  canonical value -- "nuts" maps to every tree-nut entry plus peanuts, "dairy" maps to milk,
  "shellfish" maps to crustaceans and molluscs. Empty array if no allergy was stated.
- search_query: the question rewritten as a short search phrase for just the dish/food itself
  -- strip out anything already captured by dietary, price_max_gbp, or allergens_exclude
  above (don't repeat "vegan", "under £6", or allergy wording) and strip filler words ("a",
  "do you have", "what's in"). Examples: "a vegan starter under £6" -> "starter";
  "a spicy noodle dish under £8" -> "spicy noodle dish"; "what time do you open" -> "what
  time do you open" (nothing to strip for an FAQ question). EXCEPTION: a "(gluten-free
  recipe)" or "(vegan recipe)" suffix is part of that dish's own name on this menu -- two
  different recipes can share the same display name, disambiguated only by this suffix -- so
  if the guest names a dish that way, KEEP the suffix verbatim in search_query even though it
  reads like dietary wording. Example: "I'm vegan, is the yasai cha han (vegan recipe) safe"
  -> search_query "yasai cha han (vegan recipe)", NOT "yasai cha han" (stripping it searches
  for a different recipe with different allergens than the one actually asked about). Never
  return an empty string -- fall back to the original question if nothing else to extract.
- category_hint: zero or more names from the menu category list below that best match any
  course-type language in the question (e.g. "starter", "small plate", "main", "dessert",
  "drink") -- the guest's word for a course type rarely matches this menu's own category
  names exactly (there is no category literally called "starters"), so use your judgement
  about which real categories a guest asking for that course type would actually mean.
  Leave empty if the question already names a specific dish, or names no course type, or is
  an "faq" question (this list is menu categories only). EXCEPTION: the category literally
  named "drinks" is the kids' menu's drinks section specifically, not general beverages --
  for a guest asking about a drink without saying "kids", use the actual adult beverage
  categories instead ("coffee + tea", "soft drinks", "freshly made juices", "beers + cider",
  "wine + sake", "cocktails"), picking whichever most closely matches what they asked for.
- gluten_free_only: true ONLY when the guest asks for the restaurant's own gluten-free
  menu/section by name ("what's on your gluten-free menu", "gluten-free options"). This is a
  positive filter for that curated section -- separate from allergens_exclude, which is the
  safety exclusion for a guest describing an actual allergy or intolerance. Both can be true
  together (e.g. "I'm coeliac, what's on the gluten-free menu").
- kcal_max: a calorie ceiling as a number. Honour both a firm number ("under 500 calories")
  and qualitative wording ("a low-calorie main" -> use a sensible reference like 500). null
  if calories were not mentioned at all.
- protein_min_g: a protein floor in grams as a number. Honour both a firm number ("at least
  20g protein") and qualitative wording ("a high-protein dish" -> use a sensible reference
  like 20). null if protein was not mentioned at all.
- alcohol_free: true ONLY when the guest explicitly wants a non-alcoholic / alcohol-free
  drink.

If the user message begins with "Recent conversation so far:" followed by prior guest/
assistant turns and then "Guest's new message:", treat only the text after "Guest's new
message:" as the question to classify and extract every field from -- use the prior turns
only to resolve pronouns or implicit references in that new message (e.g. "what about the
vegan one?" naming a dish mentioned earlier), never to pull price/allergy/dietary wording
that belonged to a previous turn instead of this one.

Canonical allergens: {", ".join(ALLOWED_ALLERGENS)}
Menu categories: {", ".join(category_index.categories)}
""".strip()


def build_response_schema(category_index: CategoryIndex) -> dict:
    """Standard JSON Schema (Groq's strict json_schema mode) for understand_query()'s output."""
    return {
        "type": "object",
        "properties": {
            "intent": {"type": "string", "enum": ["menu", "faq"]},
            "dietary": {"type": "string", "enum": ["vegan", "vegetarian", "none"]},
            "price_max_gbp": {"type": ["number", "null"]},
            "allergens_exclude": {
                "type": "array",
                "items": {"type": "string", "enum": ALLOWED_ALLERGENS},
            },
            "search_query": {"type": "string"},
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
            "dietary",
            "price_max_gbp",
            "allergens_exclude",
            "search_query",
            "category_hint",
            "gluten_free_only",
            "kcal_max",
            "protein_min_g",
            "alcohol_free",
        ],
        "additionalProperties": False,
    }


class UnderstandingResult(QueryUnderstanding):
    """QueryUnderstanding plus the two fields app/retrieval.py never needs: the classified
    intent (used to pick generation tone/temperature) and this call's own token usage."""

    intent: Literal["menu", "faq"]
    usage: Usage


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
        return {
            "intent": "menu",
            "dietary": None,
            "price_max_gbp": None,
            "allergens_exclude": [],
            "search_query": question,
            "category_hint": [],
            "gluten_free_only": False,
            "kcal_max": None,
            "protein_min_g": None,
            "alcohol_free": False,
            "usage": zero_usage(),
        }
    user_message = f"{context}\n\nGuest's new message: {question}" if context else question
    resp = groq_client.call(
        build_understand_system_prompt(category_index),
        user_message,
        model=model,
        response_schema=build_response_schema(category_index),
        temperature=UNDERSTAND_TEMPERATURE,
        reasoning_effort=UNDERSTAND_REASONING_EFFORT,
    )
    parsed = json.loads(resp["text"])
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
    return {
        "intent": parsed.get("intent") if parsed.get("intent") in ("menu", "faq") else "menu",
        "dietary": None if dietary == "none" else dietary,
        "price_max_gbp": parsed.get("price_max_gbp"),
        "allergens_exclude": allergens,
        "search_query": search_query,
        "category_hint": category_hint,
        "gluten_free_only": bool(parsed.get("gluten_free_only")),
        "kcal_max": parsed.get("kcal_max"),
        "protein_min_g": parsed.get("protein_min_g"),
        "alcohol_free": alcohol_free,
        "usage": resp["usage"],
    }
