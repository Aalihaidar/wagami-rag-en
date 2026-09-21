from types import SimpleNamespace
from typing import Any

import pytest

from app.agent.catalog import build_catalog
from app.agent.llm import LLMResponse, Usage
from app.agent.understanding import (
    ALLOWED_ALLERGENS,
    INTENTS,
    CategoryIndex,
    build_response_schema,
    build_understand_system_prompt,
    load_category_index,
    parse_understanding,
    understand_query,
)

SAMPLE_HISTORY_CONTEXT = (
    "Recent conversation so far (most recent last):\nGuest: is yasai cha han vegan\nAssistant: no"
)


def make_category_index() -> CategoryIndex:
    return CategoryIndex(
        categories=["bao buns", "gyoza", "ramen", "soft drinks", "beers + cider"],
        siblings={"bao buns": {"bao buns", "gyoza"}, "gyoza": {"bao buns", "gyoza"}},
        alcoholic_only={"beers + cider"},
    )


class FakeGroqClient:
    def __init__(self, text: str) -> None:
        self._text = text
        self.calls: list[dict[str, Any]] = []

    def call(self, system_prompt: str, user_prompt: str, **kwargs: Any) -> LLMResponse:
        self.calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt, **kwargs})
        usage: Usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        return {"text": self._text, "usage": usage}


class FakeCategoryKB:
    """Stands in for the Weaviate collection's .iterator() -- load_category_index() only
    reads return_properties off each row's .properties, per its own docstring."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def iterator(self, return_properties: list[str]) -> list[Any]:
        return [SimpleNamespace(properties=row) for row in self._rows]


def test_load_category_index_does_not_expand_drinks_siblings() -> None:
    """Regression test for a live bug: "is there coffee?" expanded category_hint to all six
    adult-beverage categories (coffee + tea, wine + sake, beers + cider, cocktails, soft
    drinks, freshly made juices) because they all share the "drinks" parent in category_path,
    and sibling expansion treated them as interchangeable the same way it correctly does for
    course-type groups (bao buns / gyoza / lighter bites / big flavour bites). That diluted
    both hybrid retrieval and the rerank query text badly enough that every genuine coffee/tea
    row lost to unrelated wine/juice/cider rows. Drink sub-categories are mutually exclusive
    drink TYPES, not synonyms, so "drinks" must be excluded from sibling expansion specifically.
    """
    kb = FakeCategoryKB(
        [
            {
                "item_type": "menu_item",
                "category": "bao buns",
                "category_path": ["sides", "bao buns"],
            },
            {"item_type": "menu_item", "category": "gyoza", "category_path": ["sides", "gyoza"]},
            {
                "item_type": "menu_item",
                "category": "coffee + tea",
                "category_path": ["drinks", "coffee + tea"],
            },
            {
                "item_type": "menu_item",
                "category": "wine + sake",
                "category_path": ["drinks", "wine + sake"],
            },
            {"item_type": "faq", "category": "faqs", "category_path": ["faqs"]},
        ]
    )
    index = load_category_index(kb)

    # Course-type siblings still expand -- this is the behavior the mechanism exists for.
    assert index.siblings["bao buns"] == {"bao buns", "gyoza"}
    assert index.siblings["gyoza"] == {"bao buns", "gyoza"}

    # Drink categories must NOT be expanded into each other.
    assert "coffee + tea" not in index.siblings
    assert "wine + sake" not in index.siblings

    # alcoholic_only is computed straight from parent_groups, not from the sibling exclusion
    # above, so it's unaffected by this fix.
    assert index.alcoholic_only == {"wine + sake"}


def test_understand_query_fallback_when_no_groq_client() -> None:
    result = understand_query(
        "a vegan starter under £6",
        category_index=make_category_index(),
        groq_client=None,
        model="openai/gpt-oss-120b",
    )
    assert result["intent"] == "menu"
    assert result["search_query"] == "a vegan starter under £6"
    assert result["dietary"] is None
    assert result["usage"] == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def test_understand_query_expands_category_siblings() -> None:
    import json

    payload = json.dumps(
        {
            "intent": "menu",
            "dietary": "none",
            "price_max_gbp": None,
            "allergens_exclude": [],
            "search_query": "starter",
            "category_hint": ["bao buns"],
            "gluten_free_only": False,
            "kcal_max": None,
            "protein_min_g": None,
            "alcohol_free": False,
        }
    )
    client = FakeGroqClient(payload)
    result = understand_query(
        "a starter",
        category_index=make_category_index(),
        groq_client=client,  # type: ignore[arg-type]
        model="openai/gpt-oss-120b",
    )
    assert result["category_hint"] == ["bao buns", "gyoza"]
    assert result["usage"]["total_tokens"] == 15


def test_understand_query_alcohol_free_strips_alcoholic_category_hints() -> None:
    import json

    payload = json.dumps(
        {
            "intent": "menu",
            "dietary": "none",
            "price_max_gbp": None,
            "allergens_exclude": [],
            "search_query": "a drink",
            "category_hint": ["beers + cider", "soft drinks"],
            "gluten_free_only": False,
            "kcal_max": None,
            "protein_min_g": None,
            "alcohol_free": True,
        }
    )
    client = FakeGroqClient(payload)
    result = understand_query(
        "a non-alcoholic drink",
        category_index=make_category_index(),
        groq_client=client,  # type: ignore[arg-type]
        model="openai/gpt-oss-120b",
    )
    assert result["alcohol_free"] is True
    assert result["category_hint"] == ["soft drinks"]


def test_understand_query_ignores_unknown_allergen_and_category_values() -> None:
    import json

    payload = json.dumps(
        {
            "intent": "menu",
            "dietary": "none",
            "price_max_gbp": None,
            "allergens_exclude": ["not-a-real-allergen", "milk"],
            "search_query": "a dish",
            "category_hint": ["not-a-real-category", "ramen"],
            "gluten_free_only": False,
            "kcal_max": None,
            "protein_min_g": None,
            "alcohol_free": False,
        }
    )
    client = FakeGroqClient(payload)
    result = understand_query(
        "a dish with milk allergy",
        category_index=make_category_index(),
        groq_client=client,  # type: ignore[arg-type]
        model="openai/gpt-oss-120b",
    )
    assert result["allergens_exclude"] == ["milk"]
    assert result["category_hint"] == ["ramen"]


def test_build_understand_system_prompt_keeps_recipe_suffix_instruction() -> None:
    # The source notebook's own EXCEPTION clause line-wraps mid-phrase ("(gluten-free\n  recipe)"),
    # so check on whitespace-collapsed text rather than an exact substring.
    prompt = " ".join(build_understand_system_prompt(make_category_index()).split())
    assert "(gluten-free recipe)" in prompt
    assert "(vegan recipe)" in prompt
    assert "KEEP the suffix verbatim" in prompt
    assert "ramen" in prompt  # category list is embedded live


def test_build_response_schema_uses_live_categories_and_allergens() -> None:
    schema = build_response_schema(make_category_index())
    assert schema["properties"]["category_hint"]["items"]["enum"] == [
        "bao buns",
        "gyoza",
        "ramen",
        "soft drinks",
        "beers + cider",
    ]
    assert schema["properties"]["allergens_exclude"]["items"]["enum"] == ALLOWED_ALLERGENS


def test_understand_query_wraps_user_message_with_history_context() -> None:
    import json

    payload = json.dumps(
        {
            "intent": "menu",
            "dietary": "vegan",
            "price_max_gbp": None,
            "allergens_exclude": [],
            "search_query": "yasai cha han (vegan recipe)",
            "category_hint": [],
            "gluten_free_only": False,
            "kcal_max": None,
            "protein_min_g": None,
            "alcohol_free": False,
        }
    )
    client = FakeGroqClient(payload)
    understand_query(
        "what about the vegan one",
        category_index=make_category_index(),
        groq_client=client,  # type: ignore[arg-type]
        model="openai/gpt-oss-120b",
        context=SAMPLE_HISTORY_CONTEXT,
    )
    sent = client.calls[0]["user_prompt"]
    assert sent.startswith("Recent conversation so far")
    assert sent.endswith("Guest's new message: what about the vegan one")


def test_understand_query_no_context_sends_bare_question() -> None:
    import json

    payload = json.dumps(
        {
            "intent": "menu",
            "dietary": None,
            "price_max_gbp": None,
            "allergens_exclude": [],
            "search_query": "ramen",
            "category_hint": [],
            "gluten_free_only": False,
            "kcal_max": None,
            "protein_min_g": None,
            "alcohol_free": False,
        }
    )
    client = FakeGroqClient(payload)
    understand_query(
        "a ramen dish",
        category_index=make_category_index(),
        groq_client=client,  # type: ignore[arg-type]
        model="openai/gpt-oss-120b",
    )
    assert client.calls[0]["user_prompt"] == "a ramen dish"


def test_understand_query_search_query_fallback_never_leaks_history_context() -> None:
    """Even with a history context, a missing/empty search_query must fall back to the bare
    current question -- never to the wrapped user_message, which would leak old turns into
    the text used for retrieval."""
    import json

    payload = json.dumps(
        {
            "intent": "menu",
            "dietary": None,
            "price_max_gbp": None,
            "allergens_exclude": [],
            "search_query": "",
            "category_hint": [],
            "gluten_free_only": False,
            "kcal_max": None,
            "protein_min_g": None,
            "alcohol_free": False,
        }
    )
    client = FakeGroqClient(payload)
    result = understand_query(
        "what about the vegan one",
        category_index=make_category_index(),
        groq_client=client,  # type: ignore[arg-type]
        model="openai/gpt-oss-120b",
        context=SAMPLE_HISTORY_CONTEXT,
    )
    assert result["search_query"] == "what about the vegan one"


# ---- routing: intents, browse fields, and the safety net ---------------------------------------

USAGE: Usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


def browse_index() -> CategoryIndex:
    rows = [
        {
            "item_type": "menu_item",
            "name": "Flat White",
            "category": "coffee + tea",
            "category_path": ["drinks", "coffee + tea"],
        },
        {
            "item_type": "menu_item",
            "name": "Roku G+T",
            "category": "cocktails",
            "category_path": ["drinks", "cocktails"],
        },
        {
            "item_type": "menu_item",
            "name": "Gyoza",
            "category": "gyoza",
            "category_path": ["sides", "gyoza"],
        },
    ]
    catalog = build_catalog(rows)
    return CategoryIndex(
        categories=catalog.categories, siblings={}, alcoholic_only=set(), catalog=catalog
    )


def understanding_json(**overrides: Any) -> str:
    import json

    base: dict[str, Any] = {
        "intent": "menu",
        "browse_group": "none",
        "browse_category": "none",
        "dietary": "none",
        "price_max_gbp": None,
        "allergens_exclude": [],
        "search_query": "x",
        "category_hint": [],
        "gluten_free_only": False,
        "kcal_max": None,
        "protein_min_g": None,
        "alcohol_free": False,
    }
    return json.dumps({**base, **overrides})


def parse(**overrides: Any) -> Any:
    text = understanding_json(**overrides)
    return parse_understanding(text, USAGE, "the question", browse_index())


def test_the_schema_offers_every_intent_and_the_real_groups_and_categories() -> None:
    schema = build_response_schema(browse_index())

    assert schema["properties"]["intent"]["enum"] == list(INTENTS)
    assert schema["properties"]["browse_group"]["enum"] == ["drinks", "sides", "none"]
    assert schema["properties"]["browse_category"]["enum"] == [
        "cocktails",
        "coffee + tea",
        "gyoza",
        "none",
    ]
    assert {"intent", "browse_group", "browse_category"} <= set(schema["required"])


def test_an_intent_the_model_invents_falls_back_to_menu() -> None:
    assert parse(intent="chitchat")["intent"] == "menu"


@pytest.mark.parametrize("intent", ["greeting", "off_topic", "menu", "faq"])
def test_the_other_intents_pass_through(intent: str) -> None:
    assert parse(intent=intent)["intent"] == intent


def test_a_browse_keeps_the_group_and_category_the_guest_named() -> None:
    result = parse(intent="menu_browse", browse_group="drinks", browse_category="cocktails")

    assert result["intent"] == "menu_browse"
    assert (result["browse_group"], result["browse_category"]) == ("drinks", "cocktails")


def test_none_and_unknown_names_become_no_selection() -> None:
    result = parse(intent="menu_browse", browse_group="none", browse_category="pizza")

    assert result["intent"] == "menu_browse"
    assert (result["browse_group"], result["browse_category"]) == (None, None)


def test_browse_names_are_dropped_for_every_other_intent() -> None:
    result = parse(intent="menu", browse_group="drinks", browse_category="cocktails")

    assert (result["browse_group"], result["browse_category"]) == (None, None)


@pytest.mark.parametrize(
    "requirement",
    [
        {"dietary": "vegan"},
        {"dietary": "vegetarian"},
        {"allergens_exclude": ["milk"]},
        {"price_max_gbp": 6},
        {"kcal_max": 500},
        {"protein_min_g": 20},
        {"alcohol_free": True},
    ],
)
def test_a_browse_with_a_requirement_is_downgraded_to_a_search(requirement: dict) -> None:
    """Listing a group ignores allergens, diets and every other requirement, so a message that
    states one must go through retrieval, where the filters (allergen exclusion above all) are
    enforced in code -- whatever the model classified it as."""
    result = parse(intent="menu_browse", browse_group="drinks", **requirement)

    assert result["intent"] == "menu"
    assert result["browse_group"] is None and result["browse_category"] is None


def test_naming_the_gluten_free_section_alone_is_still_a_browse() -> None:
    result = parse(intent="menu_browse", browse_group="drinks", gluten_free_only=True)

    assert result["intent"] == "menu_browse"


def test_browsing_is_off_when_there_is_no_catalog_to_list_from() -> None:
    no_catalog = CategoryIndex(categories=["gyoza"], siblings={}, alcoholic_only=set())
    result = parse_understanding(
        understanding_json(intent="menu_browse", browse_category="gyoza"), USAGE, "q", no_catalog
    )

    assert result["intent"] == "menu"


def test_the_no_key_fallback_is_a_plain_search_with_no_browse_selection() -> None:
    result = understand_query("hello", category_index=browse_index(), groq_client=None, model="m")

    assert result["intent"] == "menu"
    assert (result["browse_group"], result["browse_category"]) == (None, None)


def test_the_prompt_sent_to_the_model_carries_the_structure_of_the_knowledge_base() -> None:
    client = FakeGroqClient(understanding_json(intent="greeting"))

    understand_query("hi", category_index=browse_index(), groq_client=client, model="m")  # type: ignore[arg-type]

    system_prompt = client.calls[0]["system_prompt"]
    assert "Knowledge base structure" in system_prompt
    assert "- drinks: 2 categories: cocktails; coffee + tea" in system_prompt


def test_load_category_index_builds_the_catalog_from_the_same_rows() -> None:
    kb = FakeCategoryKB(
        [
            {
                "item_type": "menu_item",
                "name": "Gyoza",
                "category": "gyoza",
                "category_path": ["sides", "gyoza"],
            },
            {
                "item_type": "faq",
                "name": "Hours?",
                "category": "faqs",
                "category_path": ["faqs", "hours"],
            },
        ]
    )

    index = load_category_index(kb)

    assert index.catalog.groups == ["sides"]
    assert index.catalog.items_of("sides", "gyoza") == ["Gyoza"]
    assert index.catalog.item_type_counts == {"faq": 1, "menu_item": 1}


# ---- an off-topic verdict on a real dish question is overruled -----------------------------------


def test_off_topic_is_overruled_when_the_message_names_a_menu_item() -> None:
    """Seen live: "tell me about the roku g+t" was classified off_topic and the guest got the
    polite decline instead of an answer. A message that names a real dish is searched."""
    result = parse_understanding(
        understanding_json(intent="off_topic"), USAGE, "tell me about the roku g+t", browse_index()
    )

    assert result["intent"] == "menu"
    assert result["search_query"] == "x"


def test_off_topic_stands_when_no_menu_item_is_named() -> None:
    result = parse_understanding(
        understanding_json(intent="off_topic"), USAGE, "how do I change a car tyre", browse_index()
    )

    assert result["intent"] == "off_topic"


def test_off_topic_is_left_alone_when_there_is_no_catalog_to_check_against() -> None:
    no_catalog = CategoryIndex(categories=["gyoza"], siblings={}, alcoholic_only=set())
    result = parse_understanding(
        understanding_json(intent="off_topic"), USAGE, "tell me about the roku g+t", no_catalog
    )

    assert result["intent"] == "off_topic"


def test_naming_a_dish_does_not_change_a_greeting_or_a_faq() -> None:
    assert parse(intent="greeting")["intent"] == "greeting"
    assert parse(intent="faq")["intent"] == "faq"


def test_the_prompt_tells_the_model_to_prefer_menu_over_off_topic_when_unsure() -> None:
    prompt = " ".join(
        build_understand_system_prompt(browse_index()).split()
    )  # collapse the hand-wrapping

    assert "Use it only when the message is clearly unrelated" in prompt
    assert 'a wrong "off_topic" turns a real guest away' in prompt
