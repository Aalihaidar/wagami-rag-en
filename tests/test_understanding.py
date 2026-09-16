from typing import Any

from app.agent.llm import LLMResponse, Usage
from app.agent.understanding import (
    ALLOWED_ALLERGENS,
    CategoryIndex,
    build_response_schema,
    build_understand_system_prompt,
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
