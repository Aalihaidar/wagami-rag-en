from app.agent.generation import (
    build_context,
    build_user_prompt,
    format_row,
    temperature_for,
    tone_for,
)
from app.retrieval import MenuRow


def make_row(
    name: str,
    *,
    item_type: str = "menu_item",
    description: str | None = "tasty",
    price_gbp: float | None = 9.5,
    is_gluten_free_listed: bool = False,
    dietary_tags: list[str] | None = None,
    allergens_contains: list[str] | None = None,
    allergens_may_contain: list[str] | None = None,
) -> MenuRow:
    return {
        "uuid": f"uuid-{name}",
        "score": 0.5,
        "properties": {
            "name": name,
            "item_type": item_type,
            "description": description,
            "category": "ramen",
            "price_gbp": price_gbp,
            "kcal": 500.0,
            "protein_g": 20.0,
            "abv_percent": None,
            "is_gluten_free_listed": is_gluten_free_listed,
            "dietary_tags": dietary_tags or [],
            "allergens_contains": allergens_contains or [],
            "allergens_may_contain": allergens_may_contain or [],
        },
    }


def test_tone_and_temperature_by_intent() -> None:
    assert tone_for("faq") != tone_for("menu")
    assert temperature_for("faq") == 0.8
    assert temperature_for("menu") == 0.2


def test_format_row_faq() -> None:
    row = make_row("what time do you open", item_type="faq", description="9:00-24:00, daily.")
    line = format_row(row)
    assert line.startswith("- FAQ | Q: what time do you open")
    assert "9:00-24:00" in line


def test_format_row_menu_item_includes_all_fields() -> None:
    row = make_row(
        "vegan ramen",
        dietary_tags=["vegan"],
        allergens_contains=["soya"],
        allergens_may_contain=["sesame"],
    )
    line = format_row(row)
    assert "vegan ramen" in line
    assert "£9.50" in line
    assert "500 kcal" in line
    assert "dietary_tags: vegan" in line
    assert "allergens_contains: soya" in line
    assert "allergens_may_contain: sesame" in line


def test_format_row_handles_missing_price_and_abv() -> None:
    row = make_row("still water", price_gbp=None)
    line = format_row(row)
    assert "price: not listed" in line
    assert "ABV not listed" in line


def test_build_context_empty_when_no_ranked_hits() -> None:
    assert build_context([]) == "(no matching rows retrieved)"


def test_build_context_joins_formatted_rows() -> None:
    ranked = [{"row": make_row("ramen"), "rerank": 0.9, "hybrid": 0.5}]
    context = build_context(ranked)  # type: ignore[arg-type]
    assert "ramen" in context


def test_build_user_prompt_plain() -> None:
    prompt = build_user_prompt("what's in the ramen", "(no matching rows retrieved)", [])
    assert "QUESTION: what's in the ramen" in prompt
    assert "NOTE" not in prompt


def test_build_user_prompt_includes_relaxed_fields_note() -> None:
    prompt = build_user_prompt("a low-cal high-protein main", "CONTEXT ROW", ["kcal_max"])
    assert "NOTE: no result matched every part of the question" in prompt
    assert "kcal_max" in prompt


def test_build_user_prompt_includes_excluded_top_match_note() -> None:
    prompt = build_user_prompt(
        "is the yasai cha han (vegan recipe) vegan",
        "CONTEXT ROW",
        [],
        {"name": "yasai cha han (vegan recipe)", "reason": "is not tagged vegan"},
    )
    assert "yasai cha han (vegan recipe)" in prompt
    assert "is not tagged vegan" in prompt
    assert "do NOT say you don't have information" in prompt
