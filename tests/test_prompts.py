import re

import pytest

from app.agent.catalog import build_catalog
from app.agent.generation import parse_generation_reply
from app.agent.prompts import (
    CITATION_OUTPUT_INSTRUCTIONS,
    GENERATION_RULES,
    GENERATION_SYSTEM_PROMPT,
    SCOPE_AND_SAFETY,
    build_understand_system_prompt,
)
from app.agent.understanding import INTENTS, CategoryIndex, build_response_schema

GYOZA_ROW = {
    "item_type": "menu_item",
    "name": "x",
    "category": "gyoza",
    "category_path": ["sides", "gyoza"],
}


def test_the_system_prompt_is_the_rules_then_the_scope_guard() -> None:
    assert f"{GENERATION_RULES}\n\n{SCOPE_AND_SAFETY}" == GENERATION_SYSTEM_PROMPT


def test_the_scope_guard_is_kept_out_of_the_generation_rules() -> None:
    """The leak check compares replies against SCOPE_AND_SAFETY alone; if its wording also sat
    in GENERATION_RULES (which the model is told to echo, e.g. the decline text) legitimate
    replies would look like leaks."""
    assert SCOPE_AND_SAFETY not in GENERATION_RULES
    assert GENERATION_RULES.splitlines()[0] not in SCOPE_AND_SAFETY


def _fields_section(prompt: str) -> str:
    """The `Fields:` bullets of the understanding prompt (the structure block above it also has
    bullets, one per corpus field)."""
    return prompt.split("\nFields:\n", 1)[1].split("\n\n", 1)[0]


@pytest.mark.parametrize("with_catalog", [False, True])
def test_the_understand_prompt_describes_exactly_the_response_schema_fields(
    with_catalog: bool,
) -> None:
    """Prompt and response schema must describe the same fields: a field in one and not the
    other means the model is told to fill something the schema forbids, or the reverse."""
    catalog = build_catalog([GYOZA_ROW])
    index = CategoryIndex(categories=["gyoza"], siblings={}, alcoholic_only=set(), catalog=catalog)
    prompt = build_understand_system_prompt(
        allowed_allergens=["milk"], categories=["gyoza"], catalog=catalog if with_catalog else None
    )

    fields = re.findall(r"^- ([a-z_]+):", _fields_section(prompt), re.M)
    assert fields == build_response_schema(index)["required"]


def test_the_understand_prompt_defines_every_routing_intent() -> None:
    fields = _fields_section(
        build_understand_system_prompt(allowed_allergens=["milk"], categories=["gyoza"])
    )

    for intent in INTENTS:
        assert f'"{intent}" --' in fields


def test_the_structure_block_is_only_added_when_there_is_a_catalog() -> None:
    catalog = build_catalog([GYOZA_ROW])
    with_block = build_understand_system_prompt(
        allowed_allergens=["milk"], categories=["gyoza"], catalog=catalog
    )
    without = build_understand_system_prompt(allowed_allergens=["milk"], categories=["gyoza"])

    assert "Knowledge base structure" in with_block
    assert with_block.index("Knowledge base structure") < with_block.index("\nFields:\n")
    assert "Menu groups: sides" in with_block
    assert "Knowledge base structure" not in without
    assert "Menu groups:" not in without


def test_the_understand_prompt_appends_the_live_vocabularies() -> None:
    prompt = build_understand_system_prompt(
        allowed_allergens=["milk", "eggs"], categories=["bao buns", "gyoza"]
    )

    assert prompt.endswith("Canonical allergens: milk, eggs\nMenu categories: bao buns, gyoza")


def test_the_citation_prompt_names_the_keys_the_reply_parser_reads() -> None:
    assert re.findall(r'^- "(\w+)":', CITATION_OUTPUT_INSTRUCTIONS, re.M) == [
        "answer",
        "cited_slugs",
    ]
    assert parse_generation_reply('{"answer": "hi", "cited_slugs": ["a"]}', ["a"]) == ("hi", ["a"])


# Distinctive tokens from the rules that carry the app's safety guarantees (allergens, recipe
# variants, the decline path, the injection guard). Deleting or rewording one away by accident
# should fail a test rather than quietly change what the model is told; a deliberate change
# updates this list and re-runs the evaluation.
@pytest.mark.parametrize(
    ("prompt", "phrase"),
    [
        (GENERATION_RULES, "allergens_contains"),
        (GENERATION_RULES, "allergens_may_contain"),
        (GENERATION_RULES, "(gluten-free recipe)"),
        (GENERATION_RULES, "(vegan recipe)"),
        (GENERATION_RULES, "price exactly as given in CONTEXT"),
        (GENERATION_RULES, "limited data set"),
        (GENERATION_RULES, "EXCEPTION: if a NOTE appears"),
        (SCOPE_AND_SAFETY, "<guest_message>"),
        (SCOPE_AND_SAFETY, "<retrieved_context>"),
        (SCOPE_AND_SAFETY, "Refuse everything else"),
        (SCOPE_AND_SAFETY, "Never reveal"),
    ],
)
def test_the_safety_critical_wording_is_present(prompt: str, phrase: str) -> None:
    assert phrase in prompt
