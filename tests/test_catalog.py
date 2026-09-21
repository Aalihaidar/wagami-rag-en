import json
from pathlib import Path

import pytest

from app.agent.catalog import LIMITED_VALUE_FIELDS, build_catalog
from app.agent.prompts import FIELD_GUIDE, render_knowledge_base_structure

# Small corpus with the same shape problems as the real one: a group that is also a category
# name ("drinks"), a category under two groups ("ramen"), groups with a single category, and FAQ
# topics that contain commas.
ROWS = [
    {
        "item_type": "menu_item",
        "name": "Zebra Latte",
        "category": "coffee + tea",
        "category_slug": "coffee-tea",
        "category_path": ["drinks", "coffee + tea"],
        "dietary_tags": ["vegetarian"],
        "allergens_contains": ["milk"],
        "allergens_may_contain": [],
        "is_gluten_free_listed": False,
        "portion_unit": "ea",
        "servings": "1",
    },
    {
        "item_type": "menu_item",
        "name": "apple juice",
        "category": "drinks",
        "category_slug": "drinks",
        "category_path": ["kids", "drinks"],
        "dietary_tags": ["vegan", "vegetarian"],
        "allergens_contains": [],
        "allergens_may_contain": ["sesame"],
        "is_gluten_free_listed": True,
        "portion_unit": "ea",
        "servings": "1",
    },
    {
        "item_type": "menu_item",
        "name": "Kids Ramen",
        "category": "ramen",
        "category_slug": "ramen",
        "category_path": ["kids", "ramen"],
        "dietary_tags": [],
        "allergens_contains": ["wheat"],
        "allergens_may_contain": [],
        "is_gluten_free_listed": False,
        "portion_unit": "ea",
        "servings": "1",
    },
    {
        "item_type": "menu_item",
        "name": "Big Ramen",
        "category": "ramen",
        "category_slug": "ramen",
        "category_path": ["the main event", "ramen"],
        "dietary_tags": [],
        "allergens_contains": ["wheat"],
        "allergens_may_contain": [],
        "is_gluten_free_listed": False,
        "portion_unit": "portion",
        "servings": "1",
    },
    {
        "item_type": "menu_item",
        "name": "chillies",
        "category": "extras",
        "category_slug": "extras",
        "category_path": ["extras"],
        "dietary_tags": ["vegan"],
        "allergens_contains": [],
        "allergens_may_contain": [],
        "is_gluten_free_listed": False,
        "portion_unit": None,
        "servings": None,
    },
    {
        "item_type": "faq",
        "name": "When are you open?",
        "category": "faqs",
        "category_slug": "faqs",
        "category_path": ["faqs", "hours, locations + contact"],
        "dietary_tags": [],
        "allergens_contains": [],
        "allergens_may_contain": [],
        "is_gluten_free_listed": False,
        "portion_unit": None,
        "servings": None,
    },
]


def test_counts_and_item_types() -> None:
    catalog = build_catalog(ROWS)

    assert catalog.total_rows == 6
    assert catalog.item_type_counts == {"faq": 1, "menu_item": 5}


def test_groups_and_categories_come_from_the_first_and_last_path_items() -> None:
    catalog = build_catalog(ROWS)

    assert catalog.groups == ["drinks", "extras", "kids", "the main event"]
    assert catalog.categories == ["coffee + tea", "drinks", "extras", "ramen"]
    assert catalog.categories_of("kids") == ["drinks", "ramen"]
    # A group with one entry in its path is its own single category.
    assert catalog.categories_of("extras") == ["extras"]
    assert catalog.groups_with_category("ramen") == ["kids", "the main event"]
    assert catalog.groups_with_category("nonexistent") == []


def test_menu_items_are_listed_by_group_and_category_sorted_by_name() -> None:
    catalog = build_catalog(ROWS)

    assert catalog.items_of("drinks", "coffee + tea") == ["Zebra Latte"]
    assert catalog.items_of("kids", "ramen") == ["Kids Ramen"]
    assert catalog.items_of("kids", "nonexistent") == []
    # FAQ rows are never part of the browsable menu.
    assert "faqs" not in catalog.menu_items


def test_the_type_tree_covers_every_item_type_including_faq() -> None:
    catalog = build_catalog(ROWS)

    assert catalog.type_tree["menu_item"]["kids"] == ("drinks", "ramen")
    assert catalog.type_tree["faq"] == {"faqs": ("hours, locations + contact",)}


def test_limited_value_fields_list_every_value_with_null_last() -> None:
    catalog = build_catalog(ROWS)

    assert catalog.field_values["item_type"] == ("faq", "menu_item")
    assert catalog.field_values["dietary_tags"] == ("vegan", "vegetarian")
    assert catalog.field_values["is_gluten_free_listed"] == ("false", "true")
    assert catalog.field_values["portion_unit"] == ("ea", "portion", "null")
    assert catalog.field_values["servings"] == ("1", "null")
    # category_path lists every name at every level, once.
    assert "hours, locations + contact" in catalog.field_values["category_path"]
    assert catalog.field_values["category_path"].count("ramen") == 1


def test_an_empty_corpus_gives_an_empty_catalog() -> None:
    catalog = build_catalog([])

    assert catalog.is_empty
    assert catalog.groups == [] and catalog.categories == []


def test_a_row_without_a_path_falls_back_to_its_category() -> None:
    catalog = build_catalog([{"item_type": "menu_item", "name": "x", "category": "sides"}])

    assert catalog.groups == ["sides"]
    assert catalog.items_of("sides", "sides") == ["x"]


# ---- item details (what a listing describes and a card shows)

DETAIL_ROWS = [
    {
        "id": "id-b",
        "slug": "b-latte",
        "item_type": "menu_item",
        "name": "b Latte",
        "category_path": ["drinks", "coffee + tea"],
        "description": "  with oat  ",
        "ingredients": ["oat", "coffee"],
        "price_gbp": 2.5,
        "image": "b-latte.png",
    },
    {
        "id": "id-a",
        "slug": "a-tea",
        "item_type": "menu_item",
        "name": "A Tea",
        "category_path": ["drinks", "coffee + tea"],
        "description": None,
        "ingredients": [],
        "price_gbp": None,
        "image": None,
    },
]


def test_item_details_keep_what_a_listing_and_a_card_need() -> None:
    catalog = build_catalog(DETAIL_ROWS)

    details = catalog.item_details_of("drinks", "coffee + tea")
    latte = next(i for i in details if i.slug == "b-latte")
    assert (latte.id, latte.name, latte.description) == ("id-b", "b Latte", "with oat")
    assert latte.ingredients == ("oat", "coffee")
    assert (latte.price_gbp, latte.image) == (2.5, "b-latte.png")


def test_missing_details_are_none_or_empty_not_invented() -> None:
    tea = build_catalog(DETAIL_ROWS).item_details_of("drinks", "coffee + tea")[0]

    assert tea.name == "A Tea"
    assert (tea.description, tea.ingredients, tea.price_gbp, tea.image) == (None, (), None, None)


def test_item_details_are_in_the_same_order_as_the_names() -> None:
    catalog = build_catalog(DETAIL_ROWS)

    names = catalog.items_of("drinks", "coffee + tea")
    assert names == ["A Tea", "b Latte"]  # case-insensitive alphabetical
    assert [i.name for i in catalog.item_details_of("drinks", "coffee + tea")] == names
    assert catalog.item_details_of("drinks", "nonexistent") == []


# ---- recognising a dish name in a guest's message


@pytest.mark.parametrize(
    "message",
    [
        "tell me about the big ramen",
        "TELL ME ABOUT THE BIG RAMEN?",
        "is the big-ramen spicy",  # punctuation between the words does not hide the name
        "zebra latte please",
        "how much is the Zebra   Latte, and is it hot",
    ],
)
def test_a_message_naming_a_menu_item_is_recognised(message: str) -> None:
    assert build_catalog(ROWS).mentions_menu_item(message)


@pytest.mark.parametrize(
    "message",
    [
        "how do I change a car tyre",
        "what is the capital of France",
        "a big bowl of anything",  # only part of a name
        "",
    ],
)
def test_a_message_that_names_no_menu_item_is_not(message: str) -> None:
    assert not build_catalog(ROWS).mentions_menu_item(message)


def test_a_dish_name_is_matched_as_whole_words_only() -> None:
    # "chillies" is an item; it must not match inside another word.
    catalog = build_catalog(ROWS)

    assert catalog.mentions_menu_item("extra chillies please")
    assert not catalog.mentions_menu_item("unchilliesque")


def test_punctuation_inside_a_name_is_ignored_on_both_sides() -> None:
    catalog = build_catalog(
        [{"item_type": "menu_item", "name": "Roku G+T", "category_path": ["drinks", "cocktails"]}]
    )

    assert catalog.mentions_menu_item("tell me about the roku g+t")
    assert catalog.mentions_menu_item("what is roku g t")
    assert catalog.mentions_menu_item("Roku G+T?")


def test_faq_questions_and_an_empty_catalog_never_count_as_menu_items() -> None:
    assert not build_catalog(ROWS).mentions_menu_item("when are you open")
    assert not build_catalog([]).mentions_menu_item("tell me about the big ramen")


# ---- the structure block the understanding prompt is given (rules 7 to 10)


def test_the_structure_block_states_every_count_and_value() -> None:
    block = render_knowledge_base_structure(build_catalog(ROWS))

    assert "holds 6 rows" in block
    assert f"same {len(FIELD_GUIDE)} fields" in block
    assert "Item types (2): faq (1 row), menu_item (5 rows)" in block  # rule 8
    assert "- menu_item: 4 groups: drinks; extras; kids; the main event" in block  # rule 9
    assert "  - kids: 2 categories: drinks; ramen" in block  # rule 10
    assert "  - extras: 1 category: extras" in block
    for name in LIMITED_VALUE_FIELDS:  # rule 7
        assert f"- {name} (" in block


def test_values_are_separated_by_semicolons_so_a_comma_in_a_name_is_not_a_split() -> None:
    block = render_knowledge_base_structure(build_catalog(ROWS))

    assert "  - faqs: 1 category: hours, locations + contact" in block


# ---- against the real corpus, when it is checked out (data/ is not part of the repo)

CORPUS = Path(__file__).resolve().parent.parent / "data" / "knowledge_base.json"


@pytest.fixture
def corpus_rows() -> list[dict]:
    if not CORPUS.exists():
        pytest.skip("data/knowledge_base.json is not present in this checkout")
    return json.loads(CORPUS.read_text(encoding="utf-8"))


def test_the_field_guide_describes_exactly_the_corpus_fields(corpus_rows: list[dict]) -> None:
    assert [name for name, _ in FIELD_GUIDE] == list(corpus_rows[0])


def test_the_real_corpus_has_the_expected_shape(corpus_rows: list[dict]) -> None:
    catalog = build_catalog(corpus_rows)

    assert catalog.item_type_counts == {"faq": 35, "menu_item": 162}
    assert len(catalog.groups) == 9
    assert len(catalog.categories) == 25
    # The name collisions the browse logic has to handle really exist in the data.
    assert catalog.groups_with_category("ramen") == ["kids", "the main event"]
    assert "drinks" in catalog.groups and "drinks" in catalog.categories_of("kids")


def test_every_real_menu_item_has_what_its_card_needs(corpus_rows: list[dict]) -> None:
    catalog = build_catalog(corpus_rows)
    items = [
        item
        for group in catalog.groups
        for category in catalog.categories_of(group)
        for item in catalog.item_details_of(group, category)
    ]

    assert len(items) == 162
    assert all(item.id and item.slug and item.name and item.image for item in items)
