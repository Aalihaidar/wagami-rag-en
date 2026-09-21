import pytest

from app.agent.browse import (
    DIRECT_INTENTS,
    browse_answer,
    browse_reply,
    direct_answer,
    direct_reply,
)
from app.agent.catalog import MenuCatalog, build_catalog
from app.agent.prompts import GREETING_REPLY, OFF_TOPIC_REPLY


def _row(name: str, path: list[str]) -> dict:
    return {"item_type": "menu_item", "name": name, "category": path[-1], "category_path": path}


@pytest.fixture
def catalog() -> MenuCatalog:
    return build_catalog(
        [
            _row("Lychee Sangria", ["drinks", "cocktails"]),
            _row("Roku G+T", ["drinks", "cocktails"]),
            _row("Flat White", ["drinks", "coffee + tea"]),
            _row("Kids Juice", ["kids", "drinks"]),  # a category named like a group
            _row("Kids Ramen", ["kids", "ramen"]),
            _row("Big Ramen", ["the main event", "ramen"]),  # `ramen` under two groups
            _row("Gyoza", ["sides", "gyoza"]),
            _row("Chillies", ["extras"]),  # a group with a single category
            _row("Sauce", ["extras"]),
            _row("Bibimbap", ["limited time only", "buldak"]),
        ]
    )


def test_no_group_or_category_lists_the_groups_and_asks_which(catalog: MenuCatalog) -> None:
    reply = browse_reply(catalog, None, None)

    assert "Our menu is organised into these categories:" in reply
    for group in catalog.groups:
        assert f"- {group}\n" in reply + "\n"
    assert reply.endswith("What kind of these would you like to see?")


def test_a_group_lists_its_categories_without_searching(catalog: MenuCatalog) -> None:
    reply = browse_reply(catalog, "drinks", None)

    assert reply.startswith("In drinks we have these sub-categories:")
    assert "- cocktails\n- coffee + tea" in reply
    assert "Lychee Sangria" not in reply  # categories, not items, at this level


def test_a_category_lists_all_its_items(catalog: MenuCatalog) -> None:
    reply = browse_reply(catalog, None, "cocktails")

    assert reply.startswith("Here is everything in cocktails (drinks):")
    assert "- Lychee Sangria\n- Roku G+T" in reply
    assert reply.endswith("Would you like to know more about any of these?")


def test_a_group_with_one_category_skips_straight_to_its_items(catalog: MenuCatalog) -> None:
    extras = browse_reply(catalog, "extras", None)
    limited = browse_reply(catalog, "limited time only", None)

    assert extras.startswith("Here is everything in extras:")
    assert "- Chillies\n- Sauce" in extras
    assert "Here is everything in buldak (limited time only):" in limited


def test_a_bare_name_that_is_a_group_means_the_group(catalog: MenuCatalog) -> None:
    # "drinks" is a group AND the kids' category; a guest saying just "drinks" wants the group.
    reply = browse_reply(catalog, None, "drinks")

    assert reply.startswith("In drinks we have these sub-categories:")


def test_naming_the_group_reaches_the_category_that_shares_its_name(catalog: MenuCatalog) -> None:
    reply = browse_reply(catalog, "kids", "drinks")

    assert reply.startswith("Here is everything in drinks (kids):")
    assert "- Kids Juice" in reply


def test_a_category_in_two_groups_asks_which_one_is_meant(catalog: MenuCatalog) -> None:
    reply = browse_reply(catalog, None, "ramen")

    assert reply == (
        "We have ramen in more than one part of the menu: kids and the main event. "
        "Which one would you like to see?"
    )
    assert "- Kids Ramen" in browse_reply(catalog, "kids", "ramen")


def test_a_group_that_does_not_contain_the_category_is_ignored(catalog: MenuCatalog) -> None:
    # The model named a group and a category that do not go together: trust the category.
    reply = browse_reply(catalog, "sides", "cocktails")

    assert reply.startswith("Here is everything in cocktails (drinks):")


def test_an_unknown_name_falls_back_to_the_overview_instead_of_guessing(
    catalog: MenuCatalog,
) -> None:
    assert browse_reply(catalog, None, "pizza").startswith("Our menu is organised into")
    assert browse_reply(catalog, "brunch", None).startswith("Our menu is organised into")


def test_lists_are_bulleted_one_per_line(catalog: MenuCatalog) -> None:
    lines = browse_reply(catalog, "drinks", None).splitlines()

    assert [line for line in lines if line.startswith("- ")] == ["- cocktails", "- coffee + tea"]


def test_greeting_and_off_topic_have_fixed_replies(catalog: MenuCatalog) -> None:
    assert direct_reply({"intent": "greeting"}, catalog) == GREETING_REPLY
    assert direct_reply({"intent": "off_topic"}, catalog) == OFF_TOPIC_REPLY


def test_the_fixed_replies_say_what_the_rules_require() -> None:
    assert "how can i help you with our restaurant or menu" in GREETING_REPLY.lower()
    # Off-topic: a polite pointer to what the assistant can help with. The evaluation's decline
    # detector recognises this exact phrasing.
    assert "can only help with questions about our restaurant and menu" in OFF_TOPIC_REPLY


def test_direct_reply_reads_the_browse_fields_off_the_understanding(catalog: MenuCatalog) -> None:
    understanding = {"intent": "menu_browse", "browse_group": "drinks", "browse_category": None}

    assert direct_reply(understanding, catalog).startswith("In drinks we have")


def test_only_greeting_off_topic_and_browse_skip_retrieval() -> None:
    assert {"greeting", "off_topic", "menu_browse"} == DIRECT_INTENTS


# ---- item listings: described in the text, with a card each ------------------------------------


@pytest.fixture
def detailed() -> MenuCatalog:
    def row(name: str, path: list[str], **details: object) -> dict:
        return {
            "id": f"id-{name}",
            "slug": name.lower().replace(" ", "-"),
            "item_type": "menu_item",
            "name": name,
            "category": path[-1],
            "category_path": path,
            **details,
        }

    return build_catalog(
        [
            row(
                "Iced Latte",
                ["drinks", "coffee + tea"],
                description="sweeten with cane syrup",
                ingredients=["sweeten with cane syrup", "milk", "coffee"],
                price_gbp=2.5,
                image="iced-latte.png",
            ),
            row(
                "Double Espresso",
                ["drinks", "coffee + tea"],
                ingredients=["coffee"],
                price_gbp=2.5,
                image="espresso.png",
            ),
            row("Plain Water", ["drinks", "coffee + tea"]),  # nothing known, and no picture
            row("Gyoza", ["sides", "gyoza"], price_gbp=6.0, image="gyoza.png"),
            row("Bao", ["sides", "bao buns"], price_gbp=5.0, image="bao.png"),
        ]
    )


def test_a_category_listing_describes_each_item_like_a_single_dish_answer(
    detailed: MenuCatalog,
) -> None:
    text = browse_answer(detailed, "drinks", "coffee + tea").text

    assert text.startswith("Here is everything in coffee + tea (drinks):")
    assert "- Double Espresso: ingredients: coffee; price: £2.50." in text
    # The description is said once: it is also the first "ingredient" in the corpus.
    assert "- Iced Latte: sweeten with cane syrup; ingredients: milk, coffee; price: £2.50." in text
    assert "\n- Plain Water\n" in text  # nothing is known about it, so just its name
    assert text.endswith("Would you like to know more about any of these?")


def test_a_category_listing_returns_a_card_for_each_item_that_has_an_image(
    detailed: MenuCatalog,
) -> None:
    cards = browse_answer(detailed, "drinks", "coffee + tea").cards

    assert [c["name"] for c in cards] == ["Double Espresso", "Iced Latte"]  # not Plain Water
    latte = cards[1]
    assert latte == {
        "id": "id-Iced Latte",
        "slug": "iced-latte",
        "name": "Iced Latte",
        "description": "sweeten with cane syrup",
        "ingredients": ["sweeten with cane syrup", "milk", "coffee"],
        "price_gbp": 2.5,
        "image": "iced-latte.png",
    }


def test_choosing_a_group_with_one_category_also_lists_items_with_cards(
    detailed: MenuCatalog,
) -> None:
    only = build_catalog(
        [
            {
                "id": "x",
                "slug": "chillies",
                "item_type": "menu_item",
                "name": "Chillies",
                "category_path": ["extras"],
                "price_gbp": 1.0,
                "image": "c.png",
            }
        ]
    )

    answer = browse_answer(only, "extras", None)

    assert answer.text.startswith("Here is everything in extras:")
    assert [c["name"] for c in answer.cards] == ["Chillies"]


@pytest.mark.parametrize(
    ("group", "category"),
    [(None, None), ("sides", None), (None, "ramen"), ("nowhere", None), (None, "pizza")],
)
def test_lists_of_groups_and_categories_and_questions_carry_no_cards(
    detailed: MenuCatalog, group: str | None, category: str | None
) -> None:
    assert browse_answer(detailed, group, category).cards == []


def test_the_unaltered_text_helpers_agree_with_the_answer(detailed: MenuCatalog) -> None:
    answer = browse_answer(detailed, "sides", "gyoza")

    assert browse_reply(detailed, "sides", "gyoza") == answer.text
    understanding = {"intent": "menu_browse", "browse_group": "sides", "browse_category": "gyoza"}
    assert direct_reply(understanding, detailed) == answer.text
    assert direct_answer(understanding, detailed).cards == answer.cards


def test_greetings_and_off_topic_replies_carry_no_cards(detailed: MenuCatalog) -> None:
    assert direct_answer({"intent": "greeting"}, detailed).cards == []
    assert direct_answer({"intent": "off_topic"}, detailed).cards == []


def test_each_sentence_of_the_description_is_said_once_but_real_ingredients_stay() -> None:
    """The corpus splits a description into fragments and lists them as ingredients. A sentence
    is not repeated among the ingredients; an ingredient the description merely mentions is."""
    catalog = build_catalog(
        [
            {
                "item_type": "menu_item",
                "name": "Duck gyoza",
                "category_path": ["sides", "gyoza"],
                "description": "fried. sweet cherry hoisin sauce",
                "ingredients": ["fried", "sweet cherry hoisin sauce", "wheat", "soya"],
                "price_gbp": 9.35,
            },
            {
                "item_type": "menu_item",
                "name": "Americano",
                "category_path": ["drinks", "coffee"],
                "description": "served black or with milk",
                "ingredients": ["served black or with milk", "milk", "coffee"],
                "price_gbp": 2.5,
            },
        ]
    )
    duck = browse_answer(catalog, "sides", "gyoza").text
    americano = browse_answer(catalog, "drinks", "coffee").text

    duck_line = (
        "- Duck gyoza: fried. sweet cherry hoisin sauce; ingredients: wheat, soya; price: £9.35."
    )
    assert duck_line in duck
    americano_line = (
        "- Americano: served black or with milk; ingredients: milk, coffee; price: £2.50."
    )
    assert americano_line in americano
