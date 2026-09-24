"""Item cards: which dishes an answer contains (rules R-13 and C-21 in docs/LLM_RULES.md)."""

from typing import Any

from app.agent.cards import card_for_item, card_for_row, cards_for_answer, cited_items_from_ranked
from app.agent.catalog import NUTRITION_FIELDS, MenuItem
from app.retrieval import MenuRow, RerankHit


def make_row(
    name: str,
    *,
    item_type: str = "menu_item",
    slug: str | None = None,
    description: str | None = "tasty",
    ingredients: list[str] | None = None,
    price_gbp: float | None = 9.5,
    image: str = "",
    dietary_tags: list[str] | None = None,
    allergens_contains: list[str] | None = None,
    allergens_may_contain: list[str] | None = None,
    **details: Any,
) -> MenuRow:
    return {
        "uuid": f"uuid-{name}",
        "score": 0.5,
        "properties": {
            "name": name,
            "slug": slug or name.lower().replace(" ", "-"),
            "item_type": item_type,
            "description": description,
            "ingredients": ingredients or [],
            "price_gbp": price_gbp,
            "image": image,
            "dietary_tags": dietary_tags or [],
            "allergens_contains": allergens_contains or [],
            "allergens_may_contain": allergens_may_contain or [],
            **details,
        },
    }


def make_hit(row: MenuRow) -> RerankHit:
    return {"row": row, "rerank": 0.9, "hybrid": 0.9}


def hit(name: str, *, image: str = "x.png", **kwargs: Any) -> RerankHit:
    return make_hit(make_row(name, image=image, **kwargs))


def slugs_of(items: list[Any]) -> list[str]:
    return [i["slug"] for i in items]


def cards(
    answer: str,
    ranked: list[RerankHit],
    cited: list[str] | None = None,
    *,
    screened_out: str | None = None,
) -> list[str]:
    return slugs_of(
        cards_for_answer(answer, ranked=ranked, cited_slugs=cited or [], screened_out=screened_out)
    )


# ---- cited_items_from_ranked: the cards for a set of slugs ---------------------------------------


def test_cited_items_from_ranked_includes_description_ingredients_and_price() -> None:
    """The card carries name/description/ingredients/price/image plus the rest of the dish's row
    -- dietary tags, allergens, nutrition, category, portion, ABV -- for the single-dish detail
    view (rule C-26). A listing's or a multi-dish answer's gallery card still only *displays* the
    first group -- the model can also state any of this in the answer text itself (Section 4)."""
    ramen = make_row(
        "vegan ramen",
        description="Rich miso broth.",
        ingredients=["tofu", "miso", "soya"],
        price_gbp=9.5,
        image="r.png",
        dietary_tags=["vegan", "vegetarian"],
        allergens_contains=["soya"],
        category="ramen",
        category_path=["the main event", "ramen"],
        kcal=512.0,
        protein_g=20.1,
        salt_g=4,  # an int from the store still comes out as a float
        is_gluten_free_listed=True,
        portion_value=1.0,
        portion_unit="ea",
        servings="1",
        abv_percent=None,
    )
    espresso = make_row(
        "double espresso", description=None, ingredients=["coffee"], price_gbp=2.5, image="e.png"
    )
    faq = make_row("what time do you open", item_type="faq", image="")  # never cited
    no_image = make_row("no image dish", image="")  # never cited -- nothing to show a card for

    items = cited_items_from_ranked(
        [make_hit(ramen), make_hit(espresso), make_hit(faq), make_hit(no_image)],
        ["vegan-ramen", "double-espresso"],
    )

    assert items == [
        {
            "id": "uuid-vegan ramen",
            "slug": "vegan-ramen",
            "name": "vegan ramen",
            "description": "Rich miso broth.",
            "ingredients": ["tofu", "miso", "soya"],
            "price_gbp": 9.5,
            "image": "the-main-event/ramen/r.png",
            "dietary_tags": ["vegan", "vegetarian"],
            "allergens_contains": ["soya"],
            "allergens_may_contain": [],
            "category": "ramen",
            "category_path": ["the main event", "ramen"],
            "nutrition": {
                **{name: None for name in NUTRITION_FIELDS},
                "kcal": 512.0,
                "protein_g": 20.1,
                "salt_g": 4.0,
            },
            "is_gluten_free_listed": True,
            "portion_value": 1.0,
            "portion_unit": "ea",
            "servings": "1",
            "abv_percent": None,
        },
        {
            "id": "uuid-double espresso",
            "slug": "double-espresso",
            "name": "double espresso",
            "description": None,  # omitted card line, not a placeholder string
            "ingredients": ["coffee"],
            "price_gbp": 2.5,
            "image": "e.png",
            "dietary_tags": [],
            "allergens_contains": [],
            "allergens_may_contain": [],
            "category": None,
            "category_path": [],
            "nutrition": {name: None for name in NUTRITION_FIELDS},
            "is_gluten_free_listed": False,
            "portion_value": None,
            "portion_unit": None,
            "servings": None,
            "abv_percent": None,
        },
    ]


def test_card_for_item_carries_dietary_tags_and_allergens() -> None:
    """The direct-route card (app/agent/browse.py's listings) needs the same fields as a search
    answer's card, for the single-dish detail view (rule C-26)."""
    nutrition = tuple((name, 1.5 if name == "kcal" else None) for name in NUTRITION_FIELDS)
    item = MenuItem(
        id="id-1",
        slug="vegan-ramen",
        name="Vegan Ramen",
        description="Rich miso broth.",
        ingredients=("tofu", "miso", "soya"),
        price_gbp=9.5,
        image="drinks/ramen/vegan-ramen.png",
        dietary_tags=("vegan", "vegetarian"),
        allergens_contains=("soya",),
        allergens_may_contain=("sesame",),
        category="ramen",
        category_path=("the main event", "ramen"),
        nutrition=nutrition,
        is_gluten_free_listed=True,
        portion_value=1.0,
        portion_unit="ea",
        servings="1",
        abv_percent=0.0,
    )

    assert card_for_item(item) == {
        "id": "id-1",
        "slug": "vegan-ramen",
        "name": "Vegan Ramen",
        "description": "Rich miso broth.",
        "ingredients": ["tofu", "miso", "soya"],
        "price_gbp": 9.5,
        "image": "drinks/ramen/vegan-ramen.png",
        "dietary_tags": ["vegan", "vegetarian"],
        "allergens_contains": ["soya"],
        "allergens_may_contain": ["sesame"],
        "category": "ramen",
        "category_path": ["the main event", "ramen"],
        "nutrition": dict(nutrition),
        "is_gluten_free_listed": True,
        "portion_value": 1.0,
        "portion_unit": "ea",
        "servings": "1",
        "abv_percent": 0.0,
    }


def test_card_for_item_with_no_image_is_none() -> None:
    item = MenuItem(id="id-1", slug="no-photo", name="No Photo")

    assert card_for_item(item) is None


def test_cited_items_from_ranked_excludes_dishes_not_cited_by_the_model() -> None:
    """A reply about one dish must not surface cards for every other reranked candidate --
    the bug this filter exists to fix. Only rows the generation call's own `cited_slugs`
    output names get a card, so a dish the model retrieved but never actually discussed is
    excluded even though it's still one of the reranked hits."""
    espresso = make_row(
        "double espresso", description=None, ingredients=["coffee"], price_gbp=2.5, image="e.png"
    )
    latte = make_row(
        "latte - whole milk", ingredients=["milk", "coffee"], price_gbp=2.5, image="l.png"
    )

    items = cited_items_from_ranked([make_hit(espresso), make_hit(latte)], ["double-espresso"])

    assert [item["name"] for item in items] == ["double espresso"]


def test_cited_items_from_ranked_includes_a_dish_referred_to_implicitly() -> None:
    """`cited_slugs` is how a pronoun/implicit reference ("it", "that one") back to a dish
    already named still gets a card -- the model resolves the reference itself and reports
    the slug, rather than this function trying to detect it from the answer text."""
    espresso = make_row(
        "double espresso", description=None, ingredients=["coffee"], price_gbp=2.5, image="e.png"
    )

    items = cited_items_from_ranked(
        [make_hit(espresso)], ["double-espresso"]
    )  # e.g. answer: "It's £2.50." -- no literal name in the text at all

    assert [item["name"] for item in items] == ["double espresso"]


def test_cited_items_from_ranked_gives_one_card_per_slug() -> None:
    """The same dish listed under two menu sections is two rows with one slug: one card."""
    first = make_row("Edamame", slug="edamame", image="edamame.png")
    second = make_row("Edamame", slug="edamame", image="edamame.png")

    items = cited_items_from_ranked([make_hit(first), make_hit(second)], ["edamame"])

    assert [i["slug"] for i in items] == ["edamame"]


def test_cited_items_from_ranked_ignores_a_slug_not_in_ranked() -> None:
    """Defense-in-depth: even if a malformed/hallucinated slug slipped past the json_schema
    enum constraint, a slug that doesn't match any reranked row must never produce a card."""
    espresso = make_row(
        "double espresso", description=None, ingredients=["coffee"], price_gbp=2.5, image="e.png"
    )

    items = cited_items_from_ranked([make_hit(espresso)], ["not-a-real-slug"])

    assert items == []


# ---- layer 1: the model's citation ---------------------------------------------------------------


def test_a_cited_dish_the_answer_only_refers_to_as_it_gets_its_card() -> None:
    assert cards("It is £2.50.", [hit("Double Espresso")], ["double-espresso"]) == [
        "double-espresso"
    ]


def test_a_cited_slug_that_was_not_retrieved_and_is_not_named_gets_no_card() -> None:
    assert cards("Here you go.", [hit("Double Espresso")], ["iced-latte"]) == []


# ---- layer 2: a full dish name in the answer -----------------------------------------------------


def test_a_dish_the_answer_names_gets_its_card_even_if_it_was_not_cited() -> None:
    assert cards("The Double Espresso is £2.50.", [hit("Double Espresso")]) == ["double-espresso"]


def test_retrieved_dishes_the_answer_does_not_talk_about_get_no_card() -> None:
    ranked = [hit("Double Espresso"), hit("Iced Latte")]

    assert cards("The Iced Latte is £3.", ranked) == ["iced-latte"]


def test_a_name_inside_a_longer_dish_name_is_not_a_mention() -> None:
    ranked = [hit("Coke"), hit("Diet Coke")]

    assert cards("A Diet Coke is £2.", ranked) == ["diet-coke"]


def test_every_dish_a_comparison_names_gets_a_card_in_relevance_order() -> None:
    ranked = [hit("Double Espresso"), hit("Iced Latte")]

    answer = "The Iced Latte has more milk than the Double Espresso."
    assert cards(answer, ranked) == ["double-espresso", "iced-latte"]


def test_faq_rows_and_dishes_without_an_image_never_get_a_card() -> None:
    ranked = [hit("Do you take bookings", item_type="faq"), hit("Plain Water", image="")]

    assert cards("Do you take bookings? Plain Water is free.", ranked, ["plain-water"]) == []


def test_a_dish_the_search_did_not_return_gets_no_card_even_if_the_answer_names_it() -> None:
    """Dish descriptions list components that are themselves menu items ("katsu curry sauce"), so
    only this turn's retrieved dishes are looked for in an answer."""
    ranked = [hit("Chicken Katsu Curry")]
    answer = "The Chicken Katsu Curry comes with katsu curry sauce or a Miso Soup."

    assert cards(answer, ranked) == ["chicken-katsu-curry"]


def test_a_dish_named_only_as_a_component_of_another_named_dish_gets_no_card() -> None:
    """Live: the answer about the katsu curry lists the sauce among its ingredients, and the sauce
    is itself a menu item that was retrieved."""
    curry = hit("Chicken Katsu Curry", ingredients=["panko chicken", "katsu curry sauce"])
    sauce = hit("Katsu Curry Sauce")
    answer = "Chicken Katsu Curry: panko chicken, katsu curry sauce. Price: £16.45."

    assert cards(answer, [curry, sauce]) == ["chicken-katsu-curry"]


def test_a_component_the_model_cited_still_gets_its_card() -> None:
    curry = hit("Chicken Katsu Curry", ingredients=["panko chicken", "katsu curry sauce"])
    sauce = hit("Katsu Curry Sauce")
    answer = "The Chicken Katsu Curry has katsu curry sauce; the sauce is also sold on its own."

    assert cards(answer, [curry, sauce], ["katsu-curry-sauce"]) == [
        "chicken-katsu-curry",
        "katsu-curry-sauce",
    ]


def test_two_dishes_that_do_not_contain_each_other_both_get_a_card() -> None:
    ranked = [
        hit("Double Espresso", ingredients=["coffee"]),
        hit("Iced Latte", ingredients=["milk"]),
    ]

    assert cards("The Double Espresso and the Iced Latte.", ranked) == [
        "double-espresso",
        "iced-latte",
    ]


def test_the_same_dish_listed_twice_gets_one_card() -> None:
    ranked = [hit("Edamame", slug="edamame"), hit("Edamame", slug="edamame")]

    assert cards("Edamame is £6.", ranked) == ["edamame"]


# ---- a dish a filter screened out never gets a card ----------------------------------------------


def test_a_screened_out_dish_gets_no_card_even_when_the_answer_explains_why() -> None:
    ranked = [hit("Vegan Ramen")]
    answer = "The Chicken Katsu Curry contains milk, so it isn't suitable. Try the Vegan Ramen."

    assert cards(answer, ranked, screened_out="Chicken Katsu Curry") == ["vegan-ramen"]


def test_a_shortened_name_inside_the_screened_out_dishs_name_is_not_a_mention() -> None:
    """Asked for vegan, the search screened out "Katsu Curry" and kept "Katsu Curry (vegan
    recipe)". A sentence about the first must not put the second one's card under it."""
    ranked = [hit("Katsu Curry (vegan recipe)")]
    answer = "The Katsu Curry is not tagged vegan."

    assert cards(answer, ranked, screened_out="Katsu Curry") == []
    assert cards(
        "The Katsu Curry (vegan recipe) is vegan.", ranked, screened_out="Katsu Curry"
    ) == ["katsu-curry-(vegan-recipe)"]


# ---- layer 3: a shortened name -------------------------------------------------------------------


def test_a_shortened_dish_name_gets_its_card() -> None:
    ranked = [hit("Gochujang Pork Belly Ramen (gluten-free recipe)")]

    assert cards("The pork belly ramen is £12.", ranked) == [
        "gochujang-pork-belly-ramen-(gluten-free-recipe)"
    ]


def test_a_shortened_name_two_dishes_share_is_not_guessed() -> None:
    ranked = [hit("Chicken Katsu Curry"), hit("Vegan Katsu Curry")]

    assert cards("The katsu curry is £12.", ranked) == []


def test_a_shortened_name_that_is_one_of_the_dishes_ingredients_is_not_a_mention() -> None:
    ranked = [hit("Yasai Ramen Rice Noodles", ingredients=["rice noodles", "tofu"])]

    assert cards("Made with rice noodles and tofu.", ranked) == []


def test_recipe_wording_alone_is_not_a_shortened_name() -> None:
    ranked = [hit("Pork Ramen (gluten-free recipe)"), hit("Vegan Curry (vegan recipe)")]

    assert cards("Everything here is gluten free, or a vegan recipe.", ranked) == []


# ---- nothing is guessed when the answer names no dish and the model cited none ------------------


def test_an_answer_that_says_it_and_names_nothing_gets_no_card() -> None:
    ranked = [hit("Double Espresso"), hit("Iced Latte")]

    assert cards("It's £2.50 and has no milk.", ranked) == []


def test_an_answer_that_names_no_dish_gets_no_card_from_the_search_alone() -> None:
    assert cards("Thanks for asking.", [hit("Double Espresso")]) == []
    assert cards("It is £2.50.", []) == []


def test_a_cards_image_is_the_photo_in_its_rows_category_folder() -> None:
    """A row's `image` is a bare filename; the photo sits in the folder of its group and
    category, which the card's image path carries."""
    row = make_row("Tofu Firecracker", image="tofu-firecracker.png")
    row["properties"]["category_path"] = ["the main event", "curries"]

    card = card_for_row(row)

    assert card is not None
    assert card["image"] == "the-main-event/curries/tofu-firecracker.png"
