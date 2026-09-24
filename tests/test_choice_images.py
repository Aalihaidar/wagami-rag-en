"""Paths in the menu image tree: dish photos and group/category covers (rule C-23)."""

from app.agent.choice_images import (
    COVER_NAME,
    category_image_filename,
    dish_image_path,
    group_image_filename,
    image_folder,
    slugify,
)


def test_slugify_lower_cases_and_hyphenates() -> None:
    assert slugify("Wine + Sake") == "wine-sake"
    assert slugify("The Main Event") == "the-main-event"
    assert slugify("gyoza") == "gyoza"


def test_slugify_collapses_any_run_of_punctuation_to_one_hyphen() -> None:
    assert slugify("Coffee + Tea!!") == "coffee-tea"
    assert slugify("  Sides  ") == "sides"
    assert slugify("Bao / Buns") == "bao-buns"


def test_a_category_folder_is_inside_its_groups_folder() -> None:
    assert image_folder("drinks") == "drinks"
    assert image_folder("drinks", "wine + sake") == "drinks/wine-sake"


def test_a_category_named_like_its_group_is_the_groups_own_folder() -> None:
    """extras, lunch time and desserts + sweet treats have one category of their own name, whose
    photos sit straight in the group's folder."""
    assert image_folder("lunch time", "lunch time") == "lunch-time"


def test_a_groups_card_shows_the_cover_in_its_folder() -> None:
    assert group_image_filename("drinks") == "drinks/cover.png"
    assert group_image_filename("the main event") == "the-main-event/cover.png"


def test_a_categorys_card_shows_the_cover_in_its_folder() -> None:
    assert category_image_filename("drinks", "cocktails") == "drinks/cocktails/cover.png"
    assert category_image_filename("Drinks", "Coffee + Tea") == "drinks/coffee-tea/cover.png"


def test_a_group_and_a_category_with_the_same_name_get_different_covers() -> None:
    """ "drinks" is both a group and, under "kids", a category -- a guest browsing "kids" must
    not see the top-level drinks group's own picture."""
    assert group_image_filename("drinks") != category_image_filename("kids", "drinks")
    assert category_image_filename("kids", "drinks") == f"kids/drinks/{COVER_NAME}"


def test_two_categories_that_share_a_name_get_their_own_covers() -> None:
    """ "ramen" is a category under both "kids" and "the main event"; each has its own folder."""
    assert category_image_filename("kids", "ramen") == "kids/ramen/cover.png"
    assert category_image_filename("the main event", "ramen") == "the-main-event/ramen/cover.png"


def test_a_dish_photo_is_in_its_rows_category_folder() -> None:
    assert dish_image_path(["drinks", "cocktails"], "roku.png") == "drinks/cocktails/roku.png"
    assert dish_image_path(["extras"], "chillies.png") == "extras/chillies.png"


def test_a_dish_with_no_category_path_keeps_its_bare_filename() -> None:
    assert dish_image_path([], "x.png") == "x.png"
