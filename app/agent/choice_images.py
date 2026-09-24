"""Where every picture lives in the image tree: a dish's photo, and the cover picture of a menu
group or category (rule C-23).

The pictures are arranged by the menu's own structure, under a `menu/` folder
(`data/images/menu/` locally; `IMAGE_BASE_URL` points at the same folder once deployed, so every
path here is relative to it):

    <group>/<category>/<dish photo>      a group with several categories (drinks/cocktails/...)
    <group>/<dish photo>                 a group whose one category has its own name (extras/...)
    <group>/cover.png                    the group's card in the menu overview
    <group>/<category>/cover.png         the category's card in its group's list of categories

Folder names are slugify()'d group and category names. A category's folder sits inside its
group's, so two categories that share a name ("ramen" under both "kids" and "the main event")
each have their own picture. This module is the one place that turns names into these paths, so
nothing else has to agree on the convention. A corpus row's own `image` field is only the dish
photo's bare filename; its folder comes from the row's `category_path`.

A picture that has not been uploaded is not an error: the chat page drops a picture that fails to
load and shows the card with just its name (see chat.js's buildChoiceCard()).
"""

import re
from collections.abc import Sequence

# The root of the picture tree, inside data/images/ (and what IMAGE_BASE_URL points at).
MENU_IMAGE_DIR = "menu"
# Each group and category folder's own card picture.
COVER_NAME = "cover.png"


def slugify(name: str) -> str:
    """Lower-cased, hyphen-joined form of `name`, safe for a folder name: runs of anything that
    isn't a letter or digit collapse to one hyphen, and a leading or trailing one is dropped.
    "Wine + Sake" -> "wine-sake". Matches this corpus's own `category_slug` convention."""
    return re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")


def image_folder(group: str, category: str | None = None) -> str:
    """The folder of a group, or of one of its categories. A category named like its group is
    the group's folder itself (menu/extras/, not menu/extras/extras/)."""
    group_slug = slugify(group)
    if category is None or slugify(category) == group_slug:
        return group_slug
    return f"{group_slug}/{slugify(category)}"


def group_image_filename(group: str) -> str:
    """The picture for a menu group's card in the menu overview."""
    return f"{image_folder(group)}/{COVER_NAME}"


def category_image_filename(group: str, category: str) -> str:
    """The picture for a category's card in its group's list of categories."""
    return f"{image_folder(group, category)}/{COVER_NAME}"


def dish_image_path(category_path: Sequence[str], image: str) -> str:
    """Where a dish's photo is, from its row's `category_path` (group first, category last) and
    its `image` filename. A row with no path keeps the bare filename."""
    if not category_path:
        return image
    return f"{image_folder(category_path[0], category_path[-1])}/{image}"
