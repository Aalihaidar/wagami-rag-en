#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Reorganise data/images/ into a menu/<category>/<sub-category>/ tree, moving each dish photo out
of the flat top level into its place. "Category" is the app's **group** (category_path[0]:
drinks, kids, ...), "sub-category" is its **category** field (cocktails, ramen, ...) -- see
app/agent/catalog.py's own vocabulary note. Folder names are lower-cased with "-" for spaces (and
any other run of non-alphanumeric characters), via app.agent.choice_images.slugify() for the
group and each row's own `category_slug` for the sub-category, so this can never disagree with
those two other places.

Before nesting: deletes every one of the 25 flat <category_slug>/ folders
scripts/organize_images_by_category.py made, and data/images/thumbs/ if scripts/make_thumbnails.py
was ever run -- both by exact name, never a general "remove every subfolder" sweep, so it can't
take a categories/ folder of hand-uploaded group/category pictures (or anything else) with it.

After this runs, no photo is left at the flat data/images/<file>.png location any more. The app
builds each photo's nested path from the row's category_path
(app/agent/choice_images.py's dish_image_path()), with IMAGE_BASE_URL pointing at menu/. Run
scripts/flatten_same_named_categories.py next. This script only moves files.

Usage:
    uv run python scripts/nest_images_by_category.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent.choice_images import slugify  # noqa: E402

KNOWLEDGE_BASE = Path(__file__).resolve().parent.parent / "data" / "knowledge_base.json"
IMAGES_DIR = Path(__file__).resolve().parent.parent / "data" / "images"
MENU_DIR = IMAGES_DIR / "menu"
THUMBS_DIR = IMAGES_DIR / "thumbs"
# The exact 25 folders scripts/organize_images_by_category.py creates (one per category_slug in
# the corpus this repo ships with) -- named explicitly, not swept, so a folder this script does
# not recognise (categories/, or anything else) is left alone.
FLAT_CATEGORY_FOLDERS = [
    "bao-buns",
    "beers-cider",
    "big-flavour-bites",
    "buldak",
    "cocktails",
    "coffee-tea",
    "curries",
    "desserts",
    "desserts-sweet-treats",
    "donburi",
    "drinks",
    "extras",
    "freshly-made-juices",
    "gyoza",
    "katsu",
    "lighter-bites",
    "lunch-time",
    "noodles",
    "ramen",
    "rice",
    "sides",
    "soft-drinks",
    "teppanyaki",
    "the-main-event",
    "wine-sake",
]


def remove_tree(path: Path) -> bool:
    if not path.exists():
        return False
    for child in sorted(path.rglob("*"), reverse=True):
        if child.is_file():
            child.unlink()
        else:
            child.rmdir()
    path.rmdir()
    return True


def main() -> int:
    if not KNOWLEDGE_BASE.exists():
        print(f"Not found: {KNOWLEDGE_BASE}", file=sys.stderr)
        return 1

    removed = [name for name in FLAT_CATEGORY_FOLDERS if remove_tree(IMAGES_DIR / name)]
    thumbs_removed = remove_tree(THUMBS_DIR)
    print(
        f"removed {len(removed)} flat category folder(s)"
        + (" (thumbs/ too)" if thumbs_removed else "")
    )

    with KNOWLEDGE_BASE.open(encoding="utf-8") as fh:
        rows = json.load(fh)

    moved = already_nested = 0
    missing: list[str] = []
    for row in rows:
        if row.get("item_type") != "menu_item":
            continue
        image = row.get("image")
        slug = row.get("category_slug")
        path = row.get("category_path")
        if not image or not slug or not isinstance(path, list) or not path:
            continue
        target_dir = MENU_DIR / slugify(str(path[0])) / slug
        target = target_dir / image
        source = IMAGES_DIR / image
        if target.exists():
            already_nested += 1
            continue
        if not source.exists():
            missing.append(image)
            continue
        target_dir.mkdir(parents=True, exist_ok=True)
        source.rename(target)
        moved += 1

    print(f"{moved} photo(s) moved into {MENU_DIR}, {already_nested} already nested")
    if missing:
        print(
            f"{len(missing)} row(s) named a photo that wasn't at the flat location:",
            file=sys.stderr,
        )
        for name in missing:
            print(f"  {name}", file=sys.stderr)

    leftover = sorted(p.name for p in IMAGES_DIR.glob("*.png"))
    if leftover:
        print(
            f"{len(leftover)} photo(s) still at the flat top level (no matching corpus row?):",
            file=sys.stderr,
        )
        for name in leftover:
            print(f"  {name}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
