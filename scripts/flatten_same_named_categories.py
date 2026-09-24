#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Collapse a redundant menu/<name>/<name>/ nesting under data/images/menu/ (built by
scripts/nest_images_by_category.py): where a group's only sub-category folder shares its exact
name (a group with one sub-category of the same name -- "extras", "lunch time", "desserts + sweet
treats" in this corpus), move that sub-category's photos up into the group folder directly and
remove the now-empty sub-category folder, so menu/extras/extras/*.png becomes menu/extras/*.png.

Driven by what is actually on disk, not a hardcoded list of names, so it stays correct if the
corpus's groups/categories change. A group with more than one sub-category is never touched, even
if by chance one of them shared the group's name.

Usage:
    uv run python scripts/flatten_same_named_categories.py
"""

import sys
from pathlib import Path

MENU_DIR = Path(__file__).resolve().parent.parent / "data" / "images" / "menu"


def main() -> int:
    if not MENU_DIR.exists():
        print(f"Not found: {MENU_DIR}", file=sys.stderr)
        return 1

    flattened = 0
    for group_dir in sorted(p for p in MENU_DIR.iterdir() if p.is_dir()):
        same_named = group_dir / group_dir.name
        if not same_named.is_dir():
            continue
        photos = sorted(same_named.glob("*.png"))
        for photo in photos:
            photo.rename(group_dir / photo.name)
        same_named.rmdir()
        flattened += 1
        print(
            f"{same_named.relative_to(MENU_DIR.parent.parent)} -> "
            f"{group_dir.relative_to(MENU_DIR.parent.parent)}/ ({len(photos)} photo(s))"
        )

    print(f"\n{flattened} folder(s) flattened")
    return 0


if __name__ == "__main__":
    sys.exit(main())
