#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Copy every dish photo in data/images/ into a subfolder named after its category, for browsing:
data/images/<category_slug>/<the dish's own image file>.

Reads which photo belongs to which category from data/knowledge_base.json's own `category_slug`
field (each menu row's own value, not recomputed), so the folders always match the corpus. A
category that spans two groups with the same name (`ramen` under both `kids` and `the main
event`; `desserts + sweet treats` under both its own group and `gluten free`) gets one folder
with every dish from both.

Copies, not moves: nothing else in the app reads a photo from a category subfolder (a menu row's
`image` field is still a bare filename, and app/main.py's `_image_url()` still appends it directly
to the flat folder), so leaving the originals in place at the top level keeps the app working
exactly as it does today. This is purely an extra, organized view for browsing by hand.

Usage:
    uv run python scripts/organize_images_by_category.py
"""

import json
import shutil
import sys
from pathlib import Path

KNOWLEDGE_BASE = Path(__file__).resolve().parent.parent / "data" / "knowledge_base.json"
IMAGES_DIR = Path(__file__).resolve().parent.parent / "data" / "images"


def main() -> int:
    if not KNOWLEDGE_BASE.exists():
        print(f"Not found: {KNOWLEDGE_BASE}", file=sys.stderr)
        return 1
    with KNOWLEDGE_BASE.open(encoding="utf-8") as fh:
        rows = json.load(fh)

    copied = skipped_existing = 0
    missing: list[str] = []
    folders: set[str] = set()

    for row in rows:
        if row.get("item_type") != "menu_item":
            continue
        image = row.get("image")
        slug = row.get("category_slug")
        if not image or not slug:
            continue
        source = IMAGES_DIR / image
        if not source.exists():
            missing.append(image)
            continue
        folder = IMAGES_DIR / slug
        folder.mkdir(exist_ok=True)
        folders.add(slug)
        target = folder / image
        if target.exists():
            skipped_existing += 1
            continue
        shutil.copy2(source, target)
        copied += 1

    print(f"{len(folders)} category folders under {IMAGES_DIR}")
    print(f"{copied} photo(s) copied, {skipped_existing} already there")
    if missing:
        print(f"{len(missing)} row(s) named a photo that isn't in {IMAGES_DIR}:", file=sys.stderr)
        for name in missing:
            print(f"  {name}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
