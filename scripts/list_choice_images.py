#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
List the pictures the chat page needs: the cover of each menu group and of each category a guest
can browse to (rule C-23 in docs/LLM_RULES.md), and every dish photo, at their paths in the
menu/ image tree (app/agent/choice_images.py) -- and say which are missing from data/images/menu/.

Reads the actual `data/knowledge_base.json`, so the list always matches whatever corpus is
loaded rather than a hand-written one going stale the moment the menu's structure changes. A
group with exactly one category goes straight to that category's item listing (R-05), so it shows
no category card, and only its own cover is needed.

Nothing here reaches Weaviate, Groq or any other live service; it only reads local files.

Usage:
    uv run python scripts/list_choice_images.py
    uv run python scripts/list_choice_images.py --knowledge-base path/to/knowledge_base.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent.browse import browse_answer  # noqa: E402
from app.agent.catalog import build_catalog  # noqa: E402
from app.agent.choice_images import MENU_IMAGE_DIR, group_image_filename  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_KNOWLEDGE_BASE = ROOT / "data" / "knowledge_base.json"
MENU_DIR = ROOT / "data" / "images" / MENU_IMAGE_DIR


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--knowledge-base", type=Path, default=DEFAULT_KNOWLEDGE_BASE)
    args = parser.parse_args()

    if not args.knowledge_base.exists():
        print(f"Not found: {args.knowledge_base}", file=sys.stderr)
        return 1
    with args.knowledge_base.open(encoding="utf-8") as fh:
        catalog = build_catalog(json.load(fh))
    if catalog.is_empty:
        print(f"No menu rows in {args.knowledge_base}", file=sys.stderr)
        return 1

    covers: list[str] = []
    for group in catalog.groups:
        covers.append(group_image_filename(group))
        answer = browse_answer(catalog, group, None)
        if answer.choices:
            covers.extend(card["image"] for card in answer.choices["cards"])
    photos = sorted(
        {
            item.image
            for group in catalog.groups
            for category in catalog.categories_of(group)
            for item in catalog.item_details_of(group, category)
            if item.image
        }
    )

    print(f"Paths are relative to <IMAGE_BASE_URL> (data/images/{MENU_IMAGE_DIR}/ locally).\n")
    print(f"Covers -- {len(covers)} files:")
    for path in covers:
        print(f"  {path}")
    missing = [p for p in [*covers, *photos] if not (MENU_DIR / p).is_file()]
    print(f"\nDish photos -- {len(photos)} files.")
    if missing:
        print(f"\nMissing from {MENU_DIR} -- {len(missing)} files:")
        for path in missing:
            print(f"  {path}")
    else:
        print(f"\nAll {len(covers) + len(photos)} files are present in {MENU_DIR}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
