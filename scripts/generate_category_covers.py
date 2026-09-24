#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Build one representative `cover.png` for every category and sub-category folder under
data/images/menu/, from the dish photos already there. Driven entirely by what's on disk (a
category is a top-level folder under menu/, a sub-category is one of its subfolders), not a
hardcoded list, so it stays correct if the tree changes.

For a sub-category folder (or a category folder with no sub-category subfolders of its own, e.g.
one a guest browses straight into -- see scripts/nest_images_by_category.py): pick its four
highest-resolution dish photos.

For a category folder that does have sub-category subfolders: pick one photo -- the
highest-resolution one -- from each sub-category, preferring the four sub-categories whose own
best photo is highest-resolution if there are more than four. If that comes up short of four
(a category with fewer than four sub-categories), fill the rest with the next-best photos from
anywhere in the category.

Either way, once four candidate photos are chosen (repeating one if a folder truly doesn't have
four distinct photos at all, so the 2x2 layout below always has something in every quadrant):
resize all four down to the smallest one's own size (never upscale), lay them out in a 2x2 square
with nothing cropped, then shrink that square back down to one photo's size -- so `cover.png`
ends up a small collage of four, at a normal photo's own resolution rather than four times it.

`cover.png` is the fixed name this writes and always skips reading back (a re-run never treats an
earlier run's own cover as one of the four source photos). It is the picture the chat app shows
on each group's and category's card (app/agent/choice_images.py's COVER_NAME).

Usage:
    uv run python scripts/generate_category_covers.py
"""

import sys
from pathlib import Path

from PIL import Image

MENU_DIR = Path(__file__).resolve().parent.parent / "data" / "images" / "menu"
COVER_NAME = "cover.png"
TILES = 4


def real_photos(folder: Path) -> list[Path]:
    """The dish photos directly in `folder`, excluding any cover this script already wrote."""
    return sorted(p for p in folder.glob("*.png") if p.name != COVER_NAME)


def rank(path: Path) -> tuple[int, int, str]:
    """Sort key for "best (highest pixel area, then larger file, then name) first"."""
    with Image.open(path) as im:
        area = im.width * im.height
    return (-area, -path.stat().st_size, path.name)


def by_resolution(paths: list[Path]) -> list[Path]:
    return sorted(paths, key=rank)


def pick(candidates: list[Path], n: int = TILES) -> list[Path]:
    """The best `n` of `candidates`, best first; repeats (cycling) if fewer than `n` exist, so
    there is always something for every quadrant of the 2x2 layout."""
    if not candidates:
        return []
    ranked = by_resolution(candidates)
    return [ranked[i % len(ranked)] for i in range(n)]


def choose_photos(folder: Path) -> list[Path]:
    subcategories = sorted(p for p in folder.iterdir() if p.is_dir())
    if not subcategories:
        return pick(real_photos(folder))

    per_subcategory = []
    for sub in subcategories:
        photos = real_photos(sub)
        if photos:
            per_subcategory.append((by_resolution(photos)[0], photos))
    per_subcategory.sort(key=lambda entry: rank(entry[0]))  # best-photo-first order
    selected = [best for best, _ in per_subcategory[:TILES]]

    if len(selected) < TILES:
        pool = [p for _, photos in per_subcategory for p in photos if p not in selected]
        for photo in by_resolution(pool):
            if len(selected) >= TILES:
                break
            selected.append(photo)

    return pick(selected) if selected else []


def make_cover(photos: list[Path], target: Path) -> None:
    images = [Image.open(p).convert("RGBA") for p in photos]
    try:
        side = min(min(im.width, im.height) for im in images)
        resized = [
            im if im.size == (side, side) else im.resize((side, side), Image.Resampling.LANCZOS)
            for im in images
        ]
        canvas = Image.new("RGBA", (side * 2, side * 2), (0, 0, 0, 0))
        for tile, (x, y) in zip(resized, [(0, 0), (side, 0), (0, side), (side, side)], strict=True):
            canvas.paste(tile, (x, y), tile)
        final = canvas.resize((side, side), Image.Resampling.LANCZOS)
        target.parent.mkdir(parents=True, exist_ok=True)
        final.save(target, format="PNG")
    finally:
        for im in images:
            im.close()


def main() -> int:
    if not MENU_DIR.exists():
        print(f"Not found: {MENU_DIR}", file=sys.stderr)
        return 1

    made = 0
    for category in sorted(p for p in MENU_DIR.iterdir() if p.is_dir()):
        subcategories = sorted(p for p in category.iterdir() if p.is_dir())
        for sub in subcategories:
            photos = pick(real_photos(sub))
            if not photos:
                print(f"skipped {sub.relative_to(MENU_DIR)} -- no photos", file=sys.stderr)
                continue
            make_cover(photos, sub / COVER_NAME)
            note = " (repeated -- fewer than 4 distinct photos)" if len(set(photos)) < TILES else ""
            print(f"{(sub / COVER_NAME).relative_to(MENU_DIR.parent.parent)}{note}")
            made += 1

        photos = choose_photos(category)
        if not photos:
            print(f"skipped {category.relative_to(MENU_DIR)} -- no photos", file=sys.stderr)
            continue
        make_cover(photos, category / COVER_NAME)
        note = (
            " (repeated -- fewer than 4 distinct photos available)"
            if len(set(photos)) < TILES
            else ""
        )
        print(f"{(category / COVER_NAME).relative_to(MENU_DIR.parent.parent)}{note}")
        made += 1

    print(f"\n{made} cover(s) made")
    return 0


if __name__ == "__main__":
    sys.exit(main())
