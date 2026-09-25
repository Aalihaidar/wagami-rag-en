#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Make the small thumbnails the chat page shows on the cards for menu groups and categories.

Reads every photo in data/images/ and writes a small WebP version of it to data/images/thumbs/
(where app/thumbnails.py says it is). A photo whose thumbnail is already there and newer is
skipped, so re-running after adding photos only does the new ones. Like the photos, the thumbnails
are local: data/images/ is not committed, so publish thumbs/ wherever the photos are published
(the same base URL, IMAGE_BASE_URL).

Usage:
    uv run python scripts/make_thumbnails.py            # only what is missing or out of date
    uv run python scripts/make_thumbnails.py --force    # redo everything
    uv run python scripts/make_thumbnails.py --size 480 --quality 80
"""

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageOps

from app.thumbnails import THUMBNAIL_DIR, THUMBNAIL_SIZE, thumbnail_filename

IMAGES_DIR = Path(__file__).resolve().parent.parent / "data" / "images"
PHOTO_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
DEFAULT_QUALITY = 78


def make_thumbnail(
    source: Path, target: Path, *, size: int = THUMBNAIL_SIZE, quality: int = DEFAULT_QUALITY
) -> None:
    """Write a WebP copy of `source` whose longest side is at most `size` pixels (never enlarged,
    aspect ratio kept, photo orientation applied) to `target`, creating its folder."""
    with Image.open(source) as opened:
        image = ImageOps.exif_transpose(opened)
        image.thumbnail((size, size), Image.Resampling.LANCZOS)
        if image.mode not in ("RGB", "RGBA"):
            image = image.convert("RGBA" if "transparency" in image.info else "RGB")
        target.parent.mkdir(parents=True, exist_ok=True)
        image.save(target, format="WEBP", quality=quality, method=6)


def is_fresh(source: Path, target: Path) -> bool:
    return target.exists() and target.stat().st_mtime >= source.stat().st_mtime


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--images-dir", type=Path, default=IMAGES_DIR)
    parser.add_argument("--size", type=int, default=THUMBNAIL_SIZE, help="longest side, in pixels")
    parser.add_argument("--quality", type=int, default=DEFAULT_QUALITY, help="WebP quality, 1-100")
    parser.add_argument("--force", action="store_true", help="redo thumbnails that are up to date")
    args = parser.parse_args()

    photos = sorted(
        p for p in args.images_dir.iterdir() if p.is_file() and p.suffix.lower() in PHOTO_SUFFIXES
    )
    if not photos:
        print(f"No photos in {args.images_dir}", file=sys.stderr)
        return 1

    made = skipped = 0
    before = after = 0
    for photo in photos:
        target = args.images_dir / thumbnail_filename(photo.name)
        if not args.force and is_fresh(photo, target):
            skipped += 1
            continue
        make_thumbnail(photo, target, size=args.size, quality=args.quality)
        made += 1
        before += photo.stat().st_size
        after += target.stat().st_size

    print(f"{made} made, {skipped} already up to date, in {args.images_dir / THUMBNAIL_DIR}")
    if made:
        print(
            f"photos {before / 1024:.0f} KB -> thumbnails {after / 1024:.0f} KB "
            f"({after / made / 1024:.0f} KB each on average)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
