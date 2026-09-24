"""Small WebP versions of a folder of images, in a `thumbs` subfolder next to them, made by
scripts/make_thumbnails.py -- this module is the one place that says where a given image's
thumbnail is, so a script that writes them and any code that points at them cannot disagree.

Not currently wired to anything: it was built for the dish photos on a group/category list's
cards (a menu overview with four photos on each of nine cards, at ~280 KB each, downloaded around
10 MB), before those cards moved to one cover picture per group/category folder instead
(app/agent/choice_images.py) -- item cards have never used it, since they show the full photo (it
opens in the zoom view). Kept as a general-purpose shrink tool: point scripts/make_thumbnails.py's
`--images-dir` at any folder of images (for instance a menu/ folder whose cover pictures
turn out large) to make small versions of them the same way.
"""

from pathlib import PurePosixPath

# The folder, next to the photos, that holds the thumbnails.
THUMBNAIL_DIR = "thumbs"
# The longest side of a thumbnail, in pixels: a card's picture is at most about 230 CSS pixels
# wide, so this stays sharp on a high-density screen.
THUMBNAIL_SIZE = 360


def thumbnail_filename(image: str) -> str:
    """Where a dish photo's thumbnail is, relative to the photos' own location:
    `banana-katsu--c154f076.png` -> `thumbs/banana-katsu--c154f076.webp`."""
    return f"{THUMBNAIL_DIR}/{PurePosixPath(image).stem}.webp"
