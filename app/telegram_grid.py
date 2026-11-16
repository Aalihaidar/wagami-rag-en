"""One picture for a Telegram list of groups or categories: each card's cover with its name
printed under it, laid out in a grid -- the Telegram counterpart of the chat page's picture
cards (rule R-14). An inline button can only hold text, so the pictures go into the photo the
buttons hang under.

Pure image work, no I/O: the covers arrive as bytes (None for one that couldn't be fetched,
which gets a plain tile so its name still lines up with its button).
"""

import io

from PIL import Image, ImageDraw, ImageFont

TILE = 240  # the covers' own size (scripts/generate_category_covers.py)
GAP = 16
LABEL_HEIGHT = 58  # two lines of FONT_SIZE
FONT_SIZE = 22
BACKGROUND = (255, 255, 255)  # the covers are cut out on white
PLACEHOLDER = (238, 238, 238)
TEXT = (34, 34, 34)
JPEG_QUALITY = 85


def _columns(count: int) -> int:
    if count <= 1:
        return 1
    return 2 if count in (2, 4) else 3


def _tile(image_bytes: bytes | None) -> Image.Image:
    if image_bytes is not None:
        try:
            with Image.open(io.BytesIO(image_bytes)) as source:
                picture = source.convert("RGBA")
            picture.thumbnail((TILE, TILE))
            tile = Image.new("RGB", (TILE, TILE), BACKGROUND)
            offset = ((TILE - picture.width) // 2, (TILE - picture.height) // 2)
            tile.paste(picture, offset, picture)
            return tile
        except OSError, ValueError:
            pass  # not an image after all: same as a missing one
    return Image.new("RGB", (TILE, TILE), PLACEHOLDER)


def _label_lines(
    draw: ImageDraw.ImageDraw, name: str, font: ImageFont.FreeTypeFont | ImageFont.ImageFont
) -> list[str]:
    """`name` wrapped to the tile's width, at most two lines, the second cut with "…"."""
    lines: list[str] = []
    current = ""
    for word in name.split():
        candidate = f"{current} {word}".strip()
        if current and draw.textlength(candidate, font=font) > TILE:
            lines.append(current)
            current = word
        else:
            current = candidate
    lines.append(current)
    if len(lines) > 2:
        lines = [lines[0], " ".join(lines[1:])]
    if draw.textlength(lines[-1], font=font) > TILE:
        last = lines[-1]
        while last and draw.textlength(last + "…", font=font) > TILE:
            last = last[:-1]
        lines[-1] = last.rstrip() + "…"
    return lines


def build_grid(cards: list[tuple[str, bytes | None]]) -> bytes:
    """A JPEG of `cards` (name, cover bytes or None) in reading order."""
    columns = _columns(len(cards))
    rows = -(-len(cards) // columns)
    width = columns * TILE + (columns + 1) * GAP
    height = rows * (TILE + LABEL_HEIGHT) + (rows + 1) * GAP
    grid = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(grid)
    font = ImageFont.load_default(size=FONT_SIZE)

    for index, (name, image_bytes) in enumerate(cards):
        row, column = divmod(index, columns)
        x = GAP + column * (TILE + GAP)
        y = GAP + row * (TILE + LABEL_HEIGHT + GAP)
        grid.paste(_tile(image_bytes), (x, y))
        for line_number, line in enumerate(_label_lines(draw, name, font)):
            draw.text(
                (x + TILE / 2, y + TILE + 4 + line_number * (FONT_SIZE + 4)),
                line,
                font=font,
                fill=TEXT,
                anchor="ma",
            )

    out = io.BytesIO()
    grid.save(out, format="JPEG", quality=JPEG_QUALITY)
    return out.getvalue()
