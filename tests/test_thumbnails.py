import importlib.util
import os
from pathlib import Path
from types import ModuleType

import pytest
from PIL import Image

from app.thumbnails import THUMBNAIL_DIR, THUMBNAIL_SIZE, thumbnail_filename


def load_script() -> ModuleType:
    """scripts/ is not a package, so load the script the way python would run it."""
    path = Path(__file__).resolve().parent.parent / "scripts" / "make_thumbnails.py"
    spec = importlib.util.spec_from_file_location("make_thumbnails", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = load_script()


# ---- where a thumbnail is ------------------------------------------------------------------------


def test_a_thumbnail_is_a_webp_in_the_thumbs_folder_named_after_the_photo() -> None:
    assert thumbnail_filename("banana-katsu--c154f076.png") == "thumbs/banana-katsu--c154f076.webp"
    assert thumbnail_filename("photo.jpeg") == "thumbs/photo.webp"
    assert thumbnail_filename("a.b.png") == "thumbs/a.b.webp"  # only the last suffix is replaced
    assert THUMBNAIL_DIR == "thumbs"


# ---- making one ----------------------------------------------------------------------------------


def save_photo(path: Path, size: tuple[int, int], mode: str = "RGBA") -> Path:
    if mode == "RGBA":
        image = Image.new("RGBA", size, (200, 30, 30, 255))
        image.paste((0, 0, 0, 0), (0, 0, size[0] // 2, size[1]))  # a transparent left half
    else:
        image = Image.new(mode, size)
    image.save(path)
    return path


def test_a_large_photo_is_shrunk_to_the_size_with_its_shape_kept(tmp_path: Path) -> None:
    photo = save_photo(tmp_path / "wide.png", (1200, 600))

    script.make_thumbnail(photo, tmp_path / "thumbs" / "wide.webp")

    with Image.open(tmp_path / "thumbs" / "wide.webp") as thumb:
        assert thumb.format == "WEBP"
        assert thumb.size == (THUMBNAIL_SIZE, THUMBNAIL_SIZE // 2)


def test_a_small_photo_is_never_enlarged(tmp_path: Path) -> None:
    photo = save_photo(tmp_path / "small.png", (200, 200))

    script.make_thumbnail(photo, tmp_path / "small.webp")

    with Image.open(tmp_path / "small.webp") as thumb:
        assert thumb.size == (200, 200)


def test_transparency_survives_because_the_photos_are_cutouts(tmp_path: Path) -> None:
    photo = save_photo(tmp_path / "cutout.png", (400, 400))

    script.make_thumbnail(photo, tmp_path / "cutout.webp")

    with Image.open(tmp_path / "cutout.webp") as thumb:
        assert thumb.mode == "RGBA"
        alpha = thumb.getchannel("A")
        width, height = thumb.size
        assert alpha.getpixel((5, height // 2)) == 0  # the transparent half
        assert alpha.getpixel((width - 5, height // 2)) == 255  # the opaque half


def test_a_palette_photo_is_converted_and_the_folder_is_created(tmp_path: Path) -> None:
    photo = save_photo(tmp_path / "palette.png", (500, 500), mode="P")

    target = tmp_path / "deep" / "thumbs" / "palette.webp"
    script.make_thumbnail(photo, target)

    assert target.exists()
    with Image.open(target) as thumb:
        assert thumb.mode in ("RGB", "RGBA")


def test_a_thumbnail_is_far_smaller_than_a_photo_of_the_kind_it_comes_from(
    tmp_path: Path,
) -> None:
    noisy = Image.effect_noise((1000, 1000), 60).convert("RGBA")  # hard to compress, like a photo
    photo = tmp_path / "noisy.png"
    noisy.save(photo)

    script.make_thumbnail(photo, tmp_path / "noisy.webp")

    assert (tmp_path / "noisy.webp").stat().st_size < photo.stat().st_size / 5


# ---- the script ----------------------------------------------------------------------------------


def run(images_dir: Path, *extra: str) -> int:
    import sys

    argv = sys.argv
    sys.argv = ["make_thumbnails.py", "--images-dir", str(images_dir), *extra]
    try:
        return script.main()
    finally:
        sys.argv = argv


def test_the_script_writes_each_thumbnail_where_the_app_looks_for_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    save_photo(tmp_path / "a--1.png", (800, 800))
    save_photo(tmp_path / "b--2.png", (300, 300))
    (tmp_path / "notes.txt").write_text("not a photo")

    assert run(tmp_path) == 0

    for photo in ("a--1.png", "b--2.png"):
        assert (tmp_path / thumbnail_filename(photo)).exists()
    assert sorted(p.name for p in (tmp_path / THUMBNAIL_DIR).iterdir()) == [
        "a--1.webp",
        "b--2.webp",
    ]
    assert "2 made, 0 already up to date" in capsys.readouterr().out


def test_a_second_run_only_does_what_is_missing_or_out_of_date(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    save_photo(tmp_path / "a--1.png", (800, 800))
    run(tmp_path)
    capsys.readouterr()

    save_photo(tmp_path / "c--3.png", (800, 800))  # a new photo
    run(tmp_path)
    assert "1 made, 1 already up to date" in capsys.readouterr().out

    stale = tmp_path / thumbnail_filename("a--1.png")  # the photo was replaced by a newer one
    os.utime(stale, (1, 1))
    run(tmp_path)
    assert "1 made, 1 already up to date" in capsys.readouterr().out

    run(tmp_path, "--force")
    assert "3 made" not in capsys.readouterr().out  # only two photos, both redone


def test_the_script_reports_a_folder_with_no_photos(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run(tmp_path) == 1
    assert "No photos" in capsys.readouterr().err
