"""Tests for android_app_graph.screen_image: every screenshot is encoded once, at
capture, as lossless WebP of at most 1,000,000 pixels, its pixels below the status bar
are hashed, and an image without a pixel brighter than the black level is black."""

from __future__ import annotations

import base64
import io

import pytest
from PIL import Image, ImageDraw

from android_app_graph.screen_image import (
    BLACK_LEVEL,
    MAX_PIXELS,
    MEDIA_TYPE,
    encode_screenshot,
    is_black,
    pixel_hash,
    status_bar_rows,
)


def _noisy_image(size: tuple[int, int], mode: str = "RGB") -> Image.Image:
    """An image whose every channel varies from pixel to pixel, so a pixel-for-pixel
    comparison cannot pass by accident."""
    width, height = size
    noise = Image.effect_noise(size, 64)
    across = Image.linear_gradient("L").resize(size)
    down = Image.linear_gradient("L").rotate(90).resize((width, height))
    image = Image.merge("RGB", (noise, across, down))
    return image.convert(mode)


def _png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _decode(webp: bytes) -> Image.Image:
    with Image.open(io.BytesIO(webp)) as image:
        assert image.format == "WEBP"
        image.load()
        return image


def test_a_screenshot_over_the_limit_is_shrunk_with_lanczos_and_kept_lossless() -> None:
    original = _noisy_image((1250, 1000))
    # sqrt(1_000_000 / 1_250_000) scales 1250x1000 to 1118.03x894.43, truncated.
    expected = original.resize((1118, 894), Image.Resampling.LANCZOS)

    webp = encode_screenshot(_png(original))

    assert webp[:4] == b"RIFF"
    assert webp[8:12] == b"WEBP"
    decoded = _decode(webp)
    assert decoded.size == (1118, 894)
    assert decoded.width * decoded.height <= MAX_PIXELS
    assert decoded.mode == expected.mode
    assert decoded.tobytes() == expected.tobytes()


def test_a_screenshot_within_the_limit_keeps_its_size_and_pixels() -> None:
    original = _noisy_image((400, 300))

    decoded = _decode(encode_screenshot(_png(original)))

    assert decoded.size == (400, 300)
    assert decoded.tobytes() == original.tobytes()


def test_an_opaque_rgba_screenshot_keeps_its_pixels() -> None:
    """``screencap -p`` writes RGBA; lossless WebP keeps every pixel of an opaque one."""
    original = _noisy_image((300, 200), mode="RGBA")

    decoded = _decode(encode_screenshot(_png(original)))

    assert decoded.convert("RGBA").tobytes() == original.tobytes()


def test_base64_and_raw_bytes_encode_to_the_same_webp() -> None:
    png = _png(_noisy_image((64, 48)))
    assert encode_screenshot(base64.b64encode(png).decode("ascii")) == encode_screenshot(png)


def test_the_media_type_is_webp() -> None:
    assert MEDIA_TYPE == "image/webp"


@pytest.mark.parametrize(
    ("status_bar", "image_height", "rows"),
    [
        # ceil(121 * 1490 / 2400) = 76 rows, and 3 that LANCZOS blended with them.
        pytest.param([0, 0, 1080, 121], 1490, 79, id="shrunk"),
        pytest.param([0, 0, 1080, 121], 2400, 121, id="unshrunk"),
        pytest.param([0, 0, 1080, 2400], 1490, 1490, id="never past the image"),
        pytest.param(None, 1490, 0, id="no status bar"),
    ],
)
def test_the_status_bar_rows_of_an_encoded_image(
    status_bar: list[int] | None, image_height: int, rows: int
) -> None:
    assert status_bar_rows(status_bar, screenshot_height=2400, image_height=image_height) == rows


def _webp(image: Image.Image) -> bytes:
    return encode_screenshot(_png(image))


def _with_clock(image: Image.Image, minute: int, *, top: int = 2) -> Image.Image:
    """``image`` with a small "clock" of ``minute`` pixels drawn ``top`` rows down."""
    clocked = image.copy()
    ImageDraw.Draw(clocked).rectangle((4, top, 4 + minute, top + 3), fill=(255, 0, 0))
    return clocked


def test_a_screen_that_differs_only_inside_the_band_hashes_alike() -> None:
    screen = _noisy_image((64, 48))
    before, after = _webp(_with_clock(screen, 1)), _webp(_with_clock(screen, 9))

    assert before != after
    assert pixel_hash(before, 8) == pixel_hash(after, 8)
    assert pixel_hash(before, 0) != pixel_hash(after, 0)


def test_one_pixel_outside_the_band_changes_the_hash() -> None:
    screen = _noisy_image((64, 48))
    changed = screen.copy()
    changed.putpixel((30, 8), (0, 0, 0) if screen.getpixel((30, 8)) != (0, 0, 0) else (1, 1, 1))

    assert pixel_hash(_webp(screen), 8) != pixel_hash(_webp(changed), 8)


def test_the_hash_covers_the_size_of_the_pixels() -> None:
    """Two white images of 12 pixels have the same bytes in either shape."""
    wide, tall = Image.new("RGB", (6, 2), "white"), Image.new("RGB", (3, 4), "white")
    assert wide.tobytes() == tall.tobytes()

    assert pixel_hash(_png(wide), 0) != pixel_hash(_png(tall), 0)


def test_the_hash_covers_the_mode_of_the_pixels() -> None:
    """A gray image and a palette image whose pixels are all 0 have the same bytes."""
    gray = Image.new("L", (4, 4), 0)
    palette = Image.new("P", (4, 4), 0)
    assert gray.tobytes() == palette.tobytes()

    assert pixel_hash(_png(gray), 0) != pixel_hash(_png(palette), 0)


def test_an_opaque_screenshot_whose_every_pixel_is_zero_is_black() -> None:
    """``screencap -p`` captures a secure window as opaque RGBA pixels of 0."""
    assert is_black(_webp(Image.new("RGBA", (64, 48), (0, 0, 0, 255))))


@pytest.mark.parametrize("channel", [0, 1, 2], ids=["red", "green", "blue"])
@pytest.mark.parametrize(("level", "black"), [(BLACK_LEVEL, True), (BLACK_LEVEL + 1, False)])
def test_one_channel_of_one_pixel_above_the_black_level_makes_a_screenshot_not_black(
    channel: int, level: int, black: bool
) -> None:
    dim = Image.new("RGB", (64, 48), (BLACK_LEVEL, BLACK_LEVEL, BLACK_LEVEL))
    pixel = [BLACK_LEVEL] * 3
    pixel[channel] = level
    dim.putpixel((40, 30), tuple(pixel))

    assert is_black(_webp(dim)) is black


def test_the_black_level_is_8() -> None:
    assert BLACK_LEVEL == 8
