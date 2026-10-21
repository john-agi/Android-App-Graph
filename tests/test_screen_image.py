"""Tests for android_app_graph.screen_image: every screenshot is encoded once, at
capture, as lossless WebP of at most 1,000,000 pixels."""

from __future__ import annotations

import base64
import io

from PIL import Image

from android_app_graph.screen_image import MAX_PIXELS, MEDIA_TYPE, encode_screenshot


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
