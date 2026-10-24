"""The one image format a screenshot takes after capture, the hash of its pixels, and
whether it is black.

Every screenshot is encoded once, when it is captured, as lossless WebP of at most
``MAX_PIXELS``: the size the models receive, and a format the chat and embedding
providers both accept. Lossless keeps the pixels the models see unchanged, so two
captures of one screen hash alike once the status bar, whose clock and icons change
on their own, is left out.
"""

from __future__ import annotations

import base64
import hashlib
import io
import math
from typing import TYPE_CHECKING

from PIL import Image, ImageStat

if TYPE_CHECKING:
    from collections.abc import Sequence

MEDIA_TYPE = "image/webp"
MAX_PIXELS = 1_000_000
# Shrinking with LANCZOS blends each source row into the 3 output rows on either side of
# it, so the rows just below a shrunk status bar still carry some of its pixels.
_LANCZOS_REACH_ROWS = 3
# A secure window captures as pixels of exactly 0; the margin also counts a capture
# that is black to the eye.
BLACK_LEVEL = 8


def _shrink_to_max_pixels(image: Image.Image) -> Image.Image:
    """Return ``image`` scaled down with LANCZOS to at most ``MAX_PIXELS``, keeping its
    aspect ratio and mode, or ``image`` itself when it is no larger.
    """
    width, height = image.size
    total = width * height
    if total <= MAX_PIXELS:
        return image
    scale = math.sqrt(MAX_PIXELS / total)
    return image.resize((int(width * scale), int(height * scale)), Image.Resampling.LANCZOS)


def encode_screenshot(screenshot: bytes | str) -> bytes:
    """Return ``screenshot``, image bytes such as a PNG or their base64, as lossless
    WebP of at most ``MAX_PIXELS``.
    """
    data = (
        base64.b64decode(screenshot, validate=True) if isinstance(screenshot, str) else screenshot
    )
    with Image.open(io.BytesIO(data)) as image:
        buffer = io.BytesIO()
        _shrink_to_max_pixels(image).save(buffer, format="WEBP", lossless=True)
    return buffer.getvalue()


def status_bar_rows(
    status_bar: Sequence[int] | None, *, screenshot_height: int, image_height: int
) -> int:
    """Return how many top rows of an encoded image show the status bar, whose frame
    ``[left, top, right, bottom]`` is in the pixels of a screenshot ``screenshot_height``
    high, or ``0`` without a frame.

    The bar's bottom is scaled to the image and rounded up; a shrunk image adds the
    rows that LANCZOS blended with the bar.
    """
    if status_bar is None:
        return 0
    rows = math.ceil(status_bar[3] * image_height / screenshot_height)
    if image_height < screenshot_height:
        rows += _LANCZOS_REACH_ROWS
    return max(0, min(rows, image_height))


def pixel_hash(image: bytes, band_rows: int) -> str:
    """Return the SHA-256, as hex, of ``image``, encoded bytes such as the WebP
    ``encode_screenshot`` returns, decoded and without its top ``band_rows`` rows.

    The mode and the size of what remains are hashed with its pixels, so two images
    whose pixel bytes coincide in another layout hash apart.
    """
    with Image.open(io.BytesIO(image)) as decoded:
        width, height = decoded.size
        kept = decoded.crop((0, band_rows, width, height))
    digest = hashlib.sha256(f"{kept.mode} {kept.width}x{kept.height}\n".encode("ascii"))
    digest.update(kept.tobytes())
    return digest.hexdigest()


def is_black(image: bytes) -> bool:
    """Return whether no colour channel of any pixel of ``image``, encoded bytes such as
    the WebP ``encode_screenshot`` returns, is above ``BLACK_LEVEL``."""
    with Image.open(io.BytesIO(image)) as decoded:
        extrema = ImageStat.Stat(decoded.convert("RGB")).extrema
    return all(highest <= BLACK_LEVEL for _, highest in extrema)


def mask_regions(
    screenshot_b64: str, regions: Sequence[Sequence[int]], native_size: tuple[int, int]
) -> str:
    """Hide system-owned feedback from model input while retaining the raw screenshot.

    The mask follows native input coordinates even when the model image is shrunk.
    """
    if not regions:
        return screenshot_b64
    with Image.open(io.BytesIO(base64.b64decode(screenshot_b64, validate=True))) as image:
        masked = image.convert("RGB")
        for left, top, right, bottom in regions:
            box = (
                math.floor(left * image.width / native_size[0]),
                math.floor(top * image.height / native_size[1]),
                math.ceil(right * image.width / native_size[0]),
                math.ceil(bottom * image.height / native_size[1]),
            )
            masked.paste((128, 128, 128), box)
        buffer = io.BytesIO()
        masked.save(buffer, format="WEBP", lossless=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")
