"""The one image format a screenshot takes after capture, and the hash of its pixels.

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

from PIL import Image

if TYPE_CHECKING:
    from collections.abc import Sequence

MEDIA_TYPE = "image/webp"
MAX_PIXELS = 1_000_000
# Shrinking with LANCZOS blends each source row into the 3 output rows on either side of
# it, so the rows just below a shrunk status bar still carry some of its pixels.
_LANCZOS_REACH_ROWS = 3


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
