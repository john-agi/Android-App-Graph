"""The one image format a screenshot takes after capture.

Every screenshot is encoded once, when it is captured, as lossless WebP of at most
``MAX_PIXELS``: the size the models receive, and a format the chat and embedding
providers both accept. Lossless keeps the pixels the models see unchanged.
"""

from __future__ import annotations

import base64
import io
import math

from PIL import Image

MEDIA_TYPE = "image/webp"
MAX_PIXELS = 1_000_000


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
