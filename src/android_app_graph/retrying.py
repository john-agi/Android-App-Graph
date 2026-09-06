"""Shared retry-with-backoff policy for every remote call this project makes.

``adapters.aitk_translator`` (chat completion, page describe/state, action agent)
and ``embedding_cache`` (image embedding computation) each retried a failing
remote call the same way -- a fixed number of retries, a base delay doubling
per attempt, re-raise on the last attempt -- so this is the single place that
owns that policy, and the two callers cannot drift on it.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

logger = logging.getLogger(__name__)

DEFAULT_RETRIES = 2
DEFAULT_BASE_DELAY_SECONDS = 2.0


def call_with_retry[T](
    label: str,
    func: Callable[[], T],
    *,
    retries: int = DEFAULT_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY_SECONDS,
) -> T:
    """Run ``func``, retrying up to ``retries`` times with exponential backoff.

    A retry loop is a boundary: the traceback is logged (``exc_info=True``) on
    every failed attempt except the last, where the exception is re-raised
    instead of being swallowed.
    """
    attempts = retries + 1
    for attempt in range(attempts):
        try:
            return func()
        except Exception:
            if attempt >= attempts - 1:
                raise
            delay = base_delay * (2**attempt)
            logger.warning(
                "%s failed; retrying in %.1fs (%d/%d).",
                label,
                delay,
                attempt + 1,
                attempts - 1,
                exc_info=True,
            )
            time.sleep(delay)
    msg = f"call_with_retry exhausted its attempts without raising: {label}"
    raise RuntimeError(msg)
