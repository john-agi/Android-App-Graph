"""Ask a model once more for output it cannot use; README.md#requests-and-recovery.

Transport failures never reach this loop as retryable: the OpenRouter SDK has
already retried them with its own policy, and retrying here too would multiply
the attempts.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from openrouter.errors import ResponseValidationError

logger = logging.getLogger(__name__)

# Upstream UI-KOBE's V2_PARSE_RETRIES = 1: one more attempt after the first.
UNUSABLE_OUTPUT_ATTEMPTS = 2
_SUCCESS_STATUS_CODES = range(200, 300)


class UnusableOutputError(ValueError):
    """A completed response the caller cannot use; asking once more may help."""


def _is_unusable(exc: Exception) -> bool:
    if isinstance(exc, ResponseValidationError):
        return exc.status_code in _SUCCESS_STATUS_CODES
    return isinstance(exc, UnusableOutputError)


def call_with_retry[T](label: str, func: Callable[[], T]) -> T:
    """Run ``func``, asking once more when it raises for unusable output.

    Unusable output is an ``UnusableOutputError`` or the SDK's
    ``ResponseValidationError`` on a 2xx response; anything else propagates at
    once. A retry loop is a boundary: the first failure is logged with its
    traceback, and the last one is re-raised instead of being swallowed.
    """
    for attempt in range(1, UNUSABLE_OUTPUT_ATTEMPTS + 1):
        try:
            return func()
        except (UnusableOutputError, ResponseValidationError) as exc:
            if attempt == UNUSABLE_OUTPUT_ATTEMPTS or not _is_unusable(exc):
                raise
            logger.warning(
                "%s returned unusable output; asking again (%d/%d).",
                label,
                attempt + 1,
                UNUSABLE_OUTPUT_ATTEMPTS,
                exc_info=True,
            )
    msg = f"call_with_retry exhausted its attempts without raising: {label}"
    raise RuntimeError(msg)
