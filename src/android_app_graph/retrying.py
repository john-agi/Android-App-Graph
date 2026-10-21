"""Bound each request in time and ask a model once more for output it cannot use;
README.md#requests-and-recovery.

Transport failures never reach the re-ask as retryable: the OpenRouter SDK has
already retried them within ``capped_retries``, and retrying here too would
multiply the attempts.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator
from typing import override

import httpx
from openrouter.errors import ResponseValidationError
from openrouter.utils import BackoffStrategy, RetryConfig

logger = logging.getLogger(__name__)

# The first request and one more: every reply is billed, and each caller falls back or
# raises once both are unusable. capped_retries gives a timed-out attempt the same one more.
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


def capped_retries(request_timeout: float) -> RetryConfig:
    """Return the SDK's default retry policy, ended once twice ``request_timeout`` has passed.

    The interval, cap and exponent are the SDK's own. Twice the timeout leaves a
    timed-out attempt exactly one more, as the application asks once more for
    unusable output; the SDK's default would keep retrying for an hour.
    """
    max_elapsed_ms = round(2 * request_timeout * 1000)
    return RetryConfig(
        "backoff",
        BackoffStrategy(500, 60_000, 1.5, max_elapsed_ms),
        retry_connection_errors=True,
    )


def _read_timeout(request: httpx.Request) -> float | None:
    timeout = request.extensions.get("timeout")
    read = timeout.get("read") if isinstance(timeout, dict) else None
    return float(read) if isinstance(read, int | float) else None


class _DeadlineStream(httpx.SyncByteStream):
    """A response body that raises ``httpx.ReadTimeout`` once its deadline has passed."""

    def __init__(
        self,
        inner: httpx.SyncByteStream,
        request: httpx.Request,
        *,
        limit: float,
        deadline: float,
        clock: Callable[[], float],
    ) -> None:
        self._inner = inner
        self._request = request
        self._limit = limit
        self._deadline = deadline
        self._clock = clock

    @override
    def __iter__(self) -> Iterator[bytes]:
        for chunk in self._inner:
            if self._clock() > self._deadline:
                self._inner.close()
                msg = f"no complete response within {self._limit:g} s"
                raise httpx.ReadTimeout(msg, request=self._request)
            yield chunk

    @override
    def close(self) -> None:
        self._inner.close()


class DeadlineTransport(httpx.BaseTransport):
    """End a response still arriving after its read timeout; README.md#requests-and-recovery.

    httpx applies the read timeout to each chunk, so a provider that keeps a
    connection open with whitespace never trips it. The deadline starts when the
    request is handed over, so connecting and the headers count too. A request
    without a read timeout has no deadline, and total silence is still httpx's
    own read timeout.
    """

    def __init__(
        self, inner: httpx.BaseTransport, *, clock: Callable[[], float] | None = None
    ) -> None:
        self._inner = inner
        # Looked up per instance, not bound as the default, so a test that fakes
        # time.monotonic reaches the transport open_client builds.
        self._clock = time.monotonic if clock is None else clock

    @override
    def handle_request(self, request: httpx.Request) -> httpx.Response:
        limit = _read_timeout(request)
        started = self._clock()
        response = self._inner.handle_request(request)
        if limit is None or not isinstance(response.stream, httpx.SyncByteStream):
            return response
        return httpx.Response(
            response.status_code,
            headers=response.headers,
            stream=_DeadlineStream(
                response.stream,
                request,
                limit=limit,
                deadline=started + limit,
                clock=self._clock,
            ),
            extensions=response.extensions,
        )

    @override
    def close(self) -> None:
        self._inner.close()
