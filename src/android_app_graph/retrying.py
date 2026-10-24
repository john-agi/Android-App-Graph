"""Bound each request in time, send it again after a failed attempt, and ask a model
once more for output it cannot use; README.md#requests-and-recovery.

Transport failures never reach the re-ask as retryable: ``send_within_deadline`` has
already sent them again within its deadline, and retrying here too would multiply the
attempts.
"""

from __future__ import annotations

import contextlib
import logging
import math
import random
import time
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from http import HTTPStatus
from typing import override

import httpx
from openrouter.errors import OpenRouterError, ResponseValidationError

logger = logging.getLogger(__name__)

# The first request and one more: every reply is billed, and each caller falls back or
# raises once both are unusable. send_within_deadline gives a timed-out attempt the same
# one more.
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


# The OpenRouter SDK's default backoff: 0.5 s, growing 1.5 times per retry, at most 60 s.
_FIRST_BACKOFF = 0.5
_BACKOFF_GROWTH = 1.5
_LONGEST_BACKOFF = 60.0


def _backoff(retries: int) -> float:
    """Return the seconds to wait before retry ``retries + 1``, jitter included."""
    jitter = random.uniform(0, 1)  # noqa: S311  # spreads retries out; guards no secret
    exponential = _FIRST_BACKOFF * math.pow(_BACKOFF_GROWTH, retries)
    return min(exponential + jitter, _LONGEST_BACKOFF)


def _retry_after(headers: httpx.Headers) -> float | None:
    """Return the seconds a ``Retry-After`` header asks for, given as a delay or a date."""
    value = headers.get("retry-after", "")
    with contextlib.suppress(ValueError):
        return float(value)
    with contextlib.suppress(TypeError, ValueError):
        return (parsedate_to_datetime(value) - datetime.now(UTC)).total_seconds()
    return None


def _failed_attempt(
    error: OpenRouterError | httpx.TimeoutException | httpx.NetworkError,
) -> str | None:
    """Say how an attempt that raised ``error`` failed when sending it again may help:
    it timed out, lost its connection, or was answered with HTTP 429 or a 5XX status."""
    if isinstance(error, httpx.TimeoutException):
        return "timed out"
    if isinstance(error, httpx.NetworkError):
        return "hit a connection error"
    if error.status_code == HTTPStatus.TOO_MANY_REQUESTS:
        return "was rate limited (HTTP 429)"
    if error.status_code >= HTTPStatus.INTERNAL_SERVER_ERROR:
        return f"failed with HTTP {error.status_code}"
    return None


def send_within_deadline[T](call_type: str, request_timeout: float, send: Callable[[], T]) -> T:
    """Return what ``send`` returns, sending it again after a failed attempt until the
    deadline passes; README.md#requests-and-recovery.

    ``send`` makes one attempt. The deadline is twice ``request_timeout`` after the
    first attempt, which leaves a timed-out attempt exactly one more, as the application
    asks once more for unusable output. An attempt that ``_failed_attempt`` describes is
    sent again after its ``Retry-After`` when that is positive and ends by the deadline,
    else after the backoff cut short to end at the deadline, so none is sent after it.
    Each such attempt is logged with ``call_type``, its number and how long it took; one
    that leaves no time propagates, as does every other error.
    """
    budget = 2 * request_timeout
    started = time.monotonic()
    attempt = 1
    while True:
        attempt_started = time.monotonic()
        try:
            return send()
        except (OpenRouterError, httpx.TimeoutException, httpx.NetworkError) as error:
            failure = _failed_attempt(error)
            if failure is None:
                raise
            ended = time.monotonic()
            took = ended - attempt_started
            time_left = budget - (ended - started)
            if time_left <= 0:
                logger.warning(
                    "%s: attempt %d %s after %.1f s; its deadline has passed",
                    call_type,
                    attempt,
                    failure,
                    took,
                )
                raise
            retry_after = (
                _retry_after(error.headers) if isinstance(error, OpenRouterError) else None
            )
            wait = (
                retry_after
                if retry_after is not None and 0 < retry_after <= time_left
                else min(_backoff(attempt - 1), time_left)
            )
        logger.warning(
            "%s: attempt %d %s after %.1f s; sending again in %.1f s",
            call_type,
            attempt,
            failure,
            took,
            wait,
        )
        time.sleep(wait)
        attempt += 1


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
