"""Tests for android_app_graph.retrying.call_with_retry.

The single re-ask for unusable output shared by utils.vlm_utils (chat
completions, page descriptions, text embeddings, action responses) and
embedding_cache.compute_embedding_with_retry; this module owns its tests,
including which errors are re-asked, so the callers cannot drift on it.
"""

from __future__ import annotations

import httpx
import pytest
from openrouter.errors import OpenRouterError, ResponseValidationError

from android_app_graph import retrying


def test_call_with_retry_returns_the_first_success() -> None:
    calls: list[int] = []

    def once() -> str:
        calls.append(1)
        return "ok"

    assert retrying.call_with_retry("label", once) == "ok"
    assert len(calls) == 1


def test_call_with_retry_asks_once_more_after_unusable_output(no_sleep: list[float]) -> None:
    attempts: list[int] = []

    def flaky() -> str:
        attempts.append(1)
        if len(attempts) == 1:
            raise retrying.UnusableOutputError("unreadable")
        return "ok"

    assert retrying.call_with_retry("label", flaky) == "ok"
    assert len(attempts) == 2
    assert no_sleep == []


def test_call_with_retry_reraises_after_the_second_attempt() -> None:
    attempts: list[int] = []

    def always_unusable() -> str:
        attempts.append(1)
        raise retrying.UnusableOutputError("unreadable")

    with pytest.raises(retrying.UnusableOutputError, match="unreadable"):
        retrying.call_with_retry("label", always_unusable)
    assert len(attempts) == retrying.UNUSABLE_OUTPUT_ATTEMPTS == 2


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("boom"),
        ValueError("not marked unusable"),
        httpx.ReadTimeout("timed out"),
        httpx.ConnectError("connection refused"),
        OpenRouterError("rate limited", httpx.Response(429)),
        OpenRouterError("upstream failed", httpx.Response(503)),
        OpenRouterError("unexpected provider error", httpx.Response(200)),
    ],
)
def test_call_with_retry_never_retries_anything_else(error: Exception) -> None:
    attempts: list[int] = []

    def request() -> str:
        attempts.append(1)
        raise error

    with pytest.raises(type(error)):
        retrying.call_with_retry("sdk", request)
    assert attempts == [1]


@pytest.mark.parametrize(
    ("status", "re_asked"), [(200, True), (299, True), (300, False), (503, False)]
)
def test_only_a_validation_failure_on_a_2xx_response_is_re_asked(
    status: int, re_asked: bool
) -> None:
    attempts: list[int] = []

    def request() -> str:
        attempts.append(1)
        raise ResponseValidationError("invalid schema", httpx.Response(status), ValueError("x"))

    with pytest.raises(ResponseValidationError):
        retrying.call_with_retry("sdk", request)
    assert len(attempts) == (2 if re_asked else 1)


def test_call_with_retry_logs_the_traceback_on_a_re_ask(caplog: pytest.LogCaptureFixture) -> None:
    attempts: list[int] = []

    def flaky() -> str:
        attempts.append(1)
        if len(attempts) == 1:
            msg = "unreadable"
            raise retrying.UnusableOutputError(msg)
        return "ok"

    with caplog.at_level("WARNING"):
        assert retrying.call_with_retry("action agent", flaky) == "ok"
    assert "action agent returned unusable output; asking again (2/2)" in caplog.text
    assert caplog.records[0].exc_info is not None
