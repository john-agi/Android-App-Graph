"""Tests for android_app_graph.retrying.call_with_retry.

The single retry-with-backoff policy shared by adapters.aitk_translator (chat
completion, page describe/state, action agent) and
embedding_cache.compute_embedding_with_retry; this module owns its tests so
the callers cannot drift on the policy.
"""

from __future__ import annotations

import pytest

from android_app_graph import retrying


def test_call_with_retry_returns_the_first_success() -> None:
    calls: list[int] = []

    def once() -> str:
        calls.append(1)
        return "ok"

    assert retrying.call_with_retry("label", once) == "ok"
    assert len(calls) == 1


@pytest.mark.usefixtures("no_sleep")
def test_call_with_retry_recovers_after_a_failure() -> None:
    attempts: list[int] = []

    def flaky() -> str:
        attempts.append(1)
        if len(attempts) < 3:
            raise RuntimeError("boom")
        return "ok"

    assert retrying.call_with_retry("label", flaky) == "ok"
    assert len(attempts) == 3


def test_call_with_retry_reraises_after_the_last_attempt(no_sleep: list[float]) -> None:
    attempts: list[int] = []

    def always_fails() -> str:
        attempts.append(1)
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        retrying.call_with_retry("label", always_fails)
    assert len(attempts) == retrying.DEFAULT_RETRIES + 1
    assert no_sleep == [2.0, 4.0]


def test_call_with_retry_honors_custom_retries_and_delay(no_sleep: list[float]) -> None:
    attempts: list[int] = []

    def always_fails() -> str:
        attempts.append(1)
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        retrying.call_with_retry("label", always_fails, retries=1, base_delay=0.5)
    assert len(attempts) == 2
    assert no_sleep == [0.5]


@pytest.mark.usefixtures("no_sleep")
def test_call_with_retry_logs_the_traceback_on_a_retried_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    attempts: list[int] = []

    def flaky() -> str:
        attempts.append(1)
        if len(attempts) == 1:
            msg = "boom"
            raise RuntimeError(msg)
        return "ok"

    with caplog.at_level("WARNING"):
        assert retrying.call_with_retry("action agent", flaky) == "ok"
    assert "action agent" in caplog.text
    assert caplog.records[0].exc_info is not None
