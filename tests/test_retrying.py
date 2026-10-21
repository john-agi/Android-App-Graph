"""Tests for android_app_graph.retrying.

``call_with_retry`` is the single re-ask for unusable output shared by
utils.vlm_utils' chat completions, page descriptions, text embeddings and
action responses; this module owns its tests, including which errors are
re-asked, so the callers cannot drift on it. ``DeadlineTransport`` and
``capped_retries`` bound how long one request and the SDK's retries of it take.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from typing import override

import httpx
import pytest
from openrouter.errors import OpenRouterError, ResponseValidationError

from android_app_graph import retrying
from android_app_graph.utils import ModelSettings, open_client
from android_app_graph.utils.vlm_utils import send_chat

_URL = "https://openrouter.ai/api/v1/chat/completions"
_WHITESPACE = b"\n         \n"


class _Clock:
    """Seconds that pass only when a test moves them on."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class _Trickle(httpx.SyncByteStream):
    """A body of whitespace sent every 3 s, as OpenRouter sends while a model works.

    The default runs for 50 minutes, far past any deadline under test, so a
    missing deadline fails the test instead of hanging it.
    """

    def __init__(self, clock: _Clock, chunks: int = 1_000) -> None:
        self.clock = clock
        self.chunks = chunks
        self.sent = 0
        self.closed = False

    @override
    def __iter__(self) -> Iterator[bytes]:
        while self.sent < self.chunks:
            self.clock.now += 3
            self.sent += 1
            yield _WHITESPACE

    @override
    def close(self) -> None:
        self.closed = True


def _deadline_client(
    respond: Callable[[httpx.Request], httpx.Response], clock: _Clock
) -> httpx.Client:
    deadline = retrying.DeadlineTransport(httpx.MockTransport(respond), clock=clock)
    return httpx.Client(transport=deadline, timeout=60)


def test_a_reply_still_trickling_in_after_its_read_timeout_raises_a_read_timeout() -> None:
    """The 21st chunk, at 63 s, is the first to arrive after the 60 s deadline."""
    clock = _Clock()
    trickle = _Trickle(clock)
    with (
        _deadline_client(lambda _request: httpx.Response(200, stream=trickle), clock) as client,
        pytest.raises(httpx.ReadTimeout, match=r"^no complete response within 60 s$"),
    ):
        client.get(_URL)
    assert trickle.closed
    assert trickle.sent == 21


def test_a_reply_complete_before_its_deadline_passes_through_unchanged() -> None:
    clock = _Clock()
    sent = httpx.Response(201, headers={"x-request-id": "req-1"}, json={"ok": True})

    def reply(_request: httpx.Request) -> httpx.Response:
        clock.now += 59
        return sent

    with _deadline_client(reply, clock) as client:
        response = client.get(_URL)
    assert response.status_code == 201
    assert response.headers.multi_items() == sent.headers.multi_items()
    assert response.headers["x-request-id"] == "req-1"
    assert response.json() == {"ok": True}


def test_a_read_timeout_from_the_wrapped_transport_propagates_unchanged() -> None:
    raised: list[httpx.ReadTimeout] = []

    def silent(request: httpx.Request) -> httpx.Response:
        raised.append(httpx.ReadTimeout("no bytes for 60 s", request=request))
        raise raised[0]

    with _deadline_client(silent, _Clock()) as client, pytest.raises(httpx.ReadTimeout) as caught:
        client.get(_URL)
    assert caught.value is raised[0]


def test_a_request_without_a_read_timeout_has_no_deadline() -> None:
    clock = _Clock()
    trickle = _Trickle(clock, chunks=100)
    with _deadline_client(lambda _request: httpx.Response(200, stream=trickle), clock) as client:
        response = client.get(_URL, timeout=None)
    assert response.content == _WHITESPACE * 100
    assert clock.now == 300


def test_closing_the_deadline_transport_closes_the_transport_it_wraps() -> None:
    closed: list[bool] = []

    class Wrapped(httpx.MockTransport):
        @override
        def close(self) -> None:
            closed.append(True)

    wrapped = Wrapped(lambda _request: httpx.Response(200))
    with httpx.Client(transport=retrying.DeadlineTransport(wrapped)):
        assert closed == []
    assert closed == [True]


def test_open_client_ends_a_trickling_reply_and_the_sdk_tries_it_only_twice(
    monkeypatch: pytest.MonkeyPatch, no_sleep: list[float]
) -> None:
    """Both clocks are fake: the SDK measures its retry time with ``time.time``, so
    with a real clock and patched sleeps it would retry without end.
    """
    clock = _Clock()
    monkeypatch.setattr(time, "monotonic", clock)
    monkeypatch.setattr(time, "time", clock)
    trickles: list[_Trickle] = []

    def trickle(_transport: httpx.HTTPTransport, _request: httpx.Request) -> httpx.Response:
        trickles.append(_Trickle(clock))
        return httpx.Response(
            200, headers={"content-type": "application/json"}, stream=trickles[-1]
        )

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", trickle)
    with open_client("test-key") as client, pytest.raises(httpx.ReadTimeout):
        send_chat(
            client,
            call_type="chat",
            settings=ModelSettings("provider/model", request_timeout=60),
            messages=[{"role": "user", "content": "Write a 1,500-word essay."}],
        )
    assert [(t.sent, t.closed) for t in trickles] == [(21, True), (21, True)]
    assert len(no_sleep) == 1


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
            msg = "unreadable"
            raise retrying.UnusableOutputError(msg)
        return "ok"

    assert retrying.call_with_retry("label", flaky) == "ok"
    assert len(attempts) == 2
    assert no_sleep == []


def test_call_with_retry_reraises_after_the_second_attempt() -> None:
    attempts: list[int] = []

    def always_unusable() -> str:
        attempts.append(1)
        msg = "unreadable"
        raise retrying.UnusableOutputError(msg)

    with pytest.raises(retrying.UnusableOutputError, match="unreadable"):
        retrying.call_with_retry("label", always_unusable)
    assert len(attempts) == retrying.UNUSABLE_OUTPUT_ATTEMPTS == 2


@pytest.mark.parametrize(
    "error",
    [
        ValueError("not marked unusable"),
        httpx.ReadTimeout("timed out"),
        OpenRouterError("rate limited", httpx.Response(429)),
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
        msg = "invalid schema"
        raise ResponseValidationError(msg, httpx.Response(status), ValueError("x"))

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
