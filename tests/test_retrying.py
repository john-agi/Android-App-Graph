"""Tests for android_app_graph.retrying.

``call_with_retry`` is the single re-ask for unusable output shared by
utils.vlm_utils' chat completions, page descriptions, text embeddings and
action responses; this module owns its tests, including which errors are
re-asked, so the callers cannot drift on it. ``DeadlineTransport`` and
``send_within_deadline`` bound how long one request and its retries take, and
``send_within_deadline`` sends a request again after a failed attempt and logs
each one.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from typing import override

import httpx
import pytest
from openrouter.errors import OpenRouterError, ResponseValidationError, TooManyRequestsResponseError

from android_app_graph import retrying
from android_app_graph.utils import ModelSettings, open_client
from android_app_graph.utils.vlm_utils import message_text, send_chat

_URL = "https://openrouter.ai/api/v1/chat/completions"
_WHITESPACE = b"\n         \n"
_RATE_LIMITED = {"error": {"code": 429, "message": "rate limited"}}


class _Clock:
    """Seconds that pass only when a test moves them on or sleeps."""

    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    """One fake clock for the request's deadline, each attempt's duration and every sleep."""
    fake = _Clock()
    monkeypatch.setattr(time, "monotonic", fake)
    monkeypatch.setattr(time, "time", fake)
    monkeypatch.setattr(time, "sleep", fake.sleep)
    return fake


def _rate_limited(headers: dict[str, str] | None = None) -> httpx.Response:
    return httpx.Response(429, json=_RATE_LIMITED, headers=headers)


def _chat(answer: Callable[[httpx.Request], httpx.Response], *, request_timeout: float = 60) -> str:
    """Send one chat request through ``open_client`` to a provider that answers with ``answer``."""
    with (
        httpx.Client(transport=httpx.MockTransport(answer)) as http_client,
        open_client("test-key", http_client=http_client) as client,
    ):
        reply = send_chat(
            client,
            call_type="chat",
            settings=ModelSettings("provider/model", request_timeout=request_timeout),
            messages=[{"role": "user", "content": "hi"}],
        )
    return message_text(reply)


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


def _verify_through_open_client(
    monkeypatch: pytest.MonkeyPatch, answers: list[Callable[[], httpx.Response]]
) -> str:
    """Send one verifier request through the transport ``open_client`` builds, the
    transport it wraps answering each attempt with the next of ``answers``."""

    def answer(_transport: httpx.HTTPTransport, _request: httpx.Request) -> httpx.Response:
        return answers.pop(0)()

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", answer)
    with open_client("test-key") as client:
        reply = send_chat(
            client,
            call_type="node_verify",
            settings=ModelSettings("provider/model", request_timeout=60),
            messages=[{"role": "user", "content": "Write a 1,500-word essay."}],
        )
    return message_text(reply)


def _trickling(clock: _Clock, trickles: list[_Trickle]) -> Callable[[], httpx.Response]:
    def respond() -> httpx.Response:
        trickles.append(_Trickle(clock))
        return httpx.Response(
            200, headers={"content-type": "application/json"}, stream=trickles[-1]
        )

    return respond


def _retry_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.name == retrying.__name__]


def test_each_trickling_reply_ends_at_its_timeout_and_every_attempt_is_logged(
    monkeypatch: pytest.MonkeyPatch, clock: _Clock, caplog: pytest.LogCaptureFixture
) -> None:
    """Each reply's 21st chunk, at 63 s, is the first after its 60 s timeout; the second
    attempt ends past the request's 120 s deadline, so it is the last."""
    trickles: list[_Trickle] = []
    answers = [_trickling(clock, trickles)] * 3
    with caplog.at_level("WARNING"), pytest.raises(httpx.ReadTimeout):
        _verify_through_open_client(monkeypatch, answers)
    assert [(t.sent, t.closed) for t in trickles] == [(21, True), (21, True)]
    assert _retry_warnings(caplog) == [
        f"node_verify: attempt 1 timed out after 63.0 s; sending again in {clock.slept[0]:.1f} s",
        "node_verify: attempt 2 timed out after 63.0 s; its deadline has passed",
    ]


def test_a_timed_out_attempt_is_logged_and_the_attempt_after_it_answers(
    monkeypatch: pytest.MonkeyPatch,
    clock: _Clock,
    chat_body: dict[str, object],
    caplog: pytest.LogCaptureFixture,
) -> None:
    answers = [_trickling(clock, []), lambda: httpx.Response(200, json=chat_body)]
    with caplog.at_level("WARNING"):
        assert _verify_through_open_client(monkeypatch, answers) == "test reply"
    assert answers == []
    assert _retry_warnings(caplog) == [
        f"node_verify: attempt 1 timed out after 63.0 s; sending again in {clock.slept[0]:.1f} s"
    ]


def _connection_reset(request: httpx.Request) -> httpx.Response:
    msg = "connection reset"
    raise httpx.ConnectError(msg, request=request)


@pytest.mark.parametrize(
    ("fail", "failure"),
    [
        (
            lambda _request: httpx.Response(503, json={"error": {"code": 503, "message": "busy"}}),
            "failed with HTTP 503",
        ),
        (lambda _request: _rate_limited(), "was rate limited (HTTP 429)"),
        (_connection_reset, "hit a connection error"),
    ],
    ids=["5xx", "429", "dropped"],
)
def test_each_attempt_sent_again_is_logged_with_its_call_type_number_and_time(
    fail: Callable[[httpx.Request], httpx.Response],
    failure: str,
    clock: _Clock,
    chat_body: dict[str, object],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The failed attempt takes 2 s."""
    attempts: list[httpx.Request] = []

    def answer(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        if len(attempts) > 1:
            return httpx.Response(200, json=chat_body)
        clock.now += 2
        return fail(request)

    with caplog.at_level("WARNING"):
        assert _chat(answer) == "test reply"
    assert len(attempts) == 2
    assert _retry_warnings(caplog) == [
        f"chat: attempt 1 {failure} after 2.0 s; sending again in {clock.slept[0]:.1f} s"
    ]


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


@pytest.mark.parametrize(
    "rate_limited",
    [
        _rate_limited,
        lambda: httpx.Response(429, text="Too Many Requests"),
        lambda: httpx.Response(429, json={"unexpected": "shape"}),
    ],
    ids=["json", "text", "unexpected-json"],
)
def test_a_429_with_any_body_is_sent_again(
    rate_limited: Callable[[], httpx.Response], clock: _Clock, chat_body: dict[str, object]
) -> None:
    answers = [rate_limited(), httpx.Response(200, json=chat_body)]
    assert _chat(lambda _request: answers.pop(0)) == "test reply"
    assert answers == []
    assert len(clock.slept) == 1


@pytest.mark.parametrize("status", [429, 503])
@pytest.mark.parametrize(("retry_after", "waited"), [("30", 30), ("120", 120)])
def test_a_429_or_5xx_waits_as_long_as_its_retry_after_asks_when_that_ends_by_the_deadline(
    status: int, retry_after: str, waited: float, clock: _Clock, chat_body: dict[str, object]
) -> None:
    """At a 60 s timeout the deadline is 120 s after the first attempt."""
    refused = httpx.Response(status, json=_RATE_LIMITED, headers={"Retry-After": retry_after})
    answers = [refused, httpx.Response(200, json=chat_body)]
    assert _chat(lambda _request: answers.pop(0)) == "test reply"
    assert clock.slept == [waited]


@pytest.mark.parametrize("status", [429, 503])
@pytest.mark.parametrize(
    "retry_after", ["121", "0", "-5", "soon", "nan", "Wed, 21 Oct 2026 07:28:00 -0000"]
)
def test_a_retry_after_that_ends_past_the_deadline_or_is_no_delay_gives_way_to_the_backoff(
    status: int, retry_after: str, clock: _Clock, chat_body: dict[str, object]
) -> None:
    """A date in ``-0000``, which names no time zone, is not a delay."""
    refused = httpx.Response(status, json=_RATE_LIMITED, headers={"Retry-After": retry_after})
    answers = [refused, httpx.Response(200, json=chat_body)]
    assert _chat(lambda _request: answers.pop(0)) == "test reply"
    assert len(clock.slept) == 1
    assert 0.5 <= clock.slept[0] <= 1.5


def test_a_429_waits_until_the_date_its_retry_after_names(
    no_sleep: list[float], chat_body: dict[str, object]
) -> None:
    sent_at = datetime.now(UTC)
    retry_at = (sent_at + timedelta(seconds=30)).replace(microsecond=0)
    header = {"Retry-After": format_datetime(retry_at, usegmt=True)}
    answers = [_rate_limited(header), httpx.Response(200, json=chat_body)]
    assert _chat(lambda _request: answers.pop(0)) == "test reply"
    answered_at = datetime.now(UTC)
    assert len(no_sleep) == 1
    assert (retry_at - answered_at).total_seconds() <= no_sleep[0]
    assert no_sleep[0] <= (retry_at - sent_at).total_seconds()


def test_429s_past_the_deadline_raise_and_each_one_sent_again_is_logged(
    clock: _Clock, caplog: pytest.LogCaptureFixture
) -> None:
    """Each answer takes 50 s: the third ends past the 120 s deadline, so it is the last."""
    requests: list[httpx.Request] = []

    def slow_rate_limit(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        clock.now += 50
        return _rate_limited()

    with (
        caplog.at_level("WARNING", logger=retrying.__name__),
        pytest.raises(TooManyRequestsResponseError, match="rate limited"),
    ):
        _chat(slow_rate_limit)
    assert len(requests) == 3
    assert len(clock.slept) == 2
    assert 0.5 <= clock.slept[0] <= 1.5
    assert 0.75 <= clock.slept[1] <= 1.75
    rate_limited = "was rate limited (HTTP 429) after 50.0 s"
    assert _retry_warnings(caplog) == [
        f"chat: attempt 1 {rate_limited}; sending again in {clock.slept[0]:.1f} s",
        f"chat: attempt 2 {rate_limited}; sending again in {clock.slept[1]:.1f} s",
        f"chat: attempt 3 {rate_limited}; its deadline has passed",
    ]


def test_a_backoff_that_would_cross_the_deadline_ends_at_it_and_no_attempt_starts_after(
    clock: _Clock,
) -> None:
    """The first 429 arrives 0.25 s before the 120 s deadline, less than any backoff;
    the 429 that arrives at the deadline leaves no time, so it raises at once."""
    started_at: list[float] = []

    def rate_limit(_request: httpx.Request) -> httpx.Response:
        started_at.append(clock.now)
        clock.now += 119.75 if len(started_at) == 1 else 0
        return _rate_limited()

    with pytest.raises(TooManyRequestsResponseError):
        _chat(rate_limit)
    assert clock.slept == [0.25]
    assert started_at == [0, 120]


def test_the_429_backoff_grows_to_at_most_a_minute_between_tries(clock: _Clock) -> None:
    with pytest.raises(TooManyRequestsResponseError):
        _chat(lambda _request: _rate_limited(), request_timeout=300)
    assert max(clock.slept) == 60
    assert sum(clock.slept) == pytest.approx(600)


def test_a_5xx_after_a_429_is_sent_again_only_until_the_same_deadline(clock: _Clock) -> None:
    """Each answer takes 50 s. After the 429 and about 1 s of backoff, 69 s are left, so
    a 503 is sent once more but not a third time, which a fresh 120 s would allow."""
    statuses = [429, 503, 503, 503]

    def answer(_request: httpx.Request) -> httpx.Response:
        clock.now += 50
        status = statuses.pop(0)
        return httpx.Response(status, json={"error": {"code": status, "message": "busy"}})

    with pytest.raises(OpenRouterError) as caught:
        _chat(answer)
    assert caught.value.status_code == 503
    assert statuses == [503]


@pytest.mark.parametrize("status", [400, 402])
def test_any_other_4xx_raises_at_once(status: int, no_sleep: list[float]) -> None:
    requests: list[httpx.Request] = []

    def refuse(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(status, json={"error": {"code": status, "message": "refused"}})

    with pytest.raises(OpenRouterError) as caught:
        _chat(refuse)
    assert caught.value.status_code == status
    assert len(requests) == 1
    assert no_sleep == []
