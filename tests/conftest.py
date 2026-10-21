"""Shared pytest configuration: Hypothesis profiles, credential and network guards,
and fixtures that answer OpenRouter SDK requests from an offline mock transport."""

from __future__ import annotations

import base64
import io
import os
import socket
import subprocess
import time
from collections.abc import Callable, Generator, Iterator
from contextlib import contextmanager
from typing import Any, NoReturn

import httpx
import pytest
from hypothesis import settings
from openrouter import OpenRouter
from PIL import Image

from android_app_graph import cli, device
from android_app_graph.commands import audit
from android_app_graph.device import DeviceController
from android_app_graph.utils import open_client

# Hypothesis's built-in "ci" profile, its own choice when CI is set, is deterministic,
# keeps no example database and has no deadline; CI runs it with more examples.
# Local runs have no deadline either: how long one example takes depends on how busy
# the machine is, not on the code under test.
settings.register_profile("ci", settings.get_profile("ci"), max_examples=500)
settings.register_profile("dev", settings.get_profile("default"), deadline=None)
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "ci" if os.getenv("CI") else "dev"))

# The OpenRouter SDK also reads OPENROUTER_HTTP_REFERER, OPENROUTER_X_OPEN_ROUTER_* and
# OPENROUTER_DEBUG, so the whole prefix is cleared, not only the key.
_CREDENTIAL_PREFIXES = ("APP_GRAPH_", "OPENROUTER_")
_CREDENTIAL_NAMES = frozenset({"GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"})


@pytest.fixture(autouse=True)
def _clear_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove provider credentials and SDK settings so tests never use real keys or endpoints."""
    for name in list(os.environ):
        if name.startswith(_CREDENTIAL_PREFIXES) or name in _CREDENTIAL_NAMES:
            monkeypatch.delenv(name)


def _deny_network(*_args: object, **_kwargs: object) -> NoReturn:
    msg = "network access is disabled in the test suite"
    raise RuntimeError(msg)


@pytest.fixture(autouse=True, scope="session")
def _no_network() -> Iterator[None]:
    """Run the whole session with outbound networking disabled."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(socket, "getaddrinfo", _deny_network)
        mp.setattr(socket, "create_connection", _deny_network)
        mp.setattr(socket.socket, "connect", _deny_network)
        mp.setattr(socket.socket, "connect_ex", _deny_network)
        yield


@pytest.fixture
def model_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit model choices for tests of behavior other than missing configuration."""
    for role, model in (
        ("PAGE_DETAIL", "openai/gpt-6.1-sol"),
        ("EMBEDDING", "google/gemini-embedding-2"),
        ("INSTRUCTION", "openai/gpt-6.1-sol"),
        ("ACTION", "qwen/qwen3.8-flash"),
    ):
        monkeypatch.setenv(f"APP_GRAPH_{role}_MODEL", model)


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record sleeps instead of waiting for them.

    The OpenRouter SDK's retry back-off and the explorer's settling pauses call
    ``time.sleep`` on the shared stdlib module, so patching it here covers them
    all. A test must never let the SDK keep failing on a real clock: it retries
    5XX responses and timeouts until twice the request's timeout has passed on
    ``time.time``, which no patched sleep moves on.
    """
    delays: list[float] = []
    monkeypatch.setattr(time, "sleep", delays.append)
    return delays


@pytest.fixture
def adb_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve adb to the bare name ``adb``, so a test that replaces ``subprocess.run``
    neither needs adb installed nor depends on where it is.
    """
    monkeypatch.setattr(device, "adb_executable", lambda: "adb")


@pytest.fixture
def device_log(monkeypatch: pytest.MonkeyPatch, adb_on_path: None) -> list[object]:
    """Fake controllers append their actions here too, so a test sees adb commands
    and device actions in the order the device would have received them.
    """
    del adb_on_path  # requested for its side effect on device.adb_executable
    log: list[object] = []

    def run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        log.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    return log


@pytest.fixture
def router_requests() -> list[httpx.Request]:
    return []


@pytest.fixture
def router_responses() -> list[httpx.Response]:
    return []


@pytest.fixture
def router_http(
    router_requests: list[httpx.Request], router_responses: list[httpx.Response]
) -> Iterator[httpx.Client]:
    def respond(request: httpx.Request) -> httpx.Response:
        router_requests.append(request)
        return router_responses.pop(0)

    with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:
        yield http_client


@pytest.fixture
def router_client(router_http: httpx.Client) -> Iterator[OpenRouter]:
    with open_client("test-router-key", http_client=router_http) as client:
        yield client


@pytest.fixture
def opened_clients(router_http: httpx.Client, monkeypatch: pytest.MonkeyPatch) -> list[OpenRouter]:
    """Every client a command opens, each over the mock transport."""
    opened: list[OpenRouter] = []

    @contextmanager
    def open_routed(api_key: str) -> Generator[OpenRouter]:
        with open_client(api_key, http_client=router_http) as client:
            opened.append(client)
            yield client

    for module in (cli, audit):
        monkeypatch.setattr(module, "open_client", open_routed)
    return opened


@pytest.fixture
def chat_body() -> dict[str, object]:
    return {
        "id": "chat-test",
        "created": 0,
        "model": "test-model",
        "object": "chat.completion",
        "system_fingerprint": None,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "test reply",
                },
            }
        ],
        "usage": {
            "prompt_tokens": 7,
            "completion_tokens": 3,
            "total_tokens": 10,
            "cost": 0.00042,
            "prompt_tokens_details": {"cached_tokens": 2},
            "completion_tokens_details": {"reasoning_tokens": 1},
        },
    }


@pytest.fixture
def chat_response(chat_body: dict[str, object]) -> Callable[..., httpx.Response]:
    """Build ``chat_body`` with one assistant message made of the given fields.

    ``tool_arguments`` adds a native tool call carrying that JSON; ``content``
    left out is omitted from the message rather than sent as null.
    """

    def build(
        *, tool_arguments: str | None = None, finish_reason: str = "stop", **message: object
    ) -> httpx.Response:
        if tool_arguments is not None:
            message["tool_calls"] = [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "mobile_use", "arguments": tool_arguments},
                }
            ]
        choice = {
            "index": 0,
            "finish_reason": finish_reason,
            "message": {"role": "assistant", **message},
        }
        return httpx.Response(200, json={**chat_body, "choices": [choice]})

    return build


@pytest.fixture
def chat_usage_totals() -> dict[str, float]:
    """What one ``chat_body`` reply adds to a TokenTracker."""
    return {
        "calls": 1,
        "prompt_tokens": 7,
        "cached_prompt_tokens": 2,
        "completion_tokens": 3,
        "reasoning_tokens": 1,
        "total_tokens": 10,
        "cost": 0.00042,
    }


@pytest.fixture
def embedding_usage_totals() -> dict[str, float]:
    """What one ``embedding_body`` reply adds to a TokenTracker."""
    return {
        "calls": 1,
        "prompt_tokens": 5,
        "cached_prompt_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 5,
        "cost": 0.00001,
    }


@pytest.fixture
def embedding_body() -> dict[str, object]:
    return {
        "object": "list",
        "model": "google/gemini-embedding-2",
        "data": [{"object": "embedding", "index": 0, "embedding": [0.25, 0.75]}],
        "usage": {"prompt_tokens": 5, "total_tokens": 5, "cost": 0.00001},
    }


@pytest.fixture
def png_screenshot() -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), "white").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class _ScreenSequence:
    w = 1080
    h = 1920

    def __init__(self, *screens: dict[str, Any]) -> None:
        self.config: dict[str, Any] = {"device": {"udid": "test-device"}}
        self._screens = list(screens)

    def get_state(self) -> dict[str, Any]:
        return self._screens[0]

    def exe_action(self, action: dict[str, Any]) -> None:
        assert len(self._screens) > 1, f"No screen scripted after {action}"
        self._screens.pop(0)


@pytest.fixture
def two_screen_device(png_screenshot: str) -> DeviceController:
    # Two activities of one package: identify_state still compares the second screen
    # with the first, but cannot answer it from its last-screen cache.
    search = {
        "activity": "com.demo.app.Search",
        "screenshot": png_screenshot,
        "package": "com.demo.app",
    }
    return _ScreenSequence(search, {**search, "activity": "com.demo.app.Results"})
