"""Shared pytest configuration: Hypothesis profiles, credential and network guards."""

from __future__ import annotations

import base64
import io
import os
import socket
import time
from collections.abc import Iterator, Mapping
from typing import Any, NoReturn

import httpx
import pytest
from hypothesis import settings
from openrouter import OpenRouter
from PIL import Image

from android_app_graph.adapters import aitk_translator
from android_app_graph.utils import make_client, vlm_utils

# The built-in "ci" profile (deterministic, no deadline, no example database) is
# selected automatically when CI is set; re-registered here with more examples.
settings.register_profile("ci", settings.get_profile("ci"), max_examples=500)
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "ci" if os.getenv("CI") else "default"))

_CREDENTIAL_PREFIXES = ("APP_GRAPH_",)
_CREDENTIAL_NAMES = frozenset(
    {"GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY"}
)


@pytest.fixture(autouse=True)
def _clear_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove provider credentials so tests can never use real keys or endpoints."""
    for name in list(os.environ):
        if name.startswith(_CREDENTIAL_PREFIXES) or name in _CREDENTIAL_NAMES:
            monkeypatch.delenv(name)


def _deny_network(*_args: Any, **_kwargs: Any) -> NoReturn:
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
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record retry back-off delays instead of waiting for them.

    The retry-with-backoff policy (android_app_graph.retrying, used by
    adapters.aitk_translator and embedding_cache) does ``import time`` and calls
    ``time.sleep(...)``, so patching the shared stdlib module here covers every
    caller of it without a per-caller fixture.
    """
    delays: list[float] = []
    monkeypatch.setattr(time, "sleep", delays.append)
    return delays


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
    client, _ = make_client(
        {"api_key": "test-router-key"}, http_client=router_http, role="page_detail"
    )
    with client:
        yield client


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
        "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
    }


@pytest.fixture
def embedding_body() -> dict[str, object]:
    return {
        "object": "list",
        "model": "google/gemini-embedding-2",
        "data": [{"object": "embedding", "index": 0, "embedding": [0.25, 0.75]}],
        "usage": {"prompt_tokens": 5, "total_tokens": 5},
    }


@pytest.fixture
def png_screenshot() -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), "white").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


@pytest.fixture
def image_router(router_http: httpx.Client, monkeypatch: pytest.MonkeyPatch) -> None:
    def factory(
        cfg: Mapping[str, object] | None, *, default_model: str, role: str
    ) -> tuple[OpenRouter, str]:
        return make_client(cfg, default_model=default_model, role=role, http_client=router_http)

    monkeypatch.setattr(vlm_utils, "make_client", factory)
    monkeypatch.setattr(aitk_translator, "make_client", factory)


@pytest.fixture
def owned_router_clients(
    router_requests: list[httpx.Request],
    router_responses: list[httpx.Response],
    monkeypatch: pytest.MonkeyPatch,
) -> list[httpx.Client]:
    clients: list[httpx.Client] = []

    def respond(request: httpx.Request) -> httpx.Response:
        router_requests.append(request)
        return router_responses.pop(0)

    class TrackingClient(httpx.Client):
        def __init__(self, *, trust_env: bool = True) -> None:
            super().__init__(trust_env=trust_env, transport=httpx.MockTransport(respond))
            clients.append(self)

    monkeypatch.setattr(httpx, "Client", TrackingClient)
    return clients
