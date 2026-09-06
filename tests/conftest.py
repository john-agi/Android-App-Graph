"""Shared pytest configuration: Hypothesis profiles, credential and network guards."""

from __future__ import annotations

import os
import socket
import time
from collections.abc import Iterator
from typing import Any, NoReturn

import pytest
from hypothesis import settings

# The built-in "ci" profile (deterministic, no deadline, no example database) is
# selected automatically when CI is set; re-registered here with more examples.
settings.register_profile("ci", settings.get_profile("ci"), max_examples=500)
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "ci" if os.getenv("CI") else "default"))

_CREDENTIAL_PREFIXES = ("APP_GRAPH_",)
_CREDENTIAL_NAMES = frozenset({"GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"})


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
