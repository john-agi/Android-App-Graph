"""Property and example tests for android_app_graph.utils."""

from __future__ import annotations

import httpx
import pytest
from hypothesis import given
from hypothesis import strategies as st

from android_app_graph.utils import make_client, resolve_env


def test_none_passes_through() -> None:
    assert resolve_env(None) is None


@given(st.text(alphabet=st.characters(exclude_characters="$")))
def test_strings_without_dollar_are_returned_unchanged(value: str) -> None:
    """Only a full ${NAME} match is resolved; anything else is returned as given."""
    assert resolve_env(value) == value


def test_reference_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_GRAPH_TEST_VALUE", "hello")
    assert resolve_env("${APP_GRAPH_TEST_VALUE}") == "hello"


def test_reference_to_unset_variable_is_none() -> None:
    assert resolve_env("${APP_GRAPH_TEST_UNSET}") is None


def test_make_client_defaults() -> None:
    client, model = make_client({"api_key": "test-key"})
    assert model == "gpt-4o"
    assert client.api_key == "test-key"


def test_make_client_reads_the_config() -> None:
    client, model = make_client(
        {"api_key": "test-key", "base_url": "http://localhost:8000/v1", "model": "qwen"}
    )
    assert model == "qwen"
    assert client.api_key == "test-key"
    assert str(client.base_url) == "http://localhost:8000/v1/"


def test_make_client_falls_back_to_the_openai_api_key_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")
    client, _ = make_client({})
    assert client.api_key == "from-env"


def test_make_client_resolves_env_references(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_GRAPH_TEST_MODEL", "from-env")
    _, model = make_client({"api_key": "test-key", "model": "${APP_GRAPH_TEST_MODEL}"})
    assert model == "from-env"


def test_make_client_forwards_timeout_and_max_retries() -> None:
    client, _ = make_client({"api_key": "test-key"}, timeout=12.0, max_retries=3)
    assert client.timeout == 12.0
    assert client.max_retries == 3


def test_make_client_forwards_the_http_client() -> None:
    http_client = httpx.Client(trust_env=False)
    client, _ = make_client({"api_key": "test-key"}, http_client=http_client)
    assert client._client is http_client  # the SDK exposes no public accessor for this
