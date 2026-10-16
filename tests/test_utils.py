"""Property and example tests for android_app_graph.utils."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import httpx
import pytest
from hypothesis import given
from hypothesis import strategies as st
from openrouter import OpenRouter
from openrouter.components.security import Security

from android_app_graph.kobe import Kobe
from android_app_graph.utils import client_scope, make_client, resolve_env


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


def test_make_client_reads_the_config() -> None:
    client, model = make_client(
        {"api_key": "test-key", "base_url": "http://localhost:8000/v1", "model": "qwen"},
        role="page_detail",
    )
    assert model == "qwen"
    assert isinstance(client.sdk_configuration.security, Security)
    assert client.sdk_configuration.security.api_key == "test-key"
    assert client.sdk_configuration.server_url == "http://localhost:8000/v1"


def test_make_client_falls_back_to_the_openrouter_api_key_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "from-env")
    client, _ = make_client({"model": "openai/gpt-6.1-sol"}, role="page_detail")
    assert isinstance(client.sdk_configuration.security, Security)
    assert client.sdk_configuration.security.api_key == "from-env"


def test_make_client_resolves_env_references(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_GRAPH_TEST_MODEL", "from-env")
    _, model = make_client(
        {"api_key": "test-key", "model": "${APP_GRAPH_TEST_MODEL}"}, role="page_detail"
    )
    assert model == "from-env"


def test_make_client_forwards_timeout_and_disables_sdk_retries() -> None:
    client, _ = make_client(
        {"model": "openai/gpt-6.1-sol", "api_key": "test-key"}, timeout=12.0, role="page_detail"
    )
    assert client.sdk_configuration.timeout_ms == 12_000
    assert client.sdk_configuration.retry_config is None


def test_client_scope_leaves_an_external_sdk_client_open(router_http: httpx.Client) -> None:
    client = OpenRouter(api_key="test-key", client=router_http)
    with client_scope(client) as scoped:
        assert scoped is client
    assert client.sdk_configuration.client is router_http


def test_make_client_forwards_the_http_client() -> None:
    http_client = httpx.Client(trust_env=False)
    client, _ = make_client(
        {"model": "openai/gpt-6.1-sol", "api_key": "test-key"},
        http_client=http_client,
        role="page_detail",
    )
    assert client.sdk_configuration.client is http_client


@pytest.mark.parametrize(
    "api_key", ["explicit-key", "${APP_GRAPH_TEST_KEY}", "${APP_GRAPH_TEST_UNSET}", None, ""]
)
def test_credential_precedence_on_the_wire(
    api_key: str | None,
    router_http: httpx.Client,
    router_requests: list[httpx.Request],
    router_responses: list[httpx.Response],
    embedding_body: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "shared-key")
    monkeypatch.setenv("OPENAI_API_KEY", "wrong-provider-key")
    monkeypatch.setenv("APP_GRAPH_TEST_KEY", "role-key")
    router_responses.append(httpx.Response(200, json=embedding_body))
    client, model = make_client(
        {"api_key": api_key, "model": "google/gemini-embedding-2"},
        http_client=router_http,
        role="embedding",
    )
    client.embeddings.generate(model=model, input="hello")
    expected = {"explicit-key": "explicit-key", "${APP_GRAPH_TEST_KEY}": "role-key"}.get(
        api_key or "", "shared-key"
    )
    assert router_requests[0].headers["authorization"] == f"Bearer {expected}"
    assert str(router_requests[0].url) == "https://openrouter.ai/api/v1/embeddings"


def test_missing_openrouter_key_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "wrong-provider-key")
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        make_client({"model": "openai/gpt-6.1-sol"}, role="page_detail")


@pytest.mark.parametrize("role", ["PAGE_DETAIL", "INSTRUCTION", "ACTION", "EMBEDDING"])
def test_role_model_environment_reference_is_required(
    role: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "shared-key")
    cfg = {"model": f"${{APP_GRAPH_{role}_MODEL}}"}
    with pytest.raises(ValueError, match=f"APP_GRAPH_{role}_MODEL"):
        make_client(cfg, role=role.lower())
    monkeypatch.setenv(f"APP_GRAPH_{role}_MODEL", "provider/comparison-model")
    _, model = make_client(cfg, role=role.lower())
    assert model == "provider/comparison-model"


def test_custom_endpoint_and_timeout_are_sent_to_transport(
    router_http: httpx.Client,
    router_requests: list[httpx.Request],
    router_responses: list[httpx.Response],
    embedding_body: dict[str, object],
) -> None:
    router_responses.append(httpx.Response(200, json=embedding_body))
    client, model = make_client(
        {
            "api_key": "role-key",
            "base_url": "https://proxy.example/api/v1/",
            "model": "google/gemini-embedding-2",
            "request_timeout": 12,
        },
        http_client=router_http,
        role="embedding",
    )
    client.embeddings.generate(model=model, input="hello")
    request = router_requests[0]
    assert str(request.url) == "https://proxy.example/api/v1/embeddings"
    assert request.extensions["timeout"] == {
        "connect": 12.0,
        "read": 12.0,
        "write": 12.0,
        "pool": 12.0,
    }
    assert json.loads(request.content)["model"] == model


class _IdleDevice:
    def __init__(self) -> None:
        self.w = 100
        self.h = 200
        self.config: dict[str, object] = {}

    def get_state(self) -> dict[str, object]:
        raise AssertionError("model setup must not access the device")

    def exe_action(self, action: dict[str, object]) -> None:
        raise AssertionError(f"model setup must not execute device action {action}")


@pytest.mark.parametrize("references", [False, True])
def test_exploration_resolves_all_role_models_before_device_access(
    references: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "shared-router-key")
    config: dict[str, object] = {}
    for role in ("page_detail", "embedding", "instruction", "action"):
        if references:
            config[role] = {"model": f"${{APP_GRAPH_{role.upper()}_MODEL}}"}
        monkeypatch.setenv(f"APP_GRAPH_{role.upper()}_MODEL", f"provider/{role}")
    agent = Kobe(
        _IdleDevice(),
        "demo",
        "com.demo.app",
        logging.getLogger("test"),
        vlm_config=config,
        graph_dir=tmp_path,
    )
    assert agent.page_detail_model == "provider/page_detail"
    assert agent.embedding_model == "provider/embedding"
    assert agent.instruction_model == "provider/instruction"
    assert agent.action_model == "provider/action"
    assert agent.graph.embedding_model == agent.embedding_model
    assert agent.graph.page_detail_model == agent.page_detail_model
