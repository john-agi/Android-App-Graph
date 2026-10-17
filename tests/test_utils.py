"""Property and example tests for android_app_graph.utils."""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest
from hypothesis import given
from hypothesis import strategies as st
from openrouter import OpenRouter
from openrouter.components.security import Security

from android_app_graph.utils import (
    REMOVED_KEYS,
    ROLES,
    ConfiguredOpenRouter,
    client_scope,
    make_client,
    resolve_env,
    resolve_model_roles,
    vlm_utils,
)


@pytest.fixture
def router_key(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    return "test-key"


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


def test_make_client_reads_the_model_and_the_openrouter_key(router_key: str) -> None:
    client, model = make_client({"model": "qwen"}, role="page_detail")
    assert model == "qwen"
    assert isinstance(client.sdk_configuration.security, Security)
    assert client.sdk_configuration.security.api_key == router_key
    assert client.sdk_configuration.server_url is None


@pytest.mark.usefixtures("router_key")
def test_make_client_forwards_timeout_and_disables_sdk_retries() -> None:
    client, _ = make_client({"model": "openai/gpt-6.1-sol"}, timeout=12.0, role="page_detail")
    assert client.sdk_configuration.timeout_ms == 12_000
    assert client.sdk_configuration.retry_config is None


def test_client_scope_leaves_an_external_sdk_client_open(router_http: httpx.Client) -> None:
    client = OpenRouter(api_key="test-key", client=router_http)
    with client_scope(client) as scoped:
        assert scoped is client
    assert client.sdk_configuration.client is router_http


@pytest.mark.usefixtures("router_key")
def test_make_client_forwards_the_http_client() -> None:
    http_client = httpx.Client(trust_env=False)
    client, _ = make_client(
        {"model": "openai/gpt-6.1-sol"}, http_client=http_client, role="page_detail"
    )
    assert client.sdk_configuration.client is http_client


def test_the_openrouter_key_and_endpoint_are_the_only_ones_on_the_wire(
    router_http: httpx.Client,
    router_requests: list[httpx.Request],
    router_responses: list[httpx.Response],
    embedding_body: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "shared-key")
    monkeypatch.setenv("OPENAI_API_KEY", "wrong-provider-key")
    router_responses.append(httpx.Response(200, json=embedding_body))
    client, model = make_client(
        {"model": "google/gemini-embedding-2"}, http_client=router_http, role="embedding"
    )
    client.embeddings.generate(model=model, input="hello")
    assert router_requests[0].headers["authorization"] == "Bearer shared-key"
    assert str(router_requests[0].url) == "https://openrouter.ai/api/v1/embeddings"


@pytest.mark.parametrize("key", [None, "", " \t "])
@pytest.mark.parametrize("role", ROLES)
def test_a_missing_or_empty_openrouter_key_is_actionable(
    role: str, key: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "wrong-provider-key")
    if key is not None:
        monkeypatch.setenv("OPENROUTER_API_KEY", key)
    with pytest.raises(ValueError, match=r"^OPENROUTER_API_KEY is required"):
        make_client({"model": "openai/gpt-6.1-sol"}, role=role)


@pytest.mark.parametrize("key", REMOVED_KEYS)
@pytest.mark.parametrize("role", ROLES)
def test_removed_endpoint_and_credential_keys_are_rejected_in_every_role_block(
    role: str, key: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "dummy-secret-never-log"
    monkeypatch.setenv("APP_GRAPH_ACTION_MODEL", "qwen/qwen3.8-flash")
    monkeypatch.setenv("GEMINI_API_KEY", secret)
    with pytest.raises(ValueError, match=rf"{role}\.{key} was removed") as caught:
        resolve_model_roles({role: {key: "${GEMINI_API_KEY}"}}, roles=("action",))
    assert "OPENROUTER_API_KEY" in str(caught.value)
    assert secret not in str(caught.value)


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


@pytest.mark.usefixtures("router_key")
def test_request_timeout_is_sent_to_transport(
    router_http: httpx.Client,
    router_requests: list[httpx.Request],
    router_responses: list[httpx.Response],
    embedding_body: dict[str, object],
) -> None:
    router_responses.append(httpx.Response(200, json=embedding_body))
    client, model = make_client(
        {"model": "google/gemini-embedding-2", "request_timeout": 12},
        http_client=router_http,
        role="embedding",
    )
    client.embeddings.generate(model=model, input="hello")
    request = router_requests[0]
    assert request.extensions["timeout"] == {
        "connect": 12.0,
        "read": 12.0,
        "write": 12.0,
        "pool": 12.0,
    }
    assert json.loads(request.content)["model"] == model


@pytest.mark.usefixtures("router_key")
def test_make_client_defaults_to_a_sixty_second_timeout() -> None:
    client, model = make_client({"model": "qwen/qwen3.8-flash"}, role="action")
    assert model == "qwen/qwen3.8-flash"
    assert client.sdk_configuration.retry_config is None
    assert client.sdk_configuration.timeout_ms == 60_000


@pytest.mark.parametrize(
    ("cfg", "timeout_ms"),
    [({"timeout": 5}, 5_000), ({"request_timeout": 12, "timeout": 5}, 12_000)],
)
@pytest.mark.usefixtures("router_key")
def test_make_client_prefers_request_timeout_over_its_timeout_alias(
    cfg: dict[str, object], timeout_ms: int
) -> None:
    client, _ = make_client({"model": "qwen/qwen3.8-flash", **cfg}, role="action")
    assert client.sdk_configuration.timeout_ms == timeout_ms


@pytest.mark.parametrize("timeout", ["invalid", "nan", "0"])
def test_invalid_timeout_is_a_configuration_error(timeout: str) -> None:
    with pytest.raises(
        ValueError, match=r"action\.request_timeout must be a positive finite number"
    ):
        make_client({"model": "qwen/qwen3.8-flash", "request_timeout": timeout}, role="action")


@pytest.mark.parametrize("temperature", [float("nan"), -0.1, 2.1, "warm", "", True])
def test_make_client_rejects_an_invalid_temperature(temperature: object) -> None:
    with pytest.raises(ValueError, match=r"action\.temperature must be a finite number"):
        make_client({"model": "qwen/qwen3.8-flash", "temperature": temperature}, role="action")


@pytest.mark.parametrize("temperature", [0, 2])
@pytest.mark.usefixtures("router_key")
def test_make_client_accepts_the_temperature_bounds(temperature: int) -> None:
    client, _ = make_client(
        {"model": "qwen/qwen3.8-flash", "temperature": temperature}, role="action"
    )
    assert isinstance(client, ConfiguredOpenRouter)
    assert client.temperature == temperature


@pytest.mark.parametrize(
    ("role", "value", "variable"),
    [
        *(
            ("action", value, "APP_GRAPH_ACTION_MODEL")
            for value in (None, "", " \t ", "${APP_GRAPH_UNSET_MODEL}", 123)
        ),
        ("image_embedding", None, "APP_GRAPH_EMBEDDING_MODEL"),
    ],
)
def test_client_rejects_missing_model(
    role: str, value: object, variable: str, owned_router_clients: list[httpx.Client]
) -> None:
    with pytest.raises(ValueError, match=rf"{role}\.model is required: set {variable}"):
        make_client({"model": value}, role=role)
    assert owned_router_clients == []


@pytest.mark.parametrize("value", ["", " \t ", "${APP_GRAPH_UNSET_MODEL}"])
def test_blank_environment_model_is_rejected(value: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_GRAPH_ACTION_MODEL", value)
    with pytest.raises(ValueError, match="APP_GRAPH_ACTION_MODEL"):
        make_client({}, role="action")


@pytest.mark.parametrize("value", ["", " \t ", "${APP_GRAPH_UNSET_MODEL}"])
def test_invalid_explicit_model_cannot_fall_back_to_environment(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("APP_GRAPH_ACTION_MODEL", "provider/action")
    with pytest.raises(ValueError, match=r"action.model"):
        make_client({"model": value}, role="action")


@pytest.mark.parametrize("scope", ["explicit", "failed", "implicit"])
@pytest.mark.usefixtures("router_key")
def test_owned_http_client_closed_on_sdk_scope_exit(
    scope: str,
    owned_router_clients: list[httpx.Client],
    router_responses: list[httpx.Response],
    chat_response: Callable[..., httpx.Response],
) -> None:
    client, model = make_client({"model": "qwen/qwen3.8-flash"}, role="action")
    if scope == "implicit":
        router_responses.append(chat_response(content="test reply"))
        reply = vlm_utils.send_chat(
            client, model=model, messages=[{"role": "user", "content": "hi"}]
        )
        assert vlm_utils.message_text(reply) == "test reply"
    elif scope == "failed":
        with pytest.raises(RuntimeError, match="scope failed"), client:
            raise RuntimeError("scope failed")
    else:
        with client:
            assert not owned_router_clients[0].is_closed
    assert len(owned_router_clients) == 1
    assert owned_router_clients[0].is_closed


@pytest.mark.parametrize("fail_inner", [False, True])
@pytest.mark.usefixtures("router_key")
def test_nested_scopes_keep_outer_connection_alive(
    fail_inner: bool,
    owned_router_clients: list[httpx.Client],
    router_responses: list[httpx.Response],
    chat_body: dict[str, object],
) -> None:
    client, model = make_client({"model": "qwen/qwen3.8-flash"}, role="action")
    router_responses.extend(httpx.Response(200, json=chat_body) for _ in range(2))
    with client:
        connection = client.sdk_configuration.client
        try:
            with client:
                vlm_utils.send_chat(client, model=model, messages=[])
                if fail_inner:
                    raise RuntimeError("inner failed")
        except RuntimeError:
            assert fail_inner
        assert isinstance(connection, httpx.Client)
        assert not connection.is_closed
        vlm_utils.send_chat(client, model=model, messages=[])
        assert client.sdk_configuration.client is connection
    assert len(owned_router_clients) == 1
    assert connection.is_closed
