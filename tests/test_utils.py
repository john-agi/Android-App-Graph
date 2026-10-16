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

from android_app_graph import utils
from android_app_graph.utils import client_scope, make_client, resolve_env, vlm_utils


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


@pytest.mark.parametrize(
    "role", ["page_detail", "instruction", "action", "embedding", "image_embedding"]
)
def test_missing_openrouter_key_is_actionable(role: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "wrong-provider-key")
    with pytest.raises(ValueError, match=rf"{role}\.api_key.*OPENROUTER_API_KEY"):
        make_client({"model": "openai/gpt-6.1-sol"}, role=role)


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


def test_make_client_defaults_to_a_sixty_second_timeout() -> None:
    client, model = make_client(
        {"model": "qwen/qwen3.8-flash", "api_key": "test-key"}, role="action"
    )
    assert model == "qwen/qwen3.8-flash"
    assert client.sdk_configuration.retry_config is None
    assert client.sdk_configuration.timeout_ms == 60_000


@pytest.mark.parametrize(
    ("cfg", "timeout_ms"),
    [({"timeout": 5}, 5_000), ({"request_timeout": 12, "timeout": 5}, 12_000)],
)
def test_make_client_prefers_request_timeout_over_its_timeout_alias(
    cfg: dict[str, object], timeout_ms: int
) -> None:
    client, _ = make_client(
        {"model": "qwen/qwen3.8-flash", "api_key": "test-key", **cfg}, role="action"
    )
    assert client.sdk_configuration.timeout_ms == timeout_ms


@pytest.mark.parametrize("timeout", ["invalid", "nan", "0"])
def test_invalid_timeout_is_a_configuration_error(timeout: str) -> None:
    with pytest.raises(
        ValueError, match=r"action\.request_timeout must be a positive finite number"
    ):
        make_client(
            {"model": "qwen/qwen3.8-flash", "api_key": "test", "request_timeout": timeout},
            role="action",
        )


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
        make_client({"api_key": "test-key", "model": value}, role=role)
    assert owned_router_clients == []


@pytest.mark.parametrize("value", ["", " \t ", "${APP_GRAPH_UNSET_MODEL}"])
def test_blank_environment_model_is_rejected(value: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_GRAPH_ACTION_MODEL", value)
    with pytest.raises(ValueError, match="APP_GRAPH_ACTION_MODEL"):
        make_client({"api_key": "test-key"}, role="action")


@pytest.mark.parametrize("value", ["", " \t ", "${APP_GRAPH_UNSET_MODEL}"])
def test_invalid_explicit_model_cannot_fall_back_to_environment(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("APP_GRAPH_ACTION_MODEL", "provider/action")
    with pytest.raises(ValueError, match=r"action.model"):
        make_client({"api_key": "test-key", "model": value}, role="action")


@pytest.mark.parametrize(
    ("role", "endpoint", "message"),
    [
        (
            "embedding",
            {"base_url": "https://generativelanguage.googleapis.com/v1beta"},
            "native Google embedding endpoints",
        ),
        (
            "embedding",
            {"base_url": "https://us-central1-aiplatform.googleapis.com/v1"},
            "native Google embedding endpoints",
        ),
        (
            "image_embedding",
            {"native_base_url": "https://generativelanguage.googleapis.com/v1beta"},
            r"image_embedding\.native_base_url is unsupported",
        ),
    ],
)
def test_native_google_embedding_configuration_fails_before_client_creation(
    role: str,
    endpoint: dict[str, str],
    message: str,
    owned_router_clients: list[httpx.Client],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "dummy-google-secret-never-log"
    monkeypatch.setenv("GEMINI_API_KEY", secret)
    with pytest.raises(ValueError, match=message) as caught:
        make_client(
            {**endpoint, "api_key": "${GEMINI_API_KEY}", "model": "gemini-embedding-2"}, role=role
        )
    assert secret not in str(caught.value)
    assert f"{role}.base_url" in str(caught.value)
    assert "OPENROUTER_API_KEY" in str(caught.value)
    assert owned_router_clients == []


@pytest.mark.parametrize("endpoint", ["ftp://other.example/v1", "https:///v1"])
def test_invalid_endpoint_fails_before_client_creation(
    endpoint: str, owned_router_clients: list[httpx.Client]
) -> None:
    with pytest.raises(ValueError, match="absolute HTTP or HTTPS"):
        make_client(
            {"model": "qwen/qwen3.8-flash", "base_url": endpoint, "api_key": "test"}, role="action"
        )
    assert owned_router_clients == []


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://other.example/v1",
        "http://openrouter.ai/api/v1",
        "https://openrouter.ai:444/api/v1",
        "https://openrouter.ai:0/api/v1",
    ],
)
def test_shared_router_key_never_falls_back_to_a_different_origin(
    endpoint: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-secret")
    with pytest.raises(ValueError, match=r"action\.api_key") as caught:
        make_client({"model": "openai/gpt-6.1-sol", "base_url": endpoint}, role="action")
    assert "router-secret" not in str(caught.value)


def test_unresolved_endpoint_cannot_redirect_an_explicit_credential() -> None:
    with pytest.raises(ValueError, match="endpoint") as caught:
        make_client(
            {"base_url": "${APP_GRAPH_ENDPOINT}", "api_key": "endpoint-secret"}, role="action"
        )
    assert "endpoint-secret" not in str(caught.value)


@pytest.mark.parametrize(
    ("endpoint", "trust_env"),
    [
        ("https://router.example/v1", True),
        ("http://localhost:8000/v1", False),
        ("http://127.0.0.1:8000/v1", False),
        ("http://[::1]:8000/v1", False),
    ],
)
def test_only_loopback_endpoints_ignore_the_proxy_environment(
    endpoint: str, trust_env: bool
) -> None:
    client, _ = make_client(
        {"model": "qwen/qwen3.8-flash", "api_key": "test", "base_url": endpoint}, role="action"
    )
    assert isinstance(client.sdk_configuration.client, httpx.Client)
    assert client.sdk_configuration.client.trust_env is trust_env
    client.sdk_configuration.client.close()


@pytest.mark.parametrize("borrowed", [False, True])
def test_factory_initialization_failure_keeps_correct_owner(
    borrowed: bool,
    owned_router_clients: list[httpx.Client],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supplied = httpx.Client() if borrowed else None

    def fail_sdk(**_kwargs: object) -> OpenRouter:
        raise RuntimeError("SDK construction failed")

    monkeypatch.setattr(utils, "ConfiguredOpenRouter", fail_sdk)
    with pytest.raises(RuntimeError, match="SDK construction failed"):
        make_client(
            {"model": "qwen/qwen3.8-flash", "api_key": "test"}, role="action", http_client=supplied
        )
    assert len(owned_router_clients) == 1
    assert owned_router_clients[0].is_closed is not borrowed
    if supplied is not None:
        supplied.close()


@pytest.mark.parametrize("scope", ["explicit", "failed", "implicit"])
def test_owned_http_client_closed_on_sdk_scope_exit(
    scope: str,
    owned_router_clients: list[httpx.Client],
    router_responses: list[httpx.Response],
    chat_response: Callable[..., httpx.Response],
) -> None:
    client, model = make_client({"model": "qwen/qwen3.8-flash", "api_key": "test"}, role="action")
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
def test_nested_scopes_keep_outer_connection_alive(
    fail_inner: bool,
    owned_router_clients: list[httpx.Client],
    router_responses: list[httpx.Response],
    chat_body: dict[str, object],
) -> None:
    client, model = make_client({"model": "qwen/qwen3.8-flash", "api_key": "test"}, role="action")
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


def test_borrowed_client_keeps_its_policy_and_owner_across_sdk_scopes(
    router_http: httpx.Client,
    router_responses: list[httpx.Response],
    chat_response: Callable[..., httpx.Response],
) -> None:
    events: list[str] = []
    router_http.event_hooks["response"].append(lambda _response: events.append("response"))
    client, model = make_client(
        {"model": "openai/gpt-6.1-sol", "api_key": "test"},
        role="page_detail",
        http_client=router_http,
    )
    router_responses.extend(chat_response(content="test reply") for _ in range(2))
    for _ in range(2):
        with client:
            assert client.sdk_configuration.client is router_http
            vlm_utils.send_chat(client, model=model, messages=[])
        assert not router_http.follow_redirects
        assert not router_http.is_closed
    assert events == ["response", "response"]


@pytest.mark.parametrize(
    ("destination", "foreign"),
    [("/canonical", False), ("https://other.example/canonical", True)],
)
def test_owned_clients_follow_redirects_after_reopening(
    destination: str,
    foreign: bool,
    owned_router_clients: list[httpx.Client],
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    embedding_body: dict[str, object],
) -> None:
    client, _ = make_client(
        {"model": "google/gemini-embedding-2", "api_key": "test-key"}, role="embedding"
    )
    for _ in range(2):
        router_responses.extend(
            [
                httpx.Response(307, headers={"location": destination}),
                httpx.Response(200, json=embedding_body),
            ]
        )
        with client:
            assert vlm_utils.get_embedding(client, "home", model="google/gemini-embedding-2") == [
                0.25,
                0.75,
            ]
        first, redirected = router_requests[-2:]
        assert first.method == redirected.method == "POST"
        assert first.content == redirected.content
        assert first.headers["authorization"] == "Bearer test-key"
        assert redirected.headers.get("authorization") == (None if foreign else "Bearer test-key")
    assert len(router_requests) == 4
    assert len(owned_router_clients) == 2
    assert all(client.is_closed for client in owned_router_clients)
