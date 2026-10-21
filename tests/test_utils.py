"""Property and example tests for android_app_graph.utils."""

from __future__ import annotations

import io
import json
import logging
import time
from collections.abc import Callable, Iterator

import httpx
import pytest
from hypothesis import given
from hypothesis import strategies as st
from openrouter import OpenRouter
from openrouter.components.security import Security
from openrouter.types import UNSET
from rich.console import Console

from android_app_graph.utils import (
    REMOVED_KEYS,
    ROLES,
    ModelSettings,
    open_client,
    resolve_api_key,
    resolve_audit_timeout,
    resolve_env,
    resolve_model_settings,
    vlm_utils,
)
from android_app_graph.utils import logging as app_logging


def _settings(role: str, cfg: dict[str, object]) -> ModelSettings:
    return resolve_model_settings({role: cfg}, roles=(role,))[role]


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


def test_open_client_sends_the_key_and_lets_the_sdk_own_its_http_client() -> None:
    client = open_client("test-key")
    assert isinstance(client.sdk_configuration.security, Security)
    assert client.sdk_configuration.security.api_key == "test-key"
    assert client.sdk_configuration.server_url is None
    http_client = client.sdk_configuration.client
    assert isinstance(http_client, httpx.Client)
    assert not client.sdk_configuration.client_supplied
    with client:
        assert not http_client.is_closed
    assert http_client.is_closed
    assert client.sdk_configuration.client is None


def test_open_client_keeps_the_sdk_default_retry_policy() -> None:
    assert open_client("test-key").sdk_configuration.retry_config is UNSET


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
    client = open_client(resolve_api_key(), http_client=router_http)
    client.embeddings.generate(model="google/gemini-embedding-2", input="hello")
    assert router_requests[0].headers["authorization"] == "Bearer shared-key"
    assert str(router_requests[0].url) == "https://openrouter.ai/api/v1/embeddings"


@pytest.mark.parametrize("key", [None, "", " \t "])
def test_a_missing_or_empty_openrouter_key_is_actionable(
    key: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "wrong-provider-key")
    if key is not None:
        monkeypatch.setenv("OPENROUTER_API_KEY", key)
    with pytest.raises(ValueError, match=r"^OPENROUTER_API_KEY is required"):
        resolve_api_key()


@pytest.mark.parametrize("key", REMOVED_KEYS)
@pytest.mark.parametrize("role", ROLES)
def test_removed_endpoint_and_credential_keys_are_rejected_in_every_role_block(
    role: str, key: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "dummy-secret-never-log"  # noqa: S105  # fake credential; the test proves it is never logged
    monkeypatch.setenv("APP_GRAPH_ACTION_MODEL", "qwen/qwen3.8-flash")
    monkeypatch.setenv("GEMINI_API_KEY", secret)
    with pytest.raises(ValueError, match=rf"{role}\.{key} was removed") as caught:
        resolve_model_settings({role: {key: "${GEMINI_API_KEY}"}}, roles=("action",))
    assert "OPENROUTER_API_KEY" in str(caught.value)
    assert secret not in str(caught.value)


def test_role_model_environment_reference_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg: dict[str, object] = {"model": "${APP_GRAPH_ACTION_MODEL}"}
    with pytest.raises(ValueError, match="APP_GRAPH_ACTION_MODEL"):
        _settings("action", cfg)
    monkeypatch.setenv("APP_GRAPH_ACTION_MODEL", "provider/comparison-model")
    assert _settings("action", cfg).model == "provider/comparison-model"


def test_one_client_sends_each_roles_model_temperature_and_timeout(
    router_client: OpenRouter,
    router_requests: list[httpx.Request],
    router_responses: list[httpx.Response],
    chat_response: Callable[..., httpx.Response],
    embedding_body: dict[str, object],
) -> None:
    router_responses.extend(
        [
            chat_response(content="one"),
            chat_response(content="two"),
            httpx.Response(200, json=embedding_body),
        ]
    )
    for settings in (
        ModelSettings("qwen/qwen3.8-flash", temperature=0.0, request_timeout=2.5),
        ModelSettings("openai/gpt-6.1-sol", request_timeout=7),
    ):
        vlm_utils.send_chat(
            router_client,
            call_type="chat",
            settings=settings,
            messages=[{"role": "user", "content": "hi"}],
        )
    vlm_utils.get_embedding(router_client, "home", "google/gemini-embedding-2", request_timeout=12)
    bodies = [json.loads(request.content) for request in router_requests]
    assert [(body["model"], body.get("temperature")) for body in bodies] == [
        ("qwen/qwen3.8-flash", 0.0),
        ("openai/gpt-6.1-sol", None),
        ("google/gemini-embedding-2", None),
    ]
    assert [request.extensions["timeout"]["read"] for request in router_requests] == [2.5, 7, 12]
    assert router_client.sdk_configuration.client is not None


def test_settings_default_to_a_sixty_second_timeout_and_no_temperature() -> None:
    assert _settings("action", {"model": "qwen/qwen3.8-flash"}) == ModelSettings(
        "qwen/qwen3.8-flash", None, 60.0
    )


@pytest.mark.parametrize(
    ("cfg", "timeout"),
    [({"timeout": 5}, 5.0), ({"request_timeout": 12, "timeout": 5}, 12.0)],
)
def test_settings_prefer_request_timeout_over_its_timeout_alias(
    cfg: dict[str, object], timeout: float
) -> None:
    assert _settings("action", {"model": "qwen/qwen3.8-flash", **cfg}).request_timeout == timeout


@pytest.mark.parametrize("timeout", ["invalid", "nan", "0"])
def test_invalid_timeout_is_a_configuration_error(timeout: str) -> None:
    with pytest.raises(
        ValueError, match=r"action\.request_timeout must be a positive finite number"
    ):
        _settings("action", {"model": "qwen/qwen3.8-flash", "request_timeout": timeout})


@pytest.mark.parametrize("temperature", [float("nan"), -0.1, 2.1, "warm", "", True])
def test_an_invalid_temperature_is_a_configuration_error(temperature: object) -> None:
    with pytest.raises(ValueError, match=r"action\.temperature must be a finite number"):
        _settings("action", {"model": "qwen/qwen3.8-flash", "temperature": temperature})


@pytest.mark.parametrize("temperature", [0, 2])
def test_settings_accept_the_temperature_bounds(temperature: int) -> None:
    settings = _settings("action", {"model": "qwen/qwen3.8-flash", "temperature": temperature})
    assert settings.temperature == temperature


@pytest.mark.parametrize("value", [None, "", " \t ", "${APP_GRAPH_UNSET_MODEL}", 123])
def test_settings_reject_a_missing_model(value: object) -> None:
    with pytest.raises(ValueError, match=r"action\.model is required: set APP_GRAPH_ACTION_MODEL"):
        _settings("action", {"model": value})


@pytest.mark.parametrize("value", ["", " \t ", "${APP_GRAPH_UNSET_MODEL}"])
def test_blank_environment_model_is_rejected(value: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_GRAPH_ACTION_MODEL", value)
    with pytest.raises(ValueError, match="APP_GRAPH_ACTION_MODEL"):
        _settings("action", {})


@pytest.mark.parametrize("value", ["", " \t ", "${APP_GRAPH_UNSET_MODEL}"])
def test_invalid_explicit_model_cannot_fall_back_to_environment(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("APP_GRAPH_ACTION_MODEL", "provider/action")
    with pytest.raises(ValueError, match=r"action.model"):
        _settings("action", {"model": value})


@pytest.mark.parametrize(("value", "seconds"), [(None, 300.0), (120, 120.0), ("45.5", 45.5)])
def test_the_audit_timeout_defaults_to_five_minutes(value: object, seconds: float) -> None:
    page_detail: dict[str, object] = {} if value is None else {"audit_request_timeout": value}
    assert resolve_audit_timeout({"page_detail": page_detail}) == seconds


@pytest.mark.parametrize("value", ["soon", "nan", 0, -1])
def test_an_invalid_audit_timeout_is_reported_with_the_page_detail_settings(value: object) -> None:
    config: dict[str, object] = {
        "page_detail": {"model": "openai/gpt-6.1-sol", "audit_request_timeout": value}
    }
    with pytest.raises(ValueError, match=r"page_detail\.audit_request_timeout must be a positive"):
        resolve_model_settings(config, roles=("page_detail",))
    resolve_model_settings({**config, "embedding": {"model": "e"}}, roles=("embedding",))


def test_every_invalid_setting_is_reported_on_one_line() -> None:
    config: dict[str, object] = {
        "page_detail": {"model": "openai/gpt-6.1-sol", "request_timeout": -1},
        "action": {"model": "qwen/qwen3.8-flash", "temperature": 3, "timeout": "soon"},
        "instruction": {"base_url": "https://provider.example/v1"},
    }
    with pytest.raises(ValueError, match=r"^Invalid vlm configuration: ") as caught:
        resolve_model_settings(config, roles=("page_detail", "embedding", "action"))
    message = str(caught.value)
    assert "\n" not in message
    for problem in (
        "instruction.base_url was removed",
        "page_detail.request_timeout",
        "embedding.model is required",
        "action.temperature",
        "action.request_timeout",
    ):
        assert problem in message


@pytest.fixture
def local_time_east_of_utc() -> Iterator[None]:
    """Make local time UTC+05:45, so a log time in UTC, or off by whole hours,
    cannot pass for local wall-clock time on a machine that runs in UTC."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("TZ", "NPT-05:45")
        time.tzset()
        try:
            yield
        finally:
            mp.undo()
            time.tzset()


@pytest.mark.usefixtures("local_time_east_of_utc")
def test_the_log_time_is_local_wall_clock_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """Operators read the log time as local time, so it must not switch to UTC."""
    created = 1_700_000_000.0
    local_time = time.strftime("%H:%M:%S", time.localtime(created))
    assert local_time != time.strftime("%H:%M:%S", time.gmtime(created))
    output = io.StringIO()
    monkeypatch.setattr(app_logging, "_console", Console(file=output, color_system=None))
    record = logging.LogRecord(
        "android_app_graph.kobe", logging.INFO, __file__, 1, "explored", None, None
    )
    record.created = created

    app_logging.ColorHandler().emit(record)

    assert output.getvalue().split(" ", 1)[0] == local_time
