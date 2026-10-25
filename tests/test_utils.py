"""Property and example tests for android_app_graph.utils."""

from __future__ import annotations

import io
import json
import logging
import re
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import httpx
import pytest
import yaml
from hypothesis import given
from hypothesis import strategies as st
from openrouter import OpenRouter
from openrouter.components.security import Security
from rich.console import Console

from android_app_graph.utils import (
    DEFAULT_MATCHING,
    REQUEST_TIMEOUTS,
    ROLES,
    MatchingSettings,
    ModelSettings,
    check_config_keys,
    open_client,
    resolve_api_key,
    resolve_audit_timeout,
    resolve_env,
    resolve_matching_settings,
    resolve_model_settings,
    resolve_unreachable_limit,
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


def test_open_client_sends_the_key_and_closes_the_http_client_it_built() -> None:
    with open_client("test-key") as client:
        assert isinstance(client.sdk_configuration.security, Security)
        assert client.sdk_configuration.security.api_key == "test-key"
        assert client.sdk_configuration.server_url is None
        http_client = client.sdk_configuration.client
        assert isinstance(http_client, httpx.Client)
        assert http_client.follow_redirects
        assert not http_client.is_closed
    assert http_client.is_closed
    assert client.sdk_configuration.client is None


def test_open_client_leaves_an_http_client_it_was_given_open(router_http: httpx.Client) -> None:
    with open_client("test-key", http_client=router_http) as client:
        assert client.sdk_configuration.client is router_http
    assert client.sdk_configuration.client is None
    assert not router_http.is_closed


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
    with open_client(resolve_api_key(), http_client=router_http) as client:
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


_EVERY_ACCEPTED_KEY: dict[str, object] = {
    "device": {"udid": "emulator-5554", "avd_name": "Pixel", "unreachable_limit": 1800},
    "experiment": {
        "graph_dir": "graphs",
        "max_steps": 10,
        "resume_from": "auto",
        "coverage_checkpoint_steps": 5,
        "coverage_checkpoint_top_k": 3,
    },
    "apps": [
        {"name": "demo", "package_name": "com.demo"},
        {"name": "other", "package_name": "com.other", "launch_activity": "com.other/.Main"},
    ],
    "vlm": {
        "page_detail": {
            "model": "m",
            "temperature": 0,
            "request_timeout": 30,
            "timeout": 30,
            "audit_request_timeout": 300,
        },
        "embedding": {"model": "m", "request_timeout": 30, "timeout": 30},
        "image_embedding": {"model": "m", "request_timeout": 30, "timeout": 30},
        "instruction": {"model": "m", "temperature": 0, "request_timeout": 30, "timeout": 30},
        "action": {"model": "m", "temperature": 0, "request_timeout": 30, "timeout": 30},
        "matching": {"verifier_calls": 5, "image_candidates": 3, "text_candidates": 2},
    },
}


def test_every_key_a_command_uses_is_accepted() -> None:
    check_config_keys(_EVERY_ACCEPTED_KEY)


def test_every_committed_config_is_accepted() -> None:
    configs = sorted((Path(__file__).resolve().parents[1] / "configs").glob("*.yaml"))
    assert configs
    for config in configs:
        with config.open(encoding="utf-8") as f:
            check_config_keys(yaml.safe_load(f))


def test_an_unknown_top_level_key_is_rejected_naming_it() -> None:
    with pytest.raises(ValueError, match=r"^Unknown configuration keys: devices$"):
        check_config_keys({"devices": {"udid": "emulator-5554"}})


def test_an_unknown_vlm_key_is_rejected_naming_its_path() -> None:
    with pytest.raises(ValueError, match=r"^Unknown configuration keys: vlm\.similarity$"):
        check_config_keys({"vlm": {"similarity": 0.9}})


def test_an_unknown_role_key_is_rejected_naming_its_path_and_never_its_value() -> None:
    value = "sk-or-v1-not-a-real-key"
    with pytest.raises(
        ValueError, match=r"^Unknown configuration keys: vlm\.instruction\.api_key$"
    ) as caught:
        check_config_keys({"vlm": {"instruction": {"model": "m", "api_key": value}}})
    assert value not in str(caught.value)


@pytest.mark.parametrize(
    ("config", "path"),
    [
        ({"vlm": {"embedding": {"temperature": 0}}}, "vlm.embedding.temperature"),
        ({"vlm": {"action": {"audit_request_timeout": 9}}}, "vlm.action.audit_request_timeout"),
        ({"vlm": {"vision": {"model": "m"}}}, "vlm.vision"),
        ({"experiment": {"name": "demo"}}, "experiment.name"),
        ({"apps": [{"name": "demo", "package": "com.demo"}]}, "apps[0].package"),
        ({"apps": {"name": "demo"}}, "apps.name"),
        ({"vlm": [{"action": {}}]}, "vlm[0].action"),
        ({1: "one"}, "1"),
    ],
)
def test_a_key_no_command_uses_is_rejected(config: dict[object, object], path: str) -> None:
    with pytest.raises(ValueError, match=rf"^Unknown configuration keys: {re.escape(path)}$"):
        check_config_keys(config)


def test_every_unknown_key_is_reported_at_once_in_file_order() -> None:
    config: dict[str, object] = {
        "device": {"udid": "emulator-5554", "serial": "emulator-5554"},
        "apps": [
            {"name": "demo", "package_name": "com.demo"},
            {"name": "other", "package_name": "com.other", "launch_activty": "com.other/.Main"},
        ],
        "vlm": {"action": {"model": "m", "max_tokens": 512}},
        "extra": True,
    }
    with pytest.raises(ValueError, match=r"^Unknown configuration keys: ") as caught:
        check_config_keys(config)
    assert str(caught.value) == (
        "Unknown configuration keys: device.serial, apps[1].launch_activty, "
        "vlm.action.max_tokens, extra"
    )


@pytest.mark.parametrize("config", [None, {"vlm": None, "apps": None, "device": "emulator"}])
def test_a_missing_or_empty_section_has_no_unknown_keys(config: object) -> None:
    check_config_keys(config)


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


def test_settings_set_no_timeout_or_temperature_the_config_leaves_out() -> None:
    assert _settings("action", {"model": "qwen/qwen3.8-flash"}) == ModelSettings(
        "qwen/qwen3.8-flash", None, None
    )


_TIMEOUTS_BY_ROLE: dict[str, dict[str, float]] = {
    "page_detail": {
        "page_describe_and_state": 55,
        "node_verify": 35,
        "replay_source": 35,
        "normalize_edge": 50,
        "exploration_target": 25,
        "node_merge_audit": 300,
        "graph_audit": 300,
    },
    "embedding": {"embedding": 15},
    "image_embedding": {"image_embedding": 15},
    "instruction": {"instruction": 70, "action_description": 70},
    "action": {"action": 35},
}


def test_each_role_s_settings_give_each_of_its_call_types_its_own_timeout() -> None:
    """The timeouts README.md#requests-and-recovery states, for every call type."""
    settings = resolve_model_settings({role: {"model": "m"} for role in ROLES}, roles=ROLES)
    assert {
        role: {call_type: settings[role].timeout(call_type) for call_type in timeouts}
        for role, timeouts in _TIMEOUTS_BY_ROLE.items()
    } == _TIMEOUTS_BY_ROLE
    assert {
        call_type: timeout
        for timeouts in _TIMEOUTS_BY_ROLE.values()
        for call_type, timeout in timeouts.items()
    } == REQUEST_TIMEOUTS


def test_a_role_s_configured_timeout_is_the_timeout_of_each_of_its_call_types() -> None:
    settings = _settings("page_detail", {"model": "m", "timeout": "12"})
    assert {settings.timeout(call_type) for call_type in _TIMEOUTS_BY_ROLE["page_detail"]} == {12.0}


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
def test_settings_reject_a_missing_model(role: str, value: object, variable: str) -> None:
    with pytest.raises(ValueError, match=rf"{role}\.model is required: set {variable}"):
        _settings(role, {"model": value})


def test_image_embedding_settings_inherit_the_embedding_model_and_timeout() -> None:
    config: dict[str, object] = {
        "embedding": {"model": "voyageai/voyage-multimodal-3.5", "timeout": 9},
        "image_embedding": {},
    }
    assert resolve_model_settings(config, roles=("image_embedding",)) == {
        "image_embedding": ModelSettings("voyageai/voyage-multimodal-3.5", None, 9.0)
    }


def test_image_embeddings_read_the_embedding_model_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("APP_GRAPH_EMBEDDING_MODEL", "voyageai/voyage-multimodal-3.5")
    settings = resolve_model_settings({}, roles=("embedding", "image_embedding"))
    assert settings["image_embedding"] == settings["embedding"]
    assert settings["image_embedding"].model == "voyageai/voyage-multimodal-3.5"


@pytest.mark.parametrize("timeout_key", ["request_timeout", "timeout"])
def test_an_image_embedding_block_overrides_what_it_would_inherit(timeout_key: str) -> None:
    config: dict[str, object] = {
        "embedding": {"model": "provider/text", "request_timeout": 7.5},
        "image_embedding": {"model": "provider/image", timeout_key: 2.25},
    }
    assert resolve_model_settings(config, roles=("image_embedding",)) == {
        "image_embedding": ModelSettings("provider/image", None, 2.25)
    }


def test_an_explicit_invalid_image_embedding_model_does_not_inherit() -> None:
    config: dict[str, object] = {
        "embedding": {"model": "provider/text"},
        "image_embedding": {"model": ""},
    }
    with pytest.raises(ValueError, match=r"image_embedding\.model is required"):
        resolve_model_settings(config, roles=("image_embedding",))


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


@pytest.mark.parametrize(
    ("device", "seconds"),
    [({}, 1800.0), ({"unreachable_limit": 600}, 600.0), ({"unreachable_limit": "90.5"}, 90.5)],
)
def test_the_unreachable_limit_defaults_to_thirty_minutes(
    device: dict[str, object], seconds: float
) -> None:
    assert resolve_unreachable_limit({"device": device}) == seconds


@pytest.mark.parametrize("value", ["soon", "nan", 0, -1])
def test_an_invalid_unreachable_limit_is_a_configuration_error(value: object) -> None:
    with pytest.raises(
        ValueError,
        match=r"^device\.unreachable_limit must be a positive finite number of seconds$",
    ):
        resolve_unreachable_limit({"device": {"unreachable_limit": value}})


def test_every_invalid_setting_is_reported_on_one_line() -> None:
    config: dict[str, object] = {
        "page_detail": {"model": "openai/gpt-6.1-sol", "request_timeout": -1},
        "action": {"model": "qwen/qwen3.8-flash", "temperature": 3, "timeout": "soon"},
    }
    with pytest.raises(ValueError, match=r"^Invalid vlm configuration: ") as caught:
        resolve_model_settings(config, roles=("page_detail", "embedding", "action"))
    message = str(caught.value)
    assert "\n" not in message
    for problem in (
        "page_detail.request_timeout",
        "embedding.model is required",
        "action.temperature",
        "action.request_timeout",
    ):
        assert problem in message


def test_matching_settings_the_config_leaves_out_take_their_defaults() -> None:
    assert resolve_matching_settings({}) == DEFAULT_MATCHING == MatchingSettings(3, 3, 2)
    assert resolve_matching_settings({"matching": None}) == DEFAULT_MATCHING


def test_the_matching_settings_take_what_the_config_sets() -> None:
    assert resolve_matching_settings({"matching": {"verifier_calls": 5, "text_candidates": 0}}) == (
        MatchingSettings(verifier_calls=5, image_candidates=3, text_candidates=0)
    )


def test_every_invalid_matching_setting_is_reported_on_one_line() -> None:
    matching = {"verifier_calls": -1, "image_candidates": 2.5, "text_candidates": True}
    with pytest.raises(ValueError, match=r"^Invalid vlm configuration: ") as caught:
        resolve_matching_settings({"matching": matching})
    assert str(caught.value) == (
        "Invalid vlm configuration: "
        "vlm.matching.verifier_calls must be a whole number of 0 or more; "
        "vlm.matching.image_candidates must be a whole number of 0 or more; "
        "vlm.matching.text_candidates must be a whole number of 0 or more"
    )


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
