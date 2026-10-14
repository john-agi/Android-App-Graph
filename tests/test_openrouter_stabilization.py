"""Exercise SDK recovery through public entry points and nested resource scopes."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import httpx
import pytest

from android_app_graph.adapters import aitk_translator
from android_app_graph.commands import embed
from android_app_graph.kobe import Kobe
from android_app_graph.utils import make_client, vlm_utils


def description_response(body: dict[str, object], text: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            **body,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": text},
                }
            ],
        },
    )


@pytest.fixture
def kobe(
    tmp_path: Path,
    png_screenshot: str,
    owned_router_clients: list[httpx.Client],
    monkeypatch: pytest.MonkeyPatch,
) -> Kobe:
    class Device:
        w: int = 100
        h: int = 200

        def __init__(self) -> None:
            self.config: dict[str, object] = {}

        def get_state(self) -> dict[str, object]:
            return {"activity": "com.demo.Home", "screenshot": png_screenshot}

        def exe_action(self, action: dict[str, object]) -> None:
            raise AssertionError(f"No device action expected: {action}")

    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    built = Kobe(Device(), "demo", "com.demo", logging.getLogger("test"), graph_dir=tmp_path)
    assert all(client.is_closed for client in owned_router_clients)
    return built


@pytest.mark.parametrize("entry", ["runtime", "cli"])
@pytest.mark.parametrize("timeout_key", ["request_timeout", "timeout"])
@pytest.mark.parametrize("inherit", [False, True])
def test_image_config_timeout_reaches_actual_request(
    entry: str,
    timeout_key: str,
    inherit: bool,
    tmp_path: Path,
    owned_router_clients: list[httpx.Client],
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    embedding_body: dict[str, object],
    png_screenshot: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import base64

    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    config: dict[str, object] = {
        "embedding": {timeout_key: 7.5},
        "image_embedding": {} if inherit else {timeout_key: 2.25},
    }
    router_responses.append(httpx.Response(200, json=embedding_body))
    if entry == "runtime":
        runtime = aitk_translator.UIKobeV2Translator(graph_dir=str(tmp_path), vlm_config=config)
        assert runtime._compute_runtime_image_embedding_with_retry(
            png_screenshot, "demo", "home"
        ) == [0.25, 0.75]
    else:
        graph = tmp_path / "demo" / "demo.json"
        graph.parent.mkdir()
        graph.write_text(json.dumps({"nodes": [{"id": "home"}], "edges": []}))
        screenshots = graph.parent / "demo_screenshots"
        screenshots.mkdir()
        (screenshots / "home.png").write_bytes(base64.b64decode(png_screenshot))
        path = tmp_path / "config.yaml"
        path.write_text(json.dumps({"vlm": config, "experiment": {"graph_dir": str(tmp_path)}}))
        assert embed.main(["--config", str(path), "--app", "demo"]) == 0
    assert len(router_requests) == 1
    assert router_requests[0].extensions["timeout"] == dict.fromkeys(
        ("connect", "read", "write", "pool"), 7.5 if inherit else 2.25
    )
    assert all(client.is_closed for client in owned_router_clients)


def test_complete_chat_reply_does_not_decode_large_request(
    owned_router_clients: list[httpx.Client],
    router_responses: list[httpx.Response],
    chat_body: dict[str, object],
) -> None:
    from unittest.mock import patch

    client, model = make_client({"api_key": "test"}, role="action")
    router_responses.append(httpx.Response(200, json=chat_body))
    with patch("android_app_graph.utils.json.loads", wraps=json.loads) as loads:
        vlm_utils.send_chat(
            client, model=model, messages=[{"role": "user", "content": "A" * 4_000_000}]
        )
    assert all(len(call.args[0]) < 4_000_000 for call in loads.call_args_list)
    assert all(client.is_closed for client in owned_router_clients)


@pytest.mark.parametrize("timeout", ["nan", "inf", "0", "-1", "invalid"])
def test_invalid_timeout_is_a_configuration_error(
    timeout: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(ValueError, match="request_timeout must be a positive finite number"):
        make_client({"api_key": "test", "request_timeout": timeout}, role="action")
    config = tmp_path / "invalid.yaml"
    config.write_text(json.dumps({"vlm": {"image_embedding": {"timeout": timeout}}}))
    with pytest.raises(SystemExit) as caught:
        embed.main(["--config", str(config)])
    assert caught.value.code == 2
    assert (
        "image_embedding.request_timeout must be a positive finite number"
        in capsys.readouterr().err
    )


@pytest.mark.parametrize("schema_failure", [False, True])
def test_v2_turn_distinguishes_bad_actions_from_broken_endpoint_and_can_resume(
    schema_failure: bool,
    tmp_path: Path,
    owned_router_clients: list[httpx.Client],
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    chat_body: dict[str, object],
    png_screenshot: str,
    no_sleep: list[float],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openrouter.errors import ResponseValidationError

    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setattr(aitk_translator.device, "soft_keyboard_hint", lambda: "")
    runtime = aitk_translator.UIKobeV2Translator(graph_dir=str(tmp_path))
    runtime._app_opened = True
    state: dict[str, object] = {"screenshot": png_screenshot, "activity": "com.demo.Home"}
    history: dict[str, object] = {"actions": ["started"]}
    for turn in range(2):
        router_responses.extend(
            [
                description_response(chat_body, "nothing"),
                description_response(chat_body, "Tap Search"),
            ]
        )
        if turn == 0:
            router_responses.extend(
                httpx.Response(
                    200, json={key: val for key, val in chat_body.items() if key != "id"}
                )
                if schema_failure
                else description_response(chat_body, '{"action":"system_button","button":"Power"}')
                for _ in range(3)
            )
        else:
            router_responses.append(
                description_response(chat_body, '{"action":"click","coordinate":[500,500]}')
            )
        if turn == 0 and schema_failure:
            with pytest.raises(ResponseValidationError):
                runtime.to_agent("Find search", state, history)
            assert runtime._memory.actions == []
        else:
            result = json.loads(runtime.to_agent("Find search", state, history))
            assert result["aitk_action"] == (
                {"action": "wait", "time": 1}
                if turn == 0
                else {"action": "tap", "x": 540, "y": 960}
            )
        assert all(client.is_closed for client in owned_router_clients)
    assert len(router_requests) == 8
    assert len(owned_router_clients) == 12
    assert no_sleep == [2.0, 4.0]


def test_public_graph_and_navigation_recover_after_request_failure(
    kobe: Kobe,
    png_screenshot: str,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    owned_router_clients: list[httpx.Client],
    chat_body: dict[str, object],
    embedding_body: dict[str, object],
    no_sleep: list[float],
) -> None:
    from openrouter.errors import OpenRouterError

    router_responses.append(httpx.Response(401, json={"error": {"message": "bad key"}}))
    with pytest.raises(OpenRouterError):
        kobe.graph.identify_state("com.demo.Home", png_screenshot)
    assert len(kobe.graph.graph) == 0
    assert all(client.is_closed for client in owned_router_clients)
    for _ in range(2):
        router_responses.extend(
            [
                description_response(chat_body, '{"page_description":"home"}'),
                httpx.Response(200, json=embedding_body),
            ]
        )
    node = kobe.graph.identify_state("com.demo.Home", png_screenshot)
    kobe.graph._last_identify_cache = None
    kobe.graph.graph.nodes[node].pop("reference_screenshot", None)
    assert kobe.navigate_to_node(node) == node
    assert len(router_requests) == 5
    assert no_sleep == []
    assert all(client.is_closed for client in owned_router_clients)


@pytest.mark.parametrize("fail_inner", [False, True])
def test_nested_scopes_keep_outer_connection_alive(
    fail_inner: bool,
    owned_router_clients: list[httpx.Client],
    router_responses: list[httpx.Response],
    chat_body: dict[str, object],
) -> None:
    client, model = make_client({"api_key": "test"}, role="action")
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


@pytest.mark.parametrize("malformed", [False, True])
def test_exploration_wait_recovery_respects_step_limit(
    malformed: bool,
    kobe: Kobe,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    owned_router_clients: list[httpx.Client],
    chat_body: dict[str, object],
    embedding_body: dict[str, object],
    no_sleep: list[float],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("android_app_graph.kobe.soft_keyboard_hint", lambda: "")
    router_responses.extend(
        [
            description_response(chat_body, '{"page_description":"home"}'),
            httpx.Response(200, json=embedding_body),
        ]
    )
    for _ in range(2):
        router_responses.append(description_response(chat_body, "Tap Search"))
        router_responses.extend(
            description_response(
                chat_body, '{"action":"invented"}' if malformed else '{"action":"wait"}'
            )
            for _ in range(3 if malformed else 1)
        )
    kobe.explore(max_steps=2)
    assert kobe.graph.total_steps_completed == 2
    assert json.loads(kobe.graph_path.read_text())["total_steps_completed"] == 2
    assert len(router_requests) == (10 if malformed else 6)
    assert len(kobe.graph.graph.edges) == 0
    assert no_sleep == ([2.0, 4.0, 1, 2.0, 4.0, 1] if malformed else [1, 1])
    assert all(client.is_closed for client in owned_router_clients)


@pytest.mark.parametrize("recover", [False, True])
@pytest.mark.parametrize(
    "malformed",
    [
        '{"page_description":',
        "not JSON",
        "{}",
        '{"page_description":""}',
        '{"page_description":"home","state":[]}',
        '{"page_description":"home","elements":[null]}',
    ],
)
def test_malformed_description_cannot_become_a_graph_node(
    recover: bool,
    malformed: str,
    kobe: Kobe,
    png_screenshot: str,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    chat_body: dict[str, object],
    embedding_body: dict[str, object],
    no_sleep: list[float],
    owned_router_clients: list[httpx.Client],
) -> None:
    router_responses.extend(
        description_response(chat_body, malformed) for _ in range(1 if recover else 3)
    )
    if recover:
        router_responses.extend(
            [
                description_response(chat_body, '{"page_description":"home"}'),
                httpx.Response(200, json=embedding_body),
            ]
        )
        node = kobe.graph.identify_state("com.demo.Home", png_screenshot)
        assert kobe.graph.graph.nodes[node]["page_description"] == "home"
        assert len(kobe.graph.graph) == 1
        assert no_sleep == [2.0]
    else:
        with pytest.raises((ValueError, TypeError)):
            kobe.graph.identify_state("com.demo.Home", png_screenshot)
        assert len(kobe.graph.graph) == 0
        assert kobe.graph._last_identify_cache is None
        assert no_sleep == [2.0, 4.0]
    assert len(router_requests) == 3
    assert all(client.is_closed for client in owned_router_clients)
