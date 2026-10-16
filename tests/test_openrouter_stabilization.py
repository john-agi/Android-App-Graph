"""Exercise SDK recovery through public entry points and nested resource scopes."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from android_app_graph.adapters import aitk_translator
from android_app_graph.commands import embed


@pytest.mark.parametrize("entry", ["runtime", "cli"])
@pytest.mark.parametrize("timeout_key", ["request_timeout", "timeout"])
@pytest.mark.parametrize("inherit", [False, True])
@pytest.mark.usefixtures("model_environment")
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


@pytest.mark.parametrize("timeout", ["nan", "inf", "0", "-1", "invalid"])
@pytest.mark.usefixtures("model_environment")
def test_invalid_timeout_is_a_configuration_error(
    timeout: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
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
@pytest.mark.usefixtures("model_environment")
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
    chat_response: Callable[..., httpx.Response],
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
                chat_response(content="nothing"),
                chat_response(content="Tap Search"),
            ]
        )
        if turn == 0:
            router_responses.extend(
                httpx.Response(
                    200, json={key: val for key, val in chat_body.items() if key != "id"}
                )
                if schema_failure
                else chat_response(content='{"action":"system_button","button":"Power"}')
                for _ in range(3)
            )
        else:
            router_responses.append(
                chat_response(content='{"action":"click","coordinate":[500,500]}')
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
