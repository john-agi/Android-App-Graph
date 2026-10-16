"""Exercise SDK recovery through public entry points and nested resource scopes."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from android_app_graph.commands import embed


@pytest.mark.parametrize("timeout_key", ["request_timeout", "timeout"])
@pytest.mark.parametrize("inherit", [False, True])
@pytest.mark.usefixtures("model_environment")
def test_image_config_timeout_reaches_actual_request(
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
