"""Offline regressions for the OpenRouter migration's recovery and configuration."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from openrouter import OpenRouter
from openrouter.errors import OpenRouterError

from android_app_graph import embedding_cache
from android_app_graph.adapters import aitk_translator
from android_app_graph.embedding_cache import resolve_image_embedding_settings
from android_app_graph.utils import make_client, vlm_utils
from android_app_graph.utils.graph_manager import GraphManager


def action_response(
    body: dict[str, object], content: str, *, native: bool = False
) -> httpx.Response:
    message: dict[str, object] = {"role": "assistant", "content": content}
    if native:
        message["tool_calls"] = [
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "mobile_action",
                    "arguments": '{"action":"click","coordinate":[500,500]}',
                },
            }
        ]
    return httpx.Response(
        200,
        json={
            **body,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls" if native else "stop",
                    "message": message,
                }
            ],
        },
    )


@pytest.mark.parametrize(
    "first", ['{"action":"click","coordinate":["500","500"]}', "not JSON", "[]"]
)
def test_malformed_action_requests_another_response(
    first: str,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    chat_body: dict[str, object],
    png_screenshot: str,
    no_sleep: list[float],
) -> None:
    router_responses.extend(
        [
            action_response(chat_body, first),
            action_response(chat_body, '{"action":"click","coordinate":[500,500]}'),
        ]
    )
    action, _ = vlm_utils.predict_next_action(
        router_client, png_screenshot, "tap", screen_w=100, screen_h=200
    )
    assert action == {"action": "tap", "x": 50, "y": 100}
    assert len(router_requests) == 2
    assert no_sleep == [2.0]


def test_action_http_and_parse_failures_share_one_budget(
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    chat_body: dict[str, object],
    png_screenshot: str,
    no_sleep: list[float],
) -> None:
    router_responses.extend(
        [
            httpx.Response(429, json={"error": {"code": 429, "message": "busy"}}),
            action_response(chat_body, '{"action":"click","coordinate":["500","500"]}'),
            action_response(chat_body, '{"action":"wait"}'),
        ]
    )
    assert vlm_utils.predict_next_action(router_client, png_screenshot, "wait")[0] == {
        "action": "wait",
        "time": 3,
    }
    assert len(router_requests) == 3
    assert no_sleep == [2.0, 4.0]


@pytest.mark.parametrize("status", [400, 401, 402, 403, 404, 422, 408, 429, 500, 503])
@pytest.mark.parametrize("operation", ["chat", "text", "image", "action"])
def test_http_status_controls_attempts_and_backoff(
    status: int,
    operation: str,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    png_screenshot: str,
    no_sleep: list[float],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router_responses.extend(
        httpx.Response(status, json={"error": {"code": status, "message": "failed"}})
        for _ in range(3)
    )

    def request() -> None:
        if operation == "chat":
            vlm_utils.send_chat(
                router_client, model="test", messages=[{"role": "user", "content": "hi"}]
            )
        elif operation == "text":
            vlm_utils.get_embedding(router_client, "hi")
        elif operation == "action":
            vlm_utils.predict_next_action(router_client, png_screenshot, "tap")
        else:
            monkeypatch.setattr(
                embedding_cache,
                "get_openrouter_image_embedding",
                lambda *_args, **_kwargs: vlm_utils.get_image_embedding(
                    router_client, png_screenshot
                ),
            )
            embedding_cache.compute_embedding_with_retry(
                "test",
                png_screenshot,
                model="test",
                base_url="https://openrouter.ai/api/v1",
                app_name="demo",
                node_id="home",
            )

    with pytest.raises(OpenRouterError):
        request()
    transient = status == 408 or status == 429 or status >= 500
    assert len(router_requests) == (3 if transient else 1)
    assert no_sleep == ([2.0, 4.0] if transient else [])


@pytest.mark.parametrize(
    ("endpoint", "trust_env"),
    [
        ("https://openrouter.ai/api/v1", True),
        ("https://router.example/v1", True),
        ("http://localhost:8000/v1", False),
        ("http://127.0.0.1:8000/v1", False),
        ("http://[::1]:8000/v1", False),
    ],
)
def test_v2_endpoint_proxy_policy(
    endpoint: str, trust_env: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:8080")
    client, _ = aitk_translator._make_no_proxy_client({"api_key": "test", "base_url": endpoint})
    assert isinstance(client.sdk_configuration.client, httpx.Client)
    assert client.sdk_configuration.client.trust_env is trust_env
    client.sdk_configuration.client.close()


@pytest.mark.parametrize("native", [False, True])
def test_action_text_observations_survive_native_calls(
    native: bool,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    chat_body: dict[str, object],
    png_screenshot: str,
) -> None:
    text = (
        'Thought: Search is visible\nAction: Tap Search\n<tool_call>{"action":"wait"}</tool_call>'
    )
    router_responses.append(action_response(chat_body, text, native=native))
    action, history = vlm_utils.predict_next_action(
        router_client, png_screenshot, "tap", screen_w=100, screen_h=200
    )
    assert history == "Search is visible | Tap Search"
    assert action == (
        {"action": "tap", "x": 50, "y": 100} if native else {"action": "wait", "time": 3}
    )


@pytest.mark.parametrize("temperature", [None, 0.0, 0.7])
def test_optional_role_temperature_reaches_chat(
    temperature: float | None,
    router_http: httpx.Client,
    router_requests: list[httpx.Request],
    router_responses: list[httpx.Response],
    chat_body: dict[str, object],
) -> None:
    cfg: dict[str, object] = {"api_key": "test"}
    if temperature is not None:
        cfg["temperature"] = temperature
    client, model = make_client(cfg, http_client=router_http)
    router_responses.append(httpx.Response(200, json=chat_body))
    vlm_utils.send_chat(client, model=model, messages=[{"role": "user", "content": "hi"}])
    body = json.loads(router_requests[-1].content)
    if temperature is None:
        assert "temperature" not in body
    else:
        assert body["temperature"] == temperature


def test_temperatures_are_independent_across_roles(
    tmp_path: Path,
    router_http: httpx.Client,
    router_requests: list[httpx.Request],
    router_responses: list[httpx.Response],
    chat_body: dict[str, object],
    png_screenshot: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def factory(
        cfg: dict[str, object] | None, *, default_model: str, role: str = "chat"
    ) -> tuple[OpenRouter, str]:
        return make_client(cfg, default_model=default_model, role=role, http_client=router_http)

    monkeypatch.setattr(aitk_translator, "make_client", factory)
    translator = aitk_translator.UIKobeV2Translator(
        graph_dir=str(tmp_path),
        vlm_config={
            "action": {"api_key": "test", "temperature": 0.0},
            "page_detail": {"api_key": "test", "temperature": 0.4},
            "instruction": {"api_key": "test", "temperature": 0.8},
        },
    )
    router_responses.extend(
        [
            action_response(chat_body, '{"action":"wait"}'),
            action_response(chat_body, '{"page_description":"home","state":{},"elements":[]}'),
            httpx.Response(200, json=chat_body),
        ]
    )
    vlm_utils.predict_next_action(
        translator.model_client, png_screenshot, "wait", model=translator.model_name
    )
    vlm_utils.describe_page_and_state(
        translator.desc_client, png_screenshot, model=translator.desc_model
    )
    translator._ask_with_screenshot("plan", None, str, "test")
    assert [json.loads(request.content)["temperature"] for request in router_requests] == [
        0.0,
        0.4,
        0.8,
    ]


def test_malformed_action_exhausts_budget_and_falls_back(
    router_client: OpenRouter,
    router_requests: list[httpx.Request],
    router_responses: list[httpx.Response],
    chat_body: dict[str, object],
    png_screenshot: str,
    no_sleep: list[float],
) -> None:
    router_responses.extend(
        action_response(chat_body, '{"action":"click","coordinate":["bad",500]}') for _ in range(3)
    )
    assert vlm_utils.predict_next_action(router_client, png_screenshot, "tap") == (
        {"action": "wait", "time": 1},
        "Failed to parse action response -> wait",
    )
    assert len(router_requests) == 3
    assert no_sleep == [2.0, 4.0]


def test_instruction_credentials_error_names_role(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"instruction.*OPENROUTER_API_KEY"):
        aitk_translator.UIKobeV2Translator(
            graph_dir=str(tmp_path),
            vlm_config={
                "action": {"api_key": "test"},
                "page_detail": {"api_key": "test"},
            },
        )


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://generativelanguage.googleapis.com/v1beta",
        "https://us-central1-aiplatform.googleapis.com/v1",
    ],
)
def test_native_google_embedding_config_rejected_before_requests(endpoint: str) -> None:
    with pytest.raises(ValueError, match=r"embedding.*OpenRouter.*OPENROUTER_API_KEY"):
        make_client(
            {"api_key": "test", "base_url": endpoint}, default_model="google/gemini-embedding-2"
        )
    with pytest.raises(ValueError, match=r"image_embedding.*OpenRouter"):
        resolve_image_embedding_settings({"image_embedding": {"base_url": endpoint}})


@pytest.mark.parametrize("storage", ["inline", "sidecar"])
@pytest.mark.parametrize("model", [None, "provider/old-same-dimensions"])
@pytest.mark.parametrize("operation", ["find", "identify"])
def test_text_embedding_unknown_or_mismatched_provenance_blocks_comparison(
    storage: str,
    model: str | None,
    operation: str,
    tmp_path: Path,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    chat_body: dict[str, object],
    embedding_body: dict[str, object],
    png_screenshot: str,
) -> None:
    path = tmp_path / "demo.json"
    node: dict[str, object] = {
        "id": "home",
        "activity": "com.demo.Home",
        "page_description": "home",
    }
    if storage == "inline":
        node["description_embedding"] = [0.25, 0.75]
    if model is not None:
        node["description_embedding_model"] = model
    path.write_text(json.dumps({"nodes": [node], "edges": []}), encoding="utf-8")
    if storage == "sidecar":
        path.with_suffix(".emb.json").write_text(
            json.dumps(
                {
                    "home": [0.25, 0.75]
                    if model is None
                    else {"model": model, "embedding": [0.25, 0.75]}
                }
            ),
            encoding="utf-8",
        )
    manager = GraphManager(page_detail_client=router_client, embedding_client=router_client)
    manager.load_graph(path)
    router_responses.extend(
        [
            action_response(chat_body, '{"page_description":"home","state":{},"elements":[]}'),
            httpx.Response(200, json=embedding_body),
        ]
        if operation == "identify"
        else [httpx.Response(200, json=embedding_body)]
    )

    def compare() -> None:
        if operation == "find":
            manager.find_node_by_description("home")
        else:
            manager.identify_state("com.demo.Home", png_screenshot)

    with pytest.raises(ValueError, match=r"Text embedding.*Rebuild or re-explore"):
        compare()


def test_text_embedding_model_survives_save_load(
    tmp_path: Path,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    embedding_body: dict[str, object],
) -> None:
    manager = GraphManager(embedding_client=router_client)
    manager.graph.add_node(
        "home",
        description_embedding=[0.25, 0.75],
        description_embedding_model=manager.embedding_model,
    )
    path = tmp_path / "demo.json"
    manager.save_graph(path)
    saved = json.loads(path.with_suffix(".emb.json").read_text(encoding="utf-8"))
    assert saved["home"]["model"] == "google/gemini-embedding-2"
    manager.load_graph(path)
    router_responses.append(httpx.Response(200, json=embedding_body))
    assert manager.find_node_by_description("home") == [("home", pytest.approx(1.0))]
