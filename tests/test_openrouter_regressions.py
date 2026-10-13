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


@pytest.mark.parametrize("content", ["[]", "null", '"screen"', "42", "true"])
def test_page_description_retries_non_object_json(
    content: str,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    chat_body: dict[str, object],
    png_screenshot: str,
    no_sleep: list[float],
) -> None:
    router_responses.extend(
        [
            action_response(chat_body, content),
            action_response(chat_body, '{"page_description":"home","state":{},"elements":[]}'),
        ]
    )
    assert vlm_utils.describe_page_and_state(router_client, png_screenshot) == ("home", {}, [])
    assert len(router_requests) == 2
    assert no_sleep == [2.0]


@pytest.mark.parametrize("recover", [False, True])
def test_page_description_http_and_parse_errors_share_one_budget(
    recover: bool,
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
            action_response(chat_body, "[]"),
            action_response(chat_body, '{"page_description":"home"}' if recover else "null"),
        ]
    )
    if recover:
        assert vlm_utils.describe_page_and_state(router_client, png_screenshot) == ("home", {}, [])
    else:
        with pytest.raises(TypeError, match=r"page description.*JSON object"):
            vlm_utils.describe_page_and_state(router_client, png_screenshot)
    assert len(router_requests) == 3
    assert no_sleep == [2.0, 4.0]


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


@pytest.mark.parametrize("status", [400, 401, 402, 403, 404, 422, 408, 409, 425, 429, 500, 503])
@pytest.mark.parametrize("operation", ["chat", "text", "image", "action", "description"])
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
        elif operation == "description":
            vlm_utils.describe_page_and_state(router_client, png_screenshot)
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
    transient = status in {408, 409, 425, 429} or status >= 500
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
    client, _ = make_client({"api_key": "test", "base_url": endpoint}, role="action")
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
    client, model = make_client(cfg, http_client=router_http, role="page_detail")
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
        cfg: dict[str, object] | None, *, default_model: str, role: str
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
            {"api_key": "test", "base_url": endpoint},
            default_model="google/gemini-embedding-2",
            role="embedding",
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
    router_requests: list[httpx.Request],
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

    for _ in range(2):
        with pytest.raises(ValueError, match=r"Text embedding.*Rebuild or re-explore"):
            compare()
        assert router_requests == []


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


@pytest.mark.parametrize("operation", ["chat", "text", "image", "action"])
@pytest.mark.parametrize("recover", [False, True])
def test_sdk_validation_uses_bounded_application_recovery(
    operation: str,
    recover: bool,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    chat_body: dict[str, object],
    embedding_body: dict[str, object],
    png_screenshot: str,
    no_sleep: list[float],
) -> None:
    from openrouter.errors import ResponseValidationError

    good = embedding_body if operation in {"text", "image"} else chat_body
    bad = {key: value for key, value in good.items() if key != "model"}
    router_responses.extend(httpx.Response(200, json=bad) for _ in range(1 if recover else 3))
    if recover:
        router_responses.append(
            httpx.Response(200, json=good)
            if operation != "action"
            else action_response(chat_body, '{"action":"wait","time":2}')
        )

    def request() -> object:
        if operation == "action":
            return vlm_utils.predict_next_action(router_client, png_screenshot, "wait")[0]
        if operation == "image":
            return embedding_cache.compute_embedding_with_retry(
                "test",
                png_screenshot,
                model="model",
                base_url="https://openrouter.ai/api/v1",
                app_name="demo",
                node_id="n1",
                client=router_client,
            )
        if operation == "text":
            return vlm_utils.get_embedding(router_client, "hello")
        return vlm_utils.send_chat(
            router_client, model="test", messages=[{"role": "user", "content": "hello"}]
        )

    if recover or operation == "action":
        result = request()
        if operation == "action":
            assert result == {"action": "wait", "time": 2 if recover else 1}
        elif operation in {"text", "image"}:
            assert result == [0.25, 0.75]
        else:
            assert result is not None
    else:
        with pytest.raises(ResponseValidationError) as caught:
            request()
        assert caught.value.status_code == 200
    assert len(router_requests) == (2 if recover else 3)
    assert no_sleep == ([2.0] if recover else [2.0, 4.0])


@pytest.mark.parametrize("last", ["success", "schema", "output", "http"])
def test_action_schema_http_and_output_share_total_budget(
    last: str,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    chat_body: dict[str, object],
    png_screenshot: str,
    no_sleep: list[float],
) -> None:
    bad = {key: value for key, value in chat_body.items() if key != "model"}
    http_error = httpx.Response(503, json={"error": {"code": 503, "message": "busy"}})
    router_responses.extend(
        [
            http_error,
            httpx.Response(200, json=bad),
            {
                "success": action_response(chat_body, '{"action":"click","coordinate":[500,300]}'),
                "schema": httpx.Response(200, json=bad),
                "output": action_response(chat_body, "unreadable"),
                "http": http_error,
            }[last],
        ]
    )
    if last == "http":
        with pytest.raises(OpenRouterError):
            vlm_utils.predict_next_action(router_client, png_screenshot, "tap")
    else:
        action, _ = vlm_utils.predict_next_action(router_client, png_screenshot, "tap")
        assert action == (
            {"action": "tap", "x": 540, "y": 576}
            if last == "success"
            else {"action": "wait", "time": 1}
        )
    assert len(router_requests) == 3
    assert no_sleep == [2.0, 4.0]


@pytest.mark.parametrize("endpoint", ["https://openrouter.ai/api/v1", "http://localhost:8080/v1"])
@pytest.mark.parametrize("fingerprint", ["omitted", None, "fp_test"])
def test_chat_fingerprint_compatibility_is_scoped(
    endpoint: str,
    fingerprint: str | None,
    router_http: httpx.Client,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    chat_body: dict[str, object],
    no_sleep: list[float],
) -> None:
    body = dict(chat_body)
    if fingerprint == "omitted":
        del body["system_fingerprint"]
    else:
        body["system_fingerprint"] = fingerprint
    client, model = make_client(
        {"api_key": "test", "base_url": endpoint}, role="page_detail", http_client=router_http
    )
    router_responses.append(httpx.Response(200, json=body))
    result = vlm_utils.send_chat(client, model=model, messages=[{"role": "user", "content": "hi"}])
    assert result.system_fingerprint == (None if fingerprint == "omitted" else fingerprint)
    assert len(router_requests) == 1
    assert no_sleep == []
    assert not router_http.is_closed


@pytest.mark.parametrize("limit", [None, 17, aitk_translator.V2_CHAT_MAX_TOKENS])
def test_chat_omits_unspecified_token_limit_at_strict_endpoint(
    limit: int | None, chat_body: dict[str, object], no_sleep: list[float]
) -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        payload = json.loads(request.content)
        if "max_tokens" in payload and payload["max_tokens"] is None:
            return httpx.Response(400, json={"error": {"code": 400, "message": "null forbidden"}})
        return httpx.Response(200, json=chat_body)

    with httpx.Client(transport=httpx.MockTransport(respond)) as borrowed:
        client, model = make_client({"api_key": "test"}, role="instruction", http_client=borrowed)
        vlm_utils.send_chat(
            client, model=model, messages=[{"role": "user", "content": "hi"}], max_tokens=limit
        )
    payload = json.loads(requests[0].content)
    if limit is None:
        assert "max_tokens" not in payload
    else:
        assert payload["max_tokens"] == limit
    assert len(requests) == 1
    assert no_sleep == []


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("recover", [False, True])
@pytest.mark.parametrize(
    "args",
    [
        {"action": "click", "coordinate": {"x": 500, "y": 300}},
        {"action": "click"},
        {"action": "click", "coordinate": []},
        {"action": "click", "coordinate": [500]},
        {"action": "long_press", "coordinate": []},
        {"action": "long_press", "coordinate": [True, 300]},
        {"action": "click", "coordinate": ["500", 300]},
        {"action": "click", "coordinate": [None, 300]},
        {"action": "click", "coordinate": [float("nan"), 300]},
        {"action": "long_press", "coordinate": [500, float("inf")]},
        {"action": "swipe", "coordinate": {"x": 500, "y": 300}, "coordinate2": [0, 0]},
        {"action": "swipe", "coordinate": [500, 300], "coordinate2": [500]},
        {"action": "swipe", "coordinate": [500, 300]},
        {"action": "swipe"},
        {"action": "swipe", "coordinate2": [500, 300]},
        {"action": "click", "coordinate": [10**400, 300]},
        {"action": "swipe", "coordinate": [], "coordinate2": [0, 0], "direction": "up"},
    ],
)
def test_malformed_coordinates_recover_or_wait(
    native: bool,
    recover: bool,
    args: dict[str, object],
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    chat_body: dict[str, object],
    png_screenshot: str,
    no_sleep: list[float],
) -> None:
    response = action_response(chat_body, json.dumps(args), native=native)
    if native:
        body = response.json()
        body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = json.dumps(args)
        response = httpx.Response(200, json=body)
    router_responses.extend(response for _ in range(1 if recover else 3))
    if recover:
        router_responses.append(
            action_response(chat_body, '{"action":"click","coordinate":[500,300]}')
        )
    action, _ = vlm_utils.predict_next_action(
        router_client, png_screenshot, "tap", screen_w=100, screen_h=200
    )
    assert action == (
        {"action": "tap", "x": 50, "y": 60} if recover else {"action": "wait", "time": 1}
    )
    assert len(router_requests) == (2 if recover else 3)
    assert no_sleep == ([2.0] if recover else [2.0, 4.0])


@pytest.mark.parametrize("role", ["embedding", "image_embedding"])
@pytest.mark.parametrize("entry", ["resolver", "factory", "offline", "runtime"])
def test_legacy_native_url_rejected_before_requests(
    role: str,
    entry: str,
    tmp_path: Path,
    router_requests: list[httpx.Request],
    router_http: httpx.Client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from android_app_graph.commands import embed

    secret = "dummy-google-secret-never-log"
    monkeypatch.setenv("GEMINI_API_KEY", secret)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-router-key")
    cfg = {
        "native_base_url": "https://generativelanguage.googleapis.com/v1beta",
        "api_key": "${GEMINI_API_KEY}",
        "model": "gemini-embedding-2",
    }

    def configure() -> None:
        if entry == "factory":
            make_client(cfg, role=role, http_client=router_http)
        elif entry == "resolver":
            resolve_image_embedding_settings({role: cfg})
        elif entry == "runtime":
            aitk_translator.UIKobeV2Translator(graph_dir=str(tmp_path), vlm_config={role: cfg})
        else:
            path = tmp_path / "legacy.yaml"
            path.write_text(
                f"vlm:\n  {role}:\n    native_base_url: https://generativelanguage.googleapis.com/v1beta\n    api_key: '${{GEMINI_API_KEY}}'\n    model: gemini-embedding-2\n"
            )
            embed.load_image_embedding_settings(
                path,
                graphs_override=None,
                api_key_override=None,
                model_override=None,
                base_url_override=None,
            )

    with pytest.raises(ValueError, match=rf"{role}\.native_base_url") as caught:
        configure()
    assert secret not in str(caught.value)
    assert "base_url" in str(caught.value)
    assert "OPENROUTER_API_KEY" in str(caught.value)
    assert router_requests == []


@pytest.mark.parametrize("exceptional", [False, True])
def test_owned_http_client_closed_on_sdk_scope_exit(exceptional: bool) -> None:
    client, _ = make_client({"api_key": "test"}, role="action")
    owned = client.sdk_configuration.client
    assert isinstance(owned, httpx.Client)
    try:
        with client:
            if exceptional:
                raise RuntimeError("scope failed")
    except RuntimeError:
        assert exceptional
    assert owned.is_closed


def test_borrowed_client_and_hooks_survive_sdk_scope(
    router_http: httpx.Client, router_responses: list[httpx.Response], chat_body: dict[str, object]
) -> None:
    events: list[str] = []
    router_http.event_hooks["response"].append(lambda _response: events.append("response"))
    client, model = make_client({"api_key": "test"}, role="page_detail", http_client=router_http)
    body = {key: value for key, value in chat_body.items() if key != "system_fingerprint"}
    router_responses.append(httpx.Response(200, json=body))
    with client:
        vlm_utils.send_chat(client, model=model, messages=[])
    assert not router_http.is_closed
    assert events == ["response"]


@pytest.mark.parametrize("interrupted", [False, True])
def test_screenshot_batch_reuses_and_closes_client_with_partial_progress(
    interrupted: bool,
    tmp_path: Path,
    owned_router_clients: list[httpx.Client],
    router_requests: list[httpx.Request],
    router_responses: list[httpx.Response],
    embedding_body: dict[str, object],
    no_sleep: list[float],
) -> None:
    from collections.abc import Iterator

    graph = tmp_path / "demo.json"
    cache: dict[str, list[float]] = {}
    router_responses.extend(
        [
            httpx.Response(503, json={"error": {"code": 503, "message": "busy"}}),
            httpx.Response(200, json=embedding_body),
            httpx.Response(200, json=embedding_body),
        ]
    )

    def candidates() -> Iterator[tuple[str, str]]:
        yield "n1", "shot-1"
        if interrupted:
            raise KeyboardInterrupt
        yield "n2", "shot-2"

    def compute() -> None:
        embedding_cache.compute_missing_image_embeddings(
            graph,
            cache,
            candidates(),
            api_key="test",
            model="model",
            base_url="http://localhost:8080/v1",
            app_name="demo",
        )

    if interrupted:
        with pytest.raises(KeyboardInterrupt):
            compute()
    else:
        compute()
    assert len(router_requests) == (2 if interrupted else 3)
    assert no_sleep == [2.0]
    assert len(owned_router_clients) == 1
    assert all(client.is_closed for client in owned_router_clients)
    assert (
        embedding_cache.load_image_embeddings(graph, model="model", node_ids={"n1", "n2"}) == cache
    )
    assert set(cache) == ({"n1"} if interrupted else {"n1", "n2"})


@pytest.mark.parametrize("exceptional", [False, True])
def test_v2_invoked_callback_closes_clients_and_reuses_image_client(
    exceptional: bool,
    tmp_path: Path,
    owned_router_clients: list[httpx.Client],
    router_responses: list[httpx.Response],
    embedding_body: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    built = aitk_translator.UIKobeV2Translator(graph_dir=str(tmp_path))
    assert all(client.is_closed for client in owned_router_clients)
    initial_count = len(owned_router_clients)
    router_responses.extend(httpx.Response(200, json=embedding_body) for _ in range(4))

    def step(_task: str, _state: dict[str, object], _history: dict[str, object]) -> str:
        for screenshot in ("one", "two"):
            assert built._compute_runtime_image_embedding_with_retry(
                screenshot, "demo", "current"
            ) == [0.25, 0.75]
        if exceptional:
            raise RuntimeError("callback failed")
        return "ok"

    monkeypatch.setattr(built, "_step", step)
    for _ in range(2):
        if exceptional:
            with pytest.raises(RuntimeError, match="callback failed"):
                built.to_agent("task", {}, {})
        else:
            assert built.to_agent("task", {}, {}) == "ok"
        assert all(client.is_closed for client in owned_router_clients)
    assert len(owned_router_clients) == initial_count + 8


def test_v2_partial_initialization_closes_created_clients(
    tmp_path: Path, owned_router_clients: list[httpx.Client]
) -> None:
    with pytest.raises(ValueError, match="page_detail"):
        aitk_translator.UIKobeV2Translator(
            graph_dir=str(tmp_path), vlm_config={"action": {"api_key": "test"}}
        )
    assert len(owned_router_clients) == 1
    assert owned_router_clients[0].is_closed


@pytest.mark.parametrize("exceptional", [False, True])
def test_audit_cli_closes_owned_clients_on_exit(
    exceptional: bool,
    tmp_path: Path,
    owned_router_clients: list[httpx.Client],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from android_app_graph.commands import audit

    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    graph_dir = tmp_path / "graphs"
    app_dir = graph_dir / "demo"
    app_dir.mkdir(parents=True)
    (app_dir / "demo.json").write_text('{"nodes":[],"edges":[]}')
    config = tmp_path / "config.yaml"
    config.write_text(
        f"experiment:\n  graph_dir: {graph_dir}\napps:\n  - name: demo\n    package_name: com.demo\n"
    )

    def audit_graph(_graph: GraphManager, _app: str, _path: Path) -> dict[str, object]:
        if exceptional:
            raise RuntimeError("audit failed")
        return {"issues": []}

    monkeypatch.setattr(audit, "run_audit", audit_graph)
    monkeypatch.setattr(audit, "normalize_and_save_audited_graph", lambda *_args: 0)
    if exceptional:
        with pytest.raises(RuntimeError, match="audit failed"):
            audit.main(["-c", str(config)])
    else:
        assert audit.main(["-c", str(config)]) == 0
    assert len(owned_router_clients) == 2
    assert all(client.is_closed for client in owned_router_clients)


@pytest.mark.parametrize("exceptional", [False, True])
def test_exploration_lifecycle_closes_owned_clients(
    exceptional: bool,
    tmp_path: Path,
    owned_router_clients: list[httpx.Client],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import logging

    from android_app_graph.kobe import Kobe

    class IdleDevice:
        w: int = 100
        h: int = 200
        config: dict[str, object]

        def __init__(self) -> None:
            self.config = {}

        def get_state(self) -> dict[str, object]:
            raise RuntimeError("device failed")

        def exe_action(self, action: dict[str, object]) -> None:
            raise AssertionError(f"No device actions expected: {action}")

    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    built = Kobe(IdleDevice(), "demo", "com.demo", logging.getLogger("test"), graph_dir=tmp_path)
    assert len(owned_router_clients) == 4
    assert all(client.is_closed for client in owned_router_clients)
    if exceptional:
        with pytest.raises(RuntimeError, match="device failed"):
            built.explore(max_steps=1)
    else:
        built.explore(max_steps=0)
    assert len(owned_router_clients) == 8
    assert all(client.is_closed for client in owned_router_clients)


def test_omitted_fingerprint_does_not_default_another_required_field(
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    chat_body: dict[str, object],
    no_sleep: list[float],
) -> None:
    body = {
        key: value for key, value in chat_body.items() if key not in {"system_fingerprint", "model"}
    }
    router_responses.extend(
        [
            httpx.Response(200, json=body),
            httpx.Response(
                200,
                json={
                    key: value for key, value in chat_body.items() if key != "system_fingerprint"
                },
            ),
        ]
    )
    reply = vlm_utils.send_chat(router_client, model="test", messages=[])
    assert vlm_utils.message_text(reply) == "test reply"
    assert len(router_requests) == 2
    assert no_sleep == [2.0]


@pytest.mark.parametrize("native", [False, True])
def test_omitted_fingerprint_preserves_multipart_and_native_actions(
    native: bool,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    chat_body: dict[str, object],
    png_screenshot: str,
) -> None:
    body = action_response(
        chat_body,
        'Thought: Search\nAction: Tap\n<tool_call>{"action":"click","coordinate":[500,500]}</tool_call>',
        native=native,
    ).json()
    del body["system_fingerprint"]
    body["choices"][0]["message"]["content"] = [
        {"type": "text", "text": body["choices"][0]["message"]["content"]}
    ]
    router_responses.append(httpx.Response(200, json=body))
    action, history = vlm_utils.predict_next_action(router_client, png_screenshot, "tap")
    assert action == {"action": "tap", "x": 540, "y": 960}
    assert history == "Search | Tap"
    assert len(router_requests) == 1


@pytest.mark.parametrize("borrowed", [False, True])
def test_factory_initialization_failure_keeps_correct_owner(
    borrowed: bool,
    owned_router_clients: list[httpx.Client],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from android_app_graph import utils

    supplied = httpx.Client() if borrowed else None

    def fail_sdk(**_kwargs: object) -> OpenRouter:
        raise RuntimeError("SDK construction failed")

    monkeypatch.setattr(utils, "ConfiguredOpenRouter", fail_sdk)
    with pytest.raises(RuntimeError, match="SDK construction failed"):
        make_client({"api_key": "test"}, role="action", http_client=supplied)
    assert len(owned_router_clients) == 1
    assert owned_router_clients[0].is_closed is not borrowed
    if supplied is not None:
        supplied.close()


@pytest.mark.parametrize(
    "role", ["page_detail", "instruction", "action", "embedding", "image_embedding"]
)
def test_missing_key_names_explicit_configuration_role(role: str) -> None:
    with pytest.raises(ValueError, match=rf"{role}\.api_key.*OPENROUTER_API_KEY"):
        make_client({}, role=role, default_model="google/gemini-embedding-2")


def test_image_batches_keep_endpoint_credentials_and_models_separate(
    tmp_path: Path,
    owned_router_clients: list[httpx.Client],
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    embedding_body: dict[str, object],
) -> None:
    router_responses.extend(httpx.Response(200, json=embedding_body) for _ in range(4))
    for index in range(2):
        result = embedding_cache.compute_missing_image_embeddings(
            tmp_path / f"graph-{index}.json",
            {},
            [("n1", "one"), ("n2", "two")],
            api_key=f"key-{index}",
            model=f"model-{index}",
            base_url=f"https://endpoint-{index}.example/v1",
            app_name="demo",
        )
        assert result.computed == 2
    assert len(owned_router_clients) == 2
    assert all(client.is_closed for client in owned_router_clients)
    for index, request in enumerate(router_requests):
        setting = index // 2
        assert request.headers["authorization"] == f"Bearer key-{setting}"
        assert str(request.url) == f"https://endpoint-{setting}.example/v1/embeddings"
        assert json.loads(request.content)["model"] == f"model-{setting}"


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ({"action": "click", "coordinate": [500.5, 300.5]}, {"action": "tap", "x": 50, "y": 60}),
        (
            {"action": "long_press", "coordinate": [500, 300]},
            {"action": "long_press", "x": 50, "y": 60, "duration": 1000},
        ),
        (
            {"action": "long_press", "coordinate": [500, 300], "time": 250, "duration": 750},
            {"action": "long_press", "x": 50, "y": 60, "duration": 250},
        ),
        (
            {
                "action": "swipe",
                "coordinate": [500, 300],
                "coordinate2": [250, 800],
                "duration": 750,
            },
            {"action": "swipe", "x1": 50, "y1": 60, "x2": 25, "y2": 160, "duration": 750},
        ),
        (
            {"action": "swipe", "direction": "up"},
            {"action": "swipe", "x1": 50, "y1": 150, "x2": 50, "y2": 50, "duration": 1000},
        ),
        (
            {"action": "swipe", "direction": "down"},
            {"action": "swipe", "x1": 50, "y1": 50, "x2": 50, "y2": 150, "duration": 1000},
        ),
        (
            {"action": "swipe", "direction": "left"},
            {"action": "swipe", "x1": 80, "y1": 100, "x2": 20, "y2": 100, "duration": 1000},
        ),
        (
            {"action": "swipe", "direction": "right"},
            {"action": "swipe", "x1": 20, "y1": 100, "x2": 80, "y2": 100, "duration": 1000},
        ),
        ({"action": "end"}, {"action": "end"}),
        ({"action": "terminate"}, {"action": "end"}),
    ],
)
def test_valid_coordinate_actions_and_intentional_end_remain_single_requests(
    native: bool,
    args: dict[str, object],
    expected: dict[str, object],
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    chat_body: dict[str, object],
    png_screenshot: str,
    no_sleep: list[float],
) -> None:
    body = action_response(
        chat_body, f"<tool_call>{json.dumps(args)}</tool_call>", native=native
    ).json()
    if native:
        body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = json.dumps(args)
    router_responses.append(httpx.Response(200, json=body))
    assert (
        vlm_utils.predict_next_action(
            router_client, png_screenshot, "act", screen_w=100, screen_h=200
        )[0]
        == expected
    )
    assert len(router_requests) == 1
    assert no_sleep == []


@pytest.mark.parametrize("status", [400, 401, 402, 403, 404, 422])
def test_malformed_http_error_never_returns_action_fallback(
    status: int,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    png_screenshot: str,
    no_sleep: list[float],
) -> None:
    router_responses.append(httpx.Response(status, json={"error": {"message": "bad"}}))
    with pytest.raises(OpenRouterError) as caught:
        vlm_utils.predict_next_action(router_client, png_screenshot, "tap")
    assert caught.value.status_code == status
    assert len(router_requests) == 1
    assert no_sleep == []
