"""Compatibility and request-boundary regressions for the OpenRouter migration."""

from __future__ import annotations

import json

import httpx
import pytest
from openrouter import OpenRouter

from android_app_graph.embedding_cache import resolve_image_embedding_settings
from android_app_graph.utils import DEFAULT_BASE_URL, make_client, vlm_utils
from android_app_graph.utils.graph_manager import GraphManager


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("direction", ["up", "down", "left", "right"])
@pytest.mark.parametrize(
    "coordinates",
    [
        {"coordinate": None},
        {"coordinate": None, "coordinate2": None},
        {"coordinate": [500, 300]},
        {"coordinate2": [500, 300]},
        {"coordinate": [500, 300], "coordinate2": None},
    ],
)
def test_directional_swipes_accept_optional_coordinates(
    native: bool,
    direction: str,
    coordinates: dict[str, object],
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    chat_body: dict[str, object],
    png_screenshot: str,
    no_sleep: list[float],
) -> None:
    args = {"action": "swipe", "direction": direction, **coordinates}
    message: dict[str, object] = {"role": "assistant", "content": json.dumps(args)}
    if native:
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "swipe",
                    "type": "function",
                    "function": {"name": "mobile_use", "arguments": json.dumps(args)},
                }
            ],
        }
    body = {
        **chat_body,
        "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
    }
    router_responses.extend(httpx.Response(200, json=body) for _ in range(3))
    action, _ = vlm_utils.predict_next_action(
        router_client,
        png_screenshot,
        "swipe",
        screen_w=100,
        screen_h=200,
        model="qwen/qwen3.8-flash",
    )
    endpoints = {
        "up": (50, 150, 50, 50),
        "down": (50, 50, 50, 150),
        "left": (80, 100, 20, 100),
        "right": (20, 100, 80, 100),
    }
    x1, y1, x2, y2 = endpoints[direction]
    assert action == {"action": "swipe", "x1": x1, "y1": y1, "x2": x2, "y2": y2, "duration": 1000}
    assert len(router_requests) == 1
    assert no_sleep == []


@pytest.mark.parametrize("coordinate", [[], [500], [True, 500], [float("nan"), 300]])
def test_direction_does_not_hide_invalid_partial_coordinates(coordinate: object) -> None:
    with pytest.raises(ValueError, match="coordinates"):
        vlm_utils._tool_call_to_aitk(
            {"action": "swipe", "direction": "up", "coordinate": coordinate}
        )


@pytest.mark.parametrize("vector", [[], [float("nan")], [float("inf")], "base64"])
@pytest.mark.parametrize("recover", [False, True])
def test_text_vector_validation_uses_the_request_retry_budget(
    vector: object,
    recover: bool,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    embedding_body: dict[str, object],
    no_sleep: list[float],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracker = vlm_utils.TokenTracker()
    monkeypatch.setattr(vlm_utils, "token_tracker", tracker)
    bad = {
        **embedding_body,
        "data": [{"object": "embedding", "index": 0, "embedding": vector}],
    }
    router_responses.extend(
        httpx.Response(200, text=json.dumps(bad), headers={"content-type": "application/json"})
        for _ in range(1 if recover else 3)
    )
    if recover:
        router_responses.append(httpx.Response(200, json=embedding_body))
        assert vlm_utils.get_embedding(
            router_client, "home", model="google/gemini-embedding-2"
        ) == [0.25, 0.75]
    else:
        with pytest.raises(ValueError, match="embedding"):
            vlm_utils.get_embedding(router_client, "home", model="google/gemini-embedding-2")
    assert len(router_requests) == (2 if recover else 3)
    assert no_sleep == ([2.0] if recover else [2.0, 4.0])
    assert tracker.snapshot()["calls"] == int(recover)


def test_text_http_schema_and_vector_failures_share_one_budget(
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    embedding_body: dict[str, object],
    no_sleep: list[float],
) -> None:
    router_responses.extend(
        [
            httpx.Response(503, json={"error": {"code": 503, "message": "busy"}}),
            httpx.Response(
                200, json={key: value for key, value in embedding_body.items() if key != "model"}
            ),
            httpx.Response(200, json={**embedding_body, "data": []}),
            httpx.Response(200, json=embedding_body),
        ]
    )
    with pytest.raises(ValueError, match="one embedding"):
        vlm_utils.get_embedding(router_client, "home", model="google/gemini-embedding-2")
    assert len(router_requests) == 3
    assert len(router_responses) == 1
    assert no_sleep == [2.0, 4.0]


@pytest.mark.parametrize("other_package", [False, True])
def test_graph_preflight_allows_empty_vectors_and_unrelated_packages(
    other_package: bool,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    chat_body: dict[str, object],
    embedding_body: dict[str, object],
    png_screenshot: str,
) -> None:
    manager = GraphManager(
        page_detail_client=router_client,
        embedding_client=router_client,
        page_detail_model="openai/gpt-6.1-sol",
        embedding_model="google/gemini-embedding-2",
    )
    manager.graph.add_node(
        "old",
        activity="com.other.Home" if other_package else "com.demo.Home",
        description_embedding=[0.25, 0.75] if other_package else [],
    )
    body = {
        **chat_body,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": '{"page_description":"home","state":{},"elements":[]}',
                },
            }
        ],
    }
    router_responses.extend(
        [httpx.Response(200, json=body), httpx.Response(200, json=embedding_body)]
    )
    node = manager.identify_state("com.demo.Home", png_screenshot)
    assert manager.graph.nodes[node]["description_embedding_model"] == manager.embedding_model
    assert len(router_requests) == 2


@pytest.mark.parametrize(
    ("image_url", "override_url", "image_key", "override_key", "expected_key"),
    [
        (None, None, None, None, "text-key"),
        ("https://text.example:443/images", None, None, None, "text-key"),
        ("https://image.example/v1", None, None, None, None),
        ("http://text.example/v1", None, None, None, None),
        ("https://text.example:444/v1", None, None, None, None),
        ("https://image.example/v1", None, "image-key", None, "image-key"),
        ("https://image.example/v1", None, "${APP_GRAPH_UNSET}", None, None),
        ("https://image.example/v1", "https://third.example/v1", "image-key", None, None),
        ("https://image.example/v1", "https://third.example/v1", "image-key", "cli-key", "cli-key"),
        (
            "https://image.example/v1",
            "https://image.example:443/v2",
            "image-key",
            None,
            "image-key",
        ),
        (None, DEFAULT_BASE_URL, None, None, "router-key"),
    ],
)
@pytest.mark.usefixtures("model_environment")
def test_image_credential_inheritance_stays_with_its_origin(
    image_url: str | None,
    override_url: str | None,
    image_key: str | None,
    override_key: str | None,
    expected_key: str | None,
    router_http: httpx.Client,
    router_requests: list[httpx.Request],
    router_responses: list[httpx.Response],
    embedding_body: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-key")
    settings = resolve_image_embedding_settings(
        {
            "embedding": {"base_url": "https://text.example/v1", "api_key": "text-key"},
            "image_embedding": {"base_url": image_url, "api_key": image_key},
        },
        base_url_override=override_url,
        api_key_override=override_key,
    )
    assert settings.api_key == expected_key
    if expected_key is None:
        with pytest.raises(ValueError, match=r"image_embedding\.api_key"):
            make_client(settings._asdict(), role="image_embedding", http_client=router_http)
        assert router_requests == []
    else:
        client, _ = make_client(settings._asdict(), role="image_embedding", http_client=router_http)
        router_responses.append(httpx.Response(200, json=embedding_body))
        vlm_utils.get_image_embedding(client, "image", model="google/gemini-embedding-2")
        request = router_requests[0]
        assert request.url == httpx.URL(settings.base_url.rstrip("/") + "/embeddings")
        assert request.headers["authorization"] == f"Bearer {expected_key}"


@pytest.mark.parametrize(
    "role", ["page_detail", "instruction", "action", "embedding", "image_embedding"]
)
@pytest.mark.parametrize(
    "endpoint",
    [
        "https://other.example/v1",
        "http://openrouter.ai/api/v1",
        "https://openrouter.ai:444/api/v1",
        "https://openrouter.ai:0/api/v1",
    ],
)
@pytest.mark.usefixtures("model_environment")
def test_shared_router_key_never_falls_back_to_a_different_origin(
    role: str,
    endpoint: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-secret")
    with pytest.raises(ValueError, match=rf"{role}\.api_key") as caught:
        make_client({"model": "openai/gpt-6.1-sol", "base_url": endpoint}, role=role)
    assert "router-secret" not in str(caught.value)
    assert (
        resolve_image_embedding_settings({"image_embedding": {"base_url": endpoint}}).api_key
        is None
    )


@pytest.mark.parametrize("status", [307, 308])
@pytest.mark.parametrize(
    ("destination", "foreign"),
    [
        ("/canonical", False),
        ("https://other.example/canonical", True),
        ("http://openrouter.ai/canonical", True),
    ],
)
def test_owned_clients_preserve_redirects_and_strip_foreign_credentials_after_reopening(
    status: int,
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
                httpx.Response(status, headers={"location": destination}),
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


@pytest.mark.parametrize("endpoint", ["relative/path", "ftp://other.example/v1", "https:///v1"])
def test_invalid_endpoint_fails_before_client_creation(endpoint: str) -> None:
    with pytest.raises(ValueError, match="absolute HTTP or HTTPS"):
        make_client(
            {"model": "qwen/qwen3.8-flash", "base_url": endpoint, "api_key": "test"}, role="action"
        )


def test_borrowed_client_retains_redirect_policy_and_ownership(router_http: httpx.Client) -> None:
    client, _ = make_client(
        {"model": "qwen/qwen3.8-flash", "api_key": "test"}, role="action", http_client=router_http
    )
    for _ in range(2):
        with client:
            assert client.sdk_configuration.client is router_http
            assert not router_http.follow_redirects
        assert not router_http.is_closed


@pytest.mark.parametrize("entry", ["factory", "embedding", "image_embedding", "override"])
@pytest.mark.parametrize("empty", [False, True])
def test_unresolved_endpoint_cannot_redirect_an_explicit_credential(
    entry: str,
    empty: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if empty:
        monkeypatch.setenv("APP_GRAPH_ENDPOINT", "")
    cfg = {"base_url": "${APP_GRAPH_ENDPOINT}", "api_key": "endpoint-secret"}

    def configure() -> None:
        if entry == "factory":
            make_client(cfg, role="action")
        elif entry == "override":
            resolve_image_embedding_settings(
                {}, base_url_override="${APP_GRAPH_ENDPOINT}", api_key_override="endpoint-secret"
            )
        else:
            resolve_image_embedding_settings({entry: cfg})

    with pytest.raises(ValueError, match="endpoint") as caught:
        configure()
    assert "endpoint-secret" not in str(caught.value)
