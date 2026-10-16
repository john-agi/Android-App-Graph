"""Compatibility and request-boundary regressions for the OpenRouter migration."""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest
from openrouter import OpenRouter

from android_app_graph.embedding_cache import resolve_image_embedding_settings
from android_app_graph.utils import DEFAULT_BASE_URL, make_client, vlm_utils
from android_app_graph.utils.graph_manager import GraphManager


@pytest.mark.parametrize("other_package", [False, True])
def test_graph_preflight_allows_empty_vectors_and_unrelated_packages(
    other_package: bool,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    embedding_body: dict[str, object],
    png_screenshot: str,
    chat_response: Callable[..., httpx.Response],
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
    router_responses.extend(
        [
            chat_response(content='{"page_description":"home","state":{},"elements":[]}'),
            httpx.Response(200, json=embedding_body),
        ]
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
    endpoint: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-secret")
    assert (
        resolve_image_embedding_settings({"image_embedding": {"base_url": endpoint}}).api_key
        is None
    )


@pytest.mark.parametrize("entry", ["embedding", "image_embedding", "override"])
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
        if entry == "override":
            resolve_image_embedding_settings(
                {}, base_url_override="${APP_GRAPH_ENDPOINT}", api_key_override="endpoint-secret"
            )
        else:
            resolve_image_embedding_settings({entry: cfg})

    with pytest.raises(ValueError, match="endpoint") as caught:
        configure()
    assert "endpoint-secret" not in str(caught.value)
