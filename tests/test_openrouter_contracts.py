"""Compatibility and request-boundary regressions for the OpenRouter migration."""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest
from openrouter import OpenRouter

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
