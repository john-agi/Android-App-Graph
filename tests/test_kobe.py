"""Tests for android_app_graph.kobe: the shared client, model setup and exploration recovery."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
from openrouter import OpenRouter
from openrouter.errors import OpenRouterError

from android_app_graph.kobe import Kobe

_ROLES = ("page_detail", "embedding", "instruction", "action")


class _Device:
    """A DeviceController whose screen never changes; any device action fails the test."""

    w = 100
    h = 200

    def __init__(self, state: dict[str, object] | Exception) -> None:
        self.config: dict[str, object] = {}
        self._state = state

    def get_state(self) -> dict[str, object]:
        if isinstance(self._state, Exception):
            raise self._state
        return self._state

    def exe_action(self, action: dict[str, object]) -> None:
        raise AssertionError(f"No device action expected: {action}")


@pytest.fixture
def kobe(tmp_path: Path, png_screenshot: str, router_client: OpenRouter) -> Kobe:
    return Kobe(
        _Device({"activity": "com.demo.Home", "screenshot": png_screenshot}),
        "demo",
        "com.demo",
        logging.getLogger("test"),
        graph_dir=tmp_path,
        vlm_config={role: {"model": f"provider/{role}"} for role in _ROLES},
        client=router_client,
    )


def test_exploration_resolves_all_role_models_before_device_access(
    tmp_path: Path, router_client: OpenRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    for role in _ROLES:
        monkeypatch.setenv(f"APP_GRAPH_{role.upper()}_MODEL", f"provider/{role}")
    agent = Kobe(
        _Device(AssertionError("model setup must not access the device")),
        "demo",
        "com.demo.app",
        logging.getLogger("test"),
        graph_dir=tmp_path,
        vlm_config={"action": {"temperature": 0.0, "timeout": 9}},
        client=router_client,
    )
    assert agent.page_detail.model == "provider/page_detail"
    assert agent.instruction.model == "provider/instruction"
    assert (agent.action.model, agent.action.temperature) == ("provider/action", 0.0)
    assert agent.action.request_timeout == 9.0
    assert agent.graph.embedding_model == "provider/embedding"
    assert agent.graph.page_detail == agent.page_detail
    assert agent.client is agent.graph.client is router_client


@pytest.mark.parametrize("exceptional", [False, True])
@pytest.mark.usefixtures("model_environment")
def test_exploration_uses_the_client_it_is_given_and_leaves_it_open(
    exceptional: bool,
    tmp_path: Path,
    router_client: OpenRouter,
    router_http: httpx.Client,
) -> None:
    built = Kobe(
        _Device(RuntimeError("device failed")),
        "demo",
        "com.demo",
        logging.getLogger("test"),
        graph_dir=tmp_path,
        client=router_client,
    )
    if exceptional:
        with pytest.raises(RuntimeError, match="device failed"):
            built.explore(max_steps=1)
    else:
        built.explore(max_steps=0)
    assert router_client.sdk_configuration.client is router_http


def test_public_graph_and_navigation_recover_after_request_failure(
    kobe: Kobe,
    png_screenshot: str,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    chat_response: Callable[..., httpx.Response],
    embedding_body: dict[str, object],
    no_sleep: list[float],
) -> None:
    router_responses.append(httpx.Response(401, json={"error": {"message": "bad key"}}))
    with pytest.raises(OpenRouterError):
        kobe.graph.identify_state("com.demo.Home", png_screenshot)
    assert len(kobe.graph.graph) == 0
    assert kobe.graph._last_identify_cache is None
    for _ in range(2):
        router_responses.extend(
            [
                chat_response(content='{"page_description":"home"}'),
                httpx.Response(200, json=embedding_body),
            ]
        )
    node = kobe.graph.identify_state("com.demo.Home", png_screenshot)
    kobe.graph._last_identify_cache = None
    kobe.graph.graph.nodes[node].pop("reference_screenshot", None)
    assert kobe.navigate_to_node(node) == node
    assert len(router_requests) == 5
    assert no_sleep == []


@pytest.mark.parametrize("malformed", [False, True])
def test_exploration_wait_recovery_respects_step_limit(
    malformed: bool,
    kobe: Kobe,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    chat_response: Callable[..., httpx.Response],
    embedding_body: dict[str, object],
    no_sleep: list[float],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("android_app_graph.kobe.soft_keyboard_hint", lambda: "")
    router_responses.extend(
        [
            chat_response(content='{"page_description":"home"}'),
            httpx.Response(200, json=embedding_body),
        ]
    )
    for _ in range(2):
        router_responses.append(chat_response(content="Tap Search"))
        router_responses.extend(
            chat_response(content='{"action":"invented"}' if malformed else '{"action":"wait"}')
            for _ in range(2 if malformed else 1)
        )
    kobe.explore(max_steps=2)
    assert kobe.graph.total_steps_completed == 2
    assert json.loads(kobe.graph_path.read_text())["total_steps_completed"] == 2
    step = ["provider/instruction", *["provider/action"] * (2 if malformed else 1)]
    assert [json.loads(request.content)["model"] for request in router_requests] == [
        "provider/page_detail",
        "provider/embedding",
        *step,
        *step,
    ]
    assert len(kobe.graph.graph.edges) == 0
    assert no_sleep == [1, 1]


def test_coverage_checkpoint_asks_the_page_detail_model_where_to_continue(
    kobe: Kobe,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    chat_response: Callable[..., httpx.Response],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kobe.graph.graph.add_edge("s0_home", "s1_settings")
    router_responses.extend(
        [
            chat_response(content='{"issues":[],"summary":"no duplicates"}'),
            chat_response(content='{"node_id":"s1_settings","reason":"unexplored"}'),
        ]
    )
    monkeypatch.setattr(kobe, "navigate_to_node", lambda node: node)
    assert kobe._maybe_continue_from_coverage_checkpoint(50, 100, "s0_home") == "s1_settings"
    assert [json.loads(request.content)["model"] for request in router_requests] == [
        "provider/page_detail"
    ] * 2
