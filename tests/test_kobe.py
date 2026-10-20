"""Tests for android_app_graph.kobe: the shared client, model setup and exploration recovery."""

from __future__ import annotations

import base64
import io
import json
import logging
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
from openrouter import OpenRouter
from openrouter.errors import OpenRouterError
from PIL import Image

from android_app_graph.device import DeviceController
from android_app_graph.kobe import Kobe
from android_app_graph.utils.graph_manager import EXTERNAL_WEB_NODE

_ROLES = ("page_detail", "embedding", "instruction", "action")
_TAP_REPLY = '{"action":"click","coordinate":[500,500]}'


def _screen(activity: str, color: str) -> dict[str, object]:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), color).save(buffer, format="PNG")
    return {"activity": activity, "screenshot": base64.b64encode(buffer.getvalue()).decode("ascii")}


HOME_SCREEN = _screen("com.demo.Home", "white")


class _Screens:
    """A DeviceController that records each action and then shows the next screen, if any."""

    w = 100
    h = 200

    def __init__(self, *screens: dict[str, object]) -> None:
        self.config: dict[str, object] = {}
        self._screens = screens
        self.actions: list[dict[str, object]] = []

    def get_state(self) -> dict[str, object]:
        return self._screens[min(len(self.actions), len(self._screens) - 1)]

    def exe_action(self, action: dict[str, object]) -> None:
        self.actions.append(action)


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


def _explorer(device: DeviceController, tmp_path: Path, router_client: OpenRouter) -> Kobe:
    return Kobe(
        device,
        "demo",
        "com.demo",
        logging.getLogger("test"),
        graph_dir=tmp_path,
        vlm_config={role: {"model": f"provider/{role}"} for role in _ROLES},
        client=router_client,
    )


@pytest.fixture
def kobe(tmp_path: Path, png_screenshot: str, router_client: OpenRouter) -> Kobe:
    return _explorer(
        _Device({"activity": "com.demo.Home", "screenshot": png_screenshot}),
        tmp_path,
        router_client,
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


@pytest.mark.usefixtures("model_environment")
def test_exploration_uses_the_client_it_is_given_and_leaves_it_open(
    tmp_path: Path,
    router_client: OpenRouter,
    router_http: httpx.Client,
) -> None:
    built = Kobe(
        _Device(AssertionError("a finished exploration must not access the device")),
        "demo",
        "com.demo",
        logging.getLogger("test"),
        graph_dir=tmp_path,
        client=router_client,
    )
    built.explore(max_steps=0)
    assert router_client.sdk_configuration.client is router_http


def test_a_rejected_page_description_leaves_no_node_or_cached_identity(
    kobe: Kobe,
    png_screenshot: str,
    router_responses: list[httpx.Response],
    no_sleep: list[float],
) -> None:
    router_responses.append(httpx.Response(401, json={"error": {"message": "bad key"}}))
    with pytest.raises(OpenRouterError):
        kobe.graph.identify_state("com.demo.Home", png_screenshot)
    assert len(kobe.graph.graph) == 0
    assert kobe.graph._last_identify_cache is None
    assert no_sleep == []


def test_navigating_to_the_start_node_identifies_the_screen_again(
    kobe: Kobe,
    png_screenshot: str,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    chat_response: Callable[..., httpx.Response],
    embedding_body: dict[str, object],
) -> None:
    for _ in range(2):
        router_responses.extend(
            [
                chat_response(content='{"page_description":"home"}'),
                httpx.Response(200, json=embedding_body),
            ]
        )
    node = kobe.graph.identify_state("com.demo.Home", png_screenshot)
    # The cache would skip the model, and a reference screenshot would add a verify request.
    kobe.graph._last_identify_cache = None
    kobe.graph.graph.nodes[node].pop("reference_screenshot", None)
    assert kobe.navigate_to_node(node) == node
    assert len(router_requests) == 4


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


@pytest.mark.usefixtures("no_sleep")
def test_an_in_app_screen_that_mentions_a_privacy_policy_gets_a_node_and_an_edge(
    tmp_path: Path,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    chat_response: Callable[..., httpx.Response],
    embedding_body: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A "privacy policy" keyword in a description used to make the app's own page external."""
    monkeypatch.setattr("android_app_graph.kobe.soft_keyboard_hint", lambda: "")
    about = _screen("com.demo.SettingsActivity", "gray")
    device = _Screens(HOME_SCREEN, about)
    explorer = _explorer(device, tmp_path, router_client)
    orthogonal = {"object": "embedding", "index": 0, "embedding": [0.75, -0.25]}
    router_responses.extend(
        [
            chat_response(content='{"page_description":"home"}'),
            httpx.Response(200, json=embedding_body),
            chat_response(content="Tap the About row"),
            chat_response(content=_TAP_REPLY),
            chat_response(
                content='{"page_description":"About page with privacy policy link",'
                '"is_external_web":false}'
            ),
            httpx.Response(200, json={**embedding_body, "data": [orthogonal]}),
        ]
    )

    explorer.explore(max_steps=1)

    assert [action["action"] for action in device.actions] == ["tap"]
    assert list(explorer.graph.graph.edges) == [
        ("s0_home", "s1_about_page_with_privacy_policy_link")
    ]


@pytest.mark.usefixtures("no_sleep")
def test_an_external_web_page_gets_an_edge_to_the_shared_node_and_a_back_press(
    tmp_path: Path,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    chat_response: Callable[..., httpx.Response],
    embedding_body: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("android_app_graph.kobe.soft_keyboard_hint", lambda: "")
    browser = _screen("com.demo.ArticleActivity", "blue")
    device = _Screens(HOME_SCREEN, browser, HOME_SCREEN)
    explorer = _explorer(device, tmp_path, router_client)
    router_responses.extend(
        [
            chat_response(content='{"page_description":"home"}'),
            httpx.Response(200, json=embedding_body),
            chat_response(content="Tap the privacy policy link"),
            chat_response(content=_TAP_REPLY),
            chat_response(content='{"page_description":"news article","is_external_web":true}'),
        ]
    )

    explorer.explore(max_steps=1)

    assert set(explorer.graph.graph) == {"s0_home", EXTERNAL_WEB_NODE}
    assert list(explorer.graph.graph.edges) == [("s0_home", EXTERNAL_WEB_NODE)]
    assert [action["action"] for action in device.actions] == ["tap", "back"]


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
