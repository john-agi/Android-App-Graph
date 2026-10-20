"""Tests for android_app_graph.kobe: the shared client, model setup and exploration recovery."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import override

import httpx
import pytest
from openrouter import OpenRouter
from openrouter.errors import OpenRouterError

from android_app_graph.kobe import Kobe

_ROLES = ("page_detail", "embedding", "instruction", "action")
_FORCE_STOP = ["adb", "-s", "test-device", "shell", "am", "force-stop", "com.demo"]
_PACKAGE_LAUNCH = [
    "adb",
    "-s",
    "test-device",
    "shell",
    "monkey",
    "-p",
    "com.demo",
    "-c",
    "android.intent.category.LAUNCHER",
    "1",
]
_ACTIVITY_LAUNCH = ["adb", "-s", "test-device", "shell", "am", "start", "-n", "com.demo/.Main"]


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


class _ReplayDevice(_Device):
    def __init__(self, state: dict[str, object], log: list[object]) -> None:
        super().__init__(state)
        self.config = {"device": {"udid": "test-device"}}
        self._log = log

    @override
    def exe_action(self, action: dict[str, object]) -> None:
        self._log.append(action)


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


@pytest.fixture
def replaying_kobe(
    tmp_path: Path, png_screenshot: str, router_client: OpenRouter, device_log: list[object]
) -> Kobe:
    return Kobe(
        _ReplayDevice({"activity": "com.demo.Home", "screenshot": png_screenshot}, device_log),
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


@pytest.mark.usefixtures("no_sleep")
def test_navigating_to_the_start_node_restarts_the_app_and_identifies_the_screen_again(
    replaying_kobe: Kobe,
    png_screenshot: str,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    chat_response: Callable[..., httpx.Response],
    embedding_body: dict[str, object],
    device_log: list[object],
) -> None:
    for _ in range(2):
        router_responses.extend(
            [
                chat_response(content='{"page_description":"home"}'),
                httpx.Response(200, json=embedding_body),
            ]
        )
    node = replaying_kobe.graph.identify_state("com.demo.Home", png_screenshot)
    # The cache would skip the model, and a reference screenshot would add a verify request.
    replaying_kobe.graph._last_identify_cache = None
    replaying_kobe.graph.graph.nodes[node].pop("reference_screenshot", None)
    assert replaying_kobe.navigate_to_node(node) == node
    assert len(router_requests) == 4
    assert device_log == [_FORCE_STOP, _PACKAGE_LAUNCH]


@pytest.mark.usefixtures("no_sleep")
def test_navigation_restarts_the_app_through_its_launch_activity(
    replaying_kobe: Kobe, device_log: list[object], monkeypatch: pytest.MonkeyPatch
) -> None:
    replaying_kobe.launch_activity = "com.demo/.Main"
    replaying_kobe.graph.graph.add_node("s0_home")
    monkeypatch.setattr(replaying_kobe.graph, "identify_state", lambda *_args: "s0_home")
    assert replaying_kobe.navigate_to_node("s0_home") == "s0_home"
    assert device_log == [_FORCE_STOP, _ACTIVITY_LAUNCH]


def test_navigation_without_a_recorded_path_leaves_the_app_running(
    replaying_kobe: Kobe, device_log: list[object]
) -> None:
    replaying_kobe.graph.graph.add_nodes_from(["s0_home", "s1_orphan"])
    with pytest.raises(ValueError, match="No path from s0_home to s1_orphan"):
        replaying_kobe.navigate_to_node("s1_orphan")
    assert device_log == []


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


class _ScriptedDevice:
    w = 100
    h = 200

    def __init__(self, *screens: dict[str, object]) -> None:
        self.config: dict[str, object] = {}
        self._screens = screens
        self.actions: list[dict[str, object]] = []

    def get_state(self) -> dict[str, object]:
        return self._screens[len(self.actions)]

    def exe_action(self, action: dict[str, object]) -> None:
        self.actions.append(action)


def _explore_one_step_the_action_model_ends(
    kobe: Kobe,
    device: _ScriptedDevice,
    router_responses: list[httpx.Response],
    chat_response: Callable[..., httpx.Response],
    embedding_body: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(kobe, "controller", device)
    monkeypatch.setattr("android_app_graph.kobe.soft_keyboard_hint", lambda: "")
    page = {
        "page_description": "home",
        "elements": [{"description": "Settings icon", "position": "top right"}],
    }
    router_responses.extend(
        [
            chat_response(content=json.dumps(page)),
            httpx.Response(200, json=embedding_body),
            chat_response(content="Tap the Settings icon"),
            chat_response(content='{"action":"end"}'),
        ]
    )
    kobe.explore(max_steps=1)


@pytest.mark.usefixtures("no_sleep")
def test_an_action_model_end_is_recorded_as_the_back_press_it_executes(
    kobe: Kobe,
    png_screenshot: str,
    router_responses: list[httpx.Response],
    chat_response: Callable[..., httpx.Response],
    embedding_body: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    home: dict[str, object] = {
        "activity": "com.demo.Home",
        "screenshot": png_screenshot,
        "package": "com.demo",
    }
    device = _ScriptedDevice(home, home)
    caplog.set_level(logging.INFO, logger="test")
    _explore_one_step_the_action_model_ends(
        kobe, device, router_responses, chat_response, embedding_body, monkeypatch
    )
    assert device.actions == [{"action": "back"}]
    [(source, _target, edge)] = kobe.graph.graph.edges(data=True)
    assert edge["actions"] == [[{"action": "back"}]]
    assert edge["instructions"] == ["Press back"]
    [element] = kobe.graph.graph.nodes[source]["interactable_elements"]
    assert element["explored"] is True
    assert '"Tap the Settings icon"' in caplog.text


@pytest.mark.usefixtures("no_sleep")
def test_an_action_model_end_whose_back_press_leaves_the_app_is_recorded_as_press_back(
    kobe: Kobe,
    png_screenshot: str,
    router_responses: list[httpx.Response],
    chat_response: Callable[..., httpx.Response],
    embedding_body: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home: dict[str, object] = {
        "activity": "com.demo.Home",
        "screenshot": png_screenshot,
        "package": "com.demo",
    }
    launcher: dict[str, object] = {
        "activity": "com.android.launcher3.Launcher",
        "screenshot": png_screenshot,
        "package": "com.android.launcher3",
    }
    device = _ScriptedDevice(home, launcher, home)
    _explore_one_step_the_action_model_ends(
        kobe, device, router_responses, chat_response, embedding_body, monkeypatch
    )
    [(_source, target, edge)] = kobe.graph.graph.edges(data=True)
    assert target == "ext_com.android.launcher3"
    assert edge["instructions"] == ["Press back"]


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


@pytest.mark.usefixtures("no_sleep")
def test_coverage_checkpoint_restarts_the_app_before_replaying_to_its_target(
    replaying_kobe: Kobe,
    router_responses: list[httpx.Response],
    chat_response: Callable[..., httpx.Response],
    device_log: list[object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The replay from the start node used to run on whatever screen was showing."""
    open_settings = {"action": "tap", "coordinate": [10, 20]}
    open_detail = {"action": "tap", "coordinate": [30, 40]}
    replaying_kobe.graph.graph.add_nodes_from(["s0_home", "s1_settings", "s2_detail"])
    replaying_kobe.graph.add_edge("s0_home", "s1_settings", action=[open_settings])
    replaying_kobe.graph.add_edge("s1_settings", "s2_detail", action=[open_detail])
    router_responses.extend(
        [
            chat_response(content='{"issues":[],"summary":"no duplicates"}'),
            chat_response(content='{"node_id":"s1_settings","reason":"unexplored"}'),
        ]
    )
    monkeypatch.setattr(replaying_kobe.graph, "identify_state", lambda *_args: "s1_settings")
    assert (
        replaying_kobe._maybe_continue_from_coverage_checkpoint(50, 100, "s2_detail")
        == "s1_settings"
    )
    assert device_log == [_FORCE_STOP, _PACKAGE_LAUNCH, open_settings]
