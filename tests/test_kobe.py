"""Tests for android_app_graph.kobe: the shared client, model setup and exploration recovery."""

from __future__ import annotations

import base64
import io
import json
import logging
import re
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, NoReturn, override

import httpx
import pytest
from openrouter import OpenRouter
from openrouter.errors import OpenRouterError
from PIL import Image

from android_app_graph.device import (
    DeviceController,
    DeviceUnreachableError,
    InputRejectedError,
    PermissionGrants,
    StillOutsideAppError,
)
from android_app_graph.graph_files import identify_file_path, read_json_lines
from android_app_graph.kobe import ExplorerConfig, Kobe, NoPathError
from android_app_graph.payloads import as_object_dict
from android_app_graph.run_environment import UnsafeReplayError
from android_app_graph.screen_image import encode_screenshot
from android_app_graph.step_log import Observation, StepRecorder
from android_app_graph.utils import DEFAULT_UNREACHABLE_LIMIT, MatchingSettings
from android_app_graph.utils.graph_manager import EXTERNAL_WEB_NODE, EdgeDetails
from android_app_graph.utils.vlm_utils import ActionPlan, ActionRequest, PlanRequest

_ROLES = ("page_detail", "embedding", "instruction", "action")
_FORCE_STOP = ["adb", "-s", "test-device", "shell", "am", "force-stop", "com.demo"]
_RESOLVE_LAUNCHER = [
    *("adb", "-s", "test-device", "shell", "pm", "resolve-activity", "--brief"),
    *("-a", "android.intent.action.MAIN", "-c", "android.intent.category.LAUNCHER", "com.demo"),
]


def _clear_task_start(component: str) -> list[str]:
    return ["adb", "-s", "test-device", "shell", "am", "start", "-n", component, "-f", "0x10008000"]


_HOME_START = _clear_task_start("com.demo/.Home")
# The run's first restart asks the device for the launcher activity; later ones reuse it.
_FIRST_RESTART = [_FORCE_STOP, _RESOLVE_LAUNCHER, _HOME_START]
_RESTART = [_FORCE_STOP, _HOME_START]
_TAP_REPLY = '{"action":"click","coordinate":[500,500],"required_pointers":1}'


def _screen(activity: str, color: str) -> dict[str, object]:
    buffer = io.BytesIO()
    Image.new("RGB", (100, 200), color).save(buffer, format="PNG")
    screenshot = base64.b64encode(buffer.getvalue()).decode("ascii")
    return {"activity": activity, "screenshot": screenshot, "package": "com.demo"}


HOME_SCREEN = _screen("com.demo.Home", "white")


class _Screens:
    """A DeviceController that records each action and then shows the next screen, if any."""

    w = 100
    h = 200

    def __init__(self, *screens: dict[str, object]) -> None:
        self.config: dict[str, object] = {"device": {"udid": "test-device"}}
        self._screens = screens
        self.actions: list[dict[str, object]] = []

    def get_state(self, **_kwargs: float | None) -> dict[str, object]:
        return self._screens[min(len(self.actions), len(self._screens) - 1)]

    def exe_action(self, action: dict[str, object]) -> None:
        self.actions.append(action)


class _Device:
    """A DeviceController whose screen never changes; any device action fails the test."""

    w = 100
    h = 200

    def __init__(self, state: dict[str, object] | Exception) -> None:
        self.config: dict[str, object] = {"device": {"udid": "test-device"}}
        self._state = state

    def get_state(self, **_kwargs: float | None) -> dict[str, object]:
        if isinstance(self._state, Exception):
            raise self._state
        return self._state

    def exe_action(self, action: dict[str, object]) -> None:
        msg = f"No device action expected: {action}"
        raise AssertionError(msg)


class _ReplayDevice(_Device):
    def __init__(self, state: dict[str, object], log: list[object]) -> None:
        super().__init__(state)
        self._log = log

    @override
    def exe_action(self, action: dict[str, object]) -> None:
        self._log.append(action)


def _identify_lines(explorer: Kobe) -> list[dict[str, Any]]:
    """The lines of the explorer's identify file, beside its graph file, in order."""
    path = identify_file_path(explorer.graph_path)
    assert explorer.graph.identify_file == path
    return [dict(line) for line in read_json_lines(path)]


def _explorer(
    device: DeviceController,
    tmp_path: Path,
    router_client: OpenRouter,
    *,
    launch_activity: str | None = None,
    unreachable_limit: float = DEFAULT_UNREACHABLE_LIMIT,
) -> Kobe:
    return Kobe(
        device,
        ExplorerConfig(
            app_name="demo",
            package_name="com.demo",
            graph_dir=tmp_path,
            vlm_config={role: {"model": f"provider/{role}"} for role in _ROLES},
            launch_activity=launch_activity,
            unreachable_limit=unreachable_limit,
        ),
        client=router_client,
        logger=logging.getLogger("test"),
    )


@pytest.fixture
def kobe(tmp_path: Path, png_screenshot: str, router_client: OpenRouter) -> Kobe:
    return _explorer(
        _Device({"activity": "com.demo.Home", "screenshot": png_screenshot, "package": "com.demo"}),
        tmp_path,
        router_client,
    )


@pytest.fixture
def replaying_kobe(
    tmp_path: Path, png_screenshot: str, router_client: OpenRouter, device_log: list[object]
) -> Kobe:
    return _explorer(
        _ReplayDevice(
            {"activity": "com.demo.Home", "screenshot": png_screenshot, "package": "com.demo"},
            device_log,
        ),
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
        ExplorerConfig(
            app_name="demo",
            package_name="com.demo.app",
            graph_dir=tmp_path,
            vlm_config={"action": {"temperature": 0.0, "timeout": 9}},
        ),
        client=router_client,
        logger=logging.getLogger("test"),
    )
    assert agent.page_detail.model == "provider/page_detail"
    assert agent.instruction.model == "provider/instruction"
    assert (agent.action.model, agent.action.temperature) == ("provider/action", 0.0)
    assert agent.action.request_timeout == 9.0
    assert agent.graph.embedding is not None
    assert agent.graph.embedding.model == "provider/embedding"
    assert agent.graph.image_embedding == agent.graph.embedding
    assert agent.graph.page_detail == agent.page_detail
    assert agent.client is agent.graph.client is router_client


def test_exploration_matches_screens_under_the_configured_settings(
    tmp_path: Path, router_client: OpenRouter, write_graph_file: Callable[..., None]
) -> None:
    """A new graph takes the config's matching settings; a saved graph must have been built
    with them."""
    vlm_config: dict[str, object] = {role: {"model": f"provider/{role}"} for role in _ROLES}
    vlm_config["matching"] = {"verifier_calls": 5}
    config = ExplorerConfig(
        app_name="demo", package_name="com.demo", graph_dir=tmp_path, vlm_config=vlm_config
    )
    device = _Device(AssertionError("loading must not read the device"))
    explorer = Kobe(device, config, client=router_client, logger=logging.getLogger("test"))
    assert explorer.graph.matching == MatchingSettings(verifier_calls=5)
    graph_path = tmp_path / "demo" / "demo.json"
    graph_path.parent.mkdir(exist_ok=True)
    write_graph_file(graph_path, {"nodes": [{"id": "s0_home"}], "edges": []}, "provider/embedding")

    with pytest.raises(ValueError, match="was built with the matching settings"):
        Kobe(device, config, client=router_client, logger=logging.getLogger("test"))


@pytest.mark.usefixtures("model_environment")
def test_exploration_uses_the_client_it_is_given_and_leaves_it_open(
    tmp_path: Path,
    router_client: OpenRouter,
    router_http: httpx.Client,
) -> None:
    built = Kobe(
        _Device(AssertionError("a finished exploration must not access the device")),
        ExplorerConfig(app_name="demo", package_name="com.demo", graph_dir=tmp_path),
        client=router_client,
        logger=logging.getLogger("test"),
    )
    built.explore(max_steps=0)
    assert router_client.sdk_configuration.client is router_http


class _ReplayedError(Exception):
    """Ends exploration once it has chosen the node to replay to."""


def _replays(explorer: Kobe, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Make the explorer's replays end exploration; return the list each replay's target
    is appended to."""
    replayed: list[str] = []

    def replay(node_id: str) -> str:
        replayed.append(node_id)
        raise _ReplayedError

    monkeypatch.setattr(explorer, "navigate_to_node", replay)
    return replayed


def _write_form_and_dialog_graph(
    tmp_path: Path,
    write_graph_file: Callable[..., None],
    write_image_vectors: Callable[..., None],
) -> None:
    """Save the graph of a home screen leading to a form with unexplored fields, which
    ranks first, and to a dialog with no out-edges and nothing left to explore, which
    ranks second."""
    fields = [
        {"description": description, "explored": False} for description in ("Name", "APN", "Proxy")
    ]
    document: dict[str, object] = {
        "nodes": [
            {"id": "s0_home", "activity": "com.demo.Home"},
            {"id": "s1_form", "activity": "com.demo.Form", "interactable_elements": fields},
            {"id": "s2_dialog", "activity": "com.demo.Form"},
        ],
        "edges": [
            {"source": "s0_home", "target": "s1_form", "actions": [{"action": "tap"}]},
            {"source": "s0_home", "target": "s2_dialog", "actions": [{"action": "swipe"}]},
            {"source": "s1_form", "target": "s0_home", "actions": [{"action": "back"}]},
        ],
    }
    graph_path = tmp_path / "demo" / "demo.json"
    graph_path.parent.mkdir()
    write_graph_file(graph_path, document, embedding_model="provider/embedding")
    write_image_vectors(
        graph_path.parent / "steps",
        [f"{node}-observation" for node in ("s0_home", "s1_form", "s2_dialog")],
        model="provider/embedding",
    )


def test_resume_from_auto_replays_to_the_top_ranked_exploration_target(
    tmp_path: Path,
    router_client: OpenRouter,
    monkeypatch: pytest.MonkeyPatch,
    write_graph_file: Callable[..., None],
    write_image_vectors: Callable[..., None],
) -> None:
    """``auto`` resumes at the form with unexplored fields, not at the dialog with no
    out-edges and nothing left to explore."""
    _write_form_and_dialog_graph(tmp_path, write_graph_file, write_image_vectors)
    explorer = _explorer(
        _Device(AssertionError("the replay must start before the device is read")),
        tmp_path,
        router_client,
    )
    replayed = _replays(explorer, monkeypatch)

    with pytest.raises(_ReplayedError):
        explorer.explore(max_steps=1, resume_from="auto")

    assert replayed == ["s1_form"]


@pytest.mark.usefixtures("no_sleep")
def test_resume_from_auto_passes_over_a_node_whose_last_replay_missed(
    tmp_path: Path,
    router_client: OpenRouter,
    device_log: list[object],
    monkeypatch: pytest.MonkeyPatch,
    write_graph_file: Callable[..., None],
    write_image_vectors: Callable[..., None],
) -> None:
    """The replay to the form lands on the home screen. The checkpoint then offers the
    dialog and the start screen, and a run on the saved graph resumes at the dialog."""
    _write_form_and_dialog_graph(tmp_path, write_graph_file, write_image_vectors)
    explorer = _explorer(_ReplayDevice(HOME_SCREEN, device_log), tmp_path, router_client)
    _identifying(explorer, monkeypatch, "s0_home", "s0_home")

    assert explorer.navigate_to_node("s1_form") == "s0_home"
    offered = explorer.graph.get_exploration_target_candidates(
        package_name="com.demo", always_offer_start=True
    )
    assert [candidate["node_id"] for candidate in offered] == ["s2_dialog", "s0_home"]
    explorer.save_graph()

    resumed = _explorer(
        _Device(AssertionError("the replay must start before the device is read")),
        tmp_path,
        router_client,
    )
    replayed = _replays(resumed, monkeypatch)

    with pytest.raises(_ReplayedError):
        resumed.explore(max_steps=1, resume_from="auto")

    assert replayed == ["s2_dialog"]


def test_a_rejected_page_description_leaves_no_node(
    kobe: Kobe,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    chat_response: Callable[..., httpx.Response],
    embedding_body: dict[str, object],
    no_sleep: list[float],
) -> None:
    """Identifying the rejected screen again asks the model again."""
    router_responses.extend(
        [
            httpx.Response(200, json=embedding_body),
            chat_response(
                content='{"page_description": "home", "surface": {"kind": "page", "relation": "unknown", "return_depth": 0}}'
            ),
            httpx.Response(200, json=embedding_body),
            *(
                response
                for _ in range(2)
                for response in (
                    httpx.Response(200, json=embedding_body),
                    httpx.Response(401, json={"error": {"message": "bad key"}}),
                )
            ),
        ]
    )
    home = kobe.graph.identify_state(Observation.from_state(kobe.controller.get_state()))

    for _ in range(2):
        settings = {**kobe.controller.get_state(), "activity": "com.demo.Settings"}
        with pytest.raises(OpenRouterError):
            kobe.graph.identify_state(Observation.from_state({**settings, "pixel_hash": "other"}))
        assert list(kobe.graph.graph) == [home]

    assert [json.loads(request.content)["model"] for request in router_requests[3:]] == [
        "provider/embedding",
        "provider/page_detail",
    ] * 2
    assert no_sleep == []


@pytest.mark.usefixtures("no_sleep")
def test_navigating_to_the_start_node_restarts_the_app_and_identifies_the_screen_again(
    replaying_kobe: Kobe,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    chat_response: Callable[..., httpx.Response],
    embedding_body: dict[str, object],
    device_log: list[object],
    step_records: Callable[[Path], list[dict[str, Any]]],
) -> None:
    router_responses.extend(
        [
            httpx.Response(200, json=embedding_body),
            chat_response(
                content='{"page_description": "home", "surface": {"kind": "page", "relation": "unknown", "return_depth": 0}}'
            ),
            httpx.Response(200, json=embedding_body),
            # The replay ends on the start node's own pixels: an exact match.
            httpx.Response(200, json=embedding_body),
        ]
    )
    home = Observation.from_state(replaying_kobe.controller.get_state())
    node = replaying_kobe.graph.identify_state(home)
    assert replaying_kobe.navigate_to_node(node) == node
    assert len(router_requests) == 4
    assert device_log == _FIRST_RESTART
    (*_, replayed, match) = step_records(replaying_kobe.controller.steps)
    assert (match["kind"], match["source"]) == ("match", "navigation_replay")
    assert (match["observation"], match["expected_observation"]) == (
        replayed["id"],
        home.observation_id,
    )
    (*_, decision) = _identify_lines(replaying_kobe)
    assert decision["observation"] == replayed["id"]
    assert (decision["identify"]["stage"], decision["identify"]["expected_node"]) == (
        "exact",
        node,
    )


@pytest.mark.usefixtures("no_sleep")
def test_navigation_restarts_the_app_through_its_launch_activity(
    tmp_path: Path,
    router_client: OpenRouter,
    device_log: list[object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = _ReplayDevice(HOME_SCREEN, device_log)
    explorer = _explorer(device, tmp_path, router_client, launch_activity="com.demo/.Main")
    explorer.graph.graph.add_node("s0_home", replay_misses=0)
    monkeypatch.setattr(explorer.graph, "identify_state", lambda *_args, **_kwargs: "s0_home")
    assert explorer.navigate_to_node("s0_home") == "s0_home"
    assert device_log == [_FORCE_STOP, _clear_task_start("com.demo/.Main")]


def test_navigation_without_a_recorded_path_leaves_the_app_running(
    replaying_kobe: Kobe, device_log: list[object]
) -> None:
    replaying_kobe.graph.graph.add_nodes_from(["s0_home", "s1_orphan"], replay_misses=0)
    with pytest.raises(ValueError, match="No path from s0_home to s1_orphan"):
        replaying_kobe.navigate_to_node("s1_orphan")
    assert device_log == []


@pytest.mark.usefixtures("no_sleep")
@pytest.mark.parametrize("manual_resume", [False, True])
@pytest.mark.parametrize("aliased", [False, True])
@pytest.mark.parametrize("target", ["ext_com.foreign", EXTERNAL_WEB_NODE, "s1_boundary"])
def test_external_replay_target_is_rejected_before_device_or_model_access(
    kobe: Kobe,
    device_log: list[object],
    router_requests: list[httpx.Request],
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    *,
    aliased: bool,
    manual_resume: bool,
) -> None:
    """External destinations retain their recorded path and reject explicit replay."""
    kobe.graph.graph.add_node("s0_home", replay_misses=0)
    kobe.graph.graph.add_node(target, is_external=target == "s1_boundary", replay_misses=0)
    kobe.graph.add_edge("s0_home", target, action={"action": "tap", "coordinate": [30, 40]})
    requested = target
    if aliased:
        requested = "s2_alias"
        kobe.graph.graph.add_node(requested)
        assert kobe.graph.merge_nodes(target, requested)
    path = kobe.graph.find_path("s0_home", target)
    assert path is not None
    before = kobe.graph.graph.copy()
    monkeypatch.setattr(
        kobe.controller,
        "get_state",
        lambda **_kwargs: pytest.fail("An external replay target requires no observation"),
    )

    if manual_resume:
        with pytest.raises(ValueError, match="External replay target"):
            kobe.explore(max_steps=1, resume_from=requested)
    else:
        with pytest.raises(ValueError, match="External replay target"):
            kobe.navigate_to_node(requested)

    assert device_log == []
    assert router_requests == []
    assert kobe.graph.find_path("s0_home", target) == path
    assert dict(kobe.graph.graph.nodes(data=True)) == dict(before.nodes(data=True))
    assert dict(kobe.graph.graph.edges) == dict(before.edges)


@pytest.mark.usefixtures("no_sleep")
@pytest.mark.parametrize("action", [{"action": "tap", "coordinate": [30, 40]}, {"action": "menu"}])
def test_navigation_replays_an_edge_action_recorded_as_a_single_dict(
    replaying_kobe: Kobe,
    device_log: list[object],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    action: dict[str, object],
    write_graph_file: Callable[..., None],
) -> None:
    """An edge action saved as one dict reaches the device as it is."""
    graph_file = tmp_path / "saved.json"
    write_graph_file(
        graph_file,
        {"nodes": [{"id": node} for node in ("s0_home", "s1_detail")]},
        embedding_model="provider/embedding",
    )
    replaying_kobe.graph.load_graph(graph_file)
    replaying_kobe.graph.add_edge("s0_home", "s1_detail", action=action)
    replaying_kobe.graph.save_graph(graph_file)
    replaying_kobe.graph.load_graph(graph_file)
    _identifying(replaying_kobe, monkeypatch, "s0_home", "s1_detail")

    assert replaying_kobe.navigate_to_node("s1_detail") == "s1_detail"
    assert device_log == [*_FIRST_RESTART, action]


def _explore_sending(
    explorer: Kobe, monkeypatch: pytest.MonkeyPatch, *actions: dict[str, Any]
) -> None:
    """Explore one step per action, the action model sending ``actions`` in turn. Edge
    normalization, which the run's end asks for an edge with two actions, finds no
    template."""
    sent = iter(actions)
    monkeypatch.setattr("android_app_graph.kobe.plan_next_action", _planner("Act", []))
    monkeypatch.setattr(
        "android_app_graph.kobe.predict_next_action", lambda **_kwargs: (next(sent), "")
    )
    monkeypatch.setattr(
        "android_app_graph.utils.graph_manager.normalize_edge",
        lambda *_args, **_kwargs: {"is_template": False},
    )
    monkeypatch.setattr("android_app_graph.kobe.soft_keyboard_hint", lambda _udid: "")
    monkeypatch.setattr(
        "android_app_graph.utils.graph_manager.verify_replay_source",
        lambda *_args, **_kwargs: (True, "fixture arrival applies"),
    )
    explorer.explore(max_steps=len(actions))


@pytest.mark.usefixtures("no_sleep")
def test_a_replay_scrolls_a_list_as_exploration_did_before_tapping_a_row_on_it(
    replaying_kobe: Kobe, device_log: list[object], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tap exploration sent after a swipe that scrolled the list is replayed after that
    swipe, so it lands on the row it was sent to. A row opened in between and left by
    Back, which found the list as the swipe left it, is not replayed."""
    open_list = {"action": "tap", "x": 50, "y": 20}
    scroll_down = {"action": "swipe", "x1": 50, "y1": 180, "x2": 50, "y2": 40}
    open_row = {"action": "tap", "x": 50, "y": 60}
    back = {"action": "back"}
    open_other_row = {"action": "tap", "x": 50, "y": 190}
    top, scrolled = {"first_visible_item": "Google"}, {"first_visible_item": "Default apps"}
    _list_reached_from_home(replaying_kobe, "s2_detail", "s3_other_detail")
    _identifying(
        replaying_kobe,
        monkeypatch,
        *("s0_home", ("s1_list", top), ("s1_list", scrolled), "s2_detail"),
        *(("s1_list", scrolled), "s3_other_detail", "s0_home"),
        *(("s1_list", top), ("s1_list", scrolled), "s3_other_detail"),
    )
    _explore_sending(
        replaying_kobe, monkeypatch, open_list, scroll_down, open_row, back, open_other_row
    )
    device_log.clear()

    assert replaying_kobe.navigate_to_node("s3_other_detail") == "s3_other_detail"
    assert device_log == [*_FIRST_RESTART, open_list, scroll_down, open_other_row]


@pytest.mark.usefixtures("no_sleep")
def test_a_round_trip_by_back_that_changed_nothing_is_left_out_of_the_setup(
    replaying_kobe: Kobe, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Back from a row returns to the list in the state the list was in when exploration
    arrived there, so the next tap on the list needs nothing sent before it."""
    open_list, open_row, back, open_other_row = (
        {"action": "tap", "x": 50, "y": 20},
        {"action": "tap", "x": 50, "y": 60},
        {"action": "back"},
        {"action": "tap", "x": 50, "y": 90},
    )
    top = {"first_visible_item": "Google"}
    _list_reached_from_home(replaying_kobe, "s2_detail", "s3_other_detail")
    _identifying(
        replaying_kobe,
        monkeypatch,
        *("s0_home", ("s1_list", top), "s2_detail", ("s1_list", top), "s3_other_detail"),
    )

    _explore_sending(replaying_kobe, monkeypatch, open_list, open_row, back, open_other_row)

    assert _setups(replaying_kobe)[("s1_list", "s3_other_detail")] == [[]]


@pytest.mark.usefixtures("no_sleep")
def test_a_menu_opened_and_closed_again_is_left_out_of_the_setup(
    replaying_kobe: Kobe, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Closing a menu undoes opening it, so the next tap on the page needs neither."""
    open_page, open_menu, close_menu, open_row = (
        {"action": "tap", "x": 50, "y": 20},
        {"action": "tap", "x": 90, "y": 10},
        {"action": "back"},
        {"action": "tap", "x": 50, "y": 90},
    )
    closed, opened = {"open_menu": None}, {"open_menu": "overflow menu"}
    _list_reached_from_home(replaying_kobe, "s2_detail")
    _identifying(
        replaying_kobe,
        monkeypatch,
        *("s0_home", ("s1_list", closed), ("s1_list", opened), ("s1_list", closed), "s2_detail"),
    )

    _explore_sending(replaying_kobe, monkeypatch, open_page, open_menu, close_menu, open_row)

    assert _setups(replaying_kobe)[("s1_list", "s2_detail")] == [[]]


@pytest.mark.usefixtures("no_sleep")
def test_a_replay_opens_a_menu_on_a_form_filled_through_a_dialog_before_choosing_its_option(
    replaying_kobe: Kobe, device_log: list[object], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An option of a form's menu, the CA certificate menu that appears once a dialog of
    its own has set the form's security, is replayed after that dialog's round trip and
    the tap that opened the menu. The menu opened and closed before that is not replayed."""
    open_form = {"action": "tap", "x": 50, "y": 20}
    open_security = {"action": "tap", "x": 50, "y": 60}
    choose_enterprise = {"action": "tap", "x": 50, "y": 110}
    open_certificates = {"action": "tap", "x": 50, "y": 90}
    close_menu = {"action": "back"}
    install_certificates = {"action": "tap", "x": 50, "y": 130}
    form, dialog, installer = "s1_add_network", "s2_security_dialog", "s3_certificate_installer"
    unsecured = {"security": "None", "open_menu": None}
    enterprise = {"security": "WPA/WPA2-Enterprise", "open_menu": None}
    menu_open = {**enterprise, "open_menu": "CA certificate dropdown"}
    for node in ("s0_home", form, dialog, installer):
        replaying_kobe.graph.graph.add_node(node, package="com.demo", replay_misses=0)
    _identifying(
        replaying_kobe,
        monkeypatch,
        *("s0_home", (form, unsecured), dialog, (form, enterprise), (form, menu_open)),
        *((form, enterprise), (form, menu_open), installer, "s0_home"),
        *((form, unsecured), dialog, (form, enterprise), (form, menu_open), installer),
    )
    _explore_sending(
        replaying_kobe,
        monkeypatch,
        *(open_form, open_security, choose_enterprise, open_certificates, close_menu),
        *(open_certificates, install_certificates),
    )
    device_log.clear()

    assert replaying_kobe.navigate_to_node(installer) == installer
    assert device_log == [
        *_FIRST_RESTART,
        *(open_form, open_security, choose_enterprise, open_certificates, install_certificates),
    ]


_OPEN_DRAWER = {"action": "tap", "x": 5, "y": 10}
_OPEN_RECENT = {"action": "tap", "x": 30, "y": 40}
_DRAWER_CLOSED = {"open_menu": None}
_DRAWER_OPEN = {"open_menu": "navigation drawer"}


@pytest.mark.usefixtures("no_sleep")
def test_a_replay_opens_the_drawer_before_tapping_a_drawer_row_whose_setup_is_unknown(
    replaying_kobe: Kobe, device_log: list[object], monkeypatch: pytest.MonkeyPatch
) -> None:
    """How the device came to be on the home screen is unknown, so is the setup of every
    action sent there. The tap on a row of the drawer, which is part of the home screen's
    state, is replayed after the recorded tap that opened the drawer."""
    _list_reached_from_home(replaying_kobe, "s2_recent")
    _identifying(
        replaying_kobe,
        monkeypatch,
        *(("s0_home", _DRAWER_CLOSED), ("s0_home", _DRAWER_OPEN), "s2_recent"),
        *("s0_home", ("s0_home", _DRAWER_OPEN), "s2_recent"),
    )
    _explore_sending(replaying_kobe, monkeypatch, _OPEN_DRAWER, _OPEN_RECENT)
    device_log.clear()

    assert replaying_kobe.navigate_to_node("s2_recent") == "s2_recent"
    assert device_log == [*_FIRST_RESTART, _OPEN_DRAWER, _OPEN_RECENT]


@pytest.mark.usefixtures("no_sleep")
def test_a_drawer_row_tapped_in_a_drawer_no_recorded_action_opened_is_in_no_path(
    replaying_kobe: Kobe, device_log: list[object], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The drawer was open on the screen exploration began on, so no recorded action
    opens it: a replay cannot tap its row, and the checkpoint does not offer the screen
    only that tap reaches."""
    _list_reached_from_home(replaying_kobe, "s2_recent")
    _identifying(
        replaying_kobe,
        monkeypatch,
        *(("s0_home", _DRAWER_OPEN), "s2_recent", "s0_home", "s2_recent"),
    )
    _explore_sending(replaying_kobe, monkeypatch, _OPEN_RECENT)
    device_log.clear()

    with pytest.raises(NoPathError, match="No path from s0_home to s2_recent"):
        replaying_kobe.navigate_to_node("s2_recent")
    assert device_log == []
    offered = replaying_kobe.graph.get_exploration_target_candidates(package_name="com.demo")
    assert [candidate["node_id"] for candidate in offered] == ["s0_home"]


def _setups(explorer: Kobe) -> dict[tuple[str, str], list[object]]:
    return {(u, v): data["setups"] for u, v, data in explorer.graph.graph.edges(data=True)}


@pytest.mark.usefixtures("no_sleep")
def test_exploration_records_each_action_s_setup_unknown_on_the_screen_it_began_on(
    replaying_kobe: Kobe, monkeypatch: pytest.MonkeyPatch
) -> None:
    """How the device came to be on the first screen is unknown, so is the setup of an
    action sent there; a screen arrived at from elsewhere starts its setup empty."""
    scroll, open_list, open_row, back, open_other_row = (
        {"action": "swipe", "x1": 50, "y1": 180, "x2": 50, "y2": 40},
        {"action": "tap", "x": 50, "y": 20},
        {"action": "tap", "x": 50, "y": 60},
        {"action": "back"},
        {"action": "tap", "x": 50, "y": 90},
    )
    _list_reached_from_home(replaying_kobe, "s2_detail", "s3_other_detail")
    _identifying(
        replaying_kobe,
        monkeypatch,
        *("s0_home", "s0_home", "s1_list", "s2_detail", "s1_list", "s3_other_detail"),
    )

    _explore_sending(replaying_kobe, monkeypatch, scroll, open_list, open_row, back, open_other_row)

    assert _setups(replaying_kobe) == {
        ("s0_home", "s0_home"): [None],
        ("s0_home", "s1_list"): [None],
        ("s1_list", "s2_detail"): [[]],
        ("s2_detail", "s1_list"): [[]],
        ("s1_list", "s3_other_detail"): [[]],
    }


@pytest.mark.parametrize("web_page", [False, True], ids=["another app", "outside web page"])
@pytest.mark.usefixtures("no_sleep")
def test_after_a_step_that_leaves_the_app_the_setup_of_the_next_action_is_unknown(
    web_page: bool,
    tmp_path: Path,
    router_client: OpenRouter,
    screens_by_log: Callable[..., DeviceController],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Back may return to the screen the step left or a restart to the start screen, and
    neither is an action a replay could send."""
    open_list, scroll, open_link, open_row = (
        {"action": "tap", "x": 50, "y": 20},
        {"action": "swipe", "x1": 50, "y1": 180, "x2": 50, "y2": 40},
        {"action": "tap", "x": 50, "y": 60},
        {"action": "tap", "x": 50, "y": 90},
    )
    opened = _screen("com.demo.ArticleActivity", "blue") if web_page else _SEARCH_SCREEN
    outside = EXTERNAL_WEB_NODE if web_page else f"ext_{_SEARCH}"
    on_the_page = [EXTERNAL_WEB_NODE] if web_page else []
    explorer = _explorer(
        screens_by_log(HOME_SCREEN, HOME_SCREEN, HOME_SCREEN, opened, HOME_SCREEN),
        tmp_path,
        router_client,
    )
    top, scrolled = {"first_visible_item": "Google"}, {"first_visible_item": "Default apps"}
    _list_reached_from_home(explorer, "s2_detail", *on_the_page)
    _identifying(
        explorer,
        monkeypatch,
        *("s0_home", ("s1_list", top), ("s1_list", scrolled), *on_the_page, "s1_list"),
        "s2_detail",
    )

    _explore_sending(explorer, monkeypatch, open_list, scroll, open_link, open_row)

    assert _setups(explorer) == {
        ("s0_home", "s1_list"): [None],
        ("s1_list", "s1_list"): [[]],
        ("s1_list", outside): [[scroll]],
        ("s1_list", "s2_detail"): [None],
    }


@pytest.mark.parametrize(
    ("after_restart", "reached", "setup"),
    [
        pytest.param(["s0_home", "s1_list"], "s1_list", [], id="replay reached"),
        pytest.param(["s0_home", "s0_home"], "s0_home", None, id="replay missed"),
        pytest.param(["s1_list"], "s1_list", [], id="app opened on the node"),
    ],
)
@pytest.mark.usefixtures("no_sleep", "device_log")
def test_a_resumed_exploration_knows_the_setup_only_on_the_node_its_replay_reached(
    after_restart: list[str],
    reached: str,
    setup: list[object] | None,
    replaying_kobe: Kobe,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The launch arrives at the screen the app opens on, so a replay that sends nothing
    because the app opened on its node has reached it as one that sends its path has."""
    scroll = {"action": "swipe", "x1": 50, "y1": 180, "x2": 50, "y2": 40}
    _list_reached_from_home(replaying_kobe)
    replaying_kobe.graph.add_edge("s0_home", "s1_list", action=[{"action": "tap", "x": 1, "y": 2}])
    _identifying(replaying_kobe, monkeypatch, *after_restart, reached)
    monkeypatch.setattr("android_app_graph.kobe.plan_next_action", _planner("Scroll", []))
    monkeypatch.setattr("android_app_graph.kobe.predict_next_action", lambda **_kw: (scroll, ""))
    monkeypatch.setattr("android_app_graph.kobe.soft_keyboard_hint", lambda _udid: "")

    replaying_kobe.explore(max_steps=1, resume_from="s1_list")

    assert _setups(replaying_kobe)[(reached, reached)] == [setup]


_SCROLL_CLOCKS = {"action": "swipe", "x1": 50, "y1": 150, "x2": 50, "y2": 60}
_OPEN_ALARMS = {"action": "tap", "coordinate": [10, 190]}
_SCROLL_ALARMS = {"action": "swipe", "x1": 50, "y1": 170, "x2": 50, "y2": 30}
_OPEN_ALARM = {"action": "tap", "coordinate": [50, 40]}
_CLOCK_NODES = ("s0_alarms", "s1_clock", "s2_alarm", "s3_whats_new")


def _clock_graph(explorer: Kobe) -> None:
    """Give the graph Clock's alarms tab, its start node, reached from the clock tab, an
    alarm reached from the alarms tab, and a What's new screen no path leads from, each
    with a member; each action has a setup."""
    graph = explorer.graph
    for node in _CLOCK_NODES:
        graph.graph.add_node(node, replay_misses=0, observations=[f"{node}-observation"])
    graph.add_edge(
        "s1_clock", "s0_alarms", action=[_OPEN_ALARMS], details=EdgeDetails(setup=[_SCROLL_CLOCKS])
    )
    graph.add_edge(
        "s0_alarms", "s2_alarm", action=[_OPEN_ALARM], details=EdgeDetails(setup=[_SCROLL_ALARMS])
    )


@pytest.mark.parametrize(
    ("opened_on", "replayed"),
    [
        pytest.param("s0_alarms", [_SCROLL_ALARMS, _OPEN_ALARM], id="start node"),
        pytest.param(
            "s1_clock",
            [_SCROLL_CLOCKS, _OPEN_ALARMS, _SCROLL_ALARMS, _OPEN_ALARM],
            id="node with a path",
        ),
        pytest.param("s3_whats_new", [_SCROLL_ALARMS, _OPEN_ALARM], id="node without a path"),
    ],
)
@pytest.mark.usefixtures("no_sleep")
def test_a_replay_starts_from_the_node_the_app_opened_on(
    opened_on: str,
    replayed: list[object],
    replaying_kobe: Kobe,
    device_log: list[object],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An app can reopen on the screen used last instead of its start screen. The replay
    identifies the screen the restart shows, expecting the start node, and replays from
    that node when a recorded path leads from it, from the start node otherwise, each
    action after its setup."""
    _clock_graph(replaying_kobe)
    intermediate = ["s1_clock", "s0_alarms"] if opened_on == "s1_clock" else []
    expected = _identifying(
        replaying_kobe, monkeypatch, opened_on, *intermediate, opened_on, "s2_alarm"
    )
    caplog.set_level(logging.INFO, logger="test")

    reached = replaying_kobe.navigate_to_node("s2_alarm")

    assert device_log == [*_FIRST_RESTART, *replayed]
    intermediate_expected = [None, "s0_alarms"] if opened_on == "s1_clock" else []
    assert expected == ["s0_alarms", *intermediate_expected, None, "s2_alarm"]
    assert reached == "s2_alarm"
    opened_elsewhere = f"demo opened on {opened_on}, not on the start node s0_alarms."
    assert (opened_elsewhere in caplog.text) == (opened_on != "s0_alarms")


@pytest.mark.parametrize(
    ("opened_on", "target", "replayed", "misses"),
    [
        pytest.param(
            "s1_clock",
            "s2_alarm",
            [_SCROLL_CLOCKS, _OPEN_ALARMS, _SCROLL_ALARMS, _OPEN_ALARM],
            1,
            id="from a node with a path",
        ),
        pytest.param(
            "s3_whats_new",
            "s2_alarm",
            [_SCROLL_ALARMS, _OPEN_ALARM],
            1,
            id="from a node without a path",
        ),
        pytest.param("s3_whats_new", "s0_alarms", [], 1, id="to the start node, sending nothing"),
        pytest.param("s2_alarm", "s2_alarm", [], 0, id="app opened on the target"),
    ],
)
@pytest.mark.usefixtures("no_sleep")
def test_a_replay_from_the_screen_the_app_opened_on_counts_a_miss_where_it_lands_elsewhere(
    opened_on: str,
    target: str,
    replayed: list[object],
    misses: int,
    replaying_kobe: Kobe,
    device_log: list[object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The app opens on a node other than the start node, and the replay ends on that
    node. Unless it is the target, the target counts one miss, whether the replay started
    from that node, from the start node or sent nothing; no other node's count changes."""
    _clock_graph(replaying_kobe)
    _identifying(replaying_kobe, monkeypatch, *([opened_on] * (len(replayed) + 1)))

    reached = replaying_kobe.navigate_to_node(target)

    assert reached == opened_on
    assert device_log == [*_FIRST_RESTART, *replayed]
    assert dict(replaying_kobe.graph.graph.nodes(data="replay_misses")) == {
        node: misses if node == target else 0 for node in _CLOCK_NODES
    }


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
    monkeypatch.setattr("android_app_graph.kobe.soft_keyboard_hint", lambda _udid: "")
    router_responses.extend(
        [
            httpx.Response(200, json=embedding_body),
            chat_response(
                content='{"page_description": "home", "surface": {"kind": "page", "relation": "unknown", "return_depth": 0}}'
            ),
            httpx.Response(200, json=embedding_body),
        ]
    )
    for _ in range(2):
        router_responses.append(
            chat_response(
                content='{"instruction":"Tap Search","element":null,"required_pointers":1}'
            )
        )
        router_responses.extend(
            chat_response(
                content='{"action":"invented","required_pointers":0}'
                if malformed
                else '{"action":"wait","required_pointers":0}'
            )
            for _ in range(2 if malformed else 1)
        )
    kobe.explore(max_steps=2)
    assert kobe.graph.total_steps_completed == 2
    assert json.loads(kobe.graph_path.read_text())["total_steps_completed"] == 2
    step = ["provider/instruction", *["provider/action"] * (2 if malformed else 1)]
    assert [json.loads(request.content)["model"] for request in router_requests] == [
        "provider/embedding",
        "provider/page_detail",
        "provider/embedding",
        *step,
        *step,
    ]
    assert len(kobe.graph.graph.edges) == 0
    assert no_sleep == [1, 1]


@pytest.mark.usefixtures("no_sleep")
def test_an_exploration_step_asks_the_explored_device_whether_its_keyboard_is_up(
    kobe: Kobe,
    router_responses: list[httpx.Response],
    chat_response: Callable[..., httpx.Response],
    embedding_body: dict[str, object],
    device_log: list[object],
) -> None:
    router_responses.extend(
        [
            httpx.Response(200, json=embedding_body),
            chat_response(
                content='{"page_description": "home", "surface": {"kind": "page", "relation": "unknown", "return_depth": 0}}'
            ),
            httpx.Response(200, json=embedding_body),
            chat_response(
                content='{"instruction":"Tap Search","element":null,"required_pointers":1}'
            ),
            chat_response(content='{"action":"wait","required_pointers":0}'),
        ]
    )
    kobe.explore(max_steps=1)
    assert device_log == [["adb", "-s", "test-device", "shell", "dumpsys", "input_method"]]


@pytest.mark.usefixtures("no_sleep")
def test_an_in_app_screen_that_mentions_a_privacy_policy_gets_a_node_and_an_edge(
    tmp_path: Path,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    chat_response: Callable[..., httpx.Response],
    embedding_body: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A "privacy policy" keyword in a description does not make the app's own page external."""
    monkeypatch.setattr("android_app_graph.kobe.soft_keyboard_hint", lambda _udid: "")
    about = _screen("com.demo.SettingsActivity", "gray")
    device = _Screens(HOME_SCREEN, about)
    explorer = _explorer(device, tmp_path, router_client)
    orthogonal = {"object": "embedding", "index": 0, "embedding": [0.75, -0.25]}
    router_responses.extend(
        [
            httpx.Response(200, json=embedding_body),
            chat_response(
                content='{"page_description": "home", "surface": {"kind": "page", "relation": "unknown", "return_depth": 0}}'
            ),
            httpx.Response(200, json=embedding_body),
            chat_response(
                content='{"element":null,"instruction":"Tap the About row","required_pointers":1}'
            ),
            chat_response(content=_TAP_REPLY),
            httpx.Response(200, json=embedding_body),
            chat_response(
                content='{"page_description": "About page with privacy policy link", "is_external_web": false, "surface": {"kind": "page", "relation": "unknown", "return_depth": 0}}'
            ),
            httpx.Response(200, json={**embedding_body, "data": [orthogonal]}),
            chat_response(content=_DIFFERENT),
        ]
    )

    explorer.explore(max_steps=1)

    assert [action["action"] for action in device.actions] == ["tap"]
    assert list(explorer.graph.graph.edges) == [("s0", "s1")]


_ORTHOGONAL = {"object": "embedding", "index": 0, "embedding": [0.75, -0.25]}
_DIFFERENT = '{"same":false,"reason":"another page"}'


@pytest.mark.usefixtures("no_sleep")
def test_an_exploration_step_is_recorded_in_the_step_log(
    tmp_path: Path,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    chat_response: Callable[..., httpx.Response],
    embedding_body: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    step_records: Callable[[Path], list[dict[str, Any]]],
) -> None:
    """Each screen read is a WebP image and an observation record, each screen matched has
    a match record, an action names the observation before it, and the step's edge is a
    transition naming both screens, the planner's instruction and element, the action and
    the nodes the screens became. What placed each screen on its node, its describe
    output and decision, goes to the identify file beside the graph, not to the step
    log."""
    monkeypatch.setattr("android_app_graph.kobe.soft_keyboard_hint", lambda _udid: "")
    explorer = _explorer(
        _Screens(HOME_SCREEN, _screen("com.demo.SettingsActivity", "gray")),
        tmp_path,
        router_client,
    )
    home = {
        "page_description": "home",
        "elements": [{"description": "About", "position": "top"}],
        "surface": {"kind": "page", "relation": "unknown", "return_depth": 0},
    }
    router_responses.extend(
        [
            httpx.Response(200, json=embedding_body),
            chat_response(content=json.dumps(home)),
            httpx.Response(200, json=embedding_body),
            chat_response(
                content='{"element":1,"instruction":"Tap the About row","required_pointers":1}'
            ),
            chat_response(content=_TAP_REPLY),
            httpx.Response(200, json=embedding_body),
            chat_response(
                content='{"page_description": "about", "surface": {"kind": "page", "relation": "unknown", "return_depth": 0}}'
            ),
            httpx.Response(200, json={**embedding_body, "data": [_ORTHOGONAL]}),
            chat_response(content=_DIFFERENT),
        ]
    )

    explorer.explore(max_steps=1)

    steps_dir = tmp_path / "demo" / "steps"
    records = step_records(steps_dir)
    assert [(record["id"], record["kind"]) for record in records] == [
        ("000000", "observation"),
        ("000001", "match"),
        ("000002", "action"),
        ("000003", "observation"),
        ("000004", "match"),
        ("000005", "transition"),
    ]
    assert {(record["run_kind"], record["source"]) for record in records} == {
        ("explore", "main_loop")
    }
    assert not any("describe" in record or "identify" in record for record in records)
    assert sorted(path.name for path in steps_dir.glob("*.webp")) == ["000000.webp", "000003.webp"]
    assert [
        (r["observation"], r["expected_observation"]) for r in records if r["kind"] == "match"
    ] == [
        ("000000", None),
        ("000003", None),
    ]
    tap = {"action": "tap", "x": 50, "y": 100}
    assert (records[2]["observation"], records[2]["action"]) == ("000000", tap)
    transition = {key: records[5][key] for key in ("before", "after", "instruction", "element")}
    assert transition == {
        "before": "000000",
        "after": "000003",
        "instruction": "Tap the About row",
        "element": 1,
    }
    assert (records[5]["action"], records[5]["from_node"], records[5]["to_node"]) == (
        tap,
        "s0",
        "s1",
    )
    assert explorer.graph.graph.nodes["s1"]["observations"] == ["000003"]
    lines = _identify_lines(explorer)
    assert [(line["observation"], line["identify"]["node"]) for line in lines] == [
        ("000000", "s0"),
        ("000003", "s1"),
    ]
    assert [line["describe"]["page_description"] for line in lines] == ["home", "about"]


@pytest.mark.usefixtures("no_sleep")
def test_an_external_web_page_is_recorded_on_ext_web_and_left_by_external_web_recovery(
    tmp_path: Path,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    chat_response: Callable[..., httpx.Response],
    embedding_body: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    step_records: Callable[[Path], list[dict[str, Any]]],
) -> None:
    """The page's screen becomes a member of ext_web, the transition leads there, and the
    back press off the page and the screen read and matched after it are external-web
    recovery. That match expects the screenshot the step started from."""
    monkeypatch.setattr("android_app_graph.kobe.soft_keyboard_hint", lambda _udid: "")
    browser = _screen("com.demo.ArticleActivity", "blue")
    explorer = _explorer(_Screens(HOME_SCREEN, browser, HOME_SCREEN), tmp_path, router_client)
    router_responses.extend(
        [
            httpx.Response(200, json=embedding_body),
            chat_response(
                content='{"page_description": "home", "surface": {"kind": "page", "relation": "unknown", "return_depth": 0}}'
            ),
            httpx.Response(200, json=embedding_body),
            chat_response(
                content='{"element":null,"instruction":"Tap the privacy policy link","required_pointers":1}'
            ),
            chat_response(content=_TAP_REPLY),
            httpx.Response(200, json=embedding_body),
            chat_response(
                content='{"page_description": "news article", "is_external_web": true, "surface": {"kind": "page", "relation": "unknown", "return_depth": 0}}'
            ),
            # Back shows home again: an exact match.
            httpx.Response(200, json=embedding_body),
        ]
    )

    explorer.explore(max_steps=1)

    records = step_records(explorer.controller.steps)
    assert [(record["kind"], record["source"]) for record in records] == [
        ("observation", "main_loop"),
        ("match", "main_loop"),
        ("action", "main_loop"),
        ("observation", "main_loop"),
        ("match", "main_loop"),
        ("transition", "main_loop"),
        ("action", "external_web_recovery"),
        ("observation", "external_web_recovery"),
        ("match", "external_web_recovery"),
    ]
    assert (records[5]["after"], records[5]["to_node"]) == ("000003", EXTERNAL_WEB_NODE)
    assert explorer.graph.graph.nodes[EXTERNAL_WEB_NODE]["observations"] == ["000003"]
    assert (records[8]["observation"], records[8]["expected_observation"]) == ("000007", "000000")
    (*_, back) = _identify_lines(explorer)
    assert (back["identify"]["stage"], back["identify"]["expected_node"]) == ("exact", "s0")


@pytest.mark.parametrize(
    ("action_reply", "executed", "recorded"),
    [
        (_TAP_REPLY, "tap", "Tap the privacy policy link"),
        ('{"action":"end","required_pointers":0}', "back", "Press back"),
    ],
)
@pytest.mark.usefixtures("no_sleep")
def test_an_external_web_page_gets_an_edge_to_the_shared_node_and_a_back_press(
    action_reply: str,
    executed: str,
    recorded: str,
    tmp_path: Path,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    chat_response: Callable[..., httpx.Response],
    embedding_body: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The edge's target observation is the page's own description."""
    monkeypatch.setattr("android_app_graph.kobe.soft_keyboard_hint", lambda _udid: "")
    browser = _screen("com.demo.ArticleActivity", "blue")
    device = _Screens(HOME_SCREEN, browser, HOME_SCREEN)
    explorer = _explorer(device, tmp_path, router_client)
    router_responses.extend(
        [
            httpx.Response(200, json=embedding_body),
            chat_response(
                content='{"page_description": "home", "surface": {"kind": "page", "relation": "unknown", "return_depth": 0}}'
            ),
            httpx.Response(200, json=embedding_body),
            chat_response(
                content='{"element":null,"instruction":"Tap the privacy policy link","required_pointers":1}'
            ),
            chat_response(content=action_reply),
            httpx.Response(200, json=embedding_body),
            chat_response(
                content='{"page_description": "news article", "is_external_web": true, "surface": {"kind": "page", "relation": "unknown", "return_depth": 0}}'
            ),
            httpx.Response(200, json=embedding_body),
        ]
    )

    explorer.explore(max_steps=1)

    assert set(explorer.graph.graph) == {"s0", EXTERNAL_WEB_NODE}
    assert list(explorer.graph.graph.edges) == [("s0", EXTERNAL_WEB_NODE)]
    edge = explorer.graph.graph.edges["s0", EXTERNAL_WEB_NODE]
    assert (edge["instructions"], edge["target_observations"]) == ([recorded], ["news article"])
    assert [action["action"] for action in device.actions] == [executed, "back"]


_BACK: dict[str, object] = {"action": "back"}


@pytest.mark.parametrize(
    ("back_presses_on_the_page", "leaving"),
    [(1, [_BACK, _BACK]), (3, [_BACK, _BACK, _BACK, *_FIRST_RESTART])],
)
@pytest.mark.usefixtures("no_sleep")
def test_the_explorer_presses_back_until_it_leaves_an_external_web_page(
    back_presses_on_the_page: int,
    leaving: list[object],
    tmp_path: Path,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    chat_response: Callable[..., httpx.Response],
    embedding_body: dict[str, object],
    device_log: list[object],
    screens_by_log: Callable[..., DeviceController],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The explorer keeps pressing back while the page still shows, so the next step acts
    from an app screen and no edge out of the shared ext_web node gives find_path a
    shortcut between unrelated screens."""
    monkeypatch.setattr("android_app_graph.kobe.soft_keyboard_hint", lambda _udid: "")
    web_page = _screen("com.demo.ArticleActivity", "blue")
    settings = _screen("com.demo.Settings", "gray")
    device = screens_by_log(HOME_SCREEN, *[web_page] * len(leaving), HOME_SCREEN, settings)
    explorer = _explorer(device, tmp_path, router_client)
    external = '{"page_description": "news article", "is_external_web": true, "surface": {"kind": "page", "relation": "unknown", "return_depth": 0}}'
    orthogonal = {"object": "embedding", "index": 0, "embedding": [0.75, -0.25]}
    router_responses.extend(
        [
            httpx.Response(200, json=embedding_body),
            chat_response(
                content='{"page_description": "home", "surface": {"kind": "page", "relation": "unknown", "return_depth": 0}}'
            ),
            httpx.Response(200, json=embedding_body),
            chat_response(
                content='{"element":null,"instruction":"Tap the privacy policy link","required_pointers":1}'
            ),
            chat_response(content=_TAP_REPLY),
            *[
                response
                for _ in range(1 + back_presses_on_the_page)
                for response in (
                    httpx.Response(200, json=embedding_body),
                    chat_response(content=external),
                )
            ],
            # Home again, before or after a restart: an exact match.
            httpx.Response(200, json=embedding_body),
            chat_response(
                content='{"element":null,"instruction":"Tap Settings","required_pointers":1}'
            ),
            chat_response(content=_TAP_REPLY),
            httpx.Response(200, json=embedding_body),
            chat_response(
                content='{"page_description": "settings", "surface": {"kind": "page", "relation": "unknown", "return_depth": 0}}'
            ),
            httpx.Response(200, json={**embedding_body, "data": [orthogonal]}),
            chat_response(content=_DIFFERENT),
        ]
    )

    explorer.explore(max_steps=2)

    assert set(explorer.graph.graph.edges) == {("s0", EXTERNAL_WEB_NODE), ("s0", "s1")}
    tap = {"action": "tap", "x": 50, "y": 100}
    assert device_log == [tap, *leaving, tap]


@pytest.mark.parametrize(
    ("element", "explored", "warned"),
    [(2, ["Compose FAB"], False), (None, [], False), (3, [], True)],
)
@pytest.mark.usefixtures("no_sleep")
def test_a_step_marks_exactly_the_element_the_planner_names(
    element: int | None,
    explored: list[str],
    warned: bool,
    kobe: Kobe,
    router_responses: list[httpx.Response],
    chat_response: Callable[..., httpx.Response],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A tap on the FAB never marks the dark mode toggle, whose "on" is inside "button"."""
    kobe.graph.graph.add_node(
        "s0_home",
        interactable_elements=[
            {"description": "Turn on dark mode toggle", "position": "top", "explored": False},
            {"description": "Compose FAB", "position": "bottom right", "explored": False},
        ],
        replay_misses=0,
    )
    kobe.graph.graph.add_node("s1_compose", replay_misses=0)
    screens = iter(["s0_home", "s1_compose"])
    monkeypatch.setattr(kobe.graph, "identify_state", lambda *_args, **_kwargs: next(screens))
    _describing_by_node_id(kobe, monkeypatch)
    monkeypatch.setattr(kobe.controller, "exe_action", lambda _action: None)
    monkeypatch.setattr("android_app_graph.kobe.soft_keyboard_hint", lambda _udid: "")
    instruction = "Tap the floating action button in the bottom right"
    router_responses.extend(
        [
            chat_response(
                content=json.dumps(
                    {"required_pointers": 1, "element": element, "instruction": instruction}
                )
            ),
            chat_response(
                content="",
                tool_arguments='{"action":"click","coordinate":[900,900],"required_pointers":1}',
            ),
            chat_response(content='{"is_template":false}'),
        ]
    )

    with caplog.at_level(logging.WARNING):
        kobe.explore(max_steps=1)

    elements = kobe.graph.graph.nodes["s0_home"]["interactable_elements"]
    assert [e["description"] for e in elements if e["explored"]] == explored
    assert ("is not among the 2 unexplored element(s) on s0_home" in caplog.text) is warned
    assert kobe.graph.graph["s0_home"]["s1_compose"]["instructions"] == [instruction]


def test_a_crash_mid_exploration_is_logged_with_its_traceback(
    kobe: Kobe, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    failure = RuntimeError("planner down")

    def _crash(**_kwargs: object) -> NoReturn:
        raise failure

    kobe.graph.graph.add_node("s0_home", interactable_elements=[], replay_misses=0)
    monkeypatch.setattr(kobe.graph, "identify_state", lambda *_args, **_kwargs: "s0_home")
    _describing_by_node_id(kobe, monkeypatch)
    monkeypatch.setattr("android_app_graph.kobe.soft_keyboard_hint", lambda _udid: "")
    monkeypatch.setattr("android_app_graph.kobe.plan_next_action", _crash)

    with caplog.at_level(logging.ERROR, logger="test"), pytest.raises(RuntimeError):
        kobe.explore(max_steps=1)

    (crash,) = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert crash.getMessage() == "Exploration crashed at step 0. Graph saved. Re-run to resume."
    assert isinstance(crash.exc_info, tuple)
    assert crash.exc_info[1] is failure


class _ScriptedDevice:
    w = 100
    h = 200

    def __init__(self, *screens: dict[str, object]) -> None:
        self.config: dict[str, object] = {"device": {"udid": "test-device"}}
        self._screens = screens
        self.actions: list[dict[str, object]] = []

    def get_state(self, **_kwargs: float | None) -> dict[str, object]:
        return self._screens[len(self.actions)]

    def exe_action(self, action: dict[str, object]) -> None:
        self.actions.append(action)


def _recording(device: DeviceController, kobe: Kobe) -> StepRecorder:
    """Record ``device`` in the step log of ``kobe``, as Kobe records the device it is given."""
    return StepRecorder(
        device,
        kobe.controller.steps,
        run_kind="explore",
        source="main_loop",
        target_package=kobe.package_name,
    )


def _explore_one_step_the_action_model_ends(
    kobe: Kobe,
    device: DeviceController,
    router_responses: list[httpx.Response],
    chat_response: Callable[..., httpx.Response],
    embedding_body: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(kobe, "controller", _recording(device, kobe))
    monkeypatch.setattr("android_app_graph.kobe.soft_keyboard_hint", lambda _udid: "")
    page = {
        "page_description": "home",
        "elements": [
            {"description": "Search box", "position": "top left"},
            {"description": "Settings icon", "position": "top right"},
        ],
        "surface": {"kind": "page", "relation": "unknown", "return_depth": 0},
    }
    plan = {"required_pointers": 1, "element": 2, "instruction": "Tap the Settings icon"}
    router_responses.extend(
        [
            httpx.Response(200, json=embedding_body),
            chat_response(content=json.dumps(page)),
            httpx.Response(200, json=embedding_body),
            chat_response(content=json.dumps(plan)),
            chat_response(content='{"action":"end","required_pointers":0}'),
            # Back shows home again: an exact match.
            httpx.Response(200, json=embedding_body),
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
    step_records: Callable[[Path], list[dict[str, Any]]],
) -> None:
    """The edge records the back press as "Press back"; the step log's transition keeps
    what the planner asked for next to the back press that ran."""
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
    elements = kobe.graph.graph.nodes[source]["interactable_elements"]
    assert [e["description"] for e in elements if e["explored"]] == ["Settings icon"]
    assert '"Tap the Settings icon"' in caplog.text
    (transition,) = [r for r in step_records(kobe.controller.steps) if r["kind"] == "transition"]
    assert (transition["instruction"], transition["element"], transition["action"]) == (
        "Tap the Settings icon",
        2,
        {"action": "back"},
    )


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


@pytest.mark.parametrize("relaunched", [False, True], ids=["back press", "relaunch"])
@pytest.mark.usefixtures("no_sleep")
def test_a_step_that_leaves_the_app_records_its_screen_on_the_app_s_node(
    relaunched: bool,
    kobe: Kobe,
    png_screenshot: str,
    router_responses: list[httpx.Response],
    chat_response: Callable[..., httpx.Response],
    embedding_body: dict[str, object],
    device_log: list[object],
    screens_by_log: Callable[..., DeviceController],
    monkeypatch: pytest.MonkeyPatch,
    step_records: Callable[[Path], list[dict[str, Any]]],
) -> None:
    """The other app's node takes the screen the step reached as its member, unmatched, the
    transition leads to that node, and the back press home and the screens read after it,
    before or after a restart, are external-app recovery. The screen the app is back on is
    matched expecting the screenshot the step started from."""
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
    outside = [launcher, launcher] if relaunched else [launcher]
    _explore_one_step_the_action_model_ends(
        kobe,
        screens_by_log(home, *outside, home),
        router_responses,
        chat_response,
        embedding_body,
        monkeypatch,
    )

    records = step_records(kobe.controller.steps)
    assert [(record["kind"], record["source"]) for record in records] == [
        ("observation", "main_loop"),
        ("match", "main_loop"),
        ("action", "main_loop"),
        ("observation", "main_loop"),
        ("transition", "main_loop"),
        ("action", "external_app_recovery"),
        ("observation", "external_app_recovery"),
        *(
            [("system_event", "external_app_recovery"), ("observation", "external_app_recovery")]
            if relaunched
            else []
        ),
        ("match", "main_loop"),
    ]
    assert (records[4]["after"], records[4]["to_node"]) == ("000003", "ext_com.android.launcher3")
    assert kobe.graph.graph.nodes["ext_com.android.launcher3"]["observations"] == ["000003"]
    assert (records[-1]["observation"], records[-1]["expected_observation"]) == (
        records[-2]["id"],
        "000000",
    )
    restart = _FIRST_RESTART if relaunched else []
    assert device_log == [_BACK, _BACK, *restart]


@pytest.mark.usefixtures("no_sleep")
def test_a_step_that_starts_a_node_is_recorded_from_the_node_the_verifier_rejected(
    tmp_path: Path,
    two_screen_device: DeviceController,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    chat_response: Callable[..., httpx.Response],
    embedding_body: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The verifier rejects the node the step left and refines only the new screen's
    description, so the step's edge goes from that node, under the ID it had, and its
    element is marked explored."""
    monkeypatch.setattr("android_app_graph.kobe.soft_keyboard_hint", lambda _udid: "")
    agent = Kobe(
        two_screen_device,
        ExplorerConfig(
            app_name="demo",
            package_name="com.demo.app",
            graph_dir=tmp_path,
            vlm_config={role: {"model": f"provider/{role}"} for role in _ROLES},
        ),
        client=router_client,
        logger=logging.getLogger("test"),
    )
    search_page = {
        "page_description": "Chat search page",
        "elements": [{"description": "Search button", "position": "top"}],
        "surface": {"kind": "page", "relation": "unknown", "return_depth": 0},
    }
    verdict = {
        "same": False,
        "reason": "a query has been entered",
        "new_description": "Chat search results page",
    }
    router_responses.extend(
        [
            httpx.Response(200, json=embedding_body),
            chat_response(content=json.dumps(search_page)),
            httpx.Response(200, json=embedding_body),
            chat_response(
                content='{"element":1,"instruction":"Tap the Search button","required_pointers":1}'
            ),
            chat_response(
                content="",
                tool_arguments='{"action":"click","coordinate":[500,500],"required_pointers":1}',
            ),
            httpx.Response(200, json=embedding_body),
            chat_response(
                content='{"page_description": "Chat search results", "surface": {"kind": "page", "relation": "unknown", "return_depth": 0}}'
            ),
            httpx.Response(200, json=embedding_body),
            chat_response(content=json.dumps(verdict)),
            *[httpx.Response(200, json=embedding_body) for _ in range(2)],
        ]
    )

    with caplog.at_level(logging.WARNING, logger="android_app_graph"):
        agent.explore(max_steps=1)

    assert agent.graph.graph.has_edge("s0", "s1")
    assert agent.graph.graph.nodes["s0"]["interactable_elements"] == [
        {"description": "Search button", "position": "top", "explored": True}
    ]
    assert "add_edge: missing node(s)" not in caplog.text


_WAIT_REPLY = '{"action":"wait","required_pointers":0}'


def _planner(
    instruction: str, planned_from: list[object], *, required_pointers: int = 1
) -> Callable[..., ActionPlan]:
    """A ``plan_next_action`` that records the page description of each screen it plans from."""

    def plan(*, request: PlanRequest, **_kwargs: object) -> ActionPlan:
        planned_from.append(request.page_description)
        return ActionPlan(instruction, None, required_pointers)

    return plan


@pytest.mark.usefixtures("no_sleep")
def test_the_planner_and_a_new_edge_take_the_screen_s_own_description(
    tmp_path: Path,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    chat_response: Callable[..., httpx.Response],
    embedding_body: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The results screen joins the search page's node on the verifier's word: the edge
    it ends records the results screen's description, and the next step plans from it,
    not from the node's."""
    search = _screen("com.demo.Search", "white")
    results = _screen("com.demo.Search", "gray")
    explorer = _explorer(_Screens(search, results), tmp_path, router_client)
    planned_from: list[object] = []
    monkeypatch.setattr("android_app_graph.kobe.plan_next_action", _planner("Search", planned_from))
    monkeypatch.setattr("android_app_graph.kobe.soft_keyboard_hint", lambda _udid: "")
    same = '{"same":true,"reason":"one screen","new_description":null}'
    router_responses.extend(
        [
            httpx.Response(200, json=embedding_body),
            chat_response(
                content='{"page_description": "Chat search page", "surface": {"kind": "page", "relation": "unknown", "return_depth": 0}}'
            ),
            httpx.Response(200, json=embedding_body),
            chat_response(content=_TAP_REPLY),
            httpx.Response(200, json=embedding_body),
            chat_response(
                content='{"page_description": "Chat search results", "surface": {"kind": "page", "relation": "unknown", "return_depth": 0}}'
            ),
            httpx.Response(200, json=embedding_body),
            chat_response(content=same),
            chat_response(content=_TAP_REPLY),
            # The results show again: an exact match.
            httpx.Response(200, json=embedding_body),
        ]
    )

    explorer.explore(max_steps=2)

    node = "s0"
    assert list(explorer.graph.graph) == [node]
    assert planned_from == ["Chat search page", "Chat search results"]
    assert explorer.graph.graph[node][node]["target_observations"] == ["Chat search results"]


@pytest.mark.usefixtures("no_sleep")
def test_resuming_from_a_node_plans_the_first_step_from_the_node_the_replay_reached(
    kobe: Kobe,
    router_responses: list[httpx.Response],
    chat_response: Callable[..., httpx.Response],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A replay that arrives at another node continues from where it arrived."""
    kobe.graph.graph.add_node("s0_home", replay_misses=0)
    kobe.graph.graph.add_node("s1_settings", replay_misses=0)
    replayed: list[str] = []

    def replay(node_id: str) -> str:
        replayed.append(node_id)
        return "s0_home"

    planned_from: list[object] = []
    monkeypatch.setattr(kobe, "navigate_to_node", replay)
    _describing_by_node_id(kobe, monkeypatch)
    monkeypatch.setattr("android_app_graph.kobe.plan_next_action", _planner("Wait", planned_from))
    monkeypatch.setattr("android_app_graph.kobe.soft_keyboard_hint", lambda _udid: "")
    router_responses.append(chat_response(content=_WAIT_REPLY))

    kobe.explore(max_steps=1, resume_from="s1_settings")

    assert replayed == ["s1_settings"]
    assert planned_from == ["s0_home"]


@pytest.mark.usefixtures("no_sleep")
def test_auto_resume_with_only_external_nodes_starts_from_the_current_screen(
    kobe: Kobe,
    router_responses: list[httpx.Response],
    chat_response: Callable[..., httpx.Response],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kobe.graph.graph.add_node("ext_com.foreign", replay_misses=0)
    assert kobe.graph.get_resume_target(package_name="com.demo") is None

    def identify(_observation: Observation, **_kwargs: object) -> str:
        kobe.graph.graph.add_node("s0_home", replay_misses=0)
        return "s0_home"

    planned_from: list[object] = []
    monkeypatch.setattr(kobe.graph, "identify_state", identify)
    _describing_by_node_id(kobe, monkeypatch)
    monkeypatch.setattr("android_app_graph.kobe.plan_next_action", _planner("Wait", planned_from))
    monkeypatch.setattr("android_app_graph.kobe.soft_keyboard_hint", lambda _udid: "")
    router_responses.append(chat_response(content=_WAIT_REPLY))

    kobe.explore(max_steps=1, resume_from="auto")

    assert planned_from == ["s0_home"]


@pytest.mark.usefixtures("no_sleep")
def test_resuming_from_a_node_missing_from_the_graph_starts_from_the_current_screen(
    kobe: Kobe,
    router_responses: list[httpx.Response],
    chat_response: Callable[..., httpx.Response],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    kobe.graph.graph.add_node("s0_home", replay_misses=0)
    planned_from: list[object] = []
    monkeypatch.setattr(kobe.graph, "identify_state", lambda *_args, **_kwargs: "s0_home")
    _describing_by_node_id(kobe, monkeypatch)
    monkeypatch.setattr("android_app_graph.kobe.plan_next_action", _planner("Wait", planned_from))
    monkeypatch.setattr("android_app_graph.kobe.soft_keyboard_hint", lambda _udid: "")
    router_responses.append(chat_response(content=_WAIT_REPLY))

    kobe.explore(max_steps=1, resume_from="s9_removed")

    assert "Node s9_removed not found in graph. Starting from current screen." in caplog.text
    assert planned_from == ["s0_home"]


@pytest.mark.parametrize("interruption", [RuntimeError, KeyboardInterrupt])
def test_a_resumed_exploration_that_stops_saves_and_logs_the_steps_it_completed(
    interruption: type[BaseException],
    kobe: Kobe,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A step that raises is not counted, so a re-run repeats it."""
    kobe.graph.graph.add_node("s0_home", replay_misses=0)
    kobe.graph.total_steps_completed = 3

    def stop_planning(**_kwargs: object) -> tuple[str, int | None]:
        msg = "planning stopped"
        raise interruption(msg)

    monkeypatch.setattr(kobe.graph, "identify_state", lambda *_args, **_kwargs: "s0_home")
    _describing_by_node_id(kobe, monkeypatch)
    monkeypatch.setattr("android_app_graph.kobe.plan_next_action", stop_planning)
    monkeypatch.setattr("android_app_graph.kobe.soft_keyboard_hint", lambda _udid: "")
    caplog.set_level(logging.INFO, logger="test")

    with pytest.raises(interruption, match="planning stopped"):
        kobe.explore(max_steps=5)

    assert "Resuming exploration from step 3 (target: 5 steps total)" in caplog.text
    assert json.loads(kobe.graph_path.read_text())["total_steps_completed"] == 3
    assert "at step 3" in caplog.text


@pytest.mark.parametrize("resume_from", ["auto", "s0_home"])
def test_a_run_resumed_from_a_node_logs_the_step_it_starts_from_before_the_replay(
    resume_from: str,
    kobe: Kobe,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    kobe.graph.graph.add_node("s0_home", package="com.demo", replay_misses=0)
    kobe.graph.total_steps_completed = 3
    logged_before_the_replay: list[str] = []

    def replay(_node_id: str) -> str:
        logged_before_the_replay.extend(caplog.messages)
        raise _ReplayedError

    monkeypatch.setattr(kobe, "navigate_to_node", replay)
    caplog.set_level(logging.INFO, logger="test")

    with pytest.raises(_ReplayedError):
        kobe.explore(max_steps=5, resume_from=resume_from)

    assert "Resuming exploration from step 3 (target: 5 steps total)" in logged_before_the_replay


@pytest.mark.parametrize(
    ("detail_after", "delta", "observation"),
    [
        ({"sort": "price"}, {"sort": {"before": "name", "after": "price"}}, "s0_home (sort=price)"),
        ({"sort": None}, {"sort": {"before": "name", "after": None}}, "s0_home"),
    ],
)
@pytest.mark.usefixtures("no_sleep")
def test_a_step_that_stays_on_its_screen_records_the_details_it_changed(
    detail_after: dict[str, object],
    delta: dict[str, object],
    observation: str,
    tmp_path: Path,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    chat_response: Callable[..., httpx.Response],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The self-loop's target observation names the details shown after the step, leaving
    out those with no value."""
    explorer = _explorer(_Screens(HOME_SCREEN), tmp_path, router_client)
    graph = explorer.graph.graph
    graph.add_node("s0_home", last_detail_snapshot={"sort": "name"}, replay_misses=0)
    details = iter([{"sort": "name"}, detail_after])

    def identify(*_args: object, **_kwargs: object) -> str:
        graph.nodes["s0_home"]["last_detail_snapshot"] = next(details)
        return "s0_home"

    monkeypatch.setattr(explorer.graph, "identify_state", identify)
    _describing_by_node_id(explorer, monkeypatch)
    monkeypatch.setattr("android_app_graph.kobe.plan_next_action", _planner("Sort by price", []))
    monkeypatch.setattr("android_app_graph.kobe.soft_keyboard_hint", lambda _udid: "")
    router_responses.append(chat_response(content=_TAP_REPLY))

    explorer.explore(max_steps=1)

    edge = graph.edges["s0_home", "s0_home"]
    assert edge["schema_deltas"] == [delta]
    assert edge["target_observations"] == [observation]


@pytest.mark.usefixtures("no_sleep")
def test_exploration_with_coverage_checkpoints_off_sends_only_the_steps_requests(
    tmp_path: Path,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    chat_response: Callable[..., httpx.Response],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No step is followed by a merge audit or a choice of where to continue."""
    explorer = _explorer(_Screens(HOME_SCREEN), tmp_path, router_client)
    explorer.coverage_checkpoint_steps = 0
    explorer.graph.graph.add_node("s0_home", replay_misses=0)
    monkeypatch.setattr(explorer.graph, "identify_state", lambda *_args, **_kwargs: "s0_home")
    _describing_by_node_id(explorer, monkeypatch)
    monkeypatch.setattr("android_app_graph.kobe.soft_keyboard_hint", lambda _udid: "")
    for _ in range(2):
        router_responses.extend(
            [
                chat_response(
                    content='{"element":null,"instruction":"Tap Sort","required_pointers":1}'
                ),
                chat_response(content=_TAP_REPLY),
            ]
        )

    explorer.explore(max_steps=2)

    assert [json.loads(request.content)["model"] for request in router_requests] == [
        "provider/instruction",
        "provider/action",
    ] * 2


def test_coverage_checkpoint_asks_the_page_detail_model_where_to_continue(
    kobe: Kobe,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    chat_response: Callable[..., httpx.Response],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kobe.graph.graph.add_nodes_from(["s0_home", "s1_settings"], replay_misses=0)
    kobe.graph.add_edge("s0_home", "s1_settings", {"action": "tap"})
    router_responses.extend(
        [
            chat_response(content='{"node_id":"s1_settings","reason":"unexplored"}'),
        ]
    )
    monkeypatch.setattr(kobe, "navigate_to_node", lambda node: node)
    assert kobe.maybe_continue_from_coverage_checkpoint(50, 100, "s0_home") == "s1_settings"
    assert [json.loads(request.content)["model"] for request in router_requests] == [
        "provider/page_detail"
    ]


def test_coverage_checkpoint_offers_the_start_screen_ranked_below_its_top_k(
    kobe: Kobe,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    chat_response: Callable[..., httpx.Response],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Three forms with more unexplored fields outrank the start screen, which the model
    is offered anyway while it has unexplored sections."""
    graph = kobe.graph.graph
    graph.add_node(
        "s0_home",
        visit_count=5,
        interactable_elements=[
            {"description": description, "explored": False}
            for description in ("Display", "Storage")
        ],
        replay_misses=0,
    )
    for node_id in ("s1_form", "s2_form", "s3_form"):
        graph.add_node(
            node_id,
            interactable_elements=[
                {"description": description, "explored": False}
                for description in ("Name", "Address", "Port", "Username", "Password")
            ],
            replay_misses=0,
        )
        kobe.graph.add_edge("s0_home", node_id, {"action": "tap"})
    kobe.coverage_checkpoint_top_k = 2
    router_responses.extend(
        [
            chat_response(content='{"node_id":"s0_home","reason":"top-level sections"}'),
        ]
    )
    monkeypatch.setattr(kobe, "navigate_to_node", lambda node: node)

    assert kobe.maybe_continue_from_coverage_checkpoint(50, 100, "s1_form") == "s0_home"

    bodies = [json.loads(request.content) for request in router_requests]
    assert [body["response_format"]["json_schema"]["name"] for body in bodies] == [
        "exploration_target",
    ]
    assert "node_id=s0_home" in bodies[0]["messages"][0]["content"]


@pytest.mark.usefixtures("no_sleep")
def test_coverage_checkpoint_restarts_the_app_before_replaying_to_its_target(
    replaying_kobe: Kobe,
    router_responses: list[httpx.Response],
    chat_response: Callable[..., httpx.Response],
    device_log: list[object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The replay from the start node runs on a restarted app, not whatever screen is showing."""
    open_settings = {"action": "tap", "coordinate": [10, 20]}
    open_detail = {"action": "tap", "coordinate": [30, 40]}
    replaying_kobe.graph.graph.add_nodes_from(
        ["s0_home", "s1_settings", "s2_detail"], replay_misses=0
    )
    replaying_kobe.graph.add_edge("s0_home", "s1_settings", action=[open_settings])
    replaying_kobe.graph.add_edge("s1_settings", "s2_detail", action=[open_detail])
    router_responses.extend(
        [
            chat_response(content='{"node_id":"s1_settings","reason":"unexplored"}'),
        ]
    )
    _identifying(replaying_kobe, monkeypatch, "s0_home", "s1_settings")
    assert (
        replaying_kobe.maybe_continue_from_coverage_checkpoint(50, 100, "s2_detail")
        == "s1_settings"
    )
    assert device_log == [*_FIRST_RESTART, open_settings]


def _fail_navigation(node: str) -> str:
    msg = f"No path from s0_list to {node} in the graph."
    raise NoPathError(msg)


@pytest.mark.usefixtures("no_sleep")
@pytest.mark.parametrize("navigation_fails", [False, True])
def test_a_checkpoint_that_merges_away_the_current_node_continues_from_the_kept_node(
    navigation_fails: bool,
    replaying_kobe: Kobe,
    router_responses: list[httpx.Response],
    chat_response: Callable[..., httpx.Response],
    monkeypatch: pytest.MonkeyPatch,
    add_member: Callable[..., None],
    embedding_body: dict[str, object],
) -> None:
    """A checkpoint that does not navigate hands back the ID its merge audit kept, not the
    one it removed, so the next step acts from a node still in the graph."""
    graph = replaying_kobe.graph
    for node_id in ("s0_list", "s1_list_dup"):
        graph.graph.add_node(node_id, replay_misses=0)
        add_member(graph, node_id)
        router_responses.append(httpx.Response(200, json=embedding_body))
    graph.embed_missing_members()
    graph.graph.add_node("s2_detail", replay_misses=0)
    screens = iter(["s1_list_dup", "s1_list_dup", "s2_detail"])
    monkeypatch.setattr(graph, "identify_state", lambda *_args, **_kwargs: next(screens))
    _describing_by_node_id(replaying_kobe, monkeypatch)
    replaying_kobe.coverage_checkpoint_steps = 1
    if navigation_fails:
        graph.add_edge("s1_list_dup", "s2_detail", {"action": "tap"})
        monkeypatch.setattr(replaying_kobe, "navigate_to_node", _fail_navigation)
    duplicate = {"type": "merge_nodes", "node_a": "n0", "node_b": "n1"}
    target = '{"node_id":"s2_detail","reason":"unexplored"}'
    click = '{"action":"click","coordinate":[500,500],"required_pointers":1}'
    router_responses.extend(
        [
            chat_response(
                content='{"element":null,"instruction":"Tap the first item","required_pointers":1}'
            ),
            chat_response(content="", tool_arguments=click),
            chat_response(content=json.dumps({"issues": [duplicate], "summary": "duplicate"})),
            chat_response(content='{"same":true,"reason":"same list"}'),
            *([chat_response(content=target)] if navigation_fails else []),
            chat_response(
                content='{"element":null,"instruction":"Open the first item","required_pointers":1}'
            ),
            chat_response(content="", tool_arguments=click),
        ]
    )

    replaying_kobe.explore(max_steps=2)

    assert graph.graph["s0_list"]["s2_detail"]["instructions"][-1] == "Open the first item"


def test_a_checkpoint_merge_keeps_the_screen_the_device_is_on_out_of_the_candidates(
    kobe: Kobe,
    router_responses: list[httpx.Response],
    chat_response: Callable[..., httpx.Response],
    monkeypatch: pytest.MonkeyPatch,
    add_member: Callable[..., None],
    embedding_body: dict[str, object],
) -> None:
    """The checkpoint leaves the node the device is on out of its candidates, so it never
    replays the device to the screen it is already on."""
    for node_id in ("s1_list", "s2_list_dup"):
        kobe.graph.graph.add_node(node_id, replay_misses=0)
        add_member(kobe.graph, node_id)
        router_responses.append(httpx.Response(200, json=embedding_body))
    kobe.graph.embed_missing_members()
    kobe.graph.graph.add_node("s0_home", replay_misses=0)
    kobe.graph.add_edge("s0_home", "s1_list", {"action": "tap"})
    duplicate = {"type": "merge_nodes", "node_a": "n0", "node_b": "n1"}
    router_responses.extend(
        [
            chat_response(content=json.dumps({"issues": [duplicate], "summary": "duplicate"})),
            chat_response(content='{"same":true,"reason":"same list"}'),
            chat_response(content='{"node_id":"s1_list","reason":"unexplored"}'),
        ]
    )
    monkeypatch.setattr(kobe, "navigate_to_node", lambda node: node)
    _describing_by_node_id(kobe, monkeypatch)
    assert kobe.maybe_continue_from_coverage_checkpoint(50, 100, "s2_list_dup") == "s0_home"


class _ReplayScript:
    """A DeviceController that shows ``screens`` one state read at a time, logs its
    actions to the device log and fails ``broken`` as an adb error would."""

    w = 100
    h = 200

    def __init__(
        self,
        log: list[object],
        *screens: dict[str, object],
        broken: dict[str, object] | None = None,
    ) -> None:
        self.config: dict[str, object] = {"device": {"udid": "test-device"}}
        self._log = log
        self._screens = iter(screens)
        self._broken = broken

    def get_state(self, **_kwargs: float | None) -> dict[str, object]:
        return next(self._screens)

    def exe_action(self, action: dict[str, object]) -> None:
        self._log.append(action)
        if action == self._broken:
            msg = f"adb could not run {action}"
            raise RuntimeError(msg)


def _adb_refusing_once(
    log: list[object], *refused: list[str]
) -> Callable[..., subprocess.CompletedProcess[str]]:
    """A ``subprocess.run`` that logs each command and fails each of ``refused`` once."""

    def run(
        command: list[str], *, check: bool = False, **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        log.append(command)
        returncode = 1 if command in refused and log.count(command) == 1 else 0
        if check and returncode:
            raise subprocess.CalledProcessError(returncode, command)
        return subprocess.CompletedProcess(command, returncode, stdout="", stderr="")

    return run


def _in_app(activity: str, color: str) -> dict[str, object]:
    return {**_screen(f"com.demo.{activity}", color), "package": "com.demo"}


@pytest.mark.usefixtures("no_sleep")
def test_a_checkpoint_whose_replay_fails_continues_from_the_screen_the_device_is_on(
    tmp_path: Path,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    chat_response: Callable[..., httpx.Response],
    device_log: list[object],
    monkeypatch: pytest.MonkeyPatch,
    step_records: Callable[[Path], list[dict[str, Any]]],
) -> None:
    """A replay that fails after the app restart continues from the screen the device is
    on: the next step plans from that screen's node and screenshot, and its setup is
    unknown. The step log tells the replay's actions from the checkpoint's own screen
    reads."""
    monkeypatch.setattr("android_app_graph.kobe.soft_keyboard_hint", lambda _udid: "")
    open_list = {"action": "tap", "coordinate": [10, 20]}
    open_item: dict[str, object] = {"action": "tap", "coordinate": [30, 40]}
    home, settings = _in_app("Home", "white"), _in_app("Settings", "gray")
    item_list, sort_menu = _in_app("List", "blue"), _in_app("Sort", "green")
    device = _ReplayScript(
        device_log,
        home,
        settings,
        home,
        item_list,
        item_list,
        item_list,
        sort_menu,
        broken=open_item,
    )
    explorer = _explorer(device, tmp_path, router_client)
    explorer.coverage_checkpoint_steps = 1
    graph = explorer.graph
    nodes = {
        "com.demo.Home": "s0_home",
        "com.demo.List": "s1_list",
        "com.demo.Settings": "s3_settings",
        "com.demo.Sort": "s4_sort_menu",
    }
    graph.graph.add_nodes_from([*nodes.values(), "s2_detail"], replay_misses=0)
    graph.add_edge("s0_home", "s1_list", action=[open_list])
    graph.add_edge("s1_list", "s2_detail", action=[open_item])
    monkeypatch.setattr(
        graph, "identify_state", lambda observation, **_kwargs: nodes[observation.activity]
    )
    _describing_by_node_id(explorer, monkeypatch)
    router_responses.extend(
        [
            chat_response(
                content='{"element":null,"instruction":"Open settings","required_pointers":1}'
            ),
            chat_response(content=_TAP_REPLY),
            chat_response(content='{"node_id":"s2_detail","reason":"unexplored"}'),
            chat_response(
                content='{"element":null,"instruction":"Open the sort menu","required_pointers":1}'
            ),
            chat_response(content=_TAP_REPLY),
        ]
    )

    explorer.explore(max_steps=2)

    assert list(graph.graph.in_edges("s4_sort_menu")) == [("s1_list", "s4_sort_menu")]
    assert graph.graph["s1_list"]["s4_sort_menu"]["setups"] == [None]
    list_webp = base64.b64encode(encode_screenshot(str(item_list["screenshot"]))).decode("ascii")
    list_image = f"data:image/webp;base64,{list_webp}"
    assert all(list_image in request.content.decode() for request in router_requests[-2:])
    tap = {"action": "tap", "x": 50, "y": 100}
    assert device_log == [tap, *_FIRST_RESTART, open_list, open_item, tap]
    step = [
        ("action", "main_loop"),
        ("observation", "main_loop"),
        ("match", "main_loop"),
        ("transition", "main_loop"),
    ]
    assert [(r["kind"], r["source"]) for r in step_records(explorer.controller.steps)] == [
        ("observation", "main_loop"),
        ("match", "main_loop"),
        *step,
        ("system_event", "navigation_replay"),
        ("observation", "navigation_replay"),
        ("match", "navigation_replay"),
        ("action", "navigation_replay"),
        ("observation", "navigation_replay"),
        ("match", "navigation_replay"),
        ("action", "navigation_replay"),
        ("observation", "coverage_checkpoint"),
        ("match", "coverage_checkpoint"),
        ("observation", "coverage_checkpoint"),
        *step,
    ]


@pytest.mark.usefixtures("no_sleep")
def _settings_and_detail(explorer: Kobe) -> None:
    """Give ``explorer`` the path home → settings → detail, one tap each."""
    explorer.graph.graph.add_nodes_from(["s0_home", "s1_settings", "s2_detail"], replay_misses=0)
    explorer.graph.add_edge("s0_home", "s1_settings", action=[{"action": "tap", "x": 1, "y": 2}])
    explorer.graph.add_edge("s1_settings", "s2_detail", action=[{"action": "tap", "x": 3, "y": 4}])


def test_a_checkpoint_whose_launch_fails_restarts_the_app_once_more_and_continues_there(
    tmp_path: Path,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    chat_response: Callable[..., httpx.Response],
    device_log: list[object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A launch that fails after the force-stop leaves the launcher on the screen."""
    launcher = {
        **_screen("com.android.launcher3.Launcher", "black"),
        "package": "com.android.launcher3",
    }
    device = _ReplayScript(device_log, launcher, _in_app("Home", "white"))
    explorer = _explorer(device, tmp_path, router_client, launch_activity="com.demo/.Main")
    _settings_and_detail(explorer)
    router_responses.extend(
        [
            chat_response(content='{"node_id":"s1_settings","reason":"unexplored"}'),
        ]
    )
    main_start = _clear_task_start("com.demo/.Main")
    monkeypatch.setattr(subprocess, "run", _adb_refusing_once(device_log, main_start))
    nodes = {"com.demo.Home": "s0_home"}
    monkeypatch.setattr(
        explorer.graph, "identify_state", lambda observation, **_kwargs: nodes[observation.activity]
    )

    assert explorer.maybe_continue_from_coverage_checkpoint(50, 100, "s2_detail") == "s0_home"
    assert device_log == [_FORCE_STOP, main_start, _FORCE_STOP, main_start]


@pytest.mark.usefixtures("no_sleep")
def test_a_checkpoint_whose_replay_fails_on_a_web_page_presses_back_off_it(
    tmp_path: Path,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    chat_response: Callable[..., httpx.Response],
    device_log: list[object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The next step never starts from ``ext_web``, even when the app's own web view
    shows the outside page."""
    open_list = {"action": "tap", "coordinate": [10, 20]}
    open_item: dict[str, object] = {"action": "tap", "coordinate": [30, 40]}
    web_page, item_list = _in_app("Article", "orange"), _in_app("List", "blue")
    device = _ReplayScript(
        device_log, _in_app("Home", "white"), web_page, web_page, item_list, broken=open_item
    )
    explorer = _explorer(device, tmp_path, router_client)
    explorer.graph.graph.add_nodes_from(["s0_home", "s1_list", "s2_detail"], replay_misses=0)
    explorer.graph.add_edge("s0_home", "s1_list", action=[open_list])
    explorer.graph.add_edge("s1_list", "s2_detail", action=[open_item])
    router_responses.extend(
        [
            chat_response(content='{"node_id":"s2_detail","reason":"unexplored"}'),
        ]
    )
    nodes = {
        "com.demo.Home": "s0_home",
        "com.demo.Article": EXTERNAL_WEB_NODE,
        "com.demo.List": "s1_list",
    }
    monkeypatch.setattr(
        explorer.graph, "identify_state", lambda observation, **_kwargs: nodes[observation.activity]
    )

    assert explorer.maybe_continue_from_coverage_checkpoint(50, 100, "s0_home") == "s1_list"
    back = {"action": "back"}
    assert device_log == [*_FIRST_RESTART, open_list, open_item, back]


@pytest.mark.usefixtures("no_sleep")
@pytest.mark.parametrize("force_stop_refused", [False, True], ids=["no path", "force-stop refused"])
def test_a_checkpoint_that_fails_before_the_restart_keeps_the_steps_node_and_state(
    force_stop_refused: bool,
    tmp_path: Path,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    chat_response: Callable[..., httpx.Response],
    device_log: list[object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Neither failure changes the screen, so the step's node and state still hold."""
    explorer = _explorer(_ReplayScript(device_log), tmp_path, router_client)
    _settings_and_detail(explorer)
    router_responses.extend(
        [
            chat_response(content='{"node_id":"s1_settings","reason":"unexplored"}'),
        ]
    )
    if force_stop_refused:
        monkeypatch.setattr(subprocess, "run", _adb_refusing_once(device_log, _FORCE_STOP))
    else:
        monkeypatch.setattr(explorer.graph, "find_path", lambda *_args: None)

    assert explorer.maybe_continue_from_coverage_checkpoint(50, 100, "s2_detail") is None
    assert device_log == ([_FORCE_STOP] if force_stop_refused else [])


_DIALOG_ACTIVITY = (
    "com.google.android.permissioncontroller/"
    "com.android.permissioncontroller.permission.ui.GrantPermissionsActivity"
)


def _explore_into_a_permission_dialog(
    kobe: Kobe,
    png_screenshot: str,
    router_responses: list[httpx.Response],
    chat_response: Callable[..., httpx.Response],
    embedding_body: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> _ScriptedDevice:
    monkeypatch.setattr("android_app_graph.kobe.soft_keyboard_hint", lambda _udid: "")
    home: dict[str, object] = {
        "activity": "com.demo.Home",
        "screenshot": png_screenshot,
        "package": "com.demo",
    }
    dialog: dict[str, object] = {
        "activity": _DIALOG_ACTIVITY,
        "screenshot": png_screenshot,
        "package": "com.google.android.permissioncontroller",
        "system_surface": "permission",
        "permission_requester": "com.demo",
        "xml": '<hierarchy><node resource-id="com.android.permissioncontroller:id/permission_allow_button" enabled="true" bounds="[0,0][100,100]"/></hierarchy>',
    }
    device = _ScriptedDevice(home, dialog, home)
    monkeypatch.setattr(
        "android_app_graph.device.grant_runtime_permissions",
        lambda *_args: PermissionGrants(
            ("android.permission.CAMERA",), ("android.permission.CAMERA",)
        ),
    )
    monkeypatch.setattr(
        "android_app_graph.device.run_adb",
        lambda args, **_kwargs: subprocess.CompletedProcess(
            args,
            0,
            "  Permission [android.permission.CAMERA] (1):\n    sourcePackage=android prot=dangerous\n",
            "",
        ),
    )
    monkeypatch.setattr(kobe, "controller", _recording(device, kobe))
    router_responses.extend(
        [
            httpx.Response(200, json=embedding_body),
            chat_response(
                content='{"page_description": "home", "surface": {"kind": "page", "relation": "unknown", "return_depth": 0}}'
            ),
            httpx.Response(200, json=embedding_body),
            chat_response(
                content='{"element":null,"instruction":"Tap Take photo","required_pointers":1}'
            ),
            chat_response(
                content='{"action":"click","coordinate":[500,500],"required_pointers":1}'
            ),
            httpx.Response(200, json=embedding_body),
        ]
    )

    kobe.explore(max_steps=1)
    return device


@pytest.mark.usefixtures("no_sleep")
def test_a_target_permission_is_resolved_without_app_coverage(
    kobe: Kobe,
    png_screenshot: str,
    router_responses: list[httpx.Response],
    chat_response: Callable[..., httpx.Response],
    embedding_body: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The existing target grant policy resolves the prompt without a denial."""
    device = _explore_into_a_permission_dialog(
        kobe, png_screenshot, router_responses, chat_response, embedding_body, monkeypatch
    )

    tap = {"action": "tap", "x": 16, "y": 16}
    assert device.actions == [tap, {"action": "tap", "x": 50, "y": 50}]
    graph = kobe.graph.graph
    nodes = {data["activity"]: node for node, data in graph.nodes(data=True)}
    assert set(nodes) == {"com.demo.Home", _DIALOG_ACTIVITY}
    dialog_node = graph.nodes[nodes[_DIALOG_ACTIVITY]]
    assert dialog_node["is_external"] is True
    assert dialog_node["system_surface"] == "permission"
    edge = graph.edges[nodes["com.demo.Home"], nodes[_DIALOG_ACTIVITY]]
    assert edge["actions"] == [[tap]]


@pytest.mark.usefixtures("no_sleep")
def test_a_permission_boundary_is_excluded_from_all_app_targets(
    kobe: Kobe,
    png_screenshot: str,
    router_requests: list[httpx.Request],
    router_responses: list[httpx.Response],
    chat_response: Callable[..., httpx.Response],
    embedding_body: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A replay from the start screen cannot land on a dialog. The dialog node's own
    package keeps it out of both targets although it would rank first."""
    _explore_into_a_permission_dialog(
        kobe, png_screenshot, router_responses, chat_response, embedding_body, monkeypatch
    )

    graph = kobe.graph
    nodes = {data.get("package"): node for node, data in graph.graph.nodes(data=True)}
    assert set(nodes) == {"com.demo", None}
    home = nodes["com.demo"]
    assert graph.get_resume_target() == home
    assert graph.get_resume_target(package_name="com.demo") == home
    checkpoint = graph.get_exploration_target_candidates(package_name="com.demo")
    assert [candidate["node_id"] for candidate in checkpoint] == [home]
    described_without_known_screens = [
        request
        for request in router_requests
        if b"No known-screen descriptions are supplied" in request.content
    ]
    assert len(described_without_known_screens) == 1


_SEARCH = "com.google.android.settings.intelligence"
_SEARCH_SCREEN = {**_screen(f"{_SEARCH}.SearchActivity", "orange"), "package": _SEARCH}
_TAP = {"action": "tap", "x": 50, "y": 100}


def _explore_from_home_into_another_app(
    explorer: Kobe,
    router_responses: list[httpx.Response],
    chat_response: Callable[..., httpx.Response],
    embedding_body: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    *identifying: httpx.Response,
    max_steps: int = 1,
) -> list[object]:
    """Explore with a first step that taps from home into another app and steps that wait
    after it. ``identifying`` answers the model requests that identifying the screen the
    first step ends on makes. Returns the page description each step planned from."""
    planned_from: list[object] = []
    monkeypatch.setattr("android_app_graph.kobe.plan_next_action", _planner("Search", planned_from))
    monkeypatch.setattr("android_app_graph.kobe.soft_keyboard_hint", lambda _udid: "")
    router_responses.extend(
        [
            httpx.Response(200, json=embedding_body),
            chat_response(
                content='{"page_description": "home", "surface": {"kind": "page", "relation": "unknown", "return_depth": 0}}'
            ),
            httpx.Response(200, json=embedding_body),
            chat_response(content=_TAP_REPLY),
            *identifying,
            *[chat_response(content=_WAIT_REPLY) for _ in range(max_steps - 1)],
        ]
    )
    explorer.explore(max_steps=max_steps)
    return planned_from


def _node_packages(explorer: Kobe) -> dict[str, object]:
    return {node: data.get("package") for node, data in explorer.graph.graph.nodes(data=True)}


@pytest.mark.parametrize(
    ("after_the_tap", "leaving"),
    [
        ([_SEARCH_SCREEN, HOME_SCREEN], [_BACK]),
        (
            [*[_SEARCH_SCREEN] * 4, HOME_SCREEN],
            [_BACK, *_FIRST_RESTART],
        ),
    ],
    ids=["back returns", "back leaves the other app in front"],
)
@pytest.mark.usefixtures("no_sleep")
def test_a_step_into_another_app_returns_to_the_app_before_the_next_step(
    after_the_tap: list[dict[str, object]],
    leaving: list[object],
    tmp_path: Path,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    chat_response: Callable[..., httpx.Response],
    embedding_body: dict[str, object],
    device_log: list[object],
    screens_by_log: Callable[..., DeviceController],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Back can close only the keyboard and leave the other app's screen up. Launching
    the app would then resume its task with that screen still on top, so the explorer
    restarts the app with its task cleared, and gives the other app's screen no node."""
    explorer = _explorer(screens_by_log(HOME_SCREEN, *after_the_tap), tmp_path, router_client)

    planned_from = _explore_from_home_into_another_app(
        explorer,
        router_responses,
        chat_response,
        embedding_body,
        monkeypatch,
        # Home again, before or after a restart: an exact match.
        httpx.Response(200, json=embedding_body),
        max_steps=2,
    )

    assert device_log == [_TAP, *leaving]
    assert planned_from == ["home", "home"]
    home_webp = base64.b64encode(encode_screenshot(str(HOME_SCREEN["screenshot"])))
    home_image = f"data:image/webp;base64,{home_webp.decode('ascii')}"
    assert home_image in router_requests[-1].content.decode()
    assert _node_packages(explorer) == {"s0": "com.demo", f"ext_{_SEARCH}": None}


@pytest.mark.parametrize(
    ("after_the_tap", "reached"),
    [
        pytest.param(_SEARCH_SCREEN, f"ext_{_SEARCH}", id="another app"),
        pytest.param(
            _screen("com.demo.ArticleActivity", "blue"), EXTERNAL_WEB_NODE, id="outside web page"
        ),
    ],
)
@pytest.mark.usefixtures("no_sleep")
def test_a_step_that_leaves_the_app_marks_the_element_the_planner_names(
    after_the_tap: dict[str, object],
    reached: str,
    tmp_path: Path,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    chat_response: Callable[..., httpx.Response],
    embedding_body: dict[str, object],
    screens_by_log: Callable[..., DeviceController],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The element is explored as after a step that stays in the app, so the planner is
    not offered it again and the coverage ranking no longer counts it."""
    monkeypatch.setattr("android_app_graph.kobe.soft_keyboard_hint", lambda _udid: "")
    explorer = _explorer(
        screens_by_log(HOME_SCREEN, after_the_tap, HOME_SCREEN), tmp_path, router_client
    )
    home = {
        "page_description": "home",
        "elements": [
            {"description": "Dark mode toggle", "position": "top"},
            {"description": "Search button", "position": "top right"},
        ],
        "surface": {"kind": "page", "relation": "unknown", "return_depth": 0},
    }
    identifying_the_web_page = [
        httpx.Response(200, json=embedding_body),
        chat_response(
            content='{"page_description": "news article", "is_external_web": true, "surface": {"kind": "page", "relation": "unknown", "return_depth": 0}}'
        ),
    ]
    router_responses.extend(
        [
            httpx.Response(200, json=embedding_body),
            chat_response(content=json.dumps(home)),
            httpx.Response(200, json=embedding_body),
            chat_response(
                content='{"element":2,"instruction":"Tap the Search button","required_pointers":1}'
            ),
            chat_response(content=_TAP_REPLY),
            *(identifying_the_web_page if reached == EXTERNAL_WEB_NODE else []),
            # Back shows home again: an exact match.
            httpx.Response(200, json=embedding_body),
        ]
    )

    explorer.explore(max_steps=1)

    assert list(explorer.graph.graph.edges) == [("s0", reached)]
    elements = explorer.graph.graph.nodes["s0"]["interactable_elements"]
    assert [e["description"] for e in elements if e["explored"]] == ["Search button"]


@pytest.mark.usefixtures("no_sleep")
def test_a_keyboard_in_front_after_the_back_press_counts_as_the_app(
    tmp_path: Path,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    chat_response: Callable[..., httpx.Response],
    embedding_body: dict[str, object],
    device_log: list[object],
    screens_by_log: Callable[..., DeviceController],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    keyboard = {**_screen("com.demo.Home", "gray"), "package": "com.demo"}
    explorer = _explorer(
        screens_by_log(HOME_SCREEN, _SEARCH_SCREEN, keyboard), tmp_path, router_client
    )

    _explore_from_home_into_another_app(
        explorer,
        router_responses,
        chat_response,
        embedding_body,
        monkeypatch,
        httpx.Response(200, json=embedding_body),
        chat_response(
            content='{"page_description": "home with the keyboard shown", "surface": {"kind": "page", "relation": "unknown", "return_depth": 0}}'
        ),
        httpx.Response(200, json=embedding_body),
        chat_response(content='{"same":true}'),
    )

    assert device_log == [_TAP, _BACK]
    assert _node_packages(explorer) == {
        "s0": "com.demo",
        f"ext_{_SEARCH}": None,
    }


@pytest.mark.usefixtures("no_sleep")
def test_another_app_still_in_front_after_the_restart_stops_exploration_naming_it(
    tmp_path: Path,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    chat_response: Callable[..., httpx.Response],
    embedding_body: dict[str, object],
    device_log: list[object],
    screens_by_log: Callable[..., DeviceController],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    explorer = _explorer(screens_by_log(HOME_SCREEN, _SEARCH_SCREEN), tmp_path, router_client)

    with pytest.raises(StillOutsideAppError, match=_SEARCH):
        _explore_from_home_into_another_app(
            explorer, router_responses, chat_response, embedding_body, monkeypatch
        )

    assert device_log == [_TAP, _BACK, *_FIRST_RESTART]
    assert _node_packages(explorer) == {"s0": "com.demo", f"ext_{_SEARCH}": None}


@pytest.mark.usefixtures("no_sleep")
def test_a_replay_that_ends_in_another_app_restarts_the_app_instead_of_identifying_it(
    tmp_path: Path,
    router_client: OpenRouter,
    device_log: list[object],
    screens_by_log: Callable[..., DeviceController],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    open_detail = {"action": "tap", "coordinate": [30, 40]}
    after_the_replay = [_SEARCH_SCREEN, _SEARCH_SCREEN, HOME_SCREEN]
    device = screens_by_log(*[HOME_SCREEN] * 4, *after_the_replay)
    explorer = _explorer(device, tmp_path, router_client)
    _list_reached_from_home(explorer, "s1_detail")
    explorer.graph.add_edge("s0_home", "s1_detail", action=[open_detail])
    expected = _identifying(explorer, monkeypatch, "s0_home", "s0_home")

    assert explorer.navigate_to_node("s1_detail") == "s0_home"
    assert device_log == [*_FIRST_RESTART, open_detail, *_RESTART]
    assert expected == ["s0_home", "s0_home"]


type _Visit = str | tuple[str, Mapping[str, object]]


def _describing_by_node_id(explorer: Kobe, monkeypatch: pytest.MonkeyPatch) -> None:
    """Describe each node, and the screen last identified as it, by the node's ID, and
    each member by its observation ID, as a fake identify_state or a node added by hand
    gives no member a description."""
    monkeypatch.setattr(explorer.graph, "latest_description", lambda node_id: node_id)
    monkeypatch.setattr(explorer.graph, "node_descriptions", lambda node_id: node_id)
    monkeypatch.setattr(explorer.graph, "description", lambda observation_id: observation_id)


def _identifying(explorer: Kobe, monkeypatch: pytest.MonkeyPatch, *visits: _Visit) -> list[object]:
    """Make the explorer's identify_state answer ``visits`` in turn, each a node, or a node
    with the state the visit records as the node's latest, each screen described by its
    node's ID; return the list the node of each call's expected screenshot is appended
    to: the node answered for it, or the node that lists it."""
    expected: list[object] = []
    answers = iter(visits)
    answered: dict[str, str] = {}

    def identify(
        observation: Observation, *, expected_observation: str | None = None, **_kwargs: object
    ) -> str:
        known = answered.get(expected_observation or "")
        expected.append(known or explorer.graph.node_of(expected_observation))
        visit = next(answers)
        node, state = (visit, None) if isinstance(visit, str) else visit
        if state is not None:
            explorer.graph.graph.nodes[node]["last_detail_snapshot"] = dict(state)
        answered[observation.observation_id] = node
        return node

    monkeypatch.setattr(explorer.graph, "identify_state", identify)
    _describing_by_node_id(explorer, monkeypatch)
    return expected


def _list_reached_from_home(explorer: Kobe, *extra_nodes: str) -> None:
    for node in ("s0_home", "s1_list", *extra_nodes):
        explorer.graph.graph.add_node(
            node,
            package="com.demo",
            replay_misses=0,
            observations=[f"{node}-observation"],
        )


@pytest.mark.parametrize(
    ("answers", "expected"),
    [
        pytest.param(["s1_list"], ["s1_list"], id="back returns"),
        pytest.param(
            [EXTERNAL_WEB_NODE] * 3 + ["s0_home"],
            ["s1_list", "s1_list", "s1_list", "s0_home"],
            id="restart",
        ),
    ],
)
@pytest.mark.usefixtures("no_sleep", "device_log")
def test_leaving_a_web_page_expects_the_screen_that_opened_it_and_a_restart_the_start_node(
    answers: list[str],
    expected: list[object],
    tmp_path: Path,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    chat_response: Callable[..., httpx.Response],
    screens_by_log: Callable[..., DeviceController],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    explorer = _explorer(screens_by_log(HOME_SCREEN), tmp_path, router_client)
    _list_reached_from_home(explorer, EXTERNAL_WEB_NODE)
    identified = _identifying(explorer, monkeypatch, "s1_list", EXTERNAL_WEB_NODE, *answers)
    monkeypatch.setattr("android_app_graph.kobe.plan_next_action", _planner("Open the link", []))
    monkeypatch.setattr("android_app_graph.kobe.soft_keyboard_hint", lambda _udid: "")
    router_responses.append(chat_response(content=_TAP_REPLY))

    explorer.explore(max_steps=1)

    assert identified == [None, None, *expected]


@pytest.mark.usefixtures("no_sleep", "device_log")
def test_back_from_another_app_expects_the_screen_the_step_left(
    tmp_path: Path,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    chat_response: Callable[..., httpx.Response],
    screens_by_log: Callable[..., DeviceController],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    explorer = _explorer(
        screens_by_log(HOME_SCREEN, _SEARCH_SCREEN, HOME_SCREEN), tmp_path, router_client
    )
    _list_reached_from_home(explorer)
    identified = _identifying(explorer, monkeypatch, "s1_list", "s1_list")
    monkeypatch.setattr("android_app_graph.kobe.plan_next_action", _planner("Search", []))
    monkeypatch.setattr("android_app_graph.kobe.soft_keyboard_hint", lambda _udid: "")
    router_responses.append(chat_response(content=_TAP_REPLY))

    explorer.explore(max_steps=1)

    assert identified == [None, "s1_list"]


@pytest.mark.usefixtures("no_sleep")
def test_every_counted_step_logs_its_number_and_then_its_tokens(
    tmp_path: Path,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    chat_response: Callable[..., httpx.Response],
    screens_by_log: Callable[..., DeviceController],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A step that stays in the app, one into another app, one onto an outside web page and
    a wait each log ``Step N/M``, so counting those lines counts the steps a run took."""
    list_screen = _screen("com.demo.List", "gray")
    web_page = _screen("com.demo.ArticleActivity", "blue")
    screens = (HOME_SCREEN, list_screen, _SEARCH_SCREEN, list_screen, web_page, list_screen)
    explorer = _explorer(screens_by_log(*screens), tmp_path, router_client)
    _list_reached_from_home(explorer, EXTERNAL_WEB_NODE)
    _identifying(
        explorer, monkeypatch, "s0_home", "s1_list", "s1_list", EXTERNAL_WEB_NODE, "s1_list"
    )
    monkeypatch.setattr("android_app_graph.kobe.plan_next_action", _planner("Open it", []))
    monkeypatch.setattr("android_app_graph.kobe.soft_keyboard_hint", lambda _udid: "")
    router_responses.extend(chat_response(content=_TAP_REPLY) for _ in range(3))
    router_responses.append(chat_response(content=_WAIT_REPLY))
    caplog.set_level(logging.INFO, logger="test")

    explorer.explore(max_steps=4)

    messages = caplog.messages
    numbered = [i for i, message in enumerate(messages) if re.search(r"Step [0-9]*/", message)]
    assert [messages[i].split(" | ")[0] for i in numbered] == [
        "Step 1/4: s0_home --[tap]--> s1_list",
        f"Step 2/4: s1_list --[tap]--> ext_{_SEARCH}",
        f"Step 3/4: s1_list --[tap]--> {EXTERNAL_WEB_NODE}",
        "Step 4/4: s1_list --[wait]--> s1_list",
    ]
    assert [messages[i + 1].split(":")[0] for i in numbered] == ["  Tokens this step"] * 4


@pytest.mark.usefixtures("no_sleep", "device_log")
def test_a_replay_expects_the_node_it_replays_to(
    replaying_kobe: Kobe, monkeypatch: pytest.MonkeyPatch
) -> None:
    _list_reached_from_home(replaying_kobe)
    replaying_kobe.graph.add_edge("s0_home", "s1_list", action=[{"action": "tap", "x": 1, "y": 2}])
    identified = _identifying(replaying_kobe, monkeypatch, "s0_home", "s1_list", "s0_home")

    replaying_kobe.navigate_to_node("s1_list")
    replaying_kobe.navigate_to_node("s0_home")

    assert identified == ["s0_home", "s1_list", "s0_home"]


_BACK_REPLY = '{"action":"system_button","button":"Back","required_pointers":0}'
_OPEN_LIST = {"action": "tap", "coordinate": [10, 20]}


def _planning_back(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Make the planner choose Back on every screen; return the list the page description
    of each screen it plans from is appended to."""
    planned_from: list[object] = []
    monkeypatch.setattr(
        "android_app_graph.kobe.plan_next_action",
        _planner("Go back", planned_from, required_pointers=0),
    )
    monkeypatch.setattr("android_app_graph.kobe.soft_keyboard_hint", lambda _udid: "")
    return planned_from


def _requested_models(router_requests: list[httpx.Request]) -> list[str]:
    return [json.loads(request.content)["model"] for request in router_requests]


def _numbered_steps(messages: list[str]) -> list[str]:
    """Return the logged lines ``grep 'Step [0-9]*/'`` counts, without their graph summary."""
    return [message.split(" | ")[0] for message in messages if re.search(r"Step [0-9]*/", message)]


@pytest.mark.parametrize("saturated", [False, True], ids=["guarded-back", "saturated"])
@pytest.mark.parametrize("merge_at", [None, "arrival", "audit"])
@pytest.mark.usefixtures("no_sleep")
def test_ranked_arrival_keeps_its_due_merge_audit_and_next_local_step(
    kobe: Kobe,
    saturated: bool,
    merge_at: str | None,
    router_responses: list[httpx.Response],
    chat_response: Callable[..., httpx.Response],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The counted continuation retains its arrival through a scheduled merge audit."""
    _list_reached_from_home(kobe, "s2_alias")
    _identifying(kobe, monkeypatch, "s0_home")
    planned_from = _planning_back(monkeypatch)
    kobe.graph.total_steps_completed = 49
    if saturated:
        monkeypatch.setattr(kobe, "_self_loops_in_a_row", {"s0_home": 15})
    else:
        router_responses.append(chat_response(content=_BACK_REPLY))
    router_responses.append(chat_response(content=_WAIT_REPLY))
    choices: list[str] = []
    navigations: list[str] = []
    audits: list[int] = []

    def choose(current: str) -> str:
        choices.append(current)
        return "s1_list" if current == "s0_home" else "s0_home"

    def navigate(target: str) -> str:
        navigations.append(target)
        if merge_at == "arrival":
            assert kobe.graph.merge_nodes("s2_alias", "s1_list")
            return "s2_alias"
        return target

    def audit(*, app_name: str) -> dict[str, int]:
        assert app_name == "demo"
        audits.append(kobe.graph.total_steps_completed)
        if merge_at == "audit":
            assert kobe.graph.merge_nodes("s2_alias", "s1_list")
        return {"merged_count": int(merge_at == "audit")}

    monkeypatch.setattr(kobe, "_choose_continuation_target", choose)
    monkeypatch.setattr(kobe, "navigate_to_node", navigate)
    monkeypatch.setattr(kobe.graph, "run_node_merge_audit", audit)
    caplog.set_level(logging.INFO, logger="test")

    kobe.explore(max_steps=51)

    arrival = "s2_alias" if merge_at else "s1_list"
    assert navigations == ["s1_list"]
    assert choices == ["s0_home"]
    assert audits == [50]
    assert planned_from == ([] if saturated else ["s0_home"]) + [arrival]
    assert _numbered_steps(caplog.messages) == [
        f"Step 50/51: s0_home --[continue]--> {'s2_alias' if merge_at == 'arrival' else 's1_list'}",
        f"Step 51/51: {arrival} --[wait]--> {arrival}",
    ]
    saved = json.loads(kobe.graph_path.read_text())
    assert saved["total_steps_completed"] == 51


@pytest.mark.parametrize("arrival", [None, "s2_other"], ids=["unreachable", "wrong-arrival"])
@pytest.mark.usefixtures("no_sleep")
def test_unsuccessful_ranked_arrival_keeps_ordinary_checkpoint_relocation(
    kobe: Kobe,
    arrival: str | None,
    router_responses: list[httpx.Response],
    chat_response: Callable[..., httpx.Response],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An unsuccessful continuation leaves the due checkpoint free to select a target."""
    _list_reached_from_home(kobe, "s2_other")
    _identifying(kobe, monkeypatch, "s0_home")
    planned_from = _planning_back(monkeypatch)
    kobe.graph.total_steps_completed = 49
    monkeypatch.setattr(kobe, "_self_loops_in_a_row", {"s0_home": 15})
    router_responses.append(chat_response(content=_WAIT_REPLY))
    choices: list[str] = []
    navigations: list[str] = []
    audits: list[int] = []

    def choose(current: str) -> str:
        choices.append(current)
        return "s1_list" if current == "s0_home" else "s0_home"

    def navigate(target: str) -> str:
        navigations.append(target)
        if len(navigations) == 1:
            if arrival is None:
                raise NoPathError(target)
            return arrival
        return target

    def audit(*, app_name: str) -> dict[str, int]:
        assert app_name == "demo"
        audits.append(kobe.graph.total_steps_completed)
        return {"merged_count": 0}

    monkeypatch.setattr(kobe, "_choose_continuation_target", choose)
    monkeypatch.setattr(kobe, "navigate_to_node", navigate)
    monkeypatch.setattr(kobe.graph, "run_node_merge_audit", audit)
    caplog.set_level(logging.INFO, logger="test")

    kobe.explore(max_steps=51)

    assert choices == ["s0_home", arrival or "s0_home"]
    assert navigations == ["s1_list", "s0_home" if arrival else "s1_list"]
    assert audits == [50]
    assert planned_from == [navigations[-1]]
    assert len(_numbered_steps(caplog.messages)) == 2
    assert kobe.graph.total_steps_completed == 51


@pytest.mark.parametrize(
    ("interval", "maximum", "start"),
    [(0, 51, 49), (50, 50, 49), (50, 50, 48)],
    ids=["disabled", "final-step", "off-boundary"],
)
@pytest.mark.usefixtures("no_sleep")
def test_ranked_arrival_preserves_checkpoint_and_step_limits(
    kobe: Kobe,
    interval: int,
    maximum: int,
    start: int,
    router_responses: list[httpx.Response],
    chat_response: Callable[..., httpx.Response],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An arrival audits only at an enabled boundary before the final step."""
    _list_reached_from_home(kobe)
    _identifying(kobe, monkeypatch, "s0_home")
    planned_from = _planning_back(monkeypatch)
    kobe.graph.total_steps_completed = start
    kobe.coverage_checkpoint_steps = interval
    monkeypatch.setattr(kobe, "_self_loops_in_a_row", {"s0_home": 15})
    monkeypatch.setattr(kobe, "_choose_continuation_target", lambda _current: "s1_list")
    monkeypatch.setattr(kobe, "navigate_to_node", lambda target: target)
    router_responses.append(chat_response(content=_WAIT_REPLY))
    audits: list[str] = []

    def audit(*, app_name: str) -> dict[str, int]:
        audits.append(app_name)
        return {"merged_count": 0}

    monkeypatch.setattr(kobe.graph, "run_node_merge_audit", audit)

    kobe.explore(max_steps=maximum)

    assert not audits
    assert planned_from == ["s1_list"] * (maximum - start - 1)
    assert json.loads(kobe.graph_path.read_text())["total_steps_completed"] == maximum


@pytest.mark.parametrize("error", [KeyboardInterrupt, RuntimeError])
@pytest.mark.usefixtures("no_sleep")
def test_ranked_arrival_audit_stop_preserves_the_counted_continuation(
    kobe: Kobe,
    error: type[BaseException],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An interrupted audit saves the completed continuation without planning another step."""
    _list_reached_from_home(kobe)
    _identifying(kobe, monkeypatch, "s0_home")
    planned_from = _planning_back(monkeypatch)
    kobe.graph.total_steps_completed = 49
    monkeypatch.setattr(kobe, "_self_loops_in_a_row", {"s0_home": 15})
    monkeypatch.setattr(kobe, "_choose_continuation_target", lambda _current: "s1_list")
    monkeypatch.setattr(kobe, "navigate_to_node", lambda target: target)
    audits: list[str] = []

    def audit(*, app_name: str) -> NoReturn:
        audits.append(app_name)
        raise error

    monkeypatch.setattr(kobe.graph, "run_node_merge_audit", audit)
    caplog.set_level(logging.INFO, logger="test")

    with pytest.raises(error):
        kobe.explore(max_steps=51)

    assert audits == ["demo"]
    assert not planned_from
    assert _numbered_steps(caplog.messages) == ["Step 50/51: s0_home --[continue]--> s1_list"]
    assert json.loads(kobe.graph_path.read_text())["total_steps_completed"] == 50


@pytest.mark.usefixtures("no_sleep")
def test_ranked_arrival_obeys_the_next_steps_saturation_rule(
    kobe: Kobe,
    router_responses: list[httpx.Response],
    chat_response: Callable[..., httpx.Response],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Retaining an arrival permits a fresh continuation on the following counted step."""
    _list_reached_from_home(kobe)
    _identifying(kobe, monkeypatch, "s0_home")
    planned_from = _planning_back(monkeypatch)
    kobe.graph.total_steps_completed = 49
    monkeypatch.setattr(kobe, "_self_loops_in_a_row", {"s0_home": 15, "s1_list": 15})
    monkeypatch.setattr(
        kobe,
        "_choose_continuation_target",
        lambda node: "s1_list" if node == "s0_home" else "s0_home",
    )
    monkeypatch.setattr(kobe, "navigate_to_node", lambda target: target)
    router_responses.append(chat_response(content=_WAIT_REPLY))
    audits: list[int] = []

    def audit(*, app_name: str) -> dict[str, int]:
        assert app_name == "demo"
        audits.append(kobe.graph.total_steps_completed)
        return {"merged_count": 0}

    monkeypatch.setattr(kobe.graph, "run_node_merge_audit", audit)
    caplog.set_level(logging.INFO, logger="test")

    kobe.explore(max_steps=52)

    assert audits == [50]
    assert planned_from == ["s0_home"]
    assert _numbered_steps(caplog.messages) == [
        "Step 50/52: s0_home --[continue]--> s1_list",
        "Step 51/52: s1_list --[continue]--> s0_home",
        "Step 52/52: s0_home --[wait]--> s0_home",
    ]


@pytest.mark.usefixtures("no_sleep")
def test_ranked_arrival_audit_outage_resumes_from_the_recovered_screen(
    kobe: Kobe,
    router_responses: list[httpx.Response],
    chat_response: Callable[..., httpx.Response],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Recovery replaces the arrival with a fresh observation after an audit outage."""
    _list_reached_from_home(kobe, "s2_recovered")
    _identifying(kobe, monkeypatch, "s0_home")
    planned_from = _planning_back(monkeypatch)
    kobe.graph.total_steps_completed = 49
    monkeypatch.setattr(kobe, "_self_loops_in_a_row", {"s0_home": 15})
    monkeypatch.setattr(kobe, "_choose_continuation_target", lambda _current: "s1_list")
    monkeypatch.setattr(kobe, "navigate_to_node", lambda target: target)
    router_responses.append(chat_response(content=_WAIT_REPLY))
    audits: list[str] = []
    recoveries: list[subprocess.TimeoutExpired] = []

    def audit(*, app_name: str) -> NoReturn:
        audits.append(app_name)
        command = "adb"
        raise subprocess.TimeoutExpired(command, 30)

    def recover(error: subprocess.TimeoutExpired) -> tuple[str, dict[str, object]]:
        recoveries.append(error)
        return "s2_recovered", as_object_dict(kobe.controller.get_state())

    monkeypatch.setattr(kobe.graph, "run_node_merge_audit", audit)
    monkeypatch.setattr(kobe, "_recover_from_outage", recover)
    caplog.set_level(logging.INFO, logger="test")

    kobe.explore(max_steps=51)

    assert audits == ["demo"]
    assert len(recoveries) == 1
    assert planned_from == ["s2_recovered"]
    assert _numbered_steps(caplog.messages) == [
        "Step 50/51: s0_home --[continue]--> s1_list",
        "Step 51/51: s2_recovered --[wait]--> s2_recovered",
    ]
    assert json.loads(kobe.graph_path.read_text())["total_steps_completed"] == 51


@pytest.mark.parametrize(
    "action_reply", [_BACK_REPLY, '{"action":"end","required_pointers":0}'], ids=["back", "end"]
)
@pytest.mark.usefixtures("no_sleep")
def test_back_on_the_start_screen_continues_from_a_ranked_target_instead(
    action_reply: str,
    tmp_path: Path,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    chat_response: Callable[..., httpx.Response],
    device_log: list[object],
    screens_by_log: Callable[..., DeviceController],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    step_records: Callable[[Path], list[dict[str, Any]]],
) -> None:
    """Back on the start screen would leave the app, which then restarts onto the same
    screen. The Back is not sent: the coverage checkpoint's target choice runs at once,
    without its merge audit, its replay reaches the target, and the next step plans from
    there. The step counts, and its numbered line shows ``continue`` to the target, so two
    planned steps end a two-step run with two numbered lines."""
    explorer = _explorer(screens_by_log(HOME_SCREEN), tmp_path, router_client)
    _list_reached_from_home(explorer)
    explorer.graph.add_edge("s0_home", "s1_list", action=[_OPEN_LIST])
    expected = _identifying(explorer, monkeypatch, "s0_home", "s0_home", "s1_list")
    planned_from = _planning_back(monkeypatch)
    router_responses.extend(
        [
            chat_response(content=action_reply),
            chat_response(content='{"node_id":"s1_list","reason":"unexplored"}'),
            chat_response(content=_WAIT_REPLY),
        ]
    )
    caplog.set_level(logging.INFO, logger="test")

    explorer.explore(max_steps=2)

    assert device_log == [*_FIRST_RESTART, _OPEN_LIST]
    assert expected == [None, "s0_home", "s1_list"]
    assert planned_from == ["s0_home", "s1_list"]
    assert _requested_models(router_requests) == [
        "provider/action",
        "provider/page_detail",
        "provider/action",
    ]
    target_choice = json.loads(router_requests[1].content)
    assert target_choice["response_format"]["json_schema"]["name"] == "exploration_target"
    assert "node_id=s0_home" not in target_choice["messages"][0]["content"]
    assert list(explorer.graph.graph.edges) == [("s0_home", "s1_list")]
    assert (
        "Back on s0_home leaves the app; continuing from ranked target s1_list instead."
        in caplog.text
    )
    assert _numbered_steps(caplog.messages) == [
        "Step 1/2: s0_home --[continue]--> s1_list",
        "Step 2/2: s1_list --[wait]--> s1_list",
    ]
    assert [(r["kind"], r["source"]) for r in step_records(explorer.controller.steps)] == [
        ("observation", "main_loop"),
        ("match", "main_loop"),
        ("system_event", "navigation_replay"),
        ("observation", "navigation_replay"),
        ("match", "navigation_replay"),
        ("action", "navigation_replay"),
        ("observation", "navigation_replay"),
        ("match", "navigation_replay"),
        ("observation", "coverage_checkpoint"),
        ("observation", "main_loop"),
    ]


@pytest.mark.parametrize("screen", ["s0_home", "s1_list"], ids=["start screen", "other screen"])
@pytest.mark.usefixtures("no_sleep")
def test_back_is_pressed_on_other_screens_and_when_no_other_node_is_ranked(
    screen: str,
    tmp_path: Path,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    chat_response: Callable[..., httpx.Response],
    device_log: list[object],
    screens_by_log: Callable[..., DeviceController],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No recorded edge leads from the start screen to the list, so on the start screen no
    other node is ranked. On the list the start screen is ranked, and Back is pressed all
    the same, as on any screen other than the start screen. Either way the Back is
    recorded and no target is chosen."""
    explorer = _explorer(screens_by_log(HOME_SCREEN), tmp_path, router_client)
    _list_reached_from_home(explorer)
    _identifying(explorer, monkeypatch, screen, "s0_home")
    _planning_back(monkeypatch)
    router_responses.append(chat_response(content=_BACK_REPLY))

    explorer.explore(max_steps=1)

    assert device_log == [_BACK]
    assert _requested_models(router_requests) == ["provider/action"]
    assert explorer.graph.graph.edges[screen, "s0_home"]["actions"] == [[_BACK]]


@pytest.mark.usefixtures("no_sleep")
def test_back_on_the_start_screen_whose_replay_cannot_restart_the_app_stays_there(
    tmp_path: Path,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    chat_response: Callable[..., httpx.Response],
    device_log: list[object],
    screens_by_log: Callable[..., DeviceController],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A refused force-stop leaves the start screen showing, as at a coverage checkpoint:
    Back is still not pressed, the step counts with the start screen as the node it
    continues from, and the next step plans from the start screen."""
    explorer = _explorer(screens_by_log(HOME_SCREEN), tmp_path, router_client)
    _list_reached_from_home(explorer)
    explorer.graph.add_edge("s0_home", "s1_list", action=[_OPEN_LIST])
    expected = _identifying(explorer, monkeypatch, "s0_home", "s0_home")
    monkeypatch.setattr(subprocess, "run", _adb_refusing_once(device_log, _FORCE_STOP))
    planned_from = _planning_back(monkeypatch)
    router_responses.extend(
        [
            chat_response(content=_BACK_REPLY),
            chat_response(content='{"node_id":"s1_list","reason":"unexplored"}'),
            chat_response(content=_WAIT_REPLY),
        ]
    )
    caplog.set_level(logging.INFO, logger="test")

    explorer.explore(max_steps=2)

    assert device_log == [_FORCE_STOP]
    assert expected == [None]
    assert planned_from == ["s0_home", "s0_home"]
    assert list(explorer.graph.graph.edges) == [("s0_home", "s1_list")]
    assert _numbered_steps(caplog.messages) == [
        "Step 1/2: s0_home --[continue]--> s0_home",
        "Step 2/2: s0_home --[wait]--> s0_home",
    ]


_LAUNCHER = "ext_com.android.launcher3"
_OPEN_DETAIL = {"action": "tap", "coordinate": [30, 40]}


def _edge_to_the_launcher(explorer: Kobe, source: str, action: dict[str, object]) -> None:
    """Give the graph the start screen, a list and a detail screen, each reached from the
    start screen, and an edge by ``action`` from ``source`` to the launcher's node."""
    _list_reached_from_home(explorer, "s2_detail")
    explorer.graph.graph.add_node(_LAUNCHER, is_external=True, replay_misses=0)
    explorer.graph.add_edge("s0_home", "s1_list", action=[_OPEN_LIST])
    explorer.graph.add_edge("s0_home", "s2_detail", action=[_OPEN_DETAIL])
    explorer.graph.add_edge(source, _LAUNCHER, action=[action])


@pytest.mark.usefixtures("no_sleep")
def test_back_where_a_recorded_back_has_left_the_app_continues_from_a_ranked_target_instead(
    tmp_path: Path,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    chat_response: Callable[..., httpx.Response],
    device_log: list[object],
    screens_by_log: Callable[..., DeviceController],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An app that reopens on the tab used last makes that tab's Back leave the app as
    the start screen's does. Once a Back from the list has been recorded leaving the app,
    a Back planned there is not sent: exploration continues from a ranked target."""
    explorer = _explorer(screens_by_log(HOME_SCREEN), tmp_path, router_client)
    _edge_to_the_launcher(explorer, "s1_list", _BACK)
    expected = _identifying(explorer, monkeypatch, "s1_list", "s0_home", "s2_detail")
    planned_from = _planning_back(monkeypatch)
    router_responses.extend(
        [
            chat_response(content=_BACK_REPLY),
            chat_response(content='{"node_id":"s2_detail","reason":"unexplored"}'),
            chat_response(content=_WAIT_REPLY),
        ]
    )
    caplog.set_level(logging.INFO, logger="test")

    explorer.explore(max_steps=2)

    assert device_log == [*_FIRST_RESTART, _OPEN_DETAIL]
    assert expected == [None, "s0_home", "s2_detail"]
    assert planned_from == ["s1_list", "s2_detail"]
    assert _requested_models(router_requests) == [
        "provider/action",
        "provider/page_detail",
        "provider/action",
    ]
    assert list(explorer.graph.graph.edges(data="visit_count")) == [
        ("s0_home", "s1_list", 1),
        ("s0_home", "s2_detail", 1),
        ("s1_list", _LAUNCHER, 1),
    ]
    assert (
        "Back on s1_list leaves the app; continuing from ranked target s2_detail instead."
        in caplog.text
    )
    assert _numbered_steps(caplog.messages) == [
        "Step 1/2: s1_list --[continue]--> s2_detail",
        "Step 2/2: s2_detail --[wait]--> s2_detail",
    ]


@pytest.mark.parametrize(
    ("source", "action"),
    [
        pytest.param("s1_list", _OPEN_LIST, id="a tap on the list left the app"),
        pytest.param("s2_detail", _BACK, id="Back on another screen left the app"),
    ],
)
@pytest.mark.usefixtures("no_sleep")
def test_back_is_pressed_where_no_recorded_back_has_left_the_app(
    source: str,
    action: dict[str, object],
    tmp_path: Path,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    chat_response: Callable[..., httpx.Response],
    device_log: list[object],
    screens_by_log: Callable[..., DeviceController],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only an edge by Back from the list to an ``ext_`` node shows that Back leaves the
    app there, so without one the list's Back is pressed and recorded."""
    explorer = _explorer(screens_by_log(HOME_SCREEN), tmp_path, router_client)
    _edge_to_the_launcher(explorer, source, action)
    _identifying(explorer, monkeypatch, "s1_list", "s0_home")
    _planning_back(monkeypatch)
    router_responses.append(chat_response(content=_BACK_REPLY))

    explorer.explore(max_steps=1)

    assert device_log == [_BACK]
    assert _requested_models(router_requests) == ["provider/action"]
    assert explorer.graph.graph.edges["s1_list", "s0_home"]["actions"] == [[_BACK]]


_BACK_LEAVES_THE_APP = (
    "Back on this screen leaves the app for the home screen, so choose an action on this "
    "screen instead, even where a rule below says to go back."
)


@pytest.mark.parametrize(
    ("screen", "source", "action", "told"),
    [
        pytest.param("s0_home", "s1_list", _OPEN_LIST, True, id="the start screen"),
        pytest.param("s1_list", "s1_list", _BACK, True, id="a recorded Back left the app"),
        pytest.param("s1_list", "s1_list", _OPEN_LIST, False, id="a tap left the app"),
        pytest.param("s1_list", "s2_detail", _BACK, False, id="another screen's Back left"),
    ],
)
@pytest.mark.usefixtures("no_sleep")
def test_the_planner_is_told_that_back_leaves_the_app_where_it_does(
    screen: str,
    source: str,
    action: dict[str, object],
    told: bool,
    tmp_path: Path,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    chat_response: Callable[..., httpx.Response],
    screens_by_log: Callable[..., DeviceController],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The planner's prompt says that Back leaves the app on the start screen and on a
    screen with an edge by Back to an ``ext_`` node, the screens where a planned Back
    continues from a ranked target, and on no other screen."""
    explorer = _explorer(screens_by_log(HOME_SCREEN), tmp_path, router_client)
    _edge_to_the_launcher(explorer, source, action)
    _identifying(explorer, monkeypatch, screen)
    monkeypatch.setattr("android_app_graph.kobe.soft_keyboard_hint", lambda _udid: "")
    router_responses.extend(
        [
            chat_response(
                content='{"element":null,"instruction":"Scroll down","required_pointers":1}'
            ),
            chat_response(content=_WAIT_REPLY),
        ]
    )

    explorer.explore(max_steps=1)

    assert _requested_models(router_requests) == ["provider/instruction", "provider/action"]
    prompt = json.loads(router_requests[0].content)["messages"][-1]["content"][0]["text"]
    assert (_BACK_LEAVES_THE_APP in prompt) is told


@pytest.mark.usefixtures("no_sleep")
def test_back_on_the_start_screen_passes_over_a_target_whose_last_replay_missed(
    tmp_path: Path,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    chat_response: Callable[..., httpx.Response],
    device_log: list[object],
    screens_by_log: Callable[..., DeviceController],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The replay to the list that Back on the start screen continues with lands on the
    start screen and counts a miss on the list. With the list out of the ranking, the
    second Back on the start screen is pressed without a model call to choose a target."""
    explorer = _explorer(screens_by_log(HOME_SCREEN), tmp_path, router_client)
    _list_reached_from_home(explorer)
    explorer.graph.add_edge("s0_home", "s1_list", action=[_OPEN_LIST])
    expected = _identifying(explorer, monkeypatch, *["s0_home"] * 4)
    planned_from = _planning_back(monkeypatch)
    router_responses.extend(
        [
            chat_response(content=_BACK_REPLY),
            chat_response(content='{"node_id":"s1_list","reason":"unexplored"}'),
            chat_response(content=_BACK_REPLY),
        ]
    )

    explorer.explore(max_steps=2)

    assert device_log == [*_FIRST_RESTART, _OPEN_LIST, _BACK]
    assert expected == [None, "s0_home", "s1_list", None]
    assert planned_from == ["s0_home"] * 2
    assert _requested_models(router_requests) == [
        "provider/action",
        "provider/page_detail",
        "provider/action",
    ]
    assert explorer.graph.graph.nodes["s1_list"]["replay_misses"] == 1


_SCROLL_DOWN_PLAN = '{"element":null,"instruction":"Scroll down","required_pointers":1}'


@pytest.mark.usefixtures("no_sleep", "device_log")
def test_repeated_internal_route_informs_the_planner_without_replacing_its_next_action(
    tmp_path: Path,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    chat_response: Callable[..., httpx.Response],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated observed form/dialog navigation retains a useful local action."""
    dialog = _screen("com.demo.Home", "gray")
    picker = _screen("com.demo.Home", "blue")
    device = _Screens(HOME_SCREEN, dialog, HOME_SCREEN, dialog, HOME_SCREEN, picker)
    explorer = _explorer(device, tmp_path, router_client)
    explorer.graph.graph.add_node("s0_seed", package="com.demo", replay_misses=0)
    for node in ("s1_form", "s2_dialog", "s3_picker"):
        explorer.graph.graph.add_node(node, package="com.demo", replay_misses=0)
    explorer.graph.add_edge("s1_form", "s2_dialog", action=[{"action": "back"}])
    explorer.graph.add_edge("s2_dialog", "s1_form", action=[{"action": "back"}])
    _identifying(
        explorer,
        monkeypatch,
        "s1_form",
        "s2_dialog",
        "s1_form",
        "s2_dialog",
        "s1_form",
        "s3_picker",
    )
    _without_templates(monkeypatch)
    monkeypatch.setattr("android_app_graph.kobe.soft_keyboard_hint", lambda _udid: "")
    for _ in range(4):
        router_responses.extend(
            [
                chat_response(
                    content='{"element":null,"instruction":"Press Back","required_pointers":0}'
                ),
                chat_response(content=_BACK_REPLY),
            ]
        )
    router_responses.extend(
        [
            chat_response(
                content='{"element":null,"instruction":"Open the date picker","required_pointers":1}'
            ),
            chat_response(content=_TAP_REPLY),
        ]
    )

    router_responses.extend(
        [
            chat_response(content='{"element":null,"instruction":"Wait","required_pointers":0}'),
            chat_response(content=_WAIT_REPLY),
        ]
    )

    explorer.explore(max_steps=6)

    prompts = [
        str(json.loads(request.content)["messages"][-1]["content"][0]["text"])
        for request in router_requests
        if json.loads(request.content)["model"] == "provider/instruction"
    ]
    assert all("Recent repeated route" not in prompt for prompt in prompts[:4])
    assert "Recent repeated route" in prompts[4]
    assert "The executor supports at most 1 pointer" in prompts[4]
    assert "s1_form --back--> s2_dialog --back--> s1_form" in prompts[4]
    assert "Hidden state may still change" in prompts[4]
    assert device.actions == [*[{"action": "back"}] * 4, {"action": "tap", "x": 50, "y": 100}]
    assert "Recent repeated route" not in prompts[5]
    assert explorer.graph.total_steps_completed == 6
    assert explorer.graph.graph.has_edge("s1_form", "s3_picker")


@pytest.mark.parametrize("boundary", ["wait", "unsupported", "resume", "checkpoint"])
@pytest.mark.usefixtures("no_sleep", "device_log")
def test_ordinary_route_context_restarts_at_exploration_boundaries(
    boundary: str,
    tmp_path: Path,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    chat_response: Callable[..., httpx.Response],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Waits, fresh exploration and checkpoint boundaries keep history contiguous."""
    dialog = _screen("com.demo.Home", "gray")
    device = _Screens(HOME_SCREEN, dialog, HOME_SCREEN, dialog, HOME_SCREEN, dialog)
    explorer = _explorer(device, tmp_path, router_client)
    for node in ("s0_seed", "s1_form", "s2_dialog"):
        explorer.graph.graph.add_node(node, package="com.demo", replay_misses=0)
    explorer.graph.add_edge("s1_form", "s2_dialog", action=[{"action": "back"}])
    explorer.graph.add_edge("s2_dialog", "s1_form", action=[{"action": "back"}])
    visits = ["s1_form", "s2_dialog", "s1_form", "s2_dialog", "s1_form"]
    if boundary == "resume":
        visits.append("s1_form")
    _identifying(explorer, monkeypatch, *visits, "s2_dialog")
    _without_templates(monkeypatch)
    monkeypatch.setattr("android_app_graph.kobe.soft_keyboard_hint", lambda _udid: "")
    if boundary == "checkpoint":
        explorer.coverage_checkpoint_steps = 4
        monkeypatch.setattr(
            explorer, "maybe_continue_from_coverage_checkpoint", lambda *_args: None
        )
    for _ in range(4):
        router_responses.extend(
            [
                chat_response(
                    content='{"element":null,"instruction":"Press Back","required_pointers":0}'
                ),
                chat_response(content=_BACK_REPLY),
            ]
        )
    if boundary == "wait":
        router_responses.extend(
            [
                chat_response(
                    content='{"element":null,"instruction":"Wait","required_pointers":0}'
                ),
                chat_response(content=_WAIT_REPLY),
            ]
        )
    if boundary == "unsupported":
        router_responses.append(
            chat_response(
                content='{"element":null,"instruction":"Pinch inward","required_pointers":2}'
            )
        )
    router_responses.extend(
        [
            chat_response(
                content='{"element":null,"instruction":"Press Back","required_pointers":0}'
            ),
            chat_response(content=_BACK_REPLY),
        ]
    )

    if boundary == "resume":
        explorer.explore(max_steps=4)
    explorer.explore(max_steps=6 if boundary in {"wait", "unsupported"} else 5)

    prompts = [
        str(json.loads(request.content)["messages"][-1]["content"][0]["text"])
        for request in router_requests
        if json.loads(request.content)["model"] == "provider/instruction"
    ]
    if boundary in {"wait", "unsupported"}:
        assert "Recent repeated route" in prompts[4]
    assert "Recent repeated route" not in prompts[-1]
    assert device.actions == [{"action": "back"}] * 5
    assert explorer.graph.total_steps_completed == (6 if boundary in {"wait", "unsupported"} else 5)
    assert explorer.graph.graph.number_of_edges() == 2
    if boundary == "unsupported":
        records = [
            json.loads(path.read_text()) for path in explorer.controller.steps.glob("[0-9]*.json")
        ]
        assert sum(record["kind"] == "unsupported_action" for record in records) == 1


def _toggles(explorer: Kobe, *days: int) -> None:
    """Record a self-loop on the start screen with one action for each of ``days``, in
    order, each with the instruction ``Toggle day <day>``."""
    for index, day in enumerate(days):
        explorer.graph.add_edge(
            "s0_home",
            "s0_home",
            action=[{"action": "tap", "coordinate": [index, 0]}],
            details=EdgeDetails(instruction=f"Toggle day {day}", target_observation="Alarms"),
        )


def _without_templates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the edge normalization the run's end asks for find no template."""
    monkeypatch.setattr(
        "android_app_graph.utils.graph_manager.normalize_edge",
        lambda *_args, **_kwargs: {"is_template": False},
    )


def _tried_in_planner_prompt(router_requests: list[httpx.Request]) -> list[str]:
    """Return the lines of the first request's prompt that list the instructions tried."""
    request = json.loads(router_requests[0].content)
    prompt = str(request["messages"][-1]["content"][0]["text"])
    return prompt.split("where they led:\n", 1)[1].split("\n\n", 1)[0].splitlines()


@pytest.mark.parametrize(
    ("days", "listed", "count_line"),
    [
        pytest.param(
            [*range(1, 33), 2],
            [*range(4, 33), 2],
            ["- 2 more tried earlier also stayed on this screen"],
            id="32 distinct instructions",
        ),
        pytest.param([*range(1, 31), 2], [1, *range(3, 31), 2], [], id="30 distinct instructions"),
    ],
)
@pytest.mark.usefixtures("no_sleep")
def test_the_planner_lists_a_self_loop_s_30_most_recent_distinct_instructions_and_counts_the_rest(
    days: list[int],
    listed: list[int],
    count_line: list[str],
    tmp_path: Path,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    chat_response: Callable[..., httpx.Response],
    screens_by_log: Callable[..., DeviceController],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every action of a self-loop leads back to the screen the planner sees, so its
    prompt lists the self-loop's 30 most recent distinct instructions, oldest first and
    without where they led, and then how many more were tried. An instruction tried twice
    counts once, at its latest action."""
    explorer = _explorer(screens_by_log(HOME_SCREEN), tmp_path, router_client)
    _list_reached_from_home(explorer)
    _toggles(explorer, *days)
    _identifying(explorer, monkeypatch, "s0_home")
    _without_templates(monkeypatch)
    monkeypatch.setattr("android_app_graph.kobe.soft_keyboard_hint", lambda _udid: "")
    router_responses.extend(
        [chat_response(content=_SCROLL_DOWN_PLAN), chat_response(content=_WAIT_REPLY)]
    )

    explorer.explore(max_steps=1)

    assert _tried_in_planner_prompt(router_requests) == [
        *(f'- "Toggle day {day}" → stayed on this screen' for day in listed),
        *count_line,
    ]


@pytest.mark.usefixtures("no_sleep")
def test_the_planner_lists_every_action_tried_toward_another_screen_and_where_it_led(
    tmp_path: Path,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    chat_response: Callable[..., httpx.Response],
    screens_by_log: Callable[..., DeviceController],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """However many actions an edge to another screen holds, the planner's prompt lists
    each of them with the observation it recorded, and the self-loop's instructions after
    them."""
    explorer = _explorer(screens_by_log(HOME_SCREEN), tmp_path, router_client)
    _list_reached_from_home(explorer)
    _toggles(explorer, 1)
    for row in range(35):
        explorer.graph.add_edge(
            "s0_home",
            "s1_list",
            action=[{"action": "tap", "coordinate": [0, row]}],
            details=EdgeDetails(instruction=f"Open row {row}", target_observation=f"Row {row}"),
        )
    _identifying(explorer, monkeypatch, "s0_home")
    _without_templates(monkeypatch)
    monkeypatch.setattr("android_app_graph.kobe.soft_keyboard_hint", lambda _udid: "")
    router_responses.extend(
        [chat_response(content=_SCROLL_DOWN_PLAN), chat_response(content=_WAIT_REPLY)]
    )

    explorer.explore(max_steps=1)

    assert _tried_in_planner_prompt(router_requests) == [
        *(f'- "Open row {row}" → led to: "Row {row}"' for row in range(35)),
        '- "Toggle day 1" → stayed on this screen',
    ]


@pytest.mark.usefixtures("no_sleep")
def test_after_15_self_loops_on_a_screen_its_next_step_continues_from_a_ranked_target(
    tmp_path: Path,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    chat_response: Callable[..., httpx.Response],
    device_log: list[object],
    screens_by_log: Callable[..., DeviceController],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Fifteen steps in a row on the start screen each stay on it, so the sixteenth plans
    nothing there: it continues from a ranked target, as a Back that leaves the app does,
    and counts as a ``continue`` step. The continuation ends the start screen's run of
    self-loops, so the next step there plans again."""
    explorer = _explorer(screens_by_log(HOME_SCREEN), tmp_path, router_client)
    _list_reached_from_home(explorer)
    explorer.graph.add_edge("s0_home", "s1_list", action=[_OPEN_LIST])
    _identifying(explorer, monkeypatch, *["s0_home"] * 17, "s1_list", "s0_home")
    planned_from: list[object] = []
    monkeypatch.setattr("android_app_graph.kobe.plan_next_action", _planner("Toggle", planned_from))
    monkeypatch.setattr("android_app_graph.kobe.soft_keyboard_hint", lambda _udid: "")
    router_responses.extend(chat_response(content=_TAP_REPLY) for _ in range(15))
    router_responses.extend(
        [
            chat_response(content='{"node_id":"s1_list","reason":"unexplored"}'),
            chat_response(content=_TAP_REPLY),
            chat_response(content=_WAIT_REPLY),
        ]
    )
    caplog.set_level(logging.INFO, logger="test")

    explorer.explore(max_steps=18)

    assert planned_from == [*["s0_home"] * 15, "s1_list", "s0_home"]
    assert _requested_models(router_requests) == [
        *["provider/action"] * 15,
        "provider/page_detail",
        "provider/action",
        "provider/action",
    ]
    tap = device_log[0]
    assert device_log == [*[tap] * 15, *_FIRST_RESTART, _OPEN_LIST, tap]
    assert (
        "The last 15 steps on s0_home stayed on it; continuing from ranked target s1_list "
        "instead." in caplog.text
    )
    assert _numbered_steps(caplog.messages)[14:] == [
        "Step 15/18: s0_home --[tap]--> s0_home",
        "Step 16/18: s0_home --[continue]--> s1_list",
        "Step 17/18: s1_list --[tap]--> s0_home",
        "Step 18/18: s0_home --[wait]--> s0_home",
    ]


@pytest.mark.usefixtures("no_sleep", "device_log")
def test_after_15_self_loops_on_a_screen_with_no_other_node_ranked_its_next_step_plans(
    tmp_path: Path,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    chat_response: Callable[..., httpx.Response],
    screens_by_log: Callable[..., DeviceController],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """No recorded edge leads from the start screen to the list, so no other node is
    ranked, and the step after fifteen self-loops on the start screen plans and acts there
    as any other step does."""
    explorer = _explorer(screens_by_log(HOME_SCREEN), tmp_path, router_client)
    _list_reached_from_home(explorer)
    _identifying(explorer, monkeypatch, *["s0_home"] * 17)
    planned_from: list[object] = []
    monkeypatch.setattr("android_app_graph.kobe.plan_next_action", _planner("Toggle", planned_from))
    monkeypatch.setattr("android_app_graph.kobe.soft_keyboard_hint", lambda _udid: "")
    router_responses.extend(chat_response(content=_TAP_REPLY) for _ in range(16))
    caplog.set_level(logging.INFO, logger="test")

    explorer.explore(max_steps=16)

    assert planned_from == ["s0_home"] * 16
    assert _requested_models(router_requests) == ["provider/action"] * 16
    assert "The last 15 steps on s0_home stayed on it, but no other node is ranked." in caplog.text
    assert _numbered_steps(caplog.messages)[-1] == "Step 16/16: s0_home --[tap]--> s0_home"


@pytest.mark.usefixtures("no_sleep", "device_log")
def test_a_step_to_another_screen_starts_the_count_of_self_loops_again(
    tmp_path: Path,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    chat_response: Callable[..., httpx.Response],
    screens_by_log: Callable[..., DeviceController],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Seven self-loops on the start screen, a step to the list and back, and eight more
    self-loops make fifteen, but a step that left the start screen ends the first seven's
    run, so the next step there plans."""
    explorer = _explorer(screens_by_log(HOME_SCREEN), tmp_path, router_client)
    _list_reached_from_home(explorer)
    explorer.graph.add_edge("s0_home", "s1_list", action=[_OPEN_LIST])
    _identifying(explorer, monkeypatch, *["s0_home"] * 8, "s1_list", *["s0_home"] * 9)
    planned_from: list[object] = []
    monkeypatch.setattr("android_app_graph.kobe.plan_next_action", _planner("Toggle", planned_from))
    monkeypatch.setattr("android_app_graph.kobe.soft_keyboard_hint", lambda _udid: "")
    router_responses.extend(chat_response(content=_TAP_REPLY) for _ in range(17))
    router_responses.append(chat_response(content=_WAIT_REPLY))
    caplog.set_level(logging.INFO, logger="test")

    explorer.explore(max_steps=18)

    assert planned_from == [*["s0_home"] * 8, "s1_list", *["s0_home"] * 9]
    assert _requested_models(router_requests) == ["provider/action"] * 18
    assert _numbered_steps(caplog.messages)[-1] == "Step 18/18: s0_home --[wait]--> s0_home"


def test_exploration_embeds_the_members_a_loaded_graph_has_no_image_vector_for(
    tmp_path: Path,
    router_client: OpenRouter,
    router_requests: list[httpx.Request],
    router_responses: list[httpx.Response],
    embedding_body: dict[str, object],
    write_graph_file: Callable[..., None],
    write_image_vectors: Callable[..., None],
) -> None:
    """The home screen was embedded before; the list never was, and the other app's
    screen is not embedded at all."""
    graph_path = tmp_path / "demo" / "demo.json"
    graph_path.parent.mkdir()
    nodes = [{"id": "s0_home"}, {"id": "s1_list"}, {"id": "ext_com.other", "is_external": True}]
    write_graph_file(
        graph_path, {"nodes": nodes, "edges": []}, embedding_model="provider/embedding"
    )
    steps_dir = graph_path.parent / "steps"
    write_image_vectors(steps_dir, ["s0_home-observation"], model="provider/embedding")
    router_responses.append(httpx.Response(200, json=embedding_body))

    _explorer(_Device(AssertionError("loading must not read the device")), tmp_path, router_client)

    (request,) = router_requests
    body = json.loads(request.content)
    assert body["model"] == "provider/embedding"
    assert body["input"][0]["content"][0]["image_url"]["url"].startswith("data:image/webp;base64,")
    assert json.loads((steps_dir / "image_emb.json").read_text(encoding="utf-8")) == {
        "model": "provider/embedding",
        "embeddings": {"s0_home-observation": [1.0, 0.0], "s1_list-observation": [0.25, 0.75]},
    }


def test_exploration_embeds_every_member_again_when_the_vectors_are_from_another_model(
    tmp_path: Path,
    router_client: OpenRouter,
    router_requests: list[httpx.Request],
    router_responses: list[httpx.Response],
    embedding_body: dict[str, object],
    write_graph_file: Callable[..., None],
    write_image_vectors: Callable[..., None],
) -> None:
    graph_path = tmp_path / "demo" / "demo.json"
    graph_path.parent.mkdir()
    write_graph_file(
        graph_path,
        {"nodes": [{"id": "s0_home"}, {"id": "s1_list"}], "edges": []},
        embedding_model="provider/embedding",
    )
    steps_dir = graph_path.parent / "steps"
    members = ["s0_home-observation", "s1_list-observation"]
    write_image_vectors(steps_dir, members, model="provider/older-embedding")
    router_responses.extend(httpx.Response(200, json=embedding_body) for _ in members)

    _explorer(_Device(AssertionError("loading must not read the device")), tmp_path, router_client)

    assert len(router_requests) == len(members)
    assert json.loads((steps_dir / "image_emb.json").read_text(encoding="utf-8")) == {
        "model": "provider/embedding",
        "embeddings": {member: [0.25, 0.75] for member in members},
    }


@pytest.mark.parametrize(
    "vector_file_name",
    [
        pytest.param("steps/image_emb.json", id="image vectors"),
        pytest.param("demo_text_emb.json", id="text vectors"),
    ],
)
def test_a_damaged_vector_file_stops_exploration_before_any_model_request(
    vector_file_name: str,
    tmp_path: Path,
    router_client: OpenRouter,
    router_requests: list[httpx.Request],
    router_responses: list[httpx.Response],
    embedding_body: dict[str, object],
    write_graph_file: Callable[[Path, dict[str, object]], None],
) -> None:
    graph_path = tmp_path / "demo" / "demo.json"
    graph_path.parent.mkdir()
    write_graph_file(graph_path, {"nodes": [{"id": "s0_home"}], "edges": []})
    vector_file = graph_path.parent / vector_file_name
    truncated = '{"model": "provider/embedding", "embeddings": {"s0_home-observation": [1.0,'
    vector_file.write_text(truncated, encoding="utf-8")
    router_responses.append(httpx.Response(200, json=embedding_body))

    with pytest.raises(
        ValueError, match=rf"{re.escape(vector_file.name)} cannot be read or parsed"
    ):
        _explorer(
            _Device(AssertionError("loading must not read the device")), tmp_path, router_client
        )

    assert router_requests == []
    assert vector_file.read_text(encoding="utf-8") == truncated


@pytest.mark.usefixtures("no_sleep")
def test_every_screen_identify_state_takes_is_embedded_and_its_decision_recorded(
    tmp_path: Path,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    chat_response: Callable[..., httpx.Response],
    embedding_body: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    step_records: Callable[[Path], list[dict[str, Any]]],
) -> None:
    """The home screen is described, the step's screen matched exactly, both decisions are
    in the identify file, and the vectors of both are saved with the graph, under the
    image-embedding model."""
    monkeypatch.setattr("android_app_graph.kobe.soft_keyboard_hint", lambda _udid: "")
    explorer = _explorer(_Screens(HOME_SCREEN), tmp_path, router_client)
    router_responses.extend(
        [
            httpx.Response(200, json=embedding_body),
            chat_response(
                content='{"page_description": "home", "surface": {"kind": "page", "relation": "unknown", "return_depth": 0}}'
            ),
            httpx.Response(200, json=embedding_body),
            chat_response(
                content='{"element":null,"instruction":"Scroll down","required_pointers":1}'
            ),
            chat_response(content=_TAP_REPLY),
            httpx.Response(200, json=embedding_body),
        ]
    )

    explorer.explore(max_steps=1)

    observations = [
        r for r in step_records(explorer.controller.steps) if r["kind"] == "observation"
    ]
    lines = _identify_lines(explorer)
    assert [line["observation"] for line in lines] == [r["id"] for r in observations]
    assert [line["identify"]["stage"] for line in lines] == ["new", "exact"]
    assert [line["describe"]["page_description"] for line in lines] == ["home", "home"]
    assert all(len(r["pixel_hash"]) == 64 and r["status_bar_rows"] == 0 for r in observations)
    stored = json.loads((explorer.controller.steps / "image_emb.json").read_text("utf-8"))
    assert stored["model"] == "provider/embedding"
    assert sorted(stored["embeddings"]) == [r["id"] for r in observations]


@pytest.mark.usefixtures("no_sleep")
def test_exploration_that_starts_in_another_app_restarts_the_app_and_stops_if_it_stays(
    tmp_path: Path,
    router_client: OpenRouter,
    device_log: list[object],
    screens_by_log: Callable[..., DeviceController],
) -> None:
    explorer = _explorer(screens_by_log(_SEARCH_SCREEN), tmp_path, router_client)

    with pytest.raises(StillOutsideAppError, match=_SEARCH):
        explorer.explore(max_steps=1)

    assert device_log == _FIRST_RESTART
    assert list(explorer.graph.graph) == []


_PROBE = ["adb", "-s", "test-device", "shell", "true"]


def _timed_out(*args: str, timeout: float) -> subprocess.TimeoutExpired:
    return subprocess.TimeoutExpired(["adb", "-s", "test-device", *args], timeout)


def _dump_timed_out() -> subprocess.TimeoutExpired:
    """What a screen read raises when its UI dump times out twice."""
    return _timed_out("exec-out", "uiautomator dump /dev/stdout", timeout=30)


def _read_failed(diagnostic: str) -> subprocess.CalledProcessError:
    return subprocess.CalledProcessError(
        1, ["adb", "-s", "test-device", "shell", "dumpsys", "window", "-a"], stderr=diagnostic
    )


class _StallingDevice:
    """A DeviceController that answers each read with the next of ``reads``, raising one
    that is an exception as adb does for a device that does not answer, and logs each
    action to the device log, timing out the actions ``stalled_actions`` numbers from 1."""

    w = 100
    h = 200

    def __init__(
        self,
        log: list[object],
        reads: Sequence[dict[str, object] | Exception],
        stalled_actions: tuple[int, ...] = (),
    ) -> None:
        self.config: dict[str, object] = {"device": {"udid": "test-device"}}
        self._log = log
        self._reads = iter(reads)
        self._stalled_actions = stalled_actions
        self._actions = 0

    def get_state(self, **_kwargs: float | None) -> dict[str, object]:
        read = next(self._reads)
        if isinstance(read, Exception):
            raise read
        return read

    def exe_action(self, action: dict[str, object]) -> None:
        self._log.append(action)
        self._actions += 1
        if self._actions in self._stalled_actions:
            stalled = _timed_out("shell", "input", "tap", "50", "100", timeout=15)
            raise stalled


def _adb_probed(log: list[object], *probes: str) -> Callable[..., subprocess.CompletedProcess[str]]:
    """A ``subprocess.run`` that logs each command and answers the device's quick reads as
    ``probes`` says in turn, the last for every read after it: ``"answer"``, ``"timeout"``
    or ``"offline"``, which adb reports with exit status 1. Every other command succeeds."""

    def run(
        command: list[str], *, timeout: float | None = None, **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        log.append(command)
        outcome = "answer"
        if command == _PROBE:
            outcome = probes[min(log.count(_PROBE), len(probes)) - 1]
        if outcome == "timeout":
            raise subprocess.TimeoutExpired(command, timeout or 0)
        returncode = 1 if outcome == "offline" else 0
        return subprocess.CompletedProcess(command, returncode, stdout="", stderr="")

    return run


_LIST_SCREEN = _screen("com.demo.List", "gray")
_DETAIL_SCREEN = _screen("com.demo.Detail", "blue")


@pytest.mark.parametrize(
    ("reads", "stalled_actions", "probes", "sleeps"),
    [
        *[
            pytest.param(
                [HOME_SCREEN, _LIST_SCREEN, _read_failed(diagnostic), _LIST_SCREEN, _DETAIL_SCREEN],
                (2,),
                ["answer"],
                [0.5, 30, 0.5],
                id=f"recovery read exits nonzero: {diagnostic or 'no diagnostics'}",
            )
            for diagnostic in (
                "error: device offline",
                "error: closed",
                "android.os.DeadObjectException",
                "",
            )
        ],
        pytest.param(
            [HOME_SCREEN, _LIST_SCREEN, _dump_timed_out(), _LIST_SCREEN, _DETAIL_SCREEN],
            (),
            ["timeout", "answer"],
            [0.5, 0.5, 30, 0.5],
            id="a read times out twice",
        ),
        pytest.param(
            [HOME_SCREEN, _LIST_SCREEN, _LIST_SCREEN, _DETAIL_SCREEN],
            (2,),
            ["offline", "answer"],
            [0.5, 30, 0.5],
            id="an input times out",
        ),
        pytest.param(
            [HOME_SCREEN, _LIST_SCREEN, _dump_timed_out(), _LIST_SCREEN, _DETAIL_SCREEN],
            (2,),
            ["answer"],
            [0.5, 30, 0.5],
            id="the read after the device answers times out",
        ),
    ],
)
@pytest.mark.usefixtures("adb_on_path")
def test_a_device_that_stops_answering_and_answers_again_continues_from_the_screen_it_shows(
    reads: list[dict[str, object] | Exception],
    stalled_actions: tuple[int, ...],
    probes: list[str],
    sleeps: list[float],
    tmp_path: Path,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    chat_response: Callable[..., httpx.Response],
    no_sleep: list[float],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    step_records: Callable[[Path], list[dict[str, Any]]],
) -> None:
    """The step the outage interrupted counts as ``interrupted`` and records no edge, since
    what its action did is unknown; the next step plans from the screen the device shows
    when it answers again, where the setup of its action is unknown too. The screen read
    after the outage is recorded with the source ``device_recovery``."""
    log: list[object] = []
    monkeypatch.setattr(subprocess, "run", _adb_probed(log, *probes))
    monkeypatch.setattr(time, "monotonic", lambda: sum(no_sleep))
    explorer = _explorer(_StallingDevice(log, reads, stalled_actions), tmp_path, router_client)
    _list_reached_from_home(explorer, "s2_detail")
    identified = _identifying(explorer, monkeypatch, "s0_home", "s1_list", "s1_list", "s2_detail")
    planned_from: list[object] = []
    monkeypatch.setattr(
        "android_app_graph.kobe.plan_next_action", _planner("Open it", planned_from)
    )
    monkeypatch.setattr("android_app_graph.kobe.soft_keyboard_hint", lambda _udid: "")
    router_responses.extend(chat_response(content=_TAP_REPLY) for _ in range(3))
    caplog.set_level(logging.INFO, logger="test")

    explorer.explore(max_steps=3)

    assert _numbered_steps(caplog.messages) == [
        "Step 1/3: s0_home --[tap]--> s1_list",
        "Step 2/3: s1_list --[interrupted]--> s1_list",
        "Step 3/3: s1_list --[tap]--> s2_detail",
    ]
    assert planned_from == ["s0_home", "s1_list", "s1_list"]
    assert identified == [None, None, None, None]
    assert list(explorer.graph.graph.edges) == [("s0_home", "s1_list"), ("s1_list", "s2_detail")]
    assert explorer.graph.graph["s1_list"]["s2_detail"]["setups"] == [None]
    assert log == [_TAP, _TAP, *[_PROBE] * 2, _TAP]
    assert no_sleep == sleeps
    in_app = [("action", "main_loop"), ("observation", "main_loop"), ("match", "main_loop")]
    assert [(r["kind"], r["source"]) for r in step_records(explorer.controller.steps)] == [
        ("observation", "main_loop"),
        ("match", "main_loop"),
        *in_app,
        ("transition", "main_loop"),
        ("action", "main_loop"),
        ("observation", "device_recovery"),
        ("match", "device_recovery"),
        *in_app,
        ("transition", "main_loop"),
    ]
    assert json.loads(explorer.graph_path.read_text("utf-8"))["total_steps_completed"] == 3


@pytest.mark.parametrize(
    ("limit", "minutes", "checks"),
    [
        pytest.param(DEFAULT_UNREACHABLE_LIMIT, "30", 60, id="default"),
        pytest.param(90, "1.5", 3, id="configured"),
    ],
)
@pytest.mark.usefixtures("adb_on_path")
def test_a_device_that_does_not_answer_within_the_limit_stops_the_run_with_the_graph_saved(
    limit: float,
    minutes: str,
    checks: int,
    tmp_path: Path,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    chat_response: Callable[..., httpx.Response],
    no_sleep: list[float],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The device is checked at once and then every 30 s, up to the limit, by default 30
    minutes. The run then stops with an error naming the device and the limit, and logs
    it without a traceback after saving the graph."""
    log: list[object] = []
    monkeypatch.setattr(subprocess, "run", _adb_probed(log, "timeout"))
    monkeypatch.setattr(time, "monotonic", lambda: sum(no_sleep))
    device = _StallingDevice(log, [HOME_SCREEN, _LIST_SCREEN], stalled_actions=(2,))
    explorer = _explorer(device, tmp_path, router_client, unreachable_limit=limit)
    _list_reached_from_home(explorer)
    _identifying(explorer, monkeypatch, "s0_home", "s1_list")
    monkeypatch.setattr("android_app_graph.kobe.plan_next_action", _planner("Open it", []))
    monkeypatch.setattr("android_app_graph.kobe.soft_keyboard_hint", lambda _udid: "")
    router_responses.extend(chat_response(content=_TAP_REPLY) for _ in range(2))
    caplog.set_level(logging.INFO, logger="test")
    unreachable = f"Device test-device unreachable for {minutes} minutes"

    with pytest.raises(DeviceUnreachableError) as raised:
        explorer.explore(max_steps=3)

    assert str(raised.value) == unreachable
    assert log == [_TAP, _TAP, *[_PROBE] * checks]
    assert no_sleep == [0.5, *[30] * checks]
    saved = json.loads(explorer.graph_path.read_text("utf-8"))
    assert saved["total_steps_completed"] == 1
    assert [(edge["source"], edge["target"]) for edge in saved["edges"]] == [("s0_home", "s1_list")]
    stopped = f"{unreachable}. Exploration stopped at step 1. Graph saved. Re-run to resume."
    (record,) = [record for record in caplog.records if record.getMessage() == stopped]
    assert record.exc_info is None
    assert not [record for record in caplog.records if record.levelno >= logging.ERROR]


@pytest.mark.usefixtures("no_sleep", "adb_on_path")
@pytest.mark.parametrize("nonzero_read", [False, True])
def test_a_device_that_stops_answering_after_a_counted_step_counts_no_other_step(
    nonzero_read: bool,
    tmp_path: Path,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    chat_response: Callable[..., httpx.Response],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An outage during the coverage checkpoint that follows a step leaves that step as it
    was counted, and the next step plans from the screen the device shows when it answers."""
    log: list[object] = []
    monkeypatch.setattr(subprocess, "run", _adb_probed(log, "answer"))
    reads: list[dict[str, object] | Exception] = [HOME_SCREEN, _LIST_SCREEN]
    if nonzero_read:
        reads.append(_read_failed(""))
    reads.extend([_LIST_SCREEN, _DETAIL_SCREEN])
    explorer = _explorer(_StallingDevice(log, reads), tmp_path, router_client)
    _list_reached_from_home(explorer, "s2_detail")
    _identifying(explorer, monkeypatch, "s0_home", "s1_list", "s1_list", "s2_detail")
    planned_from: list[object] = []
    monkeypatch.setattr(
        "android_app_graph.kobe.plan_next_action", _planner("Open it", planned_from)
    )
    monkeypatch.setattr("android_app_graph.kobe.soft_keyboard_hint", lambda _udid: "")
    checkpoints: list[int] = []

    def checkpoint(step: int, _max_steps: int, _current_node: str) -> str | None:
        checkpoints.append(step)
        if len(checkpoints) == 1:
            stalled = _timed_out("shell", "am", "force-stop", "com.demo", timeout=30)
            raise stalled
        return None

    monkeypatch.setattr(explorer, "maybe_continue_from_coverage_checkpoint", checkpoint)
    router_responses.extend(chat_response(content=_TAP_REPLY) for _ in range(2))
    caplog.set_level(logging.INFO, logger="test")

    explorer.explore(max_steps=2)

    assert _numbered_steps(caplog.messages) == [
        "Step 1/2: s0_home --[tap]--> s1_list",
        "Step 2/2: s1_list --[tap]--> s2_detail",
    ]
    assert checkpoints == [1, 2]
    assert planned_from == ["s0_home", "s1_list"]
    assert log == [_TAP, *[_PROBE] * (2 if nonzero_read else 1), _TAP]


def test_a_checkpoint_whose_navigation_times_out_reads_nothing_more_from_the_device(
    tmp_path: Path,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    chat_response: Callable[..., httpx.Response],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A device that stops answering during the replay is waited for by the exploration
    loop, not read again as after a navigation that fails for another reason."""
    explorer = _explorer(
        _Device(AssertionError("the screen was read after the device stopped answering")),
        tmp_path,
        router_client,
    )
    _settings_and_detail(explorer)
    router_responses.extend(
        [
            chat_response(content='{"node_id":"s1_settings","reason":"unexplored"}'),
        ]
    )
    stalled = _timed_out("shell", "input", "tap", "1", "2", timeout=15)

    def navigate(_node: str) -> str:
        raise stalled

    monkeypatch.setattr(explorer, "navigate_to_node", navigate)

    with pytest.raises(subprocess.TimeoutExpired) as raised:
        explorer.maybe_continue_from_coverage_checkpoint(50, 100, "s2_detail")

    assert raised.value is stalled


def _recovery_explorer(
    log: list[object],
    reads: Sequence[dict[str, object] | Exception],
    tmp_path: Path,
    router_client: OpenRouter,
    monkeypatch: pytest.MonkeyPatch,
    *,
    limit: float = DEFAULT_UNREACHABLE_LIMIT,
) -> Kobe:
    explorer = _explorer(
        _StallingDevice(log, [HOME_SCREEN, *reads]),
        tmp_path,
        router_client,
        unreachable_limit=limit,
    )
    _list_reached_from_home(explorer)
    _identifying(explorer, monkeypatch, "s0_home", "s1_list")

    def stall(_position: object, _max_steps: int) -> NoReturn:
        raise _dump_timed_out()

    monkeypatch.setattr(explorer, "_explore_step", stall)
    return explorer


@pytest.mark.usefixtures("adb_on_path")
@pytest.mark.parametrize(
    "diagnostic", ["Permission denied", "invalid option -- bad", "Can't find service: window"]
)
@pytest.mark.parametrize(
    "read", [["dumpsys", "window", "-a"], ["dumpsys", "activity", "activities"], ["wm", "size"]]
)
def test_permanent_recovery_read_failures_stop_with_command_diagnostics(
    diagnostic: str,
    read: list[str],
    tmp_path: Path,
    router_client: OpenRouter,
    no_sleep: list[float],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    log: list[object] = []
    monkeypatch.setattr(subprocess, "run", _adb_probed(log, "answer"))
    monkeypatch.setattr(time, "monotonic", lambda: sum(no_sleep))
    failure = subprocess.CalledProcessError(
        1, ["adb", "-s", "test-device", "shell", *read], stderr=diagnostic
    )
    explorer = _recovery_explorer(log, [failure], tmp_path, router_client, monkeypatch)

    with pytest.raises(subprocess.CalledProcessError) as raised:
        explorer.explore(max_steps=1)

    assert raised.value is failure
    assert log == [_PROBE]
    assert no_sleep == []
    assert any(
        " ".join(read) in line and "exit 1" in line and diagnostic in line
        for line in caplog.messages
    )


@pytest.mark.usefixtures("adb_on_path")
def test_unknown_recovery_failures_share_one_confirmation_allowance(
    tmp_path: Path,
    router_client: OpenRouter,
    no_sleep: list[float],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    log: list[object] = []
    monkeypatch.setattr(subprocess, "run", _adb_probed(log, "answer"))
    monkeypatch.setattr(time, "monotonic", lambda: sum(no_sleep))
    first = _read_failed("")
    second = subprocess.CalledProcessError(
        1, ["adb", "-s", "test-device", "exec-out", "uiautomator dump /dev/stdout"]
    )
    explorer = _recovery_explorer(
        log, [first, _dump_timed_out(), second], tmp_path, router_client, monkeypatch
    )

    with pytest.raises(subprocess.CalledProcessError) as raised:
        explorer.explore(max_steps=1)

    assert raised.value is second
    assert log == [_PROBE] * 3
    assert no_sleep == [30, 30]
    assert any("diagnostics absent" in line for line in caplog.messages)


@pytest.mark.usefixtures("adb_on_path")
def test_shell_answers_and_read_timeouts_share_one_recovery_deadline(
    tmp_path: Path,
    router_client: OpenRouter,
    no_sleep: list[float],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log: list[object] = []
    monkeypatch.setattr(subprocess, "run", _adb_probed(log, "answer"))
    monkeypatch.setattr(time, "monotonic", lambda: sum(no_sleep))
    reads = [_dump_timed_out() for _ in range(4)]
    explorer = _recovery_explorer(log, reads, tmp_path, router_client, monkeypatch, limit=45)

    with pytest.raises(DeviceUnreachableError, match=r"unreachable for 0\.75 minutes"):
        explorer.explore(max_steps=1)

    assert no_sleep == [30, 15]
    assert log == [_PROBE, _PROBE]


@pytest.mark.usefixtures("adb_on_path")
def test_recovery_identification_failure_is_outside_device_retry_policy(
    tmp_path: Path,
    router_client: OpenRouter,
    no_sleep: list[float],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log: list[object] = []
    monkeypatch.setattr(subprocess, "run", _adb_probed(log, "answer"))
    monkeypatch.setattr(time, "monotonic", lambda: sum(no_sleep))
    explorer = _recovery_explorer(log, [HOME_SCREEN], tmp_path, router_client, monkeypatch)
    failure = subprocess.TimeoutExpired("model", 1)

    identified: list[object] = []

    def identify(state: dict[str, object]) -> tuple[str, dict[str, object]]:
        identified.append(state)
        if len(identified) == 1:
            return "s0_home", state
        raise failure

    monkeypatch.setattr(explorer, "_identify", identify)
    with pytest.raises(subprocess.TimeoutExpired) as raised:
        explorer.explore(max_steps=1)

    assert raised.value is failure
    assert log == [_PROBE]
    assert no_sleep == []


@pytest.mark.usefixtures("adb_on_path")
@pytest.mark.parametrize(
    ("read_command", "stderr", "stdout", "failures", "diagnostic", "excluded", "stops"),
    [
        (
            ["shell", "dumpsys", "window", "-a"],
            b"error: closed",
            None,
            2,
            "error: closed",
            "",
            False,
        ),
        (
            ["exec-out", "uiautomator dump /dev/stdout"],
            "android.os.DeadObjectException",
            None,
            2,
            "DeadObjectException",
            "",
            False,
        ),
        (
            ["shell", "dumpsys", "window", "-a"],
            None,
            "Window{private-screen}\nerror: device offline",
            2,
            "error: device offline",
            "private-screen",
            False,
        ),
        (
            ["exec-out", "screencap -p"],
            None,
            b"PNG-private-screen",
            1,
            "diagnostics absent",
            "PNG-private-screen",
            False,
        ),
        (
            ["shell", "dumpsys", "window", "-a"],
            b"unclassified " + b"x" * 1100 + b"private-tail",
            None,
            1,
            "unclassified",
            "private-tail",
            False,
        ),
        (
            ["shell", "dumpsys", "window", "-a"],
            None,
            "Permission denied\nWindow{private-screen}",
            1,
            "Permission denied",
            "private-screen",
            True,
        ),
        (
            ["exec-out", "uiautomator dump /dev/stdout"],
            "/system/bin/sh: uiautomator: inaccessible or not found",
            None,
            1,
            "inaccessible or not found",
            "",
            True,
        ),
    ],
)
def test_recovery_uses_bounded_error_diagnostics_and_preserves_observation_payloads(
    read_command: list[str],
    stderr: str | bytes | None,
    stdout: str | bytes | None,
    failures: int,
    diagnostic: str,
    excluded: str,
    stops: bool,
    tmp_path: Path,
    router_client: OpenRouter,
    no_sleep: list[float],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    log: list[object] = []
    monkeypatch.setattr(subprocess, "run", _adb_probed(log, "answer"))
    monkeypatch.setattr(time, "monotonic", lambda: sum(no_sleep))
    failure = subprocess.CalledProcessError(
        1, ["adb", "-s", "test-device", *read_command], output=stdout, stderr=stderr
    )
    explorer = _recovery_explorer(
        log, [*[failure] * failures, HOME_SCREEN], tmp_path, router_client, monkeypatch
    )

    if stops:
        with pytest.raises(subprocess.CalledProcessError) as raised:
            explorer.explore(max_steps=1)
        assert raised.value is failure
        assert no_sleep == []
    else:
        explorer.explore(max_steps=1)
        assert no_sleep == [30] * failures

    messages = [line for line in caplog.messages if "read:" in line]
    assert len(messages) == failures
    assert all(diagnostic in line and len(line) < 1300 for line in messages)
    if excluded:
        assert excluded not in caplog.text


@pytest.mark.usefixtures("adb_on_path")
@pytest.mark.parametrize(
    "command",
    [
        [
            "adb",
            "-s",
            "test-device",
            "shell",
            "pm",
            "grant",
            "com.demo",
            "android.permission.CAMERA",
        ],
        ["adb", "-s", "other-device", "shell", "dumpsys", "window", "-a"],
        ["adb", "-s", "other-device", "shell", "dumpsys", "activity", "activities"],
        ["adb", "-s", "test-device", "shell", "input", "tap", "10", "20"],
        ["adb", "-s", "other-device", "shell", "wm", "size"],
        ["adb", "-s", "test-device", "shell", "wm", "density", "reset"],
        ["adb", "-s", "test-device", "shell", "settings", "put", "system", "font_scale", "1"],
    ],
)
def test_recovery_stops_nonzero_commands_outside_its_observation_reads(
    command: list[str],
    tmp_path: Path,
    router_client: OpenRouter,
    no_sleep: list[float],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log: list[object] = []
    monkeypatch.setattr(subprocess, "run", _adb_probed(log, "answer"))
    monkeypatch.setattr(time, "monotonic", lambda: sum(no_sleep))
    failure = subprocess.CalledProcessError(1, command, stderr="error: device offline")
    explorer = _recovery_explorer(log, [failure], tmp_path, router_client, monkeypatch)

    with pytest.raises(subprocess.CalledProcessError) as raised:
        explorer.explore(max_steps=1)

    assert raised.value is failure
    assert log == [_PROBE]
    assert no_sleep == []


@pytest.mark.parametrize("unfocused", [{"focused_field": None}, {}])
@pytest.mark.usefixtures("no_sleep")
def test_typing_plans_use_current_focus_and_keep_field_selection_atomic(
    unfocused: dict[str, object],
    tmp_path: Path,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    chat_response: Callable[..., httpx.Response],
    embedding_body: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each tab's current description supplies focus to both models before a field tap or Type."""
    device = _Screens(*[_screen("com.demo.Form", color) for color in ("white", "gray", "blue")])
    explorer = _explorer(device, tmp_path, router_client)
    monkeypatch.setattr(explorer.graph, "normalize_all_edges", lambda: 0)
    instructions = iter(["Tap Options", "Tap Note", "Type 'Grocery rewards card'"])
    actions: list[dict[str, object]] = [
        {"action": "tap", "x": 20, "y": 40},
        {"action": "tap", "x": 30, "y": 60},
        {"action": "type", "text": "Grocery rewards card"},
    ]
    proposed = iter(actions)
    plans: list[PlanRequest] = []
    grounding: list[ActionRequest] = []

    def plan(*, request: PlanRequest, **_kwargs: object) -> ActionPlan:
        plans.append(request)
        return ActionPlan(next(instructions), None, 1)

    def act(*, request: ActionRequest, **_kwargs: object) -> tuple[dict[str, object], str]:
        grounding.append(request)
        return next(proposed), ""

    monkeypatch.setattr("android_app_graph.kobe.plan_next_action", plan)
    monkeypatch.setattr("android_app_graph.kobe.predict_next_action", act)
    monkeypatch.setattr(
        "android_app_graph.kobe.soft_keyboard_hint", lambda _udid: "keyboard visible"
    )
    for index, state in enumerate(
        [{"focused_field": "Name"}, unfocused, {"focused_field": "Note"}]
    ):
        router_responses.extend(
            [
                httpx.Response(200, json=embedding_body),
                chat_response(
                    content=json.dumps(
                        {
                            "page_description": "Card editor",
                            "state": state,
                            "surface": {"kind": "page", "relation": "unknown", "return_depth": 0},
                        }
                    )
                ),
                httpx.Response(200, json=embedding_body),
            ]
        )
        if index:
            router_responses.append(chat_response(content='{"same":true,"reason":"one form"}'))
    router_responses.append(httpx.Response(200, json=embedding_body))

    explorer.explore(max_steps=3)

    middle = "null" if "focused_field" in unfocused else "unknown (not reported)"
    for plan_request, action_request, focus in zip(
        plans, grounding, ['"Name"', middle, '"Note"'], strict=True
    ):
        expected = f"Current screenshot description reports focused_field: {focus}."
        assert expected in plan_request.input_status
        assert plan_request.input_status in action_request.instruction
    assert device.actions == actions
    assert explorer.graph.total_steps_completed == 3
    assert explorer.graph.graph.nodes["s0"]["state_schema"]["focused_field"][0] == "Name"


class _TimedFeedbackDevice(_ScriptedDevice):
    def __init__(self, *screens: dict[str, object]) -> None:
        super().__init__(*screens)
        self.reads = 0

    @override
    def get_state(self, **_kwargs: float | None) -> dict[str, object]:
        state = self._screens[min(self.reads, len(self._screens) - 1)]
        self.reads += 1
        return state


@pytest.mark.usefixtures("no_sleep")
@pytest.mark.parametrize("max_steps", [1, 2])
def test_system_feedback_defers_a_covered_action_within_the_step_budget(
    kobe: Kobe, png_screenshot: str, monkeypatch: pytest.MonkeyPatch, max_steps: int
) -> None:
    app: dict[str, object] = {
        "package": "com.demo",
        "activity": "Home",
        "screenshot": png_screenshot,
    }
    feedback = {**app, "system_overlays": [[0, 100, 100, 150]]}
    device = _TimedFeedbackDevice(feedback, app, app)
    monkeypatch.setattr(kobe, "controller", _recording(device, kobe))
    kobe.graph.graph.add_node(
        "s0",
        package="com.demo",
        activity="Home",
        state_schema={},
        last_detail_snapshot={},
        observations=[],
        visit_count=1,
        replay_misses=0,
        interactable_elements=[{"description": "App Share", "position": "top", "explored": False}],
    )
    monkeypatch.setattr(kobe.graph, "identify_state", lambda *_args, **_kwargs: "s0")
    monkeypatch.setattr(kobe.graph, "latest_description", lambda _node: "Calculator")
    monkeypatch.setattr(kobe.graph, "normalize_all_edges", lambda: 0)
    monkeypatch.setattr("android_app_graph.kobe.soft_keyboard_hint", lambda _udid: "")
    pending: list[bool] = []

    def plan(*, request: PlanRequest, **_kwargs: object) -> ActionPlan:
        pending.append(
            request.unexplored_elements[0]["explored"] if request.unexplored_elements else True
        )
        return ActionPlan("Tap app Share", 1, 1)

    actions = iter([{"action": "tap", "x": 50, "y": 125}, {"action": "tap", "x": 10, "y": 20}])
    monkeypatch.setattr("android_app_graph.kobe.plan_next_action", plan)
    monkeypatch.setattr(
        "android_app_graph.kobe.predict_next_action", lambda **_kwargs: (next(actions), "")
    )
    kobe.explore(max_steps=max_steps)
    assert kobe.graph.total_steps_completed == max_steps
    assert pending == [False] * max_steps
    assert device.actions == ([] if max_steps == 1 else [{"action": "tap", "x": 10, "y": 20}])
    assert kobe.graph.graph.number_of_edges() == max_steps - 1
    assert kobe.graph.graph.nodes["s0"]["interactable_elements"][0]["explored"] is (max_steps == 2)
    records = [json.loads(path.read_text()) for path in kobe.controller.steps.glob("*.json")]
    skipped = [
        r for r in records if r["kind"] == "system_event" and r.get("status") == "action_skipped"
    ]
    assert len(skipped) == 1


@pytest.mark.usefixtures("no_sleep")
def test_replay_stops_before_a_system_feedback_target_and_records_its_actual_node(
    kobe: Kobe, monkeypatch: pytest.MonkeyPatch, device_log: list[object]
) -> None:
    _settings_and_detail(kobe)
    screen = {**HOME_SCREEN, "system_overlays": [[0, 0, 100, 100]]}
    device = _TimedFeedbackDevice(screen, screen)
    monkeypatch.setattr(kobe, "controller", _recording(device, kobe))
    monkeypatch.setattr(kobe.graph, "identify_state", lambda *_args, **_kwargs: "s0_home")
    assert kobe.navigate_to_node("s2_detail") == "s0_home"
    assert kobe.graph.graph.nodes["s2_detail"]["replay_misses"] == 1
    assert device.actions == []
    assert device_log == _FIRST_RESTART


@pytest.mark.usefixtures("no_sleep")
def test_startup_dismisses_a_focused_shade_and_retains_its_boundary_identity(
    kobe: Kobe, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = HOME_SCREEN
    shade = {
        **app,
        "package": "com.android.systemui",
        "activity": "NotificationShade",
        "system_surface": "notification_shade",
    }
    device = _TimedFeedbackDevice(shade, app)
    monkeypatch.setattr(kobe, "controller", _recording(device, kobe))
    kobe.graph.graph.add_node(
        "s0",
        package="com.demo",
        activity="Home",
        observations=[],
        state_schema={},
        last_detail_snapshot={},
        visit_count=1,
        replay_misses=0,
    )
    monkeypatch.setattr(kobe.graph, "identify_state", lambda *_args, **_kwargs: "s0")
    monkeypatch.setattr(kobe.graph, "latest_description", lambda _node: "Home")
    monkeypatch.setattr("android_app_graph.kobe.soft_keyboard_hint", lambda _udid: "")
    monkeypatch.setattr(
        "android_app_graph.kobe.plan_next_action", lambda **_kwargs: ActionPlan("Wait", None, 0)
    )
    monkeypatch.setattr(
        "android_app_graph.kobe.predict_next_action", lambda **_kwargs: ({"action": "wait"}, "")
    )
    kobe.explore(max_steps=1)
    boundary = kobe.graph.graph.nodes["ext_com.android.systemui"]
    assert boundary["system_surface"] == "notification_shade"
    assert boundary["is_external"] is True
    assert kobe.graph.get_resume_target() == "s0"
    assert device.actions == [{"action": "back"}]


@pytest.mark.usefixtures("no_sleep")
def test_an_unresolved_startup_permission_saves_its_boundary_before_stopping(
    kobe: Kobe, monkeypatch: pytest.MonkeyPatch
) -> None:
    from android_app_graph.device import UnresolvedPermissionError

    prompt = {
        **HOME_SCREEN,
        "package": "com.android.permissioncontroller",
        "activity": "GrantPermissionsActivity",
        "system_surface": "permission",
        "permission_requester": "com.android.inputmethod.latin",
    }
    device = _TimedFeedbackDevice(prompt)
    monkeypatch.setattr(kobe, "controller", _recording(device, kobe))
    with pytest.raises(UnresolvedPermissionError, match="not verified as target"):
        kobe.explore(max_steps=1)
    assert kobe.graph_path.is_file()
    assert device.actions == []
    assert list(kobe.graph.graph) == ["ext_com.android.permissioncontroller"]


@pytest.mark.usefixtures("no_sleep")
@pytest.mark.parametrize(
    "surface", ["notification_shade", "permission", "notification_shade_restart"]
)
def test_system_recovery_preserves_the_replay_expectation(
    kobe: Kobe, monkeypatch: pytest.MonkeyPatch, surface: str
) -> None:
    app = {**HOME_SCREEN, "activity": "com.demo.Detail"}
    restart = surface == "notification_shade_restart"
    boundary = {
        **app,
        "package": "com.android.systemui",
        "system_surface": "notification_shade" if restart else surface,
    }
    device = (
        _TimedFeedbackDevice(HOME_SCREEN, boundary, boundary, HOME_SCREEN)
        if restart
        else _TimedFeedbackDevice(HOME_SCREEN, boundary, app)
    )
    monkeypatch.setattr(kobe, "controller", _recording(device, kobe))
    kobe.graph.graph.add_nodes_from(["s0_home", "s1_detail"], replay_misses=0)
    kobe.graph.add_edge("s0_home", "s1_detail", action=[{"action": "tap", "x": 1, "y": 2}])
    monkeypatch.setattr("android_app_graph.kobe.restart_app", lambda *_args: None)
    monkeypatch.setattr(
        kobe.graph,
        "identify_state",
        lambda observation, **_kwargs: (
            "s1_detail" if observation.activity == "com.demo.Detail" else "s0_home"
        ),
    )
    monkeypatch.setattr(kobe.graph, "earliest_member", lambda node: node + "-observation")
    if surface == "permission":
        monkeypatch.setattr(
            "android_app_graph.kobe.resolve_permission_prompt",
            lambda controller, *_args: controller.get_state(),
        )
    monkeypatch.setattr(
        "android_app_graph.device.restart_into_app",
        lambda controller, *_args: controller.get_state(),
    )
    assert kobe.navigate_to_node("s1_detail") == ("s0_home" if restart else "s1_detail")
    matches = [json.loads(p.read_text()) for p in sorted(kobe.controller.steps.glob("*.json"))]
    assert [r["expected_observation"] for r in matches if r["kind"] == "match"] == [
        "s0_home-observation",
        "s0_home-observation" if restart else "s1_detail-observation",
    ]


@pytest.mark.parametrize("phase", ["replay", "capture"])
@pytest.mark.parametrize("failure", ["permission", "overlay"])
def test_resume_preserves_graph_evidence_when_system_recovery_stops(
    kobe: Kobe, monkeypatch: pytest.MonkeyPatch, phase: str, failure: str
) -> None:
    from android_app_graph.device import UnresolvedPermissionError
    from android_app_graph.system_surfaces import UnresolvedSystemOverlayError

    error = UnresolvedPermissionError if failure == "permission" else UnresolvedSystemOverlayError
    kobe.graph.graph.add_node("s0", replay_misses=0)

    def stop(*_args: object, **_kwargs: object) -> NoReturn:
        kobe.graph.graph.add_node("ext_boundary", is_external=True, replay_misses=0)
        msg = "unresolved boundary"
        raise error(msg)

    monkeypatch.setattr(kobe, "navigate_to_node", stop if phase == "replay" else lambda _node: "s0")
    if phase == "capture":
        monkeypatch.setattr(kobe.controller, "get_state", stop)
    with pytest.raises(error, match="unresolved boundary"):
        kobe.explore(max_steps=1, resume_from="s0")
    assert kobe.graph_path.is_file()
    assert "ext_boundary" in kobe.graph_path.read_text()


def test_an_ime_target_records_its_own_settings_as_app_coverage(
    tmp_path: Path, png_screenshot: str, router_client: OpenRouter
) -> None:
    package = "com.touchtype.swiftkey"
    device = _Device(
        {
            "activity": package + ".Settings",
            "screenshot": png_screenshot,
            "package": package,
            "system_surface": "ime_settings",
        }
    )
    explorer = Kobe(
        device,
        ExplorerConfig(
            app_name="keyboard",
            package_name=package,
            graph_dir=tmp_path,
            vlm_config={role: {"model": f"provider/{role}"} for role in _ROLES},
        ),
        client=router_client,
        logger=logging.getLogger("test"),
    )
    state = explorer.controller.get_state()
    assert state["system_surface"] is None
    assert Observation.from_state(state).system_surface is None
    record = json.loads((explorer.controller.steps / "000000.json").read_text())
    assert record["package"] == package
    assert record["system_surface"] is None


@pytest.mark.usefixtures("no_sleep")
@pytest.mark.parametrize("resume_from", ["auto", "s1_settings"])
def test_display_rejected_resume_identifies_the_current_screen_and_explores(
    kobe: Kobe,
    router_responses: list[httpx.Response],
    chat_response: Callable[..., httpx.Response],
    monkeypatch: pytest.MonkeyPatch,
    resume_from: str,
) -> None:
    kobe.graph.graph.add_node("s0_home", replay_misses=0)
    kobe.graph.graph.add_node("s1_settings", replay_misses=0)
    replayed: list[str] = []

    def replay(node_id: str) -> NoReturn:
        replayed.append(node_id)
        message = "Coordinate replay requires its recorded display state or fresh grounding"
        raise UnsafeReplayError(message)

    monkeypatch.setattr(kobe, "navigate_to_node", replay)
    monkeypatch.setattr(kobe.graph, "get_resume_target", lambda **_kwargs: "s1_settings")
    monkeypatch.setattr(kobe.graph, "identify_state", lambda *_args, **_kwargs: "s0_home")
    _describing_by_node_id(kobe, monkeypatch)
    planned_from: list[object] = []
    monkeypatch.setattr("android_app_graph.kobe.plan_next_action", _planner("Wait", planned_from))
    monkeypatch.setattr("android_app_graph.kobe.soft_keyboard_hint", lambda _udid: "")
    router_responses.append(chat_response(content=_WAIT_REPLY))

    kobe.explore(max_steps=1, resume_from=resume_from)

    assert replayed == ["s1_settings"]
    assert planned_from == ["s0_home"]
    assert kobe.graph.total_steps_completed == 1


@pytest.mark.usefixtures("no_sleep")
def test_unreachable_planned_inputs_consume_steps_and_preserve_unexplored_targets(
    tmp_path: Path,
    router_client: OpenRouter,
    monkeypatch: pytest.MonkeyPatch,
    step_records: Callable[[Path], list[dict[str, Any]]],
) -> None:
    regions = {"display": [0, 0, 720, 1280], "tap": [[0, 1184, 720, 1280]], "gesture": []}
    device = _Screens({**HOME_SCREEN, "input_regions": regions})
    explorer = _explorer(device, tmp_path, router_client)
    _list_reached_from_home(explorer)
    explorer.graph.graph.nodes["s0_home"]["interactable_elements"] = [
        {"description": "Import & Export", "explored": False}
    ]
    _identifying(explorer, monkeypatch, "s0_home")
    monkeypatch.setattr("android_app_graph.kobe.soft_keyboard_hint", lambda _udid: "")
    planning_hints: list[str] = []

    def plan(*, request: PlanRequest, **_kwargs: object) -> ActionPlan:
        planning_hints.append(request.input_status)
        return ActionPlan("Open Import & Export", 1, 1)

    monkeypatch.setattr("android_app_graph.kobe.plan_next_action", plan)
    action = {"action": "tap", "x": 257, "y": 1190}
    monkeypatch.setattr(
        "android_app_graph.kobe.predict_next_action", lambda **_kwargs: (action, "")
    )

    def reject(_action: dict[str, object]) -> None:
        reason = "App input targets a system touch region"
        raise InputRejectedError(reason, None)

    monkeypatch.setattr(device, "exe_action", reject)
    explorer.explore(max_steps=2)

    assert explorer.graph.total_steps_completed == 2
    assert len(planning_hints) == 2
    assert all("1184" not in hint for hint in planning_hints)
    assert "rejected" not in planning_hints[0]
    assert "Previous tap was rejected: App input targets a system touch region" in planning_hints[1]
    assert device.actions == []
    assert list(explorer.graph.graph.edges) == []
    assert explorer.graph.get_unexplored_elements("s0_home") == [
        {"description": "Import & Export", "explored": False}
    ]
    records = step_records(explorer.controller.steps)
    assert len([record for record in records if record["kind"] == "action_rejected"]) == 2
    assert not any(record["kind"] == "transition" for record in records)
    assert records[0]["input_regions"] == regions


@pytest.mark.usefixtures("no_sleep")
@pytest.mark.parametrize("in_setup", [False, True])
def test_a_replay_stops_at_an_unreachable_input_and_identifies_its_actual_arrival(
    in_setup: bool,
    tmp_path: Path,
    router_client: OpenRouter,
    monkeypatch: pytest.MonkeyPatch,
    device_log: list[object],
) -> None:
    device = _Screens(HOME_SCREEN)
    explorer = _explorer(device, tmp_path, router_client)
    _list_reached_from_home(explorer)
    blocked = {"action": "tap", "x": 257, "y": 1190}
    later = {"action": "tap", "x": 50, "y": 100}
    explorer.graph.add_edge(
        "s0_home",
        "s1_list",
        action=[later] if in_setup else [blocked, later],
        details=EdgeDetails(setup=[blocked] if in_setup else []),
    )
    _identifying(explorer, monkeypatch, "s0_home", "s0_home")

    def reject(_action: dict[str, object]) -> None:
        reason = "App input targets a system touch region"
        raise InputRejectedError(reason, None)

    monkeypatch.setattr(device, "exe_action", reject)
    assert explorer.navigate_to_node("s1_list") == "s0_home"
    assert device.actions == []
    assert device_log == _FIRST_RESTART
    assert explorer.graph.graph.nodes["s1_list"]["replay_misses"] == 1


@pytest.mark.usefixtures("no_sleep")
@pytest.mark.parametrize("pointers", [None, 3])
def test_unsupported_gestures_count_once_without_input_edges_or_element_credit(
    pointers: int | None,
    kobe: Kobe,
    monkeypatch: pytest.MonkeyPatch,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    chat_response: Callable[..., httpx.Response],
    step_records: Callable[[Path], list[dict[str, Any]]],
) -> None:
    instruction = "Swipe upward with three fingers through the center of the settings list"
    element = {"description": "Settings list", "position": "center", "explored": False}
    kobe.graph.graph.add_node("s0", replay_misses=0, interactable_elements=[element])
    monkeypatch.setattr(kobe.graph, "identify_state", lambda *_args, **_kwargs: "s0")
    _describing_by_node_id(kobe, monkeypatch)
    monkeypatch.setattr("android_app_graph.kobe.soft_keyboard_hint", lambda _udid: "")
    checkpoints: list[int] = []

    def checkpoint(
        step: int,
        _limit: int,
        node: str,
        state: dict[str, Any],
        *,
        preserve_arrival: bool,
    ) -> tuple[str, dict[str, Any]]:
        assert not preserve_arrival
        checkpoints.append(step)
        return node, state

    monkeypatch.setattr(kobe, "_apply_coverage_checkpoint_after_step", checkpoint)
    router_responses.extend(
        chat_response(
            content=json.dumps(
                {"instruction": instruction, "element": 1, "required_pointers": pointers}
            )
        )
        for _ in range(3)
    )
    kobe.explore(max_steps=3)
    records = step_records(kobe.controller.steps)
    unsupported = [r for r in records if r["kind"] == "unsupported_action"]
    assert len(unsupported) == 3
    assert [r["instruction"] for r in unsupported] == [instruction] * 3
    assert all(r["planned_required_pointers"] == pointers and r["reason"] for r in unsupported)
    assert all(r["observation"] is not None for r in unsupported)
    assert not any(r["kind"] in {"action", "transition"} for r in records)
    assert kobe.graph.total_steps_completed == 3
    assert checkpoints == [1, 2, 3]
    assert kobe.graph.graph.number_of_edges() == 0
    assert kobe.graph.graph.nodes["s0"]["interactable_elements"] == [element]
    assert kobe.graph_path.exists()
    assert len(router_requests) == 3
    assert (
        "Previous instruction was unsupported"
        in json.loads(router_requests[1].content)["messages"][0]["content"][0]["text"]
    )


@pytest.mark.usefixtures("no_sleep")
def test_fresh_graph_entry_establishes_observed_preparation(
    replaying_kobe: Kobe, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A new graph retains its first observed entry as known preparation state."""
    kobe = replaying_kobe
    states = iter(({"folder": "root"}, {"folder": "Pictures"}))

    def identify(state: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        kobe.graph.graph.add_node(
            "s0",
            package="com.demo",
            replay_misses=0,
            observations=[state["observation_id"]],
            last_detail_snapshot=next(states),
        )
        return "s0", state

    monkeypatch.setattr(kobe, "_identify", identify)
    _describing_by_node_id(kobe, monkeypatch)
    _explore_sending(kobe, monkeypatch, {"action": "tap", "x": 20, "y": 30})
    assert kobe.graph.graph["s0"]["s0"]["setups"] == [[]]


@pytest.mark.usefixtures("no_sleep")
@pytest.mark.parametrize("applicable", [True, False])
def test_navigation_checks_preparation_before_sending_any_recorded_input(
    replaying_kobe: Kobe,
    device_log: list[object],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    write_graph_file: Callable[..., None],
    applicable: bool,
) -> None:
    graph_file = tmp_path / "saved.json"
    write_graph_file(
        graph_file,
        {"nodes": [{"id": node} for node in ("s0_home", "s1_detail")]},
        embedding_model="provider/embedding",
    )
    graph = replaying_kobe.graph
    graph.load_graph(graph_file)
    setup = {"action": "type", "text": "Pictures"}
    action = {"action": "tap", "x": 30, "y": 40}
    anchor = Observation.from_state(replaying_kobe.controller.get_state()).observation_id
    graph.steps_dir = replaying_kobe.controller.steps
    graph.graph.nodes["s0_home"]["observations"].append(anchor)
    graph.add_edge(
        "s0_home", "s1_detail", action, details=EdgeDetails(setup=[setup], setup_observation=anchor)
    )
    _identifying(replaying_kobe, monkeypatch, "s0_home", "s1_detail" if applicable else "s0_home")
    monkeypatch.setattr(
        "android_app_graph.utils.graph_manager.verify_replay_source",
        lambda *_args, **_kwargs: (applicable, "actual entry fixture"),
    )
    reached = replaying_kobe.navigate_to_node("s1_detail")
    assert reached == ("s1_detail" if applicable else "s0_home")
    assert device_log == [*_FIRST_RESTART, *([setup, action] if applicable else [])]
    assert graph.graph.nodes["s1_detail"]["replay_misses"] == (0 if applicable else 1)


@pytest.mark.usefixtures("no_sleep")
@pytest.mark.parametrize(
    ("answers", "prepared"), [([True], False), ([False, True], True), ([False, False], None)]
)
def test_navigation_uses_the_shortest_applicable_entry_and_preserves_the_other_route(
    replaying_kobe: Kobe,
    device_log: list[object],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    write_graph_file: Callable[..., None],
    answers: list[bool],
    prepared: bool | None,
) -> None:
    graph_file = tmp_path / "saved.json"
    write_graph_file(
        graph_file,
        {"nodes": [{"id": node} for node in ("s0_home", "s1_detail")]},
        embedding_model="provider/embedding",
    )
    graph = replaying_kobe.graph
    graph.load_graph(graph_file)
    graph.steps_dir = replaying_kobe.controller.steps
    anchors = [
        Observation.from_state(replaying_kobe.controller.get_state()).observation_id
        for _ in range(2)
    ]
    graph.graph.nodes["s0_home"]["observations"].extend(anchors)
    setup = {"action": "type", "text": "Pictures"}
    action = {"action": "tap", "x": 30, "y": 40}
    graph.add_edge(
        "s0_home",
        "s1_detail",
        action,
        details=EdgeDetails(setup=[setup], setup_observation=anchors[0]),
    )
    graph.add_edge(
        "s0_home", "s1_detail", action, details=EdgeDetails(setup=[], setup_observation=anchors[1])
    )
    _identifying(
        replaying_kobe, monkeypatch, "s0_home", "s0_home" if prepared is None else "s1_detail"
    )
    replies = iter(answers)
    checked: list[object] = []

    def verify(*_args: object, **kwargs: object) -> tuple[bool, str]:
        checked.append(kwargs["actions"])
        assert device_log == _FIRST_RESTART
        return next(replies), "entry fixture"

    monkeypatch.setattr("android_app_graph.utils.graph_manager.verify_replay_source", verify)
    reached = replaying_kobe.navigate_to_node("s1_detail")
    assert reached == ("s0_home" if prepared is None else "s1_detail")
    assert len(checked) == len(answers)
    expected = [] if prepared is None else [setup, action] if prepared else [action]
    assert device_log == [*_FIRST_RESTART, *expected]
