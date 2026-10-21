"""Tests for android_app_graph.kobe: the shared client, model setup and exploration recovery."""

from __future__ import annotations

import base64
import io
import json
import logging
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any, NoReturn, override

import httpx
import pytest
from openrouter import OpenRouter
from openrouter.errors import OpenRouterError
from PIL import Image

from android_app_graph.device import DeviceController, StillOutsideAppError
from android_app_graph.kobe import ExplorerConfig, Kobe, NoPathError
from android_app_graph.screen_image import encode_screenshot
from android_app_graph.step_log import Observation, StepRecorder
from android_app_graph.utils.graph_manager import EXTERNAL_WEB_NODE
from android_app_graph.utils.vlm_utils import PlanRequest

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
_TAP_REPLY = '{"action":"click","coordinate":[500,500]}'


def _screen(activity: str, color: str) -> dict[str, object]:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), color).save(buffer, format="PNG")
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

    def get_state(self) -> dict[str, object]:
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

    def get_state(self) -> dict[str, object]:
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


def _explorer(
    device: DeviceController,
    tmp_path: Path,
    router_client: OpenRouter,
    *,
    launch_activity: str | None = None,
) -> Kobe:
    return Kobe(
        device,
        ExplorerConfig(
            app_name="demo",
            package_name="com.demo",
            graph_dir=tmp_path,
            vlm_config={role: {"model": f"provider/{role}"} for role in _ROLES},
            launch_activity=launch_activity,
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
    assert agent.graph.embedding_model == "provider/embedding"
    assert agent.graph.image_embedding == agent.graph.embedding
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
        ExplorerConfig(app_name="demo", package_name="com.demo", graph_dir=tmp_path),
        client=router_client,
        logger=logging.getLogger("test"),
    )
    built.explore(max_steps=0)
    assert router_client.sdk_configuration.client is router_http


@pytest.mark.parametrize("saved_package", [None, "com.stale"])
def test_exploration_records_the_configured_package_name(
    saved_package: str | None,
    tmp_path: Path,
    png_screenshot: str,
    router_client: OpenRouter,
) -> None:
    """The configured package wins over whatever, if anything, an earlier graph saved."""
    if saved_package is not None:
        graph_path = tmp_path / "demo" / "demo.json"
        graph_path.parent.mkdir()
        graph_path.write_text(
            json.dumps({"package_name": saved_package, "nodes": [], "edges": []}),
            encoding="utf-8",
        )
    built = Kobe(
        _Device({"activity": "com.demo.Home", "screenshot": png_screenshot}),
        ExplorerConfig(
            app_name="demo",
            package_name="com.demo",
            graph_dir=tmp_path,
            vlm_config={role: {"model": f"provider/{role}"} for role in _ROLES},
        ),
        client=router_client,
        logger=logging.getLogger("test"),
    )
    built.save_graph()
    saved = json.loads((tmp_path / "demo" / "demo.json").read_text(encoding="utf-8"))
    assert saved["package_name"] == "com.demo"


class _ReplayedError(Exception):
    """Ends exploration once it has chosen the node to replay to."""


@pytest.mark.parametrize("saved_package", [None, "com.whatsapp"])
def test_resume_from_auto_replays_to_a_node_of_a_graph_saved_without_node_packages(
    saved_package: str | None,
    tmp_path: Path,
    router_client: OpenRouter,
    monkeypatch: pytest.MonkeyPatch,
    write_graph_file: Callable[[Path, dict[str, object]], None],
    write_image_vectors: Callable[..., None],
) -> None:
    """A graph whose nodes record no package resumes at a node of ``com.whatsapp``,
    although no activity's first three dot-segments equal the package."""
    document: dict[str, object] = {
        "nodes": [
            {"id": "s0_home", "activity": "com.whatsapp.home.ui.HomeActivity"},
            {"id": "s1_conversation", "activity": "com.whatsapp.Conversation"},
        ],
        "edges": [{"source": "s0_home", "target": "s1_conversation"}],
    }
    if saved_package is not None:
        document["package_name"] = saved_package
    graph_path = tmp_path / "whatsapp" / "whatsapp.json"
    graph_path.parent.mkdir()
    write_graph_file(graph_path, document)
    write_image_vectors(
        graph_path.parent / "steps",
        ["s0_home-observation", "s1_conversation-observation"],
        model="provider/embedding",
    )
    explorer = Kobe(
        _Device(AssertionError("the replay must start before the device is read")),
        ExplorerConfig(
            app_name="whatsapp",
            package_name="com.whatsapp",
            graph_dir=tmp_path,
            vlm_config={role: {"model": f"provider/{role}"} for role in _ROLES},
        ),
        client=router_client,
        logger=logging.getLogger("test"),
    )
    replayed: list[str] = []

    def replay(node_id: str) -> str:
        replayed.append(node_id)
        raise _ReplayedError

    monkeypatch.setattr(explorer, "navigate_to_node", replay)

    with pytest.raises(_ReplayedError):
        explorer.explore(max_steps=1, resume_from="auto")

    assert replayed == ["s1_conversation"]


def test_resume_from_auto_replays_to_the_top_ranked_exploration_target(
    tmp_path: Path,
    router_client: OpenRouter,
    monkeypatch: pytest.MonkeyPatch,
    write_graph_file: Callable[[Path, dict[str, object]], None],
    write_image_vectors: Callable[..., None],
) -> None:
    """``auto`` resumes at the form with unexplored fields, not at the dialog with no
    out-edges and nothing left to explore."""
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
            {"source": "s0_home", "target": "s1_form"},
            {"source": "s0_home", "target": "s2_dialog"},
            {"source": "s1_form", "target": "s0_home"},
        ],
    }
    graph_path = tmp_path / "demo" / "demo.json"
    graph_path.parent.mkdir()
    write_graph_file(graph_path, document)
    write_image_vectors(
        graph_path.parent / "steps",
        [f"{node}-observation" for node in ("s0_home", "s1_form", "s2_dialog")],
        model="provider/embedding",
    )
    explorer = _explorer(
        _Device(AssertionError("the replay must start before the device is read")),
        tmp_path,
        router_client,
    )
    replayed: list[str] = []

    def replay(node_id: str) -> str:
        replayed.append(node_id)
        raise _ReplayedError

    monkeypatch.setattr(explorer, "navigate_to_node", replay)

    with pytest.raises(_ReplayedError):
        explorer.explore(max_steps=1, resume_from="auto")

    assert replayed == ["s1_form"]


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
            chat_response(content='{"page_description":"home"}'),
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
            chat_response(content='{"page_description":"home"}'),
            httpx.Response(200, json=embedding_body),
            # The replay ends on the start node's own pixels: an exact match.
            httpx.Response(200, json=embedding_body),
        ]
    )
    node = replaying_kobe.graph.identify_state(
        Observation.from_state(replaying_kobe.controller.get_state())
    )
    assert replaying_kobe.navigate_to_node(node) == node
    assert len(router_requests) == 4
    assert device_log == _FIRST_RESTART
    (*_, replayed) = step_records(replaying_kobe.controller.steps)
    assert (replayed["source"], replayed["identify"]["stage"]) == ("navigation_replay", "exact")
    assert replayed["identify"]["expected_node"] == node


@pytest.mark.usefixtures("no_sleep")
def test_navigation_restarts_the_app_through_its_launch_activity(
    tmp_path: Path,
    router_client: OpenRouter,
    device_log: list[object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = _ReplayDevice(HOME_SCREEN, device_log)
    explorer = _explorer(device, tmp_path, router_client, launch_activity="com.demo/.Main")
    explorer.graph.graph.add_node("s0_home")
    monkeypatch.setattr(explorer.graph, "identify_state", lambda *_args, **_kwargs: "s0_home")
    assert explorer.navigate_to_node("s0_home") == "s0_home"
    assert device_log == [_FORCE_STOP, _clear_task_start("com.demo/.Main")]


def test_navigation_without_a_recorded_path_leaves_the_app_running(
    replaying_kobe: Kobe, device_log: list[object]
) -> None:
    replaying_kobe.graph.graph.add_nodes_from(["s0_home", "s1_orphan"])
    with pytest.raises(ValueError, match="No path from s0_home to s1_orphan"):
        replaying_kobe.navigate_to_node("s1_orphan")
    assert device_log == []


@pytest.mark.usefixtures("no_sleep")
def test_navigation_sends_nothing_for_an_edge_without_actions_and_replays_the_rest(
    replaying_kobe: Kobe,
    device_log: list[object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    write_graph_file: Callable[[Path, dict[str, object]], None],
) -> None:
    """A loaded graph file can hold an edge with no actions; the replay passes over it."""
    open_detail = {"action": "tap", "coordinate": [30, 40]}
    graph_file = tmp_path / "hand_edited.json"
    write_graph_file(
        graph_file,
        {
            "nodes": [{"id": node} for node in ("s0_splash", "s1_home", "s2_detail")],
            "edges": [
                {"source": "s0_splash", "target": "s1_home", "actions": []},
                {"source": "s1_home", "target": "s2_detail", "actions": [[open_detail]]},
            ],
        },
    )
    replaying_kobe.graph.load_graph(graph_file)
    monkeypatch.setattr(
        replaying_kobe.graph, "identify_state", lambda *_args, **_kwargs: "s2_detail"
    )

    assert replaying_kobe.navigate_to_node("s2_detail") == "s2_detail"
    assert device_log == [*_FIRST_RESTART, open_detail]


@pytest.mark.usefixtures("no_sleep")
def test_navigation_replays_an_edge_action_recorded_as_a_single_dict(
    replaying_kobe: Kobe, device_log: list[object], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An edge action stored as one dict, not a list of steps, reaches the device as it is."""
    open_detail = {"action": "tap", "coordinate": [30, 40]}
    replaying_kobe.graph.graph.add_nodes_from(["s0_home", "s1_detail"])
    replaying_kobe.graph.add_edge("s0_home", "s1_detail", action=open_detail)
    monkeypatch.setattr(
        replaying_kobe.graph, "identify_state", lambda *_args, **_kwargs: "s1_detail"
    )

    assert replaying_kobe.navigate_to_node("s1_detail") == "s1_detail"
    assert device_log == [*_FIRST_RESTART, open_detail]


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
            chat_response(content='{"page_description":"home"}'),
            httpx.Response(200, json=embedding_body),
            chat_response(content="Tap Search"),
            chat_response(content='{"action":"wait"}'),
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
            chat_response(content='{"page_description":"home"}'),
            httpx.Response(200, json=embedding_body),
            chat_response(content='{"element":null,"instruction":"Tap the About row"}'),
            chat_response(content=_TAP_REPLY),
            httpx.Response(200, json=embedding_body),
            chat_response(
                content='{"page_description":"About page with privacy policy link",'
                '"is_external_web":false}'
            ),
            httpx.Response(200, json={**embedding_body, "data": [orthogonal]}),
            chat_response(content=_DIFFERENT),
        ]
    )

    explorer.explore(max_steps=1)

    assert [action["action"] for action in device.actions] == ["tap"]
    assert list(explorer.graph.graph.edges) == [
        ("s0_home", "s1_about_page_with_privacy_policy_link")
    ]


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
    """Each screen read is a WebP image and an observation record, an action names the
    observation before it, and the step's edge is a transition naming both screens, the
    planner's instruction and element, the action and the nodes the screens became."""
    monkeypatch.setattr("android_app_graph.kobe.soft_keyboard_hint", lambda _udid: "")
    explorer = _explorer(
        _Screens(HOME_SCREEN, _screen("com.demo.SettingsActivity", "gray")),
        tmp_path,
        router_client,
    )
    home = {"page_description": "home", "elements": [{"description": "About", "position": "top"}]}
    router_responses.extend(
        [
            httpx.Response(200, json=embedding_body),
            chat_response(content=json.dumps(home)),
            httpx.Response(200, json=embedding_body),
            chat_response(content='{"element":1,"instruction":"Tap the About row"}'),
            chat_response(content=_TAP_REPLY),
            httpx.Response(200, json=embedding_body),
            chat_response(content='{"page_description":"about"}'),
            httpx.Response(200, json={**embedding_body, "data": [_ORTHOGONAL]}),
            chat_response(content=_DIFFERENT),
        ]
    )

    explorer.explore(max_steps=1)

    steps_dir = tmp_path / "demo" / "steps"
    records = step_records(steps_dir)
    assert [(record["id"], record["kind"]) for record in records] == [
        ("000000", "observation"),
        ("000001", "action"),
        ("000002", "observation"),
        ("000003", "transition"),
    ]
    assert {(record["run_kind"], record["source"]) for record in records} == {
        ("explore", "main_loop")
    }
    assert sorted(path.name for path in steps_dir.glob("*.webp")) == ["000000.webp", "000002.webp"]
    tap = {"action": "tap", "x": 50, "y": 100}
    assert (records[1]["observation"], records[1]["action"]) == ("000000", tap)
    transition = {key: records[3][key] for key in ("before", "after", "instruction", "element")}
    assert transition == {
        "before": "000000",
        "after": "000002",
        "instruction": "Tap the About row",
        "element": 1,
    }
    assert (records[3]["action"], records[3]["from_node"], records[3]["to_node"]) == (
        tap,
        "s0_home",
        "s1_about",
    )
    assert explorer.graph.graph.nodes["s1_about"]["observations"] == ["000002"]


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
    back press off the page and the screen read after it are external-web recovery."""
    monkeypatch.setattr("android_app_graph.kobe.soft_keyboard_hint", lambda _udid: "")
    browser = _screen("com.demo.ArticleActivity", "blue")
    explorer = _explorer(_Screens(HOME_SCREEN, browser, HOME_SCREEN), tmp_path, router_client)
    router_responses.extend(
        [
            httpx.Response(200, json=embedding_body),
            chat_response(content='{"page_description":"home"}'),
            httpx.Response(200, json=embedding_body),
            chat_response(content='{"element":null,"instruction":"Tap the privacy policy link"}'),
            chat_response(content=_TAP_REPLY),
            httpx.Response(200, json=embedding_body),
            chat_response(content='{"page_description":"news article","is_external_web":true}'),
            # Back shows home again: an exact match.
            httpx.Response(200, json=embedding_body),
        ]
    )

    explorer.explore(max_steps=1)

    records = step_records(explorer.controller.steps)
    assert [(record["kind"], record["source"]) for record in records] == [
        ("observation", "main_loop"),
        ("action", "main_loop"),
        ("observation", "main_loop"),
        ("transition", "main_loop"),
        ("action", "external_web_recovery"),
        ("observation", "external_web_recovery"),
    ]
    assert (records[3]["after"], records[3]["to_node"]) == ("000002", EXTERNAL_WEB_NODE)
    assert explorer.graph.graph.nodes[EXTERNAL_WEB_NODE]["observations"] == ["000002"]


@pytest.mark.parametrize(
    ("action_reply", "executed", "recorded"),
    [
        (_TAP_REPLY, "tap", "Tap the privacy policy link"),
        ('{"action":"end"}', "back", "Press back"),
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
    monkeypatch.setattr("android_app_graph.kobe.soft_keyboard_hint", lambda _udid: "")
    browser = _screen("com.demo.ArticleActivity", "blue")
    device = _Screens(HOME_SCREEN, browser, HOME_SCREEN)
    explorer = _explorer(device, tmp_path, router_client)
    router_responses.extend(
        [
            httpx.Response(200, json=embedding_body),
            chat_response(content='{"page_description":"home"}'),
            httpx.Response(200, json=embedding_body),
            chat_response(content='{"element":null,"instruction":"Tap the privacy policy link"}'),
            chat_response(content=action_reply),
            httpx.Response(200, json=embedding_body),
            chat_response(content='{"page_description":"news article","is_external_web":true}'),
            httpx.Response(200, json=embedding_body),
        ]
    )

    explorer.explore(max_steps=1)

    assert set(explorer.graph.graph) == {"s0_home", EXTERNAL_WEB_NODE}
    assert list(explorer.graph.graph.edges) == [("s0_home", EXTERNAL_WEB_NODE)]
    assert explorer.graph.graph.edges["s0_home", EXTERNAL_WEB_NODE]["instructions"] == [recorded]
    assert [action["action"] for action in device.actions] == [executed, "back"]


_BACK = {"action": "back"}


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
    external = '{"page_description":"news article","is_external_web":true}'
    orthogonal = {"object": "embedding", "index": 0, "embedding": [0.75, -0.25]}
    router_responses.extend(
        [
            httpx.Response(200, json=embedding_body),
            chat_response(content='{"page_description":"home"}'),
            httpx.Response(200, json=embedding_body),
            chat_response(content='{"element":null,"instruction":"Tap the privacy policy link"}'),
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
            chat_response(content='{"element":null,"instruction":"Tap Settings"}'),
            chat_response(content=_TAP_REPLY),
            httpx.Response(200, json=embedding_body),
            chat_response(content='{"page_description":"settings"}'),
            httpx.Response(200, json={**embedding_body, "data": [orthogonal]}),
            chat_response(content=_DIFFERENT),
        ]
    )

    explorer.explore(max_steps=2)

    assert set(explorer.graph.graph.edges) == {
        ("s0_home", EXTERNAL_WEB_NODE),
        ("s0_home", "s1_settings"),
    }
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
        page_description="home",
        interactable_elements=[
            {"description": "Turn on dark mode toggle", "position": "top", "explored": False},
            {"description": "Compose FAB", "position": "bottom right", "explored": False},
        ],
    )
    kobe.graph.graph.add_node("s1_compose", page_description="compose")
    screens = iter(["s0_home", "s1_compose"])
    monkeypatch.setattr(kobe.graph, "identify_state", lambda *_args, **_kwargs: next(screens))
    monkeypatch.setattr(kobe.controller, "exe_action", lambda _action: None)
    monkeypatch.setattr("android_app_graph.kobe.soft_keyboard_hint", lambda _udid: "")
    instruction = "Tap the floating action button in the bottom right"
    router_responses.extend(
        [
            chat_response(content=json.dumps({"element": element, "instruction": instruction})),
            chat_response(content="", tool_arguments='{"action":"click","coordinate":[900,900]}'),
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

    kobe.graph.graph.add_node("s0_home", page_description="home", interactable_elements=[])
    monkeypatch.setattr(kobe.graph, "identify_state", lambda *_args, **_kwargs: "s0_home")
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

    def get_state(self) -> dict[str, object]:
        return self._screens[len(self.actions)]

    def exe_action(self, action: dict[str, object]) -> None:
        self.actions.append(action)


def _recording(device: DeviceController, kobe: Kobe) -> StepRecorder:
    """Record ``device`` in the step log of ``kobe``, as Kobe records the device it is given."""
    return StepRecorder(device, kobe.controller.steps, run_kind="explore", source="main_loop")


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
    }
    plan = {"element": 2, "instruction": "Tap the Settings icon"}
    router_responses.extend(
        [
            httpx.Response(200, json=embedding_body),
            chat_response(content=json.dumps(page)),
            httpx.Response(200, json=embedding_body),
            chat_response(content=json.dumps(plan)),
            chat_response(content='{"action":"end"}'),
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
    """The other app's node takes the screen the step reached as its member, the transition
    leads to that node, and the back press home and the screens read after it, before or
    after a restart, are external-app recovery."""
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
        ("action", "main_loop"),
        ("observation", "main_loop"),
        ("transition", "main_loop"),
        ("action", "external_app_recovery"),
        ("observation", "external_app_recovery"),
        *([("observation", "external_app_recovery")] if relaunched else []),
    ]
    assert (records[3]["after"], records[3]["to_node"]) == ("000002", "ext_com.android.launcher3")
    assert kobe.graph.graph.nodes["ext_com.android.launcher3"]["observations"] == ["000002"]
    restart = _FIRST_RESTART if relaunched else []
    assert device_log == [_BACK, _BACK, *restart]


@pytest.mark.usefixtures("no_sleep")
def test_a_step_is_recorded_on_the_node_identify_state_renamed_while_leaving_it(
    tmp_path: Path,
    two_screen_device: DeviceController,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    chat_response: Callable[..., httpx.Response],
    embedding_body: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The explorer acts from the node's new ID after identify_state renames it, so the
    step's edge is recorded and its element marked explored."""
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
    }
    verdict = {
        "same": False,
        "reason": "a query has been entered",
        "existing_description": "Chat search page with recent searches",
        "new_description": "Chat search results page",
    }
    router_responses.extend(
        [
            httpx.Response(200, json=embedding_body),
            chat_response(content=json.dumps(search_page)),
            httpx.Response(200, json=embedding_body),
            chat_response(content='{"element":1,"instruction":"Tap the Search button"}'),
            chat_response(content="", tool_arguments='{"action":"click","coordinate":[500,500]}'),
            httpx.Response(200, json=embedding_body),
            chat_response(content='{"page_description":"Chat search results"}'),
            httpx.Response(200, json=embedding_body),
            chat_response(content=json.dumps(verdict)),
            *[httpx.Response(200, json=embedding_body) for _ in range(2)],
        ]
    )

    with caplog.at_level(logging.WARNING, logger="android_app_graph"):
        agent.explore(max_steps=1)

    renamed, results = "s0_chat_search_page_with_recent_searches", "s1_chat_search_results_page"
    assert agent.graph.graph.has_edge(renamed, results)
    assert agent.graph.graph.nodes[renamed]["interactable_elements"] == [
        {"description": "Search button", "position": "top", "explored": True}
    ]
    assert "add_edge: missing node(s)" not in caplog.text


_WAIT_REPLY = '{"action":"wait"}'


def _planner(instruction: str, planned_from: list[object]) -> Callable[..., tuple[str, int | None]]:
    """A ``plan_next_action`` that records the page description of each screen it plans from."""

    def plan(*, request: PlanRequest, **_kwargs: object) -> tuple[str, int | None]:
        planned_from.append(request.page_description)
        return instruction, None

    return plan


@pytest.mark.usefixtures("no_sleep")
def test_resuming_from_a_node_plans_the_first_step_from_the_node_the_replay_reached(
    kobe: Kobe,
    router_responses: list[httpx.Response],
    chat_response: Callable[..., httpx.Response],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A replay that arrives at another node continues from where it arrived."""
    kobe.graph.graph.add_node("s0_home", page_description="home")
    kobe.graph.graph.add_node("s1_settings", page_description="settings")
    replayed: list[str] = []

    def replay(node_id: str) -> str:
        replayed.append(node_id)
        return "s0_home"

    planned_from: list[object] = []
    monkeypatch.setattr(kobe, "navigate_to_node", replay)
    monkeypatch.setattr("android_app_graph.kobe.plan_next_action", _planner("Wait", planned_from))
    monkeypatch.setattr("android_app_graph.kobe.soft_keyboard_hint", lambda _udid: "")
    router_responses.append(chat_response(content=_WAIT_REPLY))

    kobe.explore(max_steps=1, resume_from="s1_settings")

    assert replayed == ["s1_settings"]
    assert planned_from == ["home"]


@pytest.mark.usefixtures("no_sleep")
def test_resuming_from_a_node_missing_from_the_graph_starts_from_the_current_screen(
    kobe: Kobe,
    router_responses: list[httpx.Response],
    chat_response: Callable[..., httpx.Response],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    kobe.graph.graph.add_node("s0_home", page_description="home")
    planned_from: list[object] = []
    monkeypatch.setattr(kobe.graph, "identify_state", lambda *_args, **_kwargs: "s0_home")
    monkeypatch.setattr("android_app_graph.kobe.plan_next_action", _planner("Wait", planned_from))
    monkeypatch.setattr("android_app_graph.kobe.soft_keyboard_hint", lambda _udid: "")
    router_responses.append(chat_response(content=_WAIT_REPLY))

    kobe.explore(max_steps=1, resume_from="s9_removed")

    assert "Node s9_removed not found in graph. Starting from current screen." in caplog.text
    assert planned_from == ["home"]


@pytest.mark.parametrize("interruption", [RuntimeError, KeyboardInterrupt])
def test_a_resumed_exploration_that_stops_saves_and_logs_the_steps_it_completed(
    interruption: type[BaseException],
    kobe: Kobe,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A step that raises is not counted, so a re-run repeats it."""
    kobe.graph.graph.add_node("s0_home", page_description="home")
    kobe.graph.total_steps_completed = 3

    def stop_planning(**_kwargs: object) -> tuple[str, int | None]:
        msg = "planning stopped"
        raise interruption(msg)

    monkeypatch.setattr(kobe.graph, "identify_state", lambda *_args, **_kwargs: "s0_home")
    monkeypatch.setattr("android_app_graph.kobe.plan_next_action", stop_planning)
    monkeypatch.setattr("android_app_graph.kobe.soft_keyboard_hint", lambda _udid: "")
    caplog.set_level(logging.INFO, logger="test")

    with pytest.raises(interruption, match="planning stopped"):
        kobe.explore(max_steps=5)

    assert "Resuming exploration from step 3 (target: 5 steps total)" in caplog.text
    assert json.loads(kobe.graph_path.read_text())["total_steps_completed"] == 3
    assert "at step 3" in caplog.text


@pytest.mark.parametrize(
    ("detail_after", "delta", "observation"),
    [
        ({"sort": "price"}, {"sort": {"before": "name", "after": "price"}}, "home (sort=price)"),
        ({"sort": None}, {"sort": {"before": "name", "after": None}}, "home"),
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
    graph.add_node("s0_home", page_description="home", last_detail_snapshot={"sort": "name"})
    details = iter([{"sort": "name"}, detail_after])

    def identify(*_args: object, **_kwargs: object) -> str:
        graph.nodes["s0_home"]["last_detail_snapshot"] = next(details)
        return "s0_home"

    monkeypatch.setattr(explorer.graph, "identify_state", identify)
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
    explorer.graph.graph.add_node("s0_home", page_description="home")
    monkeypatch.setattr(explorer.graph, "identify_state", lambda *_args, **_kwargs: "s0_home")
    monkeypatch.setattr("android_app_graph.kobe.soft_keyboard_hint", lambda _udid: "")
    for _ in range(2):
        router_responses.extend(
            [
                chat_response(content='{"element":null,"instruction":"Tap Sort"}'),
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
    kobe.graph.graph.add_edge("s0_home", "s1_settings")
    router_responses.extend(
        [
            chat_response(content='{"issues":[],"summary":"no duplicates"}'),
            chat_response(content='{"node_id":"s1_settings","reason":"unexplored"}'),
        ]
    )
    monkeypatch.setattr(kobe, "navigate_to_node", lambda node: node)
    assert kobe.maybe_continue_from_coverage_checkpoint(50, 100, "s0_home") == "s1_settings"
    assert [json.loads(request.content)["model"] for request in router_requests] == [
        "provider/page_detail"
    ] * 2


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
    )
    for node_id in ("s1_form", "s2_form", "s3_form"):
        graph.add_node(
            node_id,
            interactable_elements=[
                {"description": description, "explored": False}
                for description in ("Name", "Address", "Port", "Username", "Password")
            ],
        )
        graph.add_edge("s0_home", node_id)
    kobe.coverage_checkpoint_top_k = 2
    router_responses.extend(
        [
            chat_response(content='{"issues":[],"summary":"no duplicates"}'),
            chat_response(content='{"node_id":"s0_home","reason":"top-level sections"}'),
        ]
    )
    monkeypatch.setattr(kobe, "navigate_to_node", lambda node: node)

    assert kobe.maybe_continue_from_coverage_checkpoint(50, 100, "s1_form") == "s0_home"

    bodies = [json.loads(request.content) for request in router_requests]
    assert [body["response_format"]["json_schema"]["name"] for body in bodies] == [
        "node_merge_audit",
        "exploration_target",
    ]
    assert "node_id=s0_home" in bodies[1]["messages"][0]["content"]


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
    replaying_kobe.graph.graph.add_nodes_from(["s0_home", "s1_settings", "s2_detail"])
    replaying_kobe.graph.add_edge("s0_home", "s1_settings", action=[open_settings])
    replaying_kobe.graph.add_edge("s1_settings", "s2_detail", action=[open_detail])
    router_responses.extend(
        [
            chat_response(content='{"issues":[],"summary":"no duplicates"}'),
            chat_response(content='{"node_id":"s1_settings","reason":"unexplored"}'),
        ]
    )
    monkeypatch.setattr(
        replaying_kobe.graph, "identify_state", lambda *_args, **_kwargs: "s1_settings"
    )
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
    add_representative: Callable[..., None],
) -> None:
    """A checkpoint that does not navigate hands back the ID its merge audit kept, not the
    one it removed, so the next step acts from a node still in the graph."""
    graph = replaying_kobe.graph
    for node_id in ("s0_list", "s1_list_dup"):
        graph.graph.add_node(node_id, page_description="Item list")
        add_representative(graph, node_id)
    graph.graph.add_node("s2_detail", page_description="Item detail")
    screens = iter(["s1_list_dup", "s1_list_dup", "s2_detail"])
    monkeypatch.setattr(graph, "identify_state", lambda *_args, **_kwargs: next(screens))
    replaying_kobe.coverage_checkpoint_steps = 1
    if navigation_fails:
        graph.graph.add_edge("s1_list_dup", "s2_detail")
        monkeypatch.setattr(replaying_kobe, "navigate_to_node", _fail_navigation)
    duplicate = {"type": "merge_nodes", "node_a": "s0_list", "node_b": "s1_list_dup"}
    target = '{"node_id":"s2_detail","reason":"unexplored"}'
    click = '{"action":"click","coordinate":[500,500]}'
    router_responses.extend(
        [
            chat_response(content='{"element":null,"instruction":"Tap the first item"}'),
            chat_response(content="", tool_arguments=click),
            chat_response(content=json.dumps({"issues": [duplicate], "summary": "duplicate"})),
            chat_response(content='{"same":true,"reason":"same list"}'),
            *([chat_response(content=target)] if navigation_fails else []),
            chat_response(content='{"element":null,"instruction":"Open the first item"}'),
            chat_response(content="", tool_arguments=click),
        ]
    )

    replaying_kobe.explore(max_steps=2)

    assert graph.graph["s0_list"]["s2_detail"]["instructions"] == ["Open the first item"]


def test_a_checkpoint_merge_keeps_the_screen_the_device_is_on_out_of_the_candidates(
    kobe: Kobe,
    router_responses: list[httpx.Response],
    chat_response: Callable[..., httpx.Response],
    monkeypatch: pytest.MonkeyPatch,
    add_representative: Callable[..., None],
) -> None:
    """The checkpoint leaves the node the device is on out of its candidates, so it never
    replays the device to the screen it is already on."""
    for node_id in ("s1_list", "s2_list_dup"):
        kobe.graph.graph.add_node(node_id, page_description="Item list")
        add_representative(kobe.graph, node_id)
    kobe.graph.graph.add_edge("s0_home", "s1_list")
    duplicate = {"type": "merge_nodes", "node_a": "s1_list", "node_b": "s2_list_dup"}
    router_responses.extend(
        [
            chat_response(content=json.dumps({"issues": [duplicate], "summary": "duplicate"})),
            chat_response(content='{"same":true,"reason":"same list"}'),
            chat_response(content='{"node_id":"s1_list","reason":"unexplored"}'),
        ]
    )
    monkeypatch.setattr(kobe, "navigate_to_node", lambda node: node)
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

    def get_state(self) -> dict[str, object]:
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


_NO_DUPLICATES = '{"issues":[],"summary":"no duplicates"}'


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
    on: the next step plans from that screen's node and screenshot. The step log tells
    the replay's actions from the checkpoint's own screen reads."""
    monkeypatch.setattr("android_app_graph.kobe.soft_keyboard_hint", lambda _udid: "")
    open_list = {"action": "tap", "coordinate": [10, 20]}
    open_item: dict[str, object] = {"action": "tap", "coordinate": [30, 40]}
    home, settings = _in_app("Home", "white"), _in_app("Settings", "gray")
    item_list, sort_menu = _in_app("List", "blue"), _in_app("Sort", "green")
    # The checkpoint reads the list screen once to identify it and once for the next step.
    device = _ReplayScript(
        device_log, home, settings, item_list, item_list, sort_menu, broken=open_item
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
    graph.graph.add_nodes_from([*nodes.values(), "s2_detail"], page_description="a screen")
    graph.add_edge("s0_home", "s1_list", action=[open_list])
    graph.add_edge("s1_list", "s2_detail", action=[open_item])
    monkeypatch.setattr(
        graph, "identify_state", lambda observation, **_kwargs: nodes[observation.activity]
    )
    router_responses.extend(
        [
            chat_response(content='{"element":null,"instruction":"Open settings"}'),
            chat_response(content=_TAP_REPLY),
            chat_response(content=_NO_DUPLICATES),
            chat_response(content='{"node_id":"s2_detail","reason":"unexplored"}'),
            chat_response(content='{"element":null,"instruction":"Open the sort menu"}'),
            chat_response(content=_TAP_REPLY),
        ]
    )

    explorer.explore(max_steps=2)

    assert list(graph.graph.in_edges("s4_sort_menu")) == [("s1_list", "s4_sort_menu")]
    list_webp = base64.b64encode(encode_screenshot(str(item_list["screenshot"]))).decode("ascii")
    list_image = f"data:image/webp;base64,{list_webp}"
    assert all(list_image in request.content.decode() for request in router_requests[-2:])
    tap = {"action": "tap", "x": 50, "y": 100}
    assert device_log == [tap, *_FIRST_RESTART, open_list, open_item, tap]
    step = [("action", "main_loop"), ("observation", "main_loop"), ("transition", "main_loop")]
    assert [(r["kind"], r["source"]) for r in step_records(explorer.controller.steps)] == [
        ("observation", "main_loop"),
        *step,
        ("action", "navigation_replay"),
        ("action", "navigation_replay"),
        ("observation", "coverage_checkpoint"),
        ("observation", "coverage_checkpoint"),
        *step,
    ]


@pytest.mark.usefixtures("no_sleep")
def test_a_checkpoint_whose_launch_fails_restarts_the_app_once_more_and_continues_there(
    tmp_path: Path,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    chat_response: Callable[..., httpx.Response],
    device_log: list[object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A launch that fails after the force-stop, through the launcher intent too, leaves
    the launcher on the screen."""
    launcher = {
        **_screen("com.android.launcher3.Launcher", "black"),
        "package": "com.android.launcher3",
    }
    device = _ReplayScript(device_log, launcher, _in_app("Home", "white"))
    explorer = _explorer(device, tmp_path, router_client, launch_activity="com.demo/.Main")
    explorer.graph.graph.add_edge("s0_home", "s1_settings")
    explorer.graph.graph.add_edge("s1_settings", "s2_detail")
    router_responses.extend(
        [
            chat_response(content=_NO_DUPLICATES),
            chat_response(content='{"node_id":"s1_settings","reason":"unexplored"}'),
        ]
    )
    main_start = _clear_task_start("com.demo/.Main")
    monkeypatch.setattr(
        subprocess, "run", _adb_refusing_once(device_log, main_start, _PACKAGE_LAUNCH)
    )
    nodes = {"com.demo.Home": "s0_home"}
    monkeypatch.setattr(
        explorer.graph, "identify_state", lambda observation, **_kwargs: nodes[observation.activity]
    )

    assert explorer.maybe_continue_from_coverage_checkpoint(50, 100, "s2_detail") == "s0_home"
    assert device_log == [_FORCE_STOP, main_start, _PACKAGE_LAUNCH, _FORCE_STOP, main_start]


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
    device = _ReplayScript(device_log, web_page, item_list, broken=open_item)
    explorer = _explorer(device, tmp_path, router_client)
    explorer.graph.graph.add_nodes_from(["s0_home", "s1_list", "s2_detail"])
    explorer.graph.add_edge("s0_home", "s1_list", action=[open_list])
    explorer.graph.add_edge("s1_list", "s2_detail", action=[open_item])
    router_responses.extend(
        [
            chat_response(content=_NO_DUPLICATES),
            chat_response(content='{"node_id":"s2_detail","reason":"unexplored"}'),
        ]
    )
    nodes = {"com.demo.Article": EXTERNAL_WEB_NODE, "com.demo.List": "s1_list"}
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
    explorer.graph.graph.add_edge("s0_home", "s1_settings")
    explorer.graph.graph.add_edge("s1_settings", "s2_detail")
    router_responses.extend(
        [
            chat_response(content=_NO_DUPLICATES),
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
    }
    device = _ScriptedDevice(home, dialog, home)
    monkeypatch.setattr(kobe, "controller", _recording(device, kobe))
    router_responses.extend(
        [
            httpx.Response(200, json=embedding_body),
            chat_response(content='{"page_description":"home"}'),
            httpx.Response(200, json=embedding_body),
            chat_response(content='{"element":null,"instruction":"Tap Take photo"}'),
            chat_response(content='{"action":"click","coordinate":[500,500]}'),
            # The dialog shows home's pixels, but of another package: no exact match.
            httpx.Response(200, json=embedding_body),
            chat_response(content='{"page_description":"camera permission dialog"}'),
            httpx.Response(200, json=embedding_body),
        ]
    )

    kobe.explore(max_steps=1)
    return device


@pytest.mark.usefixtures("no_sleep")
def test_a_permission_dialog_is_recorded_as_a_screen_of_the_app(
    kobe: Kobe,
    png_screenshot: str,
    router_responses: list[httpx.Response],
    chat_response: Callable[..., httpx.Response],
    embedding_body: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pressing back on a permission dialog would deny the permission."""
    device = _explore_into_a_permission_dialog(
        kobe, png_screenshot, router_responses, chat_response, embedding_body, monkeypatch
    )

    tap = {"action": "tap", "x": 50, "y": 100}
    assert device.actions == [tap]
    graph = kobe.graph.graph
    nodes = {data["activity"]: node for node, data in graph.nodes(data=True)}
    assert set(nodes) == {"com.demo.Home", _DIALOG_ACTIVITY}
    dialog_node = graph.nodes[nodes[_DIALOG_ACTIVITY]]
    assert dialog_node["page_description"] == "camera permission dialog"
    assert "is_external" not in dialog_node
    edge = graph.edges[nodes["com.demo.Home"], nodes[_DIALOG_ACTIVITY]]
    assert edge["actions"] == [[tap]]


@pytest.mark.usefixtures("no_sleep")
def test_a_permission_dialog_is_matched_only_with_dialogs_and_never_replayed_to(
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
    nodes = {data["package"]: node for node, data in graph.graph.nodes(data=True)}
    assert set(nodes) == {"com.demo", "com.google.android.permissioncontroller"}
    home = nodes["com.demo"]
    assert graph.get_resume_target() == nodes["com.google.android.permissioncontroller"]
    assert graph.get_resume_target(package_name="com.demo") == home
    checkpoint = graph.get_exploration_target_candidates(package_name="com.demo")
    assert [candidate["node_id"] for candidate in checkpoint] == [home]
    described_without_known_screens = [
        request
        for request in router_requests
        if b"No screens have been discovered" in request.content
    ]
    assert len(described_without_known_screens) == 2


_SEARCH = "com.google.android.settings.intelligence"
_SEARCH_SCREEN = {**_screen(f"{_SEARCH}.SearchActivity", "orange"), "package": _SEARCH}
_KEYBOARD = "com.google.android.inputmethod.latin"
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
            chat_response(content='{"page_description":"home"}'),
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
    assert _node_packages(explorer) == {"s0_home": "com.demo", f"ext_{_SEARCH}": None}


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
    keyboard = {**_screen("com.demo.Home", "gray"), "package": _KEYBOARD}
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
        chat_response(content='{"page_description":"home with the keyboard shown"}'),
        httpx.Response(200, json=embedding_body),
    )

    assert device_log == [_TAP, _BACK]
    assert _node_packages(explorer) == {
        "s0_home": "com.demo",
        f"ext_{_SEARCH}": None,
        "s1_home_with_the_keyboard_shown": _KEYBOARD,
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
    assert _node_packages(explorer) == {"s0_home": "com.demo", f"ext_{_SEARCH}": None}


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
    explorer.graph.graph.add_nodes_from(["s0_home", "s1_detail"])
    explorer.graph.add_edge("s0_home", "s1_detail", action=[open_detail])
    expected = _identifying(explorer, monkeypatch, "s0_home")

    assert explorer.navigate_to_node("s1_detail") == "s0_home"
    assert device_log == [*_FIRST_RESTART, open_detail, *_RESTART]
    assert expected == ["s0_home"]


def _identifying(explorer: Kobe, monkeypatch: pytest.MonkeyPatch, *nodes: str) -> list[object]:
    """Make the explorer's identify_state answer ``nodes`` in turn; return the list the
    expected node of each call is appended to."""
    expected: list[object] = []
    answers = iter(nodes)

    def identify(_observation: Observation, *, expected_node: str | None = None) -> str:
        expected.append(expected_node)
        return next(answers)

    monkeypatch.setattr(explorer.graph, "identify_state", identify)
    return expected


def _list_reached_from_home(explorer: Kobe, *extra_nodes: str) -> None:
    for node in ("s0_home", "s1_list", *extra_nodes):
        explorer.graph.graph.add_node(node, page_description=node, package="com.demo")


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


@pytest.mark.usefixtures("no_sleep", "device_log")
def test_a_replay_expects_the_node_it_replays_to(
    replaying_kobe: Kobe, monkeypatch: pytest.MonkeyPatch
) -> None:
    _list_reached_from_home(replaying_kobe)
    replaying_kobe.graph.add_edge("s0_home", "s1_list", action=[{"action": "tap", "x": 1, "y": 2}])
    identified = _identifying(replaying_kobe, monkeypatch, "s1_list", "s0_home")

    replaying_kobe.navigate_to_node("s1_list")
    replaying_kobe.navigate_to_node("s0_home")

    assert identified == ["s1_list", "s0_home"]


def test_exploration_embeds_the_members_a_loaded_graph_has_no_image_vector_for(
    tmp_path: Path,
    router_client: OpenRouter,
    router_requests: list[httpx.Request],
    router_responses: list[httpx.Response],
    embedding_body: dict[str, object],
    write_graph_file: Callable[[Path, dict[str, object]], None],
    write_image_vectors: Callable[..., None],
) -> None:
    """The home screen was embedded before; the list never was, and the other app's
    screen is not embedded at all."""
    graph_path = tmp_path / "demo" / "demo.json"
    graph_path.parent.mkdir()
    nodes = [{"id": "s0_home"}, {"id": "s1_list"}, {"id": "ext_com.other", "is_external": True}]
    write_graph_file(graph_path, {"nodes": nodes, "edges": []})
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


@pytest.mark.usefixtures("no_sleep")
def test_every_screen_identify_state_takes_is_embedded_and_annotated(
    tmp_path: Path,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    chat_response: Callable[..., httpx.Response],
    embedding_body: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    step_records: Callable[[Path], list[dict[str, Any]]],
) -> None:
    """The home screen is described, the step's screen matched exactly, and the vectors of
    both are saved with the graph, under the image-embedding model."""
    monkeypatch.setattr("android_app_graph.kobe.soft_keyboard_hint", lambda _udid: "")
    explorer = _explorer(_Screens(HOME_SCREEN), tmp_path, router_client)
    router_responses.extend(
        [
            httpx.Response(200, json=embedding_body),
            chat_response(content='{"page_description":"home"}'),
            httpx.Response(200, json=embedding_body),
            chat_response(content='{"element":null,"instruction":"Scroll down"}'),
            chat_response(content=_TAP_REPLY),
            httpx.Response(200, json=embedding_body),
        ]
    )

    explorer.explore(max_steps=1)

    observations = [
        r for r in step_records(explorer.controller.steps) if r["kind"] == "observation"
    ]
    assert [r["identify"]["stage"] for r in observations] == ["new", "exact"]
    assert [r["describe"]["page_description"] for r in observations] == ["home", "home"]
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
