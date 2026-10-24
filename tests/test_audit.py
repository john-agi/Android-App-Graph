"""app-graph-audit parsing and pure helpers, exercised without adb or an emulator."""

from __future__ import annotations

import base64
import copy
import io
import json
import logging
from collections.abc import Callable, Iterator, Mapping
from contextlib import nullcontext
from pathlib import Path
from typing import Any, NoReturn, override

import httpx
import pytest
from openrouter import OpenRouter
from PIL import Image

from android_app_graph.commands import audit
from android_app_graph.device import DeviceController
from android_app_graph.graph_files import (
    append_json_line,
    graph_steps_dir,
    identify_file_path,
    observation_image_path,
    save_embeddings,
    text_embeddings_path,
)
from android_app_graph.run_environment import UnsafeReplayError
from android_app_graph.screen_image import encode_screenshot
from android_app_graph.step_log import Observation, StepRecorder
from android_app_graph.utils import DEFAULT_MATCHING, ModelSettings, graph_manager, open_client
from android_app_graph.utils.graph_manager import EXTERNAL_WEB_NODE, EdgeDetails, GraphManager
from android_app_graph.utils.vlm_utils import ActionRequest, PlanRequest

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


@pytest.fixture
def client() -> Iterator[OpenRouter]:
    """An OpenRouter client that is never called: every VLM helper is monkeypatched."""
    with open_client("not-a-real-key") as opened:
        yield opened


def _inspection() -> dict[str, object]:
    return {"execution": {"effect": "inspection", "reason": "Inspect the recorded app screen."}}


def _write_config(tmp_path: Path, graph_dir: str = "graphs") -> Path:
    config = tmp_path / "explore.yaml"
    config.write_text(
        f"experiment:\n  graph_dir: {graph_dir}\napps:\n  - name: demo\n    package_name: com.demo\n",
        encoding="utf-8",
    )
    return config


def _graph_with_nodes(*nodes: tuple[str, dict[str, Any]]) -> GraphManager:
    """Return a GraphManager holding the given nodes, each with a ``replay_misses`` of 0
    and, unless its attributes list its members, one member observation, and no VLM
    clients."""
    graph = GraphManager()
    for node_id, attrs in nodes:
        member = {"observations": [f"{node_id}-observation"]}
        graph.graph.add_node(node_id, replay_misses=0, **(member | attrs))
    return graph


def _png(shade: int) -> str:
    """Return a small base64 PNG screenshot of one gray ``shade``."""
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (shade, shade, shade)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _webp(png: str) -> str:
    """Return ``png`` as the base64 WebP a StepRecorder hands over."""
    return base64.b64encode(encode_screenshot(png)).decode("ascii")


def test_audit_parser_defaults() -> None:
    args = audit.build_parser().parse_args([])
    assert args.config == Path("configs/explore.yaml")
    assert args.app is None
    assert args.re_explore is False
    assert args.steps_per_issue == 5


@pytest.mark.parametrize(
    ("kind", "advice", "effect"),
    [
        ("explore_node", "Test call initiation only with a controlled test number", "side_effect"),
        ("explore_node", "Confirm Clear call history", "side_effect"),
        ("explore_node", "Place a call to the controlled test number", "inspection"),
        ("explore_node", "Confirm deletion of the contact", "inspection"),
        ("retry_edge", "Send the message to Alice", "inspection"),
        ("retry_edge", "Tap the final Delete button", "side_effect"),
        ("retry_edge", "", "inspection"),
        ("explore_node", "Try the unlabeled action", "uncertain"),
        ("explore_node", "Inspect the dialer", None),
    ],
)
@pytest.mark.usefixtures("_fake_device")
def test_audit_execution_policy_skips_commitments_before_navigation(
    kind: str,
    advice: str,
    effect: str | None,
    client: OpenRouter,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    device_log: list[object],
) -> None:
    graph, graph_path = _two_node_graph(tmp_path)
    monkeypatch.setattr(graph, "identify_state", lambda *_args, **_kwargs: "s0_home")
    issue: dict[str, object] = {
        "type": kind,
        "node": "s0_home",
        "source_node": "s0_home",
        "target_node": "s1_detail",
        "instruction": advice,
        "expected_pages": [advice],
    }
    if effect is not None:
        issue["execution"] = {"effect": effect, "reason": "Assessed proposed operation."}
    controller = FakeController("com.demo")

    [result] = audit.re_explore_issues(
        graph,
        [issue],
        target=audit.ReExploreTarget(
            app_name="demo", package_name="com.demo", graph_path=graph_path
        ),
        session=audit.ReExploreSession(
            controller, ModelSettings("instruction-model"), ModelSettings("action-model")
        ),
        client=client,
    )

    assert result["status"] == "skipped"
    assert "policy" in result["reason"]
    assert result["issue"] == issue
    assert controller.actions == []
    assert device_log == []


@pytest.mark.usefixtures("_fake_device")
def test_audit_plans_consume_the_step_budget_when_a_commitment_is_skipped(
    client: OpenRouter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph, graph_path = _two_node_graph(tmp_path)
    monkeypatch.setattr(graph, "identify_state", lambda *_args, **_kwargs: "s0_home")
    planned = iter(
        ["Confirm deletion of the contact", "Confirm deletion of the contact", "Press Cancel"]
    )
    descriptions: list[str] = []

    def plan(*, request: PlanRequest, **_kwargs: object) -> tuple[str, None]:
        descriptions.append(request.page_description)
        return next(planned), None

    monkeypatch.setattr(audit, "plan_next_action", plan)
    grounded: list[str] = []

    def act(*, request: ActionRequest, **_kwargs: object) -> tuple[dict[str, str], str]:
        grounded.append(request.instruction)
        return {"action": "back"}, ""

    monkeypatch.setattr(audit, "predict_next_action", act)
    controller = FakeController("com.demo")
    [result] = _re_explore(
        graph,
        [{**_inspection(), "type": "explore_node", "node": "s0_home", "expected_pages": []}],
        controller,
        client,
        graph_path.parent,
        steps_per_issue=3,
    )

    assert "Excluded by the inspection policy" not in descriptions[0]
    assert "Excluded by the inspection policy" in descriptions[1]
    assert "Confirm deletion of the contact" in descriptions[1]
    assert result["steps"][0]["status"] == "skipped"
    assert "policy" in result["steps"][0]["reason"]
    assert descriptions[2].count("Confirm deletion of the contact") == 1
    assert result["steps"][1]["status"] == "skipped"
    assert result["steps"][2]["step"] == 3
    assert len(grounded) == 1
    assert grounded[0].startswith("Press Cancel\n")
    assert controller.actions == [({"action": "back"}, True)]


@pytest.mark.usefixtures("_fake_device")
def test_audit_navigation_skips_a_recorded_commitment_before_restarting(
    client: OpenRouter, tmp_path: Path, device_log: list[object]
) -> None:
    graph, graph_path = _two_node_graph(tmp_path)
    graph.graph["s0_home"]["s1_detail"]["instructions"] = ["Send the file to Alice"]
    controller = FakeController("com.demo")
    [result] = _re_explore(
        graph,
        [{**_inspection(), "type": "explore_node", "node": "s1_detail", "expected_pages": []}],
        controller,
        client,
        graph_path.parent,
    )
    assert result["status"] == "skipped"
    assert "policy" in result["reason"]
    assert controller.actions == []
    assert device_log == []


@pytest.mark.usefixtures("_fake_device")
def test_navigation_checks_the_chosen_action_from_several_recorded_actions(
    client: OpenRouter, tmp_path: Path, device_log: list[object]
) -> None:
    graph, graph_path = _two_node_graph(tmp_path)
    edge = graph.graph["s0_home"]["s1_detail"]
    edge["instructions"] = ["Open the detail page"]
    edge["num_steps"] = [3]
    graph.add_edge(
        "s0_home",
        "s1_detail",
        [{"action": "tap", "x": 30, "y": 40}],
        details=EdgeDetails(instruction="Send the selected file", target_observation="Detail"),
    )
    controller = FakeController("com.demo")
    [result] = _re_explore(
        graph,
        [{**_inspection(), "type": "explore_node", "node": "s1_detail", "expected_pages": []}],
        controller,
        client,
        graph_path.parent,
    )
    assert result["status"] == "skipped"
    assert "policy" in result["reason"]
    assert controller.actions == []
    assert device_log == []


@pytest.mark.usefixtures("model_environment", "opened_clients", "no_sleep", "device_log")
def test_non_executable_audit_advice_leaves_permissions_and_launch_untouched(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    add_member: Callable[..., None],
    write_image_vectors: Callable[..., None],
) -> None:
    graph_dir = tmp_path / "graphs"
    app_dir = graph_dir / "demo"
    _save_with_members(_flagged_graph(), app_dir / "demo.json", add_member, write_image_vectors)
    config = tmp_path / "audit.yaml"
    config.write_text(
        json.dumps(
            {
                "device": {"udid": "test-device"},
                "experiment": {"graph_dir": str(graph_dir)},
                "apps": [{"name": "demo", "package_name": "com.demo"}],
            }
        )
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    controller = FakeController("com.demo")
    _install_fake_adb(monkeypatch, controller)
    issue = {"type": "explore_node", "node": "s0_home", "expected_pages": ["Make a call"]}
    monkeypatch.setattr(audit, "run_audit", lambda *_args: {"issues": [issue]})
    monkeypatch.setattr(GraphManager, "normalize_all_edges", lambda _self: 0)

    def forbidden(*_args: object, **_kwargs: object) -> NoReturn:
        msg = "Non-executable advice leaves the device untouched"
        raise AssertionError(msg)

    monkeypatch.setattr(audit, "grant_runtime_permissions", forbidden)
    monkeypatch.setattr(audit, "launch_app", forbidden)
    monkeypatch.setattr(audit, "AdbDisplayAccess", forbidden)
    monkeypatch.setattr(audit, "RunEnvironment", forbidden)
    assert audit.main(["--config", str(config), "--re-explore"]) == 0
    [result] = json.loads((app_dir / "demo_re_explore.json").read_text())
    assert result["status"] == "skipped"
    assert result["execution_policy"]["executable"] is False
    assert controller.actions == []


def test_basic_audit_saves_local_execution_decisions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = GraphManager()
    issues = [
        {**_inspection(), "type": "explore_node", "expected_pages": ["Open the dialer"]},
        {**_inspection(), "type": "retry_edge", "instruction": "Send the file"},
        {"type": "merge_nodes", "node_a": "s0", "node_b": "s1"},
    ]
    monkeypatch.setattr(graph, "run_audit", lambda **_kwargs: {"issues": issues})
    path = tmp_path / "audit.json"
    audit.run_audit(graph, "demo", path)
    saved = json.loads(path.read_text())["issues"]
    assert saved[0]["execution_policy"]["executable"] is True
    assert saved[1]["execution_policy"]["executable"] is False
    assert "execution_policy" not in saved[2]


def test_audit_parser_overrides() -> None:
    args = audit.build_parser().parse_args(
        ["-c", "custom.yaml", "--app", "demo", "--re-explore", "--steps-per-issue", "9"]
    )
    assert args.config == Path("custom.yaml")
    assert args.app == "demo"
    assert args.re_explore is True
    assert args.steps_per_issue == 9


def test_audit_main_rejects_missing_config(capsys: pytest.CaptureFixture[str]) -> None:
    """A missing config file exits 2 before anything is read or logging is set up."""
    with pytest.raises(SystemExit) as excinfo:
        audit.main(["-c", "does-not-exist.yaml"])
    assert excinfo.value.code == 2
    assert "config file not found" in capsys.readouterr().err


def test_audit_main_rejects_app_with_re_explore(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """select_apps_for_audit's ValueError is reported through parser.error (exit 2)."""
    config = _write_config(tmp_path)
    with pytest.raises(SystemExit) as excinfo:
        audit.main(["-c", str(config), "--app", "demo", "--re-explore"])
    assert excinfo.value.code == 2
    assert "--app can only be used for basic audit" in capsys.readouterr().err


def test_select_apps_without_app_returns_config_apps(tmp_path: Path) -> None:
    config = {"apps": [{"name": "a", "package_name": "com.a"}]}
    assert audit.select_apps_for_audit(config, tmp_path) == config["apps"]


def test_select_apps_without_app_and_without_apps_key(tmp_path: Path) -> None:
    assert audit.select_apps_for_audit({}, tmp_path) == []


def test_select_apps_rejects_app_with_re_explore(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="--app can only be used for basic audit"):
        audit.select_apps_for_audit({}, tmp_path, app_name="demo", re_explore=True)


def test_select_apps_requires_an_existing_graph(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No graph found at"):
        audit.select_apps_for_audit({}, tmp_path, app_name="demo")


def test_select_apps_finds_graph_folder_by_name(tmp_path: Path) -> None:
    graph_path = tmp_path / "demo" / "demo.json"
    graph_path.parent.mkdir(parents=True)
    graph_path.write_text("{}", encoding="utf-8")
    assert audit.select_apps_for_audit({}, tmp_path, app_name="demo") == [{"name": "demo"}]


def test_run_audit_writes_the_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    graph = GraphManager()
    result = {
        "issues": [
            {"instruction": "Inspect the source page", **_inspection(), "type": "retry_edge"}
        ],
        "summary": "one issue",
    }
    audited: list[str] = []

    def fake_run_audit(app_name: str) -> dict[str, Any]:
        audited.append(app_name)
        return result

    monkeypatch.setattr(graph, "run_audit", fake_run_audit)

    report_path = tmp_path / "reports" / "demo_audit.json"
    returned = audit.run_audit(graph, "demo", report_path)

    assert returned == result
    assert audited == ["demo"]
    assert json.loads(report_path.read_text(encoding="utf-8")) == result


def test_normalize_and_save_audited_graph(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    graph = _graph_with_nodes(("s0_home", {}), ("s1_detail", {}))
    steps: list[object] = []

    def fake_normalize() -> int:
        steps.append("normalize all edges")
        return 4

    monkeypatch.setattr(graph, "normalize_all_edges", fake_normalize)
    monkeypatch.setattr(graph, "save_graph", steps.append)

    audited_path = tmp_path / "demo_audited.json"
    assert audit.normalize_and_save_audited_graph(graph, audited_path) == 4
    assert steps == ["normalize all edges", audited_path]


def _merge_issue(node_a: str = "s0_home", node_b: str = "s1_home") -> dict[str, Any]:
    return {"type": "merge_nodes", "node_a": node_a, "node_b": node_b}


class FakeController:
    """A DeviceController that records actions instead of touching a device."""

    w = 1080
    h = 1920

    def __init__(self, package_name: str) -> None:
        self.config: dict[str, Any] = {"device": {"udid": "test-device"}}
        self._package_name = package_name
        self.actions: list[tuple[dict[str, Any], bool]] = []

    def get_state(self, **_kwargs: float | None) -> dict[str, Any]:
        return {
            "activity": "MainActivity",
            "screenshot": _png(0),
            "package": self._package_name,
        }

    def exe_action(self, action: dict[str, Any], save_flag: bool = True) -> None:
        self.actions.append((action, save_flag))


class _LoggingController(FakeController):
    def __init__(self, package_name: str, log: list[object]) -> None:
        super().__init__(package_name)
        self._log = log

    @override
    def exe_action(self, action: dict[str, Any], save_flag: bool = True) -> None:
        super().exe_action(action, save_flag)
        self._log.append(action)


def _two_node_graph(tmp_path: Path) -> tuple[GraphManager, Path]:
    """Write and load a two-node, one-edge graph under *tmp_path*."""
    graph_path = tmp_path / "demo" / "demo.json"
    graph_path.parent.mkdir(parents=True)
    graph_path.write_text(
        json.dumps(
            {
                "next_id": 2,
                "matching": DEFAULT_MATCHING._asdict(),
                "nodes": [
                    {
                        "id": "s0_home",
                        "activity": "com.demo.Home",
                        "replay_misses": 0,
                        "observations": ["s0_home-observation"],
                    },
                    {
                        "id": "s1_detail",
                        "activity": "com.demo.Detail",
                        "replay_misses": 0,
                        "observations": ["s1_detail-observation"],
                    },
                ],
                "edges": [
                    {
                        "source": "s0_home",
                        "target": "s1_detail",
                        "actions": [{"action": "tap", "coordinate": [10, 20]}],
                        "instructions": ["open the detail page"],
                        "target_observations": [""],
                        "num_steps": [1],
                        "schema_deltas": [None],
                        "setups": [None],
                        "open_menus": [None],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    for node_id in ("s0_home", "s1_detail"):
        _write_member(graph_path, node_id)
    graph = GraphManager()
    graph.load_graph(graph_path)
    return graph, graph_path


type _Visit = str | tuple[str, Mapping[str, object]]


def _describing_by_node_id(graph: GraphManager, monkeypatch: pytest.MonkeyPatch) -> None:
    """Describe each node, and the screen last identified as it, by the node's ID, as a
    fake identify_state or a node added by hand gives no member a description."""
    monkeypatch.setattr(graph, "latest_description", lambda node_id: node_id)
    monkeypatch.setattr(graph, "node_descriptions", lambda node_id: node_id)


def _identifying(
    graph: GraphManager, monkeypatch: pytest.MonkeyPatch, *visits: _Visit
) -> list[str | None]:
    """Make identify_state answer ``visits`` in turn, each a node, or a node with the state
    the visit records as the node's latest; return the list the node of each call's
    expected screenshot is appended to: the node answered for it, or the node that lists
    it."""
    expected: list[str | None] = []
    answers = iter(visits)
    answered: dict[str, str] = {}

    def identify(observation: Observation, *, expected_observation: str | None = None) -> str:
        known = answered.get(expected_observation or "")
        expected.append(known or graph.node_of(expected_observation))
        visit = next(answers)
        node, state = (visit, None) if isinstance(visit, str) else visit
        if state is not None:
            graph.graph.nodes[node]["last_detail_snapshot"] = dict(state)
        answered[observation.observation_id] = node
        return node

    monkeypatch.setattr(graph, "identify_state", identify)
    return expected


def _write_member(graph_path: Path, node_id: str) -> None:
    """Put the member a node of ``graph_path`` lists in its step log, its image and its
    record with a pixel hash, and its describe output in the graph's identify file, which
    describes it as ``<node_id> screen``."""
    steps_dir = graph_steps_dir(graph_path)
    steps_dir.mkdir(exist_ok=True)
    observation_id = f"{node_id}-observation"
    observation_image_path(steps_dir, observation_id).write_bytes(encode_screenshot(_png(0)))
    record = {"id": observation_id, "kind": "observation", "pixel_hash": f"hash-{observation_id}"}
    (steps_dir / f"{observation_id}.json").write_text(json.dumps(record), encoding="utf-8")
    describe = {
        "page_description": f"{node_id} screen",
        "state": {},
        "elements": [],
        "is_external_web": False,
    }
    append_json_line(
        identify_file_path(graph_path), {"observation": observation_id, "describe": describe}
    )


@pytest.fixture
def _fake_device(
    monkeypatch: pytest.MonkeyPatch, no_sleep: list[float], device_log: list[object]
) -> None:
    """Silence every side effect re_explore_issues has outside the graph."""
    del no_sleep, device_log  # requested for their side effects on time.sleep and subprocess.run
    monkeypatch.setattr(audit, "plan_next_action", lambda **_kwargs: ("tap the first button", None))
    monkeypatch.setattr(audit, "predict_next_action", lambda **_kwargs: ({"action": "end"}, ""))


@pytest.mark.usefixtures("_fake_device")
def test_re_explore_issues_retries_explores_and_skips(
    client: OpenRouter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph, graph_path = _two_node_graph(tmp_path)
    monkeypatch.setattr(graph, "identify_state", lambda *_args, **_kwargs: "s1_detail")
    controller = FakeController("com.demo")
    models: list[tuple[str, str]] = []

    def plan(*, settings: ModelSettings, **_kwargs: object) -> tuple[str, None]:
        models.append(("plan", settings.model))
        return "tap the first button", None

    def act(*, settings: ModelSettings, **_kwargs: object) -> tuple[dict[str, str], str]:
        models.append(("act", settings.model))
        return {"action": "end"}, ""

    monkeypatch.setattr(audit, "plan_next_action", plan)
    monkeypatch.setattr(audit, "predict_next_action", act)

    out_path = tmp_path / "demo" / "demo_audited.json"
    results = audit.re_explore_issues(
        graph,
        [
            {
                "instruction": "Inspect the source page",
                **_inspection(),
                "type": "retry_edge",
                "source_node": "s1_detail",
                "target_node": "s0_home",
            },
            {
                **_inspection(),
                "type": "explore_node",
                "node": "s0_home",
                "expected_pages": ["Settings"],
            },
            {
                "instruction": "Inspect the source page",
                **_inspection(),
                "type": "retry_edge",
                "source_node": "s9_missing",
                "target_node": "s0_home",
            },
        ],
        target=audit.ReExploreTarget(app_name="demo", package_name="com.demo", graph_path=out_path),
        session=audit.ReExploreSession(
            controller,
            ModelSettings("instruction-model"),
            ModelSettings("action-model"),
            steps_per_issue=1,
        ),
        client=client,
    )

    assert [r["status"] for r in results] == ["instruction_not_found", "explored", "skipped"]
    assert results[2]["reason"] == "source not found"
    assert models == [("plan", "instruction-model"), ("act", "action-model")] * 2
    # "end" is rewritten to a back press, and every action is recorded with save_flag.
    assert {action["action"] for action, _save_flag in controller.actions} <= {"tap", "back"}
    assert all(save_flag for _action, save_flag in controller.actions)
    assert out_path.exists()
    assert graph_path.exists()


@pytest.mark.usefixtures("_fake_device")
def test_re_explore_issues_records_an_action_model_end_as_press_back(
    client: OpenRouter,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    graph, _graph_path = _two_node_graph(tmp_path)
    monkeypatch.setattr(graph, "identify_state", lambda *_args, **_kwargs: "s0_home")
    controller = FakeController("com.demo")
    caplog.set_level(logging.INFO, logger=audit.logger.name)

    [result] = audit.re_explore_issues(
        graph,
        [
            {
                **_inspection(),
                "type": "explore_node",
                "node": "s0_home",
                "expected_pages": ["Settings"],
            }
        ],
        target=audit.ReExploreTarget(
            app_name="demo",
            package_name="com.demo",
            graph_path=tmp_path / "demo" / "demo_audited.json",
        ),
        session=audit.ReExploreSession(
            controller,
            ModelSettings("instruction-model"),
            ModelSettings("action-model"),
            steps_per_issue=1,
        ),
        client=client,
    )

    assert controller.actions == [({"action": "back"}, True)]
    edge = graph.graph["s0_home"]["s0_home"]
    assert edge["actions"] == [[{"action": "back"}]]
    assert edge["instructions"] == ["Press back"]
    assert [step["instruction"] for step in result["steps"]] == ["Press back"]
    assert '"tap the first button"' in caplog.text


_STEP_FIELDS = ("before", "after", "instruction", "element", "action", "from_node", "to_node")


@pytest.mark.usefixtures("_fake_device")
def test_a_re_exploration_step_asks_the_audited_device_whether_its_keyboard_is_up(
    client: OpenRouter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, device_log: list[object]
) -> None:
    graph, graph_path = _two_node_graph(tmp_path)
    monkeypatch.setattr(graph, "identify_state", lambda _observation, **_kwargs: "s1_detail")

    _re_explore(
        graph,
        [{**_inspection(), "type": "explore_node", "node": "s1_detail", "expected_pages": []}],
        _LoggingController("com.demo", device_log),
        client,
        graph_path.parent,
        steps_per_issue=1,
    )

    probes = [entry for entry in device_log if isinstance(entry, list) and "input_method" in entry]
    assert probes == [["adb", "-s", "test-device", "shell", "dumpsys", "input_method"]]


@pytest.mark.usefixtures("_fake_device")
def test_a_re_exploration_step_is_recorded_as_an_audit_transition(
    client: OpenRouter,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    device_log: list[object],
    step_records: Callable[[Path], list[dict[str, Any]]],
) -> None:
    """The replay to the node is recorded as a navigation replay, its start expecting the
    start node's earliest member and its end the node's; the step from it is an audit
    transition from the screen it planned on to the screen it reached, matched with no
    expectation."""
    graph, graph_path = _two_node_graph(tmp_path)
    reached = iter(["s0_home", "s1_detail", "s0_home"])
    monkeypatch.setattr(graph, "identify_state", lambda _observation, **_kwargs: next(reached))
    monkeypatch.setattr(audit, "soft_keyboard_hint", lambda _udid: "")
    tap = {"action": "tap", "x": 1, "y": 2}
    monkeypatch.setattr(audit, "predict_next_action", lambda **_kwargs: (tap, ""))

    _re_explore(
        graph,
        [{**_inspection(), "type": "explore_node", "node": "s1_detail", "expected_pages": []}],
        _LoggingController("com.demo", device_log),
        client,
        graph_path.parent,
        steps_per_issue=1,
    )

    records = step_records(graph_path.parent / "steps")
    assert [(record["kind"], record["source"]) for record in records] == [
        ("observation", "navigation_replay"),
        ("match", "navigation_replay"),
        ("action", "navigation_replay"),
        ("observation", "navigation_replay"),
        ("match", "navigation_replay"),
        ("observation", "re_explore"),
        ("action", "re_explore"),
        ("observation", "re_explore"),
        ("match", "re_explore"),
        ("transition", "re_explore"),
    ]
    assert {record["run_kind"] for record in records} == {"audit"}
    assert records[2]["observation"] == records[0]["id"]
    matches = [record for record in records if record["kind"] == "match"]
    assert [(r["observation"], r["expected_observation"]) for r in matches] == [
        ("000000", "s0_home-observation"),
        ("000003", "s1_detail-observation"),
        ("000007", None),
    ]
    assert {key: records[9][key] for key in _STEP_FIELDS} == {
        "before": "000005",
        "after": "000007",
        "instruction": "tap the first button",
        "element": None,
        "action": tap,
        "from_node": "s1_detail",
        "to_node": "s0_home",
    }


@pytest.mark.usefixtures("_fake_device")
def test_re_explore_issues_skips_unreachable_nodes(
    client: OpenRouter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph, _graph_path = _two_node_graph(tmp_path)
    monkeypatch.setattr(graph, "find_path", lambda *_args: None)
    controller = FakeController("com.demo")

    results = audit.re_explore_issues(
        graph,
        [
            {
                "instruction": "Inspect the source page",
                **_inspection(),
                "type": "retry_edge",
                "source_node": "s1_detail",
                "target_node": "s0_home",
            },
            {**_inspection(), "type": "explore_node", "node": "s0_home", "expected_pages": []},
        ],
        target=audit.ReExploreTarget(
            app_name="demo",
            package_name="com.demo",
            graph_path=tmp_path / "demo" / "demo_audited.json",
        ),
        session=audit.ReExploreSession(
            controller,
            ModelSettings("instruction-model"),
            ModelSettings("action-model"),
            steps_per_issue=1,
        ),
        client=client,
    )

    assert [r["status"] for r in results] == ["unverified", "skipped"]
    assert results[0]["reached"] is None
    assert results[1]["reason"] == "unreachable"


@pytest.mark.usefixtures("_fake_device")
def test_re_explore_records_an_edge_to_an_external_web_page_and_presses_back(
    client: OpenRouter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph, graph_path = _two_node_graph(tmp_path)
    # identify_state is faked below, so add the node it would create.
    graph.graph.add_node(EXTERNAL_WEB_NODE, replay_misses=0)
    identified = iter(["s0_home", "s1_detail", EXTERNAL_WEB_NODE, "s1_detail", "s0_home"])
    monkeypatch.setattr(graph, "identify_state", lambda *_args, **_kwargs: next(identified))
    _describing_by_node_id(graph, monkeypatch)
    monkeypatch.setattr(
        audit, "predict_next_action", lambda **_kwargs: ({"action": "tap", "x": 1, "y": 2}, "")
    )
    controller = FakeController("com.demo")

    (result,) = audit.re_explore_issues(
        graph,
        [{**_inspection(), "type": "explore_node", "node": "s1_detail", "expected_pages": []}],
        target=audit.ReExploreTarget(
            app_name="demo", package_name="com.demo", graph_path=graph_path
        ),
        session=audit.ReExploreSession(
            controller,
            ModelSettings("instruction-model"),
            ModelSettings("action-model"),
            steps_per_issue=2,
        ),
        client=client,
    )

    assert [(step["from"], step["to"]) for step in result["steps"]] == [
        ("s1_detail", EXTERNAL_WEB_NODE),
        ("s1_detail", "s0_home"),
    ]
    assert graph.graph.has_edge("s1_detail", EXTERNAL_WEB_NODE)
    assert [action["action"] for action, _save_flag in controller.actions] == [
        "tap",
        "tap",
        "back",
        "tap",
    ]


_BACK = {"action": "back"}


@pytest.mark.parametrize(
    ("back_presses_on_the_page", "leaving"),
    [(1, [_BACK, _BACK]), (3, [_BACK, _BACK, _BACK, *_RESTART])],
)
@pytest.mark.usefixtures("_fake_device")
def test_re_explore_presses_back_until_it_leaves_an_external_web_page(
    back_presses_on_the_page: int,
    leaving: list[object],
    client: OpenRouter,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    device_log: list[object],
) -> None:
    """Re-explore keeps pressing back while the page still shows, so the next step acts
    from an app screen and records no edge out of the shared ext_web node."""
    graph, graph_path = _two_node_graph(tmp_path)
    # identify_state is faked below, so add the node it would create.
    graph.graph.add_node(EXTERNAL_WEB_NODE, replay_misses=0)
    on_the_page = [EXTERNAL_WEB_NODE] * (1 + back_presses_on_the_page)
    expected = _identifying(
        graph, monkeypatch, "s0_home", "s1_detail", *on_the_page, "s1_detail", "s0_home"
    )
    _describing_by_node_id(graph, monkeypatch)
    monkeypatch.setattr(audit, "soft_keyboard_hint", lambda _udid: "")
    tap = {"action": "tap", "x": 1, "y": 2}
    monkeypatch.setattr(audit, "predict_next_action", lambda **_kwargs: (tap, ""))

    (result,) = audit.re_explore_issues(
        graph,
        [{**_inspection(), "type": "explore_node", "node": "s1_detail", "expected_pages": []}],
        target=audit.ReExploreTarget(
            app_name="demo", package_name="com.demo", graph_path=graph_path
        ),
        session=audit.ReExploreSession(
            _LoggingController("com.demo", device_log),
            ModelSettings("instruction-model"),
            ModelSettings("action-model"),
            steps_per_issue=2,
        ),
        client=client,
    )

    assert [(step["from"], step["to"]) for step in result["steps"]] == [
        ("s1_detail", EXTERNAL_WEB_NODE),
        ("s1_detail", "s0_home"),
    ]
    assert graph.graph.out_degree(EXTERNAL_WEB_NODE) == 0
    replay = {"action": "tap", "coordinate": [10, 20]}
    assert device_log == [*_FIRST_RESTART, replay, tap, *leaving, tap]
    # The replay expects the start node and then its target, each back press the page's
    # opener, a restart home.
    restarted = back_presses_on_the_page == 3
    back = ["s1_detail"] * (back_presses_on_the_page + (0 if restarted else 1))
    restart = ["s0_home"] if restarted else []
    assert expected == ["s0_home", "s1_detail", None, *back, *restart, None]


@pytest.mark.usefixtures("_fake_device")
def test_re_explore_records_the_edge_from_the_node_the_verifier_rejected(
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
    description, so the step's edge goes from that node, under the ID it had."""
    monkeypatch.setattr(
        audit, "predict_next_action", lambda **_kwargs: ({"action": "tap", "x": 1, "y": 2}, "")
    )
    graph = GraphManager(
        client=router_client,
        page_detail=ModelSettings("provider/page_detail"),
        embedding=ModelSettings("provider/embedding"),
        image_embedding=ModelSettings("provider/embedding"),
    )
    router_responses.extend(
        [
            httpx.Response(200, json=embedding_body),
            chat_response(content='{"page_description":"Chat search page"}'),
            httpx.Response(200, json=embedding_body),
        ]
    )
    # The re-exploration below records its screens in the same step log.
    graph.steps_dir = tmp_path / "steps"
    graph.identify_file = identify_file_path(tmp_path / "demo_audited.json")
    recorder = StepRecorder(
        two_screen_device, graph.steps_dir, run_kind="audit", source="re_explore"
    )
    start = graph.identify_state(Observation.from_state(recorder.get_state()))
    verdict = {
        "same": False,
        "reason": "a query has been entered",
        "new_description": "Chat search results page",
    }
    router_responses.extend(
        [
            # The replay to the start node ends on its screen: an exact match.
            httpx.Response(200, json=embedding_body),
            httpx.Response(200, json=embedding_body),
            chat_response(content='{"page_description":"Chat search results"}'),
            httpx.Response(200, json=embedding_body),
            chat_response(content=json.dumps(verdict)),
            *[httpx.Response(200, json=embedding_body) for _ in range(2)],
        ]
    )
    with caplog.at_level(logging.WARNING, logger="android_app_graph"):
        (result,) = audit.re_explore_issues(
            graph,
            [
                {
                    **_inspection(),
                    "type": "explore_node",
                    "node": start,
                    "expected_pages": ["Search results"],
                }
            ],
            target=audit.ReExploreTarget(
                app_name="demo",
                package_name="com.demo.app",
                graph_path=tmp_path / "demo_audited.json",
            ),
            session=audit.ReExploreSession(
                two_screen_device,
                ModelSettings("provider/instruction"),
                ModelSettings("provider/action"),
                steps_per_issue=1,
            ),
            client=router_client,
        )

    assert [(step["from"], step["to"]) for step in result["steps"]] == [(start, "s1")]
    assert graph.graph.has_edge(start, "s1")
    assert "add_edge: missing node(s)" not in caplog.text


def _recording_planner(planned_from: list[str]) -> Callable[..., tuple[str, None]]:
    """A ``plan_next_action`` that records the page description of each screen it plans
    from, hint included."""

    def plan(*, request: PlanRequest, **_kwargs: object) -> tuple[str, None]:
        planned_from.append(request.page_description)
        return "Tap the Search button", None

    return plan


@pytest.mark.usefixtures("_fake_device")
def test_re_exploration_plans_from_and_records_each_screen_s_own_description(
    tmp_path: Path,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    chat_response: Callable[..., httpx.Response],
    embedding_body: dict[str, object],
    screens_by_log: Callable[..., DeviceController],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The results screen joins the search page's node on the verifier's word: the edge
    it ends records the results screen's description, and the next step plans from it,
    not from the node's."""
    planned_from: list[str] = []
    input_statuses: list[str] = []
    grounded: list[str] = []

    def plan(*, request: PlanRequest, **_kwargs: object) -> tuple[str, None]:
        planned_from.append(request.page_description)
        input_statuses.append(request.input_status)
        return "Tap the Search button", None

    def act(*, request: ActionRequest, **_kwargs: object) -> tuple[dict[str, object], str]:
        grounded.append(request.instruction)
        return {"action": "tap", "x": 1, "y": 2}, ""

    monkeypatch.setattr(audit, "plan_next_action", plan)
    monkeypatch.setattr(audit, "predict_next_action", act)
    monkeypatch.setattr(audit, "soft_keyboard_hint", lambda _udid: "")
    search = {"activity": "com.demo.Search", "screenshot": _png(255), "package": "com.demo"}
    # The restart before the replay sends three adb commands, and the first tap after it
    # shows the results.
    device = screens_by_log(*[search] * 4, {**search, "screenshot": _png(128)})
    graph = GraphManager(
        client=router_client,
        page_detail=ModelSettings("provider/page_detail"),
        embedding=ModelSettings("provider/embedding"),
        image_embedding=ModelSettings("provider/embedding"),
    )
    graph.steps_dir = tmp_path / "steps"
    graph.identify_file = identify_file_path(tmp_path / "demo_audited.json")
    router_responses.extend(
        [
            httpx.Response(200, json=embedding_body),
            chat_response(
                content='{"page_description":"Chat search page","state":{"focused_field":"Search"}}'
            ),
            httpx.Response(200, json=embedding_body),
        ]
    )
    recorder = StepRecorder(device, graph.steps_dir, run_kind="audit", source="re_explore")
    start = graph.identify_state(Observation.from_state(recorder.get_state()))
    same = '{"same":true,"reason":"one screen","new_description":null}'
    router_responses.extend(
        [
            # The replay ends on the search page: an exact match.
            httpx.Response(200, json=embedding_body),
            httpx.Response(200, json=embedding_body),
            chat_response(
                content='{"page_description":"Chat search results","state":{"focused_field":null}}'
            ),
            httpx.Response(200, json=embedding_body),
            chat_response(content=same),
            # The results show again: an exact match.
            httpx.Response(200, json=embedding_body),
        ]
    )

    audit.re_explore_issues(
        graph,
        [
            {
                **_inspection(),
                "type": "explore_node",
                "node": start,
                "expected_pages": ["Search results"],
            }
        ],
        target=audit.ReExploreTarget(
            app_name="demo", package_name="com.demo", graph_path=tmp_path / "demo_audited.json"
        ),
        session=audit.ReExploreSession(
            device,
            ModelSettings("provider/instruction"),
            ModelSettings("provider/action"),
            steps_per_issue=2,
        ),
        client=router_client,
    )

    hint = (
        "\n[AUDIT HINT: This screen might be able to lead to pages like 'Search results'. "
        "Check if any elements on this screen should be interacted with.]"
    )
    assert planned_from == [
        f"Chat search page{hint}\n{audit.INSPECTION_HINT}",
        f"Chat search results{hint}\n{audit.INSPECTION_HINT}",
    ]
    assert graph.graph[start][start]["target_observations"] == ["Chat search results"]

    for status, instruction, focus in zip(
        input_statuses, grounded, ['"Search"', "null"], strict=True
    ):
        assert f"Current screenshot description reports focused_field: {focus}." in status
        assert status in instruction


@pytest.mark.usefixtures("no_sleep", "device_log")
def test_re_exploration_lists_a_self_loop_s_30_most_recent_instructions_as_exploration_does(
    tmp_path: Path,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    chat_response: Callable[..., httpx.Response],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A re-exploration step plans with exploration's prompt, which lists the 30 most
    recent distinct instructions of the screen's self-loop and counts the rest."""
    graph, graph_path = _two_node_graph(tmp_path)
    for day in range(31):
        graph.add_edge(
            "s1_detail",
            "s1_detail",
            action=[{"action": "tap", "coordinate": [day, 0]}],
            details=EdgeDetails(instruction=f"Toggle day {day}", target_observation="Detail"),
        )
    monkeypatch.setattr(graph, "identify_state", lambda _observation, **_kwargs: "s1_detail")
    monkeypatch.setattr(audit, "soft_keyboard_hint", lambda _udid: "")
    monkeypatch.setattr(audit, "predict_next_action", lambda **_kwargs: ({"action": "wait"}, ""))
    router_responses.append(chat_response(content='{"element":null,"instruction":"Scroll down"}'))

    _re_explore(
        graph,
        [{**_inspection(), "type": "explore_node", "node": "s1_detail", "expected_pages": []}],
        FakeController("com.demo"),
        router_client,
        graph_path.parent,
        steps_per_issue=1,
    )

    (plan,) = [json.loads(request.content) for request in router_requests]
    prompt = plan["messages"][-1]["content"][0]["text"]
    tried = prompt.split("where they led:\n", 1)[1].split("\n\n", 1)[0].splitlines()
    assert tried == [
        *(f'- "Toggle day {day}" → stayed on this screen' for day in range(1, 31)),
        "- 1 more tried earlier also stayed on this screen",
    ]


@pytest.mark.usefixtures("_fake_device")
def test_the_retry_hint_names_the_flagged_target_by_its_members_descriptions_with_counts(
    client: OpenRouter,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    write_member_graph: Callable[..., None],
) -> None:
    """The steps after a retried edge plan from the source screen's own description, with
    a hint that names the edge's target by every distinct description of its members."""
    graph_path = tmp_path / "demo" / "demo.json"
    write_member_graph(
        graph_path,
        {
            "s0_home": [("000001", [1.0, 0.0]), ("000002", [1.0, 0.0]), ("000003", [1.0, 0.0])],
            "s1_detail": [("000004", [0.0, 1.0])],
        },
        edges=(
            {
                "source": "s0_home",
                "target": "s1_detail",
                "actions": [[{"action": "tap", "x": 10, "y": 20}]],
                "instructions": [_FLAGGED],
                "target_observations": ["Detail page"],
                "num_steps": [1],
                "schema_deltas": [None],
                "setups": [None],
                "open_menus": [None],
                "visit_count": 1,
            },
        ),
        described={
            "000001": "Home",
            "000002": "Home feed",
            "000003": "Home feed",
            "000004": "Detail page",
        },
    )
    graph = GraphManager()
    graph.load_graph(graph_path)
    _identifying(graph, monkeypatch, "s0_home", "s1_detail", "s1_detail")
    planned_from: list[str] = []
    monkeypatch.setattr(audit, "plan_next_action", _recording_planner(planned_from))
    monkeypatch.setattr(audit, "soft_keyboard_hint", lambda _udid: "")

    (result,) = _re_explore(
        graph,
        [_retry("open the home page", "s1_detail", "s0_home")],
        FakeController("com.demo"),
        client,
        graph_path.parent,
        steps_per_issue=1,
    )

    target = '"Home feed" (2), "Home" (1)'
    hint = (
        f"A previous attempt to reach {target} from this screen may have gone to the wrong "
        f"place. Check if any elements on this screen could lead to {target}."
    )
    assert result["status"] == "instruction_not_found"
    assert planned_from == [f"Detail page\n[AUDIT HINT: {hint}]\n{audit.INSPECTION_HINT}"]


_LAUNCHER = "ext_com.android.launcher3"
_BACK_LEAVES_THE_APP = (
    "Back on this screen leaves the app for the home screen, so choose an action on this "
    "screen instead, even where a rule below says to go back."
)


@pytest.mark.parametrize(
    ("node", "left_by", "told"),
    [
        pytest.param("s0_home", {"action": "tap", "x": 1, "y": 2}, True, id="the start screen"),
        pytest.param("s1_detail", {"action": "back"}, True, id="a recorded Back left the app"),
        pytest.param("s1_detail", {"action": "tap", "x": 1, "y": 2}, False, id="a tap left"),
    ],
)
@pytest.mark.usefixtures("no_sleep", "device_log")
def test_re_exploration_tells_the_planner_that_back_leaves_the_app_where_it_does(
    node: str,
    left_by: dict[str, object],
    told: bool,
    tmp_path: Path,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    chat_response: Callable[..., httpx.Response],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A re-exploration step plans with exploration's prompt, which says that Back leaves
    the app on the start screen and on a screen with an edge by Back to an ``ext_`` node,
    and on no other screen."""
    graph, graph_path = _two_node_graph(tmp_path)
    graph.graph.add_node(_LAUNCHER, is_external=True, replay_misses=0)
    graph.add_edge("s1_detail", _LAUNCHER, action=[left_by])
    monkeypatch.setattr(graph, "identify_state", lambda _observation, **_kwargs: node)
    monkeypatch.setattr(audit, "soft_keyboard_hint", lambda _udid: "")
    monkeypatch.setattr(audit, "predict_next_action", lambda **_kwargs: ({"action": "wait"}, ""))
    router_responses.append(chat_response(content='{"element":null,"instruction":"Scroll down"}'))

    _re_explore(
        graph,
        [{**_inspection(), "type": "explore_node", "node": node, "expected_pages": []}],
        FakeController("com.demo"),
        router_client,
        graph_path.parent,
        steps_per_issue=1,
    )

    (plan,) = [json.loads(request.content) for request in router_requests]
    assert plan["model"] == "instruction-model"
    assert (_BACK_LEAVES_THE_APP in plan["messages"][-1]["content"][0]["text"]) is told


@pytest.mark.usefixtures("_fake_device")
def test_re_explore_issues_returns_early_for_an_empty_graph(
    client: OpenRouter, tmp_path: Path
) -> None:
    assert (
        audit.re_explore_issues(
            GraphManager(),
            [
                {
                    "instruction": "Inspect the source page",
                    **_inspection(),
                    "type": "retry_edge",
                    "source_node": "s0_home",
                }
            ],
            target=audit.ReExploreTarget(
                app_name="demo", package_name="com.demo", graph_path=tmp_path / "demo_audited.json"
            ),
            session=audit.ReExploreSession(
                FakeController("com.demo"),
                ModelSettings("instruction-model"),
                ModelSettings("action-model"),
            ),
            client=client,
        )
        == []
    )


_MAIN_RESTART = [_FORCE_STOP, _clear_task_start("com.demo/.Main")]


@pytest.mark.parametrize(
    ("launch_activity", "first_restart", "restart"),
    [(None, _FIRST_RESTART, _RESTART), ("com.demo/.Main", _MAIN_RESTART, _MAIN_RESTART)],
)
@pytest.mark.usefixtures("_fake_device")
def test_re_explore_issues_restarts_the_app_before_each_replay(
    launch_activity: str | None,
    first_restart: list[list[str]],
    restart: list[list[str]],
    client: OpenRouter,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    device_log: list[object],
) -> None:
    """Each replay starts from a restarted app, not the screen the previous issue ended on,
    and the run asks the device for the launcher activity once."""
    graph, _graph_path = _two_node_graph(tmp_path)
    # Each issue replays from home to the detail page, where its one step stays.
    _identifying(graph, monkeypatch, *("s0_home", "s1_detail", "s1_detail") * 2)
    monkeypatch.setattr(audit, "soft_keyboard_hint", lambda _udid: "")

    results = audit.re_explore_issues(
        graph,
        [
            {
                "instruction": "Inspect the source page",
                **_inspection(),
                "type": "retry_edge",
                "source_node": "s1_detail",
                "target_node": "s0_home",
            },
            {
                **_inspection(),
                "type": "explore_node",
                "node": "s1_detail",
                "expected_pages": ["Settings"],
            },
        ],
        target=audit.ReExploreTarget(
            app_name="demo",
            package_name="com.demo",
            graph_path=tmp_path / "demo_audited.json",
            launch_activity=launch_activity,
        ),
        session=audit.ReExploreSession(
            _LoggingController("com.demo", device_log),
            ModelSettings("instruction-model"),
            ModelSettings("action-model"),
            steps_per_issue=1,
        ),
        client=client,
    )

    assert [r["status"] for r in results] == ["instruction_not_found", "explored"]
    replay = {"action": "tap", "coordinate": [10, 20]}
    explore_step = {"action": "back"}
    assert device_log == [*first_restart, replay, explore_step, *restart, replay, explore_step]


@pytest.mark.usefixtures("_fake_device")
def test_re_explore_records_a_permission_dialog_instead_of_pressing_back(
    client: OpenRouter,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    device_log: list[object],
) -> None:
    graph, _graph_path = _two_node_graph(tmp_path)
    graph.graph.add_node("s2_camera_permission", replay_misses=0)
    reached = iter(["s0_home", "s1_detail", "s2_camera_permission"])
    monkeypatch.setattr(graph, "identify_state", lambda *_args, **_kwargs: next(reached))
    _describing_by_node_id(graph, monkeypatch)
    monkeypatch.setattr(audit, "soft_keyboard_hint", lambda _udid: "")
    tap = {"action": "tap", "x": 1, "y": 2}
    monkeypatch.setattr(audit, "predict_next_action", lambda **_kwargs: (tap, ""))

    (result,) = audit.re_explore_issues(
        graph,
        [{**_inspection(), "type": "explore_node", "node": "s1_detail", "expected_pages": []}],
        target=audit.ReExploreTarget(
            app_name="demo",
            package_name="com.demo",
            graph_path=tmp_path / "demo" / "demo_audited.json",
        ),
        session=audit.ReExploreSession(
            _LoggingController("com.google.android.permissioncontroller", device_log),
            ModelSettings("instruction-model"),
            ModelSettings("action-model"),
            steps_per_issue=1,
        ),
        client=client,
    )

    assert result["steps"] == [
        {
            "step": 1,
            "from": "s1_detail",
            "to": "s2_camera_permission",
            "instruction": "tap the first button",
            "action": tap,
        }
    ]
    assert graph.graph.has_edge("s1_detail", "s2_camera_permission")
    replay = {"action": "tap", "coordinate": [10, 20]}
    assert device_log == [*_FIRST_RESTART, replay, tap]


_FLAGGED = "open the detail page"
_DETAIL_TAP = {"action": "tap", "x": 10, "y": 20}
_GEAR_TAP = {"action": "tap", "x": 1000, "y": 80}
_SWIPE = {"action": "swipe", "x1": 540, "y1": 1500, "x2": 540, "y2": 500, "duration": 400}


def _flagged_graph() -> GraphManager:
    """Home reaches Detail and Settings with one tap each; issues flag the tap to Detail."""
    graph = _graph_with_nodes(("s0_home", {}), ("s1_detail", {}), ("s2_settings", {}))
    graph.add_edge(
        "s0_home",
        "s1_detail",
        [_DETAIL_TAP],
        details=EdgeDetails(instruction=_FLAGGED, target_observation="Detail"),
    )
    graph.add_edge(
        "s0_home",
        "s2_settings",
        [_GEAR_TAP],
        details=EdgeDetails(instruction="open settings", target_observation="Settings"),
    )
    return graph


def _retry(
    instruction: str = _FLAGGED, source: str = "s0_home", target: str = "s1_detail"
) -> dict[str, Any]:
    return {
        **_inspection(),
        "type": "retry_edge",
        "source_node": source,
        "target_node": target,
        "instruction": instruction,
    }


def _edges(graph: GraphManager) -> dict[tuple[str, str], dict[str, Any]]:
    return {(u, v): copy.deepcopy(dict(data)) for u, v, data in graph.graph.edges(data=True)}


class _ReplayController(_LoggingController):
    """Shows a new screenshot after every action, from *package_after_action* once one has run."""

    def __init__(self, log: list[object], package_after_action: str = "com.demo") -> None:
        super().__init__("com.demo", log)
        self._package_after_action = package_after_action

    @override
    def get_state(self, **_kwargs: float | None) -> dict[str, Any]:
        state = super().get_state()
        if self.actions:
            state["package"] = self._package_after_action
        return {**state, "screenshot": _png(len(self.actions))}


def _re_explore(
    graph: GraphManager,
    issues: list[dict[str, Any]],
    controller: DeviceController,
    client: OpenRouter,
    tmp_path: Path,
    steps_per_issue: int = 0,
) -> list[dict[str, Any]]:
    return audit.re_explore_issues(
        graph,
        issues,
        target=audit.ReExploreTarget(
            app_name="demo", package_name="com.demo", graph_path=tmp_path / "demo_audited.json"
        ),
        session=audit.ReExploreSession(
            controller,
            ModelSettings("instruction-model"),
            ModelSettings("action-model"),
            steps_per_issue=steps_per_issue,
        ),
        client=client,
    )


@pytest.mark.usefixtures("_fake_device")
def test_re_explore_moves_a_flagged_action_to_the_edge_toward_the_node_it_reaches(
    client: OpenRouter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, device_log: list[object]
) -> None:
    """A flagged edge whose action reaches another node moves that action to the edge toward
    it and is removed once empty; the action keeps the target observation it recorded."""
    graph = _flagged_graph()
    expected = _identifying(graph, monkeypatch, "s0_home", "s2_settings", "s0_home", "s0_home")
    _describing_by_node_id(graph, monkeypatch)
    monkeypatch.setattr(audit, "soft_keyboard_hint", lambda _udid: "")

    (result,) = _re_explore(
        graph, [_retry()], _ReplayController(device_log), client, tmp_path, steps_per_issue=1
    )

    # The replays expect the source, the flagged action its edge's target.
    assert expected == ["s0_home", "s1_detail", "s0_home", None]

    assert (result["status"], result["moved_to"]) == ("moved", "s2_settings")
    assert not graph.graph.has_edge("s0_home", "s1_detail")
    edge = graph.graph["s0_home"]["s2_settings"]
    assert edge["actions"] == [[_GEAR_TAP], [_DETAIL_TAP]]
    assert edge["instructions"] == ["open settings", _FLAGGED]
    assert edge["target_observations"] == ["Settings", "Detail"]
    assert [(step["from"], step["to"]) for step in result["steps"]] == [("s0_home", "s0_home")]
    assert device_log == [*_FIRST_RESTART, _DETAIL_TAP, *_RESTART, _BACK]


@pytest.mark.usefixtures("_fake_device")
def test_a_retried_edge_is_recorded_as_a_transition_to_the_node_it_reaches(
    client: OpenRouter,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    device_log: list[object],
    step_records: Callable[[Path], list[dict[str, Any]]],
) -> None:
    """The flagged action and the screens before and after it are an edge retry, whose
    transition carries the edge's instruction and the node the action reached, the screen
    after it matched expecting the earliest member of the edge's target; the replays
    around it are navigation replays."""
    graph = _flagged_graph()
    reached = iter(["s0_home", "s2_settings", "s0_home"])
    monkeypatch.setattr(graph, "identify_state", lambda _observation, **_kwargs: next(reached))
    _describing_by_node_id(graph, monkeypatch)

    _re_explore(graph, [_retry()], _ReplayController(device_log), client, tmp_path)

    records = step_records(tmp_path / "steps")
    assert [(record["kind"], record["source"]) for record in records] == [
        ("observation", "navigation_replay"),
        ("match", "navigation_replay"),
        ("observation", "edge_retry"),
        ("action", "edge_retry"),
        ("observation", "edge_retry"),
        ("match", "edge_retry"),
        ("transition", "edge_retry"),
        ("observation", "navigation_replay"),
        ("match", "navigation_replay"),
    ]
    assert (records[5]["observation"], records[5]["expected_observation"]) == (
        "000004",
        "s1_detail-observation",
    )
    assert {key: records[6][key] for key in _STEP_FIELDS} == {
        "before": "000002",
        "after": "000004",
        "instruction": _FLAGGED,
        "element": None,
        "action": [_DETAIL_TAP],
        "from_node": "s0_home",
        "to_node": "s2_settings",
    }


@pytest.mark.parametrize(
    ("label", "outcome"),
    [
        ('Tap the "Detail" card', {"status": "relabelled", "instruction": 'Tap the "Detail" card'}),
        (None, {"status": "relabel_failed"}),
    ],
)
@pytest.mark.usefixtures("_fake_device")
def test_re_explore_relabels_only_the_flagged_instruction_when_its_action_reaches_the_target(
    label: str | None,
    outcome: dict[str, str],
    client: OpenRouter,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    device_log: list[object],
) -> None:
    graph = _flagged_graph()
    graph.add_edge(
        "s0_home",
        "s1_detail",
        [_SWIPE],
        details=EdgeDetails(instruction="scroll to the detail card"),
    )
    reached = iter(["s0_home", "s1_detail", "s0_home"])
    monkeypatch.setattr(graph, "identify_state", lambda *_args, **_kwargs: next(reached))
    _describing_by_node_id(graph, monkeypatch)
    described: list[dict[str, object]] = []

    def describe(**kwargs: object) -> str | None:
        described.append(kwargs)
        return label

    monkeypatch.setattr(audit, "describe_executed_action", describe)

    (result,) = _re_explore(graph, [_retry()], _ReplayController(device_log), client, tmp_path)

    assert result == {"issue": _retry(), **outcome, "steps": []}
    edge = graph.graph["s0_home"]["s1_detail"]
    assert edge["actions"] == [[_DETAIL_TAP], [_SWIPE]]
    assert edge["instructions"] == [label or _FLAGGED, "scroll to the detail card"]
    (call,) = described
    assert call["screenshot_b64"] == _webp(_png(0))
    assert call["action"] == [_DETAIL_TAP]
    assert call["screen_size"] == (1080, 1920)
    assert call["settings"] == ModelSettings("instruction-model")


@pytest.mark.parametrize("arrives_at", [None, "s2_settings"])
@pytest.mark.usefixtures("_fake_device")
def test_re_explore_leaves_the_graph_unchanged_when_the_replay_misses_the_source(
    arrives_at: str | None,
    client: OpenRouter,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    device_log: list[object],
) -> None:
    graph = _flagged_graph()
    _describing_by_node_id(graph, monkeypatch)
    if arrives_at is None:
        monkeypatch.setattr(graph, "find_path", lambda *_args: None)
    monkeypatch.setattr(graph, "identify_state", lambda *_args, **_kwargs: arrives_at)
    before = _edges(graph)
    controller = _ReplayController(device_log)

    (result,) = _re_explore(graph, [_retry()], controller, client, tmp_path, steps_per_issue=1)

    assert result == {"issue": _retry(), "status": "unverified", "reached": arrives_at}
    assert _edges(graph) == before
    assert controller.actions == []


@pytest.mark.parametrize(
    "issue",
    [
        _retry(instruction="open the details screen"),
        _retry(target="s2_settings"),
        _retry(target="s9_missing"),
    ],
    ids=["paraphrased", "on another edge", "no such edge"],
)
@pytest.mark.usefixtures("_fake_device")
def test_re_explore_explores_from_the_source_when_the_flagged_instruction_is_not_on_the_edge(
    issue: dict[str, Any],
    client: OpenRouter,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    device_log: list[object],
) -> None:
    graph = _flagged_graph()
    graph.add_edge("s0_home", "s1_detail", [_SWIPE])
    monkeypatch.setattr(graph, "identify_state", lambda *_args, **_kwargs: "s0_home")
    _describing_by_node_id(graph, monkeypatch)
    monkeypatch.setattr(audit, "soft_keyboard_hint", lambda _udid: "")
    before = _edges(graph)

    (result,) = _re_explore(
        graph, [issue], _ReplayController(device_log), client, tmp_path, steps_per_issue=1
    )

    assert result["status"] == "instruction_not_found"
    assert [(step["from"], step["to"]) for step in result["steps"]] == [("s0_home", "s0_home")]
    assert {key: _edges(graph)[key] for key in before} == before
    assert device_log == [*_FIRST_RESTART, _BACK]


@pytest.mark.parametrize(
    ("issue", "package", "status", "instructions"),
    [
        (
            _retry("open help", target="ext_com.browser"),
            "com.browser",
            "relabelled",
            {"s1_detail": [_FLAGGED], "ext_com.browser": ["Tap Help"]},
        ),
        (
            _retry(),
            "com.browser",
            "moved",
            {"ext_com.browser": ["open help", _FLAGGED]},
        ),
        (
            _retry(),
            "com.unknown",
            "unverified",
            {"s1_detail": [_FLAGGED], "ext_com.browser": ["open help"]},
        ),
    ],
    ids=["to the target's app", "to another app's node", "to an app the graph has no node for"],
)
@pytest.mark.usefixtures("_fake_device")
def test_re_explore_takes_a_flagged_action_that_leaves_the_app_to_that_app_s_node(
    issue: dict[str, Any],
    package: str,
    status: str,
    instructions: dict[str, list[str]],
    client: OpenRouter,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    device_log: list[object],
) -> None:
    graph = _flagged_graph()
    graph.graph.add_node("ext_com.browser", replay_misses=0)
    graph.add_edge(
        "s0_home", "ext_com.browser", [_GEAR_TAP], details=EdgeDetails(instruction="open help")
    )
    monkeypatch.setattr(graph, "identify_state", lambda *_args, **_kwargs: "s0_home")
    _describing_by_node_id(graph, monkeypatch)
    monkeypatch.setattr(audit, "describe_executed_action", lambda **_kwargs: "Tap Help")

    (result,) = _re_explore(
        graph, [issue], _ReplayController(device_log, package), client, tmp_path
    )

    assert result["status"] == status
    out_of_home = {
        target: data["instructions"]
        for _source, target, data in graph.graph.out_edges("s0_home", data=True)
    }
    assert out_of_home == {"s2_settings": ["open settings"], **instructions}
    assert "ext_com.unknown" not in graph.graph


@pytest.mark.usefixtures("_fake_device")
def test_re_explore_identifies_the_screen_a_flagged_action_reaches_by_the_device_s_package(
    client: OpenRouter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, device_log: list[object]
) -> None:
    """A system dialog on top of the app is a screen of its own package, not an ext_ node."""
    graph = _flagged_graph()
    graph.graph.add_node("s3_camera_permission", replay_misses=0)
    reached = iter(["s0_home", "s3_camera_permission", "s0_home"])
    identified: list[tuple[str, str, str]] = []

    def identify(observation: Observation, **_kwargs: object) -> str:
        identified.append((observation.activity, observation.screenshot_b64, observation.package))
        return next(reached)

    monkeypatch.setattr(graph, "identify_state", identify)
    _describing_by_node_id(graph, monkeypatch)
    dialog = "com.google.android.permissioncontroller"

    (result,) = _re_explore(
        graph, [_retry()], _ReplayController(device_log, dialog), client, tmp_path
    )

    assert (result["status"], result["moved_to"]) == ("moved", "s3_camera_permission")
    assert identified[:2] == [
        ("MainActivity", _webp(_png(0)), "com.demo"),
        ("MainActivity", _webp(_png(1)), dialog),
    ]


@pytest.mark.usefixtures("_fake_device")
def test_re_explore_moves_a_flagged_action_that_opens_an_outside_web_page_to_the_ext_web_edge(
    client: OpenRouter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, device_log: list[object]
) -> None:
    """The restart before the next replay leaves the page, so no back press is needed."""
    graph = _flagged_graph()
    # identify_state is faked below, so add the node it would create.
    graph.graph.add_node(EXTERNAL_WEB_NODE, replay_misses=0)
    reached = iter(["s0_home", EXTERNAL_WEB_NODE, "s0_home"])
    monkeypatch.setattr(graph, "identify_state", lambda *_args, **_kwargs: next(reached))
    _describing_by_node_id(graph, monkeypatch)

    (result,) = _re_explore(graph, [_retry()], _ReplayController(device_log), client, tmp_path)

    assert (result["status"], result["moved_to"]) == ("moved", EXTERNAL_WEB_NODE)
    assert not graph.graph.has_edge("s0_home", "s1_detail")
    assert graph.graph["s0_home"][EXTERNAL_WEB_NODE]["instructions"] == [_FLAGGED]
    assert graph.graph.out_degree(EXTERNAL_WEB_NODE) == 0
    assert device_log == [*_FIRST_RESTART, _DETAIL_TAP, *_RESTART]


@pytest.mark.usefixtures("_fake_device")
def test_re_explore_skips_a_flagged_edge_out_of_an_external_node(
    client: OpenRouter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, device_log: list[object]
) -> None:
    graph = _flagged_graph()
    graph.graph.add_node(EXTERNAL_WEB_NODE, replay_misses=0)
    graph.add_edge(
        "s0_home", EXTERNAL_WEB_NODE, [_GEAR_TAP], details=EdgeDetails(instruction="open help")
    )
    # add_edge refuses an edge out of an ext_ node, but a loaded graph file can hold one.
    graph.graph.add_edge(
        EXTERNAL_WEB_NODE, "s0_home", actions=[[_BACK]], instructions=["go home"], num_steps=[1]
    )
    monkeypatch.setattr(graph, "identify_state", lambda *_args, **_kwargs: EXTERNAL_WEB_NODE)
    before = _edges(graph)
    issue = _retry("go home", source=EXTERNAL_WEB_NODE, target="s0_home")

    (result,) = _re_explore(
        graph, [issue], _ReplayController(device_log), client, tmp_path, steps_per_issue=1
    )

    assert result == {"issue": issue, "status": "skipped", "reason": "external source"}
    assert _edges(graph) == before
    assert device_log == []


@pytest.mark.usefixtures("_fake_device")
def test_re_explore_does_not_explore_when_the_replay_after_a_correction_misses_the_source(
    client: OpenRouter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, device_log: list[object]
) -> None:
    graph = _flagged_graph()
    _describing_by_node_id(graph, monkeypatch)
    reached = iter(["s0_home", "s2_settings", "s2_settings"])
    monkeypatch.setattr(graph, "identify_state", lambda *_args, **_kwargs: next(reached))

    (result,) = _re_explore(
        graph, [_retry()], _ReplayController(device_log), client, tmp_path, steps_per_issue=1
    )

    assert (result["status"], result["steps"]) == ("moved", [])
    assert device_log == [*_FIRST_RESTART, _DETAIL_TAP, *_RESTART]


@pytest.mark.usefixtures("_fake_device")
def test_re_explore_resolves_each_issue_s_nodes_when_it_reaches_that_issue(
    client: OpenRouter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, device_log: list[object]
) -> None:
    """The audit merges nodes before re-exploring, so an issue can name a node a merge
    removed."""
    graph = _flagged_graph()
    graph.graph.add_node("s3_home_copy", replay_misses=0)
    graph.graph.add_node("s4_detail_copy", replay_misses=0)
    graph.merge_nodes("s0_home", "s3_home_copy")
    graph.merge_nodes("s1_detail", "s4_detail_copy")
    reached = iter(["s0_home", "s1_detail", "s0_home", "s0_home", "s2_settings"])
    monkeypatch.setattr(graph, "identify_state", lambda *_args, **_kwargs: next(reached))
    _describing_by_node_id(graph, monkeypatch)
    monkeypatch.setattr(audit, "describe_executed_action", lambda **_kwargs: "Tap the card")

    results = _re_explore(
        graph,
        [
            _retry(source="s3_home_copy", target="s4_detail_copy"),
            {**_inspection(), "type": "explore_node", "node": "s2_settings", "expected_pages": []},
        ],
        _ReplayController(device_log),
        client,
        tmp_path,
    )

    assert [r["status"] for r in results] == ["relabelled", "explored"]
    assert graph.graph["s0_home"]["s1_detail"]["instructions"] == ["Tap the card"]
    assert results[1]["navigated_to"] == "s2_settings"
    assert device_log == [*_FIRST_RESTART, _DETAIL_TAP, *_RESTART, *_RESTART, _GEAR_TAP]


_SCROLL_CLOCKS = {"action": "swipe", "x1": 540, "y1": 1200, "x2": 540, "y2": 600}
_OPEN_ALARMS = {"action": "tap", "x": 100, "y": 1800}
_SCROLL_ALARMS = {"action": "swipe", "x1": 540, "y1": 1500, "x2": 540, "y2": 300}
_OPEN_ALARM = {"action": "tap", "x": 540, "y": 400}


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
@pytest.mark.usefixtures("_fake_device")
def test_re_explore_replays_from_the_node_the_app_opened_on(
    opened_on: str,
    replayed: list[object],
    client: OpenRouter,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    device_log: list[object],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An app can reopen on the screen used last instead of its start screen. The replay
    identifies the screen the restart shows, expecting the start node, and replays from
    that node when a recorded path leads from it, from the start node otherwise, each
    action after its setup."""
    graph = _graph_with_nodes(
        *((node, {}) for node in ("s0_alarms", "s1_clock", "s2_alarm", "s3_whats_new"))
    )
    _describing_by_node_id(graph, monkeypatch)
    graph.add_edge(
        "s1_clock", "s0_alarms", [_OPEN_ALARMS], details=EdgeDetails(setup=[_SCROLL_CLOCKS])
    )
    graph.add_edge(
        "s0_alarms", "s2_alarm", [_OPEN_ALARM], details=EdgeDetails(setup=[_SCROLL_ALARMS])
    )
    expected = _identifying(graph, monkeypatch, opened_on, "s2_alarm")
    caplog.set_level(logging.INFO, logger=audit.logger.name)

    (result,) = _re_explore(
        graph,
        [{**_inspection(), "type": "explore_node", "node": "s2_alarm", "expected_pages": []}],
        _LoggingController("com.demo", device_log),
        client,
        tmp_path,
    )

    assert device_log == [*_FIRST_RESTART, *replayed]
    assert expected == ["s0_alarms", "s2_alarm"]
    assert result["navigated_to"] == "s2_alarm"
    opened_elsewhere = f"demo opened on {opened_on}, not on the start node s0_alarms"
    assert (opened_elsewhere in caplog.text) == (opened_on != "s0_alarms")


@pytest.mark.usefixtures("_fake_device")
def test_re_explore_replays_each_action_after_its_setup(
    client: OpenRouter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, device_log: list[object]
) -> None:
    graph = _graph_with_nodes(("s0_home", {}), ("s1_detail", {}))
    _describing_by_node_id(graph, monkeypatch)
    graph.add_edge("s0_home", "s1_detail", [_DETAIL_TAP], details=EdgeDetails(setup=[_SWIPE]))
    _identifying(graph, monkeypatch, "s0_home", "s1_detail")
    issue = {**_inspection(), "type": "explore_node", "node": "s1_detail", "expected_pages": []}

    (result,) = _re_explore(
        graph, [issue], _LoggingController("com.demo", device_log), client, tmp_path
    )

    assert result["navigated_to"] == "s1_detail"
    assert device_log == [*_FIRST_RESTART, _SWIPE, _DETAIL_TAP]


@pytest.mark.usefixtures("_fake_device")
def test_re_explore_sends_a_flagged_action_s_setup_before_it_and_moves_the_setup_with_it(
    client: OpenRouter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, device_log: list[object]
) -> None:
    graph = _flagged_graph()
    graph.graph["s0_home"]["s1_detail"]["setups"] = [[_SWIPE]]
    _identifying(graph, monkeypatch, "s0_home", "s2_settings", "s0_home")
    _describing_by_node_id(graph, monkeypatch)

    (result,) = _re_explore(graph, [_retry()], _ReplayController(device_log), client, tmp_path)

    assert (result["status"], result["moved_to"]) == ("moved", "s2_settings")
    assert graph.graph["s0_home"]["s2_settings"]["setups"] == [None, [_SWIPE]]
    assert device_log == [*_FIRST_RESTART, _SWIPE, _DETAIL_TAP, *_RESTART]


@pytest.mark.usefixtures("_fake_device")
def test_re_explore_leaves_a_flagged_action_unverified_when_no_recorded_action_opens_its_menu(
    client: OpenRouter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, device_log: list[object]
) -> None:
    """The flagged tap was sent in a menu with an unknown setup, so the retry could not
    send it in that menu: it sends nothing and leaves the graph as it was."""
    graph = _flagged_graph()
    _describing_by_node_id(graph, monkeypatch)
    graph.graph["s0_home"]["s1_detail"]["open_menus"] = ["overflow menu"]
    edges = _edges(graph)
    _identifying(graph, monkeypatch, "s0_home")

    (result,) = _re_explore(graph, [_retry()], _ReplayController(device_log), client, tmp_path)

    assert result == {"issue": _retry(), "status": "unverified", "reached": "s0_home"}
    assert device_log == _FIRST_RESTART
    assert _edges(graph) == edges


@pytest.mark.usefixtures("_fake_device")
def test_re_explore_records_the_menu_open_over_the_screen_a_step_is_sent_on(
    client: OpenRouter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, device_log: list[object]
) -> None:
    row_tap = {"action": "tap", "x": 540, "y": 1700}
    graph = _flagged_graph()
    drawer = {"open_menu": "navigation drawer"}
    _identifying(graph, monkeypatch, "s0_home", ("s1_detail", drawer), "s2_settings")
    _describing_by_node_id(graph, monkeypatch)
    monkeypatch.setattr(audit, "soft_keyboard_hint", lambda _udid: "")
    monkeypatch.setattr(audit, "predict_next_action", lambda **_kwargs: (row_tap, ""))
    issue = {**_inspection(), "type": "explore_node", "node": "s1_detail", "expected_pages": []}

    _re_explore(
        graph,
        [issue],
        _LoggingController("com.demo", device_log),
        client,
        tmp_path,
        steps_per_issue=1,
    )

    assert graph.graph["s1_detail"]["s2_settings"]["open_menus"] == ["navigation drawer"]


@pytest.mark.parametrize(
    ("after_restart", "reached", "scroll_setup", "row_setup"),
    [
        (["s0_home", "s1_detail"], "s1_detail", [], [_SWIPE]),
        (["s0_home", "s0_home"], "s0_home", None, None),
        (["s1_detail"], "s1_detail", [], [_SWIPE]),
    ],
    ids=["replay reached", "replay missed", "app opened on the node"],
)
@pytest.mark.usefixtures("_fake_device")
def test_re_explore_records_each_step_s_setup_known_only_on_the_node_its_replay_reached(
    after_restart: list[str],
    reached: str,
    scroll_setup: list[object] | None,
    row_setup: list[object] | None,
    client: OpenRouter,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    device_log: list[object],
) -> None:
    """The launch arrives at the screen the app opens on, so a replay that sends nothing
    because the app opened on its node has reached it as one that sends its path has."""
    row_tap = {"action": "tap", "x": 540, "y": 1700}
    graph = _flagged_graph()
    scrolled = {"first_visible_item": "Storage"}
    _identifying(graph, monkeypatch, *after_restart, (reached, scrolled), "s2_settings")
    _describing_by_node_id(graph, monkeypatch)
    monkeypatch.setattr(audit, "soft_keyboard_hint", lambda _udid: "")
    sent = iter([_SWIPE, row_tap])
    monkeypatch.setattr(audit, "predict_next_action", lambda **_kwargs: (next(sent), ""))
    issue = {**_inspection(), "type": "explore_node", "node": "s1_detail", "expected_pages": []}

    _re_explore(
        graph,
        [issue],
        _LoggingController("com.demo", device_log),
        client,
        tmp_path,
        steps_per_issue=2,
    )

    edges = graph.graph[reached]
    assert edges[reached]["setups"] == [scroll_setup]
    assert (edges["s2_settings"]["actions"][-1], edges["s2_settings"]["setups"][-1]) == (
        [row_tap],
        row_setup,
    )


@pytest.mark.usefixtures("_fake_device")
def test_re_explore_takes_no_step_from_a_screen_the_graph_has_no_node_for(
    client: OpenRouter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, device_log: list[object]
) -> None:
    graph, _graph_path = _two_node_graph(tmp_path)
    monkeypatch.setattr(graph, "identify_state", lambda *_args, **_kwargs: "s9_unknown")
    issue = {**_inspection(), "type": "explore_node", "node": "s1_detail", "expected_pages": []}

    (result,) = _re_explore(
        graph,
        [issue],
        _LoggingController("com.demo", device_log),
        client,
        tmp_path,
        steps_per_issue=1,
    )

    assert result == {
        "issue": issue,
        "status": "explored",
        "navigated_to": "s9_unknown",
        "steps": [],
    }
    replay = {"action": "tap", "coordinate": [10, 20]}
    assert device_log == [*_FIRST_RESTART, replay]


@pytest.mark.usefixtures("_fake_device")
def test_re_explore_spends_a_step_on_an_action_model_wait_without_recording_it(
    client: OpenRouter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, device_log: list[object]
) -> None:
    graph, _graph_path = _two_node_graph(tmp_path)
    monkeypatch.setattr(graph, "identify_state", lambda *_args, **_kwargs: "s0_home")
    monkeypatch.setattr(audit, "soft_keyboard_hint", lambda _udid: "")
    predictions: list[dict[str, object]] = []

    def wait(**kwargs: object) -> tuple[dict[str, str], str]:
        predictions.append(kwargs)
        return {"action": "wait"}, ""

    monkeypatch.setattr(audit, "predict_next_action", wait)
    before = _edges(graph)
    issue = {**_inspection(), "type": "explore_node", "node": "s0_home", "expected_pages": []}

    (result,) = _re_explore(
        graph,
        [issue],
        _LoggingController("com.demo", device_log),
        client,
        tmp_path,
        steps_per_issue=2,
    )

    assert result == {"issue": issue, "status": "explored", "navigated_to": "s0_home", "steps": []}
    assert len(predictions) == 2
    assert _edges(graph) == before
    assert device_log == _FIRST_RESTART


_HOME_SCREEN = {"activity": "com.demo.Home", "screenshot": _png(200), "package": "com.demo"}
_DETAIL_SCREEN = {**_HOME_SCREEN, "activity": "com.demo.Detail", "screenshot": _png(150)}
_BROWSER_SCREEN = {
    "activity": "com.browser.Main",
    "screenshot": _png(100),
    "package": "com.browser",
}


@pytest.mark.parametrize(
    ("after_the_tap", "leaving"),
    [
        ([_BROWSER_SCREEN, _HOME_SCREEN], [_BACK]),
        (
            [_BROWSER_SCREEN, _BROWSER_SCREEN, _BROWSER_SCREEN, _HOME_SCREEN],
            [_BACK, *_RESTART],
        ),
    ],
    ids=["back returns", "back leaves the other app in front"],
)
@pytest.mark.usefixtures("_fake_device")
def test_re_explore_returns_to_the_app_and_records_nothing_when_a_step_leaves_it(
    after_the_tap: list[dict[str, Any]],
    leaving: list[object],
    client: OpenRouter,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    device_log: list[object],
    screens_by_log: Callable[..., DeviceController],
) -> None:
    """Only the app's screens are identified, and the next step starts from one."""
    graph, _graph_path = _two_node_graph(tmp_path)
    nodes = {"com.demo.Home": "s0_home", "com.demo.Detail": "s1_detail"}
    expected: list[str | None] = []

    answered: dict[str, str] = {}

    def identify(observation: Observation, *, expected_observation: str | None = None) -> str:
        known = answered.get(expected_observation or "")
        expected.append(known or graph.node_of(expected_observation))
        answered[observation.observation_id] = nodes[observation.activity]
        return nodes[observation.activity]

    monkeypatch.setattr(graph, "identify_state", identify)
    monkeypatch.setattr(audit, "soft_keyboard_hint", lambda _udid: "")
    tap = {"action": "tap", "x": 1, "y": 2}
    monkeypatch.setattr(audit, "predict_next_action", lambda **_kwargs: (tap, ""))
    issue = {**_inspection(), "type": "explore_node", "node": "s0_home", "expected_pages": []}
    # The replay's restart shows home, and the second step's tap opens the detail page.
    controller = screens_by_log(*[_HOME_SCREEN] * 4, *after_the_tap, _DETAIL_SCREEN)

    (result,) = _re_explore(graph, [issue], controller, client, tmp_path, steps_per_issue=2)

    assert [(step["step"], step["from"], step["to"]) for step in result["steps"]] == [
        (2, "s0_home", "s1_detail")
    ]
    assert set(graph.graph.edges) == {("s0_home", "s1_detail")}
    assert device_log == [*_FIRST_RESTART, tap, *leaving, tap]
    # Back from the other app expects the screen the step left.
    assert expected == ["s0_home", "s0_home", None]


@pytest.mark.parametrize(
    "after_the_tap",
    [_BROWSER_SCREEN, {**_HOME_SCREEN, "activity": "com.demo.Article", "screenshot": _png(50)}],
    ids=["another app", "outside web page"],
)
@pytest.mark.usefixtures("_fake_device")
def test_re_explore_records_the_setup_of_the_step_after_one_that_leaves_the_app_as_unknown(
    after_the_tap: dict[str, Any],
    client: OpenRouter,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    screens_by_log: Callable[..., DeviceController],
) -> None:
    """Back may return to the screen the step left or a restart to the start screen, and
    neither is an action a replay could send."""
    graph, _graph_path = _two_node_graph(tmp_path)
    graph.graph.add_node(EXTERNAL_WEB_NODE, replay_misses=0)
    nodes = {
        "com.demo.Home": "s0_home",
        "com.demo.Detail": "s1_detail",
        "com.demo.Article": EXTERNAL_WEB_NODE,
    }
    monkeypatch.setattr(
        graph, "identify_state", lambda observation, **_kwargs: nodes[observation.activity]
    )
    _describing_by_node_id(graph, monkeypatch)
    monkeypatch.setattr(audit, "soft_keyboard_hint", lambda _udid: "")
    tap = {"action": "tap", "x": 1, "y": 2}
    monkeypatch.setattr(audit, "predict_next_action", lambda **_kwargs: (tap, ""))
    issue = {**_inspection(), "type": "explore_node", "node": "s0_home", "expected_pages": []}
    controller = screens_by_log(*[_HOME_SCREEN] * 4, after_the_tap, _HOME_SCREEN, _DETAIL_SCREEN)

    _re_explore(graph, [issue], controller, client, tmp_path, steps_per_issue=2)

    edge = graph.graph["s0_home"]["s1_detail"]
    assert (edge["actions"][-1], edge["setups"][-1]) == ([tap], None)


@pytest.mark.usefixtures("_fake_device")
def test_re_explore_skips_a_missing_node_and_ignores_an_issue_of_another_type(
    client: OpenRouter, tmp_path: Path, device_log: list[object]
) -> None:
    graph, _graph_path = _two_node_graph(tmp_path)
    missing = {
        **_inspection(),
        "type": "explore_node",
        "node": "s9_missing",
        "expected_pages": ["Settings"],
    }

    results = _re_explore(
        graph,
        [missing, {"type": "something_else", "node": "s0_home"}],
        _LoggingController("com.demo", device_log),
        client,
        tmp_path,
        steps_per_issue=1,
    )

    assert results == [{"issue": missing, "status": "skipped", "reason": "node not found"}]
    assert device_log == []


def _install_fake_adb(monkeypatch: pytest.MonkeyPatch, controller: DeviceController) -> None:
    monkeypatch.setattr(audit, "AdbController", lambda _config, _logger: controller)


@pytest.mark.usefixtures("model_environment", "opened_clients", "no_sleep", "device_log")
def test_re_explore_saves_its_corrections_and_the_outcome_of_each_issue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    add_member: Callable[..., None],
    write_image_vectors: Callable[..., None],
) -> None:
    graph_dir = tmp_path / "graphs"
    app_dir = graph_dir / "demo"
    _save_with_members(_flagged_graph(), app_dir / "demo.json", add_member, write_image_vectors)
    config = tmp_path / "audit.yaml"
    config.write_text(
        json.dumps(
            {
                "device": {"udid": "test-device"},
                "experiment": {"graph_dir": str(graph_dir)},
                "apps": [{"name": "demo", "package_name": "com.demo"}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    _install_fake_adb(monkeypatch, FakeController("com.demo"))
    issues = [_retry(), _retry("open settings", target="s2_settings")]
    monkeypatch.setattr(audit, "run_audit", lambda *_args: {"issues": issues})
    reached = iter(["s0_home", "s2_settings", "s0_home"] * 2)
    identify_files: list[Path | None] = []

    monkeypatch.setattr(
        audit, "RunEnvironment", lambda controller, *_args, **_kwargs: nullcontext(controller)
    )

    def identify(self: GraphManager, *_args: object, **_kwargs: object) -> str:
        identify_files.append(self.identify_file)
        return next(reached)

    monkeypatch.setattr(GraphManager, "identify_state", identify)
    monkeypatch.setattr(GraphManager, "normalize_all_edges", lambda _self: 0)
    monkeypatch.setattr(audit, "describe_executed_action", lambda **_kwargs: "Tap the gear icon")
    explored_decisions = (app_dir / "demo_identify.jsonl").read_bytes()

    assert audit.main(["--config", str(config), "--re-explore", "--steps-per-issue", "0"]) == 0

    # The audited copy's decisions start from the explored graph's, and its own go there.
    assert set(identify_files) == {app_dir / "demo_audited_identify.jsonl"}
    assert (app_dir / "demo_audited_identify.jsonl").read_bytes() == explored_decisions
    assert (app_dir / "demo_identify.jsonl").read_bytes() == explored_decisions
    outcomes = json.loads((app_dir / "demo_re_explore.json").read_text(encoding="utf-8"))
    assert [(o["issue"], o["status"]) for o in outcomes] == [
        (issues[0], "moved"),
        (issues[1], "relabelled"),
    ]
    audited = json.loads((app_dir / "demo_audited.json").read_text(encoding="utf-8"))
    (edge,) = audited["edges"]
    assert (edge["source"], edge["target"]) == ("s0_home", "s2_settings")
    assert edge["actions"] == [[_GEAR_TAP], [_DETAIL_TAP]]
    assert edge["instructions"] == ["Tap the gear icon", _FLAGGED]


@pytest.mark.usefixtures("model_environment", "opened_clients")
def test_audit_rejects_invalid_matching_settings_before_reading_a_graph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "audit.yaml"
    config.write_text("apps: []\nvlm:\n  matching:\n    verifier_calls: -1\n", encoding="utf-8")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    with pytest.raises(SystemExit) as caught:
        audit.main(["--config", str(config)])

    assert caught.value.code == 2
    assert "vlm.matching.verifier_calls must be a whole number" in capsys.readouterr().err


@pytest.mark.parametrize("re_explore", [False, True])
def test_audit_requires_only_the_models_it_uses(
    re_explore: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    opened_clients: list[OpenRouter],
) -> None:
    config = tmp_path / "audit.yaml"
    config.write_text("apps: []\n", encoding="utf-8")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("APP_GRAPH_PAGE_DETAIL_MODEL", "provider/detail")
    monkeypatch.setenv("APP_GRAPH_EMBEDDING_MODEL", "provider/embedding")
    args = ["--config", str(config)]
    if re_explore:
        with pytest.raises(SystemExit) as caught:
            audit.main([*args, "--re-explore"])
        assert caught.value.code == 2
        error = capsys.readouterr().err
        assert "APP_GRAPH_INSTRUCTION_MODEL" in error
        assert "APP_GRAPH_ACTION_MODEL" in error
        assert opened_clients == []
    else:
        assert audit.main(args) == 0


@pytest.mark.parametrize("exceptional", [False, True])
@pytest.mark.usefixtures("model_environment")
def test_audit_cli_opens_one_client_and_closes_it_on_exit(
    exceptional: bool,
    tmp_path: Path,
    opened_clients: list[OpenRouter],
    router_http: httpx.Client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    graph_dir = tmp_path / "graphs"
    app_dir = graph_dir / "demo"
    app_dir.mkdir(parents=True)
    document = {"matching": DEFAULT_MATCHING._asdict(), "nodes": [], "edges": []}
    (app_dir / "demo.json").write_text(json.dumps(document))
    config = _write_config(tmp_path, str(graph_dir))

    def audit_graph(graph: GraphManager, _app: str, _path: Path) -> dict[str, object]:
        assert opened_clients == [graph.client]
        assert opened_clients[0].sdk_configuration.client is router_http
        if exceptional:
            msg = "audit failed"
            raise RuntimeError(msg)
        return {"issues": []}

    monkeypatch.setattr(audit, "run_audit", audit_graph)
    monkeypatch.setattr(audit, "normalize_and_save_audited_graph", lambda *_args: 0)
    if exceptional:
        with pytest.raises(RuntimeError, match="audit failed"):
            audit.main(["-c", str(config)])
    else:
        assert audit.main(["-c", str(config)]) == 0
    assert len(opened_clients) == 1
    assert opened_clients[0].sdk_configuration.client is None


@pytest.mark.usefixtures("model_environment", "opened_clients")
def test_the_audit_embeds_the_members_the_explored_graph_has_no_image_vector_for(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    router_requests: list[httpx.Request],
    router_responses: list[httpx.Response],
    embedding_body: dict[str, object],
    write_graph_file: Callable[[Path, dict[str, Any]], None],
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    app_dir = tmp_path / "graphs" / "demo"
    app_dir.mkdir(parents=True)
    write_graph_file(app_dir / "demo.json", {"nodes": [{"id": "s0_home"}], "edges": []})
    router_responses.append(httpx.Response(200, json=embedding_body))
    monkeypatch.setattr(audit, "run_audit", lambda *_args: {"issues": []})
    config = _write_config(tmp_path, str(tmp_path / "graphs"))

    assert audit.main(["-c", str(config), "--app", "demo"]) == 0

    assert [str(request.url) for request in router_requests] == [
        "https://openrouter.ai/api/v1/embeddings"
    ]
    assert json.loads((app_dir / "steps" / "image_emb.json").read_text(encoding="utf-8")) == {
        "model": "google/gemini-embedding-2",
        "embeddings": {"s0_home-observation": [0.25, 0.75]},
    }


@pytest.mark.usefixtures("model_environment", "opened_clients")
def test_the_audited_copy_starts_its_text_vectors_from_the_explored_graph_s(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    router_requests: list[httpx.Request],
    write_graph_file: Callable[[Path, dict[str, Any]], None],
    write_image_vectors: Callable[..., None],
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    app_dir = tmp_path / "graphs" / "demo"
    app_dir.mkdir(parents=True)
    write_graph_file(app_dir / "demo.json", {"nodes": [{"id": "s0_home"}], "edges": []})
    member = "s0_home-observation"
    write_image_vectors(app_dir / "steps", [member])
    explored = {"model": "google/gemini-embedding-2", "embeddings": {member: [1.0, 0.0]}}
    (app_dir / "demo_text_emb.json").write_text(json.dumps(explored), encoding="utf-8")
    monkeypatch.setattr(audit, "run_audit", lambda *_args: {"issues": []})
    config = _write_config(tmp_path, str(tmp_path / "graphs"))

    assert audit.main(["-c", str(config), "--app", "demo"]) == 0

    assert router_requests == []
    audited = json.loads((app_dir / "demo_audited_text_emb.json").read_text(encoding="utf-8"))
    assert audited == explored
    assert json.loads((app_dir / "demo_text_emb.json").read_text(encoding="utf-8")) == explored


@pytest.mark.usefixtures("model_environment", "opened_clients")
def test_audit_skips_a_configured_app_without_a_graph_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    config = _write_config(tmp_path, str(tmp_path / "graphs"))

    assert audit.main(["-c", str(config)]) == 0

    assert list(tmp_path.rglob("*")) == [config]


@pytest.mark.parametrize("same", [True, False])
@pytest.mark.usefixtures("model_environment", "opened_clients")
def test_audit_saves_the_merge_results_and_an_audited_graph_only_after_a_merge(
    same: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    add_member: Callable[..., None],
    write_image_vectors: Callable[..., None],
) -> None:
    """Each merge result names the sources that proposed the pair and the member of each
    node the verifier was shown."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    app_dir = tmp_path / "graphs" / "demo"
    _save_with_members(
        _graph_with_nodes(("s0_home", {}), ("s1_home", {})),
        app_dir / "demo.json",
        add_member,
        write_image_vectors,
    )
    config = _write_config(tmp_path, str(tmp_path / "graphs"))
    issue = _merge_issue()
    monkeypatch.setattr(audit, "run_audit", lambda *_args: {"issues": [issue]})
    monkeypatch.setattr(
        graph_manager, "verify_same_node", lambda *_args, **_kwargs: {"same": same, "reason": "r"}
    )
    monkeypatch.setattr(audit, "normalize_and_save_audited_graph", lambda *_args: 0)

    assert audit.main(["-c", str(config)]) == 0

    outcome = {"status": "merged", "kept": "s0_home"} if same else {"status": "kept_separate"}
    assert json.loads((app_dir / "demo_merge.json").read_text(encoding="utf-8")) == {
        "rounds": 1,
        "asked": 1,
        "deduced": 0,
        "merges": [{"kept": "s0_home", "removed": ["s1_home"]}] if same else [],
        "pairs": [
            {
                "node_a": "s0_home",
                "node_b": "s1_home",
                "sources": ["text_audit"],
                "findings": [{"source": "text_audit", "issue": issue}],
                "members": {"s0_home": "s0_home-observation", "s1_home": "s1_home-observation"},
                "image_score": 0.0,
                "asked": True,
                "round": 1,
                "same": same,
                "reason": "r",
                "reused": False,
                **outcome,
            }
        ],
    }
    assert (app_dir / "demo_audited.json").exists() == same


_BROWSER = "ext_com.android.chrome"


def _graph_with_a_dead_end(dead_end: str) -> GraphManager:
    """Home and List lead to each other; *dead_end* and the browser lead nowhere."""
    graph = _graph_with_nodes(("s0_home", {}), ("s1_list", {}), (dead_end, {}), (_BROWSER, {}))
    graph.add_edge("s0_home", "s1_list", {"action": "tap", "x": 10, "y": 20})
    graph.add_edge("s1_list", "s0_home", {"action": "back"})
    graph.add_edge("s1_list", dead_end, {"action": "tap", "x": 30, "y": 40})
    graph.add_edge("s0_home", _BROWSER, {"action": "tap", "x": 50, "y": 60})
    return graph


def _save_with_members(
    graph: GraphManager,
    path: Path,
    add_member: Callable[..., None],
    write_image_vectors: Callable[..., None],
) -> None:
    """Save *graph* to *path* with a member for every node, its image and image vector in
    the step log beside it and its describe output and text vector in the files beside
    the graph, so the audit can load it without embedding any. The vectors are
    orthogonal, so no two of the screens look or read alike to the audit."""
    graph.steps_dir = graph_steps_dir(path)
    graph.identify_file = identify_file_path(path)
    for node_id in list(graph.graph):
        add_member(graph, node_id)
    graph.save_graph(path)
    observations = [f"{node_id}-observation" for node_id in graph.graph]
    write_image_vectors(graph.steps_dir, observations)
    save_embeddings(
        text_embeddings_path(path),
        {
            observation: [float(index == position) for position in range(len(observations))]
            for index, observation in enumerate(observations)
        },
        model="google/gemini-embedding-2",
    )


def _no_path_to_start_line(app_dir: Path, in_app: int) -> str:
    return (
        f"{in_app} in-app and 1 external node(s) of {app_dir / 'demo_audited.json'} have no "
        f"path back to s0_home; listed in {app_dir / 'demo_audit.json'}"
    )


def _audit_with_the_model_reporting(
    issues: list[dict[str, Any]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Audit the saved demo graph while the model reports *issues*; return the app directory."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    # setup_logging replaces the root handlers, caplog's among them.
    monkeypatch.setattr(audit, "setup_logging", lambda **_kwargs: None)
    monkeypatch.setattr(
        GraphManager, "run_audit", lambda _self, **_kwargs: {"issues": issues, "summary": "ok"}
    )
    monkeypatch.setattr(GraphManager, "normalize_all_edges", lambda _self: 0)
    graph_dir = tmp_path / "graphs"
    config = _write_config(tmp_path, str(graph_dir))

    assert audit.main(["--config", str(config), "--app", "demo"]) == 0
    return graph_dir / "demo"


@pytest.mark.usefixtures("model_environment", "opened_clients")
def test_audit_adds_the_nodes_with_no_path_to_the_start_node_to_the_model_s_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    router_requests: list[httpx.Request],
    add_member: Callable[..., None],
    write_image_vectors: Callable[..., None],
) -> None:
    _save_with_members(
        _graph_with_a_dead_end("s2_dialog"),
        tmp_path / "graphs" / "demo" / "demo.json",
        add_member,
        write_image_vectors,
    )
    caplog.set_level(logging.INFO, logger=audit.logger.name)

    app_dir = _audit_with_the_model_reporting([], tmp_path, monkeypatch)

    report = json.loads((app_dir / "demo_audit.json").read_text(encoding="utf-8"))
    assert report == {
        "issues": [],
        "summary": "ok",
        "no_path_to_start": {
            "graph": "demo_audited.json",
            "start_node": "s0_home",
            "in_app": ["s2_dialog"],
            "external": [_BROWSER],
        },
    }
    assert router_requests == []
    line = _no_path_to_start_line(app_dir, in_app=1)
    assert (audit.logger.name, logging.WARNING, line) in caplog.record_tuples


@pytest.mark.usefixtures("model_environment", "opened_clients")
def test_audit_reports_no_path_to_the_start_node_for_the_graph_after_its_merges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    add_member: Callable[..., None],
    write_image_vectors: Callable[..., None],
) -> None:
    """A dead end merged into a node with a path back is gone from the audited graph."""
    app_dir = tmp_path / "graphs" / "demo"
    explored = _graph_with_a_dead_end("s2_list")
    assert explored.nodes_without_path_to_start().in_app == ["s2_list"]
    _save_with_members(explored, app_dir / "demo.json", add_member, write_image_vectors)
    monkeypatch.setattr(
        graph_manager,
        "verify_same_node",
        lambda *_args, **_kwargs: {"same": True, "reason": "same list"},
    )
    caplog.set_level(logging.INFO, logger=audit.logger.name)

    _audit_with_the_model_reporting([_merge_issue("s1_list", "s2_list")], tmp_path, monkeypatch)

    report = json.loads((app_dir / "demo_audit.json").read_text(encoding="utf-8"))
    assert report["no_path_to_start"] == {
        "graph": "demo_audited.json",
        "start_node": "s0_home",
        "in_app": [],
        "external": [_BROWSER],
    }
    line = _no_path_to_start_line(app_dir, in_app=0)
    assert (audit.logger.name, logging.INFO, line) in caplog.record_tuples


def _deny_device_work(monkeypatch: pytest.MonkeyPatch) -> None:
    def touched(*_args: object, **_kwargs: object) -> NoReturn:
        msg = "settings must be validated before any device work"
        raise AssertionError(msg)

    monkeypatch.setattr(audit, "AdbController", touched)
    monkeypatch.setattr(audit, "launch_app", touched)


@pytest.mark.parametrize(
    ("role_settings", "key", "message"),
    [
        ({"instruction": {"request_timeout": 0}}, "test", "instruction.request_timeout must be"),
        ({"page_detail": {"audit_request_timeout": -5}}, "test", "audit_request_timeout must be"),
        (
            {"page_detail": {"temprature": 0}},
            "test",
            "Unknown configuration keys: vlm.page_detail.temprature",
        ),
        ({}, None, "OPENROUTER_API_KEY is required"),
    ],
)
@pytest.mark.usefixtures("model_environment")
def test_audit_rejects_invalid_settings_before_any_device_work(
    role_settings: dict[str, object],
    key: str | None,
    message: str,
    tmp_path: Path,
    opened_clients: list[OpenRouter],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if key is not None:
        monkeypatch.setenv("OPENROUTER_API_KEY", key)
    _deny_device_work(monkeypatch)
    config = tmp_path / "audit.yaml"
    config.write_text(json.dumps({"apps": [], "vlm": role_settings}), encoding="utf-8")
    with pytest.raises(SystemExit) as caught:
        audit.main(["--config", str(config), "--re-explore"])
    assert caught.value.code == 2
    error = capsys.readouterr().err
    assert "Traceback" not in error
    (line,) = [line for line in error.splitlines() if line.startswith("app-graph-audit: error:")]
    assert message in line
    assert error.rstrip().endswith(line)
    assert opened_clients == []


@pytest.mark.usefixtures("model_environment")
def test_re_explore_session_shares_the_audit_client(
    tmp_path: Path, opened_clients: list[OpenRouter], monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_adb(monkeypatch, FakeController("com.demo"))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    config = tmp_path / "audit.yaml"
    config.write_text("apps: []\n", encoding="utf-8")
    assert audit.main(["--config", str(config), "--re-explore"]) == 0
    assert len(opened_clients) == 1
    assert opened_clients[0].sdk_configuration.client is None


@pytest.mark.usefixtures("model_environment", "opened_clients", "no_sleep")
def test_re_explore_grants_the_requested_permissions_before_launching_the_app(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_adb(monkeypatch, FakeController("com.demo"))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    graph_dir = tmp_path / "graphs"
    (graph_dir / "demo").mkdir(parents=True)
    document = {"matching": DEFAULT_MATCHING._asdict(), "nodes": [], "edges": []}
    (graph_dir / "demo" / "demo.json").write_text(json.dumps(document), encoding="utf-8")
    config = tmp_path / "audit.yaml"
    config.write_text(
        json.dumps(
            {
                "device": {"udid": "emulator-5554"},
                "experiment": {"graph_dir": str(graph_dir)},
                "apps": [{"name": "demo", "package_name": "com.demo"}],
            }
        ),
        encoding="utf-8",
    )
    events: list[tuple[object, ...]] = []

    monkeypatch.setattr(
        audit, "RunEnvironment", lambda controller, *_args, **_kwargs: nullcontext(controller)
    )

    def launch(*_args: object) -> str:
        events.append(("launch",))
        return "com.demo/.Home"

    def re_explore(
        *_args: object, target: audit.ReExploreTarget, **_kwargs: object
    ) -> list[dict[str, Any]]:
        events.append(("re-explore", target.launch_activity))
        return []

    monkeypatch.setattr(
        audit,
        "run_audit",
        lambda *_args: {"issues": [{**_inspection(), "type": "explore_node", "node": "s0"}]},
    )
    monkeypatch.setattr(
        audit, "grant_runtime_permissions", lambda *args: events.append(("grant", *args))
    )
    monkeypatch.setattr(audit, "launch_app", launch)
    monkeypatch.setattr(audit, "re_explore_issues", re_explore)
    monkeypatch.setattr(audit, "normalize_and_save_audited_graph", lambda *_args: 0)

    assert audit.main(["--config", str(config), "--re-explore"]) == 0
    # Re-exploration restarts the app through the component its launch started.
    assert events == [
        ("grant", "emulator-5554", "com.demo"),
        ("launch",),
        ("re-explore", "com.demo/.Home"),
    ]


@pytest.mark.parametrize("reached", ["s1_detail", "s2_settings"])
@pytest.mark.usefixtures("_fake_device")
def test_a_policy_block_after_correction_keeps_the_correction_outcome(
    reached: str,
    client: OpenRouter,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    device_log: list[object],
) -> None:
    graph = _flagged_graph()
    graph.graph.add_node("s3_recents", replay_misses=0, observations=["recents"])
    graph.add_edge(
        "s3_recents",
        "s0_home",
        [_GEAR_TAP],
        details=EdgeDetails(instruction='Tap "Send file" in the foreground menu.'),
    )
    _identifying(graph, monkeypatch, "s0_home", reached, "s3_recents")
    _describing_by_node_id(graph, monkeypatch)
    monkeypatch.setattr(audit, "describe_executed_action", lambda **_kwargs: "Open Detail")

    (result,) = _re_explore(
        graph, [_retry()], _ReplayController(device_log), client, tmp_path, steps_per_issue=1
    )

    assert result["status"] == ("moved" if reached == "s2_settings" else "relabelled")
    assert result["steps"] == []
    assert "policy" in result["reason"]
    assert _DETAIL_TAP in device_log
    if reached == "s2_settings":
        assert result["moved_to"] == reached
        assert not graph.graph.has_edge("s0_home", "s1_detail")
    else:
        assert result["instruction"] == "Open Detail"
        assert graph.graph["s0_home"]["s1_detail"]["instructions"] == ["Open Detail"]


@pytest.mark.parametrize("retry", [False, True])
@pytest.mark.usefixtures("_fake_device")
def test_recorded_setup_commitments_are_checked_before_execution(
    *,
    retry: bool,
    client: OpenRouter,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    device_log: list[object],
) -> None:
    graph = _flagged_graph()
    graph.add_edge(
        "s0_home",
        "s0_home",
        [_SWIPE],
        details=EdgeDetails(instruction='Tap "Send file", the second option in the menu.'),
    )
    graph.graph["s0_home"]["s1_detail"]["setups"] = [[_SWIPE]]
    _identifying(graph, monkeypatch, "s0_home", "s1_detail", "s0_home")
    _describing_by_node_id(graph, monkeypatch)
    monkeypatch.setattr(audit, "describe_executed_action", lambda **_kwargs: "Open Detail")
    issue = (
        _retry()
        if retry
        else {**_inspection(), "type": "explore_node", "node": "s1_detail", "expected_pages": []}
    )

    (result,) = _re_explore(graph, [issue], _ReplayController(device_log), client, tmp_path)

    assert result["status"] == "skipped"
    assert "policy" in result["reason"]
    assert _SWIPE not in device_log
    assert _DETAIL_TAP not in device_log
    assert graph.graph["s0_home"]["s1_detail"]["instructions"] == [_FLAGGED]


@pytest.mark.parametrize("retry", [False, True])
@pytest.mark.usefixtures("_fake_device")
def test_recorded_confirmation_uses_the_source_screen_context(
    *,
    retry: bool,
    client: OpenRouter,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    device_log: list[object],
) -> None:
    graph = _flagged_graph()
    graph.graph["s0_home"]["s1_detail"]["instructions"] = ['Tap "OK"']
    _identifying(graph, monkeypatch, "s0_home", "s1_detail", "s0_home")
    monkeypatch.setattr(
        graph, "latest_description", lambda _node: "Delete contact confirmation dialog"
    )
    monkeypatch.setattr(audit, "describe_executed_action", lambda **_kwargs: "Open Detail")
    issue = (
        _retry('Tap "OK"')
        if retry
        else {**_inspection(), "type": "explore_node", "node": "s1_detail", "expected_pages": []}
    )

    (result,) = _re_explore(graph, [issue], _ReplayController(device_log), client, tmp_path)

    assert result["status"] == "skipped"
    assert "policy" in result["reason"]
    assert _DETAIL_TAP not in device_log
    assert device_log == []


def test_re_exploration_reports_unsafe_display_replay_as_skipped(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    client: OpenRouter,
) -> None:
    graph = _graph_with_nodes(("s0", {"is_start": True, "activity": "Main", "package": "com.demo"}))

    def refuse(*_args: object) -> NoReturn:
        message = "Recorded display differs"
        raise UnsafeReplayError(message)

    monkeypatch.setattr(audit, "_retry_edge", refuse)
    session = audit.ReExploreSession(
        FakeController("com.demo"), ModelSettings("fake"), ModelSettings("fake")
    )
    results = audit.re_explore_issues(
        graph,
        [{**_inspection(), "type": "retry_edge", "instruction": "Open Settings"}],
        target=audit.ReExploreTarget(
            app_name="demo", package_name="com.demo", graph_path=tmp_path / "demo.json"
        ),
        session=session,
        client=client,
    )
    assert results == [
        {
            "issue": {**_inspection(), "type": "retry_edge", "instruction": "Open Settings"},
            "status": "skipped",
            "reason": "Recorded display differs",
        }
    ]


@pytest.mark.usefixtures("_fake_device")
def test_a_display_rejected_return_replay_retains_the_applied_edge_correction(
    client: OpenRouter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, device_log: list[object]
) -> None:
    graph = _flagged_graph()
    monkeypatch.setattr(graph, "identify_state", lambda *_args, **_kwargs: "s2_settings")
    replays = 0

    def replay(_run: object, target: str) -> str:
        nonlocal replays
        replays += 1
        if replays == 2:
            message = "Recorded display differs"
            raise UnsafeReplayError(message)
        return target

    monkeypatch.setattr(audit, "_replay_to", replay)
    (result,) = _re_explore(
        graph, [_retry()], _ReplayController(device_log), client, tmp_path, steps_per_issue=1
    )
    assert result["status"] == "moved"
    assert result["moved_to"] == "s2_settings"
    assert result["steps"] == []
    assert result["reason"] == "Recorded display differs"
    assert graph.graph["s0_home"]["s2_settings"]["instructions"][-1] == _FLAGGED
    assert not graph.graph.has_edge("s0_home", "s1_detail")
    assert device_log == [_DETAIL_TAP]


@pytest.mark.usefixtures("model_environment", "opened_clients", "no_sleep")
def test_re_explore_reports_a_missing_run_baseline_without_a_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _install_fake_adb(monkeypatch, FakeController("com.demo"))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    graph_dir = tmp_path / "graphs"
    (graph_dir / "demo").mkdir(parents=True)
    graph_path = graph_dir / "demo" / "demo.json"
    graph_path.write_text(
        json.dumps({"matching": DEFAULT_MATCHING._asdict(), "nodes": [], "edges": []})
    )
    config = tmp_path / "audit.yaml"
    config.write_text(
        json.dumps(
            {
                "device": {"udid": "test-device"},
                "experiment": {"graph_dir": str(graph_dir)},
                "apps": [{"name": "demo", "package_name": "com.demo"}],
            }
        )
    )
    monkeypatch.setattr(
        audit, "run_audit", lambda *_args: {"issues": [{"type": "explore_node", "node": "s0"}]}
    )
    assert audit.main(["--config", str(config), "--re-explore"]) == 1
    assert "original run baseline" in caplog.text
    assert "Traceback" not in caplog.text
    assert not graph_path.with_name("demo_run.json").exists()
