"""app-graph-audit parsing and pure helpers, exercised without adb or an emulator."""

from __future__ import annotations

import base64
import copy
import io
import json
import logging
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, NoReturn, override

import httpx
import pytest
from openrouter import OpenRouter
from PIL import Image

from android_app_graph.commands import audit
from android_app_graph.device import DeviceController
from android_app_graph.graph_files import (
    graph_steps_dir,
    observation_image_path,
    save_embeddings,
    text_embeddings_path,
)
from android_app_graph.screen_image import encode_screenshot
from android_app_graph.step_log import Observation, StepRecorder
from android_app_graph.utils import ModelSettings, graph_manager, open_client
from android_app_graph.utils.graph_manager import EXTERNAL_WEB_NODE, EdgeDetails, GraphManager

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


def _write_config(tmp_path: Path, graph_dir: str = "graphs") -> Path:
    config = tmp_path / "explore.yaml"
    config.write_text(
        f"experiment:\n  graph_dir: {graph_dir}\napps:\n  - name: demo\n    package_name: com.demo\n",
        encoding="utf-8",
    )
    return config


def _graph_with_nodes(*nodes: tuple[str, dict[str, Any]]) -> GraphManager:
    """Return a GraphManager holding the given nodes and no VLM clients."""
    graph = GraphManager()
    for node_id, attrs in nodes:
        graph.graph.add_node(node_id, **attrs)
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
    result = {"issues": [{"type": "retry_edge"}], "summary": "one issue"}
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

    def get_state(self) -> dict[str, Any]:
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
                "nodes": [
                    {
                        "id": "s0_home",
                        "activity": "com.demo.Home",
                        "page_description": "Home",
                        "observations": ["s0_home-observation"],
                        "representative_observation": "s0_home-observation",
                    },
                    {
                        "id": "s1_detail",
                        "activity": "com.demo.Detail",
                        "page_description": "Detail",
                        "observations": ["s1_detail-observation"],
                        "representative_observation": "s1_detail-observation",
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
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    for node_id in ("s0_home", "s1_detail"):
        _write_representative(graph_path, node_id)
    graph = GraphManager()
    graph.load_graph(graph_path)
    return graph, graph_path


def _identifying(
    graph: GraphManager, monkeypatch: pytest.MonkeyPatch, *nodes: str
) -> list[str | None]:
    """Make identify_state answer ``nodes`` in turn; return the list the expected node of
    each call is appended to."""
    expected: list[str | None] = []
    answers = iter(nodes)

    def identify(_observation: Observation, *, expected_node: str | None = None) -> str:
        expected.append(expected_node)
        return next(answers)

    monkeypatch.setattr(graph, "identify_state", identify)
    return expected


def _write_representative(graph_path: Path, node_id: str) -> None:
    """Put the representative a node of ``graph_path`` names in its step log: its image,
    and its record with a pixel hash and a describe output."""
    steps_dir = graph_steps_dir(graph_path)
    steps_dir.mkdir(exist_ok=True)
    observation_id = f"{node_id}-observation"
    observation_image_path(steps_dir, observation_id).write_bytes(encode_screenshot(_png(0)))
    record = {
        "id": observation_id,
        "kind": "observation",
        "pixel_hash": f"hash-{observation_id}",
        "describe": {"page_description": "", "state": {}, "elements": [], "is_external_web": False},
    }
    (steps_dir / f"{observation_id}.json").write_text(json.dumps(record), encoding="utf-8")


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
            {"type": "retry_edge", "source_node": "s1_detail", "target_node": "s0_home"},
            {"type": "explore_node", "node": "s0_home", "expected_pages": ["Settings"]},
            {"type": "retry_edge", "source_node": "s9_missing", "target_node": "s0_home"},
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
        [{"type": "explore_node", "node": "s0_home", "expected_pages": ["Settings"]}],
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
        [{"type": "explore_node", "node": "s1_detail", "expected_pages": []}],
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
    """The replay to the node is recorded as a navigation replay; the step from it is an
    audit transition from the screen it planned on to the screen it reached."""
    graph, graph_path = _two_node_graph(tmp_path)
    reached = iter(["s0_home", "s1_detail", "s0_home"])
    monkeypatch.setattr(graph, "identify_state", lambda _observation, **_kwargs: next(reached))
    monkeypatch.setattr(audit, "soft_keyboard_hint", lambda _udid: "")
    tap = {"action": "tap", "x": 1, "y": 2}
    monkeypatch.setattr(audit, "predict_next_action", lambda **_kwargs: (tap, ""))

    _re_explore(
        graph,
        [{"type": "explore_node", "node": "s1_detail", "expected_pages": []}],
        _LoggingController("com.demo", device_log),
        client,
        graph_path.parent,
        steps_per_issue=1,
    )

    records = step_records(graph_path.parent / "steps")
    assert [(record["kind"], record["source"]) for record in records] == [
        ("observation", "navigation_replay"),
        ("action", "navigation_replay"),
        ("observation", "navigation_replay"),
        ("observation", "re_explore"),
        ("action", "re_explore"),
        ("observation", "re_explore"),
        ("transition", "re_explore"),
    ]
    assert {record["run_kind"] for record in records} == {"audit"}
    assert records[1]["observation"] == records[0]["id"]
    assert {key: records[6][key] for key in _STEP_FIELDS} == {
        "before": "000003",
        "after": "000005",
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
            {"type": "retry_edge", "source_node": "s1_detail", "target_node": "s0_home"},
            {"type": "explore_node", "node": "s0_home", "expected_pages": []},
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
    graph.graph.add_node(EXTERNAL_WEB_NODE)
    identified = iter(["s0_home", "s1_detail", EXTERNAL_WEB_NODE, "s1_detail", "s0_home"])
    monkeypatch.setattr(graph, "identify_state", lambda *_args, **_kwargs: next(identified))
    monkeypatch.setattr(
        audit, "predict_next_action", lambda **_kwargs: ({"action": "tap", "x": 1, "y": 2}, "")
    )
    controller = FakeController("com.demo")

    (result,) = audit.re_explore_issues(
        graph,
        [{"type": "explore_node", "node": "s1_detail", "expected_pages": []}],
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
    graph.graph.add_node(EXTERNAL_WEB_NODE, page_description="[external web page]")
    on_the_page = [EXTERNAL_WEB_NODE] * (1 + back_presses_on_the_page)
    expected = _identifying(
        graph, monkeypatch, "s0_home", "s1_detail", *on_the_page, "s1_detail", "s0_home"
    )
    monkeypatch.setattr(audit, "soft_keyboard_hint", lambda _udid: "")
    tap = {"action": "tap", "x": 1, "y": 2}
    monkeypatch.setattr(audit, "predict_next_action", lambda **_kwargs: (tap, ""))

    (result,) = audit.re_explore_issues(
        graph,
        [{"type": "explore_node", "node": "s1_detail", "expected_pages": []}],
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
def test_re_explore_records_the_edge_from_a_node_identify_state_renamed(
    tmp_path: Path,
    two_screen_device: DeviceController,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    chat_response: Callable[..., httpx.Response],
    embedding_body: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """re_explore_issues acts from the node's new ID after identify_state renames it, so
    the step's edge is recorded."""
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
    recorder = StepRecorder(
        two_screen_device, graph.steps_dir, run_kind="audit", source="re_explore"
    )
    start = graph.identify_state(Observation.from_state(recorder.get_state()))
    verdict = {
        "same": False,
        "reason": "a query has been entered",
        "existing_description": "Chat search page with recent searches",
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
            [{"type": "explore_node", "node": start, "expected_pages": ["Search results"]}],
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

    renamed, results = "s0_chat_search_page_with_recent_searches", "s1_chat_search_results_page"
    assert [(step["from"], step["to"]) for step in result["steps"]] == [(renamed, results)]
    assert graph.graph.has_edge(renamed, results)
    assert "add_edge: missing node(s)" not in caplog.text


@pytest.mark.usefixtures("_fake_device")
def test_re_explore_issues_returns_early_for_an_empty_graph(
    client: OpenRouter, tmp_path: Path
) -> None:
    assert (
        audit.re_explore_issues(
            GraphManager(),
            [{"type": "retry_edge", "source_node": "s0_home"}],
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
            {"type": "retry_edge", "source_node": "s1_detail", "target_node": "s0_home"},
            {"type": "explore_node", "node": "s1_detail", "expected_pages": ["Settings"]},
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
    graph.graph.add_node("s2_camera_permission", page_description="Camera permission dialog")
    reached = iter(["s0_home", "s1_detail", "s2_camera_permission"])
    monkeypatch.setattr(graph, "identify_state", lambda *_args, **_kwargs: next(reached))
    monkeypatch.setattr(audit, "soft_keyboard_hint", lambda _udid: "")
    tap = {"action": "tap", "x": 1, "y": 2}
    monkeypatch.setattr(audit, "predict_next_action", lambda **_kwargs: (tap, ""))

    (result,) = audit.re_explore_issues(
        graph,
        [{"type": "explore_node", "node": "s1_detail", "expected_pages": []}],
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
    graph = _graph_with_nodes(
        ("s0_home", {"page_description": "Home"}),
        ("s1_detail", {"page_description": "Detail"}),
        ("s2_settings", {"page_description": "Settings"}),
    )
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
    def get_state(self) -> dict[str, Any]:
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
    it and is removed once empty."""
    graph = _flagged_graph()
    expected = _identifying(graph, monkeypatch, "s0_home", "s2_settings", "s0_home", "s0_home")
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
    assert edge["target_observations"] == ["Settings", "Settings"]
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
    transition carries the edge's instruction and the node the action reached; the replays
    around it are navigation replays."""
    graph = _flagged_graph()
    reached = iter(["s0_home", "s2_settings", "s0_home"])
    monkeypatch.setattr(graph, "identify_state", lambda _observation, **_kwargs: next(reached))

    _re_explore(graph, [_retry()], _ReplayController(device_log), client, tmp_path)

    records = step_records(tmp_path / "steps")
    assert [(record["kind"], record["source"]) for record in records] == [
        ("observation", "navigation_replay"),
        ("observation", "edge_retry"),
        ("action", "edge_retry"),
        ("observation", "edge_retry"),
        ("transition", "edge_retry"),
        ("observation", "navigation_replay"),
    ]
    assert {key: records[4][key] for key in _STEP_FIELDS} == {
        "before": "000001",
        "after": "000003",
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
        _retry(instruction=""),
    ],
    ids=["paraphrased", "on another edge", "no such edge", "empty"],
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
    graph.graph.add_node("ext_com.browser", page_description="[external app: com.browser]")
    graph.add_edge(
        "s0_home", "ext_com.browser", [_GEAR_TAP], details=EdgeDetails(instruction="open help")
    )
    monkeypatch.setattr(graph, "identify_state", lambda *_args, **_kwargs: "s0_home")
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
    graph.graph.add_node("s3_camera_permission", page_description="Camera permission dialog")
    reached = iter(["s0_home", "s3_camera_permission", "s0_home"])
    identified: list[tuple[str, str, str]] = []

    def identify(observation: Observation, **_kwargs: object) -> str:
        identified.append((observation.activity, observation.screenshot_b64, observation.package))
        return next(reached)

    monkeypatch.setattr(graph, "identify_state", identify)
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
    graph.graph.add_node(EXTERNAL_WEB_NODE, page_description="[external web page]")
    reached = iter(["s0_home", EXTERNAL_WEB_NODE, "s0_home"])
    monkeypatch.setattr(graph, "identify_state", lambda *_args, **_kwargs: next(reached))

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
    graph.graph.add_node(EXTERNAL_WEB_NODE, page_description="[external web page]")
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
    """The audit merges nodes before re-exploring, and identify_state renames them during it."""
    graph = _flagged_graph()
    graph.client, graph.embedding = client, ModelSettings("provider/embedding")
    monkeypatch.setattr(graph_manager, "get_embedding", lambda *_args, **_kwargs: [1.0])
    graph.graph.add_node("s3_home_copy", page_description="Home")
    graph.graph.add_node("s4_detail_copy", page_description="Detail")
    graph.merge_nodes("s0_home", "s3_home_copy")
    graph.merge_nodes("s1_detail", "s4_detail_copy")
    reached = iter(["s0_start", "s1_detail", "s0_start", "s0_start", "s2_app_settings"])

    def identify(*_args: object, **_kwargs: object) -> str:
        if "s0_home" in graph.graph:
            graph.rename_node("s0_home", "Start")
            graph.rename_node("s2_settings", "App settings")
        return next(reached)

    monkeypatch.setattr(graph, "identify_state", identify)
    monkeypatch.setattr(audit, "describe_executed_action", lambda **_kwargs: "Tap the card")

    results = _re_explore(
        graph,
        [
            _retry(source="s3_home_copy", target="s4_detail_copy"),
            {"type": "explore_node", "node": "s2_settings", "expected_pages": []},
        ],
        _ReplayController(device_log),
        client,
        tmp_path,
    )

    assert [r["status"] for r in results] == ["relabelled", "explored"]
    assert graph.graph["s0_start"]["s1_detail"]["instructions"] == ["Tap the card"]
    assert results[1]["navigated_to"] == "s2_app_settings"
    assert device_log == [*_FIRST_RESTART, _DETAIL_TAP, *_RESTART, *_RESTART, _GEAR_TAP]


@pytest.mark.usefixtures("_fake_device")
def test_re_explore_sends_nothing_for_an_edge_without_actions_and_replays_the_rest(
    client: OpenRouter,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    device_log: list[object],
    write_graph_file: Callable[[Path, dict[str, Any]], None],
) -> None:
    """A loaded graph file can hold an edge with no actions; the replay passes over it."""
    graph_path = tmp_path / "hand_edited.json"
    write_graph_file(
        graph_path,
        {
            "nodes": [{"id": node} for node in ("s0_splash", "s1_home", "s2_detail")],
            "edges": [
                {"source": "s0_splash", "target": "s1_home", "actions": []},
                {"source": "s1_home", "target": "s2_detail", "actions": [[_DETAIL_TAP]]},
            ],
        },
    )
    graph = GraphManager()
    graph.load_graph(graph_path)
    _identifying(graph, monkeypatch, "s0_splash", "s2_detail")

    (result,) = _re_explore(
        graph,
        [{"type": "explore_node", "node": "s2_detail", "expected_pages": []}],
        _LoggingController("com.demo", device_log),
        client,
        tmp_path,
    )

    assert (result["status"], result["navigated_to"]) == ("explored", "s2_detail")
    assert device_log == [*_FIRST_RESTART, _DETAIL_TAP]


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
        [{"type": "explore_node", "node": "s2_alarm", "expected_pages": []}],
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
def test_re_explore_replays_an_edge_without_actions_by_executing_nothing_for_it(
    client: OpenRouter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, device_log: list[object]
) -> None:
    graph = _graph_with_nodes(
        ("s0_home", {"page_description": "Home"}), ("s1_detail", {"page_description": "Detail"})
    )
    graph.graph.add_edge("s0_home", "s1_detail")
    _identifying(graph, monkeypatch, "s0_home", "s1_detail")
    issue = {"type": "explore_node", "node": "s1_detail", "expected_pages": []}

    (result,) = _re_explore(
        graph, [issue], _LoggingController("com.demo", device_log), client, tmp_path
    )

    assert result == {
        "issue": issue,
        "status": "explored",
        "navigated_to": "s1_detail",
        "steps": [],
    }
    assert device_log == _FIRST_RESTART


@pytest.mark.usefixtures("_fake_device")
def test_re_explore_replays_each_action_after_its_setup(
    client: OpenRouter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, device_log: list[object]
) -> None:
    graph = _graph_with_nodes(
        ("s0_home", {"page_description": "Home"}), ("s1_detail", {"page_description": "Detail"})
    )
    graph.add_edge("s0_home", "s1_detail", [_DETAIL_TAP], details=EdgeDetails(setup=[_SWIPE]))
    _identifying(graph, monkeypatch, "s0_home", "s1_detail")
    issue = {"type": "explore_node", "node": "s1_detail", "expected_pages": []}

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

    (result,) = _re_explore(graph, [_retry()], _ReplayController(device_log), client, tmp_path)

    assert (result["status"], result["moved_to"]) == ("moved", "s2_settings")
    assert graph.graph["s0_home"]["s2_settings"]["setups"] == [None, [_SWIPE]]
    assert device_log == [*_FIRST_RESTART, _SWIPE, _DETAIL_TAP, *_RESTART]


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
    _identifying(graph, monkeypatch, *after_restart, reached, "s2_settings")
    monkeypatch.setattr(audit, "soft_keyboard_hint", lambda _udid: "")
    sent = iter([_SWIPE, row_tap])
    monkeypatch.setattr(audit, "predict_next_action", lambda **_kwargs: (next(sent), ""))
    issue = {"type": "explore_node", "node": "s1_detail", "expected_pages": []}

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
def test_re_explore_to_the_start_node_renamed_on_the_screen_the_app_opened_on_reaches_it(
    client: OpenRouter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, device_log: list[object]
) -> None:
    """Expecting the start node on the screen the app opened on, identification can reject
    and rename it. A replay that then reaches it under its new ID has reached the issue's
    node, so the setup of the next step there is known."""
    graph = _flagged_graph()
    graph.add_edge("s1_detail", "s0_home", [{"action": "back"}])
    graph.client, graph.embedding = client, ModelSettings("provider/embedding")
    monkeypatch.setattr(graph_manager, "get_embedding", lambda *_args, **_kwargs: [1.0])
    answers = iter(["s1_detail", "s0_start", "s0_start"])

    def identify(_observation: Observation, *, expected_node: str | None = None) -> str:
        if expected_node == "s0_home":
            graph.rename_node("s0_home", "Start")
        return next(answers)

    monkeypatch.setattr(graph, "identify_state", identify)
    monkeypatch.setattr(audit, "soft_keyboard_hint", lambda _udid: "")
    monkeypatch.setattr(audit, "predict_next_action", lambda **_kwargs: (_SWIPE, ""))
    issue = {"type": "explore_node", "node": "s0_home", "expected_pages": []}

    (result,) = _re_explore(
        graph,
        [issue],
        _LoggingController("com.demo", device_log),
        client,
        tmp_path,
        steps_per_issue=1,
    )

    assert result["navigated_to"] == "s0_start"
    assert graph.graph["s0_start"]["s0_start"]["setups"] == [[]]


@pytest.mark.usefixtures("_fake_device")
def test_re_explore_takes_no_step_from_a_screen_the_graph_has_no_node_for(
    client: OpenRouter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, device_log: list[object]
) -> None:
    graph, _graph_path = _two_node_graph(tmp_path)
    monkeypatch.setattr(graph, "identify_state", lambda *_args, **_kwargs: "s9_unknown")
    issue = {"type": "explore_node", "node": "s1_detail", "expected_pages": []}

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
    issue = {"type": "explore_node", "node": "s0_home", "expected_pages": []}

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

    def identify(observation: Observation, *, expected_node: str | None = None) -> str:
        expected.append(expected_node)
        return nodes[observation.activity]

    monkeypatch.setattr(graph, "identify_state", identify)
    monkeypatch.setattr(audit, "soft_keyboard_hint", lambda _udid: "")
    tap = {"action": "tap", "x": 1, "y": 2}
    monkeypatch.setattr(audit, "predict_next_action", lambda **_kwargs: (tap, ""))
    issue = {"type": "explore_node", "node": "s0_home", "expected_pages": []}
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
    graph.graph.add_node(EXTERNAL_WEB_NODE, page_description="[external web page]")
    nodes = {
        "com.demo.Home": "s0_home",
        "com.demo.Detail": "s1_detail",
        "com.demo.Article": EXTERNAL_WEB_NODE,
    }
    monkeypatch.setattr(
        graph, "identify_state", lambda observation, **_kwargs: nodes[observation.activity]
    )
    monkeypatch.setattr(audit, "soft_keyboard_hint", lambda _udid: "")
    tap = {"action": "tap", "x": 1, "y": 2}
    monkeypatch.setattr(audit, "predict_next_action", lambda **_kwargs: (tap, ""))
    issue = {"type": "explore_node", "node": "s0_home", "expected_pages": []}
    controller = screens_by_log(*[_HOME_SCREEN] * 4, after_the_tap, _HOME_SCREEN, _DETAIL_SCREEN)

    _re_explore(graph, [issue], controller, client, tmp_path, steps_per_issue=2)

    edge = graph.graph["s0_home"]["s1_detail"]
    assert (edge["actions"][-1], edge["setups"][-1]) == ([tap], None)


@pytest.mark.usefixtures("_fake_device")
def test_re_explore_skips_a_missing_node_and_ignores_an_issue_of_another_type(
    client: OpenRouter, tmp_path: Path, device_log: list[object]
) -> None:
    graph, _graph_path = _two_node_graph(tmp_path)
    missing = {"type": "explore_node", "node": "s9_missing", "expected_pages": ["Settings"]}

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
    add_representative: Callable[..., None],
    write_image_vectors: Callable[..., None],
) -> None:
    graph_dir = tmp_path / "graphs"
    app_dir = graph_dir / "demo"
    _save_with_representatives(
        _flagged_graph(), app_dir / "demo.json", add_representative, write_image_vectors
    )
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
    monkeypatch.setattr(
        GraphManager, "identify_state", lambda _self, *_args, **_kwargs: next(reached)
    )
    monkeypatch.setattr(GraphManager, "normalize_all_edges", lambda _self: 0)
    monkeypatch.setattr(audit, "describe_executed_action", lambda **_kwargs: "Tap the gear icon")

    assert audit.main(["--config", str(config), "--re-explore", "--steps-per-issue", "0"]) == 0

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
    (app_dir / "demo.json").write_text('{"nodes":[],"edges":[]}')
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
    add_representative: Callable[..., None],
    write_image_vectors: Callable[..., None],
) -> None:
    """Each merge result names the sources that proposed the pair and the member of each
    node the verifier was shown."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    app_dir = tmp_path / "graphs" / "demo"
    _save_with_representatives(
        _graph_with_nodes(("s0_home", {}), ("s1_home", {})),
        app_dir / "demo.json",
        add_representative,
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

    outcome = (
        {"status": "merged", "kept": "s0_home", "removed": "s1_home"}
        if same
        else {"status": "kept_separate"}
    )
    assert json.loads((app_dir / "demo_merge.json").read_text(encoding="utf-8")) == [
        {
            "node_a": "s0_home",
            "node_b": "s1_home",
            "sources": ["text_audit"],
            "findings": [{"source": "text_audit", "issue": issue}],
            "members": {"s0_home": "s0_home-observation", "s1_home": "s1_home-observation"},
            "image_score": 0.0,
            "reason": "r",
            "reused": False,
            **outcome,
        }
    ]
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


def _save_with_representatives(
    graph: GraphManager,
    path: Path,
    add_representative: Callable[..., None],
    write_image_vectors: Callable[..., None],
) -> None:
    """Save *graph* to *path* with a representative for every node, its image and image
    vector in the step log beside it and its text vector in the file beside the graph, so
    the audit can load it without embedding any."""
    graph.steps_dir = graph_steps_dir(path)
    for node_id in list(graph.graph):
        add_representative(graph, node_id)
    graph.save_graph(path)
    representatives = [f"{node_id}-observation" for node_id in graph.graph]
    write_image_vectors(graph.steps_dir, representatives)
    save_embeddings(
        text_embeddings_path(path),
        {representative: [1.0, 0.0] for representative in representatives},
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
    add_representative: Callable[..., None],
    write_image_vectors: Callable[..., None],
) -> None:
    _save_with_representatives(
        _graph_with_a_dead_end("s2_dialog"),
        tmp_path / "graphs" / "demo" / "demo.json",
        add_representative,
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
    add_representative: Callable[..., None],
    write_image_vectors: Callable[..., None],
) -> None:
    """A dead end merged into a node with a path back is gone from the audited graph."""
    app_dir = tmp_path / "graphs" / "demo"
    explored = _graph_with_a_dead_end("s2_list")
    assert explored.nodes_without_path_to_start().in_app == ["s2_list"]
    _save_with_representatives(
        explored, app_dir / "demo.json", add_representative, write_image_vectors
    )
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
    (graph_dir / "demo" / "demo.json").write_text('{"nodes":[],"edges":[]}', encoding="utf-8")
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

    def re_explore(*_args: object, **_kwargs: object) -> list[dict[str, Any]]:
        events.append(("re-explore",))
        return []

    monkeypatch.setattr(
        audit, "run_audit", lambda *_args: {"issues": [{"type": "explore_node", "node": "s0"}]}
    )
    monkeypatch.setattr(
        audit, "grant_runtime_permissions", lambda *args: events.append(("grant", *args))
    )
    monkeypatch.setattr(audit, "launch_app", lambda *_args: events.append(("launch",)))
    monkeypatch.setattr(audit, "re_explore_issues", re_explore)
    monkeypatch.setattr(audit, "normalize_and_save_audited_graph", lambda *_args: 0)

    assert audit.main(["--config", str(config), "--re-explore"]) == 0
    assert events == [("grant", "emulator-5554", "com.demo"), ("launch",), ("re-explore",)]
