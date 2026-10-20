"""app-graph-audit parsing and pure helpers, exercised without AITK, adb or an emulator."""

from __future__ import annotations

import copy
import json
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any, NoReturn, override

import httpx
import pytest
from openrouter import OpenRouter

from android_app_graph.commands import audit
from android_app_graph.device import DeviceController
from android_app_graph.utils import ModelSettings, graph_manager, open_client
from android_app_graph.utils.graph_manager import EXTERNAL_WEB_NODE, GraphManager

PAGE_DETAIL = ModelSettings("gpt-test")
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


@pytest.fixture
def client() -> OpenRouter:
    """An OpenRouter client that is never called: every VLM helper is monkeypatched."""
    return open_client("not-a-real-key")


def _unreachable_verifier(**_kwargs: Any) -> dict[str, Any]:
    msg = "verify_same_node must not be called for a skipped merge candidate"
    raise AssertionError(msg)


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
    assert audit.select_apps_for_audit({}, tmp_path, app_name="demo") == [
        {"name": "demo", "package_name": ""}
    ]


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
    normalized: list[str] = []

    def fake_normalize(node_id: str) -> int:
        normalized.append(node_id)
        return 2

    saved: list[Path] = []
    monkeypatch.setattr(graph, "normalize_node_edges", fake_normalize)
    monkeypatch.setattr(graph, "save_graph", saved.append)

    audited_path = tmp_path / "demo_audited.json"
    assert audit.normalize_and_save_audited_graph(graph, audited_path) == 4
    assert sorted(normalized) == ["s0_home", "s1_detail"]
    assert saved == [audited_path]


def _merge_issue(node_a: str = "s0_home", node_b: str = "s1_home") -> dict[str, Any]:
    return {"type": "merge_nodes", "node_a": node_a, "node_b": node_b}


def test_verify_and_merge_nodes_merges_confirmed_pair(
    client: OpenRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = _graph_with_nodes(
        ("s0_home", {"reference_screenshot": "aaa", "page_description": "Home", "visit_count": 3}),
        ("s1_home", {"reference_screenshot": "bbb", "page_description": "Start", "visit_count": 1}),
    )
    seen: dict[str, Any] = {}

    def fake_verify(**kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs)
        return {"same": True, "reason": "identical screens"}

    monkeypatch.setattr(audit, "verify_same_node", fake_verify)

    (result,) = audit.verify_and_merge_nodes(graph, [_merge_issue()], client, PAGE_DETAIL)

    assert result["status"] == "merged"
    assert result["kept"] == "s0_home"
    assert result["removed"] == "s1_home"
    assert result["reason"] == "identical screens"
    # The description of the more-visited node is used as the reference.
    assert seen["existing_description"] == "Home"
    assert seen["settings"] == PAGE_DETAIL
    assert "s1_home" not in graph.graph


def test_verify_and_merge_nodes_keeps_separate(
    client: OpenRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = _graph_with_nodes(
        ("s0_home", {"reference_screenshot": "aaa", "page_description": "Home"}),
        ("s1_home", {"reference_screenshot": "bbb", "page_description": "Settings"}),
    )
    monkeypatch.setattr(
        audit,
        "verify_same_node",
        lambda **_kwargs: {"same": False, "reason": "different content"},
    )

    (result,) = audit.verify_and_merge_nodes(graph, [_merge_issue()], client, PAGE_DETAIL)

    assert result["status"] == "kept_separate"
    assert result["reason"] == "different content"
    assert set(graph.graph) == {"s0_home", "s1_home"}


def test_verify_and_merge_nodes_skips_missing_screenshot(
    client: OpenRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = _graph_with_nodes(
        ("s0_home", {"reference_screenshot": "aaa"}),
        ("s1_home", {}),
    )
    monkeypatch.setattr(audit, "verify_same_node", _unreachable_verifier)

    (result,) = audit.verify_and_merge_nodes(graph, [_merge_issue()], client, PAGE_DETAIL)

    assert result["status"] == "skipped"
    assert result["reason"] == "missing screenshot"


def test_verify_and_merge_nodes_skips_a_self_pair(
    client: OpenRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An auditor naming one node twice used to delete it."""
    graph = _graph_with_nodes(
        ("s0_home", {"reference_screenshot": "aaa", "page_description": "Home", "visit_count": 3}),
    )
    monkeypatch.setattr(audit, "verify_same_node", _unreachable_verifier)

    issue = _merge_issue(node_a="s0_home", node_b="s0_home")
    (result,) = audit.verify_and_merge_nodes(graph, [issue], client, PAGE_DETAIL)

    assert result["status"] == "skipped"
    assert result["reason"] == "same node"
    assert set(graph.graph) == {"s0_home"}
    assert graph.graph.nodes["s0_home"]["visit_count"] == 3


def test_verify_and_merge_nodes_skips_removed_node(
    client: OpenRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = _graph_with_nodes(("s0_home", {"reference_screenshot": "aaa"}))
    monkeypatch.setattr(audit, "verify_same_node", _unreachable_verifier)

    (result,) = audit.verify_and_merge_nodes(graph, [_merge_issue()], client, PAGE_DETAIL)

    assert result["status"] == "skipped"
    assert result["reason"] == "node removed"


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
            "screenshot": "c2NyZWVu",
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
                    {"id": "s0_home", "activity": "com.demo.Home", "page_description": "Home"},
                    {
                        "id": "s1_detail",
                        "activity": "com.demo.Detail",
                        "page_description": "Detail",
                    },
                ],
                "edges": [
                    {
                        "source": "s0_home",
                        "target": "s1_detail",
                        "actions": [{"action": "tap", "coordinate": [10, 20]}],
                        "instructions": ["open the detail page"],
                        "num_steps": [1],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    graph = GraphManager()
    graph.load_graph(graph_path)
    return graph, graph_path


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
        controller,
        "demo",
        "com.demo",
        out_path,
        client,
        ModelSettings("instruction-model"),
        ModelSettings("action-model"),
        steps_per_issue=1,
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
        controller,
        "demo",
        "com.demo",
        tmp_path / "demo" / "demo_audited.json",
        client,
        ModelSettings("instruction-model"),
        ModelSettings("action-model"),
        steps_per_issue=1,
    )

    assert controller.actions == [({"action": "back"}, True)]
    edge = graph.graph["s0_home"]["s0_home"]
    assert edge["actions"] == [[{"action": "back"}]]
    assert edge["instructions"] == ["Press back"]
    assert [step["instruction"] for step in result["steps"]] == ["Press back"]
    assert '"tap the first button"' in caplog.text


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
        controller,
        "demo",
        "com.demo",
        tmp_path / "demo" / "demo_audited.json",
        client,
        ModelSettings("instruction-model"),
        ModelSettings("action-model"),
        steps_per_issue=1,
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
    identified = iter(["s1_detail", EXTERNAL_WEB_NODE, "s1_detail", "s0_home"])
    monkeypatch.setattr(graph, "identify_state", lambda *_args: next(identified))
    monkeypatch.setattr(
        audit, "predict_next_action", lambda **_kwargs: ({"action": "tap", "x": 1, "y": 2}, "")
    )
    controller = FakeController("com.demo")

    (result,) = audit.re_explore_issues(
        graph,
        [{"type": "explore_node", "node": "s1_detail", "expected_pages": []}],
        controller,
        "demo",
        "com.demo",
        graph_path,
        client,
        ModelSettings("instruction-model"),
        ModelSettings("action-model"),
        steps_per_issue=2,
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
    [(1, [_BACK, _BACK]), (3, [_BACK, _BACK, _BACK, _FORCE_STOP, _PACKAGE_LAUNCH])],
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
    """A page still showing after one back press used to become the node the next step
    acted from, recording an edge out of the shared ext_web node."""
    graph, graph_path = _two_node_graph(tmp_path)
    # identify_state is faked below, so add the node it would create.
    graph.graph.add_node(EXTERNAL_WEB_NODE, page_description="[external web page]")
    on_the_page = [EXTERNAL_WEB_NODE] * (1 + back_presses_on_the_page)
    identified = iter(["s1_detail", *on_the_page, "s1_detail", "s0_home"])
    monkeypatch.setattr(graph, "identify_state", lambda *_args: next(identified))
    monkeypatch.setattr(audit, "soft_keyboard_hint", lambda: "")
    tap = {"action": "tap", "x": 1, "y": 2}
    monkeypatch.setattr(audit, "predict_next_action", lambda **_kwargs: (tap, ""))

    (result,) = audit.re_explore_issues(
        graph,
        [{"type": "explore_node", "node": "s1_detail", "expected_pages": []}],
        _LoggingController("com.demo", device_log),
        "demo",
        "com.demo",
        graph_path,
        client,
        ModelSettings("instruction-model"),
        ModelSettings("action-model"),
        steps_per_issue=2,
    )

    assert [(step["from"], step["to"]) for step in result["steps"]] == [
        ("s1_detail", EXTERNAL_WEB_NODE),
        ("s1_detail", "s0_home"),
    ]
    assert graph.graph.out_degree(EXTERNAL_WEB_NODE) == 0
    replay = {"action": "tap", "coordinate": [10, 20]}
    assert device_log == [_FORCE_STOP, _PACKAGE_LAUNCH, replay, tap, *leaving, tap]


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
    """re_explore_issues used to keep the old ID of the node it acted from, so the
    step's edge was dropped."""
    monkeypatch.setattr(
        audit, "predict_next_action", lambda **_kwargs: ({"action": "tap", "x": 1, "y": 2}, "")
    )
    graph = GraphManager(
        client=router_client,
        page_detail=ModelSettings("provider/page_detail"),
        embedding=ModelSettings("provider/embedding"),
    )
    router_responses.extend(
        [
            chat_response(content='{"page_description":"Chat search page"}'),
            httpx.Response(200, json=embedding_body),
        ]
    )
    screen = two_screen_device.get_state()
    start = graph.identify_state(screen["activity"], screen["screenshot"])
    verdict = {
        "same": False,
        "reason": "a query has been entered",
        "existing_description": "Chat search page with recent searches",
        "new_description": "Chat search results page",
    }
    router_responses.extend(
        [
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
            two_screen_device,
            "demo",
            "com.demo.app",
            tmp_path / "demo_audited.json",
            router_client,
            ModelSettings("provider/instruction"),
            ModelSettings("provider/action"),
            steps_per_issue=1,
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
            FakeController("com.demo"),
            "demo",
            "com.demo",
            tmp_path / "demo_audited.json",
            client,
            ModelSettings("instruction-model"),
            ModelSettings("action-model"),
        )
        == []
    )


@pytest.mark.parametrize(
    ("launch_activity", "launch"), [(None, _PACKAGE_LAUNCH), ("com.demo/.Main", _ACTIVITY_LAUNCH)]
)
@pytest.mark.usefixtures("_fake_device")
def test_re_explore_issues_restarts_the_app_before_each_replay(
    launch_activity: str | None,
    launch: list[str],
    client: OpenRouter,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    device_log: list[object],
) -> None:
    """The launcher intent used to resume the screen the previous issue ended on."""
    graph, _graph_path = _two_node_graph(tmp_path)
    monkeypatch.setattr(graph, "identify_state", lambda *_args, **_kwargs: "s1_detail")
    monkeypatch.setattr(audit, "soft_keyboard_hint", lambda: "")

    results = audit.re_explore_issues(
        graph,
        [
            {"type": "retry_edge", "source_node": "s1_detail", "target_node": "s0_home"},
            {"type": "explore_node", "node": "s1_detail", "expected_pages": ["Settings"]},
        ],
        _LoggingController("com.demo", device_log),
        "demo",
        "com.demo",
        tmp_path / "demo_audited.json",
        client,
        ModelSettings("instruction-model"),
        ModelSettings("action-model"),
        steps_per_issue=1,
        launch_activity=launch_activity,
    )

    assert [r["status"] for r in results] == ["instruction_not_found", "explored"]
    replay = {"action": "tap", "coordinate": [10, 20]}
    explore_step = {"action": "back"}
    assert device_log == [_FORCE_STOP, launch, replay, explore_step] * 2


@pytest.mark.usefixtures("_fake_device")
def test_re_explore_records_a_permission_dialog_instead_of_pressing_back(
    client: OpenRouter,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    device_log: list[object],
) -> None:
    graph, _graph_path = _two_node_graph(tmp_path)
    graph.graph.add_node("s2_camera_permission", page_description="Camera permission dialog")
    reached = iter(["s1_detail", "s2_camera_permission"])
    monkeypatch.setattr(graph, "identify_state", lambda *_args: next(reached))
    monkeypatch.setattr(audit, "soft_keyboard_hint", lambda: "")
    tap = {"action": "tap", "x": 1, "y": 2}
    monkeypatch.setattr(audit, "predict_next_action", lambda **_kwargs: (tap, ""))

    (result,) = audit.re_explore_issues(
        graph,
        [{"type": "explore_node", "node": "s1_detail", "expected_pages": []}],
        _LoggingController("com.google.android.permissioncontroller", device_log),
        "demo",
        "com.demo",
        tmp_path / "demo" / "demo_audited.json",
        client,
        ModelSettings("instruction-model"),
        ModelSettings("action-model"),
        steps_per_issue=1,
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
    assert device_log == [_FORCE_STOP, _PACKAGE_LAUNCH, replay, tap]


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
        "s0_home", "s1_detail", [_DETAIL_TAP], instruction=_FLAGGED, target_observation="Detail"
    )
    graph.add_edge(
        "s0_home",
        "s2_settings",
        [_GEAR_TAP],
        instruction="open settings",
        target_observation="Settings",
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
        return {**state, "screenshot": f"screen-{len(self.actions)}"}


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
        controller,
        "demo",
        "com.demo",
        tmp_path / "demo_audited.json",
        client,
        ModelSettings("instruction-model"),
        ModelSettings("action-model"),
        steps_per_issue=steps_per_issue,
    )


@pytest.mark.usefixtures("_fake_device")
def test_re_explore_moves_a_flagged_action_to_the_edge_toward_the_node_it_reaches(
    client: OpenRouter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, device_log: list[object]
) -> None:
    """re_explore_issues used to leave a flagged edge as it was and only explore around it."""
    graph = _flagged_graph()
    reached = iter(["s0_home", "s2_settings", "s0_home", "s0_home"])
    monkeypatch.setattr(graph, "identify_state", lambda *_args: next(reached))
    monkeypatch.setattr(audit, "soft_keyboard_hint", lambda: "")

    (result,) = _re_explore(
        graph, [_retry()], _ReplayController(device_log), client, tmp_path, steps_per_issue=1
    )

    assert (result["status"], result["moved_to"]) == ("moved", "s2_settings")
    assert not graph.graph.has_edge("s0_home", "s1_detail")
    edge = graph.graph["s0_home"]["s2_settings"]
    assert edge["actions"] == [[_GEAR_TAP], [_DETAIL_TAP]]
    assert edge["instructions"] == ["open settings", _FLAGGED]
    assert edge["target_observations"] == ["Settings", "Settings"]
    assert [(step["from"], step["to"]) for step in result["steps"]] == [("s0_home", "s0_home")]
    restart = [_FORCE_STOP, _PACKAGE_LAUNCH]
    assert device_log == [*restart, _DETAIL_TAP, *restart, _BACK]


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
    graph.add_edge("s0_home", "s1_detail", [_SWIPE], instruction="scroll to the detail card")
    reached = iter(["s0_home", "s1_detail", "s0_home"])
    monkeypatch.setattr(graph, "identify_state", lambda *_args: next(reached))
    described: list[dict[str, Any]] = []

    def describe(**kwargs: Any) -> str | None:
        described.append(kwargs)
        return label

    monkeypatch.setattr(audit, "describe_executed_action", describe)

    (result,) = _re_explore(graph, [_retry()], _ReplayController(device_log), client, tmp_path)

    assert result == {"issue": _retry(), **outcome, "steps": []}
    edge = graph.graph["s0_home"]["s1_detail"]
    assert edge["actions"] == [[_DETAIL_TAP], [_SWIPE]]
    assert edge["instructions"] == [label or _FLAGGED, "scroll to the detail card"]
    (call,) = described
    assert call["screenshot_b64"] == "screen-0"
    assert call["action"] == [_DETAIL_TAP]
    assert (call["screen_w"], call["screen_h"]) == (1080, 1920)
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
    monkeypatch.setattr(graph, "identify_state", lambda *_args: arrives_at)
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
    monkeypatch.setattr(graph, "identify_state", lambda *_args: "s0_home")
    monkeypatch.setattr(audit, "soft_keyboard_hint", lambda: "")
    before = _edges(graph)

    (result,) = _re_explore(
        graph, [issue], _ReplayController(device_log), client, tmp_path, steps_per_issue=1
    )

    assert result["status"] == "instruction_not_found"
    assert [(step["from"], step["to"]) for step in result["steps"]] == [("s0_home", "s0_home")]
    assert {key: _edges(graph)[key] for key in before} == before
    assert device_log == [_FORCE_STOP, _PACKAGE_LAUNCH, _BACK]


@pytest.mark.parametrize(
    ("issue", "package", "status", "help_instruction"),
    [
        (_retry("open help", target="ext_com.browser"), "com.browser", "relabelled", "Tap Help"),
        (_retry(), "com.unknown", "unverified", "open help"),
    ],
    ids=["to the target's app", "to an app the graph has no node for"],
)
@pytest.mark.usefixtures("_fake_device")
def test_re_explore_takes_a_flagged_action_that_leaves_the_app_to_that_app_s_node(
    issue: dict[str, Any],
    package: str,
    status: str,
    help_instruction: str,
    client: OpenRouter,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    device_log: list[object],
) -> None:
    graph = _flagged_graph()
    graph.graph.add_node("ext_com.browser", page_description="[external app: com.browser]")
    graph.add_edge("s0_home", "ext_com.browser", [_GEAR_TAP], instruction="open help")
    monkeypatch.setattr(graph, "identify_state", lambda *_args: "s0_home")
    monkeypatch.setattr(audit, "describe_executed_action", lambda **_kwargs: "Tap Help")

    (result,) = _re_explore(
        graph, [issue], _ReplayController(device_log, package), client, tmp_path
    )

    assert result["status"] == status
    assert graph.graph["s0_home"]["ext_com.browser"]["instructions"] == [help_instruction]
    assert graph.graph["s0_home"]["s1_detail"]["instructions"] == [_FLAGGED]
    assert "ext_com.unknown" not in graph.graph


@pytest.mark.usefixtures("_fake_device")
def test_re_explore_does_not_explore_when_the_replay_after_a_correction_misses_the_source(
    client: OpenRouter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, device_log: list[object]
) -> None:
    graph = _flagged_graph()
    reached = iter(["s0_home", "s2_settings", "s2_settings"])
    monkeypatch.setattr(graph, "identify_state", lambda *_args: next(reached))

    (result,) = _re_explore(
        graph, [_retry()], _ReplayController(device_log), client, tmp_path, steps_per_issue=1
    )

    assert (result["status"], result["steps"]) == ("moved", [])
    restart = [_FORCE_STOP, _PACKAGE_LAUNCH]
    assert device_log == [*restart, _DETAIL_TAP, *restart]


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
    reached = iter(["s0_start", "s1_detail", "s0_start", "s2_app_settings"])

    def identify(*_args: object) -> str:
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
    restart = [_FORCE_STOP, _PACKAGE_LAUNCH]
    assert device_log == [*restart, _DETAIL_TAP, *restart, *restart, _GEAR_TAP]


def _install_fake_adb(monkeypatch: pytest.MonkeyPatch, controller: DeviceController) -> None:
    adb_module = ModuleType("aitk.utils.adb_controller")
    monkeypatch.setattr(
        adb_module, "ADBController", lambda _config, _logger: controller, raising=False
    )
    monkeypatch.setitem(sys.modules, adb_module.__name__, adb_module)


@pytest.mark.usefixtures("model_environment", "opened_clients", "no_sleep", "device_log")
def test_re_explore_saves_its_corrections_and_the_outcome_of_each_issue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph_dir = tmp_path / "graphs"
    app_dir = graph_dir / "demo"
    _flagged_graph().save_graph(app_dir / "demo.json")
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
    monkeypatch.setattr(GraphManager, "identify_state", lambda _self, *_args: next(reached))
    monkeypatch.setattr(GraphManager, "normalize_node_edges", lambda _self, _node_id: 0)
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
            raise RuntimeError("audit failed")
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
def test_audit_of_one_app_keeps_the_package_name_it_loaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--app`` audits without the config's apps, so the package must come from the file."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    graph_dir = tmp_path / "graphs"
    app_dir = graph_dir / "demo"
    app_dir.mkdir(parents=True)
    (app_dir / "demo.json").write_text(
        json.dumps({"package_name": "com.demo.app", "nodes": [], "edges": []}), encoding="utf-8"
    )
    config = _write_config(tmp_path, str(graph_dir))
    monkeypatch.setattr(audit, "run_audit", lambda *_args: {"issues": []})

    assert audit.main(["-c", str(config), "--app", "demo"]) == 0

    audited = json.loads((app_dir / "demo_audited.json").read_text(encoding="utf-8"))
    assert audited["package_name"] == "com.demo.app"


def _deny_device_work(monkeypatch: pytest.MonkeyPatch) -> None:
    def touched(*_args: object, **_kwargs: object) -> NoReturn:
        raise AssertionError("settings must be validated before any device work")

    adb_module = ModuleType("aitk.utils.adb_controller")
    monkeypatch.setattr(adb_module, "ADBController", touched, raising=False)
    monkeypatch.setitem(sys.modules, adb_module.__name__, adb_module)
    monkeypatch.setattr(audit, "launch_app", touched)


@pytest.mark.parametrize(
    ("role_settings", "key", "message"),
    [
        ({"instruction": {"request_timeout": 0}}, "test", "instruction.request_timeout must be"),
        ({"page_detail": {"audit_request_timeout": -5}}, "test", "audit_request_timeout must be"),
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
