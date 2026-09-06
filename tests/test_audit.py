"""app-graph-audit parsing and pure helpers, exercised without AITK, adb or an emulator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from openai import OpenAI

from android_app_graph.commands import audit
from android_app_graph.utils.graph_manager import GraphManager


@pytest.fixture
def client() -> OpenAI:
    """An OpenAI client that is never called: every VLM helper is monkeypatched."""
    return OpenAI(api_key="not-a-real-key")


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
    client: OpenAI, monkeypatch: pytest.MonkeyPatch
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

    (result,) = audit.verify_and_merge_nodes(graph, [_merge_issue()], client, "gpt-test")

    assert result["status"] == "merged"
    assert result["kept"] == "s0_home"
    assert result["removed"] == "s1_home"
    assert result["reason"] == "identical screens"
    # The description of the more-visited node is used as the reference.
    assert seen["existing_description"] == "Home"
    assert seen["model"] == "gpt-test"
    assert "s1_home" not in graph.graph


def test_verify_and_merge_nodes_keeps_separate(
    client: OpenAI, monkeypatch: pytest.MonkeyPatch
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

    (result,) = audit.verify_and_merge_nodes(graph, [_merge_issue()], client, "gpt-test")

    assert result["status"] == "kept_separate"
    assert result["reason"] == "different content"
    assert set(graph.graph) == {"s0_home", "s1_home"}


def test_verify_and_merge_nodes_skips_missing_screenshot(
    client: OpenAI, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = _graph_with_nodes(
        ("s0_home", {"reference_screenshot": "aaa"}),
        ("s1_home", {}),
    )
    monkeypatch.setattr(audit, "verify_same_node", _unreachable_verifier)

    (result,) = audit.verify_and_merge_nodes(graph, [_merge_issue()], client, "gpt-test")

    assert result["status"] == "skipped"
    assert result["reason"] == "missing screenshot"


def test_verify_and_merge_nodes_skips_a_self_pair(
    client: OpenAI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression for #48: an auditor naming one node twice used to delete it."""
    graph = _graph_with_nodes(
        ("s0_home", {"reference_screenshot": "aaa", "page_description": "Home", "visit_count": 3}),
    )
    monkeypatch.setattr(audit, "verify_same_node", _unreachable_verifier)

    issue = _merge_issue(node_a="s0_home", node_b="s0_home")
    (result,) = audit.verify_and_merge_nodes(graph, [issue], client, "gpt-test")

    assert result["status"] == "skipped"
    assert result["reason"] == "same node"
    assert set(graph.graph) == {"s0_home"}
    assert graph.graph.nodes["s0_home"]["visit_count"] == 3


def test_verify_and_merge_nodes_skips_removed_node(
    client: OpenAI, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = _graph_with_nodes(("s0_home", {"reference_screenshot": "aaa"}))
    monkeypatch.setattr(audit, "verify_same_node", _unreachable_verifier)

    (result,) = audit.verify_and_merge_nodes(graph, [_merge_issue()], client, "gpt-test")

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


class _FakeCompletedProcess:
    """The subset of subprocess.CompletedProcess the auditor reads."""

    returncode = 0
    stdout = ""


def _fake_run(*_args: object, **_kwargs: object) -> _FakeCompletedProcess:
    return _FakeCompletedProcess()


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
def _fake_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """Silence every side effect re_explore_issues has outside the graph."""
    monkeypatch.setattr(audit.subprocess, "run", _fake_run)
    monkeypatch.setattr(audit.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(audit, "plan_next_action", lambda **_kwargs: "tap the first button")
    monkeypatch.setattr(audit, "predict_next_action", lambda **_kwargs: ({"action": "end"}, ""))


@pytest.mark.usefixtures("_fake_device")
def test_re_explore_issues_retries_explores_and_skips(
    client: OpenAI, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph, graph_path = _two_node_graph(tmp_path)
    monkeypatch.setattr(graph, "identify_state", lambda *_args, **_kwargs: "s1_detail")
    controller = FakeController("com.demo")

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
        "instruction-model",
        client,
        "action-model",
        steps_per_issue=1,
    )

    assert [r["status"] for r in results] == ["retried", "explored", "skipped"]
    assert results[2]["reason"] == "source not found"
    # "end" is rewritten to a back press, and every action is recorded with save_flag.
    assert {action["action"] for action, _save_flag in controller.actions} <= {"tap", "back"}
    assert all(save_flag for _action, save_flag in controller.actions)
    assert out_path.exists()
    assert graph_path.exists()


@pytest.mark.usefixtures("_fake_device")
def test_re_explore_issues_skips_unreachable_nodes(
    client: OpenAI, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
        "instruction-model",
        client,
        "action-model",
        steps_per_issue=1,
    )

    assert [r["status"] for r in results] == ["skipped", "skipped"]
    assert {r["reason"] for r in results} == {"unreachable"}


@pytest.mark.usefixtures("_fake_device")
def test_re_explore_issues_returns_early_for_an_empty_graph(client: OpenAI, tmp_path: Path) -> None:
    assert (
        audit.re_explore_issues(
            GraphManager(),
            [{"type": "retry_edge", "source_node": "s0_home"}],
            FakeController("com.demo"),
            "demo",
            "com.demo",
            tmp_path / "demo_audited.json",
            client,
            "instruction-model",
            client,
            "action-model",
        )
        == []
    )
