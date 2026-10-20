"""app-graph-audit parsing and pure helpers, exercised without AITK, adb or an emulator."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, NoReturn, override

import httpx
import pytest
from openrouter import OpenRouter

from android_app_graph.commands import audit
from android_app_graph.utils import ModelSettings, open_client
from android_app_graph.utils.graph_manager import GraphManager

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
    """Regression for #48: an auditor naming one node twice used to delete it."""
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
    """A FakeController that also appends each action to ``log``."""

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
    monkeypatch.setattr(audit, "plan_next_action", lambda **_kwargs: "tap the first button")
    monkeypatch.setattr(audit, "predict_next_action", lambda **_kwargs: ({"action": "end"}, ""))


@pytest.mark.usefixtures("_fake_device")
def test_re_explore_issues_retries_explores_and_skips(
    client: OpenRouter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph, graph_path = _two_node_graph(tmp_path)
    monkeypatch.setattr(graph, "identify_state", lambda *_args, **_kwargs: "s1_detail")
    controller = FakeController("com.demo")
    models: list[tuple[str, str]] = []

    def plan(*, settings: ModelSettings, **_kwargs: object) -> str:
        models.append(("plan", settings.model))
        return "tap the first button"

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

    assert [r["status"] for r in results] == ["retried", "explored", "skipped"]
    assert results[2]["reason"] == "source not found"
    assert models == [("plan", "instruction-model"), ("act", "action-model")] * 2
    # "end" is rewritten to a back press, and every action is recorded with save_flag.
    assert {action["action"] for action, _save_flag in controller.actions} <= {"tap", "back"}
    assert all(save_flag for _action, save_flag in controller.actions)
    assert out_path.exists()
    assert graph_path.exists()


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

    assert [r["status"] for r in results] == ["skipped", "skipped"]
    assert {r["reason"] for r in results} == {"unreachable"}


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


def _re_explore_detail_twice(
    graph: GraphManager,
    controller: FakeController,
    client: OpenRouter,
    out_path: Path,
    **options: str,
) -> list[dict[str, Any]]:
    """Run one retry_edge and one explore_node issue, both replaying to s1_detail.

    ``options`` reach re_explore_issues only when given, so the default launch is
    tested without naming ``launch_activity``.
    """
    return audit.re_explore_issues(
        graph,
        [
            {"type": "retry_edge", "source_node": "s1_detail", "target_node": "s0_home"},
            {"type": "explore_node", "node": "s1_detail", "expected_pages": ["Settings"]},
        ],
        controller,
        "demo",
        "com.demo",
        out_path,
        client,
        ModelSettings("instruction-model"),
        ModelSettings("action-model"),
        steps_per_issue=1,
        **options,
    )


@pytest.mark.usefixtures("_fake_device")
def test_re_explore_issues_restarts_the_app_before_each_replay(
    client: OpenRouter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, device_log: list[object]
) -> None:
    """Regression for #79: the launcher intent resumed the screen the last issue ended on."""
    graph, _graph_path = _two_node_graph(tmp_path)
    monkeypatch.setattr(graph, "identify_state", lambda *_args, **_kwargs: "s1_detail")
    monkeypatch.setattr(audit, "soft_keyboard_hint", lambda: "")
    controller = _LoggingController("com.demo", device_log)

    results = _re_explore_detail_twice(graph, controller, client, tmp_path / "demo_audited.json")

    assert [r["status"] for r in results] == ["retried", "explored"]
    replay = {"action": "tap", "coordinate": [10, 20]}
    explore_step = {"action": "back"}
    assert device_log == [_FORCE_STOP, _PACKAGE_LAUNCH, replay, explore_step] * 2


@pytest.mark.usefixtures("_fake_device")
def test_re_explore_issues_restarts_the_app_through_its_launch_activity(
    client: OpenRouter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, device_log: list[object]
) -> None:
    graph, _graph_path = _two_node_graph(tmp_path)
    monkeypatch.setattr(graph, "identify_state", lambda *_args, **_kwargs: "s1_detail")
    monkeypatch.setattr(audit, "soft_keyboard_hint", lambda: "")
    controller = FakeController("com.demo")

    _re_explore_detail_twice(
        graph,
        controller,
        client,
        tmp_path / "demo_audited.json",
        launch_activity="com.demo/.Main",
    )

    assert device_log == [_FORCE_STOP, _ACTIVITY_LAUNCH] * 2


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
    adb_module = ModuleType("aitk.utils.adb_controller")
    monkeypatch.setattr(
        adb_module,
        "ADBController",
        lambda _config, _logger: FakeController("com.demo"),
        raising=False,
    )
    monkeypatch.setitem(sys.modules, adb_module.__name__, adb_module)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    config = tmp_path / "audit.yaml"
    config.write_text("apps: []\n", encoding="utf-8")
    assert audit.main(["--config", str(config), "--re-explore"]) == 0
    assert len(opened_clients) == 1
    assert opened_clients[0].sdk_configuration.client is None
