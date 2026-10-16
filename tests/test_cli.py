"""app-graph argument parsing, model checks and run order, without AITK, adb or an emulator."""

from __future__ import annotations

import builtins
import json
import logging
import sys
from pathlib import Path
from types import ModuleType

import pytest

from android_app_graph import cli
from android_app_graph.cli import build_parser, main


def test_help_exits_zero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """argparse handles --help inside parse_args() and exits 0 before aitk is imported."""
    monkeypatch.setattr(sys, "argv", ["app-graph", "--help"])
    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 0
    assert "--config" in capsys.readouterr().out


def test_parser_defaults() -> None:
    args = build_parser().parse_args([])
    assert args.config == Path("configs/explore.yaml")
    assert args.max_steps is None
    assert args.resume_from is None


def test_parser_overrides() -> None:
    args = build_parser().parse_args(["-c", "custom.yaml", "--max-steps", "5", "-r", "auto"])
    assert args.config == Path("custom.yaml")
    assert args.max_steps == 5
    assert args.resume_from == "auto"


@pytest.mark.usefixtures("no_sleep")
def test_main_passes_resolved_models_to_exploration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    roles = ("page_detail", "embedding", "instruction", "action")
    expected_models = {role: {"model": f"provider/{role}"} for role in roles}
    config = tmp_path / "explore.yaml"
    config.write_text(
        json.dumps(
            {
                "vlm": {role: {"model": f"${{APP_GRAPH_{role.upper()}_MODEL}}"} for role in roles},
                "apps": [{"name": "demo", "package_name": "com.demo"}],
                "experiment": {"graph_dir": str(tmp_path / "graphs")},
            }
        ),
        encoding="utf-8",
    )
    for role in roles:
        monkeypatch.setenv(f"APP_GRAPH_{role.upper()}_MODEL", f"provider/{role}")
    events: list[str] = []

    class Device:
        def __init__(self, _config: dict[str, object], _logger: logging.Logger) -> None:
            events.append("device")

        def _terminate_all_apps(self) -> None:
            events.append("terminate")

    class Explorer:
        def __init__(
            self,
            controller: Device,
            app_name: str,
            package_name: str,
            _logger: logging.Logger,
            *,
            vlm_config: dict[str, object],
            graph_dir: str,
            max_steps: int,
            **_kwargs: object,
        ) -> None:
            assert isinstance(controller, Device)
            assert (app_name, package_name) == ("demo", "com.demo")
            assert vlm_config == expected_models
            assert graph_dir == str(tmp_path / "graphs")
            assert max_steps == 3

        def explore(self, *, resume_from: str) -> None:
            assert resume_from == "auto"
            events.append("explore")

        def save_graph(self) -> None:
            events.append("save")

    def check_avd(_manager: object, config: dict[str, object], _logger: logging.Logger) -> None:
        assert config["vlm"] == expected_models
        events.append("avd")

    adb_module = ModuleType("aitk.utils.adb_controller")
    avd_module = ModuleType("aitk.utils.avd_manager")
    monkeypatch.setattr(adb_module, "ADBController", Device, raising=False)
    monkeypatch.setattr(avd_module, "AVDManager", lambda _logger: object(), raising=False)
    monkeypatch.setitem(sys.modules, adb_module.__name__, adb_module)
    monkeypatch.setitem(sys.modules, avd_module.__name__, avd_module)
    monkeypatch.setattr(cli, "Kobe", Explorer)
    monkeypatch.setattr(cli, "check_avd", check_avd)
    monkeypatch.setattr(cli, "launch_app", lambda *_args: events.append("launch"))
    monkeypatch.setattr(
        sys, "argv", ["app-graph", "-c", str(config), "--max-steps", "3", "--resume-from", "auto"]
    )
    main()
    assert events == ["avd", "device", "terminate", "launch", "explore", "save"]


def test_explorer_reports_all_missing_models_before_importing_device_tools(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "explore.yaml"
    config.write_text("vlm: {}\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["app-graph", "--config", str(config)])
    original_import = builtins.__import__

    def guarded_import(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        assert not name.startswith("aitk"), "configuration must be checked before device setup"
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(SystemExit) as caught:
        main()
    assert caught.value.code == 2
    error = capsys.readouterr().err
    for role in ("PAGE_DETAIL", "EMBEDDING", "INSTRUCTION", "ACTION"):
        assert f"APP_GRAPH_{role}_MODEL" in error
    assert "Traceback" not in error
