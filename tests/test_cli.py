"""app-graph argument parsing, model checks and run order, without AITK, adb or an emulator."""

from __future__ import annotations

import builtins
import json
import logging
import sys
from pathlib import Path
from types import ModuleType
from typing import NoReturn

import httpx
import pytest
from openrouter import OpenRouter

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
def test_main_opens_one_client_for_every_app_and_closes_it_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    opened_clients: list[OpenRouter],
    router_http: httpx.Client,
) -> None:
    roles = ("page_detail", "embedding", "instruction", "action")
    vlm = {role: {"model": f"${{APP_GRAPH_{role.upper()}_MODEL}}"} for role in roles}
    config = tmp_path / "explore.yaml"
    config.write_text(
        json.dumps(
            {
                "device": {"udid": "emulator-5554"},
                "vlm": vlm,
                "apps": [
                    {"name": "demo", "package_name": "com.demo"},
                    {
                        "name": "other",
                        "package_name": "com.other",
                        "launch_activity": "com.other/.Main",
                    },
                ],
                "experiment": {"graph_dir": str(tmp_path / "graphs")},
            }
        ),
        encoding="utf-8",
    )
    for role in roles:
        monkeypatch.setenv(f"APP_GRAPH_{role.upper()}_MODEL", f"provider/{role}")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    events: list[str] = []
    clients: list[OpenRouter] = []
    launch_activities: list[str | None] = []

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
            client: OpenRouter,
            launch_activity: str | None,
            **_kwargs: object,
        ) -> None:
            assert isinstance(controller, Device)
            assert package_name == f"com.{app_name}"
            assert vlm_config == vlm
            assert graph_dir == str(tmp_path / "graphs")
            assert max_steps == 3
            assert client.sdk_configuration.client is router_http
            clients.append(client)
            launch_activities.append(launch_activity)

        def explore(self, *, resume_from: str) -> None:
            assert resume_from == "auto"
            events.append("explore")

        def save_graph(self) -> None:
            events.append("save")

    def check_avd(_manager: object, config: dict[str, object], _logger: logging.Logger) -> None:
        assert config["vlm"] == vlm
        events.append("avd")

    adb_module = ModuleType("aitk.utils.adb_controller")
    avd_module = ModuleType("aitk.utils.avd_manager")
    monkeypatch.setattr(adb_module, "ADBController", Device, raising=False)
    monkeypatch.setattr(avd_module, "AVDManager", lambda _logger: object(), raising=False)
    monkeypatch.setitem(sys.modules, adb_module.__name__, adb_module)
    monkeypatch.setitem(sys.modules, avd_module.__name__, avd_module)
    monkeypatch.setattr(cli, "Kobe", Explorer)
    monkeypatch.setattr(cli, "check_avd", check_avd)
    grants: list[tuple[str, str]] = []

    def grant(udid: str, package_name: str) -> None:
        grants.append((udid, package_name))
        events.append("grant")

    monkeypatch.setattr(cli, "grant_runtime_permissions", grant)
    monkeypatch.setattr(cli, "launch_app", lambda *_args: events.append("launch"))
    monkeypatch.setattr(
        sys, "argv", ["app-graph", "-c", str(config), "--max-steps", "3", "--resume-from", "auto"]
    )
    main()
    assert events == ["avd", "device", *["terminate", "grant", "launch", "explore", "save"] * 2]
    assert grants == [("emulator-5554", "com.demo"), ("emulator-5554", "com.other")]
    assert launch_activities == [None, "com.other/.Main"]
    assert opened_clients == clients[:1] == clients[1:]
    assert opened_clients[0].sdk_configuration.client is None


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


@pytest.mark.parametrize(
    ("role_settings", "key", "message"),
    [
        ({"action": {"temperature": "warm"}}, "test", "action.temperature must be"),
        ({}, None, "OPENROUTER_API_KEY is required"),
    ],
)
@pytest.mark.usefixtures("model_environment")
def test_explorer_rejects_invalid_settings_before_any_device_work(
    role_settings: dict[str, object],
    key: str | None,
    message: str,
    tmp_path: Path,
    opened_clients: list[OpenRouter],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def touched(*_args: object, **_kwargs: object) -> NoReturn:
        raise AssertionError("settings must be validated before any device work")

    if key is not None:
        monkeypatch.setenv("OPENROUTER_API_KEY", key)
    monkeypatch.setattr(cli, "check_avd", touched)
    monkeypatch.setattr(cli, "launch_app", touched)
    monkeypatch.setattr(cli, "Kobe", touched)
    for name, attribute in (
        ("aitk.utils.adb_controller", "ADBController"),
        ("aitk.utils.avd_manager", "AVDManager"),
    ):
        module = ModuleType(name)
        monkeypatch.setattr(module, attribute, touched, raising=False)
        monkeypatch.setitem(sys.modules, name, module)
    config = tmp_path / "explore.yaml"
    config.write_text(
        json.dumps({"vlm": role_settings, "apps": [{"name": "demo", "package_name": "com.demo"}]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", ["app-graph", "--config", str(config)])
    with pytest.raises(SystemExit) as caught:
        main()
    assert caught.value.code == 2
    error = capsys.readouterr().err
    assert "Traceback" not in error
    (line,) = [line for line in error.splitlines() if line.startswith("app-graph: error:")]
    assert message in line
    assert error.rstrip().endswith(line)
    assert opened_clients == []


class _NoRunningAvd:
    def __init__(self) -> None:
        self.duplicated: list[str] = []

    def get_running_avd_list(self) -> None:
        return None

    def duplicate_avd(self, avd_name: str) -> None:
        self.duplicated.append(avd_name)


@pytest.mark.usefixtures("no_sleep")
def test_check_avd_starts_the_duplicate_with_the_emulator_on_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started: list[list[str]] = []
    monkeypatch.setattr(cli.shutil, "which", lambda name: f"/opt/android-sdk/emulator/{name}")
    monkeypatch.setattr(cli.subprocess, "Popen", lambda cmd, **_kwargs: started.append(cmd))
    avds = _NoRunningAvd()

    cli.check_avd(avds, {"device": {"avd_name": "Pixel"}}, logging.getLogger("test"))

    assert avds.duplicated == ["Pixel"]
    emulator = "/opt/android-sdk/emulator/emulator"
    assert started == [[emulator, "-avd", "Pixel_dup", "-no-snapshot"]]


def test_check_avd_raises_file_not_found_naming_the_emulator_when_it_is_not_on_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``FileNotFoundError`` is what ``subprocess.Popen`` raises for a missing program."""

    def must_not_start(*_args: object, **_kwargs: object) -> NoReturn:
        msg = "the emulator started although it is not on PATH"
        raise AssertionError(msg)

    monkeypatch.setattr(cli.shutil, "which", lambda _name: None)
    monkeypatch.setattr(cli.subprocess, "Popen", must_not_start)
    with pytest.raises(FileNotFoundError, match=r"^emulator was not found on PATH$"):
        cli.check_avd(_NoRunningAvd(), {"device": {"avd_name": "Pixel"}}, logging.getLogger("test"))
