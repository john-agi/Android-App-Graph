"""app-graph argument parsing, model checks and run order, without adb or an emulator."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import NoReturn

import httpx
import pytest
from openrouter import OpenRouter

from android_app_graph import cli, device
from android_app_graph.cli import build_parser, main
from android_app_graph.kobe import ExplorerConfig


def test_help_exits_zero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """argparse handles --help inside parse_args() and exits 0 before any device work."""
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
                "device": {"udid": "emulator-5554", "unreachable_limit": 600},
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
        def __init__(
            self, _config: dict[str, object], _logger: logging.Logger, **_kwargs: object
        ) -> None:
            events.append("device")

        def terminate_all_apps(self, _recovery: device.StartupRecovery) -> None:
            events.append("terminate")

    class Explorer:
        def __init__(
            self,
            controller: Device,
            config: ExplorerConfig,
            *,
            client: OpenRouter,
            logger: logging.Logger,
        ) -> None:
            del logger
            assert isinstance(controller, Device)
            assert config.package_name == f"com.{config.app_name}"
            assert config.vlm_config == vlm
            assert config.graph_dir == str(tmp_path / "graphs")
            assert config.max_steps == 3
            assert config.unreachable_limit == 600
            assert client.sdk_configuration.client is router_http
            clients.append(client)
            launch_activities.append(config.launch_activity)

        def explore(self, *, resume_from: str) -> None:
            assert resume_from == "auto"
            events.append("explore")

        def save_graph(self) -> None:
            events.append("save")

    def check_avd(_manager: object, config: dict[str, object], _logger: logging.Logger) -> None:
        assert config["vlm"] == vlm
        events.append("avd")

    monkeypatch.setattr(cli, "AdbController", Device)
    monkeypatch.setattr(cli, "EmulatorManager", lambda _logger: object())
    monkeypatch.setattr(cli, "Kobe", Explorer)
    monkeypatch.setattr(cli, "check_avd", check_avd)
    grants: list[tuple[str, str]] = []

    def grant(udid: str, package_name: str, _recovery: device.StartupRecovery) -> None:
        grants.append((udid, package_name))
        events.append("grant")

    monkeypatch.setattr(cli, "grant_runtime_permissions", grant)

    def launch(
        _config: dict[str, object],
        app: dict[str, str],
        _logger: logging.Logger,
        _recovery: device.StartupRecovery,
    ) -> str:
        events.append("launch")
        return app.get("launch_activity", f"{app['package_name']}/.Home")

    monkeypatch.setattr(cli, "launch_app", launch)
    monkeypatch.setattr(
        sys, "argv", ["app-graph", "-c", str(config), "--max-steps", "3", "--resume-from", "auto"]
    )
    main()
    assert events == ["avd", "device", *["terminate", "grant", "launch", "explore", "save"] * 2]
    assert grants == [("emulator-5554", "com.demo"), ("emulator-5554", "com.other")]
    # The explorer restarts each app through the component its launch started.
    assert launch_activities == ["com.demo/.Home", "com.other/.Main"]
    assert opened_clients == clients[:1] == clients[1:]
    assert opened_clients[0].sdk_configuration.client is None


def _deny_device_work(monkeypatch: pytest.MonkeyPatch) -> None:
    def touched(*_args: object, **_kwargs: object) -> NoReturn:
        msg = "settings must be validated before any device work"
        raise AssertionError(msg)

    for name in ("EmulatorManager", "check_avd", "AdbController", "launch_app", "Kobe"):
        monkeypatch.setattr(cli, name, touched)


def test_explorer_reports_all_missing_models_before_any_device_work(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "explore.yaml"
    config.write_text("vlm: {}\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["app-graph", "--config", str(config)])
    _deny_device_work(monkeypatch)
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
        (
            {"matching": {"text_candidates": "two"}},
            "test",
            "vlm.matching.text_candidates must be a whole number of 0 or more",
        ),
        (
            {"instruction": {"max_tokens": 512}},
            "test",
            "Unknown configuration keys: vlm.instruction.max_tokens",
        ),
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
    if key is not None:
        monkeypatch.setenv("OPENROUTER_API_KEY", key)
    _deny_device_work(monkeypatch)
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


@pytest.mark.usefixtures("model_environment")
def test_explorer_rejects_an_invalid_unreachable_limit_before_any_device_work(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    _deny_device_work(monkeypatch)
    config = tmp_path / "explore.yaml"
    config.write_text(
        json.dumps({"device": {"udid": "emulator-5554", "unreachable_limit": 0}}), encoding="utf-8"
    )
    monkeypatch.setattr(sys, "argv", ["app-graph", "--config", str(config)])

    with pytest.raises(SystemExit) as caught:
        main()

    assert caught.value.code == 2
    error = capsys.readouterr().err
    assert "device.unreachable_limit must be a positive finite number of seconds" in error


@pytest.mark.parametrize("unreachable", ["adb server", "device"])
@pytest.mark.usefixtures("model_environment", "opened_clients", "no_sleep")
def test_main_exits_with_the_error_when_adb_or_the_device_cannot_be_reached(
    unreachable: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The error ends the run with status 1 and no traceback, after the explorer saved its
    graph."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    config = tmp_path / "explore.yaml"
    app = {"name": "demo", "package_name": "com.demo"}
    config.write_text(
        json.dumps({"device": {"udid": "emulator-5554"}, "apps": [app]}), encoding="utf-8"
    )
    server = device.AdbServerUnreachableError("adb devices timed out after 30 s")
    stays = device.DeviceUnreachableError("Device emulator-5554 unreachable for 30 minutes")
    saved: list[str] = []

    def check_avd(*_args: object) -> None:
        if unreachable == "adb server":
            raise server

    class Device:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def terminate_all_apps(self, _recovery: device.StartupRecovery) -> None:
            pass

    class Explorer:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def explore(self, **_kwargs: object) -> NoReturn:
            raise stays

        def save_graph(self) -> None:
            saved.append("saved")

    monkeypatch.setattr(cli, "check_avd", check_avd)
    monkeypatch.setattr(cli, "EmulatorManager", lambda _logger: object())
    monkeypatch.setattr(cli, "AdbController", Device)
    monkeypatch.setattr(cli, "grant_runtime_permissions", lambda *_args: None)
    monkeypatch.setattr(cli, "launch_app", lambda *_args: None)
    monkeypatch.setattr(cli, "Kobe", Explorer)
    monkeypatch.setattr(sys, "argv", ["app-graph", "-c", str(config)])

    with pytest.raises(SystemExit) as caught:
        main()

    error = server if unreachable == "adb server" else stays
    assert caught.value.code == f"app-graph: error: {error}"
    assert saved == ([] if unreachable == "adb server" else ["saved"])


class _NoRunningAvd:
    def __init__(self) -> None:
        self.duplicated: list[str] = []

    def get_running_avd_list(self) -> list[dict[str, object]]:
        return []

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


@pytest.mark.parametrize(
    ("unreachable", "cause"),
    [
        pytest.param(
            subprocess.CompletedProcess(
                ["adb", "devices"],
                1,
                stdout="",
                stderr="* cannot start server on remote host\n"
                "adb: failed to check server version: cannot connect to daemon at "
                "tcp:100.64.0.7:5037: Connection refused\n",
            ),
            "* cannot start server on remote host adb: failed to check server version: "
            "cannot connect to daemon at tcp:100.64.0.7:5037: Connection refused",
            id="refused",
        ),
        pytest.param(None, "adb devices timed out after 30 s", id="timed out"),
    ],
)
@pytest.mark.usefixtures("adb_on_path")
def test_check_avd_names_an_adb_server_it_cannot_reach_and_copies_and_starts_nothing(
    unreachable: subprocess.CompletedProcess[str] | None,
    cause: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A device list adb cannot read says nothing about which devices run, so no AVD is
    copied or started on this machine."""
    monkeypatch.setenv("ADB_SERVER_SOCKET", "tcp:100.64.0.7:5037")
    commands: list[list[str]] = []

    def run(
        command: list[str], *, timeout: float | None = None, **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        if unreachable is None:
            raise subprocess.TimeoutExpired(command, timeout or 0)
        return unreachable

    def must_not_start(*_args: object, **_kwargs: object) -> NoReturn:
        msg = "an emulator started although adb cannot reach its server"
        raise AssertionError(msg)

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(cli.subprocess, "Popen", must_not_start)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    (tmp_path / ".android" / "avd").mkdir(parents=True)
    logger = logging.getLogger("test")

    with pytest.raises(device.AdbServerUnreachableError) as raised:
        cli.check_avd(device.EmulatorManager(logger), {"device": {"avd_name": "Pixel"}}, logger)

    assert str(raised.value) == f"adb cannot reach its server at tcp:100.64.0.7:5037: {cause}"
    assert commands == [["adb", "devices"]]
    assert list((tmp_path / ".android" / "avd").iterdir()) == []
