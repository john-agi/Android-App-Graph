"""Tests for android_app_graph.device, with adb, sleeps and the home directory faked."""

from __future__ import annotations

import base64
import logging
import re
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, NoReturn, override

import pytest

from android_app_graph import device

_FORCE_STOP = ["adb", "-s", "test-device", "shell", "am", "force-stop", "com.demo"]
_PM = ["adb", "-s", "test-device", "shell", "pm"]
_LAUNCHER_INTENT = ["-a", "android.intent.action.MAIN", "-c", "android.intent.category.LAUNCHER"]
_RESOLVE_LAUNCHER = [*_PM, "resolve-activity", "--brief", *_LAUNCHER_INTENT, "com.demo"]
_QUERY_LAUNCHERS = [*_PM, "query-activities", "--brief", *_LAUNCHER_INTENT, "com.demo"]
_CONFIG = {"device": {"udid": "test-device"}}


class _Device:
    w = 100
    h = 200

    def __init__(self) -> None:
        self.config: dict[str, Any] = {"device": {"udid": "test-device"}}

    def get_state(self, **_kwargs: float | None) -> dict[str, Any]:
        msg = "restarting the app must not read the screen"
        raise AssertionError(msg)

    def exe_action(self, action: dict[str, Any]) -> None:
        msg = f"restarting the app must not act on the screen: {action}"
        raise AssertionError(msg)


def _clear_task_start(component: str) -> list[str]:
    return ["adb", "-s", "test-device", "shell", "am", "start", "-n", component, "-f", "0x10008000"]


def test_restart_app_resolves_the_launcher_activity_once_and_starts_it_in_a_cleared_task(
    device_log: list[object], no_sleep: list[float]
) -> None:
    """Every restart of the run starts the same component, so each lands on the same screen."""
    app = device.AppLaunch("demo", "com.demo")
    for _ in range(2):
        device.restart_app(_Device(), app, logging.getLogger("test"))
    home = _clear_task_start("com.demo/.Home")
    assert device_log == [_FORCE_STOP, _RESOLVE_LAUNCHER, home, _FORCE_STOP, home]
    assert no_sleep == [4, 4]


def test_restart_app_starts_the_configured_launch_activity_in_a_cleared_task(
    device_log: list[object], no_sleep: list[float]
) -> None:
    app = device.AppLaunch("demo", "com.demo", "com.demo/.Main")
    device.restart_app(_Device(), app, logging.getLogger("test"))
    assert device_log == [_FORCE_STOP, _clear_task_start("com.demo/.Main")]
    assert no_sleep == [4]


# Both in the layout of the Pixel 9 emulator's `pm ... --brief` output.
_SYSTEM_CHOOSER = (
    "priority=0 preferredOrder=0 match=0x0 specificIndex=-1 isDefault=false\n"
    "android/com.android.internal.app.ResolverActivity\n"
)
_TWO_LAUNCHER_ACTIVITIES = (
    "2 activities found:\n"
    "  Activity #0:\n"
    "    priority=0 preferredOrder=0 match=0x108000 specificIndex=-1 isDefault=true\n"
    "    com.demo/.Home\n"
    "  Activity #1:\n"
    "    priority=0 preferredOrder=0 match=0x108000 specificIndex=-1 isDefault=true\n"
    "    com.demo/.Widgets\n"
)


def _answer_pm(
    monkeypatch: pytest.MonkeyPatch, device_log: list[object], *, resolved: str, listed: str
) -> None:
    """Answer ``pm resolve-activity`` with ``resolved`` and ``pm query-activities`` with
    ``listed``; every other adb command succeeds without output."""
    outputs = {"resolve-activity": resolved, "query-activities": listed}

    def run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        device_log.append(command)
        stdout = outputs.get(command[5], "")
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(device.subprocess, "run", run)


@pytest.mark.usefixtures("no_sleep")
def test_restart_app_starts_the_first_of_several_launcher_activities_and_names_the_others(
    monkeypatch: pytest.MonkeyPatch, device_log: list[object], caplog: pytest.LogCaptureFixture
) -> None:
    _answer_pm(monkeypatch, device_log, resolved=_SYSTEM_CHOOSER, listed=_TWO_LAUNCHER_ACTIVITIES)
    app = device.AppLaunch("demo", "com.demo")
    with caplog.at_level(logging.WARNING, logger="test"):
        device.restart_app(_Device(), app, logging.getLogger("test"))
    start = _clear_task_start("com.demo/.Home")
    assert device_log == [_FORCE_STOP, _RESOLVE_LAUNCHER, _QUERY_LAUNCHERS, start]
    warning = (
        "com.demo has several launcher activities; starting it through com.demo/.Home, "
        "not com.demo/.Widgets. Set launch_activity to choose another."
    )
    assert caplog.messages == [warning]


@pytest.mark.usefixtures("no_sleep")
def test_restart_app_starts_the_single_queried_launcher_with_an_empty_warning_log(
    monkeypatch: pytest.MonkeyPatch, device_log: list[object], caplog: pytest.LogCaptureFixture
) -> None:
    _answer_pm(
        monkeypatch,
        device_log,
        resolved=_SYSTEM_CHOOSER,
        listed="1 activities found:\n  Activity #0:\n    com.demo/.Home\n",
    )
    app = device.AppLaunch("demo", "com.demo")
    with caplog.at_level(logging.WARNING, logger="test"):
        device.restart_app(_Device(), app, logging.getLogger("test"))
    assert device_log == [
        _FORCE_STOP,
        _RESOLVE_LAUNCHER,
        _QUERY_LAUNCHERS,
        _clear_task_start("com.demo/.Home"),
    ]
    assert caplog.messages == []


@pytest.mark.usefixtures("no_sleep")
def test_restart_app_without_a_launcher_activity_asks_for_a_launch_activity_and_starts_nothing(
    monkeypatch: pytest.MonkeyPatch, device_log: list[object]
) -> None:
    _answer_pm(
        monkeypatch, device_log, resolved="No activity found\n", listed="No activities found\n"
    )
    app = device.AppLaunch("demo", "com.demo")
    with pytest.raises(
        device.NoLauncherActivityError,
        match=r"^com\.demo has no launcher activity; set launch_activity for it in the config$",
    ):
        device.restart_app(_Device(), app, logging.getLogger("test"))
    assert device_log == [_FORCE_STOP, _RESOLVE_LAUNCHER, _QUERY_LAUNCHERS]


_STARTING_GONE = "Starting: Intent { flg=0x10008000 cmp=com.demo/.Gone }\n"
_GONE = "Error type 3\nError: Activity class {com.demo/com.demo.Gone} does not exist.\n"
_GONE_ON_ONE_LINE = "Error type 3 Error: Activity class {com.demo/com.demo.Gone} does not exist."
_START_REFUSALS = pytest.mark.parametrize(
    ("returncode", "stdout", "stderr", "error"),
    [
        (1, _STARTING_GONE, _GONE, _GONE_ON_ONE_LINE),
        # An adb that reports no exit status prints both streams to stdout.
        (0, _STARTING_GONE + _GONE, "", f"{_STARTING_GONE.strip()} {_GONE_ON_ONE_LINE}"),
    ],
    ids=["exit status", "error line"],
)


def _refuse(
    monkeypatch: pytest.MonkeyPatch,
    device_log: list[object],
    refused: list[str],
    reply: tuple[int, str, str],
) -> None:
    """Answer ``refused`` with ``reply``, its exit status, stdout and stderr; every other
    adb command succeeds without output."""
    returncode, stdout, stderr = reply

    def run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        device_log.append(command)
        if command == refused:
            return subprocess.CompletedProcess(command, returncode, stdout=stdout, stderr=stderr)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(device.subprocess, "run", run)


@_START_REFUSALS
def test_restart_app_raises_naming_the_component_when_am_start_cannot_start_it(
    returncode: int,
    stdout: str,
    stderr: str,
    error: str,
    monkeypatch: pytest.MonkeyPatch,
    device_log: list[object],
    no_sleep: list[float],
) -> None:
    start = _clear_task_start("com.demo/.Gone")
    _refuse(monkeypatch, device_log, start, (returncode, stdout, stderr))
    app = device.AppLaunch("demo", "com.demo", "com.demo/.Gone")
    with pytest.raises(RuntimeError) as raised:
        device.restart_app(_Device(), app, logging.getLogger("test"))
    assert str(raised.value) == f"am start could not start com.demo/.Gone: {error}"
    assert device_log == [_FORCE_STOP, start]
    assert no_sleep == []


@pytest.mark.usefixtures("adb_on_path")
def test_restart_app_launches_nothing_when_the_force_stop_fails(
    monkeypatch: pytest.MonkeyPatch, no_sleep: list[float]
) -> None:
    commands: list[list[str]] = []

    def offline(
        command: list[str], *, check: bool = False, **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        if check:
            raise subprocess.CalledProcessError(1, command)
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="device offline")

    monkeypatch.setattr(device.subprocess, "run", offline)
    app = device.AppLaunch("demo", "com.demo")
    with pytest.raises(device.ForceStopRefusedError):
        device.restart_app(_Device(), app, logging.getLogger("test"))
    assert commands == [_FORCE_STOP]
    assert no_sleep == []


def _start(component: str) -> list[str]:
    return ["adb", "-s", "test-device", "shell", "am", "start", "-n", component]


@pytest.mark.parametrize(
    ("app", "launches", "component"),
    [
        pytest.param(
            {"name": "demo", "package_name": "com.demo"},
            [_RESOLVE_LAUNCHER, _start("com.demo/.Home")],
            "com.demo/.Home",
            id="launcher activity",
        ),
        pytest.param(
            {"name": "demo", "package_name": "com.demo", "launch_activity": "com.demo/.Main"},
            [_start("com.demo/.Main")],
            "com.demo/.Main",
            id="launch activity",
        ),
    ],
)
def test_launch_app_starts_the_launch_activity_or_else_the_launcher_activity_and_returns_it(
    app: dict[str, str], launches: list[list[str]], component: str, device_log: list[object]
) -> None:
    """The launch returns the component it started, which the run's restarts start too."""
    assert (
        device.launch_app(_CONFIG, app, logging.getLogger("test"), _startup_recovery()) == component
    )
    assert device_log == launches


def test_launch_app_without_a_launcher_activity_raises_naming_the_package_and_starts_nothing(
    monkeypatch: pytest.MonkeyPatch, device_log: list[object]
) -> None:
    _answer_pm(
        monkeypatch, device_log, resolved="No activity found\n", listed="No activities found\n"
    )
    with pytest.raises(
        device.NoLauncherActivityError,
        match=r"^com\.demo has no launcher activity; set launch_activity for it in the config$",
    ):
        device.launch_app(
            _CONFIG,
            {"name": "demo", "package_name": "com.demo"},
            logging.getLogger("test"),
            _startup_recovery(),
        )
    assert device_log == [_RESOLVE_LAUNCHER, _QUERY_LAUNCHERS]


@_START_REFUSALS
def test_launch_app_raises_naming_the_component_when_am_start_cannot_start_it(
    returncode: int,
    stdout: str,
    stderr: str,
    error: str,
    monkeypatch: pytest.MonkeyPatch,
    device_log: list[object],
) -> None:
    start = _start("com.demo/.Gone")
    _refuse(monkeypatch, device_log, start, (returncode, stdout, stderr))
    app = {"name": "demo", "package_name": "com.demo", "launch_activity": "com.demo/.Gone"}
    with pytest.raises(RuntimeError) as raised:
        device.launch_app(_CONFIG, app, logging.getLogger("test"), _startup_recovery())
    assert str(raised.value) == f"am start could not start com.demo/.Gone: {error}"
    assert device_log == [start]


def test_adb_executable_raises_file_not_found_naming_adb_when_adb_is_not_on_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``FileNotFoundError`` is what ``subprocess`` raises for a missing program, so
    callers that tolerate a missing adb by catching ``OSError`` still do."""
    monkeypatch.setattr(device.shutil, "which", lambda _name: None)
    with pytest.raises(FileNotFoundError, match=r"^adb was not found on PATH$"):
        device.adb_executable()


def test_run_adb_runs_the_adb_on_path_with_the_given_options_in_text_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A call that names no timeout gets the 30 s every read gets."""
    calls: list[tuple[list[str], dict[str, object]]] = []

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(device.shutil, "which", lambda name: f"/opt/platform-tools/{name}")
    monkeypatch.setattr(device.subprocess, "run", run)
    device.run_adb(
        ["shell", "dumpsys", "input_method"],
        output="capture",
        errors="replace",
        timeout=3,
        check=True,
    )
    device.run_adb(["shell", "am", "force-stop", "com.demo"], output="discard", check=False)

    adb = "/opt/platform-tools/adb"
    assert calls == [
        (
            [adb, "shell", "dumpsys", "input_method"],
            {
                "capture_output": True,
                "stdout": None,
                "stderr": None,
                "text": True,
                "encoding": "utf-8",
                "errors": "replace",
                "timeout": 3,
                "check": True,
            },
        ),
        (
            [adb, "shell", "am", "force-stop", "com.demo"],
            {
                "capture_output": False,
                "stdout": subprocess.DEVNULL,
                "stderr": subprocess.DEVNULL,
                "text": True,
                "encoding": "utf-8",
                "errors": None,
                "timeout": 30,
                "check": False,
            },
        ),
    ]


class _CompletedProcess:
    """The subset of subprocess.CompletedProcess the probe reads."""

    def __init__(self, stdout: str) -> None:
        self.stdout = stdout


@pytest.mark.usefixtures("adb_on_path")
def test_soft_keyboard_hint_asks_the_named_device_when_several_are_connected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With more than one device connected, adb refuses a command that names none, so
    the probe names the device it asks with ``-s``."""
    commands: list[list[str]] = []

    def several_devices(command: list[str], **_kwargs: object) -> _CompletedProcess:
        commands.append(command)
        names_a_device = command[1:3] == ["-s", "emulator-5558"]
        return _CompletedProcess("mInputShown=true" if names_a_device else "")

    monkeypatch.setattr(device.subprocess, "run", several_devices)
    hint = device.soft_keyboard_hint("emulator-5558")
    assert "soft keyboard is currently visible" in hint
    assert commands == [["adb", "-s", "emulator-5558", "shell", "dumpsys", "input_method"]]


@pytest.mark.usefixtures("adb_on_path")
def test_soft_keyboard_hint_when_the_keyboard_is_shown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        device.subprocess, "run", lambda *_a, **_k: _CompletedProcess("mInputShown=true")
    )
    hint = device.soft_keyboard_hint("test-device")
    assert "soft keyboard is currently visible" in hint
    assert "focused" not in hint
    assert "ready for typing" not in hint


@pytest.mark.usefixtures("adb_on_path")
def test_soft_keyboard_hint_when_the_keyboard_is_hidden(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(device.subprocess, "run", lambda *_a, **_k: _CompletedProcess(""))
    assert device.soft_keyboard_hint("test-device") == ""


@pytest.mark.usefixtures("adb_on_path")
def test_soft_keyboard_hint_survives_a_missing_adb(monkeypatch: pytest.MonkeyPatch) -> None:
    def _no_adb(*_args: object, **_kwargs: object) -> _CompletedProcess:
        msg = "adb"
        raise FileNotFoundError(msg)

    monkeypatch.setattr(device.subprocess, "run", _no_adb)
    assert device.soft_keyboard_hint("test-device") == ""


def test_soft_keyboard_hint_is_empty_when_adb_is_not_on_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def must_not_run(*_args: object, **_kwargs: object) -> NoReturn:
        msg = "adb ran although it is not on PATH"
        raise AssertionError(msg)

    monkeypatch.setattr(device.shutil, "which", lambda _name: None)
    monkeypatch.setattr(device.subprocess, "run", must_not_run)
    assert device.soft_keyboard_hint("test-device") == ""


@pytest.mark.usefixtures("adb_on_path")
def test_soft_keyboard_hint_survives_undecodable_dumpsys_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``text=True`` decodes adb's stdout; undecodable bytes must not raise out of a
    probe that has to stay best-effort.
    """

    def _bad_bytes(*_args: object, **_kwargs: object) -> _CompletedProcess:
        msg = "invalid start byte"
        encoding = "utf-8"
        raise UnicodeDecodeError(encoding, b"\xff", 0, 1, msg)

    monkeypatch.setattr(device.subprocess, "run", _bad_bytes)
    assert device.soft_keyboard_hint("test-device") == ""


@pytest.mark.usefixtures("adb_on_path")
def test_soft_keyboard_hint_reads_past_a_replacement_character(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With ``errors="replace"`` a decode failure yields a replacement character
    (U+FFFD) next to the real output instead of raising; the hint is still found.
    """
    monkeypatch.setattr(
        device.subprocess,
        "run",
        lambda *_a, **_k: _CompletedProcess("�mInputShown=true"),
    )
    hint = device.soft_keyboard_hint("test-device")
    assert "soft keyboard is currently visible" in hint


def test_keyboard_status_when_the_keyboard_is_shown() -> None:
    assert device.keyboard_status(" (Note: the soft keyboard is visible.)") == (
        "Soft keyboard is visible; identify the intended editable field from the current screen."
    )


def test_keyboard_status_when_the_keyboard_is_hidden() -> None:
    assert device.keyboard_status("") == (
        "OS keyboard visibility is unknown. Inspect the current screen for the intended editable field."
    )


@pytest.mark.parametrize(
    ("package", "outside"),
    [
        ("com.android.chrome", True),
        ("com.demo", False),
        ("com.google.android.inputmethod.latin", True),
        ("com.google.android.permissioncontroller", True),
        ("", False),
    ],
)
def test_only_another_app_in_front_is_outside_the_app(package: str, outside: bool) -> None:
    assert device.is_outside_app(package, "com.demo") is outside


@pytest.mark.usefixtures("no_sleep")
def test_return_to_app_lets_a_refused_force_stop_through(
    screens_by_log: Callable[..., device.DeviceController],
    device_log: list[object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    other_app = {
        "activity": "com.android.chrome.Main",
        "screenshot": "",
        "package": "com.android.chrome",
    }
    controller = screens_by_log(other_app)

    def refuse(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        device_log.append(command)
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="device offline")

    monkeypatch.setattr(device.subprocess, "run", refuse)
    with pytest.raises(device.ForceStopRefusedError):
        device.return_to_app(
            controller, device.AppLaunch("demo", "com.demo"), logging.getLogger("test")
        )
    assert device_log == [{"action": "back"}, _FORCE_STOP]


# Hand-written in the layout of `adb shell dumpsys package`, not captured from a device.
_DUMPSYS_PACKAGE = Path(__file__).parent / "data" / "dumpsys_package.txt"
_REQUESTED_PERMISSIONS = [
    "android.permission.INTERNET",
    "android.permission.ACCESS_NETWORK_STATE",
    "android.permission.CAMERA",
    "android.permission.ACCESS_FINE_LOCATION",
    "android.permission.ACCESS_COARSE_LOCATION",
    "android.permission.POST_NOTIFICATIONS",
    "android.permission.READ_EXTERNAL_STORAGE",
    "android.permission.WAKE_LOCK",
    "com.google.android.c2dm.permission.RECEIVE",
    "com.example.app.DYNAMIC_RECEIVER_NOT_EXPORTED_PERMISSION",
]
_REFUSAL = (
    "Exception occurred while executing 'grant':\n"
    "java.lang.SecurityException: Permission android.permission.INTERNET requested by "
    "com.example.app is not a changeable permission type\n"
    "\tat com.android.server.pm.permission.PermissionManagerServiceImpl"
    ".grantRuntimePermissionInternal(PermissionManagerServiceImpl.java:1433)\n"
    "\tat com.android.server.pm.permission.PermissionManagerServiceImpl"
    ".grantRuntimePermission(PermissionManagerServiceImpl.java:1364)\n"
)


@pytest.mark.usefixtures("adb_on_path")
def test_each_requested_permission_is_granted_and_a_refusal_is_logged_and_skipped(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    dumpsys_package = _DUMPSYS_PACKAGE.read_text(encoding="utf-8")
    commands: list[list[str]] = []

    def adb(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(args)
        if "dumpsys" in args:
            return subprocess.CompletedProcess(args, 0, stdout=dumpsys_package, stderr="")
        if args[-1] == "android.permission.INTERNET":
            return subprocess.CompletedProcess(args, 255, stdout="", stderr=_REFUSAL)
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(device.subprocess, "run", adb)
    with caplog.at_level(logging.INFO, logger="android_app_graph.device"):
        device.grant_runtime_permissions("emulator-5554", "com.example.app", _startup_recovery())

    shell = ["adb", "-s", "emulator-5554", "shell"]
    assert commands == [
        [*shell, "dumpsys", "package", "com.example.app"],
        *([*shell, "pm", "grant", "com.example.app", name] for name in _REQUESTED_PERMISSIONS),
    ]
    refusal = (
        "Could not grant android.permission.INTERNET to com.example.app: "
        "Exception occurred while executing 'grant': java.lang.SecurityException: "
        "Permission android.permission.INTERNET requested by com.example.app "
        "is not a changeable permission type"
    )
    assert caplog.messages == [refusal, "Granted 9 of 10 requested permissions to com.example.app"]


_UDID = "emulator-5554"
_ADB = ["adb", "-s", _UDID]
_INPUT = [*_ADB, "shell", "input"]
_WM_SIZE = ("-s", _UDID, "shell", "wm", "size")
_UI_DUMP = ("-s", _UDID, "exec-out", "uiautomator dump /dev/stdout")
_SCREENCAP = ("-s", _UDID, "exec-out", "screencap -p")
_DUMPSYS_WINDOW = ("-s", _UDID, "shell", "dumpsys", "window", "-a")


class _FakeAdb:
    """Stands in for ``subprocess.run`` and ``time.sleep``.

    ``events`` records each adb argv and each sleep in the order they happened, and
    ``timeouts`` the timeout of each adb call. ``replies`` maps the arguments after
    ``adb`` to an exit status and raw stdout; any other command exits 0 with no
    output. ``queued`` maps the arguments after ``adb`` to replies of the same form,
    one for each of their next calls, before ``replies`` answers them. Stdout stays
    bytes unless the caller asks for text, which must name its
    encoding: the host's locale differs between machines, and Android writes UTF-8.
    ``stalls`` maps the arguments after ``adb`` to how many of their next calls stall
    until their timeout expires. Every call must name a timeout: a call without one
    blocks for as long as the device does not answer.
    """

    def __init__(self) -> None:
        self.events: list[list[str] | float] = []
        self.timeouts: list[float | None] = []
        self.replies: dict[tuple[str, ...], tuple[int, bytes]] = {}
        self.queued: dict[tuple[str, ...], list[tuple[int, bytes]]] = {}
        self.stalls: dict[tuple[str, ...], int] = {}

    def run(
        self,
        command: list[str],
        *,
        check: bool = False,
        text: bool = False,
        encoding: str | None = None,
        timeout: float | None = None,
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[str] | subprocess.CompletedProcess[bytes]:
        if timeout is None:
            msg = f"adb call without a timeout: {command}"
            raise AssertionError(msg)
        self.events.append(command)
        self.timeouts.append(timeout)
        args = tuple(command[1:])
        if self.stalls.get(args, 0) > 0:
            self.stalls[args] -= 1
            raise subprocess.TimeoutExpired(command, timeout)
        queued = self.queued.get(args)
        returncode, stdout = queued.pop(0) if queued else self.replies.get(args, (0, b""))
        if check and returncode != 0:
            raise subprocess.CalledProcessError(returncode, command)
        if not text:
            return subprocess.CompletedProcess(command, returncode, stdout, b"")
        if encoding is None:
            msg = f"text output without an encoding depends on the locale: {command}"
            raise AssertionError(msg)
        return subprocess.CompletedProcess(command, returncode, stdout.decode(encoding), "")


@pytest.fixture
def adb(monkeypatch: pytest.MonkeyPatch, adb_on_path: None) -> _FakeAdb:
    del adb_on_path  # requested for its side effect on device.adb_executable
    fake = _FakeAdb()
    monkeypatch.setattr(device.subprocess, "run", fake.run)
    monkeypatch.setattr(device.time, "sleep", fake.events.append)
    return fake


@pytest.fixture
def controller(adb: _FakeAdb) -> device.AdbController:
    adb.replies[_WM_SIZE] = (0, b"Physical size: 1080x2400\n")
    built = device.AdbController(
        {"device": {"udid": _UDID}}, logging.getLogger("test"), unreachable_limit=1800
    )
    adb.events.clear()
    adb.timeouts.clear()
    adb.replies[_DUMPSYS_WINDOW] = (
        0,
        (
            b"  WindowInsetsStateController\n    InsetsState\n"
            b"      mDisplayFrame=Rect(0, 0 - 1080, 2400)\n"
            b"      InsetsSource type=tappableElement frame=[0,0][0,0] visible=true\n"
        ),
    )
    return built


def test_adb_controller_reads_the_screen_size_of_the_configured_device(adb: _FakeAdb) -> None:
    adb.replies[_WM_SIZE] = (0, b"Physical size: 1080x2400\n")
    config = {"device": {"udid": _UDID}}

    controller = device.AdbController(config, logging.getLogger("test"), unreachable_limit=1800)

    assert (controller.w, controller.h) == (1080, 2400)
    assert controller.config is config
    assert adb.events == [["adb", *_WM_SIZE]]


def test_adb_controller_raises_when_wm_size_reports_no_size(adb: _FakeAdb) -> None:
    adb.replies[_WM_SIZE] = (0, b"Physical size: unknown\n")
    with pytest.raises(ValueError, match=r"^Failed to get screen size$"):
        device.AdbController(
            {"device": {"udid": _UDID}}, logging.getLogger("test"), unreachable_limit=1800
        )


# PNG's signature contains CR LF, so any newline translation would change the base64.
_PNG = b"\x89PNG\r\n\x1a\n"

_BUTTON_NAVIGATION = (
    "  WindowInsetsStateController\n"
    "    InsetsState\n"
    "      mDisplayFrame=Rect(0, 0 - 720, 1280)\n"
    "        InsetsSource id=1 type=navigationBars frame=[0,1184][720,1280] visible=true\n"
    "        InsetsSource id=6 type=tappableElement frame=[0,1184][720,1280] visible=true\n"
    "        InsetsSource id=5 type=mandatorySystemGestures frame=[0,1184][720,1280] visible=true\n"
)
_GESTURE_NAVIGATION = (
    "  WindowInsetsStateController\n"
    "    InsetsState\n"
    "      mDisplayFrame=Rect(0, 0 - 1080, 2424)\n"
    "        InsetsSource id=1 type=navigationBars frame=[0,2361][1080,2424] visible=true\n"
    "        InsetsSource id=6 type=tappableElement frame=[0,0][0,0] visible=true\n"
    "        InsetsSource id=4 type=systemGestures frame=[0,0][78,2424] visible=true\n"
    "        InsetsSource id=24 type=systemGestures frame=[1002,0][1080,2424] visible=true\n"
    "        InsetsSource id=5 type=mandatorySystemGestures frame=[0,2340][1080,2424] visible=true\n"
)


@pytest.mark.parametrize(
    ("window", "action", "reason"),
    [
        (_BUTTON_NAVIGATION, {"action": "tap", "x": 299, "y": 1227}, "system touch region"),
        (_BUTTON_NAVIGATION, {"action": "tap", "x": 257, "y": 1190}, "system touch region"),
        (_BUTTON_NAVIGATION, {"action": "long_press", "x": 360, "y": 1250}, "system touch region"),
        (
            _GESTURE_NAVIGATION,
            {"action": "swipe", "x1": 54, "y1": 800, "x2": 700, "y2": 800},
            "system gesture region",
        ),
        (
            _GESTURE_NAVIGATION,
            {"action": "swipe", "x1": 500, "y1": 2400, "x2": 500, "y2": 900},
            "system gesture region",
        ),
        (
            (
                "  WindowInsetsStateController\n    InsetsState\n"
                "      mDisplayFrame=Rect(0, 0 - 1280, 720)\n"
                "      InsetsSource type=tappableElement frame=[0,0][0,0] visible=true\n"
            ),
            {"action": "tap", "x": 331, "y": 1002},
            "outside the display",
        ),
    ],
    ids=[
        "music preview",
        "obscured row",
        "held button",
        "back gesture",
        "home gesture",
        "rotation",
    ],
)
def test_app_coordinate_inputs_reject_system_regions_without_dispatching(
    controller: device.AdbController,
    adb: _FakeAdb,
    window: str,
    action: dict[str, object],
    reason: str,
) -> None:
    adb.replies[_DUMPSYS_WINDOW] = (0, window.encode())

    with pytest.raises(ValueError, match=reason):
        controller.exe_action(action)

    assert not any(isinstance(event, list) and event[:5] == _INPUT for event in adb.events)


@pytest.mark.parametrize(
    ("window", "action"),
    [
        (_BUTTON_NAVIGATION, {"action": "tap", "x": 257, "y": 1183}),
        (_GESTURE_NAVIGATION, {"action": "tap", "x": 20, "y": 2380}),
        (_GESTURE_NAVIGATION, {"action": "long_press", "x": 20, "y": 800}),
        (
            _GESTURE_NAVIGATION,
            {"action": "swipe", "x1": 100, "y1": 900, "x2": 900, "y2": 900},
        ),
        (
            _BUTTON_NAVIGATION,
            {"action": "swipe", "x1": 20, "y1": 700, "x2": 500, "y2": 700},
        ),
    ],
    ids=["revealed row", "near-bottom tap", "edge hold", "interior swipe", "button-mode drawer"],
)
def test_app_inputs_preserve_valid_content_coordinates(
    controller: device.AdbController,
    adb: _FakeAdb,
    window: str,
    action: dict[str, object],
) -> None:
    adb.replies[_DUMPSYS_WINDOW] = (0, window.encode())
    controller.exe_action(action)
    dispatched = [event for event in adb.events if isinstance(event, list) and event[:5] == _INPUT]
    assert len(dispatched) == 1
    if action["action"] == "tap":
        assert dispatched[0][-2:] == [str(action["x"]), str(action["y"])]


def test_coordinate_input_defers_when_current_geometry_is_unavailable(
    controller: device.AdbController, adb: _FakeAdb
) -> None:
    adb.replies[_DUMPSYS_WINDOW] = (0, b"mCurrentFocus=null\n")
    with pytest.raises(device.InputRejectedError, match="regions are unavailable") as error:
        controller.exe_action({"action": "tap", "x": 10, "y": 20})
    assert error.value.regions is None
    assert adb.events == [["adb", *_DUMPSYS_WINDOW]]


def test_a_rotation_between_observation_and_dispatch_defers_the_input(
    controller: device.AdbController, adb: _FakeAdb
) -> None:
    adb.replies[_DUMPSYS_WINDOW] = (0, _BUTTON_NAVIGATION.encode())
    state = controller.get_state()
    assert state["input_regions"]["display"] == [0, 0, 720, 1280]
    assert (controller.w, controller.h) == (720, 1280)
    adb.events.clear()
    adb.replies[_DUMPSYS_WINDOW] = (
        0,
        (
            b"WindowInsetsStateController\nInsetsState\nmDisplayFrame=Rect(0, 0 - 1280, 720)\n"
            b"InsetsSource type=tappableElement frame=[0,0][0,0] visible=true\n"
        ),
    )
    with pytest.raises(device.InputRejectedError, match="Display geometry changes") as error:
        controller.exe_action({"action": "tap", "x": 100, "y": 200})
    assert error.value.regions == {"display": [0, 0, 1280, 720], "tap": [], "gesture": []}
    assert adb.events == [["adb", *_DUMPSYS_WINDOW]]


@pytest.mark.parametrize(
    ("dumpsys_window", "package", "activity"),
    [
        pytest.param(
            "  mCurrentFocus=Window{3553291 u0 com.supercook.app/com.supercook.app.MainActivity}\n"
            "  mFocusedApp=ActivityRecord{5baa84e u0 com.other.app/.Other t9}\n",
            "com.supercook.app",
            "com.supercook.app.MainActivity",
            id="mCurrentFocus",
        ),
        pytest.param(
            "  mCurrentFocus=null\n"
            "  mFocusedApp=ActivityRecord{5baa84e u0 "
            "com.supercook.app/com.supercook.app.Main t9}\n",
            "com.supercook.app",
            "com.supercook.app.Main",
            id="mFocusedApp",
        ),
        pytest.param(
            "  mCurrentFocus=Window{3553291 u0 com.supercook.app/.Main}\n",
            "com.supercook.app",
            "com.supercook.app.Main",
            id="shorthand activity",
        ),
        pytest.param("WINDOW MANAGER WINDOWS (dumpsys window windows)\n", "unknown", "unknown"),
    ],
)
def test_get_state_returns_the_xml_screenshot_package_and_activity(
    controller: device.AdbController,
    adb: _FakeAdb,
    dumpsys_window: str,
    package: str,
    activity: str,
) -> None:
    xml = '<hierarchy><node text="Café" /></hierarchy>'
    adb.replies[_UI_DUMP] = (0, xml.encode())
    adb.replies[_SCREENCAP] = (0, _PNG)
    adb.replies[_DUMPSYS_WINDOW] = (0, dumpsys_window.encode())

    assert controller.get_state() == {
        "xml": xml,
        "screenshot": base64.b64encode(_PNG).decode("ascii"),
        "package": package,
        "activity": activity,
        "status_bar": None,
        "system_surface": None,
        "system_overlays": [],
        "permission_requester": None,
        "underlying_package": (
            "com.other.app"
            if "com.other.app" in dumpsys_window
            else "com.supercook.app"
            if "mFocusedApp" in dumpsys_window
            else "unknown"
        ),
        "input_regions": None,
    }
    assert adb.events == [["adb", *_UI_DUMP], ["adb", *_DUMPSYS_WINDOW], ["adb", *_SCREENCAP]]


def _dumpsys_window(name: str) -> str:
    return (Path(__file__).parent / "data" / f"dumpsys_window_{name}.txt").read_text(
        encoding="utf-8"
    )


def test_get_state_reads_the_status_bar_from_the_dump_it_reads_the_activity_from(
    controller: device.AdbController, adb: _FakeAdb
) -> None:
    adb.replies[_DUMPSYS_WINDOW] = (0, _dumpsys_window("api36_redroid").encode())

    state = controller.get_state()

    assert (state["package"], state["status_bar"]) == ("com.android.settings", [0, 0, 720, 48])
    assert adb.events.count(["adb", *_DUMPSYS_WINDOW]) == 1


# Trimmed device captures: Redroid on API 36, pixel_9 and pixel_8a on API 37 emulators.
@pytest.mark.parametrize(
    ("fixture", "frame"),
    [
        ("api36_redroid", [0, 0, 720, 48]),
        ("api37_pixel_9", [0, 0, 1080, 142]),
        ("api37_pixel_8a", [0, 0, 1080, 121]),
    ],
)
def test_the_status_bar_frame_is_the_first_status_bar_insets_source(
    fixture: str, frame: list[int]
) -> None:
    assert device.status_bar_frame(_dumpsys_window(fixture)) == frame


@pytest.mark.parametrize(
    "dumpsys_window",
    [
        pytest.param(
            "        InsetsSource id=bf360000 type=statusBars frame=[0,0][1080,121] "
            "visible=false flags= sideHint=TOP\n",
            id="hidden",
        ),
        pytest.param(
            "        InsetsSource id=1 type=statusBars frame=[0,2300][1080,2400] visible=true\n",
            id="not at the top",
        ),
        pytest.param(
            "  mCurrentFocus=Window{3553291 u0 com.supercook.app/.Main}\n"
            "        InsetsSource id=27 type=displayCutout frame=[0,0][1080,121] visible=true\n",
            id="not reported",
        ),
    ],
)
def test_a_hidden_misplaced_or_missing_status_bar_gives_no_frame(dumpsys_window: str) -> None:
    assert device.status_bar_frame(dumpsys_window) is None


@pytest.mark.parametrize("failing", [_UI_DUMP, _SCREENCAP, _DUMPSYS_WINDOW])
def test_get_state_raises_when_a_read_fails(
    controller: device.AdbController, adb: _FakeAdb, failing: tuple[str, ...]
) -> None:
    adb.replies[failing] = (1, b"")
    with pytest.raises(subprocess.CalledProcessError):
        controller.get_state()


_STATE_READS = [["adb", *_UI_DUMP], ["adb", *_DUMPSYS_WINDOW], ["adb", *_SCREENCAP]]
_EACH_STATE_READ = [
    pytest.param(_UI_DUMP, "-s emulator-5554 exec-out 'uiautomator dump /dev/stdout'", id="dump"),
    pytest.param(_SCREENCAP, "-s emulator-5554 exec-out 'screencap -p'", id="screenshot"),
    pytest.param(_DUMPSYS_WINDOW, "-s emulator-5554 shell dumpsys window -a", id="dumpsys window"),
]


def test_get_state_gives_each_read_a_30_second_timeout(
    controller: device.AdbController, adb: _FakeAdb
) -> None:
    controller.get_state()
    assert adb.events == _STATE_READS
    assert adb.timeouts == [30, 30, 30]


@pytest.mark.parametrize(("stalled", "command"), _EACH_STATE_READ)
def test_get_state_runs_a_read_that_times_out_once_more_and_logs_it(
    controller: device.AdbController,
    adb: _FakeAdb,
    caplog: pytest.LogCaptureFixture,
    stalled: tuple[str, ...],
    command: str,
) -> None:
    adb.replies[_UI_DUMP] = (0, b"<hierarchy />")
    adb.replies[_SCREENCAP] = (0, _PNG)
    adb.replies[_DUMPSYS_WINDOW] = (0, b"  mCurrentFocus=Window{1 u0 com.demo/.Main}\n")
    adb.stalls[stalled] = 1

    with caplog.at_level(logging.WARNING, logger="test"):
        state = controller.get_state()

    assert state == {
        "xml": "<hierarchy />",
        "screenshot": base64.b64encode(_PNG).decode("ascii"),
        "package": "com.demo",
        "activity": "com.demo.Main",
        "status_bar": None,
        "system_surface": None,
        "system_overlays": [],
        "permission_requester": None,
        "underlying_package": "unknown",
        "input_regions": None,
    }
    retried = _STATE_READS.index(["adb", *stalled])
    assert adb.events == [*_STATE_READS[: retried + 1], *_STATE_READS[retried:]]
    assert adb.timeouts == [30, 30, 30, 30]
    assert caplog.messages == [f"adb {command} timed out after 30 s; retrying once"]


@pytest.mark.parametrize("stalled", [_UI_DUMP, _SCREENCAP, _DUMPSYS_WINDOW])
def test_get_state_raises_naming_the_command_when_a_read_times_out_twice(
    controller: device.AdbController, adb: _FakeAdb, stalled: tuple[str, ...]
) -> None:
    adb.stalls[stalled] = 2

    with pytest.raises(subprocess.TimeoutExpired) as raised:
        controller.get_state()

    assert (raised.value.cmd, raised.value.timeout) == (["adb", *stalled], 30)
    stopped = _STATE_READS.index(["adb", *stalled])
    assert adb.events == [*_STATE_READS[: stopped + 1], ["adb", *stalled]]


@pytest.mark.parametrize(
    ("action", "events"),
    [
        ({"action": "tap", "x": 10, "y": 20}, [[*_INPUT, "tap", "10", "20"]]),
        (
            {"action": "long_press", "x": 10, "y": 20, "duration": 3000},
            [[*_INPUT, "swipe", "10", "20", "10", "20", "1000"]],
        ),
        (
            {"action": "swipe", "x1": 1, "y1": 2, "x2": 3, "y2": 4, "duration": 500},
            [[*_INPUT, "swipe", "1", "2", "3", "4", "500"]],
        ),
        (
            {"action": "swipe", "x1": 1, "y1": 2, "x2": 3, "y2": 4, "time": 700},
            [[*_INPUT, "swipe", "1", "2", "3", "4", "700"]],
        ),
        (
            {"action": "swipe", "x1": 1, "y1": 2, "x2": 3, "y2": 4},
            [[*_INPUT, "swipe", "1", "2", "3", "4", "1000"]],
        ),
        ({"action": "enter"}, [[*_INPUT, "keyevent", "KEYCODE_ENTER"], 1]),
        ({"action": "back"}, [[*_INPUT, "keyevent", "KEYCODE_BACK"], 1]),
        ({"action": "home"}, [[*_INPUT, "keyevent", "KEYCODE_HOME"], 1]),
        ({"action": "menu"}, [[*_INPUT, "keyevent", "KEYCODE_MENU"], 1]),
        ({"action": "wait", "duration": 2}, [2]),
        ({"action": "wait", "time": 5}, [5]),
        ({"action": "wait"}, [3]),
        ({"action": "end"}, []),
    ],
)
def test_each_action_sends_its_adb_command_and_pause(
    controller: device.AdbController,
    adb: _FakeAdb,
    action: dict[str, Any],
    events: list[list[str] | float],
) -> None:
    controller.exe_action(action)
    observed = (
        [["adb", *_DUMPSYS_WINDOW], *events]
        if action["action"] in {"tap", "long_press", "swipe"}
        else events
    )
    assert adb.events == observed


@pytest.mark.parametrize(
    "action",
    [
        {"action": "tap", "x": 10, "y": 20},
        {"action": "long_press", "x": 10, "y": 20},
        {"action": "swipe", "x1": 1, "y1": 2, "x2": 3, "y2": 4},
        {"action": "back"},
        {"action": "menu"},
        {"action": "type", "text": "Hi"},
    ],
)
def test_every_input_command_gives_adb_15_seconds(
    controller: device.AdbController, adb: _FakeAdb, action: dict[str, Any]
) -> None:
    controller.exe_action(action)
    assert adb.timeouts
    assert all(
        timeout == 15
        for event, timeout in zip(
            (event for event in adb.events if isinstance(event, list)), adb.timeouts, strict=True
        )
        if isinstance(event, list) and "input" in event
    )
    assert all(
        timeout == 30
        for event, timeout in zip(
            (event for event in adb.events if isinstance(event, list)), adb.timeouts, strict=True
        )
        if isinstance(event, list) and "dumpsys" in event
    )


def test_an_input_that_times_out_raises_naming_the_command_and_is_not_sent_again(
    controller: device.AdbController, adb: _FakeAdb
) -> None:
    """A timed-out input may have reached the device, so sending it again could act twice."""
    tap = [*_INPUT, "tap", "10", "20"]
    adb.stalls[tuple(tap[1:])] = 1

    with pytest.raises(subprocess.TimeoutExpired) as raised:
        controller.exe_action({"action": "tap", "x": 10, "y": 20})

    assert (raised.value.cmd, raised.value.timeout) == (tap, 15)
    assert adb.events == [["adb", *_DUMPSYS_WINDOW], tap]


_CLEAR_FIELD = [
    [*_INPUT, "keyboard", "keycombination", "KEYCODE_CTRL_LEFT", "KEYCODE_A"],
    [*_INPUT, "keyboard", "keyevent", "KEYCODE_DEL"],
]


@pytest.mark.parametrize("command_index", [0, 1], ids=["select-all", "delete-selection"])
@pytest.mark.parametrize("failure", ["exit", "timeout"])
def test_clearing_failure_stops_type_without_retrying_input(
    controller: device.DeviceController, adb: _FakeAdb, command_index: int, failure: str
) -> None:
    command = _CLEAR_FIELD[command_index]
    args = tuple(command[1:])
    if failure == "timeout":
        adb.stalls[args] = 1
        error = subprocess.TimeoutExpired
    else:
        adb.replies[args] = (1, b"")
        error = subprocess.CalledProcessError

    with pytest.raises(error) as raised:
        controller.exe_action({"action": "type", "text": "New"})

    assert raised.value.cmd == command
    assert adb.events == _CLEAR_FIELD[: command_index + 1]


@pytest.mark.parametrize(
    ("text", "typed"),
    [
        pytest.param("Hi there!", [[*_INPUT, "text", "'Hi there!'"], 1], id="one quoted call"),
        pytest.param(
            '!?:;"&%$<>{}[]|~^', [[*_INPUT, "text", "'!?:;\"&%$<>{}[]|~^'"], 1], id="punctuation"
        ),
        pytest.param("it's", [[*_INPUT, "text", "'it'\"'\"'s'"], 1], id="single quote"),
        pytest.param(
            "50%sale", [[*_INPUT, "text", "50%"], 1, [*_INPUT, "text", "sale"], 1], id="%s split"
        ),
        pytest.param("%%s", [[*_INPUT, "text", "%%"], 1, [*_INPUT, "text", "s"], 1], id="%%s"),
        pytest.param("", [], id="empty"),
    ],
)
def test_type_clears_the_field_then_types_the_text_quoted_for_the_device_shell(
    controller: device.AdbController,
    adb: _FakeAdb,
    text: str,
    typed: list[list[str] | float],
) -> None:
    controller.exe_action({"action": "type", "text": text})
    assert adb.events == [*_CLEAR_FIELD, *typed]


def test_type_raises_when_adb_cannot_type_a_character(
    controller: device.AdbController, adb: _FakeAdb
) -> None:
    """``input text`` exits non-zero for a character outside the virtual keyboard map."""
    adb.replies[("-s", _UDID, "shell", "input", "text", "'é'")] = (255, b"")
    with pytest.raises(subprocess.CalledProcessError):
        controller.exe_action({"action": "type", "text": "é"})
    assert adb.events == [*_CLEAR_FIELD, [*_INPUT, "text", "'é'"]]


def test_end_logs_that_the_task_finished(
    controller: device.AdbController, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="test"):
        controller.exe_action({"action": "end"})
    assert caplog.messages == ["Task Finished."]


@pytest.mark.parametrize("name", ["double_tap", "open"])
def test_an_unknown_action_raises_value_error_naming_it(
    controller: device.AdbController, adb: _FakeAdb, name: str
) -> None:
    with pytest.raises(ValueError, match=rf"^Unknown device action: '{name}'$"):
        controller.exe_action({"action": name, "x": 1, "y": 2})
    assert adb.events == []


def _ui_dump(name: str) -> bytes:
    """Return ``uiautomator dump`` output saved in ``tests/data``, trailer included."""
    return (Path(__file__).parent / "data" / f"uiautomator_{name}.xml").read_bytes()


# Captured on a Redroid container, Android 16, 720x1280, with Launcher3: the list
# scrolled to its end, Clear all at [72,512][264,608] beside the oldest app's card;
# the list on open, the newest app's card in the middle and no Clear all; the
# recent apps screen with no app, "No recent items"; and the home screen.
_CLEAR_ALL_SHOWN = _ui_dump("recents_api36_redroid_clear_all")
_CARDS_ONLY = _ui_dump("recents_api36_redroid_cards")
_NO_RECENT_APPS = _ui_dump("recents_api36_redroid_empty")
_HOME_SCREEN = _ui_dump("home_api36_redroid")
_OPEN_RECENT_APPS = [*_INPUT, "keyevent", "KEYCODE_APP_SWITCH"]
_LOOK = ["adb", *_UI_DUMP]
_LOOK_AND_SWIPE = [_LOOK, [*_INPUT, "swipe", "108", "1200", "972", "1200", "300"], 1]


@pytest.mark.parametrize(
    ("dump", "tap"),
    [
        pytest.param(_CLEAR_ALL_SHOWN, ["168", "560"], id="Launcher3 id"),
        pytest.param(
            b'<hierarchy><node text="Clear all" '
            b'resource-id="com.google.android.apps.nexuslauncher:id/clear_all" '
            b'bounds="[48,1100][352,1240]" /></hierarchy>',
            ["200", "1170"],
            id="text",
        ),
        pytest.param(
            b'<hierarchy><node bounds="[0,10][100,30]" text="CLEAR ALL" /></hierarchy>',
            ["50", "20"],
            id="text in capitals",
        ),
    ],
)
def test_terminate_all_apps_taps_the_clear_all_button_the_recent_apps_show(
    controller: device.AdbController, adb: _FakeAdb, dump: bytes, tap: list[str]
) -> None:
    adb.replies[_UI_DUMP] = (0, dump)
    controller.terminate_all_apps(
        device.StartupRecovery(controller, 1800, logging.getLogger("test"))
    )
    assert adb.events == [_OPEN_RECENT_APPS, 1, _LOOK, [*_INPUT, "tap", *tap], 1.5]


def test_terminate_all_apps_swipes_toward_clear_all_until_it_appears(
    controller: device.AdbController, adb: _FakeAdb
) -> None:
    """Each swipe moves the list by one app, toward the oldest, which Clear all follows."""
    adb.queued[_UI_DUMP] = [(0, _CARDS_ONLY), (0, _CARDS_ONLY)]
    adb.replies[_UI_DUMP] = (0, _CLEAR_ALL_SHOWN)
    controller.terminate_all_apps(
        device.StartupRecovery(controller, 1800, logging.getLogger("test"))
    )
    assert adb.events == [
        _OPEN_RECENT_APPS,
        1,
        *_LOOK_AND_SWIPE * 2,
        _LOOK,
        [*_INPUT, "tap", "168", "560"],
        1.5,
    ]


def test_terminate_all_apps_presses_home_without_a_warning_when_no_app_is_in_recent_apps(
    controller: device.AdbController, adb: _FakeAdb, caplog: pytest.LogCaptureFixture
) -> None:
    """The recent apps screen shows no app card, so there is nothing to clear."""
    adb.replies[_UI_DUMP] = (0, _NO_RECENT_APPS)
    with caplog.at_level(logging.WARNING, logger="test"):
        controller.terminate_all_apps(
            device.StartupRecovery(controller, 1800, logging.getLogger("test"))
        )
    assert adb.events == [_OPEN_RECENT_APPS, 1, _LOOK, [*_INPUT, "keyevent", "KEYCODE_HOME"], 1]
    assert caplog.messages == []


@pytest.mark.parametrize(
    "dump",
    [
        pytest.param(_CARDS_ONLY, id="app cards"),
        # Only the recent apps screen tells that no app is in it, not a screen without it.
        pytest.param(_HOME_SCREEN, id="home screen"),
    ],
)
def test_terminate_all_apps_presses_home_and_warns_when_clear_all_never_appears(
    controller: device.AdbController,
    adb: _FakeAdb,
    caplog: pytest.LogCaptureFixture,
    dump: bytes,
) -> None:
    adb.replies[_UI_DUMP] = (0, dump)
    with caplog.at_level(logging.WARNING, logger="test"):
        controller.terminate_all_apps(
            device.StartupRecovery(controller, 1800, logging.getLogger("test"))
        )
    assert adb.events == [
        _OPEN_RECENT_APPS,
        1,
        *_LOOK_AND_SWIPE * 10,
        _LOOK,
        [*_INPUT, "keyevent", "KEYCODE_HOME"],
        1,
    ]
    assert caplog.messages == [
        "Clear all did not appear in recent apps after 10 swipes; pressing Home."
    ]


_DEVICES = ("devices",)


def _avd_name(udid: str) -> tuple[str, ...]:
    return ("-s", udid, "emu", "avd", "name")


def test_running_devices_name_an_emulator_by_its_avd_and_a_phone_by_none(adb: _FakeAdb) -> None:
    adb.replies[_DEVICES] = (
        0,
        b"List of devices attached\nemulator-5554\tdevice\nR58M123\tunauthorized\n\n",
    )
    adb.replies[_avd_name("emulator-5554")] = (0, b"Pixel_dup\n")
    adb.replies[_avd_name("R58M123")] = (1, b"")

    running = device.EmulatorManager(logging.getLogger("test")).get_running_avd_list()

    assert running == [
        {"udid": "emulator-5554", "status": "device", "name": "Pixel_dup"},
        {"udid": "R58M123", "status": "unauthorized", "name": None},
    ]
    assert adb.events == [
        ["adb", *_DEVICES],
        ["adb", *_avd_name("emulator-5554")],
        ["adb", *_avd_name("R58M123")],
    ]


def test_no_running_devices_is_an_empty_list(adb: _FakeAdb) -> None:
    adb.replies[_DEVICES] = (0, b"List of devices attached\n\n")
    assert device.EmulatorManager(logging.getLogger("test")).get_running_avd_list() == []
    assert adb.events == [["adb", *_DEVICES]]


_SERVER_VARIABLES = ("ADB_SERVER_SOCKET", "ANDROID_ADB_SERVER_ADDRESS", "ANDROID_ADB_SERVER_PORT")


@pytest.mark.parametrize(
    ("environment", "address"),
    [
        pytest.param(
            {"ADB_SERVER_SOCKET": "tcp:100.64.0.7:5037"}, "tcp:100.64.0.7:5037", id="socket"
        ),
        pytest.param(
            {"ANDROID_ADB_SERVER_ADDRESS": "10.0.2.2", "ANDROID_ADB_SERVER_PORT": "5038"},
            "tcp:10.0.2.2:5038",
            id="host and port",
        ),
        pytest.param({}, "tcp:localhost:5037", id="default"),
    ],
)
def test_running_devices_raise_naming_the_server_adb_cannot_reach(
    adb: _FakeAdb,
    monkeypatch: pytest.MonkeyPatch,
    environment: dict[str, str],
    address: str,
) -> None:
    for name in _SERVER_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    adb.replies[_DEVICES] = (
        1,
        b"adb: failed to check server version: cannot connect to daemon: Connection refused\n",
    )

    with pytest.raises(device.AdbServerUnreachableError) as raised:
        device.EmulatorManager(logging.getLogger("test")).get_running_avd_list()

    assert str(raised.value) == (
        f"adb cannot reach its server at {address}: "
        "adb: failed to check server version: cannot connect to daemon: Connection refused"
    )
    assert adb.events == [["adb", *_DEVICES]]


def test_running_devices_raise_naming_the_server_when_adb_devices_times_out(
    adb: _FakeAdb, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ADB_SERVER_SOCKET", "tcp:100.64.0.7:5037")
    adb.stalls[_DEVICES] = 1

    with pytest.raises(device.AdbServerUnreachableError) as raised:
        device.EmulatorManager(logging.getLogger("test")).get_running_avd_list()

    assert str(raised.value) == (
        "adb cannot reach its server at tcp:100.64.0.7:5037: adb devices timed out after 30 s"
    )
    assert adb.events == [["adb", *_DEVICES]]


def _write_avd(avd_root_dir: Path, *, avd_ini: str) -> None:
    avd_dir = avd_root_dir / "Pixel.avd"
    avd_dir.mkdir(parents=True)
    (avd_root_dir / "Pixel.ini").write_text(avd_ini, encoding="utf-8")
    (avd_dir / "config.ini").write_text(
        "AvdId=Pixel\n"
        "avd.ini.displayname=Pixel\n"
        "abi.type=armeabi-v7a\n"
        "hw.cpu.arch=arm\n"
        "hw.ramSize=2048\n"
        "skin.path=/elsewhere/skins/pixel_7\n",
        encoding="utf-8",
    )
    (avd_dir / "hardware-qemu.ini").write_text(
        "hw.cpu.arch=arm\n"
        "disk.cachePartition.path=/elsewhere/cache.img\n"
        "disk.dataPartition.path=/elsewhere/userdata-qemu.img\n"
        "disk.encryptionKeyPartition.path=/elsewhere/encryptionkey.img\n"
        "android.sdk.root=/elsewhere/sdk\n"
        "android.avd.home=/elsewhere/avd\n"
        "hw.lcd.width=1080\n",
        encoding="utf-8",
    )
    (avd_dir / "userdata-qemu.img").write_bytes(b"\x00data")


@pytest.mark.parametrize("target", ["android-36", "android-37"])
@pytest.mark.parametrize(
    ("system", "sdk_parts", "abi", "cpu_arch"),
    [
        ("Darwin", ("Library", "Android", "sdk"), "arm64-v8a", "arm64"),
        ("Linux", ("Android", "Sdk"), "x86_64", "x86_64"),
        ("Windows", ("AppData", "Local", "Android", "Sdk"), "x86_64", "x86_64"),
    ],
)
def test_duplicate_avd_copies_the_avd_and_points_its_files_at_the_copy_and_the_sdk(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    system: str,
    sdk_parts: tuple[str, ...],
    abi: str,
    cpu_arch: str,
    target: str,
) -> None:
    monkeypatch.setattr(device.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(device.platform, "system", lambda: system)
    root = tmp_path / ".android" / "avd"
    avd_ini = f"avd.ini.encoding=UTF-8\npath=/elsewhere/Pixel.avd\ntarget={target}\n"
    _write_avd(root, avd_ini=avd_ini)
    original_config = (root / "Pixel.avd" / "config.ini").read_text(encoding="utf-8")

    device.EmulatorManager(logging.getLogger("test")).duplicate_avd("Pixel")

    sdk = tmp_path.joinpath(*sdk_parts)
    dup = root / "Pixel_dup.avd"
    via_parent = root / ".." / "avd" / "Pixel_dup.avd"
    assert (root / "Pixel_dup.ini").read_text(encoding="utf-8") == (
        f"avd.ini.encoding=UTF-8\npath={dup}\npath.rel=avd/Pixel_dup.avd\ntarget={target}"
    )
    assert (dup / "config.ini").read_text(encoding="utf-8") == (
        "AvdId=Pixel_dup\n"
        "avd.ini.displayname=Pixel_dup\n"
        f"abi.type={abi}\n"
        f"hw.cpu.arch={cpu_arch}\n"
        "hw.ramSize=2048\n"
        f"skin.path={sdk / 'skins' / 'pixel_7'}\n"
    )
    assert (dup / "hardware-qemu.ini").read_text(encoding="utf-8") == (
        f"hw.cpu.arch={abi}\n"
        f"disk.cachePartition.path={via_parent / 'cache.img'}\n"
        f"{via_parent / 'userdata-qemu.img'}\n"
        f"{via_parent / 'encryptionkey.img'}\n"
        f"{sdk}\n"
        f"{root}\n"
        "hw.lcd.width=1080\n"
    )
    assert (dup / "userdata-qemu.img").read_bytes() == b"\x00data"
    assert (root / "Pixel.ini").read_text(encoding="utf-8") == avd_ini
    assert (root / "Pixel.avd" / "config.ini").read_text(encoding="utf-8") == original_config


@pytest.mark.parametrize(
    "avd_ini",
    [
        pytest.param("avd.ini.encoding=UTF-8\n", id="missing target"),
        pytest.param("target=\n", id="empty target"),
        pytest.param("target= \t\n", id="blank target"),
    ],
)
def test_duplicate_avd_requires_a_nonempty_target_to_create_the_copy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, avd_ini: str
) -> None:
    monkeypatch.setattr(device.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(device.platform, "system", lambda: "Linux")
    root = tmp_path / ".android" / "avd"
    _write_avd(root, avd_ini=avd_ini)
    original_config = (root / "Pixel.avd" / "config.ini").read_text(encoding="utf-8")

    message = re.escape(f"AVD metadata '{root / 'Pixel.ini'}' requires a non-empty target.")
    with pytest.raises(ValueError, match=rf"^{message}$"):
        device.EmulatorManager(logging.getLogger("test")).duplicate_avd("Pixel")

    assert not (root / "Pixel_dup.ini").exists()
    assert not (root / "Pixel_dup.avd").exists()
    assert (root / "Pixel.ini").read_text(encoding="utf-8") == avd_ini
    assert (root / "Pixel.avd" / "config.ini").read_text(encoding="utf-8") == original_config


def test_emulator_manager_needs_the_avd_directory_only_to_duplicate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(device.Path, "home", lambda: tmp_path)
    manager = device.EmulatorManager(logging.getLogger("test"))
    missing = re.escape(f"AVD root directory '{tmp_path / '.android' / 'avd'}' not found.")
    with pytest.raises(FileNotFoundError, match=rf"^{missing}$"):
        manager.duplicate_avd("Pixel")


def test_duplicate_avd_raises_value_error_on_an_unsupported_os(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(device.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(device.platform, "system", lambda: "FreeBSD")
    root = tmp_path / ".android" / "avd"
    _write_avd(root, avd_ini="target=android-36\n")
    with pytest.raises(ValueError, match=r"^Unsupported OS: FreeBSD$"):
        device.EmulatorManager(logging.getLogger("test")).duplicate_avd("Pixel")
    assert not (root / "Pixel_dup.avd").exists()


@pytest.mark.parametrize(
    ("times", "stalls", "timeouts", "expired"),
    [
        ([100, 110, 125], 0, [30, 25, 10], False),
        ([100, 125], 2, [30, 10], True),
        ([100, 135], 1, [30], True),
        ([135], 0, [], True),
    ],
)
def test_observation_reads_and_retries_use_the_remaining_deadline(
    controller: device.AdbController,
    adb: _FakeAdb,
    monkeypatch: pytest.MonkeyPatch,
    times: list[float],
    stalls: int,
    timeouts: list[float],
    expired: bool,
) -> None:
    clock = iter(times)
    # SDK finalizers also read the stdlib clock while these finite readings belong to adb.
    monkeypatch.setattr(device, "time", SimpleNamespace(monotonic=lambda: next(clock)))
    adb.stalls[_UI_DUMP] = stalls

    if expired:
        with pytest.raises(subprocess.TimeoutExpired):
            controller.get_state(deadline=135)
    else:
        controller.get_state(deadline=135)

    assert adb.timeouts == timeouts


def test_recovery_probe_is_bounded_by_its_remaining_time(
    adb: _FakeAdb,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    elapsed: list[float] = []
    timeouts: list[float] = []
    monkeypatch.setattr(device.time, "monotonic", lambda: sum(elapsed))

    def timeout(command: list[str], *, timeout: float, **_kwargs: object) -> NoReturn:
        timeouts.append(timeout)
        elapsed.append(timeout)
        raise subprocess.TimeoutExpired(command, timeout)

    monkeypatch.setattr(subprocess, "run", timeout)
    with pytest.raises(device.DeviceUnreachableError, match="test-device unreachable"):
        device.wait_for_device_state(_Device(), 2, logging.getLogger("test"))

    assert timeouts == [2]
    assert adb.events == []


def _startup_recovery() -> device.StartupRecovery:
    class Controller:
        w = 100
        h = 200

        def __init__(self) -> None:
            self.config: dict[str, object] = {"device": {"udid": "emulator-5554"}}

        def get_state(self, *, deadline: float | None = None) -> dict[str, object]:
            del deadline
            pytest.fail("healthy startup requires no recovery observation")

        def exe_action(self, action: dict[str, object]) -> None:
            del action
            pytest.fail("startup recovery sends no input")

    return device.StartupRecovery(Controller(), 1800, logging.getLogger("test"))


def test_startup_screen_size_timeout_recovers_before_controller_initialization(
    adb: _FakeAdb,
) -> None:
    adb.stalls[_WM_SIZE] = 1
    adb.replies[_WM_SIZE] = (0, b"Physical size: 1080x2400\n")

    controller = device.AdbController(
        {"device": {"udid": _UDID}}, logging.getLogger("test"), unreachable_limit=45
    )

    assert (controller.w, controller.h) == (1080, 2400)
    assert adb.events == [
        ["adb", *_WM_SIZE],
        [*_ADB, "shell", "true"],
        *_STATE_READS,
        ["adb", *_WM_SIZE],
    ]


@pytest.mark.parametrize("delivered", [False, True])
def test_startup_rechecks_recent_apps_after_an_uncertain_app_switch(
    delivered: bool, controller: device.AdbController, adb: _FakeAdb
) -> None:
    adb.stalls[tuple(_OPEN_RECENT_APPS[1:])] = 1
    adb.queued[_UI_DUMP] = [(0, _CARDS_ONLY if delivered else _HOME_SCREEN)]
    adb.replies[_UI_DUMP] = (0, _CLEAR_ALL_SHOWN)

    controller.terminate_all_apps(device.StartupRecovery(controller, 45, logging.getLogger("test")))

    assert adb.events.count(_OPEN_RECENT_APPS) == (1 if delivered else 2)
    assert adb.events.count([*_INPUT, "tap", "168", "560"]) == 1
    assert [*_ADB, "shell", "true"] in adb.events


@pytest.mark.parametrize("delivered", [False, True])
def test_startup_rechecks_clear_all_and_uses_its_current_coordinates_after_a_tap_timeout(
    delivered: bool, controller: device.AdbController, adb: _FakeAdb
) -> None:
    tap = [*_INPUT, "tap", "168", "560"]
    moved = b'<hierarchy><node text="Clear all" bounds="[0,0][200,100]" /></hierarchy>'
    adb.stalls[tuple(tap[1:])] = 1
    adb.queued[_UI_DUMP] = [(0, _CLEAR_ALL_SHOWN), (0, _HOME_SCREEN if delivered else moved)]

    controller.terminate_all_apps(device.StartupRecovery(controller, 45, logging.getLogger("test")))

    assert adb.events.count(tap) == 1
    assert adb.events.count([*_INPUT, "tap", "100", "50"]) == (0 if delivered else 1)
    assert [*_ADB, "shell", "true"] in adb.events


def test_startup_observes_recent_apps_after_an_uncertain_swipe(
    controller: device.AdbController, adb: _FakeAdb
) -> None:
    swipe = [*_INPUT, "swipe", "108", "1200", "972", "1200", "300"]
    adb.stalls[tuple(swipe[1:])] = 1
    adb.queued[_UI_DUMP] = [(0, _CARDS_ONLY)]
    adb.replies[_UI_DUMP] = (0, _CLEAR_ALL_SHOWN)

    controller.terminate_all_apps(device.StartupRecovery(controller, 45, logging.getLogger("test")))

    assert adb.events.count(swipe) == 1
    assert adb.events.count([*_INPUT, "tap", "168", "560"]) == 1
    assert adb.events.count(_LOOK) == 3


def test_repeated_grant_timeouts_share_a_deadline_while_observations_succeed(
    controller: device.AdbController, adb: _FakeAdb, monkeypatch: pytest.MonkeyPatch
) -> None:
    grant = ("-s", _UDID, "shell", "pm", "grant", "com.demo", "android.permission.CAMERA")
    adb.replies[("-s", _UDID, "shell", "dumpsys", "package", "com.demo")] = (
        0,
        b"  requested permissions:\n    android.permission.CAMERA\n",
    )
    adb.stalls[grant] = 10
    elapsed = [0.0]
    monkeypatch.setattr(device.time, "monotonic", lambda: elapsed[0])

    original = device.run_adb

    def run(
        args: Sequence[str],
        *,
        output: Literal["capture", "discard"],
        check: bool,
        timeout: float = 30,
        errors: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        try:
            return original(args, output=output, check=check, timeout=timeout, errors=errors)
        except subprocess.TimeoutExpired as exc:
            elapsed[0] += exc.timeout
            raise

    monkeypatch.setattr(device, "run_adb", run)
    with pytest.raises(device.DeviceUnreachableError, match=r"unreachable for 0\.75 minutes"):
        device.grant_runtime_permissions(
            _UDID, "com.demo", device.StartupRecovery(controller, 45, logging.getLogger("test"))
        )

    grant_timeouts = [
        timeout
        for event, timeout in zip(
            (event for event in adb.events if isinstance(event, list)), adb.timeouts, strict=True
        )
        if event == ["adb", *grant]
    ]
    assert grant_timeouts == [30, 30, 15]
    assert elapsed == [75]


@pytest.mark.usefixtures("adb_on_path")
def test_startup_keeps_one_confirmation_allowance_across_repeated_phase_timeouts(
    no_sleep: list[float], monkeypatch: pytest.MonkeyPatch
) -> None:
    unknown = subprocess.CalledProcessError(
        1, ["adb", "-s", "test-device", "shell", "dumpsys", "window"]
    )
    reads = iter([unknown, {"package": "com.demo"}, unknown])

    class Controller(_Device):
        @override
        def get_state(self, **_kwargs: float | None) -> dict[str, object]:
            result = next(reads)
            if isinstance(result, subprocess.CalledProcessError):
                raise result
            return dict(result)

    monkeypatch.setattr(device.time, "monotonic", lambda: sum(no_sleep))
    monkeypatch.setattr(
        device, "run_adb", lambda args, **_kwargs: subprocess.CompletedProcess(args, 0)
    )
    attempts: list[float | None] = []

    def grant(deadline: float | None, _state: dict[str, object] | None) -> None:
        attempts.append(deadline)
        raise subprocess.TimeoutExpired(["pm", "grant"], 30)

    with pytest.raises(subprocess.CalledProcessError) as caught:
        device.StartupRecovery(Controller(), 90, logging.getLogger("test")).run("permission", grant)

    assert caught.value is unknown
    assert attempts == [None, 90]
    assert no_sleep == [30]


def test_startup_grants_resume_at_the_timed_out_permission(
    controller: device.AdbController, adb: _FakeAdb
) -> None:
    shell = ("-s", _UDID, "shell")
    adb.replies[(*shell, "dumpsys", "package", "com.example.app")] = (
        0,
        _DUMPSYS_PACKAGE.read_bytes(),
    )
    grants = [
        ["adb", *shell, "pm", "grant", "com.example.app", permission]
        for permission in _REQUESTED_PERMISSIONS
    ]
    adb.stalls[tuple(grants[1][1:])] = 1

    device.grant_runtime_permissions(
        _UDID, "com.example.app", device.StartupRecovery(controller, 45, logging.getLogger("test"))
    )

    assert [event for event in adb.events if isinstance(event, list) and "grant" in event] == [
        grants[0],
        grants[1],
        *grants[1:],
    ]


@pytest.mark.parametrize(
    ("focus", "window", "package", "surface"),
    [
        (
            "NotificationShade",
            "ty=NOTIFICATION_SHADE package=com.android.systemui",
            "com.android.systemui",
            "notification_shade",
        ),
        (
            "com.android.permissioncontroller/.permission.ui.GrantPermissionsActivity",
            "ty=BASE_APPLICATION package=com.android.permissioncontroller",
            "com.android.permissioncontroller",
            "permission",
        ),
        ("InputMethod", "ty=INPUT_METHOD package=com.android.inputmethod.latin", "com.demo", None),
        (
            "com.android.inputmethod.latin/.settings.SettingsActivity",
            "ty=BASE_APPLICATION package=com.android.inputmethod.latin",
            "com.android.inputmethod.latin",
            "ime_settings",
        ),
        (
            "com.android.settings/.Settings",
            "ty=BASE_APPLICATION package=com.android.settings",
            "com.android.settings",
            None,
        ),
    ],
    ids=["shade", "permission", "attached keyboard", "keyboard settings", "Settings app"],
)
def test_capture_keeps_system_window_ownership_separate_from_the_focused_app(
    controller: device.AdbController,
    adb: _FakeAdb,
    focus: str,
    window: str,
    package: str,
    surface: str | None,
) -> None:
    adb.replies[_DUMPSYS_WINDOW] = (
        0,
        (
            f"  mCurrentFocus=Window{{focus u0 {focus}}}\n"
            "  mFocusedApp=ActivityRecord{app u0 com.demo/.Home t1}\n"
            f"  Window #0 Window{{focus u0 {focus}}}:\n"
            f"    mDisplayId=0 mOwnerUid=1000 {window}\n"
            "    mHasSurface=true isOnScreen=true isVisible=true\n"
        ).encode(),
    )

    state = controller.get_state()

    assert state["package"] == package
    assert state["underlying_package"] == "com.demo"
    assert state["system_surface"] == surface


def test_clipboard_feedback_retains_app_focus_and_records_its_touch_regions(
    controller: device.AdbController, adb: _FakeAdb
) -> None:
    adb.replies[_DUMPSYS_WINDOW] = (
        0,
        (
            b"  mCurrentFocus=Window{app u0 com.demo/.Home}\n"
            b"  mFocusedApp=ActivityRecord{app u0 com.demo/.Home t1}\n"
            b"  Window #0 Window{clip u0 ClipboardOverlay}:\n"
            b"    mDisplayId=0 package=com.android.systemui\n"
            b"    mAttrs={ty=SCREENSHOT fl=NOT_FOCUSABLE NOT_TOUCH_MODAL}\n"
            b"    mTouchableInsets=3\n"
            b"    touchable region=SkRegion((-16,1040,176,1180)(200,1050,340,1180))\n"
            b"    Frames: frame=[0,0][720,1280]\n"
            b"    mHasSurface=true isVisible=true\n"
        ),
    )

    state = controller.get_state()

    assert state["package"] == "com.demo"
    assert state["system_surface"] is None
    assert state["system_overlays"] == [[-16, 1040, 176, 1180], [200, 1050, 340, 1180]]


@pytest.mark.parametrize(
    "package",
    ["com.android.systemui", "com.android.permissioncontroller", "com.android.inputmethod.latin"],
)
def test_an_independent_system_destination_is_outside_the_app(package: str) -> None:
    assert device.is_outside_app(package, "com.demo")


@pytest.mark.parametrize("geometry", ["    touchable region=SkRegion((0,0,100,100))\n", ""])
def test_dispatch_checks_new_system_feedback_before_sending_coordinates(
    controller: device.AdbController, adb: _FakeAdb, geometry: str
) -> None:
    from android_app_graph.system_surfaces import SystemOverlayActionError

    adb.replies[_DUMPSYS_WINDOW] = (
        0,
        (
            "  Window #0 Window{clip u0 ClipboardOverlay}:\n"
            "    mDisplayId=0 package=com.android.systemui ty=SCREENSHOT\n"
            + geometry
            + "    mHasSurface=true isVisible=true\n"
        ).encode(),
    )
    with pytest.raises(SystemOverlayActionError):
        controller.exe_action({"action": "tap", "x": 10, "y": 20})
    assert adb.events == [["adb", *_DUMPSYS_WINDOW]]


@pytest.mark.parametrize("stalls", [1, 2])
def test_coordinate_dispatch_retries_its_window_read_once_before_any_input(
    controller: device.AdbController, adb: _FakeAdb, stalls: int
) -> None:
    adb.stalls[_DUMPSYS_WINDOW] = stalls
    action = {"action": "tap", "x": 10, "y": 20}
    if stalls == 2:
        with pytest.raises(subprocess.TimeoutExpired):
            controller.exe_action(action)
        assert adb.events == [["adb", *_DUMPSYS_WINDOW]] * 2
    else:
        controller.exe_action(action)
        assert adb.events == [
            ["adb", *_DUMPSYS_WINDOW],
            ["adb", *_DUMPSYS_WINDOW],
            ["adb", "-s", _UDID, "shell", "input", "tap", "10", "20"],
        ]


@pytest.mark.parametrize("first", [{"action": "tap", "x": 100, "y": 200}, {"action": "back"}])
def test_replay_checks_each_inputs_current_regions_after_a_rotating_action(
    controller: device.AdbController, adb: _FakeAdb, first: dict[str, object]
) -> None:
    adb.replies[_DUMPSYS_WINDOW] = (0, _BUTTON_NAVIGATION.encode())
    controller.get_state()
    controller.exe_action(first)
    landscape = (
        "WindowInsetsStateController\nInsetsState\nmDisplayFrame=Rect(0, 0 - 1280, 720)\n"
        "InsetsSource type=tappableElement frame=[1184,0][1280,720] visible=true\n"
    )
    adb.replies[_DUMPSYS_WINDOW] = (0, landscape.encode())
    controller.exe_action({"action": "tap", "x": 1100, "y": 300})
    with pytest.raises(device.InputRejectedError, match="system touch region"):
        controller.exe_action({"action": "tap", "x": 1250, "y": 300})
    with pytest.raises(device.InputRejectedError, match="outside the display"):
        controller.exe_action({"action": "tap", "x": 100, "y": 900})
    assert [*_INPUT, "tap", "1100", "300"] in adb.events


@pytest.mark.parametrize(
    ("action", "error"),
    [
        ({"action": "tap", "x": 50, "y": 50}, "clipboard"),
        ({"action": "tap", "x": 360, "y": 1250}, "navigation"),
        ({"action": "tap", "x": 360, "y": 500}, None),
    ],
)
def test_dispatch_applies_feedback_and_navigation_policy_from_one_window_read(
    controller: device.AdbController,
    adb: _FakeAdb,
    action: dict[str, object],
    error: str | None,
) -> None:
    from android_app_graph.system_surfaces import SystemOverlayActionError

    window = (
        "  Window #0 Window{clip u0 ClipboardOverlay}:\n"
        "    mDisplayId=0 package=com.android.systemui ty=SCREENSHOT\n"
        "    touchable region=SkRegion((0,0,100,100))\n"
        "    mHasSurface=true isVisible=true\n" + _BUTTON_NAVIGATION
    )
    adb.replies[_DUMPSYS_WINDOW] = (0, window.encode())
    if error is None:
        controller.exe_action(action)
        assert adb.events == [
            ["adb", *_DUMPSYS_WINDOW],
            [*_INPUT, "tap", "360", "500"],
        ]
    else:
        failure = SystemOverlayActionError if error == "clipboard" else device.InputRejectedError
        with pytest.raises(failure):
            controller.exe_action(action)
        assert adb.events == [["adb", *_DUMPSYS_WINDOW]]
