"""Tests for android_app_graph.device, with adb, sleeps and the home directory faked."""

from __future__ import annotations

import base64
import logging
import re
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any, NoReturn

import pytest

from android_app_graph import device
from android_app_graph.ui_tree import IDLE_FAILURE, xml_document

_FORCE_STOP = ["adb", "-s", "test-device", "shell", "am", "force-stop", "com.demo"]
_PM = ["adb", "-s", "test-device", "shell", "pm"]
_LAUNCHER_INTENT = ["-a", "android.intent.action.MAIN", "-c", "android.intent.category.LAUNCHER"]
_RESOLVE_LAUNCHER = [*_PM, "resolve-activity", "--brief", *_LAUNCHER_INTENT, "com.demo"]
_QUERY_LAUNCHERS = [*_PM, "query-activities", "--brief", *_LAUNCHER_INTENT, "com.demo"]
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


class _Device:
    w = 100
    h = 200

    def __init__(self) -> None:
        self.config: dict[str, Any] = {"device": {"udid": "test-device"}}

    def get_state(self) -> dict[str, Any]:
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
        "com.demo has several launcher activities; restarting it through com.demo/.Home, "
        "not com.demo/.Widgets. Set launch_activity to choose another."
    )
    assert caplog.messages == [warning]


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


@pytest.mark.parametrize(
    ("returncode", "stdout", "stderr", "error"),
    [
        (1, _STARTING_GONE, _GONE, _GONE_ON_ONE_LINE),
        # An adb that reports no exit status prints both streams to stdout.
        (0, _STARTING_GONE + _GONE, "", f"{_STARTING_GONE.strip()} {_GONE_ON_ONE_LINE}"),
    ],
    ids=["exit status", "error line"],
)
@pytest.mark.usefixtures("no_sleep")
def test_restart_app_launches_the_package_when_am_start_cannot_start_the_activity(
    returncode: int,
    stdout: str,
    stderr: str,
    error: str,
    monkeypatch: pytest.MonkeyPatch,
    device_log: list[object],
    caplog: pytest.LogCaptureFixture,
) -> None:
    start = _clear_task_start("com.demo/.Gone")

    def run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        device_log.append(command)
        if command == start:
            return subprocess.CompletedProcess(command, returncode, stdout=stdout, stderr=stderr)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(device.subprocess, "run", run)
    app = device.AppLaunch("demo", "com.demo", "com.demo/.Gone")
    with caplog.at_level(logging.WARNING, logger="test"):
        device.restart_app(_Device(), app, logging.getLogger("test"))
    assert device_log == [_FORCE_STOP, start, _PACKAGE_LAUNCH]
    assert caplog.messages == [
        f"Activity launch failed for demo; falling back to package launch. error={error}"
    ]


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


_ACTIVITY_LAUNCH = ["adb", "-s", "test-device", "shell", "am", "start", "-n", "com.demo/.Main"]


@pytest.mark.parametrize(
    ("returncode", "launches"),
    [(0, [_ACTIVITY_LAUNCH]), (1, [_ACTIVITY_LAUNCH, _PACKAGE_LAUNCH])],
    ids=["started", "refused"],
)
def test_launch_app_starts_the_launch_activity_or_else_the_package_without_clearing_the_task(
    returncode: int,
    launches: list[list[str]],
    monkeypatch: pytest.MonkeyPatch,
    device_log: list[object],
) -> None:
    def run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        device_log.append(command)
        refused = command == _ACTIVITY_LAUNCH and returncode
        return subprocess.CompletedProcess(command, returncode if refused else 0, "", "")

    monkeypatch.setattr(device.subprocess, "run", run)
    app = {"name": "demo", "package_name": "com.demo", "launch_activity": "com.demo/.Main"}
    device.launch_app({"device": {"udid": "test-device"}}, app, logging.getLogger("test"))
    assert device_log == launches


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
                "timeout": None,
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
        "Soft keyboard is visible; a text field is focused and ready for typing."
    )


def test_keyboard_status_when_the_keyboard_is_hidden() -> None:
    assert device.keyboard_status("") == (
        "No OS keyboard signal detected. Still inspect the screenshot: "
        "a bottom input/keyboard bar can mean a text field is active."
    )


@pytest.mark.parametrize(
    "package",
    [
        "com.google.android.permissioncontroller",
        "com.android.permissioncontroller",
        "org.example.inputmethod.pinyin",
        "com.example.keyboard",
    ],
)
def test_a_system_overlay_package_is_recognised(package: str) -> None:
    assert device.is_system_overlay(package)


@pytest.mark.parametrize("package", ["com.amazon.primevideo", "com.example.timetracker"])
def test_an_app_package_is_not_a_system_overlay(package: str) -> None:
    """Both contain "ime" without being keyboards."""
    assert not device.is_system_overlay(package)


@pytest.mark.parametrize(
    ("package", "outside"),
    [
        ("com.android.chrome", True),
        ("com.demo", False),
        ("com.google.android.inputmethod.latin", False),
        ("com.google.android.permissioncontroller", False),
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
        device.grant_runtime_permissions("emulator-5554", "com.example.app")

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
_API_LEVEL = ("-s", _UDID, "shell", "getprop", "ro.build.version.sdk")
_UI_DUMP = ("-s", _UDID, "exec-out", "uiautomator dump --windows /dev/stdout")
_SCREENCAP = ("-s", _UDID, "exec-out", "screencap -p")
_DUMPSYS_WINDOW = ("-s", _UDID, "shell", "dumpsys", "window")


class _FakeAdb:
    """Stands in for ``subprocess.run`` and ``time.sleep``.

    ``events`` records each adb argv and each sleep in the order they happened, and
    ``timeouts`` the timeout of each adb call. ``replies`` maps the arguments after
    ``adb`` to an exit status and raw stdout; ``queued`` maps them to replies that the
    next calls take in turn, before ``replies``. Any other command exits 0 with no
    output. Stdout stays bytes unless the caller asks for text, which must name its
    encoding: the host's locale differs between machines, and Android writes UTF-8.
    ``stalls`` maps the arguments after ``adb`` to how many of their next calls stall
    until their timeout expires, before any reply; a stalled call without a timeout
    would never return.
    """

    def __init__(self) -> None:
        self.events: list[list[str] | float] = []
        self.timeouts: list[float | None] = []
        self.replies: dict[tuple[str, ...], tuple[int, bytes]] = {}
        self.stalls: dict[tuple[str, ...], int] = {}
        self.queued: dict[tuple[str, ...], list[tuple[int, bytes]]] = {}

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
        self.events.append(command)
        self.timeouts.append(timeout)
        args = tuple(command[1:])
        if self.stalls.get(args, 0) > 0:
            if timeout is None:
                msg = f"a stalled call without a timeout never returns: {command}"
                raise AssertionError(msg)
            self.stalls[args] -= 1
            raise subprocess.TimeoutExpired(command, timeout)
        queue = self.queued.get(args)
        returncode, stdout = queue.pop(0) if queue else self.replies.get(args, (0, b""))
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
    adb.replies[_API_LEVEL] = (0, b"37\n")
    adb.replies[_WM_SIZE] = (0, b"Physical size: 1080x2400\n")
    built = device.AdbController({"device": {"udid": _UDID}}, logging.getLogger("test"))
    adb.events.clear()
    adb.timeouts.clear()
    return built


@pytest.mark.parametrize("api_level", [b"30\n", b"37\n"])
def test_adb_controller_reads_the_screen_size_of_the_configured_device(
    adb: _FakeAdb, api_level: bytes
) -> None:
    adb.replies[_API_LEVEL] = (0, api_level)
    adb.replies[_WM_SIZE] = (0, b"Physical size: 1080x2400\n")
    config = {"device": {"udid": _UDID}}

    controller = device.AdbController(config, logging.getLogger("test"))

    assert (controller.w, controller.h) == (1080, 2400)
    assert controller.config is config
    assert adb.events == [["adb", *_API_LEVEL], ["adb", *_WM_SIZE]]


def test_adb_controller_raises_when_wm_size_reports_no_size(adb: _FakeAdb) -> None:
    adb.replies[_API_LEVEL] = (0, b"37\n")
    adb.replies[_WM_SIZE] = (0, b"Physical size: unknown\n")
    with pytest.raises(ValueError, match=r"^Failed to get screen size$"):
        device.AdbController({"device": {"udid": _UDID}}, logging.getLogger("test"))


def test_building_the_controller_below_api_30_raises_naming_the_level(adb: _FakeAdb) -> None:
    adb.replies[_API_LEVEL] = (0, b"29\n")

    with pytest.raises(RuntimeError, match=r"^emulator-5554 runs API level 29; .* API level 30"):
        device.AdbController({"device": {"udid": _UDID}}, logging.getLogger("test"))

    assert adb.events == [["adb", *_API_LEVEL]]


@pytest.mark.parametrize(
    "api_level", [b"", b"thirty\n", "\u0663\u0660\n".encode()], ids=["empty", "word", "digits"]
)
def test_building_the_controller_raises_when_the_api_level_does_not_read(
    adb: _FakeAdb, api_level: bytes
) -> None:
    adb.replies[_API_LEVEL] = (0, api_level)

    with pytest.raises(ValueError, match=r"^Failed to read the API level of emulator-5554"):
        device.AdbController({"device": {"udid": _UDID}}, logging.getLogger("test"))


# PNG's signature contains CR LF, so any newline translation would change the base64.
_PNG = b"\x89PNG\r\n\x1a\n"


def _uiautomator(name: str) -> str:
    return (Path(__file__).parent / "data" / f"uiautomator_{name}.txt").read_text(encoding="utf-8")


# A trimmed `uiautomator dump --windows` of Settings' home screen, trailing line included,
# and the same screen one tap later, with a node's text changed.
_SCREEN = _uiautomator("windows_plain_list")
_LATER = _SCREEN.replace('text="Apps"', 'text="Apps and notifications"')


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
def test_get_state_returns_the_tree_screenshot_package_and_activity(
    controller: device.AdbController,
    adb: _FakeAdb,
    dumpsys_window: str,
    package: str,
    activity: str,
) -> None:
    adb.replies[_UI_DUMP] = (0, _SCREEN.encode())
    adb.replies[_SCREENCAP] = (0, _PNG)
    adb.replies[_DUMPSYS_WINDOW] = (0, dumpsys_window.encode())

    assert controller.get_state() == {
        "xml": xml_document(_SCREEN),
        "ui_tree_error": None,
        "ui_tree_retakes": 0,
        "ui_tree_settled": True,
        "screenshot": base64.b64encode(_PNG).decode("ascii"),
        "package": package,
        "activity": activity,
        "status_bar": None,
    }
    assert adb.events == [
        ["adb", *_UI_DUMP],
        ["adb", *_SCREENCAP],
        ["adb", *_UI_DUMP],
        ["adb", *_DUMPSYS_WINDOW],
    ]


def _dumpsys_window(name: str) -> str:
    return (Path(__file__).parent / "data" / f"dumpsys_window_{name}.txt").read_text(
        encoding="utf-8"
    )


def test_get_state_reads_the_status_bar_from_the_dump_it_reads_the_activity_from(
    controller: device.AdbController, adb: _FakeAdb
) -> None:
    adb.replies[_DUMPSYS_WINDOW] = (0, _dumpsys_window("api37_pixel_9").encode())

    state = controller.get_state()

    assert (state["package"], state["status_bar"]) == ("com.android.settings", [0, 0, 1080, 142])
    assert adb.events.count(["adb", *_DUMPSYS_WINDOW]) == 1


# Trimmed from `adb shell dumpsys window`: pixel_9 and pixel_8a on API 37 emulators.
# api33 is hand-written in the layout Android 13's InsetsSource.dump() prints.
@pytest.mark.parametrize(
    ("fixture", "frame"),
    [
        ("api37_pixel_9", [0, 0, 1080, 142]),
        ("api37_pixel_8a", [0, 0, 1080, 121]),
        ("api33", [0, 0, 1080, 121]),
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
            "        InsetsSource type=ITYPE_STATUS_BAR frame=[0,0][1080,121] visible=false\n",
            id="hidden before API 34",
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


def _state_of(
    adb: _FakeAdb, controller: device.AdbController, *dumps: tuple[int, bytes]
) -> dict[str, Any]:
    """Read the state while the dumps reply ``dumps`` in turn and each screenshot is the
    next of ``b"shot-0"``, ``b"shot-1"``, ..."""
    adb.queued[_UI_DUMP] = list(dumps)
    adb.queued[_SCREENCAP] = [(0, f"shot-{n}".encode()) for n in range(len(dumps))]
    return controller.get_state()


def _reads(adb: _FakeAdb) -> list[str]:
    names: dict[tuple[str, ...], str] = {
        _UI_DUMP: "dump",
        _SCREENCAP: "screencap",
        _DUMPSYS_WINDOW: "dumpsys",
    }
    return [names[tuple(event[1:])] for event in adb.events if isinstance(event, list)]


def _shot(n: int) -> str:
    return base64.b64encode(f"shot-{n}".encode()).decode("ascii")


@pytest.mark.parametrize(
    ("dumps", "reads", "kept", "retakes", "settled"),
    [
        pytest.param(
            [_SCREEN, _LATER, _LATER],
            ["dump", "screencap", "dump", "screencap", "dump", "dumpsys"],
            (_LATER, 1),
            1,
            True,
            id="settled after one retake",
        ),
        pytest.param(
            [_SCREEN, _LATER, _SCREEN, _LATER],
            ["dump", "screencap", "dump", "screencap", "dump", "screencap", "dump", "dumpsys"],
            (_SCREEN, 2),
            2,
            False,
            id="never settled",
        ),
    ],
)
def test_the_screenshot_and_the_dump_are_retaken_while_the_two_dumps_differ(
    controller: device.AdbController,
    adb: _FakeAdb,
    dumps: list[str],
    reads: list[str],
    kept: tuple[str, int],
    retakes: int,
    settled: bool,
) -> None:
    """The last dump becomes the first of the new pair, and the dump kept is the one
    taken right before the kept screenshot."""
    state = _state_of(adb, controller, *((0, dump.encode()) for dump in dumps))

    assert _reads(adb) == reads
    xml, shot = kept
    assert (state["xml"], state["screenshot"]) == (xml_document(xml), _shot(shot))
    assert (state["ui_tree_error"], state["ui_tree_retakes"], state["ui_tree_settled"]) == (
        None,
        retakes,
        settled,
    )


def test_a_change_in_the_status_bar_alone_needs_no_retake(
    controller: device.AdbController, adb: _FakeAdb
) -> None:
    next_minute = _SCREEN.replace('text="1:33"', 'text="1:34"')

    state = _state_of(adb, controller, (0, _SCREEN.encode()), (0, next_minute.encode()))

    assert _reads(adb) == ["dump", "screencap", "dump", "dumpsys"]
    assert (state["ui_tree_retakes"], state["ui_tree_settled"]) == (0, True)


_UNUSABLE = {
    "adb failed": ((1, b""), "adb exited with status 1"),
    "empty": ((0, b""), "empty output"),
    "not utf-8": ((0, b"<displays>\xff</displays>"), "output is not UTF-8"),
    "default dump": (
        (0, _uiautomator("default_plain_list").encode()),
        "root is <hierarchy>, not <displays>",
    ),
    "truncated": (
        (0, _SCREEN[:200].encode()),
        "XML does not parse: unclosed token: line 1, column 82",
    ),
}


@pytest.mark.parametrize(
    "unusable", [pytest.param(reply, id=name) for name, (reply, _) in _UNUSABLE.items()]
)
def test_an_unusable_dump_is_retried_once(
    controller: device.AdbController, adb: _FakeAdb, unusable: tuple[int, bytes]
) -> None:
    screen = (0, _SCREEN.encode())

    state = _state_of(adb, controller, unusable, screen, screen)

    assert _reads(adb) == ["dump", "dump", "screencap", "dump", "dumpsys"]
    assert (state["xml"], state["ui_tree_error"], state["ui_tree_settled"]) == (
        xml_document(_SCREEN),
        None,
        True,
    )


@pytest.mark.parametrize(
    ("unusable", "reason"),
    [pytest.param(reply, reason, id=name) for name, (reply, reason) in _UNUSABLE.items()],
)
def test_a_dump_still_unusable_leaves_no_tree_with_the_reason_and_takes_the_screenshot(
    controller: device.AdbController, adb: _FakeAdb, unusable: tuple[int, bytes], reason: str
) -> None:
    state = _state_of(adb, controller, unusable, unusable)

    assert _reads(adb) == ["dump", "dump", "screencap", "dumpsys"]
    assert (state["xml"], state["ui_tree_error"]) == (None, reason)
    assert (state["ui_tree_retakes"], state["ui_tree_settled"]) == (0, False)
    assert state["screenshot"] == _shot(0)


def test_an_idle_failure_is_not_retried(controller: device.AdbController, adb: _FakeAdb) -> None:
    """The failed dump has already waited ten seconds for a quiet second."""
    idle = (0, _uiautomator("windows_never_idle").encode())

    state = _state_of(adb, controller, idle, (0, _SCREEN.encode()))

    assert _reads(adb) == ["dump", "screencap", "dumpsys"]
    assert (state["xml"], state["ui_tree_error"], state["screenshot"]) == (
        None,
        IDLE_FAILURE,
        _shot(0),
    )


def test_an_unusable_second_dump_keeps_the_first_unsettled(
    controller: device.AdbController, adb: _FakeAdb
) -> None:
    state = _state_of(adb, controller, (0, _SCREEN.encode()), (0, b""), (0, b""))

    assert _reads(adb) == ["dump", "screencap", "dump", "dump", "dumpsys"]
    assert (state["xml"], state["ui_tree_error"]) == (xml_document(_SCREEN), None)
    assert (state["ui_tree_retakes"], state["ui_tree_settled"]) == (0, False)


@pytest.mark.parametrize("failing", [_SCREENCAP, _DUMPSYS_WINDOW])
def test_get_state_raises_when_a_read_fails(
    controller: device.AdbController, adb: _FakeAdb, failing: tuple[str, ...]
) -> None:
    adb.replies[failing] = (1, b"")
    with pytest.raises(subprocess.CalledProcessError):
        controller.get_state()


_STATE_READS = [
    ["adb", *_UI_DUMP],
    ["adb", *_SCREENCAP],
    ["adb", *_UI_DUMP],
    ["adb", *_DUMPSYS_WINDOW],
]
_EACH_STATE_READ = [
    pytest.param(
        _UI_DUMP, "-s emulator-5554 exec-out 'uiautomator dump --windows /dev/stdout'", id="dump"
    ),
    pytest.param(_SCREENCAP, "-s emulator-5554 exec-out 'screencap -p'", id="screenshot"),
    pytest.param(_DUMPSYS_WINDOW, "-s emulator-5554 shell dumpsys window", id="dumpsys window"),
]


def test_get_state_gives_each_read_a_30_second_timeout(
    controller: device.AdbController, adb: _FakeAdb
) -> None:
    adb.replies[_UI_DUMP] = (0, _SCREEN.encode())
    controller.get_state()
    assert adb.events == _STATE_READS
    assert adb.timeouts == [30, 30, 30, 30]


@pytest.mark.parametrize(("stalled", "command"), _EACH_STATE_READ)
def test_get_state_runs_a_read_that_times_out_once_more_and_logs_it(
    controller: device.AdbController,
    adb: _FakeAdb,
    caplog: pytest.LogCaptureFixture,
    stalled: tuple[str, ...],
    command: str,
) -> None:
    adb.replies[_UI_DUMP] = (0, _SCREEN.encode())
    adb.replies[_SCREENCAP] = (0, _PNG)
    adb.replies[_DUMPSYS_WINDOW] = (0, b"  mCurrentFocus=Window{1 u0 com.demo/.Main}\n")
    adb.stalls[stalled] = 1

    with caplog.at_level(logging.WARNING, logger="test"):
        state = controller.get_state()

    assert state == {
        "xml": xml_document(_SCREEN),
        "ui_tree_error": None,
        "ui_tree_retakes": 0,
        "ui_tree_settled": True,
        "screenshot": base64.b64encode(_PNG).decode("ascii"),
        "package": "com.demo",
        "activity": "com.demo.Main",
        "status_bar": None,
    }
    retried = _STATE_READS.index(["adb", *stalled])
    assert adb.events == [*_STATE_READS[: retried + 1], *_STATE_READS[retried:]]
    assert adb.timeouts == [30, 30, 30, 30, 30]
    assert caplog.messages == [f"adb {command} timed out after 30 s; retrying once"]


@pytest.mark.parametrize("stalled", [_SCREENCAP, _DUMPSYS_WINDOW])
def test_get_state_raises_naming_the_command_when_a_read_times_out_twice(
    controller: device.AdbController, adb: _FakeAdb, stalled: tuple[str, ...]
) -> None:
    adb.replies[_UI_DUMP] = (0, _SCREEN.encode())
    adb.stalls[stalled] = 2

    with pytest.raises(subprocess.TimeoutExpired) as raised:
        controller.get_state()

    assert (raised.value.cmd, raised.value.timeout) == (["adb", *stalled], 30)
    stopped = _STATE_READS.index(["adb", *stalled])
    assert adb.events == [*_STATE_READS[: stopped + 1], ["adb", *stalled]]


def test_a_dump_that_times_out_twice_leaves_no_tree_with_the_reason_and_takes_the_screenshot(
    controller: device.AdbController, adb: _FakeAdb, caplog: pytest.LogCaptureFixture
) -> None:
    """The read has already run the dump a second time, so no third dump is tried, even
    one that would reply, and without a first dump there is no pair to confirm."""
    adb.replies[_UI_DUMP] = (0, _SCREEN.encode())
    adb.replies[_SCREENCAP] = (0, _PNG)
    adb.stalls[_UI_DUMP] = 2

    with caplog.at_level(logging.WARNING, logger="test"):
        state = controller.get_state()

    assert _reads(adb) == ["dump", "dump", "screencap", "dumpsys"]
    assert adb.timeouts == [30, 30, 30, 30]
    assert (state["xml"], state["ui_tree_error"]) == (None, "adb timed out twice")
    assert (state["ui_tree_retakes"], state["ui_tree_settled"]) == (0, False)
    assert state["screenshot"] == base64.b64encode(_PNG).decode("ascii")
    dump = "-s emulator-5554 exec-out 'uiautomator dump --windows /dev/stdout'"
    assert caplog.messages == [f"adb {dump} timed out after 30 s; retrying once"]


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
    assert adb.events == events


_CLEAR_FIELD = [[*_INPUT, "keyevent", "KEYCODE_DEL"], 0.2] * 20


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


def test_terminate_all_apps_opens_recents_swipes_and_taps_at_scaled_coordinates(
    controller: device.AdbController, adb: _FakeAdb
) -> None:
    controller.terminate_all_apps()
    assert adb.events == [
        [*_INPUT, "keyevent", "KEYCODE_APP_SWITCH"],
        1,
        [*_INPUT, "swipe", "108", "1200", "972", "1200", "100"],
        1,
        [*_INPUT, "tap", "216", "1152"],
        1.5,
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


def test_running_devices_are_none_when_adb_devices_fails(
    adb: _FakeAdb, caplog: pytest.LogCaptureFixture
) -> None:
    adb.replies[_DEVICES] = (1, b"")
    with caplog.at_level(logging.ERROR, logger="test"):
        running = device.EmulatorManager(logging.getLogger("test")).get_running_avd_list()
    assert running is None
    assert caplog.messages == ["Failed to get running AVD list: "]


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
) -> None:
    monkeypatch.setattr(device.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(device.platform, "system", lambda: system)
    root = tmp_path / ".android" / "avd"
    avd_ini = "avd.ini.encoding=UTF-8\npath=/elsewhere/Pixel.avd\ntarget=android-34\n"
    _write_avd(root, avd_ini=avd_ini)
    original_config = (root / "Pixel.avd" / "config.ini").read_text(encoding="utf-8")

    device.EmulatorManager(logging.getLogger("test")).duplicate_avd("Pixel")

    sdk = tmp_path.joinpath(*sdk_parts)
    dup = root / "Pixel_dup.avd"
    via_parent = root / ".." / "avd" / "Pixel_dup.avd"
    assert (root / "Pixel_dup.ini").read_text(encoding="utf-8") == (
        f"avd.ini.encoding=UTF-8\npath={dup}\npath.rel=avd/Pixel_dup.avd\ntarget=android-34"
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


def test_duplicate_avd_targets_android_35_when_the_avd_names_no_target(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(device.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(device.platform, "system", lambda: "Linux")
    root = tmp_path / ".android" / "avd"
    _write_avd(root, avd_ini="avd.ini.encoding=UTF-8\n")

    device.EmulatorManager(logging.getLogger("test")).duplicate_avd("Pixel")

    assert (root / "Pixel_dup.ini").read_text(encoding="utf-8").endswith("\ntarget=android-35")


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
    _write_avd(root, avd_ini="target=android-34\n")
    with pytest.raises(ValueError, match=r"^Unsupported OS: FreeBSD$"):
        device.EmulatorManager(logging.getLogger("test")).duplicate_avd("Pixel")
    assert not (root / "Pixel_dup.avd").exists()
