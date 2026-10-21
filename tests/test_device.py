"""Tests for android_app_graph.device, with adb, sleeps and the home directory faked."""

from __future__ import annotations

import base64
import logging
import re
import subprocess
from pathlib import Path
from typing import Any, NoReturn

import pytest

from android_app_graph import device

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


def test_restart_app_force_stops_then_launches_and_waits_for_the_screen(
    device_log: list[object], no_sleep: list[float]
) -> None:
    device.restart_app(_Device(), "demo", "com.demo", None, logging.getLogger("test"))
    assert device_log == [_FORCE_STOP, _PACKAGE_LAUNCH]
    assert no_sleep == [4]


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
    with pytest.raises(device.ForceStopRefusedError):
        device.restart_app(_Device(), "demo", "com.demo", None, logging.getLogger("test"))
    assert commands == [_FORCE_STOP]
    assert no_sleep == []


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
def test_soft_keyboard_hint_when_the_keyboard_is_shown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        device.subprocess, "run", lambda *_a, **_k: _CompletedProcess("mInputShown=true")
    )
    hint = device.soft_keyboard_hint()
    assert "soft keyboard is currently visible" in hint


@pytest.mark.usefixtures("adb_on_path")
def test_soft_keyboard_hint_when_the_keyboard_is_hidden(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(device.subprocess, "run", lambda *_a, **_k: _CompletedProcess(""))
    assert device.soft_keyboard_hint() == ""


@pytest.mark.usefixtures("adb_on_path")
def test_soft_keyboard_hint_survives_a_missing_adb(monkeypatch: pytest.MonkeyPatch) -> None:
    def _no_adb(*_args: object, **_kwargs: object) -> _CompletedProcess:
        msg = "adb"
        raise FileNotFoundError(msg)

    monkeypatch.setattr(device.subprocess, "run", _no_adb)
    assert device.soft_keyboard_hint() == ""


def test_soft_keyboard_hint_is_empty_when_adb_is_not_on_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def must_not_run(*_args: object, **_kwargs: object) -> NoReturn:
        msg = "adb ran although it is not on PATH"
        raise AssertionError(msg)

    monkeypatch.setattr(device.shutil, "which", lambda _name: None)
    monkeypatch.setattr(device.subprocess, "run", must_not_run)
    assert device.soft_keyboard_hint() == ""


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
    assert device.soft_keyboard_hint() == ""


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
    hint = device.soft_keyboard_hint()
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
_UI_DUMP = ("-s", _UDID, "exec-out", "uiautomator dump /dev/stdout")
_SCREENCAP = ("-s", _UDID, "exec-out", "screencap -p")
_DUMPSYS_WINDOW = ("-s", _UDID, "shell", "dumpsys", "window")


class _FakeAdb:
    """Stands in for ``subprocess.run`` and ``time.sleep``.

    ``events`` records each adb argv and each sleep in the order they happened.
    ``replies`` maps the arguments after ``adb`` to an exit status and raw stdout;
    any other command exits 0 with no output. Stdout stays bytes unless the caller
    asks for text, which must name its encoding: the host's locale differs between
    machines, and Android writes UTF-8.
    """

    def __init__(self) -> None:
        self.events: list[list[str] | float] = []
        self.replies: dict[tuple[str, ...], tuple[int, bytes]] = {}

    def run(
        self,
        command: list[str],
        *,
        check: bool = False,
        text: bool = False,
        encoding: str | None = None,
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[str] | subprocess.CompletedProcess[bytes]:
        self.events.append(command)
        returncode, stdout = self.replies.get(tuple(command[1:]), (0, b""))
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
    built = device.AdbController({"device": {"udid": _UDID}}, logging.getLogger("test"))
    adb.events.clear()
    return built


def test_adb_controller_reads_the_screen_size_of_the_configured_device(adb: _FakeAdb) -> None:
    adb.replies[_WM_SIZE] = (0, b"Physical size: 1080x2400\n")
    config = {"device": {"udid": _UDID}}

    controller = device.AdbController(config, logging.getLogger("test"))

    assert (controller.w, controller.h) == (1080, 2400)
    assert controller.config is config
    assert adb.events == [["adb", *_WM_SIZE]]


def test_adb_controller_raises_when_wm_size_reports_no_size(adb: _FakeAdb) -> None:
    adb.replies[_WM_SIZE] = (0, b"Physical size: unknown\n")
    with pytest.raises(ValueError, match=r"^Failed to get screen size$"):
        device.AdbController({"device": {"udid": _UDID}}, logging.getLogger("test"))


# PNG's signature contains CR LF, so any newline translation would change the base64.
_PNG = b"\x89PNG\r\n\x1a\n"


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
    }
    assert adb.events == [["adb", *_UI_DUMP], ["adb", *_SCREENCAP], ["adb", *_DUMPSYS_WINDOW]]


@pytest.mark.parametrize("failing", [_UI_DUMP, _SCREENCAP, _DUMPSYS_WINDOW])
def test_get_state_raises_when_a_read_fails(
    controller: device.AdbController, adb: _FakeAdb, failing: tuple[str, ...]
) -> None:
    adb.replies[failing] = (1, b"")
    with pytest.raises(subprocess.CalledProcessError):
        controller.get_state()


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
