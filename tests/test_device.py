"""Tests for android_app_graph.device.

The single soft-keyboard probe shared by kobe, commands.audit and the AITK
translator; this module owns its tests so the three call sites do not each
pin their own copy of the adb probe.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any

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
        raise AssertionError("restarting the app must not read the screen")

    def exe_action(self, action: dict[str, Any]) -> None:
        raise AssertionError(f"restarting the app must not act on the screen: {action}")


def test_restart_app_force_stops_then_launches_and_waits_for_the_screen(
    device_log: list[object], no_sleep: list[float]
) -> None:
    device.restart_app(_Device(), "demo", "com.demo", None, logging.getLogger("test"))
    assert device_log == [_FORCE_STOP, _PACKAGE_LAUNCH]
    assert no_sleep == [4]


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
    with pytest.raises(subprocess.CalledProcessError):
        device.restart_app(_Device(), "demo", "com.demo", None, logging.getLogger("test"))
    assert commands == [_FORCE_STOP]
    assert no_sleep == []


class _CompletedProcess:
    """The subset of subprocess.CompletedProcess the probe reads."""

    def __init__(self, stdout: str) -> None:
        self.stdout = stdout


def test_soft_keyboard_hint_when_the_keyboard_is_shown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        device.subprocess, "run", lambda *_a, **_k: _CompletedProcess("mInputShown=true")
    )
    hint = device.soft_keyboard_hint()
    assert "soft keyboard is currently visible" in hint


def test_soft_keyboard_hint_when_the_keyboard_is_hidden(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(device.subprocess, "run", lambda *_a, **_k: _CompletedProcess(""))
    assert device.soft_keyboard_hint() == ""


def test_soft_keyboard_hint_survives_a_missing_adb(monkeypatch: pytest.MonkeyPatch) -> None:
    def _no_adb(*_args: Any, **_kwargs: Any) -> _CompletedProcess:
        raise FileNotFoundError("adb")

    monkeypatch.setattr(device.subprocess, "run", _no_adb)
    assert device.soft_keyboard_hint() == ""


def test_soft_keyboard_hint_survives_undecodable_dumpsys_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``text=True`` decodes adb's stdout; undecodable bytes must not raise out of a
    probe that has to stay best-effort.
    """

    def _bad_bytes(*_args: Any, **_kwargs: Any) -> _CompletedProcess:
        msg = "invalid start byte"
        raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, msg)

    monkeypatch.setattr(device.subprocess, "run", _bad_bytes)
    assert device.soft_keyboard_hint() == ""


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
