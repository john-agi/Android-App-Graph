"""Tests for android_app_graph.device: the soft-keyboard probe and the app launch helpers.

The probe is shared by kobe, commands.audit and the AITK translator, and the
launch helpers by cli, kobe and commands.audit; this module owns their tests so
the call sites do not each pin their own copy of the adb commands.
"""

from __future__ import annotations

import logging
import subprocess
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
    """A DeviceController on ``test-device`` whose screen must not be read or touched."""

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


def test_launch_app_falls_back_to_the_package_when_the_activity_fails(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    commands: list[list[str]] = []

    def run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        returncode = 1 if "start" in command else 0
        return subprocess.CompletedProcess(command, returncode, stdout="", stderr="no activity")

    monkeypatch.setattr(device.subprocess, "run", run)
    app = {"name": "demo", "package_name": "com.demo", "launch_activity": "com.demo/.Gone"}
    with caplog.at_level(logging.WARNING):
        device.launch_app({"device": {"udid": "test-device"}}, app, logging.getLogger("test"))
    assert commands == [
        ["adb", "-s", "test-device", "shell", "am", "start", "-n", "com.demo/.Gone"],
        _PACKAGE_LAUNCH,
    ]
    assert "Activity launch failed for demo" in caplog.text
    assert "error=no activity" in caplog.text


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
