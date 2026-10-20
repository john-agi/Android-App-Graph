"""Tests for android_app_graph.device's adb helpers and system-overlay check.

Each is shared by several of kobe, the CLI, commands.audit and the AITK
translator; this module owns their tests so the call sites do not each pin
their own copy.
"""

from __future__ import annotations

import logging
import subprocess
from typing import Any

import pytest

from android_app_graph import device


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
        "com.google.android.inputmethod.latin",
        "com.android.systemui",
        "android",
        "org.example.inputmethod.pinyin",
        "com.example.keyboard",
    ],
)
def test_a_system_overlay_package_is_recognised(package: str) -> None:
    assert device.is_system_overlay(package)


@pytest.mark.parametrize(
    "package", ["com.amazon.primevideo", "com.example.timetracker", "com.demo"]
)
def test_an_app_package_is_not_a_system_overlay(package: str) -> None:
    """Regression for #86: any package containing "ime" used to count as a keyboard."""
    assert not device.is_system_overlay(package)


_UDID = "emulator-5554"
_PACKAGE = "com.example.app"
_REFUSAL = (
    "Exception occurred while executing 'grant':\n"
    "java.lang.SecurityException: Permission android.permission.INTERNET requested by "
    "com.example.app is not a changeable permission type\n"
    "\tat com.android.server.pm.permission.PermissionManagerServiceImpl"
    ".grantRuntimePermissionInternal(PermissionManagerServiceImpl.java:1433)\n"
    "\tat com.android.server.pm.permission.PermissionManagerServiceImpl"
    ".grantRuntimePermission(PermissionManagerServiceImpl.java:1364)\n"
)


class _Adb:
    """A fake ``subprocess.run`` for adb that records every command.

    ``dumpsys package`` returns ``dumpsys``; ``pm grant`` succeeds unless the
    permission is in ``refused``, which fails the way ``pm`` refuses ``INTERNET``.
    """

    def __init__(
        self, dumpsys: subprocess.CompletedProcess[str], refused: frozenset[str] = frozenset()
    ) -> None:
        self.commands: list[list[str]] = []
        self._dumpsys = dumpsys
        self._refused = refused

    def __call__(self, args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        self.commands.append(args)
        if args[4:6] == ["dumpsys", "package"]:
            return self._dumpsys
        if args[-1] in self._refused:
            return subprocess.CompletedProcess(args, 255, stdout="", stderr=_REFUSAL)
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")


def test_each_requested_permission_is_granted_and_a_refusal_is_logged_and_skipped(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    dumpsys_package: str,
    requested_permissions: list[str],
) -> None:
    adb = _Adb(
        subprocess.CompletedProcess([], 0, stdout=dumpsys_package, stderr=""),
        refused=frozenset({"android.permission.INTERNET"}),
    )
    monkeypatch.setattr(device.subprocess, "run", adb)
    with caplog.at_level(logging.INFO, logger="android_app_graph.device"):
        device.grant_runtime_permissions(_UDID, _PACKAGE)
    assert adb.commands == [
        ["adb", "-s", _UDID, "shell", "dumpsys", "package", _PACKAGE],
        *(
            ["adb", "-s", _UDID, "shell", "pm", "grant", _PACKAGE, permission]
            for permission in requested_permissions
        ),
    ]
    refusal = (
        "Could not grant android.permission.INTERNET to com.example.app: "
        "Exception occurred while executing 'grant': java.lang.SecurityException: "
        "Permission android.permission.INTERNET requested by com.example.app "
        "is not a changeable permission type"
    )
    assert caplog.messages == [
        refusal,
        "Granted 9 of 10 requested permissions to com.example.app",
    ]


def test_no_permission_is_granted_when_dumpsys_fails(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    adb = _Adb(subprocess.CompletedProcess([], 1, stdout="", stderr="adb: device offline\n"))
    monkeypatch.setattr(device.subprocess, "run", adb)
    with caplog.at_level(logging.INFO, logger="android_app_graph.device"):
        device.grant_runtime_permissions(_UDID, _PACKAGE)
    assert adb.commands == [["adb", "-s", _UDID, "shell", "dumpsys", "package", _PACKAGE]]
    warning = (
        "Could not read the permissions com.example.app requests, so none are granted: "
        "adb: device offline"
    )
    assert [(record.levelno, record.getMessage()) for record in caplog.records] == [
        (logging.WARNING, warning)
    ]
