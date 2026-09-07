"""Tests for android_app_graph.device.soft_keyboard_hint.

The single soft-keyboard probe shared by kobe, commands.audit and the AITK
translator; this module owns its tests so the three call sites do not each
pin their own copy of the adb probe.
"""

from __future__ import annotations

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
