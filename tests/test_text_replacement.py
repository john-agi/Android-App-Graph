"""Keyboard replacement preserves the editor when an empty field passes shortcuts on."""

from __future__ import annotations

import logging
import shlex
import subprocess
from collections.abc import Sequence
from typing import Literal

import pytest

from android_app_graph import device


class _Editor:
    def __init__(self, value: str, selection: tuple[int, int]) -> None:
        self.value = value
        self.selection = selection
        self.application_selection = False
        self.pending: list[tuple[str, ...]] = []

    def run(
        self,
        args: Sequence[str],
        *,
        output: Literal["capture", "discard"],
        timeout: float,
        check: bool = False,
        errors: str = "strict",
    ) -> subprocess.CompletedProcess[str]:
        del output, timeout, check, errors
        command = list(args[2:])
        stdout = ""
        if command == ["shell", "wm", "size"]:
            stdout = "Physical size: 720x1280\n"
        elif command == ["exec-out", "uiautomator dump /dev/stdout"]:
            for keys in self.pending:
                self._shortcut(keys)
            self.pending.clear()
            stdout = "<hierarchy><node /></hierarchy>"
        elif command[:4] == ["shell", "input", "keyboard", "keycombination"]:
            self.pending.append(tuple(command[4:]))
        elif command == ["shell", "input", "keyboard", "keyevent", "KEYCODE_DEL"]:
            start, end = sorted(self.selection)
            if start == end:
                start = max(0, start - 1)
            self.value = self.value[:start] + self.value[end:]
            self.selection = start, start
        elif command[:3] == ["shell", "input", "text"]:
            text = shlex.split(command[3])[0]
            start, end = sorted(self.selection)
            self.value = self.value[:start] + text + self.value[end:]
            self.selection = start + len(text), start + len(text)
        else:
            pytest.fail(f"Unexpected editor transport: {command}")
        return subprocess.CompletedProcess(list(args), 0, stdout, "")

    def _shortcut(self, keys: tuple[str, ...]) -> None:
        assert "KEYCODE_CTRL_LEFT" in keys
        if keys[-1] == "KEYCODE_A":
            if self.value:
                self.selection = 0, len(self.value)
            else:
                # Android's empty TextView declines Select All; the activity receives it.
                self.application_selection = True
        else:
            assert keys[-1] in {"KEYCODE_MOVE_HOME", "KEYCODE_MOVE_END"}
            end = 0 if keys[-1] == "KEYCODE_MOVE_HOME" else len(self.value)
            start = self.selection[0] if "KEYCODE_SHIFT_LEFT" in keys else end
            self.selection = start, end


@pytest.mark.parametrize(
    ("value", "selection"),
    [
        pytest.param("", (0, 0), id="empty-search"),
        pytest.param("report", (0, 0), id="beginning"),
        pytest.param("report", (2, 2), id="middle"),
        pytest.param("report", (6, 6), id="end"),
        pytest.param("report", (1, 4), id="partial-selection"),
        pytest.param("report", (6, 0), id="backward-selection"),
        pytest.param("long value " * 40, (100, 150), id="long-selection"),
        pytest.param("first line\nsecond line\nlast line", (14, 14), id="multiline-middle"),
    ],
)
@pytest.mark.parametrize("replacement", ["Lamp", ""])
def test_type_replaces_the_editor_without_activating_application_selection(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
    selection: tuple[int, int],
    replacement: str,
) -> None:
    editor = _Editor(value, selection)
    monkeypatch.setattr(device, "run_adb", editor.run)
    monkeypatch.setattr(device.time, "sleep", lambda _seconds: None)
    controller = device.AdbController(
        {"device": {"udid": "test-device"}}, logging.getLogger("test"), unreachable_limit=30
    )

    controller.exe_action({"action": "type", "text": replacement})

    assert editor.value == replacement
    assert not editor.application_selection
