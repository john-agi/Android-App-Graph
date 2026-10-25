"""Type replacement through the real dispatcher and a standard editable-buffer model."""

from __future__ import annotations

import logging
import shlex
import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest

from android_app_graph import device
from android_app_graph.step_log import StepRecorder, read_records


class _EditableBuffer:
    """Model selection, backward deletion and insertion, without an Android widget or IME."""

    def __init__(
        self, value: str, selection: tuple[int, int], *, handles_select_all: bool = True
    ) -> None:
        self.value = value
        self.selection = selection
        self.handles_select_all = handles_select_all
        self.commands: list[list[str]] = []

    def run(self, args: Sequence[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        command = list(args)
        if command == ["-s", "buffer", "shell", "wm", "size"]:
            return subprocess.CompletedProcess(command, 0, "Physical size: 720x1280\n")
        assert command[:4] == ["-s", "buffer", "shell", "input"]
        editing = command[4:]
        self.commands.append(editing)
        match editing:
            case ["keyboard", "keycombination", "KEYCODE_CTRL_LEFT", "KEYCODE_A"]:
                if self.handles_select_all:
                    self.selection = (0, len(self.value))
            case ["keyevent", *keys] | ["keyboard", "keyevent", *keys]:
                for key in keys:
                    assert key == "KEYCODE_DEL"
                    start, end = sorted(self.selection)
                    if start == end:
                        start = max(0, start - 1)
                    self.replace(start, end, "")
            case ["text", quoted]:
                chunks = shlex.split(quoted)
                assert len(chunks) == 1
                self.replace(*sorted(self.selection), chunks[0].replace("%s", " "))
            case _:
                pytest.fail(f"Unexpected input: {editing}")
        return subprocess.CompletedProcess(command, 0, "")

    def replace(self, start: int, end: int, text: str) -> None:
        self.value = self.value[:start] + text + self.value[end:]
        self.selection = (start + len(text), start + len(text))


def _controller(
    buffer: _EditableBuffer, monkeypatch: pytest.MonkeyPatch
) -> device.DeviceController:
    monkeypatch.setattr(device, "run_adb", buffer.run)
    return device.AdbController(
        {"device": {"udid": "buffer"}}, logging.getLogger("test"), unreachable_limit=37
    )


@pytest.mark.usefixtures("no_sleep")
@pytest.mark.parametrize(
    ("initial", "selection", "requested"),
    [
        pytest.param("", (0, 0), "New", id="empty-field"),
        pytest.param("ABCDE", (0, 0), "New", id="beginning"),
        pytest.param("ABCDE", (2, 2), "New", id="middle"),
        pytest.param("ABCDE", (5, 5), "New", id="end"),
        pytest.param("ABCDEFGHIJKLMNOPQRSTUVWXYZ", (0, 0), "New", id="long-beginning"),
        pytest.param("ABCDEFGHIJKLMNOPQRSTUVWXYZ", (13, 13), "New", id="long-middle"),
        pytest.param("ABCDEFGHIJKLMNOPQRSTUVWXYZ", (26, 26), "New", id="long-end"),
        pytest.param("ABCDEFGHIJKLM\nNOPQRSTUVWXYZ", (2, 2), "New", id="multiline-middle"),
        pytest.param("ABCDEFGHIJKLM\nNOPQRSTUVWXYZ", (27, 27), "New", id="multiline-end"),
        pytest.param("ABCDEFGHIJKLMNOPQRSTUVWXYZ", (5, 10), "New", id="partial-selection"),
        pytest.param("ABCDEFGHIJKLMNOPQRSTUVWXYZ", (26, 5), "New", id="reverse-selection"),
        pytest.param("ABCDEFGHIJKLMNOPQRSTUVWXYZ", (0, 26), "New", id="full-selection"),
        pytest.param("ABCDE", (2, 2), "", id="empty-replacement"),
        pytest.param("ABCDE", (2, 2), "R'&%s", id="quoted-literal-percent-s"),
    ],
)
def test_type_replaces_the_complete_supported_field_value(
    monkeypatch: pytest.MonkeyPatch, initial: str, selection: tuple[int, int], requested: str
) -> None:
    buffer = _EditableBuffer(initial, selection)

    _controller(buffer, monkeypatch).exe_action({"action": "type", "text": requested})

    assert buffer.value == requested
    assert buffer.selection == (len(requested), len(requested))


@pytest.mark.parametrize("length", [1, 26, 10_000])
def test_type_uses_two_clearing_commands_independently_of_value_length(
    monkeypatch: pytest.MonkeyPatch, no_sleep: list[float], length: int
) -> None:
    buffer = _EditableBuffer("A" * length, (length // 2, length // 2))

    _controller(buffer, monkeypatch).exe_action({"action": "type", "text": "New"})

    assert buffer.value == "New"
    assert buffer.commands == [
        ["keyboard", "keycombination", "KEYCODE_CTRL_LEFT", "KEYCODE_A"],
        ["keyboard", "keyevent", "KEYCODE_DEL"],
        ["text", "New"],
    ]
    assert no_sleep == [1]


@pytest.mark.usefixtures("no_sleep")
def test_successful_delivery_does_not_acknowledge_an_unhandled_selection_shortcut(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A field ignoring Select All is outside the supported replacement contract."""
    buffer = _EditableBuffer("ABCDE", (2, 2), handles_select_all=False)

    _controller(buffer, monkeypatch).exe_action({"action": "type", "text": "New"})

    assert buffer.value == "ANewCDE"
    assert len(buffer.commands) == 3


@pytest.mark.usefixtures("no_sleep")
def test_type_replacement_is_one_recorded_action(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    buffer = _EditableBuffer("ABCDEFGHIJKLMNOPQRSTUVWXYZ", (13, 13))
    recorder = StepRecorder(
        _controller(buffer, monkeypatch),
        tmp_path,
        run_kind="explore",
        source="main_loop",
        target_package="com.demo",
    )

    recorder.exe_action({"action": "type", "text": "New"})

    assert buffer.value == "New"
    records = read_records(tmp_path)
    assert len(records) == 1
    assert records[0]["kind"] == "action"
    assert records[0]["action"] == {"action": "type", "text": "New"}
