"""Supported display formats and bounded, quoted adb configuration operations."""

from __future__ import annotations

import logging
import shlex
import subprocess
import time
from dataclasses import replace

import pytest

from android_app_graph import device
from android_app_graph.display_state import NIGHT_KEYS, DisplayState, NightMode


@pytest.fixture
def baseline() -> DisplayState:
    return DisplayState(
        (720, 1280),
        None,
        320,
        None,
        None,
        "no",
        "22:00",
        "06:00",
        dict.fromkeys(NIGHT_KEYS),
        (False, False),
    )


@pytest.mark.parametrize("mode", ["yes", "no", "auto", "custom_schedule", "custom_bedtime"])
def test_display_state_roundtrip_retains_absence_and_effective_mode(
    baseline: DisplayState, mode: NightMode
) -> None:
    state = replace(baseline, night_mode=mode)
    assert DisplayState.from_dict(state.to_dict()) == state
    commands = state.restoration_commands()
    assert ("cmd", "uimode", "night", mode) in commands
    assert ("wm", "size", "reset") in commands
    assert ("wm", "density", "reset") in commands
    assert ("settings", "delete", "system", "font_scale") in commands
    assert all(("settings", "delete", "secure", key) in commands for key in NIGHT_KEYS)


def test_explicit_overrides_have_exact_restoration_commands(baseline: DisplayState) -> None:
    state = replace(
        baseline,
        size_override=(1080, 1920),
        density_override=400,
        font_scale="1.25",
        night_settings=dict.fromkeys(NIGHT_KEYS, "0"),
    )
    commands = state.restoration_commands()
    assert ("wm", "size", "1080x1920") in commands
    assert ("wm", "density", "400") in commands
    assert ("settings", "put", "system", "font_scale", "1.25") in commands
    assert all(("settings", "put", "secure", key, "0") in commands for key in NIGHT_KEYS)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("physical_size", [0, 1]),
        ("physical_size", [True, 3]),
        ("physical_size", [720]),
        ("density_override", 0),
        ("physical_density", False),
        ("font_scale", 1.5),
        ("night_mode", "unknown"),
        ("night_overrides", [False]),
        ("night_overrides", [0, 1]),
        ("custom_start", "invalid"),
        ("custom_end", "25:00"),
        ("night_settings", {}),
    ],
)
def test_malformed_stored_display_configuration_is_rejected(
    baseline: DisplayState, field: str, value: object
) -> None:
    with pytest.raises(ValueError, match=r"(?i)(display|density|setting|night|hour)"):
        DisplayState.from_dict({**baseline.to_dict(), field: value})


@pytest.mark.parametrize("field", ["font_scale", "size_override", "night_settings"])
def test_missing_display_fields_are_explicit(baseline: DisplayState, field: str) -> None:
    data = baseline.to_dict()
    del data[field]
    with pytest.raises(ValueError, match="Incomplete display baseline"):
        DisplayState.from_dict(data)


def test_persistent_night_values_require_text_or_absence(baseline: DisplayState) -> None:
    values = {**baseline.night_settings, "ui_night_mode": 1}
    with pytest.raises(ValueError, match="text or null"):
        DisplayState.from_dict({**baseline.to_dict(), "night_settings": values})


class _Adb:
    def __init__(self) -> None:
        self.answers: dict[tuple[str, ...], str] = {
            ("wm", "size"): "Physical size: 720x1280\n",
            ("wm", "density"): "Physical density: 320\n",
            ("cmd", "uimode", "night"): "Night mode: no\n",
            ("dumpsys", "uimode"): "mOverrideOn/Off=false/false customStart=22:00 customEnd06:00\n",
        }
        self.calls: list[list[str]] = []
        self.failure: subprocess.CalledProcessError | None = None

    def run(
        self, args: list[str], *, output: str, check: bool, timeout: float, **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        del output, check
        assert 0 < timeout <= 3
        self.calls.append(args)
        if self.failure:
            raise self.failure
        command = tuple(args[3:])
        answer = self.answers.get(command, "null" if command[:2] == ("settings", "get") else "")
        return subprocess.CompletedProcess(args, 0, answer, "")


@pytest.fixture
def adb(monkeypatch: pytest.MonkeyPatch) -> _Adb:
    adapter = _Adb()
    monkeypatch.setattr(device, "run_adb", adapter.run)
    return adapter


def test_adb_capture_preserves_override_presence_and_custom_schedule(adb: _Adb) -> None:
    adb.answers[("wm", "size")] += "Override size: 1080x1920\n"
    adb.answers[("wm", "density")] += "Override density: 400\n"
    adb.answers[("settings", "get", "system", "font_scale")] = "1.15"
    adb.answers[("cmd", "uimode", "night")] = "Night mode: custom_schedule"
    state = device.AdbDisplayAccess("fake").capture()
    assert state.size_override == (1080, 1920)
    assert state.density_override == 400
    assert state.font_scale == "1.15"
    assert state.night_mode == "custom_schedule"
    assert state.night_settings == dict.fromkeys(NIGHT_KEYS)
    assert all(c[:3] == ["-s", "fake", "shell"] for c in adb.calls)


@pytest.mark.parametrize(
    ("command", "output"),
    [
        (("wm", "size"), "Physical size: invalid"),
        (("wm", "density"), "Physical density: nope"),
        (("wm", "size"), "Physical size: 720x1280\nOverride size: invalid"),
        (("wm", "size"), ""),
        (("wm", "density"), ""),
        (("dumpsys", "uimode"), "unsupported format"),
    ],
)
def test_unsupported_adb_configuration_never_invents_a_baseline(
    adb: _Adb, command: tuple[str, ...], output: str
) -> None:
    adb.answers[command] = output
    with pytest.raises(ValueError, match=r"(?i)(display|density|setting|night|hour)"):
        device.AdbDisplayAccess("fake").capture()


def test_expired_deadline_sends_no_configuration_read(adb: _Adb) -> None:
    with pytest.raises(subprocess.TimeoutExpired):
        device.AdbDisplayAccess("fake").capture(deadline=0)
    assert not adb.calls


def test_configuration_values_are_quoted_for_the_device_shell(adb: _Adb) -> None:
    value = "1.0; echo unsafe"
    device.AdbDisplayAccess("fake").execute(
        ("settings", "put", "system", "font_scale", value), deadline=time.monotonic() + 10
    )
    assert adb.calls[-1][-1] == shlex.quote(value)


def test_explicit_seed_directory_returns_only_named_file_hashes(adb: _Adb) -> None:
    command = ("find", "/sdcard/seed", "-type", "f", "-exec", "sha256sum", "'{}'", "+")
    adb.answers[command] = "a" * 64 + "  /sdcard/seed/a.txt\n"
    assert device.AdbDisplayAccess("fake").seed_hashes("/sdcard/seed") == {
        "/sdcard/seed/a.txt": "a" * 64
    }


@pytest.mark.parametrize("output", ["malformed", "a" * 64 + "  /data/private/a.txt"])
def test_seed_output_refuses_malformed_or_out_of_scope_files(adb: _Adb, output: str) -> None:
    command = ("find", "/sdcard/seed", "-type", "f", "-exec", "sha256sum", "'{}'", "+")
    adb.answers[command] = output
    with pytest.raises(ValueError, match="malformed"):
        device.AdbDisplayAccess("fake").seed_hashes("/sdcard/seed")


@pytest.mark.parametrize("stderr", ["No such file or directory", "Permission denied"])
def test_seed_directory_loss_and_read_failure_are_distinct(adb: _Adb, stderr: str) -> None:
    adb.failure = subprocess.CalledProcessError(1, ["find"], stderr=stderr)
    if stderr == "No such file or directory":
        assert device.AdbDisplayAccess("fake").seed_hashes("/sdcard/seed") == {}
    else:
        with pytest.raises(subprocess.CalledProcessError):
            device.AdbDisplayAccess("fake").seed_hashes("/sdcard/seed")


def test_retry_commands_include_only_mismatched_owned_values(baseline: DisplayState) -> None:
    current = replace(baseline, font_scale="1.0")
    assert baseline.changed_restoration_commands(current) == [
        ("settings", "delete", "system", "font_scale")
    ]
    current = replace(
        current,
        size_override=(1080, 1920),
        density_override=400,
        night_mode="yes",
        custom_start="21:00",
        custom_end="05:30",
        night_settings=dict.fromkeys(NIGHT_KEYS, "0"),
    )
    assert baseline.changed_restoration_commands(current) == baseline.restoration_commands()


@pytest.mark.parametrize("failures", [1, 2])
def test_display_reads_retry_one_timeout_within_the_original_deadline(
    adb: _Adb, monkeypatch: pytest.MonkeyPatch, failures: int
) -> None:
    attempts: list[float] = []
    now = 10.0

    def run(
        args: list[str], *, timeout: float, **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        nonlocal now
        attempts.append(timeout)
        if len(attempts) <= failures:
            now += 2
            raise subprocess.TimeoutExpired(args, timeout)
        return adb.run(args, output="capture", check=True, timeout=timeout, **kwargs)

    def adapter(
        args: list[str], *, output: str, check: bool, timeout: float
    ) -> subprocess.CompletedProcess[str]:
        del output, check
        return run(args, timeout=timeout)

    monkeypatch.setattr(device.time, "monotonic", lambda: now)
    monkeypatch.setattr(device, "run_adb", adapter)
    if failures == 1:
        assert device.AdbDisplayAccess("fake").capture(deadline=14).physical_size == (720, 1280)
        assert attempts[:2] == [3, 2]
    else:
        with pytest.raises(subprocess.TimeoutExpired):
            device.AdbDisplayAccess("fake").capture(deadline=14)
        assert attempts == [3, 2]


@pytest.mark.usefixtures("no_sleep")
@pytest.mark.parametrize(
    "command",
    [
        ("wm", "size"),
        ("wm", "density"),
        ("dumpsys", "uimode"),
        ("cmd", "uimode", "night"),
        ("settings", "get", "system", "font_scale"),
        *(("settings", "get", "secure", key) for key in NIGHT_KEYS),
    ],
)
def test_display_transport_failures_recover_the_complete_observation(
    adb: _Adb, monkeypatch: pytest.MonkeyPatch, command: tuple[str, ...]
) -> None:
    failed = False

    def run(
        args: list[str], *, output: str, check: bool, timeout: float
    ) -> subprocess.CompletedProcess[str]:
        nonlocal failed
        if tuple(args[3:]) == command and not failed:
            failed = True
            raise subprocess.CalledProcessError(1, ["adb", *args], stderr="error: device offline")
        return adb.run(args, output=output, check=check, timeout=timeout)

    class ObservedDisplay:
        w = 720
        h = 1280

        def __init__(self) -> None:
            self.config: dict[str, object] = {"device": {"udid": "fake"}}

        def get_state(self, *, deadline: float | None = None) -> dict[str, object]:
            return {"display": device.AdbDisplayAccess("fake").capture(deadline=deadline).to_dict()}

        def exe_action(self, action: dict[str, object]) -> None:
            pytest.fail(f"recovery sent an input: {action}")

    monkeypatch.setattr(device, "run_adb", run)
    monkeypatch.setattr(device, "_answers", lambda *_args: True)
    state = device.wait_for_device_state(ObservedDisplay(), 5, logging.getLogger("test"))
    assert failed
    assert state["display"]["physical_size"] == [720, 1280]
