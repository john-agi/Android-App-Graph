"""Focused SystemUI subpanels and ordinary app-window countercontrols."""

from __future__ import annotations

import base64
import logging
import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest

from android_app_graph import device
from android_app_graph.step_log import (
    StepRecorder,
    observation_ownership,
    read_records,
    recorded_observation,
)
from android_app_graph.system_surfaces import window_ownership
from android_app_graph.utils.graph_manager import GraphManager


def _dialog_window(title: str = "Device selection") -> str:
    return (
        f"mCurrentFocus=Window{{dialog u0 {title}}}\n"
        "mFocusedApp=ActivityRecord{app u0 com.demo/.Home t1}\n"
        f"  Window #0 Window{{dialog u0 {title}}}:\n"
        "    mDisplayId=0 mOwnerUid=10062 package=com.android.systemui\n"
        "    mAttrs={ty=STATUS_BAR_SUB_PANEL}\n"
        "    mViewVisibility=0x0\n"
        "    mHasSurface=true isOnScreen=true isVisible=true\n"
        "  Window #1 Window{app u0 com.demo/com.demo.Home}:\n"
        "    mDisplayId=0 package=com.demo ty=BASE_APPLICATION\n"
        "    mHasSurface=true isVisible=true\n"
    )


@pytest.mark.parametrize("title", ["Device selection", "MediaOutputDialog", "Unrelated panel"])
def test_a_focused_systemui_subpanel_has_a_synthetic_window_identity(title: str) -> None:
    observed = window_ownership(_dialog_window(title), "com.demo", "com.demo.Home")

    assert observed.package == "com.android.systemui"
    assert observed.activity == "SystemUIDialog"
    assert observed.system_surface == "system_dialog"
    assert observed.underlying_package == "com.demo"
    assert observed.system_overlays == ()
    assert observed.overlay_error is None


@pytest.mark.parametrize(
    ("original", "replacement"),
    [
        ("mCurrentFocus=Window{dialog", "mCurrentFocus=Window{other"),
        ("mCurrentFocus=Window{dialog", "mCurrentFocus=null # Window{dialog"),
        ("mDisplayId=0 mOwnerUid", "mDisplayId=1 mOwnerUid"),
        ("mHasSurface=true isOnScreen", "mHasSurface=false isOnScreen"),
        ("isVisible=true\n  Window #1", "isVisible=false\n  Window #1"),
        ("package=com.android.systemui", "package=com.demo"),
        ("package=com.android.systemui", "owner_unavailable=true"),
        ("ty=STATUS_BAR_SUB_PANEL", "ty=APPLICATION_PANEL"),
        ("ty=STATUS_BAR_SUB_PANEL", "ty=STATUS_BAR"),
        ("ty=STATUS_BAR_SUB_PANEL", "ty=NAVIGATION_BAR"),
        ("ty=STATUS_BAR_SUB_PANEL", "ty=NAVIGATION_BAR_PANEL"),
        ("ty=STATUS_BAR_SUB_PANEL", "ty=INPUT_METHOD"),
    ],
)
def test_only_a_verified_focused_systemui_subpanel_replaces_app_ownership(
    original: str, replacement: str
) -> None:
    dump = _dialog_window("MediaOutputDialog").replace(original, replacement)

    observed = window_ownership(dump, "com.demo", "com.demo.Home")

    assert observed.package == "com.demo"
    assert observed.activity == "com.demo.Home"
    assert observed.system_surface is None
    assert observed.underlying_package == "com.demo"


def _adapter(monkeypatch: pytest.MonkeyPatch, png_screenshot: str) -> device.DeviceController:
    def run(args: Sequence[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        if list(args[-3:]) == ["shell", "wm", "size"]:
            text = "Physical size: 720x1280\n"
        elif list(args[-3:]) == ["dumpsys", "window", "-a"]:
            text = _dialog_window()
        else:
            assert list(args[-2:]) == ["exec-out", "uiautomator dump /dev/stdout"]
            text = '<hierarchy><node package="com.android.systemui" /></hierarchy>'
        return subprocess.CompletedProcess(args, 0, text)

    def screenshot(_args: Sequence[str], _timeout: float) -> bytes:
        return base64.b64decode(png_screenshot)

    monkeypatch.setattr(device, "run_adb", run)
    monkeypatch.setattr(device, "_adb_stdout_bytes", screenshot)
    return device.AdbController(
        {"device": {"udid": "dialog-test"}}, logging.getLogger("test"), unreachable_limit=37
    )


def test_native_dialog_ownership_round_trips_into_an_external_boundary_without_models(
    monkeypatch: pytest.MonkeyPatch, png_screenshot: str, tmp_path: Path
) -> None:
    recorder = StepRecorder(
        _adapter(monkeypatch, png_screenshot),
        tmp_path,
        run_kind="explore",
        source="main_loop",
        target_package="com.demo",
    )

    state = recorder.get_state()

    assert state["package"] == "com.android.systemui"
    assert state["system_surface"] == "system_dialog"
    records = read_records(tmp_path)
    assert len(records) == 1
    persisted = recorded_observation(tmp_path, records[0])
    assert observation_ownership(records[0]).surface == "system_dialog"
    assert persisted.system_surface == "system_dialog"
    assert records[0]["underlying_package"] == "com.demo"
    graph = GraphManager()
    boundary = graph.identify_state(persisted)
    assert boundary == "ext_com.android.systemui"
    assert graph.graph.nodes[boundary]["system_surface"] == "system_dialog"
    assert graph.get_resume_target() is None
    assert graph.get_exploration_target_candidates(package_name="com.demo") == []


def test_systemui_as_the_target_records_its_own_dialog_as_app_state(
    monkeypatch: pytest.MonkeyPatch, png_screenshot: str, tmp_path: Path
) -> None:
    recorder = StepRecorder(
        _adapter(monkeypatch, png_screenshot),
        tmp_path,
        run_kind="explore",
        source="main_loop",
        target_package="com.android.systemui",
    )

    state = recorder.get_state()

    assert state["package"] == "com.android.systemui"
    assert state["system_surface"] is None
    record = read_records(tmp_path)[0]
    assert record["system_surface"] is None
    assert recorded_observation(tmp_path, record).system_surface is None
