"""System ownership, app-only image masks and bounded permission recovery."""

from __future__ import annotations

import base64
import io
import subprocess
from collections.abc import Mapping

import pytest
from PIL import Image

from android_app_graph import device
from android_app_graph.screen_image import mask_regions
from android_app_graph.step_log import Observation
from android_app_graph.system_surfaces import (
    SystemOverlayActionError,
    check_overlay_action,
    overlay_regions,
    permission_requester,
    window_ownership,
)
from android_app_graph.utils.graph_manager import GraphManager


@pytest.mark.parametrize("requester", [None, "com.android.inputmethod.latin"])
def test_an_unknown_or_other_app_permission_request_sends_no_input(
    requester: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller = _PermissionDevice([])
    events: list[Mapping[str, object]] = []

    def unexpected_grant(*_args: object) -> device.PermissionGrants:
        pytest.fail("An unverified caller has no target grant authority")

    monkeypatch.setattr(device, "grant_runtime_permissions", unexpected_grant)
    with pytest.raises(device.UnresolvedPermissionError, match="not verified as target"):
        device.resolve_permission_prompt(
            controller,
            device.AppLaunch("demo", "com.demo"),
            {**_PROMPT, "permission_requester": requester},
            events.append,
        )
    assert controller.actions == []
    assert events[-1]["status"] == "permission_unresolved"


class _PermissionDevice:
    w = 720
    h = 1280

    def __init__(self, states: list[dict[str, object]]) -> None:
        self.config: dict[str, object] = {"device": {"udid": "test-device"}}
        self.states = states
        self.actions: list[dict[str, object]] = []

    def get_state(self, *, deadline: float | None = None) -> dict[str, object]:
        del deadline
        return self.states.pop(0)

    def exe_action(self, action: dict[str, object]) -> None:
        self.actions.append(action)


_PROMPT: dict[str, object] = {
    "system_surface": "permission",
    "permission_requester": "com.demo",
    "package": "com.android.permissioncontroller",
    "activity": "GrantPermissionsActivity",
}
_APP: dict[str, object] = {"package": "com.demo", "activity": "Home", "system_surface": None}
_CAMERA = "android.permission.CAMERA"
_CATALOG = (
    "  Permission [android.permission.CAMERA] (1):\n    sourcePackage=android prot=dangerous\n"
)


def _authorized_policy(
    monkeypatch: pytest.MonkeyPatch, granted: tuple[str, ...] = (_CAMERA,)
) -> None:
    monkeypatch.setattr(
        device,
        "grant_runtime_permissions",
        lambda *_args: device.PermissionGrants((_CAMERA,), granted),
    )
    monkeypatch.setattr(
        device,
        "run_adb",
        lambda args, **_kwargs: subprocess.CompletedProcess(args, 0, _CATALOG, ""),
    )


def _allow_xml(button: str = "permission_allow_button", *, enabled: str = "true") -> str:
    return f'<hierarchy><node resource-id="com.android.permissioncontroller:id/{button}" enabled="{enabled}" bounds="[10,20][110,100]"/></hierarchy>'


@pytest.mark.parametrize(
    "button", ["permission_allow_button", "permission_allow_foreground_only_button"]
)
def test_a_verified_target_prompt_uses_existing_grants_and_its_supported_button(
    button: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _authorized_policy(monkeypatch)
    controller = _PermissionDevice([{**_PROMPT, "xml": _allow_xml(button)}, _APP])
    events: list[Mapping[str, object]] = []
    state = device.resolve_permission_prompt(
        controller, device.AppLaunch("demo", "com.demo"), _PROMPT, events.append
    )
    assert state == _APP
    assert controller.actions == [{"action": "tap", "x": 60, "y": 60}]
    assert events[1]["requested"] == [_CAMERA]
    assert events[1]["runtime_requested"] == [_CAMERA]
    assert events[-1]["status"] == "permission_resolved"


def test_a_prompt_closed_by_authorized_grants_needs_no_tap(monkeypatch: pytest.MonkeyPatch) -> None:
    _authorized_policy(monkeypatch)
    controller = _PermissionDevice([_APP])
    assert (
        device.resolve_permission_prompt(
            controller, device.AppLaunch("demo", "com.demo"), _PROMPT, lambda _event: None
        )
        == _APP
    )
    assert controller.actions == []


def test_a_repeated_permission_dialog_stops_after_one_supported_tap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _authorized_policy(monkeypatch)
    prompt = {**_PROMPT, "xml": _allow_xml()}
    controller = _PermissionDevice([prompt, prompt])
    with pytest.raises(device.UnresolvedPermissionError, match="remains unresolved"):
        device.resolve_permission_prompt(
            controller, device.AppLaunch("demo", "com.demo"), _PROMPT, lambda _event: None
        )
    assert len(controller.actions) == 1
    assert controller.states == []


@pytest.mark.parametrize(
    "xml",
    [
        _allow_xml("permission_deny_button"),
        _allow_xml(enabled="false"),
        _allow_xml().replace("[10,20][110,100]", "[110,20][10,100]"),
        '<node resource-id="com.android.permissioncontroller:id/permission_allow_button" enabled="true" bounds="missing"/>',
    ],
)
def test_an_unsupported_permission_button_stops_without_a_tap(
    xml: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _authorized_policy(monkeypatch)
    controller = _PermissionDevice([{**_PROMPT, "xml": xml}])
    with pytest.raises(device.UnresolvedPermissionError, match="remains unresolved"):
        device.resolve_permission_prompt(
            controller, device.AppLaunch("demo", "com.demo"), _PROMPT, lambda _event: None
        )
    assert controller.actions == []


def test_failed_authorized_grants_cannot_authorize_the_dialog_button(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _authorized_policy(monkeypatch, ())
    controller = _PermissionDevice([])
    with pytest.raises(device.UnresolvedPermissionError, match="not fully covered"):
        device.resolve_permission_prompt(
            controller, device.AppLaunch("demo", "com.demo"), _PROMPT, lambda _event: None
        )
    assert controller.actions == []


@pytest.mark.parametrize("caller", ["com.demo", "com.android.inputmethod.latin"])
def test_permission_requester_comes_from_its_own_activity_record(caller: str) -> None:
    dump = (
        "  topResumedActivity=ActivityRecord{p u0 com.android.permissioncontroller/.GrantPermissionsActivity t1}\n"
        "    * Hist #1: ActivityRecord{p u0 com.android.permissioncontroller/.GrantPermissionsActivity t1}\n"
        f"      launchedFromPackage={caller}\n"
        "    * Hist #0: ActivityRecord{a u0 com.demo/.Home t1}\n"
        "      launchedFromPackage=com.android.shell\n"
    )
    assert (
        permission_requester(
            dump,
            "mFocusedApp=ActivityRecord{p u0 com.android.permissioncontroller/.GrantPermissionsActivity t1}",
        )
        == caller
    )
    assert (
        permission_requester(dump, "mFocusedApp=ActivityRecord{other u0 com.other/.Prompt t1}")
        is None
    )
    assert permission_requester("", "Prompt") is None


@pytest.mark.parametrize(
    "value",
    [
        None,
        "rectangle",
        [1, 2, 3, 4],
        [[0, 0, 2]],
        [[0, 0, 0, 1]],
        [[0, 0, 1, -1]],
        [[0, 0, "bad", 1]],
    ],
)
def test_unusable_overlay_regions_are_not_coordinates(value: object) -> None:
    assert overlay_regions(value) == ()


def test_overlay_regions_preserve_native_rectangles() -> None:
    assert overlay_regions([[-2, 10, 20, 30], [40, 50, 60, 70]]) == (
        (-2, 10, 20, 30),
        (40, 50, 60, 70),
    )


@pytest.mark.parametrize(
    "action",
    [
        {"action": "tap", "x": 10, "y": 15},
        {"action": "long_press", "x": 10, "y": 15},
        {"action": "swipe", "x1": 10, "y1": 15, "x2": 50, "y2": 50},
        {"action": "swipe", "x1": 50, "y1": 50, "x2": 10, "y2": 15},
    ],
)
def test_app_coordinate_inputs_cannot_target_system_feedback(action: dict[str, object]) -> None:
    with pytest.raises(SystemOverlayActionError, match="system clipboard feedback"):
        check_overlay_action(action, [(0, 10, 20, 30)])


@pytest.mark.parametrize(
    "action",
    [
        {"action": "tap", "x": 20, "y": 15},
        {"action": "tap", "x": -1, "y": 15},
        {"action": "tap", "x": "unknown", "y": 15},
        {"action": "type", "text": "app text"},
        {"action": "wait"},
    ],
)
def test_unobscured_app_inputs_remain_available(action: dict[str, object]) -> None:
    check_overlay_action(action, [(0, 10, 20, 30)])


def test_only_the_native_feedback_region_is_hidden_in_the_model_image() -> None:
    image = Image.new("RGB", (6, 10), "blue")
    raw = io.BytesIO()
    image.save(raw, format="PNG")
    encoded = base64.b64encode(raw.getvalue()).decode("ascii")
    masked = mask_regions(encoded, [(0, 0, 6, 10)], (12, 20))
    with Image.open(io.BytesIO(base64.b64decode(masked))) as observed:
        assert observed.getpixel((2, 4)) == (128, 128, 128)
        assert observed.getpixel((3, 4)) == (0, 0, 255)
        assert observed.getpixel((2, 5)) == (0, 0, 255)
    assert mask_regions(encoded, [], (12, 20)) == encoded
    with Image.open(io.BytesIO(base64.b64decode(encoded))) as observed:
        assert observed.getpixel((2, 4)) == (0, 0, 255)


def test_invisible_system_windows_leave_the_app_owner_unchanged() -> None:
    dump = (
        "mCurrentFocus=Window{shade u0 NotificationShade}\n"
        "mFocusedApp=ActivityRecord{app u0 com.demo/.Home t1}\n"
        "  Window #0 Window{shade u0 NotificationShade}:\n"
        "    mDisplayId=0 package=com.android.systemui ty=NOTIFICATION_SHADE\n"
        "    mHasSurface=false isVisible=false\n"
    )
    ownership = window_ownership(dump, "com.demo", "com.demo.Home")
    assert ownership.package == "com.demo"
    assert ownership.system_surface is None


@pytest.mark.parametrize(
    ("surface", "package"),
    [
        ("notification_shade", "com.android.systemui"),
        ("permission", "com.android.permissioncontroller"),
        ("ime_settings", "com.android.inputmethod.latin"),
    ],
)
def test_system_boundaries_need_no_model_and_cannot_rank_as_app_targets(
    surface: str, package: str
) -> None:
    graph = GraphManager()
    observed = Observation(
        "000001", "raw", "System", package, "hash", False, system_surface=surface
    )
    boundary = graph.identify_state(observed)
    assert boundary == f"ext_{package}"
    assert graph.graph.nodes[boundary]["system_surface"] == surface
    assert graph.get_resume_target() is None
    assert graph.get_exploration_target_candidates(package_name="com.demo") == []


def test_requester_uses_the_focused_activity_token_among_duplicate_permission_activities() -> None:
    windows = "mFocusedApp=ActivityRecord{current u0 com.android.permissioncontroller/.GrantPermissionsActivity t2}"
    activities = (
        " * Hist #1: ActivityRecord{old u0 com.android.permissioncontroller/.GrantPermissionsActivity t1}\n"
        "   launchedFromPackage=com.demo\n"
        " * Hist #1: ActivityRecord{current u0 com.android.permissioncontroller/.GrantPermissionsActivity t2}\n"
        "   launchedFromPackage=com.android.inputmethod.latin\n"
    )
    assert permission_requester(activities, windows) == "com.android.inputmethod.latin"
    assert permission_requester(activities, "missing focused record") is None


def test_permission_recovery_verifies_the_target_screen_after_grants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _authorized_policy(monkeypatch)
    controller = _PermissionDevice([{**_APP, "package": "com.other"}])
    with pytest.raises(device.UnresolvedPermissionError, match="instead of target"):
        device.resolve_permission_prompt(
            controller, device.AppLaunch("demo", "com.demo"), _PROMPT, lambda _event: None
        )
    assert controller.actions == []


def test_permission_recovery_does_not_accept_a_button_from_another_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _authorized_policy(monkeypatch)
    controller = _PermissionDevice(
        [
            {
                **_PROMPT,
                "xml": _allow_xml().replace(
                    "com.android.permissioncontroller:id/", "com.other:id/"
                ),
            }
        ]
    )
    with pytest.raises(device.UnresolvedPermissionError, match="remains unresolved"):
        device.resolve_permission_prompt(
            controller, device.AppLaunch("demo", "com.demo"), _PROMPT, lambda _event: None
        )
    assert controller.actions == []


def test_permission_recovery_stops_when_the_requester_changes_after_grant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _authorized_policy(monkeypatch)
    controller = _PermissionDevice(
        [{**_PROMPT, "permission_requester": "com.other", "xml": _allow_xml()}]
    )
    with pytest.raises(device.UnresolvedPermissionError, match="remains unresolved"):
        device.resolve_permission_prompt(
            controller, device.AppLaunch("demo", "com.demo"), _PROMPT, lambda _event: None
        )
    assert controller.actions == []


def test_clipboard_without_reported_geometry_has_an_explicit_capture_error() -> None:
    dump = (
        "  Window #0 Window{clip u0 ClipboardOverlay}:\n"
        "    mDisplayId=0 package=com.android.systemui ty=SCREENSHOT\n"
        "    mHasSurface=true isVisible=true\n"
    )
    observed = window_ownership(dump, "com.demo", "Home")
    assert observed.overlay_error == "Visible ClipboardOverlay has no reported touchable region"
    assert observed.package == "com.demo"
    assert observed.system_overlays == ()
