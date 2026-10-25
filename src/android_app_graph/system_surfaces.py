"""Observed ownership of independently navigable system windows and clipboard feedback."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from android_app_graph.payloads import as_int_list, as_str

type Region = tuple[int, int, int, int]

_REGION_COORDINATES = 4

_WINDOW = re.compile(r"^\s*Window #\d+ Window\{(\S+) u\d+ ([^}]+)\}:\s*$", re.MULTILINE)
_CURRENT = re.compile(r"mCurrentFocus=Window\{(\S+) u\d+ ([^}]+)\}")
_COMPONENT = re.compile(r"([\w.]+)/([\w.]+)")
_TOUCH_REGION = re.compile(r"\((-?\d+),(-?\d+),(-?\d+),(-?\d+)\)")
_IME_PACKAGES = frozenset(
    {
        "com.android.inputmethod.latin",
        "com.google.android.inputmethod.latin",
        "com.samsung.android.honeyboard",
        "com.swiftkey.swiftkey",
        "com.touchtype.swiftkey",
    }
)
_PERMISSION_PACKAGES = frozenset(
    {"com.android.permissioncontroller", "com.google.android.permissioncontroller"}
)


@dataclass(frozen=True, slots=True)
class WindowOwnership:
    """A visible window's owner and the app beneath independently owned system UI."""

    package: str
    activity: str
    underlying_package: str
    system_surface: str | None
    system_overlays: tuple[Region, ...]
    overlay_error: str | None = None


def _component(text: str) -> tuple[str, str]:
    match = _COMPONENT.search(text)
    if match is None:
        return "unknown", "unknown"
    package, activity = as_str(match.group(1), ""), as_str(match.group(2), "")
    return package, package + activity if activity.startswith(".") else activity


def _visible(block: str) -> bool:
    return "mHasSurface=true" in block and "isVisible=true" in block


def _focused_system_surface(block: str, owner: re.Match[str] | None) -> tuple[str, str] | None:
    if "ty=NOTIFICATION_SHADE" in block:
        return "NotificationShade", "notification_shade"
    if (
        owner is not None
        and owner.group(1) == "com.android.systemui"
        and "ty=STATUS_BAR_SUB_PANEL" in block
    ):
        return "SystemUIDialog", "system_dialog"
    return None


def window_ownership(dump: str, package: str, activity: str) -> WindowOwnership:
    """Use focused and visible display-zero windows, preserving clipboard touch regions.

    Clipboard input follows explicit touchable regions within its display-sized window.
    """
    app_line = next((line for line in dump.splitlines() if "mFocusedApp=" in line), "")
    underlying, underlying_activity = _component(app_line)
    focused = _CURRENT.search(dump)
    focus_id = focused.group(1) if focused is not None else None
    matches = list(_WINDOW.finditer(dump))
    surface = None
    overlays: list[Region] = []
    overlay_error = None
    for index, match in enumerate(matches):
        block = dump[
            match.end() : matches[index + 1].start() if index + 1 < len(matches) else len(dump)
        ]
        if not re.search(r"\bmDisplayId=0\b", block) or not _visible(block):
            continue
        owner = re.search(r"\bpackage=([\w.]+)", block)
        title = match.group(2)
        if (
            title == "ClipboardOverlay"
            and owner is not None
            and owner.group(1) == "com.android.systemui"
        ):
            region_line = next(
                (line for line in block.splitlines() if "touchable region=" in line), ""
            )
            regions = [
                tuple(int(value) for value in rectangle)
                for rectangle in _TOUCH_REGION.findall(region_line)
            ]
            valid = [(r[0], r[1], r[2], r[3]) for r in regions if r[0] < r[2] and r[1] < r[3]]
            if not valid:
                overlay_error = "Visible ClipboardOverlay has no reported touchable region"
            overlays.extend(valid)
        if match.group(1) != focus_id:
            continue
        focused_surface = _focused_system_surface(block, owner)
        if focused_surface is not None:
            package = "com.android.systemui"
            activity, surface = focused_surface
        elif "ty=INPUT_METHOD" in block:
            package, activity = underlying, underlying_activity
    if package in _PERMISSION_PACKAGES and "GrantPermissionsActivity" in activity:
        surface = "permission"
    elif package in _IME_PACKAGES:
        surface = "ime_settings"
    return WindowOwnership(package, activity, underlying, surface, tuple(overlays), overlay_error)


def permission_requester(activities: str, window_dump: str) -> str | None:
    """Read the caller of the exact permission activity focused by WindowManager."""
    focused = re.search(r"mFocusedApp=ActivityRecord\{(\S+) u\d+ ", window_dump)
    if focused is None:
        return None
    token = as_str(focused.group(1), "")
    records = re.split(r"(?=^\s*\* Hist\s+#\d+: ActivityRecord\{)", activities, flags=re.MULTILINE)
    for record in records[1:]:
        first_line = record.splitlines()[0]
        if re.search(rf"ActivityRecord\{{{re.escape(token)} u\d+ ", first_line) is None:
            continue
        match = re.search(r"\blaunchedFromPackage=([\w.]+)", record)
        return (as_str(match.group(1), "") or None) if match is not None else None
    return None


def overlay_regions(value: object) -> tuple[Region, ...]:
    """Narrow a captured list of native-pixel rectangles."""
    if not isinstance(value, (list, tuple)):
        return ()
    result: list[Region] = []
    for item in value:
        coordinates = as_int_list(list(item)) if isinstance(item, (list, tuple)) else []
        if len(coordinates) == _REGION_COORDINATES:
            left, top, right, bottom = coordinates
            if left < right and top < bottom:
                result.append((left, top, right, bottom))
    return tuple(result)


class UnresolvedSystemOverlayError(RuntimeError):
    """A visible system feedback window has no usable ownership geometry."""


class SystemOverlayActionError(RuntimeError):
    """App input waits for usable ownership of system feedback."""


def check_overlay_action(action: Mapping[str, object], regions: Sequence[Region]) -> None:
    """Reject app coordinates intersecting system feedback without moving the target."""
    points: list[tuple[object, object]] = []
    if action.get("action") in {"tap", "long_press"}:
        points.append((action.get("x"), action.get("y")))
    elif action.get("action") == "swipe":
        points.extend([(action.get("x1"), action.get("y1")), (action.get("x2"), action.get("y2"))])
    for x, y in points:
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            continue
        for left, top, right, bottom in regions:
            if left <= x < right and top <= y < bottom:
                msg = f"App input ({x}, {y}) is covered by system clipboard feedback {left, top, right, bottom}"
                raise SystemOverlayActionError(msg)
