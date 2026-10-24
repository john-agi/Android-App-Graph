"""The device: what exploration and the audit need from it, and adb behind that.

``DeviceController`` and ``AvdManager`` state what the explorer, the audit and
``check_avd`` read from a device and do to it. ``AdbController`` and
``EmulatorManager`` provide them over adb and the emulator's AVD files; tests pass
fakes instead, so they need neither adb nor an emulator.
"""

from __future__ import annotations

import base64
import logging
import os
import platform
import re
import shlex
import shutil
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NamedTuple, Protocol

from android_app_graph.display_state import NIGHT_KEYS, DisplayState
from android_app_graph.payloads import as_int
from android_app_graph.system_surfaces import (
    SystemOverlayActionError,
    check_overlay_action,
    permission_requester,
    window_ownership,
)

logger = logging.getLogger(__name__)

# How long each kind of adb call may run. About one adb call in a hundred stalls between
# the host and the device, sometimes for good, and the next call succeeds. An input
# returns once the device has queued its events. Of the calls that wait for the device's
# answer, the slowest is a dump that gives up waiting for an idle screen, after about
# 12 s. A probe answers a question the run can go on without.
_INPUT_TIMEOUT_SECONDS = 15
_READ_TIMEOUT_SECONDS = 30
_PROBE_TIMEOUT_SECONDS = 3

_SOFT_KEYBOARD_HINT = " (Note: the soft keyboard is currently visible.)"


class DeviceController(Protocol):
    """The screen size, the run's config, the current screen and a way to act on it."""

    w: int
    h: int
    config: dict[str, Any]

    def get_state(self, *, deadline: float | None = None) -> dict[str, Any]: ...

    def exe_action(self, action: dict[str, Any]) -> None: ...


class AvdManager(Protocol):
    """What ``check_avd`` needs to find a running device or start an emulator."""

    def get_running_avd_list(self) -> list[dict[str, Any]]: ...

    def duplicate_avd(self, avd_name: str) -> None: ...


def adb_executable() -> str:
    """Return the path of the adb found on ``PATH``.

    A missing adb raises ``FileNotFoundError``, the error ``subprocess`` raises for a
    missing program, so callers that tolerate a missing adb by catching ``OSError``
    still do.
    """
    path = shutil.which("adb")
    if path is None:
        msg = "adb was not found on PATH"
        raise FileNotFoundError(msg)
    return path


def run_adb(
    args: Sequence[str],
    *,
    output: Literal["capture", "discard"],
    errors: str | None = None,
    timeout: float = _READ_TIMEOUT_SECONDS,
    check: bool,
) -> subprocess.CompletedProcess[str]:
    """Run the adb on ``PATH`` with ``args``, capturing its stdout and stderr as text or
    discarding both, and stop it with ``subprocess.TimeoutExpired`` after ``timeout``
    seconds.

    Text is decoded as UTF-8 whatever the host's locale, because Android writes UTF-8.
    """
    capture = output == "capture"
    sink = None if capture else subprocess.DEVNULL
    return subprocess.run(  # noqa: S603  # argv list without a shell; adb arguments come from this package's config and constants
        [adb_executable(), *args],
        capture_output=capture,
        stdout=sink,
        stderr=sink,
        text=True,
        encoding="utf-8",
        errors=errors,
        timeout=timeout,
        check=check,
    )


def _adb_stdout_text(args: Sequence[str], timeout: float) -> str:
    """Run the adb on ``PATH`` with ``args`` and return its stdout as UTF-8 text. A
    non-zero exit raises ``CalledProcessError``.
    """
    return run_adb(args, output="capture", timeout=timeout, check=True).stdout


def _adb_stdout_bytes(args: Sequence[str], timeout: float) -> bytes:
    """Run the adb on ``PATH`` with ``args`` and return its stdout undecoded, for output
    that is not text, such as a PNG. A non-zero exit raises ``CalledProcessError``.
    """
    return subprocess.run(  # noqa: S603  # argv list without a shell; adb arguments come from this package's config and constants
        [adb_executable(), *args], capture_output=True, timeout=timeout, check=True
    ).stdout


def _focused_package_activity(dumpsys_window: str) -> tuple[str, str]:
    """Read the package and activity from the first ``mCurrentFocus`` or ``mFocusedApp``
    line of ``dumpsys window`` output that names one, expanding a shorthand ``.Activity``.
    """
    for line in dumpsys_window.splitlines():
        if "mCurrentFocus" not in line and "mFocusedApp" not in line:
            continue
        match = re.search(r"[\w.]+/[\w.]+", line)
        if match:
            package, _, activity = match.group().partition("/")
            if activity.startswith("."):
                activity = package + activity
            return package, activity
    return "unknown", "unknown"


_STATUS_BAR_SOURCE = re.compile(
    r"InsetsSource\b.*?\btype=statusBars\b"
    r".*?\bframe=\[(-?\d+),(-?\d+)\]\[(-?\d+),(-?\d+)\]"
    r".*?\bvisible=(true|false)\b"
)


def status_bar_frame(dumpsys_window: str) -> list[int] | None:
    """Return the status bar's frame, ``[left, top, right, bottom]`` in screen pixels,
    from the first status-bar ``InsetsSource`` line of ``dumpsys window`` output.

    ``None`` when no line reports the bar, or the bar is hidden, as an app in immersive
    mode hides it, or does not start at the top of the screen. The frame follows the
    display cutout and the rotation, so it is read at every capture.
    """
    for line in dumpsys_window.splitlines():
        match = _STATUS_BAR_SOURCE.search(line)
        if match is None:
            continue
        left, top, right, bottom = (int(value) for value in match.groups()[:4])
        if match.group(5) != "true" or top != 0:
            return None
        return [left, top, right, bottom]
    return None


_DISPLAY_FRAME = re.compile(r"mDisplayFrame=Rect\((-?\d+), (-?\d+) - (-?\d+), (-?\d+)\)")
_INPUT_SOURCE = re.compile(
    r"^\s*InsetsSource\b.*?\btype=(tappableElement|systemGestures|mandatorySystemGestures)\b"
    r".*?\bframe=\[(-?\d+),(-?\d+)\]\[(-?\d+),(-?\d+)\].*?\bvisible=(true|false)\b"
)


class InputRegions(NamedTuple):
    """Display coordinates and the visible system touch and gesture-start regions."""

    display: tuple[int, int, int, int]
    tap: tuple[tuple[int, int, int, int], ...]
    gesture: tuple[tuple[int, int, int, int], ...]

    def as_dict(self) -> dict[str, list[int] | list[list[int]]]:
        """Return the geometry carried by observations and rejected-action records."""
        return {
            "display": list(self.display),
            "tap": [list(frame) for frame in self.tap],
            "gesture": [list(frame) for frame in self.gesture],
        }

    def reject_reason(self, action: Mapping[str, object]) -> str | None:
        """Return why an app input is unreachable at its coordinates, or ``None``."""
        swipe = action["action"] == "swipe"
        start = _input_point(action, "x1", "y1") if swipe else _input_point(action, "x", "y")
        end = _input_point(action, "x2", "y2") if swipe else start
        if start is None or end is None:
            return "App input has invalid coordinates"
        points = (start, end)
        if any(not _contains(self.display, point) for point in points):
            return "App input is outside the display"
        if any(_contains(frame, start) for frame in self.tap):
            return "App input targets a system touch region; reveal the control or choose an unobscured point"
        if swipe and any(_contains(frame, start) for frame in self.gesture):
            return "App swipe starts in a system gesture region; use an interior app control"
        return None


def _input_point(action: Mapping[str, object], x_name: str, y_name: str) -> tuple[int, int] | None:
    x, y = as_int(action.get(x_name)), as_int(action.get(y_name))
    return (x, y) if x is not None and y is not None else None


def _contains(frame: tuple[int, int, int, int], point: tuple[int, int]) -> bool:
    left, top, right, bottom = frame
    return left <= point[0] < right and top <= point[1] < bottom


def input_regions(dumpsys_window: str) -> InputRegions | None:
    """Read the first display's authoritative InsetsState, excluding provider copies.

    Tappable-element sources identify touch-consuming bars; gesture sources identify
    where swipes enter system navigation. A gesture-navigation bar can be transparent
    to ordinary app taps, so its visual navigation-bar frame is not a tap exclusion.
    """
    _, found, block = dumpsys_window.partition("WindowInsetsStateController")
    if not found:
        return None
    block = block.partition("InsetsSourceProviders:")[0]
    display = _DISPLAY_FRAME.search(block)
    if display is None:
        return None
    left, top, right, bottom = (int(value) for value in display.groups())
    taps: list[tuple[int, int, int, int]] = []
    gestures: list[tuple[int, int, int, int]] = []
    known_source = False
    for line in block.splitlines():
        source = _INPUT_SOURCE.search(line)
        if source is None:
            continue
        known_source = True
        if source.group(6) != "true":
            continue
        x1, y1, x2, y2 = (int(value) for value in source.groups()[1:5])
        if x1 >= x2 or y1 >= y2:
            continue
        frames = taps if source.group(1) == "tappableElement" else gestures
        frame = (x1, y1, x2, y2)
        if frame not in frames:
            frames.append(frame)
    return (
        InputRegions((left, top, right, bottom), tuple(taps), tuple(gestures))
        if known_source
        else None
    )


class InputRejectedError(ValueError):
    """An app input has no reachable target at its requested display coordinates."""

    def __init__(self, reason: str, regions: InputRegions | None) -> None:
        super().__init__(reason)
        self.regions = regions.as_dict() if regions is not None else None


def input_region_hint(state: Mapping[str, object]) -> str:
    """Give both model coordinate frames the same reachable-target policy."""
    if state.get("input_regions") is None:
        return ""
    return (
        "\nSystem navigation regions consume app inputs. "
        "App taps and holds stay outside system touch regions; "
        "app swipes start outside system gesture regions. "
        "Reveal an obscured control or use an unobscured point on that same control. "
        "When it cannot be reached, choose another app control or wait."
    )


_GESTURE_MS = 1000
_DEFAULT_WAIT_SECONDS = 3

_CLEAR_ALL_ID = "com.android.launcher3:id/clear_all"
# Each swipe moves the recent apps by one app, so Clear all is reached past at most this
# many apps. Every swipe adds a UI dump, about 2 s on a Redroid container.
_CLEAR_ALL_SWIPES = 10
# Long enough for a Redroid container, which renders 15 frames per second, to move the
# recent apps on every swipe.
_RECENT_APPS_SWIPE_MS = 300
_UI_NODE = re.compile(r"<node\b[^>]*>")
_NODE_ATTRIBUTE = re.compile(r'([\w-]+)="([^"]*)"')
_BOUNDS = re.compile(r"\[(-?\d+),(-?\d+)\]\[(-?\d+),(-?\d+)\]")


def _ui_nodes(ui_dump: str) -> list[dict[str, str]]:
    """Return the attributes of each node in ``uiautomator dump`` output, in order.

    uiautomator escapes ``<``, ``>`` and ``"`` inside attribute values, so each node
    tag and each attribute is one match.
    """
    return [dict(_NODE_ATTRIBUTE.findall(node)) for node in _UI_NODE.findall(ui_dump)]


def _shows_recent_apps(ui_dump: str) -> bool:
    """Recognize Launcher3's overview panel by its resource name in any package.

    Launchers built on Launcher3, including the Pixel launcher, share this view name.
    """
    return any(
        node.get("resource-id", "").endswith(":id/overview_panel") for node in _ui_nodes(ui_dump)
    )


def _shows_no_recent_apps(ui_dump: str) -> bool:
    """Return whether ``uiautomator dump`` output shows the recent apps screen with no
    app in it: an ``overview_panel`` node and no ``task_view`` node, an app's card.
    """
    resource_ids = [node.get("resource-id", "") for node in _ui_nodes(ui_dump)]
    return _shows_recent_apps(ui_dump) and not any(":id/task_view" in rid for rid in resource_ids)


def _clear_all_center(ui_dump: str) -> tuple[int, int] | None:
    """Return the centre of the recent apps' Clear all button in ``uiautomator dump``
    output: the first node whose resource id is Launcher3's ``clear_all`` or whose text,
    in any case, is "Clear all". ``None`` when no node is.
    """
    for attributes in _ui_nodes(ui_dump):
        bounds = _BOUNDS.fullmatch(attributes.get("bounds", ""))
        is_clear_all = (
            attributes.get("resource-id") == _CLEAR_ALL_ID
            or attributes.get("text", "").casefold() == "clear all"
        )
        if bounds is not None and is_clear_all:
            left, top, right, bottom = (int(value) for value in bounds.groups())
            return (left + right) // 2, (top + bottom) // 2
    return None


class AdbController:
    """Reads the screen of the device ``config["device"]["udid"]`` and acts on it over adb."""

    def __init__(
        self, config: dict[str, Any], logger: logging.Logger, *, unreachable_limit: float
    ) -> None:
        self.config = config
        self._logger = logger
        self._serial = ["-s", config["device"]["udid"]]
        recovery = StartupRecovery(self, unreachable_limit, logger)
        size = recovery.command("screen size", [*self._serial, "shell", "wm", "size"], check=True)
        match = re.search(r"(\d+)x(\d+)", size.stdout)
        if match is None:
            msg = "Failed to get screen size"
            raise ValueError(msg)
        self.w = int(match.group(1))
        self.h = int(match.group(2))
        self._observed_display: tuple[int, int, int, int] | None = None

    def get_state(self, *, deadline: float | None = None) -> dict[str, Any]:
        """Return the UI hierarchy XML, a base64 PNG screenshot, the focused package and
        activity, explicit system-boundary ownership, clipboard touch regions and the
        ``status_bar`` frame from WindowManager. The window read precedes the image so
        observed feedback ownership survives its dismissal during capture.
        ``deadline`` bounds every read and retry by the remaining monotonic time.
        """
        xml = self._ui_dump(deadline)
        dumpsys = self._read(
            _adb_stdout_text, [*self._serial, "shell", "dumpsys", "window", "-a"], deadline
        )
        png = self._read(_adb_stdout_bytes, [*self._serial, "exec-out", "screencap -p"], deadline)
        package, activity = _focused_package_activity(dumpsys)
        ownership = window_ownership(dumpsys, package, activity)
        requester = None
        if ownership.system_surface == "permission":
            activities = self._read(
                _adb_stdout_text,
                [*self._serial, "shell", "dumpsys", "activity", "activities"],
                deadline,
            )
            requester = permission_requester(activities, dumpsys)
        regions = input_regions(dumpsys)
        if regions is not None:
            self._observed_display = regions.display
            left, top, right, bottom = regions.display
            self.w, self.h = right - left, bottom - top
        return {
            "xml": xml,
            "screenshot": base64.b64encode(png).decode("ascii"),
            "package": ownership.package,
            "activity": ownership.activity,
            "underlying_package": ownership.underlying_package,
            "system_surface": ownership.system_surface,
            "system_overlays": [list(region) for region in ownership.system_overlays],
            "permission_requester": requester,
            "status_bar": status_bar_frame(dumpsys),
            **(
                {"system_overlay_error": ownership.overlay_error} if ownership.overlay_error else {}
            ),
            "input_regions": regions.as_dict() if regions is not None else None,
        }

    def _ui_dump(self, deadline: float | None = None) -> str:
        return self._read(
            _adb_stdout_text, [*self._serial, "exec-out", "uiautomator dump /dev/stdout"], deadline
        )

    def _read[T](
        self, read: Callable[[Sequence[str], float], T], args: Sequence[str], deadline: float | None
    ) -> T:
        """Read with one timeout retry, bounding each attempt by the observation deadline."""
        timeout = _observation_timeout(args, deadline)
        try:
            return read(args, timeout)
        except subprocess.TimeoutExpired:
            self._logger.warning(
                "adb %s timed out after %g s; retrying once", shlex.join(args), timeout
            )
            return read(args, _observation_timeout(args, deadline))

    def exe_action(self, action: dict[str, Any]) -> None:
        """Send ``action``, a device action from ``tool_call_to_action``, to the device.

        A ``long_press`` is held for ``_GESTURE_MS`` whatever its ``duration`` says.
        """
        if action["action"] in {"tap", "long_press", "swipe"}:
            self._check_input(action)
        match action["action"]:
            case "tap":
                self._tap(action["x"], action["y"])
            case "long_press":
                x, y = action["x"], action["y"]
                self._swipe((x, y), (x, y), _GESTURE_MS)
            case "swipe":
                duration = action.get("duration", action.get("time", _GESTURE_MS))
                self._swipe((action["x1"], action["y1"]), (action["x2"], action["y2"]), duration)
            case "type":
                self._clear()
                self._type(action["text"])
            case "enter" | "back" | "home" | "menu" as key:
                self._press(f"KEYCODE_{key.upper()}")
            case "wait":
                time.sleep(action.get("duration", action.get("time", _DEFAULT_WAIT_SECONDS)))
            case "end":
                self._logger.info("Task Finished.")
            case unknown:
                msg = f"Unknown device action: {unknown!r}"
                raise ValueError(msg)

        # A sent action can rotate the display; only the next capture grounds a new plan.
        self._observed_display = None

    def _check_input(self, action: Mapping[str, object]) -> None:
        """Check current system input geometry once, immediately before coordinate input."""
        dumpsys = self._read(
            _adb_stdout_text, [*self._serial, "shell", "dumpsys", "window", "-a"], None
        )
        package, activity = _focused_package_activity(dumpsys)
        ownership = window_ownership(dumpsys, package, activity)
        if ownership.overlay_error is not None:
            raise SystemOverlayActionError(ownership.overlay_error)
        check_overlay_action(action, ownership.system_overlays)
        regions = input_regions(dumpsys)
        if regions is None:
            reason = "System input regions are unavailable; app input is deferred"
            raise InputRejectedError(reason, None)
        if self._observed_display is not None and self._observed_display != regions.display:
            reason = "Display geometry changes after the observation; capture it again"
            raise InputRejectedError(reason, regions)
        reason = regions.reject_reason(action)
        if reason is not None:
            raise InputRejectedError(reason, regions)

    def terminate_all_apps(self, recovery: StartupRecovery) -> None:
        """Open recent apps and tap their Clear all button, which ``_scroll_to_clear_all``
        brings on screen, or press Home when they show no app.

        When Clear all does not appear, log a warning and press Home; the recent apps
        stay.
        """

        def open_recents(deadline: float | None, state: dict[str, Any] | None) -> None:
            if state is None or not _shows_recent_apps(state.get("xml", "")):
                self._press("KEYCODE_APP_SWITCH", deadline=deadline)

        recovery.run("recent apps", open_recents)
        ui_dump = self._scroll_to_clear_all(recovery)
        clear_all = _clear_all_center(ui_dump)
        if clear_all is not None:

            def clear_recents(deadline: float | None, state: dict[str, Any] | None) -> None:
                target = clear_all if state is None else _clear_all_center(state.get("xml", ""))
                if target is not None:
                    self._tap(*target, deadline=deadline)

            recovery.run("clear recent apps", clear_recents)
            time.sleep(1.5)
            return
        if not _shows_no_recent_apps(ui_dump):
            self._logger.warning(
                "Clear all did not appear in recent apps after %d swipes; pressing Home.",
                _CLEAR_ALL_SWIPES,
            )
        recovery.run(
            "home", lambda deadline, _state: self._press("KEYCODE_HOME", deadline=deadline)
        )

    def _scroll_to_clear_all(self, recovery: StartupRecovery) -> str:
        """Return the UI dump of the recent apps on screen once it shows Clear all or no
        app, swiping the list from left to right before each look after the first, at
        most ``_CLEAR_ALL_SWIPES`` times.

        In a left-to-right layout that swipe moves the list toward its oldest app, which
        Clear all follows.
        """
        ui_dump = recovery.run(
            "recent apps observation", lambda deadline, _state: self._ui_dump(deadline)
        )
        middle = int(0.5 * self.h)
        for _ in range(_CLEAR_ALL_SWIPES):
            if _clear_all_center(ui_dump) is not None or _shows_no_recent_apps(ui_dump):
                return ui_dump
            recovery.input(
                "scroll recent apps",
                lambda deadline: self._swipe(
                    (int(0.1 * self.w), middle),
                    (int(0.9 * self.w), middle),
                    _RECENT_APPS_SWIPE_MS,
                    deadline=deadline,
                ),
            )
            time.sleep(1)
            ui_dump = recovery.run(
                "recent apps observation", lambda deadline, _state: self._ui_dump(deadline)
            )
        return ui_dump

    def _input(self, *args: str, check: bool = False, deadline: float | None = None) -> None:
        """Run ``adb shell input`` with ``args``. A timeout raises
        ``subprocess.TimeoutExpired`` at once: the input may have reached the device, and
        sending it again could act twice.
        """
        run_adb(
            [*self._serial, "shell", "input", *args],
            output="discard",
            timeout=min(_INPUT_TIMEOUT_SECONDS, _observation_timeout(args, deadline)),
            check=check,
        )

    def _tap(self, x: int, y: int, *, deadline: float | None = None) -> None:
        self._input("tap", str(x), str(y), deadline=deadline)

    def _swipe(
        self,
        start: tuple[int, int],
        end: tuple[int, int],
        duration: int,
        *,
        deadline: float | None = None,
    ) -> None:
        self._input("swipe", *(str(value) for value in (*start, *end, duration)), deadline=deadline)

    def _press(self, keycode: str, *, deadline: float | None = None) -> None:
        self._input("keyevent", keycode, deadline=deadline)
        time.sleep(1)

    def _clear(self) -> None:
        self._input("keyboard", "keycombination", "KEYCODE_CTRL_LEFT", "KEYCODE_A", check=True)
        self._input("keyboard", "keyevent", "KEYCODE_DEL", check=True)

    def _type(self, text: str) -> None:
        """Type ``text`` with ``input text``, which turns each character into key events
        with the device's virtual keyboard map, adding Shift where one is needed.

        adb joins its arguments into one command line for the device's shell, so the
        text is quoted for that shell. ``input text`` reads ``%s`` as a space, so a
        ``%`` followed by ``s`` ends one call and the ``s`` starts the next. A character
        the map cannot type, such as ``é``, makes adb exit non-zero, which raises
        ``CalledProcessError``.
        """
        for chunk in re.split(r"(?<=%)(?=s)", text):
            if chunk:
                self._input("text", shlex.quote(chunk), check=True)
                time.sleep(1)


class AdbDisplayAccess:
    """Supported display configuration and explicit public seed-directory hashes."""

    def __init__(self, udid: str) -> None:
        self._serial = ["-s", udid]

    def _read(self, command: tuple[str, ...], deadline: float) -> str:
        try:
            return self._run(command, deadline)
        except subprocess.TimeoutExpired:
            logger.warning("Display read %s timed out; retrying once", shlex.join(command))
            return self._run(command, deadline)

    def _run(self, command: tuple[str, ...], deadline: float) -> str:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(command, 0)
        return run_adb(
            [*self._serial, "shell", *(shlex.quote(argument) for argument in command)],
            output="capture",
            check=True,
            timeout=min(_PROBE_TIMEOUT_SECONDS, remaining),
        ).stdout.strip()

    def capture(self, *, deadline: float | None = None) -> DisplayState:
        bound = time.monotonic() + _READ_TIMEOUT_SECONDS if deadline is None else deadline
        size = self._read(("wm", "size"), bound)
        density = self._read(("wm", "density"), bound)
        dump = self._read(("dumpsys", "uimode"), bound)
        custom = re.search(r"customStart=([0-9:.]+)\s+customEnd=?([0-9:.]+)", dump)
        overrides = re.search(r"mOverrideOn/Off=(true|false)/(true|false)", dump)
        if custom is None or overrides is None:
            msg = "The device does not expose supported custom night configuration"
            raise ValueError(msg)
        settings = {key: self._setting("secure", key, bound) for key in NIGHT_KEYS}
        return DisplayState.from_dict(
            {
                "physical_size": self._wm_value(size, "Physical size", pair=True),
                "size_override": self._wm_value(size, "Override size", pair=True),
                "physical_density": self._wm_value(density, "Physical density", pair=False),
                "density_override": self._wm_value(density, "Override density", pair=False),
                "font_scale": self._setting("system", "font_scale", bound),
                "night_mode": self._read(("cmd", "uimode", "night"), bound).removeprefix(
                    "Night mode: "
                ),
                "custom_start": custom.group(1),
                "custom_end": custom.group(2),
                "night_settings": settings,
                "night_overrides": [value == "true" for value in overrides.groups()],
            }
        )

    def _setting(self, namespace: str, key: str, deadline: float) -> str | None:
        value = self._read(("settings", "get", namespace, key), deadline)
        return None if value == "null" else value

    @staticmethod
    def _wm_value(output: str, label: str, *, pair: bool) -> object:
        pattern = rf"(?m)^{label}: (\d+)x(\d+)\s*$" if pair else rf"(?m)^{label}: (\d+)\s*$"
        match = re.search(pattern, output)
        if match is None:
            if label in output:
                msg = f"Malformed display setting: {label}"
                raise ValueError(msg)
            return None
        return [int(match[1]), int(match[2])] if pair else int(match[1])

    def execute(self, command: tuple[str, ...], *, deadline: float) -> None:
        self._run(command, deadline)

    def seed_hashes(self, root: str, *, deadline: float | None = None) -> dict[str, str]:
        bound = time.monotonic() + _READ_TIMEOUT_SECONDS if deadline is None else deadline
        try:
            output = self._read(
                ("find", root, "-type", "f", "-exec", "sha256sum", "{}", "+"), bound
            )
        except subprocess.CalledProcessError as exc:
            if "No such file or directory" in (exc.stderr or ""):
                return {}
            raise
        hashes: dict[str, str] = {}
        for line in output.splitlines():
            match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
            if match is None or not match[2].startswith(root.rstrip("/") + "/"):
                msg = "The seed directory hash output is malformed"
                raise ValueError(msg)
            hashes[match[2]] = match[1]
        return hashes


class _AvdHost(NamedTuple):
    """Where this machine keeps AVDs and the SDK, and the emulator's CPU architecture."""

    avd_root_dir: Path
    sdk_dir: Path
    abi: str
    cpu_arch: str


def _avd_host() -> _AvdHost:
    home = Path.home()
    avd_root_dir = home / ".android" / "avd"
    if not avd_root_dir.exists():
        msg = f"AVD root directory '{avd_root_dir}' not found."
        raise FileNotFoundError(msg)
    match platform.system():
        case "Darwin":
            sdk_dir = home / "Library" / "Android" / "sdk"
            return _AvdHost(avd_root_dir, sdk_dir, abi="arm64-v8a", cpu_arch="arm64")
        case "Linux":
            sdk_dir = home / "Android" / "Sdk"
            return _AvdHost(avd_root_dir, sdk_dir, abi="x86_64", cpu_arch="x86_64")
        case "Windows":
            sdk_dir = home / "AppData" / "Local" / "Android" / "Sdk"
            return _AvdHost(avd_root_dir, sdk_dir, abi="x86_64", cpu_arch="x86_64")
        case system:
            msg = f"Unsupported OS: {system}"
            raise ValueError(msg)


def _rewrite_lines(path: Path, new_lines: Mapping[str, str]) -> None:
    """Replace each line of ``path`` that starts with a key of ``new_lines`` by its value."""
    with path.open(encoding="utf-8") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        for prefix, new_line in new_lines.items():
            if line.startswith(prefix):
                lines[i] = f"{new_line}\n"
                break
    with path.open("w", encoding="utf-8") as f:
        f.writelines(lines)


def _adb_server_address() -> str:
    """Return the socket of the adb server that adb talks to, as adb reads it from the
    environment: ``ADB_SERVER_SOCKET``, or else ``ANDROID_ADB_SERVER_ADDRESS`` and
    ``ANDROID_ADB_SERVER_PORT``, which default to ``localhost`` and ``5037``."""
    host = os.environ.get("ANDROID_ADB_SERVER_ADDRESS") or "localhost"
    port = os.environ.get("ANDROID_ADB_SERVER_PORT") or "5037"
    return os.environ.get("ADB_SERVER_SOCKET") or f"tcp:{host}:{port}"


class AdbServerUnreachableError(ConnectionError):
    """adb cannot reach its server, so which devices run is unknown."""

    def __init__(self, cause: str) -> None:
        super().__init__(f"adb cannot reach its server at {_adb_server_address()}: {cause}")


class EmulatorManager:
    """Lists the devices adb sees and duplicates an AVD for ``check_avd`` to start."""

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def get_running_avd_list(self) -> list[dict[str, Any]]:
        """Return each device on ``adb devices`` with its AVD name, ``None`` for a phone.

        An ``adb devices`` that fails or times out raises ``AdbServerUnreachableError``
        naming the adb server's address and adb's error.
        """
        try:
            devices = run_adb(["devices"], output="capture", check=False)
        except subprocess.TimeoutExpired as exc:
            cause = f"adb devices timed out after {exc.timeout:g} s"
            raise AdbServerUnreachableError(cause) from exc
        if devices.returncode != 0:
            raise AdbServerUnreachableError(_adb_error(devices))
        running: list[dict[str, Any]] = []
        for line in devices.stdout.splitlines()[1:]:
            if not line:
                continue
            udid, status = line.split()
            avd = run_adb(["-s", udid, "emu", "avd", "name"], output="capture", check=False)
            name = avd.stdout.strip() if avd.returncode == 0 else None
            running.append({"udid": udid, "status": status, "name": name})
        return running

    def duplicate_avd(self, avd_name: str) -> None:
        """Copy the AVD ``avd_name`` to ``<avd_name>_dup`` and point the copy's files at
        itself and at this machine's SDK.
        """
        host = _avd_host()
        avd_ini = host.avd_root_dir / f"{avd_name}.ini"
        with avd_ini.open(encoding="utf-8") as f:
            target = next(
                (line.removeprefix("target=").strip() for line in f if line.startswith("target=")),
                "",
            )
        if not target:
            msg = f"AVD metadata '{avd_ini}' requires a non-empty target."
            raise ValueError(msg)
        dup_name = f"{avd_name}_dup"
        shutil.copytree(
            host.avd_root_dir / f"{avd_name}.avd",
            host.avd_root_dir / f"{dup_name}.avd",
            dirs_exist_ok=True,
        )
        self._rewrite_avd_ini(host, dup_name, target)
        self._rewrite_config_ini(host, dup_name)
        self._rewrite_hardware_qemu_ini(host, dup_name)

    def _rewrite_avd_ini(self, host: _AvdHost, avd_name: str, target: str) -> None:
        avd_ini = host.avd_root_dir / f"{avd_name}.ini"
        avd_ini.write_text(
            f"avd.ini.encoding=UTF-8\npath={host.avd_root_dir / f'{avd_name}.avd'}\n"
            f"path.rel=avd/{avd_name}.avd\ntarget={target}",
            encoding="utf-8",
        )
        self._logger.info("AVD init file '%s' modified.", avd_ini)

    def _rewrite_config_ini(self, host: _AvdHost, avd_name: str) -> None:
        config_ini = host.avd_root_dir / f"{avd_name}.avd" / "config.ini"
        _rewrite_lines(
            config_ini,
            {
                "AvdId=": f"AvdId={avd_name}",
                "avd.ini.displayname=": f"avd.ini.displayname={avd_name}",
                "abi.type=": f"abi.type={host.abi}",
                "hw.cpu.arch=": f"hw.cpu.arch={host.cpu_arch}",
                "skin.path=": f"skin.path={host.sdk_dir / 'skins' / 'pixel_7'}",
            },
        )
        self._logger.info("AVD config file '%s' modified.", config_ini)

    def _rewrite_hardware_qemu_ini(self, host: _AvdHost, avd_name: str) -> None:
        avd_dir = host.avd_root_dir / ".." / "avd" / f"{avd_name}.avd"
        hardware_ini = host.avd_root_dir / f"{avd_name}.avd" / "hardware-qemu.ini"
        _rewrite_lines(
            hardware_ini,
            {
                "hw.cpu.arch=": f"hw.cpu.arch={host.abi}",
                "disk.cachePartition.path=": f"disk.cachePartition.path={avd_dir / 'cache.img'}",
                # Known defect, copied unchanged: the next four lines lose their keys.
                "disk.dataPartition.path=": str(avd_dir / "userdata-qemu.img"),
                "disk.encryptionKeyPartition.path=": str(avd_dir / "encryptionkey.img"),
                "android.sdk.root=": str(host.sdk_dir),
                "android.avd.home=": str(host.avd_root_dir),
            },
        )
        self._logger.info("AVD hardware file '%s' modified.", hardware_ini)


def launch_app(
    config: dict[str, Any], app: dict[str, Any], logger: logging.Logger, recovery: StartupRecovery
) -> str:
    """Start ``app``, an entry of ``config["apps"]``, with ``am start -n`` and return the
    component started, which ``AppLaunch.component`` names: its ``launch_activity``, or
    else the package's launcher activity.

    The app's task is left as it is. Raises ``NoLauncherActivityError`` naming the
    package when it has no launcher activity, and ``ActivityStartError`` when ``am
    start`` cannot start the component.
    """
    udid = config["device"]["udid"]
    launch = AppLaunch(app["name"], app["package_name"], app.get("launch_activity"))
    component = recovery.run(
        "launcher resolution",
        lambda deadline, _state: launch.component(udid, logger, deadline=deadline),
    )
    logger.info("Launching app %s via activity %s ...", app["name"], component)

    def start(deadline: float | None, state: dict[str, Any] | None) -> None:
        if state is None or is_outside_app(state.get("package", ""), app["package_name"]):
            _start_activity(udid, component, deadline=deadline)

    recovery.run("app launch", start)
    return component


_LAUNCH_SETTLE_SECONDS = 4
_LAUNCHER_INTENT = ("-a", "android.intent.action.MAIN", "-c", "android.intent.category.LAUNCHER")
# FLAG_ACTIVITY_NEW_TASK | FLAG_ACTIVITY_CLEAR_TASK; the clear flag works only together
# with the new-task flag.
_CLEAR_TASK_FLAGS = "0x10008000"


class ForceStopRefusedError(subprocess.CalledProcessError):
    """adb refused to force-stop the app, so the device still shows what it showed."""


class NoLauncherActivityError(RuntimeError):
    """The package has no launcher activity, so starting it needs a configured
    ``launch_activity``."""


class ActivityStartError(RuntimeError):
    """``am start`` could not start the app's component."""


class AppLaunch:
    """An app that ``launch_app`` or ``restart_app`` starts, and the one component every
    start of it starts.

    Every replay begins at the graph's start node, so every restart has to land on the
    same screen. The component is the given ``launch_activity``, or else the package's
    launcher activity, resolved on first use and reused.
    """

    def __init__(self, name: str, package_name: str, launch_activity: str | None = None) -> None:
        self.name = name
        self.package_name = package_name
        self._component = launch_activity

    def component(self, udid: str, logger: logging.Logger, *, deadline: float | None = None) -> str:
        """Return the component to start, asking ``udid`` for the launcher activity on the
        first call when no ``launch_activity`` is configured."""
        if self._component is None:
            self._component = _launcher_activity(udid, self.package_name, logger, deadline=deadline)
        return self._component


def _launcher_activity(
    udid: str, package_name: str, logger: logging.Logger, *, deadline: float | None = None
) -> str:
    """Return the activity ``pm resolve-activity`` resolves the package's launcher intent
    to, or else the first that ``pm query-activities`` lists for it.

    ``pm resolve-activity`` prints ``No activity found`` for a package with none, and the
    system chooser for one with several. Raises ``NoLauncherActivityError`` when the
    package has none.
    """
    pm = ["-s", udid, "shell", "pm"]
    intent = [*_LAUNCHER_INTENT, package_name]
    resolve = [*pm, "resolve-activity", "--brief", *intent]
    resolved = run_adb(
        resolve, output="capture", check=True, timeout=_observation_timeout(resolve, deadline)
    )
    if components := _package_components(resolved.stdout, package_name):
        return components[0]
    query = [*pm, "query-activities", "--brief", *intent]
    listed = run_adb(
        query, output="capture", check=True, timeout=_observation_timeout(query, deadline)
    )
    components = _package_components(listed.stdout, package_name)
    if not components:
        msg = f"{package_name} has no launcher activity; set launch_activity for it in the config"
        raise NoLauncherActivityError(msg)
    first, *others = components
    if others:
        logger.warning(
            "%s has several launcher activities; starting it through %s, not %s. "
            "Set launch_activity to choose another.",
            package_name,
            first,
            ", ".join(others),
        )
    return first


def _package_components(pm_output: str, package_name: str) -> list[str]:
    """Return the components of ``package_name`` that ``pm ... --brief`` output names."""
    lines = (line.strip() for line in pm_output.splitlines())
    return [line for line in lines if line.startswith(f"{package_name}/")]


def restart_app(controller: DeviceController, app: AppLaunch, logger: logging.Logger) -> None:
    """Put the app on its start screen, where every replay from the start node begins.

    The force-stop discards what the app holds in memory. It leaves another package's
    activity in the app's task, such as a search screen the app opened, and launching an
    app whose task still exists brings that task to the foreground as it was
    (https://developer.android.com/guide/components/activities/tasks-and-back-stack),
    so the launch clears the task. A refused force-stop raises ``ForceStopRefusedError``
    before anything is launched, so no replay starts from an unknown screen. An ``am
    start`` that cannot start the component raises ``ActivityStartError``.
    """
    udid = controller.config["device"]["udid"]
    force_stop = ["-s", udid, "shell", "am", "force-stop", app.package_name]
    stopped = run_adb(force_stop, output="discard", check=False)
    if stopped.returncode != 0:
        raise ForceStopRefusedError(stopped.returncode, stopped.args)
    _start_activity(udid, app.component(udid, logger), "-f", _CLEAR_TASK_FLAGS)
    time.sleep(_LAUNCH_SETTLE_SECONDS)


def _start_activity(udid: str, component: str, *flags: str, deadline: float | None = None) -> None:
    """Start ``component`` on the device ``udid`` with ``am start -n`` and ``flags``.

    Raises ``ActivityStartError`` naming the component and adb's error when ``am start``
    reports that it could not start it.
    """
    args = ["-s", udid, "shell", "am", "start", "-n", component, *flags]
    started = run_adb(
        args, output="capture", check=False, timeout=_observation_timeout(args, deadline)
    )
    if _start_failed(started):
        msg = f"am start could not start {component}: {_adb_error(started)}"
        raise ActivityStartError(msg)


def _start_failed(started: subprocess.CompletedProcess[str]) -> bool:
    """Return whether ``am start`` reported that it could not start the activity.

    Its ``Error`` lines count as well as its exit status: on a device without adb's shell
    protocol, ``adb shell`` exits 0 whatever the command returned.
    """
    lines = f"{started.stdout}\n{started.stderr}".splitlines()
    return started.returncode != 0 or any(line.startswith("Error") for line in lines)


def soft_keyboard_hint(udid: str) -> str:
    """Return a note that the soft keyboard on the device ``udid`` is visible, or ``""``
    otherwise.

    Shells out to adb rather than going through ``DeviceController``, which has no
    shell method.
    """
    try:
        kb_check = run_adb(
            ["-s", udid, "shell", "dumpsys", "input_method"],
            output="capture",
            # Undecodable dumpsys output must never raise UnicodeDecodeError out of a
            # best-effort probe; replace bad bytes instead of failing to decode them.
            errors="replace",
            timeout=_PROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        logger.debug("Soft keyboard probe failed: %s", exc)
        return ""
    return _SOFT_KEYBOARD_HINT if "mInputShown=true" in kb_check.stdout else ""


def keyboard_status(hint: str) -> str:
    """Return the planner-prompt sentence for ``hint`` from ``soft_keyboard_hint``."""
    if hint:
        return "Soft keyboard is visible; identify the intended editable field from the current screen."
    return "OS keyboard visibility is unknown. Inspect the current screen for the intended editable field."


_CHECK_INTERVAL_SECONDS = 30


class DeviceUnreachableError(RuntimeError):
    """The device supplied no complete observation within the recovery deadline."""


def _observation_timeout(args: Sequence[str], deadline: float | None) -> float:
    if deadline is None:
        return _READ_TIMEOUT_SECONDS
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise subprocess.TimeoutExpired(args, 0)
    return min(_READ_TIMEOUT_SECONDS, remaining)


def _remaining_recovery_time(udid: str, deadline: float, limit: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        msg = f"Device {udid} unreachable for {limit / 60:g} minutes"
        raise DeviceUnreachableError(msg)
    return remaining


@dataclass(slots=True)
class _RecoveryEpisode:
    deadline: float
    confirmation_used: bool = False


def wait_for_device_state(
    controller: DeviceController,
    limit: float,
    logger: logging.Logger,
    *,
    episode: _RecoveryEpisode | None = None,
) -> dict[str, Any]:
    """Recover a complete observation within one deadline after a device outage.

    Only the observation reads qualify for nonzero recovery. One unclassified failure
    gets a confirmation observation; the allowance belongs to the entire episode.
    """
    udid = controller.config["device"]["udid"]
    started = time.monotonic()
    episode = episode or _RecoveryEpisode(started + limit)
    deadline = episode.deadline
    while True:
        remaining = _remaining_recovery_time(udid, deadline, limit)
        if _answers(udid, min(_PROBE_TIMEOUT_SECONDS, remaining)):
            try:
                state = controller.get_state(deadline=deadline)
            except subprocess.TimeoutExpired as timeout:
                logger.warning("Device %s observation is unavailable: %s", udid, timeout)
            except subprocess.CalledProcessError as failure:
                policy = _recovery_read_failure(udid, failure, logger)
                if policy == "stop" or (policy == "confirm" and episode.confirmation_used):
                    raise
                episode.confirmation_used |= policy == "confirm"
            else:
                _remaining_recovery_time(udid, deadline, limit)
                logger.info(
                    "Device %s observation recovered after %.0f s. Continuing from its screen.",
                    udid,
                    time.monotonic() - started,
                )
                return state
        remaining = _remaining_recovery_time(udid, deadline, limit)
        time.sleep(min(_CHECK_INTERVAL_SECONDS, remaining))


class StartupRecovery:
    """Resume one startup phase within its device-outage deadline.

    Reads and idempotent commands retry their phase. Inputs with uncertain completion
    yield the recovered screen to their caller, which inspects it before continuing.
    """

    def __init__(self, controller: DeviceController, limit: float, logger: logging.Logger) -> None:
        self.controller = controller
        self.limit = limit
        self.logger = logger

    def run[T](
        self, phase: str, operation: Callable[[float | None, dict[str, Any] | None], T]
    ) -> T:
        udid = self.controller.config["device"]["udid"]
        episode: _RecoveryEpisode | None = None
        state: dict[str, Any] | None = None
        while True:
            deadline = episode.deadline if episode is not None else None
            if deadline is not None:
                _remaining_recovery_time(udid, deadline, self.limit)
            try:
                result = operation(deadline, state)
            except subprocess.TimeoutExpired as timeout:
                if episode is None:
                    episode = _RecoveryEpisode(time.monotonic() + self.limit)
                self.logger.warning(
                    "Device %s is unreachable during startup %s: %s. Waiting up to %g minutes.",
                    udid,
                    phase,
                    timeout,
                    self.limit / 60,
                )
                state = wait_for_device_state(
                    self.controller, self.limit, self.logger, episode=episode
                )
            else:
                if deadline is not None:
                    _remaining_recovery_time(udid, deadline, self.limit)
                return result

    def command(
        self, phase: str, args: list[str], *, check: bool
    ) -> subprocess.CompletedProcess[str]:
        return self.run(
            phase,
            lambda deadline, _state: run_adb(
                args, output="capture", check=check, timeout=_observation_timeout(args, deadline)
            ),
        )

    def input(self, phase: str, action: Callable[[float | None], None]) -> None:
        self.run(phase, lambda deadline, state: action(deadline) if state is None else None)


def _answers(udid: str, timeout: float) -> bool:
    try:
        probe = run_adb(
            ["-s", udid, "shell", "true"], output="discard", timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired:
        return False
    return probe.returncode == 0


def _read_phase(command: object, udid: str) -> str | None:
    match command:
        case [str(), "-s", serial, "exec-out", "uiautomator dump /dev/stdout"] if serial == udid:
            return "hierarchy"
        case [str(), "-s", serial, "exec-out", "screencap -p"] if serial == udid:
            return "screenshot"
        case [str(), "-s", serial, "shell", "dumpsys", "window", "-a"] if serial == udid:
            return "window"
        case [str(), "-s", serial, "shell", "dumpsys", "activity", "activities"] if serial == udid:
            return "activity"
        case [str(), "-s", serial, "shell", *arguments] if serial == udid:
            display_reads = (
                ("wm", "size"),
                ("wm", "density"),
                ("dumpsys", "uimode"),
                ("cmd", "uimode", "night"),
                ("settings", "get", "system", "font_scale"),
                *(("settings", "get", "secure", key) for key in NIGHT_KEYS),
            )
            return "display" if tuple(arguments) in display_reads else None
        case _:
            return None


_DIAGNOSTIC_LIMIT = 1024
_ERROR_PREFIXES = (
    "error",
    "adb:",
    "exception",
    "java.",
    "android.os.",
    "permission",
    "securityexception",
    "invalid",
    "unknown",
    "unsupported",
    "can't find service:",
    "usage:",
)


def _diagnostic_text(value: object) -> str:
    if isinstance(value, bytes):
        return value[:_DIAGNOSTIC_LIMIT].decode("utf-8", errors="replace").strip()
    return value[:_DIAGNOSTIC_LIMIT].strip() if isinstance(value, str) else ""


def _read_diagnostic(failure: subprocess.CalledProcessError, phase: str) -> str:
    diagnostic = _diagnostic_text(failure.stderr)
    if not diagnostic and phase != "screenshot":
        diagnostic = "\n".join(
            line
            for line in _diagnostic_text(failure.stdout).splitlines()
            if line.lstrip().lower().startswith(_ERROR_PREFIXES)
        )
    return "".join(character if character.isprintable() else " " for character in diagnostic)


def _recovery_read_failure(
    udid: str, failure: subprocess.CalledProcessError, logger: logging.Logger
) -> Literal["retry", "confirm", "stop"]:
    phase = _read_phase(failure.cmd, udid)
    if phase is None:
        return "stop"
    diagnostic = _read_diagnostic(failure, phase)
    logger.warning(
        "Device %s recovery %s read: %s; exit %d; %s",
        udid,
        phase,
        shlex.join(failure.cmd),
        failure.returncode,
        diagnostic or "diagnostics absent",
    )
    lowered = diagnostic.lower()
    if any(
        marker in lowered
        for marker in (
            "permission denied",
            "permission denial",
            "device unauthorized",
            "securityexception",
            "invalid option",
            "invalid argument",
            "inaccessible or not found",
            "command not found",
            "not supported",
            "unknown option",
            "unknown command",
            "unsupported",
            "can't find service:",
            "usage:",
        )
    ):
        return "stop"
    # These failures can clear as the transport or system service returns during the outage.
    if any(
        marker in lowered
        for marker in (
            "error: device offline",
            "error: closed",
            "android.os.deadobjectexception",
        )
    ):
        return "retry"
    return "confirm"


def is_outside_app(package: str, app_package: str) -> bool:
    """Return whether the observed owner belongs to a destination outside the target app.

    Attached keyboards retain the underlying app owner at capture; independently
    navigable system windows have their own owner.
    """
    return bool(package) and package != app_package


class StillOutsideAppError(RuntimeError):
    """Another app is still in front after the app was restarted, so a step taken now
    would act on that app and record its screens as the app's."""


def restart_into_app(
    controller: DeviceController, app: AppLaunch, logger: logging.Logger
) -> dict[str, Any]:
    """Restart the app with ``restart_app`` and return the state of the screen it shows.

    Raises ``StillOutsideAppError`` naming the package in front when it still belongs
    to another app.
    """
    restart_app(controller, app, logger)
    state = controller.get_state()
    package = state.get("package", "")
    if is_outside_app(package, app.package_name):
        msg = f"{package} is still in front after restarting {app.package_name}"
        raise StillOutsideAppError(msg)
    return state


def return_to_app(
    controller: DeviceController,
    app: AppLaunch,
    logger: logging.Logger,
    *,
    on_restart: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Press Back after a step opened another app and return the state of the app's
    screen, restarting the app with ``restart_into_app`` when Back does not return to it.

    ``on_restart`` runs before the restart so callers can reset their expected screen.

    Back pops only the top activity, so the other app can stay in front, for example
    when Back only closed its keyboard. A launch would then resume the app's task with
    the other app's activity still on top, which the restart avoids by clearing the task.
    """
    controller.exe_action({"action": "back"})
    time.sleep(1)
    state = controller.get_state()
    package = state.get("package", "")
    if not is_outside_app(package, app.package_name):
        return state
    logger.warning(
        "Back left %s in front of %s. Restarting %s.", package, app.package_name, app.name
    )
    if on_restart is not None:
        on_restart()
    return restart_into_app(controller, app, logger)


class PermissionGrants(NamedTuple):
    """The target package's declared permissions and successful authorized grants."""

    requested: tuple[str, ...]
    granted: tuple[str, ...]


class UnresolvedPermissionError(RuntimeError):
    """A permission request needs a decision outside the run's verified grant policy."""


def resolve_permission_prompt(
    controller: DeviceController,
    app: AppLaunch,
    state: dict[str, Any],
    on_event: Callable[[Mapping[str, object]], None],
    recovery: StartupRecovery,
) -> dict[str, Any]:
    """Resolve one standard target-app runtime prompt under its existing grant policy.

    Caller identity comes from the focused permission activity's launch record. The
    standard Allow button only closes a request whose declared runtime permissions
    the existing policy already grants. Unknown requests stop with evidence.
    """
    requester = state.get("permission_requester")
    on_event({"status": "permission_observed", "requester": requester, "target": app.package_name})
    if requester != app.package_name:
        msg = f"Permission requester {requester!r} is not verified as target {app.package_name}"
        on_event({"status": "permission_unresolved", "reason": msg})
        raise UnresolvedPermissionError(msg)
    grants = grant_runtime_permissions(
        controller.config["device"]["udid"], app.package_name, recovery
    )
    result = run_adb(
        ["-s", controller.config["device"]["udid"], "shell", "dumpsys", "package", "permissions"],
        output="capture",
        check=True,
    )
    permission_records = re.split(r"(?=^  Permission \[)", result.stdout, flags=re.MULTILINE)
    runtime = {
        match.group(1)
        for record in permission_records
        if re.search(r"\bprot=dangerous(?:\W|$)", record)
        and (match := re.search(r"Permission \[([\w.]+)\]", record)) is not None
    }
    requested = runtime.intersection(grants.requested)
    on_event(
        {
            "status": "permission_grants",
            "requested": list(grants.requested),
            "runtime_requested": sorted(requested),
            "granted": list(grants.granted),
        }
    )
    if not requested or not requested.issubset(grants.granted):
        msg = (
            "Target runtime permission request is not fully covered by successful authorized grants"
        )
        on_event({"status": "permission_unresolved", "reason": msg})
        raise UnresolvedPermissionError(msg)
    state = controller.get_state()
    if state.get("system_surface") != "permission":
        return _resolved_permission_state(state, app.package_name, on_event)
    if state.get("permission_requester") == app.package_name:
        for node in _ui_nodes(state.get("xml", "")):
            resource = node.get("resource-id", "").partition(":id/")[2]
            if not node.get("resource-id", "").startswith(
                "com.android.permissioncontroller:id/"
            ) or resource not in {
                "permission_allow_button",
                "permission_allow_foreground_only_button",
            }:
                continue
            bounds = _BOUNDS.fullmatch(node.get("bounds", ""))
            if bounds is None or node.get("enabled") != "true":
                continue
            left, top, right, bottom = (int(v) for v in bounds.groups())
            if left >= right or top >= bottom:
                continue
            controller.exe_action(
                {"action": "tap", "x": (left + right) // 2, "y": (top + bottom) // 2}
            )
            state = controller.get_state()
            if state.get("system_surface") != "permission":
                return _resolved_permission_state(state, app.package_name, on_event)
            break
    msg = "Target permission dialog remains unresolved after its authorized grants"
    on_event({"status": "permission_unresolved", "reason": msg})
    raise UnresolvedPermissionError(msg)


def _resolved_permission_state(
    state: dict[str, Any], target: str, on_event: Callable[[Mapping[str, object]], None]
) -> dict[str, Any]:
    if state.get("package") != target:
        msg = f"Permission recovery reached {state.get('package')!r} instead of target {target}"
        on_event({"status": "permission_unresolved", "reason": msg})
        raise UnresolvedPermissionError(msg)
    on_event({"status": "permission_resolved", "package": target})
    return state


def grant_runtime_permissions(
    udid: str, package_name: str, recovery: StartupRecovery
) -> PermissionGrants:
    """Grant ``package_name`` every permission it requests.

    Startup and target-prompt recovery use the same declared-permission policy.
    ``pm grant`` refuses permissions such as ``INTERNET`` that require no runtime
    grant; each refusal is logged, and the result retains the successful grants.
    """
    shell = ["-s", udid, "shell"]
    dumpsys = recovery.command(
        "requested permissions", [*shell, "dumpsys", "package", package_name], check=True
    )
    permissions = _requested_permissions(dumpsys.stdout)
    granted: list[str] = []
    for permission in permissions:
        grant = recovery.command(
            f"permission grant {permission}",
            [*shell, "pm", "grant", package_name, permission],
            check=False,
        )
        if grant.returncode == 0:
            granted.append(permission)
        else:
            logger.info("Could not grant %s to %s: %s", permission, package_name, _adb_error(grant))
    logger.info(
        "Granted %d of %d requested permissions to %s", len(granted), len(permissions), package_name
    )
    return PermissionGrants(tuple(permissions), tuple(granted))


def _requested_permissions(dumpsys_output: str) -> list[str]:
    """Return the entries under ``requested permissions:`` in ``dumpsys package`` output,
    without suffixes such as ``: restricted=true``.
    """
    permissions: list[str] = []
    section_indent: int | None = None
    for line in dumpsys_output.splitlines():
        entry = line.strip()
        indent = len(line) - len(line.lstrip())
        if entry == "requested permissions:":
            section_indent = indent
        elif section_indent is not None and entry and indent > section_indent:
            permissions.append(entry.split(":", maxsplit=1)[0])
        else:
            section_indent = None
    return permissions


def _adb_error(result: subprocess.CompletedProcess[str]) -> str:
    """Return adb's error output on one line, without the Java stack frames ``pm`` prints."""
    lines = (line.strip() for line in (result.stderr or result.stdout).splitlines())
    return " ".join(line for line in lines if line and not line.startswith("at "))
