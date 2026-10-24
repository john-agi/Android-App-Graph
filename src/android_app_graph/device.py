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
from pathlib import Path
from typing import Any, Literal, NamedTuple, Protocol

from android_app_graph.payloads import as_int

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


def _shows_no_recent_apps(ui_dump: str) -> bool:
    """Return whether ``uiautomator dump`` output shows the recent apps screen with no
    app in it: an ``overview_panel`` node and no ``task_view`` node, an app's card.

    The ids are matched by name in any package, as Launcher3 and the launchers built on
    it, such as the Pixel launcher, name these views alike.
    """
    resource_ids = [node.get("resource-id", "") for node in _ui_nodes(ui_dump)]
    shows_recent_apps = any(rid.endswith(":id/overview_panel") for rid in resource_ids)
    return shows_recent_apps and not any(":id/task_view" in rid for rid in resource_ids)


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

    def __init__(self, config: dict[str, Any], logger: logging.Logger) -> None:
        self.config = config
        self._logger = logger
        self._serial = ["-s", config["device"]["udid"]]
        size = run_adb([*self._serial, "shell", "wm", "size"], output="capture", check=True)
        match = re.search(r"(\d+)x(\d+)", size.stdout)
        if match is None:
            msg = "Failed to get screen size"
            raise ValueError(msg)
        self.w = int(match.group(1))
        self.h = int(match.group(2))
        self._observed_display: tuple[int, int, int, int] | None = None

    def get_state(self, *, deadline: float | None = None) -> dict[str, Any]:
        """Return the UI hierarchy XML, a base64 PNG screenshot, the focused package and
        activity, both ``"unknown"`` when ``dumpsys window`` names none, and the
        ``status_bar`` frame that ``status_bar_frame`` reads from the same dump.
        ``deadline`` bounds every read and retry by the remaining monotonic time.
        """
        xml = self._ui_dump(deadline)
        png = self._read(_adb_stdout_bytes, [*self._serial, "exec-out", "screencap -p"], deadline)
        dumpsys = self._read(
            _adb_stdout_text, [*self._serial, "shell", "dumpsys", "window"], deadline
        )
        package, activity = _focused_package_activity(dumpsys)
        regions = input_regions(dumpsys)
        if regions is not None:
            self._observed_display = regions.display
            left, top, right, bottom = regions.display
            self.w, self.h = right - left, bottom - top
        return {
            "xml": xml,
            "screenshot": base64.b64encode(png).decode("ascii"),
            "package": package,
            "activity": activity,
            "status_bar": status_bar_frame(dumpsys),
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
                self._swipe(x, y, x, y, _GESTURE_MS)
            case "swipe":
                duration = action.get("duration", action.get("time", _GESTURE_MS))
                self._swipe(action["x1"], action["y1"], action["x2"], action["y2"], duration)
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
        dumpsys = self._read(_adb_stdout_text, [*self._serial, "shell", "dumpsys", "window"], None)
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

    def terminate_all_apps(self) -> None:
        """Open recent apps and tap their Clear all button, which ``_scroll_to_clear_all``
        brings on screen, or press Home when they show no app.

        When Clear all does not appear, log a warning and press Home; the recent apps
        stay.
        """
        self._press("KEYCODE_APP_SWITCH")
        ui_dump = self._scroll_to_clear_all()
        clear_all = _clear_all_center(ui_dump)
        if clear_all is not None:
            self._tap(*clear_all)
            time.sleep(1.5)
            return
        if not _shows_no_recent_apps(ui_dump):
            self._logger.warning(
                "Clear all did not appear in recent apps after %d swipes; pressing Home.",
                _CLEAR_ALL_SWIPES,
            )
        self._press("KEYCODE_HOME")

    def _scroll_to_clear_all(self) -> str:
        """Return the UI dump of the recent apps on screen once it shows Clear all or no
        app, swiping the list from left to right before each look after the first, at
        most ``_CLEAR_ALL_SWIPES`` times.

        In a left-to-right layout that swipe moves the list toward its oldest app, which
        Clear all follows.
        """
        ui_dump = self._ui_dump()
        for _ in range(_CLEAR_ALL_SWIPES):
            if _clear_all_center(ui_dump) is not None or _shows_no_recent_apps(ui_dump):
                return ui_dump
            middle = int(0.5 * self.h)
            self._swipe(int(0.1 * self.w), middle, int(0.9 * self.w), middle, _RECENT_APPS_SWIPE_MS)
            time.sleep(1)
            ui_dump = self._ui_dump()
        return ui_dump

    def _input(self, *args: str, check: bool = False) -> None:
        """Run ``adb shell input`` with ``args``. A timeout raises
        ``subprocess.TimeoutExpired`` at once: the input may have reached the device, and
        sending it again could act twice.
        """
        run_adb(
            [*self._serial, "shell", "input", *args],
            output="discard",
            timeout=_INPUT_TIMEOUT_SECONDS,
            check=check,
        )

    def _tap(self, x: int, y: int) -> None:
        self._input("tap", str(x), str(y))

    def _swipe(self, x1: int, y1: int, x2: int, y2: int, duration: int) -> None:
        self._input("swipe", str(x1), str(y1), str(x2), str(y2), str(duration))

    def _press(self, keycode: str) -> None:
        self._input("keyevent", keycode)
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


def launch_app(config: dict[str, Any], app: dict[str, Any], logger: logging.Logger) -> str:
    """Start ``app``, an entry of ``config["apps"]``, with ``am start -n`` and return the
    component started, which ``AppLaunch.component`` names: its ``launch_activity``, or
    else the package's launcher activity.

    The app's task is left as it is. Raises ``NoLauncherActivityError`` naming the
    package when it has no launcher activity, and ``ActivityStartError`` when ``am
    start`` cannot start the component.
    """
    udid = config["device"]["udid"]
    launch = AppLaunch(app["name"], app["package_name"], app.get("launch_activity"))
    component = launch.component(udid, logger)
    logger.info("Launching app %s via activity %s ...", app["name"], component)
    _start_activity(udid, component)
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

    def component(self, udid: str, logger: logging.Logger) -> str:
        """Return the component to start, asking ``udid`` for the launcher activity on the
        first call when no ``launch_activity`` is configured."""
        if self._component is None:
            self._component = _launcher_activity(udid, self.package_name, logger)
        return self._component


def _launcher_activity(udid: str, package_name: str, logger: logging.Logger) -> str:
    """Return the activity ``pm resolve-activity`` resolves the package's launcher intent
    to, or else the first that ``pm query-activities`` lists for it.

    ``pm resolve-activity`` prints ``No activity found`` for a package with none, and the
    system chooser for one with several. Raises ``NoLauncherActivityError`` when the
    package has none.
    """
    pm = ["-s", udid, "shell", "pm"]
    intent = [*_LAUNCHER_INTENT, package_name]
    resolved = run_adb([*pm, "resolve-activity", "--brief", *intent], output="capture", check=True)
    if components := _package_components(resolved.stdout, package_name):
        return components[0]
    listed = run_adb([*pm, "query-activities", "--brief", *intent], output="capture", check=True)
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


def _start_activity(udid: str, component: str, *flags: str) -> None:
    """Start ``component`` on the device ``udid`` with ``am start -n`` and ``flags``.

    Raises ``ActivityStartError`` naming the component and adb's error when ``am start``
    reports that it could not start it.
    """
    started = run_adb(
        ["-s", udid, "shell", "am", "start", "-n", component, *flags],
        output="capture",
        check=False,
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


def wait_for_device_state(
    controller: DeviceController, limit: float, logger: logging.Logger
) -> dict[str, Any]:
    """Recover a complete observation within one deadline after an exploration outage.

    Only the observation reads qualify for nonzero recovery. One unclassified failure
    gets a confirmation observation; the allowance belongs to the entire episode.
    """
    udid = controller.config["device"]["udid"]
    started = time.monotonic()
    deadline = started + limit
    confirmation_used = False
    while True:
        remaining = _remaining_recovery_time(udid, deadline, limit)
        if _answers(udid, min(_PROBE_TIMEOUT_SECONDS, remaining)):
            try:
                state = controller.get_state(deadline=deadline)
            except subprocess.TimeoutExpired as timeout:
                logger.warning("Device %s observation is unavailable: %s", udid, timeout)
            except subprocess.CalledProcessError as failure:
                policy = _recovery_read_failure(udid, failure, logger)
                if policy == "stop" or (policy == "confirm" and confirmation_used):
                    raise
                confirmation_used |= policy == "confirm"
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
        case [str(), "-s", serial, "shell", "dumpsys", "window"] if serial == udid:
            return "window"
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


_SYSTEM_OVERLAY_PACKAGES = frozenset(
    {
        "com.google.android.inputmethod.latin",
        "com.android.inputmethod.latin",
        "com.samsung.android.honeyboard",
        "com.swiftkey.swiftkey",
        "com.touchtype.swiftkey",
        "com.android.systemui",
        "com.android.permissioncontroller",
        "com.google.android.permissioncontroller",
        "com.android.packageinstaller",
        "android",
    }
)


def is_system_overlay(package: str) -> bool:
    """Return whether ``package`` is a keyboard, the status bar or a system dialog.

    These take focus on top of the app without navigating away from it, so the
    explorer records their screens, such as a permission prompt, as part of the app.
    """
    if package in _SYSTEM_OVERLAY_PACKAGES:
        return True
    lower = package.lower()
    return "inputmethod" in lower or "keyboard" in lower


def is_outside_app(package: str, app_package: str) -> bool:
    """Return whether ``package``, the package the device reports in front, belongs to
    an app other than ``app_package``.

    A system overlay sits on top of the app without leaving it, and an empty package
    names no app.
    """
    return bool(package) and package != app_package and not is_system_overlay(package)


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
    controller: DeviceController, app: AppLaunch, logger: logging.Logger
) -> dict[str, Any]:
    """Press Back after a step opened another app and return the state of the app's
    screen, restarting the app with ``restart_into_app`` when Back does not return to it.

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
    return restart_into_app(controller, app, logger)


def grant_runtime_permissions(udid: str, package_name: str) -> None:
    """Grant ``package_name`` every permission it requests.

    A permission prompt that appears during exploration would be dismissed, which
    denies the permission and leaves the feature behind it unexplored. ``pm grant``
    refuses permissions that are not runtime permissions, such as ``INTERNET``, so
    a refusal is only logged.
    """
    shell = ["-s", udid, "shell"]
    dumpsys = run_adb([*shell, "dumpsys", "package", package_name], output="capture", check=True)
    permissions = _requested_permissions(dumpsys.stdout)
    granted = 0
    for permission in permissions:
        grant = run_adb(
            [*shell, "pm", "grant", package_name, permission], output="capture", check=False
        )
        if grant.returncode == 0:
            granted += 1
        else:
            logger.info("Could not grant %s to %s: %s", permission, package_name, _adb_error(grant))
    logger.info(
        "Granted %d of %d requested permissions to %s", granted, len(permissions), package_name
    )


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
