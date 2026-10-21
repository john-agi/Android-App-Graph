"""The device: what exploration and the audit need from it, and adb behind that.

``DeviceController`` and ``AvdManager`` state what the explorer, the audit and
``check_avd`` read from a device and do to it. ``AdbController`` and
``EmulatorManager`` provide them over adb and the emulator's AVD files; tests pass
fakes instead, so they need neither adb nor an emulator.
"""

from __future__ import annotations

import base64
import logging
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

from android_app_graph.ui_tree import (
    IDLE_FAILURE,
    UiTree,
    UnusableDumpError,
    parse_tree,
    same_screen,
    xml_document,
)

logger = logging.getLogger(__name__)

_SOFT_KEYBOARD_HINT = (
    " (Note: the soft keyboard is currently visible — a text field is focused "
    "and ready for typing.)"
)


class DeviceController(Protocol):
    """The screen size, the run's config, the current screen and a way to act on it."""

    w: int
    h: int
    config: dict[str, Any]

    def get_state(self) -> dict[str, Any]: ...

    def exe_action(self, action: dict[str, Any]) -> None: ...


class AvdManager(Protocol):
    """What ``check_avd`` needs to find a running device or start an emulator."""

    def get_running_avd_list(self) -> list[dict[str, Any]] | None: ...

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
    timeout: float | None = None,
    check: bool,
) -> subprocess.CompletedProcess[str]:
    """Run the adb on ``PATH`` with ``args``, capturing its stdout and stderr as text or
    discarding both.

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


# AOSP's InsetsSource.dump(): the type is "statusBars" from API 34, "ITYPE_STATUS_BAR" before.
_STATUS_BAR_SOURCE = re.compile(
    r"InsetsSource\b.*?\btype=(?:statusBars|ITYPE_STATUS_BAR)\b"
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


# About one adb call in a hundred stalls between the host and the device, sometimes
# for good, and the next call succeeds. The slowest legitimate read is a dump that
# gives up waiting for an idle screen, after about 12 s.
_READ_TIMEOUT_SECONDS = 30

_GESTURE_MS = 1000
_DEFAULT_WAIT_SECONDS = 3
_CLEAR_KEY_PRESSES = 20
# The first release whose uiautomator dumps every window (Build.VERSION_CODES.R).
_MIN_API_LEVEL = 30
_UI_DUMP = "uiautomator dump --windows /dev/stdout"
_MAX_RETAKES = 2
_DUMP_TIMED_OUT = "adb timed out twice"
_DUMP_ERRORS_NOT_RETRIED = frozenset({IDLE_FAILURE, _DUMP_TIMED_OUT})


@dataclass(frozen=True, slots=True)
class _Dump:
    """One ``uiautomator dump``: its XML document and tree, or the reason it has none."""

    document: str | None
    tree: UiTree | None
    error: str | None


@dataclass(frozen=True, slots=True)
class _PairedCapture:
    """A screenshot, the dump kept with it, how many times the screenshot was retaken
    and whether the dumps around the kept screenshot agreed."""

    png: bytes
    dump: _Dump
    retakes: int
    settled: bool


class AdbController:
    """Reads the screen of the device ``config["device"]["udid"]`` and acts on it over adb."""

    def __init__(self, config: dict[str, Any], logger: logging.Logger) -> None:
        self.config = config
        self._logger = logger
        udid = config["device"]["udid"]
        self._serial = ["-s", udid]
        sdk = run_adb(
            [*self._serial, "shell", "getprop", "ro.build.version.sdk"],
            output="capture",
            check=True,
        ).stdout.strip()
        if not re.fullmatch(r"[0-9]+", sdk):
            msg = f"Failed to read the API level of {udid}: {sdk!r}"
            raise ValueError(msg)
        if int(sdk) < _MIN_API_LEVEL:
            msg = (
                f"{udid} runs API level {sdk}; capturing every window with uiautomator "
                f"needs API level {_MIN_API_LEVEL} (Android 11) or later"
            )
            raise RuntimeError(msg)
        size = run_adb([*self._serial, "shell", "wm", "size"], output="capture", check=True)
        match = re.search(r"(\d+)x(\d+)", size.stdout)
        if match is None:
            msg = "Failed to get screen size"
            raise ValueError(msg)
        self.w = int(match.group(1))
        self.h = int(match.group(2))

    def get_state(self) -> dict[str, Any]:
        """Return a base64 PNG screenshot with the accessibility tree of every window
        paired with it, the focused package and activity, both ``"unknown"`` when
        ``dumpsys window`` names none, and the ``status_bar`` frame that
        ``status_bar_frame`` reads from the same dump.

        ``"xml"`` is the kept dump's XML document, or ``None`` with the reason in
        ``"ui_tree_error"``; ``"ui_tree_retakes"`` counts the screenshot's retakes and
        ``"ui_tree_settled"`` says whether the dumps around it agreed.
        """
        capture = self._paired_capture()
        dumpsys = self._read(_adb_stdout_text, [*self._serial, "shell", "dumpsys", "window"])
        package, activity = _focused_package_activity(dumpsys)
        return {
            "xml": capture.dump.document,
            "ui_tree_error": capture.dump.error,
            "ui_tree_retakes": capture.retakes,
            "ui_tree_settled": capture.settled,
            "screenshot": base64.b64encode(capture.png).decode("ascii"),
            "package": package,
            "activity": activity,
            "status_bar": status_bar_frame(dumpsys),
        }

    def _paired_capture(self) -> _PairedCapture:
        """Dump, take the screenshot, and dump again; while the two dumps show different
        screens, take the screenshot and the dump again, at most ``_MAX_RETAKES`` times,
        the last dump becoming the first of the new pair.

        The dump waits for one quiet second before it reads the screen, so the
        screenshot right after it usually shows the settled screen; the second dump
        confirms nothing changed in between. The kept dump is the one before the kept
        screenshot. Without a first dump there is no pair, and the screenshot is kept
        without a tree.
        """
        before = self._dump()
        png = self._screencap()
        retakes = 0
        while before.tree is not None:
            after = self._dump()
            if after.tree is None:
                break
            if same_screen(before.tree, after.tree):
                return _PairedCapture(png, before, retakes, settled=True)
            if retakes == _MAX_RETAKES:
                break
            retakes += 1
            png = self._screencap()
            before = after
        return _PairedCapture(png, before, retakes, settled=False)

    def _screencap(self) -> bytes:
        return self._read(_adb_stdout_bytes, [*self._serial, "exec-out", "screencap -p"])

    def _dump(self) -> _Dump:
        """Dump every window, and once more when the dump is unusable, except after an
        idle failure or a timeout.

        An idle failure has already waited ten seconds for a quiet second, and a dump
        right after it fails the same way; ``_read`` has already run a dump that timed
        out once more.
        """
        dump = self._dump_once()
        if dump.tree is None and dump.error not in _DUMP_ERRORS_NOT_RETRIED:
            dump = self._dump_once()
        return dump

    def _dump_once(self) -> _Dump:
        try:
            output = self._read(_adb_stdout_bytes, [*self._serial, "exec-out", _UI_DUMP])
        except subprocess.TimeoutExpired:
            return _Dump(None, None, _DUMP_TIMED_OUT)
        except subprocess.CalledProcessError as exc:
            return _Dump(None, None, f"adb exited with status {exc.returncode}")
        try:
            text = output.decode("utf-8")
        except UnicodeDecodeError:
            return _Dump(None, None, "output is not UTF-8")
        try:
            tree = parse_tree(text)
        except UnusableDumpError as exc:
            return _Dump(None, None, exc.reason)
        return _Dump(xml_document(text), tree, None)

    def _read[T](self, read: Callable[[Sequence[str], float], T], args: Sequence[str]) -> T:
        """Return ``read(args, timeout)``, logging a warning and running it once more
        when it times out.

        A second timeout raises ``subprocess.TimeoutExpired``, which names the command
        and the timeout.
        """
        try:
            return read(args, _READ_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            self._logger.warning(
                "adb %s timed out after %g s; retrying once",
                shlex.join(args),
                _READ_TIMEOUT_SECONDS,
            )
            return read(args, _READ_TIMEOUT_SECONDS)

    def exe_action(self, action: dict[str, Any]) -> None:
        """Send ``action``, a device action from ``tool_call_to_action``, to the device.

        A ``long_press`` is held for ``_GESTURE_MS`` whatever its ``duration`` says.
        """
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
            case "enter" | "back" | "home" as key:
                self._press(f"KEYCODE_{key.upper()}")
            case "wait":
                time.sleep(action.get("duration", action.get("time", _DEFAULT_WAIT_SECONDS)))
            case "end":
                self._logger.info("Task Finished.")
            case unknown:
                msg = f"Unknown device action: {unknown!r}"
                raise ValueError(msg)

    def terminate_all_apps(self) -> None:
        """Open recent apps, swipe the list to its left end and tap where the launcher
        shows Clear all.
        """
        self._press("KEYCODE_APP_SWITCH")
        self._swipe(int(0.1 * self.w), int(0.5 * self.h), int(0.9 * self.w), int(0.5 * self.h), 100)
        time.sleep(1)
        self._tap(int(0.2 * self.w), int(0.48 * self.h))
        time.sleep(1.5)

    def _input(self, *args: str, check: bool = False) -> None:
        run_adb([*self._serial, "shell", "input", *args], output="discard", check=check)

    def _tap(self, x: int, y: int) -> None:
        self._input("tap", str(x), str(y))

    def _swipe(self, x1: int, y1: int, x2: int, y2: int, duration: int) -> None:
        self._input("swipe", str(x1), str(y1), str(x2), str(y2), str(duration))

    def _press(self, keycode: str) -> None:
        self._input("keyevent", keycode)
        time.sleep(1)

    def _clear(self) -> None:
        for _ in range(_CLEAR_KEY_PRESSES):
            self._input("keyevent", "KEYCODE_DEL")
            time.sleep(0.2)

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


class EmulatorManager:
    """Lists the devices adb sees and duplicates an AVD for ``check_avd`` to start."""

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def get_running_avd_list(self) -> list[dict[str, Any]] | None:
        """Return each device on ``adb devices`` with its AVD name, ``None`` for a phone,
        or ``None`` instead of a list when adb fails.
        """
        devices = run_adb(["devices"], output="capture", check=False)
        if devices.returncode != 0:
            self._logger.error("Failed to get running AVD list: %s", devices.stderr)
            return None
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
        dup_name = f"{avd_name}_dup"
        shutil.copytree(
            host.avd_root_dir / f"{avd_name}.avd",
            host.avd_root_dir / f"{dup_name}.avd",
            dirs_exist_ok=True,
        )
        shutil.copy(host.avd_root_dir / f"{avd_name}.ini", host.avd_root_dir / f"{dup_name}.ini")
        self._rewrite_avd_ini(host, dup_name)
        self._rewrite_config_ini(host, dup_name)
        self._rewrite_hardware_qemu_ini(host, dup_name)

    def _rewrite_avd_ini(self, host: _AvdHost, avd_name: str) -> None:
        avd_ini = host.avd_root_dir / f"{avd_name}.ini"
        with avd_ini.open(encoding="utf-8") as f:
            target = next(
                (line.removeprefix("target=").strip() for line in f if line.startswith("target=")),
                "android-35",
            )
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


def launch_app(config: dict[str, Any], app: dict[str, Any], logger: logging.Logger) -> None:
    """Launch an app using an explicit activity when configured."""
    udid = config["device"]["udid"]
    app_name = app["name"]
    package_name = app["package_name"]
    launch_activity = app.get("launch_activity")

    if launch_activity:
        logger.info("Launching app %s via activity %s ...", app_name, launch_activity)
        result = run_adb(
            [
                "-s",
                udid,
                "shell",
                "am",
                "start",
                "-n",
                launch_activity,
            ],
            output="capture",
            check=False,
        )
        if result.returncode == 0:
            return
        logger.warning(
            "Activity launch failed for %s; falling back to package launch. error=%s",
            app_name,
            (result.stderr or result.stdout or "").strip(),
        )

    logger.info("Launching app %s (%s) ...", app_name, package_name)
    run_adb(
        [
            "-s",
            udid,
            "shell",
            "monkey",
            "-p",
            package_name,
            "-c",
            "android.intent.category.LAUNCHER",
            "1",
        ],
        output="discard",
        check=True,
    )


_LAUNCH_SETTLE_SECONDS = 4
_LAUNCHER_INTENT = ("-a", "android.intent.action.MAIN", "-c", "android.intent.category.LAUNCHER")
# FLAG_ACTIVITY_NEW_TASK | FLAG_ACTIVITY_CLEAR_TASK; the clear flag works only together
# with the new-task flag.
_CLEAR_TASK_FLAGS = "0x10008000"


class ForceStopRefusedError(subprocess.CalledProcessError):
    """adb refused to force-stop the app, so the device still shows what it showed."""


class NoLauncherActivityError(RuntimeError):
    """The package has no launcher activity, so restarting it needs a configured
    ``launch_activity``."""


class AppLaunch:
    """An app that ``restart_app`` restarts, and the one component every restart starts.

    Every replay begins at the graph's start node, so every restart has to land on the
    same screen. The component is the configured ``launch_activity``, or else the
    package's launcher activity, resolved on the first restart and reused; ``monkey``
    would pick one of several launcher activities at random.
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
            "%s has several launcher activities; restarting it through %s, not %s. "
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
    before anything is launched, so no replay starts from an unknown screen. When
    ``am start`` cannot start the component, the package's launcher intent is sent instead.
    """
    config = controller.config
    udid = config["device"]["udid"]
    force_stop = ["-s", udid, "shell", "am", "force-stop", app.package_name]
    stopped = run_adb(force_stop, output="discard", check=False)
    if stopped.returncode != 0:
        raise ForceStopRefusedError(stopped.returncode, stopped.args)
    component = app.component(udid, logger)
    start = ["-s", udid, "shell", "am", "start", "-n", component, "-f", _CLEAR_TASK_FLAGS]
    started = run_adb(start, output="capture", check=False)
    if _start_failed(started):
        logger.warning(
            "Activity launch failed for %s; falling back to package launch. error=%s",
            app.name,
            _adb_error(started),
        )
        launch_app(config, {"name": app.name, "package_name": app.package_name}, logger)
    time.sleep(_LAUNCH_SETTLE_SECONDS)


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
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        logger.debug("Soft keyboard probe failed: %s", exc)
        return ""
    return _SOFT_KEYBOARD_HINT if "mInputShown=true" in kb_check.stdout else ""


def keyboard_status(hint: str) -> str:
    """Return the planner-prompt sentence for ``hint`` from ``soft_keyboard_hint``."""
    if hint:
        return "Soft keyboard is visible; a text field is focused and ready for typing."
    return (
        "No OS keyboard signal detected. Still inspect the screenshot: "
        "a bottom input/keyboard bar can mean a text field is active."
    )


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
