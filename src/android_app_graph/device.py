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
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, NamedTuple, Protocol

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


def _adb_stdout_bytes(args: Sequence[str]) -> bytes:
    """Run the adb on ``PATH`` with ``args`` and return its stdout undecoded, for output
    that is not text, such as a PNG. A non-zero exit raises ``CalledProcessError``.
    """
    return subprocess.run(  # noqa: S603  # argv list without a shell; adb arguments come from this package's config and constants
        [adb_executable(), *args], capture_output=True, check=True
    ).stdout


_GESTURE_MS = 1000
_DEFAULT_WAIT_SECONDS = 3
_CLEAR_KEY_PRESSES = 20


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

    def get_state(self) -> dict[str, Any]:
        """Return the UI hierarchy XML, a base64 PNG screenshot, and the focused package
        and activity, both ``"unknown"`` when ``dumpsys window`` names none.
        """
        xml = run_adb(
            [*self._serial, "exec-out", "uiautomator dump /dev/stdout"],
            output="capture",
            check=True,
        )
        png = _adb_stdout_bytes([*self._serial, "exec-out", "screencap -p"])
        package, activity = self._focused_package_activity()
        return {
            "xml": xml.stdout,
            "screenshot": base64.b64encode(png).decode("ascii"),
            "package": package,
            "activity": activity,
        }

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

    def _focused_package_activity(self) -> tuple[str, str]:
        """Read the package and activity from the first ``mCurrentFocus`` or
        ``mFocusedApp`` line that names one, expanding a shorthand ``.Activity``.
        """
        dumpsys = run_adb(
            [*self._serial, "shell", "dumpsys", "window"], output="capture", check=True
        )
        for line in dumpsys.stdout.splitlines():
            if "mCurrentFocus" not in line and "mFocusedApp" not in line:
                continue
            match = re.search(r"[\w.]+/[\w.]+", line)
            if match:
                package, _, activity = match.group().partition("/")
                if activity.startswith("."):
                    activity = package + activity
                return package, activity
        return "unknown", "unknown"

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


class ForceStopRefusedError(subprocess.CalledProcessError):
    """adb refused to force-stop the app, so the device still shows what it showed."""


def restart_app(
    controller: DeviceController,
    app_name: str,
    package_name: str,
    launch_activity: str | None,
    logger: logging.Logger,
) -> None:
    """Put the app on its start screen, where every replay from the start node begins.

    A launch alone does not: launching an app whose task still exists brings that
    task to the foreground as it was
    (https://developer.android.com/guide/components/activities/tasks-and-back-stack).
    A refused force-stop raises ``ForceStopRefusedError`` before anything is
    launched, so no replay starts from an unknown screen.
    """
    config = controller.config
    force_stop = ["-s", config["device"]["udid"], "shell", "am", "force-stop", package_name]
    stopped = run_adb(force_stop, output="discard", check=False)
    if stopped.returncode != 0:
        raise ForceStopRefusedError(stopped.returncode, stopped.args)
    app = {"name": app_name, "package_name": package_name, "launch_activity": launch_activity}
    launch_app(config, app, logger)
    time.sleep(_LAUNCH_SETTLE_SECONDS)


def soft_keyboard_hint() -> str:
    """Return a note that the soft keyboard is visible, or ``""`` otherwise.

    Shells out to adb rather than going through ``DeviceController``, which has no
    shell method.
    """
    try:
        kb_check = run_adb(
            ["shell", "dumpsys", "input_method"],
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
    """Return the planner-prompt sentence for ``hint`` from ``soft_keyboard_hint()``."""
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
