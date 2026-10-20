"""Structural types for the aitk device surface that android_app_graph depends on.

``aitk`` is an unannotated dependency, so everything read back from its
``ADBController`` and ``AVDManager`` is ``Unknown``. These Protocols pin down
the subset android_app_graph actually uses, which keeps the device boundary explicit and
lets tests pass a fake without adb or an emulator.
"""

from __future__ import annotations

import logging
import subprocess
import time
from typing import Any, Protocol

logger = logging.getLogger(__name__)

_SOFT_KEYBOARD_HINT = (
    " (Note: the soft keyboard is currently visible — a text field is focused "
    "and ready for typing.)"
)


class DeviceController(Protocol):
    """The subset of aitk's ADBController that android_app_graph depends on.

    Arguments android_app_graph never passes (``exe_action``'s ``save_flag``) are omitted.
    """

    w: int
    h: int
    config: dict[str, Any]

    def get_state(self) -> dict[str, Any]: ...

    def exe_action(self, action: dict[str, Any]) -> None: ...


class AvdManager(Protocol):
    """The subset of aitk's AVDManager that the CLI depends on."""

    def get_running_avd_list(self) -> list[dict[str, Any]] | None: ...

    def duplicate_avd(self, avd_name: str) -> None: ...


def launch_app(config: dict[str, Any], app: dict[str, Any], logger: logging.Logger) -> None:
    """Launch an app using an explicit activity when configured."""
    udid = config["device"]["udid"]
    app_name = app["name"]
    package_name = app["package_name"]
    launch_activity = app.get("launch_activity")

    if launch_activity:
        logger.info("Launching app %s via activity %s ...", app_name, launch_activity)
        result = subprocess.run(
            [
                "adb",
                "-s",
                udid,
                "shell",
                "am",
                "start",
                "-n",
                launch_activity,
            ],
            capture_output=True,
            text=True,
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
    subprocess.run(
        [
            "adb",
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
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )


_LAUNCH_SETTLE_SECONDS = 4


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
    A failed force-stop raises before anything is launched, so no replay starts
    from an unknown screen.
    """
    config = controller.config
    subprocess.run(
        ["adb", "-s", config["device"]["udid"], "shell", "am", "force-stop", package_name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )
    app = {"name": app_name, "package_name": package_name, "launch_activity": launch_activity}
    launch_app(config, app, logger)
    time.sleep(_LAUNCH_SETTLE_SECONDS)


def soft_keyboard_hint() -> str:
    """Return a note that the soft keyboard is visible, or ``""`` otherwise.

    Shells out to adb rather than going through ``DeviceController``: aitk's
    controller exposes no shell method, and the translator holds no controller.
    """
    try:
        kb_check = subprocess.run(
            ["adb", "shell", "dumpsys", "input_method"],
            capture_output=True,
            text=True,
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
