"""Structural types for the aitk device surface that android_app_graph depends on.

``aitk`` is an unannotated dependency, so everything read back from its
``ADBController`` and ``AVDManager`` is ``Unknown``. These Protocols pin down
the subset android_app_graph actually uses, which keeps the device boundary explicit and
lets tests pass a fake without adb or an emulator.
"""

from __future__ import annotations

import logging
import subprocess
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
    shell = ["adb", "-s", udid, "shell"]
    dumpsys = subprocess.run(
        [*shell, "dumpsys", "package", package_name],
        capture_output=True,
        text=True,
        check=True,
    )
    permissions = _requested_permissions(dumpsys.stdout)
    granted = 0
    for permission in permissions:
        grant = subprocess.run(
            [*shell, "pm", "grant", package_name, permission],
            capture_output=True,
            text=True,
            check=False,
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
