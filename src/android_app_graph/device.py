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
