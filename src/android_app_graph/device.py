"""Structural types for the aitk device surface that android_app_graph depends on.

``aitk`` is an unannotated dependency, so everything read back from its
``ADBController`` and ``AVDManager`` is ``Unknown``. These Protocols pin down
the subset android_app_graph actually uses, which keeps the device boundary explicit and
lets tests pass a fake without adb or an emulator.
"""

from __future__ import annotations

from typing import Any, Protocol


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
