"""Structural types for the aitk device surface that ui_kobe depends on.

``aitk`` is an unannotated dependency, so everything read back from its
``ADBController`` and ``AVDManager`` is ``Unknown``. These Protocols pin down
the subset ui_kobe actually uses, which keeps the device boundary explicit and
lets tests pass a fake without adb or an emulator.
"""

from __future__ import annotations

from typing import Any, Protocol


class DeviceController(Protocol):
    """The subset of aitk's ADBController that ui_kobe depends on."""

    w: int
    h: int
    config: dict[str, Any]

    def get_state(self) -> dict[str, Any]: ...

    # aitk's ADBController also takes ``save_flag: bool = True``; ui_kobe never passes it,
    # and a Protocol declares only the arguments its callers use. An implementation that
    # takes the extra defaulted parameter still satisfies this signature.
    def exe_action(self, action: dict[str, Any]) -> None: ...


class AvdManager(Protocol):
    """The subset of aitk's AVDManager that the CLI depends on."""

    def get_running_avd_list(self) -> list[dict[str, Any]] | None: ...

    def duplicate_avd(self, avd_name: str) -> None: ...
