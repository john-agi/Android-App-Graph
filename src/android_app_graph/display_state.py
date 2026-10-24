"""Supported display configuration and its explicit restoration commands."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import time as local_time
from typing import Literal

from android_app_graph.payloads import as_object_dict

type NightMode = Literal["yes", "no", "auto", "custom_schedule", "custom_bedtime"]
_PAIR_SIZE = 2
NIGHT_KEYS = (
    "ui_night_mode",
    "ui_night_mode_custom_type",
    "dark_theme_custom_start_time",
    "dark_theme_custom_end_time",
    "ui_night_mode_override_on",
    "ui_night_mode_override_off",
    "ui_night_mode_last_computed",
)


def nullable_setting(value: object) -> str | None:
    if value is None or isinstance(value, str):
        return value
    msg = "A stored setting must be text or null"
    raise ValueError(msg)


def _size(value: object) -> tuple[int, int]:
    if isinstance(value, (list, tuple)) and len(value) == _PAIR_SIZE:
        width, height = value
        if type(width) is int and type(height) is int and width > 0 and height > 0:
            return width, height
    msg = "A display size must contain two positive integers"
    raise ValueError(msg)


def _density(value: object) -> int:
    if type(value) is not int or value <= 0:
        msg = "Display density must be a positive integer"
        raise ValueError(msg)
    return value


def _time(value: object) -> str:
    if isinstance(value, str) and re.fullmatch(r"\d{2}:\d{2}(?::\d{2}(?:\.\d{1,9})?)?", value):
        local_time.fromisoformat(value)
        return value
    msg = "A custom night-mode time must be an ISO local time"
    raise ValueError(msg)


def _night_mode(value: object) -> NightMode:
    if value == "yes":
        return "yes"
    if value == "no":
        return "no"
    if value == "auto":
        return "auto"
    if value == "custom_schedule":
        return "custom_schedule"
    if value == "custom_bedtime":
        return "custom_bedtime"
    msg = f"Unsupported night mode: {value!r}"
    raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class DisplayState:
    """Physical defaults, override presence and the supported night-mode configuration."""

    physical_size: tuple[int, int]
    size_override: tuple[int, int] | None
    physical_density: int
    density_override: int | None
    font_scale: str | None
    night_mode: NightMode
    custom_start: str
    custom_end: str
    night_settings: dict[str, str | None]
    night_overrides: tuple[bool, bool]

    def to_dict(self) -> dict[str, object]:
        return {
            "physical_size": list(self.physical_size),
            "size_override": None if self.size_override is None else list(self.size_override),
            "physical_density": self.physical_density,
            "density_override": self.density_override,
            "font_scale": self.font_scale,
            "night_mode": self.night_mode,
            "custom_start": self.custom_start,
            "custom_end": self.custom_end,
            "night_settings": dict(self.night_settings),
            "night_overrides": list(self.night_overrides),
        }

    @classmethod
    def from_dict(cls, value: object) -> DisplayState:
        data = as_object_dict(value)
        required = (
            "physical_size",
            "size_override",
            "physical_density",
            "density_override",
            "font_scale",
            "night_mode",
            "custom_start",
            "custom_end",
            "night_settings",
            "night_overrides",
        )
        missing = [name for name in required if name not in data]
        if missing:
            msg = f"Incomplete display baseline: {', '.join(missing)}"
            raise ValueError(msg)
        mode = _night_mode(data["night_mode"])
        overrides = data["night_overrides"]
        if (
            not isinstance(overrides, list)
            or len(overrides) != _PAIR_SIZE
            or not all(type(v) is bool for v in overrides)
        ):
            msg = "Night overrides require two boolean values"
            raise ValueError(msg)
        settings = as_object_dict(data["night_settings"])
        if any(key not in settings for key in NIGHT_KEYS):
            msg = "Incomplete persistent night-mode settings"
            raise ValueError(msg)
        return cls(
            physical_size=_size(data["physical_size"]),
            size_override=None if data["size_override"] is None else _size(data["size_override"]),
            physical_density=_density(data["physical_density"]),
            density_override=None
            if data["density_override"] is None
            else _density(data["density_override"]),
            font_scale=nullable_setting(data["font_scale"]),
            night_mode=mode,
            custom_start=_time(data["custom_start"]),
            custom_end=_time(data["custom_end"]),
            night_settings={key: nullable_setting(settings[key]) for key in NIGHT_KEYS},
            night_overrides=(overrides[0] is True, overrides[1] is True),
        )

    def geometry(self) -> dict[str, object]:
        return {
            "physical_size": list(self.physical_size),
            "size_override": None if self.size_override is None else list(self.size_override),
            "physical_density": self.physical_density,
            "density_override": self.density_override,
            "font_scale": self.font_scale,
        }

    def restoration_commands(self) -> list[tuple[str, ...]]:
        size = "reset" if self.size_override is None else "x".join(map(str, self.size_override))
        density = "reset" if self.density_override is None else str(self.density_override)
        commands = [
            ("wm", "size", size),
            ("wm", "density", density),
            ("cmd", "uimode", "time", "start", self.custom_start),
            ("cmd", "uimode", "time", "end", self.custom_end),
            ("cmd", "uimode", "night", self.night_mode),
        ]
        values = {
            "font_scale": self.font_scale,
            **{key: self.night_settings[key] for key in NIGHT_KEYS},
        }
        for key, value in values.items():
            namespace = "system" if key == "font_scale" else "secure"
            commands.append(
                ("settings", "delete", namespace, key)
                if value is None
                else ("settings", "put", namespace, key, value)
            )
        return commands

    def changed_restoration_commands(self, current: DisplayState) -> list[tuple[str, ...]]:
        commands = self.restoration_commands()
        changes = [
            (self.size_override != current.size_override, commands[0]),
            (self.density_override != current.density_override, commands[1]),
            (self.custom_start != current.custom_start, commands[2]),
            (self.custom_end != current.custom_end, commands[3]),
            (self.night_mode != current.night_mode, commands[4]),
        ]
        values = {
            "font_scale": self.font_scale,
            **{key: self.night_settings[key] for key in NIGHT_KEYS},
        }
        for index, (key, value) in enumerate(values.items(), start=5):
            actual = current.font_scale if key == "font_scale" else current.night_settings[key]
            changes.append((value != actual, commands[index]))
        return [command for changed, command in changes if changed]
