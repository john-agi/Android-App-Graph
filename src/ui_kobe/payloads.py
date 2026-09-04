"""Narrowing helpers for JSON and other untyped payloads.

VLM responses arrive as ``json.loads`` output and the persisted graph arrives
as ``json.load`` output, so every value read out of them is ``Any``. These
helpers turn one such value into the concrete type the caller declares, with an
explicit fallback when the payload does not have the expected shape.
"""

from __future__ import annotations

from typing import Any


def as_str(value: object, default: str) -> str:
    """Return ``value`` when it is a ``str``, else ``default``."""
    return value if isinstance(value, str) else default


def as_int(value: object) -> int | None:
    """Return ``value`` as an ``int`` when it is numeric, else ``None``.

    Numeric strings are accepted because models routinely quote numbers.
    ``bool`` is rejected: ``True`` is an ``int`` in Python but never a
    meaningful count or duration in a payload.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return None
    return None


def as_int_list(value: object) -> list[int]:
    """Return ``value`` as ``list[int]`` when every item is numeric, else ``[]``."""
    if not isinstance(value, list):
        return []
    result: list[int] = []
    for item in value:
        number = as_int(item)
        if number is None:
            return []
        result.append(number)
    return result


def as_float_list(value: object) -> list[float]:
    """Return ``value`` as ``list[float]`` when every item is numeric, else ``[]``."""
    if not isinstance(value, list):
        return []
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            return []
        result.append(float(item))
    return result


def as_str_dict(value: object) -> dict[str, Any]:
    """Return ``value`` when it is a ``dict`` with ``str`` keys, else ``{}``."""
    if not isinstance(value, dict):
        return {}
    return {k: v for k, v in value.items() if isinstance(k, str)}
