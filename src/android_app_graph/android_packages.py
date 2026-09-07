"""Derive an Android app package from a full activity name.

The package is the join key between a runtime screen and a graph node, so the
graph builder and the runtime must share this one implementation.
"""

from __future__ import annotations


def package_from_activity(activity: str) -> str:
    """Return the first three dot-segments of ``activity``, without its ``/component``.

    ``com.citymapper.app/com.citymapper.app.MainActivity`` -> ``com.citymapper.app``
    """
    if "/" in activity:
        activity = activity.split("/", maxsplit=1)[0]
    parts = activity.split(".")
    if len(parts) >= 3:
        return ".".join(parts[:3])
    return activity
