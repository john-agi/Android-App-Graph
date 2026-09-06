"""Derive an Android app package from a full activity name.

The package is the join key between a runtime screen (reported as an Android
activity by adb/AITK) and a graph node (grouped by the app that produced it),
so there must be exactly one implementation of this heuristic: both
``utils.graph_manager`` (building exploration graphs) and ``adapters`` (the
AITK runtime translator) import it from here rather than keeping their own
copy.
"""

from __future__ import annotations


def package_from_activity(activity: str) -> str:
    """Extract the app package from a full Android activity name.

    ``com.citymapper.app.home.HomeActivity2`` -> ``com.citymapper.app``
    ``com.citymapper.app/com.citymapper.app.MainActivity`` -> ``com.citymapper.app``

    Heuristic: take the first 3 dot-segments (``com.company.app``). This is the
    standard Android package convention and is enough to group activities that
    belong to the same app while separating different apps.
    """
    if "/" in activity:
        activity = activity.split("/", maxsplit=1)[0]
    parts = activity.split(".")
    if len(parts) >= 3:
        return ".".join(parts[:3])
    return activity
