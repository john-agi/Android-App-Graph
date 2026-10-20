"""The Android package of a screen or of a graph node.

The package is the join key between a screen and a graph node, so the graph
builder and the runtime must read it from a graph file the same way.
"""

from __future__ import annotations

from collections.abc import Mapping


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


def saved_node_package(node: Mapping[str, object], graph_package: str) -> str:
    """Return the package of ``node``, read from a graph file.

    A node saved before nodes recorded their package, or saved from a graph that
    had no ``package_name``, gets ``graph_package`` unless it is an ``ext_``
    placeholder: the explorer records any other app's screen as one. Its activity
    cannot tell, because an app's classes need not live under its package.
    """
    package = node.get("package")
    if isinstance(package, str) and package:
        return package
    node_id = node.get("id")
    if isinstance(node_id, str) and node_id.startswith("ext_"):
        return ""
    return graph_package
