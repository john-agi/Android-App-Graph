"""The Android package of a graph node saved in a graph file."""

from __future__ import annotations

from collections.abc import Mapping


def saved_node_package(node: Mapping[str, object], graph_package: str) -> str:
    """Return the package of ``node``, read from a graph file.

    A node saved without a package is taken to be a screen of the graph's app and
    gets ``graph_package``, unless it is an ``ext_`` placeholder, which stands for
    another app or a web page and gets none. Its activity cannot tell, because an
    app's classes need not live under its package.
    """
    package = node.get("package")
    if isinstance(package, str) and package:
        return package
    node_id = node.get("id")
    if isinstance(node_id, str) and node_id.startswith("ext_"):
        return ""
    return graph_package
