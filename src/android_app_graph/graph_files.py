"""Graph-file discovery, per-node reference-screenshot lookup, atomic JSON
writing, and graph-structure validation shared by every loader.
"""

from __future__ import annotations

import base64
import json
import os
import secrets
import shutil
from pathlib import Path
from typing import Any


def write_json_atomically(
    path: Path,
    payload: object,
    *,
    indent: int | None = None,
    ensure_ascii: bool = True,
) -> None:
    """Write ``payload`` as JSON to ``path``, replacing it atomically.

    Dumped to a temporary file in the same directory and moved into place with
    ``os.replace``, so a crash mid-dump -- or a failed rename -- leaves ``path``
    as either the previous complete file or the new one, never a truncated mix
    of both and never an orphaned temp file: the replace runs inside the same
    cleanup that unlinks the temp file on any other failure.

    A fresh file gets exactly the mode ``open(path, "w")`` would give (0o666
    with the process umask applied by the kernel); a file that already exists
    keeps its current mode across the rewrite, matching what in-place
    truncation used to do, so an operator's chmod on a shared graph directory
    survives a rewrite.

    Unlike the in-place truncation this replaced, which needed write
    permission only on the target file itself, an atomic replace needs write
    permission on the containing directory too (to create and rename the temp
    file) -- the accepted cost of never leaving a truncated file behind.
    """
    tmp_path = path.parent / f"{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp"
    # tempfile.mkstemp hardcodes mode 0600, which would make a file written by
    # one user (or a CI job) unreadable to another process -- e.g. an AITK
    # runtime -- reading the same shared graph directory as a different user.
    # os.open lets the kernel apply the umask the way open(path, "w") does; never
    # os.umask, which is process-global state.
    fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=indent, ensure_ascii=ensure_ascii)
        if path.exists():
            shutil.copymode(path, tmp_path)
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def reference_screenshot_path(graph_path: Path, node_id: str) -> Path | None:
    """Return one node's reference screenshot path, or ``None`` when it has none.

    ``GraphManager.save_graph`` writes only the nodes it re-explored into
    ``<stem>_screenshots`` (``demo_audited_screenshots`` for an audited graph);
    every other node's screenshot stays where it always was, under the app
    directory's own name (``<parent.name>_screenshots``). An audited graph is
    thus split across both directories, so each node is looked up on its own
    rather than picking one directory for the whole graph. Both runtime graph
    loading and offline precomputation must resolve to the same path, or one
    sees a screenshot the other reports missing.

    The app-directory fallback only ever applies to that plain/audited pair:
    a graph with any other stem (an operator's ``demo_v1.json`` kept next to
    ``demo.json``) never falls back to it, or it would silently borrow a
    sibling graph's screenshot for a colliding node id.
    """
    stem_path = graph_path.parent / f"{graph_path.stem}_screenshots" / f"{node_id}.png"
    if stem_path.exists():
        return stem_path
    app_name = graph_path.parent.name
    if graph_path.stem not in (app_name, f"{app_name}_audited"):
        return None
    app_path = graph_path.parent / f"{app_name}_screenshots" / f"{node_id}.png"
    if app_path.exists():
        return app_path
    return None


def encode_screenshot_b64(screenshot_path: Path) -> str:
    """Read one screenshot file and return its bytes as base64 ascii.

    Split out of ``reference_screenshot_b64`` so a caller that already has a
    node's screenshot path -- offline precomputation streaming one candidate
    at a time instead of reading every uncached screenshot up front -- reads
    and encodes through the same path as runtime graph loading.
    """
    return base64.b64encode(screenshot_path.read_bytes()).decode("ascii")


def reference_screenshot_b64(graph_path: Path, node_id: str) -> str | None:
    """Return one node's reference screenshot as base64, or ``None`` when it has none."""
    screenshot_path = reference_screenshot_path(graph_path, node_id)
    if screenshot_path is None:
        return None
    return encode_screenshot_b64(screenshot_path)


def require_known_edge_endpoints(data: dict[str, Any], path: Path) -> None:
    """Raise when an edge names a node id absent from ``data["nodes"]``.

    Shared by both graph loaders (runtime and GraphManager): networkx's
    add_edge would otherwise silently create an attribute-less node for an
    undefined endpoint, so a hand-edited or truncated file would load
    quietly. See #62/#63.
    """
    node_ids = {str(node["id"]) for node in data.get("nodes", [])}
    missing_ids: set[str] = set()
    for edge in data.get("edges", []):
        missing_ids.update(
            str(endpoint)
            for endpoint in (edge["source"], edge["target"])
            if str(endpoint) not in node_ids
        )
    if missing_ids:
        msg = f"{path}: edge(s) reference node id(s) absent from the file: {sorted(missing_ids)}"
        raise ValueError(msg)


def iter_graph_files(graph_dir: Path, app_name: str | None = None) -> list[tuple[str, Path]]:
    """Return one graph JSON per app, preferring the audited graph.

    Neither runtime graph loading nor offline embedding precomputation should
    pick up audit reports, merge reports, or embedding sidecars: the app name
    comes from the graph directory name, so ``eboox_audited.json`` still loads
    as app ``eboox``. When ``app_name`` is given, only that app's directory is
    considered.
    """
    selected: list[tuple[str, Path]] = []
    if not graph_dir.exists():
        return selected
    app_dirs = (
        [graph_dir / app_name] if app_name else sorted(p for p in graph_dir.iterdir() if p.is_dir())
    )
    for app_dir in app_dirs:
        if not app_dir.is_dir():
            continue
        name = app_dir.name
        audited = app_dir / f"{name}_audited.json"
        plain = app_dir / f"{name}.json"
        if audited.exists():
            selected.append((name, audited))
        elif plain.exists():
            selected.append((name, plain))
    return selected
