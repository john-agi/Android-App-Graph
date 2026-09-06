"""Graph-file discovery, per-node reference-screenshot lookup, atomic JSON
writing, and graph-structure validation shared by every loader.
"""

from __future__ import annotations

import base64
import errno
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
    cleanup that unlinks the temp file on any other failure. The temp file is
    flushed and fsync'd before the rename, so that guarantee survives a power
    loss too, not only a process crash.

    A fresh file gets exactly the mode ``open(path, "w")`` would give (0o666
    with the process umask applied by the kernel); a file that already exists
    keeps its current mode across the rewrite, matching what in-place
    truncation used to do, so an operator's chmod on a shared graph directory
    survives a rewrite.

    Unlike the in-place truncation this replaced, which needed write
    permission only on the target file itself, an atomic replace needs write
    permission on the containing directory too (to create and rename the temp
    file) -- the accepted cost of never leaving a truncated file behind. But
    ``os.replace`` does not otherwise care whether ``path`` itself is
    writable, so without an explicit check a directory-writable rewrite would
    silently defeat an operator's ``chmod 444`` freeze on the file, where the
    in-place write this replaced would have raised ``PermissionError``. The
    check runs before the temp file is created, so a read-only target leaves
    nothing to clean up.

    A symlinked ``path`` is resolved to its real target first, so the temp
    file and the replace happen next to that file: ``os.replace`` over a
    symlink replaces the link itself, leaving whatever it pointed to
    untouched and turning ``path`` into a plain file.
    """
    if path.is_symlink():
        path = path.resolve()
    if path.exists() and not os.access(path, os.W_OK):
        # A privileged (real-uid-0) process passes this for every file
        # regardless of its mode, matching root's in-place write succeeding
        # too -- the freeze only ever bound an unprivileged writer.
        raise PermissionError(errno.EACCES, os.strerror(errno.EACCES), str(path))
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
            f.flush()
            os.fsync(f.fileno())
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


def require_graph_shape(data: object, path: Path) -> dict[str, Any]:
    """Return ``data`` once it is an object whose ``nodes``/``edges`` are lists
    of objects with string ids and endpoints; raise ``TypeError`` naming
    ``path`` otherwise.

    Shared by both graph loaders (runtime ``aitk_translator`` and
    ``GraphManager``) so the check and its message cannot drift between them.
    A hand-edited or truncated file can leave ``nodes``/``edges`` missing,
    ``null``, or holding something other than a list of objects; every such
    shape failure is reported here, naming the path, rather than surfacing
    later as a bare ``TypeError``/``AttributeError`` with no path once a
    loader iterates or indexes into it.
    """
    if not isinstance(data, dict):
        raise TypeError(f"Graph JSON must be an object: {path}")
    # JSON object keys are always strings; the comprehension is what narrows the
    # loaded object to dict[str, Any] for the type checker (as payloads does).
    graph: dict[str, Any] = {key: value for key, value in data.items() if isinstance(key, str)}
    nodes = graph.get("nodes")
    edges = graph.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise TypeError(f"Graph JSON must contain list fields 'nodes' and 'edges': {path}")
    for node in nodes:
        if not isinstance(node, dict):
            raise TypeError(f"Graph node must be an object: {node!r} in {path}")
        if not isinstance(node.get("id"), str):
            raise TypeError(f"Graph node id must be a string: {node!r} in {path}")
    for edge in edges:
        if not isinstance(edge, dict):
            raise TypeError(f"Graph edge must be an object: {edge!r} in {path}")
        if not isinstance(edge.get("source"), str) or not isinstance(edge.get("target"), str):
            raise TypeError(f"Graph edge endpoint must be a string: {edge!r} in {path}")
    return graph


def require_known_edge_endpoints(data: object, path: Path) -> dict[str, Any]:
    """Return the validated graph object; raise when an edge names a node id
    absent from ``data["nodes"]``.

    Shared by both graph loaders (runtime and GraphManager): networkx's
    add_edge would otherwise silently create an attribute-less node for an
    undefined endpoint, so a hand-edited or truncated file would load
    quietly. See #62/#63.

    ``require_graph_shape`` runs first, so every id and endpoint read below is
    already confirmed to be a string -- a missing, malformed or non-string
    one is reported there, before this ever builds the node id set.
    """
    graph = require_graph_shape(data, path)
    node_ids = {node["id"] for node in graph["nodes"]}
    missing_ids = {
        endpoint
        for edge in graph["edges"]
        for endpoint in (edge["source"], edge["target"])
        if endpoint not in node_ids
    }
    if missing_ids:
        msg = f"{path}: edge(s) reference node id(s) absent from the file: {sorted(missing_ids)}"
        raise ValueError(msg)
    return graph


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
