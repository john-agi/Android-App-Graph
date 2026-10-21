"""Graph files on disk: atomic writes, the step log's layout, per-node
reference-screenshot lookup, and graph-structure validation.
"""

from __future__ import annotations

import base64
import errno
import io
import json
import os
import secrets
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import IO, Any


def write_json_atomically(
    path: Path,
    payload: object,
    *,
    indent: int | None = None,
    ensure_ascii: bool = True,
) -> None:
    """Write ``payload`` as JSON to ``path``, replacing it atomically as
    ``write_bytes_atomically`` does.
    """

    def dump(f: IO[bytes]) -> None:
        text = io.TextIOWrapper(f, encoding="utf-8")
        try:
            json.dump(payload, text, indent=indent, ensure_ascii=ensure_ascii)
            text.flush()
        finally:
            # Leaves f open for the fsync that follows; closing the wrapper would close it.
            text.detach()

    _write_atomically(path, dump)


def write_bytes_atomically(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path``, replacing it atomically.

    Written to a temporary file in the same directory and moved into place with
    ``os.replace``, so a crash mid-write -- or a failed rename -- leaves ``path``
    as either the previous complete file or the new one, never a truncated mix
    of both and never an orphaned temp file: the replace runs inside the same
    cleanup that unlinks the temp file on any other failure. The temp file is
    flushed and fsync'd before the rename, so that guarantee survives a power
    loss too, not only a process crash.

    A fresh file gets exactly the mode ``open(path, "w")`` would give (0o666
    with the process umask applied by the kernel); a file that already exists
    keeps its current mode across the rewrite, so an operator's chmod on a
    shared graph directory survives a rewrite.

    An atomic replace needs write permission on the containing directory, not
    only on the target file (to create and rename the temp file) -- the
    accepted cost of never leaving a truncated file behind. But
    ``os.replace`` does not otherwise care whether ``path`` itself is
    writable, so an explicit check raises ``PermissionError`` for a read-only
    ``path``: without it a directory-writable rewrite would silently defeat an
    operator's ``chmod 444`` freeze on the file. The check runs before the
    temp file is created, so a read-only target leaves nothing to clean up.

    A symlinked ``path`` is resolved to its real target first, so the temp
    file and the replace happen next to that file: ``os.replace`` over a
    symlink replaces the link itself, leaving whatever it pointed to
    untouched and turning ``path`` into a plain file.

    The rename itself is fsync'd too, on the containing directory's own
    descriptor after ``os.replace`` returns: a file-only fsync guarantees the
    new content reached disk, but not that the directory entry now pointing
    at it did, so without this a power loss right after the rename could
    still resurrect the old file.
    """
    _write_atomically(path, lambda f: f.write(data))


def _write_atomically(path: Path, write: Callable[[IO[bytes]], object]) -> None:
    """Replace ``path`` with what ``write`` writes, as ``write_bytes_atomically`` describes."""
    if path.is_symlink():
        path = path.resolve()
    if path.exists() and not os.access(path, os.W_OK):
        # A privileged (real-uid-0) process passes this for every file
        # regardless of its mode, matching root's in-place write succeeding
        # too -- the freeze only ever bound an unprivileged writer.
        raise PermissionError(errno.EACCES, os.strerror(errno.EACCES), str(path))
    tmp_path = path.parent / f"{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp"
    # tempfile.mkstemp hardcodes mode 0600, which would make a file written by
    # one user (or a CI job) unreadable to another process -- e.g. the audit or
    # the plot -- reading the same shared graph directory as a different user.
    # os.open lets the kernel apply the umask the way open(path, "w") does; never
    # os.umask, which is process-global state.
    fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    try:
        with os.fdopen(fd, "wb") as f:
            write(f)
            f.flush()
            os.fsync(f.fileno())
        if path.exists():
            shutil.copymode(path, tmp_path)
        tmp_path.replace(path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    dir_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def app_graph_path(graph_dir: Path, app_name: str) -> Path:
    """Return where exploration saves the graph of ``app_name`` under ``graph_dir``."""
    return graph_dir / app_name / f"{app_name}.json"


def graph_steps_dir(graph_path: Path) -> Path:
    """Return the step log directory of the graph at ``graph_path``.

    It sits beside the graph file, so an app's explored graph and its audited copy
    share one step log, and each names its nodes' observations by ID in it.
    """
    return graph_path.parent / "steps"


def step_record_path(steps: Path, step_id: str) -> Path:
    """Return where the step log ``steps`` keeps the record of step ``step_id``."""
    return steps / f"{step_id}.json"


def observation_image_path(steps: Path, observation_id: str) -> Path:
    """Return where the step log ``steps`` keeps the WebP image of an observation."""
    return steps / f"{observation_id}.webp"


def observation_image_b64(steps: Path, observation_id: str) -> str:
    """Return the WebP image of an observation in the step log ``steps`` as base64.

    A missing image raises ``FileNotFoundError``: every node names observations
    whose images exist, so a missing one means the step log was changed by hand.
    """
    image = observation_image_path(steps, observation_id).read_bytes()
    return base64.b64encode(image).decode("ascii")


def reference_screenshot_path(graph_path: Path, node_id: str) -> Path | None:
    """Return one node's reference screenshot path, or ``None`` when it has none.

    ``GraphManager.save_graph`` writes only the nodes it re-explored into
    ``<stem>_screenshots`` (``demo_audited_screenshots`` for an audited graph);
    every other node's screenshot stays where it always was, under the app
    directory's own name (``<parent.name>_screenshots``). An audited graph is
    thus split across both directories, so each node is looked up on its own
    rather than picking one directory for the whole graph.

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


def reference_screenshot_b64(graph_path: Path, node_id: str) -> str | None:
    """Return one node's reference screenshot as base64, or ``None`` when it has none."""
    screenshot_path = reference_screenshot_path(graph_path, node_id)
    if screenshot_path is None:
        return None
    return base64.b64encode(screenshot_path.read_bytes()).decode("ascii")


def require_graph_shape(data: object, path: Path) -> dict[str, Any]:
    """Return ``data`` once it is an object whose ``nodes``/``edges`` are lists
    of objects with non-empty string ids and endpoints; raise ``TypeError``
    naming ``path`` otherwise.

    A hand-edited or truncated file can leave ``nodes``/``edges`` ``null``, or
    holding something other than a list of objects; every such shape failure
    is reported here, naming the path, rather than surfacing later as a bare
    ``TypeError``/``AttributeError`` with no path once the loader iterates or
    indexes into it. An absent key is an empty list, not a shape failure: a
    graph with no edges yet is a normal, not malformed, one.

    A node id of ``""`` is a string, so it would otherwise sail past the
    string check above and load as a node; it is rejected here, naming the
    path, like every other malformed id.
    """
    if not isinstance(data, dict):
        msg = f"Graph JSON must be an object: {path}"
        raise TypeError(msg)
    # JSON object keys are always strings; the comprehension is what narrows the
    # loaded object to dict[str, Any] for the type checker (as payloads does).
    graph: dict[str, Any] = {key: value for key, value in data.items() if isinstance(key, str)}
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    if not isinstance(nodes, list) or not isinstance(edges, list):
        msg = f"Graph JSON must contain list fields 'nodes' and 'edges': {path}"
        raise TypeError(msg)
    graph["nodes"] = nodes
    graph["edges"] = edges
    for node in nodes:
        if not isinstance(node, dict):
            msg = f"Graph node must be an object: {node!r} in {path}"
            raise TypeError(msg)
        node_id = node.get("id")
        if not isinstance(node_id, str):
            msg = f"Graph node id must be a string: {node!r} in {path}"
            raise TypeError(msg)
        if not node_id:
            msg = f"Graph node id must be a non-empty string: {node!r} in {path}"
            raise TypeError(msg)
    for edge in edges:
        if not isinstance(edge, dict):
            msg = f"Graph edge must be an object: {edge!r} in {path}"
            raise TypeError(msg)
        if not isinstance(edge.get("source"), str) or not isinstance(edge.get("target"), str):
            msg = f"Graph edge endpoint must be a string: {edge!r} in {path}"
            raise TypeError(msg)
    return graph


def require_known_edge_endpoints(data: object, path: Path) -> dict[str, Any]:
    """Return the validated graph object; raise when an edge names a node id
    absent from ``data["nodes"]``.

    networkx's add_edge would otherwise silently create an attribute-less node
    for an undefined endpoint, so a hand-edited or truncated file would load
    quietly.

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
