"""Graph files on disk: atomic writes, the step log's layout, model-tagged vector
files, and graph-structure validation.
"""

from __future__ import annotations

import base64
import errno
import io
import json
import logging
import os
import secrets
import shutil
from collections.abc import Callable, Collection, Iterable, Mapping
from pathlib import Path
from typing import IO, Any, Literal

from android_app_graph.payloads import as_float_list, as_object_dict

logger = logging.getLogger(__name__)


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


def image_embeddings_path(steps: Path) -> Path:
    """Return where the step log ``steps`` keeps its observations' image vectors."""
    return steps / "image_emb.json"


def text_embeddings_path(graph_path: Path) -> Path:
    """Return where the graph at ``graph_path`` keeps its members' description vectors.

    A description depends on the other nodes' descriptions in its prompt, so its vector
    belongs to the graph beside it rather than to the step log the graph shares.
    """
    return graph_path.with_name(f"{graph_path.stem}_text_emb.json")


def identify_file_path(graph_path: Path) -> Path:
    """Return where the graph at ``graph_path`` keeps the decisions that placed its
    screenshots, one JSON line per screenshot it identified.

    A screen's description depends on the other nodes' descriptions in its prompt, and
    its node on the matching settings and every earlier decision, so both belong to the
    graph beside them rather than to the step log the graph shares.
    """
    return graph_path.with_name(f"{graph_path.stem}_identify.jsonl")


def append_json_line(path: Path, record: Mapping[str, object]) -> None:
    """Append ``record`` to ``path`` as one line of JSON, creating the file when it is
    missing, and flush and fsync it before returning.

    A last line a crash tore off before its newline is cut off first, so the new line
    starts a line of its own.
    """
    line = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
    with path.open("a+b") as f:
        size = f.seek(0, os.SEEK_END)
        if size:
            f.seek(size - 1)
            if f.read(1) != b"\n":
                f.seek(0)
                f.truncate(f.read().rfind(b"\n") + 1)
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def read_json_lines(path: Path) -> list[dict[str, object]]:
    """Return the JSON object on each line of ``path``, in order; a missing file has none.

    A last line without its newline, torn off by a crash, is left out. Any other line
    that is not a JSON object raises ``ValueError`` naming the file and the line.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    records: list[dict[str, object]] = []
    for number, line in enumerate(text.split("\n")[:-1], start=1):
        record = _json_object(line)
        if record is None:
            msg = f"{path}: line {number} is not a JSON object"
            raise ValueError(msg)
        records.append(record)
    return records


def _json_object(line: str) -> dict[str, object] | None:
    try:
        value = json.loads(line)
    except ValueError:
        return None
    return as_object_dict(value) if isinstance(value, dict) else None


type VectorKind = Literal["image", "text"]


def _unusable_embeddings(path: Path, kind: VectorKind, problem: str) -> ValueError:
    remedy = (
        "Delete it to embed every screenshot again."
        if kind == "image"
        else "Explore the app again into a new graph directory."
    )
    return ValueError(f"{kind.capitalize()} embedding file {path} {problem}. {remedy}")


def load_embeddings(
    path: Path, *, model: str, kind: VectorKind, required: Collection[str] = ()
) -> dict[str, list[float]]:
    """Return the ``kind`` vectors of the file ``path`` by observation ID, when
    ``model`` computed them, which must include one for each of ``required``.

    The file is tagged with the model that wrote it, since a matching dimension
    cannot prove two vectors came from one model. A missing file has no vectors yet.
    An image file written by another model is empty after one warning, so every
    screenshot is embedded again; nothing embeds a description again, so a text file
    written by another model raises ``ValueError``.

    A file that cannot be used raises ``ValueError`` naming it and the problem, so
    damage is reported instead of repaired by paying for every vector again: a file
    that cannot be read or parsed, is not tagged, has no ``embeddings`` object, or holds
    a malformed or non-finite vector, and a file, missing or not, without the vector of
    an observation of ``required``. The message names such observations.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        if required:
            raise _unusable_embeddings(path, kind, "is missing") from None
        return {}
    except (OSError, ValueError) as exc:
        raise _unusable_embeddings(path, kind, f"cannot be read or parsed: {exc}") from exc
    stored = as_object_dict(raw)
    tag = stored.get("model")
    if not isinstance(tag, str):
        raise _unusable_embeddings(path, kind, "is not tagged with the model that wrote it")
    if not isinstance(stored.get("embeddings"), dict):
        raise _unusable_embeddings(path, kind, "has no embeddings object")
    if tag != model:
        if kind == "text":
            problem = f"was written by model {tag!r}, not the configured {model!r}"
            raise _unusable_embeddings(path, kind, problem)
        logger.warning(
            "Image embedding file %s was written by model %r, not the configured %r; "
            "treating it as empty, so every observation is embedded again",
            path,
            tag,
            model,
        )
        return {}
    embeddings = {
        observation_id: as_float_list(vector)
        for observation_id, vector in as_object_dict(stored.get("embeddings")).items()
    }
    malformed = [observation_id for observation_id, numbers in embeddings.items() if not numbers]
    if malformed:
        raise _unusable_embeddings(
            path,
            kind,
            f"has a malformed or non-finite vector for observation(s) {', '.join(malformed)}",
        )
    missing = [observation_id for observation_id in required if observation_id not in embeddings]
    if missing:
        raise _unusable_embeddings(
            path, kind, f"has no vector for observation(s) {', '.join(missing)}"
        )
    return embeddings


def save_embeddings(path: Path, embeddings: Mapping[str, list[float]], *, model: str) -> None:
    """Write ``embeddings``, vectors by observation ID that ``model`` computed, to
    ``path``, replacing the file atomically."""
    write_json_atomically(path, {"model": model, "embeddings": dict(embeddings)})


def observation_image_b64(steps: Path, observation_id: str) -> str:
    """Return the WebP image of an observation in the step log ``steps`` as base64.

    A missing image raises ``FileNotFoundError``: every node names observations
    whose images exist, so a missing one means the step log was changed by hand.
    """
    image = observation_image_path(steps, observation_id).read_bytes()
    return base64.b64encode(image).decode("ascii")


def require_graph_shape(data: object, path: Path) -> dict[str, Any]:
    """Return ``data`` once it is an object whose ``nodes``/``edges`` are lists
    of objects with non-empty string ids and endpoints; raise the ``ValueError``
    of ``unloadable_graph`` naming ``path`` otherwise.

    A hand-edited or truncated file can leave ``nodes``/``edges`` absent, ``null``,
    or holding something other than a list of objects; every such shape failure
    is reported here, naming the path, rather than surfacing later as a bare
    ``TypeError``/``AttributeError``/``KeyError`` with no path once the loader
    iterates or indexes into it.

    A node id of ``""`` is a string, so it would otherwise sail past the
    string check above and load as a node; it is rejected here, naming the
    path, like every other malformed id.
    """
    if not isinstance(data, dict):
        raise unloadable_graph(path, ["graph JSON must be an object"])
    # JSON object keys are always strings; the comprehension is what narrows the
    # loaded object to dict[str, Any] for the type checker (as payloads does).
    graph: dict[str, Any] = {key: value for key, value in data.items() if isinstance(key, str)}
    nodes = graph.get("nodes")
    edges = graph.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise unloadable_graph(path, ["graph JSON must contain list fields 'nodes' and 'edges'"])
    for node in nodes:
        if not isinstance(node, dict):
            raise unloadable_graph(path, [f"graph node must be an object: {node!r}"])
        node_id = node.get("id")
        if not isinstance(node_id, str):
            raise unloadable_graph(path, [f"graph node id must be a string: {node!r}"])
        if not node_id:
            raise unloadable_graph(path, [f"graph node id must be a non-empty string: {node!r}"])
    for edge in edges:
        if not isinstance(edge, dict):
            raise unloadable_graph(path, [f"graph edge must be an object: {edge!r}"])
        if not isinstance(edge.get("source"), str) or not isinstance(edge.get("target"), str):
            raise unloadable_graph(path, [f"graph edge endpoint must be a string: {edge!r}"])
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
        problem = f"edge(s) reference node id(s) absent from the file: {sorted(missing_ids)}"
        raise unloadable_graph(path, [problem])
    return graph


# A longer setup is an excursion, not a recipe for the state an action needs.
MAX_SETUP_ACTIONS = 8

PER_ACTION_LISTS = (
    "num_steps",
    "instructions",
    "target_observations",
    "schema_deltas",
    "setups",
    "open_menus",
    "action_sources",
    "setup_observations",
    "setup_alternatives",
)


def _valid_preparation(value: object) -> bool:
    entry = as_object_dict(value)
    setup, observation = entry.get("setup"), entry.get("observation")
    return (
        set(entry) == {"setup", "observation"}
        and isinstance(setup, list)
        and len(setup) <= MAX_SETUP_ACTIONS
        and all(isinstance(action, dict) for action in setup)
        and isinstance(observation, str)
        and bool(observation)
    )


def _valid_preparation_alternatives(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) <= MAX_SETUP_ACTIONS
        and all(_valid_preparation(entry) for entry in value)
    )


def edge_problems(edges: Iterable[dict[str, Any]]) -> list[str]:
    """Return what keeps the saved ``edges`` from loading: the edges without actions,
    and the edges whose per-action lists do not hold one entry per action."""
    without_actions: list[str] = []
    misaligned: list[str] = []
    invalid_anchors: list[str] = []
    for edge in edges:
        name = f"{edge['source']} -> {edge['target']}"
        actions = edge.get("actions")
        if not isinstance(actions, list) or not actions:
            without_actions.append(name)
        elif not all(
            isinstance(edge.get(key), list) and len(edge[key]) == len(actions)
            for key in PER_ACTION_LISTS
        ):
            misaligned.append(name)
        elif any(
            not _valid_preparation_alternatives(value) for value in edge["setup_alternatives"]
        ) or any(
            anchor is not None
            and (not isinstance(anchor, str) or not anchor or not isinstance(setup, list))
            for anchor, setup in zip(edge["setup_observations"], edge["setups"], strict=True)
        ):
            invalid_anchors.append(name)
    problems = []
    if without_actions:
        problems.append(f"edge(s) without actions: {without_actions}")
    if misaligned:
        problems.append(
            f"edge(s) without one entry per action in each of {', '.join(PER_ACTION_LISTS)}: "
            f"{misaligned}"
        )
    if invalid_anchors:
        problems.append(f"edge(s) with invalid preparation anchors: {invalid_anchors}")
    return problems


def member_problems(nodes: Iterable[dict[str, Any]], steps: Path) -> list[str]:
    """Return what keeps the saved ``nodes`` from loading: the nodes that do not list
    their member observations as a non-empty list of observation IDs, and the members
    whose image is missing from the step log ``steps``."""
    without_members: list[str] = []
    without_image: list[str] = []
    for node in nodes:
        observations = _listed_members(node)
        if observations is None:
            without_members.append(node["id"])
            continue
        without_image.extend(
            observation
            for observation in observations
            if not observation_image_path(steps, observation).is_file()
        )
    problems = []
    if without_members:
        problems.append(f"node(s) with no member observations: {without_members}")
    if without_image:
        problems.append(f"member(s) without an image in {steps}: {without_image}")
    return problems


def _listed_members(node: dict[str, Any]) -> list[str] | None:
    observations = node.get("observations")
    if not isinstance(observations, list):
        return None
    members: list[str] = [o for o in observations if isinstance(o, str)]
    return members if members and len(members) == len(observations) else None


def unloadable_graph(path: Path, problems: Iterable[str]) -> ValueError:
    """Return the ``ValueError`` that loading the graph at ``path`` raises for
    ``problems``, each naming what in the file is wrong."""
    return ValueError(
        f"{path}: {'; '.join(problems)}. Explore the app again into a new graph directory."
    )
