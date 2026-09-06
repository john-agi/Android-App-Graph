"""Graph-file discovery and the image-embedding sidecar cache.

Shared by runtime graph loading and offline precomputation so they cannot
drift on what counts as a usable cached vector, how a failed call is retried,
or how newly computed vectors are logged and persisted.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import secrets
import shutil
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any, NamedTuple

from android_app_graph.payloads import as_float_list, as_str_dict
from android_app_graph.retrying import call_with_retry
from android_app_graph.utils import resolve_env
from android_app_graph.utils.vlm_utils import get_gemini_native_image_embedding

logger = logging.getLogger(__name__)

IMAGE_EMBEDDING_RETRIES = 2
IMAGE_EMBEDDING_RETRY_BASE_DELAY_SECONDS = 2.0
DEFAULT_IMAGE_EMBEDDING_MODEL = "gemini-embedding-2"
DEFAULT_NATIVE_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


class ImageEmbeddingSettings(NamedTuple):
    api_key: str | None
    model: str
    base_url: str


def resolve_image_embedding_settings(
    vlm_config: dict[str, Any],
    *,
    model_override: str | None = None,
    api_key_override: str | None = None,
    base_url_override: str | None = None,
) -> ImageEmbeddingSettings:
    """Resolve the native-Gemini image-embedding model, API key and base URL.

    Shared by runtime graph loading (``aitk_translator``) and offline
    precomputation (``commands.embed``) so an override, an env-var reference or
    the GEMINI_API_KEY/GOOGLE_API_KEY fallback cannot drift between the two
    callers. A base URL that does not name a googleapis.com host falls back to
    the default, since ``get_gemini_native_image_embedding`` only speaks the
    native Gemini API.
    """
    image_embedding_cfg = vlm_config.get("image_embedding") or {}
    model = (
        resolve_env(model_override)
        or resolve_env(image_embedding_cfg.get("model"))
        or DEFAULT_IMAGE_EMBEDDING_MODEL
    )
    base_url_cfg = resolve_env(base_url_override) or resolve_env(
        image_embedding_cfg.get("native_base_url") or image_embedding_cfg.get("base_url")
    )
    base_url = (
        base_url_cfg
        if base_url_cfg and "googleapis.com" in base_url_cfg
        else DEFAULT_NATIVE_GEMINI_BASE_URL
    )
    api_key = (
        resolve_env(api_key_override)
        or resolve_env(image_embedding_cfg.get("api_key"))
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
    )
    return ImageEmbeddingSettings(api_key=api_key, model=model, base_url=base_url)


def image_embeddings_path(graph_path: Path) -> Path:
    """Return the sidecar path holding a graph's cached image embeddings."""
    return graph_path.with_suffix(".image_emb.json")


def load_image_embeddings(graph_path: Path) -> dict[str, list[float]]:
    """Return the cached embeddings for a graph, or ``{}`` when none are usable.

    A sidecar that cannot be read or parsed is an empty cache, never a reason
    to drop the app graph; a malformed or non-finite vector is dropped and logged.
    """
    emb_path = image_embeddings_path(graph_path)
    if not emb_path.exists():
        return {}
    try:
        with emb_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError) as exc:
        logger.warning("Corrupt image embedding cache %s; treating it as empty: %s", emb_path, exc)
        return {}
    if not isinstance(raw, dict):
        logger.warning(
            "Image embedding cache %s is not a JSON object; treating it as empty", emb_path
        )
        return {}
    embeddings: dict[str, list[float]] = {}
    for node_id, vector in as_str_dict(raw).items():
        numbers = as_float_list(vector)
        if numbers:
            embeddings[node_id] = numbers
        else:
            logger.warning(
                "Dropping malformed image embedding for node %s in %s", node_id, emb_path
            )
    return embeddings


def save_image_embeddings(graph_path: Path, embeddings: dict[str, list[float]]) -> None:
    """Write ``embeddings`` to the graph's sidecar file.

    Dumped to a temporary file in the same directory and moved into place with
    ``os.replace``, so a crash mid-dump -- or a failed rename -- leaves the
    sidecar as either the previous complete file or the new one, never a
    truncated mix of both and never an orphaned temp file: the replace runs
    inside the same cleanup that unlinks the temp file on any other failure.

    A fresh sidecar gets exactly the mode ``open(path, "w")`` would give (0o666
    with the process umask applied by the kernel); a sidecar that already
    exists keeps its current mode across the rewrite, matching what in-place
    truncation used to do, so an operator's chmod on a shared graph directory
    survives a rewrite.

    Unlike the in-place truncation this replaced, which needed write
    permission only on the sidecar file itself, an atomic replace needs write
    permission on the graph directory too (to create and rename the temp
    file) -- the accepted cost of never leaving a truncated sidecar behind. A
    directory the process cannot write into makes ``compute_missing_image_embeddings``
    log the failed cache write and keep the computed vectors in memory for
    the run, rather than persisting them.
    """
    target = image_embeddings_path(graph_path)
    tmp_path = target.parent / f"{target.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp"
    # tempfile.mkstemp hardcodes mode 0600, which would make a sidecar precomputed
    # by one user (or a CI job) unreadable to another process -- e.g. an AITK
    # runtime -- reading the same shared graph directory as a different user.
    # os.open lets the kernel apply the umask the way open(path, "w") does; never
    # os.umask, which is process-global state.
    fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(embeddings, f, ensure_ascii=False)
        if target.exists():
            shutil.copymode(target, tmp_path)
        os.replace(tmp_path, target)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def compute_embedding_with_retry(
    api_key: str,
    screenshot_b64: str,
    *,
    model: str,
    base_url: str,
    app_name: str,
    node_id: str,
) -> list[float]:
    """Get a native Gemini image embedding, retrying on failure with backoff."""
    return call_with_retry(
        f"[GRAPH] {app_name}/{node_id}: image embedding",
        lambda: get_gemini_native_image_embedding(
            api_key,
            screenshot_b64,
            model=model,
            base_url=base_url,
        ),
        retries=IMAGE_EMBEDDING_RETRIES,
        base_delay=IMAGE_EMBEDDING_RETRY_BASE_DELAY_SECONDS,
    )


class ImageEmbeddingRun(NamedTuple):
    computed: int
    failed: int


def compute_missing_image_embeddings(
    graph_path: Path,
    embeddings: dict[str, list[float]],
    candidates: Iterable[tuple[str, str]],
    *,
    api_key: str,
    model: str,
    base_url: str,
    app_name: str,
) -> ImageEmbeddingRun:
    """Compute an image embedding for every ``(node_id, screenshot_b64)`` candidate.

    Shared by offline precomputation (``commands.embed``) and runtime graph
    loading (``adapters.aitk_translator``) so the two cannot drift on retry,
    logging or persistence behaviour. ``embeddings`` is the cache as already
    loaded; each computed vector is added to it in place, and it is what gets
    written back, so a caller that pre-filters or prunes entries controls
    exactly what the sidecar ends up holding.

    The sidecar is written once, after the whole loop, since rewriting it after
    every node is O(N^2) for a cold cache. The write runs in a ``finally`` so a
    ``KeyboardInterrupt`` or ``SystemExit`` partway through the loop still
    persists every embedding computed before it -- those are paid API calls,
    and the interrupt itself keeps propagating. A failed write is caught as
    ``OSError`` and only logged, since a cache write failure must not drop a
    graph that loaded and computed its embeddings fine.
    """
    computed = 0
    failed = 0
    try:
        for node_id, screenshot_b64 in candidates:
            try:
                started = time.perf_counter()
                embeddings[node_id] = compute_embedding_with_retry(
                    api_key,
                    screenshot_b64,
                    model=model,
                    base_url=base_url,
                    app_name=app_name,
                    node_id=node_id,
                )
                computed += 1
                logger.info(
                    "[GRAPH] %s/%s: computed image embedding in %.1fs",
                    app_name,
                    node_id,
                    time.perf_counter() - started,
                )
            except Exception:
                failed += 1
                logger.exception(
                    "Runtime image embedding failed for graph %s node %s after retries; "
                    "continuing without this node embedding.",
                    app_name,
                    node_id,
                )
    finally:
        if computed:
            try:
                save_image_embeddings(graph_path, embeddings)
            except OSError:
                logger.exception(
                    "Failed to write image embedding cache for graph %s at %s",
                    app_name,
                    graph_path,
                )
    return ImageEmbeddingRun(computed=computed, failed=failed)


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


def reference_screenshot_b64(graph_path: Path, node_id: str) -> str | None:
    """Return one node's reference screenshot as base64, or ``None`` when it has none."""
    screenshot_path = reference_screenshot_path(graph_path, node_id)
    if screenshot_path is None:
        return None
    return base64.b64encode(screenshot_path.read_bytes()).decode("ascii")


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
