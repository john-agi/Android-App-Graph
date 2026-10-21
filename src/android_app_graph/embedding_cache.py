"""The image-embedding sidecar cache and its compute loop.

Shared by runtime graph loading and offline precomputation so they cannot
drift on what counts as a usable cached vector, how a failed call is retried,
or how newly computed vectors are logged and persisted.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping
from pathlib import Path
from typing import NamedTuple

from openrouter import OpenRouter

from android_app_graph.graph_files import encode_screenshot_b64, write_json_atomically
from android_app_graph.payloads import as_float_list, as_object_dict, as_str_dict
from android_app_graph.retrying import call_with_retry
from android_app_graph.utils import DEFAULT_REQUEST_TIMEOUT, resolve_model_settings
from android_app_graph.utils.vlm_utils import get_image_embedding

logger = logging.getLogger(__name__)


class ImageEmbeddingSettings(NamedTuple):
    model: str
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT


class ImageEmbedder(NamedTuple):
    """The client and settings every image-embedding request is sent with."""

    client: OpenRouter
    settings: ImageEmbeddingSettings


def resolve_image_embedding_settings(
    vlm_config: Mapping[str, object], *, model_override: str | None = None
) -> ImageEmbeddingSettings:
    """Resolve screenshot-embedding settings; omitted ones inherit the ``embedding`` role."""
    if model_override is not None:
        image_embedding_cfg = as_object_dict(vlm_config.get("image_embedding"))
        vlm_config = {
            **vlm_config,
            "image_embedding": {**image_embedding_cfg, "model": model_override},
        }
    settings = resolve_model_settings(vlm_config, roles=("image_embedding",))["image_embedding"]
    return ImageEmbeddingSettings(settings.model, settings.request_timeout)


def image_embeddings_path(graph_path: Path) -> Path:
    """Return the sidecar path holding a graph's cached image embeddings."""
    return graph_path.with_suffix(".image_emb.json")


def load_image_embeddings(
    graph_path: Path, *, model: str, node_ids: Collection[str]
) -> dict[str, list[float]]:
    """Return the cached embeddings for a graph, or ``{}`` when none are usable.

    A sidecar that cannot be read or parsed is an empty cache, never a reason
    to drop the app graph; a malformed or non-finite vector is dropped and logged.

    The sidecar format tags the file with the embedding model that wrote it
    (``{"model": ..., "embeddings": {...}}``), since a dimension match alone
    cannot prove two vectors came from the same model. A tag matching
    ``model`` returns its vectors; a different tag means the cached vectors
    sit in a model it did not come from. A bare ``{node_id: vector}`` file
    predates model tagging and has the same problem one step further back:
    its vectors carry no record of which model produced them at all, and
    accepting them would assert a provenance they never had. Both cases are
    treated alike -- every node recomputed as missing, ``{}`` returned after
    one warning naming the sidecar.

    An entry whose id is not in ``node_ids`` is dropped too, at
    ``logger.debug``: both callers (the runtime translator and offline
    precomputation) write this same dict back to the sidecar, so a node that
    vanished from the graph since it was cached must not survive the rewrite
    in either one. Required, not defaulted, so a caller cannot forget to pass
    the graph's current node ids and silently keep every stale entry forever.

    This function only ever reads: the prune above is an in-memory detail,
    dropped from the dict handed back and never written to disk. The sidecar
    is rewritten -- prune included -- the next time the caller's compute loop
    persists a computed vector; a vanished node's stray vector is harmless
    data until then. Logging the drop at debug rather than info keeps it from
    repeating as a visible line on every start of a fully cached graph, and
    keeps a read-only graph directory from ever seeing a write attempt from
    a load.
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

    sidecar_model = raw.get("model")
    # A legacy bare dict may hold a node called "model", so the tagged form is
    # recognised by its shape (string tag plus an embeddings object), not by the key.
    if not (isinstance(sidecar_model, str) and isinstance(raw.get("embeddings"), dict)):
        logger.warning(
            "Image embedding cache %s predates model tagging; its vectors' provenance is "
            "unknown, so they are recomputed under the current model %r",
            emb_path,
            model,
        )
        return {}
    if sidecar_model != model:
        logger.warning(
            "Image embedding cache %s was written by model %r, not the current model %r; "
            "treating it as empty so every node is recomputed",
            emb_path,
            sidecar_model,
            model,
        )
        return {}

    embeddings: dict[str, list[float]] = {}
    for node_id, vector in as_str_dict(raw.get("embeddings")).items():
        numbers = as_float_list(vector)
        if numbers:
            embeddings[node_id] = numbers
        else:
            logger.warning(
                "Dropping malformed image embedding for node %s in %s", node_id, emb_path
            )

    node_id_set = set(node_ids)
    stale_ids = [node_id for node_id in embeddings if node_id not in node_id_set]
    if stale_ids:
        logger.debug(
            "Image embedding cache %s: dropping %d vector(s) for node id(s) no longer "
            "present in the graph",
            emb_path,
            len(stale_ids),
        )
        for node_id in stale_ids:
            del embeddings[node_id]
    return embeddings


def save_image_embeddings(
    graph_path: Path, embeddings: dict[str, list[float]], *, model: str
) -> None:
    """Write ``embeddings`` to the graph's sidecar file, tagged with the
    embedding model that computed them, replacing it atomically.

    See ``graph_files.write_json_atomically`` for the write mechanism. The
    model tag is what ``load_image_embeddings`` uses to detect a model switch
    on its own, rather than relying on a dimension match (two different
    models can share a dimension) or an operator remembering ``--recompute``.
    A directory the process cannot write into makes
    ``compute_missing_image_embeddings`` log the failed cache write and keep
    the computed vectors in memory for the run, rather than persisting them.
    """
    payload = {"model": model, "embeddings": embeddings}
    write_json_atomically(image_embeddings_path(graph_path), payload, ensure_ascii=False)


def compute_embedding_with_retry(
    client: OpenRouter,
    screenshot_b64: str,
    *,
    settings: ImageEmbeddingSettings,
    app_name: str,
    node_id: str,
) -> list[float]:
    """Get an OpenRouter image embedding, asking once more for an unusable vector."""
    return call_with_retry(
        f"[GRAPH] {app_name}/{node_id}: image embedding",
        lambda: get_image_embedding(
            client,
            screenshot_b64,
            settings.model,
            request_timeout=settings.request_timeout,
        ),
    )


def iter_screenshot_candidates(
    pending: Iterable[tuple[str, Path]],
    *,
    app_name: str,
    on_failed: Callable[[str], None] | None = None,
) -> Iterator[tuple[str, str]]:
    """Encode each pending node's screenshot lazily, one at a time.

    Shared by offline precomputation (``commands.embed``) and runtime graph
    loading (``adapters.aitk_translator``) so the two cannot drift on this:
    encoding here rather than up front keeps at most one base64 payload
    resident, so a cold cache on a graph with hundreds of nodes never holds
    hundreds of MB of screenshots before the first API call.

    A candidate's path was only stat'd for existence when ``pending`` was
    built; by the time it is read and encoded here it may have become
    unreadable or gone. This generator is iterated from a plain ``for`` in
    ``compute_missing_image_embeddings``, outside that loop's own per-node
    ``except``, so an uncaught ``OSError`` here would abort the whole run --
    every remaining node and every remaining app -- rather than skip one
    node. Caught and logged here instead, a boundary this generator forms on
    behalf of its caller.
    """
    for node_id, screenshot_path in pending:
        try:
            screenshot_b64 = encode_screenshot_b64(screenshot_path)
        except OSError:
            logger.exception(
                "[GRAPH] %s/%s: reference screenshot could not be read; skipping",
                app_name,
                node_id,
            )
            if on_failed is not None:
                on_failed(node_id)
            continue
        yield node_id, screenshot_b64


class ImageEmbeddingRun(NamedTuple):
    computed: int
    failed: int


def compute_missing_image_embeddings(
    graph_path: Path,
    embeddings: dict[str, list[float]],
    candidates: Iterable[tuple[str, str]],
    *,
    embedder: ImageEmbedder,
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
                    embedder.client,
                    screenshot_b64,
                    settings=embedder.settings,
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
                save_image_embeddings(graph_path, embeddings, model=embedder.settings.model)
            except OSError:
                logger.exception(
                    "Failed to write image embedding cache for graph %s at %s",
                    app_name,
                    graph_path,
                )
    return ImageEmbeddingRun(computed=computed, failed=failed)
