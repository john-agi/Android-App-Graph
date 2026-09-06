"""The image-embedding sidecar cache and its compute loop.

Shared by runtime graph loading and offline precomputation so they cannot
drift on what counts as a usable cached vector, how a failed call is retried,
or how newly computed vectors are logged and persisted.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any, NamedTuple

from android_app_graph.graph_files import write_json_atomically
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


def load_image_embeddings(graph_path: Path, *, model: str) -> dict[str, list[float]]:
    """Return the cached embeddings for a graph, or ``{}`` when none are usable.

    A sidecar that cannot be read or parsed is an empty cache, never a reason
    to drop the app graph; a malformed or non-finite vector is dropped and logged.

    The current sidecar format tags the file with the embedding model that
    wrote it (``{"model": ..., "embeddings": {...}}``), since a dimension
    match alone cannot prove two vectors came from the same model. A tag
    matching ``model`` returns its vectors; a different tag means the cached
    vectors sit in a model it did not come from, so every node must be
    recomputed as missing -- ``{}`` is returned after one warning naming both
    models. A bare ``{node_id: vector}`` file predates model tagging (an
    older release, or a hand-edited sidecar); it is accepted the same as
    always, logged once at info level, and gets tagged the next time this
    graph's embeddings are saved.
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
    if isinstance(sidecar_model, str) and isinstance(raw.get("embeddings"), dict):
        if sidecar_model != model:
            logger.warning(
                "Image embedding cache %s was written by model %r, not the current model %r; "
                "treating it as empty so every node is recomputed",
                emb_path,
                sidecar_model,
                model,
            )
            return {}
        raw_vectors = raw.get("embeddings")
    else:
        logger.info(
            "Image embedding cache %s predates model tagging; it will be tagged the next "
            "time this graph's embeddings are saved",
            emb_path,
        )
        raw_vectors = raw

    embeddings: dict[str, list[float]] = {}
    for node_id, vector in as_str_dict(raw_vectors).items():
        numbers = as_float_list(vector)
        if numbers:
            embeddings[node_id] = numbers
        else:
            logger.warning(
                "Dropping malformed image embedding for node %s in %s", node_id, emb_path
            )
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
                save_image_embeddings(graph_path, embeddings, model=model)
            except OSError:
                logger.exception(
                    "Failed to write image embedding cache for graph %s at %s",
                    app_name,
                    graph_path,
                )
    return ImageEmbeddingRun(computed=computed, failed=failed)
