"""Graph-file discovery and image-embedding sidecar cache.

Runtime graph loading (``adapters.aitk_translator``) and offline embedding
precomputation (``commands.embed``) both need to find one JSON graph file per
app (preferring the audited file over the plain one), read/write its
``.image_emb.json`` sidecar of cached image embeddings, and compute a fresh
embedding with the same retry-with-backoff policy. This is the single place
that does all three, so the two callers cannot drift on what counts as a
usable cached vector or how a failed embedding call is retried.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from android_app_graph.payloads import as_float_list, as_str_dict
from android_app_graph.utils.vlm_utils import get_gemini_native_image_embedding

logger = logging.getLogger(__name__)

IMAGE_EMBEDDING_RETRIES = 2
IMAGE_EMBEDDING_RETRY_BASE_DELAY_SECONDS = 2.0


def image_embeddings_path(graph_path: Path) -> Path:
    """Return the sidecar path holding a graph's cached image embeddings."""
    return graph_path.with_suffix(".image_emb.json")


def load_image_embeddings(graph_path: Path) -> dict[str, list[float]]:
    """Return the cached embeddings for a graph, or ``{}`` when none are usable.

    A sidecar that fails to parse as JSON is treated as "no cache" rather than
    raising: a truncated or corrupt cache file must not take down the whole app
    graph it caches for. A malformed or empty vector for one node is dropped
    (not kept as ``[]``) and logged, naming the node.
    """
    emb_path = image_embeddings_path(graph_path)
    if not emb_path.exists():
        return {}
    try:
        with emb_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except json.JSONDecodeError:
        logger.warning("Corrupt image embedding cache %s; treating it as empty", emb_path)
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
    """Write ``embeddings`` to the graph's sidecar file."""
    with image_embeddings_path(graph_path).open("w", encoding="utf-8") as f:
        json.dump(embeddings, f, ensure_ascii=False)


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
    attempts = IMAGE_EMBEDDING_RETRIES + 1
    for attempt in range(attempts):
        try:
            return get_gemini_native_image_embedding(
                api_key,
                screenshot_b64,
                model=model,
                base_url=base_url,
            )
        # A retry loop is a boundary (AGENTS.md): re-raise on the last attempt,
        # log the traceback on every earlier one.
        except Exception:
            if attempt >= attempts - 1:
                raise
            delay = IMAGE_EMBEDDING_RETRY_BASE_DELAY_SECONDS * (2**attempt)
            logger.warning(
                "[GRAPH] %s/%s: image embedding failed; retrying in %.1fs (%d/%d).",
                app_name,
                node_id,
                delay,
                attempt + 1,
                attempts - 1,
                exc_info=True,
            )
            time.sleep(delay)
    msg = "compute_embedding_with_retry exhausted its attempts without raising"
    raise RuntimeError(msg)


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
