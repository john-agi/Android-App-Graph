"""Graph-file discovery and the image-embedding sidecar cache.

Shared by runtime graph loading and offline precomputation so they cannot
drift on what counts as a usable cached vector or how a failed call is retried.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from android_app_graph.payloads import as_float_list, as_str_dict
from android_app_graph.retrying import call_with_retry
from android_app_graph.utils.vlm_utils import get_gemini_native_image_embedding

logger = logging.getLogger(__name__)

IMAGE_EMBEDDING_RETRIES = 2
IMAGE_EMBEDDING_RETRY_BASE_DELAY_SECONDS = 2.0


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


def reference_screenshots_dir(graph_path: Path) -> Path:
    """Return the directory holding a graph's per-node reference screenshots.

    ``GraphManager.save_graph`` writes re-explored screenshots under
    ``<stem>_screenshots`` (``demo_audited_screenshots`` for an audited graph),
    so that directory is preferred; a graph nobody has re-explored keeps its
    screenshots under the app directory's own name instead. Both runtime graph
    loading and offline precomputation must resolve to the same directory, or
    one sees screenshots the other reports missing.
    """
    stem_dir = graph_path.parent / f"{graph_path.stem}_screenshots"
    if stem_dir.exists():
        return stem_dir
    return graph_path.parent / f"{graph_path.parent.name}_screenshots"


def reference_screenshot_path(graph_path: Path, node_id: str) -> Path:
    """Return one node's reference screenshot path under ``reference_screenshots_dir``."""
    return reference_screenshots_dir(graph_path) / f"{node_id}.png"


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
