"""Precompute runtime image embeddings for every graph under a root folder.

Usage:
    uv run app-graph-embed --config configs/explore.yaml --app <app_name> [--recompute]
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml

from android_app_graph.embedding_cache import (
    compute_missing_image_embeddings,
    iter_screenshot_candidates,
    load_image_embeddings,
    resolve_image_embedding_settings,
    save_image_embeddings,
)
from android_app_graph.graph_files import (
    iter_graph_files,
    reference_screenshot_path,
    require_graph_shape,
)
from android_app_graph.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def load_graph_json(graph_path: Path) -> dict[str, Any]:
    with graph_path.open("r", encoding="utf-8") as f:
        return require_graph_shape(json.load(f), graph_path)


def _pending_candidates(
    pending: list[tuple[str, Path]], *, app_name: str, summary: dict[str, int]
) -> Iterator[tuple[str, str]]:
    """Wrap the shared lazy screenshot generator, counting a failed read into
    this run's own summary -- the accounting embedding_cache's shared helper
    has no reason to know about.
    """

    def _count_failed(_node_id: str) -> None:
        summary["skipped_failed"] += 1

    yield from iter_screenshot_candidates(pending, app_name=app_name, on_failed=_count_failed)


def precompute_graph_image_embeddings(
    graph_dir: Path,
    *,
    api_key: str,
    model: str,
    base_url: str,
    app_name: str | None = None,
    recompute: bool = False,
) -> dict[str, int]:
    summary: dict[str, int] = {
        "graphs": 0,
        "reference_screenshots": 0,
        "already_cached": 0,
        "computed": 0,
        "skipped_missing_screenshot": 0,
        "skipped_failed": 0,
    }

    for current_app_name, graph_path in iter_graph_files(graph_dir, app_name):
        summary["graphs"] += 1
        logger.info("[GRAPH] %s: selected %s", current_app_name, graph_path.name)
        graph_data = load_graph_json(graph_path)
        # --recompute discards the sidecar by writing an empty tagged payload
        # through the same writer as every other sidecar write, not by unlinking
        # it: unlink() on a symlinked sidecar detaches the link and leaves the
        # shared target it points at stale. Writing it up front also keeps the
        # discard when every call afterwards fails, since the loop only rewrites
        # the sidecar once a vector succeeds.
        node_ids = {node.get("id") for node in graph_data.get("nodes", []) if node.get("id")}

        embeddings: dict[str, list[float]]
        rewrite = False
        if recompute:
            save_image_embeddings(graph_path, {}, model=model)
            embeddings = {}
        else:
            loaded = load_image_embeddings(graph_path, model=model, node_ids=node_ids)
            embeddings = loaded.vectors
            rewrite = loaded.pruned > 0

        # A stat, not a read: deciding reference_screenshots/already_cached/
        # skipped_missing_screenshot must not pay for every uncached node's
        # base64 encoding before the first API call, and a cached node's
        # screenshot is never read at all.
        pending: list[tuple[str, Path]] = []
        for node in graph_data.get("nodes", []):
            node_id = node.get("id")
            if not node_id:
                continue
            screenshot_path = reference_screenshot_path(graph_path, node_id)
            if screenshot_path is None:
                summary["skipped_missing_screenshot"] += 1
                continue
            summary["reference_screenshots"] += 1
            if node_id in embeddings:
                summary["already_cached"] += 1
                continue
            pending.append((node_id, screenshot_path))

        run = compute_missing_image_embeddings(
            graph_path,
            embeddings,
            _pending_candidates(pending, app_name=current_app_name, summary=summary),
            api_key=api_key,
            model=model,
            base_url=base_url,
            app_name=current_app_name,
            rewrite=rewrite,
        )
        summary["computed"] += run.computed
        summary["skipped_failed"] += run.failed

    return summary


def load_image_embedding_settings(
    config_path: Path,
    *,
    graphs_override: Path | None,
    api_key_override: str | None,
    model_override: str | None,
    base_url_override: str | None,
) -> tuple[Path, str | None, str, str]:
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    exp_config = config.get("experiment") or {}
    translator_args = config.get("translator_args") or {}
    graph_dir = graphs_override or Path(
        translator_args.get("graph_dir") or exp_config.get("graph_dir", "graphs")
    )

    vlm_config = translator_args.get("vlm_config") or config.get("vlm") or {}
    settings = resolve_image_embedding_settings(
        vlm_config,
        model_override=model_override,
        api_key_override=api_key_override,
        base_url_override=base_url_override,
    )
    return graph_dir, settings.api_key, settings.model, settings.base_url


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="app-graph-embed",
        description="Precompute Gemini image embeddings for graph screenshots.",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=Path("configs/explore.yaml"),
        help="YAML config used for image_embedding API settings.",
    )
    parser.add_argument(
        "--graphs",
        type=Path,
        default=None,
        help="Optional graph-root override. Defaults to experiment.graph_dir from config.",
    )
    parser.add_argument(
        "--app",
        type=str,
        default=None,
        help="Optional single app folder to precompute.",
    )
    parser.add_argument("--model", type=str, default=None, help="Override image embedding model.")
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Override native Gemini API base URL.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key or env reference. Defaults to GEMINI_API_KEY / GOOGLE_API_KEY.",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Discard the existing embedding sidecar and rebuild every vector, "
        "regardless of its model tag; for a hand-edited sidecar or a deliberate "
        "refresh. A model change, or a sidecar that predates model tagging, is "
        "otherwise detected automatically and recomputed with no flag needed.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.config.expanduser().is_file():
        parser.error(f"config file not found: {args.config}")

    graph_dir, api_key, model, base_url = load_image_embedding_settings(
        args.config.expanduser(),
        graphs_override=args.graphs.expanduser() if args.graphs else None,
        api_key_override=args.api_key,
        model_override=args.model,
        base_url_override=args.base_url,
    )
    graph_dir = graph_dir.expanduser()
    if not graph_dir.exists():
        parser.error(f"graph root does not exist: {graph_dir}")

    if not api_key:
        parser.error("missing API key: pass --api-key or set GEMINI_API_KEY/GOOGLE_API_KEY")

    setup_logging(level=logging.INFO)

    summary = precompute_graph_image_embeddings(
        graph_dir,
        api_key=api_key,
        model=model,
        base_url=base_url,
        app_name=args.app,
        recompute=args.recompute,
    )
    logger.info(
        "Done: graphs=%d reference_screenshots=%d already_cached=%d computed=%d "
        "skipped_missing_screenshot=%d skipped_failed=%d",
        summary["graphs"],
        summary["reference_screenshots"],
        summary["already_cached"],
        summary["computed"],
        summary["skipped_missing_screenshot"],
        summary["skipped_failed"],
    )
    return 0
