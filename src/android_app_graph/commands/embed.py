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
from openrouter import OpenRouter

from android_app_graph.embedding_cache import (
    ImageEmbeddingSettings,
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
from android_app_graph.utils import open_client, resolve_api_key
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
    client: OpenRouter,
    settings: ImageEmbeddingSettings,
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
        node_ids = {node["id"] for node in graph_data["nodes"]}

        embeddings: dict[str, list[float]]
        if recompute:
            try:
                # Preserve symlink targets and invalidate stale data even if every call fails.
                save_image_embeddings(graph_path, {}, model=settings.model)
            except OSError:
                # Avoid mixing fresh vectors with a cache we could not invalidate.
                logger.exception(
                    "Failed to discard image embedding cache for graph %s at %s",
                    current_app_name,
                    graph_path,
                )
                continue
            embeddings = {}
        else:
            embeddings = load_image_embeddings(graph_path, model=settings.model, node_ids=node_ids)

        # Defer encoding to keep memory bounded and avoid reading cached screenshots.
        pending: list[tuple[str, Path]] = []
        for node in graph_data["nodes"]:
            node_id = node["id"]
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
            client=client,
            settings=settings,
            app_name=current_app_name,
        )
        summary["computed"] += run.computed
        summary["skipped_failed"] += run.failed

    return summary


def load_image_embedding_settings(
    config_path: Path,
    *,
    graphs_override: Path | None,
    model_override: str | None,
) -> tuple[Path, ImageEmbeddingSettings]:
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    exp_config = config.get("experiment") or {}
    translator_args = config.get("translator_args") or {}
    graph_dir = graphs_override or Path(
        translator_args.get("graph_dir") or exp_config.get("graph_dir", "graphs")
    )

    vlm_config = translator_args.get("vlm_config") or config.get("vlm") or {}
    return graph_dir, resolve_image_embedding_settings(vlm_config, model_override=model_override)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="app-graph-embed",
        description="Precompute OpenRouter image embeddings for graph screenshots.",
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

    try:
        graph_dir, settings = load_image_embedding_settings(
            args.config.expanduser(),
            graphs_override=args.graphs.expanduser() if args.graphs else None,
            model_override=args.model,
        )
    except ValueError as exc:
        parser.error(str(exc))
    graph_dir = graph_dir.expanduser()
    if not graph_dir.exists():
        parser.error(f"graph root does not exist: {graph_dir}")
    try:
        api_key = resolve_api_key()
    except ValueError as exc:
        parser.error(str(exc))

    setup_logging(level=logging.INFO)

    with open_client(api_key) as client:
        summary = precompute_graph_image_embeddings(
            graph_dir,
            client=client,
            settings=settings,
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
