"""Precompute runtime image embeddings for every graph under a root folder."""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import time
from pathlib import Path

import yaml

from ui_kobe.utils import resolve_env
from ui_kobe.utils.vlm_utils import get_gemini_native_image_embedding

logger = logging.getLogger("ui_kobe.precompute_graph_image_embeddings")

IMAGE_EMBEDDING_RETRIES = 2
IMAGE_EMBEDDING_RETRY_BASE_DELAY_SECONDS = 2.0


def image_embeddings_path(graph_path: Path) -> Path:
    return graph_path.with_suffix(".image_emb.json")


def load_image_embeddings(graph_path: Path) -> dict[str, list[float]]:
    emb_path = image_embeddings_path(graph_path)
    if emb_path.exists():
        with open(emb_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_image_embeddings(graph_path: Path, embeddings: dict[str, list[float]]) -> None:
    with open(image_embeddings_path(graph_path), "w", encoding="utf-8") as f:
        json.dump(embeddings, f, ensure_ascii=False)


def iter_graph_files(graph_dir: Path, app_name: str | None = None) -> list[tuple[str, Path]]:
    selected = []
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


def load_graph_json(graph_path: Path) -> dict:
    with open(graph_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Graph JSON must be an object: {graph_path}")
    return data


def reference_screenshot_b64(graph_path: Path, node_id: str) -> str | None:
    screenshot_path = graph_path.parent / f"{graph_path.parent.name}_screenshots" / f"{node_id}.png"
    if not screenshot_path.exists():
        return None
    return base64.b64encode(screenshot_path.read_bytes()).decode("ascii")


def compute_embedding_with_retry(
    api_key: str,
    screenshot_b64: str,
    *,
    model: str,
    base_url: str,
    app_name: str,
    node_id: str,
) -> list[float]:
    attempts = IMAGE_EMBEDDING_RETRIES + 1
    for attempt in range(attempts):
        try:
            return get_gemini_native_image_embedding(
                api_key,
                screenshot_b64,
                model=model,
                base_url=base_url,
            )
        except Exception as exc:
            if attempt >= attempts - 1:
                raise
            delay = IMAGE_EMBEDDING_RETRY_BASE_DELAY_SECONDS * (2**attempt)
            logger.warning(
                "[GRAPH] %s/%s: image embedding failed; retrying in %.1fs (%d/%d). Error: %s",
                app_name,
                node_id,
                delay,
                attempt + 1,
                attempts - 1,
                exc,
            )
            time.sleep(delay)
    raise RuntimeError("unreachable")


def precompute_graph_image_embeddings(
    graph_dir: Path,
    *,
    api_key: str,
    model: str,
    base_url: str,
    app_name: str | None = None,
) -> dict[str, int]:
    summary = {
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
        embeddings = load_image_embeddings(graph_path)

        for node in graph_data.get("nodes", []):
            node_id = node.get("id")
            if not node_id:
                continue
            screenshot_b64 = reference_screenshot_b64(graph_path, node_id)
            if screenshot_b64 is None:
                summary["skipped_missing_screenshot"] += 1
                continue
            summary["reference_screenshots"] += 1
            if node_id in embeddings:
                summary["already_cached"] += 1
                continue

            try:
                started = time.perf_counter()
                embeddings[node_id] = compute_embedding_with_retry(
                    api_key,
                    screenshot_b64,
                    model=model,
                    base_url=base_url,
                    app_name=current_app_name,
                    node_id=node_id,
                )
                save_image_embeddings(graph_path, embeddings)
                summary["computed"] += 1
                logger.info(
                    "[GRAPH] %s/%s: computed image embedding in %.1fs",
                    current_app_name,
                    node_id,
                    time.perf_counter() - started,
                )
            except Exception as exc:
                summary["skipped_failed"] += 1
                logger.error(
                    "Runtime image embedding failed for graph %s node %s after retries; "
                    "continuing without this node embedding. Error: %s",
                    current_app_name,
                    node_id,
                    exc,
                )

    return summary


def load_image_embedding_settings(
    config_path: Path,
    *,
    graphs_override: Path | None,
    api_key_override: str | None,
    model_override: str | None,
    base_url_override: str | None,
) -> tuple[Path, str | None, str, str]:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    exp_config = config.get("experiment") or {}
    translator_args = config.get("translator_args") or {}
    graph_dir = graphs_override or Path(
        translator_args.get("graph_dir") or exp_config.get("graph_dir", "graphs")
    )

    vlm_config = translator_args.get("vlm_config") or config.get("vlm") or {}
    image_embedding_cfg = vlm_config.get("image_embedding") or {}
    model = (
        resolve_env(model_override)
        or resolve_env(image_embedding_cfg.get("model"))
        or "gemini-embedding-2"
    )
    base_url_cfg = base_url_override or resolve_env(
        image_embedding_cfg.get("native_base_url") or image_embedding_cfg.get("base_url")
    )
    base_url = (
        base_url_cfg
        if base_url_cfg and "googleapis.com" in base_url_cfg
        else "https://generativelanguage.googleapis.com/v1beta"
    )
    api_key = (
        resolve_env(api_key_override)
        or resolve_env(image_embedding_cfg.get("api_key"))
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
    )
    return graph_dir, api_key, model, base_url


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Precompute Gemini image embeddings for graph screenshots."
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
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

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

    summary = precompute_graph_image_embeddings(
        graph_dir,
        api_key=api_key,
        model=model,
        base_url=base_url,
        app_name=args.app,
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


if __name__ == "__main__":
    main()
