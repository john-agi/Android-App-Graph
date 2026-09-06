"""Android-App-Graph v2 translator for AITK.

Loop-based architecture with a single small model (7-9B):

  Step 0 — Open app.

  Step 1+ — Reactive loop (every AITK step):
    1. IDENTIFY: Match current screen to a graph node (embedding similarity
       + model picks from top-K candidates).
    2. RECORD: Model checks if the screen has important info to remember.
    3. DECIDE: Model sees current node, 1-hop neighbors (with edge
       descriptions), self-loop actions, task, and memory — picks the next
       action: stay (execute a self-loop / free action) or go to a neighbor.
    4. EXECUTE: Action agent converts the refined instruction into one low-level
       action dictionary for this AITK turn.
    5. Loop back to step 1.

  Memory module stores:
    - Completed actions (what's been done)
    - Extracted information (prices, times, names read from screens)
    - Observations (anything the model flags as task-relevant)

  Key design: every model call is multiple-choice, slot-filling, or short
  extraction — never open-ended planning over the full graph.
"""

from __future__ import annotations

import base64
import json
import logging
import math
import os
import re
import subprocess
import time
from collections.abc import Callable, Iterator
from datetime import datetime
from pathlib import Path
from typing import Any, override

import httpx
import networkx as nx
from aitk.translators.base import BaseTranslator
from openai import OpenAI

from android_app_graph.payloads import as_float_list, as_str, as_str_dict
from android_app_graph.utils import resolve_env
from android_app_graph.utils.vlm_utils import (
    describe_page_and_state,
    get_gemini_native_image_embedding,
    predict_next_action,
    strip_json_fences,
)

logger = logging.getLogger("AITK - Android-App-Graph")

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

_SEP = "-" * 60
_SEP_THICK = "=" * 60
V2_CHAT_MAX_TOKENS = 3000
V2_PARSE_RETRIES = 1
V2_API_RETRIES = 2
V2_API_RETRY_BASE_DELAY_SECONDS = 2.0
RUNTIME_IMAGE_EMBEDDING_RETRIES = 2
RUNTIME_IMAGE_EMBEDDING_RETRY_BASE_DELAY_SECONDS = 2.0


NODE_IDENTIFY_PROMPT = """\
You are looking at a mobile app screen. Below are candidate screen descriptions \
from the app's navigation graph. Pick the one that best matches what you see, \
or pick "none" if nothing matches.

Each option includes an image embedding similarity score. Higher means the \
candidate screenshot is visually closer to the current screen. The candidate \
description is precise: every word in the description was chosen to distinguish \
that screen from similar graph nodes, so use those words as important evidence.

## Candidates
{candidates}

Think easily and briefly, under 10 sentences if needed. Do not explain your reasoning. Reply with ONLY the final letter (A, B, C, ...) \
or "none". Nothing else."""


RECORD_PROMPT = """\
You are a mobile app assistant extracting information from a screen.

## Task
{task}

## Today's date
{today}

## Memory so far
{memory}

Look at the current screen. Is there any information visible that is relevant \
to the task and NOT already in memory? This includes: numbers, prices, times, \
names, status messages, confirmation texts, error messages, or answers to \
questions in the task.

If yes, reply with a short factual statement of what you found.
If nothing new is relevant, reply with exactly "nothing".

Think easily and briefly, under 10 sentences if needed. Reply with ONLY the information or "nothing". No explanations."""


DECIDE_PROMPT = """\
You are a mobile app agent navigating an app to complete a task.

## Task
{task}

## Today's date
{today}

## Memory (what you know and have done)
{memory}

## Current screen
Node: {current_node}
Description: "{current_desc}"
{state_context}
## Available actions
{options}

## Instructions
Pick the best action to make progress on the task.
- If the task is fully complete (the answer is already in memory), pick "done".
- If you need to act on the current screen (type, search, tap a button), pick \
  a self-loop action and provide the refined instruction.
- If text has already been typed and now needs submission, use the exact \
  low-level instruction "Press enter" rather than vague wording such as \
  "submit the search query".
- If you need to navigate to another screen, pick that neighbor.
- If nothing in the graph helps, pick "free" and describe what to do.

Think easily and briefly, under 10 sentences if needed. Reply with ONLY a JSON object:
{{
  "choice": "<option letter>",
  "instruction": "<refined instruction for the action agent — replace generic \
terms with specific values from the task, e.g. 'search for London Bridge' \
instead of 'search for a location'>"
}}"""


DONE_PROMPT = """\
You are a mobile app assistant. The task is complete.

## Task
{task}

## Memory (all information collected)
{memory}

Based on the memory, provide the final answer to the task.
If the task asks a question, answer it directly.
If the task asks to perform an action, confirm what was done.

Think easily and briefly, under 10 sentences if needed. Reply with ONLY the answer, nothing else. Keep it concise."""


FREE_ACTION_PLAN_PROMPT = """\
You are choosing the next single UI instruction for a mobile action agent.

## Overall task
{task}

## Why graph guidance is unavailable
{reason}

## Memory
{memory}

## Recent action history
{action_history}

Look at the screenshot and produce exactly one immediate low-level instruction \
for the next visible UI action. The execution model will receive only your \
instruction, so make it specific and local to the current screen.
If text has already been typed and now needs submission, output exactly \
"Press enter" rather than vague wording such as "submit the search query".
Never answer with vague delegation such as "take the best action", "move toward \
the task", or "continue toward the goal". If graph guidance is unavailable, you \
must still choose one concrete visible action yourself, such as "Tap the search \
bar", "Press enter", or "Tap the back arrow".

Think easily and briefly, under 10 sentences if needed. Reply with ONLY the \
one-step instruction. No explanations."""


def _load_graph_from_json(path: Path) -> tuple[nx.DiGraph, dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise TypeError(f"Runtime graph JSON must be an object: {path}")
    data = as_str_dict(raw)
    if not isinstance(data.get("nodes"), list) or not isinstance(data.get("edges"), list):
        raise TypeError(f"Runtime graph JSON must contain list fields 'nodes' and 'edges': {path}")

    screenshots_dir = path.parent / (path.stem + "_screenshots")
    if not screenshots_dir.exists() and path.stem.endswith("_audited"):
        base_stem = path.stem.removesuffix("_audited")
        base_screenshots_dir = path.parent / f"{base_stem}_screenshots"
        if base_screenshots_dir.exists():
            screenshots_dir = base_screenshots_dir
    G = nx.DiGraph()
    for node_data in data.get("nodes", []):
        node_id = str(node_data["id"])
        ref_screenshot = None
        img_path = screenshots_dir / f"{node_id}.png"
        if img_path.exists():
            ref_screenshot = base64.b64encode(img_path.read_bytes()).decode("ascii")
        G.add_node(
            node_id,
            activity=node_data.get("activity", ""),
            page_description=node_data.get("page_description", ""),
            state_schema=node_data.get("state_schema", {}),
            last_detail_snapshot=node_data.get("last_detail_snapshot", {}),
            reference_screenshot=ref_screenshot,
            visit_count=node_data.get("visit_count", 0),
        )
    for edge_data in data.get("edges", []):
        edge_attrs = {
            "actions": edge_data.get("actions", []),
            "instructions": edge_data.get("instructions", []),
            "instruction_templates": edge_data.get("instruction_templates", []),
            "target_observations": edge_data.get("target_observations", []),
            "num_steps": edge_data.get("num_steps", []),
            "visit_count": edge_data.get("visit_count", 0),
        }
        if edge_data.get("schema_deltas"):
            edge_attrs["schema_deltas"] = edge_data["schema_deltas"]
        G.add_edge(str(edge_data["source"]), str(edge_data["target"]), **edge_attrs)

    return G, data


def _load_image_embeddings(graph_path: Path) -> dict[str, list[float]]:
    emb_path = _image_embeddings_path(graph_path)
    if not emb_path.exists():
        return {}
    with open(emb_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    embeddings: dict[str, list[float]] = {}
    for node_id, vector in as_str_dict(raw).items():
        numbers = as_float_list(vector)
        if numbers:
            embeddings[node_id] = numbers
    return embeddings


def _image_embeddings_path(graph_path: Path) -> Path:
    return graph_path.with_suffix(".image_emb.json")


def _save_image_embeddings(graph_path: Path, G: nx.DiGraph) -> None:
    embeddings = {
        node_id: data["image_embedding"]
        for node_id, data in G.nodes(data=True)
        if data.get("image_embedding")
    }
    emb_path = _image_embeddings_path(graph_path)
    with open(emb_path, "w", encoding="utf-8") as f:
        json.dump(embeddings, f, ensure_ascii=False)


def _iter_runtime_graph_files(graph_dir: Path) -> list[tuple[str, Path]]:
    """Return one graph JSON per app, preferring audited graphs.

    Runtime should not load audit reports, merge reports, or embedding sidecars.
    The app name comes from the graph directory name so eboox_audited.json still
    loads as app "eboox".
    """
    graph_files: list[tuple[str, Path]] = []
    if not graph_dir.exists():
        return graph_files

    for app_dir in sorted(p for p in graph_dir.iterdir() if p.is_dir()):
        app_name = app_dir.name
        audited = app_dir / f"{app_name}_audited.json"
        original = app_dir / f"{app_name}.json"
        if audited.exists():
            graph_files.append((app_name, audited))
        elif original.exists():
            graph_files.append((app_name, original))
    return graph_files


def _package_from_activity(activity: str) -> str:
    if "/" in activity:
        activity = activity.split("/", maxsplit=1)[0]
    parts = activity.split(".")
    if len(parts) >= 3:
        return ".".join(parts[:3])
    return activity


def _extract_packages_from_graph(G: nx.DiGraph) -> set[str]:
    packages = set()
    for _, data in G.nodes(data=True):
        activity = data.get("activity", "")
        if activity:
            packages.add(_package_from_activity(activity))
    return packages


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def _parse_model_choice(raw: str, valid_letters: str) -> str | None:
    answer = (raw or "").strip().upper()
    if answer == "NONE":
        return "NONE"
    if answer in valid_letters:
        return answer

    think_end = answer.rfind("</THINK>")
    if think_end != -1:
        post_think = answer[think_end + len("</THINK>") :].strip()
        parsed = _parse_model_choice(post_think, valid_letters)
        if parsed:
            return parsed

    final_match = re.search(
        r"\b(?:FINAL\s+(?:ANSWER|DECISION)|ANSWER|DECISION)\s*:\s*(NONE|[A-Z])\b",
        answer,
    )
    if final_match:
        choice = as_str(final_match.group(1), "")
        if choice == "NONE":
            return "NONE"
        if choice in valid_letters:
            return choice

    if re.search(r"\bNONE\b", answer):
        return "NONE"
    matches = [as_str(m, "") for m in re.findall(r"\b([A-Z])\b", answer)]
    for match in reversed(matches):
        if match in valid_letters:
            return match
    return None


def _parse_record_output(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return "nothing"

    think_end = text.lower().rfind("</think>")
    if think_end != -1:
        text = text[think_end + len("</think>") :].strip()

    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE).strip()
    final_matches = list(
        re.finditer(
            r"(?im)^\s*(?:final\s+answer|answer|information)\s*:\s*(.+)$",
            text,
        )
    )
    if final_matches:
        text = text[final_matches[-1].start(1) :].strip()
    else:
        markers = [
            "Thinking Process:",
            "Reasoning:",
            "Thought Process:",
            "Analysis:",
        ]
        for marker in markers:
            if text.lower().startswith(marker.lower()):
                paragraphs = [
                    stripped
                    for p in re.split(r"\n\s*\n", text)
                    if (stripped := as_str(p, "").strip())
                ]
                if len(paragraphs) > 1:
                    text = paragraphs[-1]
                break

    text = text.strip().strip('"')
    if text.lower().startswith("nothing"):
        return "nothing"
    if len(text) > 800 or len(re.findall(r"[.!?]", text)) > 10:
        return "nothing"
    return text or "nothing"


def _parse_one_step_instruction(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return ""

    think_end = text.lower().rfind("</think>")
    if think_end != -1:
        text = text[think_end + len("</think>") :].strip()

    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE).strip()
    final_matches = list(
        re.finditer(
            r"(?im)^\s*(?:final\s+answer|answer|instruction|next\s+action)\s*:\s*(.+)$",
            text,
        )
    )
    if final_matches:
        text = text[final_matches[-1].start(1) :].strip()

    text = text.strip().strip('"')
    if len(text) > 500:
        return ""
    generic_phrases = (
        "take one immediate visible ui action",
        "best moves toward the task",
        "move toward the task",
        "continue toward the goal",
        "take the best action",
    )
    if any(phrase in text.lower() for phrase in generic_phrases):
        return ""
    return text


def _extract_json_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, _end = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return as_str_dict(obj)
    return None


def _parse_json_object(text: str) -> dict[str, Any] | None:
    """Parse ``text`` as a JSON object, else scan it for the first embedded one."""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return _extract_json_object(text)
    return as_str_dict(parsed) if isinstance(parsed, dict) else None


def _parse_decide_output(raw: str) -> dict[str, Any] | None:
    text = strip_json_fences((raw or "").strip())
    think_end = text.lower().rfind("</think>")
    if think_end != -1:
        post_think = strip_json_fences(text[think_end + len("</think>") :].strip())
        parsed = _parse_json_object(post_think)
        if parsed is not None:
            return parsed
    return _parse_json_object(text)


def _call_with_retry[T](label: str, func: Callable[[], T]) -> T:
    """Run one API operation with up to three total attempts."""
    attempts = V2_API_RETRIES + 1
    for attempt in range(attempts):
        try:
            return func()
        except Exception as exc:
            if attempt >= attempts - 1:
                raise
            delay = V2_API_RETRY_BASE_DELAY_SECONDS * (2**attempt)
            logger.warning(
                "[API] %s failed; retrying in %.1fs (%d/%d). Error: %s",
                label,
                delay,
                attempt + 1,
                attempts - 1,
                exc,
            )
            time.sleep(delay)
    raise RuntimeError("unreachable")


def _chat_completion_content(
    client: OpenAI,
    *,
    parse_retries: int = V2_PARSE_RETRIES,
    **kwargs: Any,
) -> Iterator[tuple[int, str, bool]]:
    """Yield chat completion content, retrying at the caller's parse boundary."""
    for attempt in range(parse_retries + 1):
        resp = _call_with_retry(
            "chat completion",
            lambda: client.chat.completions.create(**kwargs),
        )
        content = as_str(resp.choices[0].message.content, "")
        yield attempt, content, attempt < parse_retries


def _make_no_proxy_client(cfg: dict[str, Any] | None) -> tuple[OpenAI, str]:
    """Create an OpenAI SDK client for non-embedding v2 calls.

    Native Google image embedding uses httpx directly and keeps the environment
    proxy. The OpenAI-compatible action/page-detail endpoints are often SSH
    forwarded locally, so they must ignore HTTP(S)_PROXY.
    """
    cfg = cfg or {}
    api_key = resolve_env(cfg.get("api_key")) or os.environ.get("OPENAI_API_KEY")
    base_url = resolve_env(cfg.get("base_url"))
    model = resolve_env(cfg.get("model")) or "gpt-4o"
    request_timeout = float(cfg.get("request_timeout", cfg.get("timeout", 60)))
    max_retries = int(cfg.get("max_retries", 0))

    kwargs: dict[str, Any] = {
        "api_key": api_key,
        "http_client": httpx.Client(trust_env=False),
        "timeout": request_timeout,
        "max_retries": max_retries,
    }
    if base_url:
        kwargs["base_url"] = base_url

    return OpenAI(**kwargs), model


class Memory:
    """Task memory that stores actions taken, observations, and extracted info."""

    def __init__(self) -> None:
        self.actions: list[str] = []
        self.info: list[str] = []
        self.observations: list[str] = []

    def add_action(self, action: str) -> None:
        self.actions.append(action)
        logger.info('[MEMORY] +action: "%s"', action)

    def add_info(self, info: str) -> None:
        if info and info.lower() != "nothing":
            self.info.append(info)
            logger.info('[MEMORY] +info: "%s"', info)

    def add_observation(self, obs: str) -> None:
        if obs:
            self.observations.append(obs)
            logger.info('[MEMORY] +obs: "%s"', obs)

    def format(self) -> str:
        """Format memory for prompt inclusion."""
        lines = []
        if self.actions:
            lines.append("Actions completed:")
            for i, a in enumerate(self.actions, 1):
                lines.append(f"  {i}. {a}")
        if self.info:
            lines.append("Information collected:")
            for i, info in enumerate(self.info, 1):
                lines.append(f"  {i}. {info}")
        if self.observations:
            lines.append("Observations:")
            for i, obs in enumerate(self.observations, 1):
                lines.append(f"  {i}. {obs}")
        return "\n".join(lines) if lines else "(empty)"

    def has_content(self) -> bool:
        return bool(self.actions or self.info or self.observations)


class UIKobeV2Translator(BaseTranslator):
    """Loop-based translator: identify → record → decide → execute."""

    def __init__(
        self,
        graph_dir: str = "graphs",
        vlm_config: dict[str, Any] | None = None,
        history_window: int = 5,
        max_pixels: int = 1_000_000,
    ) -> None:
        super().__init__()
        self.graph_dir = Path(graph_dir)
        self.history_window = history_window
        self.max_pixels = max_pixels
        vlm_config = vlm_config or {}

        # The planner and the action agent share one model, configured under "action".
        self.model_client, self.model_name = _make_no_proxy_client(vlm_config.get("action"))
        self.desc_client, self.desc_model = _make_no_proxy_client(vlm_config.get("page_detail"))
        embedding_cfg = vlm_config.get("embedding") or {}
        self.emb_model = resolve_env(embedding_cfg.get("model")) or "gemini-embedding-2-preview"
        image_embedding_cfg = vlm_config.get("image_embedding") or {}
        self.image_embedding_model = (
            resolve_env(image_embedding_cfg.get("model")) or "gemini-embedding-2"
        )
        self.image_embedding_api_key = (
            resolve_env(image_embedding_cfg.get("api_key"))
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )
        base_url_cfg = resolve_env(
            image_embedding_cfg.get("native_base_url") or image_embedding_cfg.get("base_url")
        )
        self.image_embedding_base_url = (
            base_url_cfg
            if base_url_cfg and "googleapis.com" in base_url_cfg
            else "https://generativelanguage.googleapis.com/v1beta"
        )

        self._graphs: dict[str, nx.DiGraph] = {}
        self._package_to_app: dict[str, str] = {}
        self._load_all_graphs()

        self._reset_task_state()

    def _reset_task_state(self) -> None:
        self._current_graph: nx.DiGraph | None = None
        self._current_node: str | None = None
        self._app_name: str = ""
        self._app_opened = False
        self._screen_w = 1080
        self._screen_h = 1920
        self._memory = Memory()
        self._step_count = 0

    def _get_runtime_image_embedding(self, screenshot_b64: str) -> list[float]:
        """Use native Gemini image embeddings for runtime node retrieval."""
        if not self.image_embedding_api_key:
            raise RuntimeError(
                "Native Gemini image embedding requires image_embedding.api_key "
                "or GEMINI_API_KEY/GOOGLE_API_KEY."
            )
        return get_gemini_native_image_embedding(
            self.image_embedding_api_key,
            screenshot_b64,
            model=self.image_embedding_model,
            base_url=self.image_embedding_base_url,
        )

    def _compute_runtime_image_embedding_with_retry(
        self,
        screenshot_b64: str,
        app_name: str,
        node_id: str,
    ) -> list[float]:
        attempts = RUNTIME_IMAGE_EMBEDDING_RETRIES + 1
        for attempt in range(attempts):
            try:
                return self._get_runtime_image_embedding(screenshot_b64)
            except Exception as exc:
                can_retry = attempt < attempts - 1
                if not can_retry:
                    raise
                delay = RUNTIME_IMAGE_EMBEDDING_RETRY_BASE_DELAY_SECONDS * (2**attempt)
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

    def _load_all_graphs(self) -> None:
        if not self.graph_dir.exists():
            logger.warning("Graph dir %s does not exist", self.graph_dir)
            return
        logger.info("[GRAPH] Loading runtime graphs from %s", self.graph_dir)
        for app_name, graph_file in _iter_runtime_graph_files(self.graph_dir):
            try:
                logger.info("[GRAPH] %s: selected %s", app_name, graph_file.name)
                G, _ = _load_graph_from_json(graph_file)
                image_embeddings = _load_image_embeddings(graph_file)
                for node_id, emb in image_embeddings.items():
                    if node_id in G:
                        G.nodes[node_id]["image_embedding"] = emb
                ref_count = sum(
                    1 for _, data in G.nodes(data=True) if data.get("reference_screenshot")
                )
                cached_count = sum(
                    1 for _, data in G.nodes(data=True) if data.get("image_embedding")
                )
                logger.info(
                    "[GRAPH] %s: nodes=%d edges=%d reference_screenshots=%d cached_image_embeddings=%d",
                    app_name,
                    G.number_of_nodes(),
                    G.number_of_edges(),
                    ref_count,
                    cached_count,
                )
                updated_image_cache = False
                for node_id, data in G.nodes(data=True):
                    if data.get("image_embedding") or not data.get("reference_screenshot"):
                        continue
                    try:
                        started = time.perf_counter()
                        data["image_embedding"] = self._compute_runtime_image_embedding_with_retry(
                            data["reference_screenshot"],
                            app_name,
                            node_id,
                        )
                        _save_image_embeddings(graph_file, G)
                        updated_image_cache = True
                        logger.info(
                            "[GRAPH] %s/%s: computed image embedding in %.1fs",
                            app_name,
                            node_id,
                            time.perf_counter() - started,
                        )
                    except Exception:
                        logger.exception(
                            "Runtime image embedding failed for graph %s node %s after retries; "
                            "continuing without this node embedding.",
                            app_name,
                            node_id,
                        )
                if updated_image_cache:
                    logger.info("[GRAPH] %s: image embedding cache updated", app_name)
                self._graphs[app_name] = G
                for pkg in _extract_packages_from_graph(G):
                    self._package_to_app[pkg] = app_name
                logger.info(
                    "[GRAPH] %s loaded: packages=%s",
                    app_name,
                    sorted(_extract_packages_from_graph(G)),
                )
            except Exception:
                logger.exception("Failed to load graph %s", graph_file)

    def _resolve_app_from_task(self, task: str) -> str | None:
        task_lower = task.lower()
        for app_name in self._graphs:
            if app_name.lower() in task_lower:
                return app_name
        return None

    def _get_graph_for_package(self, package: str) -> nx.DiGraph | None:
        app_name = self._package_to_app.get(package)
        if not app_name:
            app_name = self._package_to_app.get(_package_from_activity(package))
        if app_name and app_name in self._graphs:
            self._app_name = app_name
            return self._graphs[app_name]
        for pkg_prefix, name in self._package_to_app.items():
            if package.startswith(pkg_prefix) or pkg_prefix.startswith(package):
                self._app_name = name
                return self._graphs.get(name)
        return None

    def _identify_node(self, activity: str, screenshot: str) -> tuple[str | None, str]:
        """Match current screen to a graph node.

        1. Get page description via VLM.
        2. Compute embedding similarity against all nodes (same package).
        3. Take top-K candidates.
        4. Ask the model to pick the best match (multiple-choice).

        Returns (node_id, page_description).
        """
        G = self._current_graph
        if G is None:
            logger.warning("[IDENTIFY] No current graph is loaded")
            return None, ""

        current_pkg = _package_from_activity(activity)
        graph_packages = sorted(_extract_packages_from_graph(G))
        logger.info(
            "[IDENTIFY] activity=%s package=%s graph_app=%s graph_packages=%s",
            activity,
            current_pkg,
            self._app_name or "(unknown)",
            graph_packages,
        )

        same_pkg_descriptions: list[str] = []
        same_pkg_keys: list[str] = []
        same_pkg_node_count = 0
        for _, data in G.nodes(data=True):
            if _package_from_activity(data.get("activity", "")) == current_pkg:
                same_pkg_node_count += 1
                desc = data.get("page_description", "")
                if desc and desc not in same_pkg_descriptions:
                    same_pkg_descriptions.append(desc)
                for k in data.get("state_schema", {}):
                    if k not in same_pkg_keys:
                        same_pkg_keys.append(k)

        logger.info(
            "[IDENTIFY] same-package graph nodes=%d known_descriptions=%d known_state_keys=%d",
            same_pkg_node_count,
            len(same_pkg_descriptions),
            len(same_pkg_keys),
        )

        started = time.perf_counter()
        page_desc, _detail_snapshot, _elements = _call_with_retry(
            "page describe and state",
            lambda: describe_page_and_state(
                self.desc_client,
                screenshot,
                existing_nodes=same_pkg_descriptions or None,
                existing_keys=same_pkg_keys or None,
                model=self.desc_model,
            ),
        )
        logger.info(
            '[IDENTIFY] screen_description="%s" describe_time=%.1fs',
            page_desc,
            time.perf_counter() - started,
        )

        try:
            started = time.perf_counter()
            query_image_emb = self._compute_runtime_image_embedding_with_retry(
                screenshot,
                self._app_name or "(current)",
                "(current screenshot)",
            )
            logger.info(
                "[IDENTIFY] current screenshot image embedding computed in %.1fs",
                time.perf_counter() - started,
            )
        except Exception as exc:
            logger.error("Runtime image embedding failed for current screenshot. Error: %s", exc)
            raise
        candidates: list[tuple[str, float, str]] = []
        for node_id, data in G.nodes(data=True):
            if _package_from_activity(data.get("activity", "")) != current_pkg:
                continue
            node_image_emb = as_float_list(data.get("image_embedding"))
            if not node_image_emb:
                continue
            sim = _cosine_similarity(query_image_emb, node_image_emb)
            candidates.append((str(node_id), sim, as_str(data.get("page_description", ""), "")))

        candidates.sort(key=lambda x: x[1], reverse=True)

        logger.info("[IDENTIFY] image candidates=%d", len(candidates))
        for nid, sim, desc in candidates[:5]:
            logger.info('[IDENTIFY] candidate sim=%.3f node=%s desc="%s"', sim, nid, desc)

        if not candidates:
            if same_pkg_node_count == 0:
                logger.warning(
                    "[IDENTIFY] no candidates: current package %s has no nodes in selected graph %s",
                    current_pkg,
                    self._app_name or "(unknown)",
                )
            else:
                missing_embeddings = sum(
                    1
                    for _, data in G.nodes(data=True)
                    if _package_from_activity(data.get("activity", "")) == current_pkg
                    and not data.get("image_embedding")
                )
                logger.warning(
                    "[IDENTIFY] no candidates: %d same-package nodes lack image embeddings",
                    missing_embeddings,
                )
            return None, page_desc

        # Take top 4 for multiple-choice. Image similarity is retrieval only;
        # the model must always make the final node-identification decision.
        top_k = candidates[:4]
        letters = "ABCDEFGH"
        candidate_text = "\n".join(
            f'{letters[i]}) similarity={sim:.3f} node={nid}: "{desc}"'
            for i, (nid, sim, desc) in enumerate(top_k)
        )

        prompt = NODE_IDENTIFY_PROMPT.format(candidates=candidate_text)

        answer = None
        for attempt, raw_answer, can_retry in _chat_completion_content(
            self.model_client,
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{screenshot}"},
                        },
                    ],
                }
            ],
            max_tokens=V2_CHAT_MAX_TOKENS,
            temperature=0.0,
        ):
            answer = _parse_model_choice(raw_answer, letters[: len(top_k)])
            if answer:
                break
            if can_retry:
                logger.warning("[IDENTIFY] parse failed on attempt %d; retrying", attempt + 1)
        logger.info("[IDENTIFY] model_pick=%s", answer)

        if answer == "NONE" or not answer:
            logger.warning(
                "[IDENTIFY] model rejected candidates; not using image similarity as final identity. best_sim=%.3f",
                candidates[0][1],
            )
            return None, page_desc

        idx = letters.index(answer)
        chosen_node = top_k[idx][0]
        logger.info("[IDENTIFY] identified node=%s", chosen_node)
        return chosen_node, page_desc

    def _record_info(self, task: str, screenshot: str) -> None:
        """Check if the current screen has task-relevant info to remember."""
        prompt = RECORD_PROMPT.format(
            task=task,
            today=datetime.now().strftime("%Y-%m-%d (%A)"),
            memory=self._memory.format(),
        )

        result = "nothing"
        for attempt, raw_record, can_retry in _chat_completion_content(
            self.model_client,
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{screenshot}"},
                        },
                    ],
                }
            ],
            max_tokens=V2_CHAT_MAX_TOKENS,
            temperature=0.0,
        ):
            result = _parse_record_output(raw_record)
            if result:
                break
            if can_retry:
                logger.warning("[RECORD] parse failed on attempt %d; retrying", attempt + 1)
        logger.info("[RECORD] parsed=%s", result)
        self._memory.add_info(result)

    @staticmethod
    def _unpack_template(tmpl: dict[str, Any] | str) -> tuple[str, str]:
        """Extract (instruction_text, observation_text) from a template entry."""
        if isinstance(tmpl, dict):
            return (
                as_str(tmpl.get("template", ""), ""),
                as_str(tmpl.get("observation_template", ""), ""),
            )
        return str(tmpl), ""

    @staticmethod
    def _format_schema_delta(delta: dict[str, Any] | None) -> str:
        """Format a state transition hint for the small runtime model."""
        if not isinstance(delta, dict) or not delta:
            return ""

        parts: list[str] = []
        for key, change in delta.items():
            if isinstance(change, dict):
                before = change.get("before", "?")
                after = change.get("after", "?")
                parts.append(f"{key}: {before} -> {after}")
            else:
                parts.append(f"{key}: {change}")
        return ", ".join(parts)

    def _edge_effect_hint(self, edge_data: dict[str, Any], index: int | None = None) -> str:
        """Return a concise effect hint from schema deltas, if available."""
        deltas = edge_data.get("schema_deltas", [])
        if not deltas:
            return ""

        if index is not None and index < len(deltas):
            return self._format_schema_delta(deltas[index])

        merged: dict[str, Any] = {}
        for delta in deltas:
            if isinstance(delta, dict):
                for key, change in delta.items():
                    merged.setdefault(key, change)
        return self._format_schema_delta(merged)

    def _build_options(self, G: nx.DiGraph, node_id: str) -> tuple[str, list[dict[str, str]]]:
        """Build the option list for the DECIDE prompt.

        Returns (formatted_text, option_list) where each option is:
        {"letter": "A", "type": "self_loop"|"neighbor"|"done"|"free",
         "node": node_id, "instruction": str, ...}
        """
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        options: list[dict[str, str]] = []
        lines: list[str] = []
        idx = 0

        letter = letters[idx]
        options.append({"letter": letter, "type": "done"})
        lines.append(f"{letter}) DONE — the task is fully complete, answer is in memory")
        idx += 1

        if G.has_edge(node_id, node_id):
            edge_data = G[node_id][node_id]
            templates = edge_data.get("instruction_templates", [])
            observations = edge_data.get("target_observations", [])

            if templates:
                for tmpl in templates:
                    letter = letters[idx]
                    tmpl_text, obs_text = self._unpack_template(tmpl)
                    effect = self._edge_effect_hint(edge_data)
                    options.append(
                        {
                            "letter": letter,
                            "type": "self_loop",
                            "node": node_id,
                            "instruction": tmpl_text,
                            "effect": effect,
                        }
                    )
                    hint = f'{letter}) Stay here — "{tmpl_text}"'
                    if obs_text:
                        hint += f" → {obs_text}"
                    if effect:
                        hint += f" | changes: {effect}"
                    lines.append(hint)
                    idx += 1
            else:
                for i, raw_instr in enumerate(edge_data.get("instructions", [])):
                    letter = letters[idx]
                    instr = as_str(raw_instr, "")
                    effect = self._edge_effect_hint(edge_data, i)
                    options.append(
                        {
                            "letter": letter,
                            "type": "self_loop",
                            "node": node_id,
                            "instruction": instr,
                            "effect": effect,
                        }
                    )
                    hint = f'{letter}) Stay here — "{instr}"'
                    if i < len(observations) and observations[i]:
                        hint += f" → {observations[i]}"
                    if effect:
                        hint += f" | changes: {effect}"
                    lines.append(hint)
                    idx += 1

        for _, raw_neighbor, edge_data in G.out_edges(node_id, data=True):
            neighbor = str(raw_neighbor)
            if neighbor == node_id:
                continue
            neighbor_desc = as_str(G.nodes[neighbor].get("page_description", neighbor), neighbor)
            templates = edge_data.get("instruction_templates", [])
            observations = edge_data.get("target_observations", [])

            instr = ""
            obs = ""
            if templates:
                instr, obs = self._unpack_template(templates[0])
            else:
                instrs = edge_data.get("instructions", [])
                if instrs:
                    instr = as_str(instrs[0], "")
                if observations:
                    obs = as_str(observations[0], "")

            letter = letters[idx]
            effect = self._edge_effect_hint(edge_data, 0)
            options.append(
                {
                    "letter": letter,
                    "type": "neighbor",
                    "node": neighbor,
                    "instruction": instr,
                    "description": neighbor_desc,
                    "effect": effect,
                }
            )
            edge_hint = f' — "{instr}"' if instr else ""
            if obs:
                edge_hint += f" → {obs}"
            if effect:
                edge_hint += f" | changes: {effect}"
            lines.append(f'{letter}) Go to "{neighbor_desc}"{edge_hint}')
            idx += 1

        letter = letters[idx]
        options.append({"letter": letter, "type": "free"})
        lines.append(f"{letter}) FREE — do something not listed above (describe it)")
        idx += 1

        return "\n".join(lines), options

    def _decide(self, G: nx.DiGraph, task: str, node_id: str, screenshot: str) -> dict[str, str]:
        """Ask the model to pick the next action.

        Returns the chosen option dict with an added "instruction" key
        (the refined instruction from the model).
        """
        current_desc = as_str(G.nodes[node_id].get("page_description", ""), "")
        today = datetime.now()

        # Show state keys (from schema) so the model knows what parameters this screen has.
        # Values are omitted — the model can read them from the screenshot directly.
        state_schema = as_str_dict(G.nodes[node_id].get("state_schema", {}))
        if state_schema:
            keys_str = ", ".join(state_schema.keys())
            state_context = f"State parameters: [{keys_str}]\n"
        else:
            state_context = ""

        options_text, options_list = self._build_options(G, node_id)

        prompt = DECIDE_PROMPT.format(
            task=task,
            today=today.strftime("%Y-%m-%d (%A)"),
            memory=self._memory.format(),
            current_node=node_id,
            current_desc=current_desc,
            state_context=state_context,
            options=options_text,
        )

        result = None
        raw = ""
        for attempt, raw, can_retry in _chat_completion_content(
            self.model_client,
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{screenshot}"},
                        },
                    ],
                }
            ],
            max_tokens=V2_CHAT_MAX_TOKENS,
            temperature=0.0,
        ):
            result = _parse_decide_output(raw)
            if result is not None:
                break
            if can_retry:
                logger.warning("[DECIDE] parse failed on attempt %d; retrying", attempt + 1)
        if result is None:
            logger.warning(
                "Failed to parse DECIDE response. raw_len=%d raw_prefix=%r",
                len(raw),
                raw[:160],
            )
            return {
                "type": "free",
                "instruction": "",
                "reason": "DECIDE response could not be parsed",
            }

        choice_letter = as_str(result.get("choice", ""), "").upper()
        instruction = as_str(result.get("instruction", ""), "")
        logger.info(
            "[DECIDE] parsed choice=%s instruction=%r",
            choice_letter,
            instruction,
        )

        for opt in options_list:
            if opt["letter"] == choice_letter:
                chosen = {**opt}
                if instruction:
                    chosen["instruction"] = instruction
                elif not chosen.get("instruction"):
                    chosen["instruction"] = task
                return chosen

        logger.warning("DECIDE: unrecognized choice '%s'", choice_letter)
        return {
            "type": "free",
            "instruction": instruction,
            "reason": f"DECIDE returned unrecognized choice {choice_letter!r}",
        }

    def _plan_free_action(
        self,
        task: str,
        screenshot: str,
        reason: str,
        action_history: list[Any],
    ) -> str:
        history_text = (
            "\n".join(str(a) for a in action_history[-5:]) if action_history else "(none)"
        )
        prompt = FREE_ACTION_PLAN_PROMPT.format(
            task=task,
            reason=reason,
            memory=self._memory.format(),
            action_history=history_text,
        )
        instruction = ""
        for attempt, raw_instruction, can_retry in _chat_completion_content(
            self.model_client,
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{screenshot}"},
                        },
                    ],
                }
            ],
            max_tokens=V2_CHAT_MAX_TOKENS,
            temperature=0.0,
        ):
            instruction = _parse_one_step_instruction(raw_instruction)
            if instruction:
                break
            if can_retry:
                logger.warning("[FREE] parse failed on attempt %d; retrying", attempt + 1)

        if instruction:
            logger.info("[FREE] planned_instruction=%s", instruction)
            return instruction

        logger.warning("[FREE] planner failed; using conservative concrete fallback")
        return "Wait briefly for the current screen to finish loading."

    def _call_action_agent(
        self,
        task: str,
        action_history: list[str],
        screenshot: str,
        overall_task: str = "",
    ) -> tuple[dict[str, Any], str, str]:
        keyboard_hint = ""
        try:
            kb_check = subprocess.run(
                ["adb", "shell", "dumpsys", "input_method"],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
            if "mInputShown=true" in kb_check.stdout:
                keyboard_hint = " (Note: the soft keyboard is currently visible — a text field is focused and ready for typing.)"
        except (OSError, subprocess.SubprocessError) as exc:
            logger.debug("Soft keyboard probe failed: %s", exc)

        aitk_action, history_entry = _call_with_retry(
            "action agent",
            lambda: predict_next_action(
                self.model_client,
                screenshot,
                task + keyboard_hint,
                screen_w=self._screen_w,
                screen_h=self._screen_h,
                action_history=action_history,
                model=self.model_name,
                overall_task=overall_task,
            ),
        )
        observation = history_entry.split(" | ")[0] if " | " in history_entry else ""
        return aitk_action, observation, history_entry

    def _make_free_instruction(self, task: str, reason: str) -> str:
        """Constrain graph fallback so the action agent still emits one local UI action."""
        app_context = (
            f"You are already inside the {self._app_name} app. "
            if self._app_name
            else "You are already inside the target app. "
        )
        return (
            f"{app_context}"
            f"Graph guidance is unavailable because {reason}. "
            f"Take exactly one immediate visible UI action that moves toward this goal: {task}. "
            "Do not press Home, do not open the app, and do not leave the app unless the screenshot clearly shows you are outside it."
        )

    def _generate_answer(self, task: str) -> str:
        prompt = DONE_PROMPT.format(
            task=task,
            memory=self._memory.format(),
        )
        answer = ""
        for attempt, raw_answer, can_retry in _chat_completion_content(
            self.model_client,
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=V2_CHAT_MAX_TOKENS,
            temperature=0.0,
        ):
            answer = raw_answer.strip()
            if answer:
                return answer
            if can_retry:
                logger.warning("[DONE] parse failed on attempt %d; retrying", attempt + 1)
        return answer

    def _step(self, task: str, state: dict[str, Any], history: dict[str, Any]) -> str:
        screenshot = as_str(state.get("screenshot", ""), "")
        activity = as_str(state.get("activity", ""), "")
        package = as_str(state.get("package", ""), "")

        if not self._app_opened:
            self._app_opened = True
            app_name = self._resolve_app_from_task(task)
            if app_name:
                self._app_name = app_name
                logger.info(_SEP_THICK)
                logger.info("[OPEN] Launching %s", app_name)
                logger.info(_SEP_THICK)
                return self._make_response(
                    f"Opening app: {app_name}",
                    {"action": "open", "app": app_name},
                )
            logger.warning("[OPEN] Could not resolve app from task: %s", task)

        if self._current_graph is None:
            if self._app_name and self._app_name in self._graphs:
                self._current_graph = self._graphs[self._app_name]
            else:
                self._current_graph = self._get_graph_for_package(package)

        G = self._current_graph

        self._step_count += 1
        logger.info(_SEP_THICK)
        logger.info("[LOOP] Step %d", self._step_count)
        logger.info(_SEP_THICK)

        node_id, page_desc = self._identify_node(activity, screenshot)
        self._current_node = node_id
        logger.info('[IDENTIFY] Node: %s — "%s"', node_id, page_desc)

        self._record_info(task, screenshot)
        logger.info("[RECORD] Memory: %s", self._memory.format())

        if node_id is not None and G is not None:
            decision = self._decide(G, task, node_id, screenshot)
        else:
            reason = (
                "no graph is loaded"
                if G is None
                else "the current screen did not match any graph node"
            )
            logger.warning("[DECIDE] Falling back to free action: %s", reason)
            fallback_instruction = self._plan_free_action(
                task,
                screenshot,
                reason,
                history.get("actions", []),
            )
            decision = {"type": "free", "instruction": fallback_instruction}

        decision_type = decision.get("type", "free")
        instruction = decision.get("instruction", task)
        if decision_type == "free" and not instruction:
            reason = decision.get(
                "reason", "graph guidance did not produce an executable instruction"
            )
            instruction = self._plan_free_action(
                task,
                screenshot,
                reason,
                history.get("actions", []),
            )
            decision["instruction"] = instruction

        logger.info(
            '[DECIDE] type=%s node=%s instruction="%s"',
            decision_type,
            node_id,
            instruction,
        )

        if decision_type == "done":
            answer = self._generate_answer(task)
            logger.info(_SEP_THICK)
            logger.info("[DONE] Answer: %s", answer)
            logger.info("[DONE] Memory: %s", self._memory.format())
            logger.info(_SEP_THICK)
            return self._make_response(
                f"Task complete. Answer: {answer}",
                {"action": "end", "answer": answer},
            )

        logger.info(_SEP)
        logger.info('ACTION AGENT one-step instruction: "%s"', instruction)
        logger.info(_SEP)
        aitk_action, obs, history_entry = self._call_action_agent(
            instruction,
            [],
            screenshot,
            overall_task=task,
        )

        self._memory.add_action(instruction)
        if obs:
            self._memory.add_observation(obs)

        # Only the DECIDE step is allowed to end the task. If the grounding
        # model emits "end" for an intermediate graph action, wait one turn
        # and let the next screenshot drive recovery.
        if aitk_action.get("action") == "end":
            logger.warning(
                'Action agent returned "end" for non-done instruction: "%s"',
                instruction,
            )
            aitk_action = {"action": "wait", "time": 1}

        return self._make_response(f"Action: {history_entry}", aitk_action)

    @staticmethod
    def _make_response(message: str, aitk_action: dict[str, Any]) -> str:
        return json.dumps(
            {
                "message": f"[Android-App-Graph] {message}",
                "aitk_action": aitk_action,
            }
        )

    @override
    def to_agent(self, task: str, state: dict[str, Any], history: dict[str, Any]) -> str:
        if not history["actions"]:
            self._reset_task_state()
        return self._step(task, state, history)

    @override
    def to_device(self, action: str, width: int, height: int) -> dict[str, Any]:
        self._screen_w = width
        self._screen_h = height
        try:
            data = json.loads(action)
        except json.JSONDecodeError:
            logger.warning("Failed to parse action: %s", action)
            return {"action": "end", "answer": "parse error"}
        aitk_action = as_str_dict(data).get("aitk_action")
        if not isinstance(aitk_action, dict):
            return {"action": "end", "answer": ""}
        return as_str_dict(aitk_action)


def register(kargs: dict[str, Any]) -> UIKobeV2Translator:
    return UIKobeV2Translator(**kargs)
