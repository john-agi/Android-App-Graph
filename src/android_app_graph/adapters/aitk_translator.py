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

import json
import logging
import re
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, NotRequired, TypedDict, override

import httpx
import networkx as nx
from aitk.translators.base import BaseTranslator
from openai import OpenAI

from android_app_graph import device
from android_app_graph.android_packages import package_from_activity
from android_app_graph.embedding_cache import (
    compute_embedding_with_retry,
    compute_missing_image_embeddings,
    iter_graph_files,
    load_image_embeddings,
    reference_screenshot_b64,
    require_known_edge_endpoints,
    resolve_image_embedding_settings,
)
from android_app_graph.payloads import as_int, as_list, as_str, as_str_dict
from android_app_graph.retrying import call_with_retry
from android_app_graph.utils import make_client
from android_app_graph.utils.vlm_utils import (
    build_image_message,
    describe_page_and_state,
    predict_next_action,
    score_by_cosine,
    strip_json_fences,
)

logger = logging.getLogger("AITK - Android-App-Graph")

_SEP = "-" * 60
_SEP_THICK = "=" * 60
V2_CHAT_MAX_TOKENS = 3000
V2_PARSE_RETRIES = 1
# Format-agnostic: it is appropriate whether the caller's parse expects a letter,
# JSON, or free text, so one reminder covers every _ask_with_screenshot caller.
V2_PARSE_RETRY_HINT = (
    "Your previous reply could not be parsed. Reply in exactly the format "
    "requested above and nothing else."
)

_OptionType = Literal["done", "self_loop", "neighbor", "free"]


class _Option(TypedDict):
    """One DECIDE menu entry; the keys present depend on ``type``."""

    letter: str
    type: _OptionType
    node: NotRequired[str]
    instruction: NotRequired[str]
    description: NotRequired[str]
    effect: NotRequired[str]


class _Decision(TypedDict):
    """The outcome of ``_decide``; a "free" fallback carries ``reason`` instead of a node."""

    type: _OptionType
    instruction: str
    letter: NotRequired[str]
    node: NotRequired[str]
    description: NotRequired[str]
    effect: NotRequired[str]
    reason: NotRequired[str]


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


def _load_graph_from_json(path: Path) -> nx.DiGraph:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise TypeError(f"Runtime graph JSON must be an object: {path}")
    data = raw
    if not isinstance(data.get("nodes"), list) or not isinstance(data.get("edges"), list):
        raise TypeError(f"Runtime graph JSON must contain list fields 'nodes' and 'edges': {path}")

    G = nx.DiGraph()
    for node_data in data.get("nodes", []):
        if not isinstance(node_data.get("id"), str):
            raise TypeError(f"Runtime graph node id must be a string: {node_data!r} in {path}")
        node_id = node_data["id"]
        ref_screenshot = reference_screenshot_b64(path, node_id)
        # `.get(key, default)` keeps a present-but-null value, so every field a reader
        # iterates or indexes is narrowed here once rather than at each read site.
        G.add_node(
            node_id,
            activity=as_str(node_data.get("activity"), ""),
            page_description=as_str(node_data.get("page_description"), ""),
            state_schema=as_str_dict(node_data.get("state_schema")),
            last_detail_snapshot=as_str_dict(node_data.get("last_detail_snapshot")),
            reference_screenshot=ref_screenshot,
            visit_count=node_data.get("visit_count", 0),
        )

    edge_specs: list[tuple[str, str, dict[str, Any]]] = []
    for edge_data in data.get("edges", []):
        if not isinstance(edge_data.get("source"), str) or not isinstance(
            edge_data.get("target"), str
        ):
            raise TypeError(
                f"Runtime graph edge endpoints must be strings: {edge_data!r} in {path}"
            )
        source = edge_data["source"]
        target = edge_data["target"]
        edge_attrs: dict[str, Any] = {
            "actions": edge_data.get("actions", []),
            "instructions": as_list(edge_data.get("instructions")),
            "instruction_templates": as_list(edge_data.get("instruction_templates")),
            "target_observations": as_list(edge_data.get("target_observations")),
            "num_steps": edge_data.get("num_steps", []),
            "visit_count": edge_data.get("visit_count", 0),
        }
        if edge_data.get("schema_deltas"):
            edge_attrs["schema_deltas"] = as_list(edge_data["schema_deltas"])
        edge_specs.append((source, target, edge_attrs))

    # Checked before any edge is added: networkx's add_edge silently creates an
    # attribute-less node for an undefined endpoint, so a graph with one bad edge
    # must load none of its edges rather than a partially-formed graph.
    require_known_edge_endpoints(data, path)

    for source, target, edge_attrs in edge_specs:
        G.add_edge(source, target, **edge_attrs)

    return G


def _extract_packages_from_graph(G: nx.DiGraph) -> set[str]:
    packages = set()
    for _, data in G.nodes(data=True):
        activity = data.get("activity", "")
        if activity:
            packages.add(package_from_activity(activity))
    return packages


def _parse_model_choice(raw: str, valid_letters: str) -> str | None:
    text = (raw or "").strip()
    answer = text.upper()
    if answer == "NONE":
        return "NONE"
    if len(answer) == 1 and answer in valid_letters:
        return answer

    # Case-insensitive here only to find the tag; the slice keeps original case
    # so the free-text fallbacks below see what the model actually wrote.
    think_end = text.lower().rfind("</think>")
    if think_end != -1:
        post_think = text[think_end + len("</think>") :].strip()
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

    # Free text beyond an explicit "Answer:"/</think> form: the last answer signal
    # in the text wins, whether that is a named letter or a "NONE" rejection,
    # because a model states its conclusion last ("None of the others fit, so B"
    # answers B; "Neither A nor B match; none of them." answers NONE). Uppercase
    # letters are scanned in the original case, so .upper() cannot turn a
    # lowercase article ("a") into a false option letter.
    last_letter_pos = -1
    last_letter = ""
    for letter_match in re.finditer(r"\b([A-Z])\b", text):
        letter = as_str(letter_match.group(1), "")
        # A standalone "A"/"I" immediately followed by a lowercase word is the
        # English article or pronoun, not a named answer letter; that case is left
        # to the strict-format retry (V2_PARSE_RETRY_HINT) rather than guessed.
        if letter in ("A", "I") and re.match(r"\s+[a-z]", text[letter_match.end() :]):
            continue
        if letter in valid_letters:
            last_letter_pos, last_letter = letter_match.start(), letter

    last_none_pos = -1
    # Scanned on the original text, not its .upper() copy: upper-casing can change
    # a string's length (e.g. "ß" -> "SS"), which would shift the offsets compared
    # against the letter positions above.
    for none_match in re.finditer(r"\bNONE\b", text, re.IGNORECASE):
        last_none_pos = none_match.start()

    if last_letter_pos == -1 and last_none_pos == -1:
        return None
    return "NONE" if last_none_pos > last_letter_pos else last_letter


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
            return obj
    return None


def _parse_json_object(text: str) -> dict[str, Any] | None:
    """Parse ``text`` as a JSON object, else scan it for the first embedded one."""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return _extract_json_object(text)
    return parsed if isinstance(parsed, dict) else None


def _parse_decide_output(raw: str) -> dict[str, Any] | None:
    text = strip_json_fences((raw or "").strip())
    think_end = text.lower().rfind("</think>")
    if think_end != -1:
        post_think = strip_json_fences(text[think_end + len("</think>") :].strip())
        parsed = _parse_json_object(post_think)
        if parsed is not None:
            return parsed
    return _parse_json_object(text)


def _chat_completion_content(
    client: OpenAI,
    **kwargs: Any,
) -> str:
    """Run one chat completion and return its content."""
    resp = call_with_retry(
        "[API] chat completion",
        lambda: client.chat.completions.create(**kwargs),
    )
    # A filtered or refused completion can come back with zero choices; that
    # is empty content for the parse-retry to handle, not an IndexError.
    return as_str(resp.choices[0].message.content, "") if resp.choices else ""


def _make_no_proxy_client(cfg: dict[str, Any] | None) -> tuple[OpenAI, str]:
    """Create an OpenAI SDK client for non-embedding v2 calls.

    Native Google image embedding uses httpx directly and keeps the environment
    proxy. The OpenAI-compatible action/page-detail endpoints are often SSH
    forwarded locally, so they must ignore HTTP(S)_PROXY.
    """
    cfg = cfg or {}
    request_timeout = float(cfg.get("request_timeout", cfg.get("timeout", 60)))
    max_retries = int(cfg.get("max_retries", 0))
    return make_client(
        cfg,
        http_client=httpx.Client(trust_env=False),
        timeout=request_timeout,
        max_retries=max_retries,
    )


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


class UIKobeV2Translator(BaseTranslator):
    """Loop-based translator: identify → record → decide → execute."""

    def __init__(
        self,
        graph_dir: str = "graphs",
        vlm_config: dict[str, Any] | None = None,
        max_pixels: int = 1_000_000,
    ) -> None:
        super().__init__()
        # Set here, not at import time, so importing this module has no global side effect.
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)

        self.graph_dir = Path(graph_dir)
        # Accepted because AITK forwards every translator_args key and its controller.yaml
        # sets max_pixels; this translator sends screenshots unresized.
        self.max_pixels = max_pixels
        vlm_config = vlm_config or {}

        # The planner and the action agent share one model, configured under "action".
        self.model_client, self.model_name = _make_no_proxy_client(vlm_config.get("action"))
        self.desc_client, self.desc_model = _make_no_proxy_client(vlm_config.get("page_detail"))
        image_embedding_settings = resolve_image_embedding_settings(vlm_config)
        self.image_embedding_model = image_embedding_settings.model
        self.image_embedding_api_key = image_embedding_settings.api_key
        self.image_embedding_base_url = image_embedding_settings.base_url

        self._graphs: dict[str, nx.DiGraph] = {}
        self._package_to_app: dict[str, str] = {}
        self._load_all_graphs()

        # A device property: only to_device updates it, and a task reset must not.
        self._screen_w = 1080
        self._screen_h = 1920
        self._reset_task_state()

    def _reset_task_state(self) -> None:
        self._current_graph: nx.DiGraph | None = None
        self._app_name: str = ""
        self._app_opened = False
        self._memory = Memory()
        self._step_count = 0

    def _compute_runtime_image_embedding_with_retry(
        self,
        screenshot_b64: str,
        app_name: str,
        node_id: str,
    ) -> list[float]:
        """Use native Gemini image embeddings for runtime node retrieval, with retry."""
        if not self.image_embedding_api_key:
            raise RuntimeError(
                "Native Gemini image embedding requires image_embedding.api_key "
                "or GEMINI_API_KEY/GOOGLE_API_KEY."
            )
        return compute_embedding_with_retry(
            self.image_embedding_api_key,
            screenshot_b64,
            model=self.image_embedding_model,
            base_url=self.image_embedding_base_url,
            app_name=app_name,
            node_id=node_id,
        )

    def _compute_missing_image_embeddings(
        self,
        G: nx.DiGraph,
        app_name: str,
        graph_file: Path,
        embeddings: dict[str, list[float]],
    ) -> None:
        """Compute an image embedding for every node with a screenshot but no cached vector.

        ``embeddings`` is the sidecar as already loaded and pruned to this graph's
        nodes by the caller, which has already copied every entry it holds onto
        ``G``; ``compute_missing_image_embeddings`` adds each newly computed
        vector to it in place and persists it (once, not per node -- see its own
        docstring), and only those newly computed vectors are copied back onto
        ``G`` here -- the caller's entries are on ``G`` already.

        A missing ``image_embedding.api_key`` is checked once here, before the
        retry loop starts, rather than inside it: raising per node would log a
        full traceback for every candidate instead of one clear error for the graph.
        """
        candidates = [
            (node_id, data["reference_screenshot"])
            for node_id, data in G.nodes(data=True)
            if data.get("reference_screenshot") and not data.get("image_embedding")
        ]
        if not candidates:
            return
        if not self.image_embedding_api_key:
            logger.error(
                "Native Gemini image embedding requires image_embedding.api_key "
                "or GEMINI_API_KEY/GOOGLE_API_KEY."
            )
            return
        compute_missing_image_embeddings(
            graph_file,
            embeddings,
            candidates,
            api_key=self.image_embedding_api_key,
            model=self.image_embedding_model,
            base_url=self.image_embedding_base_url,
            app_name=app_name,
        )
        for node_id, _screenshot in candidates:
            if node_id in embeddings:
                G.nodes[node_id]["image_embedding"] = embeddings[node_id]

    def _load_all_graphs(self) -> None:
        if not self.graph_dir.exists():
            logger.warning("Graph dir %s does not exist", self.graph_dir)
            return
        logger.info("[GRAPH] Loading runtime graphs from %s", self.graph_dir)
        for app_name, graph_file in iter_graph_files(self.graph_dir):
            try:
                logger.info("[GRAPH] %s: selected %s", app_name, graph_file.name)
                G = _load_graph_from_json(graph_file)
                # Pruned to this graph's current nodes: compute_missing_image_embeddings
                # writes this dict back verbatim, and an entry for a node that vanished
                # from the graph since the sidecar was written must not survive the
                # rewrite.
                embeddings = {
                    node_id: emb
                    for node_id, emb in load_image_embeddings(graph_file).items()
                    if node_id in G
                }
                for node_id, emb in embeddings.items():
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
                self._compute_missing_image_embeddings(G, app_name, graph_file, embeddings)
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
        if not package:
            return None
        app_name = self._package_to_app.get(package)
        if not app_name:
            app_name = self._package_to_app.get(package_from_activity(package))
        if app_name and app_name in self._graphs:
            self._app_name = app_name
            return self._graphs[app_name]
        for pkg_prefix, name in self._package_to_app.items():
            if package.startswith(pkg_prefix) or pkg_prefix.startswith(package):
                self._app_name = name
                return self._graphs.get(name)
        return None

    def _ask_with_screenshot[T](
        self,
        prompt: str,
        screenshot: str | None,
        parse: Callable[[str], T | None],
        tag: str,
    ) -> T | None:
        """Ask the model one question, retrying at the caller's parse boundary.

        ``parse`` must return ``None`` to request a retry; any other value
        (including a falsy one such as ``{}`` or ``""``) is accepted as success.
        Pass ``screenshot`` to ground the question in a screenshot; ``None`` sends
        the prompt as plain text. A retry resends the identical prompt at
        temperature 0.0, so a reply the parser could not read would otherwise come
        back the same way; every attempt after the first appends a strict-format
        reminder (``V2_PARSE_RETRY_HINT``) to give the retry an actual chance.
        """
        result: T | None = None
        raw = ""
        for attempt in range(V2_PARSE_RETRIES + 1):
            text = prompt if attempt == 0 else f"{prompt}\n\n{V2_PARSE_RETRY_HINT}"
            content: str | list[Any] = text
            if screenshot is not None:
                content = [{"type": "text", "text": text}, build_image_message(screenshot)]

            raw = _chat_completion_content(
                self.model_client,
                model=self.model_name,
                messages=[{"role": "user", "content": content}],
                max_tokens=V2_CHAT_MAX_TOKENS,
                temperature=0.0,
            )
            result = parse(raw)
            if result is not None:
                return result
            if attempt < V2_PARSE_RETRIES:
                logger.warning("%s parse failed on attempt %d; retrying", tag, attempt + 1)
        logger.warning(
            "%s all parse attempts exhausted. raw_len=%d raw_prefix=%r", tag, len(raw), raw[:160]
        )
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

        current_pkg = package_from_activity(activity)
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
            if package_from_activity(data.get("activity", "")) == current_pkg:
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
        page_desc, _detail_snapshot, _elements = call_with_retry(
            "[API] page describe and state",
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
        # Stored already narrowed; re-narrowing it per step doubled this loop's cost.
        same_pkg_image_candidates = (
            (str(node_id), data.get("image_embedding"))
            for node_id, data in G.nodes(data=True)
            if package_from_activity(data.get("activity", "")) == current_pkg
        )
        scored = score_by_cosine(
            query_image_emb,
            same_pkg_image_candidates,
            scope="[IDENTIFY] image-embedding",
            remedy="; the sidecar was written by a different embedding model, "
            "rerun app-graph-embed --recompute",
        )
        candidates: list[tuple[str, float, str]] = [
            (node_id, sim, G.nodes[node_id].get("page_description", "")) for node_id, sim in scored
        ]
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
                    if package_from_activity(data.get("activity", "")) == current_pkg
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

        answer = self._ask_with_screenshot(
            prompt,
            screenshot,
            lambda raw: _parse_model_choice(raw, letters[: len(top_k)]),
            "[IDENTIFY]",
        )
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

        # _parse_record_output never returns None, so this never retries: one completion.
        result = (
            self._ask_with_screenshot(prompt, screenshot, _parse_record_output, "[RECORD]")
            or "nothing"
        )
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

    def _build_self_loop_candidates(self, G: nx.DiGraph, node_id: str) -> list[tuple[_Option, str]]:
        """Return (option, menu line) pairs for the self-loop, lettered later by the cap."""
        if not G.has_edge(node_id, node_id):
            return []
        edge_data = G[node_id][node_id]
        templates = edge_data.get("instruction_templates", [])
        observations = edge_data.get("target_observations", [])
        candidates: list[tuple[_Option, str]] = []

        if templates:
            for tmpl in templates:
                tmpl_text, obs_text = self._unpack_template(tmpl)
                effect = self._edge_effect_hint(edge_data)
                option: _Option = {
                    "letter": "",
                    "type": "self_loop",
                    "node": node_id,
                    "instruction": tmpl_text,
                    "effect": effect,
                }
                hint = f'Stay here — "{tmpl_text}"'
                if obs_text:
                    hint += f" → {obs_text}"
                if effect:
                    hint += f" | changes: {effect}"
                candidates.append((option, hint))
        else:
            for i, raw_instr in enumerate(edge_data.get("instructions", [])):
                instr = as_str(raw_instr, "")
                effect = self._edge_effect_hint(edge_data, i)
                option: _Option = {
                    "letter": "",
                    "type": "self_loop",
                    "node": node_id,
                    "instruction": instr,
                    "effect": effect,
                }
                hint = f'Stay here — "{instr}"'
                if i < len(observations) and observations[i]:
                    hint += f" → {observations[i]}"
                if effect:
                    hint += f" | changes: {effect}"
                candidates.append((option, hint))

        return candidates

    def _build_neighbor_candidates(
        self, G: nx.DiGraph, node_id: str
    ) -> list[tuple[int, _Option, str]]:
        """Return (visit_count, option, menu line) per neighbour, lettered later by the cap."""
        candidates: list[tuple[int, _Option, str]] = []
        for _, raw_neighbor, edge_data in G.out_edges(node_id, data=True):
            neighbor = str(raw_neighbor)
            if neighbor == node_id:
                continue
            neighbor_desc = G.nodes[neighbor].get("page_description", neighbor)
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

            effect = self._edge_effect_hint(edge_data, 0)
            option: _Option = {
                "letter": "",
                "type": "neighbor",
                "node": neighbor,
                "instruction": instr,
                "description": neighbor_desc,
                "effect": effect,
            }
            edge_hint = f' — "{instr}"' if instr else ""
            if obs:
                edge_hint += f" → {obs}"
            if effect:
                edge_hint += f" | changes: {effect}"
            line = f'Go to "{neighbor_desc}"{edge_hint}'
            visit_count = as_int(edge_data.get("visit_count")) or 0
            candidates.append((visit_count, option, line))
        return candidates

    def _build_options(self, G: nx.DiGraph, node_id: str) -> tuple[str, list[_Option]]:
        """Build the option list for the DECIDE prompt.

        A-Z gives 26 slots: DONE and FREE are reserved, and the other 24 are split
        between self-loop instructions and neighbours. Neighbours are the only way
        to navigate the graph (GraphManager adds one self-loop instruction per
        distinct self-loop action, e.g. each typed query on a search screen, so
        self-loop instructions alone can otherwise fill every slot), so when both
        would overflow the 24 slots, neighbours keep at least half of them
        (ranked by visit_count, the only usefulness signal) and self-loop
        instructions take the rest.
        """
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        max_middle_options = len(letters) - 2  # reserve DONE and FREE

        self_loop_candidates = self._build_self_loop_candidates(G, node_id)
        neighbor_candidates = self._build_neighbor_candidates(G, node_id)

        kept_self_loop = self_loop_candidates
        kept_neighbors = [(opt, line) for _, opt, line in neighbor_candidates]
        total_middle = len(self_loop_candidates) + len(neighbor_candidates)

        if total_middle > max_middle_options:
            neighbor_reserve = min(len(neighbor_candidates), max_middle_options // 2)
            self_loop_budget = min(len(self_loop_candidates), max_middle_options - neighbor_reserve)
            neighbor_budget = max_middle_options - self_loop_budget
            kept_self_loop = self_loop_candidates[:self_loop_budget]
            ranked = sorted(
                range(len(neighbor_candidates)),
                key=lambda i: (-neighbor_candidates[i][0], i),
            )
            keep_indices = sorted(ranked[:neighbor_budget])
            kept_neighbors = [
                (neighbor_candidates[i][1], neighbor_candidates[i][2]) for i in keep_indices
            ]
            logger.warning(
                "[DECIDE] %s: %d candidate actions exceed the %d-letter menu; "
                "dropped %d least-useful self-loop instruction(s)/neighbour(s), "
                "keeping neighbours to at least half the menu.",
                node_id,
                total_middle,
                max_middle_options,
                total_middle - len(kept_self_loop) - len(kept_neighbors),
            )

        options: list[_Option] = []
        lines: list[str] = []
        idx = 0

        letter = letters[idx]
        options.append({"letter": letter, "type": "done"})
        lines.append(f"{letter}) DONE — the task is fully complete, answer is in memory")
        idx += 1

        for option, line in (*kept_self_loop, *kept_neighbors):
            letter = letters[idx]
            option["letter"] = letter
            options.append(option)
            lines.append(f"{letter}) {line}")
            idx += 1

        letter = letters[idx]
        options.append({"letter": letter, "type": "free"})
        lines.append(f"{letter}) FREE — do something not listed above (describe it)")
        idx += 1

        return "\n".join(lines), options

    def _decide(self, G: nx.DiGraph, task: str, node_id: str, screenshot: str) -> _Decision:
        """Ask the model to pick the next action.

        Returns the chosen option dict with an added "instruction" key
        (the refined instruction from the model).
        """
        current_desc = G.nodes[node_id].get("page_description", "")
        today = datetime.now()

        # Show state keys (from schema) so the model knows what parameters this screen has.
        # Values are omitted — the model can read them from the screenshot directly.
        state_schema = G.nodes[node_id].get("state_schema", {})
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

        result = self._ask_with_screenshot(prompt, screenshot, _parse_decide_output, "[DECIDE]")
        if result is None:
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
                if opt["type"] == "free":
                    # An empty instruction is what makes _step re-plan a FREE pick; the
                    # task-text fallback would hide that. Graph-built options may be empty
                    # by construction, so they keep it.
                    resolved_instruction = instruction
                else:
                    resolved_instruction = instruction or opt.get("instruction") or task
                chosen: _Decision = {
                    "type": opt["type"],
                    "letter": opt["letter"],
                    "instruction": resolved_instruction,
                }
                if "node" in opt:
                    chosen["node"] = opt["node"]
                if "description" in opt:
                    chosen["description"] = opt["description"]
                if "effect" in opt:
                    chosen["effect"] = opt["effect"]
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
        instruction = self._ask_with_screenshot(
            prompt,
            screenshot,
            lambda raw: _parse_one_step_instruction(raw) or None,
            "[FREE]",
        )

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
        keyboard_hint = device.soft_keyboard_hint()

        aitk_action, history_entry = call_with_retry(
            "[API] action agent",
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

    def _generate_answer(self, task: str) -> str:
        prompt = DONE_PROMPT.format(
            task=task,
            memory=self._memory.format(),
        )
        answer = self._ask_with_screenshot(prompt, None, lambda raw: raw.strip() or None, "[DONE]")
        return answer or ""

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
        logger.info('[IDENTIFY] Node: %s — "%s"', node_id, page_desc)

        self._record_info(task, screenshot)
        logger.info("[RECORD] Memory: %s", self._memory.format())

        decision: _Decision
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

        decision_type = decision["type"]
        instruction = decision["instruction"]
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
        if not history.get("actions", []):
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
        if not isinstance(aitk_action, dict) or "action" not in aitk_action:
            return {"action": "end", "answer": ""}
        return aitk_action


def register(kargs: dict[str, Any]) -> UIKobeV2Translator:
    return UIKobeV2Translator(**kargs)
