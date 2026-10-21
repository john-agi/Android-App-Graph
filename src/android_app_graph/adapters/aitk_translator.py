"""AITK v2 translator; setup and runtime flow: README.md#use-android-app-graph-with-aitk."""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, NotRequired, TypedDict, override

import networkx as nx
from aitk.translators.base import BaseTranslator
from openrouter import OpenRouter

from android_app_graph import device
from android_app_graph.android_packages import saved_node_package
from android_app_graph.embedding_cache import (
    compute_embedding_with_retry,
    compute_missing_image_embeddings,
    iter_screenshot_candidates,
    load_image_embeddings,
    resolve_image_embedding_settings,
)
from android_app_graph.graph_files import (
    iter_graph_files,
    reference_screenshot_path,
    require_known_edge_endpoints,
)
from android_app_graph.payloads import as_int, as_list, as_str, as_str_dict
from android_app_graph.utils import (
    ModelSettings,
    open_client,
    resolve_api_key,
    resolve_model_settings,
)
from android_app_graph.utils.vlm_utils import (
    TruncatedReplyError,
    build_image_message,
    message_text,
    predict_next_action,
    score_by_cosine,
    send_chat,
    strip_json_fences,
)

if TYPE_CHECKING:
    from openrouter.components.chatmessages import ChatMessagesTypedDict
    from openrouter.components.chatusermessage import ChatUserMessageContentTypedDict

logger = logging.getLogger("AITK - Android-App-Graph")

_SEP = "-" * 60
_SEP_THICK = "=" * 60
V2_PARSE_RETRIES = 1
# Format-agnostic: it is appropriate whether the caller's parse expects a letter,
# JSON, or free text, so one reminder covers every _ask_with_screenshot caller.
V2_PARSE_RETRY_HINT = (
    "Your previous reply could not be parsed. Reply in exactly the format "
    "requested above and nothing else."
)

RUNTIME_ROLES = ("action", "image_embedding")

OptionType = Literal["done", "self_loop", "neighbor", "free"]


class Option(TypedDict):
    """One DECIDE menu entry; the keys present depend on ``type``."""

    letter: str
    type: OptionType
    instruction: NotRequired[str]


class Decision(TypedDict):
    """The outcome of ``decide``, read by ``_step`` for ``type`` and ``instruction``;
    a "free" fallback additionally carries ``reason``."""

    type: OptionType
    instruction: str
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

Think easily and briefly, under 10 sentences if needed. Reply with ONLY a JSON object:
{{"information": "<short factual statement of what you found>"}}
or, when no new information is relevant:
{{"information": null}}"""


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


def load_graph_from_json(path: Path) -> nx.DiGraph:
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    # Run before anything else below: a graph with one bad node or edge must
    # reject before the node loop's per-node screenshot path lookup, and
    # before any edge is added.
    data = require_known_edge_endpoints(raw, path)

    edge_specs: list[tuple[str, str, dict[str, Any]]] = []
    for edge_data in data.get("edges", []):
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

    package_name = as_str(data.get("package_name"), "")
    graph = nx.DiGraph(package_name=package_name)
    for node_data in data.get("nodes", []):
        node_id = node_data["id"]
        # A path, not the screenshot itself: reading and base64-encoding every
        # node's screenshot at load time kept every one of them resident for the
        # whole AITK session even though only _compute_missing_image_embeddings
        # (a candidate with no cached vector) ever reads the bytes. The path is
        # looked up once here (a stat, not a read) and encoded lazily there,
        # the same as offline precomputation already does.
        screenshot_path = reference_screenshot_path(path, node_id)
        # `.get(key, default)` keeps a present-but-null value, so every field a reader
        # iterates or indexes is narrowed here once rather than at each read site.
        graph.add_node(
            node_id,
            package=saved_node_package(node_data, package_name),
            activity=as_str(node_data.get("activity"), ""),
            page_description=as_str(node_data.get("page_description"), ""),
            state_schema=as_str_dict(node_data.get("state_schema")),
            last_detail_snapshot=as_str_dict(node_data.get("last_detail_snapshot")),
            reference_screenshot_path=screenshot_path,
            visit_count=node_data.get("visit_count", 0),
        )

    for source, target, edge_attrs in edge_specs:
        graph.add_edge(source, target, **edge_attrs)

    return graph


def _graph_package(graph: nx.DiGraph) -> str:
    return as_str(graph.graph.get("package_name"), "")


type _ParsedGraph = tuple[str, Path, nx.DiGraph]


def _parse_graphs_by_package(graph_dir: Path) -> dict[str, _ParsedGraph]:
    """Each step selects the graph whose ``package_name`` is the foreground package,
    so a file without one could never be selected and files that share one could
    not be told apart. Both are left out here, before any of their screenshots is
    embedded.
    """
    parsed: dict[str, list[_ParsedGraph]] = {}
    for app_name, graph_file in iter_graph_files(graph_dir):
        logger.info("[GRAPH] %s: selected %s", app_name, graph_file.name)
        try:
            graph = load_graph_from_json(graph_file)
        except Exception:
            logger.exception("Failed to load graph %s", graph_file)
            continue
        package = _graph_package(graph)
        if package:
            parsed.setdefault(package, []).append((app_name, graph_file, graph))
        else:
            logger.error("[GRAPH] Not loading %s: it has no package_name", graph_file)
    unique: dict[str, _ParsedGraph] = {}
    for package, graphs in parsed.items():
        if len(graphs) == 1:
            unique[package] = graphs[0]
        else:
            logger.error(
                "[GRAPH] Not loading %s: they share package_name %s",
                ", ".join(str(graph_file) for _app, graph_file, _graph in graphs),
                package,
            )
    return unique


def after_last_think_tag(text: str) -> str | None:
    """Return the text after the last ``</think>`` tag, stripped, or ``None`` if absent.

    Found with a case-insensitive regex on the original ``text``, never on
    ``text.lower()``: case-folding can change a string's length (e.g. U+0130
    "İ" lowers to two code points), which would shift the tag's offset out of
    step with the original text it is used to slice.
    """
    last_end = -1
    for match in re.finditer(r"</think>", text, re.IGNORECASE):
        last_end = match.end()
    return text[last_end:].strip() if last_end != -1 else None


# An article is followed by a noun phrase, never by a finite verb, so "A is the
# login screen" names a letter while "A login screen is visible" does not. A
# closed list of the verbs a model puts after an answer letter, not a parser:
# a verb missing from it costs one strict-format retry, never a wrong pick.
_VERBS_THAT_CANNOT_FOLLOW_AN_ARTICLE = frozenset(
    {
        "is",
        "was",
        "matches",
        "shows",
        "looks",
        "seems",
        "appears",
        "fits",
        "has",
        "does",
        "would",
        "should",
        "corresponds",
        "represents",
        "displays",
    }
)


def _reads_as_article(letter: str, text: str, start: int, end: int) -> bool:
    """True when ``letter``, matched in ``text`` at ``[start, end)``, reads as
    the English article/pronoun "a"/"I" rather than a named answer letter.

    Both must hold: the letter is followed on the same line by a lowercase
    word, AND it is sentence-initial -- at the start of the text, or preceded
    (ignoring spaces/tabs) by a newline or one of ``. ! ? :`` (the colon
    covers "Answer: A login screen ..." and "Candidates: A looks ..."). The
    article reading only makes sense there; mid-sentence, "the best match is
    A because it shows a home screen" names a letter even though "A" precedes
    a lowercase word. That article-looking case is left to the strict-format
    retry (V2_PARSE_RETRY_HINT) rather than guessed. A line break ends the
    sentence, so a letter alone on its line ("Answer: A", then an explanation)
    is the answer.
    """
    if letter not in ("A", "I"):
        return False
    following = re.match(r"[ \t]+([a-z]+)", text[end:])
    if following is None or following.group(1) in _VERBS_THAT_CANNOT_FOLLOW_AN_ARTICLE:
        return False
    prefix = text[:start].rstrip(" \t")
    return not prefix or prefix[-1] in ".!?:\n"


def last_answer_signal(text: str, valid_letters: str) -> str | None:
    """Return free text's one answer signal -- a named letter or "NONE" -- or
    ``None`` when the text carries no signal, or carries both kinds.

    The parser decides only when the reply carries one kind of answer signal:
    when only letters are present, the last one wins, because a model states
    its conclusion last ("None of the others fit, so B" answers B); when only
    "NONE" is present, it answers NONE. When a reply carries both a letter and
    a "NONE" -- "B is the best match. None of the others match." as much as
    "None of the others fit, so B" -- their order carries no reliable meaning
    and no text rule can tell a rejection-then-letter from a
    letter-then-rejection, so the reply is ambiguous: this returns ``None``,
    which sends the caller back to the strict-format retry (asking for
    exactly one letter or NONE) instead of guessing.
    """
    has_letter = False
    last_letter = ""
    for letter_match in re.finditer(r"\b([A-Z])\b", text):
        letter = as_str(letter_match.group(1), "")
        if _reads_as_article(letter, text, letter_match.start(), letter_match.end()):
            continue
        if letter in valid_letters:
            has_letter, last_letter = True, letter

    has_none = re.search(r"\bNONE\b", text, re.IGNORECASE) is not None

    if has_letter and has_none:
        return None
    if has_none:
        return "NONE"
    return last_letter or None


def parse_model_choice(raw: str, valid_letters: str) -> str | None:
    text = (raw or "").strip()
    answer = text.upper()
    if answer == "NONE":
        return "NONE"
    if len(answer) == 1 and answer in valid_letters:
        return answer

    post_think = after_last_think_tag(text)
    if post_think is not None:
        parsed = parse_model_choice(post_think, valid_letters)
        if parsed:
            return parsed

    # The label is trusted as written, with no article check and no scan of what
    # follows it: a justification ("Answer: A because none of the others
    # match") and a rejection ("Answer: A login screen is shown, none of them
    # match") are textually indistinguishable after the label, so no rule
    # written against the remainder can separate them without breaking one of
    # the two. Only the keyword ("answer"/"decision"/"final answer"/"final
    # decision") is case-insensitive; the captured letter itself is not, so a
    # lowercase article right after the colon ("Answer: a login screen...")
    # never counts as a named letter and falls through to the free-text scan
    # below instead. The LAST label wins, not the first: a model that revises
    # itself states the revision last ("Answer: A\n...\nFinal answer: C"
    # answers C).
    final_matches = list(
        re.finditer(
            r"(?i:\b(?:FINAL\s+(?:ANSWER|DECISION)|ANSWER|DECISION)\s*:\s*)((?i:NONE)|[A-Z])\b",
            text,
        )
    )
    final_match = final_matches[-1] if final_matches else None
    if final_match:
        choice = as_str(final_match.group(1), "").upper()
        if choice == "NONE":
            return "NONE"
        if choice in valid_letters:
            return choice

    # Free text beyond an explicit "Answer:"/</think> form.
    return last_answer_signal(text, valid_letters)


def parse_one_step_instruction(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return ""

    post_think = after_last_think_tag(text)
    if post_think is not None:
        text = post_think

    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE).strip()
    final_matches = list(
        re.finditer(
            r"(?im)^\s*(?:final\s+answer|answer|instruction|next\s+action)\s*:\s*(.+)$",
            text,
        )
    )
    if final_matches:
        text = text[final_matches[-1].start(1) :].strip()

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


def extract_json_object(text: str) -> dict[str, Any] | None:
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


def parse_json_object(text: str) -> dict[str, Any] | None:
    """Parse ``text`` as a JSON object, else scan it for the first embedded one."""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return extract_json_object(text)
    return parsed if isinstance(parsed, dict) else None


def parse_json_reply(raw: str) -> dict[str, Any] | None:
    """Parse a RECORD or DECIDE reply's JSON object, preferring anything after ``</think>``.

    ``</think>`` is located on the raw text before fences are stripped, never
    after: stripping fences first can eat a fenced JSON draft inside the think
    block along with the tag itself (``strip_json_fences`` matches the first
    fence pair it finds), which would hide the fenced final answer that
    follows the tag and return the draft instead.
    """
    text = (raw or "").strip()
    post_think = after_last_think_tag(text)
    if post_think is not None:
        parsed = parse_json_object(strip_json_fences(post_think))
        if parsed is not None:
            return parsed
    return parse_json_object(strip_json_fences(text))


class RecordReplyError(ValueError):
    """A RECORD reply the runtime cannot use; the run stops rather than guess at a fact."""


def _unreadable_record_reply(problem: str, raw: str) -> RecordReplyError:
    return RecordReplyError(f"[RECORD] {problem} (reply begins {raw[:160]!r})")


def parse_record_output(raw: str) -> str:
    """Return the fact a RECORD reply states, or ``""`` when it states none.

    Raises instead of returning ``None``: ``_ask_with_screenshot`` asks again on
    ``None``, and RECORD is asked only once.
    """
    reply = parse_json_reply(raw)
    if reply is None:
        msg = "the reply has no JSON object"
        raise _unreadable_record_reply(msg, raw)
    if "information" not in reply:
        msg = 'the JSON object has no "information" key'
        raise _unreadable_record_reply(msg, raw)
    information = reply["information"]
    if information is None:
        return ""
    if not isinstance(information, str):
        msg = '"information" is neither a string nor null'
        raise _unreadable_record_reply(msg, raw)
    if not information.strip():
        msg = '"information" is empty'
        raise _unreadable_record_reply(msg, raw)
    return information


def chat_completion_content(
    client: OpenRouter,
    *,
    settings: ModelSettings,
    messages: list[ChatMessagesTypedDict],
) -> str:
    """Run a typed completion, account for usage and return assistant text."""
    return message_text(
        send_chat(client, call_type="v2_chat", settings=settings, messages=messages)
    )


def unpack_template(tmpl: dict[str, Any] | str) -> tuple[str, str]:
    """Extract (instruction_text, observation_text) from a template entry."""
    if isinstance(tmpl, dict):
        return (
            as_str(tmpl.get("template", ""), ""),
            as_str(tmpl.get("observation_template", ""), ""),
        )
    return str(tmpl), ""


def format_schema_delta(delta: dict[str, Any] | None) -> str:
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


def edge_effect_hint(edge_data: dict[str, Any], index: int | None = None) -> str:
    """Return a concise effect hint from schema deltas, if available."""
    deltas = edge_data.get("schema_deltas", [])
    if not deltas:
        return ""

    if index is not None:
        return format_schema_delta(deltas[index]) if index < len(deltas) else ""

    merged: dict[str, Any] = {}
    for delta in deltas:
        if isinstance(delta, dict):
            for key, change in delta.items():
                merged.setdefault(key, change)
    return format_schema_delta(merged)


def _build_self_loop_candidates(graph: nx.DiGraph, node_id: str) -> list[tuple[Option, str]]:
    """Return (option, menu line) pairs for the self-loop, lettered later by the cap."""
    if not graph.has_edge(node_id, node_id):
        return []
    edge_data = graph[node_id][node_id]
    templates = edge_data.get("instruction_templates", [])
    observations = edge_data.get("target_observations", [])
    candidates: list[tuple[Option, str]] = []

    if templates:
        for tmpl in templates:
            tmpl_text, obs_text = unpack_template(tmpl)
            effect = edge_effect_hint(edge_data)
            option: Option = {
                "letter": "",
                "type": "self_loop",
                "instruction": tmpl_text,
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
            if not instr:
                continue
            effect = edge_effect_hint(edge_data, i)
            option: Option = {
                "letter": "",
                "type": "self_loop",
                "instruction": instr,
            }
            hint = f'Stay here — "{instr}"'
            if i < len(observations) and observations[i]:
                hint += f" → {observations[i]}"
            if effect:
                hint += f" | changes: {effect}"
            candidates.append((option, hint))

    return candidates


def _build_neighbor_candidates(graph: nx.DiGraph, node_id: str) -> list[tuple[int, Option, str]]:
    """Return (visit_count, option, menu line) per neighbour, lettered later by the cap."""
    candidates: list[tuple[int, Option, str]] = []
    for _, raw_neighbor, edge_data in graph.out_edges(node_id, data=True):
        neighbor = str(raw_neighbor)
        if neighbor == node_id:
            continue
        neighbor_desc = graph.nodes[neighbor].get("page_description", neighbor)
        templates = edge_data.get("instruction_templates", [])
        observations = edge_data.get("target_observations", [])

        instr = ""
        obs = ""
        if templates:
            instr, obs = unpack_template(templates[0])
        else:
            instrs = edge_data.get("instructions", [])
            if instrs:
                instr = as_str(instrs[0], "")
            if observations:
                obs = as_str(observations[0], "")

        effect = edge_effect_hint(edge_data, 0)
        option: Option = {
            "letter": "",
            "type": "neighbor",
            "instruction": instr,
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


def build_options(graph: nx.DiGraph, node_id: str) -> tuple[str, list[Option]]:
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

    self_loop_candidates = _build_self_loop_candidates(graph, node_id)
    neighbor_candidates = _build_neighbor_candidates(graph, node_id)

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

    options: list[Option] = []
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


def make_response(message: str, aitk_action: dict[str, Any]) -> str:
    return json.dumps(
        {
            "message": f"[Android-App-Graph] {message}",
            "aitk_action": aitk_action,
        }
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
        if info:
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
        logging.getLogger("openrouter").setLevel(logging.WARNING)

        self.graph_dir = Path(graph_dir)
        # Accepted because AITK forwards every translator_args key and its controller.yaml
        # sets max_pixels; this translator sends screenshots unresized.
        self.max_pixels = max_pixels
        vlm_config = vlm_config or {}
        settings = resolve_model_settings(vlm_config, roles=RUNTIME_ROLES)
        self.action = settings["action"]
        self.image_embedding = resolve_image_embedding_settings(vlm_config)
        # AITK has no translator teardown, so the client lives as long as the
        # translator; the SDK closes it when the translator is collected or at exit.
        self.client = open_client(resolve_api_key())
        self._graphs: dict[str, nx.DiGraph] = {}
        self._app_by_package: dict[str, str] = {}
        self._load_all_graphs()

        # A device property: only to_device updates it, and a task reset must not.
        self._screen_w = 1080
        self._screen_h = 1920
        self.reset_task_state()

    @property
    def graphs(self) -> Mapping[str, nx.DiGraph]:
        """The loaded graphs by app name, as a read-only view."""
        return MappingProxyType(self._graphs)

    @property
    def memory(self) -> Memory:
        """The current task's memory; ``reset_task_state`` starts a new one."""
        return self._memory

    def reset_task_state(self) -> None:
        self._app_opened = False
        self._memory = Memory()
        self._step_count = 0

    def _compute_runtime_image_embedding_with_retry(
        self,
        screenshot_b64: str,
        app_name: str,
        node_id: str,
    ) -> list[float]:
        """Use OpenRouter image embeddings for runtime node retrieval, with retry."""
        return compute_embedding_with_retry(
            self.client,
            screenshot_b64,
            settings=self.image_embedding,
            app_name=app_name,
            node_id=node_id,
        )

    def _compute_missing_image_embeddings(
        self,
        graph: nx.DiGraph,
        app_name: str,
        graph_file: Path,
        embeddings: dict[str, list[float]],
    ) -> None:
        """Compute an image embedding for every node with a screenshot but no cached vector.

        ``embeddings`` is the sidecar as already loaded and pruned to this graph's
        nodes by the caller, which has already copied every entry it holds onto
        ``graph``; ``compute_missing_image_embeddings`` adds each newly computed
        vector to it in place and persists it (once, not per node -- see its own
        docstring), and only those newly computed vectors are copied back onto
        ``graph`` here -- the caller's entries are on ``graph`` already.
        """
        pending = [
            (node_id, data["reference_screenshot_path"])
            for node_id, data in graph.nodes(data=True)
            if data.get("reference_screenshot_path") and not data.get("image_embedding")
        ]
        if not pending:
            return
        # Encoded lazily, one screenshot at a time, the same as offline
        # precomputation: at most one base64 payload resident, never every
        # uncached node's at once.
        candidates = iter_screenshot_candidates(pending, app_name=app_name)
        compute_missing_image_embeddings(
            graph_file,
            embeddings,
            candidates,
            client=self.client,
            settings=self.image_embedding,
            app_name=app_name,
        )
        for node_id, _screenshot_path in pending:
            if node_id in embeddings:
                graph.nodes[node_id]["image_embedding"] = embeddings[node_id]

    def _load_all_graphs(self) -> None:
        if not self.graph_dir.exists():
            logger.warning("Graph dir %s does not exist", self.graph_dir)
            return
        logger.info("[GRAPH] Loading runtime graphs from %s", self.graph_dir)
        for package, (app_name, graph_file, graph) in _parse_graphs_by_package(
            self.graph_dir
        ).items():
            try:
                embeddings = load_image_embeddings(
                    graph_file, model=self.image_embedding.model, node_ids=set(graph.nodes)
                )
                for node_id, emb in embeddings.items():
                    graph.nodes[node_id]["image_embedding"] = emb
                ref_count = sum(
                    1 for _, data in graph.nodes(data=True) if data.get("reference_screenshot_path")
                )
                cached_count = sum(
                    1 for _, data in graph.nodes(data=True) if data.get("image_embedding")
                )
                logger.info(
                    "[GRAPH] %s: nodes=%d edges=%d reference_screenshots=%d cached_image_embeddings=%d",
                    app_name,
                    graph.number_of_nodes(),
                    graph.number_of_edges(),
                    ref_count,
                    cached_count,
                )
                self._compute_missing_image_embeddings(graph, app_name, graph_file, embeddings)
                self._graphs[app_name] = graph
                self._app_by_package[package] = app_name
                logger.info("[GRAPH] %s loaded: package=%s", app_name, package)
            except Exception:
                logger.exception("Failed to load graph %s", graph_file)

    def _resolve_app_from_task(self, task: str) -> str | None:
        task_lower = task.lower()
        for app_name in self._graphs:
            if app_name.lower() in task_lower:
                return app_name
        return None

    def _graph_for_package(self, package: str) -> tuple[str, nx.DiGraph] | None:
        """Return the app name and graph whose ``package_name`` is ``package``, if any."""
        app_name = self._app_by_package.get(package)
        if app_name is None:
            return None
        return app_name, self._graphs[app_name]

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
        the prompt as plain text. Every attempt after the first appends a strict-format reminder
        (``V2_PARSE_RETRY_HINT``) to help the model repair an unreadable reply. A reply cut
        off at its token limit returns ``None`` at once: asking again would be cut off too.
        """
        result: T | None = None
        raw = ""
        for attempt in range(V2_PARSE_RETRIES + 1):
            text = prompt if attempt == 0 else f"{prompt}\n\n{V2_PARSE_RETRY_HINT}"
            content: ChatUserMessageContentTypedDict = text
            if screenshot is not None:
                content = [{"type": "text", "text": text}, build_image_message(screenshot)]

            try:
                raw = chat_completion_content(
                    self.client,
                    settings=self.action,
                    messages=[{"role": "user", "content": content}],
                )
            except TruncatedReplyError:
                logger.warning("%s reply hit its token limit; not asking again", tag)
                return None
            result = parse(raw)
            if result is not None:
                return result
            if attempt < V2_PARSE_RETRIES:
                logger.warning("%s parse failed on attempt %d; retrying", tag, attempt + 1)
        logger.warning(
            "%s all parse attempts exhausted. raw_len=%d raw_prefix=%r", tag, len(raw), raw[:160]
        )
        return None

    def identify_node(self, activity: str, screenshot: str, package: str) -> str | None:
        """Match the current screen to a graph node, or return ``None`` when none matches.

        1. Embed the screenshot and score it against the cached screenshot embeddings
           of the nodes of the foreground ``package``.
        2. Ask the model to pick one of the top-K candidates, each shown with its
           description and similarity, or to reject them all.

        No page description is generated: the paper's runtime identifies nodes by
        screenshot, and upstream UI-KOBE's describe call here never fed a decision.
        """
        selected = self._graph_for_package(package)
        if selected is None:
            logger.warning("[IDENTIFY] No current graph is loaded")
            return None
        app_name, graph = selected

        logger.info(
            "[IDENTIFY] activity=%s package=%s graph_app=%s graph_package=%s",
            activity,
            package,
            app_name,
            _graph_package(graph),
        )

        same_pkg_node_count = sum(
            1 for _, data in graph.nodes(data=True) if data.get("package") == package
        )
        logger.info("[IDENTIFY] same-package graph nodes=%d", same_pkg_node_count)

        try:
            started = time.perf_counter()
            query_image_emb = self._compute_runtime_image_embedding_with_retry(
                screenshot,
                app_name,
                "(current screenshot)",
            )
            logger.info(
                "[IDENTIFY] current screenshot image embedding computed in %.1fs",
                time.perf_counter() - started,
            )
        except Exception:
            logger.exception("Runtime image embedding failed for current screenshot.")
            raise
        # Stored already narrowed; re-narrowing it per step doubled this loop's cost.
        same_pkg_image_candidates = (
            (str(node_id), data.get("image_embedding"))
            for node_id, data in graph.nodes(data=True)
            if data.get("package") == package
        )
        scored = score_by_cosine(
            query_image_emb,
            same_pkg_image_candidates,
            scope="[IDENTIFY] image-embedding",
            remedy="; the sidecar was hand-edited or is stale, rerun app-graph-embed --recompute",
        )
        candidates: list[tuple[str, float, str]] = [
            (node_id, sim, graph.nodes[node_id].get("page_description", ""))
            for node_id, sim in scored
        ]
        candidates.sort(key=lambda x: x[1], reverse=True)

        logger.info("[IDENTIFY] image candidates=%d", len(candidates))
        for nid, sim, desc in candidates[:5]:
            logger.info('[IDENTIFY] candidate sim=%.3f node=%s desc="%s"', sim, nid, desc)

        if not candidates:
            if same_pkg_node_count == 0:
                logger.warning(
                    "[IDENTIFY] no candidates: current package %s has no nodes in selected graph %s",
                    package,
                    app_name,
                )
            else:
                logger.warning(
                    "[IDENTIFY] no candidates: none of the %d same-package node(s) has a "
                    "usable image embedding (missing, or cached at a stale dimension)",
                    same_pkg_node_count,
                )
            return None

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
            lambda raw: parse_model_choice(raw, letters[: len(top_k)]),
            "[IDENTIFY]",
        )
        logger.info("[IDENTIFY] model_pick=%s", answer)

        if answer == "NONE" or not answer:
            logger.warning(
                "[IDENTIFY] model rejected candidates; not using image similarity as final identity. best_sim=%.3f",
                candidates[0][1],
            )
            return None

        idx = letters.index(answer)
        chosen_node = top_k[idx][0]
        logger.info("[IDENTIFY] identified node=%s", chosen_node)
        return chosen_node

    def record_info(self, task: str, screenshot: str) -> None:
        """Store the task-relevant fact the current screen shows, if it shows one.

        Raises ``RecordReplyError`` for a reply it cannot read or one cut off at
        its token limit, after one request: a fact RECORD guessed at, or one it
        silently lost, would mislead every later decision and the final answer.
        """
        prompt = RECORD_PROMPT.format(
            task=task,
            today=datetime.now().astimezone().strftime("%Y-%m-%d (%A)"),
            memory=self._memory.format(),
        )

        # parse_record_output never returns None, so None here can only mean
        # that the reply hit its token limit.
        info = self._ask_with_screenshot(prompt, screenshot, parse_record_output, "[RECORD]")
        if info is None:
            msg = "[RECORD] the reply hit its token limit"
            raise RecordReplyError(msg)
        logger.info("[RECORD] parsed=%r", info)
        self._memory.add_info(info)

    def decide(self, graph: nx.DiGraph, task: str, node_id: str, screenshot: str) -> Decision:
        """Ask the model to pick the next action.

        Returns the chosen option's type together with its resolved
        instruction (the model's own instruction, falling back to the
        option's graph-built one, then to the task text).
        """
        current_desc = graph.nodes[node_id].get("page_description", "")
        today = datetime.now().astimezone()

        # Show state keys (from schema) so the model knows what parameters this screen has.
        # Values are omitted — the model can read them from the screenshot directly.
        state_schema = graph.nodes[node_id].get("state_schema", {})
        if state_schema:
            keys_str = ", ".join(state_schema.keys())
            state_context = f"State parameters: [{keys_str}]\n"
        else:
            state_context = ""

        options_text, options_list = build_options(graph, node_id)

        prompt = DECIDE_PROMPT.format(
            task=task,
            today=today.strftime("%Y-%m-%d (%A)"),
            memory=self._memory.format(),
            current_node=node_id,
            current_desc=current_desc,
            state_context=state_context,
            options=options_text,
        )

        result = self._ask_with_screenshot(prompt, screenshot, parse_json_reply, "[DECIDE]")
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
                return {
                    "type": opt["type"],
                    "instruction": resolved_instruction,
                }

        logger.warning("DECIDE: unrecognized choice '%s'", choice_letter)
        return {
            "type": "free",
            "instruction": instruction,
            "reason": f"DECIDE returned unrecognized choice {choice_letter!r}",
        }

    def plan_free_action(
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
            lambda raw: parse_one_step_instruction(raw) or None,
            "[FREE]",
        )

        if instruction:
            logger.info("[FREE] planned_instruction=%s", instruction)
            return instruction

        logger.warning("[FREE] planner failed; using conservative concrete fallback")
        return "Wait briefly for the current screen to finish loading."

    def call_action_agent(
        self,
        task: str,
        action_history: list[str],
        screenshot: str,
        overall_task: str = "",
    ) -> tuple[dict[str, Any], str, str]:
        keyboard_hint = device.soft_keyboard_hint()

        aitk_action, history_entry = predict_next_action(
            self.client,
            screenshot,
            task + keyboard_hint,
            screen_w=self._screen_w,
            screen_h=self._screen_h,
            action_history=action_history,
            settings=self.action,
            overall_task=overall_task,
        )
        observation = history_entry.split(" | ")[0] if " | " in history_entry else ""
        return aitk_action, observation, history_entry

    def generate_answer(self, task: str) -> str:
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
                logger.info(_SEP_THICK)
                logger.info("[OPEN] Launching %s", app_name)
                logger.info(_SEP_THICK)
                return make_response(
                    f"Opening app: {app_name}",
                    {"action": "open", "app": app_name},
                )
            logger.warning("[OPEN] Could not resolve app from task: %s", task)

        selected = self._graph_for_package(package)
        graph = None if selected is None else selected[1]

        self._step_count += 1
        logger.info(_SEP_THICK)
        logger.info("[LOOP] Step %d", self._step_count)
        logger.info(_SEP_THICK)

        node_id = self.identify_node(activity, screenshot, package)
        logger.info("[IDENTIFY] Node: %s", node_id)

        self.record_info(task, screenshot)
        logger.info("[RECORD] Memory: %s", self._memory.format())

        decision: Decision
        if node_id is not None and graph is not None:
            decision = self.decide(graph, task, node_id, screenshot)
        else:
            reason = (
                "no graph is loaded"
                if graph is None
                else "the current screen did not match any graph node"
            )
            logger.warning("[DECIDE] Falling back to free action: %s", reason)
            fallback_instruction = self.plan_free_action(
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
            instruction = self.plan_free_action(
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
            answer = self.generate_answer(task)
            logger.info(_SEP_THICK)
            logger.info("[DONE] Answer: %s", answer)
            logger.info("[DONE] Memory: %s", self._memory.format())
            logger.info(_SEP_THICK)
            return make_response(
                f"Task complete. Answer: {answer}",
                {"action": "end", "answer": answer},
            )

        logger.info(_SEP)
        logger.info('ACTION AGENT one-step instruction: "%s"', instruction)
        logger.info(_SEP)
        aitk_action, obs, history_entry = self.call_action_agent(
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

        return make_response(f"Action: {history_entry}", aitk_action)

    @override
    def to_agent(self, task: str, state: dict[str, Any], history: dict[str, Any]) -> str:
        if not history.get("actions", []):
            self.reset_task_state()
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
