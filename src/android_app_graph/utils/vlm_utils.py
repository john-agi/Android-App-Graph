"""VLM utilities for page description, detail extraction, and action prediction."""

from __future__ import annotations

import json
import logging
import math
import re
import threading
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import httpx
from openrouter import OpenRouter
from openrouter.components.chatcontenttext import ChatContentText
from openrouter.components.chatresult import ChatResult
from openrouter.components.chatusage import (
    ChatUsage,
    ChatUsageCompletionTokensDetails,
    ChatUsagePromptTokensDetails,
)
from openrouter.components.costdetails import CostDetails
from openrouter.operations.createembeddings import CreateEmbeddingsResponseBody
from openrouter.types import UNSET

from android_app_graph.payloads import as_float_list, as_int, as_object_dict, as_str, as_str_dict
from android_app_graph.retrying import UnusableOutputError, call_with_retry, send_within_deadline
from android_app_graph.screen_image import MEDIA_TYPE
from android_app_graph.utils import REQUEST_TIMEOUTS, ModelSettings

if TYPE_CHECKING:
    from openrouter.components.chatcontentimage import ChatContentImageTypedDict
    from openrouter.components.chatcontentitems import ChatContentItemsTypedDict
    from openrouter.components.chatmessages import ChatMessagesTypedDict
    from openrouter.components.chatrequest import ResponseFormatTypedDict
    from openrouter.operations.createembeddings import (
        CreateEmbeddingsResponse,
        CreateEmbeddingsUsage,
    )

logger = logging.getLogger(__name__)

_COORDINATE_AXES = 2


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Return the cosine similarity of ``a`` and ``b``, clamped to ``[-1.0, 1.0]``.

    Raises ``ValueError`` when the vectors have different lengths: scoring
    them anyway (as plain ``zip`` would, truncating to the shorter one) treats
    the overlapping prefix of two unrelated embedding spaces as comparable,
    e.g. a cached vector computed under a since-changed embedding model.
    A zero norm has no direction and scores 0.0. Subnormal rounding can push
    the ratio outside the clamp, and a non-finite ratio scores 0.0 so a NaN
    never wins.
    """
    if len(a) != len(b):
        msg = f"cosine_similarity: vectors have different lengths ({len(a)} vs {len(b)})"
        raise ValueError(msg)
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if not norm_a or not norm_b:
        return 0.0
    similarity = dot / (norm_a * norm_b)
    if not math.isfinite(similarity):
        return 0.0
    return max(-1.0, min(1.0, similarity))


def score_by_cosine[K](
    query: list[float],
    candidates: Iterable[tuple[K, list[float] | None]],
    *,
    scope: str,
    remedy: str = "",
) -> list[tuple[K, float]]:
    """Score each candidate against ``query`` by cosine similarity, input order kept.

    A missing or empty vector is skipped silently: that is a candidate no
    embedding was ever computed for, not a stale one. A non-empty vector whose
    length differs from ``query`` is skipped too, but its dimension is
    collected and logged in a single warning once scoring is done -- an
    embedding model changed after these vectors were cached, so they sit in a
    different space than the fresh query; scoring them anyway would rank
    candidates as garbage with no signal anything was wrong, and one line beats
    a warning per candidate.
    """
    scored: list[tuple[K, float]] = []
    stale_count = 0
    stale_dims: set[int] = set()
    for key, vector in candidates:
        if not vector:
            continue
        if len(vector) != len(query):
            stale_count += 1
            stale_dims.add(len(vector))
            continue
        scored.append((key, cosine_similarity(query, vector)))

    if stale_count:
        logger.warning(
            "%s: embedding cache is stale for %d vector(s): query dim=%d, cached dim(s)=%s%s",
            scope,
            stale_count,
            len(query),
            sorted(stale_dims),
            remedy,
        )

    return scored


_USAGE_COUNTERS = (
    "calls",
    "prompt_tokens",
    "cached_prompt_tokens",
    "completion_tokens",
    "reasoning_tokens",
    "total_tokens",
    "cost",
)


def _cached_prompt_tokens(usage: ChatUsage | CreateEmbeddingsUsage) -> int:
    details = usage.prompt_tokens_details if isinstance(usage, ChatUsage) else None
    if isinstance(details, ChatUsagePromptTokensDetails) and isinstance(details.cached_tokens, int):
        return details.cached_tokens
    return 0


def _reasoning_tokens(usage: ChatUsage | CreateEmbeddingsUsage | None) -> int | None:
    details = usage.completion_tokens_details if isinstance(usage, ChatUsage) else None
    if not isinstance(details, ChatUsageCompletionTokensDetails):
        return None
    return details.reasoning_tokens if isinstance(details.reasoning_tokens, int) else None


def _cost(usage: ChatUsage | CreateEmbeddingsUsage) -> float:
    cost = usage.cost if isinstance(usage.cost, float) else 0.0
    # On a request sent with the account's own provider key (BYOK), `cost` is only
    # OpenRouter's charge; the provider bills `upstream_inference_cost` to that key.
    # https://openrouter.ai/docs/guides/administration/usage-accounting
    details = usage.cost_details
    if usage.is_byok and isinstance(details, CostDetails):
        upstream = details.upstream_inference_cost
        cost += upstream if isinstance(upstream, float) else 0.0
    return cost


class TokenTracker:
    """Accumulates token usage and cost across all API calls, broken down by call type."""

    def __init__(self) -> None:
        self._usage: dict[str, dict[str, float]] = {}
        # call_concurrently's worker threads record at the same time; nothing reads the
        # totals until its pool has finished.
        self._lock = threading.Lock()

    def record(
        self,
        call_type: str,
        model: str,
        usage: ChatUsage | CreateEmbeddingsUsage | None,
    ) -> None:
        """Record the usage OpenRouter reported for one completed request.

        Args:
            call_type: e.g. "page_describe_and_state", "embedding", "instruction", "action"
            model: The model name used.
            usage: The response's usage; cached prompt tokens and reasoning tokens
                count only when the provider reports them, and cost is in USD.
        """
        if usage is None:
            return
        with self._lock:
            totals = self._usage.setdefault(
                f"{call_type}:{model}", dict.fromkeys(_USAGE_COUNTERS, 0)
            )
            totals["calls"] += 1
            totals["prompt_tokens"] += usage.prompt_tokens
            totals["cached_prompt_tokens"] += _cached_prompt_tokens(usage)
            totals["completion_tokens"] += (
                usage.completion_tokens if isinstance(usage, ChatUsage) else 0
            )
            totals["reasoning_tokens"] += _reasoning_tokens(usage) or 0
            totals["total_tokens"] += usage.total_tokens
            totals["cost"] += _cost(usage)

    def snapshot(self) -> dict[str, float]:
        """Return current totals as a flat dict."""
        total = dict.fromkeys(_USAGE_COUNTERS, 0.0)
        for totals in self._usage.values():
            for counter in _USAGE_COUNTERS:
                total[counter] += totals[counter]
        return total

    def snapshot_by_type(self) -> dict[str, int]:
        """Return total_tokens per call_type (aggregated across models).

        E.g. {"page_describe_and_state": 1200, "embedding": 20, ...}
        """
        by_type: dict[str, int] = {}
        for key, totals in self._usage.items():
            call_type = key.split(":")[0]
            by_type[call_type] = by_type.get(call_type, 0) + int(totals["total_tokens"])
        return by_type

    def summary(self) -> dict[str, dict[str, float]]:
        """Return accumulated usage by call_type:model."""
        return dict(self._usage)

    def print_summary(self) -> None:
        """Log a formatted summary table of all token usage and cost."""
        if not self._usage:
            logger.info("No API calls recorded.")
            return

        def row(label: str, totals: dict[str, float]) -> str:
            return (
                f"{label:<40} {totals['calls']:>6.0f} {totals['prompt_tokens']:>10.0f} "
                f"{totals['cached_prompt_tokens']:>8.0f} {totals['completion_tokens']:>10.0f} "
                f"{totals['reasoning_tokens']:>9.0f} {totals['total_tokens']:>10.0f} "
                f"{totals['cost']:>12.6f}"
            )

        width = 112
        lines = ["", "=" * width, "TOKEN USAGE SUMMARY", "=" * width]
        lines.append(
            f"{'Call Type':<40} {'Calls':>6} {'Prompt':>10} {'Cached':>8} {'Completion':>10} "
            f"{'Reasoning':>9} {'Total':>10} {'Cost (USD)':>12}"
        )
        lines.append("-" * width)
        lines.extend(row(key, totals) for key, totals in sorted(self._usage.items()))
        lines.append("-" * width)
        lines.append(row("TOTAL", self.snapshot()))
        lines.append("=" * width)

        logger.info("\n".join(lines))


token_tracker = TokenTracker()

MAX_REQUESTS_IN_FLIGHT = 16


def call_concurrently[T, R](
    label: str, call: Callable[[T], R], inputs: Sequence[T], apply: Callable[[T, R], None]
) -> None:
    """Run ``call`` on every input, ``MAX_REQUESTS_IN_FLIGHT`` at a time on worker threads,
    and ``apply`` each result on the calling thread, in input order.

    ``call`` must not touch what ``apply`` changes. When a call raises, the calls not yet
    started are cancelled, the results of the others are still applied, and then the first
    error in input order is raised. ``label`` names the calls in the log.
    """
    error: BaseException | None = None
    pool = ThreadPoolExecutor(max_workers=MAX_REQUESTS_IN_FLIGHT)
    try:
        futures = [pool.submit(call, item) for item in inputs]
        for item, future in zip(inputs, futures, strict=True):
            if future.cancelled():
                continue
            failure = future.exception()
            if failure is None:
                apply(item, future.result())
            elif error is None:
                error = failure
                cancelled = sum(pending.cancel() for pending in futures)
                logger.warning("%s failed; cancelled %d call(s) not yet started", label, cancelled)
            else:
                logger.warning("%s failed too: %s: %s", label, type(failure).__name__, failure)
    finally:
        # Cancelled here too, so an apply that raises or an interrupt sends no queued call.
        pool.shutdown(cancel_futures=True)
    if error is not None:
        raise error


def strip_json_fences(text: str) -> str:
    """Remove ```json ... ``` markdown fences if present."""
    text = text.strip()
    m = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if m:
        return as_str(m.group(1), text).strip()
    return text


NORMALIZE_EDGE_PROMPT = """\
You are helping build a reusable navigation graph for a mobile app. Your job is to \
normalize action instructions and their resulting observations into TEMPLATES by \
replacing specific values with named placeholders.

Given one or more concrete action instructions recorded on one edge of the graph (and \
optionally their target observations), produce a single normalized template that captures \
the common pattern, or answer that they are different actions.

Rules:
- Replace specific user-provided values (search queries, names, addresses, numbers, dates, \
  item selections) with descriptive {{placeholder}} names like {{query}}, {{location}}, \
  {{item_name}}, {{date}}, {{amount}}, etc.
- Keep the action verb and UI element references as-is (e.g. "Tap the search bar" stays unchanged).
- If only one instruction is given, still normalize it — replace any concrete value with a placeholder.
- If multiple instructions are given, find the common pattern and merge them.
- For target observations, normalize the same way — replace specific result content with placeholders.
- If the instruction has NO specific values to replace (e.g. "Tap the settings icon", \
  "Press back"), return it unchanged with is_template: false.
- A template is ONE instruction. When the instructions are different actions that no one \
  instruction with placeholders describes (e.g. a toggle, a scroll and a menu on one screen), \
  set different_actions to true and leave every other field empty. Never join several actions \
  into one template, by line breaks, numbering, "|", "OR" or "or", and never answer with a \
  template made of placeholders alone or of a verb and placeholders, such as \
  "{{action}} {{target_element}}" or "Tap {{target}}".

Return a JSON object:
{{
  "different_actions": true/false (true when the instructions are different actions),
  "instruction_template": "the normalized instruction template",
  "observation_template": "the normalized observation template (or empty string if none)",
  "is_template": true/false (true if any placeholders were inserted),
  "param_names": ["list", "of", "placeholder", "names"] (empty if is_template is false),
  "examples": [
    {{"instruction": "original instruction 1", "params": {{"query": "egg"}}}},
    {{"instruction": "original instruction 2", "params": {{"query": "milk"}}}}
  ]
}}

Examples:
- Input: ["Search for egg", "Search for milk"]
  Output: {{"different_actions": false, "instruction_template": "Search for {{query}}", "observation_template": "search results for {{query}}", "is_template": true, "param_names": ["query"], "examples": [{{"instruction": "Search for egg", "params": {{"query": "egg"}}}}, {{"instruction": "Search for milk", "params": {{"query": "milk"}}}}]}}

- Input: ["Tap the Settings icon"]
  Output: {{"different_actions": false, "instruction_template": "Tap the Settings icon", "observation_template": "", "is_template": false, "param_names": [], "examples": []}}

- Input: ["Type 'Waterloo' in the start location field", "Type '221B Baker Street' in the start location field"]
  Output: {{"different_actions": false, "instruction_template": "Type '{{location}}' in the start location field", "observation_template": "start location field shows {{location}}", "is_template": true, "param_names": ["location"], "examples": [{{"instruction": "Type 'Waterloo' in the start location field", "params": {{"location": "Waterloo"}}}}, {{"instruction": "Type '221B Baker Street' in the start location field", "params": {{"location": "221B Baker Street"}}}}]}}

- Input: ["Tap the three-dot overflow menu in the top-right corner", "Swipe upward on the alarm list", "Tap the switch of the 7:00 AM alarm"]
  Output: {{"different_actions": true, "instruction_template": "", "observation_template": "", "is_template": false, "param_names": [], "examples": []}}"""

PAGE_DESCRIBE_AND_STATE_PROMPT = """\
You are an expert at analyzing mobile app screenshots. You are helping build a navigation \
graph of an app where each node is a distinct SCREEN TYPE (template).

{known_descriptions_section}

{existing_keys_section}

Analyze this screenshot and return a JSON object with exactly four keys:

1. "page_description": The screen TYPE in under 10 words. This is a reusable label for the \
screen template — it must NOT contain any user-specific content (search queries, item names, \
addresses, dates, prices, usernames, etc.). Think of it as the screen's class name. \
Good: "search results page", "route planning page", "location search page" \
Bad: "search results for Big Ben", "route from home to work", "John's profile page" \
All user-specific data belongs in "state", never in "page_description". \
The on-screen keyboard, which input field has focus, how far a list is scrolled, \
an overflow menu, dropdown, picker, popup or context menu open over the screen, and which \
of its sections are expanded never change "page_description"; record them in "state". \
A dialog, bottom sheet or full-screen picker that has its own purpose, such as a \
"data usage warning dialog", is a screen of its own: describe the dialog, and record a \
menu open over it in "state". \
Always describe what you actually see. \
{description_hint}

2. "state": A FLAT JSON object capturing ONLY the DYNAMIC state of this screen — values \
that change through user interaction. This is where ALL user-specific content goes.

For the "state" object, focus on:
- "first_visible_item": the first list/menu item visible on screen (scroll position)
- "last_visible_item": the last list/menu item visible on screen
- "focused_field": the input field that has focus, or null when none has
- "keyboard_visible": whether the on-screen keyboard is shown
- "open_menu": the overflow menu, dropdown, picker, popup or context menu open over the \
  screen, named by what opened it (e.g. "overflow menu", "Security dropdown"), or null \
  when none is open
- "expanded_sections": on a screen with collapsible sections, the ones shown expanded, \
  comma-separated (e.g. "Advanced options"), or null when none is
- User-controlled state: active tab, search query, applied filters, sort order, toggle states, \
  selected items, input field contents
- Navigation state: which sub-section or category is currently shown

Do NOT include in "state":
- Static UI elements that never change (titles, fixed icons, permanent buttons, status bar)
- Summaries or descriptions of list items
- System status (time, battery, signal)
- Boolean flags for whether fixed elements are visible
- Any value that would be identical every time this screen is opened

State rules:
- Keys should be snake_case parameter names.
- Values should be the CURRENT value observed (strings, numbers, or booleans).
- If a parameter exists but you cannot determine its value, set it to null.
- Do NOT nest objects or use arrays inside "state". Keep it flat.

3. "elements": A list of INTERACTABLE elements visible on this screen. Each element is an \
object with two keys:
- "description": Short label for the element (e.g. "search box", "share button", "Home tab"). \
  Use the screen's own text/labels when available. Keep it under 8 words.
- "position": General region on screen (e.g. "top left", "center", "bottom navigation bar", \
  "top right corner"). Do NOT use pixel coordinates.

Element rules:
- Include only elements a user can tap, type into, or swipe — buttons, tabs, icons, input \
  fields, menu items, toggles, links, list items, etc.
- For STRUCTURAL elements (always present on this screen type): list each one individually \
  (e.g. each bottom nav tab, each toolbar icon, the search bar, the back button).
- For CONTENT elements (vary with data — e.g. search result items, product cards, list entries): \
  group them as ONE entry (e.g. "search result items" not each individual result). \
  Only include one representative entry for the group.
- Do NOT include non-interactable text, decorative images, status bar icons, or background areas.
- Order roughly top-to-bottom, left-to-right as they appear on screen.

4. "is_external_web": true only when this screen shows web content from outside the app, \
such as a browser or a custom tab that displays an address bar or a URL. The app's own \
screens are false, including screens it renders in a web view and screens that mention \
or link to a website, a privacy policy or terms of service.

Return ONLY the JSON object, nothing else.
Example:
{{"page_description": "product search results", "state": {{"search_query": "keyboard", "filter_brand": "logitech"}}, \
"elements": [{{"description": "back button", "position": "top left"}}, \
{{"description": "search box", "position": "top center"}}, \
{{"description": "filter button", "position": "top right"}}, \
{{"description": "sort dropdown", "position": "below search box"}}, \
{{"description": "product result items", "position": "center list"}}, \
{{"description": "Home tab", "position": "bottom navigation bar"}}, \
{{"description": "Cart tab", "position": "bottom navigation bar"}}, \
{{"description": "Account tab", "position": "bottom navigation bar"}}], \
"is_external_web": false}}"""

NODE_VERIFY_PROMPT = """\
You are comparing two mobile app screenshots to decide if they are the SAME screen or DIFFERENT screens.

Each screenshot is shown with the description its screen was given.

Two screenshots are the SAME screen if they serve the same purpose and a user would consider \
them the same page — even if the content (list items, search results, text in fields) differs. \
For example, a search results page showing different queries is still the same screen. \
They are also the SAME screen when they differ only in whether the on-screen keyboard is shown, \
which input field has focus, how far a list is scrolled, whether an overflow menu, dropdown, \
picker, popup or context menu is open over the screen, or which of its sections are expanded \
or collapsed. A menu open over a dialog leaves it the same dialog.

Two screenshots are DIFFERENT screens if they serve different purposes in the app's workflow, \
even if they look visually similar. Pay close attention to:
- Screen titles, headers, or labels (e.g. "Choose start location" vs "Choose destination")
- The stage in a multi-step flow (e.g. step 1 vs step 2 of a booking)
- Different form sections or tabs that change the screen's function
- A dialog, bottom sheet or full-screen picker that has its own purpose, such as a \
"Data usage warning" dialog, is a DIFFERENT screen from the page beneath it.
- A search page and a search result list page is considered different usually, though sometimes the search page also provide some suggestions.

A destination the app names in its own navigation, such as a tab, a bottom navigation bar \
item or a navigation drawer entry, is a screen of its own. Two screenshots that show \
different destinations are DIFFERENT screens, even when they share a layout and serve the \
same purpose: Downloads, Recent and Internal storage in a file manager are three screens, \
and so are Alarm and Timer in a clock. Content inside one destination is the SAME screen as \
that destination: a subfolder opened inside it, a different search query or a different list. \
The selected tab or navigation item, or the first entry of a breadcrumb trail, shows which \
destination a screenshot is in.

Answer with ONLY a JSON object:

If they are the SAME screen:
{"same": true, "reason": "brief explanation"}

If they are DIFFERENT screens, you must also provide a slightly more specific description of the \
second screenshot that distinguishes it from the first. Usually adding one or two extra \
distinguishing words is enough. Keep the description concise (under 10 words). \
Name the screen by its purpose, never by its current state: leave out the on-screen keyboard, \
which input field has focus, how far a list is scrolled, any overflow menu, dropdown, picker, \
popup or context menu open over it, and which of its sections are expanded, even when the \
screenshot shows them:
{"same": false, "reason": "brief explanation", \
"new_description": "more specific description for the second screenshot"}"""

_SEVERITY: dict[str, Any] = {
    "type": "string",
    "enum": ["high", "medium", "low"],
    "description": "How likely and how harmful the issue is.",
}

_EDGE_TEMPLATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "different_actions": {
            "type": "boolean",
            "description": "Whether the instructions are different actions that no one "
            "instruction with placeholders describes; every other field is then empty.",
        },
        "instruction_template": {
            "type": "string",
            "description": "The instruction with each user-provided value replaced by a "
            "{placeholder}, or the instruction unchanged when there is none.",
        },
        "observation_template": {
            "type": "string",
            "description": "The observation normalized the same way, or an empty string.",
        },
        "is_template": {
            "type": "boolean",
            "description": "Whether any placeholder was inserted.",
        },
        "param_names": {
            "type": "array",
            "items": {"type": "string"},
            "description": "The placeholder names; empty when is_template is false.",
        },
        "examples": {
            "type": "array",
            "description": "Each original instruction with the value each placeholder took.",
            "items": {
                "type": "object",
                "properties": {
                    "instruction": {"type": "string", "description": "An original instruction."},
                    "params": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                        "description": "Placeholder name to the value it took.",
                    },
                },
                "required": ["instruction", "params"],
                "additionalProperties": False,
            },
        },
    },
    "required": [
        "different_actions",
        "instruction_template",
        "observation_template",
        "is_template",
        "param_names",
        "examples",
    ],
    "additionalProperties": False,
}

_PAGE_DESCRIPTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "page_description": {
            "type": "string",
            "description": "The screen type in under 10 words, with no user-specific content.",
        },
        "state": {
            "type": "object",
            "additionalProperties": {"type": ["string", "number", "boolean", "null"]},
            "description": "Flat dynamic state: snake_case keys mapped to the current value, "
            "or null when it cannot be read.",
        },
        "elements": {
            "type": "array",
            "description": "Interactable elements, roughly top to bottom and left to right.",
            "items": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "A short label, under 8 words, using the screen's text.",
                    },
                    "position": {
                        "type": "string",
                        "description": "The general screen region, never pixel coordinates.",
                    },
                },
                "required": ["description", "position"],
                "additionalProperties": False,
            },
        },
        "is_external_web": {
            "type": "boolean",
            "description": "True only for web content from outside the app, such as a browser "
            "or custom tab showing an address bar or URL; the app's own web views are false.",
        },
    },
    "required": ["page_description", "state", "elements", "is_external_web"],
    "additionalProperties": False,
}

_NODE_VERIFICATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "same": {
            "type": "boolean",
            "description": "Whether the two screenshots show the same screen.",
        },
        "reason": {"type": "string", "description": "A brief explanation."},
        "new_description": {
            "type": ["string", "null"],
            "description": "When the screens differ, a more specific description of the second "
            "screenshot that tells it apart from the first; null when they are the same.",
        },
    },
    "required": ["same", "reason", "new_description"],
    "additionalProperties": False,
}

NEXT_ACTION_PLAN_PROMPT = """\
You are an exploration planner for a mobile app. Your goal is to systematically explore \
the app to discover as many DISTINCT screens and functionalities as possible.

App name: {app_name}
Current screen description: "{page_description}"
Current input status: {input_status}
{back_leaves_app_line}
Instructions already tried from this screen and where they led:
{explored_instructions}

{unexplored_elements_section}
EXPLORATION STRATEGY — follow these rules carefully:

1. **Bottom navigation bar.** If this screen has a bottom menu/navigation bar \
(e.g. Home, Search, Profile, Settings tabs), those tabs are HIGH PRIORITY and should \
be explored first — each tab usually leads to a completely different section of the \
app, maximizing screen diversity with minimal effort.

2. **Text input.** If this screen has a text input field (search bar, form \
field, title field, add-item field, etc.) AND no text input action has been tried \
yet (check the explored instructions list), your action should involve that text field. \
IMPORTANT: give only ONE atomic action at a time:
   - If the text field is NOT yet focused/active: instruct to TAP it first. \
     (e.g. "Tap on the search bar", "Tap on the title field")
   - If a text field IS focused/active, instruct to TYPE the value. Treat any \
     of these as active/focused: a visible keyboard, a blinking cursor, or a \
     bottom-of-screen keyboard/input bar indicating the keyboard has been activated \
     even if the full keyboard is not visible. (e.g. "Type 'stewed beef'", "Type 'Lunch'")
     Minimize-keyboard buttons, keyboard-switcher buttons, and keyboard controls at \
     the bottom of the screen are NOT app elements to explore; they only indicate \
     that the keyboard/input field is active, so you can directly type strings.
   - If text has been typed and needs submission: instruct to tap the submit/search \
     button or press enter. (e.g. "Tap the search button", "Press enter")
   Pick relevant sample values for the app's domain.
   Similarly, if this screen is a **search results page** (i.e. the previous action \
was a search) and no result item has been tapped yet, your action should be \
to tap one representative result item to discover the detail page.

3. **Focus on active/foreground elements.** If an overlay, popup, bottom sheet, \
dialog, filter panel, or sort menu is visible on screen, the background behind it is \
typically greyed out and NOT interactable. You MUST interact only with the foreground \
overlay elements (e.g. select a filter option, tap a sort choice, dismiss the dialog). \
If the current page displays a bright foreground overlay and dimmed or covered \
background elements, only the bright overlay elements are current explorable elements. \
The dimmed/covered elements belong to the back page, not the current explorable surface. \
Do NOT attempt to tap anything in the greyed-out background — those taps will either \
fail or simply dismiss the overlay.

4. **Prioritize structural diversity.** Choose actions that lead to genuinely DIFFERENT \
types of screens (e.g. a settings page vs a detail page vs a search page). Do NOT \
tap on repetitive items that clearly lead to the same type of page. For example, \
in a product list, tapping one product is enough — tapping every product wastes steps \
since they all open the same product detail template. Similarly, in a settings menu, \
each individual setting leads to a unique page, so those ARE worth exploring.

5. **Recognize functional equivalence.** Items in a list/grid that share the same \
visual structure (e.g. search results, contacts, emails, products) are usually \
instances of the same template. One representative tap is sufficient. Instead, \
look for structurally different elements: navigation tabs, toolbar icons, filters, \
sort buttons, floating action buttons, menu drawers, etc.

6. **Go back when the screen is fully explored.** If most interactive elements on \
this screen have already been explored (check the instructions list above), press the \
Back button to return to the parent screen and explore other branches. Do NOT \
keep tapping unexplored repetitive items just because they haven't been clicked.

7. **Prioritize the app's major features.** Think about what "{app_name}" is for and \
what its main user flows are. For example, a transport app's major features include \
route planning, live departures, and saved places — not the "About" page. \
Actively seek out the app's core features by exploring buttons, tabs, and entry points \
that lead to primary functionality. Spend more steps on screens central to the app's \
purpose and fewer on secondary pages like "About", "Terms of Service", "Help", \
"Feedback", or deep settings sub-menus. If the current screen is a secondary page, \
go back quickly to explore more important branches.

8. **Scroll only when valuable.** Scroll down only if you believe new TYPES of \
elements or sections will appear (e.g. more menu categories). Do not scroll just \
to see more of the same list items.

9. **One atomic action per instruction.** Each instruction should be a SINGLE \
action — one tap, one type, one swipe, or one back. Do NOT combine multiple \
actions (e.g. do NOT say "Tap the search bar and type 'beef'"). Examples:
   - "Tap the search bar at the top of the screen" (just tap, don't type yet)
   - "Type 'stewed beef'" (only if the field is already focused)
   - "Press Enter"
   - "Tap the 'Settings' menu item in the left side drawer"

10. **Be specific enough for coordinate grounding.** The action model only sees this \
one instruction plus the screenshot, so describe the target precisely. For tap/select \
actions, include visible anchors such as screen location, label, icon shape, color, \
nearby text, list position, or container when they help disambiguate. If you refer to \
text visible on the screenshot, wrap the exact text in quotes. Prefer "Tap the blue \
search icon in the top-right corner" over "Tap search", and "Tap the \"Price: Low to \
High\" option in the sort menu" over "Select low price".

11. **Recover English language immediately.** If the app language has accidentally \
changed away from English, your next action MUST be to switch the app back to English \
immediately. Ignore the no-repeat rule for this language-recovery action; repeating \
a previous language-setting instruction is allowed when needed to restore English.

12. **Never repeat a previous instruction.** Look at the "Instructions already tried" \
list carefully. Do NOT output an instruction that is semantically the same as one \
already listed (such as search with different words or selecting search result), \
even if worded differently. If all useful actions have been tried, go back.

13. **Go back from external websites.** If you see a website URL / address bar \
displayed at the top or bottom of the screen (e.g. "example.com", \
"https://..."), this is usually an in-app browser showing an external website, \
NOT part of the app itself. Always go back immediately — do not explore external \
web content.

14. **NEVER perform dangerous or irreversible actions.** Do NOT: register accounts, \
sign up, log in, create passwords, submit forms with personal data, make purchases \
and checkout, send messages or emails (create and compose is fine), delete data, \
change account settings, or perform any action that has real-world side effects. \
You are ONLY exploring the app's UI structure — navigate and observe, but never \
commit or submit anything. If a screen requires login/signup to proceed, go back instead.

Reply with ONLY a JSON object with two keys:
- "element": the number of the element in the list of elements NOT interacted with yet \
that your instruction acts on, or null when it acts on none of them (for example going \
back, scrolling, or typing into an already focused field).
- "instruction": a single ATOMIC action. Each instruction = one tap, one type, one \
swipe, or one back. Never combine steps.

Example instructions:
- Tap the "Network & internet" menu item near the top of the settings list
- Tap the rounded search bar at the top of the screen
- Type 'stewed beef'
- Press Enter
- Tap the first search result card in the center list
- Select the "Price: Low to High" option from the sort menu
- Scroll down to reveal more settings
- Go back to the previous screen

Example reply:
{{"element": 2, "instruction": "Tap the rounded search bar at the top of the screen"}}"""

BACK_LEAVES_APP_LINE = (
    "Back on this screen leaves the app for the home screen, so choose an action on this "
    "screen instead, even where a rule below says to go back.\n"
)

NEXT_ACTION_SYSTEM_PROMPT = """\

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{{"type": "function", "function": {{"name": "mobile_use", "description": "Use a touchscreen to interact with a mobile device, and take screenshots.\\n* This is an interface to a mobile device with touchscreen. You can perform actions like clicking, typing, swiping, etc.\\n* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.\\n* The screen's resolution is 999x999.\\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.", "parameters": {{"properties": {{"action": {{"description": "The action to perform. The available actions are:\\n* `click`: Click the point on the screen with coordinate (x, y).\\n* `long_press`: Press the point on the screen with coordinate (x, y) for specified seconds.\\n* `swipe`: Swipe from the starting point with coordinate (x, y) to the end point with coordinates2 (x2, y2).\\n* `type`: Input the specified text into the activated input box.\\n* `system_button`: Press the system button.\\n* `wait`: Wait specified seconds for the change to happen.\\n* `end`: Signal that the current task is completed. Use this when the task goal has been achieved.", "enum": ["click", "long_press", "swipe", "type", "system_button", "wait", "end"], "type": "string"}}, "coordinate": {{"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=click`, `action=long_press`, and `action=swipe`.", "type": "array"}}, "coordinate2": {{"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=swipe`.", "type": "array"}}, "text": {{"description": "Required only by `action=type`.", "type": "string"}}, "time": {{"description": "The seconds to wait. Required only by `action=long_press` and `action=wait`.", "type": "number"}}, "button": {{"description": "Back means returning to the previous interface, Home means returning to the desktop, Menu means opening the application options menu, and Enter means pressing the enter. Required only by `action=system_button`", "enum": ["Back", "Home", "Menu", "Enter"], "type": "string"}}}}, "required": ["action"], "type": "object"}}}}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>


Response format for every step:
1) Thought: one concise sentence explaining the next move (no multi-step reasoning).
2) Action: a short imperative describing what to do in the UI.
3) A single <tool_call>...</tool_call> block containing only the JSON: {{"name": <function-name>, "arguments": <args-json-object>}}.

Rules:
- Output exactly in the order: Thought, Action, <tool_call>.
- Be brief: one sentence for Thought, one for Action.
- If an overlay, popup, bottom sheet, dialog, or filter panel is visible, \
the background is greyed out and NOT interactable. Interact ONLY with \
the foreground overlay elements.
- IMPORTANT: If the task is a single action like "scroll down", "swipe", or "tap X", do it ONCE \
and then call `end`. Do NOT repeat the same action or add extra actions.
- Do NOT overreact: if the instruction requires only one action (e.g. "tap Settings", \
"go back"), perform that single action and immediately call `end`. Only perform \
multiple steps when the instruction genuinely requires them (e.g. "search for X" \
needs tap → type → submit).
- If the instruction says you are already inside the target app, do NOT press Home, \
do NOT open the app, and do NOT leave the app. Use one visible in-app control that \
moves toward the requested goal. Only use Home if the instruction explicitly asks \
for Home or the screenshot clearly shows you are outside the target app.
- Do not output anything else outside those three parts."""

NEXT_ACTION_USER_PROMPT = """\
You are interacting with a mobile app.

Your task: {instruction}
{overall_task_section}\
{history_section}\
Look at the screenshot and execute the SINGLE next action to complete the task above.
Focus on the task instruction. The overall goal is only for context when the instruction is ambiguous."""

NEXT_ACTION_RETRY_PROMPT = """\
Your previous reply could not be used: {reason}. Reply again with Thought, Action, and \
one <tool_call> block containing only JSON: {{"name": "mobile_use", "arguments": {{...}}}}."""


def _structured_output(
    name: str, schema: dict[str, Any], *, strict: bool = True
) -> ResponseFormatTypedDict:
    """Ask for JSON matching ``schema``; parsers still validate, since not every provider does."""
    return {
        "type": "json_schema",
        "json_schema": {"name": name, "strict": strict, "schema_": schema},
    }


class GenerationFailedError(UnusableOutputError):
    """The provider reported ``finish_reason: "error"``: generation failed partway."""


class TruncatedReplyError(Exception):
    """The reply hit its token limit (``finish_reason: "length"``); asking again cannot help."""


def _send_chat_once(
    client: OpenRouter,
    *,
    call_type: str,
    settings: ModelSettings,
    messages: list[ChatMessagesTypedDict],
    response_format: ResponseFormatTypedDict | None = None,
) -> ChatResult:
    """Send one completion with its call type's timeout, record its usage and check how
    it finished.

    A ``response_format`` brings the response-healing plugin with it;
    README.md#requests-and-recovery.
    """
    timeout = settings.timeout(call_type)
    response: object = send_within_deadline(
        call_type,
        timeout,
        lambda: client.chat.send(
            model=settings.model,
            messages=messages,
            response_format=response_format,
            plugins=None if response_format is None else [{"id": "response-healing"}],
            temperature=settings.temperature if settings.temperature is not None else UNSET,
            stream=False,
            timeout_ms=int(timeout * 1000),
        ),
    )
    if not isinstance(response, ChatResult):
        msg = "Expected a non-streaming OpenRouter chat response"
        raise TypeError(msg)
    token_tracker.record(call_type, settings.model, response.usage)
    finish_reason = response.choices[0].finish_reason if response.choices else None
    if finish_reason == "error":
        msg = f"{call_type}: the provider reported finish_reason=error"
        raise GenerationFailedError(msg)
    if finish_reason == "length":
        reasoning_tokens = _reasoning_tokens(response.usage)
        logger.warning(
            "%s: the reply hit its token limit (finish_reason=length%s); not asking again",
            call_type,
            "" if reasoning_tokens is None else f", reasoning_tokens={reasoning_tokens}",
        )
        msg = f"{call_type}: the reply hit its token limit"
        raise TruncatedReplyError(msg)
    if finish_reason == "content_filter":
        logger.warning(
            "%s: the provider's content filter stopped the reply (finish_reason=content_filter)",
            call_type,
        )
    return response


def send_chat(
    client: OpenRouter,
    *,
    call_type: str,
    settings: ModelSettings,
    messages: list[ChatMessagesTypedDict],
    response_format: ResponseFormatTypedDict | None = None,
) -> ChatResult:
    """Send a non-streaming completion, asking once more for unusable output.

    Unusable output is a failed generation (``finish_reason: "error"``) or a 2xx
    reply the SDK cannot validate. ``call_type`` names the call in usage records
    and logs, and chooses its timeout. A truncated reply raises ``TruncatedReplyError``
    and a request that still times out at its deadline raises
    ``httpx.TimeoutException``, both for the caller's final fallback.
    """
    return call_with_retry(
        f"[API] {call_type}",
        lambda: _send_chat_once(
            client,
            call_type=call_type,
            settings=settings,
            messages=messages,
            response_format=response_format,
        ),
    )


def message_text(resp: ChatResult) -> str:
    """Read text, multipart text, null or omitted content from the SDK response."""
    if not resp.choices:
        return ""
    content = resp.choices[0].message.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "\n".join(part.text for part in content if isinstance(part, ChatContentText)).strip()
    return ""


def build_image_message(screenshot_b64: str) -> ChatContentImageTypedDict:
    """Build an OpenRouter image_url message part from a base64 screenshot, which
    ``screen_image`` encoded when it was captured."""
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{MEDIA_TYPE};base64,{screenshot_b64}",
        },
    }


def _parse_page_description(
    text: str,
) -> tuple[str, dict[str, object], list[dict[str, str]], bool]:
    result = json.loads(strip_json_fences(text))
    if not isinstance(result, dict):
        msg = "Expected page description to be a JSON object"
        raise TypeError(msg)
    payload = as_object_dict(result)
    description = payload.get("page_description")
    if not isinstance(description, str) or not description.strip():
        msg = "Expected a non-empty page_description string"
        raise ValueError(msg)
    state = payload.get("state", {})
    if not isinstance(state, dict):
        msg = "Expected page state to be a JSON object"
        raise TypeError(msg)
    raw_elements = payload.get("elements", [])
    if not isinstance(raw_elements, list):
        msg = "Expected page elements to be a JSON array"
        raise TypeError(msg)
    elements: list[dict[str, str]] = []
    for raw_element in raw_elements:
        element = as_object_dict(raw_element)
        label, position = element.get("description"), element.get("position")
        if not isinstance(label, str) or not isinstance(position, str):
            msg = "Expected element description and position strings"
            raise TypeError(msg)
        elements.append({"description": label, "position": position})
    is_external_web = payload.get("is_external_web", False)
    if not isinstance(is_external_web, bool):
        msg = "Expected is_external_web to be a boolean"
        raise TypeError(msg)
    return description, as_object_dict(state), elements, is_external_web


def describe_page_and_state(
    client: OpenRouter,
    screenshot_b64: str,
    known_descriptions: list[str] | None = None,
    existing_keys: list[str] | None = None,
    *,
    settings: ModelSettings,
) -> tuple[str, dict[str, object], list[dict[str, str]], bool]:
    """Describe a screen, retrying failed requests and malformed descriptions.

    The model always writes its own description based on what it sees. The
    descriptions of the screens already seen are provided as context so the model
    can disambiguate similar-looking screens (e.g. "start location search" vs "end
    location search").

    Args:
        known_descriptions: A selection of distinct descriptions of screens already
            seen in the same package.
        existing_keys: State keys already seen across same-package nodes.

    Returns:
        (page_description, state_dict, elements_list, is_external_web)
        where elements_list is a list of {"description": str, "position": str} dicts
        and is_external_web is true only for web content from outside the app.
    """
    if known_descriptions:
        listed = "\n".join(f'  - "{d}"' for d in known_descriptions)
        known_descriptions_section = (
            f"These selected screens have already been discovered in this app:\n{listed}"
        )
        description_hint = (
            "If this screen is clearly the same type as one in the list, output the same "
            "description. If it is similar but serves a different purpose (e.g. a different "
            "step in a flow), write a description that clearly distinguishes it. Pay "
            "attention to screen titles, headers, labels, and what stage of a multi-step "
            "flow this is."
        )
    else:
        known_descriptions_section = "No known-screen descriptions are supplied for this screen."
        description_hint = "Focus on the screen's purpose. Use precise words."

    if existing_keys:
        keys_str = ", ".join(f'"{k}"' for k in existing_keys)
        existing_keys_section = (
            f"The following state keys have been identified from previous visits to "
            f"screens in this app:\n[{keys_str}]\n"
            f"Include any of these keys that are relevant to this screen, with their "
            f"current values. You may also add new keys."
        )
    else:
        existing_keys_section = (
            "No state keys have been identified yet. Identify all the "
            "dynamic parameters that define this screen's state."
        )

    prompt = PAGE_DESCRIBE_AND_STATE_PROMPT.format(
        known_descriptions_section=known_descriptions_section,
        existing_keys_section=existing_keys_section,
        description_hint=description_hint,
    )
    messages: list[ChatMessagesTypedDict] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                build_image_message(screenshot_b64),
            ],
        }
    ]

    def request_description() -> tuple[str, dict[str, object], list[dict[str, str]], bool]:
        resp = _send_chat_once(
            client,
            call_type="page_describe_and_state",
            settings=settings,
            messages=messages,
            # Not strict: "state" has free-form keys, which strict mode cannot express
            # because it needs every key listed and additionalProperties false.
            response_format=_structured_output(
                "page_description", _PAGE_DESCRIPTION_SCHEMA, strict=False
            ),
        )
        try:
            return _parse_page_description(message_text(resp))
        except (TypeError, ValueError) as exc:
            msg = "Malformed page description"
            raise UnusableOutputError(msg) from exc

    page_description, state, elements, is_external_web = call_with_retry(
        "[API] page describe and state", request_description
    )
    logger.debug("Page description: %s", page_description)
    logger.debug("Page state: %s", state)
    logger.debug("Elements: %d found", len(elements))
    return page_description, state, elements, is_external_web


def _no_verifier_answer(reason: str) -> dict[str, Any]:
    logger.warning("node_verify: no answer (%s)", reason)
    return {"same": False, "answered": False, "reason": reason}


@dataclass(frozen=True, slots=True)
class DescribedScreenshot:
    """A screenshot the verifier is shown, and the description its screen was given."""

    screenshot_b64: str
    description: str


def _shown(label: str, screenshot: DescribedScreenshot) -> list[ChatContentItemsTypedDict]:
    return [
        {"type": "text", "text": f'{label}, described as "{screenshot.description}":'},
        build_image_message(screenshot.screenshot_b64),
    ]


def verify_same_node(
    client: OpenRouter,
    *,
    existing: DescribedScreenshot,
    new: DescribedScreenshot,
    settings: ModelSettings,
) -> dict[str, Any]:
    """Compare two screenshots, each shown with its own description, to decide if they
    are the same screen.

    If they are different, the verifier also provides a more specific description of
    the new screen, so a vague description like "location search page" gets refined to
    "end location search page".

    A request that still times out at its deadline is asked once more, with a
    deadline of its own for sending it again.

    Returns a dict with keys:
        - "same": bool
        - "reason": str
        - "new_description": str (only when same=False) — refined name for the new screen
        - "answered": False (only when the verifier gave no answer: the request timed out
          twice, the reply was truncated or it could not be parsed) — "same" is then False
    """

    def ask() -> ChatResult:
        return send_chat(
            client,
            call_type="node_verify",
            settings=settings,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": NODE_VERIFY_PROMPT},
                        *_shown("First screenshot (existing node)", existing),
                        *_shown("Second screenshot (new screen)", new),
                    ],
                }
            ],
            response_format=_structured_output("node_verification", _NODE_VERIFICATION_SCHEMA),
        )

    try:
        try:
            resp = ask()
        except httpx.TimeoutException:
            logger.warning("node_verify: asking once more")
            resp = ask()
    except httpx.TimeoutException:
        return _no_verifier_answer("timed out")
    except TruncatedReplyError:
        return _no_verifier_answer("reply truncated")
    raw = strip_json_fences(message_text(resp))
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Failed to parse verify JSON. Raw:\n%s", raw)
        return _no_verifier_answer("parse error")

    logger.debug("Node verify: %s", result)
    return result


_MAX_INSTRUCTIONS_SENT = 20

_PLACEHOLDER = re.compile(r"\{+[^{}]*\}+")
_ACTION_LIST = re.compile(r"[\n\r|]|\bOR\b|^\d+[.)]\s")


def _is_one_instruction(template: str) -> bool:
    """Whether ``template`` has a word outside its placeholders, and no line break, "|",
    "OR" or leading number that lists several actions."""
    text = template.strip()
    return any(c.isalpha() for c in _PLACEHOLDER.sub("", text)) and not _ACTION_LIST.search(text)


def _different_actions() -> dict[str, Any]:
    return {
        "different_actions": True,
        "instruction_template": "",
        "observation_template": "",
        "is_template": False,
        "param_names": [],
        "examples": [],
    }


def _latest_distinct(recorded: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Each distinct instruction of ``recorded`` with the observation it led to last, in the
    order of their last recordings."""
    latest: dict[str, str] = {}
    for instruction, observation in recorded:
        latest.pop(instruction, None)
        latest[instruction] = observation
    return list(latest.items())


def normalize_edge(
    client: OpenRouter,
    instructions: list[str],
    target_observations: list[str] | None = None,
    *,
    settings: ModelSettings,
) -> dict[str, Any]:
    """Normalize edge instructions into a reusable template.

    Takes one or more concrete instructions (e.g. "search for egg", "search for milk")
    and produces a template with placeholders (e.g. "search for {query}"). It sends the
    20 most recent distinct instructions, each with the observation it led to last, and the
    count of the older ones.

    Args:
        instructions: List of concrete instruction strings to normalize, in the order
            they were recorded, at least one of them non-empty.
        target_observations: Optional list of corresponding target observations.

    Returns a dict with keys:
        - different_actions: bool — whether the instructions are different actions, which
          share no template; a reply whose template is placeholders alone or lists several
          actions counts as different actions, and every other field is then empty
        - instruction_template: str — the normalized template
        - observation_template: str — normalized observation (may be empty)
        - is_template: bool — whether any placeholders were inserted
        - param_names: list[str] — placeholder names
        - examples: list[dict] — original instructions with extracted param values
    """
    observations = target_observations or []
    distinct = _latest_distinct(
        [
            (inst, observations[i] if i < len(observations) else "")
            for i, inst in enumerate(instructions)
            if inst
        ]
    )
    sent = distinct[-_MAX_INSTRUCTIONS_SENT:]
    older = len(distinct) - len(sent)
    input_text = (
        f"Instructions to normalize (the {len(sent)} most recent; "
        f"{older} older ones are not shown):\n"
        if older
        else "Instructions to normalize:\n"
    )
    for inst, obs in sent:
        input_text += f'- Instruction: "{inst}"'
        if obs:
            input_text += f' → Observation: "{obs}"'
        input_text += "\n"

    first_instruction, first_observation = sent[0]
    unchanged = {
        "different_actions": False,
        "instruction_template": first_instruction,
        "observation_template": first_observation,
        "is_template": False,
        "param_names": [],
        "examples": [],
    }
    try:
        resp = send_chat(
            client,
            call_type="normalize_edge",
            settings=settings,
            messages=[
                {
                    "role": "user",
                    "content": f"{NORMALIZE_EDGE_PROMPT}\n\n{input_text}",
                }
            ],
            # Not strict: each example's "params" maps free-form placeholder names.
            response_format=_structured_output(
                "edge_template", _EDGE_TEMPLATE_SCHEMA, strict=False
            ),
        )
    except TruncatedReplyError, httpx.TimeoutException:
        return unchanged
    raw = strip_json_fences(message_text(resp))
    try:
        result = as_str_dict(json.loads(raw))
    except json.JSONDecodeError:
        logger.warning("Failed to parse normalize JSON. Raw:\n%s", raw)
        return unchanged

    logger.debug("Normalize edge: %s", result)
    template = as_str(result.get("instruction_template"), "")
    if result.get("different_actions") is True or not _is_one_instruction(template):
        return _different_actions()
    return result


GRAPH_AUDIT_PROMPT = """\
You are an expert at analyzing mobile app navigation graphs. You are reviewing a graph \
that was automatically built by an exploration agent. The agent sometimes clicks wrong \
buttons, so some edges may go to the wrong destination.

## App: {app_name}

## Graph structure
{graph_description}

## Your task
Find three types of issues:

### 1. merge_nodes — Node pairs that might be the same screen
Two nodes that look like they represent the same screen type, just described differently. \
This often happens when the exploration agent visits the same screen from different paths \
and the describer gives slightly different names. \
For example:
- "start location search" and "start location suggestions" (same screen, different names)
- "route planner form" and "plan a journey page" (same screen, paraphrased)
- "settings list" and "app settings" (same screen)
Do NOT flag nodes that are clearly different screens with similar names (e.g. \
"start location search" and "end location search" are different).
For each, identify both node IDs and explain why they might be the same.

### 2. retry_edge — Suspicious edges that need re-testing
An edge's instruction says one thing but it leads to an unexpected node. This often \
happens when the action agent clicked the wrong button during exploration. \
For example:
- "tap search bar" leading to "settings page" instead of "search page"
- "select a route" leading back to "home page" instead of "route detail page"
- An instruction that clearly describes navigating to node X, but the edge goes to node Y
For each, identify the source node and the specific edge instruction that looks wrong.

### 3. explore_node — Nodes missing expected edges
A node that should logically have more outgoing edges based on its description. \
For example:
- A "route planner page" with no edge to a "route results page"
- A "settings page" with only 1 explored action when it likely has many options
- A "search results page" with no edge for tapping a result item
For each, describe what edges/actions are likely missing.

Return a JSON object:
{{
  "issues": [
    {{
      "type": "merge_nodes",
      "severity": "high" | "medium" | "low",
      "description": "why these two nodes might be the same screen",
      "node_a": "first node_id",
      "node_b": "second node_id"
    }},
    {{
      "type": "retry_edge",
      "severity": "high" | "medium" | "low",
      "description": "why this edge looks wrong",
      "source_node": "node_id where the edge starts",
      "target_node": "node_id where the edge currently goes (wrongly)",
      "instruction": "the edge instruction that looks mismatched"
    }},
    {{
      "type": "explore_node",
      "severity": "high" | "medium" | "low",
      "description": "what is likely missing from this node",
      "node": "node_id that needs more exploration",
      "expected_pages": ["page that should be reachable from this node", "another reachable page"]
    }}
  ],
  "summary": "one-paragraph overall assessment of graph quality"
}}"""


NODE_MERGE_AUDIT_PROMPT = """\
You are reviewing mobile app graph nodes produced by an exploration agent.

## App
{app_name}

## Graph nodes
{graph_description}

## Task
Find node pairs that may represent the same screen and should be verified for merging.
Only report likely duplicate screen templates. Do NOT report different screens that merely \
look similar, such as start search vs destination search, or different selected tabs.

Return ONLY a JSON object:
{{
  "issues": [
    {{
      "type": "merge_nodes",
      "severity": "high" | "medium" | "low",
      "description": "why these nodes may be the same screen",
      "node_a": "first node_id",
      "node_b": "second node_id"
    }}
  ],
  "summary": "short summary"
}}"""


_MERGE_NODES_ISSUE: dict[str, Any] = {
    "type": "object",
    "properties": {
        "type": {"type": "string", "enum": ["merge_nodes"], "description": "The issue type."},
        "severity": _SEVERITY,
        "description": {
            "type": "string",
            "description": "Why these two nodes may be the same screen.",
        },
        "node_a": {"type": "string", "description": "The first node_id."},
        "node_b": {"type": "string", "description": "The second node_id."},
    },
    "required": ["type", "severity", "description", "node_a", "node_b"],
    "additionalProperties": False,
}

_NODE_MERGE_AUDIT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "issues": {
            "type": "array",
            "items": _MERGE_NODES_ISSUE,
            "description": "Likely duplicate screen templates to verify for merging.",
        },
        "summary": {"type": "string", "description": "A short summary."},
    },
    "required": ["issues", "summary"],
    "additionalProperties": False,
}

_GRAPH_AUDIT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "issues": {
            "type": "array",
            "description": "Every merge_nodes, retry_edge and explore_node issue found.",
            "items": {
                "anyOf": [
                    _MERGE_NODES_ISSUE,
                    {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": ["retry_edge"],
                                "description": "The issue type.",
                            },
                            "severity": _SEVERITY,
                            "description": {
                                "type": "string",
                                "description": "Why this edge looks wrong.",
                            },
                            "source_node": {
                                "type": "string",
                                "description": "The node_id where the edge starts.",
                            },
                            "target_node": {
                                "type": "string",
                                "description": "The node_id the edge currently goes to.",
                            },
                            "instruction": {
                                "type": "string",
                                "description": "The edge instruction that looks mismatched.",
                            },
                        },
                        "required": [
                            "type",
                            "severity",
                            "description",
                            "source_node",
                            "target_node",
                            "instruction",
                        ],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": ["explore_node"],
                                "description": "The issue type.",
                            },
                            "severity": _SEVERITY,
                            "description": {
                                "type": "string",
                                "description": "What is likely missing from this node.",
                            },
                            "node": {
                                "type": "string",
                                "description": "The node_id that needs more exploration.",
                            },
                            "expected_pages": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Pages that should be reachable from this node.",
                            },
                        },
                        "required": ["type", "severity", "description", "node", "expected_pages"],
                        "additionalProperties": False,
                    },
                ]
            },
        },
        "summary": {
            "type": "string",
            "description": "A one-paragraph overall assessment of graph quality.",
        },
    },
    "required": ["issues", "summary"],
    "additionalProperties": False,
}


def audit_merge_nodes(
    client: OpenRouter,
    graph_summary: str,
    app_name: str = "",
    *,
    settings: ModelSettings,
) -> dict[str, Any]:
    """Ask a model only for duplicate-node merge candidates."""
    prompt = NODE_MERGE_AUDIT_PROMPT.format(
        app_name=app_name,
        graph_description=graph_summary,
    )
    try:
        resp = send_chat(
            client,
            call_type="node_merge_audit",
            settings=settings,
            messages=[{"role": "user", "content": prompt}],
            response_format=_structured_output("node_merge_audit", _NODE_MERGE_AUDIT_SCHEMA),
        )
    except TruncatedReplyError, httpx.TimeoutException:
        return {"issues": [], "summary": "reply truncated"}
    raw = strip_json_fences(message_text(resp))
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Failed to parse node merge audit JSON. Raw:\n%s", raw)
        return {"issues": [], "summary": "parse error"}

    issues = [issue for issue in result.get("issues", []) if issue.get("type") == "merge_nodes"]
    return {
        "issues": issues,
        "summary": result.get("summary", ""),
    }


EXPLORATION_TARGET_PROMPT = """\
You are helping a mobile app exploration agent decide where to continue exploring.

## App
{app_name}

## Candidate nodes
{candidates}

## Task
Pick exactly one node that is most urgent to continue exploring from.
Prefer nodes with many unexplored elements, low outgoing edge count, important UI surfaces \
such as settings/search/menus/tabs, and nodes that look under-covered.
A candidate marked start screen is the app's entry screen; its unexplored elements usually \
open top-level sections.

Return ONLY a JSON object:
{{
  "node_id": "candidate node_id",
  "reason": "short reason"
}}"""


def _exploration_target_schema(node_ids: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "node_id": {
                "type": "string",
                "enum": node_ids,
                "description": "The candidate node_id most urgent to continue exploring from.",
            },
            "reason": {"type": "string", "description": "A short reason."},
        },
        "required": ["node_id", "reason"],
        "additionalProperties": False,
    }


def select_exploration_target(
    client: OpenRouter,
    app_name: str,
    candidates: list[dict[str, Any]],
    *,
    settings: ModelSettings,
) -> str | None:
    """Select a node for periodic live exploration coverage balancing."""
    if not candidates:
        return None

    candidate_lines = []
    valid_ids = {c.get("node_id", "") for c in candidates}
    for i, candidate in enumerate(candidates, 1):
        unexplored_desc = candidate.get("unexplored_element_descriptions", [])
        unexplored_text = ", ".join(d for d in unexplored_desc if d) or "none listed"
        line = (
            f"{i}. node_id={candidate.get('node_id', '')} | "
            f"descriptions={candidate.get('descriptions', '')} | "
            f"visits={candidate.get('visit_count', 0)} | "
            f"out_edges={candidate.get('out_degree', 0)} | "
            f"unexplored={candidate.get('unexplored_elements', 0)}/"
            f"{candidate.get('total_elements', 0)} | "
            f"score={candidate.get('score', 0)} | "
            f"unexplored_elements=[{unexplored_text}]"
        )
        if candidate.get("is_start_node"):
            line += " | start screen"
        candidate_lines.append(line)

    prompt = EXPLORATION_TARGET_PROMPT.format(
        app_name=app_name,
        candidates="\n".join(candidate_lines),
    )
    try:
        resp = send_chat(
            client,
            call_type="exploration_target",
            settings=settings,
            messages=[{"role": "user", "content": prompt}],
            response_format=_structured_output(
                "exploration_target", _exploration_target_schema(sorted(valid_ids))
            ),
        )
    except TruncatedReplyError, httpx.TimeoutException:
        return None

    raw = strip_json_fences(message_text(resp))
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Failed to parse exploration target JSON. Raw:\n%s", raw)
        return None

    node_id = result.get("node_id")
    reason = result.get("reason", "")
    if isinstance(node_id, str) and node_id in valid_ids:
        logger.info("Selected exploration target %s: %s", node_id, reason)
        return node_id

    logger.warning("Exploration target model selected invalid node_id=%s", node_id)
    return None


def audit_graph(
    client: OpenRouter,
    graph_summary: str,
    app_name: str = "",
    *,
    settings: ModelSettings,
) -> dict[str, Any]:
    """Ask a model to review the graph structure for anomalies.

    Args:
        graph_summary: Formatted text describing all nodes and edges.
        app_name: Name of the app for context.

    Returns a dict with "issues" list and "summary" string.
    """
    prompt = GRAPH_AUDIT_PROMPT.format(
        app_name=app_name,
        graph_description=graph_summary,
    )
    try:
        resp = send_chat(
            client,
            call_type="graph_audit",
            settings=settings,
            messages=[{"role": "user", "content": prompt}],
            response_format=_structured_output("graph_audit", _GRAPH_AUDIT_SCHEMA),
        )
    except TruncatedReplyError, httpx.TimeoutException:
        return {"issues": [], "summary": "reply truncated"}
    finish_reason = resp.choices[0].finish_reason if resp.choices else ""
    raw_original = message_text(resp)
    raw = strip_json_fences(raw_original)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning(
            "Failed to parse audit JSON. finish_reason=%s raw_len=%d raw_repr=%r",
            finish_reason,
            len(raw_original),
            raw_original,
        )
        return {"issues": [], "summary": "parse error"}


def _embedding_vector(
    response: CreateEmbeddingsResponse, *, call_type: str, model: str
) -> list[float]:
    """Record a completed response's usage, then return its one finite vector."""
    if not isinstance(response, CreateEmbeddingsResponseBody):
        msg = "Expected an OpenRouter JSON embedding response"
        raise TypeError(msg)
    token_tracker.record(call_type, model, response.usage)
    if len(response.data) != 1:
        msg = "Expected one embedding vector from OpenRouter"
        raise ValueError(msg)
    vector = as_float_list(response.data[0].embedding)
    if not vector:
        msg = "OpenRouter returned an empty, non-finite or non-float embedding"
        raise UnusableOutputError(msg)
    return vector


def get_embedding(
    client: OpenRouter,
    text: str,
    model: str,
    *,
    request_timeout: float = REQUEST_TIMEOUTS["embedding"],
) -> list[float]:
    """Get a text embedding through the official OpenRouter SDK."""

    def request_embedding() -> list[float]:
        response = send_within_deadline(
            "embedding",
            request_timeout,
            lambda: client.embeddings.generate(
                model=model,
                input=text,
                encoding_format="float",
                timeout_ms=int(request_timeout * 1000),
            ),
        )
        return _embedding_vector(response, call_type="embedding", model=model)

    return call_with_retry("[API] text embedding", request_embedding)


def get_image_embedding(
    client: OpenRouter,
    screenshot_b64: str,
    model: str,
    *,
    request_timeout: float = REQUEST_TIMEOUTS["image_embedding"],
) -> list[float]:
    """Embed a screenshot, which ``screen_image`` encoded when it was captured, as
    OpenRouter's multimodal content array:
    https://openrouter.ai/docs/api_reference/embeddings.
    """

    def request_embedding() -> list[float]:
        response = send_within_deadline(
            "image_embedding",
            request_timeout,
            lambda: client.embeddings.generate(
                model=model,
                input=[
                    {
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{MEDIA_TYPE};base64,{screenshot_b64}"},
                            }
                        ]
                    }
                ],
                encoding_format="float",
                timeout_ms=int(request_timeout * 1000),
            ),
        )
        return _embedding_vector(response, call_type="image_embedding", model=model)

    return call_with_retry("[API] image embedding", request_embedding)


def _parse_tool_call(raw: str) -> dict[str, Any] | None:
    """Extract the JSON arguments from a <tool_call> block, even one missing its </tool_call>."""
    m = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", raw, re.DOTALL)
    inner = m.group(1).strip() if m else ""
    start = raw.find("<tool_call>")
    if not inner and start == -1:
        return None

    # A closed block must hold only JSON; an unclosed one ends wherever its JSON object does.
    try:
        if inner:
            parsed = json.loads(strip_json_fences(inner))
        else:
            remainder = raw[start + len("<tool_call>") :].lstrip()
            parsed, _end = json.JSONDecoder().raw_decode(remainder)
    except json.JSONDecodeError:
        return None
    # A native tool call arrives as {"name": ..., "arguments": {...}}; only the arguments matter.
    if "arguments" in parsed:
        return parsed["arguments"]
    return parsed


def _scale(coord: object, screen_w: int, screen_h: int) -> tuple[int, int]:
    """Scale a coordinate on the model's 0-1000 scale to device pixels."""
    if not isinstance(coord, list) or len(coord) != _COORDINATE_AXES:
        msg = "Action coordinates must be two finite numbers"
        raise ValueError(msg)
    values = as_float_list(coord)
    if not values:
        msg = "Action coordinates must be two finite numbers"
        raise ValueError(msg)
    x = int(values[0] / 1000 * screen_w)
    y = int(values[1] / 1000 * screen_h)
    return x, y


def _duration(args: dict[str, object]) -> int:
    """Read a swipe's or long press's duration: a numeric time, else duration, else 1000."""
    for key in ("time", "duration"):
        duration = as_int(args.get(key))
        if duration is not None:
            return duration
    return 1000


def _click(args: dict[str, object], screen_w: int, screen_h: int) -> dict[str, object]:
    x, y = _scale(args.get("coordinate"), screen_w, screen_h)
    return {"action": "tap", "x": x, "y": y}


def _long_press(args: dict[str, object], screen_w: int, screen_h: int) -> dict[str, object]:
    x, y = _scale(args.get("coordinate"), screen_w, screen_h)
    return {"action": "long_press", "x": x, "y": y, "duration": _duration(args)}


_SWIPE_ENDPOINTS: dict[str, Callable[[int, int], tuple[int, int, int, int]]] = {
    "up": lambda w, h: (w // 2, h * 3 // 4, w // 2, h // 4),
    "down": lambda w, h: (w // 2, h // 4, w // 2, h * 3 // 4),
    "left": lambda w, h: (w * 4 // 5, h // 2, w // 5, h // 2),
    "right": lambda w, h: (w // 5, h // 2, w * 4 // 5, h // 2),
}


def _direction_endpoints(
    direction: object, screen_w: int, screen_h: int
) -> tuple[int, int, int, int]:
    if isinstance(direction, str):
        endpoints = _SWIPE_ENDPOINTS.get(direction)
        if endpoints is not None:
            return endpoints(screen_w, screen_h)
    msg = "Swipe requires a supported direction"
    raise ValueError(msg)


def _swipe(args: dict[str, object], screen_w: int, screen_h: int) -> dict[str, object]:
    start, end = args.get("coordinate"), args.get("coordinate2")
    direction = args.get("direction")
    if start is not None and end is not None:
        x1, y1 = _scale(start, screen_w, screen_h)
        x2, y2 = _scale(end, screen_w, screen_h)
    elif direction is not None:
        # A malformed coordinate is an error even when the direction alone would do.
        for coordinate in (start, end):
            if coordinate is not None:
                _scale(coordinate, screen_w, screen_h)
        x1, y1, x2, y2 = _direction_endpoints(direction, screen_w, screen_h)
    else:
        msg = "Swipe requires coordinates or a supported direction"
        raise ValueError(msg)
    return {"action": "swipe", "x1": x1, "y1": y1, "x2": x2, "y2": y2, "duration": _duration(args)}


def _type(args: dict[str, object], _screen_w: int, _screen_h: int) -> dict[str, object]:
    text = args.get("text")
    if isinstance(text, str):
        return {"action": "type", "text": text}
    msg = "Type requires a text string"
    raise ValueError(msg)


_SYSTEM_BUTTONS: dict[str, str] = {"Back": "back", "Home": "home", "Menu": "menu", "Enter": "enter"}


def _system_button(args: dict[str, object], _screen_w: int, _screen_h: int) -> dict[str, object]:
    button = args.get("button", "")
    if isinstance(button, str) and button in _SYSTEM_BUTTONS:
        return {"action": _SYSTEM_BUTTONS[button]}
    msg = "System button must be Back, Home, Menu or Enter"
    raise ValueError(msg)


def _wait(args: dict[str, object], _screen_w: int, _screen_h: int) -> dict[str, object]:
    return {"action": "wait", "time": args.get("time") or args.get("duration") or 3}


def _end(_args: dict[str, object], _screen_w: int, _screen_h: int) -> dict[str, object]:
    return {"action": "end"}


_MOBILE_USE_ACTIONS: dict[str, Callable[[dict[str, object], int, int], dict[str, object]]] = {
    "click": _click,
    "long_press": _long_press,
    "swipe": _swipe,
    "type": _type,
    "system_button": _system_button,
    "wait": _wait,
    "end": _end,
    "terminate": _end,
    "answer": _end,
    "error": _end,
}


def tool_call_to_action(
    args: dict[str, object], screen_w: int = 1080, screen_h: int = 1920
) -> dict[str, object]:
    """Convert mobile_use arguments using the contract in README.md#action-responses."""
    nested = args.get("arguments")
    if isinstance(nested, dict):
        args = {**args, **as_object_dict(nested)}

    action = args.get("action", "")
    # A JSON list or object as the action is unhashable, and must stay a ValueError.
    convert = _MOBILE_USE_ACTIONS.get(action) if isinstance(action, str) else None
    if convert is None:
        msg = f"Unknown action: {action}"
        raise ValueError(msg)
    return convert(args, screen_w, screen_h)


_NEXT_ACTION_PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "element": {
            "type": ["integer", "null"],
            "description": "The number of the listed unexplored element the instruction acts "
            "on, or null when it acts on none of them.",
        },
        "instruction": {
            "type": "string",
            "description": "One atomic action: one tap, one type, one swipe, or one back.",
        },
    },
    "required": ["element", "instruction"],
    "additionalProperties": False,
}


# A self-loop gains an action for every state its screen can be put in, such as each day an
# alarm repeats on, and each of them leads back to the screen the planner already sees.
SELF_LOOP_INSTRUCTIONS_LISTED = 30
# Steps in a row that each stay on one node change only its screen's state and find no new screen.
SATURATING_SELF_LOOP_STEPS = 15


@dataclass(frozen=True, kw_only=True)
class PlanRequest:
    """The screen ``plan_next_action`` plans the next instruction for.

    Attributes:
        node: The node of the screen. The explored edge back to it is its self-loop, which
            the prompt lists without where its actions led.
        explored_edges: List of edge dicts from GraphManager.get_all_edges_from_node(),
            each with its "target", its "actions" and, one entry per action, their
            "instructions" and "target_observations".
        unexplored_elements: List of element dicts not yet interacted with.
        input_status: Runtime hint about whether a text field/keyboard is active.
        back_leaves_app: Whether Back on this screen leaves the app, which the prompt
            then says.
    """

    screenshot_b64: str
    page_description: str
    node: str
    explored_edges: list[dict[str, Any]]
    app_name: str = ""
    unexplored_elements: list[dict[str, Any]] | None = None
    input_status: str = "unknown from screenshot"
    back_leaves_app: bool = False


def _tried_actions(edge: dict[str, Any]) -> list[tuple[str, str]]:
    """Return each action tried on an explored edge with the target observation it
    recorded: by its quoted instruction, or, on an edge that recorded no instruction, by
    the action itself."""
    observations = [as_str(observation, "") for observation in edge.get("target_observations", [])]
    named = [
        (f'"{instruction}"', observation)
        for instruction, observation in zip(edge.get("instructions", []), observations, strict=True)
        if instruction
    ]
    return named or [
        (json.dumps(action), observation)
        for action, observation in zip(edge.get("actions", []), observations, strict=True)
    ]


def _self_loop_lines(edge: dict[str, Any]) -> list[str]:
    """Return the prompt's lines for a self-loop: its ``SELF_LOOP_INSTRUCTIONS_LISTED`` most
    recent distinct instructions, oldest first, and a line counting the others."""
    distinct = [tried for tried, _ in _latest_distinct(_tried_actions(edge))]
    listed = distinct[-SELF_LOOP_INSTRUCTIONS_LISTED:]
    lines = [f"- {tried} → stayed on this screen" for tried in listed]
    if earlier := len(distinct) - len(listed):
        lines.append(f"- {earlier} more tried earlier also stayed on this screen")
    return lines


def _explored_lines(request: PlanRequest) -> list[str]:
    """Return the prompt's lines for the actions tried from the screen: every action of an
    edge to another node with the observation it led to, then the self-loop's lines."""
    lines = [
        f'- {tried} → led to: "{observation}"'
        for edge in request.explored_edges
        if edge["target"] != request.node
        for tried, observation in _tried_actions(edge)
    ]
    for edge in request.explored_edges:
        if edge["target"] == request.node:
            lines.extend(_self_loop_lines(edge))
    return lines


def plan_next_action(
    client: OpenRouter,
    request: PlanRequest,
    *,
    settings: ModelSettings,
) -> tuple[str, int | None]:
    """Use a reasoning VLM to decide WHAT to do next as a natural language instruction.

    This separates the "what" (exploration strategy) from the "how" (coordinate grounding).
    Each instruction is ONE atomic action (one tap, one type, one swipe, or one back).

    Returns:
        ``(instruction, element)``, where ``element`` is the number of the listed
        unexplored element the instruction acts on, or None.
    """

    instructions_str = (
        "\n".join(_explored_lines(request)) or "(none — this screen has not been explored yet)"
    )

    if request.unexplored_elements:
        elem_lines = []
        for number, elem in enumerate(request.unexplored_elements, start=1):
            desc = elem.get("description", "?")
            pos = elem.get("position", "?")
            elem_lines.append(f"{number}. {desc} ({pos})")
        unexplored_section = (
            "Elements on this screen that have NOT been interacted with yet:\n"
            + "\n".join(elem_lines)
            + "\n"
        )
    else:
        unexplored_section = ""

    prompt = NEXT_ACTION_PLAN_PROMPT.format(
        app_name=request.app_name,
        page_description=request.page_description,
        input_status=request.input_status,
        back_leaves_app_line=BACK_LEAVES_APP_LINE if request.back_leaves_app else "",
        explored_instructions=instructions_str,
        unexplored_elements_section=unexplored_section,
    )

    messages: list[ChatMessagesTypedDict] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                build_image_message(request.screenshot_b64),
            ],
        }
    ]
    resp = send_chat(
        client,
        call_type="instruction",
        settings=settings,
        messages=messages,
        response_format=_structured_output("next_action_plan", _NEXT_ACTION_PLAN_SCHEMA),
    )
    raw = message_text(resp)
    try:
        plan = as_object_dict(json.loads(strip_json_fences(raw)))
    except json.JSONDecodeError:
        plan = {}

    instruction = as_str(plan.get("instruction"), raw).strip()
    element = as_int(plan.get("element"))
    logger.info("Planned instruction: %s (element: %s)", instruction, element)
    return instruction, element


ACTION_DESCRIPTION_PROMPT = """\
You are labelling one step of a mobile app exploration graph. The screenshot shows the \
screen before the step. The step performed this action, with coordinates in pixels on a \
{screen_w}x{screen_h} screen, measured from the top-left corner:
{action}

"tap" and "long_press" act on the element at x, y; "swipe" moves from x1, y1 to x2, y2; \
"type" enters its text into the focused field; "back", "home" and "enter" press that \
button; "menu" opens the application options menu. A list is a sequence of such actions.

Write the instruction this step carried out, as one atomic action that names the element \
it acts on. If you refer to text visible on the screenshot, wrap the exact text in quotes, \
and add an anchor such as the element's screen location, icon shape or container. For a \
swipe, give its direction and what it scrolls; for typing, give the text typed.

Example instructions:
- Tap the "Network & internet" menu item near the top of the settings list
- Swipe up on the search results list
- Type 'stewed beef'
- Press back

Example reply:
{{"instruction": "Tap the rounded search bar at the top of the screen"}}"""

_ACTION_DESCRIPTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "instruction": {
            "type": "string",
            "description": "The executed action as one atomic instruction naming its element.",
        },
    },
    "required": ["instruction"],
    "additionalProperties": False,
}


def describe_executed_action(
    client: OpenRouter,
    screenshot_b64: str,
    action: dict[str, Any] | list[dict[str, Any]],
    screen_size: tuple[int, int],
    *,
    settings: ModelSettings,
) -> str | None:
    """Write the instruction for a device action from the screenshot taken before it ran.

    ``screen_size`` is the device's ``(width, height)`` in pixels, which the action's
    coordinates are measured in. Returns None when the reply is truncated, not a JSON
    object or has a blank instruction, or when the request still times out at its
    deadline.
    """
    screen_w, screen_h = screen_size
    prompt = ACTION_DESCRIPTION_PROMPT.format(
        screen_w=screen_w, screen_h=screen_h, action=json.dumps(action)
    )
    try:
        resp = send_chat(
            client,
            call_type="action_description",
            settings=settings,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        build_image_message(screenshot_b64),
                    ],
                }
            ],
            response_format=_structured_output("action_description", _ACTION_DESCRIPTION_SCHEMA),
        )
    except TruncatedReplyError, httpx.TimeoutException:
        return None
    raw = strip_json_fences(message_text(resp))
    try:
        reply = as_object_dict(json.loads(raw))
    except json.JSONDecodeError:
        logger.warning("Failed to parse action description JSON. Raw:\n%s", raw)
        return None
    instruction = as_str(reply.get("instruction"), "").strip()
    logger.info("Described executed action %s as: %s", action, instruction)
    return instruction or None


def _labelled_line(raw: str, label: str) -> str:
    """Return the text after the reply's first ``label:``, up to the end of its line."""
    m = re.search(rf"{re.escape(label)}:\s*(.+?)(?:\n|$)", raw)
    return as_str(m.group(1), "").strip() if m else ""


def _native_arguments(arguments: str) -> dict[str, object] | None:
    """Decode a native tool call's arguments, or return None unless they are a JSON object."""
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return None
    return as_object_dict(parsed) if isinstance(parsed, dict) else None


def _raw_json_arguments(raw: str) -> dict[str, object] | None:
    """Decode a reply that is bare JSON, unwrapping its ``arguments`` like a <tool_call> block."""
    try:
        parsed = json.loads(strip_json_fences(raw))
    except json.JSONDecodeError:
        logger.warning("Failed to parse action agent response. Raw:\n%s", raw)
        return None
    if isinstance(parsed, dict) and "arguments" in parsed:
        parsed = parsed["arguments"]
    return as_object_dict(parsed) if isinstance(parsed, dict) else None


def parse_agent_response(resp: ChatResult) -> tuple[dict[str, object] | None, str, str]:
    """Parse an action agent VLM response.

    Handles both JSON in a <tool_call> block and native tool calling.

    Returns:
        (tool_args, thought, action_desc) or (None, "", "") on failure.
    """
    if not resp.choices:
        return None, "", ""
    raw = message_text(resp)
    tool_calls = resp.choices[0].message.tool_calls
    if not raw and not tool_calls:
        logger.error("Action agent returned empty response (no content, no tool_calls)")
        return None, "", ""
    logger.debug("  Agent raw output: %s", raw)

    thought = _labelled_line(raw, "Thought")
    action_desc = _labelled_line(raw, "Action")

    if tool_calls:
        logger.info("  Agent used native tool calling")
        native_args = _native_arguments(tool_calls[0].function.arguments)
        if native_args is None:
            return None, thought, action_desc
        return native_args, thought, action_desc or str(native_args)

    tool_args = _parse_tool_call(raw)
    if tool_args is not None:
        return as_object_dict(tool_args), thought, action_desc
    return _raw_json_arguments(raw), thought, action_desc


class _MalformedActionOutputError(UnusableOutputError):
    """An action reply that cannot become an action, kept so the retry can show it to the model."""

    def __init__(self, reason: str, reply: str) -> None:
        super().__init__(reason)
        self.reason = reason
        self.reply = reply

    def follow_up(self) -> list[ChatMessagesTypedDict]:
        """Return the failed reply and why it failed, to append to the original messages."""
        retry: ChatMessagesTypedDict = {
            "role": "user",
            "content": NEXT_ACTION_RETRY_PROMPT.format(reason=self.reason),
        }
        if not self.reply:
            # Nothing to show, and some providers reject an empty assistant message.
            return [retry]
        return [{"role": "assistant", "content": self.reply}, retry]


def _unparsed_reason(raw: str) -> str:
    """Say why a reply yielded no tool-call arguments."""
    block = re.search(r"<tool_call>(.*?)(?:</tool_call>|\Z)", raw, re.DOTALL)
    if block and "<function=" in block.group(1):
        return "the <tool_call> block used <function=...> tags instead of JSON"
    return "it had no <tool_call> block with valid JSON"


def _read_action(resp: ChatResult, screen_w: int, screen_h: int) -> tuple[dict[str, object], str]:
    """Turn an action reply into a device action and its history entry."""
    raw = message_text(resp)
    try:
        tool_args, thought, action_desc = parse_agent_response(resp)
    except (TypeError, ValueError, AttributeError, IndexError, OverflowError) as exc:
        raise _MalformedActionOutputError(_unparsed_reason(raw), raw) from exc
    if tool_args is None:
        raise _MalformedActionOutputError(_unparsed_reason(raw), raw)
    try:
        action = tool_call_to_action(tool_args, screen_w=screen_w, screen_h=screen_h)
    except (TypeError, ValueError, AttributeError, IndexError, OverflowError) as exc:
        reason = f"the mobile_use arguments were invalid: {exc}"
        raise _MalformedActionOutputError(reason, raw) from exc
    history_entry = f"{thought} | {action_desc}" if thought else (action_desc or str(action))
    logger.info("  Agent action: %s → device action: %s", tool_args.get("action"), action)
    return as_object_dict(action), history_entry


@dataclass(frozen=True, kw_only=True)
class ActionRequest:
    """The step ``predict_next_action`` asks the action agent to take.

    Attributes:
        instruction: The high-level task instruction.
        screen_w: Actual device screen width in pixels.
        screen_h: Actual device screen height in pixels.
        action_history: List of NL descriptions of previous actions taken.
        overall_task: Optional real task for supplementary context.
    """

    screenshot_b64: str
    instruction: str
    screen_w: int = 1080
    screen_h: int = 1920
    action_history: list[str] | None = None
    overall_task: str = ""


def predict_next_action(
    client: OpenRouter,
    request: ActionRequest,
    *,
    settings: ModelSettings,
) -> tuple[dict[str, Any], str]:
    """Call the action agent VLM for one step.

    The agent sees the instruction, action history, and current screenshot,
    and decides the next action (or "end" if the task is complete).

    Returns:
        (device_action, history_entry):
        - device_action: device action dict (may be {"action": "end"})
        - history_entry: NL description for the agent's action history
    """

    system_prompt = NEXT_ACTION_SYSTEM_PROMPT.format()

    if request.action_history:
        history_text = "\n".join(f"Step {i + 1}: {h}" for i, h in enumerate(request.action_history))
        history_section = f"Task progress (actions already taken):\n{history_text}\n\n"
    else:
        history_section = ""

    if request.overall_task and request.overall_task != request.instruction:
        overall_task_section = f"(Overall goal for context: {request.overall_task})\n\n"
    else:
        overall_task_section = ""

    user_prompt = NEXT_ACTION_USER_PROMPT.format(
        instruction=request.instruction,
        overall_task_section=overall_task_section,
        history_section=history_section,
    )

    messages: list[ChatMessagesTypedDict] = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                build_image_message(request.screenshot_b64),
            ],
        },
    ]

    failure: _MalformedActionOutputError | None = None

    def request_action() -> tuple[dict[str, object], str]:
        nonlocal failure
        # At temperature 0 the identical request can repeat the same mistake, so the retry
        # says what was wrong. A failed generation or a response the SDK cannot validate is
        # sent again unchanged, since the model did nothing wrong. One request per attempt
        # keeps every kind of failure within the one re-ask.
        retry: list[ChatMessagesTypedDict] = [] if failure is None else failure.follow_up()
        resp = _send_chat_once(
            client, call_type="action", settings=settings, messages=[*messages, *retry]
        )
        try:
            return _read_action(resp, request.screen_w, request.screen_h)
        except _MalformedActionOutputError as exc:
            failure = exc
            raise

    try:
        return call_with_retry("[API] action response", request_action)
    except _MalformedActionOutputError, TruncatedReplyError, httpx.TimeoutException:
        logger.warning("Action agent gave no usable action, falling back to 'wait'", exc_info=True)
        return {"action": "wait", "time": 1}, "Failed to parse action response -> wait"
