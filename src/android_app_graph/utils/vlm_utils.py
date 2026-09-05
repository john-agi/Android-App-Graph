"""VLM utilities for page description, detail extraction, and action prediction."""

from __future__ import annotations

import base64
import io
import json
import logging
import math
import re
from typing import TYPE_CHECKING, Any

import httpx
from openai import OpenAI, OpenAIError
from PIL import Image

from android_app_graph.payloads import as_float_list, as_int, as_str

if TYPE_CHECKING:
    from openai.types import CompletionUsage
    from openai.types.chat import (
        ChatCompletion,
        ChatCompletionContentPartImageParam,
        ChatCompletionMessageParam,
    )
    from openai.types.create_embedding_response import Usage as EmbeddingUsage

logger = logging.getLogger(__name__)

MAX_PIXELS = 1_000_000


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Return the cosine similarity of ``a`` and ``b``, clamped to ``[-1.0, 1.0]``.

    A zero-norm vector has no direction, so the similarity is defined as ``0.0``
    rather than raising a division error. Subnormal floats can otherwise push the
    unclamped ratio slightly outside ``[-1.0, 1.0]`` (a violation of
    Cauchy-Schwarz that is a floating-point rounding artifact, not a real
    similarity), so the result is clamped before it is returned.
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if not norm_a or not norm_b:
        return 0.0
    return max(-1.0, min(1.0, dot / (norm_a * norm_b)))


# ---------------------------------------------------------------------------
# Token usage tracker
# ---------------------------------------------------------------------------


class TokenTracker:
    """Accumulates token usage across all API calls, broken down by call type."""

    def __init__(self) -> None:
        self._usage: dict[str, dict[str, int]] = {}

    def record(
        self,
        call_type: str,
        model: str,
        usage: CompletionUsage | EmbeddingUsage | None,
    ) -> None:
        """Record token usage from an API response.

        Args:
            call_type: e.g. "page_description", "page_detail", "embedding",
                       "instruction", "action"
            model: The model name used.
            usage: The response.usage object (has prompt_tokens, completion_tokens, total_tokens).
        """
        if usage is None:
            return
        key = f"{call_type}:{model}"
        if key not in self._usage:
            self._usage[key] = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "calls": 0,
            }
        self._usage[key]["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
        self._usage[key]["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0
        self._usage[key]["total_tokens"] += getattr(usage, "total_tokens", 0) or 0
        self._usage[key]["calls"] += 1

    def snapshot(self) -> dict[str, int]:
        """Return current totals as a flat dict."""
        total = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "calls": 0,
        }
        for u in self._usage.values():
            for k in total:
                total[k] += u[k]
        return total

    def snapshot_by_type(self) -> dict[str, int]:
        """Return total_tokens per call_type (aggregated across models).

        E.g. {"page_description": 500, "page_detail": 1200, "embedding": 20, ...}
        """
        by_type: dict[str, int] = {}
        for key, u in self._usage.items():
            call_type = key.split(":")[0]
            by_type[call_type] = by_type.get(call_type, 0) + u["total_tokens"]
        return by_type

    def summary(self) -> dict[str, dict[str, int]]:
        """Return accumulated usage by call_type:model."""
        return dict(self._usage)

    def print_summary(self) -> None:
        """Log a formatted summary table of all token usage."""
        if not self._usage:
            logger.info("No API calls recorded.")
            return

        total_prompt = 0
        total_completion = 0
        total_all = 0
        total_calls = 0

        lines = ["", "=" * 80, "TOKEN USAGE SUMMARY", "=" * 80]
        lines.append(
            f"{'Call Type':<40} {'Calls':>6} {'Prompt':>10} {'Completion':>10} {'Total':>10}"
        )
        lines.append("-" * 80)

        for key, u in sorted(self._usage.items()):
            lines.append(
                f"{key:<40} {u['calls']:>6} {u['prompt_tokens']:>10} "
                f"{u['completion_tokens']:>10} {u['total_tokens']:>10}"
            )
            total_prompt += u["prompt_tokens"]
            total_completion += u["completion_tokens"]
            total_all += u["total_tokens"]
            total_calls += u["calls"]

        lines.append("-" * 80)
        lines.append(
            f"{'TOTAL':<40} {total_calls:>6} {total_prompt:>10} "
            f"{total_completion:>10} {total_all:>10}"
        )
        lines.append("=" * 80)

        logger.info("\n".join(lines))


# Global tracker instance
token_tracker = TokenTracker()


def _resize_screenshot(
    screenshot_b64: str, max_pixels: int = MAX_PIXELS
) -> tuple[str, int, int, float]:
    """Resize a base64 screenshot so total pixels <= max_pixels, preserving aspect ratio.

    Returns:
        (resized_b64, new_width, new_height, scale_factor)
        scale_factor is orig/new, i.e. multiply resized coords by this to get original coords.
        If no resize needed, scale_factor is 1.0.
    """
    img = Image.open(io.BytesIO(base64.b64decode(screenshot_b64)))
    w, h = img.size
    total = w * h
    if total <= max_pixels:
        return screenshot_b64, w, h, 1.0

    scale = math.sqrt(max_pixels / total)
    new_w = int(w * scale)
    new_h = int(h * scale)
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    logger.debug(
        "Screenshot resized: %dx%d -> %dx%d (scale_back=%.3f)",
        w,
        h,
        new_w,
        new_h,
        1.0 / scale,
    )

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii"), new_w, new_h, 1.0 / scale


def strip_json_fences(text: str) -> str:
    """Remove ```json ... ``` markdown fences if present."""
    text = text.strip()
    m = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if m:
        return as_str(m.group(1), text).strip()
    return text


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

NORMALIZE_EDGE_PROMPT = """\
You are helping build a reusable navigation graph for a mobile app. Your job is to \
normalize action instructions and their resulting observations into TEMPLATES by \
replacing specific values with named placeholders.

Given one or more concrete action instructions (and optionally their target observations), \
produce a single normalized template that captures the common pattern.

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

Return a JSON object:
{{
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
  Output: {{"instruction_template": "Search for {{query}}", "observation_template": "search results for {{query}}", "is_template": true, "param_names": ["query"], "examples": [{{"instruction": "Search for egg", "params": {{"query": "egg"}}}}, {{"instruction": "Search for milk", "params": {{"query": "milk"}}}}]}}

- Input: ["Tap the Settings icon"]
  Output: {{"instruction_template": "Tap the Settings icon", "observation_template": "", "is_template": false, "param_names": [], "examples": []}}

- Input: ["Type 'Waterloo' in the start location field", "Type '221B Baker Street' in the start location field"]
  Output: {{"instruction_template": "Type '{{location}}' in the start location field", "observation_template": "start location field shows {{location}}", "is_template": true, "param_names": ["location"], "examples": [{{"instruction": "Type 'Waterloo' in the start location field", "params": {{"location": "Waterloo"}}}}, {{"instruction": "Type '221B Baker Street' in the start location field", "params": {{"location": "221B Baker Street"}}}}]}}"""

PAGE_DESCRIBE_AND_STATE_PROMPT = """\
You are an expert at analyzing mobile app screenshots. You are helping build a navigation \
graph of an app where each node is a distinct SCREEN TYPE (template).

{existing_nodes_section}

{existing_keys_section}

Analyze this screenshot and return a JSON object with exactly three keys:

1. "page_description": The screen TYPE in under 10 words. This is a reusable label for the \
screen template — it must NOT contain any user-specific content (search queries, item names, \
addresses, dates, prices, usernames, etc.). Think of it as the screen's class name. \
Good: "search results page", "route planning page", "location search page" \
Bad: "search results for Big Ben", "route from home to work", "John's profile page" \
All user-specific data belongs in "state", never in "page_description". \
Always describe what you actually see. \
{description_hint}

2. "state": A FLAT JSON object capturing ONLY the DYNAMIC state of this screen — values \
that change through user interaction. This is where ALL user-specific content goes.

For the "state" object, focus on:
- "first_visible_item": the first list/menu item visible on screen (scroll position)
- "last_visible_item": the last list/menu item visible on screen
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
{{"description": "Account tab", "position": "bottom navigation bar"}}]}}"""

NODE_VERIFY_PROMPT = """\
You are comparing two mobile app screenshots to decide if they are the SAME screen or DIFFERENT screens.

Both screens were given this description: "{existing_description}"

Two screenshots are the SAME screen if they serve the same purpose and a user would consider \
them the same page — even if the content (list items, search results, text in fields) differs. \
For example, a search results page showing different queries is still the same screen.

Two screenshots are DIFFERENT screens if they serve different purposes in the app's workflow, \
even if they look visually similar. Pay close attention to:
- Screen titles, headers, or labels (e.g. "Choose start location" vs "Choose destination")
- Which input field is active or being filled
- The stage in a multi-step flow (e.g. step 1 vs step 2 of a booking)
- Different form sections or tabs that change the screen's function
- Any dropdown menu, overflow menu, popup menu, context menu, sort/filter menu, \
or foreground menu panel that appears or disappears. A screen with such a menu \
open is a DIFFERENT screen from the same underlying page with the menu closed.
- A search page and a search result list page is considered different usually, though sometimes the search page also provide some suggestions.

Answer with ONLY a JSON object:

If they are the SAME screen:
{{"same": true, "reason": "brief explanation"}}

If they are DIFFERENT screens, you must also provide slightly more specific descriptions that \
distinguish them from each other. Usually adding one or two extra distinguishing words is enough. \
Keep the descriptions concise (under 10 words each) and clearly differentiate the two screens. \
Exception: if a dropdown/overflow/menu panel is visible, the description may be a bit longer \
if needed, but it MUST explicitly name what menu is shown (e.g. "results page with sort menu open", \
"home page with overflow menu open"):
{{"same": false, "reason": "brief explanation", \
"existing_description": "more specific description for the first screenshot", \
"new_description": "more specific description for the second screenshot"}}"""

NEXT_ACTION_PLAN_PROMPT = """\
You are an exploration planner for a mobile app. Your goal is to systematically explore \
the app to discover as many DISTINCT screens and functionalities as possible.

App name: {app_name}
Current screen description: "{page_description}"
Current input status: {input_status}

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

Reply with ONLY the instruction as a plain text string — a single ATOMIC action. \
Each instruction = one tap, one type, one swipe, or one back. Never combine steps.

Examples:
- Tap the "Network & internet" menu item near the top of the settings list
- Tap the rounded search bar at the top of the screen
- Type 'stewed beef'
- Press Enter
- Tap the first search result card in the center list
- Select the "Price: Low to High" option from the sort menu
- Scroll down to reveal more settings
- Go back to the previous screen

Reply with ONLY the JSON, nothing else."""

NEXT_ACTION_SYSTEM_PROMPT = """\
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{{"type": "function", "function": {{"name": "mobile_use", "description": "Use a touchscreen to interact with a mobile device, and take screenshots.\\n* This is an interface to a mobile device with touchscreen. You can perform actions like clicking, typing, swiping, etc.\\n* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.\\n* The screen's resolution is {screen_w}x{screen_h}.\\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.", "parameters": {{"properties": {{"action": {{"description": "The action to perform. The available actions are:\\n* `click`: Click the point on the screen with coordinate (x, y).\\n* `long_press`: Press the point on the screen with coordinate (x, y) for specified seconds.\\n* `swipe`: Swipe from the starting point with coordinate (x, y) to the end point with coordinates2 (x2, y2).\\n* `type`: Input the specified text into the activated input box.\\n* `system_button`: Press the system button.\\n* `wait`: Wait specified seconds for the change to happen.\\n* `end`: Signal that the current task is completed. Use this when the task goal has been achieved.", "enum": ["click", "long_press", "swipe", "type", "system_button", "wait", "end"], "type": "string"}}, "coordinate": {{"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=click`, `action=long_press`, and `action=swipe`.", "type": "array"}}, "coordinate2": {{"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=swipe`.", "type": "array"}}, "text": {{"description": "Required only by `action=type`.", "type": "string"}}, "time": {{"description": "The seconds to wait. Required only by `action=long_press` and `action=wait`.", "type": "number"}}, "button": {{"description": "Back means returning to the previous interface, Home means returning to the desktop, Menu means opening the application background menu, and Enter means pressing the enter. Required only by `action=system_button`", "enum": ["Back", "Home", "Menu", "Enter"], "type": "string"}}}}, "required": ["action"], "type": "object"}}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>

# Response format

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

DEFAULT_PAGE_DETAIL_MODEL = "gpt-5.4"
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-2-preview"
DEFAULT_INSTRUCTION_MODEL = "gpt-5.4"
DEFAULT_ACTION_MODEL = "qwen3.5-plus-2026-02-15"

# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------


def _message_text(resp: ChatCompletion) -> str:
    """Return the assistant text of a completion, stripped.

    ``message.content`` is ``None`` when the model answered with tool calls or
    with nothing at all; the callers all treat unparsable text as a parse
    failure and fall back, so an empty string is the honest value here.
    """
    content = resp.choices[0].message.content
    return content.strip() if content is not None else ""


def _build_image_message(screenshot_b64: str) -> ChatCompletionContentPartImageParam:
    """Build an OpenAI image_url message part from a base64 screenshot."""
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/png;base64,{screenshot_b64}",
        },
    }


def describe_page_and_state(
    client: OpenAI,
    screenshot_b64: str,
    existing_nodes: list[str] | None = None,
    existing_keys: list[str] | None = None,
    model: str = DEFAULT_PAGE_DETAIL_MODEL,
) -> tuple[str, dict[str, Any], list[dict[str, str]]]:
    """Call VLM once to get a fresh page description, dynamic state, and interactable elements.

    The model always writes its own description based on what it sees. Existing
    node descriptions are provided as context so the model can disambiguate
    similar-looking screens (e.g. "start location search" vs "end location search").

    Args:
        existing_nodes: Descriptions of nodes already in the graph (same package).
        existing_keys: State keys already seen across same-package nodes.

    Returns:
        (page_description, state_dict, elements_list)
        where elements_list is a list of {"description": str, "position": str} dicts.
    """
    screenshot_b64, *_ = _resize_screenshot(screenshot_b64)

    # Build existing nodes section
    if existing_nodes:
        nodes_list = "\n".join(f'  - "{d}"' for d in existing_nodes)
        existing_nodes_section = (
            f"The following screens have already been discovered in this app:\n{nodes_list}"
        )
        description_hint = (
            "If this screen is clearly the same type as one in the list, output the same "
            "description. If it is similar but serves a different purpose (e.g. different "
            "step in a flow, different input field focused), write a description that "
            "clearly distinguishes it. Pay attention to screen titles, headers, labels, "
            "which field is active, and what stage of a multi-step flow this is."
        )
    else:
        existing_nodes_section = "No screens have been discovered yet in this app."
        description_hint = "Focus on the screen's purpose. Use precise words."

    # Build existing keys section
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
        existing_nodes_section=existing_nodes_section,
        existing_keys_section=existing_keys_section,
        description_hint=description_hint,
    )
    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                _build_image_message(screenshot_b64),
            ],
        }
    ]
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    token_tracker.record("page_describe_and_state", model, resp.usage)
    raw = strip_json_fences(_message_text(resp))
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Failed to parse page+state JSON. Raw:\n%s", raw)
        return raw[:100], {}, []

    page_description = as_str(result.get("page_description"), "unknown screen")
    state = result.get("state", {})
    if not isinstance(state, dict):
        state = {}
    elements = result.get("elements", [])
    if not isinstance(elements, list):
        elements = []
    logger.debug("Page description: %s", page_description)
    logger.debug("Page state: %s", state)
    logger.debug("Elements: %d found", len(elements))
    return page_description, state, elements


def verify_same_node(
    client: OpenAI,
    screenshot_new_b64: str,
    screenshot_existing_b64: str,
    existing_description: str,
    model: str = DEFAULT_PAGE_DETAIL_MODEL,
) -> dict[str, Any]:
    """Compare two screenshots to decide if they are the same screen.

    If they are different, the verifier also provides more specific descriptions
    for both the existing node and the new screen, so vague descriptions like
    "location search page" get refined to "start location search page" and
    "end location search page".

    Returns a dict with keys:
        - "same": bool
        - "reason": str
        - "existing_description": str (only when same=False) — refined name for existing node
        - "new_description": str (only when same=False) — refined name for new screen
    """
    screenshot_new_b64, *_ = _resize_screenshot(screenshot_new_b64)
    screenshot_existing_b64, *_ = _resize_screenshot(screenshot_existing_b64)

    prompt = NODE_VERIFY_PROMPT.format(existing_description=existing_description)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "text", "text": "First screenshot (existing node):"},
                    _build_image_message(screenshot_existing_b64),
                    {"type": "text", "text": "Second screenshot (new screen):"},
                    _build_image_message(screenshot_new_b64),
                ],
            }
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    token_tracker.record("node_verify", model, resp.usage)
    raw = strip_json_fences(_message_text(resp))
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Failed to parse verify JSON. Raw:\n%s", raw)
        return {"same": False, "reason": "parse error"}

    logger.debug("Node verify: %s", result)
    return result


def normalize_edge(
    client: OpenAI,
    instructions: list[str],
    target_observations: list[str] | None = None,
    model: str = DEFAULT_PAGE_DETAIL_MODEL,
) -> dict[str, Any]:
    """Normalize edge instructions into a reusable template.

    Takes one or more concrete instructions (e.g. "search for egg", "search for milk")
    and produces a template with placeholders (e.g. "search for {query}").

    Args:
        instructions: List of concrete instruction strings to normalize.
        target_observations: Optional list of corresponding target observations.

    Returns a dict with keys:
        - instruction_template: str — the normalized template
        - observation_template: str — normalized observation (may be empty)
        - is_template: bool — whether any placeholders were inserted
        - param_names: list[str] — placeholder names
        - examples: list[dict] — original instructions with extracted param values
    """
    input_text = "Instructions to normalize:\n"
    for i, inst in enumerate(instructions):
        obs = target_observations[i] if target_observations and i < len(target_observations) else ""
        input_text += f'- Instruction: "{inst}"'
        if obs:
            input_text += f' → Observation: "{obs}"'
        input_text += "\n"

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": f"{NORMALIZE_EDGE_PROMPT}\n\n{input_text}",
            }
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    token_tracker.record("normalize_edge", model, resp.usage)
    raw = strip_json_fences(_message_text(resp))
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Failed to parse normalize JSON. Raw:\n%s", raw)
        return {
            "instruction_template": instructions[0],
            "observation_template": target_observations[0] if target_observations else "",
            "is_template": False,
            "param_names": [],
            "examples": [],
        }

    logger.debug("Normalize edge: %s", result)
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
look similar, such as start search vs destination search, opened menu vs closed page, or \
different selected tabs.

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


def audit_merge_nodes(
    client: OpenAI,
    graph_summary: str,
    app_name: str = "",
    model: str = DEFAULT_PAGE_DETAIL_MODEL,
) -> dict[str, Any]:
    """Ask a model only for duplicate-node merge candidates."""
    prompt = NODE_MERGE_AUDIT_PROMPT.format(
        app_name=app_name,
        graph_description=graph_summary,
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    token_tracker.record("node_merge_audit", model, resp.usage)
    raw = strip_json_fences((resp.choices[0].message.content or "").strip())
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

Return ONLY a JSON object:
{{
  "node_id": "candidate node_id",
  "reason": "short reason"
}}"""


def select_exploration_target(
    client: OpenAI,
    app_name: str,
    candidates: list[dict[str, Any]],
    model: str = DEFAULT_PAGE_DETAIL_MODEL,
) -> str | None:
    """Select a node for periodic live exploration coverage balancing."""
    if not candidates:
        return None

    candidate_lines = []
    valid_ids = {c.get("node_id", "") for c in candidates}
    for i, candidate in enumerate(candidates, 1):
        unexplored_desc = candidate.get("unexplored_element_descriptions", [])
        unexplored_text = ", ".join(d for d in unexplored_desc if d) or "none listed"
        candidate_lines.append(
            f"{i}. node_id={candidate.get('node_id', '')} | "
            f'description="{candidate.get("page_description", "")}" | '
            f"visits={candidate.get('visit_count', 0)} | "
            f"out_edges={candidate.get('out_degree', 0)} | "
            f"unexplored={candidate.get('unexplored_elements', 0)}/"
            f"{candidate.get('total_elements', 0)} | "
            f"score={candidate.get('score', 0)} | "
            f"unexplored_elements=[{unexplored_text}]"
        )

    prompt = EXPLORATION_TARGET_PROMPT.format(
        app_name=app_name,
        candidates="\n".join(candidate_lines),
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    token_tracker.record("exploration_target", model, resp.usage)

    raw = strip_json_fences(_message_text(resp))
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
    client: OpenAI,
    graph_summary: str,
    app_name: str = "",
    model: str = DEFAULT_PAGE_DETAIL_MODEL,
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
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    token_tracker.record("graph_audit", model, resp.usage)
    choice = resp.choices[0]
    finish_reason = getattr(choice, "finish_reason", "")
    raw_original = (choice.message.content or "").strip()
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


def get_embedding(client: OpenAI, text: str, model: str = DEFAULT_EMBEDDING_MODEL) -> list[float]:
    """Get an embedding vector for a text string."""
    resp = client.embeddings.create(model=model, input=text)
    token_tracker.record("embedding", model, resp.usage)
    return resp.data[0].embedding


def get_image_embedding(
    client: OpenAI, screenshot_b64: str, model: str = DEFAULT_EMBEDDING_MODEL
) -> list[float]:
    """Get an embedding vector for a screenshot.

    Multimodal embedding providers expose image inputs through different
    OpenAI-compatible shapes. Try common OpenAI image_url and Gemini content
    shapes before surfacing the provider error.
    """
    data_url = f"data:image/png;base64,{screenshot_b64}"
    # Provider-specific shapes that the SDK's own `input` type does not describe.
    attempts: list[Any] = [
        [{"type": "image_url", "image_url": {"url": data_url}}],
        {"type": "image_url", "image_url": {"url": data_url}},
        [{"mime_type": "image/png", "data": screenshot_b64}],
        {"mime_type": "image/png", "data": screenshot_b64},
        [{"parts": [{"inline_data": {"mime_type": "image/png", "data": screenshot_b64}}]}],
        {"parts": [{"inline_data": {"mime_type": "image/png", "data": screenshot_b64}}]},
        [
            {
                "content": {
                    "parts": [{"inline_data": {"mime_type": "image/png", "data": screenshot_b64}}]
                }
            }
        ],
        {
            "content": {
                "parts": [{"inline_data": {"mime_type": "image/png", "data": screenshot_b64}}]
            }
        },
    ]

    errors = []
    for input_payload in attempts:
        try:
            resp = client.embeddings.create(model=model, input=input_payload)
            token_tracker.record("image_embedding", model, resp.usage)
            return resp.data[0].embedding
        except (OpenAIError, TypeError, ValueError) as exc:
            errors.append(f"{type(exc).__name__}: {exc}")

    raise RuntimeError(
        "Image embedding request failed for all known payload formats: " + " | ".join(errors)
    )


def get_gemini_native_image_embedding(
    api_key: str,
    screenshot_b64: str,
    model: str = "gemini-embedding-2",
    base_url: str = "https://generativelanguage.googleapis.com/v1beta",
) -> list[float]:
    """Get an image embedding through Gemini's native REST API."""
    base_url = base_url.rstrip("/")
    model_name = model if model.startswith("models/") else f"models/{model}"
    url = f"{base_url}/{model_name}:embedContent"
    payload = {
        "content": {
            "parts": [
                {
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": screenshot_b64,
                    },
                }
            ],
        },
    }
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }
    resp = httpx.post(url, headers=headers, json=payload, timeout=60)
    try:
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(f"{resp.status_code} {resp.text}") from exc

    data = resp.json()
    values = data.get("embedding", {}).get("values")
    if not values and data.get("embeddings"):
        values = data["embeddings"][0].get("values")
    embedding = as_float_list(values)
    if not embedding:
        raise RuntimeError(
            f"Gemini native image embedding response missing embedding values: {data}"
        )
    return embedding


def _parse_tool_call(raw: str) -> dict[str, Any] | None:
    """Extract the JSON arguments from a <tool_call>...</tool_call> block."""
    m = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", raw, re.DOTALL)
    inner = m.group(1).strip() if m else ""

    if not inner:
        start = raw.find("<tool_call>")
        if start == -1:
            return None
        remainder = raw[start + len("<tool_call>") :].lstrip()
        decoder = json.JSONDecoder()
        try:
            parsed, _end = decoder.raw_decode(remainder)
        except json.JSONDecodeError:
            return None
        if "arguments" in parsed:
            return parsed["arguments"]
        return parsed

    inner = strip_json_fences(inner)
    try:
        parsed = json.loads(inner)
    except json.JSONDecodeError:
        return None
    # Handle {"name": "mobile_use", "arguments": {...}}
    if "arguments" in parsed:
        return parsed["arguments"]
    return parsed


def _tool_call_to_aitk(
    args: dict[str, Any], screen_w: int = 1080, screen_h: int = 1920
) -> dict[str, Any]:
    """Convert a mobile_use tool-call argument dict to AITK action format.

    Follows the same conversion logic as qwen3_vl.py's to_device() and the
    AITK UIKobeTranslator._convert_action().

    The VLM outputs coordinates in a normalized 0–1000 range.
    We convert them to actual device pixels: x_pixel = x_norm / 1000 * screen_w.
    """
    if isinstance(args.get("arguments"), dict):
        args = {**args, **args["arguments"]}

    action = args.get("action", "")

    def _scale(coord: list[Any]) -> tuple[int, int]:
        x = int(coord[0] / 1000 * screen_w)
        y = int(coord[1] / 1000 * screen_h)
        return x, y

    def _get_duration() -> int:
        for key in ("time", "duration"):
            duration = as_int(args.get(key))
            if duration is not None:
                return duration
        return 1000

    if action == "click":
        coord = args.get("coordinate")
        if not coord or len(coord) < 2:
            return {"action": "end"}
        x, y = _scale(coord)
        return {"action": "tap", "x": x, "y": y}

    if action == "long_press":
        coord = args.get("coordinate")
        if not coord or len(coord) < 2:
            return {"action": "end"}
        x, y = _scale(coord)
        return {"action": "long_press", "x": x, "y": y, "duration": _get_duration()}

    if action == "swipe":
        if args.get("coordinate") is not None and args.get("coordinate2") is not None:
            x1, y1 = _scale(args["coordinate"])
            x2, y2 = _scale(args["coordinate2"])
        elif args.get("direction") is not None:
            direction = args["direction"]
            if direction == "up":
                x1, y1 = screen_w // 2, screen_h * 3 // 4
                x2, y2 = screen_w // 2, screen_h // 4
            elif direction == "down":
                x1, y1 = screen_w // 2, screen_h // 4
                x2, y2 = screen_w // 2, screen_h * 3 // 4
            elif direction == "left":
                x1, y1 = screen_w * 4 // 5, screen_h // 2
                x2, y2 = screen_w // 5, screen_h // 2
            elif direction == "right":
                x1, y1 = screen_w // 5, screen_h // 2
                x2, y2 = screen_w * 4 // 5, screen_h // 2
            else:
                return {"action": "end"}
        else:
            return {"action": "end"}
        return {
            "action": "swipe",
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "duration": _get_duration(),
        }

    if action == "type":
        if "text" in args:
            return {"action": "type", "text": args["text"]}
        return {"action": "end"}

    if action == "system_button":
        button = args.get("button", "")
        if button == "Back":
            return {"action": "back"}
        if button == "Home":
            return {"action": "home"}
        if button == "Enter":
            return {"action": "enter"}
        return {"action": "end"}

    if action == "wait":
        t = args.get("time") or args.get("duration") or 3
        return {"action": "wait", "time": t}

    if action in ("end", "terminate"):
        return {"action": "end"}

    if action == "answer":
        return {"action": "end"}

    if action == "error":
        return {"action": "end"}

    # Unknown action
    logger.warning("Unknown action: %s", action)
    return {"action": "end"}


def plan_next_action(
    client: OpenAI,
    screenshot_b64: str,
    page_description: str,
    explored_edges: list[dict[str, Any]],
    app_name: str = "",
    unexplored_elements: list[dict[str, Any]] | None = None,
    input_status: str = "unknown from screenshot",
    model: str = DEFAULT_INSTRUCTION_MODEL,
) -> str:
    """Use a reasoning VLM to decide WHAT to do next as a natural language instruction.

    This separates the "what" (exploration strategy) from the "how" (coordinate grounding).
    Each instruction is ONE atomic action (one tap, one type, one swipe, or one back).

    Args:
        explored_edges: List of edge dicts from GraphManager.get_all_edges_from_node(),
            each with "target_description", "instructions", "actions", "visit_count".
        unexplored_elements: List of element dicts not yet interacted with.
        input_status: Runtime hint about whether a text field/keyboard is active.

    Returns:
        A plain text instruction string.
    """
    screenshot_b64, *_ = _resize_screenshot(screenshot_b64)

    # Format explored instructions with their outcomes
    if explored_edges:
        lines = []
        for edge in explored_edges:
            target_desc = edge.get("target_description", "unknown")
            instructions = edge.get("instructions", [])
            if instructions:
                for instr in instructions:
                    lines.append(f'- "{instr}" → led to: "{target_desc}"')
            else:
                # Fallback for edges without instruction metadata
                for action in edge.get("actions", []):
                    lines.append(f'- {json.dumps(action)} → led to: "{target_desc}"')
        instructions_str = (
            "\n".join(lines) if lines else "(none — this screen has not been explored yet)"
        )
    else:
        instructions_str = "(none — this screen has not been explored yet)"

    # Format unexplored elements section
    if unexplored_elements:
        elem_lines = []
        for elem in unexplored_elements:
            desc = elem.get("description", "?")
            pos = elem.get("position", "?")
            elem_lines.append(f"- {desc} ({pos})")
        unexplored_section = (
            "Elements on this screen that have NOT been interacted with yet:\n"
            + "\n".join(elem_lines)
            + "\n"
        )
    else:
        unexplored_section = ""

    prompt = NEXT_ACTION_PLAN_PROMPT.format(
        app_name=app_name,
        page_description=page_description,
        input_status=input_status,
        explored_instructions=instructions_str,
        unexplored_elements_section=unexplored_section,
    )

    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                _build_image_message(screenshot_b64),
            ],
        }
    ]
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
    )

    token_tracker.record("instruction", model, resp.usage)
    raw = _message_text(resp)

    # The planner now returns plain text; try to extract from JSON if it still outputs one
    raw_clean = strip_json_fences(raw)
    try:
        result = json.loads(raw_clean)
        if isinstance(result, dict):
            instruction = as_str(result.get("instruction"), raw)
        else:
            instruction = str(result)
    except json.JSONDecodeError:
        instruction = raw

    # Strip surrounding quotes if present
    instruction = instruction.strip().strip('"').strip("'")

    logger.info("Planned instruction: %s", instruction)
    return instruction


def _parse_agent_response(resp: ChatCompletion) -> tuple[dict[str, Any] | None, str, str]:
    """Parse an action agent VLM response.

    Handles both XML <tool_call> responses and native tool calling.

    Returns:
        (tool_args, thought, action_desc) or (None, "", "") on failure.
    """
    raw_content = resp.choices[0].message.content
    if raw_content is None:
        # Model returned no text content — check for native tool calls
        tool_calls = resp.choices[0].message.tool_calls
        if tool_calls:
            logger.info("  Agent used native tool calling")
            tool_call = tool_calls[0]
            if tool_call.type != "function":
                logger.error("Action agent returned an unsupported %s tool call", tool_call.type)
                return None, "", ""
            try:
                tool_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                return None, "", ""
            return tool_args, "", str(tool_args)
        logger.error("Action agent returned empty response (no content, no tool_calls)")
        return None, "", ""

    raw = raw_content.strip()
    logger.debug("  Agent raw output: %s", raw)

    # Extract Thought and Action lines
    thought = ""
    thought_m = re.search(r"Thought:\s*(.+?)(?:\n|$)", raw)
    if thought_m:
        thought = thought_m.group(1).strip()

    action_desc = ""
    action_m = re.search(r"Action:\s*(.+?)(?:\n|$)", raw)
    if action_m:
        action_desc = action_m.group(1).strip()

    # Parse <tool_call> block
    tool_args = _parse_tool_call(raw)
    if tool_args is not None:
        return tool_args, thought, action_desc

    # Fallback: try parsing whole response as JSON
    raw_json = strip_json_fences(raw)
    try:
        parsed = json.loads(raw_json)
        if "arguments" in parsed:
            parsed = parsed["arguments"]
        return parsed, thought, action_desc
    except json.JSONDecodeError:
        logger.warning("Failed to parse action agent response. Raw:\n%s", raw)
        return None, thought, action_desc


def predict_next_action(
    client: OpenAI,
    screenshot_b64: str,
    instruction: str,
    screen_w: int = 1080,
    screen_h: int = 1920,
    action_history: list[str] | None = None,
    model: str = DEFAULT_ACTION_MODEL,
    overall_task: str = "",
) -> tuple[dict[str, Any], str]:
    """Call the action agent VLM for one step.

    The agent sees the instruction, action history, and current screenshot,
    and decides the next action (or "end" if the task is complete).

    Args:
        instruction: The high-level task instruction.
        screen_w: Actual device screen width in pixels.
        screen_h: Actual device screen height in pixels.
        action_history: List of NL descriptions of previous actions taken.
        overall_task: Optional real task for supplementary context.

    Returns:
        (aitk_action, history_entry):
        - aitk_action: action dict in AITK format (may be {"action": "end"})
        - history_entry: NL description for the agent's action history
    """
    screenshot_b64, resized_w, resized_h, _scale_back = _resize_screenshot(screenshot_b64)

    system_prompt = NEXT_ACTION_SYSTEM_PROMPT.format(
        screen_w=resized_w,
        screen_h=resized_h,
    )

    # Build history section
    if action_history:
        history_text = "\n".join(f"Step {i + 1}: {h}" for i, h in enumerate(action_history))
        history_section = f"Task progress (actions already taken):\n{history_text}\n\n"
    else:
        history_section = ""

    # Build overall task context section
    if overall_task and overall_task != instruction:
        overall_task_section = f"(Overall goal for context: {overall_task})\n\n"
    else:
        overall_task_section = ""

    user_prompt = NEXT_ACTION_USER_PROMPT.format(
        instruction=instruction,
        overall_task_section=overall_task_section,
        history_section=history_section,
    )

    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                _build_image_message(screenshot_b64),
            ],
        },
    ]
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
    )

    token_tracker.record("action", model, resp.usage)

    tool_args, thought, action_desc = _parse_agent_response(resp)
    if tool_args is None:
        logger.warning("Action agent parse failure, falling back to 'wait'")
        return {"action": "wait", "time": 1}, "Failed to parse action response -> wait"

    action = _tool_call_to_aitk(tool_args, screen_w=screen_w, screen_h=screen_h)

    # Build history entry
    history_entry = f"{thought} | {action_desc}" if thought else (action_desc or str(action))

    logger.info("  Agent action: %s → AITK: %s", tool_args.get("action"), action)

    return action, history_entry
