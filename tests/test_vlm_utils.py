"""Tests for android_app_graph.utils.vlm_utils.

Its cosine scoring is used by graph_manager, whose node-retrieval tests exercise
it end to end. Its OpenRouter helpers run here over the real SDK and an offline
transport: request serialization, response parsing, action conversion, and the
retry budget that HTTP, schema and parse failures share.
"""

from __future__ import annotations

import base64
import json
import logging
import math
import re
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, override

import httpx
import pytest
from hypothesis import given
from hypothesis import strategies as st
from openrouter import OpenRouter
from openrouter.chat import Chat
from openrouter.components.chatresult import ChatResult
from openrouter.components.chatusage import ChatUsage
from openrouter.embeddings import Embeddings
from openrouter.errors import ResponseValidationError
from openrouter.operations.createembeddings import CreateEmbeddingsUsage

from android_app_graph.payloads import as_object_dict, as_str, as_str_dict
from android_app_graph.retrying import UNUSABLE_OUTPUT_ATTEMPTS, UnusableOutputError
from android_app_graph.screen_image import encode_screenshot
from android_app_graph.utils import ModelSettings, open_client, resolve_model_settings
from android_app_graph.utils import vlm_utils as vlm
from android_app_graph.utils.vlm_utils import cosine_similarity, score_by_cosine

PAGE_DETAIL = ModelSettings("openai/gpt-6.1-sol")
ACTION = ModelSettings("qwen/qwen3.8-flash")
OTHER = ModelSettings("provider/other", request_timeout=60)


@pytest.fixture
def webp_screenshot(png_screenshot: str) -> str:
    """A screenshot as the step log hands it over: encoded once at capture."""
    return base64.b64encode(encode_screenshot(png_screenshot)).decode("ascii")


def test_build_image_message_labels_the_screenshot_as_webp(webp_screenshot: str) -> None:
    assert vlm.build_image_message(webp_screenshot) == {
        "type": "image_url",
        "image_url": {"url": f"data:image/webp;base64,{webp_screenshot}"},
    }


def test_cosine_similarity_of_identical_vectors_is_one() -> None:
    assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_cosine_similarity_of_orthogonal_vectors_is_zero() -> None:
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_of_opposite_vectors_is_minus_one() -> None:
    assert cosine_similarity([1.0, 1.0], [-1.0, -1.0]) == pytest.approx(-1.0)


@pytest.mark.parametrize(("a", "b"), [([0.0, 0.0], [1.0, 1.0]), ([1.0, 1.0], [0.0, 0.0])])
def test_cosine_similarity_with_a_zero_vector_is_zero(a: list[float], b: list[float]) -> None:
    """A zero norm has no direction, so the similarity is defined as 0.0 rather than NaN."""
    assert cosine_similarity(a, b) == 0.0


def test_cosine_similarity_rejects_vectors_of_different_length() -> None:
    """Two vectors of different dimension are never scored against each other.

    ``zip`` would otherwise silently truncate to the shorter vector, scoring a
    cached embedding from a different model's space against the overlapping
    prefix and ranking it as garbage with no signal that anything went wrong.
    """
    with pytest.raises(ValueError, match=r"2 vs 1|1 vs 2"):
        cosine_similarity([1.0, 0.0], [1.0])


@pytest.mark.parametrize(
    ("a", "b"),
    [
        # Subnormal floats break Cauchy-Schwarz under naive floating-point rounding:
        # the unclamped ratio can land slightly outside [-1.0, 1.0].
        ([2.5e-162], [3.4e-162]),
        ([1.0], [8.30800819665626e-160]),
    ],
)
def test_cosine_similarity_clamps_subnormal_rounding_artifacts(
    a: list[float], b: list[float]
) -> None:
    assert -1.0 <= cosine_similarity(a, b) <= 1.0


_finite_float = st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)
# Same length for both vectors: cosine_similarity rejects a mismatch outright,
# and this property is about the score, not that guard.
_same_length_vector_pair = st.integers(min_value=1, max_value=8).flatmap(
    lambda n: st.tuples(
        st.lists(_finite_float, min_size=n, max_size=n),
        st.lists(_finite_float, min_size=n, max_size=n),
    )
)


@given(vectors=_same_length_vector_pair)
def test_cosine_similarity_is_bounded_and_symmetric(
    vectors: tuple[list[float], list[float]],
) -> None:
    a, b = vectors
    similarity = cosine_similarity(a, b)
    assert -1.0 <= similarity <= 1.0
    assert similarity == pytest.approx(cosine_similarity(b, a))
    assert not math.isnan(similarity)


def test_cosine_similarity_with_a_nan_element_is_zero() -> None:
    """A NaN component scores 0.0 and never wins a similarity search.

    A naive clamp (``max(-1.0, min(1.0, nan))``) would return 1.0 instead.
    """
    assert cosine_similarity([math.nan, 1.0], [1.0, 1.0]) == 0.0


def test_cosine_similarity_never_returns_nan() -> None:
    assert not math.isnan(cosine_similarity([math.nan], [1.0]))
    assert not math.isnan(cosine_similarity([math.inf], [math.inf]))


def test_score_by_cosine_skips_missing_and_empty_vectors_without_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``None`` and ``[]`` both mean "no vector was ever computed for this
    candidate" -- an ordinary, silent case, not the stale-dimension one.
    """
    candidates = [("a", None), ("b", []), ("c", [1.0, 0.0])]
    with caplog.at_level("WARNING"):
        scored = score_by_cosine([1.0, 0.0], candidates, scope="test")
    assert scored == [("c", pytest.approx(1.0))]
    assert caplog.text == ""


def test_score_by_cosine_skips_a_wrong_dimension_vector_and_warns_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    candidates = [("a", [1.0, 0.0, 0.0]), ("b", [1.0, 0.0]), ("c", [0.0, 1.0, 0.0])]
    with caplog.at_level("WARNING"):
        scored = score_by_cosine([1.0, 0.0], candidates, scope="my-scope")
    assert scored == [("b", pytest.approx(1.0))]
    stale_warnings = [m for m in caplog.messages if "my-scope" in m]
    assert len(stale_warnings) == 1
    assert "query dim=2" in stale_warnings[0]
    assert "dim(s)=[3]" in stale_warnings[0]


def test_score_by_cosine_appends_the_remedy_to_the_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    candidates = [("a", [1.0, 0.0, 0.0])]
    with caplog.at_level("WARNING"):
        score_by_cosine([1.0, 0.0], candidates, scope="my-scope", remedy="; do X")
    assert "; do X" in caplog.text


def test_score_by_cosine_keeps_the_scored_order_equal_to_the_input_order() -> None:
    candidates = [("c", [0.0, 1.0]), ("a", [1.0, 0.0]), ("b", [1.0, 1.0])]
    scored = score_by_cosine([1.0, 0.0], candidates, scope="test")
    assert [key for key, _ in scored] == ["c", "a", "b"]


_CLICK = '{"action":"click","coordinate":[500,500]}'


@pytest.fixture
def tracked_usage(monkeypatch: pytest.MonkeyPatch) -> vlm.TokenTracker:
    tracker = vlm.TokenTracker()
    monkeypatch.setattr(vlm, "token_tracker", tracker)
    return tracker


def test_token_tracker_sums_cost_cached_and_reasoning_tokens_and_logs_them(
    caplog: pytest.LogCaptureFixture,
) -> None:
    tracker = vlm.TokenTracker()
    with caplog.at_level("INFO"):
        tracker.print_summary()
    assert "No API calls recorded." in caplog.text
    chat = ChatUsage.model_validate(
        {
            "prompt_tokens": 100,
            "completion_tokens": 40,
            "total_tokens": 140,
            "cost": 0.0125,
            "prompt_tokens_details": {"cached_tokens": 64},
            "completion_tokens_details": {"reasoning_tokens": 30},
        }
    )
    bare = ChatUsage.model_validate({"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3})
    embedding = CreateEmbeddingsUsage.model_validate(
        {"prompt_tokens": 5, "total_tokens": 5, "cost": 0.0005}
    )
    tracker.record("action", "qwen/qwen3.8-flash", chat)
    tracker.record("action", "qwen/qwen3.8-flash", bare)
    tracker.record("embedding", "google/gemini-embedding-2", embedding)
    tracker.record("embedding", "google/gemini-embedding-2", None)
    assert tracker.summary()["action:qwen/qwen3.8-flash"] == {
        "calls": 2,
        "prompt_tokens": 101,
        "cached_prompt_tokens": 64,
        "completion_tokens": 42,
        "reasoning_tokens": 30,
        "total_tokens": 143,
        "cost": 0.0125,
    }
    assert tracker.snapshot()["cost"] == pytest.approx(0.013)
    assert tracker.snapshot_by_type() == {"action": 143, "embedding": 5}
    caplog.clear()
    with caplog.at_level("INFO"):
        tracker.print_summary()
    assert "Cached" in caplog.text
    assert "Reasoning" in caplog.text
    assert "Cost (USD)" in caplog.text
    (total,) = [line for line in caplog.text.splitlines() if line.startswith("TOTAL")]
    assert total.split()[1:] == ["3", "106", "64", "42", "30", "148", "0.013000"]


def _cost_details(upstream: float) -> dict[str, float]:
    return {
        "upstream_inference_prompt_cost": upstream / 2,
        "upstream_inference_completions_cost": upstream / 2,
        "upstream_inference_cost": upstream,
    }


def test_token_tracker_adds_the_provider_bill_of_a_byok_request_to_its_cost() -> None:
    tracker = vlm.TokenTracker()
    byok_chat = ChatUsage.model_validate(
        {
            "prompt_tokens": 65,
            "completion_tokens": 26,
            "total_tokens": 91,
            "cost": 0,
            "is_byok": True,
            "cost_details": _cost_details(0.00002197),
        }
    )
    byok_embedding = CreateEmbeddingsUsage.model_validate(
        {
            "prompt_tokens": 5,
            "total_tokens": 5,
            "cost": 0.000001,
            "is_byok": True,
            "cost_details": _cost_details(0.00001),
        }
    )
    credit_chat = ChatUsage.model_validate(
        {
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
            "cost": 0.0003,
            "is_byok": False,
            "cost_details": _cost_details(0.0003),
        }
    )
    tracker.record("action", "qwen/qwen3.8-flash", byok_chat)
    tracker.record("action", "qwen/qwen3.8-flash", credit_chat)
    tracker.record("embedding", "voyageai/voyage-multimodal-3.5", byok_embedding)
    summary = tracker.summary()
    assert summary["action:qwen/qwen3.8-flash"]["cost"] == pytest.approx(0.00032197)
    assert summary["embedding:voyageai/voyage-multimodal-3.5"]["cost"] == pytest.approx(0.000011)


# How long a blocked call waits to be let go before it fails the test instead of hanging.
RELEASE_TIMEOUT = 10.0


class CallFailedError(RuntimeError):
    """Raised by the call for the item it names."""


class OpenOnMessage(logging.Handler):
    """Sets *event* when a record's message contains *text*."""

    def __init__(self, text: str, event: threading.Event) -> None:
        super().__init__()
        self.text = text
        self.event = event

    @override
    def emit(self, record: logging.LogRecord) -> None:
        if self.text in record.getMessage():
            self.event.set()


def test_call_concurrently_cancels_the_calls_not_yet_started_once_a_call_fails(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Item 0 fails once item 1 has started, and every other started call waits for the
    cancellation, so item 1 is running when the cancellation comes and no item past the
    first one the freed worker may take ever starts."""
    in_flight = vlm.MAX_REQUESTS_IN_FLIGHT
    item_1_started = threading.Event()
    cancelled = threading.Event()
    started: list[int] = []
    applied: list[tuple[int, int]] = []

    def call(item: int) -> int:
        started.append(item)
        if item == 0:
            assert item_1_started.wait(RELEASE_TIMEOUT)
            raise CallFailedError(item)
        if item == 1:
            item_1_started.set()
        assert cancelled.wait(RELEASE_TIMEOUT)
        if item == 1:
            raise CallFailedError(item)
        return item * 10

    vlm_logger = logging.getLogger(vlm.__name__)
    handler = OpenOnMessage("not yet started", cancelled)
    vlm_logger.addHandler(handler)
    try:
        with (
            caplog.at_level(logging.WARNING, logger=vlm.__name__),
            pytest.raises(CallFailedError) as raised,
        ):
            vlm.call_concurrently(
                "test_call",
                call,
                range(3 * in_flight),
                lambda item, result: applied.append((item, result)),
            )
    finally:
        vlm_logger.removeHandler(handler)

    assert raised.value.args == (0,)
    assert max(started) <= in_flight
    assert applied == [(item, item * 10) for item in sorted(started) if item > 1]
    messages = [record.getMessage() for record in caplog.records]
    assert any(message.startswith("test_call failed; cancelled ") for message in messages)
    assert "test_call failed too: CallFailedError: 1" in messages


def test_call_concurrently_sends_no_queued_call_after_an_error_on_the_calling_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An error in ``apply``, like an interrupt, cancels the queued calls rather than waiting
    for all of them; the started calls wait until the pool's shutdown has cancelled them."""
    in_flight = vlm.MAX_REQUESTS_IN_FLIGHT
    shut_down = threading.Event()
    started: list[int] = []

    class SignallingPool(ThreadPoolExecutor):
        @override
        def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
            super().shutdown(wait=False, cancel_futures=cancel_futures)
            shut_down.set()
            super().shutdown(wait=wait)

    def call(item: int) -> int:
        started.append(item)
        if item > 0:
            assert shut_down.wait(RELEASE_TIMEOUT)
        return item

    def apply(_item: int, _result: int) -> None:
        raise CallFailedError

    monkeypatch.setattr(vlm, "ThreadPoolExecutor", SignallingPool)

    with pytest.raises(CallFailedError):
        vlm.call_concurrently("test_call", call, range(3 * in_flight), apply)

    assert max(started) <= in_flight


_STRUCTURED: dict[str, tuple[str, bool]] = {
    "page": ("page_description", False),
    "verify": ("node_verification", True),
    "normalize": ("edge_template", False),
    "merge": ("node_merge_audit", True),
    "audit": ("graph_audit", True),
    "target": ("exploration_target", True),
    "planner": ("next_action_plan", True),
    "describe": ("action_description", True),
}


def _verify_itself(client: OpenRouter, screenshot: str, description: str) -> dict[str, Any]:
    """Ask the verifier whether ``screenshot`` and a copy of it show one screen, both
    described as ``description``."""
    shown = vlm.DescribedScreenshot(screenshot, description)
    return vlm.verify_same_node(client, existing=shown, new=shown, settings=PAGE_DETAIL)


_ROLE_HELPERS: dict[str, Callable[[OpenRouter, str], object]] = {
    "page": lambda client, screenshot: vlm.describe_page_and_state(
        client, screenshot, settings=PAGE_DETAIL
    ),
    "verify": lambda client, screenshot: _verify_itself(client, screenshot, "home"),
    "normalize": lambda client, _screenshot: vlm.normalize_edge(
        client, ["search milk"], settings=PAGE_DETAIL
    ),
    "merge": lambda client, _screenshot: vlm.audit_merge_nodes(
        client, "home", settings=PAGE_DETAIL
    ),
    "audit": lambda client, _screenshot: vlm.audit_graph(client, "home", settings=PAGE_DETAIL),
    "target": lambda client, _screenshot: vlm.select_exploration_target(
        client, "demo", [{"node_id": "home"}], settings=PAGE_DETAIL
    ),
    "planner": lambda client, screenshot: vlm.plan_next_action(
        client,
        vlm.PlanRequest(
            screenshot_b64=screenshot,
            page_description="home",
            node="s0",
            explored_edges=[],
            app_name="demo",
        ),
        settings=PAGE_DETAIL,
    ),
    "describe": lambda client, screenshot: vlm.describe_executed_action(
        client, screenshot, {"action": "back"}, (1080, 1920), settings=PAGE_DETAIL
    ),
    "action": lambda client, screenshot: vlm.predict_next_action(
        client, vlm.ActionRequest(screenshot_b64=screenshot, instruction="wait"), settings=ACTION
    ),
}


@pytest.mark.parametrize(
    ("operation", "call_type", "model", "content", "expected"),
    [
        (
            "page",
            "page_describe_and_state",
            "openai/gpt-6.1-sol",
            (
                '{"page_description":"home","state":{"tab":"all"},'
                '"elements":[{"description":"Search","position":"top"}]}'
            ),
            ("home", {"tab": "all"}, [{"description": "Search", "position": "top"}], False),
        ),
        ("verify", "node_verify", "openai/gpt-6.1-sol", '{"same":true}', {"same": True}),
        (
            "normalize",
            "normalize_edge",
            "openai/gpt-6.1-sol",
            '{"instruction_template":"search {query}"}',
            {"instruction_template": "search {query}"},
        ),
        (
            "merge",
            "node_merge_audit",
            "openai/gpt-6.1-sol",
            '{"issues":[],"summary":"fine"}',
            {"issues": [], "summary": "fine"},
        ),
        (
            "audit",
            "graph_audit",
            "openai/gpt-6.1-sol",
            '{"issues":[],"summary":"fine"}',
            {"issues": [], "summary": "fine"},
        ),
        ("target", "exploration_target", "openai/gpt-6.1-sol", '{"node_id":"home"}', "home"),
        (
            "planner",
            "instruction",
            "openai/gpt-6.1-sol",
            '```json\n{"element":null,"instruction":"Tap Search"}\n```',
            ("Tap Search", None),
        ),
        (
            "describe",
            "action_description",
            "openai/gpt-6.1-sol",
            '{"instruction":"Tap Search"}',
            "Tap Search",
        ),
        (
            "action",
            "action",
            "qwen/qwen3.8-flash",
            '{"action":"wait"}',
            ({"action": "wait", "time": 3}, "{'action': 'wait', 'time': 3}"),
        ),
    ],
)
def test_role_helpers_serialize_and_parse_real_sdk_responses(
    operation: str,
    call_type: str,
    model: str,
    content: str,
    expected: object,
    router_client: OpenRouter,
    router_requests: list[httpx.Request],
    router_responses: list[httpx.Response],
    webp_screenshot: str,
    tracked_usage: vlm.TokenTracker,
    chat_response: Callable[..., httpx.Response],
    chat_usage_totals: dict[str, float],
) -> None:
    """Each role sends the captured WebP screenshot unchanged, labelled image/webp."""
    router_responses.append(chat_response(content=content))
    result = _ROLE_HELPERS[operation](router_client, webp_screenshot)
    assert result == expected
    body = json.loads(router_requests[0].content)
    assert body["model"] == model
    assert body["stream"] is False
    assert "temperature" not in body
    if operation in {"page", "verify", "planner", "describe", "action"}:
        parts = body["messages"][-1]["content"]
        images = [part for part in parts if part["type"] == "image_url"]
        assert len(images) == (2 if operation == "verify" else 1)
        assert images[0]["image_url"]["url"] == f"data:image/webp;base64,{webp_screenshot}"
    if operation in _STRUCTURED:
        name, strict = _STRUCTURED[operation]
        json_schema = body["response_format"]["json_schema"]
        assert body["response_format"]["type"] == "json_schema"
        assert (json_schema["name"], json_schema["strict"]) == (name, strict)
        assert json_schema["schema"]["type"] == "object"
        assert body["plugins"] == [{"id": "response-healing"}]
    else:
        assert "response_format" not in body
        assert "plugins" not in body
    assert "provider" not in body
    assert tracked_usage.summary()[f"{call_type}:{model}"] == chat_usage_totals


def test_a_native_tool_call_without_text_content_is_still_read(
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    png_screenshot: str,
    chat_response: Callable[..., httpx.Response],
) -> None:
    router_responses.append(chat_response(content="", tool_arguments=_CLICK))
    action, history = vlm.predict_next_action(
        router_client,
        vlm.ActionRequest(
            screenshot_b64=png_screenshot, instruction="tap", screen_w=100, screen_h=200
        ),
        settings=ACTION,
    )
    assert action == {"action": "tap", "x": 50, "y": 100}
    assert history == "{'action': 'click', 'coordinate': [500, 500]}"


@pytest.mark.parametrize("native", [False, True])
def test_menu_tool_calls_convert_to_the_menu_device_action(
    native: bool, chat_response: Callable[..., httpx.Response]
) -> None:
    arguments = '{"action": "system_button", "button": "Menu"}'
    reply = ChatResult.model_validate(
        chat_response(
            content="" if native else f"<tool_call>{arguments}</tool_call>",
            tool_arguments=arguments if native else None,
        ).json()
    )

    args, _, _ = vlm.parse_agent_response(reply)

    assert args is not None
    assert vlm.tool_call_to_action(args) == {"action": "menu"}


def test_action_prompt_declares_mobile_use_and_the_benchmarked_screen_line(
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    png_screenshot: str,
    chat_response: Callable[..., httpx.Response],
) -> None:
    router_responses.append(chat_response(content='<tool_call>{"action":"wait"}</tool_call>'))
    vlm.predict_next_action(
        router_client,
        vlm.ActionRequest(
            screenshot_b64=png_screenshot, instruction="wait", screen_w=1080, screen_h=2400
        ),
        settings=ACTION,
    )
    system = as_str_dict(json.loads(router_requests[0].content)["messages"][0])
    assert system["role"] == "system"
    prompt = as_str(system["content"], "")
    tools = prompt.split("<tools>\n", 1)[1].split("\n</tools>", 1)[0]
    assert json.loads(tools)["function"]["name"] == "mobile_use"
    assert "The screen's resolution is 999x999." in prompt
    assert "32x32" not in prompt
    assert "1080" not in prompt
    assert "application options menu" in prompt


@pytest.mark.parametrize(
    "tool_call",
    [
        '{"name": "mobile_use", "arguments": {"action": "click", "coordinate": [752, 170]}}',
        '{"action": "click", "coordinate": [752, 170]}',
    ],
)
def test_a_tool_call_missing_its_closing_tag_is_still_read(
    tool_call: str,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    png_screenshot: str,
    chat_response: Callable[..., httpx.Response],
) -> None:
    content = (
        "Thought: The search icon is at the top.\nAction: Tap the search icon.\n<tool_call>\n"
        + tool_call
    )
    router_responses.append(chat_response(content=content))
    action, history = vlm.predict_next_action(
        router_client,
        vlm.ActionRequest(
            screenshot_b64=png_screenshot, instruction="tap", screen_w=1000, screen_h=2000
        ),
        settings=ACTION,
    )
    assert action == {"action": "tap", "x": 752, "y": 340}
    assert history == "The search icon is at the top. | Tap the search icon."
    assert len(router_requests) == 1


def _strict_violations(schema: object, path: str = "$") -> list[str]:
    """Where a schema breaks strict mode: every object closed, every property required."""
    if isinstance(schema, list):
        return [
            violation
            for index, item in enumerate(schema)
            for violation in _strict_violations(item, f"{path}[{index}]")
        ]
    node = as_object_dict(schema)
    violations: list[str] = []
    if node.get("type") == "object":
        properties = as_object_dict(node.get("properties"))
        if node.get("additionalProperties") is not False:
            violations.append(f"{path} is open")
        required = node.get("required")
        required_names = sorted(map(str, required)) if isinstance(required, list) else []
        if required_names != sorted(properties):
            violations.append(f"{path} leaves a property optional")
        violations.extend(
            f"{path}.{key} has no description"
            for key, value in properties.items()
            if "description" not in as_object_dict(value)
        )
    for key, value in node.items():
        violations.extend(_strict_violations(value, f"{path}.{key}"))
    return violations


def _sent_schemas(
    router_client: OpenRouter,
    router_requests: list[httpx.Request],
    router_responses: list[httpx.Response],
    png_screenshot: str,
    chat_response: Callable[..., httpx.Response],
) -> dict[str, dict[str, Any]]:
    router_responses.extend(
        chat_response(content=content)
        for content in (
            '{"page_description":"home","state":{},"elements":[]}',
            '{"same":true}',
            "{}",
            "{}",
            "{}",
            "{}",
            "{}",
        )
    )
    vlm.describe_page_and_state(router_client, png_screenshot, settings=PAGE_DETAIL)
    _verify_itself(router_client, png_screenshot, "home")
    vlm.normalize_edge(router_client, ["search milk"], settings=PAGE_DETAIL)
    vlm.audit_merge_nodes(router_client, "home", settings=PAGE_DETAIL)
    vlm.audit_graph(router_client, "home", settings=PAGE_DETAIL)
    vlm.select_exploration_target(
        router_client, "demo", [{"node_id": "b"}, {"node_id": "a"}], settings=PAGE_DETAIL
    )
    vlm.describe_executed_action(
        router_client, png_screenshot, {"action": "back"}, (1080, 1920), settings=PAGE_DETAIL
    )
    schemas = [
        json.loads(request.content)["response_format"]["json_schema"] for request in router_requests
    ]
    return {schema["name"]: schema for schema in schemas}


def test_strict_schemas_close_every_object_and_free_form_ones_keep_their_format(
    router_client: OpenRouter,
    router_requests: list[httpx.Request],
    router_responses: list[httpx.Response],
    png_screenshot: str,
    chat_response: Callable[..., httpx.Response],
) -> None:
    schemas = _sent_schemas(
        router_client, router_requests, router_responses, png_screenshot, chat_response
    )
    strict = {name: schema for name, schema in schemas.items() if schema["strict"]}
    assert sorted(strict) == [
        "action_description",
        "exploration_target",
        "graph_audit",
        "node_merge_audit",
        "node_verification",
    ]
    for name, schema in strict.items():
        assert _strict_violations(schema["schema"]) == [], name
    assert schemas["exploration_target"]["schema"]["properties"]["node_id"]["enum"] == ["a", "b"]
    verification = schemas["node_verification"]["schema"]
    assert sorted(verification["properties"]) == ["new_description", "reason", "same"]
    assert sorted(verification["required"]) == ["new_description", "reason", "same"]
    assert verification["properties"]["new_description"]["type"] == ["string", "null"]
    state = schemas["page_description"]["schema"]["properties"]["state"]
    assert state["additionalProperties"] == {"type": ["string", "number", "boolean", "null"]}
    examples = schemas["edge_template"]["schema"]["properties"]["examples"]
    assert examples["items"]["properties"]["params"]["additionalProperties"] == {"type": "string"}


def test_the_page_description_asks_for_and_reads_the_external_web_flag(
    router_client: OpenRouter,
    router_requests: list[httpx.Request],
    router_responses: list[httpx.Response],
    png_screenshot: str,
    chat_response: Callable[..., httpx.Response],
) -> None:
    router_responses.append(
        chat_response(content='{"page_description":"news article","is_external_web":true}')
    )
    *_, is_external_web = vlm.describe_page_and_state(
        router_client, png_screenshot, settings=PAGE_DETAIL
    )
    assert is_external_web is True
    body = json.loads(router_requests[0].content)
    schema = body["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["is_external_web"]["type"] == "boolean"
    assert "is_external_web" in schema["required"]
    assert '"is_external_web"' in body["messages"][0]["content"][0]["text"]


def test_graph_audit_assesses_the_effect_of_both_kinds_of_device_proposal(
    router_client: OpenRouter,
    router_requests: list[httpx.Request],
    router_responses: list[httpx.Response],
    chat_response: Callable[..., httpx.Response],
) -> None:
    router_responses.append(chat_response(content='{"issues":[],"summary":"Inspection complete"}'))
    vlm.audit_graph(router_client, "Phone keypad and deletion confirmation", settings=PAGE_DETAIL)
    [request] = [json.loads(request.content) for request in router_requests]
    variants = request["response_format"]["json_schema"]["schema"]["properties"]["issues"]["items"][
        "anyOf"
    ]
    for variant in variants:
        if variant["properties"]["type"]["enum"][0] == "merge_nodes":
            continue
        assert "execution" in variant["required"]
        assessment = variant["properties"]["execution"]
        assert assessment["properties"]["effect"]["enum"] == [
            "inspection",
            "side_effect",
            "uncertain",
        ]
        assert assessment["required"] == ["effect", "reason"]
    prompt = request["messages"][0]["content"]
    assert "controlled test number" in prompt
    assert "confirmation dialog followed" in prompt
    assert "A recorded action is" in prompt


def test_the_page_description_keeps_keyboard_focus_and_scroll_in_state(
    router_client: OpenRouter,
    router_requests: list[httpx.Request],
    router_responses: list[httpx.Response],
    png_screenshot: str,
    chat_response: Callable[..., httpx.Response],
) -> None:
    for known_descriptions in (None, ["Add Wi-Fi network dialog"]):
        router_responses.append(
            chat_response(content='{"page_description":"Add Wi-Fi network dialog"}')
        )
        vlm.describe_page_and_state(
            router_client,
            png_screenshot,
            known_descriptions=known_descriptions,
            settings=PAGE_DETAIL,
        )
    for request in router_requests:
        prompt = json.loads(request.content)["messages"][0]["content"][0]["text"]
        assert 'never change "page_description"' in prompt
        assert '"focused_field"' in prompt
        assert '"keyboard_visible"' in prompt
        assert "input field focused" not in prompt
        assert "which field is active" not in prompt


def test_the_verifier_treats_keyboard_focus_and_scroll_as_the_same_screen(
    router_client: OpenRouter,
    router_requests: list[httpx.Request],
    router_responses: list[httpx.Response],
    png_screenshot: str,
    chat_response: Callable[..., httpx.Response],
) -> None:
    router_responses.append(chat_response(content='{"same":true,"reason":"keyboard only"}'))
    _verify_itself(router_client, png_screenshot, "Add Wi-Fi network dialog")
    prompt = json.loads(router_requests[0].content)["messages"][0]["content"][0]["text"]
    assert "differ only in whether the on-screen keyboard is shown" in prompt
    assert "Which input field is active" not in prompt


def test_the_page_description_keeps_open_menus_and_expanded_sections_in_state(
    router_client: OpenRouter,
    router_requests: list[httpx.Request],
    router_responses: list[httpx.Response],
    png_screenshot: str,
    chat_response: Callable[..., httpx.Response],
) -> None:
    """A menu open over a page and its expanded sections are recorded as the state keys
    ``open_menu`` and ``expanded_sections``, never in the page's description, while a
    dialog or full-screen picker with its own purpose is described as a screen."""
    for known_descriptions in (None, ["Add Wi-Fi network dialog"]):
        router_responses.append(
            chat_response(content='{"page_description":"Add Wi-Fi network dialog"}')
        )
        vlm.describe_page_and_state(
            router_client,
            png_screenshot,
            known_descriptions=known_descriptions,
            settings=PAGE_DETAIL,
        )
    for request in router_requests:
        prompt = json.loads(request.content)["messages"][0]["content"][0]["text"]
        kept_out = re.search(r'[^.]*never change "page_description"', prompt)
        assert kept_out is not None
        for kind in ("menu", "dropdown", "picker", "popup", "expanded"):
            assert kind in kept_out.group(), kind
        assert '"open_menu"' in prompt
        assert '"expanded_sections"' in prompt
        assert "full-screen picker that has its own purpose" in prompt


def test_the_verifier_counts_an_open_menu_or_expanded_section_as_the_same_screen(
    router_client: OpenRouter,
    router_requests: list[httpx.Request],
    router_responses: list[httpx.Response],
    png_screenshot: str,
    chat_response: Callable[..., httpx.Response],
) -> None:
    """Screens that differ only by an open menu or an expanded section are the SAME
    screen, the DIFFERENT cases name no menu, and a dialog with its own purpose is
    DIFFERENT from the page beneath it."""
    router_responses.append(chat_response(content='{"same":true,"reason":"menu only"}'))
    _verify_itself(router_client, png_screenshot, "Wi-Fi settings")
    prompt = json.loads(router_requests[0].content)["messages"][0]["content"][0]["text"]
    same_cases, different_cases = prompt.split("Two screenshots are DIFFERENT screens", 1)
    same_cases = same_cases[same_cases.index("Two screenshots are the SAME screen") :]
    different_cases = different_cases[: different_cases.index("Answer with ONLY")]
    for kind in ("overflow menu", "dropdown", "picker", "popup", "context menu", "expanded"):
        assert kind in same_cases, kind
    assert "menu" not in different_cases
    assert "its own purpose" in different_cases


def test_the_verifier_keeps_page_state_out_of_its_refined_descriptions(
    router_client: OpenRouter,
    router_requests: list[httpx.Request],
    router_responses: list[httpx.Response],
    png_screenshot: str,
    chat_response: Callable[..., httpx.Response],
) -> None:
    """The description the verifier writes for the new screen when the two differ leaves
    out what "page_description" leaves out, so a screen with a menu open is named after
    its page's purpose."""
    router_responses.append(chat_response(content='{"same":true,"reason":"same page"}'))
    _verify_itself(router_client, png_screenshot, "Wi-Fi Direct")
    prompt = json.loads(router_requests[0].content)["messages"][0]["content"][0]["text"]
    refinement = prompt[prompt.index("If they are DIFFERENT screens") :]
    for kind in ("keyboard", "focus", "scrolled", "menu", "dropdown", "picker", "expanded"):
        assert kind in refinement, kind


def test_the_verifier_is_shown_each_screenshot_with_its_own_description(
    router_client: OpenRouter,
    router_requests: list[httpx.Request],
    router_responses: list[httpx.Response],
    png_screenshot: str,
    chat_response: Callable[..., httpx.Response],
    shown_descriptions: Callable[[httpx.Request], list[str]],
) -> None:
    """The existing screenshot comes first, and when the two differ the verifier refines
    only the new screen's description."""
    router_responses.append(chat_response(content='{"same":true,"reason":"one screen"}'))
    other = base64.b64encode(b"another screenshot").decode("ascii")

    vlm.verify_same_node(
        router_client,
        existing=vlm.DescribedScreenshot(png_screenshot, "Wi-Fi settings"),
        new=vlm.DescribedScreenshot(other, "Network settings"),
        settings=PAGE_DETAIL,
    )

    (request,) = router_requests
    assert shown_descriptions(request) == [
        'First screenshot (existing node), described as "Wi-Fi settings":',
        'Second screenshot (new screen), described as "Network settings":',
    ]
    content = json.loads(request.content)["messages"][0]["content"]
    images = [part["image_url"]["url"] for part in content if part["type"] == "image_url"]
    assert [image.rpartition(",")[2] for image in images] == [png_screenshot, other]
    refinement = content[0]["text"][content[0]["text"].index("If they are DIFFERENT screens") :]
    assert "description of the second screenshot" in refinement
    assert refinement.count("_description") == 1


def test_the_verifier_tells_navigation_destinations_apart_and_keeps_their_content_together(
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    png_screenshot: str,
    chat_response: Callable[..., httpx.Response],
    verifier_prompts: Callable[[], list[str]],
) -> None:
    """Destinations the app names in its navigation, such as Files' Downloads and
    Internal storage, are DIFFERENT screens even when they share a layout, and a
    subfolder, query or list inside one destination is the SAME screen as it."""
    router_responses.append(chat_response(content='{"same":false,"reason":"drawer entry"}'))
    _verify_itself(router_client, png_screenshot, "Downloads file list")
    (prompt,) = verifier_prompts()
    for navigation in ("a tab", "a bottom navigation bar item", "a navigation drawer entry"):
        assert navigation in prompt, navigation
    assert "different destinations are DIFFERENT screens, even when they share a layout" in prompt
    assert "Downloads, Recent and Internal storage" in prompt
    assert (
        "Content inside one destination is the SAME screen as that destination: a subfolder "
        "opened inside it, a different search query or a different list."
    ) in prompt


def test_the_merge_audit_may_report_a_page_with_its_menu_open(
    router_client: OpenRouter,
    router_requests: list[httpx.Request],
    router_responses: list[httpx.Response],
    chat_response: Callable[..., httpx.Response],
) -> None:
    """The look-alike screens the merge audit leaves unreported have different purposes,
    so it may report a page and the same page with a menu open for the verifier."""
    router_responses.append(chat_response(content='{"issues":[],"summary":"fine"}'))
    vlm.audit_merge_nodes(router_client, "home", settings=PAGE_DETAIL)
    prompt = json.loads(router_requests[0].content)["messages"][0]["content"]
    assert "menu" not in prompt


@pytest.mark.parametrize(
    ("operation", "reply", "expected"),
    [
        ("target", '{"node_id":"elsewhere","reason":"invented"}', None),
        (
            "merge",
            '{"issues":[{"type":"retry_edge","node_a":"a","node_b":"b"}],"summary":"s"}',
            {"issues": [], "summary": "s"},
        ),
    ],
)
def test_parsers_reject_replies_that_violate_the_schema(
    operation: str,
    reply: str,
    expected: object,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    chat_response: Callable[..., httpx.Response],
) -> None:
    router_responses.append(chat_response(content=reply))
    result = (
        vlm.select_exploration_target(
            router_client, "demo", [{"node_id": "home"}], settings=PAGE_DETAIL
        )
        if operation == "target"
        else vlm.audit_merge_nodes(router_client, "home", settings=PAGE_DETAIL)
    )
    assert result == expected


def test_the_exploration_target_request_marks_only_the_start_screen(
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    chat_response: Callable[..., httpx.Response],
) -> None:
    router_responses.append(chat_response(content='{"node_id":"s0_home","reason":"sections"}'))
    selected = vlm.select_exploration_target(
        router_client,
        "demo",
        [
            {"node_id": "s1_form", "is_start_node": False},
            {"node_id": "s2_form", "is_start_node": False},
            {"node_id": "s0_home", "is_start_node": True},
        ],
        settings=PAGE_DETAIL,
    )

    assert selected == "s0_home"
    prompt = json.loads(router_requests[0].content)["messages"][0]["content"]
    lines = [line for line in prompt.splitlines() if "node_id=" in line]
    assert [line.split(" | ", 1)[0] for line in lines] == [
        "1. node_id=s1_form",
        "2. node_id=s2_form",
        "3. node_id=s0_home",
    ]
    assert [line.endswith(" | start screen") for line in lines] == [False, False, True]
    assert "marked start screen" in prompt


def test_the_exploration_target_request_names_each_candidate_by_its_member_descriptions(
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    chat_response: Callable[..., httpx.Response],
) -> None:
    router_responses.append(chat_response(content='{"node_id":"s1_form","reason":"fields"}'))
    descriptions = '"APN form" (2), "Access point editor" (1)'

    vlm.select_exploration_target(
        router_client,
        "demo",
        [{"node_id": "s1_form", "descriptions": descriptions}],
        settings=PAGE_DETAIL,
    )

    prompt = json.loads(router_requests[0].content)["messages"][0]["content"]
    (line,) = [line for line in prompt.splitlines() if "node_id=" in line]
    assert line.split(" | ")[:2] == ["1. node_id=s1_form", f"descriptions={descriptions}"]


@pytest.mark.parametrize(
    "content",
    [None, "omitted", [{"type": "text", "text": "  hello "}, {"type": "text", "text": "world "}]],
)
def test_sdk_assistant_content_variants(
    content: object,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    chat_response: Callable[..., httpx.Response],
) -> None:
    router_responses.append(
        chat_response() if content == "omitted" else chat_response(content=content)
    )
    response = vlm.send_chat(
        router_client,
        call_type="chat",
        settings=OTHER,
        messages=[{"role": "user", "content": "hi"}],
    )
    assert vlm.message_text(response) == ("hello \nworld" if isinstance(content, list) else "")


@pytest.mark.parametrize("image", [False, True])
def test_text_and_image_embeddings_use_the_documented_sdk_format(
    image: bool,
    router_client: OpenRouter,
    router_requests: list[httpx.Request],
    router_responses: list[httpx.Response],
    embedding_body: dict[str, object],
    embedding_usage_totals: dict[str, float],
    tracked_usage: vlm.TokenTracker,
) -> None:
    router_responses.append(httpx.Response(200, json=embedding_body))
    vector = (
        vlm.get_image_embedding(router_client, "shot", model="voyageai/voyage-multimodal-3.5")
        if image
        else vlm.get_embedding(router_client, "home", model="voyageai/voyage-multimodal-3.5")
    )
    assert vector == [0.25, 0.75]
    body = json.loads(router_requests[0].content)
    assert body["model"] == "voyageai/voyage-multimodal-3.5"
    assert body["encoding_format"] == "float"
    assert body["input"] == (
        [{"content": [{"type": "image_url", "image_url": {"url": "data:image/webp;base64,shot"}}]}]
        if image
        else "home"
    )
    assert tracked_usage.snapshot() == embedding_usage_totals
    assert tracked_usage.snapshot_by_type() == {"image_embedding" if image else "embedding": 5}


def test_an_unusable_image_embedding_is_asked_for_once_more(
    router_client: OpenRouter,
    router_requests: list[httpx.Request],
    router_responses: list[httpx.Response],
    embedding_body: dict[str, object],
    tracked_usage: vlm.TokenTracker,
) -> None:
    router_responses.extend(
        [
            httpx.Response(
                200, json={**embedding_body, "data": [{"object": "embedding", "embedding": []}]}
            ),
            httpx.Response(200, json=embedding_body),
        ]
    )
    assert vlm.get_image_embedding(router_client, "shot", model="provider/image") == [0.25, 0.75]
    assert len(router_requests) == 2
    assert tracked_usage.summary()["image_embedding:provider/image"]["calls"] == 2


@pytest.mark.parametrize(
    "data",
    [
        [{"object": "embedding", "embedding": "base64-vector"}],
        [{"object": "embedding", "embedding": [math.inf]}],
    ],
)
def test_sdk_embedding_shapes_are_rejected_explicitly(
    data: list[dict[str, object]],
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    embedding_body: dict[str, object],
) -> None:
    embedding_body["data"] = data
    router_responses.extend(
        httpx.Response(
            200, text=json.dumps(embedding_body), headers={"content-type": "application/json"}
        )
        for _ in range(UNUSABLE_OUTPUT_ATTEMPTS)
    )
    with pytest.raises(UnusableOutputError, match="embedding"):
        vlm.get_embedding(router_client, "home", model="google/gemini-embedding-2")


def _request(operation: str, client: OpenRouter, png_screenshot: str) -> object:
    match operation:
        case "chat":
            reply = vlm.send_chat(
                client,
                call_type="chat",
                settings=OTHER,
                messages=[{"role": "user", "content": "hi"}],
            )
            return vlm.message_text(reply)
        case "text":
            return vlm.get_embedding(client, "home", model="google/gemini-embedding-2")
        case "image":
            return vlm.get_image_embedding(
                client, png_screenshot, model="google/gemini-embedding-2"
            )
        case "action":
            return vlm.predict_next_action(
                client,
                vlm.ActionRequest(screenshot_b64=png_screenshot, instruction="tap"),
                settings=ACTION,
            )[0]
        case _:
            return vlm.describe_page_and_state(client, png_screenshot, settings=PAGE_DETAIL)[0]


_GOOD_REPLIES: dict[str, str] = {
    "chat": "test reply",
    "action": '{"action":"click","coordinate":[500,500]}',
    "description": '{"page_description":"home"}',
}
_RESULTS: dict[str, object] = {
    "chat": "test reply",
    "text": [0.25, 0.75],
    "image": [0.25, 0.75],
    "action": {"action": "tap", "x": 540, "y": 960},
    "description": "home",
}
_CALL_TYPES: dict[str, str] = {
    "chat": "chat",
    "text": "embedding",
    "image": "image_embedding",
    "action": "action",
    "description": "page_describe_and_state",
}


@pytest.mark.parametrize("operation", ["chat", "text", "image", "action", "description"])
def test_a_429_is_sent_again_after_a_short_backoff_and_logged_with_its_call_type(
    operation: str,
    router_client: OpenRouter,
    router_requests: list[httpx.Request],
    router_responses: list[httpx.Response],
    embedding_body: dict[str, object],
    png_screenshot: str,
    no_sleep: list[float],
    chat_response: Callable[..., httpx.Response],
    tracked_usage: vlm.TokenTracker,
    caplog: pytest.LogCaptureFixture,
) -> None:
    router_responses.extend(
        [
            httpx.Response(429, json={"error": {"code": 429, "message": "rate limited"}}),
            httpx.Response(200, json=embedding_body)
            if operation in {"text", "image"}
            else chat_response(content=_GOOD_REPLIES[operation]),
        ]
    )
    with caplog.at_level("WARNING"):
        assert _request(operation, router_client, png_screenshot) == _RESULTS[operation]
    assert len(router_requests) == 2
    assert router_requests[1].content == router_requests[0].content
    assert len(no_sleep) == 1
    assert 0.5 <= no_sleep[0] <= 1.5
    assert re.search(
        rf"{_CALL_TYPES[operation]}: attempt 1 was rate limited \(HTTP 429\) after \d+\.\d s; "
        rf"sending again in {no_sleep[0]:.1f} s",
        caplog.text,
    )
    assert tracked_usage.snapshot()["calls"] == 1


def test_a_503_is_sent_again_after_a_short_backoff(
    router_client: OpenRouter,
    router_requests: list[httpx.Request],
    router_responses: list[httpx.Response],
    png_screenshot: str,
    no_sleep: list[float],
    chat_response: Callable[..., httpx.Response],
    tracked_usage: vlm.TokenTracker,
) -> None:
    router_responses.extend(
        [
            httpx.Response(503, json={"error": {"code": 503, "message": "busy"}}),
            chat_response(content=_GOOD_REPLIES["chat"]),
        ]
    )
    assert _request("chat", router_client, png_screenshot) == _RESULTS["chat"]
    assert len(router_requests) == 2
    assert len(no_sleep) == 1
    assert 0.5 <= no_sleep[0] <= 1.5
    assert tracked_usage.snapshot()["calls"] == 1


@pytest.mark.parametrize("failure", ["timeout", "dropped"])
def test_a_timeout_or_a_dropped_connection_is_sent_again_until_it_recovers(
    failure: str, chat_body: dict[str, object], no_sleep: list[float]
) -> None:
    attempts: list[httpx.Request] = []

    def flaky(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        if len(attempts) == 1 and failure == "timeout":
            msg = "provider timed out"
            raise httpx.ReadTimeout(msg, request=request)
        if len(attempts) == 1:
            msg = "connection reset"
            raise httpx.ConnectError(msg, request=request)
        return httpx.Response(200, json=chat_body)

    with (
        httpx.Client(transport=httpx.MockTransport(flaky)) as http_client,
        open_client("test-key", http_client=http_client) as client,
    ):
        reply = vlm.send_chat(
            client,
            call_type="chat",
            settings=OTHER,
            messages=[{"role": "user", "content": "hi"}],
        )
    assert vlm.message_text(reply) == "test reply"
    assert len(attempts) == 2
    assert len(no_sleep) == 1


def _recording[**P, R](calls: list[dict[str, object]], method: Callable[P, R]) -> Callable[P, R]:
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        calls.append(dict(kwargs))
        return method(*args, **kwargs)

    return wrapper


def _send_every_model_call(
    client: OpenRouter,
    screenshot: str,
    settings: dict[str, ModelSettings],
    responses: list[httpx.Response],
    reply: Callable[..., httpx.Response],
    embedding_body: dict[str, object],
) -> None:
    """Make one call of every call type, each with its role's ``settings``."""
    responses.extend(
        [
            reply(content='{"page_description":"home"}'),
            reply(content='{"same":true}'),
            reply(content="{}"),
            reply(content="{}"),
            reply(content="{}"),
            reply(content="{}"),
            reply(content="{}"),
            reply(content="Tap Search"),
            reply(content='{"action":"wait"}'),
            httpx.Response(200, json=embedding_body),
            httpx.Response(200, json=embedding_body),
        ]
    )
    page_detail = settings["page_detail"]
    shown = vlm.DescribedScreenshot(screenshot, "home")
    vlm.describe_page_and_state(client, screenshot, settings=page_detail)
    vlm.verify_same_node(client, existing=shown, new=shown, settings=page_detail)
    vlm.normalize_edge(client, ["search milk"], settings=page_detail)
    vlm.audit_merge_nodes(client, "home", settings=page_detail)
    vlm.select_exploration_target(client, "demo", [{"node_id": "x"}], settings=page_detail)
    vlm.audit_graph(client, "home", settings=page_detail)
    vlm.describe_executed_action(
        client, screenshot, {"action": "back"}, (1, 1), settings=settings["instruction"]
    )
    vlm.plan_next_action(
        client,
        vlm.PlanRequest(
            screenshot_b64=screenshot, page_description="home", node="s0", explored_edges=[]
        ),
        settings=settings["instruction"],
    )
    vlm.predict_next_action(
        client,
        vlm.ActionRequest(screenshot_b64=screenshot, instruction="wait"),
        settings=settings["action"],
    )
    vlm.get_embedding(client, "home", model=settings["embedding"].model)
    vlm.get_image_embedding(client, screenshot, model=settings["image_embedding"].model)


def test_each_call_takes_its_call_types_own_timeout_when_its_role_sets_none(
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    png_screenshot: str,
    chat_response: Callable[..., httpx.Response],
    embedding_body: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The timeouts README.md#requests-and-recovery states, in the order of
    ``_send_every_model_call``: describe, verifier, normalization, merge audit, target
    choice, graph audit, relabelling, planner, action, text and image embeddings."""
    sent: list[dict[str, object]] = []
    monkeypatch.setattr(Chat, "send", _recording(sent, Chat.send))
    monkeypatch.setattr(Embeddings, "generate", _recording(sent, Embeddings.generate))
    roles = ("page_detail", "embedding", "image_embedding", "instruction", "action")
    settings = resolve_model_settings(
        {role: {"model": f"provider/{role}"} for role in roles}, roles=roles
    )
    _send_every_model_call(
        router_client, png_screenshot, settings, router_responses, chat_response, embedding_body
    )
    assert [kwargs["timeout_ms"] for kwargs in sent] == [
        55_000,
        35_000,
        50_000,
        300_000,
        25_000,
        300_000,
        70_000,
        70_000,
        35_000,
        15_000,
        15_000,
    ]


def test_a_role_s_configured_timeout_is_the_timeout_of_every_one_of_its_calls(
    router_client: OpenRouter,
    router_requests: list[httpx.Request],
    router_responses: list[httpx.Response],
    png_screenshot: str,
    chat_response: Callable[..., httpx.Response],
    embedding_body: dict[str, object],
) -> None:
    """The chat calls; GraphManager hands the embedding calls their role's timeout."""
    config: dict[str, object] = {
        "page_detail": {"model": "provider/page", "request_timeout": 12},
        "instruction": {"model": "provider/instruction", "timeout": 13},
        "action": {"model": "provider/action", "request_timeout": 14},
        "embedding": {"model": "provider/embedding"},
    }
    roles = ("page_detail", "embedding", "image_embedding", "instruction", "action")
    settings = resolve_model_settings(config, roles=roles)
    _send_every_model_call(
        router_client, png_screenshot, settings, router_responses, chat_response, embedding_body
    )
    reads = [request.extensions["timeout"]["read"] for request in router_requests]
    assert reads[:9] == [12, 12, 12, 12, 12, 12, 13, 13, 14]


@pytest.mark.parametrize("operation", ["audit", "planner", "action"])
def test_a_reply_without_choices_keeps_parse_fallbacks(
    operation: str,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    chat_body: dict[str, object],
    png_screenshot: str,
) -> None:
    chat_body["choices"] = []
    chat_body.pop("usage")
    router_responses.extend(
        httpx.Response(200, json=chat_body) for _ in range(2 if operation == "action" else 1)
    )
    if operation == "audit":
        assert vlm.audit_graph(router_client, "home", settings=PAGE_DETAIL) == {
            "issues": [],
            "summary": "parse error",
        }
    elif operation == "planner":
        assert vlm.plan_next_action(
            router_client,
            vlm.PlanRequest(
                screenshot_b64=png_screenshot, page_description="home", node="s0", explored_edges=[]
            ),
            settings=PAGE_DETAIL,
        ) == ("", None)
    else:
        assert vlm.predict_next_action(
            router_client,
            vlm.ActionRequest(screenshot_b64=png_screenshot, instruction="tap"),
            settings=ACTION,
        ) == (
            {"action": "wait", "time": 1},
            "Failed to parse action response -> wait",
        )
    assert router_responses == []
    assert len(router_requests) == (2 if operation == "action" else 1)


@pytest.mark.parametrize("operation", ["chat", "text", "image", "action"])
@pytest.mark.parametrize("recover", [False, True])
def test_sdk_validation_of_a_2xx_reply_is_re_asked_once(
    operation: str,
    recover: bool,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    chat_body: dict[str, object],
    embedding_body: dict[str, object],
    png_screenshot: str,
    no_sleep: list[float],
    chat_response: Callable[..., httpx.Response],
) -> None:
    good = embedding_body if operation in {"text", "image"} else chat_body
    missing = "system_fingerprint" if operation == "chat" else "model"
    bad = {key: value for key, value in good.items() if key != missing}
    router_responses.append(httpx.Response(200, json=bad))
    router_responses.append(
        httpx.Response(200, json=bad)
        if not recover
        else chat_response(content=_GOOD_REPLIES[operation])
        if operation == "action"
        else httpx.Response(200, json=good)
    )
    if recover:
        assert _request(operation, router_client, png_screenshot) == _RESULTS[operation]
    else:
        with pytest.raises(ResponseValidationError) as caught:
            _request(operation, router_client, png_screenshot)
        assert caught.value.status_code == 200
    assert len(router_requests) == 2
    assert router_requests[1].content == router_requests[0].content
    assert no_sleep == []


@pytest.mark.parametrize("recover", [False, True])
def test_an_empty_or_non_finite_vector_is_re_asked_once(
    recover: bool,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    embedding_body: dict[str, object],
    tracked_usage: vlm.TokenTracker,
) -> None:
    bad = {**embedding_body, "data": [{"object": "embedding", "index": 0, "embedding": []}]}
    router_responses.append(httpx.Response(200, json=bad))
    router_responses.append(httpx.Response(200, json=embedding_body if recover else bad))
    if recover:
        assert vlm.get_embedding(router_client, "home", model="google/gemini-embedding-2") == [
            0.25,
            0.75,
        ]
    else:
        with pytest.raises(UnusableOutputError, match="embedding"):
            vlm.get_embedding(router_client, "home", model="google/gemini-embedding-2")
    assert len(router_requests) == 2
    assert tracked_usage.snapshot()["calls"] == 2


def test_a_wrong_vector_count_is_not_re_asked(
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    embedding_body: dict[str, object],
) -> None:
    router_responses.append(httpx.Response(200, json={**embedding_body, "data": []}))
    with pytest.raises(ValueError, match="one embedding"):
        vlm.get_embedding(router_client, "home", model="google/gemini-embedding-2")
    assert len(router_requests) == 1


@pytest.mark.parametrize(
    ("malformed", "error", "message"),
    [
        ('{"page_description":', json.JSONDecodeError, "Expecting value"),
        ("[]", TypeError, "page description to be a JSON object"),
        ("{}", ValueError, "non-empty page_description"),
        ('{"page_description":""}', ValueError, "non-empty page_description"),
        ('{"page_description":"home","state":[]}', TypeError, "page state"),
        ('{"page_description":"home","elements":{}}', TypeError, "page elements"),
        ('{"page_description":"home","elements":[null]}', TypeError, "element description"),
        ('{"page_description":"home","is_external_web":"yes"}', TypeError, "is_external_web"),
    ],
)
def test_a_malformed_page_description_is_re_asked_once_then_raises(
    malformed: str,
    error: type[Exception],
    message: str,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    png_screenshot: str,
    chat_response: Callable[..., httpx.Response],
) -> None:
    router_responses.extend(chat_response(content=malformed) for _ in range(2))
    with pytest.raises(UnusableOutputError, match="Malformed page description") as caught:
        vlm.describe_page_and_state(router_client, png_screenshot, settings=PAGE_DETAIL)
    assert isinstance(caught.value.__cause__, error)
    assert message in str(caught.value.__cause__)
    assert len(router_requests) == 2


@pytest.mark.parametrize("operation", ["verify", "action", "description", "chat"])
@pytest.mark.parametrize("recover", [False, True])
def test_a_failed_generation_is_re_asked_once_and_never_becomes_a_fallback(
    operation: str,
    recover: bool,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    png_screenshot: str,
    chat_response: Callable[..., httpx.Response],
    tracked_usage: vlm.TokenTracker,
) -> None:
    good = {"verify": '{"same":true,"reason":"identical"}', **_GOOD_REPLIES}[operation]
    router_responses.append(chat_response(content="partial", finish_reason="error"))
    router_responses.append(
        chat_response(content=good) if recover else chat_response(content="", finish_reason="error")
    )

    def request() -> object:
        if operation == "verify":
            return _verify_itself(router_client, png_screenshot, "home")["same"]
        return _request(operation, router_client, png_screenshot)

    if recover:
        assert request() == {"verify": True, **_RESULTS}[operation]
    else:
        with pytest.raises(vlm.GenerationFailedError, match="finish_reason=error"):
            request()
    assert len(router_requests) == 2
    assert router_requests[1].content == router_requests[0].content
    assert tracked_usage.snapshot()["calls"] == 2


def _final_fallback(operation: str, client: OpenRouter, png_screenshot: str) -> object:
    match operation:
        case "verify":
            result: object = _verify_itself(client, png_screenshot, "home")
        case "normalize":
            normalized = vlm.normalize_edge(client, ["search milk"], settings=PAGE_DETAIL)
            result = {key: normalized[key] for key in ("instruction_template", "is_template")}
        case "merge_audit":
            result = vlm.audit_merge_nodes(client, "home", settings=PAGE_DETAIL)
        case "audit":
            result = vlm.audit_graph(client, "home", settings=PAGE_DETAIL)
        case "target":
            result = vlm.select_exploration_target(
                client, "demo", [{"node_id": "x"}], settings=PAGE_DETAIL
            )
        case "describe":
            result = vlm.describe_executed_action(
                client, png_screenshot, {"action": "back"}, (1, 1), settings=PAGE_DETAIL
            )
        case _:
            result = vlm.predict_next_action(
                client,
                vlm.ActionRequest(screenshot_b64=png_screenshot, instruction="tap"),
                settings=ACTION,
            )[0]
    return result


def _truncated_reply(chat_response: Callable[..., httpx.Response]) -> httpx.Response:
    response = chat_response(content='{"same": tr', finish_reason="length")
    body = response.json()
    body["usage"]["completion_tokens_details"] = {"reasoning_tokens": 2950}
    return httpx.Response(200, json=body)


@pytest.mark.parametrize(
    ("operation", "fallback"),
    [
        ("verify", {"same": False, "answered": False, "reason": "reply truncated"}),
        ("normalize", {"instruction_template": "search milk", "is_template": False}),
        ("merge_audit", {"issues": [], "summary": "reply truncated"}),
        ("audit", {"issues": [], "summary": "reply truncated"}),
        ("target", None),
        ("describe", None),
        ("action", {"action": "wait", "time": 1}),
    ],
)
def test_a_truncated_reply_is_never_re_asked_and_takes_the_final_fallback(
    operation: str,
    fallback: object,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    png_screenshot: str,
    chat_response: Callable[..., httpx.Response],
    caplog: pytest.LogCaptureFixture,
) -> None:
    router_responses.append(_truncated_reply(chat_response))
    with caplog.at_level("WARNING"):
        result = _final_fallback(operation, router_client, png_screenshot)
    assert result == fallback
    assert len(router_requests) == 1
    assert "finish_reason=length, reasoning_tokens=2950" in caplog.text


def test_normalize_edge_pairs_each_instruction_with_its_own_observation(
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    chat_response: Callable[..., httpx.Response],
) -> None:
    router_responses.append(_truncated_reply(chat_response))

    normalized = vlm.normalize_edge(
        router_client,
        ["", "search milk", "search eggs"],
        target_observations=["unpaired", "milk results", ""],
        settings=PAGE_DETAIL,
    )

    prompt = json.loads(router_requests[0].content)["messages"][-1]["content"]
    assert prompt.endswith(
        "Instructions to normalize:\n"
        '- Instruction: "search milk" → Observation: "milk results"\n'
        '- Instruction: "search eggs"\n'
    )
    assert normalized["instruction_template"] == "search milk"
    assert normalized["observation_template"] == "milk results"


def test_normalize_edge_sends_the_20_most_recent_distinct_instructions_and_counts_the_rest(
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    chat_response: Callable[..., httpx.Response],
) -> None:
    """An instruction recorded again is recent, with the observation it led to last."""
    router_responses.append(_truncated_reply(chat_response))

    vlm.normalize_edge(
        router_client,
        [f"open row {n}" for n in range(25)] + ["open row 3"],
        target_observations=[f"row {n}" for n in range(25)] + ["row 3 again"],
        settings=PAGE_DETAIL,
    )

    prompt = json.loads(router_requests[0].content)["messages"][-1]["content"]
    assert prompt.endswith(
        "Instructions to normalize (the 20 most recent; 5 older ones are not shown):\n"
        + "".join(f'- Instruction: "open row {n}" → Observation: "row {n}"\n' for n in range(6, 25))
        + '- Instruction: "open row 3" → Observation: "row 3 again"\n'
    )


def test_the_edge_template_schema_asks_whether_the_instructions_are_different_actions(
    router_client: OpenRouter,
    router_requests: list[httpx.Request],
    router_responses: list[httpx.Response],
    png_screenshot: str,
    chat_response: Callable[..., httpx.Response],
) -> None:
    schema = _sent_schemas(
        router_client, router_requests, router_responses, png_screenshot, chat_response
    )["edge_template"]["schema"]
    assert schema["properties"]["different_actions"]["type"] == "boolean"
    assert "different_actions" in schema["required"]


@pytest.mark.parametrize("operation", ["description", "planner"])
def test_a_truncated_reply_without_a_fallback_raises_at_once(
    operation: str,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    png_screenshot: str,
    chat_response: Callable[..., httpx.Response],
) -> None:
    router_responses.append(chat_response(content="", finish_reason="length"))

    def request() -> object:
        if operation == "description":
            return vlm.describe_page_and_state(router_client, png_screenshot, settings=PAGE_DETAIL)
        return vlm.plan_next_action(
            router_client,
            vlm.PlanRequest(
                screenshot_b64=png_screenshot, page_description="home", node="s0", explored_edges=[]
            ),
            settings=PAGE_DETAIL,
        )

    with pytest.raises(vlm.TruncatedReplyError, match="token limit"):
        request()
    assert len(router_requests) == 1


@pytest.mark.parametrize(
    ("operation", "call_type", "fallback"),
    [
        (
            "normalize",
            "normalize_edge",
            {"instruction_template": "search milk", "is_template": False},
        ),
        ("merge_audit", "node_merge_audit", {"issues": [], "summary": "reply truncated"}),
        ("audit", "graph_audit", {"issues": [], "summary": "reply truncated"}),
        ("target", "exploration_target", None),
        ("describe", "action_description", None),
        ("action", "action", {"action": "wait", "time": 1}),
    ],
)
def test_a_request_that_still_times_out_takes_the_truncated_reply_fallback(
    operation: str,
    call_type: str,
    fallback: object,
    timing_out_client: OpenRouter,
    timed_out_requests: list[httpx.Request],
    png_screenshot: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("WARNING"):
        result = _final_fallback(operation, timing_out_client, png_screenshot)
    assert result == fallback
    assert re.search(
        rf"{call_type}: attempt 2 timed out after \d+\.0 s; its deadline has passed", caplog.text
    )
    assert len(timed_out_requests) == 2


def test_node_verification_that_times_out_is_asked_once_more_then_has_no_answer(
    timing_out_client: OpenRouter,
    timed_out_requests: list[httpx.Request],
    png_screenshot: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("WARNING"):
        verdict = _verify_itself(timing_out_client, png_screenshot, "home")
    assert verdict == {"same": False, "answered": False, "reason": "timed out"}
    assert (
        caplog.text.count("node_verify: attempt 2 timed out after 36.0 s; its deadline has passed")
        == 2
    )
    assert "node_verify: asking once more" in caplog.text
    assert "node_verify: no answer (timed out)" in caplog.text
    assert len(timed_out_requests) == 4
    assert timed_out_requests[2].content == timed_out_requests[0].content


def test_node_verification_takes_the_answer_to_its_second_ask(
    monkeypatch: pytest.MonkeyPatch,
    no_sleep: list[float],
    png_screenshot: str,
    chat_response: Callable[..., httpx.Response],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The first ask times out on both of its attempts, each a second past the verifier's
    35 s timeout; the second ask answers."""
    del no_sleep  # requested for its side effect on time.sleep
    attempts: list[httpx.Request] = []

    def answer_late(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        if len(attempts) <= 2:
            msg = "no complete response within 35 s"
            raise httpx.ReadTimeout(msg, request=request)
        return chat_response(content='{"same":true,"reason":"same list"}')

    monkeypatch.setattr(time, "time", lambda: 36.0 * len(attempts))
    monkeypatch.setattr(time, "monotonic", lambda: 36.0 * len(attempts))
    with (
        caplog.at_level("WARNING"),
        httpx.Client(transport=httpx.MockTransport(answer_late)) as http_client,
        open_client("test-key", http_client=http_client) as client,
    ):
        verdict = _verify_itself(client, png_screenshot, "home")
    assert verdict == {"same": True, "reason": "same list"}
    assert len(attempts) == 3
    assert "node_verify: attempt 2 timed out after 36.0 s; its deadline has passed" in caplog.text
    assert "node_verify: asking once more" in caplog.text


def test_each_node_verification_ask_sends_a_429_again_within_a_deadline_of_its_own(
    monkeypatch: pytest.MonkeyPatch, png_screenshot: str, caplog: pytest.LogCaptureFixture
) -> None:
    """Each ask is answered 429 with a ``Retry-After`` that ends at its 70 s deadline, twice
    the verifier's 35 s timeout, and the request sent again then times out at 35 s, so no
    answer comes after 210 s."""
    elapsed: list[float] = [0.0]
    attempts: list[httpx.Request] = []

    def rate_limit_then_time_out(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        if len(attempts) % 2:
            return httpx.Response(
                429,
                json={"error": {"code": 429, "message": "rate limited"}},
                headers={"retry-after": "70"},
            )
        elapsed[0] += 35
        msg = "no complete response within 35 s"
        raise httpx.ReadTimeout(msg, request=request)

    def sleep(seconds: float) -> None:
        elapsed[0] += seconds

    monkeypatch.setattr(time, "monotonic", lambda: elapsed[0])
    monkeypatch.setattr(time, "time", lambda: elapsed[0])
    monkeypatch.setattr(time, "sleep", sleep)
    with (
        caplog.at_level("WARNING"),
        httpx.Client(transport=httpx.MockTransport(rate_limit_then_time_out)) as http_client,
        open_client("test-key", http_client=http_client) as client,
    ):
        verdict = _verify_itself(client, png_screenshot, "home")
    assert verdict == {"same": False, "answered": False, "reason": "timed out"}
    assert len(attempts) == 4
    rate_limited = "node_verify: attempt 1 was rate limited (HTTP 429) after 0.0 s"
    assert caplog.text.count(f"{rate_limited}; sending again in 70.0 s") == 2
    assert "node_verify: asking once more" in caplog.text
    assert elapsed[0] == 210


def test_node_verification_with_an_unparseable_reply_has_no_answer(
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    png_screenshot: str,
    chat_response: Callable[..., httpx.Response],
    caplog: pytest.LogCaptureFixture,
) -> None:
    router_responses.append(chat_response(content="The screens look the same."))
    with caplog.at_level("WARNING"):
        verdict = _verify_itself(router_client, png_screenshot, "home")
    assert verdict == {"same": False, "answered": False, "reason": "parse error"}
    assert "node_verify: no answer (parse error)" in caplog.text
    assert len(router_requests) == 1


def test_a_page_description_that_still_times_out_raises(
    timing_out_client: OpenRouter, png_screenshot: str
) -> None:
    with pytest.raises(httpx.ReadTimeout, match="no complete response"):
        vlm.describe_page_and_state(timing_out_client, png_screenshot, settings=PAGE_DETAIL)


def test_a_content_filtered_reply_is_logged_by_name_and_handled_as_partial_content(
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    png_screenshot: str,
    chat_response: Callable[..., httpx.Response],
    caplog: pytest.LogCaptureFixture,
) -> None:
    router_responses.append(chat_response(content="Tap", finish_reason="content_filter"))
    with caplog.at_level("WARNING"):
        assert vlm.plan_next_action(
            router_client,
            vlm.PlanRequest(
                screenshot_b64=png_screenshot, page_description="home", node="s0", explored_edges=[]
            ),
            settings=PAGE_DETAIL,
        ) == ("Tap", None)
    assert "instruction: the provider's content filter stopped the reply" in caplog.text
    assert len(router_requests) == 1


def test_the_planner_numbers_the_unexplored_elements_and_returns_the_one_it_names(
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    png_screenshot: str,
    chat_response: Callable[..., httpx.Response],
) -> None:
    instruction = "Tap the floating action button in the bottom right"
    router_responses.append(
        chat_response(content=json.dumps({"element": 2, "instruction": instruction}))
    )
    assert vlm.plan_next_action(
        router_client,
        vlm.PlanRequest(
            screenshot_b64=png_screenshot,
            page_description="home",
            node="s0",
            explored_edges=[],
            unexplored_elements=[
                {"description": "Turn on dark mode toggle", "position": "top"},
                {"description": "Compose FAB", "position": "bottom right"},
            ],
        ),
        settings=PAGE_DETAIL,
    ) == (instruction, 2)
    prompt = json.loads(router_requests[0].content)["messages"][-1]["content"][0]["text"]
    assert "1. Turn on dark mode toggle (top)\n2. Compose FAB (bottom right)\n" in prompt


def test_the_planner_says_where_each_tried_action_led_by_its_own_target_observation(
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    png_screenshot: str,
    chat_response: Callable[..., httpx.Response],
) -> None:
    """An edge with instructions lists the actions that have one, and an edge without
    lists every action, each with the observation it recorded."""
    router_responses.append(chat_response(content='{"element":null,"instruction":"Go back"}'))
    explored_edges: list[dict[str, Any]] = [
        {
            "target": "s1_network",
            "actions": [{"action": "tap"}, {"action": "swipe"}],
            "instructions": ["Tap Wi-Fi", ""],
            "target_observations": ["Wi-Fi details (ssid=home)", "Network list"],
        },
        {
            "target": "s2_about",
            "actions": [{"action": "back"}],
            "instructions": [""],
            "target_observations": ["About phone"],
        },
    ]

    vlm.plan_next_action(
        router_client,
        vlm.PlanRequest(
            screenshot_b64=png_screenshot,
            page_description="Settings",
            node="s0_settings",
            explored_edges=explored_edges,
        ),
        settings=PAGE_DETAIL,
    )

    prompt = json.loads(router_requests[0].content)["messages"][-1]["content"][0]["text"]
    tried = prompt.split("where they led:\n", 1)[1].split("\n\n", 1)[0]
    assert tried.splitlines() == [
        '- "Tap Wi-Fi" → led to: "Wi-Fi details (ssid=home)"',
        '- {"action": "back"} → led to: "About phone"',
    ]


@pytest.mark.parametrize(
    ("reply", "instruction"),
    [
        ('{"element": null, "instruction": "Type \'stewed beef\'"}', "Type 'stewed beef'"),
        ('{"element": null, "instruction": "Tap \\"Save\\""}', 'Tap "Save"'),
        ('"Tap Search"', '"Tap Search"'),
        ("'Type 'stewed beef''", "'Type 'stewed beef''"),
    ],
)
def test_the_planner_keeps_every_quote_in_its_instruction(
    reply: str,
    instruction: str,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    png_screenshot: str,
    chat_response: Callable[..., httpx.Response],
) -> None:
    router_responses.append(chat_response(content=reply))
    assert vlm.plan_next_action(
        router_client,
        vlm.PlanRequest(
            screenshot_b64=png_screenshot, page_description="home", node="s0", explored_edges=[]
        ),
        settings=PAGE_DETAIL,
    ) == (instruction, None)


@pytest.mark.parametrize(
    ("reply", "instruction"),
    [
        ('{"instruction": "\\"Swipe up\\" on the results list"}', '"Swipe up" on the results list'),
        ('{"instruction": " "}', None),
        ('["Swipe up"]', None),
        ("Swipe up on the results list", None),
    ],
    ids=["quotes kept", "blank", "not an object", "not JSON"],
)
@pytest.mark.parametrize(
    "action",
    [
        [{"action": "swipe", "x1": 540, "y1": 1500, "x2": 540, "y2": 500}],
        {"action": "menu"},
    ],
)
def test_an_executed_action_is_described_from_its_screen_or_not_at_all(
    reply: str,
    instruction: str | None,
    action: dict[str, object] | list[dict[str, object]],
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    png_screenshot: str,
    chat_response: Callable[..., httpx.Response],
) -> None:
    router_responses.append(chat_response(content=reply))

    described = vlm.describe_executed_action(
        router_client, png_screenshot, action, (1080, 1920), settings=PAGE_DETAIL
    )

    assert described == instruction
    prompt = json.loads(router_requests[0].content)["messages"][-1]["content"][0]["text"]
    assert "1080x1920 screen" in prompt
    assert json.dumps(action) in prompt
    assert '"menu" opens the application options menu' in prompt


@pytest.mark.parametrize("multipart", [False, True])
@pytest.mark.parametrize("native", [False, True])
def test_action_text_observations_survive_native_calls(
    native: bool,
    multipart: bool,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    png_screenshot: str,
    chat_response: Callable[..., httpx.Response],
) -> None:
    text = (
        'Thought: Search is visible\nAction: Tap Search\n<tool_call>{"action":"wait"}</tool_call>'
    )
    router_responses.append(
        chat_response(
            content=[{"type": "text", "text": text}] if multipart else text,
            tool_arguments=_CLICK if native else None,
        )
    )
    action, history = vlm.predict_next_action(
        router_client,
        vlm.ActionRequest(
            screenshot_b64=png_screenshot, instruction="tap", screen_w=100, screen_h=200
        ),
        settings=ACTION,
    )
    assert history == "Search is visible | Tap Search"
    assert action == (
        {"action": "tap", "x": 50, "y": 100} if native else {"action": "wait", "time": 3}
    )
    assert len(router_requests) == 1


_UP_SWIPE = {"action": "swipe", "x1": 50, "y1": 150, "x2": 50, "y2": 50, "duration": 1000}


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ({"action": "click", "coordinate": [500.5, 300.5]}, {"action": "tap", "x": 50, "y": 60}),
        (
            {"action": "long_press", "coordinate": [500, 300]},
            {"action": "long_press", "x": 50, "y": 60, "duration": 1000},
        ),
        (
            {"action": "long_press", "coordinate": [500, 300], "time": 250, "duration": 750},
            {"action": "long_press", "x": 50, "y": 60, "duration": 250},
        ),
        (
            {
                "action": "swipe",
                "coordinate": [500, 300],
                "coordinate2": [250, 800],
                "duration": 750,
            },
            {"action": "swipe", "x1": 50, "y1": 60, "x2": 25, "y2": 160, "duration": 750},
        ),
        ({"action": "swipe", "direction": "up"}, _UP_SWIPE),
        ({"action": "swipe", "direction": "up", "coordinate": [500, 300]}, _UP_SWIPE),
        (
            {"action": "swipe", "direction": "down"},
            {"action": "swipe", "x1": 50, "y1": 50, "x2": 50, "y2": 150, "duration": 1000},
        ),
        (
            {"action": "swipe", "direction": "left"},
            {"action": "swipe", "x1": 80, "y1": 100, "x2": 20, "y2": 100, "duration": 1000},
        ),
        (
            {"action": "swipe", "direction": "right"},
            {"action": "swipe", "x1": 20, "y1": 100, "x2": 80, "y2": 100, "duration": 1000},
        ),
        ({"action": "end"}, {"action": "end"}),
        ({"action": "terminate"}, {"action": "end"}),
    ],
)
def test_valid_coordinate_actions_and_intentional_end_remain_single_requests(
    args: dict[str, object],
    expected: dict[str, object],
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    png_screenshot: str,
    no_sleep: list[float],
    chat_response: Callable[..., httpx.Response],
) -> None:
    router_responses.append(chat_response(content=f"<tool_call>{json.dumps(args)}</tool_call>"))
    action, _ = vlm.predict_next_action(
        router_client,
        vlm.ActionRequest(
            screenshot_b64=png_screenshot, instruction="act", screen_w=100, screen_h=200
        ),
        settings=ACTION,
    )
    assert action == expected
    assert len(router_requests) == 1
    assert no_sleep == []


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ({"action": "swipe", "direction": "north"}, "Swipe requires a supported direction"),
        ({"action": "type"}, "Type requires a text string"),
        ({"action": "system_button", "button": "Power"}, "Back, Home, Menu or Enter"),
        ({"action": "invented"}, "Unknown action"),
        ({"action": "click"}, "two finite numbers"),
        ({"action": "click", "coordinate": [500]}, "two finite numbers"),
        ({"action": "click", "coordinate": [float("nan"), 300]}, "two finite numbers"),
        ({"action": "swipe", "coordinate": [500, 300]}, "coordinates or a supported direction"),
        ({"action": "swipe", "coordinate": [500, 300], "coordinate2": [500]}, "two finite numbers"),
        ({"action": "swipe", "direction": "north", "coordinate": [500]}, "two finite numbers"),
        ({"action": ["click"]}, "Unknown action"),
        ({"action": "swipe", "direction": ["up"]}, "Swipe requires a supported direction"),
        ({"action": "system_button", "button": ["Back"]}, "Back, Home, Menu or Enter"),
    ],
)
def test_malformed_tool_arguments_are_rejected(args: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        vlm.tool_call_to_action(args)


def test_direction_does_not_hide_invalid_partial_coordinates() -> None:
    with pytest.raises(ValueError, match="coordinates"):
        vlm.tool_call_to_action({"action": "swipe", "direction": "up", "coordinate": [500]})


@pytest.mark.parametrize(
    "message",
    [
        {"content": ""},
        {"content": "not JSON"},
        {"content": "[]"},
        {"content": '{"name":"mobile_use","arguments":[]}'},
        {"content": "", "tool_arguments": "not JSON"},
        {"content": "", "tool_arguments": "[]"},
    ],
)
def test_agent_reply_without_an_arguments_object_is_unusable(
    message: dict[str, str], chat_response: Callable[..., httpx.Response]
) -> None:
    reply = ChatResult.model_validate(chat_response(**message).json())
    assert vlm.parse_agent_response(reply) == (None, "", "")


def test_raw_json_tool_call_is_unwrapped_to_its_arguments(
    chat_response: Callable[..., httpx.Response],
) -> None:
    reply = ChatResult.model_validate(
        chat_response(content='{"name":"mobile_use","arguments":{"action":"wait"}}').json()
    )
    assert vlm.parse_agent_response(reply) == ({"action": "wait"}, "", "")


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ({"action": "type", "text": "milk"}, {"action": "type", "text": "milk"}),
        ({"action": "system_button", "button": "Back"}, {"action": "back"}),
        ({"action": "system_button", "button": "Home"}, {"action": "home"}),
        ({"action": "system_button", "button": "Menu"}, {"action": "menu"}),
        ({"action": "system_button", "button": "Enter"}, {"action": "enter"}),
        ({"action": "answer", "text": "9 EUR"}, {"action": "end"}),
        ({"action": "error"}, {"action": "end"}),
        ({"action": "wait"}, {"action": "wait", "time": 3}),
        ({"action": "wait", "duration": 2}, {"action": "wait", "time": 2}),
        ({"action": "wait", "time": "5"}, {"action": "wait", "time": "5"}),
        (
            {"name": "mobile_use", "arguments": {"action": "click", "coordinate": [500, 500]}},
            {"action": "tap", "x": 540, "y": 960},
        ),
    ],
)
def test_tool_arguments_convert_to_device_actions(
    args: dict[str, object], expected: dict[str, object]
) -> None:
    assert vlm.tool_call_to_action(args) == expected


_XML_CLICK = """\
Thought: The search icon is at the top.
Action: Tap the search icon.
<tool_call>
<function=mobile_use>
<parameter=action>
click
</parameter>
<parameter=coordinate>
[752, 170]
</parameter>
</tool_call>"""
_NO_JSON = "it had no <tool_call> block with valid JSON"
_XML_TAGS = "the <tool_call> block used <function=...> tags instead of JSON"


@pytest.mark.parametrize(
    ("reply", "reason"),
    [
        (_XML_CLICK, _XML_TAGS),
        (_XML_CLICK.removesuffix("\n</tool_call>"), _XML_TAGS),
        ("Thought: Search is visible\nAction: Tap Search", _NO_JSON),
        ('<tool_call>{"action": click}</tool_call>', _NO_JSON),
        ("<tool_call>5</tool_call>", _NO_JSON),
        ("", _NO_JSON),
        (
            '<tool_call>{"name": "mobile_use", "arguments": {"action": "type"}}</tool_call>',
            "the mobile_use arguments were invalid: Type requires a text string",
        ),
    ],
)
def test_an_unusable_action_reply_is_retried_with_what_was_wrong(
    reply: str,
    reason: str,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    png_screenshot: str,
    no_sleep: list[float],
    chat_response: Callable[..., httpx.Response],
) -> None:
    router_responses.append(chat_response(content=reply))
    router_responses.append(chat_response(content=f"<tool_call>{_CLICK}</tool_call>"))
    action, _ = vlm.predict_next_action(
        router_client,
        vlm.ActionRequest(
            screenshot_b64=png_screenshot, instruction="tap", screen_w=100, screen_h=200
        ),
        settings=ACTION,
    )
    assert action == {"action": "tap", "x": 50, "y": 100}
    first, retry = (json.loads(request.content)["messages"] for request in router_requests)
    shown = [{"role": "assistant", "content": reply}] if reply else []
    assert retry == [
        *first,
        *shown,
        {
            "role": "user",
            "content": f"Your previous reply could not be used: {reason}. Reply again with "
            "Thought, Action, and one <tool_call> block containing only JSON: "
            '{"name": "mobile_use", "arguments": {...}}.',
        },
    ]
    assert no_sleep == []


def test_a_second_unusable_action_reply_falls_back_to_wait(
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    png_screenshot: str,
    no_sleep: list[float],
    chat_response: Callable[..., httpx.Response],
) -> None:
    router_responses.extend(chat_response(content="Tap Search") for _ in range(2))
    assert vlm.predict_next_action(
        router_client,
        vlm.ActionRequest(screenshot_b64=png_screenshot, instruction="tap"),
        settings=ACTION,
    ) == (
        {"action": "wait", "time": 1},
        "Failed to parse action response -> wait",
    )
    assert len(router_requests) == 2
    assert no_sleep == []


@pytest.mark.parametrize("recover", [False, True])
def test_action_retries_and_the_single_re_ask_are_separate(
    recover: bool,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    chat_body: dict[str, object],
    png_screenshot: str,
    no_sleep: list[float],
    chat_response: Callable[..., httpx.Response],
) -> None:
    bad = {key: value for key, value in chat_body.items() if key != "model"}
    router_responses.extend(
        [
            httpx.Response(503, json={"error": {"code": 503, "message": "busy"}}),
            httpx.Response(200, json=bad),
            chat_response(content='{"action":"click","coordinate":[500,300]}')
            if recover
            else httpx.Response(200, json=bad),
        ]
    )
    if recover:
        action, _ = vlm.predict_next_action(
            router_client,
            vlm.ActionRequest(screenshot_b64=png_screenshot, instruction="tap"),
            settings=ACTION,
        )
        assert action == {"action": "tap", "x": 540, "y": 576}
    else:
        with pytest.raises(ResponseValidationError):
            vlm.predict_next_action(
                router_client,
                vlm.ActionRequest(screenshot_b64=png_screenshot, instruction="tap"),
                settings=ACTION,
            )
    assert len(router_requests) == 3
    assert len(no_sleep) == 1


def test_text_planning_establishes_the_intended_field_before_typing(
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    png_screenshot: str,
    chat_response: Callable[..., httpx.Response],
) -> None:
    router_responses.append(chat_response(content='{"element":null,"instruction":"Tap Note"}'))
    vlm.plan_next_action(
        router_client,
        vlm.PlanRequest(
            screenshot_b64=png_screenshot, page_description="Options", node="s0", explored_edges=[]
        ),
        settings=PAGE_DETAIL,
    )
    prompt = json.loads(router_requests[0].content)["messages"][-1]["content"][0]["text"]
    assert "Keyboard visibility alone establishes no field identity or focus." in prompt
    assert "tap the intended field as a separate action, then observe before typing" in prompt
    assert "current screenshot description" in prompt
    assert "current caret or field highlight" in prompt
