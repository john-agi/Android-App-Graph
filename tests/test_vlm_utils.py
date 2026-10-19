"""Tests for android_app_graph.utils.vlm_utils.

Its cosine scoring is shared by graph_manager and aitk_translator, whose
node-retrieval tests exercise it end to end. Its OpenRouter helpers run here over
the real SDK and an offline transport: request serialization, response parsing,
action conversion, and the retry budget that HTTP, schema and parse failures share.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from hypothesis import given
from hypothesis import strategies as st
from openrouter import OpenRouter
from openrouter.chat import Chat
from openrouter.components.chatresult import ChatResult
from openrouter.components.chatusage import ChatUsage
from openrouter.embeddings import Embeddings
from openrouter.errors import OpenRouterError, ResponseValidationError
from openrouter.operations.createembeddings import CreateEmbeddingsUsage
from openrouter.types import UNSET

from android_app_graph.payloads import as_list, as_object_dict, as_str, as_str_dict
from android_app_graph.retrying import UnusableOutputError
from android_app_graph.utils import ModelSettings, open_client
from android_app_graph.utils import vlm_utils as vlm
from android_app_graph.utils.vlm_utils import cosine_similarity, score_by_cosine

PAGE_DETAIL = ModelSettings("openai/gpt-6.1-sol")
ACTION = ModelSettings("qwen/qwen3.8-flash")
OTHER = ModelSettings("provider/other")


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


_STRUCTURED: dict[str, tuple[str, bool]] = {
    "page": ("page_description", False),
    "verify": ("node_verification", True),
    "normalize": ("edge_template", False),
    "merge": ("node_merge_audit", True),
    "audit": ("graph_audit", True),
    "target": ("exploration_target", True),
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
            ("home", {"tab": "all"}, [{"description": "Search", "position": "top"}]),
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
            '```json\n{"instruction":"Tap Search"}\n```',
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
    png_screenshot: str,
    tracked_usage: vlm.TokenTracker,
    chat_response: Callable[..., httpx.Response],
    chat_usage_totals: dict[str, float],
) -> None:
    router_responses.append(chat_response(content=content))
    result: object
    match operation:
        case "page":
            result = vlm.describe_page_and_state(
                router_client, png_screenshot, settings=PAGE_DETAIL
            )
        case "verify":
            result = vlm.verify_same_node(
                router_client, png_screenshot, png_screenshot, "home", settings=PAGE_DETAIL
            )
        case "normalize":
            result = vlm.normalize_edge(router_client, ["search milk"], settings=PAGE_DETAIL)
        case "merge":
            result = vlm.audit_merge_nodes(router_client, "home", settings=PAGE_DETAIL)
        case "audit":
            result = vlm.audit_graph(router_client, "home", settings=PAGE_DETAIL)
        case "target":
            result = vlm.select_exploration_target(
                router_client, "demo", [{"node_id": "home"}], settings=PAGE_DETAIL
            )
        case "planner":
            result = vlm.plan_next_action(
                router_client,
                png_screenshot,
                "home",
                [],
                app_name="demo",
                settings=PAGE_DETAIL,
            )
        case _:
            result = vlm.predict_next_action(router_client, png_screenshot, "wait", settings=ACTION)
    assert result == expected
    body = json.loads(router_requests[0].content)
    assert body["model"] == model
    assert body["stream"] is False
    assert "temperature" not in body
    if operation in {"page", "verify", "planner", "action"}:
        parts = body["messages"][-1]["content"]
        images = [part for part in parts if part["type"] == "image_url"]
        assert len(images) == (2 if operation == "verify" else 1)
        assert images[0]["image_url"]["url"] == f"data:image/png;base64,{png_screenshot}"
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
    action, _ = vlm.predict_next_action(
        router_client, png_screenshot, "tap", screen_w=100, screen_h=200, settings=ACTION
    )
    assert action == {"action": "tap", "x": 50, "y": 100}


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
        png_screenshot,
        "wait",
        screen_w=1080,
        screen_h=2400,
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


def test_xml_parameter_values_that_are_not_json_stay_strings() -> None:
    raw = (
        "<tool_call>\n<function=mobile_use>\n<parameter=action>\ntype\n</parameter>\n"
        "<parameter=text>\nstewed beef\n</parameter>\n<parameter=time>\n2\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    assert vlm._parse_tool_call(raw) == {"action": "type", "text": "stewed beef", "time": 2}


def test_a_truncated_xml_parameter_is_dropped() -> None:
    raw = "<tool_call>\n<function=mobile_use>\n<parameter=action>\nclick\n</parameter>\n"
    raw += "<parameter=coordinate>\n[752,"
    assert vlm._parse_tool_call(raw) == {"action": "click"}


@pytest.mark.parametrize("truncated", [False, True])
def test_xml_parameter_tool_calls_become_device_actions(
    truncated: bool,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    png_screenshot: str,
    chat_response: Callable[..., httpx.Response],
) -> None:
    content = _XML_CLICK.removesuffix("</tool_call>") if truncated else _XML_CLICK
    router_responses.append(chat_response(content=content))
    action, history = vlm.predict_next_action(
        router_client,
        png_screenshot,
        "tap",
        screen_w=1000,
        screen_h=2000,
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
        if sorted(map(str, as_list(node.get("required")))) != sorted(properties):
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
        )
    )
    vlm.describe_page_and_state(router_client, png_screenshot, settings=PAGE_DETAIL)
    vlm.verify_same_node(
        router_client, png_screenshot, png_screenshot, "home", settings=PAGE_DETAIL
    )
    vlm.normalize_edge(router_client, ["search milk"], settings=PAGE_DETAIL)
    vlm.audit_merge_nodes(router_client, "home", settings=PAGE_DETAIL)
    vlm.audit_graph(router_client, "home", settings=PAGE_DETAIL)
    vlm.select_exploration_target(
        router_client, "demo", [{"node_id": "b"}, {"node_id": "a"}], settings=PAGE_DETAIL
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
        "exploration_target",
        "graph_audit",
        "node_merge_audit",
        "node_verification",
    ]
    for name, schema in strict.items():
        assert _strict_violations(schema["schema"]) == [], name
    assert schemas["exploration_target"]["schema"]["properties"]["node_id"]["enum"] == ["a", "b"]
    assert schemas["node_verification"]["schema"]["properties"]["new_description"]["type"] == [
        "string",
        "null",
    ]
    state = schemas["page_description"]["schema"]["properties"]["state"]
    assert state["additionalProperties"] == {"type": ["string", "number", "boolean", "null"]}
    examples = schemas["edge_template"]["schema"]["properties"]["examples"]
    assert examples["items"]["properties"]["params"]["additionalProperties"] == {"type": "string"}


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
def test_parsers_still_reject_replies_that_violate_the_schema(
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
        vlm.get_image_embedding(router_client, "shot", model="google/gemini-embedding-2")
        if image
        else vlm.get_embedding(router_client, "home", model="google/gemini-embedding-2")
    )
    assert vector == [0.25, 0.75]
    body = json.loads(router_requests[0].content)
    assert body["model"] == "google/gemini-embedding-2"
    assert body["encoding_format"] == "float"
    assert body["input"] == (
        [{"content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,shot"}}]}]
        if image
        else "home"
    )
    assert tracked_usage.snapshot() == embedding_usage_totals
    assert tracked_usage.snapshot_by_type() == {"image_embedding" if image else "embedding": 5}


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
    router_responses.append(
        httpx.Response(
            200, text=json.dumps(embedding_body), headers={"content-type": "application/json"}
        )
    )
    with pytest.raises(ValueError, match="embedding"):
        vlm.get_image_embedding(router_client, "shot", model="google/gemini-embedding-2")


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
        case "action":
            return vlm.predict_next_action(client, png_screenshot, "tap", settings=ACTION)[0]
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
    "action": {"action": "tap", "x": 540, "y": 960},
    "description": "home",
}


@pytest.mark.parametrize("operation", ["chat", "text", "action", "description"])
def test_a_429_is_retried_by_neither_the_sdk_nor_the_app(
    operation: str,
    router_client: OpenRouter,
    router_requests: list[httpx.Request],
    router_responses: list[httpx.Response],
    png_screenshot: str,
    no_sleep: list[float],
    tracked_usage: vlm.TokenTracker,
) -> None:
    router_responses.append(
        httpx.Response(429, json={"error": {"code": 429, "message": "upstream failed"}})
    )
    with pytest.raises(OpenRouterError, match="upstream failed"):
        _request(operation, router_client, png_screenshot)
    assert len(router_requests) == 1
    assert no_sleep == []
    assert tracked_usage.snapshot()["calls"] == 0


def test_the_sdk_retries_a_503_after_a_short_backoff(
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
def test_the_sdk_retries_a_timeout_or_a_dropped_connection_once_it_recovers(
    failure: str, chat_body: dict[str, object], no_sleep: list[float]
) -> None:
    attempts: list[httpx.Request] = []

    def flaky(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        if len(attempts) == 1 and failure == "timeout":
            raise httpx.ReadTimeout("provider timed out", request=request)
        if len(attempts) == 1:
            raise httpx.ConnectError("connection reset", request=request)
        return httpx.Response(200, json=chat_body)

    with httpx.Client(transport=httpx.MockTransport(flaky)) as http_client:
        reply = vlm.send_chat(
            open_client("test-key", http_client=http_client),
            call_type="chat",
            settings=OTHER,
            messages=[{"role": "user", "content": "hi"}],
        )
    assert vlm.message_text(reply) == "test reply"
    assert len(attempts) == 2
    assert len(no_sleep) == 1


def test_no_request_overrides_the_sdk_retry_policy(
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    png_screenshot: str,
    chat_response: Callable[..., httpx.Response],
    embedding_body: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert router_client.sdk_configuration.retry_config is UNSET
    sent: list[dict[str, object]] = []
    chat_send = Chat.send
    generate = Embeddings.generate

    def spy_send(self: Chat, **kwargs: Any) -> object:
        sent.append(kwargs)
        return chat_send(self, **kwargs)

    def spy_generate(self: Embeddings, **kwargs: Any) -> object:
        sent.append(kwargs)
        return generate(self, **kwargs)

    monkeypatch.setattr(Chat, "send", spy_send)
    monkeypatch.setattr(Embeddings, "generate", spy_generate)
    router_responses.extend(
        [
            chat_response(content='{"page_description":"home"}'),
            chat_response(content='{"same":true}'),
            chat_response(content="{}"),
            chat_response(content="{}"),
            chat_response(content="{}"),
            chat_response(content="{}"),
            chat_response(content="Tap Search"),
            chat_response(content='{"action":"wait"}'),
            httpx.Response(200, json=embedding_body),
            httpx.Response(200, json=embedding_body),
        ]
    )
    vlm.describe_page_and_state(router_client, png_screenshot, settings=PAGE_DETAIL)
    vlm.verify_same_node(
        router_client, png_screenshot, png_screenshot, "home", settings=PAGE_DETAIL
    )
    vlm.normalize_edge(router_client, ["search milk"], settings=PAGE_DETAIL)
    vlm.audit_merge_nodes(router_client, "home", settings=PAGE_DETAIL)
    vlm.select_exploration_target(router_client, "demo", [{"node_id": "x"}], settings=PAGE_DETAIL)
    vlm.audit_graph(router_client, "home", settings=PAGE_DETAIL)
    vlm.plan_next_action(router_client, png_screenshot, "home", [], settings=PAGE_DETAIL)
    vlm.predict_next_action(router_client, png_screenshot, "wait", settings=ACTION)
    vlm.get_embedding(router_client, "home", model="google/gemini-embedding-2")
    vlm.get_image_embedding(router_client, "shot", model="google/gemini-embedding-2")
    assert len(sent) == 10
    assert all("retries" not in kwargs and "timeout_ms" in kwargs for kwargs in sent)


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
        assert (
            vlm.plan_next_action(router_client, png_screenshot, "home", [], settings=PAGE_DETAIL)
            == ""
        )
    else:
        assert vlm.predict_next_action(router_client, png_screenshot, "tap", settings=ACTION) == (
            {"action": "wait", "time": 1},
            "Failed to parse action response -> wait",
        )
    assert router_responses == []
    assert len(router_requests) == (2 if operation == "action" else 1)


@pytest.mark.parametrize("limit", [None, 17])
def test_chat_omits_unspecified_token_limit_at_strict_endpoint(
    limit: int | None,
    chat_body: dict[str, object],
    no_sleep: list[float],
) -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        payload = json.loads(request.content)
        if "max_tokens" in payload and payload["max_tokens"] is None:
            return httpx.Response(400, json={"error": {"code": 400, "message": "null forbidden"}})
        return httpx.Response(200, json=chat_body)

    with httpx.Client(transport=httpx.MockTransport(respond)) as borrowed:
        vlm.send_chat(
            open_client("test", http_client=borrowed),
            call_type="chat",
            settings=PAGE_DETAIL,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=limit,
        )
    payload = json.loads(requests[0].content)
    if limit is None:
        assert "max_tokens" not in payload
    else:
        assert payload["max_tokens"] == limit
    assert len(requests) == 1
    assert no_sleep == []


@pytest.mark.parametrize("operation", ["chat", "text", "action"])
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
    good = embedding_body if operation == "text" else chat_body
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
            return vlm.verify_same_node(
                router_client, png_screenshot, png_screenshot, "home", settings=PAGE_DETAIL
            )["same"]
        return _request(operation, router_client, png_screenshot)

    if recover:
        assert request() == {"verify": True, **_RESULTS}[operation]
    else:
        with pytest.raises(vlm.GenerationFailedError, match="finish_reason=error"):
            request()
    assert len(router_requests) == 2
    assert router_requests[1].content == router_requests[0].content
    assert tracked_usage.snapshot()["calls"] == 2


def _truncated_reply(chat_response: Callable[..., httpx.Response]) -> httpx.Response:
    response = chat_response(content='{"same": tr', finish_reason="length")
    body = response.json()
    body["usage"]["completion_tokens_details"] = {"reasoning_tokens": 2950}
    return httpx.Response(200, json=body)


@pytest.mark.parametrize(
    ("operation", "fallback"),
    [
        ("verify", {"same": False, "reason": "reply truncated"}),
        ("normalize", {"instruction_template": "search milk", "is_template": False}),
        ("merge_audit", {"issues": [], "summary": "reply truncated"}),
        ("audit", {"issues": [], "summary": "reply truncated"}),
        ("target", None),
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
        match operation:
            case "verify":
                result: object = vlm.verify_same_node(
                    router_client, png_screenshot, png_screenshot, "home", settings=PAGE_DETAIL
                )
            case "normalize":
                normalized = vlm.normalize_edge(
                    router_client, ["search milk"], settings=PAGE_DETAIL
                )
                result = {key: normalized[key] for key in ("instruction_template", "is_template")}
            case "merge_audit":
                result = vlm.audit_merge_nodes(router_client, "home", settings=PAGE_DETAIL)
            case "audit":
                result = vlm.audit_graph(router_client, "home", settings=PAGE_DETAIL)
            case "target":
                result = vlm.select_exploration_target(
                    router_client, "demo", [{"node_id": "x"}], settings=PAGE_DETAIL
                )
            case _:
                result = vlm.predict_next_action(
                    router_client, png_screenshot, "tap", settings=ACTION
                )[0]
    assert result == fallback
    assert len(router_requests) == 1
    assert "finish_reason=length, reasoning_tokens=2950" in caplog.text


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
        return vlm.plan_next_action(router_client, png_screenshot, "home", [], settings=PAGE_DETAIL)

    with pytest.raises(vlm.TruncatedReplyError, match="token limit"):
        request()
    assert len(router_requests) == 1


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
        assert (
            vlm.plan_next_action(router_client, png_screenshot, "home", [], settings=PAGE_DETAIL)
            == "Tap"
        )
    assert "instruction: the provider's content filter stopped the reply" in caplog.text
    assert len(router_requests) == 1


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
        router_client, png_screenshot, "tap", screen_w=100, screen_h=200, settings=ACTION
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
        router_client, png_screenshot, "act", screen_w=100, screen_h=200, settings=ACTION
    )
    assert action == expected
    assert len(router_requests) == 1
    assert no_sleep == []


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ({"action": "swipe", "direction": "north"}, "Swipe requires a supported direction"),
        ({"action": "type"}, "Type requires a text string"),
        ({"action": "system_button", "button": "Power"}, "Back, Home or Enter"),
        ({"action": "invented"}, "Unknown action"),
        ({"action": "click"}, "two finite numbers"),
        ({"action": "click", "coordinate": [500]}, "two finite numbers"),
        ({"action": "click", "coordinate": [float("nan"), 300]}, "two finite numbers"),
        ({"action": "swipe", "coordinate": [500, 300]}, "coordinates or a supported direction"),
        ({"action": "swipe", "coordinate": [500, 300], "coordinate2": [500]}, "two finite numbers"),
    ],
)
def test_malformed_tool_arguments_are_rejected(args: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        vlm._tool_call_to_aitk(args)


def test_direction_does_not_hide_invalid_partial_coordinates() -> None:
    with pytest.raises(ValueError, match="coordinates"):
        vlm._tool_call_to_aitk({"action": "swipe", "direction": "up", "coordinate": [500]})


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
    assert vlm._parse_agent_response(reply) == (None, "", "")


def test_raw_json_tool_call_is_unwrapped_to_its_arguments(
    chat_response: Callable[..., httpx.Response],
) -> None:
    reply = ChatResult.model_validate(
        chat_response(content='{"name":"mobile_use","arguments":{"action":"wait"}}').json()
    )
    assert vlm._parse_agent_response(reply) == ({"action": "wait"}, "", "")


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ({"action": "type", "text": "milk"}, {"action": "type", "text": "milk"}),
        ({"action": "system_button", "button": "Back"}, {"action": "back"}),
        ({"action": "system_button", "button": "Home"}, {"action": "home"}),
        ({"action": "system_button", "button": "Enter"}, {"action": "enter"}),
        ({"action": "answer", "text": "9 EUR"}, {"action": "end"}),
        ({"action": "error"}, {"action": "end"}),
        (
            {"name": "mobile_use", "arguments": {"action": "click", "coordinate": [500, 500]}},
            {"action": "tap", "x": 540, "y": 960},
        ),
    ],
)
def test_tool_arguments_convert_to_aitk_actions(
    args: dict[str, object], expected: dict[str, object]
) -> None:
    assert vlm._tool_call_to_aitk(args) == expected


_NO_JSON = "it had no <tool_call> block with valid JSON"


@pytest.mark.parametrize(
    ("reply", "reason"),
    [
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
        router_client, png_screenshot, "tap", screen_w=100, screen_h=200, settings=ACTION
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


def test_a_function_tag_inside_the_tool_call_block_is_named_as_the_reason() -> None:
    assert (
        vlm._unparsed_reason("<tool_call>\n<function=mobile_use>\n<parameter=action>")
        == "the <tool_call> block used <function=...> tags instead of JSON"
    )


def test_a_second_unusable_action_reply_falls_back_to_wait(
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    png_screenshot: str,
    no_sleep: list[float],
    chat_response: Callable[..., httpx.Response],
) -> None:
    router_responses.extend(chat_response(content="Tap Search") for _ in range(2))
    assert vlm.predict_next_action(router_client, png_screenshot, "tap", settings=ACTION) == (
        {"action": "wait", "time": 1},
        "Failed to parse action response -> wait",
    )
    assert len(router_requests) == 2
    assert no_sleep == []


@pytest.mark.parametrize("recover", [False, True])
def test_action_sdk_retries_and_the_single_re_ask_are_separate(
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
        action, _ = vlm.predict_next_action(router_client, png_screenshot, "tap", settings=ACTION)
        assert action == {"action": "tap", "x": 540, "y": 576}
    else:
        with pytest.raises(ResponseValidationError):
            vlm.predict_next_action(router_client, png_screenshot, "tap", settings=ACTION)
    assert len(router_requests) == 3
    assert len(no_sleep) == 1
