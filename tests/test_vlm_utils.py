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

import httpx
import pytest
from hypothesis import given
from hypothesis import strategies as st
from openrouter import OpenRouter
from openrouter.components.chatresult import ChatResult
from openrouter.errors import OpenRouterError, ResponseValidationError

from android_app_graph.payloads import as_str, as_str_dict
from android_app_graph.utils import make_client
from android_app_graph.utils import vlm_utils as vlm
from android_app_graph.utils.vlm_utils import cosine_similarity, score_by_cosine


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
) -> None:
    router_responses.append(chat_response(content=content))
    result: object
    match operation:
        case "page":
            result = vlm.describe_page_and_state(
                router_client, png_screenshot, model="openai/gpt-6.1-sol"
            )
        case "verify":
            result = vlm.verify_same_node(
                router_client, png_screenshot, png_screenshot, "home", model="openai/gpt-6.1-sol"
            )
        case "normalize":
            result = vlm.normalize_edge(router_client, ["search milk"], model="openai/gpt-6.1-sol")
        case "merge":
            result = vlm.audit_merge_nodes(router_client, "home", model="openai/gpt-6.1-sol")
        case "audit":
            result = vlm.audit_graph(router_client, "home", model="openai/gpt-6.1-sol")
        case "target":
            result = vlm.select_exploration_target(
                router_client, "demo", [{"node_id": "home"}], model="openai/gpt-6.1-sol"
            )
        case "planner":
            result = vlm.plan_next_action(
                router_client,
                png_screenshot,
                "home",
                [],
                app_name="demo",
                model="openai/gpt-6.1-sol",
            )
        case _:
            result = vlm.predict_next_action(
                router_client, png_screenshot, "wait", model="qwen/qwen3.8-flash"
            )
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
    if operation not in {"planner", "action"}:
        assert body["response_format"] == {"type": "json_object"}
    assert tracked_usage.summary()[f"{call_type}:{model}"] == {
        "prompt_tokens": 7,
        "completion_tokens": 3,
        "total_tokens": 10,
        "calls": 1,
    }


@pytest.mark.parametrize("content", ["", "calling action"])
def test_native_tool_calls_survive_sdk_parsing(
    content: str,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    png_screenshot: str,
    chat_response: Callable[..., httpx.Response],
) -> None:
    router_responses.append(chat_response(content=content, tool_arguments=_CLICK))
    action, _ = vlm.predict_next_action(
        router_client, png_screenshot, "tap", screen_w=100, screen_h=200, model="qwen/qwen3.8-flash"
    )
    assert action == {"action": "tap", "x": 50, "y": 100}


def _sent_system_prompt(request: httpx.Request) -> str:
    system = as_str_dict(json.loads(request.content)["messages"][0])
    assert system["role"] == "system"
    return as_str(system["content"], "")


def test_action_prompt_declares_a_parseable_mobile_use_function(
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    png_screenshot: str,
    chat_response: Callable[..., httpx.Response],
) -> None:
    router_responses.append(chat_response(content='<tool_call>{"action":"wait"}</tool_call>'))
    vlm.predict_next_action(router_client, png_screenshot, "wait", model="qwen/qwen3.8-flash")
    prompt = _sent_system_prompt(router_requests[0])
    tools = prompt.split("<tools>\n", 1)[1].split("\n</tools>", 1)[0]
    assert json.loads(tools)["function"]["name"] == "mobile_use"


def test_action_prompt_states_the_benchmarked_screen_line_whatever_the_screenshot(
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
        model="qwen/qwen3.8-flash",
    )
    prompt = _sent_system_prompt(router_requests[0])
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


@pytest.mark.parametrize("truncated", [False, True])
def test_xml_parameter_tool_calls_are_read(truncated: bool) -> None:
    raw = _XML_CLICK.removesuffix("</tool_call>") if truncated else _XML_CLICK
    assert vlm._parse_tool_call(raw) == {"action": "click", "coordinate": [752, 170]}


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
        model="qwen/qwen3.8-flash",
    )
    assert action == {"action": "tap", "x": 752, "y": 340}
    assert history == "The search icon is at the top. | Tap the search icon."
    assert len(router_requests) == 1


def test_send_chat_rejects_a_streamed_response(
    router_client: OpenRouter, router_responses: list[httpx.Response]
) -> None:
    router_responses.append(
        httpx.Response(200, text="data: [DONE]\n\n", headers={"content-type": "text/event-stream"})
    )
    with pytest.raises(TypeError, match="non-streaming"):
        vlm.send_chat(
            router_client, model="provider/other", messages=[{"role": "user", "content": "hi"}]
        )


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
        router_client, model="provider/other", messages=[{"role": "user", "content": "hi"}]
    )
    assert vlm.message_text(response) == ("hello \nworld" if isinstance(content, list) else "")


@pytest.mark.parametrize("image", [False, True])
def test_text_and_image_embeddings_use_the_documented_sdk_format(
    image: bool,
    router_client: OpenRouter,
    router_requests: list[httpx.Request],
    router_responses: list[httpx.Response],
    embedding_body: dict[str, object],
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
    assert tracked_usage.snapshot() == {
        "prompt_tokens": 5,
        "completion_tokens": 0,
        "total_tokens": 5,
        "calls": 1,
    }
    assert tracked_usage.snapshot_by_type() == {"image_embedding" if image else "embedding": 5}


@pytest.mark.parametrize(
    "data",
    [
        [],
        [{"object": "embedding", "embedding": "base64-vector"}],
        [{"object": "embedding", "embedding": []}],
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


def test_non_json_embedding_response_is_rejected(
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
) -> None:
    router_responses.append(
        httpx.Response(200, text="data: unexpected", headers={"content-type": "text/event-stream"})
    )
    with pytest.raises(TypeError, match="JSON embedding response"):
        vlm.get_image_embedding(router_client, "shot", model="google/gemini-embedding-2")


@pytest.mark.parametrize("operation", ["chat", "text", "action", "description"])
@pytest.mark.parametrize(("status", "attempts"), [(429, 3), (400, 1)])
def test_sdk_errors_exhaust_bounded_application_retries(
    operation: str,
    status: int,
    attempts: int,
    router_client: OpenRouter,
    router_requests: list[httpx.Request],
    router_responses: list[httpx.Response],
    png_screenshot: str,
    no_sleep: list[float],
    tracked_usage: vlm.TokenTracker,
) -> None:
    router_responses.extend(
        httpx.Response(status, json={"error": {"code": status, "message": "upstream failed"}})
        for _ in range(attempts)
    )

    def request() -> None:
        match operation:
            case "chat":
                vlm.send_chat(
                    router_client,
                    model="provider/other",
                    messages=[{"role": "user", "content": "hi"}],
                )
            case "text":
                vlm.get_embedding(router_client, "home", model="google/gemini-embedding-2")
            case "action":
                vlm.predict_next_action(
                    router_client, png_screenshot, "tap", model="qwen/qwen3.8-flash"
                )
            case _:
                vlm.describe_page_and_state(
                    router_client, png_screenshot, model="openai/gpt-6.1-sol"
                )

    with pytest.raises(OpenRouterError, match="upstream failed"):
        request()
    assert len(router_requests) == attempts
    assert no_sleep == ([2.0, 4.0] if status == 429 else [])
    assert tracked_usage.snapshot()["calls"] == 0


@pytest.mark.parametrize("operation", ["chat", "embedding"])
def test_sdk_retry_recovers_and_counts_only_completed_responses(
    operation: str,
    router_client: OpenRouter,
    router_requests: list[httpx.Request],
    router_responses: list[httpx.Response],
    chat_body: dict[str, object],
    embedding_body: dict[str, object],
    no_sleep: list[float],
    tracked_usage: vlm.TokenTracker,
) -> None:
    router_responses.extend(
        [
            httpx.Response(429, json={"error": {"code": 429, "message": "rate limited"}}),
            httpx.Response(200, json=chat_body if operation == "chat" else embedding_body),
        ]
    )
    if operation == "chat":
        vlm.audit_graph(router_client, "home", model="openai/gpt-6.1-sol")
    else:
        vlm.get_embedding(router_client, "home", model="google/gemini-embedding-2")
    assert len(router_requests) == 2
    assert no_sleep == [2.0]
    assert tracked_usage.snapshot()["calls"] == 1


def test_transport_timeouts_have_a_bounded_retry_budget(no_sleep: list[float]) -> None:
    attempts: list[httpx.Request] = []

    def timeout(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        raise httpx.ReadTimeout("provider timed out", request=request)

    with (
        httpx.Client(transport=httpx.MockTransport(timeout)) as http_client,
        OpenRouter(api_key="test-key", client=http_client) as client,
        pytest.raises(httpx.ReadTimeout, match="provider timed out"),
    ):
        vlm.send_chat(client, model="provider/other", messages=[{"role": "user", "content": "hi"}])
    assert len(attempts) == 3
    assert no_sleep == [2.0, 4.0]


@pytest.mark.parametrize("operation", ["audit", "planner", "action"])
def test_a_reply_without_choices_keeps_parse_fallbacks(
    operation: str,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    chat_body: dict[str, object],
    png_screenshot: str,
    no_sleep: list[float],
) -> None:
    chat_body["choices"] = []
    chat_body.pop("usage")
    router_responses.extend(
        httpx.Response(200, json=chat_body) for _ in range(3 if operation == "action" else 1)
    )
    if operation == "audit":
        assert vlm.audit_graph(router_client, "home", model="openai/gpt-6.1-sol") == {
            "issues": [],
            "summary": "parse error",
        }
    elif operation == "planner":
        assert (
            vlm.plan_next_action(
                router_client, png_screenshot, "home", [], model="openai/gpt-6.1-sol"
            )
            == ""
        )
    else:
        assert vlm.predict_next_action(
            router_client, png_screenshot, "tap", model="qwen/qwen3.8-flash"
        ) == (
            {"action": "wait", "time": 1},
            "Failed to parse action response -> wait",
        )
        assert no_sleep == [2.0, 4.0]


@pytest.mark.parametrize("limit", [None, 17])
def test_chat_omits_unspecified_token_limit_at_strict_endpoint(
    limit: int | None,
    chat_body: dict[str, object],
    no_sleep: list[float],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        payload = json.loads(request.content)
        if "max_tokens" in payload and payload["max_tokens"] is None:
            return httpx.Response(400, json={"error": {"code": 400, "message": "null forbidden"}})
        return httpx.Response(200, json=chat_body)

    with httpx.Client(transport=httpx.MockTransport(respond)) as borrowed:
        client, model = make_client(
            {"model": "openai/gpt-6.1-sol"}, role="instruction", http_client=borrowed
        )
        vlm.send_chat(
            client, model=model, messages=[{"role": "user", "content": "hi"}], max_tokens=limit
        )
    payload = json.loads(requests[0].content)
    if limit is None:
        assert "max_tokens" not in payload
    else:
        assert payload["max_tokens"] == limit
    assert len(requests) == 1
    assert no_sleep == []


def test_missing_required_chat_field_is_retried(
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    chat_body: dict[str, object],
    no_sleep: list[float],
) -> None:
    incomplete = {key: value for key, value in chat_body.items() if key != "system_fingerprint"}
    router_responses.extend(
        [httpx.Response(200, json=incomplete), httpx.Response(200, json=chat_body)]
    )
    reply = vlm.send_chat(router_client, model="test", messages=[])
    assert vlm.message_text(reply) == "test reply"
    assert len(router_requests) == 2
    assert no_sleep == [2.0]


@pytest.mark.parametrize("operation", ["chat", "text", "action"])
@pytest.mark.parametrize("recover", [False, True])
def test_sdk_validation_uses_bounded_application_recovery(
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
    bad = {key: value for key, value in good.items() if key != "model"}
    router_responses.extend(httpx.Response(200, json=bad) for _ in range(1 if recover else 3))
    if recover:
        router_responses.append(
            chat_response(content='{"action":"wait","time":2}')
            if operation == "action"
            else httpx.Response(200, json=good)
        )

    def request() -> object:
        if operation == "action":
            return vlm.predict_next_action(
                router_client, png_screenshot, "wait", model="qwen/qwen3.8-flash"
            )[0]
        if operation == "text":
            return vlm.get_embedding(router_client, "hello", model="google/gemini-embedding-2")
        reply = vlm.send_chat(
            router_client, model="test", messages=[{"role": "user", "content": "hello"}]
        )
        return vlm.message_text(reply)

    if recover:
        expected: dict[str, object] = {
            "action": {"action": "wait", "time": 2},
            "text": [0.25, 0.75],
            "chat": "test reply",
        }
        assert request() == expected[operation]
    else:
        with pytest.raises(ResponseValidationError) as caught:
            request()
        assert caught.value.status_code == 200
    assert len(router_requests) == (2 if recover else 3)
    assert no_sleep == ([2.0] if recover else [2.0, 4.0])


@pytest.mark.parametrize("recover", [False, True])
def test_text_vector_validation_uses_the_request_retry_budget(
    recover: bool,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    embedding_body: dict[str, object],
    no_sleep: list[float],
    tracked_usage: vlm.TokenTracker,
) -> None:
    bad = {**embedding_body, "data": [{"object": "embedding", "index": 0, "embedding": []}]}
    router_responses.extend(httpx.Response(200, json=bad) for _ in range(1 if recover else 3))
    if recover:
        router_responses.append(httpx.Response(200, json=embedding_body))
        assert vlm.get_embedding(router_client, "home", model="google/gemini-embedding-2") == [
            0.25,
            0.75,
        ]
    else:
        with pytest.raises(ValueError, match="embedding"):
            vlm.get_embedding(router_client, "home", model="google/gemini-embedding-2")
    assert len(router_requests) == (2 if recover else 3)
    assert no_sleep == ([2.0] if recover else [2.0, 4.0])
    assert tracked_usage.snapshot()["calls"] == int(recover)


def test_text_http_schema_and_vector_failures_share_one_budget(
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    embedding_body: dict[str, object],
    no_sleep: list[float],
) -> None:
    router_responses.extend(
        [
            httpx.Response(503, json={"error": {"code": 503, "message": "busy"}}),
            httpx.Response(
                200, json={key: value for key, value in embedding_body.items() if key != "model"}
            ),
            httpx.Response(200, json={**embedding_body, "data": []}),
            httpx.Response(200, json=embedding_body),
        ]
    )
    with pytest.raises(ValueError, match="one embedding"):
        vlm.get_embedding(router_client, "home", model="google/gemini-embedding-2")
    assert len(router_requests) == 3
    assert len(router_responses) == 1
    assert no_sleep == [2.0, 4.0]


@pytest.mark.parametrize("recover", [False, True])
def test_page_description_http_and_parse_errors_share_one_budget(
    recover: bool,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    png_screenshot: str,
    no_sleep: list[float],
    chat_response: Callable[..., httpx.Response],
) -> None:
    router_responses.extend(
        [
            httpx.Response(429, json={"error": {"code": 429, "message": "busy"}}),
            chat_response(content="[]"),
            chat_response(content='{"page_description":"home"}' if recover else "null"),
        ]
    )
    if recover:
        assert vlm.describe_page_and_state(
            router_client, png_screenshot, model="openai/gpt-6.1-sol"
        ) == ("home", {}, [])
    else:
        with pytest.raises(TypeError, match=r"page description.*JSON object"):
            vlm.describe_page_and_state(router_client, png_screenshot, model="openai/gpt-6.1-sol")
    assert len(router_requests) == 3
    assert no_sleep == [2.0, 4.0]


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
def test_malformed_page_description_exhausts_its_retry_budget(
    malformed: str,
    error: type[Exception],
    message: str,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    png_screenshot: str,
    no_sleep: list[float],
    chat_response: Callable[..., httpx.Response],
) -> None:
    router_responses.extend(chat_response(content=malformed) for _ in range(3))
    with pytest.raises(error, match=message):
        vlm.describe_page_and_state(router_client, png_screenshot, model="openai/gpt-6.1-sol")
    assert len(router_requests) == 3
    assert no_sleep == [2.0, 4.0]


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
        router_client, png_screenshot, "tap", screen_w=100, screen_h=200, model="qwen/qwen3.8-flash"
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
        ({"action": "swipe", "direction": "up", "coordinate2": [500, 300]}, _UP_SWIPE),
        (
            {"action": "swipe", "direction": "up", "coordinate": None, "coordinate2": None},
            _UP_SWIPE,
        ),
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
        router_client, png_screenshot, "act", screen_w=100, screen_h=200, model="qwen/qwen3.8-flash"
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
        ({"action": "click", "coordinate": {"x": 500, "y": 300}}, "two finite numbers"),
        ({"action": "click"}, "two finite numbers"),
        ({"action": "click", "coordinate": [500]}, "two finite numbers"),
        ({"action": "click", "coordinate": [True, 300]}, "two finite numbers"),
        ({"action": "click", "coordinate": ["500", 300]}, "two finite numbers"),
        ({"action": "click", "coordinate": [float("nan"), 300]}, "two finite numbers"),
        ({"action": "click", "coordinate": [10**400, 300]}, "two finite numbers"),
        ({"action": "swipe", "coordinate": [500, 300]}, "coordinates or a supported direction"),
        ({"action": "swipe", "coordinate": [500, 300], "coordinate2": [500]}, "two finite numbers"),
    ],
)
def test_malformed_tool_arguments_are_rejected(args: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        vlm._tool_call_to_aitk(args)


@pytest.mark.parametrize("coordinate", [[500], [float("nan"), 300]])
def test_direction_does_not_hide_invalid_partial_coordinates(coordinate: object) -> None:
    with pytest.raises(ValueError, match="coordinates"):
        vlm._tool_call_to_aitk({"action": "swipe", "direction": "up", "coordinate": coordinate})


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


@pytest.mark.parametrize("recover", [False, True])
def test_malformed_action_output_requests_another_response(
    recover: bool,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    png_screenshot: str,
    no_sleep: list[float],
    chat_response: Callable[..., httpx.Response],
) -> None:
    router_responses.extend(
        chat_response(content='{"action":"click","coordinate":["bad",500]}')
        for _ in range(1 if recover else 3)
    )
    if recover:
        router_responses.append(chat_response(content='{"action":"click","coordinate":[500,300]}'))
    result = vlm.predict_next_action(
        router_client, png_screenshot, "tap", screen_w=100, screen_h=200, model="qwen/qwen3.8-flash"
    )
    if recover:
        assert result[0] == {"action": "tap", "x": 50, "y": 60}
    else:
        assert result == ({"action": "wait", "time": 1}, "Failed to parse action response -> wait")
    assert len(router_requests) == (2 if recover else 3)
    assert no_sleep == ([2.0] if recover else [2.0, 4.0])


@pytest.mark.parametrize("last", ["success", "schema", "output", "http"])
def test_action_schema_http_and_output_share_total_budget(
    last: str,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    chat_body: dict[str, object],
    png_screenshot: str,
    no_sleep: list[float],
    chat_response: Callable[..., httpx.Response],
) -> None:
    bad = {key: value for key, value in chat_body.items() if key != "model"}
    http_error = httpx.Response(503, json={"error": {"code": 503, "message": "busy"}})
    router_responses.extend(
        [
            http_error,
            httpx.Response(200, json=bad),
            {
                "success": chat_response(content='{"action":"click","coordinate":[500,300]}'),
                "schema": httpx.Response(200, json=bad),
                "output": chat_response(content="unreadable"),
                "http": http_error,
            }[last],
        ]
    )
    if last in {"http", "schema"}:
        with pytest.raises(OpenRouterError):
            vlm.predict_next_action(
                router_client, png_screenshot, "tap", model="qwen/qwen3.8-flash"
            )
    else:
        action, _ = vlm.predict_next_action(
            router_client, png_screenshot, "tap", model="qwen/qwen3.8-flash"
        )
        assert action == (
            {"action": "tap", "x": 540, "y": 576}
            if last == "success"
            else {"action": "wait", "time": 1}
        )
    assert len(router_requests) == 3
    assert no_sleep == [2.0, 4.0]


def test_malformed_http_error_never_returns_action_fallback(
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    png_screenshot: str,
    no_sleep: list[float],
) -> None:
    router_responses.append(httpx.Response(400, json={"error": {"message": "bad"}}))
    with pytest.raises(OpenRouterError) as caught:
        vlm.predict_next_action(router_client, png_screenshot, "tap", model="qwen/qwen3.8-flash")
    assert caught.value.status_code == 400
    assert len(router_requests) == 1
    assert no_sleep == []
