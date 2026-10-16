"""Tests for android_app_graph.utils.vlm_utils.cosine_similarity and score_by_cosine.

The single shared implementations used by both android_app_graph.utils.graph_manager
and android_app_graph.adapters.aitk_translator; both modules' node-retrieval tests
exercise them end to end, so this module owns their unit tests.
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
from openrouter.errors.openroutererror import OpenRouterError

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
            '{"page_description":"home","state":{"tab":"all"},"elements":[]}',
            ("home", {"tab": "all"}, []),
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


@pytest.mark.parametrize("content", [None, "", "calling action", "omitted"])
def test_native_tool_calls_survive_sdk_parsing(
    content: str | None,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    png_screenshot: str,
    chat_response: Callable[..., httpx.Response],
) -> None:
    message: dict[str, object] = {} if content == "omitted" else {"content": content}
    router_responses.append(
        chat_response(tool_arguments='{"action":"click","coordinate":[500,500]}', **message)
    )
    action, _ = vlm.predict_next_action(
        router_client, png_screenshot, "tap", screen_w=100, screen_h=200, model="qwen/qwen3.8-flash"
    )
    assert action == {"action": "tap", "x": 50, "y": 100}


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


@pytest.mark.parametrize("operation", ["chat", "embedding"])
@pytest.mark.parametrize("status", [429, 500])
def test_sdk_errors_exhaust_bounded_application_retries(
    operation: str,
    status: int,
    router_client: OpenRouter,
    router_requests: list[httpx.Request],
    router_responses: list[httpx.Response],
    no_sleep: list[float],
    tracked_usage: vlm.TokenTracker,
) -> None:
    router_responses.extend(
        httpx.Response(status, json={"error": {"code": status, "message": "upstream failed"}})
        for _ in range(3)
    )
    if operation == "chat":
        with pytest.raises(OpenRouterError, match="upstream failed"):
            vlm.send_chat(
                router_client, model="provider/other", messages=[{"role": "user", "content": "hi"}]
            )
    else:
        with pytest.raises(OpenRouterError, match="upstream failed"):
            vlm.get_embedding(router_client, "home", model="google/gemini-embedding-2")
    assert len(router_requests) == 3
    assert no_sleep == [2.0, 4.0]
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
def test_unreadable_or_refused_sdk_response_keeps_parse_fallbacks(
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
