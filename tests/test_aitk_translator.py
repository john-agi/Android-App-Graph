"""Pure helpers of the AITK translator, exercised without adb, emulator, network or keys."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import networkx as nx
import pytest
from hypothesis import given
from hypothesis import strategies as st

from android_app_graph.adapters import aitk_translator

_SCREENSHOT = b"not-really-a-png"
_LETTERS = "ABCDEFGH"


def _write_graph(
    tmp_path: Path,
    *,
    app: str = "demo",
    audited: bool = False,
    nodes: list[dict[str, Any]] | None = None,
    edges: list[dict[str, Any]] | None = None,
) -> Path:
    """Write ``<tmp_path>/<app>/<stem>.json`` and return its path."""
    app_dir = tmp_path / app
    app_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{app}_audited" if audited else app
    path = app_dir / f"{stem}.json"
    path.write_text(
        json.dumps(
            {
                "nodes": nodes
                if nodes is not None
                else [{"id": "n1", "activity": "com.a.b/.Main"}],
                "edges": edges if edges is not None else [],
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize(
    ("activity", "expected"),
    [
        ("com.example.app/.MainActivity", "com.example.app"),
        ("com.google.android.apps.maps/com.google.Main", "com.google.android"),
        ("com.example.app", "com.example.app"),
        ("two.parts", "two.parts"),
        ("single", "single"),
        ("", ""),
    ],
)
def test_package_from_activity(activity: str, expected: str) -> None:
    assert aitk_translator._package_from_activity(activity) == expected


@given(st.text())
def test_package_from_activity_keeps_at_most_three_components(activity: str) -> None:
    """The package is a dotted prefix of the activity's component, never longer."""
    package = aitk_translator._package_from_activity(activity)
    component = activity.split("/", maxsplit=1)[0]
    assert component.startswith(package)
    assert len(package.split(".")) <= max(3, len(component.split(".")))


def test_extract_packages_from_graph_skips_nodes_without_activity() -> None:
    G = nx.DiGraph()
    G.add_node("a", activity="com.example.app/.Main")
    G.add_node("b", activity="com.example.app/.Detail")
    G.add_node("c", activity="org.other.thing/.Main")
    G.add_node("d", activity="")
    G.add_node("e")
    assert aitk_translator._extract_packages_from_graph(G) == {
        "com.example.app",
        "org.other.thing",
    }


def test_extract_packages_from_empty_graph() -> None:
    assert aitk_translator._extract_packages_from_graph(nx.DiGraph()) == set()


def test_cosine_similarity_of_identical_vectors_is_one() -> None:
    assert aitk_translator._cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(
        1.0
    )


def test_cosine_similarity_of_orthogonal_vectors_is_zero() -> None:
    assert aitk_translator._cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_of_opposite_vectors_is_minus_one() -> None:
    assert aitk_translator._cosine_similarity([1.0, 1.0], [-1.0, -1.0]) == pytest.approx(-1.0)


@pytest.mark.parametrize(("a", "b"), [([0.0, 0.0], [1.0, 1.0]), ([1.0, 1.0], [0.0, 0.0])])
def test_cosine_similarity_with_a_zero_vector_is_zero(a: list[float], b: list[float]) -> None:
    """A zero norm has no direction, so the similarity is defined as 0.0 rather than NaN."""
    assert aitk_translator._cosine_similarity(a, b) == 0.0


_finite_vectors = st.lists(
    st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
    min_size=1,
    max_size=8,
)


@given(a=_finite_vectors, b=_finite_vectors)
def test_cosine_similarity_is_bounded_and_symmetric(a: list[float], b: list[float]) -> None:
    similarity = aitk_translator._cosine_similarity(a, b)
    assert -1.0 - 1e-9 <= similarity <= 1.0 + 1e-9
    assert similarity == pytest.approx(aitk_translator._cosine_similarity(b, a))
    assert not math.isnan(similarity)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("A", "A"),
        (" b \n", "B"),
        ("none", "NONE"),
        ("<think>weighing A and B</think>\nC", "C"),
        ("Final answer: B", "B"),
        ("Decision: none", "NONE"),
        ("Answer: Z", None),
        ("I see A here, but B is better", "B"),
        ("nothing matches, so none of them", "NONE"),
        ("no letters at all", None),
    ],
)
def test_parse_model_choice(raw: str, expected: str | None) -> None:
    assert aitk_translator._parse_model_choice(raw, "ABCD") == expected


def test_parse_model_choice_returns_a_falsy_answer_for_an_empty_reply() -> None:
    """Current behaviour: ``"" in valid_letters`` is true, so an empty reply echoes back.

    Every caller tests the result for truthiness before using it, so an empty reply
    still triggers a parse retry and then the "no match" path.
    """
    assert aitk_translator._parse_model_choice("", "ABCD") == ""


def test_parse_model_choice_rejects_letters_outside_the_valid_set() -> None:
    assert aitk_translator._parse_model_choice("H", "ABC") is None


@given(st.sampled_from(_LETTERS), st.sampled_from(["", " ", "\n\t"]))
def test_parse_model_choice_accepts_any_bare_valid_letter(letter: str, padding: str) -> None:
    raw = f"{padding}{letter.lower()}{padding}"
    assert aitk_translator._parse_model_choice(raw, _LETTERS) == letter


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("", "nothing"),
        ("   ", "nothing"),
        ("nothing", "nothing"),
        ("Nothing new here", "nothing"),
        ("The total is $42.50", "The total is $42.50"),
        ('"The total is $42.50"', "The total is $42.50"),
        ("<think>hmm</think>The bus leaves at 14:05", "The bus leaves at 14:05"),
        ("Answer: The bus leaves at 14:05", "The bus leaves at 14:05"),
        ("Information: seat 12A is free", "seat 12A is free"),
        ("Reasoning:\n\nfirst pass\n\nthe price is 9 EUR", "the price is 9 EUR"),
    ],
)
def test_parse_record_output(raw: str, expected: str) -> None:
    assert aitk_translator._parse_record_output(raw) == expected


def test_parse_record_output_rejects_an_overlong_answer() -> None:
    assert aitk_translator._parse_record_output("x" * 801) == "nothing"


def test_parse_record_output_rejects_a_multi_sentence_ramble() -> None:
    assert aitk_translator._parse_record_output("a. " * 11) == "nothing"


def test_parse_record_output_keeps_a_single_paragraph_after_a_marker() -> None:
    """A marker with nothing to split on is left alone: there is no last paragraph."""
    assert aitk_translator._parse_record_output("Analysis: the cart holds 3 items") == (
        "Analysis: the cart holds 3 items"
    )


@given(st.text(max_size=200))
def test_parse_record_output_is_never_empty(raw: str) -> None:
    assert aitk_translator._parse_record_output(raw) != ""


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("", ""),
        ("   ", ""),
        ("Tap the Search button", "Tap the Search button"),
        ('"Tap the Search button"', "Tap the Search button"),
        ("<think>hmm</think>Tap the Search button", "Tap the Search button"),
        ("Instruction: Tap the Search button", "Tap the Search button"),
        ("Next action: Scroll down", "Scroll down"),
        ("<think> Scroll down to the totals", "Scroll down to the totals"),
    ],
)
def test_parse_one_step_instruction(raw: str, expected: str) -> None:
    assert aitk_translator._parse_one_step_instruction(raw) == expected


def test_parse_one_step_instruction_drops_everything_before_a_closing_think_tag() -> None:
    assert aitk_translator._parse_one_step_instruction("Tap Search </think>") == ""


def test_parse_one_step_instruction_rejects_an_overlong_plan() -> None:
    assert aitk_translator._parse_one_step_instruction("x" * 501) == ""


@pytest.mark.parametrize(
    "raw",
    [
        "Take one immediate visible UI action",
        "Pick whatever best moves toward the task",
        "Continue toward the goal",
        "Take the best action",
        "Just move toward the task",
    ],
)
def test_parse_one_step_instruction_rejects_generic_restatements(raw: str) -> None:
    assert aitk_translator._parse_one_step_instruction(raw) == ""


def test_extract_json_object_finds_an_embedded_object() -> None:
    assert aitk_translator._extract_json_object('noise {"choice": "A"} tail') == {"choice": "A"}


def test_extract_json_object_skips_braces_that_do_not_decode() -> None:
    text = 'prose {not json at all} then {"choice": "B"}'
    assert aitk_translator._extract_json_object(text) == {"choice": "B"}


def test_extract_json_object_ignores_a_leading_array() -> None:
    assert aitk_translator._extract_json_object('[1, 2] {"choice": "C"}') == {"choice": "C"}


@pytest.mark.parametrize("text", ["", "no object here", "[1, 2, 3]"])
def test_extract_json_object_without_an_object(text: str) -> None:
    assert aitk_translator._extract_json_object(text) is None


def test_parse_json_object_rejects_a_top_level_array() -> None:
    assert aitk_translator._parse_json_object("[1, 2]") is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"choice": "A", "instruction": "Tap"}', {"choice": "A", "instruction": "Tap"}),
        ('```json\n{"choice": "A"}\n```', {"choice": "A"}),
        ('<think>weighing</think>{"choice": "B"}', {"choice": "B"}),
        ("not json", None),
        ("", None),
    ],
)
def test_parse_decide_output(raw: str, expected: dict[str, Any] | None) -> None:
    assert aitk_translator._parse_decide_output(raw) == expected


def test_parse_decide_output_falls_back_to_the_whole_text() -> None:
    """A </think> tag with no object after it must not hide an object before it."""
    raw = '{"choice": "A"} </think> nothing usable here'
    assert aitk_translator._parse_decide_output(raw) == {"choice": "A"}


@given(st.dictionaries(st.text(max_size=10), st.integers(), max_size=5))
def test_parse_decide_output_round_trips_json_objects(payload: dict[str, int]) -> None:
    assert aitk_translator._parse_decide_output(json.dumps(payload)) == payload


def test_load_graph_from_json_reads_nodes_and_edges(tmp_path: Path) -> None:
    path = _write_graph(
        tmp_path,
        nodes=[
            {
                "id": "n1",
                "activity": "com.a.b/.Main",
                "page_description": "home",
                "state_schema": {"query": "str"},
                "visit_count": 3,
            },
            {"id": "n2"},
        ],
        edges=[{"source": "n1", "target": "n2", "instructions": ["tap"], "visit_count": 2}],
    )
    G, data = aitk_translator._load_graph_from_json(path)

    assert set(G.nodes) == {"n1", "n2"}
    assert G.nodes["n1"]["page_description"] == "home"
    assert G.nodes["n1"]["state_schema"] == {"query": "str"}
    assert G.nodes["n1"]["visit_count"] == 3
    assert G.nodes["n1"]["reference_screenshot"] is None
    assert G.nodes["n2"]["activity"] == ""
    assert G.edges["n1", "n2"]["instructions"] == ["tap"]
    assert "schema_deltas" not in G.edges["n1", "n2"]
    assert data["nodes"][1] == {"id": "n2"}


def test_load_graph_from_json_keeps_non_empty_schema_deltas(tmp_path: Path) -> None:
    path = _write_graph(
        tmp_path,
        nodes=[{"id": "n1"}, {"id": "n2"}],
        edges=[
            {
                "source": "n1",
                "target": "n2",
                "schema_deltas": [{"cart": {"before": 0, "after": 1}}],
            },
            {"source": "n2", "target": "n1", "schema_deltas": []},
        ],
    )
    G, _ = aitk_translator._load_graph_from_json(path)
    assert G.edges["n1", "n2"]["schema_deltas"] == [{"cart": {"before": 0, "after": 1}}]
    assert "schema_deltas" not in G.edges["n2", "n1"]


def test_load_graph_from_json_embeds_reference_screenshots(tmp_path: Path) -> None:
    path = _write_graph(tmp_path, nodes=[{"id": "n1"}, {"id": "n2"}])
    screenshots = path.parent / "demo_screenshots"
    screenshots.mkdir()
    (screenshots / "n1.png").write_bytes(_SCREENSHOT)

    G, _ = aitk_translator._load_graph_from_json(path)
    import base64

    assert G.nodes["n1"]["reference_screenshot"] == base64.b64encode(_SCREENSHOT).decode("ascii")
    assert G.nodes["n2"]["reference_screenshot"] is None


def test_load_graph_from_json_falls_back_to_the_unaudited_screenshot_dir(tmp_path: Path) -> None:
    path = _write_graph(tmp_path, audited=True, nodes=[{"id": "n1"}])
    screenshots = path.parent / "demo_screenshots"
    screenshots.mkdir()
    (screenshots / "n1.png").write_bytes(_SCREENSHOT)

    G, _ = aitk_translator._load_graph_from_json(path)
    assert G.nodes["n1"]["reference_screenshot"] is not None


@pytest.mark.parametrize(
    "payload",
    ["[]", '"a string"', '{"nodes": []}', '{"edges": []}', '{"nodes": {}, "edges": []}'],
)
def test_load_graph_from_json_rejects_a_malformed_document(tmp_path: Path, payload: str) -> None:
    path = tmp_path / "demo.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(TypeError):
        aitk_translator._load_graph_from_json(path)


def test_image_embeddings_path_is_a_sidecar(tmp_path: Path) -> None:
    assert aitk_translator._image_embeddings_path(tmp_path / "demo.json") == (
        tmp_path / "demo.image_emb.json"
    )


def test_image_embeddings_round_trip(tmp_path: Path) -> None:
    graph_path = tmp_path / "demo.json"
    G = nx.DiGraph()
    G.add_node("n1", image_embedding=[0.5, 0.25])
    G.add_node("n2", image_embedding=[])
    G.add_node("n3")

    aitk_translator._save_image_embeddings(graph_path, G)
    assert aitk_translator._load_image_embeddings(graph_path) == {"n1": [0.5, 0.25]}


def test_load_image_embeddings_without_a_sidecar(tmp_path: Path) -> None:
    assert aitk_translator._load_image_embeddings(tmp_path / "demo.json") == {}


def test_load_image_embeddings_drops_malformed_entries(tmp_path: Path) -> None:
    graph_path = tmp_path / "demo.json"
    aitk_translator._image_embeddings_path(graph_path).write_text(
        json.dumps({"n1": [1.0, 2.0], "n2": "not a vector", "n3": [], "n4": [1.0, "two"]}),
        encoding="utf-8",
    )
    assert aitk_translator._load_image_embeddings(graph_path) == {"n1": [1.0, 2.0]}


def test_iter_runtime_graph_files_without_a_graph_dir(tmp_path: Path) -> None:
    assert aitk_translator._iter_runtime_graph_files(tmp_path / "absent") == []


def test_iter_runtime_graph_files_prefers_the_audited_graph(tmp_path: Path) -> None:
    _write_graph(tmp_path, app="eboox")
    audited = _write_graph(tmp_path, app="eboox", audited=True)
    assert aitk_translator._iter_runtime_graph_files(tmp_path) == [("eboox", audited)]


def test_iter_runtime_graph_files_sorts_apps_and_skips_side_files(tmp_path: Path) -> None:
    zebra = _write_graph(tmp_path, app="zebra")
    alpha = _write_graph(tmp_path, app="alpha")
    (tmp_path / "alpha" / "alpha_audit_report.json").write_text("{}", encoding="utf-8")
    (tmp_path / "alpha" / "alpha.image_emb.json").write_text("{}", encoding="utf-8")
    (tmp_path / "loose.json").write_text("{}", encoding="utf-8")
    (tmp_path / "empty").mkdir()

    assert aitk_translator._iter_runtime_graph_files(tmp_path) == [
        ("alpha", alpha),
        ("zebra", zebra),
    ]


def test_memory_starts_empty() -> None:
    memory = aitk_translator.Memory()
    assert memory.has_content() is False
    assert memory.format() == "(empty)"


@pytest.mark.parametrize("info", ["", "nothing", "Nothing", "NOTHING"])
def test_memory_ignores_empty_and_nothing_info(info: str) -> None:
    memory = aitk_translator.Memory()
    memory.add_info(info)
    assert memory.info == []
    assert memory.has_content() is False


def test_memory_ignores_an_empty_observation() -> None:
    memory = aitk_translator.Memory()
    memory.add_observation("")
    assert memory.observations == []


def test_memory_records_and_formats_every_section() -> None:
    memory = aitk_translator.Memory()
    memory.add_action("tapped Search")
    memory.add_info("the total is 9 EUR")
    memory.add_observation("a dialog appeared")

    assert memory.has_content() is True
    assert memory.format() == (
        "Actions completed:\n"
        "  1. tapped Search\n"
        "Information collected:\n"
        "  1. the total is 9 EUR\n"
        "Observations:\n"
        "  1. a dialog appeared"
    )


def test_memory_numbers_repeated_entries() -> None:
    memory = aitk_translator.Memory()
    memory.add_action("first")
    memory.add_action("second")
    assert "  1. first\n  2. second" in memory.format()


class _FakeMessage:
    def __init__(self, content: str | None) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str | None) -> None:
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str | None) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    """Stand-in for ``client.chat.completions`` that replays canned replies."""

    def __init__(self, replies: list[str | None]) -> None:
        self.replies = list(replies)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> _FakeResponse:
        self.calls.append(kwargs)
        return _FakeResponse(self.replies.pop(0) if self.replies else "")


class _FakeChat:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.completions = completions


class _FakeClient:
    def __init__(self, *replies: str | None) -> None:
        self.completions = _FakeCompletions(list(replies))
        self.chat = _FakeChat(self.completions)


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record retry back-off delays instead of waiting for them."""
    delays: list[float] = []
    monkeypatch.setattr(aitk_translator.time, "sleep", delays.append)
    return delays


def test_call_with_retry_returns_the_first_success() -> None:
    calls: list[int] = []

    def once() -> str:
        calls.append(1)
        return "ok"

    assert aitk_translator._call_with_retry("label", once) == "ok"
    assert len(calls) == 1


@pytest.mark.usefixtures("no_sleep")
def test_call_with_retry_recovers_after_a_failure() -> None:
    attempts: list[int] = []

    def flaky() -> str:
        attempts.append(1)
        if len(attempts) < 3:
            raise RuntimeError("boom")
        return "ok"

    assert aitk_translator._call_with_retry("label", flaky) == "ok"
    assert len(attempts) == 3


def test_call_with_retry_reraises_after_the_last_attempt(no_sleep: list[float]) -> None:
    attempts: list[int] = []

    def always_fails() -> str:
        attempts.append(1)
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        aitk_translator._call_with_retry("label", always_fails)
    assert len(attempts) == aitk_translator.V2_API_RETRIES + 1
    assert no_sleep == [2.0, 4.0]


def test_chat_completion_content_stops_when_the_caller_breaks() -> None:
    client = _FakeClient("first", "second")
    seen = [
        (attempt, content, can_retry)
        for attempt, content, can_retry in aitk_translator._chat_completion_content(
            client,  # ty: ignore[invalid-argument-type]  # duck-typed stand-in for OpenAI
            model="m",
        )
    ]
    assert seen == [(0, "first", True), (1, "second", False)]
    assert len(client.completions.calls) == 2


def test_chat_completion_content_turns_a_missing_body_into_an_empty_string() -> None:
    client = _FakeClient(None)
    first = next(
        iter(
            aitk_translator._chat_completion_content(
                client,  # ty: ignore[invalid-argument-type]  # duck-typed stand-in for OpenAI
                model="m",
            )
        )
    )
    assert first == (0, "", True)


def test_make_no_proxy_client_defaults() -> None:
    client, model = aitk_translator._make_no_proxy_client({"api_key": "test-key"})
    assert model == "gpt-4o"
    assert client.max_retries == 0
    assert client.timeout == 60.0


def test_make_no_proxy_client_reads_the_config() -> None:
    client, model = aitk_translator._make_no_proxy_client(
        {
            "api_key": "test-key",
            "base_url": "http://localhost:8000/v1",
            "model": "qwen",
            "request_timeout": 12,
            "max_retries": 3,
        }
    )
    assert model == "qwen"
    assert str(client.base_url) == "http://localhost:8000/v1/"
    assert client.timeout == 12.0
    assert client.max_retries == 3


def test_make_no_proxy_client_resolves_env_references(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_GRAPH_TEST_MODEL", "from-env")
    _, model = aitk_translator._make_no_proxy_client(
        {"api_key": "test-key", "model": "${APP_GRAPH_TEST_MODEL}"}
    )
    assert model == "from-env"


def test_make_no_proxy_client_prefers_the_timeout_alias() -> None:
    client, _ = aitk_translator._make_no_proxy_client({"api_key": "test-key", "timeout": 5})
    assert client.timeout == 5.0


_VLM_CONFIG: dict[str, Any] = {
    "action": {"api_key": "test-key", "model": "action-model"},
    "page_detail": {"api_key": "test-key", "model": "detail-model"},
    "embedding": {"model": "emb-model"},
    "image_embedding": {"api_key": "image-key", "model": "img-model"},
}

_DEMO_NODES: list[dict[str, Any]] = [
    {
        "id": "home",
        "activity": "com.demo.app/.HomeActivity",
        "page_description": "home screen",
        "state_schema": {"query": "str"},
    },
    {
        "id": "results",
        "activity": "com.demo.app/.ResultsActivity",
        "page_description": "results list",
    },
]

_DEMO_EDGES: list[dict[str, Any]] = [
    {
        "source": "home",
        "target": "home",
        "instruction_templates": [
            {"template": "search for {query}", "observation_template": "results shown"}
        ],
        "schema_deltas": [{"query": {"before": "", "after": "shoes"}}],
    },
    {
        "source": "home",
        "target": "results",
        "instructions": ["tap the first result"],
        "target_observations": ["the results list"],
    },
]


@pytest.fixture
def graph_dir(tmp_path: Path) -> Path:
    """A graph root holding one app whose nodes carry no reference screenshots."""
    _write_graph(tmp_path, app="demo", nodes=_DEMO_NODES, edges=_DEMO_EDGES)
    return tmp_path


@pytest.fixture
def translator(graph_dir: Path) -> aitk_translator.UIKobeV2Translator:
    return aitk_translator.register({"graph_dir": str(graph_dir), "vlm_config": _VLM_CONFIG})


def test_register_loads_every_graph(translator: aitk_translator.UIKobeV2Translator) -> None:
    assert isinstance(translator, aitk_translator.UIKobeV2Translator)
    assert set(translator._graphs) == {"demo"}
    assert translator._package_to_app == {"com.demo.app": "demo"}
    assert translator.model_name == "action-model"
    assert translator.desc_model == "detail-model"
    assert translator.emb_model == "emb-model"
    assert translator.image_embedding_model == "img-model"


def test_image_embedding_base_url_defaults_to_google(graph_dir: Path) -> None:
    config = {**_VLM_CONFIG, "image_embedding": {"base_url": "http://localhost:9000/v1"}}
    built = aitk_translator.UIKobeV2Translator(graph_dir=str(graph_dir), vlm_config=config)
    assert built.image_embedding_base_url == "https://generativelanguage.googleapis.com/v1beta"


def test_image_embedding_base_url_keeps_a_google_override(graph_dir: Path) -> None:
    config = {
        **_VLM_CONFIG,
        "image_embedding": {"native_base_url": "https://eu.googleapis.com/v1beta"},
    }
    built = aitk_translator.UIKobeV2Translator(graph_dir=str(graph_dir), vlm_config=config)
    assert built.image_embedding_base_url == "https://eu.googleapis.com/v1beta"


def test_load_all_graphs_tolerates_a_missing_graph_dir(tmp_path: Path) -> None:
    built = aitk_translator.UIKobeV2Translator(
        graph_dir=str(tmp_path / "absent"), vlm_config=_VLM_CONFIG
    )
    assert built._graphs == {}


def test_load_all_graphs_skips_a_broken_graph(tmp_path: Path) -> None:
    app_dir = tmp_path / "broken"
    app_dir.mkdir()
    (app_dir / "broken.json").write_text("{not json", encoding="utf-8")
    built = aitk_translator.UIKobeV2Translator(graph_dir=str(tmp_path), vlm_config=_VLM_CONFIG)
    assert built._graphs == {}


def test_load_all_graphs_reads_the_embedding_sidecar(graph_dir: Path) -> None:
    sidecar = graph_dir / "demo" / "demo.image_emb.json"
    sidecar.write_text(json.dumps({"home": [1.0, 0.0], "gone": [0.0, 1.0]}), encoding="utf-8")
    built = aitk_translator.UIKobeV2Translator(graph_dir=str(graph_dir), vlm_config=_VLM_CONFIG)
    assert built._graphs["demo"].nodes["home"]["image_embedding"] == [1.0, 0.0]
    assert "gone" not in built._graphs["demo"]


@pytest.mark.parametrize(
    ("task", "expected"),
    [("Open Demo and search for shoes", "demo"), ("Check the weather", None)],
)
def test_resolve_app_from_task(
    translator: aitk_translator.UIKobeV2Translator, task: str, expected: str | None
) -> None:
    assert translator._resolve_app_from_task(task) == expected


@pytest.mark.parametrize(
    "package",
    ["com.demo.app", "com.demo.app/.HomeActivity", "com.demo.app.free", "com.demo"],
)
def test_get_graph_for_package_matches(
    translator: aitk_translator.UIKobeV2Translator, package: str
) -> None:
    assert translator._get_graph_for_package(package) is translator._graphs["demo"]
    assert translator._app_name == "demo"


def test_get_graph_for_unknown_package(translator: aitk_translator.UIKobeV2Translator) -> None:
    assert translator._get_graph_for_package("org.other.thing") is None


def test_reset_task_state_clears_the_previous_task(
    translator: aitk_translator.UIKobeV2Translator,
) -> None:
    translator._app_opened = True
    translator._current_node = "home"
    translator._step_count = 7
    translator._memory.add_action("tapped Search")

    translator._reset_task_state()

    assert translator._app_opened is False
    assert translator._current_node is None
    assert translator._current_graph is None
    assert translator._step_count == 0
    assert translator._memory.has_content() is False


@pytest.mark.parametrize(
    ("template", "expected"),
    [
        (
            {"template": "search for {query}", "observation_template": "results"},
            ("search for {query}", "results"),
        ),
        ({"template": "tap"}, ("tap", "")),
        ({}, ("", "")),
        ("plain instruction", ("plain instruction", "")),
    ],
)
def test_unpack_template(template: dict[str, Any] | str, expected: tuple[str, str]) -> None:
    assert aitk_translator.UIKobeV2Translator._unpack_template(template) == expected


@pytest.mark.parametrize(
    ("delta", "expected"),
    [
        (None, ""),
        ({}, ""),
        ({"cart": {"before": 0, "after": 1}}, "cart: 0 -> 1"),
        ({"cart": {}}, "cart: ? -> ?"),
        ({"cart": "emptied"}, "cart: emptied"),
        ({"a": "x", "b": "y"}, "a: x, b: y"),
    ],
)
def test_format_schema_delta(delta: dict[str, Any] | None, expected: str) -> None:
    assert aitk_translator.UIKobeV2Translator._format_schema_delta(delta) == expected


def test_edge_effect_hint_without_deltas(translator: aitk_translator.UIKobeV2Translator) -> None:
    assert translator._edge_effect_hint({}) == ""


def test_edge_effect_hint_uses_the_indexed_delta(
    translator: aitk_translator.UIKobeV2Translator,
) -> None:
    edge = {"schema_deltas": [{"a": "one"}, {"b": "two"}]}
    assert translator._edge_effect_hint(edge, 1) == "b: two"


def test_edge_effect_hint_merges_when_the_index_is_out_of_range(
    translator: aitk_translator.UIKobeV2Translator,
) -> None:
    """Earlier deltas win: the merge keeps the first value seen for each key."""
    edge = {"schema_deltas": [{"a": "one"}, {"a": "two"}, "not a delta"]}
    assert translator._edge_effect_hint(edge, 9) == "a: one"
    assert translator._edge_effect_hint(edge) == "a: one"


def test_build_options_lists_done_self_loops_neighbours_and_free(
    translator: aitk_translator.UIKobeV2Translator,
) -> None:
    G = translator._graphs["demo"]
    text, options = translator._build_options(G, "home")

    assert [opt["type"] for opt in options] == ["done", "self_loop", "neighbor", "free"]
    assert [opt["letter"] for opt in options] == ["A", "B", "C", "D"]
    assert options[1]["instruction"] == "search for {query}"
    assert options[1]["effect"] == "query:  -> shoes"
    assert options[2]["node"] == "results"
    assert options[2]["description"] == "results list"
    assert options[2]["instruction"] == "tap the first result"

    lines = text.splitlines()
    assert lines[0].startswith("A) DONE")
    assert (
        lines[1]
        == 'B) Stay here — "search for {query}" → results shown | changes: query:  -> shoes'
    )
    assert lines[2] == 'C) Go to "results list" — "tap the first result" → the results list'
    assert lines[3].startswith("D) FREE")


def test_build_options_falls_back_to_raw_self_loop_instructions(
    translator: aitk_translator.UIKobeV2Translator,
) -> None:
    G = translator._graphs["demo"]
    G.add_edge(
        "results",
        "results",
        instructions=["scroll down", "scroll up"],
        target_observations=["more results"],
        instruction_templates=[],
        schema_deltas=[{"page": "1 -> 2"}],
    )
    text, options = translator._build_options(G, "results")

    assert [opt["type"] for opt in options] == ["done", "self_loop", "self_loop", "free"]
    assert options[1]["instruction"] == "scroll down"
    assert options[2]["instruction"] == "scroll up"
    lines = text.splitlines()
    assert lines[1] == 'B) Stay here — "scroll down" → more results | changes: page: 1 -> 2'
    # The second instruction has no delta of its own, so the merged hint is reused.
    assert lines[2] == 'C) Stay here — "scroll up" | changes: page: 1 -> 2'


def test_build_options_skips_the_self_loop_when_listing_neighbours(
    translator: aitk_translator.UIKobeV2Translator,
) -> None:
    G = translator._graphs["demo"]
    _, options = translator._build_options(G, "home")
    assert [opt for opt in options if opt["type"] == "neighbor"] == [
        {
            "letter": "C",
            "type": "neighbor",
            "node": "results",
            "instruction": "tap the first result",
            "description": "results list",
            "effect": "",
        }
    ]


def _use_model(
    monkeypatch: pytest.MonkeyPatch,
    translator: aitk_translator.UIKobeV2Translator,
    *replies: str | None,
) -> _FakeClient:
    """Answer every chat completion from ``translator.model_client`` with ``replies``."""
    client = _FakeClient(*replies)
    monkeypatch.setattr(translator, "model_client", client)
    return client


def test_record_info_stores_what_the_model_read(
    monkeypatch: pytest.MonkeyPatch, translator: aitk_translator.UIKobeV2Translator
) -> None:
    _use_model(monkeypatch, translator, "The total is 9 EUR")
    translator._record_info("buy shoes", "screenshot")
    assert translator._memory.info == ["The total is 9 EUR"]


def test_record_info_retries_then_keeps_nothing(
    monkeypatch: pytest.MonkeyPatch, translator: aitk_translator.UIKobeV2Translator
) -> None:
    client = _use_model(monkeypatch, translator, "nothing", "nothing")
    translator._record_info("buy shoes", "screenshot")
    assert translator._memory.info == []
    assert len(client.completions.calls) == 1


def test_generate_answer_returns_the_first_non_empty_reply(
    monkeypatch: pytest.MonkeyPatch, translator: aitk_translator.UIKobeV2Translator
) -> None:
    client = _use_model(monkeypatch, translator, "  ", "9 EUR")
    assert translator._generate_answer("buy shoes") == "9 EUR"
    assert len(client.completions.calls) == 2


def test_generate_answer_gives_up_after_the_retry(
    monkeypatch: pytest.MonkeyPatch, translator: aitk_translator.UIKobeV2Translator
) -> None:
    _use_model(monkeypatch, translator, "", "")
    assert translator._generate_answer("buy shoes") == ""


def test_plan_free_action_uses_the_planned_instruction(
    monkeypatch: pytest.MonkeyPatch, translator: aitk_translator.UIKobeV2Translator
) -> None:
    _use_model(monkeypatch, translator, "Tap the cart icon")
    instruction = translator._plan_free_action("buy shoes", "screenshot", "no graph", ["earlier"])
    assert instruction == "Tap the cart icon"


def test_plan_free_action_falls_back_when_the_planner_is_generic(
    monkeypatch: pytest.MonkeyPatch, translator: aitk_translator.UIKobeV2Translator
) -> None:
    _use_model(monkeypatch, translator, "Take the best action", "Take the best action")
    instruction = translator._plan_free_action("buy shoes", "screenshot", "no graph", [])
    assert instruction == "Wait briefly for the current screen to finish loading."


@pytest.mark.parametrize(
    ("app_name", "expected_prefix"),
    [
        ("demo", "You are already inside the demo app. "),
        ("", "You are already inside the target app. "),
    ],
)
def test_make_free_instruction(
    translator: aitk_translator.UIKobeV2Translator, app_name: str, expected_prefix: str
) -> None:
    translator._app_name = app_name
    instruction = translator._make_free_instruction("buy shoes", "no graph is loaded")
    assert instruction.startswith(expected_prefix)
    assert "no graph is loaded" in instruction
    assert "buy shoes" in instruction


def test_decide_returns_the_chosen_option(
    monkeypatch: pytest.MonkeyPatch, translator: aitk_translator.UIKobeV2Translator
) -> None:
    client = _use_model(
        monkeypatch, translator, '{"choice": "c", "instruction": "tap the shoes result"}'
    )
    decision = translator._decide(translator._graphs["demo"], "buy shoes", "home", "screenshot")

    assert decision["type"] == "neighbor"
    assert decision["node"] == "results"
    assert decision["instruction"] == "tap the shoes result"
    assert (
        "State parameters: [query]"
        in client.completions.calls[0]["messages"][0]["content"][0]["text"]
    )


def test_decide_keeps_the_graph_instruction_when_the_model_omits_one(
    monkeypatch: pytest.MonkeyPatch, translator: aitk_translator.UIKobeV2Translator
) -> None:
    _use_model(monkeypatch, translator, '{"choice": "C"}')
    decision = translator._decide(translator._graphs["demo"], "buy shoes", "home", "screenshot")
    assert decision["instruction"] == "tap the first result"


def test_decide_falls_back_to_the_task_for_an_option_without_an_instruction(
    monkeypatch: pytest.MonkeyPatch, translator: aitk_translator.UIKobeV2Translator
) -> None:
    _use_model(monkeypatch, translator, '{"choice": "A"}')
    decision = translator._decide(translator._graphs["demo"], "buy shoes", "home", "screenshot")
    assert decision == {"letter": "A", "type": "done", "instruction": "buy shoes"}


def test_decide_reports_an_unparseable_reply(
    monkeypatch: pytest.MonkeyPatch, translator: aitk_translator.UIKobeV2Translator
) -> None:
    client = _use_model(monkeypatch, translator, "no json", "still no json")
    decision = translator._decide(translator._graphs["demo"], "buy shoes", "home", "screenshot")
    assert decision == {
        "type": "free",
        "instruction": "",
        "reason": "DECIDE response could not be parsed",
    }
    assert len(client.completions.calls) == 2


def test_decide_reports_an_unknown_choice(
    monkeypatch: pytest.MonkeyPatch, translator: aitk_translator.UIKobeV2Translator
) -> None:
    _use_model(monkeypatch, translator, '{"choice": "Z", "instruction": "tap something"}')
    decision = translator._decide(translator._graphs["demo"], "buy shoes", "home", "screenshot")
    assert decision["type"] == "free"
    assert decision["instruction"] == "tap something"
    assert decision["reason"] == "DECIDE returned unrecognized choice 'Z'"


def test_decide_omits_state_parameters_for_a_node_without_a_schema(
    monkeypatch: pytest.MonkeyPatch, translator: aitk_translator.UIKobeV2Translator
) -> None:
    client = _use_model(monkeypatch, translator, '{"choice": "A"}')
    translator._decide(translator._graphs["demo"], "buy shoes", "results", "screenshot")
    assert (
        "State parameters" not in client.completions.calls[0]["messages"][0]["content"][0]["text"]
    )


def test_runtime_image_embedding_requires_a_key(graph_dir: Path) -> None:
    config = {**_VLM_CONFIG, "image_embedding": {"model": "img-model"}}
    built = aitk_translator.UIKobeV2Translator(graph_dir=str(graph_dir), vlm_config=config)
    with pytest.raises(RuntimeError, match="Native Gemini image embedding requires"):
        built._get_runtime_image_embedding("screenshot")


def test_runtime_image_embedding_passes_the_configured_endpoint(
    monkeypatch: pytest.MonkeyPatch, translator: aitk_translator.UIKobeV2Translator
) -> None:
    seen: dict[str, Any] = {}

    def _fake_embedding(api_key: str, screenshot_b64: str, **kwargs: Any) -> list[float]:
        seen.update({"api_key": api_key, "screenshot": screenshot_b64, **kwargs})
        return [0.5, 0.5]

    monkeypatch.setattr(aitk_translator, "get_gemini_native_image_embedding", _fake_embedding)
    assert translator._get_runtime_image_embedding("shot") == [0.5, 0.5]
    assert seen == {
        "api_key": "image-key",
        "screenshot": "shot",
        "model": "img-model",
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
    }


@pytest.mark.usefixtures("no_sleep")
def test_compute_runtime_image_embedding_retries(
    monkeypatch: pytest.MonkeyPatch, translator: aitk_translator.UIKobeV2Translator
) -> None:
    attempts: list[int] = []

    def _flaky(*_args: Any, **_kwargs: Any) -> list[float]:
        attempts.append(1)
        if len(attempts) < 3:
            raise RuntimeError("rate limited")
        return [1.0]

    monkeypatch.setattr(aitk_translator, "get_gemini_native_image_embedding", _flaky)
    assert translator._compute_runtime_image_embedding_with_retry("shot", "demo", "home") == [1.0]
    assert len(attempts) == aitk_translator.RUNTIME_IMAGE_EMBEDDING_RETRIES + 1


@pytest.mark.usefixtures("no_sleep")
def test_compute_runtime_image_embedding_gives_up(
    monkeypatch: pytest.MonkeyPatch, translator: aitk_translator.UIKobeV2Translator
) -> None:
    def _always_fails(*_args: Any, **_kwargs: Any) -> list[float]:
        raise RuntimeError("rate limited")

    monkeypatch.setattr(aitk_translator, "get_gemini_native_image_embedding", _always_fails)
    with pytest.raises(RuntimeError, match="rate limited"):
        translator._compute_runtime_image_embedding_with_retry("shot", "demo", "home")


@pytest.fixture
def identifiable(
    monkeypatch: pytest.MonkeyPatch, translator: aitk_translator.UIKobeV2Translator
) -> aitk_translator.UIKobeV2Translator:
    """A translator whose page description and image embeddings come from fakes."""
    monkeypatch.setattr(
        aitk_translator,
        "describe_page_and_state",
        lambda *_args, **_kwargs: ("a home screen", {}, []),
    )
    monkeypatch.setattr(
        aitk_translator,
        "get_gemini_native_image_embedding",
        lambda *_args, **_kwargs: [1.0, 0.0],
    )
    G = translator._graphs["demo"]
    G.nodes["home"]["image_embedding"] = [1.0, 0.0]
    G.nodes["results"]["image_embedding"] = [0.0, 1.0]
    translator._current_graph = G
    translator._app_name = "demo"
    return translator


def test_identify_node_without_a_graph(translator: aitk_translator.UIKobeV2Translator) -> None:
    assert translator._identify_node("com.demo.app/.HomeActivity", "shot") == (None, "")


def test_identify_node_picks_the_model_choice(
    monkeypatch: pytest.MonkeyPatch, identifiable: aitk_translator.UIKobeV2Translator
) -> None:
    _use_model(monkeypatch, identifiable, "A")
    assert identifiable._identify_node("com.demo.app/.HomeActivity", "shot") == (
        "home",
        "a home screen",
    )


def test_identify_node_respects_a_rejection(
    monkeypatch: pytest.MonkeyPatch, identifiable: aitk_translator.UIKobeV2Translator
) -> None:
    _use_model(monkeypatch, identifiable, "none")
    assert identifiable._identify_node("com.demo.app/.HomeActivity", "shot") == (
        None,
        "a home screen",
    )


def test_identify_node_retries_an_unparseable_pick(
    monkeypatch: pytest.MonkeyPatch, identifiable: aitk_translator.UIKobeV2Translator
) -> None:
    client = _use_model(monkeypatch, identifiable, "hmm", "B")
    node_id, _ = identifiable._identify_node("com.demo.app/.HomeActivity", "shot")
    assert node_id == "results"
    assert len(client.completions.calls) == 2


def test_identify_node_without_candidates_in_the_current_package(
    identifiable: aitk_translator.UIKobeV2Translator,
) -> None:
    assert identifiable._identify_node("org.other.thing/.Main", "shot") == (None, "a home screen")


def test_identify_node_without_cached_embeddings(
    identifiable: aitk_translator.UIKobeV2Translator,
) -> None:
    G = identifiable._graphs["demo"]
    for node in G.nodes:
        del G.nodes[node]["image_embedding"]
    assert identifiable._identify_node("com.demo.app/.HomeActivity", "shot") == (
        None,
        "a home screen",
    )


def test_identify_node_propagates_an_embedding_failure(
    monkeypatch: pytest.MonkeyPatch, identifiable: aitk_translator.UIKobeV2Translator
) -> None:
    def _always_fails(*_args: Any, **_kwargs: Any) -> list[float]:
        raise RuntimeError("embedding down")

    monkeypatch.setattr(aitk_translator, "get_gemini_native_image_embedding", _always_fails)
    monkeypatch.setattr(aitk_translator.time, "sleep", lambda _seconds: None)
    with pytest.raises(RuntimeError, match="embedding down"):
        identifiable._identify_node("com.demo.app/.HomeActivity", "shot")


class _KeyboardProbe:
    def __init__(self, stdout: str) -> None:
        self.stdout = stdout


def test_call_action_agent_reports_the_keyboard_and_splits_the_history(
    monkeypatch: pytest.MonkeyPatch, translator: aitk_translator.UIKobeV2Translator
) -> None:
    seen: dict[str, Any] = {}

    def _fake_predict(
        _client: object, _shot: str, instruction: str, **kwargs: Any
    ) -> tuple[dict[str, Any], str]:
        seen["instruction"] = instruction
        seen.update(kwargs)
        return {"action": "tap", "x": 1, "y": 2}, "tapped Search | on the home screen"

    monkeypatch.setattr(
        aitk_translator.subprocess, "run", lambda *_a, **_k: _KeyboardProbe("mInputShown=true")
    )
    monkeypatch.setattr(aitk_translator, "predict_next_action", _fake_predict)

    action, observation, entry = translator._call_action_agent(
        "type shoes", ["earlier"], "shot", overall_task="buy shoes"
    )
    assert action == {"action": "tap", "x": 1, "y": 2}
    assert observation == "tapped Search"
    assert entry == "tapped Search | on the home screen"
    assert seen["instruction"].endswith(
        "the soft keyboard is currently visible — a text field is focused and ready for typing.)"
    )
    assert seen["overall_task"] == "buy shoes"


def test_call_action_agent_survives_a_missing_adb(
    monkeypatch: pytest.MonkeyPatch, translator: aitk_translator.UIKobeV2Translator
) -> None:
    def _no_adb(*_args: Any, **_kwargs: Any) -> _KeyboardProbe:
        raise FileNotFoundError("adb")

    monkeypatch.setattr(aitk_translator.subprocess, "run", _no_adb)
    monkeypatch.setattr(
        aitk_translator,
        "predict_next_action",
        lambda *_a, **_k: ({"action": "back"}, "went back"),
    )

    action, observation, entry = translator._call_action_agent("go back", [], "shot")
    assert action == {"action": "back"}
    assert observation == ""
    assert entry == "went back"


def test_make_response_wraps_the_action() -> None:
    raw = aitk_translator.UIKobeV2Translator._make_response("Action: tapped", {"action": "back"})
    assert json.loads(raw) == {
        "message": "[Android-App-Graph] Action: tapped",
        "aitk_action": {"action": "back"},
    }


def test_to_device_records_the_screen_size(
    translator: aitk_translator.UIKobeV2Translator,
) -> None:
    action = translator.to_device(
        json.dumps({"aitk_action": {"action": "tap", "x": 1, "y": 2}}), 1440, 3120
    )
    assert action == {"action": "tap", "x": 1, "y": 2}
    assert (translator._screen_w, translator._screen_h) == (1440, 3120)


def test_to_device_on_unparseable_json(translator: aitk_translator.UIKobeV2Translator) -> None:
    assert translator.to_device("not json", 1080, 1920) == {
        "action": "end",
        "answer": "parse error",
    }


@pytest.mark.parametrize("action", ['{"message": "no action"}', '"a string"', "[1, 2]"])
def test_to_device_without_an_action_object(
    translator: aitk_translator.UIKobeV2Translator, action: str
) -> None:
    assert translator.to_device(action, 1080, 1920) == {"action": "end", "answer": ""}


@pytest.fixture
def stepping(
    monkeypatch: pytest.MonkeyPatch, identifiable: aitk_translator.UIKobeV2Translator
) -> aitk_translator.UIKobeV2Translator:
    """``identifiable`` with the action agent and the adb keyboard probe faked out."""
    monkeypatch.setattr(aitk_translator.subprocess, "run", lambda *_a, **_k: _KeyboardProbe(""))
    monkeypatch.setattr(
        aitk_translator,
        "predict_next_action",
        lambda *_a, **_k: ({"action": "tap", "x": 5, "y": 6}, "tapped the result | on home"),
    )
    return identifiable


def _step_payload() -> tuple[dict[str, Any], dict[str, Any]]:
    state = {
        "screenshot": "shot",
        "activity": "com.demo.app/.HomeActivity",
        "package": "com.demo.app",
    }
    return state, {"actions": ["earlier"]}


def test_to_agent_opens_the_app_on_the_first_turn(
    translator: aitk_translator.UIKobeV2Translator,
) -> None:
    state, _ = _step_payload()
    response = json.loads(translator.to_agent("Open demo and buy shoes", state, {"actions": []}))
    assert response["aitk_action"] == {"action": "open", "app": "demo"}
    assert translator._app_name == "demo"


def test_to_agent_runs_the_loop(
    monkeypatch: pytest.MonkeyPatch, stepping: aitk_translator.UIKobeV2Translator
) -> None:
    _use_model(
        monkeypatch,
        stepping,
        "A",
        "The total is 9 EUR",
        '{"choice": "C", "instruction": "tap the shoes result"}',
    )
    stepping._app_opened = True
    state, history = _step_payload()

    response = json.loads(stepping.to_agent("buy shoes", state, history))

    assert response["aitk_action"] == {"action": "tap", "x": 5, "y": 6}
    assert response["message"] == "[Android-App-Graph] Action: tapped the result | on home"
    assert stepping._current_node == "home"
    assert stepping._step_count == 1
    assert stepping._memory.actions == ["tap the shoes result"]
    assert stepping._memory.info == ["The total is 9 EUR"]
    assert stepping._memory.observations == ["tapped the result"]


def test_step_ends_the_task_on_a_done_decision(
    monkeypatch: pytest.MonkeyPatch, stepping: aitk_translator.UIKobeV2Translator
) -> None:
    _use_model(monkeypatch, stepping, "A", "nothing", '{"choice": "A"}', "9 EUR")
    stepping._app_opened = True
    state, history = _step_payload()

    response = json.loads(stepping._step("buy shoes", state, history))
    assert response["aitk_action"] == {"action": "end", "answer": "9 EUR"}


def test_step_downgrades_an_early_end_to_a_wait(
    monkeypatch: pytest.MonkeyPatch, stepping: aitk_translator.UIKobeV2Translator
) -> None:
    """Only the DECIDE step may end a task; a grounding "end" waits one turn instead."""
    monkeypatch.setattr(
        aitk_translator,
        "predict_next_action",
        lambda *_a, **_k: ({"action": "end", "answer": "guessed"}, "gave up | on home"),
    )
    _use_model(monkeypatch, stepping, "A", "nothing", '{"choice": "C"}')
    stepping._app_opened = True
    state, history = _step_payload()

    response = json.loads(stepping._step("buy shoes", state, history))
    assert response["aitk_action"] == {"action": "wait", "time": 1}


def test_step_plans_a_free_action_when_no_node_matches(
    monkeypatch: pytest.MonkeyPatch, stepping: aitk_translator.UIKobeV2Translator
) -> None:
    _use_model(monkeypatch, stepping, "none", "nothing", "Tap the cart icon")
    stepping._app_opened = True
    state, history = _step_payload()

    json.loads(stepping._step("buy shoes", state, history))
    assert stepping._current_node is None
    assert stepping._memory.actions == ["Tap the cart icon"]


def test_step_resolves_the_graph_from_the_package(
    monkeypatch: pytest.MonkeyPatch, stepping: aitk_translator.UIKobeV2Translator
) -> None:
    _use_model(monkeypatch, stepping, "A", "nothing", '{"choice": "C"}')
    stepping._app_opened = True
    stepping._app_name = ""
    stepping._current_graph = None
    state, history = _step_payload()

    json.loads(stepping._step("buy shoes", state, history))
    assert stepping._app_name == "demo"


def test_step_warns_when_the_task_names_no_known_app(
    monkeypatch: pytest.MonkeyPatch, stepping: aitk_translator.UIKobeV2Translator
) -> None:
    """An unresolvable app falls straight through to the reactive loop."""
    _use_model(monkeypatch, stepping, "A", "nothing", '{"choice": "C"}')
    state, history = _step_payload()

    response = json.loads(stepping._step("do something unrelated", state, history))
    assert response["aitk_action"] == {"action": "tap", "x": 5, "y": 6}
    assert stepping._app_opened is True
