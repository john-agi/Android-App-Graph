"""Tests for the AITK translator: its parsing helpers, graph loading, the decision loop,
and the runtime client lifecycle over the real SDK and an offline transport."""

from __future__ import annotations

import base64
import json
import logging
import string
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import httpx
import pytest
from hypothesis import given
from hypothesis import strategies as st
from openrouter import OpenRouter
from openrouter.errors import ResponseValidationError

from android_app_graph import embedding_cache
from android_app_graph.adapters import aitk_translator
from android_app_graph.payloads import as_list, as_str, as_str_dict
from android_app_graph.utils import ModelSettings, open_client, vlm_utils
from android_app_graph.utils.graph_manager import GraphManager

_SCREENSHOT = b"not-really-a-png"
_LETTERS = "ABCDEFGH"
_NO_INFORMATION = '{"information": null}'


class _Interrupted(BaseException):
    """Stand-in for KeyboardInterrupt/SystemExit that stays clear of pytest's own."""


def _write_graph(
    tmp_path: Path,
    *,
    app: str = "demo",
    package_name: str | None = None,
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
                "package_name": f"com.{app}.app" if package_name is None else package_name,
                "nodes": nodes
                if nodes is not None
                else [{"id": "n1", "activity": "com.a.b/.Main"}],
                "edges": edges if edges is not None else [],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_after_last_think_tag_returns_none_without_a_tag() -> None:
    assert aitk_translator._after_last_think_tag("no tag here") is None


def test_after_last_think_tag_is_case_insensitive() -> None:
    assert aitk_translator._after_last_think_tag("x</THINK>  y  ") == "y"


def test_after_last_think_tag_uses_the_last_of_two_tags() -> None:
    assert aitk_translator._after_last_think_tag("a</think>b</think>c") == "c"


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
        ("B is a match", "B"),
        ("It is clearly B, a strong match", "B"),
        ("None of the others fit, so B", None),
        ("B is the best match. None of the others match.", None),
        ("None of the candidates match. A login form is visible.", "NONE"),
        ("NONE. A settings page is showing.", "NONE"),
        ("A.", "A"),
        ("Neither A nor B match; none of them.", None),
        ("B is close, but none match", None),
        ("b is the match", None),
        ("Answer: A because none of the others match", "A"),
        ("Answer: A login screen is shown, none of them match", "A"),
        ("Answer: A matches the home screen.", "A"),
        ("Answer: A login form; B is closer", "A"),
        ("Answer: A is the home screen; B and C show lists instead.", "A"),
        ("Answer: a login screen, so the best match is C", "C"),
        ("Decision: a settings page. Final pick: B", "B"),
        ("The best match is A because it shows a home screen.", "A"),
        ("I think A is right", "A"),
        ("A is the match", "A"),
        ("A matches the home screen", "A"),
        ("A login form is visible", None),
        ("Candidates: A looks right", "A"),
        ("Candidates: A login screen looks right", None),
        ("Answer: A", "A"),
        ("Answer: a", None),
        ("answer: NONE", "NONE"),
        ("Answer: A\nbecause the screen shows a list of results", "A"),
        ("Final answer: A\nbecause it matches", "A"),
        ("Answer: A\nit matches B's layout partially", "A"),
        ("A\nis the match", "A"),
        ("Answer: A\nWait, the header says Settings.\nFinal answer: C", "C"),
        ("Answer: A\nFinal answer: C", "C"),
        ("Final answer: C\nAnswer: A", "A"),
    ],
)
def test_parse_model_choice(raw: str, expected: str | None) -> None:
    assert aitk_translator._parse_model_choice(raw, "ABCD") == expected


def test_parse_model_choice_treats_a_sentence_initial_article_as_no_explicit_answer() -> None:
    """ "A login form is visible" names no explicit form ("Final answer:"/single
    token/``</think>``), and its bare "A" is a sentence-initial article followed
    by a noun, not a named letter -- so the parse must return ``None`` and let
    the caller's retry loop ask the model again, rather than guessing the
    article as a pick. "A is the match" is a pick: an article is never followed
    by a finite verb.
    """
    assert aitk_translator._parse_model_choice("A login form is visible", "ABCD") is None


def test_parse_model_choice_trusts_an_explicit_label_as_written() -> None:
    """A justification ("Answer: A because none of the others match") and a
    rejection ("Answer: A login screen is shown, none of them match") are
    textually indistinguishable after the label -- no rule scanning what
    follows it can separate the two without misreading one of them, as three
    rounds of trying proved. The label is trusted as written instead: once an
    explicit "Answer:"/"Decision:" names an uppercase letter, that letter is
    the answer, with no article check and no scan of the remainder for a
    later contrary signal.
    """
    assert (
        aitk_translator._parse_model_choice("Answer: A because none of the others match", "ABCD")
        == "A"
    )
    assert (
        aitk_translator._parse_model_choice(
            "Answer: A login screen is shown, none of them match", "ABCD"
        )
        == "A"
    )
    assert aitk_translator._parse_model_choice("Answer: A login form; B is closer", "ABCD") == "A"


def test_parse_model_choice_explicit_label_requires_an_uppercase_letter() -> None:
    """Only the label keyword itself ("answer"/"decision"/"final answer"/"final
    decision") is matched case-insensitively; the letter after it is not, so a
    lowercase article right after the colon ("Answer: a login screen...") is
    never read as a named letter -- that is what keeps the label's own case
    sensitivity from swallowing an article the way an unscoped case-insensitive
    match would. A lowercase "a" falls through to the free-text scan instead,
    which finds the real answer stated later, or nothing at all.
    """
    assert (
        aitk_translator._parse_model_choice(
            "Answer: a login screen, so the best match is C", "ABCD"
        )
        == "C"
    )
    assert aitk_translator._parse_model_choice("Answer: a", "ABCD") is None


def test_parse_model_choice_think_tag_offset_survives_case_folding_length_change() -> None:
    """``str.lower()`` can change a string's length (U+0130 "İ" lowers to the two
    code points "i̇"), so an offset found on the lower-cased copy and used to
    slice the original text can land in the wrong place.

    Three "İ" before the tag shift the miscalculated offset three characters
    into "xyzB", chopping the adjacent filler down to a standalone "B" that the
    recursive parse then accepts on its single-letter fast path -- even though
    the untouched remainder "xyzB" has no isolated letter to find at all
    (adjacent letters share no word boundary, so the correct answer is ``None``).
    """
    raw = "İ" * 3 + "</think>xyzB"
    assert aitk_translator._parse_model_choice(raw, "ABCD") is None


def test_parse_model_choice_rejects_an_empty_reply() -> None:
    assert aitk_translator._parse_model_choice("", "ABCD") is None


def test_parse_model_choice_rejects_multiple_letters() -> None:
    """``"BC" in "ABCD"`` is a true substring test, not membership of a single letter.

    A multi-letter reply must not be accepted verbatim: it has to fall through to the
    later parsing (which finds no single trailing letter here) and end as ``None``,
    which triggers the existing retry path rather than being silently used as a pick.
    """
    assert aitk_translator._parse_model_choice("BC", "ABCD") is None


def test_parse_model_choice_rejects_letters_outside_the_valid_set() -> None:
    assert aitk_translator._parse_model_choice("H", "ABC") is None


@given(st.sampled_from(_LETTERS), st.sampled_from(["", " ", "\n\t"]))
def test_parse_model_choice_accepts_any_bare_valid_letter(letter: str, padding: str) -> None:
    raw = f"{padding}{letter.lower()}{padding}"
    assert aitk_translator._parse_model_choice(raw, _LETTERS) == letter


@given(
    st.text(alphabet=string.ascii_lowercase + " ", min_size=1, max_size=40).filter(
        lambda s: len(s.strip()) != 1
    )
)
def test_parse_model_choice_never_picks_a_letter_from_lowercase_only_prose(
    sentence: str,
) -> None:
    """A lowercase word never counts as a letter pick — only an actual uppercase
    standalone letter does — so free-form lowercase prose with no explicit
    "Answer:"/"Final answer:" form can end in ``None`` or an explicit "NONE",
    but never in one of the option letters.
    """
    result = aitk_translator._parse_model_choice(sentence, _LETTERS)
    assert result is None or result == "NONE"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"information": "The total is $42.50"}', "The total is $42.50"),
        ('{"information": null}', ""),
        ('{"information": "Nothing in the cart; total is $0"}', "Nothing in the cart; total is $0"),
        ('{"information": "The search field shows \\"milk\\""}', 'The search field shows "milk"'),
        ('{"information": " seat 12A is free "}', " seat 12A is free "),
        ('The bus is shown.\n{"information": "It leaves at 14:05"}', "It leaves at 14:05"),
        ('<think>{"information": "draft"}</think>{"information": "final"}', "final"),
        ('```json\n{"information": null}\n```', ""),
    ],
)
def test_parse_record_output(raw: str, expected: str) -> None:
    assert aitk_translator._parse_record_output(raw) == expected


_UNREADABLE_RECORD_REPLIES = [
    "",
    "nothing",
    "'nothing'",
    "The total is 9 EUR",
    '{"answer": "The total is 9 EUR"}',
    '{"information": 9}',
    '{"information": ["9 EUR"]}',
    '{"information": ""}',
    '{"information": "  "}',
]


@pytest.mark.parametrize("raw", _UNREADABLE_RECORD_REPLIES)
def test_parse_record_output_rejects_a_reply_without_its_information(raw: str) -> None:
    with pytest.raises(aitk_translator.RecordReplyError, match=r"^\[RECORD\] "):
        aitk_translator._parse_record_output(raw)


def test_parse_record_output_shows_the_start_of_a_rejected_reply() -> None:
    raw = "Nothing new is shown. " * 20
    with pytest.raises(aitk_translator.RecordReplyError) as caught:
        aitk_translator._parse_record_output(raw)
    assert str(caught.value) == (
        f"[RECORD] the reply has no JSON object (reply begins {raw[:160]!r})"
    )


_fact_without_reply_markup = st.text(
    alphabet=st.characters(exclude_characters="`<"), min_size=1
).filter(str.strip)


@given(_fact_without_reply_markup)
def test_parse_record_output_returns_any_fact_as_written(fact: str) -> None:
    """The alphabet leaves out "`" and "<": a fence or a ``</think>`` inside the fact
    is read as the reply's own markup, which no JSON round-trip has to survive."""
    assert aitk_translator._parse_record_output(json.dumps({"information": fact})) == fact


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("", ""),
        ("   ", ""),
        ("Tap the Search button", "Tap the Search button"),
        ("<think>hmm</think>Tap the Search button", "Tap the Search button"),
        ("Instruction: Tap the Search button", "Tap the Search button"),
        ("Next action: Scroll down", "Scroll down"),
        ("<think> Scroll down to the totals", "Scroll down to the totals"),
    ],
)
def test_parse_one_step_instruction(raw: str, expected: str) -> None:
    assert aitk_translator._parse_one_step_instruction(raw) == expected


@pytest.mark.parametrize("instruction", ['Tap "Save"', '"Press enter"'])
def test_parse_one_step_instruction_keeps_every_quote(instruction: str) -> None:
    """The action model reads an instruction's quotes as written, so none is removed."""
    assert aitk_translator._parse_one_step_instruction(instruction) == instruction


def test_parse_one_step_instruction_drops_everything_before_a_closing_think_tag() -> None:
    assert aitk_translator._parse_one_step_instruction("Tap Search </think>") == ""


def test_parse_one_step_instruction_rejects_an_overlong_plan() -> None:
    assert aitk_translator._parse_one_step_instruction("x" * 501) == ""


def test_parse_one_step_instruction_think_tag_offset_survives_case_folding_length_change() -> None:
    """``str.lower()`` can change a string's length (U+0130 "İ" lowers to two
    code points), so the ``</think>`` offset must come from the original text,
    not the lower-cased copy used only to find the tag.
    """
    raw = "<think>İ reasoning</think>Tap the Search button"
    assert aitk_translator._parse_one_step_instruction(raw) == "Tap the Search button"


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
def test_parse_json_reply(raw: str, expected: dict[str, Any] | None) -> None:
    assert aitk_translator._parse_json_reply(raw) == expected


def test_parse_json_reply_falls_back_to_the_whole_text() -> None:
    """A </think> tag with no object after it must not hide an object before it."""
    raw = '{"choice": "A"} </think> nothing usable here'
    assert aitk_translator._parse_json_reply(raw) == {"choice": "A"}


def test_parse_json_reply_finds_the_think_tag_before_stripping_fences() -> None:
    """Fence-stripping the whole reply before looking for ``</think>`` would let a
    fenced JSON draft *inside* the think block eat the tag itself -- ``strip_json_fences``
    matches the first fence pair it finds -- so the fenced final answer that follows the
    tag would never be reached and the draft would be returned instead. The tag must be
    found on the raw text first, and fences stripped only from the slice after it.
    """
    raw = (
        "<think>Draft:\n"
        '```json\n{"choice": "B", "instruction": "draft"}\n```\n'
        "On reflection C is right.</think>\n"
        '```json\n{"choice": "C", "instruction": "final"}\n```'
    )
    assert aitk_translator._parse_json_reply(raw) == {"choice": "C", "instruction": "final"}


def test_parse_json_reply_think_tag_offset_survives_case_folding_length_change() -> None:
    """``str.lower()`` can change a string's length (U+0130 "İ" lowers to the two
    code points "i̇"), so an offset found on the lower-cased copy and used to
    slice the original text can land in the wrong place.

    Six "İ" before the tag shift the miscalculated offset six characters into
    the reply, chopping ``'{"a": {"choice": "WRONG"}, "choice": "B"}'`` down to
    ``'{"choice": "WRONG"}, "choice": "B"}'`` -- a self-contained (wrong) object
    that gets parsed and returned immediately, never reaching the correct
    top-level object.
    """
    raw = "İ" * 6 + '</think>{"a": {"choice": "WRONG"}, "choice": "B"}'
    assert aitk_translator._parse_json_reply(raw) == {
        "a": {"choice": "WRONG"},
        "choice": "B",
    }


_key_without_backtick = st.text(alphabet=st.characters(exclude_characters="`"), max_size=10)


@given(st.dictionaries(_key_without_backtick, st.integers(), max_size=5))
def test_parse_json_reply_round_trips_json_objects(payload: dict[str, int]) -> None:
    """A key alphabet that excludes the backtick: ``strip_json_fences`` fences on ```.

    A key containing two triple-backtick runs makes fence-stripping eat part of
    the serialized JSON, which is not a property of arbitrary JSON round-tripping.
    """
    assert aitk_translator._parse_json_reply(json.dumps(payload)) == payload


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
    graph = aitk_translator._load_graph_from_json(path)

    assert set(graph.nodes) == {"n1", "n2"}
    assert graph.nodes["n1"]["page_description"] == "home"
    assert graph.nodes["n1"]["state_schema"] == {"query": "str"}
    assert graph.nodes["n1"]["visit_count"] == 3
    assert graph.nodes["n1"]["reference_screenshot_path"] is None
    assert graph.nodes["n2"]["activity"] == ""
    assert graph.edges["n1", "n2"]["instructions"] == ["tap"]
    assert "schema_deltas" not in graph.edges["n1", "n2"]


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
    graph = aitk_translator._load_graph_from_json(path)
    assert graph.edges["n1", "n2"]["schema_deltas"] == [{"cart": {"before": 0, "after": 1}}]
    assert "schema_deltas" not in graph.edges["n2", "n1"]


def test_load_graph_from_json_records_the_reference_screenshot_path(tmp_path: Path) -> None:
    """The loader stores the screenshot's path, not its bytes: nothing reads the
    screenshot at load time, only _compute_missing_image_embeddings, lazily,
    for a node that still needs a vector.
    """
    path = _write_graph(tmp_path, nodes=[{"id": "n1"}, {"id": "n2"}])
    screenshots = path.parent / "demo_screenshots"
    screenshots.mkdir()
    (screenshots / "n1.png").write_bytes(_SCREENSHOT)

    graph = aitk_translator._load_graph_from_json(path)

    assert graph.nodes["n1"]["reference_screenshot_path"] == screenshots / "n1.png"
    assert graph.nodes["n2"]["reference_screenshot_path"] is None


def test_load_graph_from_json_falls_back_to_the_unaudited_screenshot_dir(tmp_path: Path) -> None:
    path = _write_graph(tmp_path, audited=True, nodes=[{"id": "n1"}])
    screenshots = path.parent / "demo_screenshots"
    screenshots.mkdir()
    (screenshots / "n1.png").write_bytes(_SCREENSHOT)

    graph = aitk_translator._load_graph_from_json(path)
    assert graph.nodes["n1"]["reference_screenshot_path"] is not None


@pytest.mark.parametrize(
    "payload",
    ["[]", '"a string"', '{"nodes": {}, "edges": []}'],
)
def test_load_graph_from_json_rejects_a_malformed_document(tmp_path: Path, payload: str) -> None:
    path = tmp_path / "demo.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(TypeError):
        aitk_translator._load_graph_from_json(path)


def test_load_graph_from_json_accepts_a_file_with_only_nodes(tmp_path: Path) -> None:
    """An absent "edges" key is an empty list, not a rejection, as in
    require_graph_shape, which does not fail a missing "nodes" or "edges" key.
    """
    path = tmp_path / "demo.json"
    path.write_text(json.dumps({"nodes": [{"id": "n1"}]}), encoding="utf-8")

    graph = aitk_translator._load_graph_from_json(path)

    assert list(graph.nodes) == ["n1"]
    assert graph.number_of_edges() == 0


def test_load_graph_from_json_narrows_present_but_null_node_and_edge_fields(
    tmp_path: Path,
) -> None:
    """``.get(key, default)`` never applies ``default`` to a stored ``null``.

    A node/edge with an explicit JSON ``null`` for a field that downstream code
    iterates or indexes into must load as the field's empty default, not ``None``.
    """
    path = _write_graph(
        tmp_path,
        nodes=[
            {
                "id": "n1",
                "activity": None,
                "page_description": None,
                "state_schema": None,
                "last_detail_snapshot": None,
            },
            {"id": "n2"},
        ],
        edges=[
            {
                "source": "n1",
                "target": "n2",
                "instructions": None,
                "instruction_templates": None,
                "target_observations": None,
                "schema_deltas": None,
            }
        ],
    )
    graph = aitk_translator._load_graph_from_json(path)

    assert graph.nodes["n1"]["activity"] == ""
    assert graph.nodes["n1"]["page_description"] == ""
    assert graph.nodes["n1"]["state_schema"] == {}
    assert graph.nodes["n1"]["last_detail_snapshot"] == {}
    assert graph.edges["n1", "n2"]["instructions"] == []
    assert graph.edges["n1", "n2"]["instruction_templates"] == []
    assert graph.edges["n1", "n2"]["target_observations"] == []
    assert "schema_deltas" not in graph.edges["n1", "n2"]


def test_load_graph_from_json_rejects_a_non_string_node_id(tmp_path: Path) -> None:
    path = _write_graph(tmp_path, nodes=[{"id": None}])
    with pytest.raises(TypeError, match="node id must be a string"):
        aitk_translator._load_graph_from_json(path)


def test_load_graph_from_json_rejects_a_node_without_an_id(tmp_path: Path) -> None:
    """A node missing "id" entirely must surface as a path-bearing TypeError
    from the shared require_graph_shape check (via require_known_edge_endpoints),
    not a bare KeyError('id') from indexing it directly further down.
    """
    path = _write_graph(tmp_path, nodes=[{"page_description": "x"}])
    with pytest.raises(TypeError, match="node id must be a string") as excinfo:
        aitk_translator._load_graph_from_json(path)
    assert str(path) in str(excinfo.value)


def test_load_graph_from_json_rejects_a_null_nodes_field_naming_the_path(
    tmp_path: Path,
) -> None:
    """``"nodes": null`` raises a ``TypeError`` naming the path, from the shared
    shape check that covers this loader's own list check too.
    """
    path = tmp_path / "demo.json"
    path.write_text(json.dumps({"nodes": None, "edges": []}), encoding="utf-8")
    with pytest.raises(TypeError, match="must contain list fields 'nodes' and 'edges'") as excinfo:
        aitk_translator._load_graph_from_json(path)
    assert str(path) in str(excinfo.value)


@pytest.mark.parametrize(
    "edge",
    [
        {"source": None, "target": "n2"},
        {"source": "n1", "target": None},
    ],
)
def test_load_graph_from_json_rejects_a_non_string_edge_endpoint(
    tmp_path: Path, edge: dict[str, Any]
) -> None:
    path = _write_graph(tmp_path, nodes=[{"id": "n1"}, {"id": "n2"}], edges=[edge])
    with pytest.raises(TypeError, match="edge endpoint must be a string"):
        aitk_translator._load_graph_from_json(path)


def test_load_graph_from_json_rejects_an_edge_to_an_undefined_node(tmp_path: Path) -> None:
    """A hand-edited or partially written graph must not grow a phantom neighbour:
    networkx's add_edge would otherwise silently create an attribute-less node
    for any endpoint the file's ``nodes`` list does not define.
    """
    path = _write_graph(
        tmp_path,
        nodes=[{"id": "n1"}],
        edges=[{"source": "n1", "target": "ghost"}],
    )
    with pytest.raises(ValueError, match="ghost"):
        aitk_translator._load_graph_from_json(path)


def test_load_graph_from_json_rejects_an_edge_before_adding_any_edge(tmp_path: Path) -> None:
    """The check runs before any edge is added: a graph with one valid edge and
    one edge to an undefined node must load none of its edges, not the valid one.
    """
    path = _write_graph(
        tmp_path,
        nodes=[{"id": "n1"}, {"id": "n2"}],
        edges=[
            {"source": "n1", "target": "n2"},
            {"source": "n2", "target": "ghost"},
        ],
    )
    with pytest.raises(ValueError, match="ghost"):
        aitk_translator._load_graph_from_json(path)


def test_load_graph_from_json_rejects_a_bad_edge_before_looking_up_any_screenshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A graph rejected for one bad edge must not first pay for every reference
    screenshot path lookup: edge-endpoint validation runs before the node loop
    looks up any of them, not after.
    """
    path = _write_graph(
        tmp_path,
        nodes=[{"id": "n1"}],
        edges=[{"source": "n1", "target": "ghost"}],
    )
    calls = 0
    original = aitk_translator.reference_screenshot_path

    def counting_wrapper(graph_path: Path, node_id: str) -> Path | None:
        nonlocal calls
        calls += 1
        return original(graph_path, node_id)

    monkeypatch.setattr(aitk_translator, "reference_screenshot_path", counting_wrapper)

    with pytest.raises(ValueError, match="ghost"):
        aitk_translator._load_graph_from_json(path)
    assert calls == 0


def test_load_all_graphs_skips_a_graph_with_an_edge_to_an_undefined_node(tmp_path: Path) -> None:
    _write_graph(
        tmp_path,
        app="broken",
        nodes=[{"id": "n1"}],
        edges=[{"source": "n1", "target": "ghost"}],
    )
    _write_graph(tmp_path, app="demo", nodes=_DEMO_NODES, edges=_DEMO_EDGES)
    built = aitk_translator.UIKobeV2Translator(graph_dir=str(tmp_path), vlm_config=_VLM_CONFIG)
    assert set(built._graphs) == {"demo"}


def test_load_all_graphs_tolerates_a_corrupt_embedding_sidecar(graph_dir: Path) -> None:
    """A corrupt cache file must not take down the whole app graph it caches for."""
    (graph_dir / "demo" / "demo.image_emb.json").write_text("{not json", encoding="utf-8")
    built = aitk_translator.UIKobeV2Translator(graph_dir=str(graph_dir), vlm_config=_VLM_CONFIG)
    assert set(built._graphs) == {"demo"}


def test_memory_starts_empty() -> None:
    memory = aitk_translator.Memory()
    assert memory.actions == memory.info == memory.observations == []
    assert memory.format() == "(empty)"


def test_memory_ignores_empty_info() -> None:
    memory = aitk_translator.Memory()
    memory.add_info("")
    assert memory.info == []


@pytest.mark.parametrize("info", ["nothing", "Nothing is scheduled on Friday"])
def test_memory_gives_the_word_nothing_no_special_meaning(info: str) -> None:
    memory = aitk_translator.Memory()
    memory.add_info(info)
    assert memory.info == [info]


def test_memory_ignores_an_empty_observation() -> None:
    memory = aitk_translator.Memory()
    memory.add_observation("")
    assert memory.observations == []


def test_memory_records_and_formats_every_section() -> None:
    memory = aitk_translator.Memory()
    memory.add_action("tapped Search")
    memory.add_info("the total is 9 EUR")
    memory.add_observation("a dialog appeared")

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


class _RecordedCompletions:
    def __init__(self, requests: list[httpx.Request]) -> None:
        self._requests = requests

    @property
    def calls(self) -> list[dict[str, object]]:
        return [as_str_dict(json.loads(request.content)) for request in self._requests]

    def prompt_text(self, index: int = 0) -> str:
        message = as_str_dict(as_list(self.calls[index]["messages"])[0])
        part = as_str_dict(as_list(message["content"])[0])
        return as_str(part["text"], "")


def test_chat_completion_content_returns_the_reply_content(
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    chat_response: Callable[..., httpx.Response],
) -> None:
    router_responses.extend([chat_response(content="first"), chat_response(content="second")])
    content = aitk_translator._chat_completion_content(
        router_client,
        settings=ModelSettings("m"),
        messages=[{"role": "user", "content": "test"}],
    )
    assert content == "first"


def test_chat_completion_content_records_its_usage_as_v2_chat(
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    chat_response: Callable[..., httpx.Response],
    chat_usage_totals: dict[str, float],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracker = vlm_utils.TokenTracker()
    monkeypatch.setattr(vlm_utils, "token_tracker", tracker)
    router_responses.append(chat_response(content="first"))
    aitk_translator._chat_completion_content(
        router_client,
        settings=ModelSettings("m"),
        messages=[{"role": "user", "content": "test"}],
    )
    assert tracker.summary() == {"v2_chat:m": chat_usage_totals}


@pytest.fixture(autouse=True)
def _openrouter_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every translator reads OPENROUTER_API_KEY; a test of its absence deletes it."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")


_VLM_CONFIG: dict[str, Any] = {
    "action": {"model": "action-model"},
    "image_embedding": {"model": "img-model"},
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
    assert translator._app_by_package == {"com.demo.app": "demo"}
    assert translator.action.model == "action-model"
    assert translator.image_embedding.model == "img-model"
    assert not hasattr(translator, "page_detail")


@pytest.mark.usefixtures("translator")
def test_construction_quiets_the_http_client_loggers() -> None:
    """The httpx/openrouter per-request loggers are quieted on construction, not on import."""
    assert logging.getLogger("httpx").level == logging.WARNING
    assert logging.getLogger("openrouter").level == logging.WARNING


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


@pytest.mark.parametrize("package_name", [None, ""])
def test_load_all_graphs_skips_a_graph_without_a_package_name_and_names_the_file(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, package_name: str | None
) -> None:
    unnamed = tmp_path / "unnamed" / "unnamed.json"
    unnamed.parent.mkdir()
    document: dict[str, Any] = {"nodes": [{"id": "n1"}], "edges": []}
    if package_name is not None:
        document["package_name"] = package_name
    unnamed.write_text(json.dumps(document), encoding="utf-8")
    _write_graph(tmp_path, app="demo", nodes=_DEMO_NODES, edges=_DEMO_EDGES)

    with caplog.at_level(logging.ERROR, logger=aitk_translator.logger.name):
        built = aitk_translator.UIKobeV2Translator(graph_dir=str(tmp_path), vlm_config=_VLM_CONFIG)

    assert set(built._graphs) == {"demo"}
    assert built._app_by_package == {"com.demo.app": "demo"}
    (error,) = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert str(unnamed) in error.getMessage()
    assert "package_name" in error.getMessage()


def test_load_all_graphs_loads_neither_graph_that_shares_a_package_name(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    first = _write_graph(tmp_path, app="demo", package_name="com.shared.app")
    second = _write_graph(tmp_path, app="clone", package_name="com.shared.app")
    _write_graph(tmp_path, app="other")

    with caplog.at_level(logging.ERROR, logger=aitk_translator.logger.name):
        built = aitk_translator.UIKobeV2Translator(graph_dir=str(tmp_path), vlm_config=_VLM_CONFIG)

    assert set(built._graphs) == {"other"}
    assert built._app_by_package == {"com.other.app": "other"}
    (error,) = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert str(first) in error.getMessage()
    assert str(second) in error.getMessage()
    assert "com.shared.app" in error.getMessage()


def test_load_all_graphs_skips_a_graph_whose_embeddings_fail_and_keeps_the_others(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    failing = _write_graph(tmp_path, app="failing")
    _write_graph(tmp_path, app="demo", nodes=_DEMO_NODES, edges=_DEMO_EDGES)
    load_image_embeddings = aitk_translator.load_image_embeddings

    def _fail_for_one_graph(
        graph_path: Path, *, model: str, node_ids: set[str]
    ) -> dict[str, list[float]]:
        if graph_path == failing:
            raise RuntimeError("embedding store unavailable")
        return load_image_embeddings(graph_path, model=model, node_ids=node_ids)

    monkeypatch.setattr(aitk_translator, "load_image_embeddings", _fail_for_one_graph)

    with caplog.at_level(logging.ERROR, logger=aitk_translator.logger.name):
        built = aitk_translator.UIKobeV2Translator(graph_dir=str(tmp_path), vlm_config=_VLM_CONFIG)

    assert built._app_by_package == {"com.demo.app": "demo"}
    assert set(built._graphs) == {"demo"}
    assert f"Failed to load graph {failing}" in caplog.messages


def test_load_all_graphs_embeds_no_screenshot_of_a_graph_it_does_not_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for app in ("demo", "clone"):
        path = _write_graph(tmp_path, app=app, package_name="com.shared.app")
        (path.parent / f"{app}_screenshots").mkdir()
        (path.parent / f"{app}_screenshots" / "n1.png").write_bytes(_SCREENSHOT)

    def _no_embedding(*_args: object, **_kwargs: object) -> list[float]:
        raise AssertionError("a graph that is not loaded needs no image embedding")

    monkeypatch.setattr(embedding_cache, "get_image_embedding", _no_embedding)

    built = aitk_translator.UIKobeV2Translator(graph_dir=str(tmp_path), vlm_config=_VLM_CONFIG)

    assert built._graphs == {}


def test_load_all_graphs_reads_the_embedding_sidecar(graph_dir: Path) -> None:
    sidecar = graph_dir / "demo" / "demo.image_emb.json"
    sidecar.write_text(
        json.dumps({"model": "img-model", "embeddings": {"home": [1.0, 0.0], "gone": [0.0, 1.0]}}),
        encoding="utf-8",
    )
    built = aitk_translator.UIKobeV2Translator(graph_dir=str(graph_dir), vlm_config=_VLM_CONFIG)
    assert built._graphs["demo"].nodes["home"]["image_embedding"] == [1.0, 0.0]
    assert "gone" not in built._graphs["demo"]


def test_load_all_graphs_opens_no_screenshot_when_the_sidecar_covers_every_node(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A node whose vector is already cached must never have its screenshot
    opened: the runtime loader stores only the path at load time, and
    _compute_missing_image_embeddings must encode nothing when every
    candidate is already covered by the sidecar.
    """
    path = _write_graph(tmp_path, app="demo", nodes=[{"id": "n1"}, {"id": "n2"}])
    screenshots = path.parent / "demo_screenshots"
    screenshots.mkdir()
    (screenshots / "n1.png").write_bytes(b"shot-n1")
    (screenshots / "n2.png").write_bytes(b"shot-n2")
    (path.parent / "demo.image_emb.json").write_text(
        json.dumps({"model": "img-model", "embeddings": {"n1": [1.0, 0.0], "n2": [0.0, 1.0]}}),
        encoding="utf-8",
    )

    calls = 0
    original_encode = embedding_cache.encode_screenshot_b64

    def counting_encode(screenshot_path: Path) -> str:
        nonlocal calls
        calls += 1
        return original_encode(screenshot_path)

    monkeypatch.setattr(embedding_cache, "encode_screenshot_b64", counting_encode)

    built = aitk_translator.UIKobeV2Translator(graph_dir=str(tmp_path), vlm_config=_VLM_CONFIG)

    assert set(built._graphs) == {"demo"}
    assert calls == 0


@pytest.mark.usefixtures("no_sleep")
def test_load_all_graphs_streams_screenshot_reads_lazily(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each screenshot is read only right before its own embedding call, not
    all up front: a cold cache must never hold every node's base64 payload in
    memory at once.
    """
    path = _write_graph(tmp_path, app="demo", nodes=[{"id": "n1"}, {"id": "n2"}, {"id": "n3"}])
    screenshots = path.parent / "demo_screenshots"
    screenshots.mkdir()
    for node_id in ("n1", "n2", "n3"):
        (screenshots / f"{node_id}.png").write_bytes(f"shot-{node_id}".encode())

    reads_so_far = 0
    original_encode = embedding_cache.encode_screenshot_b64

    def counting_encode(screenshot_path: Path) -> str:
        nonlocal reads_so_far
        reads_so_far += 1
        return original_encode(screenshot_path)

    monkeypatch.setattr(embedding_cache, "encode_screenshot_b64", counting_encode)

    reads_at_call: list[int] = []

    def fake_embedding(*_args: object, **_kwargs: object) -> list[float]:
        reads_at_call.append(reads_so_far)
        return [0.1, 0.2]

    monkeypatch.setattr(embedding_cache, "get_image_embedding", fake_embedding)

    aitk_translator.UIKobeV2Translator(graph_dir=str(tmp_path), vlm_config=_VLM_CONFIG)

    assert reads_at_call == [1, 2, 3]


@pytest.mark.usefixtures("no_sleep")
def test_load_all_graphs_writes_the_sidecar_once_per_graph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rewriting the sidecar after every computed node is O(N^2) for a cold
    cache; it must be written once, after the whole node loop, so a node whose
    embedding call fails persistently still leaves the nodes computed around
    it persisted and the graph still loads.

    ``n4`` has no screenshot and is never a compute candidate, only a
    pre-cached sidecar entry: it exercises the sidecar-loaded copy-back
    (``_load_all_graphs``'s own loop) alongside ``n1``/``n3``'s freshly
    computed one (``_compute_missing_image_embeddings``'s), so both must land
    on ``graph`` even though only the candidates it computed are its own to copy.
    """
    path = _write_graph(
        tmp_path, app="demo", nodes=[{"id": "n1"}, {"id": "n2"}, {"id": "n3"}, {"id": "n4"}]
    )
    (path.parent / "demo.image_emb.json").write_text(
        json.dumps({"model": "img-model", "embeddings": {"n4": [9.0, 9.0]}}), encoding="utf-8"
    )
    screenshots = path.parent / "demo_screenshots"
    screenshots.mkdir()
    shots = {node_id: f"shot-{node_id}".encode() for node_id in ("n1", "n2", "n3")}
    for node_id, data in shots.items():
        (screenshots / f"{node_id}.png").write_bytes(data)
    b64_by_node = {
        node_id: base64.b64encode(data).decode("ascii") for node_id, data in shots.items()
    }

    def _flaky(
        _client: OpenRouter, screenshot_b64: str, *_args: object, **_kwargs: object
    ) -> list[float]:
        if screenshot_b64 == b64_by_node["n2"]:
            msg = "embedding failed"
            raise RuntimeError(msg)
        return [1.0, 0.0]

    monkeypatch.setattr(embedding_cache, "get_image_embedding", _flaky)

    save_calls: list[dict[str, list[float]]] = []
    original_save = embedding_cache.save_image_embeddings

    def _tracking_save(graph_file: Path, embeddings: dict[str, list[float]], *, model: str) -> None:
        save_calls.append(dict(embeddings))
        original_save(graph_file, embeddings, model=model)

    monkeypatch.setattr(embedding_cache, "save_image_embeddings", _tracking_save)

    built = aitk_translator.UIKobeV2Translator(graph_dir=str(tmp_path), vlm_config=_VLM_CONFIG)

    assert set(built._graphs) == {"demo"}
    graph = built._graphs["demo"]
    assert graph.nodes["n4"]["image_embedding"] == [9.0, 9.0]  # from the sidecar, never recomputed
    assert graph.nodes["n1"]["image_embedding"] == [1.0, 0.0]  # freshly computed
    assert graph.nodes["n3"]["image_embedding"] == [1.0, 0.0]  # freshly computed
    assert len(save_calls) == 1
    assert save_calls[0] == {"n4": [9.0, 9.0], "n1": [1.0, 0.0], "n3": [1.0, 0.0]}
    assert embedding_cache.load_image_embeddings(
        path, model="img-model", node_ids={"n1", "n2", "n3", "n4"}
    ) == {
        "n4": [9.0, 9.0],
        "n1": [1.0, 0.0],
        "n3": [1.0, 0.0],
    }


@pytest.mark.usefixtures("no_sleep")
def test_load_all_graphs_persists_progress_before_an_interrupt_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """KeyboardInterrupt/SystemExit partway through the node loop must not discard
    the embeddings computed before it -- those are paid API calls. The node loop
    runs inside a ``finally`` so whatever was computed is written to the sidecar
    before the interrupt keeps propagating, and it does keep propagating: this
    is not a reason to drop or swallow it.
    """
    path = _write_graph(tmp_path, app="demo", nodes=[{"id": "n1"}, {"id": "n2"}, {"id": "n3"}])
    screenshots = path.parent / "demo_screenshots"
    screenshots.mkdir()
    shots = {node_id: f"shot-{node_id}".encode() for node_id in ("n1", "n2", "n3")}
    for node_id, data in shots.items():
        (screenshots / f"{node_id}.png").write_bytes(data)
    b64_by_node = {
        node_id: base64.b64encode(data).decode("ascii") for node_id, data in shots.items()
    }

    def _interrupted_at_n3(
        _client: OpenRouter, screenshot_b64: str, *_args: object, **_kwargs: object
    ) -> list[float]:
        if screenshot_b64 == b64_by_node["n3"]:
            raise _Interrupted
        return [1.0, 0.0]

    monkeypatch.setattr(embedding_cache, "get_image_embedding", _interrupted_at_n3)

    with pytest.raises(_Interrupted):
        aitk_translator.UIKobeV2Translator(graph_dir=str(tmp_path), vlm_config=_VLM_CONFIG)

    assert embedding_cache.load_image_embeddings(
        path, model="img-model", node_ids={"n1", "n2", "n3"}
    ) == {
        "n1": [1.0, 0.0],
        "n2": [1.0, 0.0],
    }


@pytest.mark.usefixtures("no_sleep")
def test_load_all_graphs_survives_a_sidecar_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A cache write failure is a cache failure, never a reason to drop a graph
    that loaded and computed its embeddings fine: the graph stays registered
    with the freshly computed vectors on its nodes, and only the write is logged.
    """
    path = _write_graph(tmp_path, app="demo", nodes=[{"id": "n1"}])
    screenshots = path.parent / "demo_screenshots"
    screenshots.mkdir()
    (screenshots / "n1.png").write_bytes(b"shot-n1")

    monkeypatch.setattr(embedding_cache, "get_image_embedding", lambda *_a, **_kw: [1.0, 0.0])

    def _raise_permission_error(
        _graph_file: Path, _embeddings: dict[str, list[float]], **_kwargs: object
    ) -> None:
        msg = "Permission denied"
        raise PermissionError(msg)

    monkeypatch.setattr(embedding_cache, "save_image_embeddings", _raise_permission_error)

    with caplog.at_level("ERROR"):
        built = aitk_translator.UIKobeV2Translator(graph_dir=str(tmp_path), vlm_config=_VLM_CONFIG)

    assert set(built._graphs) == {"demo"}
    assert built._graphs["demo"].nodes["n1"]["image_embedding"] == [1.0, 0.0]
    assert "demo" in caplog.text


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
    ["com.demo.app.free", "com.demo", "com.demo.app/.HomeActivity", "org.other.thing", ""],
)
def test_select_graph_matches_no_graph_for_another_package(
    translator: aitk_translator.UIKobeV2Translator, package: str
) -> None:
    """Only the exact package selects a graph; sharing a prefix with it is not enough."""
    translator._select_graph("com.demo.app")
    translator._select_graph(package)
    assert translator._current_graph is None
    assert translator._app_name == ""


def test_reset_task_state_clears_the_previous_task(
    translator: aitk_translator.UIKobeV2Translator,
) -> None:
    translator._app_opened = True
    translator._step_count = 7
    translator._memory.add_action("tapped Search")

    translator._reset_task_state()

    assert translator._app_opened is False
    assert translator._current_graph is None
    assert translator._step_count == 0
    assert translator._memory.actions == []


def test_reset_task_state_keeps_the_screen_size_aitk_reported(
    translator: aitk_translator.UIKobeV2Translator,
) -> None:
    """Screen size is a device property that survives task boundaries.

    It only ever changes when AITK calls to_device with the real size, never on
    a new task, so a task reset must not silently reset it back to the 1080x1920
    default while _call_action_agent keeps using it for grounding.
    """
    translator.to_device(json.dumps({"aitk_action": {"action": "wait", "time": 1}}), 1440, 3120)

    translator._reset_task_state()

    assert (translator._screen_w, translator._screen_h) == (1440, 3120)


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


@pytest.mark.parametrize("index", [1, 9])
def test_edge_effect_hint_is_empty_for_an_action_without_a_delta(
    translator: aitk_translator.UIKobeV2Translator, index: int
) -> None:
    edge = {"schema_deltas": [{"a": "one"}, None]}
    assert translator._edge_effect_hint(edge, index) == ""


def test_edge_effect_hint_without_an_index_merges_every_delta(
    translator: aitk_translator.UIKobeV2Translator,
) -> None:
    """Earlier deltas win: the merge keeps the first value seen for each key."""
    edge = {"schema_deltas": [{"a": "one"}, {"a": "two"}, "not a delta"]}
    assert translator._edge_effect_hint(edge) == "a: one"


def test_build_options_lists_done_self_loops_neighbours_and_free(
    translator: aitk_translator.UIKobeV2Translator,
) -> None:
    graph = translator._graphs["demo"]
    text, options = translator._build_options(graph, "home")

    assert [opt["type"] for opt in options] == ["done", "self_loop", "neighbor", "free"]
    assert [opt["letter"] for opt in options] == ["A", "B", "C", "D"]
    assert options[1]["instruction"] == "search for {query}"
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
    graph = translator._graphs["demo"]
    graph.add_edge(
        "results",
        "results",
        instructions=["scroll down", "scroll up"],
        target_observations=["more results"],
        instruction_templates=[],
        schema_deltas=[{"page": "1 -> 2"}],
    )
    text, options = translator._build_options(graph, "results")

    assert [opt["type"] for opt in options] == ["done", "self_loop", "self_loop", "free"]
    assert options[1]["instruction"] == "scroll down"
    assert options[2]["instruction"] == "scroll up"
    lines = text.splitlines()
    assert lines[1] == 'B) Stay here — "scroll down" → more results | changes: page: 1 -> 2'
    assert lines[2] == 'C) Stay here — "scroll up"'


def test_build_options_shows_each_self_loop_action_its_own_schema_delta(
    translator: aitk_translator.UIKobeV2Translator,
) -> None:
    """Each self-loop action shows its own changes, and one that changed nothing shows none."""
    gm = GraphManager()
    gm.graph.add_node("s0_list", page_description="shopping list")
    for action, instruction, delta in (
        (
            {"action": "swipe", "direction": "up"},
            "Scroll down the list",
            {"first_visible_item": {"before": "A", "after": "K"}},
        ),
        ({"action": "click", "x": 9, "y": 40}, "Tap the info icon", None),
        (
            {"action": "type", "text": "milk"},
            "Type 'milk'",
            {"search_query": {"before": None, "after": "milk"}},
        ),
    ):
        gm.add_edge(
            "s0_list",
            "s0_list",
            action,
            instruction=instruction,
            target_observation="list",
            schema_delta=delta,
        )

    text, _ = translator._build_options(gm.graph, "s0_list")

    assert text.splitlines()[1:4] == [
        'B) Stay here — "Scroll down the list" → list | changes: first_visible_item: A -> K',
        'C) Stay here — "Tap the info icon" → list',
        "D) Stay here — \"Type 'milk'\" → list | changes: search_query: None -> milk",
    ]


def test_build_options_lists_no_self_loop_action_without_an_instruction(
    translator: aitk_translator.UIKobeV2Translator,
) -> None:
    gm = GraphManager()
    gm.graph.add_node("s0_list", page_description="shopping list")
    for action, instruction, delta in (
        ({"action": "swipe"}, "Scroll down the list", {"top_item": {"before": "A", "after": "K"}}),
        ({"action": "click"}, "", {"sort": {"before": "name", "after": "price"}}),
        ({"action": "type"}, "Type 'milk'", {"query": {"before": None, "after": "milk"}}),
    ):
        gm.add_edge("s0_list", "s0_list", action, instruction=instruction, schema_delta=delta)

    text, _ = translator._build_options(gm.graph, "s0_list")

    assert text.splitlines()[1:-1] == [
        'B) Stay here — "Scroll down the list" | changes: top_item: A -> K',
        "C) Stay here — \"Type 'milk'\" | changes: query: None -> milk",
    ]


def test_build_options_skips_the_self_loop_when_listing_neighbours(
    translator: aitk_translator.UIKobeV2Translator,
) -> None:
    graph = translator._graphs["demo"]
    _, options = translator._build_options(graph, "home")
    assert [opt for opt in options if opt["type"] == "neighbor"] == [
        {
            "letter": "C",
            "type": "neighbor",
            "instruction": "tap the first result",
        }
    ]


class _UseModel(Protocol):
    def __call__(
        self, translator: aitk_translator.UIKobeV2Translator, *replies: str | None
    ) -> _RecordedCompletions: ...


@pytest.fixture
def use_model(
    monkeypatch: pytest.MonkeyPatch,
    router_http: httpx.Client,
    router_requests: list[httpx.Request],
    router_responses: list[httpx.Response],
    chat_response: Callable[..., httpx.Response],
) -> _UseModel:
    """Answer a translator's planning completions with ``replies``, in order."""

    def answer(
        translator: aitk_translator.UIKobeV2Translator, *replies: str | None
    ) -> _RecordedCompletions:
        monkeypatch.setattr(translator, "client", open_client("test-key", http_client=router_http))
        router_responses.extend(chat_response(content=reply) for reply in replies)
        return _RecordedCompletions(router_requests)

    return answer


def test_build_options_caps_at_26_entries_and_drops_the_least_visited_neighbours(
    translator: aitk_translator.UIKobeV2Translator,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An oversized menu is trimmed to 26 entries instead of raising IndexError.

    DONE and FREE always keep their slot, and the 24 remaining slots keep the
    most-visited neighbours (edge visit_count descending) rather than an
    arbitrary subset.
    """
    graph = translator._graphs["demo"]
    for i in range(30):
        neighbor = f"n{i}"
        graph.add_node(neighbor, page_description=f"screen {i}")
        graph.add_edge(
            "home",
            neighbor,
            instructions=[f"go to {i}"],
            target_observations=[],
            instruction_templates=[],
            visit_count=i,  # n29 is visited most, n0 least
        )

    with caplog.at_level("WARNING"):
        text, options = translator._build_options(graph, "home")

    assert len(options) == 26
    assert options[0]["type"] == "done"
    assert options[-1]["type"] == "free"
    # _Option carries no node id any more, so the kept neighbours are identified
    # by their edge instruction ("go to {i}"), unique per n{i} neighbour here.
    kept_instructions = {opt["instruction"] for opt in options if opt["type"] == "neighbor"}
    # The fixture's self-loop takes one slot, leaving 23 for 31 neighbours; "results"
    # ties with n0-n6 at visit_count 0 and is dropped with them.
    assert len(kept_instructions) == 23
    assert kept_instructions == {f"go to {i}" for i in range(7, 30)}
    assert "tap the first result" not in kept_instructions
    assert text.count("\n") == len(options) - 1
    assert "home" in caplog.text


@pytest.mark.parametrize(
    ("num_self_loops", "num_neighbors", "expected_self_loop", "expected_neighbor"),
    [
        (30, 3, 21, 3),
        (5, 40, 5, 19),
        (30, 30, 12, 12),
    ],
)
def test_build_options_reserves_half_the_middle_slots_for_neighbours(
    translator: aitk_translator.UIKobeV2Translator,
    num_self_loops: int,
    num_neighbors: int,
    expected_self_loop: int,
    expected_neighbor: int,
) -> None:
    """Neighbours are the only navigation options in the menu, so a node with many
    self-loop instructions (GraphManager adds one per distinct self-loop action)
    must never be able to evict every "Go to" option: neighbours keep at least
    half of the 24 middle slots, and self-loop instructions take the rest.
    """
    graph = translator._graphs["demo"]
    graph.remove_edge("home", "results")
    graph.remove_edge("home", "home")
    graph.add_edge(
        "home",
        "home",
        instructions=[f"self loop {i}" for i in range(num_self_loops)],
        target_observations=[],
        instruction_templates=[],
        schema_deltas=[],
    )
    for i in range(num_neighbors):
        neighbor = f"n{i}"
        graph.add_node(neighbor, page_description=f"screen {i}")
        graph.add_edge(
            "home",
            neighbor,
            instructions=[f"go to {i}"],
            target_observations=[],
            instruction_templates=[],
            visit_count=i,
        )

    _text, options = translator._build_options(graph, "home")

    kept_self_loop = [opt for opt in options if opt["type"] == "self_loop"]
    kept_neighbors = [opt for opt in options if opt["type"] == "neighbor"]
    assert len(kept_self_loop) == expected_self_loop
    assert len(kept_neighbors) == expected_neighbor
    assert len(options) == 2 + expected_self_loop + expected_neighbor


def test_record_info_stores_what_the_model_read_as_written(
    use_model: _UseModel, translator: aitk_translator.UIKobeV2Translator
) -> None:
    completions = use_model(translator, '{"information": "The search field shows \\"milk\\""}')
    translator._record_info("buy milk", "screenshot")
    assert translator._memory.info == ['The search field shows "milk"']
    assert _NO_INFORMATION in completions.prompt_text()


def test_record_info_stores_a_fact_that_starts_with_the_word_nothing(
    use_model: _UseModel, translator: aitk_translator.UIKobeV2Translator
) -> None:
    """RECORD keeps a fact that starts with the word "nothing", such as the answer to a
    task like "Do I have anything on Friday?"."""
    use_model(translator, '{"information": "Nothing is scheduled on Friday"}')
    translator._record_info("Do I have anything on Friday?", "screenshot")
    assert translator._memory.info == ["Nothing is scheduled on Friday"]


def test_record_info_stores_no_information_in_a_single_call(
    use_model: _UseModel, translator: aitk_translator.UIKobeV2Translator
) -> None:
    completions = use_model(translator, _NO_INFORMATION, '{"information": "unread"}')
    translator._record_info("buy shoes", "screenshot")
    assert translator._memory.info == []
    assert len(completions.calls) == 1


@pytest.mark.parametrize("raw", _UNREADABLE_RECORD_REPLIES)
def test_record_info_raises_on_an_unreadable_reply_without_asking_again(
    raw: str, use_model: _UseModel, translator: aitk_translator.UIKobeV2Translator
) -> None:
    translator._memory.add_info("The total is 9 EUR")
    completions = use_model(translator, raw, '{"information": "unread"}')
    with pytest.raises(aitk_translator.RecordReplyError, match=r"^\[RECORD\] "):
        translator._record_info("buy shoes", "screenshot")
    assert translator._memory.info == ["The total is 9 EUR"]
    assert len(completions.calls) == 1


def test_record_info_raises_on_a_reply_cut_off_at_its_token_limit(
    use_model: _UseModel,
    translator: aitk_translator.UIKobeV2Translator,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    chat_response: Callable[..., httpx.Response],
) -> None:
    use_model(translator)
    router_responses.append(
        chat_response(content='{"information": "The total', finish_reason="length")
    )
    with pytest.raises(
        aitk_translator.RecordReplyError, match=r"^\[RECORD\] the reply hit its token limit$"
    ):
        translator._record_info("buy shoes", "screenshot")
    assert translator._memory.info == []
    assert len(router_requests) == 1


def test_generate_answer_returns_the_first_non_empty_reply(
    use_model: _UseModel, translator: aitk_translator.UIKobeV2Translator
) -> None:
    completions = use_model(translator, "  ", "9 EUR")
    assert translator._generate_answer("buy shoes") == "9 EUR"
    assert len(completions.calls) == 2


def test_generate_answer_gives_up_after_the_retry(
    use_model: _UseModel, translator: aitk_translator.UIKobeV2Translator
) -> None:
    use_model(translator, "", "")
    assert translator._generate_answer("buy shoes") == ""


def test_plan_free_action_uses_the_planned_instruction(
    use_model: _UseModel, translator: aitk_translator.UIKobeV2Translator
) -> None:
    use_model(translator, "Tap the cart icon")
    instruction = translator._plan_free_action("buy shoes", "screenshot", "no graph", ["earlier"])
    assert instruction == "Tap the cart icon"


def test_plan_free_action_falls_back_when_the_planner_is_generic(
    use_model: _UseModel, translator: aitk_translator.UIKobeV2Translator
) -> None:
    use_model(translator, "Take the best action", "Take the best action")
    instruction = translator._plan_free_action("buy shoes", "screenshot", "no graph", [])
    assert instruction == "Wait briefly for the current screen to finish loading."


def test_a_truncated_v2_reply_is_not_asked_again_and_takes_the_fallback(
    use_model: _UseModel,
    translator: aitk_translator.UIKobeV2Translator,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    chat_response: Callable[..., httpx.Response],
) -> None:
    use_model(translator)
    router_responses.append(chat_response(content="Tap the", finish_reason="length"))
    instruction = translator._plan_free_action("buy shoes", "screenshot", "no graph", [])
    assert instruction == "Wait briefly for the current screen to finish loading."
    assert len(router_requests) == 1


@pytest.mark.parametrize("recover", [False, True])
def test_a_failed_v2_generation_is_asked_again_once_then_raises(
    recover: bool,
    use_model: _UseModel,
    translator: aitk_translator.UIKobeV2Translator,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    chat_response: Callable[..., httpx.Response],
) -> None:
    use_model(translator)
    router_responses.append(chat_response(content="", finish_reason="error"))
    router_responses.append(
        chat_response(content="Tap the cart icon")
        if recover
        else chat_response(content="", finish_reason="error")
    )

    def plan() -> str:
        return translator._plan_free_action("buy shoes", "screenshot", "no graph", [])

    if recover:
        assert plan() == "Tap the cart icon"
    else:
        with pytest.raises(vlm_utils.GenerationFailedError):
            plan()
    assert len(router_requests) == 2


def test_decide_returns_the_chosen_option(
    use_model: _UseModel, translator: aitk_translator.UIKobeV2Translator
) -> None:
    completions = use_model(translator, '{"choice": "c", "instruction": "tap the shoes result"}')
    decision = translator._decide(translator._graphs["demo"], "buy shoes", "home", "screenshot")

    assert decision["type"] == "neighbor"
    assert decision["instruction"] == "tap the shoes result"
    assert "State parameters: [query]" in completions.prompt_text()


def test_decide_keeps_the_graph_instruction_when_the_model_omits_one(
    use_model: _UseModel, translator: aitk_translator.UIKobeV2Translator
) -> None:
    use_model(translator, '{"choice": "C"}')
    decision = translator._decide(translator._graphs["demo"], "buy shoes", "home", "screenshot")
    assert decision["instruction"] == "tap the first result"


def test_decide_falls_back_to_the_task_for_an_option_without_an_instruction(
    use_model: _UseModel, translator: aitk_translator.UIKobeV2Translator
) -> None:
    """DONE's instruction is never read by _step, so falling back to the whole task
    here is harmless; the type-specific fallback still applies to it."""
    use_model(translator, '{"choice": "A"}')
    decision = translator._decide(translator._graphs["demo"], "buy shoes", "home", "screenshot")
    assert decision == {"type": "done", "instruction": "buy shoes"}


def test_decide_leaves_a_free_pick_without_an_instruction_empty(
    use_model: _UseModel, translator: aitk_translator.UIKobeV2Translator
) -> None:
    """A FREE pick that writes no instruction resolves to "", not the whole task.

    FREE never carries a built-in instruction, and _step's
    ``decision_type == "free" and not instruction`` branch is the only path that
    re-plans through _plan_free_action; it is unreachable if this falls back to
    the task text instead.
    """
    use_model(translator, '{"choice": "D"}')
    decision = translator._decide(translator._graphs["demo"], "buy shoes", "home", "screenshot")
    assert decision == {"type": "free", "instruction": ""}


def test_decide_uses_the_models_instruction_for_a_free_pick(
    use_model: _UseModel, translator: aitk_translator.UIKobeV2Translator
) -> None:
    use_model(translator, '{"choice": "D", "instruction": "Tap the cart icon"}')
    decision = translator._decide(translator._graphs["demo"], "buy shoes", "home", "screenshot")
    assert decision == {"type": "free", "instruction": "Tap the cart icon"}


def test_decide_reports_an_unparseable_reply(
    use_model: _UseModel, translator: aitk_translator.UIKobeV2Translator
) -> None:
    completions = use_model(translator, "no json", "still no json")
    decision = translator._decide(translator._graphs["demo"], "buy shoes", "home", "screenshot")
    assert decision == {
        "type": "free",
        "instruction": "",
        "reason": "DECIDE response could not be parsed",
    }
    assert len(completions.calls) == 2


def test_decide_treats_an_empty_json_object_as_parsed_not_failed(
    use_model: _UseModel, translator: aitk_translator.UIKobeV2Translator
) -> None:
    """DECIDE succeeds on ``is not None``, not truthiness: ``{}`` is a real (if
    unusable) parse, so it must reach the "unrecognized choice" path and not be
    retried as a parse failure."""
    completions = use_model(translator, "{}")
    decision = translator._decide(translator._graphs["demo"], "buy shoes", "home", "screenshot")
    assert decision["type"] == "free"
    assert decision["reason"] == "DECIDE returned unrecognized choice ''"
    assert len(completions.calls) == 1


def test_decide_reports_an_unknown_choice(
    use_model: _UseModel, translator: aitk_translator.UIKobeV2Translator
) -> None:
    use_model(translator, '{"choice": "Z", "instruction": "tap something"}')
    decision = translator._decide(translator._graphs["demo"], "buy shoes", "home", "screenshot")
    assert decision["type"] == "free"
    assert decision["instruction"] == "tap something"
    assert decision["reason"] == "DECIDE returned unrecognized choice 'Z'"


def test_decide_omits_state_parameters_for_a_node_without_a_schema(
    use_model: _UseModel, translator: aitk_translator.UIKobeV2Translator
) -> None:
    completions = use_model(translator, '{"choice": "A"}')
    translator._decide(translator._graphs["demo"], "buy shoes", "results", "screenshot")
    assert "State parameters" not in completions.prompt_text()


@pytest.mark.usefixtures("no_sleep")
def test_runtime_image_embedding_propagates_an_error_without_asking_again(
    monkeypatch: pytest.MonkeyPatch, translator: aitk_translator.UIKobeV2Translator
) -> None:
    """Only unusable output is asked again (tests/test_retrying.py); any other
    error reaches the caller from the first attempt."""
    calls: list[str] = []

    def _fails(
        _client: OpenRouter, screenshot_b64: str, *_args: object, **_kwargs: object
    ) -> list[float]:
        calls.append(screenshot_b64)
        raise RuntimeError("embedding failed")

    monkeypatch.setattr(embedding_cache, "get_image_embedding", _fails)
    with pytest.raises(RuntimeError, match="embedding failed"):
        translator._compute_runtime_image_embedding_with_retry("shot", "demo", "home")
    assert calls == ["shot"]


@pytest.fixture
def identifiable(
    monkeypatch: pytest.MonkeyPatch, translator: aitk_translator.UIKobeV2Translator
) -> aitk_translator.UIKobeV2Translator:
    """A translator whose image embeddings come from a fake."""
    monkeypatch.setattr(
        embedding_cache,
        "get_image_embedding",
        lambda *_args, **_kwargs: [1.0, 0.0],
    )
    graph = translator._graphs["demo"]
    graph.nodes["home"]["image_embedding"] = [1.0, 0.0]
    graph.nodes["results"]["image_embedding"] = [0.0, 1.0]
    translator._current_graph = graph
    translator._app_name = "demo"
    return translator


def test_identify_node_without_a_graph(translator: aitk_translator.UIKobeV2Translator) -> None:
    assert translator._identify_node("com.demo.app/.HomeActivity", "shot", "com.demo.app") is None


def test_identify_node_picks_the_model_choice(
    use_model: _UseModel, identifiable: aitk_translator.UIKobeV2Translator
) -> None:
    use_model(identifiable, "A")
    assert (
        identifiable._identify_node("com.demo.app/.HomeActivity", "shot", "com.demo.app") == "home"
    )


def test_identify_node_only_asks_the_action_model_to_pick_a_candidate(
    use_model: _UseModel, identifiable: aitk_translator.UIKobeV2Translator
) -> None:
    """The paper's runtime retrieves candidates by screenshot embedding and lets the
    runtime model pick one; it never describes the screen with the page-detail model."""
    completions = use_model(identifiable, "A")

    assert (
        identifiable._identify_node("com.demo.app/.HomeActivity", "shot", "com.demo.app") == "home"
    )

    assert len(completions.calls) == 1
    assert completions.calls[0]["model"] == "action-model"
    assert completions.prompt_text().startswith("You are looking at a mobile app screen.")


def test_identify_node_skips_a_stale_dimension_embedding_and_logs_once(
    use_model: _UseModel,
    identifiable: aitk_translator.UIKobeV2Translator,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A cached image embedding from a different embedding-model dimension must
    never be scored against the fresh query vector (see cosine_similarity's
    ValueError); it is skipped and warned about once, not per candidate.
    """
    graph = identifiable._graphs["demo"]
    graph.add_node(
        "stale",
        package="com.demo.app",
        activity="com.demo.app/.HomeActivity",
        page_description="a wrong-dimension screen",
        state_schema={},
        last_detail_snapshot={},
        reference_screenshot_path=None,
        visit_count=0,
        image_embedding=[1.0, 0.0, 0.0],
    )
    use_model(identifiable, "A")

    with caplog.at_level("WARNING"):
        node_id = identifiable._identify_node("com.demo.app/.HomeActivity", "shot", "com.demo.app")

    assert node_id == "home"
    stale_warnings = [
        m for m in caplog.messages if "image-embedding: embedding cache is stale" in m
    ]
    assert len(stale_warnings) == 1
    assert "query dim=2" in stale_warnings[0]
    assert "dim(s)=[3]" in stale_warnings[0]


def test_identify_node_respects_a_rejection(
    use_model: _UseModel, identifiable: aitk_translator.UIKobeV2Translator
) -> None:
    use_model(identifiable, "none")
    assert identifiable._identify_node("com.demo.app/.HomeActivity", "shot", "com.demo.app") is None


def test_identify_node_retries_an_unparseable_pick(
    use_model: _UseModel, identifiable: aitk_translator.UIKobeV2Translator
) -> None:
    completions = use_model(identifiable, "hmm", "B")
    node_id = identifiable._identify_node("com.demo.app/.HomeActivity", "shot", "com.demo.app")
    assert node_id == "results"
    assert len(completions.calls) == 2


def test_identify_node_retries_a_multi_letter_reply_instead_of_guessing(
    use_model: _UseModel, identifiable: aitk_translator.UIKobeV2Translator
) -> None:
    """A "BC"-shaped reply must not be silently mapped to a candidate by substring luck."""
    completions = use_model(identifiable, "BC", "B")
    node_id = identifiable._identify_node("com.demo.app/.HomeActivity", "shot", "com.demo.app")
    assert node_id == "results"
    assert len(completions.calls) == 2


def test_identify_node_retry_appends_the_parse_hint_and_recovers(
    use_model: _UseModel, identifiable: aitk_translator.UIKobeV2Translator
) -> None:
    """ "A login screen is the best match" reads as the sentence-initial article,
    not a pick, and a top-candidate reply of that shape is common from a model
    that ignores the requested format (A is always the top-similarity IDENTIFY
    candidate). The
    retry must repeat the identical prompt with a strict-format reminder appended
    rather than guess, so the second attempt (still "A") is what recovers it.
    """
    completions = use_model(identifiable, "A login screen is the best match", "A")
    node_id = identifiable._identify_node("com.demo.app/.HomeActivity", "shot", "com.demo.app")

    assert node_id == "home"
    assert len(completions.calls) == 2
    first_text = completions.prompt_text()
    second_text = completions.prompt_text(1)
    assert aitk_translator.V2_PARSE_RETRY_HINT not in first_text
    assert second_text.endswith(aitk_translator.V2_PARSE_RETRY_HINT)


def test_identify_node_succeeds_on_the_first_attempt_without_a_hint(
    use_model: _UseModel, identifiable: aitk_translator.UIKobeV2Translator
) -> None:
    completions = use_model(identifiable, "A")
    node_id = identifiable._identify_node("com.demo.app/.HomeActivity", "shot", "com.demo.app")

    assert node_id == "home"
    assert len(completions.calls) == 1
    text = completions.prompt_text()
    assert aitk_translator.V2_PARSE_RETRY_HINT not in text


def test_identify_node_without_candidates_in_the_current_package(
    identifiable: aitk_translator.UIKobeV2Translator,
) -> None:
    assert identifiable._identify_node("org.other.thing/.Main", "shot", "org.other.thing") is None


def test_identify_node_without_cached_embeddings(
    identifiable: aitk_translator.UIKobeV2Translator,
) -> None:
    graph = identifiable._graphs["demo"]
    for node in graph.nodes:
        del graph.nodes[node]["image_embedding"]
    assert identifiable._identify_node("com.demo.app/.HomeActivity", "shot", "com.demo.app") is None


def test_identify_node_and_build_options_tolerate_a_graph_with_null_json_fields(
    monkeypatch: pytest.MonkeyPatch, use_model: _UseModel, tmp_path: Path
) -> None:
    """A graph loaded with present-but-null fields must not crash IDENTIFY or DECIDE.

    Before the loader narrowed these fields, ``for k in data.get("state_schema", {})``
    and ``enumerate(edge_data.get("instructions", []))`` raised ``TypeError`` whenever the
    JSON stored an explicit ``null`` rather than omitting the key.
    """
    _write_graph(
        tmp_path,
        app="demo",
        nodes=[
            {
                "id": "home",
                "activity": None,
                "page_description": "home screen",
                "state_schema": None,
            }
        ],
        edges=[
            {
                "source": "home",
                "target": "home",
                "instructions": None,
                "instruction_templates": None,
                "target_observations": None,
            }
        ],
    )
    monkeypatch.setattr(embedding_cache, "get_image_embedding", lambda *_a, **_kw: [1.0, 0.0])
    translator = aitk_translator.register({"graph_dir": str(tmp_path), "vlm_config": _VLM_CONFIG})
    graph = translator._graphs["demo"]
    graph.nodes["home"]["image_embedding"] = [1.0, 0.0]
    translator._current_graph = graph
    translator._app_name = "demo"

    _text, options = translator._build_options(graph, "home")
    assert [opt["type"] for opt in options] == ["done", "free"]

    use_model(translator, "A")
    assert translator._identify_node("", "shot", "com.demo.app") == "home"


@pytest.mark.usefixtures("no_sleep")
def test_identify_node_propagates_an_embedding_failure(
    monkeypatch: pytest.MonkeyPatch, identifiable: aitk_translator.UIKobeV2Translator
) -> None:
    def _always_fails(*_args: object, **_kwargs: object) -> list[float]:
        raise RuntimeError("embedding down")

    monkeypatch.setattr(embedding_cache, "get_image_embedding", _always_fails)
    with pytest.raises(RuntimeError, match="embedding down"):
        identifiable._identify_node("com.demo.app/.HomeActivity", "shot", "com.demo.app")


def test_call_action_agent_reports_the_keyboard_and_splits_the_history(
    monkeypatch: pytest.MonkeyPatch, translator: aitk_translator.UIKobeV2Translator
) -> None:
    seen: dict[str, object] = {}

    def _fake_predict(
        _client: object, _shot: str, instruction: str, **kwargs: object
    ) -> tuple[dict[str, Any], str]:
        seen["instruction"] = instruction
        seen.update(kwargs)
        return {"action": "tap", "x": 1, "y": 2}, "tapped Search | on the home screen"

    monkeypatch.setattr(aitk_translator.device, "soft_keyboard_hint", lambda: " (keyboard up)")
    monkeypatch.setattr(aitk_translator, "predict_next_action", _fake_predict)

    action, observation, entry = translator._call_action_agent(
        "type shoes", ["earlier"], "shot", overall_task="buy shoes"
    )
    assert action == {"action": "tap", "x": 1, "y": 2}
    assert observation == "tapped Search"
    assert entry == "tapped Search | on the home screen"
    assert seen["instruction"] == "type shoes (keyboard up)"
    assert seen["overall_task"] == "buy shoes"


def test_call_action_agent_without_a_keyboard_hint(
    monkeypatch: pytest.MonkeyPatch, translator: aitk_translator.UIKobeV2Translator
) -> None:
    monkeypatch.setattr(aitk_translator.device, "soft_keyboard_hint", lambda: "")
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


@pytest.mark.parametrize(
    "action",
    [
        '{"message": "no action"}',
        '"a string"',
        "[1, 2]",
        '{"aitk_action": {}}',
        '{"aitk_action": {"answer": "no action key"}}',
    ],
)
def test_to_device_without_an_action_object(
    translator: aitk_translator.UIKobeV2Translator, action: str
) -> None:
    assert translator.to_device(action, 1080, 1920) == {"action": "end", "answer": ""}


@pytest.fixture
def stepping(
    monkeypatch: pytest.MonkeyPatch, identifiable: aitk_translator.UIKobeV2Translator
) -> aitk_translator.UIKobeV2Translator:
    """``identifiable`` with the action agent and the adb keyboard probe faked out."""
    monkeypatch.setattr(aitk_translator.device, "soft_keyboard_hint", lambda: "")
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


def test_to_agent_without_an_actions_key_resets_state(
    translator: aitk_translator.UIKobeV2Translator,
) -> None:
    """``_step`` reads ``history.get("actions", [])``; ``to_agent`` must match it
    instead of indexing ``history["actions"]``, or a history without that key
    (a first turn AITK sends bare) raises a KeyError before ``_step`` even runs.
    """
    state, _ = _step_payload()
    response = json.loads(translator.to_agent("Open demo and buy shoes", state, {}))
    assert response["aitk_action"] == {"action": "open", "app": "demo"}
    assert translator._app_name == "demo"


def test_to_agent_runs_the_loop(
    use_model: _UseModel, stepping: aitk_translator.UIKobeV2Translator
) -> None:
    turn_replies = (
        "A",
        '{"information": "The total is 9 EUR"}',
        '{"choice": "C", "instruction": "tap the shoes result"}',
    )
    use_model(stepping, *turn_replies, *turn_replies)
    stepping._app_opened = True
    state, history = _step_payload()

    for turn in range(1, 3):
        response = json.loads(stepping.to_agent("buy shoes", state, history))
        assert response["aitk_action"] == {"action": "tap", "x": 5, "y": 6}
        assert response["message"] == "[Android-App-Graph] Action: tapped the result | on home"
        assert stepping._step_count == turn
    assert stepping._memory.actions == ["tap the shoes result"] * 2
    assert stepping._memory.info == ["The total is 9 EUR"] * 2
    assert stepping._memory.observations == ["tapped the result"] * 2


def test_to_agent_stops_the_run_on_an_unreadable_record_reply(
    use_model: _UseModel, stepping: aitk_translator.UIKobeV2Translator
) -> None:
    completions = use_model(stepping, "A", "nothing", '{"choice": "C"}')
    stepping._app_opened = True
    state, history = _step_payload()

    with pytest.raises(aitk_translator.RecordReplyError):
        stepping.to_agent("buy shoes", state, history)
    assert len(completions.calls) == 2
    assert stepping._memory.actions == []


def test_step_ends_the_task_on_a_done_decision(
    use_model: _UseModel, stepping: aitk_translator.UIKobeV2Translator
) -> None:
    use_model(stepping, "A", _NO_INFORMATION, '{"choice": "A"}', "9 EUR")
    stepping._app_opened = True
    state, history = _step_payload()

    response = json.loads(stepping._step("buy shoes", state, history))
    assert response["aitk_action"] == {"action": "end", "answer": "9 EUR"}


def test_step_downgrades_an_early_end_to_a_wait(
    monkeypatch: pytest.MonkeyPatch,
    use_model: _UseModel,
    stepping: aitk_translator.UIKobeV2Translator,
) -> None:
    """Only the DECIDE step may end a task; a grounding "end" waits one turn instead."""
    monkeypatch.setattr(
        aitk_translator,
        "predict_next_action",
        lambda *_a, **_k: ({"action": "end", "answer": "guessed"}, "gave up | on home"),
    )
    use_model(stepping, "A", _NO_INFORMATION, '{"choice": "C"}')
    stepping._app_opened = True
    state, history = _step_payload()

    response = json.loads(stepping._step("buy shoes", state, history))
    assert response["aitk_action"] == {"action": "wait", "time": 1}


def test_step_plans_a_free_action_when_no_node_matches(
    use_model: _UseModel, stepping: aitk_translator.UIKobeV2Translator
) -> None:
    use_model(stepping, "none", _NO_INFORMATION, "Tap the cart icon")
    stepping._app_opened = True
    state, history = _step_payload()

    json.loads(stepping._step("buy shoes", state, history))
    assert stepping._memory.actions == ["Tap the cart icon"]


def test_step_replans_when_decide_produces_no_instruction(
    use_model: _UseModel, stepping: aitk_translator.UIKobeV2Translator
) -> None:
    """A matched node whose DECIDE reply never parses must still re-plan through
    _plan_free_action rather than executing the action agent on an empty instruction."""
    use_model(
        stepping,
        "A",  # IDENTIFY
        _NO_INFORMATION,  # RECORD
        "not json",  # DECIDE attempt 1: fails to parse
        "still not json",  # DECIDE attempt 2: fails to parse -> type="free", instruction=""
        "Tap the search bar",  # free-action fallback plan
    )
    stepping._app_opened = True
    state, history = _step_payload()

    json.loads(stepping._step("buy shoes", state, history))
    assert stepping._memory.actions == ["Tap the search bar"]


def test_step_replans_when_a_free_pick_has_no_instruction(
    use_model: _UseModel, stepping: aitk_translator.UIKobeV2Translator
) -> None:
    """A FREE pick without an instruction re-plans through _plan_free_action.

    The alternative is sending the whole task text as one instruction and
    recording it as the completed action.
    """
    use_model(
        stepping,
        "A",  # IDENTIFY
        _NO_INFORMATION,  # RECORD
        '{"choice": "D"}',  # DECIDE picks FREE, writes no instruction
        "Tap the search bar",  # free-action fallback plan
    )
    stepping._app_opened = True
    state, history = _step_payload()

    json.loads(stepping._step("buy shoes", state, history))
    assert stepping._memory.actions == ["Tap the search bar"]


def test_step_resolves_the_graph_from_the_package(
    use_model: _UseModel, stepping: aitk_translator.UIKobeV2Translator
) -> None:
    use_model(stepping, "A", _NO_INFORMATION, '{"choice": "C"}')
    stepping._app_opened = True
    stepping._app_name = ""
    stepping._current_graph = None
    state, history = _step_payload()

    json.loads(stepping._step("buy shoes", state, history))
    assert stepping._app_name == "demo"


def test_step_uses_no_graph_while_no_graph_matches_the_foreground_package(
    use_model: _UseModel, stepping: aitk_translator.UIKobeV2Translator
) -> None:
    """The app the task named only decides what to open; the foreground decides the graph."""
    use_model(stepping, _NO_INFORMATION, "Tap the Demo icon")
    stepping._app_opened = True
    state, history = _step_payload()

    stepping._step("Open demo", {**state, "package": "com.launcher"}, history)

    assert stepping._current_graph is None
    assert stepping._memory.actions == ["Tap the Demo icon"]


_SHOP_NODES: list[dict[str, Any]] = [
    {"id": "cart", "activity": "com.shop.app/.CartActivity", "page_description": "cart"}
]


@pytest.fixture
def two_apps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> aitk_translator.UIKobeV2Translator:
    _write_graph(tmp_path, app="demo", nodes=_DEMO_NODES, edges=_DEMO_EDGES)
    _write_graph(tmp_path, app="shop", nodes=_SHOP_NODES)
    built = aitk_translator.register({"graph_dir": str(tmp_path), "vlm_config": _VLM_CONFIG})
    built._graphs["demo"].nodes["home"]["image_embedding"] = [1.0, 0.0]
    built._graphs["demo"].nodes["results"]["image_embedding"] = [0.0, 1.0]
    built._graphs["shop"].nodes["cart"]["image_embedding"] = [1.0, 0.0]
    monkeypatch.setattr(embedding_cache, "get_image_embedding", lambda *_a, **_kw: [1.0, 0.0])
    monkeypatch.setattr(aitk_translator.device, "soft_keyboard_hint", lambda: "")
    monkeypatch.setattr(
        aitk_translator,
        "predict_next_action",
        lambda *_a, **_k: ({"action": "tap", "x": 5, "y": 6}, "tapped | on screen"),
    )
    built._app_opened = True
    return built


def test_step_switches_to_the_graph_of_the_package_now_in_the_foreground(
    use_model: _UseModel, two_apps: aitk_translator.UIKobeV2Translator
) -> None:
    """The graph is chosen on every step, so a second app's steps use its own graph."""
    use_model(
        two_apps,
        *("A", _NO_INFORMATION, '{"choice": "D", "instruction": "Open the shop app"}'),
        *("A", _NO_INFORMATION, '{"choice": "B", "instruction": "Tap checkout"}'),
    )
    history = {"actions": ["earlier"]}

    two_apps._step(
        "buy shoes in shop",
        {"screenshot": "shot", "activity": "com.demo.app/.HomeActivity", "package": "com.demo.app"},
        history,
    )
    assert two_apps._current_graph is two_apps._graphs["demo"]

    two_apps._step(
        "buy shoes in shop",
        {"screenshot": "shot", "activity": "com.shop.app/.CartActivity", "package": "com.shop.app"},
        history,
    )
    assert two_apps._current_graph is two_apps._graphs["shop"]
    assert two_apps._memory.actions == ["Open the shop app", "Tap checkout"]


def _stepping_over_one_graph(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    package_name: str,
    nodes: list[dict[str, Any]],
    image_embeddings: dict[str, list[float]],
) -> aitk_translator.UIKobeV2Translator:
    _write_graph(tmp_path, app="app", package_name=package_name, nodes=nodes)
    built = aitk_translator.register({"graph_dir": str(tmp_path), "vlm_config": _VLM_CONFIG})
    for node_id, vector in image_embeddings.items():
        built._graphs["app"].nodes[node_id]["image_embedding"] = vector
    monkeypatch.setattr(embedding_cache, "get_image_embedding", lambda *_a, **_kw: [1.0, 0.0])
    monkeypatch.setattr(aitk_translator.device, "soft_keyboard_hint", lambda: "")
    monkeypatch.setattr(
        aitk_translator,
        "predict_next_action",
        lambda *_a, **_k: ({"action": "tap", "x": 5, "y": 6}, "tapped | on screen"),
    )
    built._app_opened = True
    return built


def test_step_offers_only_the_nodes_of_the_foreground_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    use_model: _UseModel,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Messages and Gmail nodes stay in separate packages although they share their
    first three dot-segments, so a Messages screen is never offered a Gmail node."""
    messages = "com.google.android.apps.messaging"
    gmail = "com.google.android.gm"
    conversations = f"{messages}.ui.ConversationListActivity"
    stepping = _stepping_over_one_graph(
        tmp_path,
        monkeypatch,
        messages,
        nodes=[
            {"id": "inbox", "activity": conversations, "package": messages},
            {"id": "mail", "activity": f"{gmail}.ConversationListActivityGmail", "package": gmail},
        ],
        image_embeddings={"inbox": [0.6, 0.8], "mail": [1.0, 0.0]},
    )
    use_model(stepping, "A", _NO_INFORMATION, '{"choice": "B", "instruction": "Open the chat"}')
    state = {"screenshot": "shot", "activity": conversations, "package": messages}

    with caplog.at_level(logging.INFO, logger=aitk_translator.logger.name):
        stepping._step("Read the latest message", state, {"actions": []})

    assert "[IDENTIFY] identified node=inbox" in caplog.messages


def test_step_identifies_an_android_world_screen_in_a_graph_saved_without_node_packages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    use_model: _UseModel,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A screen that Android World reports as ``package/class`` matches the nodes of
    its package although the graph holds AITK's class names, which for Chrome do not
    start with the package."""
    chrome = "com.android.chrome"
    tabbed = "org.chromium.chrome.browser.ChromeTabbedActivity"
    stepping = _stepping_over_one_graph(
        tmp_path,
        monkeypatch,
        chrome,
        nodes=[{"id": "tabs", "activity": tabbed}],
        image_embeddings={"tabs": [1.0, 0.0]},
    )
    use_model(stepping, "A", _NO_INFORMATION, '{"choice": "B", "instruction": "Open a new tab"}')
    state = {"screenshot": "shot", "activity": f"{chrome}/{tabbed}", "package": chrome}

    with caplog.at_level(logging.INFO, logger=aitk_translator.logger.name):
        stepping._step("Open a new tab", state, {"actions": []})

    assert "[IDENTIFY] identified node=tabs" in caplog.messages


def test_step_warns_when_the_task_names_no_known_app(
    use_model: _UseModel, stepping: aitk_translator.UIKobeV2Translator
) -> None:
    """An unresolvable app falls straight through to the reactive loop."""
    use_model(stepping, "A", _NO_INFORMATION, '{"choice": "C"}')
    state, history = _step_payload()

    response = json.loads(stepping._step("do something unrelated", state, history))
    assert response["aitk_action"] == {"action": "tap", "x": 5, "y": 6}
    assert stepping._app_opened is True


def test_last_answer_signal_finds_none_after_a_szlig_heavy_prefix() -> None:
    """``str.upper()`` can change a string's length ("ß" -> "SS"); the NONE scan
    must run directly on the original text with ``re.IGNORECASE``, never
    against an uppercased copy, or a long enough case-folding prefix could
    shift a naive offset-based scan past the token entirely.
    """
    raw = "Straße Maße Größe Füße Süße Soße Fuß: none of them match"
    assert aitk_translator._last_answer_signal(raw, "ABCD") == "NONE"


def test_last_answer_signal_returns_the_last_valid_letter() -> None:
    assert aitk_translator._last_answer_signal("I see A here, but B is better", "ABCD") == "B"


def test_last_answer_signal_is_ambiguous_when_both_a_letter_and_none_are_present() -> None:
    """A rejection-then-letter ("Neither A nor B match; none of them.") and a
    letter-then-rejection ("B is the best match. None of the others match.")
    are mirror images of each other; no positional rule can tell which one
    the model meant, so carrying both kinds of signal is ambiguous and must
    be resolved by asking again, not by picking whichever came last.
    """
    assert (
        aitk_translator._last_answer_signal("Neither A nor B match; none of them.", "ABCD") is None
    )
    assert (
        aitk_translator._last_answer_signal(
            "B is the best match. None of the others match.", "ABCD"
        )
        is None
    )


def test_last_answer_signal_skips_a_sentence_initial_article() -> None:
    """A standalone "A" followed by a lowercase word on the same line, at the
    start of the text, reads as the English article, not a named answer
    letter, so it is not a signal.
    """
    assert aitk_translator._last_answer_signal("A login form is visible", "ABCD") is None


def test_last_answer_signal_skips_an_article_after_a_colon() -> None:
    """A colon starts a new sentence too, e.g. "Candidates: A looks right", so a
    standalone "A" right after one reads as the article, not a named letter.
    """
    assert (
        aitk_translator._last_answer_signal("Candidates: A login screen looks right", "ABCD")
        is None
    )


def test_last_answer_signal_treats_a_mid_sentence_article_looking_letter_as_a_letter() -> None:
    """A standalone "A"/"I" followed by a lowercase word only reads as the
    English article/pronoun when it is sentence-initial; mid-sentence it is a
    named letter even though a lowercase word follows.
    """
    assert aitk_translator._last_answer_signal("I think A is right", "ABCD") == "A"


def test_last_answer_signal_returns_none_without_any_signal() -> None:
    assert aitk_translator._last_answer_signal("no letters at all", "ABCD") is None


def test_identify_node_no_candidates_diagnostic_counts_stale_vectors_too(
    identifiable: aitk_translator.UIKobeV2Translator, caplog: pytest.LogCaptureFixture
) -> None:
    """score_by_cosine drops a cached vector of the wrong dimension as well as a
    missing one, so the "no candidates" line must not claim every same-package
    node has an embedding when all of them were dropped as stale.
    """
    graph = identifiable._graphs["demo"]
    same_pkg = [n for n, data in graph.nodes(data=True) if data["package"] == "com.demo.app"]
    for node in same_pkg:
        graph.nodes[node]["image_embedding"] = [1.0, 0.0, 0.0]

    with caplog.at_level("WARNING"):
        assert (
            identifiable._identify_node("com.demo.app/.HomeActivity", "shot", "com.demo.app")
            is None
        )

    expected = (
        f"[IDENTIFY] no candidates: none of the {len(same_pkg)} same-package node(s) has a "
        "usable image embedding (missing, or cached at a stale dimension)"
    )
    assert [m for m in caplog.messages if "no candidates" in m] == [expected]


@pytest.mark.usefixtures("no_sleep")
def test_load_all_graphs_writes_back_a_sidecar_without_a_node_that_left_the_graph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A node removed from the graph since the sidecar was written must not
    survive a rewrite: the runtime translator shares load_image_embeddings'
    pruning with app-graph-embed, and computing this graph's one uncached
    node is what carries the prune to disk, through the compute loop's own
    write -- a load never writes on its own (see load_image_embeddings).
    """
    path = _write_graph(tmp_path, app="demo", nodes=[{"id": "n1"}])
    model = embedding_cache.resolve_image_embedding_settings(_VLM_CONFIG).model
    embedding_cache.save_image_embeddings(path, {"gone": [0.0, 1.0]}, model=model)
    screenshots = path.parent / "demo_screenshots"
    screenshots.mkdir()
    (screenshots / "n1.png").write_bytes(b"shot-n1")

    monkeypatch.setattr(embedding_cache, "get_image_embedding", lambda *_a, **_kw: [1.0, 0.0])

    built = aitk_translator.UIKobeV2Translator(graph_dir=str(tmp_path), vlm_config=_VLM_CONFIG)

    assert built._graphs["demo"].nodes["n1"]["image_embedding"] == [1.0, 0.0]
    assert json.loads(embedding_cache.image_embeddings_path(path).read_text(encoding="utf-8")) == {
        "model": model,
        "embeddings": {"n1": [1.0, 0.0]},
    }


@pytest.mark.usefixtures("opened_clients", "model_environment")
def test_v2_recomputes_a_cache_tagged_with_another_model_and_embeds_runtime_screenshots(
    tmp_path: Path,
    router_requests: list[httpx.Request],
    router_responses: list[httpx.Response],
    embedding_body: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "shared-router-key")
    graph_path = _write_graph(tmp_path)
    screenshots = graph_path.parent / "demo_screenshots"
    screenshots.mkdir()
    (screenshots / "n1.png").write_bytes(_SCREENSHOT)
    sidecar = embedding_cache.image_embeddings_path(graph_path)
    sidecar.write_text(
        json.dumps({"model": "gemini-embedding-2", "embeddings": {"n1": [1.0, 0.0]}}),
        encoding="utf-8",
    )
    router_responses.extend(httpx.Response(200, json=embedding_body) for _ in range(2))
    built = aitk_translator.UIKobeV2Translator(graph_dir=str(tmp_path))
    assert built._graphs["demo"].nodes["n1"]["image_embedding"] == [0.25, 0.75]
    assert json.loads(sidecar.read_text())["model"] == "google/gemini-embedding-2"
    assert built._compute_runtime_image_embedding_with_retry("live-shot", "demo", "current") == [
        0.25,
        0.75,
    ]
    assert len(router_requests) == 2
    for request in router_requests:
        assert request.headers["authorization"] == "Bearer shared-router-key"
        assert str(request.url) == "https://openrouter.ai/api/v1/embeddings"
    assert (
        json.loads(router_requests[1].content)["input"][0]["content"][0]["image_url"]["url"]
        == "data:image/png;base64,live-shot"
    )


@pytest.mark.usefixtures("opened_clients")
def test_v2_decisions_use_the_action_model_and_never_read_page_detail_or_instruction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    router_requests: list[httpx.Request],
    router_responses: list[httpx.Response],
    png_screenshot: str,
    chat_response: Callable[..., httpx.Response],
) -> None:
    monkeypatch.setenv("APP_GRAPH_EMBEDDING_MODEL", "google/gemini-embedding-2")
    monkeypatch.setenv("APP_GRAPH_ACTION_MODEL", "qwen/qwen3.8-flash")
    translator = aitk_translator.UIKobeV2Translator(
        graph_dir=str(tmp_path),
        vlm_config={
            "action": {"temperature": 0.3},
            "page_detail": {"model": "${APP_GRAPH_PAGE_DETAIL_MODEL}", "temperature": "hot"},
            "instruction": {"model": "${APP_GRAPH_INSTRUCTION_MODEL}", "temperature": "hot"},
        },
    )
    router_responses.extend(
        [chat_response(content=_NO_INFORMATION), chat_response(content="Tap Search")]
    )
    translator._record_info("find shoes", png_screenshot)
    assert (
        translator._plan_free_action("find shoes", png_screenshot, "missing graph path", [])
        == "Tap Search"
    )
    bodies = [as_str_dict(json.loads(request.content)) for request in router_requests]
    assert [(body["model"], body["temperature"]) for body in bodies] == [
        ("qwen/qwen3.8-flash", 0.3)
    ] * 2
    assert "max_tokens" not in bodies[1]
    assert not hasattr(translator, "page_detail")
    assert not hasattr(translator, "instruction")


def test_runtime_reports_missing_models_before_creating_clients(
    tmp_path: Path, opened_clients: list[OpenRouter]
) -> None:
    with pytest.raises(ValueError, match="Invalid vlm configuration") as caught:
        aitk_translator.UIKobeV2Translator(graph_dir=str(tmp_path))
    for role in ("EMBEDDING", "ACTION"):
        assert f"APP_GRAPH_{role}_MODEL" in str(caught.value)
    assert "PAGE_DETAIL" not in str(caught.value)
    assert "INSTRUCTION" not in str(caught.value)
    assert opened_clients == []


def test_runtime_keeps_caller_environment_references(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    config: dict[str, object] = {}
    for role in ("embedding", "action"):
        variable = f"APP_GRAPH_{role.upper()}_MODEL"
        monkeypatch.setenv(variable, f"provider/{role}")
        config[role] = {"model": f"${{{variable}}}"}
    first = aitk_translator.UIKobeV2Translator(graph_dir=str(tmp_path), vlm_config=config)
    monkeypatch.setenv("APP_GRAPH_ACTION_MODEL", "provider/new-action")
    second = aitk_translator.UIKobeV2Translator(graph_dir=str(tmp_path), vlm_config=config)
    assert first.action.model == "provider/action"
    assert second.action.model == "provider/new-action"
    assert config["action"] == {"model": "${APP_GRAPH_ACTION_MODEL}"}
    assert "image_embedding" not in config


@pytest.mark.usefixtures("model_environment")
def test_v2_runtime_without_an_openrouter_key_creates_no_client(
    tmp_path: Path, opened_clients: list[OpenRouter], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY")
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY is required"):
        aitk_translator.UIKobeV2Translator(graph_dir=str(tmp_path))
    assert opened_clients == []


@pytest.mark.usefixtures("model_environment")
def test_v2_runtime_rejects_upstream_endpoint_and_credential_keys(
    tmp_path: Path, opened_clients: list[OpenRouter], monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "dummy-google-secret-never-log"  # noqa: S105  # fake credential; the test proves it is never logged
    monkeypatch.setenv("GEMINI_API_KEY", secret)
    upstream = {
        "action": {"api_key": "${GEMINI_API_KEY}", "base_url": "https://provider.example/v1"},
        "image_embedding": {
            "native_base_url": "https://generativelanguage.googleapis.com/v1beta",
            "model": "gemini-embedding-2",
        },
        "instruction": {"api_key": "${GEMINI_API_KEY}"},
    }
    with pytest.raises(ValueError, match="was removed") as caught:
        aitk_translator.UIKobeV2Translator(graph_dir=str(tmp_path), vlm_config=upstream)
    for name in (
        "action.api_key",
        "action.base_url",
        "image_embedding.native_base_url",
        "instruction.api_key",
    ):
        assert f"{name} was removed" in str(caught.value)
    assert secret not in str(caught.value)
    assert opened_clients == []


@pytest.mark.usefixtures("model_environment", "opened_clients")
def test_v2_runtime_image_requests_use_the_image_timeout(
    tmp_path: Path,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    embedding_body: dict[str, object],
    png_screenshot: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    router_responses.append(httpx.Response(200, json=embedding_body))
    runtime = aitk_translator.UIKobeV2Translator(
        graph_dir=str(tmp_path), vlm_config={"image_embedding": {"request_timeout": 2.25}}
    )
    assert runtime._compute_runtime_image_embedding_with_retry(png_screenshot, "demo", "home") == [
        0.25,
        0.75,
    ]
    assert router_requests[0].extensions["timeout"] == dict.fromkeys(
        ("connect", "read", "write", "pool"), 2.25
    )


@pytest.mark.parametrize("schema_failure", [False, True])
@pytest.mark.usefixtures("model_environment", "opened_clients")
def test_v2_turn_distinguishes_bad_actions_from_broken_endpoint_and_can_resume(
    schema_failure: bool,
    tmp_path: Path,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    chat_body: dict[str, object],
    png_screenshot: str,
    no_sleep: list[float],
    monkeypatch: pytest.MonkeyPatch,
    chat_response: Callable[..., httpx.Response],
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setattr(aitk_translator.device, "soft_keyboard_hint", lambda: "")
    runtime = aitk_translator.UIKobeV2Translator(graph_dir=str(tmp_path))
    runtime._app_opened = True
    state: dict[str, object] = {"screenshot": png_screenshot, "activity": "com.demo.Home"}
    history: dict[str, object] = {"actions": ["started"]}
    for turn in range(2):
        router_responses.extend(
            [
                chat_response(content=_NO_INFORMATION),
                chat_response(content="Tap Search"),
            ]
        )
        if turn == 0:
            router_responses.extend(
                httpx.Response(
                    200, json={key: val for key, val in chat_body.items() if key != "id"}
                )
                if schema_failure
                else chat_response(content='{"action":"system_button","button":"Power"}')
                for _ in range(2)
            )
        else:
            router_responses.append(
                chat_response(content='{"action":"click","coordinate":[500,500]}')
            )
        if turn == 0 and schema_failure:
            with pytest.raises(ResponseValidationError):
                runtime.to_agent("Find search", state, history)
            assert runtime._memory.actions == []
        else:
            result = json.loads(runtime.to_agent("Find search", state, history))
            assert result["aitk_action"] == (
                {"action": "wait", "time": 1}
                if turn == 0
                else {"action": "tap", "x": 540, "y": 960}
            )
    assert len(router_requests) == 7
    assert no_sleep == []
