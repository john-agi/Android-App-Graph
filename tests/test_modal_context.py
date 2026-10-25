"""Modal opening provenance stays bounded and survives state changes and nested returns."""

import json

import pytest

from android_app_graph.modal_context import (
    MAX_MODAL_DEPTH,
    MAX_OPENING_ACTIONS,
    MatchInput,
    ModalContext,
    Surface,
)

TAP: tuple[dict[str, object], ...] = ({"action": "tap", "x": 100, "y": 200},)


def test_modal_state_and_nested_return_keep_the_original_opening() -> None:
    outer = ModalContext().advance("a", MatchInput("page", TAP), Surface("modal", "open"))
    edited = outer.advance("b", MatchInput("a", TAP), Surface("modal", "continue"))
    nested = edited.advance("c", MatchInput("b", TAP), Surface("modal", "open"))
    returned = nested.advance(
        "d", MatchInput("c", ({"action": "back"},)), Surface("modal", "return", 1)
    )
    assert edited == outer == returned
    assert len(nested.frames) == 2
    assert nested.origin_known
    partial = ModalContext().advance("unknown", MatchInput(), Surface("modal", "unknown"))
    partial = partial.advance("inner", MatchInput("unknown", TAP), Surface("modal", "open"))
    assert not partial.origin_known
    assert nested.frames[1].source.previous == "b"
    assert returned.frames[0].source.previous == "page"
    assert returned.advance("e", MatchInput("d", TAP), Surface()) == ModalContext()
    assert ModalContext.from_record(json.loads(json.dumps(nested.as_record()))) == nested


@pytest.mark.parametrize(
    "surface",
    [
        Surface("modal", "open"),
        Surface("modal", "continue"),
        Surface("modal", "return", 1),
        Surface("modal", "unknown"),
    ],
)
def test_first_captured_modal_has_an_explicit_unknown_origin(surface: Surface) -> None:
    context = ModalContext().advance("first", MatchInput(), surface)
    assert len(context.frames) == 1
    assert context.frames[0].source == MatchInput()


def test_nesting_overflow_replaces_an_incomplete_chain_with_unknown() -> None:
    context = ModalContext()
    for index in range(MAX_MODAL_DEPTH + 1):
        context = context.advance(str(index), MatchInput("previous", TAP), Surface("modal", "open"))
    assert len(context.frames) == 1
    assert context.frames[0].source == MatchInput()


@pytest.mark.parametrize("source", [MatchInput("previous", ()), MatchInput("previous", None)])
def test_an_opening_without_observed_input_remains_unknown(source: MatchInput) -> None:
    context = ModalContext().advance("current", source, Surface("modal", "open"))
    assert context.frames[0].source == MatchInput()


@pytest.mark.parametrize(
    "value",
    [
        None,
        {},
        {"previous": 1, "actions": []},
        {"previous": None, "actions": "tap"},
        {"previous": None, "actions": [None]},
        {"previous": None, "actions": [{}] * (MAX_OPENING_ACTIONS + 1)},
    ],
)
def test_match_input_requires_an_explicit_bounded_record(value: object) -> None:
    with pytest.raises(ValueError, match=r"Expected|requires|Only"):
        MatchInput.from_record(value)


@pytest.mark.parametrize(
    "value",
    [
        None,
        {},
        {"kind": "other", "relation": "unknown", "return_depth": 0},
        {"kind": "modal", "relation": "other", "return_depth": 0},
        {"kind": "modal", "relation": "return", "return_depth": 0},
        {"kind": "modal", "relation": "open", "return_depth": 1},
        {"kind": "modal", "relation": "return", "return_depth": True},
        {"kind": "modal", "relation": "return", "return_depth": MAX_MODAL_DEPTH + 1},
    ],
)
def test_surface_requires_a_valid_modal_relationship(value: object) -> None:
    with pytest.raises(ValueError, match=r"Expected|requires|Only"):
        Surface.from_record(value)


@pytest.mark.parametrize("value", [None, {}, [{}], [{"opened": "x"}], [{}] * (MAX_MODAL_DEPTH + 1)])
def test_saved_modal_context_requires_complete_opening_records(value: object) -> None:
    with pytest.raises(ValueError, match=r"Expected|requires|Only"):
        ModalContext.from_record(value)
