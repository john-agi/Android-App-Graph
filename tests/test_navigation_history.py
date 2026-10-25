"""Observed routes provide context without declaring useful workflows exhausted."""

from collections.abc import Sequence
from itertools import pairwise

import pytest

from android_app_graph.navigation_history import NavigationHistory, RouteState


def _record(history: NavigationHistory, states: Sequence[RouteState], action: str = "back") -> None:
    for before, after in pairwise(states):
        history.record(before, after, action, known_edge=True)


@pytest.fixture
def history() -> NavigationHistory:
    return NavigationHistory(lambda node: node)


@pytest.mark.parametrize("length", [2, 3, 4])
def test_two_closed_routes_provide_bounded_context(history: NavigationHistory, length: int) -> None:
    states = [RouteState(f"s{index}", f"pixels{index}") for index in range(length)]
    _record(history, [*states, *states, states[0]])

    feedback = history.feedback(states[0])

    assert f"last {2 * length} ordinary steps" in feedback
    assert "Hidden state may still change" in feedback
    assert "Continue a useful pending workflow" in feedback
    assert "Preserve pending edits" in feedback
    assert feedback.count("--back-->") == length


@pytest.mark.parametrize(
    "states",
    [
        [RouteState("form", "a"), RouteState("dialog", "b"), RouteState("form", "a")],
        [RouteState("form", "a")] * 9,
        [RouteState("form", "a"), RouteState("form", "b")] * 4 + [RouteState("form", "a")],
        [RouteState("form", "a"), RouteState("dialog", "b"), RouteState("picker", "c")],
        [
            RouteState("form", "a"),
            RouteState("dialog", "b"),
            RouteState("form", "changed"),
            RouteState("dialog", "b"),
            RouteState("form", "changed"),
        ],
        [
            RouteState("form", "a"),
            RouteState("dialog", "b"),
            RouteState("form", "a"),
            RouteState("picker", "c"),
            RouteState("form", "a"),
        ],
        [RouteState(str(index), str(index)) for index in range(5)] * 2 + [RouteState("0", "0")],
    ],
    ids=[
        "one cycle",
        "self loops",
        "single node changing pixels",
        "open path",
        "edited form",
        "interleaved picker",
        "long route",
    ],
)
def test_other_visits_do_not_establish_a_short_repeated_route(
    history: NavigationHistory, states: list[RouteState]
) -> None:
    _record(history, states)

    assert history.feedback(states[-1]) == ""


def test_action_variants_do_not_hide_matching_observed_routes(history: NavigationHistory) -> None:
    form, dialog = RouteState("form", "a"), RouteState("dialog", "b")
    _record(history, [form, dialog, form])
    _record(history, [form, dialog, form], "tap")

    assert "form --tap--> dialog --tap--> form" in history.feedback(form)
    assert "latest cycle actions" in history.feedback(form)


def test_new_edges_start_fresh_evidence(history: NavigationHistory) -> None:
    form, dialog = RouteState("form", "a"), RouteState("dialog", "b")
    _record(history, [form, dialog, form, dialog])
    history.record(dialog, form, "back", known_edge=False)

    assert history.feedback(form) == ""
    _record(history, [form, dialog, form])
    assert history.feedback(form) == ""
    _record(history, [form, dialog, form])
    assert history.feedback(form)


@pytest.mark.parametrize("missing", ["before", "after"])
def test_unavailable_pixels_break_continuity(history: NavigationHistory, missing: str) -> None:
    form, dialog = RouteState("form", "a"), RouteState("dialog", "b")
    _record(history, [form, dialog, form])
    history.record(
        RouteState("form", "" if missing == "before" else "a"),
        RouteState("dialog", "" if missing == "after" else "b"),
        "back",
        known_edge=True,
    )
    _record(history, [dialog, form])

    assert history.feedback(form) == ""


def test_a_different_current_observation_discards_old_context(history: NavigationHistory) -> None:
    form, dialog = RouteState("form", "a"), RouteState("dialog", "b")
    _record(history, [form, dialog, form, dialog, form])

    assert history.feedback(RouteState("form", "updated")) == ""
    assert history.feedback(form) == ""


def test_discontinuous_observations_start_fresh(history: NavigationHistory) -> None:
    form, dialog, other = (
        RouteState("form", "a"),
        RouteState("dialog", "b"),
        RouteState("other", "c"),
    )
    _record(history, [form, dialog, form])
    _record(history, [other, dialog, form])

    assert history.feedback(form) == ""


def test_aliases_invalidate_history_instead_of_rewriting_old_routes() -> None:
    aliases: dict[str, str] = {}
    history = NavigationHistory(lambda node: aliases.get(node, node))
    form, dialog = RouteState("form", "a"), RouteState("dialog", "b")
    _record(history, [form, dialog, form, dialog, form])
    aliases["dialog"] = "form"

    assert history.feedback(form) == ""


def test_an_explicit_boundary_starts_fresh(history: NavigationHistory) -> None:
    form, dialog = RouteState("form", "a"), RouteState("dialog", "b")
    _record(history, [form, dialog, form])
    history.clear()
    _record(history, [form, dialog, form])

    assert history.feedback(form) == ""


def test_history_and_action_context_stay_bounded(history: NavigationHistory) -> None:
    form, dialog = RouteState("form", "a"), RouteState("dialog", "b")
    _record(history, [form, dialog] * 500 + [form], "unrecognized" * 1000)

    feedback = history.feedback(form)

    assert "form --other--> dialog --other--> form" in feedback
    assert len(feedback) < 500
