"""Tests for android_app_graph.member_groups: unit vectors, closest pairs, single-linkage
groups and the checks planned on them, all on synthetic vectors."""

from __future__ import annotations

import itertools
import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from android_app_graph.member_groups import (
    CHECKS_PER_NODE,
    MEMBER_SIMILARITY_FLOOR,
    MemberPair,
    closest_pair,
    cosine,
    linked_groups,
    plan_member_checks,
    unit_vector,
)


def _at(degrees: float) -> list[float]:
    """The unit vector at ``degrees`` in the plane: two of them have the cosine of the
    angle between them."""
    return [math.cos(math.radians(degrees)), math.sin(math.radians(degrees))]


# Members this many degrees apart are linked; twice as far apart, they are not.
LINKED = 0.6 * math.degrees(math.acos(MEMBER_SIMILARITY_FLOOR))


def _units(**angles: float) -> dict[str, list[float]]:
    return {member: _at(degrees) for member, degrees in angles.items()}


def test_a_unit_vector_has_length_one_and_a_zero_vector_none() -> None:
    unit = unit_vector([3.0, 4.0])
    assert unit == pytest.approx([0.6, 0.8])
    assert unit_vector([0.0, 0.0]) is None


def test_the_cosine_of_unit_vectors_is_their_dot_product() -> None:
    assert cosine(_at(0), _at(60)) == pytest.approx(0.5)


def test_vectors_of_different_lengths_are_not_alike() -> None:
    assert cosine([1.0, 0.0], [1.0, 0.0, 0.0]) == 0.0


def test_the_closest_pair_takes_one_member_of_each_set() -> None:
    units = _units(a=0, b=90, c=80, d=180)
    pair = closest_pair(["a", "b"], ["c", "d"], units)
    assert pair is not None
    assert (pair.first, pair.second) == ("b", "c")
    assert pair.score == pytest.approx(cosine(_at(90), _at(80)))


def test_the_earliest_pair_wins_a_tie() -> None:
    units = _units(a=0, b=0, c=10)
    assert closest_pair(["a", "b"], ["c"], units) == MemberPair("a", "c", cosine(_at(0), _at(10)))


def test_a_set_without_vectors_has_no_closest_pair() -> None:
    assert closest_pair(["a"], ["b"], _units(a=0)) is None


def test_single_linkage_joins_a_chain_whose_ends_are_apart() -> None:
    """``b`` links ``a`` and ``c`` although they are 40 degrees apart: one group."""
    units = _units(a=0, b=LINKED, c=2 * LINKED, d=90)
    assert linked_groups(["a", "b", "c", "d"], units, MEMBER_SIMILARITY_FLOOR) == [
        ["a", "b", "c"],
        ["d"],
    ]


def test_groups_keep_the_order_of_their_members() -> None:
    units = _units(a=0, b=90, c=0, d=90)
    assert linked_groups(["a", "b", "c", "d"], units, MEMBER_SIMILARITY_FLOOR) == [
        ["a", "c"],
        ["b", "d"],
    ]


@given(st.lists(st.floats(min_value=0, max_value=180), min_size=1, max_size=12))
def test_groups_are_the_connected_components_of_the_links(angles: list[float]) -> None:
    """The groups partition the members, every pair across two groups falls below the
    floor, and every group is connected by links at the floor."""
    members = [f"m{index}" for index in range(len(angles))]
    units = {member: _at(degrees) for member, degrees in zip(members, angles, strict=True)}

    def linked(first: str, second: str) -> bool:
        return cosine(units[first], units[second]) >= MEMBER_SIMILARITY_FLOOR

    groups = linked_groups(members, units, MEMBER_SIMILARITY_FLOOR)

    assert sorted(itertools.chain.from_iterable(groups)) == sorted(members)
    for first, second in itertools.combinations(groups, 2):
        assert not any(linked(a, b) for a in first for b in second)
    for group in groups:
        reached = {group[0]}
        frontier = [group[0]]
        while frontier:
            member = frontier.pop()
            for other in group:
                if other not in reached and linked(member, other):
                    reached.add(other)
                    frontier.append(other)
        assert reached == set(group)


def test_a_single_member_node_gets_no_check() -> None:
    plan = plan_member_checks(["r"], "r", _units(r=0))
    assert plan is not None
    assert (plan.groups, plan.diffuse, plan.checks, plan.unchecked) == ([["r"]], False, [], [])


def test_a_node_whose_representative_has_no_vector_has_no_plan() -> None:
    assert plan_member_checks(["r", "a"], "r", _units(a=0)) is None


def test_members_without_a_vector_are_left_out() -> None:
    plan = plan_member_checks(["r", "a", "b"], "r", _units(r=0, b=90))
    assert plan is not None
    assert plan.groups == [["r"], ["b"]]


def test_a_lone_member_is_checked_against_its_closest_home_member() -> None:
    units = _units(r=0, h=LINKED, far=LINKED + 50)
    plan = plan_member_checks(["r", "h", "far"], "r", units)
    assert plan is not None
    (check,) = plan.checks
    assert check.group == ["far"]
    assert (check.pair.first, check.pair.second) == ("h", "far")
    assert check.pair.score == pytest.approx(cosine(_at(0), _at(50)))


def test_a_three_member_node_is_never_split() -> None:
    """Two members apart from the representative leave home under half: diffuse."""
    plan = plan_member_checks(["r", "a", "b"], "r", _units(r=0, a=90, b=90))
    assert plan is not None
    assert plan.diffuse
    assert (plan.checks, plan.unchecked) == ([], [])


def test_a_group_of_two_beside_a_home_of_two_is_a_split_check() -> None:
    units = _units(r=0, h=LINKED, a=90, b=90 + LINKED)
    plan = plan_member_checks(["r", "a", "h", "b"], "r", units)
    assert plan is not None
    assert not plan.diffuse
    (check,) = plan.checks
    assert check.group == ["a", "b"]
    assert (check.pair.first, check.pair.second) == ("h", "a")


def test_a_node_whose_home_holds_under_half_its_members_is_diffuse() -> None:
    units = _units(r=0, a=60, b=120, c=180, h=LINKED)
    plan = plan_member_checks(["r", "a", "b", "c", "h"], "r", units)
    assert plan is not None
    assert plan.diffuse
    assert plan.groups == [["r", "h"], ["a"], ["b"], ["c"]]
    assert (plan.checks, plan.unchecked) == ([], [])


def test_no_node_gets_more_than_three_checks_and_larger_groups_come_first() -> None:
    home = {f"h{index}": index * LINKED for index in range(6)}
    lone = {"x": 6 * LINKED + 30, "y": 6 * LINKED + 60, "z": 6 * LINKED + 90}
    pair = {"p": 300.0, "q": 300 + LINKED}
    units = _units(**home, **lone, **pair)
    plan = plan_member_checks([*home, *lone, *pair], "h0", units)
    assert plan is not None
    assert [check.group for check in plan.checks] == [["p", "q"], ["x"], ["y"]]
    assert len(plan.checks) == CHECKS_PER_NODE
    assert plan.unchecked == [["z"]]


def test_a_lower_floor_links_more_members() -> None:
    units = _units(r=0, a=2 * LINKED)
    default = plan_member_checks(["r", "a"], "r", units)
    lower = plan_member_checks(["r", "a"], "r", units, floor=cosine(_at(0), _at(2 * LINKED)))
    assert default is not None
    assert lower is not None
    assert (default.groups, lower.groups) == ([["r"], ["a"]], [["r", "a"]])
