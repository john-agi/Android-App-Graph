"""Tests for android_app_graph.member_groups: unit vectors, closest pairs, single-linkage
groups on images and descriptions and the checks planned on them, all on synthetic
vectors."""

from __future__ import annotations

import itertools
import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from android_app_graph.member_groups import (
    IMAGE_GROUPING_FLOOR,
    TEXT_GROUPING_FLOOR,
    MemberPair,
    Signal,
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


# Members this many degrees apart are linked by image; twice as far apart, they are not.
LINKED = 0.6 * math.degrees(math.acos(IMAGE_GROUPING_FLOOR))
# Members this many degrees apart are linked by neither images nor descriptions.
APART = 2 * math.degrees(math.acos(TEXT_GROUPING_FLOOR))


def _units(**angles: float) -> dict[str, list[float]]:
    return {member: _at(degrees) for member, degrees in angles.items()}


def _axis(index: int) -> list[float]:
    """A unit vector along one of eight axes: two different axes have a cosine of 0."""
    return [float(index == position) for position in range(8)]


def test_a_unit_vector_has_length_one_and_a_zero_vector_stays_zero() -> None:
    unit = unit_vector([3.0, 4.0])
    assert unit == pytest.approx([0.6, 0.8])
    assert unit_vector([0.0, 0.0]) == [0.0, 0.0]
    assert cosine(unit_vector([0.0, 0.0]), unit) == 0.0


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


def test_an_empty_set_has_no_closest_pair() -> None:
    assert closest_pair([], ["b"], _units(b=0)) is None


def _images(units: dict[str, list[float]]) -> list[Signal]:
    return [Signal(units, IMAGE_GROUPING_FLOOR)]


def test_single_linkage_joins_a_chain_whose_ends_are_apart() -> None:
    """``b`` links ``a`` and ``c`` although they are too far apart to link: one group."""
    units = _units(a=0, b=LINKED, c=2 * LINKED, d=180)
    assert linked_groups(["a", "b", "c", "d"], _images(units)) == [["a", "b", "c"], ["d"]]


def test_groups_keep_the_order_of_their_members() -> None:
    units = _units(a=0, b=90, c=0, d=90)
    assert linked_groups(["a", "b", "c", "d"], _images(units)) == [["a", "c"], ["b", "d"]]


def test_a_lower_floor_links_more_members() -> None:
    units = _units(first=0, a=2 * LINKED)
    lower = cosine(_at(0), _at(2 * LINKED))
    assert linked_groups(["first", "a"], _images(units)) == [["first"], ["a"]]
    assert linked_groups(["first", "a"], [Signal(units, lower)]) == [["first", "a"]]


def test_a_member_without_a_vector_of_a_signal_is_linked_by_the_other_alone() -> None:
    """``c`` has no description, so only its image, alike to ``a``'s, links it."""
    images = _units(a=0, b=90, c=0)
    descriptions = _units(a=0, b=0)
    signals = [*_images(images), Signal(descriptions, TEXT_GROUPING_FLOOR)]
    assert linked_groups(["a", "b", "c"], signals) == [["a", "b", "c"]]
    assert linked_groups(["b", "c"], signals) == [["b"], ["c"]]


_ANGLES = st.floats(min_value=0, max_value=180)


@given(st.lists(st.tuples(_ANGLES, st.none() | _ANGLES), min_size=1, max_size=12))
def test_groups_are_the_connected_components_of_the_links(
    angles: list[tuple[float, float | None]],
) -> None:
    """Two members are linked when their images or their descriptions are alike, and a
    member without a description is linked by its image alone. The groups partition the
    members, no pair across two groups is linked, and every group is connected by
    links."""
    members = [f"m{index}" for index in range(len(angles))]
    images = {member: _at(image) for member, (image, _) in zip(members, angles, strict=True)}
    descriptions = {
        member: _at(text)
        for member, (_, text) in zip(members, angles, strict=True)
        if text is not None
    }

    def linked(first: str, second: str) -> bool:
        if cosine(images[first], images[second]) >= IMAGE_GROUPING_FLOOR:
            return True
        if first not in descriptions or second not in descriptions:
            return False
        return cosine(descriptions[first], descriptions[second]) >= TEXT_GROUPING_FLOOR

    signals = [*_images(images), Signal(descriptions, TEXT_GROUPING_FLOOR)]
    groups = linked_groups(members, signals)

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
    assert plan_member_checks(["first"], _units(first=0), _units(first=0)) == []


def test_home_is_the_largest_group_so_an_earliest_member_apart_gets_a_move_check() -> None:
    """The earliest member alone below the floor, the four later ones linked: home is the
    four, and the earliest member is checked against its closest home member."""
    units = _units(first=90, a=0, b=LINKED, c=0, d=LINKED)
    (check,) = plan_member_checks(["first", "a", "b", "c", "d"], units, {})
    assert check.group == ["first"]
    assert (check.pair.first, check.pair.second) == ("b", "first")


def test_a_tie_for_the_largest_group_goes_to_the_group_of_the_earliest_member() -> None:
    units = _units(first=0, a=90, b=90 + LINKED, c=LINKED)
    (check,) = plan_member_checks(["first", "a", "b", "c"], units, {})
    assert check.group == ["a", "b"]
    assert (check.pair.first, check.pair.second) == ("c", "a")


def test_a_lone_member_is_checked_against_its_closest_home_member() -> None:
    units = _units(first=0, h=LINKED, far=LINKED + 50)
    (check,) = plan_member_checks(["first", "h", "far"], units, {})
    assert check.group == ["far"]
    assert (check.pair.first, check.pair.second) == ("h", "far")
    assert check.pair.score == pytest.approx(cosine(_at(0), _at(50)))


def test_a_check_shows_the_pair_whose_images_are_closest_whatever_the_descriptions() -> None:
    """``first``'s description is closer to ``far``'s than ``h``'s is, but ``h``'s image
    is the closer one."""
    images = _units(first=0, h=LINKED, far=LINKED + 50)
    descriptions = _units(first=0, h=90, far=40)
    (check,) = plan_member_checks(["first", "h", "far"], images, descriptions)
    assert (check.pair.first, check.pair.second) == ("h", "far")


def test_three_members_apart_from_one_another_are_two_move_checks() -> None:
    """Every group holds one member, so home is the earliest's, and no group is split."""
    units = _units(first=0, a=90, b=180)
    checks = plan_member_checks(["first", "a", "b"], units, units)
    assert [(check.group, check.pair.first) for check in checks] == [
        (["a"], "first"),
        (["b"], "first"),
    ]


def test_a_group_of_two_beside_a_home_of_two_is_a_split_check() -> None:
    units = _units(first=0, h=LINKED, a=90, b=90 + LINKED)
    (check,) = plan_member_checks(["first", "a", "h", "b"], units, {})
    assert check.group == ["a", "b"]
    assert (check.pair.first, check.pair.second) == ("h", "a")


def test_every_group_is_checked_however_few_members_home_holds() -> None:
    units = _units(first=0, a=90, b=180, c=270, h=LINKED)
    checks = plan_member_checks(["first", "a", "b", "c", "h"], units, {})
    assert [(check.group, check.pair.first) for check in checks] == [
        (["a"], "h"),
        (["b"], "h"),
        (["c"], "first"),
    ]


def test_every_group_is_checked_larger_groups_first() -> None:
    home = {f"h{index}": _axis(0) for index in range(6)}
    lone = {"x": _axis(1), "y": _axis(2), "z": _axis(3)}
    pair = {"p": _axis(4), "q": _axis(4)}
    units = {**home, **lone, **pair}
    checks = plan_member_checks([*home, *lone, *pair], units, {})
    assert [check.group for check in checks] == [["p", "q"], ["x"], ["y"], ["z"]]


@pytest.mark.parametrize(
    ("images", "descriptions", "groups"),
    [
        ({"first": 0, "a": APART}, {"first": 0, "a": APART}, [["a"]]),
        ({"first": 0, "a": APART}, {"first": 0, "a": 0}, []),
        ({"first": 0, "a": 0}, {"first": 0, "a": APART}, []),
        ({"first": 0, "a": APART}, {"first": 0}, [["a"]]),
        ({"first": 0, "a": 0}, {"first": 0}, []),
    ],
    ids=[
        "both-apart",
        "descriptions-alike",
        "images-alike",
        "no-description-images-apart",
        "no-description-images-alike",
    ],
)
def test_a_member_is_apart_only_when_both_its_image_and_its_description_are(
    images: dict[str, float], descriptions: dict[str, float], groups: list[list[str]]
) -> None:
    """A member without a description, a black one, is apart when its image is."""
    checks = plan_member_checks(["first", "a"], _units(**images), _units(**descriptions))
    assert [check.group for check in checks] == groups
