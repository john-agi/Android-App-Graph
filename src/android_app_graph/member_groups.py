"""How alike the members of graph nodes are, by their images and by their descriptions:
unit vectors, the closest pair of two member sets, and the groups a node's members form.

Four floors on the cosine of two vectors decide which checks the verifier gets, never
their outcome. Two nodes are proposed for a merge when their closest members reach
``IMAGE_MERGE_FLOOR`` by image or ``TEXT_MERGE_FLOOR`` by description. Two members of
one node are linked when their images reach ``IMAGE_GROUPING_FLOOR`` or their
descriptions reach ``TEXT_GROUPING_FLOOR``, so a member is apart only when both signals
separate it, and the node's groups are the connected components of those links:
single-linkage clustering. The merge and the grouping floors pull in opposite
directions, a lower merge floor proposing more pairs and a higher grouping floor
separating more members, so each is set on its own.

Two groups exist exactly when every pair across them falls below both grouping floors,
so the pair a check shows the verifier, the one whose images are closest across the
gap, is the strongest evidence that the groups are one screen. Each vector is scaled to
length 1 once, so a cosine is one ``math.sumprod``.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import NamedTuple

IMAGE_MERGE_FLOOR = 0.65
TEXT_MERGE_FLOOR = 0.55
IMAGE_GROUPING_FLOOR = 0.95
TEXT_GROUPING_FLOOR = 0.91


class MemberPair(NamedTuple):
    """One member of each of two sets, and the cosine of their vectors."""

    first: str
    second: str
    score: float


class MemberCheck(NamedTuple):
    """A group of a node's members checked against home, and the pair whose images are
    closest across the gap: ``pair.first`` a home member, ``pair.second`` a member of
    the group."""

    group: list[str]
    pair: MemberPair


class Signal(NamedTuple):
    """Unit vectors of one kind by member, and the cosine at which two members link."""

    units: Mapping[str, Sequence[float]]
    floor: float


def unit_vector(vector: Sequence[float]) -> list[float]:
    """Return ``vector`` scaled to length 1; a zero vector, which has no direction, stays
    zero, so its cosine with any vector is 0.0."""
    norm = math.hypot(*vector)
    if not norm:
        return list(vector)
    return [value / norm for value in vector]


def cosine(first: Sequence[float], second: Sequence[float]) -> float:
    """Return the cosine of two unit vectors; vectors of different lengths come from
    different models and score 0.0."""
    if len(first) != len(second):
        return 0.0
    return math.sumprod(first, second)


def _pairs(
    first: Iterable[str], second: Iterable[str], units: Mapping[str, Sequence[float]]
) -> Iterator[MemberPair]:
    second_units = [(member, units[member]) for member in second]
    for member in first:
        unit = units[member]
        for other, other_unit in second_units:
            yield MemberPair(member, other, cosine(unit, other_unit))


def closest_pair(
    first: Iterable[str], second: Iterable[str], units: Mapping[str, Sequence[float]]
) -> MemberPair | None:
    """Return the member of ``first`` and the member of ``second`` whose unit vectors
    in ``units`` have the highest cosine, the earliest such pair on a tie, or ``None``
    when either set is empty. Every member has a vector in ``units``."""
    return max(_pairs(first, second, units), key=_score, default=None)


def _score(pair: MemberPair) -> float:
    return pair.score


def linked_groups(members: Sequence[str], signals: Sequence[Signal]) -> list[list[str]]:
    """Return the connected components of ``members`` when two are linked by any of
    ``signals``: both have a vector in its ``units``, and their cosine reaches its
    ``floor``.

    Each group keeps the order of ``members``, and the groups are in the order of
    their first member.
    """
    parent = list(range(len(members)))
    for first, second in itertools.combinations(range(len(members)), 2):
        if any(_links(signal, members[first], members[second]) for signal in signals):
            parent[_root(parent, second)] = _root(parent, first)
    groups: dict[int, list[str]] = {}
    for index, member in enumerate(members):
        groups.setdefault(_root(parent, index), []).append(member)
    return list(groups.values())


def _links(signal: Signal, first: str, second: str) -> bool:
    units = signal.units
    if first not in units or second not in units:
        return False
    return cosine(units[first], units[second]) >= signal.floor


def _root(parent: list[int], index: int) -> int:
    """Return the root of ``index`` in the union-find forest ``parent``, pointing every
    member on the way straight at it."""
    up = parent[index]
    if up == index:
        return index
    root = _root(parent, up)
    parent[index] = root
    return root


def plan_member_checks(
    members: Sequence[str],
    images: Mapping[str, Sequence[float]],
    descriptions: Mapping[str, Sequence[float]],
) -> list[MemberCheck]:
    """Return the checks to make on the groups of the ``members`` of a node, in step
    order.

    Two members are linked when their unit image vectors in ``images`` reach
    ``IMAGE_GROUPING_FLOOR`` or their unit description vectors in ``descriptions`` reach
    ``TEXT_GROUPING_FLOOR``. Every member has an image vector; a black member, never
    described, has no description vector and is linked by its image alone.

    Home is the largest group, the one holding the earliest member on a tie, and every
    other group is checked against home, larger groups first, on the pair whose images
    are closest across them. A group of two or more members is a split check and a lone
    member a move check; a split group has at least two members and home at least as
    many, so no node of fewer than four members is split.
    """
    signals = (Signal(images, IMAGE_GROUPING_FLOOR), Signal(descriptions, TEXT_GROUPING_FLOOR))
    grouped = linked_groups(members, signals)
    home = max(grouped, key=len)
    others = sorted((group for group in grouped if group is not home), key=len, reverse=True)
    return [MemberCheck(group, max(_pairs(home, group, images), key=_score)) for group in others]
