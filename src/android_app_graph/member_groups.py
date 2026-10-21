"""How alike the member images of graph nodes are: unit vectors, the closest pair of
two member sets, and the groups a node's members form.

Two floors on the cosine of two image vectors decide which checks the verifier gets,
never their outcome. Two nodes whose closest members reach ``MERGE_FLOOR`` are
proposed for a merge. Two members of one node are linked when they reach
``GROUPING_FLOOR``, and the node's groups are the connected components of those
links: single-linkage clustering cut at distance ``1 - GROUPING_FLOOR``. The floors
pull in opposite directions, a lower merge floor proposing more pairs and a higher
grouping floor separating more members, so each is set on its own.

Two groups exist exactly when every pair across them falls below the grouping floor,
so the pair a check shows the verifier, the closest one across the gap, is the
strongest evidence that the groups are one screen. Each vector is scaled to length 1
once, so a cosine is one ``math.sumprod``.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import NamedTuple

MERGE_FLOOR = 0.79
GROUPING_FLOOR = 0.79
HOME_SHARE = 0.5
CHECKS_PER_NODE = 3


class MemberPair(NamedTuple):
    """One member of each of two sets, and the cosine of their image vectors."""

    first: str
    second: str
    score: float


class MemberCheck(NamedTuple):
    """A group of a node's members checked against home, and the closest pair across
    the gap: ``pair.first`` a home member, ``pair.second`` a member of the group."""

    group: list[str]
    pair: MemberPair


class MemberPlan(NamedTuple):
    """The groups of a node's members and the checks the audit makes on them.

    ``groups`` starts with home, the group that holds the representative, and goes
    on with the others, larger first. ``checks`` are the groups checked against
    home, ``unchecked`` the groups past the per-node cap; a diffuse node has neither.
    """

    groups: list[list[str]]
    diffuse: bool
    checks: list[MemberCheck]
    unchecked: list[list[str]]


def unit_vector(vector: Sequence[float]) -> list[float] | None:
    """Return ``vector`` scaled to length 1, or ``None`` for a zero vector, which has
    no direction."""
    norm = math.hypot(*vector)
    if not norm:
        return None
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
    second_units = [(member, units[member]) for member in second if member in units]
    for member in first:
        unit = units.get(member)
        if unit is not None:
            for other, other_unit in second_units:
                yield MemberPair(member, other, cosine(unit, other_unit))


def closest_pair(
    first: Iterable[str], second: Iterable[str], units: Mapping[str, Sequence[float]]
) -> MemberPair | None:
    """Return the member of ``first`` and the member of ``second`` whose unit vectors
    in ``units`` have the highest cosine, the earliest such pair on a tie, or ``None``
    when either set has no vector."""
    return max(_pairs(first, second, units), key=_score, default=None)


def _score(pair: MemberPair) -> float:
    return pair.score


def linked_groups(
    members: Sequence[str], units: Mapping[str, Sequence[float]], floor: float
) -> list[list[str]]:
    """Return the connected components of ``members``, every one of which has a vector
    in ``units``, when two are linked at a cosine of at least ``floor``.

    Each group keeps the order of ``members``, and the groups are in the order of
    their first member.
    """
    parent = list(range(len(members)))
    for first, second in itertools.combinations(range(len(members)), 2):
        if cosine(units[members[first]], units[members[second]]) >= floor:
            parent[_root(parent, second)] = _root(parent, first)
    groups: dict[int, list[str]] = {}
    for index, member in enumerate(members):
        groups.setdefault(_root(parent, index), []).append(member)
    return list(groups.values())


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
    representative: str,
    units: Mapping[str, Sequence[float]],
    *,
    floor: float = GROUPING_FLOOR,
) -> MemberPlan | None:
    """Return the groups of the ``members`` of a node that have a vector in ``units``,
    and the checks to make on them, or ``None`` when ``representative`` has no vector.

    A node is diffuse when home holds fewer than ``HOME_SHARE`` of those members:
    its images give no reliable signal, so nothing is checked. Otherwise every other
    group is checked against home, larger groups first, ``CHECKS_PER_NODE`` at most.
    A group of two or more members is a split check and a lone member a move check;
    home holding at least half the members means no node of fewer than four is split.
    """
    if representative not in units:
        return None
    grouped = linked_groups([member for member in members if member in units], units, floor)
    home = next(group for group in grouped if representative in group)
    others = sorted((group for group in grouped if group is not home), key=len, reverse=True)
    groups = [home, *others]
    if len(home) < HOME_SHARE * sum(map(len, groups)):
        return MemberPlan(groups, diffuse=True, checks=[], unchecked=[])
    checks = [
        MemberCheck(group, max(_pairs(home, group, units), key=_score))
        for group in others[:CHECKS_PER_NODE]
    ]
    return MemberPlan(groups, diffuse=False, checks=checks, unchecked=others[CHECKS_PER_NODE:])
