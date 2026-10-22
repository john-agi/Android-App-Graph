"""Rounds of verifier questions about which node pairs show one screen, with every
answer that follows from earlier ones deduced instead of asked.

The audit's merge proposals are entity resolution with an oracle: the verifier is the
oracle and each proposed node pair a question. Two answers can decide a third by
transitivity: A = B and B = C give A = C, and A = B and B ≠ C give A ≠ C. As Wang et
al. do [1], the pairs are ordered most likely one screen first, and a round asks every
open pair that the pairs before it could not decide even if all of them were one
screen. Once its answers are in, every open pair they decide is deduced.

A pair is therefore asked exactly when the answers on the pairs before it leave it
open, so a round never asks a pair whose answer could contradict the answers before
it. The "same" answers join the nodes into clusters, and no "different" answer ever
falls inside one.

[1] Wang, Li, Kraska, Franklin, Feng, "Leveraging Transitive Relations for Crowdsourced
    Joins", SIGMOD 2013, https://arxiv.org/abs/1408.6916.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import NamedTuple

type NodePair = tuple[str, str]
type NodeOrder = Callable[[str], tuple[int, str]]
type _Links = dict[str, list[tuple[str, int]]]


class _Forest:
    """Union-find over node IDs: each node points toward the root of its component."""

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def root(self, node: str) -> str:
        """Return the root of ``node``'s component, pointing every node on the way
        straight at it."""
        path = []
        while (parent := self._parent.setdefault(node, node)) != node:
            path.append(node)
            node = parent
        for step in path:
            self._parent[step] = node
        return node

    def join(self, first: str, second: str) -> None:
        self._parent[self.root(second)] = self.root(first)


class Deduction(NamedTuple):
    """An answer that follows from asked ones. ``basis`` holds the indices of the asked
    pairs it follows from, a path of "same" answers with at most one "different"
    answer on it, and ``after`` is the round whose answers decided it."""

    same: bool
    basis: list[int]
    after: int


@dataclass(slots=True)
class PairRounds:
    """The answers on ``pairs``, ordered most likely one screen first, as rounds of
    questions ask them or deduce them.

    Call ``next_round`` for the indices of the pairs to ask, ``record`` each answer, and
    repeat until ``next_round`` returns none: every pair is then asked or deduced.
    """

    pairs: Sequence[NodePair]
    asked: dict[int, bool] = field(default_factory=dict)
    deduced: dict[int, Deduction] = field(default_factory=dict)
    rounds: int = 0

    def next_round(self) -> list[int]:
        """Deduce every open pair the answers so far decide, then return the next
        round's questions: the open pairs that the pairs before them could not decide
        even if every open one of those were one screen. Empty when no pair is open."""
        self._deduce()
        assumed = _Forest()
        apart: list[NodePair] = []
        questions: list[int] = []
        for index, (first, second) in enumerate(self.pairs):
            same = self._answer(index)
            if same is None and not _decided(assumed, apart, first, second):
                questions.append(index)
            if same is False:
                apart.append((first, second))
            else:
                assumed.join(first, second)
        if questions:
            self.rounds += 1
        return questions

    def record(self, index: int, *, same: bool) -> None:
        """Record the verifier's answer on pair ``index``."""
        self.asked[index] = same

    def clusters(self, order: NodeOrder) -> list[list[str]]:
        """Return the clusters the "same" answers join the nodes of the asked pairs
        into, each in ``order``, the clusters in the order of their first node."""
        components = _Forest()
        for index, same in self.asked.items():
            if same:
                components.join(*self.pairs[index])
        asked_nodes = {node for index in self.asked for node in self.pairs[index]}
        by_root: dict[str, list[str]] = {}
        for node in sorted(asked_nodes, key=order):
            by_root.setdefault(components.root(node), []).append(node)
        return list(by_root.values())

    def _answer(self, index: int) -> bool | None:
        if index in self.asked:
            return self.asked[index]
        deduction = self.deduced.get(index)
        return None if deduction is None else deduction.same

    def _deduce(self) -> None:
        components = _Forest()
        links: _Links = {}
        for index, same in self.asked.items():
            if same:
                first, second = self.pairs[index]
                components.join(first, second)
                links.setdefault(first, []).append((second, index))
                links.setdefault(second, []).append((first, index))
        apart: dict[frozenset[str], int] = {}
        for index, same in self.asked.items():
            if not same:
                first, second = self.pairs[index]
                ends = frozenset((components.root(first), components.root(second)))
                apart.setdefault(ends, index)
        for index, (first, second) in enumerate(self.pairs):
            if self._answer(index) is not None:
                continue
            if components.root(first) == components.root(second):
                basis = _path(links, first, second)
                self.deduced[index] = Deduction(same=True, basis=basis, after=self.rounds)
                continue
            known = apart.get(frozenset((components.root(first), components.root(second))))
            if known is not None:
                near, far = self.pairs[known]
                if components.root(near) != components.root(first):
                    near, far = far, near
                basis = [*_path(links, first, near), known, *_path(links, far, second)]
                self.deduced[index] = Deduction(same=False, basis=basis, after=self.rounds)


def _decided(components: _Forest, apart: Sequence[NodePair], first: str, second: str) -> bool:
    """Whether ``first`` and ``second`` are in one component, or in two that a
    "different" answer in ``apart`` separates."""
    ends = {components.root(first), components.root(second)}
    return len(ends) == 1 or any(
        {components.root(near), components.root(far)} == ends for near, far in apart
    )


def _path(links: _Links, start: str, goal: str) -> list[int]:
    """Return the indices of the pairs on a shortest path of ``links`` from ``start`` to
    ``goal``, which ``links`` connects."""
    reached_by: dict[str, tuple[str, int] | None] = {start: None}
    queue = deque([start])
    while goal not in reached_by:
        node = queue.popleft()
        for other, index in links.get(node, []):
            if other not in reached_by:
                reached_by[other] = (node, index)
                queue.append(other)
    path: list[int] = []
    node = goal
    while (step := reached_by[node]) is not None:
        node, index = step
        path.append(index)
    return path[::-1]
