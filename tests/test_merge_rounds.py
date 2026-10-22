"""Tests for android_app_graph.merge_rounds: rounds of questions on node pairs, the
answers deduced from earlier ones, and the clusters the answers join the nodes into."""

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping, Sequence

from hypothesis import given
from hypothesis import strategies as st

from android_app_graph.merge_rounds import Deduction, NodePair, PairRounds

A, B, C, D = "s0_a", "s1_b", "s2_c", "s3_d"


def _order(node: str) -> tuple[int, str]:
    return int(node.split("_", 1)[0][1:]), node


def _run(rounds: PairRounds, answer: Callable[[NodePair], bool | None]) -> list[list[int]]:
    """Ask every round of ``rounds`` with ``answer``, ``None`` for no answer, and return
    each round's questions."""
    asked: list[list[int]] = []
    while questions := rounds.next_round():
        asked.append(questions)
        for index in questions:
            rounds.record(index, same=answer(rounds.pairs[index]))
    return asked


def test_two_same_answers_decide_the_third_pair_without_asking_it() -> None:
    rounds = PairRounds([(A, B), (B, C), (A, C)])

    asked = _run(rounds, lambda _pair: True)

    assert asked == [[0, 1]]
    assert rounds.deduced == {2: Deduction(same=True, basis=[0, 1], after=1)}
    assert rounds.clusters(_order) == [[A, B, C]]


def test_a_same_and_a_different_answer_decide_the_third_pair_apart() -> None:
    rounds = PairRounds([(A, B), (B, C), (A, C)])

    _run(rounds, lambda pair: pair == (A, B))

    assert rounds.asked == {0: True, 1: False}
    assert rounds.deduced == {2: Deduction(same=False, basis=[0, 1], after=1)}
    assert rounds.clusters(_order) == [[A, B], [C]]


def test_a_pair_the_round_assumes_decided_is_asked_in_the_next_round_when_it_is_not() -> None:
    """A round assumes the open pairs before a pair are one screen; a "different" answer
    on one of them leaves the pair to the next round."""
    rounds = PairRounds([(A, B), (B, C), (A, C)])

    asked = _run(rounds, lambda _pair: False)

    assert asked == [[0, 1], [2]]
    assert (rounds.rounds, rounds.deduced) == (2, {})


def test_a_path_with_two_different_answers_decides_nothing() -> None:
    rounds = PairRounds([(A, B), (C, D), (A, C), (B, D)])

    asked = _run(rounds, lambda pair: pair == (A, C))

    assert asked == [[0, 1, 2], [3]]
    assert rounds.deduced == {}


def test_a_pair_without_an_answer_decides_no_other_pair_and_is_not_asked_again() -> None:
    """No answer on A + B is neither "same" nor "different", so with B = C it says
    nothing about A + C, which the next round asks."""
    rounds = PairRounds([(A, B), (B, C), (A, C)])

    asked = _run(rounds, {(A, B): None, (B, C): True, (A, C): False}.__getitem__)

    assert asked == [[0, 1], [2]]
    assert (rounds.asked, rounds.unanswered, rounds.deduced) == ({1: True, 2: False}, {0}, {})
    assert rounds.clusters(_order) == [[A], [B, C]]


def test_the_nodes_of_a_pair_without_an_answer_are_clusters_of_their_own() -> None:
    rounds = PairRounds([(A, B)])

    _run(rounds, lambda _pair: None)

    assert rounds.clusters(_order) == [[A], [B]]


def test_the_same_answers_join_the_nodes_into_clusters_in_node_order() -> None:
    rounds = PairRounds([(C, D), (B, D), (A, B), (A, C)])

    _run(rounds, lambda pair: A not in pair)

    assert rounds.clusters(_order) == [[A], [B, C, D]]


NODES = [f"s{number}_screen" for number in range(7)]


@st.composite
def ordered_pairs(draw: st.DrawFn) -> list[NodePair]:
    nodes = NODES[: draw(st.integers(min_value=2, max_value=len(NODES)))]
    every_pair = list(itertools.combinations(nodes, 2))
    return draw(st.lists(st.sampled_from(every_pair), min_size=1, unique=True))


def _component(answers: Mapping[NodePair, bool], node: str) -> set[str]:
    reached, stack = {node}, [node]
    while stack:
        current = stack.pop()
        for (first, second), same in answers.items():
            for here, there in ((first, second), (second, first)):
                if same and here == current and there not in reached:
                    reached.add(there)
                    stack.append(there)
    return reached


def _follows(answers: Mapping[NodePair, bool], first: str, second: str) -> bool | None:
    """The answer on ``first`` and ``second`` that ``answers`` imply by transitivity, or
    ``None`` when they imply none."""
    near, far = _component(answers, first), _component(answers, second)
    if second in near:
        return True
    for (one, other), same in answers.items():
        if not same and {one, other} & near and {one, other} & far:
            return False
    return None


def _asked_one_at_a_time(
    pairs: Sequence[NodePair], answer: Callable[[NodePair], bool | None]
) -> list[NodePair]:
    asked: list[NodePair] = []
    answered: dict[NodePair, bool] = {}
    for pair in pairs:
        if _follows(answered, *pair) is None:
            asked.append(pair)
            same = answer(pair)
            if same is not None:
                answered[pair] = same
    return asked


@given(pairs=ordered_pairs(), data=st.data())
def test_rounds_ask_the_pairs_that_asking_one_at_a_time_asks(
    pairs: list[NodePair], data: st.DataObject
) -> None:
    """Whatever the verifier answers, consistent or not."""
    answers = {pair: data.draw(st.booleans()) for pair in pairs}
    rounds = PairRounds(pairs)

    _run(rounds, answers.__getitem__)

    assert [pairs[index] for index in sorted(rounds.asked)] == _asked_one_at_a_time(
        pairs, answers.__getitem__
    )
    assert sorted([*rounds.asked, *rounds.deduced]) == list(range(len(pairs)))


@given(pairs=ordered_pairs(), data=st.data())
def test_pairs_without_an_answer_are_asked_once_and_deduce_nothing(
    pairs: list[NodePair], data: st.DataObject
) -> None:
    """A pair without an answer is asked as asking one pair at a time asks it, and no
    deduction rests on it."""
    answers = {pair: data.draw(st.sampled_from([True, False, None])) for pair in pairs}
    rounds = PairRounds(pairs)

    _run(rounds, answers.__getitem__)

    asked = sorted([*rounds.asked, *rounds.unanswered])
    assert [pairs[index] for index in asked] == _asked_one_at_a_time(pairs, answers.__getitem__)
    assert sorted([*asked, *rounds.deduced]) == list(range(len(pairs)))
    for deduction in rounds.deduced.values():
        assert rounds.unanswered.isdisjoint(deduction.basis)


@given(pairs=ordered_pairs(), data=st.data())
def test_answers_asked_in_round_order_never_contradict_the_deductions(
    pairs: list[NodePair], data: st.DataObject
) -> None:
    """However inconsistent the verifier is, no asked answer is the opposite of what the
    other asked answers deduce, so no cluster of "same" answers holds a "different" one."""
    answers = {pair: data.draw(st.booleans()) for pair in pairs}
    rounds = PairRounds(pairs)

    _run(rounds, answers.__getitem__)

    asked = {pairs[index]: same for index, same in rounds.asked.items()}
    for pair, same in asked.items():
        others = {other: answer for other, answer in asked.items() if other != pair}
        assert _follows(others, *pair) in {None, same}


@given(pairs=ordered_pairs(), data=st.data())
def test_each_deduction_follows_from_a_path_of_asked_answers(
    pairs: list[NodePair], data: st.DataObject
) -> None:
    answers = {pair: data.draw(st.booleans()) for pair in pairs}
    rounds = PairRounds(pairs)

    _run(rounds, answers.__getitem__)

    for index, deduction in rounds.deduced.items():
        node, goal = pairs[index]
        for step in deduction.basis:
            first, second = pairs[step]
            assert node in pairs[step]
            node = second if node == first else first
        assert node == goal
        assert [rounds.asked[step] for step in deduction.basis].count(False) == (
            0 if deduction.same else 1
        )


@given(
    pairs=ordered_pairs(),
    screens=st.lists(st.integers(min_value=0, max_value=2), min_size=len(NODES)),
)
def test_consistent_answers_are_deduced_right_and_cluster_into_the_screens(
    pairs: list[NodePair], screens: list[int]
) -> None:
    """Each node shows one of three screens, and the verifier always tells them right."""
    screen_of = dict(zip(NODES, screens, strict=False))

    def answer(pair: NodePair) -> bool:
        return screen_of[pair[0]] == screen_of[pair[1]]

    rounds = PairRounds(pairs)

    _run(rounds, answer)

    for index, deduction in rounds.deduced.items():
        assert deduction.same == answer(pairs[index])
    confirmed = {pair: True for pair in pairs if answer(pair)}
    for cluster in rounds.clusters(_order):
        assert set(cluster) == _component(confirmed, cluster[0])
