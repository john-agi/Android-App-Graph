"""The audit from members: one set of merge proposals from the model's text and from
close member images, verified in rounds, and the split and move checks on each node's
member groups.

Every graph here has synthetic image and text vectors, and the verifier, the text
embedding and the edge normalizer are scripted fakes; describing a screen fails the
test, since the audit works from the describe output the step log stores.
"""

from __future__ import annotations

import base64
import itertools
import json
import logging
import math
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NoReturn, cast

import pytest
from openrouter import OpenRouter

from android_app_graph.commands import audit
from android_app_graph.graph_files import graph_steps_dir, text_embeddings_path
from android_app_graph.member_groups import GROUPING_FLOOR, MERGE_FLOOR
from android_app_graph.step_log import read_exploration_transitions
from android_app_graph.utils import ModelSettings, graph_manager
from android_app_graph.utils.graph_manager import (
    GraphManager,
    MemberQuestion,
    MemberVerdict,
    MemberVerdicts,
    Merge,
)

FAKE_CLIENT = cast("OpenRouter", object())  # never used: every consumer is patched
PAGE_DETAIL = ModelSettings("openai/gpt-6.1-sol")
EMBEDDING = ModelSettings("google/gemini-embedding-2")

X, Y, Z = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]


def _quarter_of_the_way_to(floor: float) -> float:
    """The cosine a quarter of the way from 1 down to ``floor``."""
    return 1 - (1 - floor) / 4


# Two members of a node this alike are linked, and two turned toward each other's
# axis are not.
LINKED = _quarter_of_the_way_to(GROUPING_FLOOR)
# Two nodes whose closest members are this alike are proposed for a merge.
PROPOSED = _quarter_of_the_way_to(MERGE_FLOOR)


def _toward(start: list[float], end: list[float], cosine: float) -> list[float]:
    """The unit vector whose cosine with ``start`` is ``cosine``, turned toward ``end``;
    ``start`` and ``end`` are orthogonal unit vectors."""
    sine = math.sqrt(1 - cosine * cosine)
    return [cosine * a + sine * b for a, b in zip(start, end, strict=True)]


def _turned(angle: float) -> list[float]:
    """The unit vector ``angle`` radians from X toward Y."""
    return [math.cos(angle), math.sin(angle), 0.0]


def _member(image_b64: str) -> str:
    return base64.b64decode(image_b64).decode("ascii")


# How long a question waits for the other questions of its round before it fails the
# test instead of hanging.
MEET_TIMEOUT = 10.0


@dataclass
class FakeModels:
    """A verifier that confirms the member pairs in ``same``, gives no answer once for
    each entry of ``unanswered``, and records each pair it is shown as
    ``(existing, new)``, a text embedding that records each text, and an edge normalizer
    whose every template is ``template``.

    A pair in ``meetings`` is answered only once every question its barrier counts has
    been asked, so a round asked one question at a time breaks the barrier. A pair in
    ``refused`` raises, as a request refused for lack of credit does.
    """

    same: set[frozenset[str]] = field(default_factory=set)
    unanswered: list[frozenset[str]] = field(default_factory=list)
    shown: list[tuple[str, str]] = field(default_factory=list)
    descriptions: list[str] = field(default_factory=list)
    embedded: list[str] = field(default_factory=list)
    template: str = "before"
    meetings: dict[frozenset[str], threading.Barrier] = field(default_factory=dict)
    refused: set[frozenset[str]] = field(default_factory=set)

    def confirm(self, first: str, second: str) -> None:
        self.same.add(frozenset((first, second)))

    def give_no_answer(self, first: str, second: str) -> None:
        self.unanswered.append(frozenset((first, second)))

    def meet(self, *pairs: tuple[str, str]) -> None:
        meeting = threading.Barrier(len(pairs), timeout=MEET_TIMEOUT)
        self.meetings.update((frozenset(pair), meeting) for pair in pairs)

    def verify_same_node(
        self,
        _client: object,
        *,
        screenshot_new_b64: str,
        screenshot_existing_b64: str,
        existing_description: str,
        **_settings: object,
    ) -> dict[str, Any]:
        pair = (_member(screenshot_existing_b64), _member(screenshot_new_b64))
        if frozenset(pair) in self.refused:
            msg = f"the verifier refused {pair}"
            raise RuntimeError(msg)
        meeting = self.meetings.get(frozenset(pair))
        if meeting is not None:
            meeting.wait()
        self.shown.append(pair)
        self.descriptions.append(existing_description)
        if frozenset(pair) in self.unanswered:
            self.unanswered.remove(frozenset(pair))
            return {"same": False, "answered": False, "reason": "timed out"}
        return {"same": frozenset(pair) in self.same, "reason": "scripted"}

    def get_embedding(
        self, _client: object, text: str, *_args: object, **_kwargs: object
    ) -> list[float]:
        self.embedded.append(text)
        return [1.0, 0.0]

    def normalize_edge(self, *_args: object, **_kwargs: object) -> dict[str, Any]:
        return {"is_template": True, "instruction_template": self.template}


def _no_describe(*_args: object, **_kwargs: object) -> NoReturn:
    msg = "the audit must not describe a screen"
    raise AssertionError(msg)


@pytest.fixture
def models(monkeypatch: pytest.MonkeyPatch) -> FakeModels:
    fake = FakeModels()
    monkeypatch.setattr(graph_manager, "verify_same_node", fake.verify_same_node)
    monkeypatch.setattr(graph_manager, "get_embedding", fake.get_embedding)
    monkeypatch.setattr(graph_manager, "normalize_edge", fake.normalize_edge)
    monkeypatch.setattr(graph_manager, "describe_page_and_state", _no_describe)
    return fake


@pytest.fixture
def graph_path(tmp_path: Path) -> Path:
    return tmp_path / "graphs" / "demo" / "demo.json"


def _load(path: Path) -> GraphManager:
    graph = GraphManager(
        client=FAKE_CLIENT, page_detail=PAGE_DETAIL, embedding=EMBEDDING, image_embedding=EMBEDDING
    )
    graph.load_graph(path)
    return graph


def _check_members(graph: GraphManager, path: Path, verdicts: MemberVerdicts) -> audit.MemberReport:
    return audit.check_members(graph, verdicts, read_exploration_transitions(graph_steps_dir(path)))


def _checks(report: audit.MemberReport) -> list[dict[str, Any]]:
    """The checks of ``report`` as ``<app_name>_members.json`` keeps them."""
    return json.loads(json.dumps(report.as_record()))["checks"]


# Two nodes whose representatives look nothing alike, but whose second members do.
CLOSE_PAIR = {
    "s0_home": [("000001", X), ("000002", Y)],
    "s1_home": [("000003", Z), ("000004", _toward(Y, Z, PROPOSED))],
}


@pytest.mark.parametrize("same", [True, False])
def test_a_close_pair_of_nodes_is_merged_only_when_the_verifier_confirms_their_closest_members(
    same: bool,
    graph_path: Path,
    models: FakeModels,
    write_member_graph: Callable[..., None],
) -> None:
    """The more-visited node's member is the existing screen, with that node's
    description; the lower-numbered node is kept."""
    write_member_graph(graph_path, CLOSE_PAIR)
    graph = _load(graph_path)
    graph.graph.nodes["s1_home"]["visit_count"] = 5
    if same:
        models.confirm("000002", "000004")

    (proposal,) = graph.merge_proposals([])
    (result,) = graph.verify_merge_proposals([proposal], {}).pairs

    assert (proposal.node_a, proposal.node_b, proposal.sources) == ("s0_home", "s1_home", ["image"])
    assert proposal.findings == [
        {"source": "image", "members": ["000002", "000004"], "score": pytest.approx(PROPOSED)}
    ]
    assert (models.shown, models.descriptions) == ([("000004", "000002")], ["Node s1_home"])
    assert result["members"] == {"s0_home": "000002", "s1_home": "000004"}
    if same:
        assert (result["status"], result["kept"]) == ("merged", "s0_home")
        assert list(graph.graph) == ["s0_home"]
        assert graph.graph.nodes["s0_home"]["observations"] == [
            "000001",
            "000002",
            "000003",
            "000004",
        ]
    else:
        assert result["status"] == "kept_separate"
        assert list(graph.graph) == ["s0_home", "s1_home"]


def test_a_pair_the_model_and_the_images_propose_is_verified_once_naming_both_sources(
    graph_path: Path, models: FakeModels, write_member_graph: Callable[..., None]
) -> None:
    write_member_graph(graph_path, CLOSE_PAIR)
    graph = _load(graph_path)
    issue = {"type": "merge_nodes", "node_a": "s1_home", "node_b": "s0_home"}

    proposals = graph.merge_proposals([issue])
    (result,) = graph.verify_merge_proposals(proposals, {}).pairs

    assert result["sources"] == ["text_audit", "image"]
    assert [finding["source"] for finding in proposals[0].findings] == ["text_audit", "image"]
    assert proposals[0].findings[0] == {"source": "text_audit", "issue": issue}
    assert models.shown == [("000002", "000004")]


def test_a_pair_whose_node_an_earlier_merge_removed_is_verified_on_the_node_that_kept_it(
    graph_path: Path, models: FakeModels, write_member_graph: Callable[..., None]
) -> None:
    """A pair is verified on the nodes that merges before the call left, so a pair whose
    two nodes earlier merges made one is skipped without a verifier call."""
    write_member_graph(
        graph_path,
        {"s0_home": [("000001", X)], "s1_home": [("000002", Y)], "s2_home": [("000003", Z)]},
    )
    graph = _load(graph_path)
    models.confirm("000001", "000002")
    models.confirm("000001", "000003")
    earlier = [{"node_a": "s0_home", "node_b": "s1_home"}]
    graph.verify_merge_proposals(graph.merge_proposals(earlier), {})
    issues = [
        {"node_a": "s1_home", "node_b": "s2_home"},
        {"node_a": "s1_home", "node_b": "s0_home"},
    ]

    report = graph.verify_merge_proposals(graph.merge_proposals(issues), {})

    assert [pair["status"] for pair in report.pairs] == ["merged", "skipped"]
    assert report.pairs[0]["members"] == {"s0_home": "000001", "s2_home": "000003"}
    assert report.merges == [Merge("s0_home", ["s2_home"])]
    assert report.pairs[1]["reason"] == "same node"
    assert models.shown == [("000001", "000002"), ("000001", "000003")]
    assert list(graph.graph) == ["s0_home"]


def test_images_propose_no_pair_with_an_external_node_or_a_node_of_another_package(
    graph_path: Path, models: FakeModels, write_member_graph: Callable[..., None]
) -> None:
    write_member_graph(
        graph_path,
        {
            "s0_home": [("000001", X)],
            "s1_dialog": [("000002", X)],
            "ext_com.other": [("000003", X)],
        },
    )
    graph = _load(graph_path)
    graph.graph.nodes["s1_dialog"]["package"] = "com.google.android.permissioncontroller"

    assert graph.merge_proposals([]) == []
    assert _check_members(graph, graph_path, {}).checks == []
    assert models.shown == []


def test_the_coverage_checkpoint_verifies_the_model_s_pairs_on_closest_members_and_no_image_pair(
    graph_path: Path,
    models: FakeModels,
    monkeypatch: pytest.MonkeyPatch,
    write_member_graph: Callable[..., None],
) -> None:
    """Image proposals would bring back at every checkpoint each pair an earlier one
    kept apart, so the checkpoint verifies only the model's pairs."""
    write_member_graph(
        graph_path, {**CLOSE_PAIR, "s2_list": [("000005", Z), ("000006", _toward(X, Z, 0.6))]}
    )
    graph = _load(graph_path)
    monkeypatch.setattr(
        graph_manager,
        "audit_merge_nodes",
        lambda *_args, **_kwargs: {"issues": [{"node_a": "s2_list", "node_b": "s0_home"}]},
    )

    result = graph.run_node_merge_audit()

    assert models.shown == [("000001", "000006")]
    assert result["merged_count"] == 0


SET_PATTERN = "com.demo.password.ChooseLockPattern"
SET_PIN = "com.demo.password.ChooseLockPassword"


def test_two_black_nodes_of_different_activities_are_kept_apart_unverified_whoever_proposes_them(
    graph_path: Path, models: FakeModels, write_member_graph: Callable[..., None]
) -> None:
    """Every black capture looks the same, so the images propose the pair, and the
    verifier would compare two identical black images and call them one screen."""
    write_member_graph(
        graph_path,
        {"s0_pattern": [("000001", X)], "s1_pin": [("000002", X)]},
        black={"000001": SET_PATTERN, "000002": SET_PIN},
    )
    graph = _load(graph_path)
    models.confirm("000001", "000002")
    issue = {"type": "merge_nodes", "node_a": "s0_pattern", "node_b": "s1_pin"}

    (result,) = graph.verify_merge_proposals(graph.merge_proposals([issue]), {}).pairs

    assert result["sources"] == ["text_audit", "image"]
    assert (result["status"], result["reason"]) == (
        "skipped",
        "black screens of different activities",
    )
    assert models.shown == []
    assert list(graph.graph) == ["s0_pattern", "s1_pin"]


# Home is the first two members; the last two look alike, and nothing like home.
SPLIT = {
    "s0_home": [
        ("000001", X),
        ("000002", _toward(X, Y, LINKED)),
        ("000003", _toward(Y, X, LINKED)),
        ("000004", Y),
    ],
    "s1_list": [("000010", Z)],
}
TAP_A, TAP_B, TAP_C, TAP_D = ({"action": "tap", "x": x, "y": 1} for x in range(4))
BACK, SCROLL = {"action": "back"}, {"action": "swipe", "direction": "up"}


def _edge(source: str, target: str, *actions: dict[str, Any], **extra: object) -> dict[str, object]:
    return {
        "source": source,
        "target": target,
        "actions": [[action] for action in actions],
        "instructions": [f"do {index}" for index in range(len(actions))],
        "target_observations": [f"{target} after {index}" for index in range(len(actions))],
        "num_steps": [1] * len(actions),
        "schema_deltas": [None] * len(actions),
        "setups": [None] * len(actions),
        "visit_count": len(actions),
        **extra,
    }


SPLIT_EDGES = (
    _edge(
        "s0_home",
        "s1_list",
        TAP_A,
        TAP_B,
        TAP_C,
        TAP_D,
        visit_count=5,
        instruction_templates=[{"template": "do {n}"}],
    ),
    _edge("s1_list", "s0_home", BACK),
    _edge("s0_home", "s0_home", SCROLL),
)
SPLIT_TRANSITIONS = (
    ("000001", "000010", TAP_A, "s0_home", "s1_list"),
    ("000003", "000010", TAP_A, "s0_home", "s1_list"),
    ("000004", "000010", TAP_B, "s0_home", "s1_list"),
    # A screenshot no node lists, such as one taken after a wait, keeps its recorded node.
    ("000099", "000010", TAP_C, "s0_home", "s1_list"),
    ("000010", "000003", BACK, "s1_list", "s0_home"),
    ("000001", "000002", SCROLL, "s0_home", "s0_home"),
    ("000003", "000004", SCROLL, "s0_home", "s0_home"),
)


def _edges(graph: GraphManager) -> dict[tuple[str, str], list[object]]:
    return {(s, t): data["actions"] for s, t, data in graph.graph.edges(data=True)}


def test_a_node_with_two_groups_is_split_and_each_action_follows_its_transitions(
    graph_path: Path, models: FakeModels, write_member_graph: Callable[..., None]
) -> None:
    write_member_graph(graph_path, SPLIT, edges=SPLIT_EDGES, transitions=SPLIT_TRANSITIONS)
    graph = _load(graph_path)

    (check,) = _checks(_check_members(graph, graph_path, {}))

    new = "s2_screen_000003"
    assert (check["kind"], check["group"], check["members"]) == (
        "split",
        ["000003", "000004"],
        ["000002", "000003"],
    )
    assert check["image_score"] == pytest.approx(2 * LINKED * math.sqrt(1 - LINKED**2))
    assert models.shown == [("000002", "000003")]
    assert graph.graph.nodes["s0_home"]["observations"] == ["000001", "000002"]
    assert graph.graph.nodes[new]["observations"] == ["000003", "000004"]
    assert _edges(graph) == {
        # TAP_A was taken from both sides, TAP_C from a screenshot no node lists, and no
        # transition names TAP_D.
        ("s0_home", "s1_list"): [[TAP_A], [TAP_C], [TAP_D]],
        (new, "s1_list"): [[TAP_A], [TAP_B]],
        (new, new): [[SCROLL]],
        ("s0_home", "s0_home"): [[SCROLL]],
        ("s1_list", new): [[BACK]],
    }
    kept = graph.graph["s0_home"]["s1_list"]
    assert (kept["instructions"], kept["instruction_templates"], kept["visit_count"]) == (
        ["do 0", "do 2", "do 3"],
        [],
        3,
    )
    assert graph.graph[new]["s1_list"]["visit_count"] == 2
    assert graph.graph["s1_list"][new]["target_observations"] == ["Screen 000003"]
    assert graph.graph["s0_home"]["s0_home"]["visit_count"] == 1
    change = check["change"]
    assert (change["new_node"], change["from"], change["to"]) == (True, "s0_home", new)
    assert {"edge": ["s1_list", "s0_home"], "action": [BACK], "now": [["s1_list", new]]} in change[
        "edge_actions"
    ]


def test_a_node_built_by_a_split_takes_its_earliest_member_s_stored_describe_output(
    graph_path: Path, models: FakeModels, write_member_graph: Callable[..., None]
) -> None:
    """Its members keep their text vectors, so no embedding is requested."""
    write_member_graph(graph_path, SPLIT)
    graph = _load(graph_path)

    _check_members(graph, graph_path, {})

    node = graph.graph.nodes["s2_screen_000003"]
    assert node["page_description"] == "Screen 000003"
    assert node["representative_observation"] == "000003"
    assert node["interactable_elements"] == [
        {"description": "Button 000003", "position": "", "explored": False}
    ]
    assert node["state_schema"] == {"seen_in": ["000003", "000004"]}
    assert node["activities"] == ["com.demo.Activity000003", "com.demo.Activity000004"]
    assert (node["visit_count"], graph.graph.nodes["s0_home"]["visit_count"]) == (2, 2)
    assert models.embedded == []


def test_a_node_whose_groups_the_verifier_calls_one_screen_is_left_whole(
    graph_path: Path, models: FakeModels, write_member_graph: Callable[..., None]
) -> None:
    write_member_graph(graph_path, SPLIT, edges=SPLIT_EDGES, transitions=SPLIT_TRANSITIONS)
    graph = _load(graph_path)
    models.confirm("000002", "000003")
    before = _edges(graph)

    (check,) = _check_members(graph, graph_path, {}).checks

    assert check["same"] is True
    assert "change" not in check
    assert list(graph.graph) == ["s0_home", "s1_list"]
    assert _edges(graph) == before


# The third member of Home looks like List's member, and nothing like Menu's.
MOVE = {
    "s0_home": [("000001", X), ("000002", _toward(X, Y, LINKED)), ("000003", Y)],
    "s1_list": [("000004", _toward(Y, Z, LINKED))],
    "s2_menu": [("000005", Z)],
}
MOVE_EDGES = (_edge("s0_home", "s2_menu", TAP_A),)
MOVE_TRANSITIONS = (("000003", "000005", TAP_A, "s0_home", "s2_menu"),)


@pytest.mark.parametrize(
    ("confirmed", "destination", "shown"),
    [
        (("000002", "000003"), "s0_home", [("000002", "000003")]),
        (("000004", "000003"), "s1_list", [("000002", "000003"), ("000004", "000003")]),
        (
            None,
            "s3_screen_000003",
            [("000002", "000003"), ("000004", "000003"), ("000005", "000003")],
        ),
    ],
    ids=["stays", "moves-to-the-confirmed-node", "moves-to-a-new-node"],
)
def test_a_far_member_moves_only_when_its_closest_home_member_is_another_screen(
    confirmed: tuple[str, str] | None,
    destination: str,
    shown: list[tuple[str, str]],
    graph_path: Path,
    models: FakeModels,
    write_member_graph: Callable[..., None],
) -> None:
    """A rejected member goes to the first node the short list confirms, without the
    node it leaves; when none does, it starts a node of its own. Its text vector is the
    short list's text query, so no embedding is requested."""
    write_member_graph(graph_path, MOVE, edges=MOVE_EDGES, transitions=MOVE_TRANSITIONS)
    graph = _load(graph_path)
    if confirmed is not None:
        models.confirm(*confirmed)

    (check,) = _checks(_check_members(graph, graph_path, {}))

    assert (check["kind"], check["group"]) == ("move", ["000003"])
    assert models.shown == shown
    assert "000003" in graph.graph.nodes[destination]["observations"]
    assert (destination, "s2_menu") in graph.graph.edges
    assert models.embedded == []
    if destination == "s0_home":
        return
    assert graph.graph.nodes["s0_home"]["observations"] == ["000001", "000002"]
    assert ("s0_home", "s2_menu") not in graph.graph.edges
    candidates = check["change"]["short_list"]["candidates"]
    # Member 000003's text vector is Y, so List's member scores LINKED and Menu's 0.
    assert [(candidate["node"], candidate["sources"]) for candidate in candidates] == [
        ("s1_list", {"image": pytest.approx(LINKED), "text": pytest.approx(LINKED)}),
        ("s2_menu", {"image": pytest.approx(0.0), "text": pytest.approx(0.0)}),
    ]


def _templates_around_member_checks(
    graph: GraphManager, path: Path, models: FakeModels
) -> dict[tuple[str, str], list[str]]:
    """Normalize every edge, check the members, normalize again, and return each edge's
    templates: ``before`` as the first pass wrote them, ``after`` as the second did."""
    graph.normalize_all_edges()
    _check_members(graph, path, {})
    models.template = "after"
    graph.normalize_all_edges()
    return {
        (s, t): [template["template"] for template in data["instruction_templates"]]
        for s, t, data in graph.graph.edges(data=True)
    }


@pytest.mark.parametrize("instruction", ["do 1", ""], ids=["instructed", "no-instruction"])
def test_a_split_has_only_the_edges_whose_actions_it_changes_normalized_again(
    instruction: str,
    graph_path: Path,
    models: FakeModels,
    write_member_graph: Callable[..., None],
) -> None:
    """An edge that loses an action, with an instruction or without one, and every edge
    of the new node are sent again. An edge whose actions all stay is not, even when its
    visit count drops, and neither is an edge of another node."""
    home_to_list, *other_edges = SPLIT_EDGES
    edges = (
        {**home_to_list, "instructions": ["do 0", instruction, "do 2", "do 3"]},
        *other_edges,
        _edge("s1_list", "s1_list", SCROLL),
    )
    write_member_graph(graph_path, SPLIT, edges=edges, transitions=SPLIT_TRANSITIONS)
    graph = _load(graph_path)

    templates = _templates_around_member_checks(graph, graph_path, models)

    new = "s2_screen_000003"
    assert templates == {
        ("s0_home", "s1_list"): ["after"],
        (new, "s1_list"): ["after"],
        (new, new): ["after"],
        ("s1_list", new): ["after"],
        ("s0_home", "s0_home"): ["before"],
        ("s1_list", "s1_list"): ["before"],
    }


def test_a_move_has_the_edge_that_gains_the_member_s_action_normalized_again(
    graph_path: Path, models: FakeModels, write_member_graph: Callable[..., None]
) -> None:
    """The edge the moved member's action joins is sent again; another edge of the node
    the member joins is not."""
    edges = (*MOVE_EDGES, _edge("s1_list", "s2_menu", TAP_B), _edge("s2_menu", "s1_list", BACK))
    write_member_graph(graph_path, MOVE, edges=edges, transitions=MOVE_TRANSITIONS)
    graph = _load(graph_path)
    models.confirm("000004", "000003")

    templates = _templates_around_member_checks(graph, graph_path, models)

    assert templates == {("s1_list", "s2_menu"): ["after"], ("s2_menu", "s1_list"): ["before"]}


def test_a_moving_member_without_an_answer_on_a_candidate_goes_on_to_the_next(
    graph_path: Path,
    models: FakeModels,
    write_member_graph: Callable[..., None],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """List's member has the moving member's image vector and description vector, so
    List is the first candidate; the check on it gets no answer, counts as a rejection
    and is recorded as no answer."""
    write_member_graph(
        graph_path,
        {**MOVE, "s1_list": [("000004", Y)]},
        edges=MOVE_EDGES,
        transitions=MOVE_TRANSITIONS,
    )
    graph = _load(graph_path)
    models.give_no_answer("000004", "000003")

    with caplog.at_level(logging.INFO):
        (check,) = _checks(_check_members(graph, graph_path, {}))

    short_list = check["change"]["short_list"]
    assert short_list["candidates"][0] == {
        "node": "s1_list",
        "sources": {"image": pytest.approx(1.0), "text": pytest.approx(1.0)},
        "closest_member": "000004",
        "image_score": pytest.approx(1.0),
    }
    assert short_list["verifications"][0] == {
        "node": "s1_list",
        "member": "000004",
        "same": False,
        "answered": False,
        "reason": "timed out",
    }
    assert "Verifier gave no answer on node s1_list against member 000004 (timed out); " in (
        caplog.text
    )
    assert check["change"]["to"] == "s3_screen_000003"
    assert models.shown == [("000002", "000003"), ("000004", "000003"), ("000005", "000003")]


@pytest.mark.parametrize(
    ("nodes", "edges", "transitions", "kind"),
    [
        (SPLIT, SPLIT_EDGES, SPLIT_TRANSITIONS, "split"),
        (MOVE, MOVE_EDGES, MOVE_TRANSITIONS, "move"),
    ],
    ids=["split", "move"],
)
def test_a_split_or_move_check_without_an_answer_leaves_the_node_whole(
    nodes: dict[str, list[tuple[str, list[float]]]],
    edges: tuple[dict[str, object], ...],
    transitions: tuple[tuple[str, str, object, str, str], ...],
    kind: str,
    graph_path: Path,
    models: FakeModels,
    write_member_graph: Callable[..., None],
) -> None:
    """A change rests on a verifier answer, and the verifier was already asked twice, so
    the check is reported as unanswered, neither "same" nor "different"."""
    write_member_graph(graph_path, nodes, edges=edges, transitions=transitions)
    graph = _load(graph_path)
    members = {node: graph.graph.nodes[node]["observations"] for node in graph.graph}
    before = _edges(graph)
    models.give_no_answer("000002", "000003")

    (check,) = _checks(_check_members(graph, graph_path, {}))

    assert (check["kind"], check["members"]) == (kind, ["000002", "000003"])
    assert (check["same"], check["reason"]) == (None, "timed out")
    assert "change" not in check
    assert {node: graph.graph.nodes[node]["observations"] for node in graph.graph} == members
    assert _edges(graph) == before
    assert models.shown == [("000002", "000003")]


# Home's last member is a black screen of the PIN activity, and Pattern's member a black
# screen of another; every black member looks the same.
BLACK_MOVE = {
    "s0_home": [("000001", X), ("000002", _toward(X, Y, LINKED)), ("000003", Z)],
    "s1_pattern": [("000004", Z)],
}
BLACK_MEMBERS = {"000003": SET_PIN, "000004": SET_PATTERN, "000005": SET_PIN}


@pytest.mark.parametrize("pin_node", [True, False], ids=["joins-the-pin-node", "starts-a-node"])
def test_a_black_member_moves_by_its_activity_not_by_the_verifier(
    pin_node: bool, graph_path: Path, models: FakeModels, write_member_graph: Callable[..., None]
) -> None:
    """The short list would rank every black node first, and the verifier would call two
    identical black images one screen."""
    nodes = {**BLACK_MOVE, "s2_pin": [("000005", Z)]} if pin_node else BLACK_MOVE
    write_member_graph(graph_path, nodes, black=BLACK_MEMBERS)
    graph = _load(graph_path)
    models.confirm("000004", "000003")

    (check,) = _checks(_check_members(graph, graph_path, {}))

    destination = "s2_pin" if pin_node else "s2_secure_screen:_chooselockpassword"
    assert models.shown == [("000001", "000003")]
    assert models.embedded == []
    change = check["change"]
    assert (change["to"], change["new_node"], change["black_activity"]) == (
        destination,
        not pin_node,
        SET_PIN,
    )
    assert "000003" in graph.graph.nodes[destination]["observations"]
    assert graph.graph.nodes["s1_pattern"]["observations"] == ["000004"]
    assert graph.text_vector("000003") is None
    if not pin_node:
        assert graph.graph.nodes[destination]["page_description"] == (
            "secure screen: ChooseLockPassword"
        )


SCROLLED, SCROLLED_TWICE = [SCROLL], [SCROLL, SCROLL]


def _actions_and_setups(graph: GraphManager) -> dict[tuple[str, str], list[tuple[object, object]]]:
    return {
        (s, t): list(zip(data["actions"], data["setups"], strict=True))
        for s, t, data in graph.graph.edges(data=True)
    }


@pytest.mark.parametrize(
    ("nodes", "edges", "transitions", "expected"),
    [
        (
            SPLIT,
            (
                _edge(
                    "s0_home",
                    "s1_list",
                    TAP_A,
                    TAP_B,
                    TAP_C,
                    TAP_D,
                    setups=[SCROLLED, [], None, SCROLLED_TWICE],
                ),
                _edge("s1_list", "s0_home", BACK, setups=[SCROLLED]),
                _edge("s0_home", "s0_home", SCROLL, setups=[[]]),
            ),
            SPLIT_TRANSITIONS,
            {
                ("s0_home", "s1_list"): [
                    ([TAP_A], SCROLLED),
                    ([TAP_C], None),
                    ([TAP_D], SCROLLED_TWICE),
                ],
                ("s2_screen_000003", "s1_list"): [([TAP_A], SCROLLED), ([TAP_B], [])],
                ("s2_screen_000003", "s2_screen_000003"): [([SCROLL], [])],
                ("s0_home", "s0_home"): [([SCROLL], [])],
                ("s1_list", "s2_screen_000003"): [([BACK], SCROLLED)],
            },
        ),
        (
            MOVE,
            (_edge("s0_home", "s2_menu", TAP_A, setups=[SCROLLED]),),
            MOVE_TRANSITIONS,
            {("s3_screen_000003", "s2_menu"): [([TAP_A], SCROLLED)]},
        ),
    ],
    ids=["split", "move"],
)
@pytest.mark.usefixtures("models")
def test_an_action_that_follows_moved_members_keeps_its_setup(
    nodes: dict[str, list[tuple[str, list[float]]]],
    edges: tuple[dict[str, object], ...],
    transitions: tuple[tuple[str, str, object, str, str], ...],
    expected: dict[tuple[str, str], list[tuple[object, object]]],
    graph_path: Path,
    write_member_graph: Callable[..., None],
) -> None:
    """Every edge an action is put on takes its setup as recorded, known, empty or
    unknown: the setup was sent on the screen the action was taken from, and the
    action stays with that screen."""
    write_member_graph(graph_path, nodes, edges=edges, transitions=transitions)
    graph = _load(graph_path)

    _check_members(graph, graph_path, {})

    assert _actions_and_setups(graph) == expected


def test_single_member_diffuse_and_vectorless_nodes_get_no_check(
    graph_path: Path, models: FakeModels, write_member_graph: Callable[..., None]
) -> None:
    write_member_graph(
        graph_path,
        {
            "s0_home": [("000001", X), ("000002", [0.0, 0.0, 0.0])],
            "s1_feed": [
                ("000003", X),
                ("000004", Y),
                ("000005", Z),
                ("000006", _toward(Y, Z, 0.7)),
            ],
            "s2_blank": [("000007", [0.0, 0.0, 0.0]), ("000008", X)],
            "ext_com.other": [("000009", X), ("000010", Y)],
        },
    )
    graph = _load(graph_path)

    report = _check_members(graph, graph_path, {})

    assert report.checks == []
    assert report.diffuse == [
        {"node": "s1_feed", "groups": [["000003"], ["000004"], ["000005"], ["000006"]]}
    ]
    assert models.shown == []


@pytest.mark.parametrize(
    ("issues", "across", "sources"),
    [
        ([{"node_a": "s0_home", "node_b": "s1_home"}], 0.0, ["text_audit"]),
        ([], (MERGE_FLOOR + GROUPING_FLOOR) / 2, ["image"]),
    ],
    ids=["text", "image-between-the-floors"],
)
def test_a_merge_below_the_grouping_floor_is_not_split_again(
    issues: list[dict[str, str]],
    across: float,
    sources: list[str],
    graph_path: Path,
    models: FakeModels,
    write_member_graph: Callable[..., None],
) -> None:
    """A merged pair whose closest members fall below the grouping floor, as a pair the
    model proposes can, or one the images propose between the merge floor and the
    grouping floor, is two groups of the merged node. Their split check asks about
    the pair the merge was verified on and gets its answer back, with no second
    verifier call."""
    linked = math.acos(LINKED)
    apart = math.acos(across)
    write_member_graph(
        graph_path,
        {
            "s0_home": [("000001", _turned(0)), ("000002", _turned(linked))],
            "s1_home": [
                ("000003", _turned(linked + apart)),
                ("000004", _turned(2 * linked + apart)),
            ],
        },
    )
    graph = _load(graph_path)
    models.confirm("000002", "000003")
    verdicts: MemberVerdicts = {}

    (proposal,) = graph.merge_proposals(issues)
    (merge,) = graph.verify_merge_proposals([proposal], verdicts).pairs
    (check,) = _check_members(graph, graph_path, verdicts).checks

    assert proposal.sources == sources
    assert merge["status"] == "merged"
    assert merge["image_score"] == pytest.approx(across)
    assert (check["kind"], check["group"]) == ("split", ["000003", "000004"])
    assert (check["members"], check["same"], check["reused"]) == (["000002", "000003"], True, True)
    assert models.shown == [("000002", "000003")]
    assert list(graph.graph) == ["s0_home"]


def test_a_member_pair_the_verifier_gave_no_answer_on_is_asked_about_again_later_in_the_run(
    graph_path: Path, models: FakeModels, write_member_graph: Callable[..., None]
) -> None:
    """Settings' far member 000004 looks like Home's only member. The merge check on the
    two gets no answer, so the nodes stay apart and the run keeps no answer for the pair.
    The member's move check then shows the verifier the same pair, which it confirms."""
    write_member_graph(
        graph_path,
        {
            "s0_home": [("000001", X)],
            "s1_settings": [
                ("000002", Y),
                ("000003", _toward(Y, Z, LINKED)),
                ("000004", _toward(X, Y, PROPOSED)),
            ],
        },
    )
    graph = _load(graph_path)
    models.give_no_answer("000001", "000004")
    models.confirm("000001", "000004")
    verdicts: MemberVerdicts = {}

    (merge,) = graph.verify_merge_proposals(graph.merge_proposals([]), verdicts).pairs

    assert (merge["status"], merge["reason"]) == ("kept_separate", "timed out")
    assert verdicts == {}

    (check,) = _checks(_check_members(graph, graph_path, verdicts))

    assert (check["kind"], check["members"], check["same"]) == ("move", ["000002", "000004"], False)
    assert check["change"]["to"] == "s0_home"
    assert check["change"]["short_list"]["verifications"] == [
        {
            "node": "s0_home",
            "member": "000001",
            "same": True,
            "answered": True,
            "reason": "scripted",
        }
    ]
    assert models.shown == [("000004", "000001"), ("000002", "000004"), ("000001", "000004")]


@pytest.mark.usefixtures("model_environment", "opened_clients")
def test_the_audit_reports_every_member_check_and_saves_the_graph_it_changed(
    tmp_path: Path,
    graph_path: Path,
    models: FakeModels,
    monkeypatch: pytest.MonkeyPatch,
    write_member_graph: Callable[..., None],
) -> None:
    """The saved copy's text vectors are the explored graph's, since the split requests
    no embedding."""
    write_member_graph(graph_path, SPLIT)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setattr(audit, "run_audit", lambda *_args: {"issues": []})
    monkeypatch.setattr(audit, "normalize_and_save_audited_graph", lambda *_args: 0)
    config = tmp_path / "audit.yaml"
    config.write_text(f"experiment:\n  graph_dir: {graph_path.parent.parent}\n", encoding="utf-8")

    assert audit.main(["-c", str(config), "--app", "demo"]) == 0

    app_dir = graph_path.parent
    report = json.loads((app_dir / "demo_members.json").read_text(encoding="utf-8"))
    assert (report["grouping_floor"], report["home_share"], report["checks_per_node"]) == (
        GROUPING_FLOOR,
        0.5,
        3,
    )
    assert [check["kind"] for check in report["checks"]] == ["split"]
    assert (report["diffuse"], report["unchecked"]) == ([], [])
    audited = json.loads((app_dir / "demo_audited.json").read_text(encoding="utf-8"))
    assert [node["id"] for node in audited["nodes"]] == ["s0_home", "s1_list", "s2_screen_000003"]
    assert not (app_dir / "demo_merge.json").exists()
    assert models.shown == [("000002", "000003")]
    assert models.embedded == []
    explored_vectors = text_embeddings_path(graph_path).read_text(encoding="utf-8")
    audited_vectors = text_embeddings_path(app_dir / "demo_audited.json").read_text(
        encoding="utf-8"
    )
    assert json.loads(audited_vectors) == json.loads(explored_vectors)


def _merge_issue(node_a: str, node_b: str) -> dict[str, str]:
    return {"type": "merge_nodes", "node_a": node_a, "node_b": node_b}


def _one_member_nodes(*angles: float) -> dict[str, list[tuple[str, list[float]]]]:
    """Nodes ``s<n>_screen`` of one member each, ``00000<n+1>``, at ``angles[n]`` radians
    from X toward Y, so the closest members of two nodes have the cosine of the angle
    between them."""
    return {
        f"s{number}_screen": [(f"{number + 1:06d}", _turned(angle))]
        for number, angle in enumerate(angles)
    }


def test_two_confirmed_pairs_decide_the_third_and_all_three_nodes_merge(
    graph_path: Path, models: FakeModels, write_member_graph: Callable[..., None]
) -> None:
    """The pair whose members are least alike comes last; the two before it are one
    round, and their answers decide it."""
    write_member_graph(graph_path, _one_member_nodes(0.0, 0.5, 1.0))
    graph = _load(graph_path)
    models.confirm("000001", "000002")
    models.confirm("000002", "000003")
    issues = [
        _merge_issue("s0_screen", "s2_screen"),
        _merge_issue("s0_screen", "s1_screen"),
        _merge_issue("s1_screen", "s2_screen"),
    ]

    report = graph.verify_merge_proposals(graph.merge_proposals(issues, sources=()), {})

    assert sorted(models.shown) == [("000001", "000002"), ("000002", "000003")]
    assert (report.rounds, report.asked, report.deduced) == (1, 2, 1)
    assert report.merges == [Merge("s0_screen", ["s1_screen", "s2_screen"])]
    deduced = report.pairs[0]
    assert deduced["image_score"] == pytest.approx(math.cos(1.0))
    assert {key: deduced[key] for key in ("asked", "round", "same", "deduced_from")} == {
        "asked": False,
        "round": 1,
        "same": True,
        "deduced_from": [["s0_screen", "s1_screen"], ["s1_screen", "s2_screen"]],
    }
    assert [(pair["status"], pair["kept"]) for pair in report.pairs] == [
        ("merged", "s0_screen")
    ] * 3
    assert list(graph.graph) == ["s0_screen"]
    assert graph.graph.nodes["s0_screen"]["observations"] == ["000001", "000002", "000003"]


def test_a_round_s_questions_are_in_flight_together(
    graph_path: Path, models: FakeModels, write_member_graph: Callable[..., None]
) -> None:
    """Three pairs with no node in common are one round."""
    write_member_graph(graph_path, _one_member_nodes(0.0, 0.1, 1.0, 1.1, 2.0, 2.1))
    graph = _load(graph_path)
    models.meet(("000001", "000002"), ("000003", "000004"), ("000005", "000006"))
    issues = [_merge_issue(f"s{n}_screen", f"s{n + 1}_screen") for n in (0, 2, 4)]

    report = graph.verify_merge_proposals(graph.merge_proposals(issues, sources=()), {})

    assert (report.rounds, report.asked, report.merges) == (1, 3, [])
    assert len(models.shown) == 3


def test_a_refused_question_raises_once_the_answers_before_it_are_kept(
    graph_path: Path, models: FakeModels, write_member_graph: Callable[..., None]
) -> None:
    write_member_graph(graph_path, _one_member_nodes(0.0, 0.5, 1.0))
    graph = _load(graph_path)
    models.confirm("000001", "000002")
    models.refused.add(frozenset(("000001", "000003")))
    verdicts: MemberVerdicts = {}
    questions = [
        MemberQuestion("000001", "000002", "Node s0_screen"),
        MemberQuestion("000001", "000003", "Node s0_screen"),
    ]

    with pytest.raises(RuntimeError, match="refused"):
        graph.verify_member_pairs(questions, verdicts)

    assert verdicts == {
        frozenset(("000001", "000002")): MemberVerdict(same=True, reason="scripted")
    }


# Three screens over six nodes; the edges between them merge in the order the merges run.
SCREEN_OF = {
    "s0_screen": 0,
    "s1_screen": 1,
    "s2_screen": 0,
    "s3_screen": 1,
    "s4_screen": 0,
    "s5_screen": 2,
}
SCREEN_EDGES = (
    _edge("s0_screen", "s1_screen", TAP_A),
    _edge("s2_screen", "s1_screen", TAP_B),
    _edge("s4_screen", "s3_screen", TAP_C),
    _edge("s1_screen", "s0_screen", BACK),
    _edge("s3_screen", "s2_screen", BACK),
    _edge("s2_screen", "s4_screen", TAP_D),
    _edge("s4_screen", "s5_screen", SCROLL),
    _edge("s3_screen", "s3_screen", SCROLL),
)
SCREEN_PAIRS = [
    ("s2_screen", "s4_screen"),
    ("s0_screen", "s1_screen"),
    ("s0_screen", "s4_screen"),
    ("s1_screen", "s3_screen"),
    ("s0_screen", "s2_screen"),
    ("s3_screen", "s4_screen"),
    ("s4_screen", "s5_screen"),
    ("s2_screen", "s3_screen"),
]


def _number(node_id: str) -> int:
    return int(node_id.split("_", 1)[0][1:])


def test_consistent_answers_give_the_graph_that_verifying_one_pair_at_a_time_gives(
    tmp_path: Path, models: FakeModels, write_member_graph: Callable[..., None]
) -> None:
    """One pair at a time, each proposal is verified on the nodes earlier merges left, and
    a confirmed pair is merged at once into its lower-numbered node."""
    path = tmp_path / "graphs" / "demo" / "demo.json"
    write_member_graph(path, _one_member_nodes(0.0, 1.4, 0.3, 1.0, 0.6, 2.0), edges=SCREEN_EDGES)
    for first, second in itertools.combinations(SCREEN_OF, 2):
        if SCREEN_OF[first] == SCREEN_OF[second]:
            models.confirm(f"{_number(first) + 1:06d}", f"{_number(second) + 1:06d}")
    graph, one_at_a_time = _load(path), _load(path)
    issues = [_merge_issue(*pair) for pair in SCREEN_PAIRS]

    report = graph.verify_merge_proposals(graph.merge_proposals(issues, sources=()), {})
    for pair in SCREEN_PAIRS:
        first, second = sorted(map(one_at_a_time.resolve_node_id, pair), key=_number)
        if first != second and SCREEN_OF[first] == SCREEN_OF[second]:
            one_at_a_time.merge_nodes(first, second)

    assert report.asked < len(SCREEN_PAIRS)
    assert list(graph.graph.nodes(data=True)) == list(one_at_a_time.graph.nodes(data=True))
    assert list(graph.graph.edges(data=True)) == list(one_at_a_time.graph.edges(data=True))
    assert [graph.resolve_node_id(node) for node in SCREEN_OF] == [
        one_at_a_time.resolve_node_id(node) for node in SCREEN_OF
    ]


def _split_node(first: int) -> list[tuple[str, list[float]]]:
    """Members ``first`` to ``first + 3`` in two groups of two, as in ``SPLIT``."""
    vectors = (X, _toward(X, Y, LINKED), _toward(Y, X, LINKED), Y)
    return [(f"{first + offset:06d}", vector) for offset, vector in enumerate(vectors)]


def _move_node(first: int) -> list[tuple[str, list[float]]]:
    """Members ``first`` to ``first + 2``, the last one alone, as in ``MOVE``'s Home."""
    vectors = (X, _toward(X, Y, LINKED), Y)
    return [(f"{first + offset:06d}", vector) for offset, vector in enumerate(vectors)]


def test_member_checks_ask_every_split_at_once_then_every_move_at_once(
    graph_path: Path, models: FakeModels, write_member_graph: Callable[..., None]
) -> None:
    write_member_graph(
        graph_path,
        {
            "s0_home": _split_node(1),
            "s1_list": _split_node(5),
            "s2_menu": _move_node(9),
            "s3_page": _move_node(12),
        },
    )
    graph = _load(graph_path)
    splits = (("000002", "000003"), ("000006", "000007"))
    moves = (("000010", "000011"), ("000013", "000014"))
    for pair in (*splits, *moves):
        models.confirm(*pair)
    models.meet(*splits)
    models.meet(*moves)

    report = _check_members(graph, graph_path, {})

    assert [(check["kind"], check["node"]) for check in report.checks] == [
        ("split", "s0_home"),
        ("split", "s1_list"),
        ("move", "s2_menu"),
        ("move", "s3_page"),
    ]
    assert (set(models.shown[:2]), set(models.shown[2:])) == (set(splits), set(moves))


def test_a_member_moves_to_the_node_a_split_made(
    graph_path: Path, models: FakeModels, write_member_graph: Callable[..., None]
) -> None:
    """Menu's lone member looks like Home's second group, which a split makes a node of
    before any member moves."""
    write_member_graph(
        graph_path,
        {
            "s0_home": _split_node(1),
            "s1_menu": [
                ("000020", Z),
                ("000021", _toward(Z, X, LINKED)),
                ("000022", _toward(Y, Z, LINKED)),
            ],
        },
    )
    graph = _load(graph_path)
    models.confirm("000004", "000022")

    split, move = _checks(_check_members(graph, graph_path, {}))

    new = "s2_screen_000003"
    assert (split["kind"], split["change"]["to"]) == ("split", new)
    assert (move["kind"], move["members"], move["change"]["to"]) == (
        "move",
        ["000020", "000022"],
        new,
    )
    assert graph.graph.nodes[new]["observations"] == ["000003", "000004", "000022"]
