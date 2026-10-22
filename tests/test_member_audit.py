"""The audit from members: one set of merge proposals from the model's text and from
close member images, and the split and move checks on each node's member groups.

Every graph here has synthetic image vectors, and the verifier and the text embedding
are scripted fakes; describing a screen fails the test, since the audit works from
the describe output the step log stores.
"""

from __future__ import annotations

import base64
import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NoReturn, cast

import pytest
from openrouter import OpenRouter

from android_app_graph.commands import audit
from android_app_graph.graph_files import graph_steps_dir
from android_app_graph.member_groups import GROUPING_FLOOR, MERGE_FLOOR
from android_app_graph.step_log import read_exploration_transitions
from android_app_graph.utils import ModelSettings, graph_manager
from android_app_graph.utils.graph_manager import GraphManager, MemberVerdicts

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


@dataclass
class FakeModels:
    """A verifier that confirms the member pairs in ``same`` and records each pair it
    is shown as ``(existing, new)``, and a text embedding that records each text."""

    same: set[frozenset[str]] = field(default_factory=set)
    shown: list[tuple[str, str]] = field(default_factory=list)
    descriptions: list[str] = field(default_factory=list)
    embedded: list[str] = field(default_factory=list)

    def confirm(self, first: str, second: str) -> None:
        self.same.add(frozenset((first, second)))

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
        self.shown.append(pair)
        self.descriptions.append(existing_description)
        return {"same": frozenset(pair) in self.same, "reason": "scripted"}

    def get_embedding(
        self, _client: object, text: str, *_args: object, **_kwargs: object
    ) -> list[float]:
        self.embedded.append(text)
        return [1.0, 0.0]


def _no_describe(*_args: object, **_kwargs: object) -> NoReturn:
    msg = "the audit must not describe a screen"
    raise AssertionError(msg)


@pytest.fixture
def models(monkeypatch: pytest.MonkeyPatch) -> FakeModels:
    fake = FakeModels()
    monkeypatch.setattr(graph_manager, "verify_same_node", fake.verify_same_node)
    monkeypatch.setattr(graph_manager, "get_embedding", fake.get_embedding)
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
    (result,) = graph.verify_merge_proposals([proposal], {})

    assert (proposal.node_a, proposal.node_b, proposal.sources) == ("s0_home", "s1_home", ["image"])
    assert proposal.findings == [
        {"source": "image", "members": ["000002", "000004"], "score": pytest.approx(PROPOSED)}
    ]
    assert (models.shown, models.descriptions) == ([("000004", "000002")], ["Node s1_home"])
    assert result["members"] == {"s0_home": "000002", "s1_home": "000004"}
    if same:
        assert (result["status"], result["kept"], result["removed"]) == (
            "merged",
            "s0_home",
            "s1_home",
        )
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
    (result,) = graph.verify_merge_proposals(proposals, {})

    assert result["sources"] == ["text_audit", "image"]
    assert [finding["source"] for finding in proposals[0].findings] == ["text_audit", "image"]
    assert proposals[0].findings[0] == {"source": "text_audit", "issue": issue}
    assert models.shown == [("000002", "000004")]


def test_a_pair_whose_node_an_earlier_merge_removed_is_verified_on_the_node_that_kept_it(
    graph_path: Path, models: FakeModels, write_member_graph: Callable[..., None]
) -> None:
    """A pair whose two nodes earlier merges made one is skipped without a verifier
    call."""
    write_member_graph(
        graph_path,
        {"s0_home": [("000001", X)], "s1_home": [("000002", Y)], "s2_home": [("000003", Z)]},
    )
    graph = _load(graph_path)
    models.confirm("000001", "000002")
    models.confirm("000001", "000003")
    issues = [
        {"node_a": "s0_home", "node_b": "s1_home"},
        {"node_a": "s1_home", "node_b": "s2_home"},
        {"node_a": "s2_home", "node_b": "s0_home"},
    ]

    results = graph.verify_merge_proposals(graph.merge_proposals(issues), {})

    assert [result["status"] for result in results] == ["merged", "merged", "skipped"]
    assert results[1]["members"] == {"s0_home": "000001", "s2_home": "000003"}
    assert (results[1]["kept"], results[1]["removed"]) == ("s0_home", "s2_home")
    assert results[2]["reason"] == "same node"
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
        "anchors": [[None] for _ in actions],
        "action_flags": [[None] for _ in actions],
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
    write_member_graph(graph_path, SPLIT)
    graph = _load(graph_path)

    _check_members(graph, graph_path, {})

    node = graph.graph.nodes["s2_screen_000003"]
    assert node["page_description"] == "Screen 000003"
    assert node["representative_observation"] == "000003"
    assert node["interactable_elements"] == [
        {
            "locator": {
                "resource_id": "com.demo:id/button",
                "class": "android.widget.Button",
                "label": "Button 000003",
                "ancestor_ids": [],
            },
            "description": "Button 000003",
            "position": "top left",
            "bounds": [0, 0, 100, 100],
            "source": "tree",
            "explored": False,
        }
    ]
    assert node["state_schema"] == {"seen_in": ["000003", "000004"]}
    assert node["activities"] == ["com.demo.Activity000003", "com.demo.Activity000004"]
    assert (node["visit_count"], graph.graph.nodes["s0_home"]["visit_count"]) == (2, 2)
    assert node["description_embedding"] == [1.0, 0.0]
    assert models.embedded == ["Screen 000003"]


@pytest.mark.usefixtures("models")
def test_a_graph_saved_after_a_split_loads_with_the_new_node_s_representative(
    graph_path: Path, write_member_graph: Callable[..., None]
) -> None:
    """The node a split builds takes its earliest member as representative, a recorded
    observation with its tree, so the graph saved after the split meets the loading rule
    on representatives."""
    write_member_graph(graph_path, SPLIT)
    graph = _load(graph_path)
    _check_members(graph, graph_path, {})
    audited = graph_path.with_name("demo_audited.json")
    graph.save_graph(audited)

    loaded = _load(audited)

    assert loaded.graph.nodes["s2_screen_000003"]["representative_observation"] == "000003"


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
    node it leaves; when none does, it starts a node of its own."""
    write_member_graph(graph_path, MOVE, edges=MOVE_EDGES, transitions=MOVE_TRANSITIONS)
    graph = _load(graph_path)
    if confirmed is not None:
        models.confirm(*confirmed)

    (check,) = _checks(_check_members(graph, graph_path, {}))

    assert (check["kind"], check["group"]) == ("move", ["000003"])
    assert models.shown == shown
    assert "000003" in graph.graph.nodes[destination]["observations"]
    assert (destination, "s2_menu") in graph.graph.edges
    if destination == "s0_home":
        assert models.embedded == []
        return
    assert graph.graph.nodes["s0_home"]["observations"] == ["000001", "000002"]
    assert ("s0_home", "s2_menu") not in graph.graph.edges
    # The member's stored description is embedded once, for the short list and the new node.
    assert models.embedded == ["Screen 000003"]
    candidates = check["change"]["short_list"]["candidates"]
    assert [candidate["node"] for candidate in candidates] == ["s1_list", "s2_menu"]


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
    (merge,) = graph.verify_merge_proposals([proposal], verdicts)
    (check,) = _check_members(graph, graph_path, verdicts).checks

    assert proposal.sources == sources
    assert merge["status"] == "merged"
    assert merge["image_score"] == pytest.approx(across)
    assert (check["kind"], check["group"]) == ("split", ["000003", "000004"])
    assert (check["members"], check["same"], check["reused"]) == (["000002", "000003"], True, True)
    assert models.shown == [("000002", "000003")]
    assert list(graph.graph) == ["s0_home"]


@pytest.mark.usefixtures("model_environment", "opened_clients")
def test_the_audit_reports_every_member_check_and_saves_the_graph_it_changed(
    tmp_path: Path,
    graph_path: Path,
    models: FakeModels,
    monkeypatch: pytest.MonkeyPatch,
    write_member_graph: Callable[..., None],
) -> None:
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
