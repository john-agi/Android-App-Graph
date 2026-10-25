"""Graph manager for Android-App-Graph exploration graphs.

A node represents a unique app state defined by:
  - Level 1: App package (as the device reports it)
  - Level 2: Its members, the screenshots placed in it, each with the description
    the VLM gave its screen
  - Level 3: State schema — a dict of parameter keys, each mapped to the set of
    observed values across all visits.  E.g. {"sort_order": ["price_low_to_high",
    "rating"], "search_query": ["keyboard", null]}

A node's ID is ``s<n>``, numbered in the order nodes are made, and never changes.

An edge represents a device action that transitions between states.

State matching compares a new screenshot with the members of the nodes of its app
package, cheapest evidence first: an exact pixel match outside the status bar, then
the node the caller expects, then a short list of nodes ranked by the image and the
description similarity of their members, each checked by the verifier against its
member closest to the screen.
Different Android activities that display the same screen are merged into a single
node; the node stores all observed activities. A black screenshot, all a secure window
captures, shows nothing to compare, so it is matched by its activity instead.

Every time a node is visited (whether new or existing), the VLM detail JSON is
used to update the node's state schema — discovering new keys and accumulating
observed values.

A node holds no image. It lists the step-log observations assigned to it, in step
order, and each use of a node's image chooses a member by its own rule, reading that
member's image from the step log. The audit compares nodes on their members:
merge proposals from the model's text and from close member images and descriptions,
verified in rounds on their closest members with every answer that follows from
earlier ones deduced, and member groups whose closest pair the verifier tells apart,
split off or moved to the node screen matching confirms.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
from collections import Counter
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, NamedTuple

import networkx as nx
from openrouter import OpenRouter

from android_app_graph.checkpoint import (
    APP_NAME_BYTES,
    CheckpointReport,
    IdentityBatch,
    IdentityNode,
    MergeHint,
    ProposalReply,
    clipped_text,
    evidence_key,
    identity_batches,
)
from android_app_graph.graph_files import (
    PER_ACTION_LISTS,
    append_json_line,
    edge_problems,
    graph_steps_dir,
    identify_file_path,
    load_embeddings,
    member_problems,
    read_json_lines,
    require_known_edge_endpoints,
    save_embeddings,
    text_embeddings_path,
    unloadable_graph,
    write_bytes_atomically,
    write_json_atomically,
)
from android_app_graph.member_groups import (
    IMAGE_MERGE_FLOOR,
    TEXT_MERGE_FLOOR,
    MemberCheck,
    MemberPair,
    closest_pair,
    plan_member_checks,
    unit_vector,
)
from android_app_graph.merge_rounds import NodePair, PairRounds
from android_app_graph.modal_context import MatchInput, ModalContext, Surface
from android_app_graph.payloads import (
    as_int_list,
    as_object_dict,
    as_str,
    as_str_dict,
)
from android_app_graph.step_log import (
    Observation,
    TransitionRecord,
    load_image_embeddings,
    observation_ownership,
    read_observation_record,
    recorded_observation,
    save_image_embeddings,
    step_order,
)
from android_app_graph.utils import (
    DEFAULT_AUDIT_TIMEOUT,
    DEFAULT_MATCHING,
    MatchingSettings,
    ModelSettings,
)
from android_app_graph.utils.vlm_utils import (
    DescribedScreenshot,
    DescribeRequest,
    audit_graph,
    audit_merge_nodes,
    call_concurrently,
    cosine_similarity,
    describe_page_and_state,
    get_embedding,
    get_image_embedding,
    node_merge_context,
    node_merge_prompt,
    node_merge_request_bound,
    node_verification_context,
    normalize_edge,
    score_by_cosine,
    token_tracker,
    verify_same_node,
)

logger = logging.getLogger(__name__)

NORMALIZE_EVERY_N_VISITS = 10
_FIXED_INSTRUCTION_MAX_WORDS = 6
# The ext_ prefix keeps it out of start-node and exploration-target selection, like ext_<package>.
EXTERNAL_WEB_NODE = "ext_web"
# A redirecting link or a browser's own history can keep an outside page up after one back press.
EXTERNAL_WEB_BACK_PRESSES = 3
# A longer setup is an excursion, not a recipe for the state an action needs.
MAX_SETUP_ACTIONS = 8
_DESCRIBE_CONTEXT_ENTRIES = 16
_DESCRIBE_CONTEXT_BYTES = 2048


class _SamePackageNodes(NamedTuple):
    """A package's in-app nodes that have members to compare with, and the state keys of
    all its nodes."""

    node_ids: list[str]
    keys: list[str]


class PathToStart(NamedTuple):
    """Nodes from which the start node cannot be reached, split by origin."""

    start_node: str | None
    in_app: list[str]
    external: list[str]


class PathStep(NamedTuple):
    """One step of a path ``find_path`` returns: the node it reaches, the recorded action
    that reaches it, and that action's setup, the device actions sent before it on the
    node the step leaves. The first step is the path's start, with the action ``{}`` and
    no setup."""

    node: str
    action: dict[str, Any] | list[dict[str, Any]]
    setup: tuple[dict[str, Any], ...]


@dataclass(frozen=True, kw_only=True)
class EdgeDetails:
    """What one traversal records next to its action in the edge's aligned lists.

    Attributes:
        instruction: The natural language instruction that produced the action.
        target_observation: Description of the screen state after performing
            the action. Edge normalization templates it next to the instruction.
        schema_delta: For self-loop edges (source == target), records which
            schema keys changed and their before/after values.
            Format: {"key": {"before": old_val, "after": new_val}, ...}
        num_steps: Number of action steps the edge takes to traverse.
            If None, defaults to len(action) for lists or 1 for single actions.
        setup: The device actions sent on the source node before the action, since
            the traversal last arrived there from elsewhere, whose effect on the node's
            state is in force when the action is sent, as ``SetupTrail`` reads them;
            None when they are unknown.
        open_menu: The menu open over the source node when the action is sent, as
            ``open_menu_in`` reads it from the node's state; None when none is.
    """

    instruction: str | None = None
    target_observation: str | None = None
    schema_delta: dict[str, Any] | None = None
    num_steps: int | None = None
    setup: Sequence[dict[str, Any]] | None = None
    open_menu: str | None = None


def open_menu_in(state: Mapping[str, object]) -> str | None:
    """Return the menu a node's state records open over the screen, its ``open_menu``,
    or None when none is."""
    menu = state.get("open_menu")
    return menu if isinstance(menu, str) and menu else None


class _ReplayOption(NamedTuple):
    """An action of an edge a replay can send: its index on the edge, the device actions
    sent on the edge's source before it, and how many device actions that makes with
    the action's own steps."""

    cost: int
    index: int
    setup: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class _NodeState:
    """A state the trail found a node in, and the index into the trail's actions where
    the actions sent in that state start."""

    start: int
    state: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _Arrival:
    """A node the trail arrived at and the states it found the node in since, in order;
    ``None`` when the trail began on the node without knowing how the device came there."""

    node: str
    states: list[_NodeState] | None


def _same_state(first: Mapping[str, object], second: Mapping[str, object]) -> bool:
    """Return whether two states of a node agree on every key; a key one of them lacks
    counts as null, as in a self-loop's schema delta."""
    return all(first.get(key) == second.get(key) for key in first.keys() | second.keys())


def _spans_in_force(states: Sequence[_NodeState]) -> list[tuple[int, int]]:
    """Return the spans ``(start, end)`` of the trail's actions that took a node from the
    first of ``states`` to the last, leaving out what was undone: the actions from a state
    to a later return to the same state."""
    spans: list[tuple[int, int]] = []
    path = [states[0].state]
    for before, after in itertools.pairwise(states):
        repeated = next(
            (index for index, state in enumerate(path) if _same_state(state, after.state)), None
        )
        if repeated is None:
            spans.append((before.start, after.start))
            path.append(after.state)
        else:
            del spans[repeated:]
            del path[repeated + 1 :]
    return spans


class SetupTrail:
    """The device actions an exploration sent since it began on a screen, and the states
    they left each node in, from which it reads each action's setup: the actions sent on
    the action's source node, since the trail last arrived there from elsewhere, whose
    effect on the node's state is in force when the action is sent.

    The trail keeps the nodes it arrived at in order. Arriving at one of them again, such
    as a form after a dialog that set one of its fields, ends a round trip: the nodes
    after it are left behind, and the round trip's actions join its setup, as a
    self-loop's do. Arriving at any other node starts its setup empty.

    A setup leaves out what was undone: when the node comes back to a state it was in
    since the trail arrived there, the actions in between are left out, such as a row of
    a list opened and left by Back, a menu opened and closed again, or a field typed and
    cleared. States compare key by key, as a self-loop's schema delta does. Node IDs are
    compared through ``resolve_node_id``, so a node merged while the trail holds it still
    matches.
    """

    def __init__(self, resolve_node_id: Callable[[str], str]) -> None:
        self._resolve_node_id = resolve_node_id
        self._actions: list[dict[str, Any]] = []
        self._arrivals: list[_Arrival] = []

    def begin(self, node: str, arrival_state: Mapping[str, object] | None) -> None:
        """Begin the trail again on ``node``, forgetting every action before.

        ``arrival_state`` is the node's state when the device has just arrived there from
        elsewhere, as when a replay reaches the node it replays to; the setup on it starts
        empty. ``None`` says how the device came to be on ``node`` is unknown: the setup of
        every action sent on it is unknown until the trail arrives there from elsewhere
        again.
        """
        self._actions = []
        states = None if arrival_state is None else [_NodeState(0, dict(arrival_state))]
        self._arrivals = [_Arrival(node, states)]

    def setup(self, node: str) -> list[dict[str, Any]] | None:
        """Return the setup of an action sent on ``node``, the node the trail is on, or
        None when it is unknown."""
        position = self._position(node)
        if position is None:
            return None
        states = self._arrivals[position].states
        if states is None:
            return None
        return [
            action for start, end in _spans_in_force(states) for action in self._actions[start:end]
        ]

    def step(
        self, actions: Sequence[dict[str, Any]], reached: str, state: Mapping[str, object]
    ) -> None:
        """Add ``actions``, sent on the node the trail is on, and ``reached``, the node
        they led to, in ``state``."""
        self._actions.extend(actions)
        reached_state = _NodeState(len(self._actions), dict(state))
        position = self._position(reached)
        if position is None:
            self._arrivals.append(_Arrival(reached, [reached_state]))
            return
        del self._arrivals[position + 1 :]
        states = self._arrivals[position].states
        if states is not None:
            states.append(reached_state)

    def _position(self, node: str) -> int | None:
        resolved = self._resolve_node_id(node)
        for position in reversed(range(len(self._arrivals))):
            if self._resolve_node_id(self._arrivals[position].node) == resolved:
                return position
        return None


@dataclass(frozen=True)
class _ScreenDescription:
    description: str
    detail_snapshot: dict[str, object]
    elements: list[dict[str, str]]


@dataclass(frozen=True)
class _EdgeToNormalize:
    """An edge's normalize_edge inputs, copied so that a worker thread never reads the graph,
    and the hash of the recorded instructions they cover."""

    source: str
    target: str
    instructions: list[str]
    target_observations: list[str] | None
    instructions_hash: str


class _Described(NamedTuple):
    """What the describe call said about a screen, as its line in the graph's identify
    file keeps it."""

    page_description: str
    state: dict[str, object]
    elements: list[dict[str, str]]
    is_external_web: bool
    surface: Surface = Surface()

    def as_record(self) -> dict[str, object]:
        return {
            "page_description": self.page_description,
            "state": self.state,
            "elements": self.elements,
            "is_external_web": self.is_external_web,
            "surface": self.surface.as_record(),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> _Described | None:
        """Return the ``describe`` field of a screenshot's line in the identify file, or
        ``None`` when it has none or it is malformed."""
        describe = as_object_dict(record.get("describe"))
        description = describe.get("page_description")
        state = describe.get("state")
        elements = describe.get("elements")
        if not isinstance(description, str) or not isinstance(state, dict):
            return None
        if not isinstance(elements, list):
            return None
        return cls(
            description,
            as_object_dict(state),
            [
                {
                    key: value
                    for key, value in as_object_dict(element).items()
                    if isinstance(value, str)
                }
                for element in elements
            ],
            describe.get("is_external_web") is True,
            Surface.from_record(describe.get("surface")),
        )


def _black_screen_described(activity: str) -> _Described:
    """Return the description a black screenshot of ``activity`` takes in place of the
    describe call: the activity's class name, with no state and no elements."""
    return _Described(
        page_description=f"secure screen: {activity.rpartition('.')[2]}",
        state={},
        elements=[],
        is_external_web=False,
    )


class _StoredScreen(NamedTuple):
    """A member's pixel hash and description: the describe output an exact match copies
    from it, or, for a black member, the one made from its ``black_activity``, by which
    a black screen joins its node."""

    pixel_hash: str
    described: _Described
    black_activity: str | None = None
    modal_context: ModalContext = ModalContext()


class ClosestMember(NamedTuple):
    """A node's member most similar to a screen by image, and that cosine."""

    observation_id: str
    score: float


@dataclass(frozen=True, slots=True)
class ScreenQuery:
    """A screen as the short list's sources rank nodes for it."""

    image_vector: list[float]
    description_embedding: list[float]


type SourceScores = Callable[[GraphManager, ScreenQuery, Sequence[str]], list[tuple[str, float]]]


@dataclass(frozen=True, slots=True)
class ShortListSource:
    """One named ranking of the candidate nodes, of which the short list takes the
    ``size`` best."""

    name: str
    size: int
    scores: SourceScores


@dataclass(slots=True)
class Candidate:
    """A node on the short list, the best rank any source gave it, the sources that
    proposed it with their scores, and its member closest to the screen."""

    node_id: str
    best_rank: int
    closest_member: ClosestMember
    sources: dict[str, float] = field(default_factory=dict)

    def as_record(self) -> dict[str, object]:
        return {
            "node": self.node_id,
            "sources": dict(self.sources),
            "closest_member": self.closest_member.observation_id,
            "image_score": self.closest_member.score,
        }


def _image_scores(
    graph: GraphManager, query: ScreenQuery, node_ids: Sequence[str]
) -> list[tuple[str, float]]:
    return [
        (node_id, graph.closest_member(node_id, query.image_vector).score) for node_id in node_ids
    ]


def _text_scores(
    graph: GraphManager, query: ScreenQuery, node_ids: Sequence[str]
) -> list[tuple[str, float]]:
    """Score each node by its member whose description vector is most similar to the
    screen's; a node whose members are all black has no score."""
    member_scores = score_by_cosine(
        query.description_embedding,
        graph.member_description_vectors(node_ids).items(),
        scope="identify_state description-embedding",
    )
    best: dict[str, float] = {}
    for (node_id, _), score in member_scores:
        best[node_id] = max(score, best.get(node_id, score))
    return list(best.items())


def short_list_sources(matching: MatchingSettings) -> tuple[ShortListSource, ...]:
    """Return the short list's sources, each proposing as many nodes as ``matching``
    gives it."""
    return (
        ShortListSource("image", matching.image_candidates, _image_scores),
        ShortListSource("text", matching.text_candidates, _text_scores),
    )


def _short_list_order(candidate: Candidate) -> tuple[int, float]:
    return candidate.best_rank, -candidate.closest_member.score


class MemberVerdict(NamedTuple):
    """The verifier's answer on whether two member observations show one screen;
    ``reused`` when an earlier check of the same pair in the run gave it, and not
    ``answered``, with ``same`` false, when the verifier gave no answer."""

    same: bool
    reason: str
    reused: bool = False
    answered: bool = True


type MemberVerdicts = dict[frozenset[str], MemberVerdict]


class MemberQuestion(NamedTuple):
    """Whether member observations ``existing`` and ``new`` show one screen."""

    existing: str
    new: str


class _MemberReply(NamedTuple):
    verdict: MemberVerdict
    evidence: str | None


def _shown_member_key(member: str, shown: DescribedScreenshot) -> str:
    return evidence_key(
        [
            member,
            shown.description,
            shown.screenshot_b64,
            shown.system_feedback,
            shown.opening_context,
            shown.opening_images,
            shown.modal_origin,
        ]
    )


class ShownMembers(NamedTuple):
    """The member of each node a merge check shows the verifier, and the cosine of
    their images."""

    existing: str
    new: str
    score: float


type MergePairs = Callable[[GraphManager], list[tuple[str, str, dict[str, object]]]]


@dataclass(frozen=True, slots=True)
class MergeSource:
    """One named way of finding node pairs that may be one screen; ``pairs`` returns
    each pair with what the source found."""

    name: str
    pairs: MergePairs


def _found(
    pairs: Iterable[tuple[str, str, MemberPair]],
) -> list[tuple[str, str, dict[str, object]]]:
    return [
        (node_a, node_b, {"members": [pair.first, pair.second], "score": pair.score})
        for node_a, node_b, pair in pairs
    ]


def _close_member_images(graph: GraphManager) -> list[tuple[str, str, dict[str, object]]]:
    return _found(graph.close_member_images())


def _close_member_descriptions(graph: GraphManager) -> list[tuple[str, str, dict[str, object]]]:
    return _found(graph.close_member_descriptions())


MERGE_SOURCES = (
    MergeSource("image", _close_member_images),
    MergeSource("text", _close_member_descriptions),
)


@dataclass(slots=True)
class MergeProposal:
    """A node pair that may be one screen, and the finding of every source that
    proposed it, each naming its ``source``."""

    node_a: str
    node_b: str
    findings: list[dict[str, object]] = field(default_factory=list)

    @property
    def sources(self) -> list[str]:
        return list(dict.fromkeys(as_str(finding.get("source"), "") for finding in self.findings))


class Merge(NamedTuple):
    """A cluster of nodes merged into its lowest-numbered node, ``kept``."""

    kept: str
    removed: list[str]


@dataclass(frozen=True, slots=True)
class MergeReport:
    """What verifying merge proposals found: each proposal's result in proposal order,
    the rounds the questions took, the pairs asked and deduced, and the merges made."""

    pairs: list[dict[str, object]]
    rounds: int
    asked: int
    deduced: int
    merges: list[Merge]

    @property
    def merged_count(self) -> int:
        """The number of nodes merged into another."""
        return sum(len(merge.removed) for merge in self.merges)

    def as_record(self) -> dict[str, object]:
        return {
            "rounds": self.rounds,
            "asked": self.asked,
            "deduced": self.deduced,
            "merges": [merge._asdict() for merge in self.merges],
            "pairs": self.pairs,
        }


class _MergeCheck(NamedTuple):
    """A proposed pair the rounds decide: its two nodes, the question it asks the
    verifier, the image cosine of the members that question shows, and the pair's
    result in the report."""

    nodes: NodePair
    question: MemberQuestion
    score: float
    result: dict[str, object]


def _most_alike_first(check: _MergeCheck) -> float:
    return -check.score


@dataclass(slots=True)
class _Decision:
    """What decided a screen's node, as the ``identify`` field of its line in the graph's
    identify file keeps it: the screenshot the caller expected the screen to show the
    screen of, and the node that screenshot belonged to then."""

    expected_observation: str | None
    expected_node: str | None
    stage: Literal["black", "exact", "expected", "short_list", "new", "external_web"] = "new"
    exact_match: str | None = None
    candidates: list[dict[str, object]] = field(default_factory=list)
    verifications: list[dict[str, object]] = field(default_factory=list)
    refinement: dict[str, object] | None = None

    def as_record(self, node_id: str) -> dict[str, object]:
        return {
            "stage": self.stage,
            "expected_observation": self.expected_observation,
            "expected_node": self.expected_node,
            "exact_match": self.exact_match,
            "candidates": self.candidates,
            "verifications": self.verifications,
            "refinement": self.refinement,
            "node": node_id,
        }


@dataclass(slots=True)
class _Identification:
    """One described screen on its way through ``identify_state``: its description and
    that description's vector, which the step's first rejection refines, and the
    decision its record gets.

    A member the audit identifies again from its stored describe output carries the
    run's ``verdicts``, which its verifier calls reuse and add to; it refines no
    description.
    """

    observation: Observation
    image_vector: list[float]
    screen: _ScreenDescription
    text_vector: list[float]
    decision: _Decision
    rejected: bool = False
    verdicts: MemberVerdicts | None = None


def _node_id(value: object) -> str:
    """Return a graph node ID as ``str``.

    ``networkx`` ships no annotations, so every node ID read back out of the
    graph is ``Unknown``. The graph only ever stores ``str`` IDs, and this is
    the one place that says so.
    """
    return value if isinstance(value, str) else str(value)


def _merge_into_schema(
    schema: dict[str, list[Any]], detail_snapshot: dict[str, Any]
) -> dict[str, list[Any]]:
    """Merge a detail snapshot into an existing state schema.

    For each key in the snapshot:
      - If the key is new, add it with a single-element list.
      - If the key exists and the value is new, append it.

    Returns the updated schema (mutated in place and also returned).
    """
    for key, value in detail_snapshot.items():
        if key not in schema:
            schema[key] = [value]
            logger.debug("Schema: new key '%s' with value %s", key, value)
        elif value not in schema[key]:
            schema[key].append(value)
            logger.debug("Schema: key '%s' got new value %s", key, value)
    return schema


def _merge_elements(node_data: dict[str, Any], new_elements: list[dict[str, str]]) -> None:
    """Merge newly observed elements into a node's existing element list.

    New elements whose description doesn't already exist are appended with
    explored=False.  Existing elements are left untouched (preserving their
    explored status).
    """
    existing = node_data.get("interactable_elements", [])
    existing_descs = {e.get("description", "").lower() for e in existing}
    for elem in new_elements:
        desc = elem.get("description", "")
        if desc.lower() not in existing_descs:
            existing.append(
                {
                    "description": desc,
                    "position": elem.get("position", ""),
                    "explored": False,
                }
            )
            existing_descs.add(desc.lower())
    node_data["interactable_elements"] = existing


def _merge_activities(keep_data: dict[str, Any], remove_data: dict[str, Any]) -> None:
    keep_activities = keep_data.get("activities", [keep_data.get("activity", "")])
    remove_activities = remove_data.get("activities", [remove_data.get("activity", "")])
    for act in remove_activities:
        if act and act not in keep_activities:
            keep_activities.append(act)
    keep_data["activities"] = keep_activities


def _merge_schema_values(keep_data: dict[str, Any], remove_data: dict[str, Any]) -> None:
    """Union *remove_data*'s state schema into *keep_data*'s.

    A schema maps each key to a list of observed values, where a detail snapshot maps
    it to one value, so ``_merge_into_schema`` would append the whole list as one value.
    """
    keep_schema = keep_data.get("state_schema", {})
    remove_schema = remove_data.get("state_schema", {})
    for key, values in remove_schema.items():
        if key not in keep_schema:
            keep_schema[key] = list(values)
        else:
            for v in values:
                if v not in keep_schema[key]:
                    keep_schema[key].append(v)
    keep_data["state_schema"] = keep_schema


def _add_member(node_data: dict[str, Any], observation_id: str) -> None:
    """Append an observation to a node's members."""
    node_data.setdefault("observations", []).append(observation_id)


def _node_number(node_id: str) -> int | None:
    """Return the ``n`` of a node ID ``s<n>``, or ``None`` for any other ID."""
    number = node_id.removeprefix("s")
    return int(number) if number != node_id and number.isdecimal() else None


def _merge_node_attributes(keep_data: dict[str, Any], remove_data: dict[str, Any]) -> None:
    """Fold a removed node's visits, members, activities, state schema and elements into
    the kept one."""
    keep_data["visit_count"] = keep_data.get("visit_count", 0) + remove_data.get("visit_count", 0)
    members = {*keep_data.get("observations", []), *remove_data.get("observations", [])}
    keep_data["observations"] = sorted(members, key=step_order)
    _merge_activities(keep_data, remove_data)
    _merge_schema_values(keep_data, remove_data)
    _merge_elements(keep_data, remove_data.get("interactable_elements", []))


def schema_delta(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, dict[str, Any]] | None:
    """Return what changed between two detail snapshots of a node, as a self-loop's
    ``schema_deltas`` entry records it, ``{"key": {"before": old, "after": new}}``, or
    ``None`` when nothing changed."""
    delta = {
        key: {"before": before.get(key), "after": after.get(key)}
        for key in before.keys() | after.keys()
        if before.get(key) != after.get(key)
    }
    return delta or None


def target_observation(page_description: str, detail_snapshot: Mapping[str, Any]) -> str:
    """Return ``"<description> (key=value, ...)"`` for the details that have a value, or
    the bare description when none has."""
    details = ", ".join(f"{k}={v}" for k, v in detail_snapshot.items() if v is not None)
    return f"{page_description} ({details})" if details else page_description


def _default_num_steps(action: object) -> int:
    return len(action) if isinstance(action, list) else 1


def _aligned_lists(edge_data: dict[str, Any]) -> dict[str, list[Any]]:
    """Readers pair entry i of every per-action list with actions[i] and skip placeholders."""
    return {key: list(edge_data.get(key, [])) for key in ("actions", *PER_ACTION_LISTS)}


def _same_action(entry: object, action: object) -> bool:
    """Whether an edge's action entry is a transition record's action; an edge keeps
    a single action as a one-action sequence."""
    return entry in (action, [action])


def _without_entries(
    aligned: dict[str, list[Any]], indices: Collection[int]
) -> dict[str, list[Any]]:
    return {
        key: [value for index, value in enumerate(values) if index not in indices]
        for key, values in aligned.items()
    }


def _setup_actions(entry: object) -> tuple[dict[str, Any], ...]:
    """Return an action's entry of ``setups`` as its device actions; an entry that is not
    a list of actions, such as the ``None`` of an unknown setup, gives none."""
    if not isinstance(entry, list) or not all(isinstance(action, dict) for action in entry):
        return ()
    return tuple(as_str_dict(action) for action in entry)


def _recorded_setup(
    setup: Sequence[dict[str, Any]] | None, source: str, target: str
) -> list[dict[str, Any]] | None:
    """Return ``setup`` as the edge from ``source`` to ``target`` records it: one longer
    than ``MAX_SETUP_ACTIONS`` is logged as dropped and recorded as unknown."""
    if setup is None:
        return None
    if len(setup) > MAX_SETUP_ACTIONS:
        logger.info(
            "Dropped the setup of an action on %s -> %s: %d actions, more than %d",
            source,
            target,
            len(setup),
            MAX_SETUP_ACTIONS,
        )
        return None
    return list(setup)


def _count_repeat(
    edge_data: dict[str, Any],
    action: dict[str, Any] | list[dict[str, Any]],
    setup: list[dict[str, Any]] | None,
) -> bool:
    """Count one more visit when the edge already holds ``action``, and return whether it
    does. A known ``setup`` replaces the action's recorded one when that is unknown or
    longer."""
    for index, recorded_action in enumerate(edge_data.get("actions", [])):
        if recorded_action == action:
            edge_data["visit_count"] = edge_data.get("visit_count", 1) + 1
            setups = edge_data["setups"]
            recorded = setups[index]
            if setup is not None and (not isinstance(recorded, list) or len(setup) < len(recorded)):
                setups[index] = setup
            return True
    return False


def _device_actions(action: object) -> tuple[dict[str, Any], ...]:
    """Return the device actions of an edge's action entry, one action or a list of them."""
    return tuple(as_str_dict(step) for step in (action if isinstance(action, list) else [action]))


def _menu_openers(loop: dict[str, Any] | None) -> dict[str, tuple[dict[str, Any], ...]]:
    """Return, by menu, the fewest device actions that open it on a node whose self-loop
    is ``loop``: a recorded self-loop action whose schema delta records the menu opening,
    after its setup. An action whose setup is unknown opens the menu only when it was
    sent with no menu open, the state a replay arrives in."""
    if loop is None:
        return {}
    aligned = _aligned_lists(loop)
    openers: dict[str, tuple[dict[str, Any], ...]] = {}
    for action, delta, setup, menu in zip(
        aligned["actions"],
        aligned["schema_deltas"],
        aligned["setups"],
        aligned["open_menus"],
        strict=True,
    ):
        opened = as_object_dict(as_object_dict(delta).get("open_menu")).get("after")
        replayable = isinstance(setup, list) or menu is None
        if not isinstance(opened, str) or not replayable:
            continue
        sent = (*_setup_actions(setup), *_device_actions(action))
        if opened not in openers or len(sent) < len(openers[opened]):
            openers[opened] = sent
    return openers


def _replay_setup(
    setup: object, menu: object, openers: Mapping[str, tuple[dict[str, Any], ...]]
) -> tuple[dict[str, Any], ...] | None:
    """Return the device actions a replay sends before an action whose ``setups`` and
    ``open_menus`` entries are ``setup`` and ``menu``: its recorded setup, or, when that
    is unknown, the opening of the menu it was sent in from ``openers``; None when no
    recorded action opens that menu."""
    if isinstance(setup, list):
        return _setup_actions(setup)
    if not isinstance(menu, str):
        return ()
    return openers.get(menu)


def _instructions_hash(instructions: list[str]) -> str:
    """SHA-256 of *instructions* as a JSON list, which keeps every instruction apart."""
    return hashlib.sha256(json.dumps(instructions).encode()).hexdigest()


def _drop_templates(edge_data: dict[str, Any]) -> None:
    """Drop the edge's templates and the record of what they covered, so that the next
    normalization rebuilds them."""
    edge_data["instruction_templates"] = []
    edge_data.pop("normalized_instructions_hash", None)


def _member_observations(node: Mapping[str, Any]) -> list[str]:
    """Return a saved node's member observation IDs, or ``[]`` when it has none or
    they are not a list of strings."""
    observations = node.get("observations")
    if not isinstance(observations, list):
        return []
    members: list[str] = [o for o in observations if isinstance(o, str)]
    return members if len(members) == len(observations) else []


def _saved_in_app_members(nodes: Iterable[Mapping[str, Any]]) -> list[str]:
    """Return the members of the saved ``nodes`` that are not external, in node order."""
    return [
        member
        for node in nodes
        if not (node["id"].startswith("ext_") or node.get("is_external"))
        for member in _member_observations(node)
    ]


def _saved_matching(graph: dict[str, Any]) -> MatchingSettings | None:
    """Return the matching settings a saved graph names, or ``None`` when it names none or
    one of them is not a whole number."""
    saved = as_object_dict(graph.get("matching"))
    values = [saved.get(setting) for setting in MatchingSettings._fields]
    counts = [value for value in values if isinstance(value, int) and not isinstance(value, bool)]
    return MatchingSettings(*counts) if len(counts) == len(values) else None


def _require_saved_fields(graph: dict[str, Any], path: Path, steps_dir: Path) -> None:
    """Raise ``ValueError`` naming ``path`` and every saved node that does not list its
    member observations or that has no integer ``replay_misses`` count, every member
    without an image in ``steps_dir``, every edge without actions or whose per-action
    lists do not hold one entry per action, and a graph that names no matching settings.
    """
    problems = member_problems(graph["nodes"], steps_dir)
    without_replay_misses = [
        node["id"] for node in graph["nodes"] if not isinstance(node.get("replay_misses"), int)
    ]
    if without_replay_misses:
        problems.append(f"node(s) without a replay_misses count: {without_replay_misses}")
    problems.extend(edge_problems(graph["edges"]))
    if _saved_matching(graph) is None:
        settings = ", ".join(MatchingSettings._fields)
        problems.append(f"no matching settings, a whole number for each of {settings}")
    if problems:
        raise unloadable_graph(path, problems)
    for node in graph["nodes"]:
        for member in node["observations"]:
            record = read_observation_record(steps_dir, member)
            if record is None:
                msg = f"member {member!r} has no observation record"
                raise unloadable_graph(path, [msg])
            ownership = observation_ownership(record)
            if ownership.error is not None or (
                ownership.surface is not None and not str(node["id"]).startswith("ext_")
            ):
                msg = f"member {member!r} has unsupported ownership for node {node['id']!r}"
                raise unloadable_graph(path, [msg])


def _raise_for_members_without(vector: str, without: Mapping[str, Sequence[str]]) -> None:
    """Raise ``ValueError`` naming each node of ``without`` and its members that have no
    ``vector``, such as ``"an image vector"``, when ``without`` names any."""
    if without:
        named = ". ".join(
            f"Node {node_id!r} has members without {vector}: {', '.join(members)}"
            for node_id, members in without.items()
        )
        raise ValueError(named)


def _stored_screens(
    nodes: list[dict[str, Any]], path: Path, steps_dir: Path, identify_file: Path
) -> dict[str, _StoredScreen]:
    """Read the pixel hash of every member of the saved in-app ``nodes`` from its record
    in ``steps_dir``, and its description from its line in ``identify_file``: its describe
    output, or, for a black screenshot, the description made from its activity.

    A member whose record holds no pixel hash, or that has no describe output and is not
    of a black screenshot, raises ``ValueError`` naming ``path`` and every such member.
    """
    lines = _identify_lines(identify_file)
    stored: dict[str, _StoredScreen] = {}
    undescribed: list[str] = []
    for member in _saved_in_app_members(nodes):
        record = read_observation_record(steps_dir, member) or {}
        black_activity = as_str(record.get("activity"), "") if record.get("black") is True else None
        described = (
            _Described.from_record(lines.get(member, {}))
            if black_activity is None
            else _black_screen_described(black_activity)
        )
        pixel_hash = as_str(record.get("pixel_hash"), "")
        if described is None or not pixel_hash:
            undescribed.append(member)
        else:
            stored[member] = _StoredScreen(
                pixel_hash,
                described,
                black_activity,
                ModalContext.from_record(lines.get(member, {}).get("modal_context"))
                if black_activity is None
                else ModalContext(),
            )
    if undescribed:
        problem = (
            f"member(s) whose record in {steps_dir} holds no pixel hash, or that have no "
            f"describe output in {identify_file} and are not of a black screenshot: "
            f"{undescribed}"
        )
        raise unloadable_graph(path, [problem])
    return stored


def _identify_lines(identify_file: Path) -> dict[str, dict[str, object]]:
    """Return the lines of ``identify_file`` by the observation each names."""
    return {as_str(line.get("observation"), ""): line for line in read_json_lines(identify_file)}


def _line_description(lines: Mapping[str, Mapping[str, object]], observation_id: str) -> str:
    """Return the description the identify file's line of ``observation_id`` gives, or
    ``""`` when it gives none."""
    line = lines.get(observation_id, {})
    describe = as_object_dict(line.get("describe"))
    copied_from = describe.get("copied_from")
    if isinstance(copied_from, str):
        return _line_description(lines, copied_from)
    refinement = as_object_dict(as_object_dict(line.get("identify")).get("refinement"))
    refined = as_str(refinement.get("new_description"), "")
    return refined or as_str(describe.get("page_description"), "")


def read_member_descriptions(
    graph_path: Path, nodes: Iterable[Mapping[str, Any]]
) -> dict[str, str]:
    """Return the description of every member of the saved in-app ``nodes`` of the graph
    at ``graph_path``, by observation ID, as the graph's identify file records it: the
    description the step's first rejection refined the screen to
    (``identify.refinement.new_description``), or else its ``describe`` output's
    ``page_description``. An exact match's is that of the member it copied
    (``describe.copied_from``), and a black screenshot's, which its record in the step
    log marks, the one made from its activity.

    A member without one raises ``ValueError`` naming ``graph_path`` and every such
    member.
    """
    steps_dir = graph_steps_dir(graph_path)
    identify_file = identify_file_path(graph_path)
    lines = _identify_lines(identify_file)
    descriptions: dict[str, str] = {}
    for member in _saved_in_app_members(nodes):
        record = read_observation_record(steps_dir, member) or {}
        descriptions[member] = (
            _black_screen_described(as_str(record.get("activity"), "")).page_description
            if record.get("black") is True
            else _line_description(lines, member)
        )
    undescribed = [member for member, description in descriptions.items() if not description]
    if undescribed:
        problem = f"member(s) whose line in {identify_file} holds no description: {undescribed}"
        raise unloadable_graph(graph_path, [problem])
    return descriptions


def distinct_descriptions(descriptions: Iterable[str]) -> str:
    """Return each distinct one of ``descriptions``, a node's member descriptions in step
    order, quoted and followed by the number of members that have it, most first, then by
    earliest member: ``"Chrome app permissions overview" (14), "App permissions settings"
    (2)``."""
    return ", ".join(
        f'"{description}" ({count})' for description, count in Counter(descriptions).most_common()
    )


class GraphManager:
    """Manages a directed graph of app UI states and transitions."""

    def __init__(
        self,
        client: OpenRouter | None = None,
        page_detail: ModelSettings | None = None,
        embedding: ModelSettings | None = None,
        image_embedding: ModelSettings | None = None,
        audit_timeout: float = DEFAULT_AUDIT_TIMEOUT,
    ) -> None:
        self.graph = nx.DiGraph()
        self.client = client
        self.page_detail = page_detail
        self.embedding = embedding
        self.image_embedding = image_embedding
        self.audit_timeout = audit_timeout
        # The settings screen matching places screenshots by: a new graph takes its owner's,
        # and load_graph requires the file's to equal them.
        self.matching = DEFAULT_MATCHING
        # The step log the nodes' observations are in, and the file each decision that
        # placed one is appended to: load_graph sets the ones beside the file it loads, and
        # the owner of a new graph sets both before identify_state.
        self.steps_dir: Path | None = None
        self.identify_file: Path | None = None
        self._next_id = 0
        self._node_aliases: dict[str, str] = {}
        self._image_vectors: dict[str, list[float]] = {}
        self._unit_vectors: dict[str, list[float]] = {}
        # By observation ID, the vector of the description each described in-app screen
        # ended with: its describe output's, or the one its step's first rejection
        # refined it to.
        self._text_vectors: dict[str, list[float]] = {}
        # By observation ID, the description each screen identify_state settles ends with,
        # the one an exact match copies from its member and a black screen's included. A
        # loaded graph holds those of its in-app members, as read_member_descriptions
        # reads them back.
        self._descriptions: dict[str, str] = {}
        self._stored_screens: dict[str, _StoredScreen] = {}
        self._checkpoint_verdicts: MemberVerdicts = {}
        self._checkpoint_evidence: dict[frozenset[str], str] = {}
        self._checkpoint_proposals: dict[str, ProposalReply] = {}
        self.total_steps_completed = 0

    def _page_detail(self) -> tuple[OpenRouter, ModelSettings] | None:
        if self.client is None or self.page_detail is None:
            return None
        return self.client, self.page_detail

    def _require_page_detail(self) -> tuple[OpenRouter, ModelSettings]:
        page_detail = self._page_detail()
        if page_detail is None:
            msg = "GraphManager needs a client and page_detail settings for this operation"
            raise RuntimeError(msg)
        return page_detail

    def _require_embedding(self) -> tuple[OpenRouter, ModelSettings]:
        if self.client is None or self.embedding is None:
            msg = "GraphManager needs a client and embedding settings for this operation"
            raise RuntimeError(msg)
        return self.client, self.embedding

    def _require_image_embedding(self) -> tuple[OpenRouter, ModelSettings]:
        if self.client is None or self.image_embedding is None:
            msg = "GraphManager needs a client and image_embedding settings for this operation"
            raise RuntimeError(msg)
        return self.client, self.image_embedding

    def _require_steps_dir(self) -> Path:
        if self.steps_dir is None:
            msg = "GraphManager needs a steps_dir to read observations"
            raise RuntimeError(msg)
        return self.steps_dir

    def _require_identify_file(self) -> Path:
        if self.identify_file is None:
            msg = "GraphManager needs an identify_file to record what places each screenshot"
            raise RuntimeError(msg)
        return self.identify_file

    def _record_decision(
        self,
        observation: Observation,
        decision: _Decision,
        node_id: str,
        describe: Mapping[str, object] | None,
    ) -> None:
        """Append to the identify file what placed ``observation`` on ``node_id``: the
        describe output it took, when it took one, and the decision."""
        line: dict[str, object] = {"observation": observation.observation_id}
        if describe is not None:
            line["describe"] = dict(describe)
        stored = self._stored_screens.get(observation.observation_id)
        line["modal_context"] = stored.modal_context.as_record() if stored is not None else []
        line["identify"] = decision.as_record(node_id)
        append_json_line(self._require_identify_file(), line)

    def node_of(self, observation_id: str | None) -> str | None:
        """Return the node that lists ``observation_id`` among its members, or ``None``
        when none does or no observation is named."""
        if observation_id is None:
            return None
        return next(
            (
                _node_id(node_id)
                for node_id, data in self.graph.nodes(data=True)
                if observation_id in _member_observations(data)
            ),
            None,
        )

    def earliest_member(self, node_id: str | None) -> str | None:
        """Return the earliest member of ``node_id``, followed through merges, or
        ``None`` when no node is named, it is gone or it has no members."""
        if node_id is None:
            return None
        node_id = self.resolve_node_id(node_id)
        if node_id not in self.graph:
            return None
        members = _member_observations(self.graph.nodes[node_id])
        return members[0] if members else None

    def _embed(self, text: str) -> list[float]:
        client, embedding = self._require_embedding()
        return get_embedding(
            client,
            text,
            embedding.model,
            request_timeout=embedding.timeout("embedding"),
        )

    def _embed_image(self, screenshot_b64: str) -> list[float]:
        client, image_embedding = self._require_image_embedding()
        return get_image_embedding(
            client,
            screenshot_b64,
            image_embedding.model,
            request_timeout=image_embedding.timeout("image_embedding"),
        )

    def image_vector(self, observation_id: str) -> list[float] | None:
        """Return the image vector of an observation, or ``None`` when it has none."""
        return self._image_vectors.get(observation_id)

    def use_image_vectors(self, vectors: Mapping[str, list[float]]) -> None:
        """Take ``vectors``, image vectors by observation ID, as the screenshots' own, so
        screen matching embeds none of those screenshots again."""
        self._image_vectors.update(vectors)

    def text_vector(self, observation_id: str) -> list[float] | None:
        """Return the vector of an observation's description, or ``None`` when it has none."""
        return self._text_vectors.get(observation_id)

    def description(self, observation_id: str) -> str | None:
        """Return the description an observation's screen ended with, or ``None`` when it
        has none."""
        return self._descriptions.get(observation_id)

    def observation_state(self, observation: Observation) -> dict[str, object]:
        """Return the description's state for this observation or its exact current image.

        A fresh audit capture can repeat an identified image. An image with no unique
        exact member supplies no reported focus, even when its node has older state.
        """
        stored = self._stored_screens.get(observation.observation_id)
        if stored is None:
            same_package = self._same_package_nodes(observation.package)
            exact = self._exact_match(observation.pixel_hash, same_package.node_ids)
            if exact is not None:
                stored = self._stored_screens[exact[1]]
        return dict(stored.described.state) if stored is not None else {}

    def latest_description(self, node_id: str) -> str:
        """Return the description of ``node_id``'s latest member, the screen
        ``identify_state`` last placed in the node: on the node the device was last
        identified on, the screen the device shows.

        A latest member without a description raises ``ValueError`` naming it.
        """
        latest = _member_observations(self.graph.nodes[node_id])[-1]
        (description,) = self._member_descriptions(node_id, [latest])
        return description

    def node_descriptions(self, node_id: str) -> str:
        """Return the distinct descriptions of ``node_id``'s members with their counts, as
        ``distinct_descriptions`` lists them; an external node is named by its ID.

        A member without a description raises ``ValueError`` naming the node and every
        such member.
        """
        if self._is_external(node_id):
            return node_id
        members = _member_observations(self.graph.nodes[node_id])
        return distinct_descriptions(self._member_descriptions(node_id, members))

    def _member_descriptions(self, node_id: str, members: Sequence[str]) -> list[str]:
        """Return the descriptions of ``members`` of ``node_id``, in order.

        A member without one raises ``ValueError`` naming the node and every such member.
        """
        undescribed = [member for member in members if member not in self._descriptions]
        _raise_for_members_without("a description", {node_id: undescribed} if undescribed else {})
        return [self._descriptions[member] for member in members]

    def closest_member(self, node_id: str, image_vector: list[float]) -> ClosestMember:
        """Return the member of ``node_id`` whose image vector is most similar to
        ``image_vector``.

        A node without members raises ``ValueError``, and so does a node with members
        that have no image vector of ``image_vector``'s dimension, naming each of them.
        """
        observations = _member_observations(self.graph.nodes[node_id])
        if not observations:
            msg = f"Node {node_id!r} has no members to compare the screen with"
            raise ValueError(msg)
        unusable = [
            member
            for member in observations
            if len(self._image_vectors.get(member, [])) != len(image_vector)
        ]
        if unusable:
            msg = (
                f"Node {node_id!r} has members without an image vector of the screen's "
                f"{len(image_vector)} dimensions: {', '.join(unusable)}. Delete the step "
                "log's image_emb.json to embed every member again."
            )
            raise ValueError(msg)
        member, score = max(
            (
                (member, cosine_similarity(image_vector, self._image_vectors[member]))
                for member in observations
            ),
            key=lambda kv: kv[1],
        )
        return ClosestMember(member, score)

    def _member_units(self, node_ids: Iterable[str]) -> dict[str, list[float]]:
        """Return the image vector of every member of ``node_ids`` scaled to length 1,
        each scaled once per loaded graph.

        A member without an image vector raises ``ValueError`` naming its node and every
        such member.
        """
        units: dict[str, list[float]] = {}
        without: dict[str, list[str]] = {}
        for node_id in node_ids:
            for member in _member_observations(self.graph.nodes[node_id]):
                vector = self._image_vectors.get(member)
                if vector is None:
                    without.setdefault(node_id, []).append(member)
                    continue
                if member not in self._unit_vectors:
                    self._unit_vectors[member] = unit_vector(vector)
                units[member] = self._unit_vectors[member]
        _raise_for_members_without("an image vector", without)
        return units

    def member_description_vectors(
        self, node_ids: Iterable[str]
    ) -> dict[tuple[str, str], list[float]]:
        """Return the description vector of every member of ``node_ids`` that is not
        black, by node and member.

        A black member is never described, so it is left out. Any other member without
        a description vector raises ``ValueError`` naming its node and every such member.
        """
        vectors: dict[tuple[str, str], list[float]] = {}
        without: dict[str, list[str]] = {}
        for node_id in node_ids:
            for member in _member_observations(self.graph.nodes[node_id]):
                stored = self._stored_screens.get(member)
                if stored is not None and stored.black_activity is not None:
                    continue
                vector = self._text_vectors.get(member)
                if vector is None:
                    without.setdefault(node_id, []).append(member)
                else:
                    vectors[node_id, member] = vector
        _raise_for_members_without("a description vector", without)
        return vectors

    def _member_description_units(self, node_ids: Iterable[str]) -> dict[str, list[float]]:
        """Return the description vector of every member of ``node_ids`` that is not
        black, scaled to length 1, as ``member_description_vectors`` reads them."""
        return {
            member: unit_vector(vector)
            for (_, member), vector in self.member_description_vectors(node_ids).items()
        }

    def close_member_images(self) -> list[tuple[str, str, MemberPair]]:
        """Return every pair of in-app nodes of one package whose closest members by image
        reach ``IMAGE_MERGE_FLOOR``, with those members, in node order.

        A member without an image vector raises ``ValueError`` naming its node and every
        such member.
        """
        by_package = self._in_app_nodes_by_package()
        units = self._member_units(itertools.chain.from_iterable(by_package))
        return self._close_member_pairs(by_package, units, IMAGE_MERGE_FLOOR)

    def close_member_descriptions(self) -> list[tuple[str, str, MemberPair]]:
        """Return every pair of in-app nodes of one package whose closest members by
        description reach ``TEXT_MERGE_FLOOR``, with those members, in node order.

        A black member, never described, is left out, so a node whose members are all
        black is in no pair. Any other member without a description vector raises
        ``ValueError`` naming its node and every such member.
        """
        by_package = self._in_app_nodes_by_package()
        units = self._member_description_units(itertools.chain.from_iterable(by_package))
        return self._close_member_pairs(by_package, units, TEXT_MERGE_FLOOR)

    def _in_app_nodes_by_package(self) -> list[list[str]]:
        """Return the in-app nodes of each package, in node order."""
        by_package: dict[str, list[str]] = {}
        for raw_id, data in self.graph.nodes(data=True):
            node_id = _node_id(raw_id)
            if not self._is_external(node_id):
                by_package.setdefault(as_str(data.get("package"), ""), []).append(node_id)
        return [sorted(node_ids, key=self._node_numeric_order) for node_ids in by_package.values()]

    def _close_member_pairs(
        self, by_package: Iterable[Sequence[str]], units: Mapping[str, list[float]], floor: float
    ) -> list[tuple[str, str, MemberPair]]:
        """Return every pair of nodes of one package of ``by_package`` whose closest
        members with a vector in ``units`` reach ``floor``, with those members."""
        found: list[tuple[str, str, MemberPair]] = []
        for node_ids in by_package:
            for node_a, node_b in itertools.combinations(node_ids, 2):
                pair = closest_pair(
                    [m for m in _member_observations(self.graph.nodes[node_a]) if m in units],
                    [m for m in _member_observations(self.graph.nodes[node_b]) if m in units],
                    units,
                )
                if pair is not None and pair.score >= floor:
                    found.append((node_a, node_b, pair))
        return found

    def closest_members(self, existing: str, new: str) -> ShownMembers:
        """Return the member of node ``existing`` and the member of node ``new`` whose
        images are the most similar, and their cosine.

        A member of either node without an image vector raises ``ValueError`` naming its
        node and every such member, and so does a node without members.
        """
        pair = closest_pair(
            self._verification_members(existing),
            self._verification_members(new),
            self._member_units([existing, new]),
        )
        if pair is None:
            msg = f"Node {existing!r} or {new!r} has no members to compare"
            raise ValueError(msg)
        return ShownMembers(pair.first, pair.second, pair.score)

    def _in_app_members(self) -> Iterator[str]:
        for node_id, data in self.graph.nodes(data=True):
            if not self._is_external(node_id):
                yield from _member_observations(data)

    def embed_missing_members(self) -> int:
        """Embed the image of every member of an in-app node that has no image vector,
        save the vectors to the step log, and return how many were embedded.

        A member has none when the step log's ``image_emb.json`` is missing, holds no
        vector for it, or is tagged with another image-embedding model.
        """
        _, image_embedding = self._require_image_embedding()
        steps_dir = self._require_steps_dir()
        missing = [m for m in dict.fromkeys(self._in_app_members()) if m not in self._image_vectors]
        if not missing:
            return 0
        logger.info(
            "Embedding the images of %d member observation(s) without an image vector: "
            "one %s request each",
            len(missing),
            image_embedding.model,
        )
        usage_key = f"image_embedding:{image_embedding.model}"
        cost_before = token_tracker.summary().get(usage_key, {}).get("cost", 0.0)
        try:
            for member in missing:
                self._image_vectors[member] = self._embed_image(self._app_member_image(member))
        finally:
            save_image_embeddings(steps_dir, self._image_vectors, model=image_embedding.model)
        cost = token_tracker.summary().get(usage_key, {}).get("cost", 0.0) - cost_before
        logger.info("Embedded %d member image(s) for $%.6f", len(missing), cost)
        return len(missing)

    def _member_observation(self, member: str) -> Observation:
        steps = self._require_steps_dir()
        record = read_observation_record(steps, member)
        if record is None:
            msg = f"Observation {member} has no record"
            raise ValueError(msg)
        return recorded_observation(steps, record)

    def _app_member_image(self, member: str) -> str:
        return self._member_observation(member).app_image

    def _described_app_member(self, member: str, description: str) -> DescribedScreenshot:
        observed = self._member_observation(member)
        context = (
            self._stored_screens[member].modal_context
            if member in self._stored_screens
            else ModalContext()
        )
        opening_text, opening_images = self._modal_evidence(member)
        return DescribedScreenshot(
            observed.app_image,
            description,
            system_feedback=bool(observed.system_overlays),
            opening_context=opening_text,
            opening_images=opening_images,
            modal_origin="known"
            if context.origin_known
            else "unknown"
            if context.frames
            else "page",
        )

    def _modal_evidence(self, member: str) -> tuple[str, tuple[tuple[str, str], ...]]:
        stored = self._stored_screens.get(member)
        if stored is None or not stored.modal_context.frames:
            return "Foreground is a page, with no active modal opening.", ()
        text = ["Active modal opening evidence (outermost first; capture IDs are not identities):"]
        images: list[tuple[str, str]] = []
        for depth, frame in enumerate(stored.modal_context.frames, 1):
            opened = self._stored_screens.get(frame.opened)
            purpose = opened.described.page_description if opened is not None else "unknown modal"
            text.append(f"Depth {depth}: {purpose}.")
            opener = frame.source.previous
            if opener is None:
                text.append("Opening destination and initiating input are unknown.")
                continue
            caption = (
                f"Depth {depth} opening destination: {self.description(opener)} "
                f"(current graph node {self.node_of(opener)}). "
                f"Observed input: {json.dumps(frame.source.actions, ensure_ascii=False)}"
            )
            images.append((caption, self._app_member_image(opener)))
        return "\n".join(text), tuple(images)

    def _description_opening(self, source: MatchInput) -> tuple[str, tuple[tuple[str, str], ...]]:
        if source.previous is None:
            return (
                "No preceding matched capture is available. A visible modal has unknown origin.",
                (),
            )
        text, images = self._modal_evidence(source.previous)
        caption = (
            f"Preceding matched capture: {self.description(source.previous)}. "
            f"Actual input since then: {json.dumps(source.actions, ensure_ascii=False)}. "
            "These inputs are observations, not evidence that a modal necessarily opened."
        )
        return text, (*images, (caption, self._app_member_image(source.previous)))

    def identify_state(
        self,
        observation: Observation,
        *,
        expected_observation: str | None = None,
        match_input: MatchInput | None = None,
    ) -> str:
        """Return the node the screen in ``observation`` belongs to, creating one when
        none does, add ``observation`` to its members, and append what decided it to the
        identify file.

        The screen is embedded by image, unless it has an image vector already, then
        compared with the members of the in-app nodes of its ``package``, cheapest
        evidence first:

        1. A page member with a describe output and the same ``pixel_hash`` decides alone,
           when it belongs to one node only: the screen takes that member's describe
           output, so neither the describe call nor the verifier runs. Modal members
           always take the describe and verification path with their opening evidence.
        2. The describe call describes the screen; web content from outside the app
           goes to ``EXTERNAL_WEB_NODE``.
        3. The expected node, the node that lists ``expected_observation``, a screenshot
           the caller expects the screen to show the screen of, is verified against its
           member closest to the screen. A screenshot no node lists names no node.
        4. The short list's candidates are verified in order against their closest
           members, up to the ``verifier_calls`` of the matching settings.

        The describe call is given a bounded selection of relevant descriptions of the
        package's in-app members, and the verifier each screenshot with its own
        description. The first confirmation decides, and a check the verifier gives no
        answer on counts as a rejection. A screen every check rejects starts a new node,
        and the step's first rejection refines the screen's description.

        A ``black`` screen, all a secure window captures, is matched by its activity
        instead of steps 1 to 4: it joins the node with a black member of its activity,
        or starts a node described from the activity's class name, with no elements. It
        takes no describe output and has no description vector, so the ``text`` source
        never ranks a node by it.
        """
        if observation.system_surface is not None:
            return self.record_external_app_visit(observation)
        client, page_detail = self._require_page_detail()
        self._require_embedding()
        self._require_steps_dir()
        self._require_identify_file()
        decision = _Decision(expected_observation, self.node_of(expected_observation))
        same_package = self._same_package_nodes(observation.package)
        image_vector = self._image_vectors.get(observation.observation_id)
        if image_vector is None:
            image_vector = self._embed_image(observation.app_image)
            self._image_vectors[observation.observation_id] = image_vector
        if observation.black:
            return self._record_black_screen(observation, same_package.node_ids, decision)
        exact = self._exact_match(observation.pixel_hash, same_package.node_ids)
        if exact is not None:
            return self._record_exact_match(observation, *exact, decision)

        match_input = match_input or MatchInput()
        previous = self._stored_screens.get(match_input.previous or "")
        if (
            previous is None
            or match_input.actions is None
            or self._member_observation(match_input.previous or "").package != observation.package
        ):
            match_input, previous = MatchInput(), None
        prior_context = previous.modal_context if previous is not None else ModalContext()
        opening_text, opening_images = self._description_opening(match_input)
        described = _Described(
            *describe_page_and_state(
                client,
                DescribeRequest(
                    screenshot_b64=observation.app_image,
                    known_descriptions=self._package_descriptions(
                        same_package.node_ids, image_vector, decision.expected_node
                    )
                    or None,
                    existing_keys=same_package.keys or None,
                    system_feedback=bool(observation.system_overlays),
                    opening_context=opening_text,
                    opening_images=opening_images,
                ),
                settings=page_detail,
            )
        )
        logger.info(
            "Describe model output: '%s' | state keys: %s | elements: %d",
            described.page_description,
            list(described.state.keys()) if described.state else "[]",
            len(described.elements),
        )
        if described.is_external_web:
            decision.stage = "external_web"
            node_id = self._record_external_web_visit(observation)
            self._descriptions[observation.observation_id] = described.page_description
        else:
            screen = _ScreenDescription(
                described.page_description, described.state, described.elements
            )
            text_vector = self._embed(described.page_description)
            identification = _Identification(
                observation, image_vector, screen, text_vector, decision
            )
            self._stored_screens[observation.observation_id] = _StoredScreen(
                observation.pixel_hash,
                described,
                modal_context=prior_context.advance(
                    observation.observation_id, match_input, described.surface
                ),
            )
            node_id = self._assign(identification)
            self._text_vectors[observation.observation_id] = identification.text_vector
            self._descriptions[observation.observation_id] = identification.screen.description
        self._record_decision(observation, decision, node_id, described.as_record())
        return node_id

    def _assign(self, identification: _Identification) -> str:
        """Return the node the identified screen joins, a new one when every check
        rejects it, and record the visit there."""
        observation = identification.observation
        node_id = self._match_screen(identification)
        if node_id is None:
            node_id = self._add_screen_node(
                identification.screen, observation.package, observation.activity
            )
        else:
            self._add_activity(node_id, observation.activity)
        screen = identification.screen
        self._record_visit(node_id, screen.detail_snapshot, screen.elements, observation)
        return node_id

    def _same_package_nodes(self, package: str) -> _SamePackageNodes:
        """Collect *package*'s in-app nodes that have members, and the state keys of all
        its nodes."""
        node_ids: list[str] = []
        keys: list[str] = []
        for raw_id, data in self.graph.nodes(data=True):
            node_id = _node_id(raw_id)
            if data.get("package") != package or self._is_external(node_id):
                continue
            if _member_observations(data):
                node_ids.append(node_id)
            for k in data.get("state_schema", {}):
                if k not in keys:
                    keys.append(k)
        return _SamePackageNodes(node_ids, keys)

    def _package_descriptions(
        self, node_ids: Sequence[str], image_vector: list[float], expected_node: str | None
    ) -> list[str]:
        """Select whole descriptions by image similarity within entry and UTF-8 budgets.

        The expected node's closest member gets first choice, then members by image
        similarity, with ties in step order. Selected descriptions appear once, in the
        step order of their first occurrence. The byte budget includes list formatting.
        """
        described: list[tuple[str, str]] = []
        expected_member: str | None = None
        for node_id in node_ids:
            if node_id == expected_node:
                expected_member = self.closest_member(node_id, image_vector).observation_id
            members = _member_observations(self.graph.nodes[node_id])
            described += zip(members, self._member_descriptions(node_id, members), strict=True)
        described.sort(key=lambda member_description: step_order(member_description[0]))
        ranked = sorted(
            described,
            key=lambda item: (
                item[0] != expected_member,
                -cosine_similarity(image_vector, self._image_vectors[item[0]]),
                step_order(item[0]),
            ),
        )
        selected: set[str] = set()
        remaining = _DESCRIBE_CONTEXT_BYTES
        for _, description in ranked:
            size = len(f'  - "{description}"\n'.encode())
            if description in selected or size > remaining:
                continue
            selected.add(description)
            remaining -= size
            if len(selected) == _DESCRIBE_CONTEXT_ENTRIES:
                break
        return list(
            dict.fromkeys(description for _, description in described if description in selected)
        )

    def _exact_match(self, pixel_hash: str, node_ids: Sequence[str]) -> tuple[str, str] | None:
        """Return the node and member with a describe output whose stored screen has
        ``pixel_hash``, when the members that have it all belong to one of ``node_ids``;
        the latest such member. A black member has no describe output to copy."""
        if not pixel_hash:
            return None
        matches: dict[str, str] = {}
        for node_id in node_ids:
            for member in self.graph.nodes[node_id].get("observations", []):
                stored = self._stored_screens.get(member)
                if (
                    stored is not None
                    and stored.black_activity is None
                    and stored.pixel_hash == pixel_hash
                ):
                    if stored.described.surface.kind == "modal":
                        return None
                    matches[node_id] = member
        if len(matches) > 1:
            logger.info(
                "Exact pixel match with members of %d nodes (%s); not deciding on it",
                len(matches),
                ", ".join(matches),
            )
        if len(matches) != 1:
            return None
        return next(iter(matches.items()))

    def _record_exact_match(
        self, observation: Observation, node_id: str, member: str, decision: _Decision
    ) -> str:
        """Add the screen to ``node_id`` as an exact copy of its ``member``, whose
        describe output, description and text vector it takes, and record that in the
        identify file."""
        described = self._stored_screens[member].described
        text_vector = self.member_description_vectors([node_id])[node_id, member]
        (description,) = self._member_descriptions(node_id, [member])
        logger.info(
            "Exact pixel match with member %s of %s; reusing its description", member, node_id
        )
        self._add_activity(node_id, observation.activity)
        self._record_visit(node_id, dict(described.state), described.elements, observation)
        self._stored_screens[observation.observation_id] = _StoredScreen(
            observation.pixel_hash, described
        )
        self._descriptions[observation.observation_id] = description
        self._text_vectors[observation.observation_id] = text_vector
        decision.stage, decision.exact_match = "exact", member
        describe = {**described.as_record(), "copied_from": member}
        self._record_decision(observation, decision, node_id, describe)
        return node_id

    def _record_black_screen(
        self, observation: Observation, node_ids: Sequence[str], decision: _Decision
    ) -> str:
        """Add the black screen in ``observation`` to the node of ``node_ids`` with a black
        member of its activity, or to a new node described from the activity's class
        name, and record that in the identify file. The screen takes no describe output,
        so it has no description vector."""
        activity = observation.activity
        described = _black_screen_described(activity)
        node_id = self._black_node(activity, node_ids)
        if node_id is None:
            node_id = self._add_screen_node(
                _ScreenDescription(described.page_description, detail_snapshot={}, elements=[]),
                observation.package,
                activity,
            )
            logger.info(
                "Black screenshot of %s starts %s: no node has a black member of that activity",
                activity,
                node_id,
            )
        else:
            logger.info(
                "Black screenshot of %s joins %s, which has a black member of that activity",
                activity,
                node_id,
            )
        self._record_visit(node_id, {}, [], observation)
        self._stored_screens[observation.observation_id] = _StoredScreen(
            observation.pixel_hash, described, activity
        )
        self._descriptions[observation.observation_id] = described.page_description
        decision.stage = "black"
        self._record_decision(observation, decision, node_id, None)
        return node_id

    def _black_node(self, activity: str, node_ids: Sequence[str]) -> str | None:
        """Return the first of ``node_ids`` with a black member of ``activity``."""
        for node_id in node_ids:
            for member in self.graph.nodes[node_id].get("observations", []):
                stored = self._stored_screens.get(member)
                if stored is not None and stored.black_activity == activity:
                    return node_id
        return None

    def _record_external_web_visit(self, observation: Observation) -> str:
        self.graph.add_node(EXTERNAL_WEB_NODE, replay_misses=0)
        node_data = self.graph.nodes[EXTERNAL_WEB_NODE]
        node_data["visit_count"] = node_data.get("visit_count", 0) + 1
        _add_member(node_data, observation.observation_id)
        return EXTERNAL_WEB_NODE

    def record_external_app_visit(self, observation: Observation) -> str:
        """Record a visit to the screen of another app in ``observation`` and return the
        ``ext_<package>`` node that stands for every screen of that app.
        """
        package = observation.package.strip()
        node_id = f"ext_{package}"
        if node_id not in self.graph:
            self.graph.add_node(
                node_id,
                activity=observation.activity,
                state_schema={},
                visit_count=0,
                replay_misses=0,
                is_external=True,
            )
        node_data = self.graph.nodes[node_id]
        if observation.system_surface is not None:
            node_data["system_surface"] = observation.system_surface
        node_data["visit_count"] = node_data.get("visit_count", 0) + 1
        _add_member(node_data, observation.observation_id)
        return node_id

    def _verifiable_node(self, node_id: str | None, package: str) -> str | None:
        """Return ``node_id`` as it is named now, when it is an in-app node of
        ``package`` with members to compare with."""
        if node_id is None:
            return None
        node_id = self.resolve_node_id(node_id)
        if node_id not in self.graph or self._is_external(node_id):
            return None
        data = self.graph.nodes[node_id]
        if data.get("package") != package or not _member_observations(data):
            return None
        return node_id

    def _match_screen(self, identification: _Identification) -> str | None:
        """Return the node the verifier confirms the identified screen belongs to: the
        expected node first, then the short list. ``None`` when it confirms none."""
        package = identification.observation.package
        expected = self._verifiable_node(identification.decision.expected_node, package)
        excluded: set[str] = set()
        if expected is not None:
            closest = self.closest_member(expected, identification.image_vector)
            confirmed = self._verify(identification, expected, closest.observation_id)
            if confirmed is not None:
                identification.decision.stage = "expected"
                return confirmed
            excluded.add(self.resolve_node_id(expected))
        return self._verify_short_list(identification, excluded)

    def _verify_short_list(
        self, identification: _Identification, excluded: Collection[str]
    ) -> str | None:
        """Verify the short list of the identified screen, built from the in-app nodes
        of its package other than ``excluded``, and return the first node confirmed."""
        node_ids = [
            node_id
            for node_id in self._same_package_nodes(identification.observation.package).node_ids
            if node_id not in excluded
        ]
        query = ScreenQuery(identification.image_vector, identification.text_vector)
        candidates = self.short_list(query, node_ids)
        identification.decision.candidates = [candidate.as_record() for candidate in candidates]
        logger.info(
            "Short list: %s",
            ", ".join(
                f"{c.node_id} ({', '.join(f'{k}={v:.3f}' for k, v in c.sources.items())})"
                for c in candidates
            )
            or "empty",
        )
        for candidate in candidates[: self.matching.verifier_calls]:
            node_id = self.resolve_node_id(candidate.node_id)
            confirmed = self._verify(
                identification, node_id, candidate.closest_member.observation_id
            )
            if confirmed is not None:
                identification.decision.stage = "short_list"
                return confirmed
        return None

    def short_list(self, query: ScreenQuery, node_ids: Sequence[str]) -> list[Candidate]:
        """Return the nodes of ``node_ids`` that a source of ``short_list_sources`` ranks
        among its best under the matching settings, once each, ordered by their best rank
        in any source and then by the image score of their closest member."""
        candidates: dict[str, Candidate] = {}
        for source in short_list_sources(self.matching):
            ranked = sorted(
                source.scores(self, query, node_ids), key=lambda kv: kv[1], reverse=True
            )
            for rank, (node_id, score) in enumerate(ranked[: source.size]):
                candidate = candidates.get(node_id)
                if candidate is None:
                    closest = self.closest_member(node_id, query.image_vector)
                    candidate = candidates[node_id] = Candidate(node_id, rank, closest)
                candidate.best_rank = min(candidate.best_rank, rank)
                candidate.sources[source.name] = score
        return sorted(candidates.values(), key=_short_list_order)

    def _verification_members(self, node_id: str) -> list[str]:
        members = _member_observations(self.graph.nodes[node_id])
        known = [
            member
            for member in members
            if member in self._stored_screens
            and self._stored_screens[member].modal_context.origin_known
        ]
        return known or members

    def _verify_screens(
        self, existing: DescribedScreenshot, new: DescribedScreenshot
    ) -> dict[str, Any]:
        if {existing.modal_origin, new.modal_origin} == {"unknown", "known"}:
            return {
                "same": False,
                "answered": False,
                "reason": "One modal opening destination is unknown; identity remains unresolved.",
            }
        client, page_detail = self._require_page_detail()
        return verify_same_node(client, existing=existing, new=new, settings=page_detail)

    def _verify(self, identification: _Identification, node_id: str, member: str) -> str | None:
        """Ask the verifier whether the identified screen is ``node_id``'s ``member``, and
        return ``node_id`` when it says so; no answer counts as a rejection, and the first
        rejection refines the screen's description."""
        witnesses = self._verification_members(node_id)
        if member not in witnesses:
            member = max(
                witnesses,
                key=lambda witness: cosine_similarity(
                    identification.image_vector, self._image_vectors[witness]
                ),
            )
        verdict = self._verifier_reply(identification, node_id, member)
        same = verdict.get("same") is True
        answered = verdict.get("answered") is not False
        reason = as_str(verdict.get("reason"), "")
        identification.decision.verifications.append(
            {
                "node": node_id,
                "member": member,
                "same": same,
                "answered": answered,
                "reason": reason,
            }
        )
        if same:
            logger.info(
                "Verifier confirmed node %s against member %s | reason: %s",
                node_id,
                member,
                reason,
            )
            return node_id
        if answered:
            logger.info(
                "Verifier rejected node %s against member %s: reason: %s", node_id, member, reason
            )
        else:
            logger.warning(
                "Verifier gave no answer on node %s against member %s (%s); it counts as a "
                "rejection",
                node_id,
                member,
                reason,
            )
        if identification.verdicts is None and not identification.rejected:
            identification.rejected = True
            self._refine(identification, node_id, verdict)
        return None

    def _verifier_reply(
        self, identification: _Identification, node_id: str, member: str
    ) -> dict[str, Any]:
        verdicts = identification.verdicts
        if verdicts is not None:
            known = self.verify_members(
                member, identification.observation.observation_id, verdicts=verdicts
            )
            return {"same": known.same, "answered": known.answered, "reason": known.reason}
        (description,) = self._member_descriptions(node_id, [member])
        return self._verify_screens(
            existing=self._described_app_member(member, description),
            new=self._described_app_member(
                identification.observation.observation_id, identification.screen.description
            ),
        )

    def verify_members(self, existing: str, new: str, *, verdicts: MemberVerdicts) -> MemberVerdict:
        """Ask the verifier whether member observations ``existing`` and ``new`` show one
        screen, and keep the answer in ``verdicts``; a pair ``verdicts`` already holds is
        answered from there."""
        (verdict,) = self.verify_member_pairs([MemberQuestion(existing, new)], verdicts)
        return verdict

    def verify_member_pairs(
        self, questions: Sequence[MemberQuestion], verdicts: MemberVerdicts
    ) -> list[MemberVerdict]:
        """Ask the verifier ``questions`` at once, up to ``MAX_REQUESTS_IN_FLIGHT`` at a
        time, each member shown with its own description, keep each answer in
        ``verdicts``, and return the answers in order.

        A pair ``verdicts`` already holds, or an earlier question names too, is answered
        from there and marked ``reused``. No answer is not kept: it says nothing about the
        two screens, and a later call asks the pair again. When a call raises, the answers
        already returned are kept in ``verdicts`` before the error is raised.
        """
        if verdicts is self._checkpoint_verdicts:
            self._validate_checkpoint_answers(questions)
        first_asked: dict[frozenset[str], int] = {}
        for index, question in enumerate(questions):
            pair = frozenset((question.existing, question.new))
            if pair not in verdicts:
                first_asked.setdefault(pair, index)
        replies: dict[frozenset[str], MemberVerdict] = {}
        if first_asked:
            replies = self._ask_verifier(
                [questions[index] for index in first_asked.values()], verdicts
            )
        answers: list[MemberVerdict] = []
        for index, question in enumerate(questions):
            pair = frozenset((question.existing, question.new))
            if first_asked.get(pair) == index:
                answers.append(replies[pair])
                continue
            logger.info(
                "Reusing the verifier's answer on members %s and %s",
                question.existing,
                question.new,
            )
            known = replies[pair] if pair in replies else verdicts[pair]
            answers.append(known._replace(reused=True))
        return answers

    def _ask_verifier(
        self, questions: Sequence[MemberQuestion], verdicts: MemberVerdicts
    ) -> dict[frozenset[str], MemberVerdict]:
        """Ask ``questions`` concurrently and return the verdict on each pair, keeping
        each answered one in ``verdicts``.

        A member without a description raises ``ValueError`` naming every such member
        before any question is asked.
        """
        _, page_detail = self._require_page_detail()
        descriptions: dict[str, str] = {}
        undescribed: list[str] = []
        for question in questions:
            for member in (question.existing, question.new):
                description = self.description(member)
                if description is None:
                    undescribed.append(member)
                else:
                    descriptions[member] = description
        if undescribed:
            named = ", ".join(dict.fromkeys(undescribed))
            msg = f"Members without a description to show the verifier: {named}"
            raise ValueError(msg)
        replies: dict[frozenset[str], MemberVerdict] = {}

        def described(member: str) -> DescribedScreenshot:
            return self._described_app_member(member, descriptions[member])

        def ask(question: MemberQuestion) -> _MemberReply:
            existing, new = described(question.existing), described(question.new)
            evidence = (
                self._checkpoint_question_key(question, existing, new, page_detail)
                if verdicts is self._checkpoint_verdicts
                else None
            )
            reply = self._verify_screens(existing, new)
            return _MemberReply(
                MemberVerdict(
                    reply.get("same") is True,
                    as_str(reply.get("reason"), ""),
                    answered=reply.get("answered") is not False
                    and isinstance(reply.get("same"), bool),
                ),
                evidence,
            )

        def keep(question: MemberQuestion, reply: _MemberReply) -> None:
            pair = frozenset((question.existing, question.new))
            verdict = reply.verdict
            replies[pair] = verdict
            if verdict.answered:
                verdicts[pair] = verdict
                if reply.evidence is not None:
                    self._checkpoint_evidence[pair] = reply.evidence

        call_concurrently("node_verify", ask, questions, keep)
        return replies

    @staticmethod
    def _checkpoint_question_key(
        question: MemberQuestion,
        existing: DescribedScreenshot,
        new: DescribedScreenshot,
        settings: ModelSettings,
    ) -> str:
        members = sorted(
            (
                _shown_member_key(question.existing, existing),
                _shown_member_key(question.new, new),
            )
        )
        return evidence_key([node_verification_context(settings), members])

    def _validate_checkpoint_answers(self, questions: Sequence[MemberQuestion]) -> None:
        _, settings = self._require_page_detail()
        for question in questions:
            pair = frozenset(question)
            if pair not in self._checkpoint_verdicts:
                continue
            descriptions = [self.description(member) for member in question]
            current = None
            if all(isinstance(description, str) for description in descriptions):
                shown = [
                    self._described_app_member(member, as_str(description, ""))
                    for member, description in zip(question, descriptions, strict=True)
                ]
                current = self._checkpoint_question_key(question, shown[0], shown[1], settings)
            if current != self._checkpoint_evidence.get(pair):
                del self._checkpoint_verdicts[pair]
                self._checkpoint_evidence.pop(pair, None)

    def _refine(
        self, identification: _Identification, node_id: str, verdict: dict[str, Any]
    ) -> None:
        """Give the screen the verifier's refined description, embedding it; the rejected
        node is left as it was."""
        refined = as_str(verdict.get("new_description"), "")
        if not refined:
            return
        identification.screen = replace(identification.screen, description=refined)
        identification.text_vector = self._embed(refined)
        identification.decision.refinement = {"node": node_id, "new_description": refined}
        logger.info("Refined the screen's description on rejecting %s: %s", node_id, refined)

    def _add_activity(self, node_id: str, activity: str) -> None:
        node_data = self.graph.nodes[node_id]
        activities = node_data.get("activities", [node_data.get("activity", "")])
        if activity not in activities:
            activities.append(activity)
            logger.info("  Node %s: added activity %s (now %d)", node_id, activity, len(activities))
        node_data["activities"] = activities

    def _add_screen_node(self, screen: _ScreenDescription, package: str, activity: str) -> str:
        node_id = self._make_node_id()
        init_elements = [
            {
                "description": e.get("description", ""),
                "position": e.get("position", ""),
                "explored": False,
            }
            for e in screen.elements
        ]
        self.graph.add_node(
            node_id,
            package=package,
            activity=activity,
            activities=[activity],
            state_schema={},
            interactable_elements=init_elements,
            last_normalized_visit_milestone=0,
            visit_count=0,
            replay_misses=0,
        )
        logger.info(
            "Created new node %s: %s (%d elements)",
            node_id,
            screen.description,
            len(init_elements),
        )
        return node_id

    def _record_visit(
        self,
        node_id: str,
        detail_snapshot: dict[str, object],
        elements: list[dict[str, str]],
        observation: Observation,
    ) -> None:
        node_data = self.graph.nodes[node_id]
        schema = node_data.get("state_schema", {})
        _merge_into_schema(schema, detail_snapshot)
        node_data["state_schema"] = schema
        node_data["last_detail_snapshot"] = detail_snapshot
        node_data["visit_count"] = node_data.get("visit_count", 0) + 1
        # Structural elements accumulate across visits, so merge rather than replace.
        _merge_elements(node_data, elements)
        _add_member(node_data, observation.observation_id)

        logger.info(
            "Node %s schema updated: %d keys, visit #%d",
            node_id,
            len(schema),
            node_data["visit_count"],
        )

    def _make_node_id(self) -> str:
        """Return the next node ID, ``s<n>``."""
        node_id = f"s{self._next_id}"
        self._next_id += 1
        return node_id

    def resolve_node_id(self, node_id: str) -> str:
        """A merge can remove any node, including one a caller still holds, so a caller
        that keeps a node ID across one passes it through here.
        ``add_edge`` deliberately does not, so a stale endpoint is still reported.
        """
        while node_id in self._node_aliases:
            node_id = self._node_aliases[node_id]
        return node_id

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        """Return node attributes or None if the node doesn't exist."""
        if node_id in self.graph:
            return dict(self.graph.nodes[node_id])
        return None

    def get_all_nodes(self) -> list[tuple[str, dict[str, Any]]]:
        """Return all (node_id, attributes) pairs."""
        return [(n, dict(d)) for n, d in self.graph.nodes(data=True)]

    def get_unexplored_elements(self, node_id: str) -> list[dict[str, Any]]:
        """Return elements on a node that have not been explored yet."""
        if node_id not in self.graph:
            return []
        elements = self.graph.nodes[node_id].get("interactable_elements", [])
        return [e for e in elements if not e.get("explored", False)]

    def mark_element_explored(self, node_id: str, element: int | None) -> None:
        """``element`` is the 1-based number the planner saw in ``get_unexplored_elements``."""
        if element is None:
            return
        unexplored = self.get_unexplored_elements(node_id)
        if not 1 <= element <= len(unexplored):
            logger.warning(
                "Element %d is not among the %d unexplored element(s) on %s; marking none",
                element,
                len(unexplored),
                node_id,
            )
            return
        named = unexplored[element - 1]
        named["explored"] = True
        logger.info("Element explored on %s: '%s'", node_id, named.get("description", ""))

    def merge_nodes(self, keep_id: str, remove_id: str) -> bool:
        """Merge *remove_id* into *keep_id*, rewiring all edges.

        - Incoming edges to *remove_id* are rewired to *keep_id*.
        - Outgoing edges from *remove_id* are rewired from *keep_id*.
        - State schemas are merged (union of keys and values).
        - Visit counts are summed, and member observations are united in step order.
        - The removed node is deleted from the graph, and its ID resolves to *keep_id*.

        Returns True if the merge was performed, False if either node is missing
        or if *keep_id* and *remove_id* name the same node.
        """
        if keep_id not in self.graph or remove_id not in self.graph:
            logger.warning("merge_nodes: missing node(s) — keep=%s, remove=%s", keep_id, remove_id)
            return False

        # Falling through is not a harmless no-op: it deletes the node.
        if keep_id == remove_id:
            logger.warning("merge_nodes: refusing to merge node %s into itself", keep_id)
            return False

        _merge_node_attributes(self.graph.nodes[keep_id], self.graph.nodes[remove_id])
        self._rewire_edges(keep_id, remove_id)
        self.graph.remove_node(remove_id)
        self._node_aliases[remove_id] = keep_id

        logger.info(
            "Merged node %s into %s (now %d nodes, %d edges)",
            remove_id,
            keep_id,
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
        )
        return True

    def _rewire_edges(self, keep_id: str, remove_id: str) -> None:
        """Move every edge of *remove_id* onto *keep_id*.

        Incoming edges move first, then outgoing ones, then the self-loop. An edge that
        lands on an existing one appends its per-action lists in that order.
        """
        for pred in list(self.graph.predecessors(remove_id)):
            if pred != remove_id:
                self._move_edge(pred, remove_id, pred, keep_id)
        for succ in list(self.graph.successors(remove_id)):
            if succ != remove_id:
                self._move_edge(remove_id, succ, keep_id, succ)
        if self.graph.has_edge(remove_id, remove_id):
            self._move_edge(remove_id, remove_id, keep_id, keep_id)

    def _move_edge(self, source: str, target: str, new_source: str, new_target: str) -> None:
        edge_data = dict(self.graph[source][target])
        self.graph.remove_edge(source, target)
        self._merge_edge_data(new_source, new_target, edge_data)

    def _merge_edge_data(self, source: str, target: str, new_data: dict[str, Any]) -> None:
        """Merge edge data into an existing edge, or create it if absent."""
        if self.graph.has_edge(source, target):
            existing = self.graph[source][target]
            incoming = _aligned_lists(new_data)
            for key, values in _aligned_lists(existing).items():
                existing[key] = values + incoming[key]
            if new_data.get("instruction_templates"):
                existing.setdefault("instruction_templates", []).extend(
                    new_data["instruction_templates"]
                )
            existing["visit_count"] = existing.get("visit_count", 0) + new_data.get(
                "visit_count", 0
            )
        else:
            self.graph.add_edge(source, target, **new_data)

    def add_edge(
        self,
        source: str,
        target: str,
        action: dict[str, Any] | list[dict[str, Any]],
        *,
        details: EdgeDetails | None = None,
    ) -> None:
        """Add a directed edge (action) from source to target.

        Args:
            action: A single device action dict, or a list of them
                (compound action sequence, e.g. [tap, type, enter]).
            details: What this traversal recorded next to its action; none by default.

        If an edge with the same action already exists, it is not duplicated.
        Instead, a visit counter is incremented, and a known setup replaces the
        action's recorded one when that is unknown or longer. A new action from another
        node clears the target's replay misses: it can give a replay another way there.

        A setup of more than ``MAX_SETUP_ACTIONS`` actions is logged as dropped and
        recorded as unknown.

        The edge is dropped with a warning when either endpoint is missing, or when the
        source is an ``ext_`` node: one such node stands for every screen of an outside
        app or the web, so an edge out of it would let ``find_path`` join screens that
        no sequence of actions connects.
        """
        # networkx would otherwise create the missing endpoint as a node with no attributes.
        if source not in self.graph or target not in self.graph:
            logger.warning("add_edge: missing node(s) — source=%s, target=%s", source, target)
            return
        if source.startswith("ext_"):
            logger.warning(
                "add_edge: refusing an edge out of external node %s to %s", source, target
            )
            return

        if details is None:
            details = EdgeDetails()
        num_steps = details.num_steps
        if num_steps is None:
            num_steps = _default_num_steps(action)
        setup = _recorded_setup(details.setup, source, target)

        if self.graph.has_edge(source, target):
            edge_data = self.graph[source][target]
            if _count_repeat(edge_data, action, setup):
                return
        else:
            self.graph.add_edge(source, target, visit_count=1)
            edge_data = self.graph[source][target]
        edge_data.update(_aligned_lists(edge_data))
        for key, value in (
            ("actions", action),
            ("num_steps", num_steps),
            ("instructions", details.instruction or ""),
            ("target_observations", details.target_observation or ""),
            ("schema_deltas", details.schema_delta),
            ("setups", setup),
            ("open_menus", details.open_menu),
        ):
            edge_data[key].append(value)
        if source != target:
            self.graph.nodes[target]["replay_misses"] = 0

        n_actions = len(edge_data.get("actions", []))
        n_templates = len(edge_data.get("instruction_templates", []))
        if source == target:
            logger.info(
                "Self-loop edge %s: %d action(s), %d template(s)",
                source,
                n_actions,
                n_templates,
            )
        else:
            logger.debug(
                "Edge added: %s -> %s (%d action(s), %d template(s))",
                source,
                target,
                n_actions,
                n_templates,
            )

    def find_edge_action(
        self, source: str, target: str, instruction: str
    ) -> tuple[int, dict[str, Any] | list[dict[str, Any]]] | None:
        """Return the index and action of the first action on the edge recorded with exactly
        *instruction*, or None. An empty *instruction* matches nothing, not the placeholder of
        an action recorded without one.
        """
        if not instruction or not self.graph.has_edge(source, target):
            return None
        edge_data = self.graph[source][target]
        recorded = zip(edge_data["actions"], edge_data["instructions"], strict=True)
        for index, (action, recorded_instruction) in enumerate(recorded):
            if recorded_instruction == instruction:
                return index, action
        return None

    def action_setup(
        self, source: str, target: str, index: int
    ) -> tuple[dict[str, Any], ...] | None:
        """Return the device actions a replay sends on *source* before action *index* of
        source→target, or None when it sends none because no recorded action opens the
        menu the action was sent in (see ``_replay_options``)."""
        return next(
            (o.setup for o in self._replay_options(source, target) if o.index == index), None
        )

    def _replay_options(self, source: str, target: str) -> list[_ReplayOption]:
        """Return the actions of source→target a replay can send, in edge order.

        Before an action, a replay sends its recorded setup. When that is unknown and a
        menu was open over *source* as the action was sent, such as the navigation
        drawer whose row the action taps, it first opens that menu with the fewest
        device actions a recorded self-loop action of *source* opened it with. An action
        sent in a menu that no recorded action opens is left out.
        """
        aligned = _aligned_lists(self.graph[source][target])
        steps = as_int_list(aligned["num_steps"]) or [
            _default_num_steps(action) for action in aligned["actions"]
        ]
        loop = self.graph[source][source] if self.graph.has_edge(source, source) else None
        openers = _menu_openers(loop)
        options: list[_ReplayOption] = []
        for index, (own, setup, menu) in enumerate(
            zip(steps, aligned["setups"], aligned["open_menus"], strict=True)
        ):
            sent = _replay_setup(setup, menu, openers)
            if sent is not None:
                options.append(_ReplayOption(own + len(sent), index, sent))
        return options

    def move_edge_action(self, source: str, target: str, index: int, new_target: str) -> bool:
        """Move action *index* of source→target, with its entry in every per-action list, to
        source→new_target, which is created when absent.

        The moved action keeps its target observation, which describes the screen it
        reached. source→target is removed when no action is left on it; otherwise its
        templates are dropped, since they were generalised from the moved instruction too,
        and the next normalization rebuilds them.

        Returns False, changing nothing, when *new_target* is missing or *source* is an
        ``ext_`` node, which ``add_edge`` gives no outgoing edge either.
        """
        if new_target not in self.graph:
            logger.warning("move_edge_action: missing node %s", new_target)
            return False
        if source.startswith("ext_"):
            logger.warning("move_edge_action: refusing an edge out of external node %s", source)
            return False

        edge_data = self.graph[source][target]
        remaining = _aligned_lists(edge_data)
        moved: dict[str, Any] = {key: [values.pop(index)] for key, values in remaining.items()}
        moved["visit_count"] = 1
        if remaining["actions"]:
            edge_data.update(remaining)
            _drop_templates(edge_data)
        else:
            self.graph.remove_edge(source, target)
        self._merge_edge_data(source, new_target, moved)
        logger.info("Moved action %d of %s -> %s to %s", index, source, target, new_target)
        return True

    def relabel_edge_action(self, source: str, target: str, index: int, instruction: str) -> None:
        """Replace the instruction of action *index* on source→target.

        The edge's templates are dropped, since they were generalised from the instruction
        this replaces, and the next normalization rebuilds them.
        """
        edge_data = self.graph[source][target]
        edge_data["instructions"][index] = instruction
        _drop_templates(edge_data)

    def member_checks(self) -> list[tuple[str, MemberCheck]]:
        """Return, in node order, the checks ``plan_member_checks`` plans on the member
        groups of every in-app node, each with its node.

        A member without an image vector raises ``ValueError`` naming its node and every
        such member, and so does a member but a black one without a description vector.
        """
        node_ids = sorted(
            (node_id for node_id in map(_node_id, self.graph) if not self._is_external(node_id)),
            key=self._node_numeric_order,
        )
        images = self._member_units(node_ids)
        descriptions = self._member_description_units(node_ids)
        return [
            (node_id, check)
            for node_id in node_ids
            for check in plan_member_checks(
                _member_observations(self.graph.nodes[node_id]), images, descriptions
            )
        ]

    def split_members(
        self, node_id: str, members: Sequence[str], transitions: Sequence[TransitionRecord]
    ) -> dict[str, object]:
        """Move ``members`` of ``node_id`` to a new node built from their stored describe
        output, move the edge actions their ``transitions`` took, and return the change.

        The new node has no text vector of its own: the ``text`` source scores it by the
        vectors its members keep, so no embedding is requested."""
        new_node = self._node_from_stored(node_id, members)
        return {"new_node": True, **self._relocate(node_id, members, new_node, transitions)}

    def move_member(
        self,
        node_id: str,
        member: str,
        *,
        verdicts: MemberVerdicts,
        transitions: Sequence[TransitionRecord],
    ) -> dict[str, object]:
        """Move ``member`` out of ``node_id`` to the node screen matching's short list
        confirms for it, or to a new node when none does, move the edge actions its
        ``transitions`` took, and return the change.

        The member's stored describe output stands in for the describe call and its text
        vector for the description's embedding, so the short list costs only its
        verifier calls; the node it leaves is not on the list, and no description is
        refined.

        A black member, which has neither, moves by its activity instead, as screen
        matching places a black screen: the short list would rank every black node
        first, and the verifier would see identical black images.
        """
        black_activity = self._stored_screens[member].black_activity
        if black_activity is not None:
            return self._move_black_member(node_id, member, black_activity, transitions)
        identification = self._stored_identification(node_id, member, verdicts)
        destination = self._verify_short_list(identification, excluded={node_id})
        new_node = destination is None
        if destination is None:
            destination = self._node_from_stored(node_id, [member])
        decision = identification.decision
        return {
            "new_node": new_node,
            **self._relocate(node_id, [member], destination, transitions),
            "short_list": {
                "candidates": decision.candidates,
                "verifications": decision.verifications,
            },
        }

    def _move_black_member(
        self,
        node_id: str,
        member: str,
        activity: str,
        transitions: Sequence[TransitionRecord],
    ) -> dict[str, object]:
        """Move the black ``member`` of ``node_id`` to another node of its package with a
        black member of ``activity``, or to a new node described from ``activity``, and
        return the change."""
        package = as_str(self.graph.nodes[node_id].get("package"), "")
        others = [n for n in self._same_package_nodes(package).node_ids if n != node_id]
        destination = self._black_node(activity, others)
        new_node = destination is None
        if destination is None:
            destination = self._node_from_stored(node_id, [member])
        return {
            "new_node": new_node,
            **self._relocate(node_id, [member], destination, transitions),
            "black_activity": activity,
        }

    def _stored_identification(
        self, node_id: str, member: str, verdicts: MemberVerdicts
    ) -> _Identification:
        stored = self._stored_screens[member]
        observation = replace(
            self._member_observation(member),
            package=as_str(self.graph.nodes[node_id].get("package"), ""),
            pixel_hash=stored.pixel_hash,
            black=stored.black_activity is not None,
        )
        described = stored.described
        screen = _ScreenDescription(described.page_description, described.state, described.elements)
        return _Identification(
            observation,
            self._image_vectors[member],
            screen,
            self.member_description_vectors([node_id])[node_id, member],
            _Decision(expected_observation=None, expected_node=None),
            verdicts=verdicts,
        )

    def _member_activity(self, member: str) -> str:
        record = read_observation_record(self._require_steps_dir(), member) or {}
        return as_str(record.get("activity"), "")

    def _node_from_stored(self, node_id: str, members: Sequence[str]) -> str:
        """Add a node for ``members`` of ``node_id`` with the elements of the earliest of
        them, from its stored describe output."""
        earliest = min(members, key=step_order)
        described = self._stored_screens[earliest].described
        screen = _ScreenDescription(described.page_description, described.state, described.elements)
        return self._add_screen_node(
            screen,
            as_str(self.graph.nodes[node_id].get("package"), ""),
            self._member_activity(earliest),
        )

    def _relocate(
        self,
        node_id: str,
        members: Sequence[str],
        destination: str,
        transitions: Sequence[TransitionRecord],
    ) -> dict[str, object]:
        matched = self._matched_transitions(node_id, transitions)
        self._move_members(node_id, members, destination)
        edge_actions = self._reassign_actions(matched)
        logger.info(
            "Moved %d member(s) of %s to %s, and %d edge action(s) with them",
            len(members),
            node_id,
            destination,
            len(edge_actions),
        )
        return {
            "from": node_id,
            "to": destination,
            "members": list(members),
            "edge_actions": edge_actions,
        }

    def _move_members(self, node_id: str, members: Sequence[str], destination: str) -> None:
        """Move ``members`` and their visits from ``node_id`` to ``destination``, which
        also takes their states and activities."""
        source_data = self.graph.nodes[node_id]
        target_data = self.graph.nodes[destination]
        moving = set(members)
        source_data["observations"] = [
            m for m in _member_observations(source_data) if m not in moving
        ]
        target_data["observations"] = sorted(
            {*_member_observations(target_data), *moving}, key=step_order
        )
        source_data["visit_count"] = max(0, source_data.get("visit_count", 0) - len(moving))
        target_data["visit_count"] = target_data.get("visit_count", 0) + len(moving)
        schema = target_data.setdefault("state_schema", {})
        for member in sorted(moving, key=step_order):
            _merge_into_schema(schema, dict(self._stored_screens[member].described.state))
            self._add_activity(destination, self._member_activity(member))

    def _transition_nodes(self, transitions: Iterable[TransitionRecord]) -> list[tuple[str, str]]:
        """Return the nodes each transition's observations belong to now: the node that
        lists each, or, for one no node lists, the node it was recorded with."""
        members_of = {
            member: _node_id(node)
            for node, data in self.graph.nodes(data=True)
            for member in _member_observations(data)
        }
        return [
            (
                members_of.get(transition.before) or self.resolve_node_id(transition.from_node),
                members_of.get(transition.after) or self.resolve_node_id(transition.to_node),
            )
            for transition in transitions
        ]

    def _matched_transitions(
        self, node_id: str, transitions: Sequence[TransitionRecord]
    ) -> dict[tuple[str, str, int], list[TransitionRecord]]:
        """Return the transitions behind each action on an edge of ``node_id``, by edge and
        action index: those whose action equals it and whose observations belong to the
        edge's endpoints. Each transition backs the first such action only."""
        endpoints = self._transition_nodes(transitions)
        claimed: set[int] = set()
        matched: dict[tuple[str, str, int], list[TransitionRecord]] = {}
        edges = {*self.graph.in_edges(node_id), *self.graph.out_edges(node_id)}
        for source, target in sorted(edges):
            for index, entry in enumerate(self.graph[source][target].get("actions", [])):
                found = [
                    t
                    for t, transition in enumerate(transitions)
                    if t not in claimed
                    and endpoints[t] == (source, target)
                    and _same_action(entry, transition.action)
                ]
                if found:
                    claimed.update(found)
                    matched[source, target, index] = [transitions[t] for t in found]
        return matched

    def _reassign_actions(
        self, matched: Mapping[tuple[str, str, int], list[TransitionRecord]]
    ) -> list[dict[str, object]]:
        """Put each matched action on every edge its transitions now give, take it off
        its edge when they no longer give that one, and return each action moved. An
        action keeps its target observation on every edge, since it describes the screen
        the action reached."""
        before = {(s, t): _aligned_lists(self.graph[s][t]) for s, t, _ in matched}
        dropped: dict[tuple[str, str], set[int]] = {}
        departed: Counter[tuple[str, str]] = Counter()
        moved: list[dict[str, object]] = []
        for (source, target, index), found in matched.items():
            now = Counter(self._transition_nodes(found))
            if set(now) == {(source, target)}:
                continue
            for (new_source, new_target), traversals in now.items():
                if (new_source, new_target) == (source, target):
                    continue
                entry = {key: [values[index]] for key, values in before[source, target].items()}
                self._merge_edge_data(new_source, new_target, {**entry, "visit_count": traversals})
                departed[source, target] += traversals
            indices = dropped.setdefault((source, target), set())
            if (source, target) not in now:
                indices.add(index)
            moved.append(
                {
                    "edge": [source, target],
                    "action": before[source, target]["actions"][index],
                    "now": sorted([*pair] for pair in now),
                }
            )
        for (source, target), indices in dropped.items():
            self._drop_actions(source, target, indices, departed[source, target])
        return moved

    def _drop_actions(
        self, source: str, target: str, indices: Collection[int], traversals: int
    ) -> None:
        """Remove the actions at ``indices`` from source→target, and ``traversals`` from
        its visit count; the edge goes once no action is left on it."""
        edge_data = self.graph[source][target]
        remaining = _without_entries(_aligned_lists(edge_data), indices)
        if not remaining["actions"]:
            self.graph.remove_edge(source, target)
            return
        edge_data.update(remaining)
        if indices:
            _drop_templates(edge_data)
        edge_data["visit_count"] = max(
            len(remaining["actions"]), edge_data.get("visit_count", 0) - traversals
        )

    def _edge_to_normalize(self, source: str, target: str) -> _EdgeToNormalize | None:
        """Return the edge's normalize_edge inputs, or None when the skip rules spare it a call:
        it has no instruction, its last normalization covered the instructions it has, or it has
        one short instruction with no value in it."""
        edge_data = self.graph[source][target]
        instructions = edge_data.get("instructions", [])
        observations = edge_data.get("target_observations", [])
        recorded = [instr for instr in instructions if instr]

        if not recorded:
            return None

        instructions_hash = _instructions_hash(recorded)
        if edge_data.get("normalized_instructions_hash") == instructions_hash:
            return None

        if len(recorded) == 1:
            instr = recorded[0]
            # Heuristic: if no quoted values, numbers, or proper nouns, unlikely to be parameterizable
            has_variable = any(c.isdigit() for c in instr) or '"' in instr or "'" in instr
            if not has_variable and len(instr.split()) <= _FIXED_INSTRUCTION_MAX_WORDS:
                edge_data["instruction_templates"] = []
                return None

        return _EdgeToNormalize(
            source,
            target,
            list(instructions),
            list(observations) if any(observations) else None,
            instructions_hash,
        )

    def _apply_edge_template(self, edge: _EdgeToNormalize, result: dict[str, Any]) -> None:
        """Write normalize_edge's *result* to the edge, and record the instructions it covered."""
        edge_data = self.graph[edge.source][edge.target]
        edge_data["normalized_instructions_hash"] = edge.instructions_hash
        if result.get("is_template"):
            edge_data["instruction_templates"] = [
                {
                    "template": result["instruction_template"],
                    "observation_template": result.get("observation_template", ""),
                    "param_names": result.get("param_names", []),
                    "examples": result.get("examples", []),
                }
            ]
            logger.info(
                "  Edge %s→%s: '%s' (%d examples)",
                edge.source,
                edge.target,
                result["instruction_template"],
                len(result.get("examples", [])),
            )
        else:
            edge_data["instruction_templates"] = []
            if result.get("different_actions"):
                logger.info("  Edge %s→%s: different actions", edge.source, edge.target)

    def _normalize_edges(self, edges: Iterable[tuple[str, str]]) -> int:
        """Normalize *edges* into templates, sending their calls concurrently.

        The skip rules choose the edges that need a call, and the results are written and
        logged in the order of *edges*, both on this thread, whichever call returns first.
        Returns the number of *edges* that hold a template afterwards, including those the
        skip rules spared a call.
        """
        client, settings = self._require_page_detail()
        edge_list = list(edges)
        chosen = [self._edge_to_normalize(source, target) for source, target in edge_list]

        def call(edge: _EdgeToNormalize) -> dict[str, Any]:
            return normalize_edge(
                client,
                edge.instructions,
                target_observations=edge.target_observations,
                settings=settings,
            )

        call_concurrently(
            "normalize_edge",
            call,
            [edge for edge in chosen if edge is not None],
            self._apply_edge_template,
        )
        return sum(
            1
            for source, target in edge_list
            if self.graph[source][target].get("instruction_templates")
        )

    def normalize_node_edges(self, node_id: str) -> int:
        """Normalize all outgoing edges from one node.

        Returns the number of outgoing edges that hold a template afterwards.
        """
        if self._page_detail() is None or node_id not in self.graph:
            return 0

        normalized_count = self._normalize_edges(self.graph.out_edges(node_id))

        logger.info(
            "Node %s normalization complete: %d/%d outgoing edges templatized",
            node_id,
            normalized_count,
            self.graph.out_degree(node_id),
        )
        return normalized_count

    def maybe_normalize_node_edges(self, node_id: str) -> bool:
        """Normalize a node's outgoing edges when it crosses a visit milestone."""
        if node_id not in self.graph:
            return False

        node_data = self.graph.nodes[node_id]
        visit_count = node_data.get("visit_count", 0)
        milestone = visit_count // NORMALIZE_EVERY_N_VISITS
        last_milestone = node_data.get("last_normalized_visit_milestone", 0)

        if milestone <= 0 or milestone <= last_milestone:
            return False

        logger.info(
            "Node %s reached visit milestone %d (visit #%d). Running batch normalization.",
            node_id,
            milestone * NORMALIZE_EVERY_N_VISITS,
            visit_count,
        )
        self.normalize_node_edges(node_id)
        node_data["last_normalized_visit_milestone"] = milestone
        return True

    def normalize_all_edges(self) -> int:
        """Batch-normalize all edges in the graph into templates.

        Call this periodically (e.g. every N exploration steps) or once after
        exploration finishes. Replaces per-edge on-the-fly normalization to
        save API costs.

        For each edge whose instructions changed since its last normalization, groups
        all instructions and their target observations, then calls the normalizer to
        produce templates with {param} placeholders. Returns the number of edges that
        hold a template afterwards.
        """
        if self._page_detail() is None:
            logger.warning("No page_detail settings — skipping normalization")
            return 0

        normalized_count = self._normalize_edges(self.graph.edges())

        logger.info(
            "Normalization complete: %d/%d edges templatized",
            normalized_count,
            self.graph.number_of_edges(),
        )
        return normalized_count

    def get_all_edges_from_node(self, node_id: str) -> list[dict[str, Any]]:
        """Return all outgoing edges from a node as a list of dicts: each edge's actions
        with their instructions and target observations, one entry per action and ``""``
        where none was recorded, and its recorded schema deltas."""
        edges = []
        for _, target, data in self.graph.out_edges(node_id, data=True):
            edge_info = {
                "target": target,
                "actions": data.get("actions", []),
                "instructions": list(data.get("instructions", [])),
                "instruction_templates": data.get("instruction_templates", []),
                "target_observations": list(data.get("target_observations", [])),
                "visit_count": data.get("visit_count", 0),
                "schema_deltas": [d for d in data.get("schema_deltas", []) if d],
            }
            edges.append(edge_info)
        return edges

    def get_explored_actions_from_node(self, node_id: str) -> list[dict[str, Any]]:
        """Return a flat list of all actions already taken from this node."""
        actions = []
        for edge in self.get_all_edges_from_node(node_id):
            actions.extend(edge.get("actions", []))
        return actions

    def _edge_weight(self, source: str, target: str, _edge_data: dict[str, Any]) -> int | None:
        """Return the fewest device actions a replay of one of the edge's actions sends,
        what it sends before the action included (used as path weight); None, which
        hides the edge from a path, when a replay can send none of its actions."""
        return min((o.cost for o in self._replay_options(source, target)), default=None)

    def find_path(self, source: str, target: str) -> list[PathStep] | None:
        """Find the shortest weighted path from source to target.

        An edge weighs as its cheapest action: the fewest steps, what a replay sends
        before it included, so a path whose actions need no setup is preferred. A path
        never takes an action sent in a menu that no recorded action opens (see
        ``_replay_options``).

        Returns:
            The path's steps, the first being the source itself. A step's action is the
            edge's cheapest action, a single dict or a list of dicts (compound action),
            with what a replay sends before it.
            Returns None if no path exists.
        """
        if source == target:
            return [PathStep(source, {}, ())]

        if source not in self.graph or target not in self.graph:
            return None

        try:
            node_path = nx.shortest_path(
                self.graph,
                source,
                target,
                weight=self._edge_weight,
            )
        except nx.NetworkXNoPath:
            return None

        result = [PathStep(_node_id(node_path[0]), {}, ())]
        for s, t in itertools.pairwise(node_path):
            best = min(self._replay_options(s, t), key=lambda option: option.cost)
            result.append(
                PathStep(_node_id(t), self.graph[s][t]["actions"][best.index], best.setup)
            )

        return result

    def find_guided_path(
        self,
        source: str,
        waypoints: list[dict[str, Any]],
    ) -> list[dict[str, Any]] | None:
        """Find a path from source through ordered waypoints (essential nodes).

        Each waypoint is a dict:
            {
                "node_id": str,
                "required_schema": {"key": "value", ...} or {}
            }

        The path chains: source → wp1 → wp2 → ... → wpN.
        At each waypoint with required_schema, self-loop edges whose
        schema_delta matches the required keys are included.

        Returns:
            A list of path steps:
            [
                {"type": "navigate", "from": src, "to": tgt,
                 "setup": [action, ...], "action": action},
                {"type": "self_loop", "node": n, "instruction": str,
                 "actions": action_seq, "schema_delta": {...},
                 "required_schema": {"key": "value"}},
                ...
            ]
            Returns None if any navigation segment has no path.
        """
        path_steps: list[dict[str, Any]] = []
        current = source

        for wp in waypoints:
            target_node = wp["node_id"]
            required_schema = wp.get("required_schema", {})

            if current != target_node:
                nav_path = self.find_path(current, target_node)
                if nav_path is None:
                    logger.warning("No path from %s to %s", current, target_node)
                    return None

                for i in range(1, len(nav_path)):
                    node_id, action, setup = nav_path[i]
                    prev_node = nav_path[i - 1].node
                    edge_data = self.graph[prev_node][node_id]
                    path_steps.append(
                        {
                            "type": "navigate",
                            "from": prev_node,
                            "to": node_id,
                            "setup": list(setup),
                            "action": action,
                            "instruction": edge_data["instructions"][0],
                            "target_observation": edge_data["target_observations"][0],
                        }
                    )

            if required_schema:
                self_loop_step = self._find_matching_self_loop(target_node, required_schema)
                if self_loop_step:
                    path_steps.append(self_loop_step)
                else:
                    path_steps.append(
                        {
                            "type": "schema_gap",
                            "node": target_node,
                            "required_schema": required_schema,
                        }
                    )

            current = target_node

        return path_steps

    def _find_matching_self_loop(
        self, node_id: str, required_schema: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Find a self-loop edge that changes the required schema keys.

        Matches by key overlap — the self-loop must change at least one of
        the required keys. Prefers the self-loop that changes the most
        required keys.

        Returns a path step dict or None.
        """
        if not self.graph.has_edge(node_id, node_id):
            return None

        edge_data = self.graph[node_id][node_id]
        deltas = edge_data.get("schema_deltas", [])
        instructions = edge_data.get("instructions", [])
        target_observations = edge_data.get("target_observations", [])
        actions = edge_data.get("actions", [])

        best_idx = -1
        best_score = 0

        for idx, delta in enumerate(deltas):
            if not delta:
                continue
            score = sum(1 for k in required_schema if k in delta)
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx < 0:
            return None

        return {
            "type": "self_loop",
            "node": node_id,
            "instruction": instructions[best_idx],
            "target_observation": target_observations[best_idx],
            "action": actions[best_idx],
            "schema_delta": deltas[best_idx],
            "required_schema": required_schema,
        }

    def _is_external(self, node_id: str) -> bool:
        return node_id.startswith("ext_") or bool(self.graph.nodes[node_id].get("is_external"))

    def get_start_node(self) -> str | None:
        """Return the initial node (the app's entry point).

        Heuristic: the in-app node of the lowest number ``n`` among the IDs ``s<n>``, or
        else the first in-app ID in sorted order.
        """
        in_app = (_node_id(n) for n in self.graph if not self._is_external(n))
        return min(in_app, key=self._node_numeric_order, default=None)

    def back_leaves_app(self, node: str) -> bool:
        """Return whether Back on ``node`` leaves the app: on the start node, and on a node
        from which Back has been recorded leaving the app, an edge by Back to an ``ext_``
        node.

        The start node is the screen the app is launched on, the root activity of its
        task, and Back there returns to the home screen
        (https://developer.android.com/guide/components/activities/tasks-and-back-stack).
        An app can also open on another screen, such as the tab used last, where Back
        then leaves the app too.
        """
        return node == self.get_start_node() or any(
            target.startswith("ext_") and [{"action": "back"}] in data.get("actions", [])
            for _, target, data in self.graph.out_edges(node, data=True)
        )

    def nodes_without_path_to_start(self) -> PathToStart:
        """Return every node from which no recorded edge path leads to the start node.

        External nodes are listed apart: an ``ext_`` node keeps no outgoing edge, so
        every one of them is listed, and only the app's own screens need a way back.
        """
        start = self.get_start_node()
        if start is None:
            return PathToStart(None, [], [])
        can_return = nx.ancestors(self.graph, start) | {start}
        missing = sorted(_node_id(n) for n in self.graph if n not in can_return)
        return PathToStart(
            start,
            in_app=[n for n in missing if not self._is_external(n)],
            external=[n for n in missing if self._is_external(n)],
        )

    def record_replay_miss(self, node_id: str) -> None:
        """Count a replay to ``node_id`` that landed on another screen.

        A miss takes the node out of ``get_exploration_target_candidates`` until
        ``add_edge`` records a new action into it from another node, which clears the
        misses.
        """
        node_id = self.resolve_node_id(node_id)
        node_data = self.graph.nodes[node_id]
        misses = node_data["replay_misses"] + 1
        node_data["replay_misses"] = misses
        logger.info(
            "Replay missed %s %d time(s) since an action into it was recorded", node_id, misses
        )

    def get_resume_target(self, package_name: str | None = None) -> str | None:
        """Return the node ``get_exploration_target_candidates`` ranks first, or ``None``
        when it ranks none."""
        candidates = self.get_exploration_target_candidates(package_name=package_name, top_k=1)
        return _node_id(candidates[0]["node_id"]) if candidates else None

    def get_exploration_target_candidates(
        self,
        package_name: str | None = None,
        top_k: int = 15,
        *,
        always_offer_start: bool = False,
    ) -> list[dict[str, Any]]:
        """Return the nodes a replay from the start node can reach that are good
        candidates for continued exploration.

        This is a lightweight coverage signal for live exploration checkpoints.
        It favors nodes with many unexplored elements, few outgoing edges, and
        lower visit counts. Each candidate's ``descriptions`` are its node's
        ``node_descriptions``, and its ``is_start_node`` says whether it is the start
        node. A node that no path of actions a replay can send reaches from the start
        node (see ``find_path``) is left out, and so is a node whose replay missed since
        an action into it was last recorded, except the start node, which a replay
        reaches by launching the app.

        With ``always_offer_start``, a start node that ranks below the first
        ``top_k`` but still has unexplored elements takes the last of their places.
        """
        start = self.get_start_node()
        if start is None:
            return []

        reachable = nx.single_source_dijkstra_path_length(
            self.graph, start, weight=self._edge_weight
        )
        candidates: list[dict[str, Any]] = []

        for node_id in reachable:
            if self._is_external(node_id):
                continue
            node_data = self.graph.nodes[node_id]

            if package_name:
                node_pkg = node_data.get("package", "")
                if node_pkg and node_pkg != package_name:
                    continue
            if node_id != start and node_data["replay_misses"]:
                continue

            elements = node_data.get("interactable_elements", [])
            unexplored = [e for e in elements if not e.get("explored", False)]
            out_degree = self.graph.out_degree(node_id)
            visit_count = node_data.get("visit_count", 0)

            # The model makes the final choice, but pre-rank by coverage need.
            score = len(unexplored) * 3
            score += max(0, 3 - out_degree)
            score += max(0, 3 - visit_count)

            candidates.append(
                {
                    "node_id": node_id,
                    "descriptions": self.node_descriptions(node_id),
                    "visit_count": visit_count,
                    "out_degree": out_degree,
                    "total_elements": len(elements),
                    "unexplored_elements": len(unexplored),
                    "unexplored_element_descriptions": [
                        e.get("description", "") for e in unexplored[:8]
                    ],
                    "score": score,
                    "is_start_node": node_id == start,
                }
            )

        candidates.sort(
            key=lambda c: (
                c["score"],
                c["unexplored_elements"],
                -c["out_degree"],
                -c["visit_count"],
            ),
            reverse=True,
        )
        ranked = candidates[:top_k]
        if always_offer_start and ranked:
            start_candidate = next(
                (c for c in candidates[len(ranked) :] if c["is_start_node"]), None
            )
            if start_candidate is not None and start_candidate["unexplored_elements"] > 0:
                ranked[-1] = start_candidate
        return ranked

    def save_graph(self, path: str | Path) -> None:
        """Save graph to a JSON file, with the matching settings that made it, its
        members' text vectors in the file ``text_embeddings_path`` names and the image
        vectors in the step log beside it."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "next_id": self._next_id,
            "total_steps_completed": self.total_steps_completed,
            "matching": self.matching._asdict(),
            "node_aliases": dict(self._node_aliases),
            "nodes": [],
            "edges": [],
        }

        for node_id, attrs in self.graph.nodes(data=True):
            activities = attrs.get("activities", [attrs.get("activity", "")])
            node_data = {
                "id": node_id,
                "package": attrs.get("package", ""),
                "activity": activities[0] if activities else "",
                "activities": activities,
                "state_schema": attrs.get("state_schema", {}),
                "last_detail_snapshot": attrs.get("last_detail_snapshot", {}),
                "interactable_elements": attrs.get("interactable_elements", []),
                "last_normalized_visit_milestone": attrs.get("last_normalized_visit_milestone", 0),
                "visit_count": attrs.get("visit_count", 0),
                "replay_misses": attrs["replay_misses"],
                "observations": attrs.get("observations", []),
            }
            data["nodes"].append(node_data)

        for source, target, attrs in self.graph.edges(data=True):
            edge_data = {
                "source": source,
                "target": target,
                "actions": attrs.get("actions", []),
                "instructions": attrs.get("instructions", []),
                "instruction_templates": attrs.get("instruction_templates", []),
                "target_observations": attrs.get("target_observations", []),
                "num_steps": attrs.get("num_steps", []),
                "visit_count": attrs.get("visit_count", 0),
                "schema_deltas": attrs.get("schema_deltas", []),
                "setups": attrs.get("setups", []),
                "open_menus": attrs.get("open_menus", []),
            }
            if attrs.get("normalized_instructions_hash"):
                edge_data["normalized_instructions_hash"] = attrs["normalized_instructions_hash"]
            data["edges"].append(edge_data)

        # Before the graph: loading requires a text vector for every described member,
        # so a crash between the two writes must not leave a member without one.
        if self.embedding is not None and self._text_vectors:
            save_embeddings(
                text_embeddings_path(path), self._text_vectors, model=self.embedding.model
            )
        write_json_atomically(path, data, indent=2, ensure_ascii=False)

        if self.image_embedding is not None and self._image_vectors:
            save_image_embeddings(
                graph_steps_dir(path), self._image_vectors, model=self.image_embedding.model
            )

        logger.info(
            "Graph saved to %s (%d nodes, %d edges)", path, len(data["nodes"]), len(data["edges"])
        )

    def load_graph(self, path: str | Path) -> None:
        """Load graph from a JSON file, with its members' text vectors when the manager
        has embedding settings, the step log beside it as the manager's ``steps_dir``,
        its identify file as the manager's ``identify_file``, its image vectors when the
        manager has image-embedding settings, and its members' stored screens and
        descriptions.

        A file that lacks a field the graph needs raises ``ValueError`` naming it and
        each node, member or edge that lacks one: a node must list its member
        observations and have an integer ``replay_misses`` count, every member must have
        an image in the step log, every member of an in-app node must have a record
        there with its pixel hash and, unless it is of a black screenshot, a describe
        output in the identify file, and a description as ``read_member_descriptions``
        reads it, every edge must hold one entry per action in each per-action list, and
        the graph must name its matching settings, which must be the manager's. With
        embedding settings, so does a damaged text-vector file, one from another model,
        and one, missing or not, without the vector of a member of an in-app node with a
        describe output.
        """
        path = Path(path)
        with path.open(encoding="utf-8") as f:
            raw = json.load(f)
        # Run before any attribute below is assigned or the graph is cleared: a
        # corrupt file must leave the previously loaded graph and counters
        # untouched, not wipe them partway through the node loop.
        data = require_known_edge_endpoints(raw, path)
        steps_dir = graph_steps_dir(path)
        _require_saved_fields(data, path, steps_dir)
        saved_matching = _saved_matching(data)
        if saved_matching != self.matching:
            msg = (
                f"{path} was built with the matching settings {saved_matching}, not the "
                f"configured {self.matching}. Set vlm.matching to the graph's settings."
            )
            raise ValueError(msg)
        identify_file = identify_file_path(path)
        try:
            stored_screens = _stored_screens(data["nodes"], path, steps_dir, identify_file)
        except (TypeError, ValueError) as exc:
            raise unloadable_graph(path, [f"Invalid member metadata: {exc}"]) from exc
        descriptions = read_member_descriptions(path, data["nodes"])
        image_vectors = (
            load_image_embeddings(steps_dir, model=self.image_embedding.model)
            if self.image_embedding is not None
            else {}
        )
        text_vectors = (
            load_embeddings(
                text_embeddings_path(path),
                model=self.embedding.model,
                kind="text",
                required=[
                    member
                    for member, stored in stored_screens.items()
                    if stored.black_activity is None
                ],
            )
            if self.embedding is not None
            else {}
        )

        self._next_id = data.get("next_id", 0)
        self.total_steps_completed = data.get("total_steps_completed", 0)
        self.steps_dir = steps_dir
        self.identify_file = identify_file
        self._image_vectors = image_vectors
        self._unit_vectors = {}
        self._text_vectors = text_vectors
        self._descriptions = descriptions
        self._stored_screens = stored_screens
        self._checkpoint_verdicts.clear()
        self._checkpoint_evidence.clear()
        self._checkpoint_proposals.clear()
        self._node_aliases = {
            old: new
            for old, new in as_str_dict(data.get("node_aliases")).items()
            if isinstance(new, str)
        }
        self.graph.clear()

        for node_data in data["nodes"]:
            node_id = node_data["id"]
            activities = node_data.get("activities", [])
            self.graph.add_node(
                node_id,
                package=as_str(node_data.get("package"), ""),
                activity=activities[0] if activities else "",
                activities=activities,
                state_schema=node_data.get("state_schema", {}),
                last_detail_snapshot=node_data.get("last_detail_snapshot", {}),
                interactable_elements=node_data.get("interactable_elements", []),
                last_normalized_visit_milestone=node_data.get("last_normalized_visit_milestone", 0),
                visit_count=node_data.get("visit_count", 0),
                replay_misses=node_data["replay_misses"],
                observations=node_data["observations"],
            )

        for edge_data in data["edges"]:
            hashed = (
                {"normalized_instructions_hash": edge_data["normalized_instructions_hash"]}
                if edge_data.get("normalized_instructions_hash")
                else {}
            )
            self.graph.add_edge(
                edge_data["source"],
                edge_data["target"],
                **{key: edge_data[key] for key in ("actions", *PER_ACTION_LISTS)},
                instruction_templates=edge_data.get("instruction_templates", []),
                visit_count=edge_data.get("visit_count", 0),
                **hashed,
            )

        logger.info(
            "Graph loaded from %s (%d nodes, %d edges)",
            path,
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
        )

    def copy_identify_file(self, path: Path) -> None:
        """Start the identify file of the copy of the graph saved at ``path`` from this
        graph's, and append every later decision there."""
        source = self._require_identify_file()
        target = identify_file_path(path)
        write_bytes_atomically(target, source.read_bytes() if source.exists() else b"")
        self.identify_file = target

    def shortest_path(self, source: str, target: str) -> list[str] | None:
        """Find shortest weighted path between two nodes. Returns list of node IDs or None."""
        try:
            node_path = nx.shortest_path(
                self.graph,
                source,
                target,
                weight=self._edge_weight,
            )
        except nx.NetworkXNoPath:
            return None
        return [_node_id(node) for node in node_path]

    def get_path_actions(self, path: list[str]) -> list[dict[str, Any]]:
        """Given a path (list of node IDs), return the sequence of actions to follow it."""
        actions = []
        for i in range(len(path) - 1):
            source, target = path[i], path[i + 1]
            actions.append(self.graph[source][target]["actions"][0])
        return actions

    def summary(self) -> dict[str, Any]:
        """Return a summary of the graph, which names each node by its ``node_descriptions``."""
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "activities": list({d.get("activity") for _, d in self.graph.nodes(data=True)}),
            "nodes": [
                {
                    "id": n,
                    "descriptions": self.node_descriptions(_node_id(n)),
                    "schema_keys": list(d.get("state_schema", {}).keys()),
                    "visit_count": d.get("visit_count", 0),
                    "elements_explored": f"{sum(1 for e in d.get('interactable_elements', []) if e.get('explored'))}/{len(d.get('interactable_elements', []))}",
                }
                for n, d in self.graph.nodes(data=True)
            ],
        }

    def format_for_audit(self) -> str:
        """Format the graph as human-readable text for LLM audit, naming each node by its
        ``node_descriptions``."""
        lines = []

        lines.append(f"## Nodes ({self.graph.number_of_nodes()})")
        for node_id, data in self.graph.nodes(data=True):
            descriptions = self.node_descriptions(_node_id(node_id))
            visits = data.get("visit_count", 0)
            out_degree = self.graph.out_degree(node_id)
            in_degree = self.graph.in_degree(node_id)
            schema_keys = list(data.get("state_schema", {}).keys())
            keys_str = f" | state: [{', '.join(schema_keys)}]" if schema_keys else ""
            elements = data.get("interactable_elements", [])
            n_explored = sum(1 for e in elements if e.get("explored", False))
            elem_str = f" | elements: {n_explored}/{len(elements)} explored" if elements else ""
            lines.append(
                f"  {node_id}: {descriptions} "
                f"(visits={visits}, out={out_degree}, in={in_degree}{keys_str}{elem_str})"
            )

        # Every instruction is listed so the auditor can spot mismatches.
        lines.append(f"\n## Edges ({self.graph.number_of_edges()})")
        for source, target, data in self.graph.edges(data=True):
            instructions = data.get("instructions", [])
            observations = data.get("target_observations", [])
            is_self = " [self-loop]" if source == target else ""
            source_descriptions = self.node_descriptions(_node_id(source))
            target_descriptions = self.node_descriptions(_node_id(target))

            lines.append(
                f"  {source} ({source_descriptions}) → {target} ({target_descriptions}){is_self}:"
            )
            for instr, obs in zip(instructions, observations, strict=True):
                if not instr:
                    continue
                obs_str = f" → {obs}" if obs else ""
                lines.append(f'    - "{instr}"{obs_str}')

        return "\n".join(lines)

    def run_audit(self, app_name: str = "") -> dict[str, Any]:
        """Run LLM audit on the graph structure.

        Returns the audit result dict with "issues" and "summary".
        """
        page_detail = self._page_detail()
        if page_detail is None:
            logger.warning("No page_detail settings — cannot audit")
            return {"issues": [], "summary": "no client available"}
        client, settings = page_detail

        graph_text = self.format_for_audit()
        logger.info("Running graph audit (%d chars)...", len(graph_text))

        result = audit_graph(
            client,
            graph_text,
            app_name=app_name,
            settings=settings._replace(request_timeout=self.audit_timeout),
        )

        logger.info("Audit summary: %s", result.get("summary", ""))
        for issue in result.get("issues", []):
            itype = issue.get("type", "?")
            severity = issue.get("severity", "?")
            desc = issue.get("description", "")
            if itype == "merge_nodes":
                logger.info(
                    "  [%s] %s: %s + %s — %s",
                    severity,
                    itype,
                    issue.get("node_a", "?"),
                    issue.get("node_b", "?"),
                    desc,
                )
            elif itype == "retry_edge":
                logger.info(
                    "  [%s] %s: %s → %s — %s (instruction: '%s')",
                    severity,
                    itype,
                    issue.get("source_node", "?"),
                    issue.get("target_node", "?"),
                    desc,
                    issue.get("instruction", ""),
                )
            elif itype == "explore_node":
                logger.info(
                    "  [%s] %s: %s — %s (expected pages: %s)",
                    severity,
                    itype,
                    issue.get("node", "?"),
                    desc,
                    issue.get("expected_pages", []),
                )

        return result

    def _merge_skip_reason(self, node_a: str, node_b: str) -> str | None:
        """Return why a proposed pair cannot be verified, or ``None`` when its members can
        be compared.

        An ``ext_`` node stands for every screen of another app or the web, so its
        members are no screens of the app to compare with. Two nodes whose members are
        all black, of different activities, would show the verifier two identical black
        images, so they stay apart as screen matching keeps them.
        """
        if node_a not in self.graph or node_b not in self.graph:
            return "node missing"
        # A source can name one node twice, and it would verify as "same".
        if node_a == node_b:
            return "same node"
        if self._is_external(node_a) or self._is_external(node_b):
            return "external node"
        pair = (node_a, node_b)
        if not all(_member_observations(self.graph.nodes[n]) for n in pair):
            return "missing screenshot"
        black_a, black_b = self._black_activities(node_a), self._black_activities(node_b)
        if black_a and black_b and black_a.isdisjoint(black_b):
            return "black screens of different activities"
        return None

    def _black_activities(self, node_id: str) -> set[str] | None:
        """Return the activities of ``node_id``'s members when every one is black, or
        ``None`` when one is not or the node has none."""
        activities: set[str] = set()
        for member in _member_observations(self.graph.nodes[node_id]):
            stored = self._stored_screens.get(member)
            if stored is None or stored.black_activity is None:
                return None
            activities.add(stored.black_activity)
        return activities or None

    @staticmethod
    def _node_numeric_order(node_id: str) -> tuple[int, str]:
        number = _node_number(node_id)
        return (10**9 if number is None else number), node_id

    def merge_proposals(
        self,
        text_issues: Iterable[Mapping[str, object]],
        *,
        sources: Sequence[MergeSource] = MERGE_SOURCES,
    ) -> list[MergeProposal]:
        """Return the node pairs that may be one screen, once per unordered pair, each
        with the finding of every source that proposed it: the model's ``merge_nodes``
        issues as ``text_audit``, then each of ``sources``.

        A pair names its lower-numbered node first, the one a merge keeps.
        """
        proposals: dict[tuple[str, str], MergeProposal] = {}

        def add(node_a: str, node_b: str, finding: dict[str, object]) -> None:
            first, second = sorted((node_a, node_b), key=self._node_numeric_order)
            proposal = proposals.setdefault((first, second), MergeProposal(first, second))
            proposal.findings.append(finding)

        for issue in text_issues:
            node_a, node_b = as_str(issue.get("node_a"), ""), as_str(issue.get("node_b"), "")
            add(node_a, node_b, {"source": "text_audit", "issue": dict(issue)})
        for source in sources:
            for node_a, node_b, found in source.pairs(self):
                add(node_a, node_b, {"source": source.name, **found})
        return list(proposals.values())

    def verify_merge_proposals(
        self, proposals: Iterable[MergeProposal], verdicts: MemberVerdicts
    ) -> MergeReport:
        """Verify the proposed pairs in rounds, deducing every answer that follows from
        earlier ones, merge each cluster of nodes the answers make into its
        lowest-numbered node, and report each pair; README.md#merge-proposals.

        A pair's nodes are followed through merges made before this call. A question
        shows the verifier the closest members of the pair's two nodes, each with its own
        description: the member of the more-visited node first, then the member of the
        other. The pairs whose members are most alike come first.
        """
        results: list[dict[str, object]] = []
        checks: list[_MergeCheck] = []
        for proposal in proposals:
            result: dict[str, object] = {
                "node_a": proposal.node_a,
                "node_b": proposal.node_b,
                "sources": proposal.sources,
                "findings": proposal.findings,
            }
            results.append(result)
            check = self._merge_check(proposal, result)
            if check is not None:
                checks.append(check)
        rounds = self._ask_merge_rounds(sorted(checks, key=_most_alike_first), verdicts)
        merges = self._merge_clusters(checks, rounds.clusters(self._node_numeric_order))
        return MergeReport(
            pairs=results,
            rounds=rounds.rounds,
            asked=len(rounds.asked) + len(rounds.unanswered),
            deduced=len(rounds.deduced),
            merges=merges,
        )

    def _merge_check(
        self, proposal: MergeProposal, result: dict[str, object]
    ) -> _MergeCheck | None:
        node_a = self.resolve_node_id(proposal.node_a)
        node_b = self.resolve_node_id(proposal.node_b)
        skip_reason = self._merge_skip_reason(node_a, node_b)
        if skip_reason is not None:
            logger.warning("Skipping merge %s + %s — %s", node_a, node_b, skip_reason)
            result.update(status="skipped", reason=skip_reason)
            return None
        existing, new = sorted(
            (node_a, node_b),
            key=lambda n: (-self.graph.nodes[n].get("visit_count", 0), self._node_numeric_order(n)),
        )
        shown = self.closest_members(existing, new)
        result.update(members={existing: shown.existing, new: shown.new}, image_score=shown.score)
        question = MemberQuestion(shown.existing, shown.new)
        return _MergeCheck((node_a, node_b), question, shown.score, result)

    def _ask_merge_rounds(
        self, checks: Sequence[_MergeCheck], verdicts: MemberVerdicts
    ) -> PairRounds:
        """Ask the verifier the rounds of questions on ``checks``, ordered most likely
        one screen first, and add each pair's answer, asked or deduced, to its result."""
        rounds = PairRounds([check.nodes for check in checks])
        while questions := rounds.next_round():
            logger.info(
                "Merge round %d: %d question(s), %d of %d pair(s) decided before it",
                rounds.rounds,
                len(questions),
                len(rounds.asked) + len(rounds.deduced),
                len(checks),
            )
            answers = self.verify_member_pairs([checks[i].question for i in questions], verdicts)
            for index, answer in zip(questions, answers, strict=True):
                same = answer.same if answer.answered else None
                outcome = "no answer" if same is None else ("same" if same else "different")
                rounds.record(index, same=same)
                check = checks[index]
                check.result.update(
                    asked=True,
                    round=rounds.rounds,
                    same=same,
                    reason=answer.reason,
                    reused=answer.reused,
                )
                logger.info(
                    "Merge check of %s + %s on members %s and %s: %s, reason: %s",
                    *check.nodes,
                    check.question.existing,
                    check.question.new,
                    outcome,
                    answer.reason,
                )
        for index, deduction in rounds.deduced.items():
            basis = [list(checks[i].nodes) for i in deduction.basis]
            checks[index].result.update(
                asked=False, round=deduction.after, same=deduction.same, deduced_from=basis
            )
            logger.info(
                "Deduced %s + %s: %s, from %s",
                *checks[index].nodes,
                "same" if deduction.same else "different",
                "; ".join(" + ".join(pair) for pair in basis),
            )
        return rounds

    def _merge_clusters(
        self, checks: Sequence[_MergeCheck], clusters: Sequence[list[str]]
    ) -> list[Merge]:
        """Merge each cluster into its first node, and give each pair its status.

        The pairs inside a cluster are merged in proposal order, each into the
        lower-numbered of its two nodes, so the action lists of edges that land on one
        another are appended in proposal order, whatever order the rounds asked in.
        """
        cluster_of = {node: cluster for cluster in clusters for node in cluster}
        for check in checks:
            first, second = check.nodes
            cluster = cluster_of[first]
            if second not in cluster:
                check.result["status"] = "kept_separate"
                continue
            check.result.update(status="merged", kept=cluster[0])
            keep, remove = sorted(
                (self.resolve_node_id(first), self.resolve_node_id(second)),
                key=self._node_numeric_order,
            )
            if keep != remove:
                self.merge_nodes(keep, remove)
        return [Merge(cluster[0], cluster[1:]) for cluster in clusters if len(cluster) > 1]

    def _checkpoint_node(self, node_id: str) -> IdentityNode:
        data = self.graph.nodes[node_id]
        members = _member_observations(data)
        descriptions = tuple(sorted({self.description(member) or "" for member in members}))

        def neighbors(nodes: Iterable[str]) -> tuple[str, ...]:
            return tuple(
                sorted(
                    {
                        description
                        for neighbor in nodes
                        if neighbor != node_id and not self._is_external(neighbor)
                        for member in _member_observations(self.graph.nodes[neighbor])
                        if (description := self.description(member))
                    }
                )
            )

        incoming = neighbors(self.graph.predecessors(node_id))
        outgoing = neighbors(self.graph.successors(node_id))
        activities = data.get("activities", [data.get("activity", "")])
        named_activities = tuple(sorted(value for value in activities if isinstance(value, str)))
        package = as_str(data.get("package"), "")
        return IdentityNode(
            node_id,
            package,
            named_activities,
            descriptions,
            incoming,
            outgoing,
        )

    def _checkpoint_groups(self) -> dict[str, list[str]]:
        packages: dict[str, list[str]] = {}
        for node in sorted(self.graph, key=self._node_numeric_order):
            node_id = _node_id(node)
            if self._is_external(node_id) or not _member_observations(self.graph.nodes[node_id]):
                continue
            package = as_str(self.graph.nodes[node_id].get("package"), "")
            packages.setdefault(package, []).append(node_id)
        return packages

    def checkpoint_batches(self) -> Iterator[IdentityBatch]:
        """Yield bounded requests covering every same-package, in-app member-node pair."""
        packages = self._checkpoint_groups()
        for package in sorted(packages):
            if len(packages[package]) > 1:
                yield from identity_batches(
                    [self._checkpoint_node(node) for node in packages[package]]
                )

    @staticmethod
    def _checkpoint_reply(value: Mapping[str, object]) -> ProposalReply:
        listed = value.get("issues")
        if not isinstance(listed, list):
            return ProposalReply((), "missing issues", answered=False)
        hints: list[MergeHint] = []
        answered = value.get("answered") is not False
        for raw in listed:
            issue = as_object_dict(raw)
            first, second = issue.get("node_a"), issue.get("node_b")
            if not isinstance(first, str) or not isinstance(second, str):
                answered = False
                continue
            hints.append(
                MergeHint(
                    first,
                    second,
                    as_str(issue.get("severity"), "medium"),
                    as_str(issue.get("description"), ""),
                )
            )
        return ProposalReply(tuple(hints), as_str(value.get("summary"), ""), answered)

    def _checkpoint_proposal_replies(
        self,
        batches: Sequence[IdentityBatch],
        client: OpenRouter,
        settings: ModelSettings,
        app_name: str,
    ) -> list[tuple[ProposalReply, bool]]:
        context = node_merge_context(settings)
        keys = [evidence_key([context, app_name, batch.evidence]) for batch in batches]
        replies = {
            index: (self._checkpoint_proposals[key], True)
            for index, key in enumerate(keys)
            if key in self._checkpoint_proposals
        }

        def call(index: int) -> ProposalReply:
            batch = batches[index]
            reply = self._checkpoint_reply(
                audit_merge_nodes(client, batch.text, app_name=app_name, settings=settings)
            )
            if any(not batch.permits(hint.first, hint.second) for hint in reply.hints):
                reply = replace(reply, answered=False)
            return reply

        def keep(index: int, reply: ProposalReply) -> None:
            replies[index] = reply, False
            if reply.answered:
                self._checkpoint_proposals[keys[index]] = reply

        call_concurrently(
            "node_merge_audit",
            call,
            [index for index in range(len(batches)) if index not in replies],
            keep,
        )
        return [replies[index] for index in range(len(batches))]

    def run_node_merge_audit(self, app_name: str = "") -> CheckpointReport:
        """Propose duplicates in bounded complete batches and reuse unchanged evidence."""
        page_detail = self._page_detail()
        if page_detail is None:
            logger.warning("No page_detail settings — cannot run node merge audit")
            return {"issues": [], "results": [], "merged_count": 0}
        client, settings = page_detail
        settings = settings._replace(request_timeout=self.audit_timeout)
        app_name = clipped_text(app_name, APP_NAME_BYTES)
        issues: list[dict[str, object]] = []
        rejected: list[dict[str, object]] = []
        summaries: list[str] = []
        groups = self._checkpoint_groups()
        node_count = self.graph.number_of_nodes()
        candidate_nodes = sum(len(nodes) for nodes in groups.values())
        eligible_pairs = sum(len(nodes) * (len(nodes) - 1) // 2 for nodes in groups.values())
        logger.info(
            "Checkpoint identity scope: %d candidate nodes, %d excluded nodes, %d eligible "
            "pairs, %d excluded pairs; JSON request bound=%d bytes",
            candidate_nodes,
            node_count - candidate_nodes,
            eligible_pairs,
            node_count * (node_count - 1) // 2 - eligible_pairs,
            node_merge_request_bound(settings),
        )
        pairs = requests = reused = unanswered = input_bytes = largest = 0
        batches = list(self.checkpoint_batches())
        replies = self._checkpoint_proposal_replies(batches, client, settings, app_name)
        for index, (batch, (reply, known)) in enumerate(zip(batches, replies, strict=True)):
            size = len(node_merge_prompt(batch.text, app_name).encode("utf-8"))
            pairs += batch.pair_count
            requests += not known
            reused += known
            unanswered += not reply.answered
            input_bytes += 0 if known else size
            largest = max(largest, size)
            summaries.append(reply.summary)
            logger.info(
                "Checkpoint proposal batch %d: %d nodes, %d candidate pairs, %d input bytes, "
                "reused=%s, answered=%s",
                index + 1,
                len(batch.nodes),
                batch.pair_count,
                size,
                known,
                reply.answered,
            )
            for hint in reply.hints:
                issue = hint.as_record(batch)
                if batch.permits(hint.first, hint.second):
                    issues.append(issue)
                else:
                    rejected.append(
                        {
                            **issue,
                            "status": "skipped",
                            "reason": (
                                "same node"
                                if hint.first == hint.second and batch.node(hint.first) is not None
                                else "pair outside request scope"
                            ),
                            "batch": index + 1,
                        }
                    )
        report = self.verify_merge_proposals(
            self.merge_proposals(issues, sources=()), self._checkpoint_verdicts
        )
        verifier_requests = sum(
            pair.get("asked") is True and pair.get("reused") is not True for pair in report.pairs
        )
        verifier_reused = sum(pair.get("reused") is True for pair in report.pairs)

        logger.info(
            "Node merge audit complete: %d issue(s), %d node(s) merged; %d candidate pairs, "
            "%d proposal requests, %d proposal cache hits, %d verifier requests, %d verifier cache hits; "
            "%d skipped proposals, %d unanswered batches",
            len(issues),
            report.merged_count,
            pairs,
            requests,
            reused,
            verifier_requests,
            verifier_reused,
            len(rejected),
            unanswered,
        )
        return {
            "issues": issues,
            "results": [*report.pairs, *rejected],
            "merged_count": report.merged_count,
            "summary": "\n".join(dict.fromkeys(summaries)),
            "candidate_pairs": pairs,
            "candidate_nodes": candidate_nodes,
            "excluded_nodes": node_count - candidate_nodes,
            "excluded_pairs": node_count * (node_count - 1) // 2 - pairs,
            "proposal_requests": requests,
            "proposal_reused": reused,
            "proposal_unanswered": unanswered,
            "proposal_input_bytes": input_bytes,
            "largest_proposal_input_bytes": largest,
            "proposal_request_byte_bound": node_merge_request_bound(settings),
            "verifier_requests": verifier_requests,
            "verifier_reused": verifier_reused,
            "skipped_proposals": len(rejected),
        }
