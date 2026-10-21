"""Graph manager for Android-App-Graph exploration graphs.

A node represents a unique app state defined by:
  - Level 1: App package (as the device reports it)
  - Level 2: High-level page description (from VLM)
  - Level 3: State schema — a dict of parameter keys, each mapped to the set of
    observed values across all visits.  E.g. {"sort_order": ["price_low_to_high",
    "rating"], "search_query": ["keyboard", null]}

An edge represents a device action that transitions between states.

State matching compares a new screenshot with the members of the nodes of its app
package, cheapest evidence first: an exact pixel match outside the status bar, then
the node the caller expects, then a short list of nodes ranked by image and text
similarity, each checked by the verifier against its member closest to the screen.
Different Android activities that display the same screen are merged into a single
node; the node stores all observed activities.

Every time a node is visited (whether new or existing), the VLM detail JSON is
used to update the node's state schema — discovering new keys and accumulating
observed values.

A node holds no image. It lists the step-log observations assigned to it, in
order, and names one of them its representative: the screen the merge audit
compares against and the plot shows, read from the step log when they need it.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Collection, Iterator, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, NamedTuple

import networkx as nx
from openrouter import OpenRouter

from android_app_graph.android_packages import saved_node_package
from android_app_graph.graph_files import (
    graph_steps_dir,
    observation_image_b64,
    observation_image_path,
    require_known_edge_endpoints,
    write_json_atomically,
)
from android_app_graph.payloads import (
    as_float_list,
    as_int_list,
    as_object_dict,
    as_str,
    as_str_dict,
)
from android_app_graph.step_log import (
    Observation,
    annotate_observation,
    load_image_embeddings,
    read_observation_record,
    save_image_embeddings,
)
from android_app_graph.utils import DEFAULT_AUDIT_TIMEOUT, ModelSettings
from android_app_graph.utils.vlm_utils import (
    audit_graph,
    audit_merge_nodes,
    describe_page_and_state,
    get_embedding,
    get_image_embedding,
    normalize_edge,
    score_by_cosine,
    token_tracker,
    verify_same_node,
)

logger = logging.getLogger(__name__)

NORMALIZE_EVERY_N_VISITS = 10
# The verifier calls the short list may make for one screen; a checked expected node adds one.
SHORT_LIST_VERIFICATIONS = 3
_FIXED_INSTRUCTION_MAX_WORDS = 6
# The ext_ prefix keeps it out of start-node and exploration-target selection, like ext_<package>.
EXTERNAL_WEB_NODE = "ext_web"
# A redirecting link or a browser's own history can keep an outside page up after one back press.
EXTERNAL_WEB_BACK_PRESSES = 3


class _SamePackageNodes(NamedTuple):
    """A package's in-app nodes that have members to compare with, and the descriptions
    and state keys of all its nodes."""

    node_ids: list[str]
    descriptions: list[str]
    keys: list[str]


class PathToStart(NamedTuple):
    """Nodes from which the start node cannot be reached, split by origin."""

    start_node: str | None
    in_app: list[str]
    external: list[str]


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
    """

    instruction: str | None = None
    target_observation: str | None = None
    schema_delta: dict[str, Any] | None = None
    num_steps: int | None = None


@dataclass(frozen=True)
class _ScreenDescription:
    page_description: str
    embedding: list[float]
    detail_snapshot: dict[str, object]
    elements: list[dict[str, str]]


class _Described(NamedTuple):
    """What the describe call said about a screen, as an observation's record keeps it."""

    page_description: str
    state: dict[str, object]
    elements: list[dict[str, str]]
    is_external_web: bool

    def as_record(self) -> dict[str, object]:
        return {
            "page_description": self.page_description,
            "state": self.state,
            "elements": self.elements,
            "is_external_web": self.is_external_web,
        }

    @classmethod
    def from_record(cls, record: dict[str, object]) -> _Described | None:
        """Return the ``describe`` field of an observation's record, or ``None`` when it
        has none or it is malformed."""
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
        )


class _StoredScreen(NamedTuple):
    """A member's pixel hash and the describe output an exact match copies from it."""

    pixel_hash: str
    described: _Described


class ClosestMember(NamedTuple):
    """A node's member most similar to a screen by image, and that cosine; ``score`` is
    ``None`` when no member has an image vector and the member is the representative."""

    observation_id: str
    score: float | None


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
        (node_id, closest.score)
        for node_id in node_ids
        if (closest := graph.closest_member(node_id, query.image_vector)).score is not None
    ]


def _text_scores(
    graph: GraphManager, query: ScreenQuery, node_ids: Sequence[str]
) -> list[tuple[str, float]]:
    return score_by_cosine(
        query.description_embedding,
        [(node_id, graph.description_embedding(node_id)) for node_id in node_ids],
        scope="identify_state description-embedding",
    )


SHORT_LIST_SOURCES = (
    ShortListSource("image", 3, _image_scores),
    ShortListSource("text", 2, _text_scores),
)


def _short_list_order(candidate: Candidate) -> tuple[int, float]:
    score = candidate.closest_member.score
    return candidate.best_rank, -score if score is not None else float("inf")


@dataclass(slots=True)
class _Decision:
    """What decided a screen's node, as the ``identify`` field of its record keeps it."""

    expected_node: str | None
    stage: Literal["exact", "expected", "short_list", "new", "external_web"] = "new"
    exact_match: str | None = None
    candidates: list[dict[str, object]] = field(default_factory=list)
    verifications: list[dict[str, object]] = field(default_factory=list)
    refinement: dict[str, object] | None = None

    def as_record(self, node_id: str) -> dict[str, object]:
        return {
            "stage": self.stage,
            "expected_node": self.expected_node,
            "exact_match": self.exact_match,
            "candidates": self.candidates,
            "verifications": self.verifications,
            "refinement": self.refinement,
            "node": node_id,
        }


@dataclass(slots=True)
class _Identification:
    """One described screen on its way through ``identify_state``: its description,
    which the step's first rejection refines, and the decision its record gets."""

    observation: Observation
    image_vector: list[float]
    screen: _ScreenDescription
    decision: _Decision
    rejected: bool = False


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
    """Append an observation to a node's members; the first member becomes its representative."""
    node_data.setdefault("observations", []).append(observation_id)
    node_data.setdefault("representative_observation", observation_id)


def _merge_node_attributes(keep_data: dict[str, Any], remove_data: dict[str, Any]) -> None:
    """Fold a removed node's visits, members, activities, state schema and elements into
    the kept one.

    The kept node's description, representative and embedding stay as they are.
    """
    keep_data["visit_count"] = keep_data.get("visit_count", 0) + remove_data.get("visit_count", 0)
    members = {*keep_data.get("observations", []), *remove_data.get("observations", [])}
    # Step IDs are zero-padded integers, so a longer ID is a later step.
    keep_data["observations"] = sorted(members, key=lambda step_id: (len(step_id), step_id))
    _merge_activities(keep_data, remove_data)
    _merge_schema_values(keep_data, remove_data)
    _merge_elements(keep_data, remove_data.get("interactable_elements", []))


def _default_num_steps(action: object) -> int:
    return len(action) if isinstance(action, list) else 1


def _aligned_lists(edge_data: dict[str, Any]) -> dict[str, list[Any]]:
    """Readers pair entry i of every per-action list with actions[i] and skip placeholders."""
    actions = list(edge_data.get("actions", []))
    num_steps = list(edge_data.get("num_steps", []))
    num_steps.extend(_default_num_steps(action) for action in actions[len(num_steps) :])
    aligned: dict[str, list[Any]] = {"actions": actions, "num_steps": num_steps}
    for key, placeholder in (
        ("instructions", ""),
        ("target_observations", ""),
        ("schema_deltas", None),
    ):
        values = list(edge_data.get(key, []))
        values.extend([placeholder] * (len(actions) - len(values)))
        aligned[key] = values
    return aligned


def _member_observations(node: dict[str, Any]) -> list[str]:
    """Return a saved node's member observation IDs, or ``[]`` when it has none or
    they are not a list of strings."""
    observations = node.get("observations")
    if not isinstance(observations, list):
        return []
    members: list[str] = [o for o in observations if isinstance(o, str)]
    return members if len(members) == len(observations) else []


def _require_node_members(nodes: list[dict[str, Any]], path: Path, steps_dir: Path) -> None:
    """Raise ``ValueError`` naming every saved node without member observations, or
    whose representative is not one of them or has no image in ``steps_dir``.
    """
    without_members: list[str] = []
    without_image: list[str] = []
    for node in nodes:
        representative = node.get("representative_observation")
        if not isinstance(representative, str) or representative not in _member_observations(node):
            without_members.append(node["id"])
        elif not observation_image_path(steps_dir, representative).is_file():
            without_image.append(node["id"])
    problems = []
    if without_members:
        problems.append(f"no member observations with a representative: {without_members}")
    if without_image:
        problems.append(f"a representative without an image in {steps_dir}: {without_image}")
    if problems:
        msg = (
            f"{path}: node(s) with {'; '.join(problems)}. Graphs saved before nodes listed "
            "their member observations cannot be loaded: explore the app again into a new "
            "graph directory."
        )
        raise ValueError(msg)


def _stored_screens(nodes: list[dict[str, Any]], steps_dir: Path) -> dict[str, _StoredScreen]:
    """Read the pixel hash and describe output of every member of the saved in-app
    ``nodes`` whose record in ``steps_dir`` holds both."""
    stored: dict[str, _StoredScreen] = {}
    for node in nodes:
        if node["id"].startswith("ext_") or node.get("is_external"):
            continue
        for member in _member_observations(node):
            record = read_observation_record(steps_dir, member)
            if record is None:
                continue
            described = _Described.from_record(record)
            pixel_hash = as_str(record.get("pixel_hash"), "")
            if described is not None and pixel_hash:
                stored[member] = _StoredScreen(pixel_hash, described)
    return stored


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
        self.package_name = ""
        self.client = client
        self.page_detail = page_detail
        self.embedding = embedding
        self.image_embedding = image_embedding
        self.audit_timeout = audit_timeout
        # The step log the nodes' observations are in: load_graph sets the one beside
        # the file it loads, and the owner of a new graph sets it before identify_state.
        self.steps_dir: Path | None = None
        self._next_id = 0
        self._node_aliases: dict[str, str] = {}
        self._image_vectors: dict[str, list[float]] = {}
        self._stored_screens: dict[str, _StoredScreen] = {}
        self.total_steps_completed = 0

    @property
    def embedding_model(self) -> str:
        if self.embedding is None:
            msg = "GraphManager needs embedding settings for this operation"
            raise RuntimeError(msg)
        return self.embedding.model

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
            msg = "GraphManager needs a steps_dir to read and annotate observations"
            raise RuntimeError(msg)
        return self.steps_dir

    def _embed(self, text: str) -> list[float]:
        client, embedding = self._require_embedding()
        return get_embedding(
            client,
            text,
            embedding.model,
            request_timeout=embedding.request_timeout,
        )

    def _embed_image(self, screenshot_b64: str) -> list[float]:
        client, image_embedding = self._require_image_embedding()
        return get_image_embedding(
            client,
            screenshot_b64,
            image_embedding.model,
            request_timeout=image_embedding.request_timeout,
        )

    def _text_embedding(self, node_id: str, data: dict[str, object]) -> list[float]:
        vector = data.get("description_embedding")
        model = data.get("description_embedding_model")
        if vector and model != self.embedding_model:
            msg = (
                f"Text embedding for node {node_id!r} has model provenance {model!r}; "
                f"the configured model is {self.embedding_model!r}. Rebuild or re-explore "
                "the graph with the configured embedding model before comparing text vectors. "
                "Do not relabel existing vectors."
            )
            raise ValueError(msg)
        return as_float_list(vector)

    def description_embedding(self, node_id: str) -> list[float]:
        """Return the embedding of a node's description, ``[]`` when it has none; one
        from another embedding model raises ``ValueError``."""
        return self._text_embedding(node_id, self.graph.nodes[node_id])

    def image_vector(self, observation_id: str) -> list[float] | None:
        """Return the image vector of an observation, or ``None`` when it has none."""
        return self._image_vectors.get(observation_id)

    def closest_member(self, node_id: str, image_vector: list[float]) -> ClosestMember:
        """Return the member of ``node_id`` whose image vector is most similar to
        ``image_vector``, or its representative when no member has a vector."""
        data = self.graph.nodes[node_id]
        scored = score_by_cosine(
            image_vector,
            [(member, self._image_vectors.get(member)) for member in data.get("observations", [])],
            scope="closest-member image-embedding",
        )
        if not scored:
            return ClosestMember(data["representative_observation"], None)
        member, score = max(scored, key=lambda kv: kv[1])
        return ClosestMember(member, score)

    def representative_image_b64(self, node_id: str) -> str:
        """Return the WebP image of a node's representative observation as base64."""
        if self.steps_dir is None:
            msg = "GraphManager needs a steps_dir to read observation images"
            raise RuntimeError(msg)
        return observation_image_b64(
            self.steps_dir, self.graph.nodes[node_id]["representative_observation"]
        )

    def _in_app_members(self) -> Iterator[str]:
        for node_id, data in self.graph.nodes(data=True):
            if not self._is_external(node_id):
                yield from _member_observations(data)

    def embed_missing_members(self) -> int:
        """Embed the image of every member of an in-app node that has no image vector,
        save the vectors to the step log, and return how many were embedded.

        Members of a graph explored before image vectors were kept, or under another
        image-embedding model, have none.
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
                self._image_vectors[member] = self._embed_image(
                    observation_image_b64(steps_dir, member)
                )
        finally:
            save_image_embeddings(steps_dir, self._image_vectors, model=image_embedding.model)
        cost = token_tracker.summary().get(usage_key, {}).get("cost", 0.0) - cost_before
        logger.info("Embedded %d member image(s) for $%.6f", len(missing), cost)
        return len(missing)

    def identify_state(self, observation: Observation, *, expected_node: str | None = None) -> str:
        """Return the node the screen in ``observation`` belongs to, creating one when
        none does, add ``observation`` to its members, and add what decided it to the
        observation's record.

        The screen is embedded by image, then compared with the members of the in-app
        nodes of its ``package``, cheapest evidence first:

        1. A member with the same ``pixel_hash`` decides alone, when it belongs to one
           node only: the screen takes that member's describe output, so neither the
           describe call nor the verifier runs.
        2. The describe call describes the screen; web content from outside the app
           goes to ``EXTERNAL_WEB_NODE``.
        3. ``expected_node``, the node the caller expects the screen to be, is verified
           against its member closest to the screen.
        4. The short list's candidates are verified in order against their closest
           members, up to ``SHORT_LIST_VERIFICATIONS``.

        The first confirmation decides. A screen every check rejects starts a new node,
        and the step's first rejection refines the two descriptions.
        """
        client, page_detail = self._require_page_detail()
        self._require_embedding()
        steps_dir = self._require_steps_dir()
        same_package = self._same_package_nodes(observation.package)
        image_vector = self._embed_image(observation.screenshot_b64)
        self._image_vectors[observation.observation_id] = image_vector
        exact = self._exact_match(observation.pixel_hash, same_package.node_ids)
        if exact is not None:
            return self._record_exact_match(observation, *exact, expected_node=expected_node)

        described = _Described(
            *describe_page_and_state(
                client,
                observation.screenshot_b64,
                existing_nodes=same_package.descriptions or None,
                existing_keys=same_package.keys or None,
                settings=page_detail,
            )
        )
        logger.info(
            "Describe model output: '%s' | state keys: %s | elements: %d",
            described.page_description,
            list(described.state.keys()) if described.state else "[]",
            len(described.elements),
        )
        screen = _ScreenDescription(
            described.page_description, [], described.state, described.elements
        )
        decision = _Decision(expected_node)
        identification = _Identification(observation, image_vector, screen, decision)
        if described.is_external_web:
            decision.stage = "external_web"
            node_id = self._record_external_web_visit(observation)
        else:
            identification.screen = replace(
                screen, embedding=self._embed(described.page_description)
            )
            node_id = self._assign(identification)
            self._stored_screens[observation.observation_id] = _StoredScreen(
                observation.pixel_hash, described
            )
        annotate_observation(
            steps_dir,
            observation.observation_id,
            {"describe": described.as_record(), "identify": decision.as_record(node_id)},
        )
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
        """Collect *package*'s in-app nodes that have members, and the descriptions and
        state keys of all its nodes.

        Every text embedding's provenance is checked, so a vector from another embedding
        model raises here, before any model request.
        """
        node_ids: list[str] = []
        descriptions: list[str] = []
        keys: list[str] = []
        for raw_id, data in self.graph.nodes(data=True):
            node_id = _node_id(raw_id)
            if data.get("package") != package or self._is_external(node_id):
                continue
            self._text_embedding(node_id, data)
            if data.get("representative_observation"):
                node_ids.append(node_id)
            desc = data.get("page_description", "")
            if desc and desc not in descriptions:
                descriptions.append(desc)
            for k in data.get("state_schema", {}):
                if k not in keys:
                    keys.append(k)
        return _SamePackageNodes(node_ids, descriptions, keys)

    def _exact_match(self, pixel_hash: str, node_ids: Sequence[str]) -> tuple[str, str] | None:
        """Return the node and member whose stored screen has ``pixel_hash``, when the
        members that have it all belong to one of ``node_ids``; the latest such member."""
        if not pixel_hash:
            return None
        matches: dict[str, str] = {}
        for node_id in node_ids:
            for member in self.graph.nodes[node_id].get("observations", []):
                stored = self._stored_screens.get(member)
                if stored is not None and stored.pixel_hash == pixel_hash:
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
        self, observation: Observation, node_id: str, member: str, *, expected_node: str | None
    ) -> str:
        """Add the screen to ``node_id`` as an exact copy of its ``member``, whose
        describe output it takes, and record that on the screen's record."""
        described = self._stored_screens[member].described
        logger.info(
            "Exact pixel match with member %s of %s; reusing its description", member, node_id
        )
        self._add_activity(node_id, observation.activity)
        self._record_visit(node_id, dict(described.state), described.elements, observation)
        self._stored_screens[observation.observation_id] = _StoredScreen(
            observation.pixel_hash, described
        )
        decision = _Decision(expected_node, "exact", exact_match=member)
        annotate_observation(
            self._require_steps_dir(),
            observation.observation_id,
            {
                "describe": {**described.as_record(), "copied_from": member},
                "identify": decision.as_record(node_id),
            },
        )
        return node_id

    def _record_external_web_visit(self, observation: Observation) -> str:
        self.graph.add_node(EXTERNAL_WEB_NODE, page_description="[external web page]")
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
                page_description=f"[external app: {package}]",
                state_schema={},
                visit_count=0,
                is_external=True,
            )
        node_data = self.graph.nodes[node_id]
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
        if data.get("package") != package or not data.get("representative_observation"):
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
        query = ScreenQuery(identification.image_vector, identification.screen.embedding)
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
        for candidate in candidates[:SHORT_LIST_VERIFICATIONS]:
            node_id = self.resolve_node_id(candidate.node_id)
            confirmed = self._verify(
                identification, node_id, candidate.closest_member.observation_id
            )
            if confirmed is not None:
                identification.decision.stage = "short_list"
                return confirmed
        return None

    def short_list(self, query: ScreenQuery, node_ids: Sequence[str]) -> list[Candidate]:
        """Return the nodes of ``node_ids`` that a source in ``SHORT_LIST_SOURCES`` ranks
        among its best, once each, ordered by their best rank in any source and then
        by the image score of their closest member."""
        candidates: dict[str, Candidate] = {}
        for source in SHORT_LIST_SOURCES:
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

    def _verify(self, identification: _Identification, node_id: str, member: str) -> str | None:
        """Ask the verifier whether the identified screen is ``node_id``'s ``member``, and
        return ``node_id`` when it says so; the first rejection refines both descriptions."""
        client, page_detail = self._require_page_detail()
        description = self.graph.nodes[node_id].get("page_description", "")
        verdict = verify_same_node(
            client,
            screenshot_new_b64=identification.observation.screenshot_b64,
            screenshot_existing_b64=observation_image_b64(self._require_steps_dir(), member),
            existing_description=description,
            settings=page_detail,
        )
        same = verdict.get("same") is True
        reason = as_str(verdict.get("reason"), "")
        identification.decision.verifications.append(
            {"node": node_id, "member": member, "same": same, "reason": reason}
        )
        if same:
            logger.info(
                "Verifier confirmed node %s against member %s: %s | reason: %s",
                node_id,
                member,
                description,
                reason,
            )
            return node_id
        logger.info(
            "Verifier rejected node %s against member %s: reason: %s", node_id, member, reason
        )
        if not identification.rejected:
            identification.rejected = True
            self._refine(identification, node_id, verdict)
        return None

    def _refine(
        self, identification: _Identification, node_id: str, verdict: dict[str, Any]
    ) -> None:
        """Give the rejected node and the screen the verifier's refined descriptions,
        re-embedding both; the node is renamed to match its new description."""
        refined_existing = as_str(verdict.get("existing_description"), "")
        refined_new = as_str(verdict.get("new_description"), "")
        renamed = node_id
        if refined_existing and refined_existing != self.graph.nodes[node_id].get(
            "page_description"
        ):
            renamed = self.rename_node(node_id, refined_existing)
        if refined_new:
            identification.screen = replace(
                identification.screen,
                page_description=refined_new,
                embedding=self._embed(refined_new),
            )
        if renamed != node_id or refined_new:
            identification.decision.refinement = {
                "node": node_id,
                "renamed_to": renamed,
                "existing_description": refined_existing,
                "new_description": refined_new,
            }
            logger.info("Refined descriptions: %s -> %s | new: %s", node_id, renamed, refined_new)

    def _add_activity(self, node_id: str, activity: str) -> None:
        node_data = self.graph.nodes[node_id]
        activities = node_data.get("activities", [node_data.get("activity", "")])
        if activity not in activities:
            activities.append(activity)
            logger.info("  Node %s: added activity %s (now %d)", node_id, activity, len(activities))
        node_data["activities"] = activities

    def _add_screen_node(self, screen: _ScreenDescription, package: str, activity: str) -> str:
        node_id = self._make_node_id(screen.page_description)
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
            page_description=screen.page_description,
            state_schema={},
            interactable_elements=init_elements,
            description_embedding=screen.embedding,
            description_embedding_model=self.embedding_model,
            last_normalized_visit_milestone=0,
            visit_count=0,
        )
        logger.info(
            "Created new node %s: %s (%d elements)",
            node_id,
            screen.page_description,
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
        # A confirmed match keeps the node's representative; a new node, including a
        # verifier-rejected split, takes this observation as its own.
        _add_member(node_data, observation.observation_id)

        logger.info(
            "Node %s schema updated: %d keys, visit #%d",
            node_id,
            len(schema),
            node_data["visit_count"],
        )

    def _make_node_id(self, page_description: str) -> str:
        """Generate a unique, human-readable node ID."""
        nid = self._next_id
        self._next_id += 1
        sanitized = page_description.lower().replace(" ", "_").replace("/", "_")[:40]
        return f"s{nid}_{sanitized}"

    @staticmethod
    def _sanitize_description_for_node_id(page_description: str) -> str:
        """Convert a page description into the human-readable node ID suffix."""
        sanitized = page_description.lower().replace(" ", "_").replace("/", "_")[:40]
        return sanitized or "unknown_screen"

    def _renamed_node_id(self, node_id: str, page_description: str) -> str:
        """Build a unique node ID that keeps the original numeric prefix."""
        prefix = node_id.split("_", 1)[0] if "_" in node_id else node_id
        base = f"{prefix}_{self._sanitize_description_for_node_id(page_description)}"
        if base == node_id or base not in self.graph:
            return base

        suffix = 2
        while f"{base}_{suffix}" in self.graph:
            suffix += 1
        return f"{base}_{suffix}"

    def rename_node(self, node_id: str, page_description: str) -> str:
        """Rename a node ID and page description together.

        The graph uses node IDs as stable edge endpoints, so a rename must
        relabel the NetworkX node rather than only editing page_description.
        """
        if node_id not in self.graph:
            logger.warning("rename_node: missing node %s", node_id)
            return node_id

        node_data = self.graph.nodes[node_id]
        old_description = node_data.get("page_description", "")
        new_node_id = self._renamed_node_id(node_id, page_description)

        node_data["page_description"] = page_description
        node_data["description_embedding"] = self._embed(page_description)
        node_data["description_embedding_model"] = self.embedding_model

        if new_node_id != node_id:
            nx.relabel_nodes(self.graph, {node_id: new_node_id}, copy=False)
            self._record_alias(node_id, new_node_id)

        logger.info(
            "Renamed node %s → %s: '%s' → '%s'",
            node_id,
            new_node_id,
            old_description,
            page_description,
        )
        return new_node_id

    def _record_alias(self, old_id: str, new_id: str) -> None:
        # new_id can be an ID that was renamed away earlier; its stale alias would
        # close a cycle that resolve_node_id never leaves.
        self._node_aliases.pop(new_id, None)
        self._node_aliases[old_id] = new_id

    def resolve_node_id(self, node_id: str) -> str:
        """``identify_state`` can rename any node and a merge can remove one, including
        one a caller still holds, so a caller that keeps a node ID across either passes
        it through here.
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
        - The kept node retains its description, representative, and embedding.
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
        self._record_alias(remove_id, keep_id)

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
        Instead, a visit counter is incremented.

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

        if self.graph.has_edge(source, target):
            edge_data = self.graph[source][target]
            for a in edge_data.get("actions", []):
                if a == action:
                    edge_data["visit_count"] = edge_data.get("visit_count", 1) + 1
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
        ):
            edge_data[key].append(value)

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
        # A graph loaded from JSON can hold more actions than instructions; an action
        # without an instruction matches nothing, so a shorter list is not an error.
        recorded = zip(
            edge_data.get("actions", []), edge_data.get("instructions", []), strict=False
        )
        for index, (action, recorded_instruction) in enumerate(recorded):
            if recorded_instruction == instruction:
                return index, action
        return None

    def move_edge_action(self, source: str, target: str, index: int, new_target: str) -> bool:
        """Move action *index* of source→target, with its entry in every per-action list, to
        source→new_target, which is created when absent.

        The moved action's observation becomes *new_target*'s description, the screen it led
        to. source→target is removed when no action is left on it; otherwise its templates are
        dropped, since they were generalised from the moved instruction too, and the next
        normalization rebuilds them.

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
        moved["target_observations"] = [self.graph.nodes[new_target].get("page_description", "")]
        moved["visit_count"] = 1
        if remaining["actions"]:
            edge_data.update(remaining)
            edge_data["instruction_templates"] = []
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
        edge_data.update(_aligned_lists(edge_data))
        edge_data["instructions"][index] = instruction
        edge_data["instruction_templates"] = []

    def _normalize_single_edge(self, source: str, target: str) -> bool:
        """Normalize one edge's instructions into a reusable template if possible."""
        page_detail = self._page_detail()
        if page_detail is None:
            return False
        client, settings = page_detail

        edge_data = self.graph[source][target]
        instructions = edge_data.get("instructions", [])
        observations = edge_data.get("target_observations", [])
        recorded = [instr for instr in instructions if instr]

        if not recorded:
            return False

        existing_templates = edge_data.get("instruction_templates", [])
        if existing_templates:
            examples = (
                existing_templates[0].get("examples", [])
                if isinstance(existing_templates[0], dict)
                else []
            )
            if len(examples) >= len(recorded):
                return False

        if len(recorded) == 1:
            instr = recorded[0]
            # Heuristic: if no quoted values, numbers, or proper nouns, unlikely to be parameterizable
            has_variable = any(c.isdigit() for c in instr) or '"' in instr or "'" in instr
            if not has_variable and len(instr.split()) <= _FIXED_INSTRUCTION_MAX_WORDS:
                edge_data["instruction_templates"] = []
                return False

        result = normalize_edge(
            client,
            instructions,
            target_observations=observations if any(observations) else None,
            settings=settings,
        )

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
                source,
                target,
                result["instruction_template"],
                len(result.get("examples", [])),
            )
            return True

        edge_data["instruction_templates"] = []
        return False

    def normalize_node_edges(self, node_id: str) -> int:
        """Normalize all outgoing edges from one node.

        Returns the number of outgoing edges that produced a parameterized template.
        """
        if self._page_detail() is None or node_id not in self.graph:
            return 0

        normalized_count = 0
        for _, target in self.graph.out_edges(node_id):
            if self._normalize_single_edge(node_id, target):
                normalized_count += 1

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

    def normalize_all_edges(self) -> None:
        """Batch-normalize all edges in the graph into templates.

        Call this periodically (e.g. every N exploration steps) or once after
        exploration finishes. Replaces per-edge on-the-fly normalization to
        save API costs.

        For each edge, groups all instructions and their target observations,
        then calls the normalizer to produce templates with {param} placeholders.
        """
        if self._page_detail() is None:
            logger.warning("No page_detail settings — skipping normalization")
            return

        normalized_count = 0
        for source, target in self.graph.edges():
            if self._normalize_single_edge(source, target):
                normalized_count += 1

        logger.info(
            "Normalization complete: %d/%d edges templatized",
            normalized_count,
            self.graph.number_of_edges(),
        )

    def get_all_edges_from_node(self, node_id: str) -> list[dict[str, Any]]:
        """Return all outgoing edges from a node as a list of dicts."""
        edges = []
        for _, target, data in self.graph.out_edges(node_id, data=True):
            edge_info = {
                "target": target,
                "target_description": self.graph.nodes[target].get("page_description", ""),
                "actions": data.get("actions", []),
                "instructions": [i for i in data.get("instructions", []) if i],
                "instruction_templates": data.get("instruction_templates", []),
                "target_observations": [o for o in data.get("target_observations", []) if o],
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

    @staticmethod
    def _edge_weight(_source: str, _target: str, edge_data: dict[str, Any]) -> int:
        """Return the minimum num_steps for an edge (used as path weight)."""
        steps = as_int_list(edge_data.get("num_steps"))
        return min(steps) if steps else 1

    def find_path(
        self, source: str, target: str
    ) -> list[tuple[str, dict[str, Any] | list[dict[str, Any]]]] | None:
        """Find the shortest weighted path from source to target.

        Edge weight = minimum num_steps recorded for that edge, so paths
        with fewer total action steps are preferred.

        Returns:
            A list of (node_id, action) tuples representing the path.
            action can be a single dict or a list of dicts (compound action).
            The first entry has action={} (it's the starting node).
            Returns None if no path exists.
        """
        if source == target:
            return [(source, {})]

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

        result: list[tuple[str, dict[str, Any] | list[dict[str, Any]]]] = [
            (_node_id(node_path[0]), {})
        ]
        for i in range(len(node_path) - 1):
            s, t = node_path[i], node_path[i + 1]
            edge_data = self.graph[s][t]
            actions = edge_data.get("actions", [])
            steps_list = edge_data.get("num_steps", [1] * len(actions))
            if actions:
                best_idx = steps_list.index(min(steps_list)) if steps_list else 0
                action = actions[best_idx]
            else:
                action = {}
            result.append((t, action))

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
                {"type": "navigate", "from": src, "to": tgt, "actions": [action, ...]},
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
                    node_id, action = nav_path[i]
                    prev_node = nav_path[i - 1][0]
                    edge_data = self.graph[prev_node][node_id]
                    instruction = ""
                    instructions = edge_data.get("instructions", [])
                    if instructions:
                        instruction = instructions[0]
                    target_obs = ""
                    target_observations = edge_data.get("target_observations", [])
                    if target_observations:
                        target_obs = target_observations[0]
                    path_steps.append(
                        {
                            "type": "navigate",
                            "from": prev_node,
                            "to": node_id,
                            "action": action,
                            "instruction": instruction,
                            "target_observation": target_obs,
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

        if not deltas:
            return None

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
            "instruction": instructions[best_idx] if best_idx < len(instructions) else "",
            "target_observation": target_observations[best_idx]
            if best_idx < len(target_observations)
            else "",
            "action": actions[best_idx] if best_idx < len(actions) else [],
            "schema_delta": deltas[best_idx],
            "required_schema": required_schema,
        }

    def _is_external(self, node_id: str) -> bool:
        return node_id.startswith("ext_") or bool(self.graph.nodes[node_id].get("is_external"))

    def get_start_node(self) -> str | None:
        """Return the initial node (the app's entry point).

        Heuristic: the node with the lowest numeric ID suffix (s0_...).
        """
        candidates: list[tuple[int, str]] = []
        for node_id in self.graph:
            if self._is_external(node_id):
                continue
            if node_id.startswith("s") and "_" in node_id:
                num_part = node_id.split("_", 1)[0][1:]
                if num_part.isdigit():
                    candidates.append((int(num_part), _node_id(node_id)))

        if candidates:
            return min(candidates)[1]

        nodes = sorted(_node_id(n) for n in self.graph if not self._is_external(n))
        return nodes[0] if nodes else None

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
        """Return reachable nodes that are good candidates for continued exploration.

        This is a lightweight coverage signal for live exploration checkpoints.
        It favors nodes with many unexplored elements, few outgoing edges, and
        lower visit counts. Each candidate's ``is_start_node`` says whether it is
        the start node.

        With ``always_offer_start``, a start node that ranks below the first
        ``top_k`` but still has unexplored elements takes the last of their places.
        """
        start = self.get_start_node()
        if start is None:
            return []

        reachable = nx.descendants(self.graph, start) | {start}
        candidates: list[dict[str, Any]] = []

        for node_id in reachable:
            if self._is_external(node_id):
                continue
            node_data = self.graph.nodes[node_id]

            if package_name:
                node_pkg = node_data.get("package", "")
                if node_pkg and node_pkg != package_name:
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
                    "page_description": node_data.get("page_description", ""),
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

    @staticmethod
    def embeddings_path(graph_path: Path) -> Path:
        """Return the companion embeddings file path for a graph JSON."""
        return graph_path.with_suffix(".emb.json")

    def save_graph(self, path: str | Path) -> None:
        """Save graph to a JSON file, with text embeddings in a separate file and the
        image vectors in the step log beside it."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "package_name": self.package_name,
            "next_id": self._next_id,
            "total_steps_completed": self.total_steps_completed,
            "node_aliases": dict(self._node_aliases),
            "nodes": [],
            "edges": [],
        }

        embeddings: dict[str, dict[str, object]] = {}

        for node_id, attrs in self.graph.nodes(data=True):
            activities = attrs.get("activities", [attrs.get("activity", "")])
            node_data = {
                "id": node_id,
                "package": attrs.get("package", ""),
                "activity": activities[0] if activities else "",
                "activities": activities,
                "page_description": attrs.get("page_description", ""),
                "state_schema": attrs.get("state_schema", {}),
                "last_detail_snapshot": attrs.get("last_detail_snapshot", {}),
                "interactable_elements": attrs.get("interactable_elements", []),
                "last_normalized_visit_milestone": attrs.get("last_normalized_visit_milestone", 0),
                "visit_count": attrs.get("visit_count", 0),
                "observations": attrs.get("observations", []),
                "representative_observation": attrs.get("representative_observation"),
            }
            data["nodes"].append(node_data)

            emb = attrs.get("description_embedding")
            if emb:
                embeddings[node_id] = {
                    "model": attrs.get("description_embedding_model") or None,
                    "embedding": emb,
                }

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
            }
            schema_deltas = attrs.get("schema_deltas")
            if schema_deltas:
                edge_data["schema_deltas"] = schema_deltas
            data["edges"].append(edge_data)

        write_json_atomically(path, data, indent=2, ensure_ascii=False)

        emb_path = self.embeddings_path(path)
        write_json_atomically(emb_path, embeddings, ensure_ascii=False)
        if self.image_embedding is not None and self._image_vectors:
            save_image_embeddings(
                graph_steps_dir(path), self._image_vectors, model=self.image_embedding.model
            )

        logger.info(
            "Graph saved to %s (%d nodes, %d edges)", path, len(data["nodes"]), len(data["edges"])
        )

    def load_graph(self, path: str | Path, *, default_package_name: str = "") -> None:
        """Load graph from a JSON file (and companion embeddings file), with the step
        log beside it as the manager's ``steps_dir``, its image vectors when the
        manager has image-embedding settings, and its members' stored screens.

        ``default_package_name`` stands in for the ``package_name`` of a file that has
        none. A node without member observations, or whose representative has no image
        in the step log, raises ``ValueError`` naming it.
        """
        path = Path(path)
        with path.open(encoding="utf-8") as f:
            raw = json.load(f)
        # Run before any attribute below is assigned or the graph is cleared: a
        # corrupt file must leave the previously loaded graph and counters
        # untouched, not wipe them partway through the node loop.
        data = require_known_edge_endpoints(raw, path)
        steps_dir = graph_steps_dir(path)
        _require_node_members(data["nodes"], path, steps_dir)
        stored_screens = _stored_screens(data["nodes"], steps_dir)
        image_vectors = (
            load_image_embeddings(steps_dir, model=self.image_embedding.model)
            if self.image_embedding is not None
            else {}
        )

        emb_path = self.embeddings_path(path)
        embeddings: dict[str, list[float]] = {}
        embedding_models: dict[str, str] = {}
        if emb_path.exists():
            with emb_path.open(encoding="utf-8") as f:
                for node_id, entry in as_str_dict(json.load(f)).items():
                    if isinstance(entry, dict):
                        tagged = as_str_dict(entry)
                        embeddings[node_id] = as_float_list(tagged.get("embedding"))
                        embedding_models[node_id] = as_str(tagged.get("model"), "")
                    else:
                        embeddings[node_id] = as_float_list(entry)

        self.package_name = as_str(data.get("package_name"), "") or default_package_name
        self._next_id = data.get("next_id", 0)
        self.total_steps_completed = data.get("total_steps_completed", 0)
        self.steps_dir = steps_dir
        self._image_vectors = image_vectors
        self._stored_screens = stored_screens
        self._node_aliases = {
            old: new
            for old, new in as_str_dict(data.get("node_aliases")).items()
            if isinstance(new, str)
        }
        self.graph.clear()

        for node_data in data.get("nodes", []):
            node_id = node_data["id"]
            emb = embeddings.get(node_id, node_data.get("description_embedding", []))
            activities = node_data.get("activities", [node_data.get("activity", "")])
            self.graph.add_node(
                node_id,
                package=saved_node_package(node_data, self.package_name),
                activity=activities[0] if activities else "",
                activities=activities,
                page_description=node_data.get("page_description", ""),
                state_schema=node_data.get("state_schema", {}),
                last_detail_snapshot=node_data.get("last_detail_snapshot", {}),
                interactable_elements=node_data.get("interactable_elements", []),
                description_embedding=emb,
                description_embedding_model=(
                    embedding_models.get(node_id, "")
                    if node_id in embeddings
                    else as_str(node_data.get("description_embedding_model"), "")
                ),
                last_normalized_visit_milestone=node_data.get("last_normalized_visit_milestone", 0),
                visit_count=node_data.get("visit_count", 0),
                observations=_member_observations(node_data),
                representative_observation=node_data["representative_observation"],
            )

        for edge_data in data.get("edges", []):
            edge_attrs = {
                "actions": edge_data.get("actions", []),
                "instructions": edge_data.get("instructions", []),
                "instruction_templates": edge_data.get("instruction_templates", []),
                "target_observations": edge_data.get("target_observations", []),
                "num_steps": edge_data.get("num_steps", []),
                "visit_count": edge_data.get("visit_count", 0),
            }
            schema_deltas = edge_data.get("schema_deltas")
            if schema_deltas:
                edge_attrs["schema_deltas"] = schema_deltas
            self.graph.add_edge(
                edge_data["source"],
                edge_data["target"],
                **edge_attrs,
            )

        logger.info(
            "Graph loaded from %s (%d nodes, %d edges)",
            path,
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
        )

    def find_node_by_description(self, query: str) -> list[tuple[str, float]]:
        """Find nodes whose page_description is most similar to the query.

        Returns a list of (node_id, similarity) sorted descending.
        """
        candidates = [
            (_node_id(node_id), self._text_embedding(_node_id(node_id), data))
            for node_id, data in self.graph.nodes(data=True)
        ]
        query_emb = self._embed(query)
        results = score_by_cosine(
            query_emb,
            candidates,
            scope="find_node_by_description description-embedding",
        )
        results.sort(key=lambda x: x[1], reverse=True)
        return results

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
            edge_data = self.graph[source][target]
            edge_actions = edge_data.get("actions", [])
            if edge_actions:
                actions.append(edge_actions[0])
        return actions

    def summary(self) -> dict[str, Any]:
        """Return a summary of the graph."""
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "activities": list({d.get("activity") for _, d in self.graph.nodes(data=True)}),
            "nodes": [
                {
                    "id": n,
                    "description": d.get("page_description", ""),
                    "schema_keys": list(d.get("state_schema", {}).keys()),
                    "visit_count": d.get("visit_count", 0),
                    "elements_explored": f"{sum(1 for e in d.get('interactable_elements', []) if e.get('explored'))}/{len(d.get('interactable_elements', []))}",
                }
                for n, d in self.graph.nodes(data=True)
            ],
        }

    def format_for_audit(self) -> str:
        """Format the graph as human-readable text for LLM audit."""
        lines = []

        lines.append(f"## Nodes ({self.graph.number_of_nodes()})")
        for node_id, data in self.graph.nodes(data=True):
            desc = data.get("page_description", "")
            visits = data.get("visit_count", 0)
            out_degree = self.graph.out_degree(node_id)
            in_degree = self.graph.in_degree(node_id)
            schema_keys = list(data.get("state_schema", {}).keys())
            keys_str = f" | state: [{', '.join(schema_keys)}]" if schema_keys else ""
            elements = data.get("interactable_elements", [])
            n_explored = sum(1 for e in elements if e.get("explored", False))
            elem_str = f" | elements: {n_explored}/{len(elements)} explored" if elements else ""
            lines.append(
                f'  {node_id}: "{desc}" '
                f"(visits={visits}, out={out_degree}, in={in_degree}{keys_str}{elem_str})"
            )

        # Every instruction is listed so the auditor can spot mismatches.
        lines.append(f"\n## Edges ({self.graph.number_of_edges()})")
        for source, target, data in self.graph.edges(data=True):
            instructions = data.get("instructions", [])
            observations = data.get("target_observations", [])
            is_self = " [self-loop]" if source == target else ""
            source_desc = self.graph.nodes[source].get("page_description", "")
            target_desc = self.graph.nodes[target].get("page_description", "")

            lines.append(f'  {source} ("{source_desc}") → {target} ("{target_desc}"){is_self}:')
            for i, instr in enumerate(instructions):
                if not instr:
                    continue
                obs = observations[i] if i < len(observations) else ""
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

    def merge_skip_reason(self, node_a: str, node_b: str) -> str | None:
        """Return why a merge candidate pair cannot be verified, or ``None`` when both
        representatives can be compared.

        An ``ext_`` node stands for every screen of another app or the web, so its
        representative is no screen of the app to compare with.
        """
        if self._is_external(node_a) or self._is_external(node_b):
            return "external node"
        pair = (node_a, node_b)
        if not all(self.graph.nodes[n].get("representative_observation") for n in pair):
            return "missing screenshot"
        return None

    @staticmethod
    def _node_numeric_order(node_id: str) -> tuple[int, str]:
        if node_id.startswith("s") and "_" in node_id:
            num = node_id.split("_", 1)[0][1:]
            if num.isdigit():
                return int(num), node_id
        return 10**9, node_id

    def run_node_merge_audit(self, app_name: str = "") -> dict[str, Any]:
        """Run a lightweight node-merge audit and merge verified duplicates."""
        page_detail = self._page_detail()
        if page_detail is None:
            logger.warning("No page_detail settings — cannot run node merge audit")
            return {"issues": [], "results": [], "merged_count": 0}
        client, settings = page_detail

        graph_text = self.format_for_audit()
        logger.info("Running node merge audit (%d chars)...", len(graph_text))
        audit_result = audit_merge_nodes(
            client,
            graph_text,
            app_name=app_name,
            settings=settings._replace(request_timeout=self.audit_timeout),
        )
        issues = audit_result.get("issues", [])
        results = []

        for issue in issues:
            node_a = issue.get("node_a", "")
            node_b = issue.get("node_b", "")
            if node_a not in self.graph or node_b not in self.graph:
                results.append(
                    {
                        "issue": issue,
                        "status": "skipped",
                        "reason": "node missing",
                    }
                )
                continue

            # An auditor can name one node twice, and it would verify as "same".
            if node_a == node_b:
                logger.warning("Skipping merge candidate %s + itself", node_a)
                results.append(
                    {
                        "issue": issue,
                        "status": "skipped",
                        "reason": "same node",
                    }
                )
                continue

            skip_reason = self.merge_skip_reason(node_a, node_b)
            if skip_reason is not None:
                results.append({"issue": issue, "status": "skipped", "reason": skip_reason})
                continue

            data_a = self.graph.nodes[node_a]
            data_b = self.graph.nodes[node_b]

            desc_a = data_a.get("page_description", "")
            desc_b = data_b.get("page_description", "")
            visits_a = data_a.get("visit_count", 0)
            visits_b = data_b.get("visit_count", 0)
            ref_desc = desc_a if visits_a >= visits_b else desc_b

            logger.info(
                "Verifying node merge candidate: %s ('%s') + %s ('%s')",
                node_a,
                desc_a,
                node_b,
                desc_b,
            )
            verify_result = verify_same_node(
                client,
                screenshot_new_b64=self.representative_image_b64(node_b),
                screenshot_existing_b64=self.representative_image_b64(node_a),
                existing_description=ref_desc,
                settings=settings,
            )
            if verify_result.get("same", False):
                keep, remove = sorted(
                    (node_a, node_b),
                    key=self._node_numeric_order,
                )
                self.merge_nodes(keep, remove)
                results.append(
                    {
                        "issue": issue,
                        "status": "merged",
                        "kept": keep,
                        "removed": remove,
                        "reason": verify_result.get("reason", ""),
                    }
                )
            else:
                results.append(
                    {
                        "issue": issue,
                        "status": "kept_separate",
                        "reason": verify_result.get("reason", ""),
                    }
                )

        merged_count = sum(1 for result in results if result.get("status") == "merged")
        logger.info(
            "Node merge audit complete: %d issue(s), %d merge(s)",
            len(issues),
            merged_count,
        )
        return {
            "issues": issues,
            "results": results,
            "merged_count": merged_count,
            "summary": audit_result.get("summary", ""),
        }
