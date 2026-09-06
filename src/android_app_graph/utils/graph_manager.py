"""Graph manager for Android-App-Graph exploration graphs.

A node represents a unique app state defined by:
  - Level 1: App package (derived from Android activity name)
  - Level 2: High-level page description (from VLM)
  - Level 3: State schema — a dict of parameter keys, each mapped to the set of
    observed values across all visits.  E.g. {"sort_order": ["price_low_to_high",
    "rating"], "search_query": ["keyboard", null]}

An edge represents an action (in AITK format) that transitions between states.

State matching uses embedding similarity on the page description within the same
app package to decide whether a new screenshot belongs to an existing node or is
new.  Different Android activities that display the same screen are merged into a
single node; the node stores all observed activities.

Every time a node is visited (whether new or existing), the VLM detail JSON is
used to update the node's state schema — discovering new keys and accumulating
observed values.
"""

from __future__ import annotations

import base64
import json
import logging
import math
from pathlib import Path
from typing import Any, NamedTuple

import networkx as nx
from openai import OpenAI

from android_app_graph.payloads import as_float_list, as_int_list, as_str_dict
from android_app_graph.utils.vlm_utils import (
    audit_graph,
    audit_merge_nodes,
    describe_page_and_state,
    get_embedding,
    normalize_edge,
    verify_same_node,
)

logger = logging.getLogger(__name__)

DEFAULT_SIMILARITY_THRESHOLD = 0.85
NORMALIZE_EVERY_N_VISITS = 10


class _IdentifyCacheEntry(NamedTuple):
    screenshot_hash: int
    activity: str
    node_id: str


def _package_from_activity(activity: str) -> str:
    """Extract the app package from a full Android activity name.

    ``com.citymapper.app.home.HomeActivity2`` → ``com.citymapper.app``
    ``com.citymapper.app/com.citymapper.app.MainActivity`` → ``com.citymapper.app``

    Heuristic: take the first 3 dot-segments (``com.company.app``).  This is the
    standard Android package convention and is enough to group activities that
    belong to the same app while separating different apps.
    """
    if "/" in activity:
        activity = activity.split("/", maxsplit=1)[0]
    parts = activity.split(".")
    if len(parts) >= 3:
        return ".".join(parts[:3])
    return activity


def _node_id(value: object) -> str:
    """Return a graph node ID as ``str``.

    ``networkx`` ships no annotations, so every node ID read back out of the
    graph is ``Unknown``. The graph only ever stores ``str`` IDs, and this is
    the one place that says so.
    """
    return value if isinstance(value, str) else str(value)


def _require_known_edge_endpoints(data: dict[str, Any], path: Path) -> None:
    """Raise when an edge names a node the same file does not define.

    networkx would create that endpoint bare, so a hand-edited or truncated file
    would load quietly.  See #62.
    """
    node_ids = {_node_id(node["id"]) for node in data.get("nodes", [])}
    for edge in data.get("edges", []):
        source = _node_id(edge["source"])
        target = _node_id(edge["target"])
        unknown = [node_id for node_id in (source, target) if node_id not in node_ids]
        if unknown:
            msg = (
                f"{path}: edge {source} -> {target} references "
                f"node(s) absent from the file: {', '.join(unknown)}"
            )
            raise ValueError(msg)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


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
        else:
            if value not in schema[key]:
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


class GraphManager:
    """Manages a directed graph of app UI states and transitions."""

    def __init__(
        self,
        page_detail_client: OpenAI | None = None,
        page_detail_model: str = "gpt-5.4",
        embedding_client: OpenAI | None = None,
        embedding_model: str = "text-embedding-3-small",
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ) -> None:
        self.graph = nx.DiGraph()
        self.page_detail_client = page_detail_client
        self.page_detail_model = page_detail_model
        self.embedding_client = embedding_client
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self._next_id = 0
        self._dirty_screenshots: set[str] = set()
        self._last_identify_cache: _IdentifyCacheEntry | None = None
        self.total_steps_completed = 0

    def _require_page_detail_client(self) -> OpenAI:
        """Return the page-detail client, or raise when none was configured."""
        if self.page_detail_client is None:
            msg = "GraphManager needs a page_detail_client for this operation"
            raise RuntimeError(msg)
        return self.page_detail_client

    def _require_embedding_client(self) -> OpenAI:
        """Return the embedding client, or raise when none was configured."""
        if self.embedding_client is None:
            msg = "GraphManager needs an embedding_client for this operation"
            raise RuntimeError(msg)
        return self.embedding_client

    def identify_state(self, activity: str, screenshot_b64: str) -> str:
        """Identify which node the current screen belongs to, or create a new one.

        Process:
        1. Collect existing same-package node descriptions and state keys.
        2. Single VLM call → model outputs a fresh description + dynamic state.
           Existing node list is provided as context for disambiguation only.
        3. Compute embedding, find best matching node by similarity.
        4. If similarity >= threshold → ALWAYS call verifier with both screenshots.
           Verifier can confirm match, reject it, and rename the existing node.
        5. If verified → update existing node; if rejected → create new node.
        6. Merge state snapshot into the node's schema.

        Matching uses app package (not exact activity) so that the same visual
        screen reached via different Android activities is merged into one node.

        Returns:
            The node ID (str) for this state.
        """
        screen_hash = hash(screenshot_b64)
        cached = self._last_identify_cache
        if (
            cached is not None
            and screen_hash == cached.screenshot_hash
            and activity == cached.activity
            and cached.node_id in self.graph
        ):
            logger.info("identify_state cache hit → %s (skipping VLM)", cached.node_id)
            return cached.node_id

        current_pkg = _package_from_activity(activity)

        same_pkg_descriptions: list[str] = []
        same_pkg_keys: list[str] = []
        for _, data in self.graph.nodes(data=True):
            if _package_from_activity(data.get("activity", "")) == current_pkg:
                desc = data.get("page_description", "")
                if desc and desc not in same_pkg_descriptions:
                    same_pkg_descriptions.append(desc)
                for k in data.get("state_schema", {}):
                    if k not in same_pkg_keys:
                        same_pkg_keys.append(k)

        page_description, detail_snapshot, elements = describe_page_and_state(
            self._require_page_detail_client(),
            screenshot_b64,
            existing_nodes=same_pkg_descriptions or None,
            existing_keys=same_pkg_keys or None,
            model=self.page_detail_model,
        )
        logger.info(
            "Describe model output: '%s' | state keys: %s | elements: %d",
            page_description,
            list(detail_snapshot.keys()) if detail_snapshot else "[]",
            len(elements),
        )

        description_embedding = get_embedding(
            self._require_embedding_client(), page_description, model=self.embedding_model
        )

        best_node_id: str | None = None
        best_similarity = -1.0
        for node_id, data in self.graph.nodes(data=True):
            if _package_from_activity(data.get("activity", "")) != current_pkg:
                continue
            existing_emb = data.get("description_embedding")
            if existing_emb is None:
                continue
            sim = _cosine_similarity(description_embedding, existing_emb)
            if sim > best_similarity:
                best_similarity = sim
                best_node_id = _node_id(node_id)

        matched_node_id: str | None = None

        if best_node_id is not None:
            logger.info(
                "Best candidate: %s (sim=%.3f, threshold=%.2f) — '%s'",
                best_node_id,
                best_similarity,
                self.similarity_threshold,
                self.graph.nodes[best_node_id].get("page_description", ""),
            )

        if best_node_id is not None and best_similarity >= self.similarity_threshold:
            candidate_data = self.graph.nodes[best_node_id]
            candidate_screenshot = candidate_data.get("reference_screenshot")

            if candidate_screenshot:
                verify_result = verify_same_node(
                    self._require_page_detail_client(),
                    screenshot_new_b64=screenshot_b64,
                    screenshot_existing_b64=candidate_screenshot,
                    existing_description=candidate_data.get("page_description", ""),
                    model=self.page_detail_model,
                )
                if verify_result.get("same", False):
                    matched_node_id = best_node_id
                    logger.info(
                        "Verifier confirmed match node %s (sim=%.3f): %s | reason: %s",
                        best_node_id,
                        best_similarity,
                        candidate_data.get("page_description"),
                        verify_result.get("reason", ""),
                    )
                else:
                    refined_existing = verify_result.get("existing_description", "")
                    refined_new = verify_result.get("new_description", "")

                    if refined_existing and refined_existing != candidate_data.get(
                        "page_description"
                    ):
                        best_node_id = self.rename_node(best_node_id, refined_existing)
                        candidate_data = self.graph.nodes[best_node_id]

                    if refined_new:
                        page_description = refined_new
                        description_embedding = get_embedding(
                            self._require_embedding_client(),
                            page_description,
                            model=self.embedding_model,
                        )

                    logger.info(
                        "Verifier rejected match node %s (sim=%.3f): reason: %s | new: %s",
                        best_node_id,
                        best_similarity,
                        verify_result.get("reason", ""),
                        page_description,
                    )
            else:
                matched_node_id = best_node_id
                logger.info(
                    "Matched node %s by embedding (sim=%.3f, no ref screenshot): %s",
                    best_node_id,
                    best_similarity,
                    candidate_data.get("page_description"),
                )

        if matched_node_id is not None:
            node_id = matched_node_id
            node_data = self.graph.nodes[node_id]
            activities = node_data.get("activities", [node_data.get("activity", "")])
            if activity not in activities:
                activities.append(activity)
                logger.info(
                    "  Node %s: added activity %s (now %d)", node_id, activity, len(activities)
                )
            node_data["activities"] = activities
        else:
            node_id = self._make_node_id(page_description)
            init_elements = [
                {
                    "description": e.get("description", ""),
                    "position": e.get("position", ""),
                    "explored": False,
                }
                for e in elements
            ]
            self.graph.add_node(
                node_id,
                activity=activity,
                activities=[activity],
                page_description=page_description,
                state_schema={},
                interactable_elements=init_elements,
                description_embedding=description_embedding,
                reference_screenshot=screenshot_b64,
                last_normalized_visit_milestone=0,
                visit_count=0,
            )
            logger.info(
                "Created new node %s: %s (%d elements)",
                node_id,
                page_description,
                len(init_elements),
            )

        node_data = self.graph.nodes[node_id]
        schema = node_data.get("state_schema", {})
        _merge_into_schema(schema, detail_snapshot)
        node_data["state_schema"] = schema
        node_data["last_detail_snapshot"] = detail_snapshot
        node_data["visit_count"] = node_data.get("visit_count", 0) + 1
        # Structural elements accumulate across visits, so merge rather than replace.
        _merge_elements(node_data, elements)
        # Only update reference screenshot for new nodes or verifier-rejected splits;
        # keep the existing screenshot when verifier confirmed a match.
        if matched_node_id is None:
            node_data["reference_screenshot"] = screenshot_b64
            self._dirty_screenshots.add(node_id)

        logger.info(
            "Node %s schema updated: %d keys, visit #%d",
            node_id,
            len(schema),
            node_data["visit_count"],
        )
        self._last_identify_cache = _IdentifyCacheEntry(screen_hash, activity, node_id)
        return node_id

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
        if "_" in node_id:
            prefix = node_id.split("_", 1)[0]
        else:
            prefix = node_id
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
        node_data["description_embedding"] = get_embedding(
            self._require_embedding_client(), page_description, model=self.embedding_model
        )

        if new_node_id != node_id:
            nx.relabel_nodes(self.graph, {node_id: new_node_id}, copy=False)
            if node_id in self._dirty_screenshots:
                self._dirty_screenshots.discard(node_id)
            if self.graph.nodes[new_node_id].get("reference_screenshot"):
                self._dirty_screenshots.add(new_node_id)
            cached = self._last_identify_cache
            if cached is not None and cached.node_id == node_id:
                self._last_identify_cache = cached._replace(node_id=new_node_id)
        else:
            if node_data.get("reference_screenshot"):
                self._dirty_screenshots.add(node_id)

        logger.info(
            "Renamed node %s → %s: '%s' → '%s'",
            node_id,
            new_node_id,
            old_description,
            page_description,
        )
        return new_node_id

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

    def mark_element_explored(self, node_id: str, instruction: str) -> None:
        """Mark the element that best matches the instruction as explored.

        Uses simple substring matching between the instruction and element
        descriptions.  The best-matching element (most words in common) is
        marked explored=True.
        """
        if node_id not in self.graph:
            return
        elements = self.graph.nodes[node_id].get("interactable_elements", [])
        if not elements:
            return

        instr_lower = instruction.lower()
        best_idx = -1
        best_score = 0
        for idx, elem in enumerate(elements):
            if elem.get("explored", False):
                continue
            desc_words = elem.get("description", "").lower().split()
            if not desc_words:
                continue
            score = sum(1 for w in desc_words if w in instr_lower)
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx >= 0 and best_score > 0:
            elements[best_idx]["explored"] = True
            logger.info(
                "Element explored on %s: '%s' (matched by instruction: '%s')",
                node_id,
                elements[best_idx]["description"],
                instruction,
            )

    def merge_nodes(self, keep_id: str, remove_id: str) -> bool:
        """Merge *remove_id* into *keep_id*, rewiring all edges.

        - Incoming edges to *remove_id* are rewired to *keep_id*.
        - Outgoing edges from *remove_id* are rewired from *keep_id*.
        - State schemas are merged (union of keys and values).
        - Visit counts are summed.
        - The kept node retains its description, screenshot, and embedding.
        - The removed node is deleted from the graph.

        Returns True if the merge was performed, False if either node is missing
        or if *keep_id* and *remove_id* name the same node.
        """
        if keep_id not in self.graph or remove_id not in self.graph:
            logger.warning("merge_nodes: missing node(s) — keep=%s, remove=%s", keep_id, remove_id)
            return False

        # Falling through is not a harmless no-op: it deletes the node.  See #48.
        if keep_id == remove_id:
            logger.warning("merge_nodes: refusing to merge node %s into itself", keep_id)
            return False

        keep_data = self.graph.nodes[keep_id]
        remove_data = self.graph.nodes[remove_id]

        keep_data["visit_count"] = keep_data.get("visit_count", 0) + remove_data.get(
            "visit_count", 0
        )

        keep_activities = keep_data.get("activities", [keep_data.get("activity", "")])
        remove_activities = remove_data.get("activities", [remove_data.get("activity", "")])
        for act in remove_activities:
            if act and act not in keep_activities:
                keep_activities.append(act)
        keep_data["activities"] = keep_activities

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

        _merge_elements(keep_data, remove_data.get("interactable_elements", []))

        for pred in list(self.graph.predecessors(remove_id)):
            if pred == remove_id:
                continue  # handle self-loops separately
            edge_data = dict(self.graph[pred][remove_id])
            self.graph.remove_edge(pred, remove_id)
            if pred == keep_id:
                self._merge_edge_data(keep_id, keep_id, edge_data)
            else:
                self._merge_edge_data(pred, keep_id, edge_data)

        for succ in list(self.graph.successors(remove_id)):
            if succ == remove_id:
                continue  # handle self-loops separately
            edge_data = dict(self.graph[remove_id][succ])
            self.graph.remove_edge(remove_id, succ)
            if succ == keep_id:
                self._merge_edge_data(keep_id, keep_id, edge_data)
            else:
                self._merge_edge_data(keep_id, succ, edge_data)

        if self.graph.has_edge(remove_id, remove_id):
            edge_data = dict(self.graph[remove_id][remove_id])
            self.graph.remove_edge(remove_id, remove_id)
            self._merge_edge_data(keep_id, keep_id, edge_data)

        self.graph.remove_node(remove_id)

        logger.info(
            "Merged node %s into %s (now %d nodes, %d edges)",
            remove_id,
            keep_id,
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
        )
        return True

    def _merge_edge_data(self, source: str, target: str, new_data: dict[str, Any]) -> None:
        """Merge edge data into an existing edge, or create it if absent."""
        if self.graph.has_edge(source, target):
            existing = self.graph[source][target]
            for key in (
                "actions",
                "instructions",
                "target_observations",
                "num_steps",
                "schema_deltas",
            ):
                if key in new_data:
                    existing.setdefault(key, []).extend(new_data[key])
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
        instruction: str | None = None,
        target_observation: str | None = None,
        schema_delta: dict[str, Any] | None = None,
        num_steps: int | None = None,
    ) -> None:
        """Add a directed edge (action) from source to target.

        Args:
            action: A single AITK action dict, or a list of action dicts
                (compound action sequence, e.g. [tap, type, enter]).
            instruction: The natural language instruction that produced this action.
            target_observation: Description of the screen state after performing
                this action. Helps the runtime model predict what will happen.
            schema_delta: For self-loop edges (source == target), records which
                schema keys changed and their before/after values.
                Format: {"key": {"before": old_val, "after": new_val}, ...}
            num_steps: Number of action steps this edge takes to traverse.
                If None, defaults to len(action) for lists or 1 for single actions.

        If an edge with the same action already exists, it is not duplicated.
        Instead, a visit counter is incremented.

        The edge is dropped with a warning when either endpoint is missing.
        """
        # networkx would otherwise create the missing endpoint bare.  See #62.
        if source not in self.graph or target not in self.graph:
            logger.warning("add_edge: missing node(s) — source=%s, target=%s", source, target)
            return

        if num_steps is None:
            num_steps = len(action) if isinstance(action, list) else 1

        if self.graph.has_edge(source, target):
            edge_data = self.graph[source][target]
            existing_actions = edge_data.get("actions", [])
            for a in existing_actions:
                if a == action:
                    edge_data["visit_count"] = edge_data.get("visit_count", 1) + 1
                    return
            existing_actions.append(action)
            edge_data["actions"] = existing_actions
            edge_data.setdefault("num_steps", []).append(num_steps)
            if instruction:
                edge_data.setdefault("instructions", []).append(instruction)
            if target_observation:
                edge_data.setdefault("target_observations", []).append(target_observation)
            if schema_delta:
                edge_data.setdefault("schema_deltas", []).append(schema_delta)
        else:
            edge_data = {
                "actions": [action],
                "num_steps": [num_steps],
                "visit_count": 1,
            }
            if instruction:
                edge_data["instructions"] = [instruction]
            if target_observation:
                edge_data["target_observations"] = [target_observation]
            if schema_delta:
                edge_data["schema_deltas"] = [schema_delta]
            self.graph.add_edge(source, target, **edge_data)

        edge_data = self.graph[source][target]
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

    def _normalize_single_edge(self, source: str, target: str) -> bool:
        """Normalize one edge's instructions into a reusable template if possible."""
        if self.page_detail_client is None:
            return False

        edge_data = self.graph[source][target]
        instructions = edge_data.get("instructions", [])
        observations = edge_data.get("target_observations", [])

        if not instructions:
            return False

        existing_templates = edge_data.get("instruction_templates", [])
        if existing_templates:
            examples = (
                existing_templates[0].get("examples", [])
                if isinstance(existing_templates[0], dict)
                else []
            )
            if len(examples) >= len(instructions):
                return False

        if len(instructions) == 1:
            instr = instructions[0]
            # Heuristic: if no quoted values, numbers, or proper nouns, unlikely to be parameterizable
            has_variable = any(c.isdigit() for c in instr) or '"' in instr or "'" in instr
            if not has_variable and len(instr.split()) <= 6:
                edge_data["instruction_templates"] = []
                return False

        result = normalize_edge(
            self.page_detail_client,
            instructions,
            target_observations=observations or None,
            model=self.page_detail_model,
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
        if self.page_detail_client is None or node_id not in self.graph:
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
        if self.page_detail_client is None:
            logger.warning("No page_detail_client — skipping normalization")
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
                "instructions": data.get("instructions", []),
                "instruction_templates": data.get("instruction_templates", []),
                "target_observations": data.get("target_observations", []),
                "visit_count": data.get("visit_count", 0),
            }
            schema_deltas = data.get("schema_deltas")
            if schema_deltas:
                edge_info["schema_deltas"] = schema_deltas
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

    def get_start_node(self) -> str | None:
        """Return the initial node (the app's entry point).

        Heuristic: the node with the lowest numeric ID suffix (s0_...).
        """
        candidates: list[tuple[int, str]] = []
        for node_id, data in self.graph.nodes(data=True):
            if node_id.startswith("ext_") or data.get("is_external"):
                continue
            if node_id.startswith("s") and "_" in node_id:
                num_part = node_id.split("_", 1)[0][1:]
                if num_part.isdigit():
                    candidates.append((int(num_part), _node_id(node_id)))

        if candidates:
            return min(candidates)[1]

        nodes = sorted(
            _node_id(n)
            for n, data in self.graph.nodes(data=True)
            if not n.startswith("ext_") and not data.get("is_external")
        )
        return nodes[0] if nodes else None

    def get_least_explored_node(self, package_name: str | None = None) -> str | None:
        """Return the node with the fewest outgoing edges (least explored).

        Only considers nodes reachable from the start node.
        Excludes external app nodes (ext_*) and nodes whose package
        doesn't match the given package_name (if provided).
        """
        start = self.get_start_node()
        if start is None:
            return None

        reachable = nx.descendants(self.graph, start) | {start}
        best_node: str | None = None
        min_out_degree = float("inf")

        for node in reachable:
            if node.startswith("ext_"):
                continue
            if self.graph.nodes[node].get("is_external"):
                continue
            if package_name:
                activity = self.graph.nodes[node].get("activity", "")
                node_pkg = _package_from_activity(activity) if activity else ""
                if node_pkg and node_pkg != package_name:
                    continue
            out_degree = self.graph.out_degree(node)
            if out_degree < min_out_degree:
                min_out_degree = out_degree
                best_node = _node_id(node)

        return best_node

    def get_exploration_target_candidates(
        self,
        package_name: str | None = None,
        top_k: int = 15,
    ) -> list[dict[str, Any]]:
        """Return reachable nodes that are good candidates for continued exploration.

        This is a lightweight coverage signal for live exploration checkpoints.
        It favors nodes with many unexplored elements, few outgoing edges, and
        lower visit counts.
        """
        start = self.get_start_node()
        if start is None:
            return []

        reachable = nx.descendants(self.graph, start) | {start}
        candidates: list[dict[str, Any]] = []

        for node_id in reachable:
            node_data = self.graph.nodes[node_id]
            if node_id.startswith("ext_") or node_data.get("is_external"):
                continue

            if package_name:
                activity = node_data.get("activity", "")
                node_pkg = _package_from_activity(activity) if activity else ""
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
        return candidates[:top_k]

    @staticmethod
    def _embeddings_path(graph_path: Path) -> Path:
        """Return the companion embeddings file path for a graph JSON."""
        return graph_path.with_suffix(".emb.json")

    def save_graph(self, path: str | Path) -> None:
        """Save graph to a JSON file, with embeddings in a separate file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "next_id": self._next_id,
            "similarity_threshold": self.similarity_threshold,
            "total_steps_completed": self.total_steps_completed,
            "nodes": [],
            "edges": [],
        }

        embeddings: dict[str, list[float]] = {}

        for node_id, attrs in self.graph.nodes(data=True):
            activities = attrs.get("activities", [attrs.get("activity", "")])
            node_data = {
                "id": node_id,
                "activity": activities[0] if activities else "",
                "activities": activities,
                "page_description": attrs.get("page_description", ""),
                "state_schema": attrs.get("state_schema", {}),
                "last_detail_snapshot": attrs.get("last_detail_snapshot", {}),
                "interactable_elements": attrs.get("interactable_elements", []),
                "last_normalized_visit_milestone": attrs.get("last_normalized_visit_milestone", 0),
                "visit_count": attrs.get("visit_count", 0),
            }
            data["nodes"].append(node_data)

            emb = attrs.get("description_embedding")
            if emb:
                embeddings[node_id] = emb

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

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        emb_path = self._embeddings_path(path)
        with open(emb_path, "w", encoding="utf-8") as f:
            json.dump(embeddings, f, ensure_ascii=False)

        if self._dirty_screenshots:
            screenshots_dir = path.parent / (path.stem + "_screenshots")
            screenshots_dir.mkdir(parents=True, exist_ok=True)
            for node_id in self._dirty_screenshots:
                if node_id in self.graph:
                    ref_b64 = self.graph.nodes[node_id].get("reference_screenshot")
                    if ref_b64:
                        img_path = screenshots_dir / f"{node_id}.png"
                        img_path.write_bytes(base64.b64decode(ref_b64))
            self._dirty_screenshots.clear()

        logger.info(
            "Graph saved to %s (%d nodes, %d edges)", path, len(data["nodes"]), len(data["edges"])
        )

    def load_graph(self, path: str | Path) -> None:
        """Load graph from a JSON file (and companion embeddings file)."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Before any mutation: a corrupt file must not leave a half-loaded graph.
        _require_known_edge_endpoints(data, path)

        # Old graphs stored embeddings inline; newer ones keep them in a companion file.
        emb_path = self._embeddings_path(path)
        embeddings: dict[str, list[float]] = {}
        if emb_path.exists():
            with open(emb_path, "r", encoding="utf-8") as f:
                embeddings = {
                    node_id: as_float_list(vector)
                    for node_id, vector in as_str_dict(json.load(f)).items()
                }

        self._next_id = data.get("next_id", 0)
        self.similarity_threshold = data.get("similarity_threshold", DEFAULT_SIMILARITY_THRESHOLD)
        self.total_steps_completed = data.get("total_steps_completed", 0)
        self.graph.clear()

        screenshots_dir = path.parent / (path.stem + "_screenshots")

        for node_data in data.get("nodes", []):
            node_id = node_data["id"]
            # Prefer companion file; fall back to inline (backwards compat)
            emb = embeddings.get(node_id, node_data.get("description_embedding", []))
            # Backwards compat: old graphs have "activity" only, new ones have "activities" list
            activities = node_data.get("activities", [node_data.get("activity", "")])
            ref_screenshot = None
            img_path = screenshots_dir / f"{node_id}.png"
            if img_path.exists():
                ref_screenshot = base64.b64encode(img_path.read_bytes()).decode("ascii")
            self.graph.add_node(
                node_id,
                activity=activities[0] if activities else "",
                activities=activities,
                page_description=node_data.get("page_description", ""),
                state_schema=node_data.get("state_schema", {}),
                last_detail_snapshot=node_data.get("last_detail_snapshot", {}),
                interactable_elements=node_data.get("interactable_elements", []),
                description_embedding=emb,
                reference_screenshot=ref_screenshot,
                last_normalized_visit_milestone=node_data.get("last_normalized_visit_milestone", 0),
                visit_count=node_data.get("visit_count", 0),
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
        query_emb = get_embedding(
            self._require_embedding_client(), query, model=self.embedding_model
        )
        results: list[tuple[str, float]] = []
        for node_id, data in self.graph.nodes(data=True):
            emb = data.get("description_embedding")
            if emb is None:
                continue
            sim = _cosine_similarity(query_emb, emb)
            results.append((_node_id(node_id), sim))
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
                obs = observations[i] if i < len(observations) else ""
                obs_str = f" → {obs}" if obs else ""
                lines.append(f'    - "{instr}"{obs_str}')

        return "\n".join(lines)

    def run_audit(self, app_name: str = "") -> dict[str, Any]:
        """Run LLM audit on the graph structure.

        Returns the audit result dict with "issues" and "summary".
        """
        if self.page_detail_client is None:
            logger.warning("No page_detail_client — cannot audit")
            return {"issues": [], "summary": "no client available"}

        graph_text = self.format_for_audit()
        logger.info("Running graph audit (%d chars)...", len(graph_text))

        result = audit_graph(
            self.page_detail_client,
            graph_text,
            app_name=app_name,
            model=self.page_detail_model,
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

    @staticmethod
    def _node_numeric_order(node_id: str) -> tuple[int, str]:
        if node_id.startswith("s") and "_" in node_id:
            num = node_id.split("_", 1)[0][1:]
            if num.isdigit():
                return int(num), node_id
        return 10**9, node_id

    def run_node_merge_audit(self, app_name: str = "") -> dict[str, Any]:
        """Run a lightweight node-merge audit and merge verified duplicates."""
        if self.page_detail_client is None:
            logger.warning("No page_detail_client — cannot run node merge audit")
            return {"issues": [], "results": [], "merged_count": 0}

        graph_text = self.format_for_audit()
        logger.info("Running node merge audit (%d chars)...", len(graph_text))
        audit_result = audit_merge_nodes(
            self.page_detail_client,
            graph_text,
            app_name=app_name,
            model=self.page_detail_model,
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

            data_a = self.graph.nodes[node_a]
            data_b = self.graph.nodes[node_b]
            screenshot_a = data_a.get("reference_screenshot")
            screenshot_b = data_b.get("reference_screenshot")
            if not screenshot_a or not screenshot_b:
                results.append(
                    {
                        "issue": issue,
                        "status": "skipped",
                        "reason": "missing screenshot",
                    }
                )
                continue

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
                self.page_detail_client,
                screenshot_new_b64=screenshot_b,
                screenshot_existing_b64=screenshot_a,
                existing_description=ref_desc,
                model=self.page_detail_model,
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
