"""Tests for ui_kobe.utils.graph_manager.

All VLM and embedding calls are replaced by scripted fakes patched onto the
graph_manager module namespace; no test needs a device, the network or keys.
"""

from __future__ import annotations

import base64
import json
import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import networkx as nx
import pytest
from hypothesis import given
from hypothesis import strategies as st
from openai import OpenAI

from ui_kobe.utils import graph_manager as gm_module
from ui_kobe.utils.graph_manager import (
    DEFAULT_SIMILARITY_THRESHOLD,
    NORMALIZE_EVERY_N_VISITS,
    GraphManager,
    _cosine_similarity,
    _merge_elements,
    _merge_into_schema,
    _package_from_activity,
)

HOME = "com.example.app.HomeActivity"
SETTINGS = "com.example.app.SettingsActivity"
OTHER_APP = "org.other.tool.MainActivity"

# Unit vectors: cosine similarity is exactly 1.0 or 0.0.
EMBEDDINGS: dict[str, list[float]] = {
    "Home screen": [1.0, 0.0, 0.0],
    "Home screen with banner": [1.0, 0.0, 0.0],
    "Home feed": [1.0, 0.0, 0.0],
    "Promo overlay": [0.0, 1.0, 0.0],
    "Settings page": [0.0, 1.0, 0.0],
    "Search results": [0.0, 0.0, 1.0],
}
UNKNOWN_EMBEDDING = [0.5, 0.5, 0.5]

FAKE_CLIENT = cast("OpenAI", object())  # never used: every consumer is patched


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


SHOT_A = b64(b"screenshot-a")
SHOT_B = b64(b"screenshot-b")


def make_manager(similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD) -> GraphManager:
    """A manager whose two clients are sentinels; every caller of them is patched."""
    return GraphManager(
        page_detail_client=FAKE_CLIENT,
        embedding_client=FAKE_CLIENT,
        similarity_threshold=similarity_threshold,
    )


def add_screen(
    gm: GraphManager,
    node_id: str,
    description: str,
    activity: str = HOME,
    **overrides: Any,
) -> None:
    """Insert a node with the attribute set identify_state would create."""
    attrs: dict[str, Any] = {
        "activity": activity,
        "activities": [activity],
        "page_description": description,
        "state_schema": {},
        "last_detail_snapshot": {},
        "interactable_elements": [],
        "description_embedding": EMBEDDINGS.get(description, UNKNOWN_EMBEDDING),
        "reference_screenshot": None,
        "last_normalized_visit_milestone": 0,
        "visit_count": 0,
    }
    attrs.update(overrides)
    gm.graph.add_node(node_id, **attrs)


@dataclass
class FakeVlm:
    """Scripted vlm_utils stand-ins; every parameter is recorded, so none is unused."""

    descriptions: list[tuple[str, dict[str, Any], list[dict[str, str]]]] = field(
        default_factory=list
    )
    verdicts: list[dict[str, Any]] = field(default_factory=list)
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def describe_page_and_state(
        self,
        client: OpenAI | None,
        screenshot_b64: str,
        existing_nodes: list[str] | None = None,
        existing_keys: list[str] | None = None,
        model: str = "",
    ) -> tuple[str, dict[str, Any], list[dict[str, str]]]:
        call = {"client": client, "screenshot": screenshot_b64, "model": model}
        call.update(existing_nodes=existing_nodes, existing_keys=existing_keys)
        self.calls.append(("describe", call))
        return self.descriptions.pop(0)

    def verify_same_node(
        self,
        client: OpenAI | None,
        screenshot_new_b64: str,
        screenshot_existing_b64: str,
        existing_description: str,
        model: str = "",
    ) -> dict[str, Any]:
        call = {"client": client, "new": screenshot_new_b64, "existing": screenshot_existing_b64}
        call.update(existing_description=existing_description, model=model)
        self.calls.append(("verify", call))
        return self.verdicts.pop(0)

    def get_embedding(self, client: OpenAI | None, text: str, model: str = "") -> list[float]:
        self.calls.append(("embed", {"client": client, "text": text, "model": model}))
        return EMBEDDINGS.get(text, UNKNOWN_EMBEDDING)

    def kinds(self) -> list[str]:
        return [kind for kind, _ in self.calls]


@dataclass
class FakeAudit:
    """Stand-in for audit_graph and audit_merge_nodes; both share this signature."""

    result: dict[str, Any] = field(default_factory=dict)
    calls: list[dict[str, Any]] = field(default_factory=list)

    def __call__(
        self,
        client: OpenAI | None,
        graph_summary: str,
        app_name: str = "",
        model: str = "",
    ) -> dict[str, Any]:
        call = {"client": client, "graph_summary": graph_summary}
        call.update(app_name=app_name, model=model)
        self.calls.append(call)
        return self.result


@pytest.fixture
def vlm(monkeypatch: pytest.MonkeyPatch) -> FakeVlm:
    fake = FakeVlm()
    monkeypatch.setattr(gm_module, "describe_page_and_state", fake.describe_page_and_state)
    monkeypatch.setattr(gm_module, "verify_same_node", fake.verify_same_node)
    monkeypatch.setattr(gm_module, "get_embedding", fake.get_embedding)
    return fake


# Suite guard


def test_network_guard_blocks_outbound_connections() -> None:
    with pytest.raises(RuntimeError, match="network access is disabled"):
        socket.create_connection(("127.0.0.1", 9), timeout=0.1)


# Module helpers


def test_package_from_activity_keeps_the_first_three_segments() -> None:
    assert _package_from_activity("com.citymapper.app.home.HomeActivity2") == "com.citymapper.app"


def test_package_from_activity_drops_the_component_after_a_slash() -> None:
    activity = "com.citymapper.app/com.citymapper.app.MainActivity"
    assert _package_from_activity(activity) == "com.citymapper.app"


def test_package_from_activity_returns_short_names_unchanged() -> None:
    assert _package_from_activity("com.example") == "com.example"


def test_package_from_activity_returns_an_empty_string_unchanged() -> None:
    assert _package_from_activity("") == ""


def test_cosine_similarity_of_parallel_vectors_is_one() -> None:
    assert _cosine_similarity([1.0, 0.0], [2.0, 0.0]) == pytest.approx(1.0)


def test_cosine_similarity_of_orthogonal_vectors_is_zero() -> None:
    assert _cosine_similarity([1.0, 0.0], [0.0, 3.0]) == pytest.approx(0.0)


def test_cosine_similarity_with_a_zero_vector_is_zero() -> None:
    assert _cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_cosine_similarity_with_a_zero_second_vector_is_zero() -> None:
    assert _cosine_similarity([1.0, 1.0], [0.0, 0.0]) == 0.0


def test_merge_into_schema_adds_a_new_key_as_a_single_element_list() -> None:
    assert _merge_into_schema({}, {"sort": "price"}) == {"sort": ["price"]}


def test_merge_into_schema_appends_only_unseen_values() -> None:
    schema: dict[str, list[Any]] = {"sort": ["price"]}
    _merge_into_schema(schema, {"sort": "price"})
    _merge_into_schema(schema, {"sort": "rating"})
    assert schema == {"sort": ["price", "rating"]}


def test_merge_into_schema_mutates_and_returns_the_same_dict() -> None:
    schema: dict[str, list[Any]] = {}
    assert _merge_into_schema(schema, {"query": None}) is schema


def test_merge_elements_appends_new_elements_as_unexplored() -> None:
    node_data: dict[str, Any] = {"interactable_elements": []}
    _merge_elements(node_data, [{"description": "Search box", "position": "10,20"}])
    assert node_data["interactable_elements"] == [
        {"description": "Search box", "position": "10,20", "explored": False}
    ]


def test_merge_elements_deduplicates_case_insensitively() -> None:
    node_data: dict[str, Any] = {
        "interactable_elements": [
            {"description": "Search box", "position": "10,20", "explored": True}
        ]
    }
    _merge_elements(node_data, [{"description": "search BOX", "position": "99,99"}])
    assert node_data["interactable_elements"] == [
        {"description": "Search box", "position": "10,20", "explored": True}
    ]


def test_merge_elements_defaults_a_missing_position_to_an_empty_string() -> None:
    node_data: dict[str, Any] = {}
    _merge_elements(node_data, [{"description": "Menu"}])
    assert node_data["interactable_elements"][0]["position"] == ""


# Construction


def test_construction_defaults() -> None:
    gm = GraphManager()
    assert gm.graph.number_of_nodes() == 0
    assert gm.page_detail_client is None
    assert gm.embedding_client is None
    assert gm.similarity_threshold == DEFAULT_SIMILARITY_THRESHOLD
    assert gm.total_steps_completed == 0


def test_identify_state_without_a_page_detail_client_raises() -> None:
    gm = GraphManager()
    with pytest.raises(RuntimeError, match="page_detail_client"):
        gm.identify_state(HOME, SHOT_A)


def test_identify_state_without_an_embedding_client_raises(vlm: FakeVlm) -> None:
    vlm.descriptions.append(("Home screen", {}, []))
    gm = GraphManager(page_detail_client=FAKE_CLIENT)
    with pytest.raises(RuntimeError, match="embedding_client"):
        gm.identify_state(HOME, SHOT_A)


# identify_state


def test_identify_state_creates_the_first_node(vlm: FakeVlm) -> None:
    vlm.descriptions.append(
        ("Home screen", {"tab": "home"}, [{"description": "Search box", "position": "10,20"}])
    )
    gm = make_manager()

    node_id = gm.identify_state(HOME, SHOT_A)

    assert node_id == "s0_home_screen"
    assert vlm.kinds() == ["describe", "embed"]
    node = gm.get_node(node_id)
    assert node is not None
    assert node["visit_count"] == 1
    assert node["state_schema"] == {"tab": ["home"]}
    assert node["reference_screenshot"] == SHOT_A
    assert node["interactable_elements"] == [
        {"description": "Search box", "position": "10,20", "explored": False}
    ]


def test_identify_state_passes_the_configured_models_and_clients(vlm: FakeVlm) -> None:
    vlm.descriptions.append(("Home screen", {}, []))
    gm = make_manager()

    gm.identify_state(HOME, SHOT_A)

    assert vlm.calls[0][1]["model"] == gm.page_detail_model
    assert vlm.calls[0][1]["client"] is FAKE_CLIENT
    assert vlm.calls[1][1]["model"] == gm.embedding_model


def test_identify_state_cache_hit_skips_the_vlm(vlm: FakeVlm) -> None:
    vlm.descriptions.append(("Home screen", {}, []))
    gm = make_manager()
    first = gm.identify_state(HOME, SHOT_A)

    second = gm.identify_state(HOME, SHOT_A)

    assert second == first
    assert vlm.kinds() == ["describe", "embed"]


def test_identify_state_verified_match_merges_into_existing_node(vlm: FakeVlm) -> None:
    vlm.descriptions.extend(
        [("Home screen", {"tab": "home"}, []), ("Home screen with banner", {"tab": "deals"}, [])]
    )
    vlm.verdicts.append({"same": True, "reason": "identical layout"})
    gm = make_manager()
    first = gm.identify_state(HOME, SHOT_A)

    second = gm.identify_state(SETTINGS, SHOT_B)

    assert second == first
    assert vlm.kinds() == ["describe", "embed", "describe", "embed", "verify"]
    assert vlm.calls[2][1]["existing_nodes"] == ["Home screen"]
    assert vlm.calls[2][1]["existing_keys"] == ["tab"]
    node = gm.get_node(first)
    assert node is not None
    assert node["activities"] == [HOME, SETTINGS]
    assert node["visit_count"] == 2
    assert node["state_schema"] == {"tab": ["home", "deals"]}
    assert node["reference_screenshot"] == SHOT_A  # kept on a confirmed match


def test_identify_state_verified_match_does_not_repeat_a_known_activity(vlm: FakeVlm) -> None:
    vlm.descriptions.extend([("Home screen", {}, []), ("Home screen with banner", {}, [])])
    vlm.verdicts.append({"same": True, "reason": "identical layout"})
    gm = make_manager()
    first = gm.identify_state(HOME, SHOT_A)

    gm.identify_state(HOME, SHOT_B)

    node = gm.get_node(first)
    assert node is not None
    assert node["activities"] == [HOME]


def test_identify_state_rejected_match_renames_existing_and_creates_new(vlm: FakeVlm) -> None:
    vlm.descriptions.extend([("Home screen", {}, []), ("Home screen with banner", {}, [])])
    vlm.verdicts.append(
        {
            "same": False,
            "reason": "overlay",
            "existing_description": "Home feed",
            "new_description": "Promo overlay",
        }
    )
    gm = make_manager()
    first = gm.identify_state(HOME, SHOT_A)

    second = gm.identify_state(HOME, SHOT_B)

    assert first not in gm.graph
    assert "s0_home_feed" in gm.graph
    assert second == "s1_promo_overlay"
    assert vlm.kinds() == ["describe", "embed", "describe", "embed", "verify", "embed", "embed"]
    renamed = gm.get_node("s0_home_feed")
    assert renamed is not None
    assert renamed["page_description"] == "Home feed"


def test_identify_state_rejected_match_without_refinements_creates_new(vlm: FakeVlm) -> None:
    vlm.descriptions.extend([("Home screen", {}, []), ("Home screen with banner", {}, [])])
    vlm.verdicts.append(
        {
            "same": False,
            "reason": "different",
            "existing_description": "Home screen",
            "new_description": "",
        }
    )
    gm = make_manager()
    first = gm.identify_state(HOME, SHOT_A)

    second = gm.identify_state(HOME, SHOT_B)

    assert first in gm.graph  # no rename: the verifier repeated the existing description
    assert second == "s1_home_screen_with_banner"
    assert vlm.kinds() == ["describe", "embed", "describe", "embed", "verify"]


def test_identify_state_matches_by_embedding_when_the_candidate_has_no_screenshot(
    vlm: FakeVlm,
) -> None:
    vlm.descriptions.append(("Home screen", {}, []))
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen")

    node_id = gm.identify_state(HOME, SHOT_A)

    assert node_id == "s0_home"
    assert vlm.kinds() == ["describe", "embed"]
    node = gm.get_node(node_id)
    assert node is not None
    assert node["reference_screenshot"] is None  # only new nodes store a screenshot


def test_identify_state_ignores_nodes_of_another_package(vlm: FakeVlm) -> None:
    vlm.descriptions.append(("Home screen", {}, []))
    gm = make_manager()
    add_screen(gm, "s0_other", "Home screen", activity=OTHER_APP)

    node_id = gm.identify_state(HOME, SHOT_A)

    assert node_id == "s0_home_screen"
    assert vlm.calls[0][1]["existing_nodes"] is None
    assert vlm.calls[0][1]["existing_keys"] is None


def test_identify_state_skips_candidates_without_an_embedding(vlm: FakeVlm) -> None:
    vlm.descriptions.append(("Home screen", {}, []))
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen", description_embedding=None)

    node_id = gm.identify_state(HOME, SHOT_A)

    assert node_id == "s0_home_screen"


def test_identify_state_below_the_threshold_creates_a_new_node(vlm: FakeVlm) -> None:
    vlm.descriptions.append(("Settings page", {}, []))
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen", reference_screenshot=SHOT_B)

    node_id = gm.identify_state(HOME, SHOT_A)

    assert node_id == "s0_settings_page"
    assert vlm.kinds() == ["describe", "embed"]


def test_identify_state_sanitizes_and_truncates_the_node_id(vlm: FakeVlm) -> None:
    description = "Home / dashboard screen showing every available promotion"
    vlm.descriptions.append((description, {}, []))
    gm = make_manager()

    node_id = gm.identify_state(HOME, SHOT_A)

    assert node_id == "s0_home___dashboard_screen_showing_every_av"
    assert len(node_id.split("_", 1)[1]) == 40


# rename_node


def test_rename_node_returns_a_missing_id_unchanged() -> None:
    gm = make_manager()
    assert gm.rename_node("s9_missing", "Anything") == "s9_missing"


def test_rename_node_keeps_the_numeric_prefix_and_relabels_edges(vlm: FakeVlm) -> None:
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen")
    add_screen(gm, "s1_settings", "Settings page")
    gm.add_edge("s0_home", "s1_settings", {"action": "tap"})

    new_id = gm.rename_node("s0_home", "Home feed")

    assert new_id == "s0_home_feed"
    assert gm.graph.has_edge("s0_home_feed", "s1_settings")
    assert vlm.kinds() == ["embed"]
    node = gm.get_node(new_id)
    assert node is not None
    assert node["description_embedding"] == EMBEDDINGS["Home feed"]


def test_rename_node_suffixes_the_id_on_collision(vlm: FakeVlm) -> None:
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen")
    add_screen(gm, "s0_settings_page", "Settings page")
    add_screen(gm, "s0_settings_page_2", "Settings page")

    assert gm.rename_node("s0_home", "Settings page") == "s0_settings_page_3"
    assert vlm.kinds() == ["embed"]


def test_rename_node_without_an_underscore_uses_the_whole_id_as_prefix(vlm: FakeVlm) -> None:
    gm = make_manager()
    add_screen(gm, "start", "Home screen")

    assert gm.rename_node("start", "Home feed") == "start_home_feed"
    assert vlm.kinds() == ["embed"]


def test_rename_node_to_the_same_id_keeps_the_screenshot_dirty(
    tmp_path: Path, vlm: FakeVlm
) -> None:
    gm = make_manager()
    add_screen(gm, "s0_home_screen", "Home screen", reference_screenshot=SHOT_A)

    assert gm.rename_node("s0_home_screen", "Home screen") == "s0_home_screen"
    assert vlm.kinds() == ["embed"]

    gm.save_graph(tmp_path / "graph.json")
    assert (tmp_path / "graph_screenshots" / "s0_home_screen.png").read_bytes() == b"screenshot-a"


def test_rename_node_moves_the_screenshot_to_the_new_id(tmp_path: Path, vlm: FakeVlm) -> None:
    vlm.descriptions.append(("Home screen", {}, []))
    gm = make_manager()
    gm.identify_state(HOME, SHOT_A)

    new_id = gm.rename_node("s0_home_screen", "Home feed")

    gm.save_graph(tmp_path / "graph.json")
    screenshots = tmp_path / "graph_screenshots"
    assert (screenshots / f"{new_id}.png").is_file()
    assert not (screenshots / "s0_home_screen.png").exists()


def test_rename_node_updates_the_identify_cache(vlm: FakeVlm) -> None:
    vlm.descriptions.append(("Home screen", {}, []))
    gm = make_manager()
    gm.identify_state(HOME, SHOT_A)

    new_id = gm.rename_node("s0_home_screen", "Home feed")

    assert gm.identify_state(HOME, SHOT_A) == new_id
    assert vlm.kinds() == ["describe", "embed", "embed"]


# Node and edge queries


def test_get_node_returns_a_copy() -> None:
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen")

    node = gm.get_node("s0_home")
    assert node is not None
    node["page_description"] = "mutated"

    assert gm.graph.nodes["s0_home"]["page_description"] == "Home screen"


def test_get_node_returns_none_for_a_missing_node() -> None:
    assert make_manager().get_node("s9_missing") is None


def test_get_all_nodes_returns_every_node_with_its_attributes() -> None:
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen")
    add_screen(gm, "s1_settings", "Settings page")

    nodes = dict(gm.get_all_nodes())

    assert sorted(nodes) == ["s0_home", "s1_settings"]
    assert nodes["s1_settings"]["page_description"] == "Settings page"


def test_get_unexplored_elements_is_empty_for_a_missing_node() -> None:
    assert make_manager().get_unexplored_elements("s9_missing") == []


def test_get_unexplored_elements_filters_out_explored_ones() -> None:
    gm = make_manager()
    add_screen(
        gm,
        "s0_home",
        "Home screen",
        interactable_elements=[
            {"description": "Search box", "explored": True},
            {"description": "Menu", "explored": False},
        ],
    )

    unexplored = gm.get_unexplored_elements("s0_home")

    assert [e["description"] for e in unexplored] == ["Menu"]


def test_mark_element_explored_marks_the_best_word_overlap() -> None:
    gm = make_manager()
    add_screen(
        gm,
        "s0_home",
        "Home screen",
        interactable_elements=[
            {"description": "settings gear", "explored": False},
            {"description": "search box", "explored": False},
        ],
    )

    gm.mark_element_explored("s0_home", "tap the search box at the top")

    elements = gm.graph.nodes["s0_home"]["interactable_elements"]
    assert elements[1]["explored"] is True
    assert elements[0]["explored"] is False


def test_mark_element_explored_ignores_already_explored_and_empty_descriptions() -> None:
    gm = make_manager()
    add_screen(
        gm,
        "s0_home",
        "Home screen",
        interactable_elements=[
            {"description": "search box", "explored": True},
            {"description": "", "explored": False},
        ],
    )

    gm.mark_element_explored("s0_home", "tap the search box")

    assert gm.get_unexplored_elements("s0_home") == [{"description": "", "explored": False}]


def test_mark_element_explored_without_overlap_is_a_no_op() -> None:
    gm = make_manager()
    add_screen(
        gm,
        "s0_home",
        "Home screen",
        interactable_elements=[{"description": "settings gear", "explored": False}],
    )

    gm.mark_element_explored("s0_home", "swipe up")

    assert gm.get_unexplored_elements("s0_home") == [
        {"description": "settings gear", "explored": False}
    ]


def test_mark_element_explored_on_a_missing_node_is_a_no_op() -> None:
    gm = make_manager()
    gm.mark_element_explored("s9_missing", "tap search")
    assert gm.graph.number_of_nodes() == 0


def test_mark_element_explored_on_a_node_without_elements_is_a_no_op() -> None:
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen")

    gm.mark_element_explored("s0_home", "tap search")

    assert gm.graph.nodes["s0_home"]["interactable_elements"] == []


def test_get_all_edges_from_node_reports_one_dict_per_outgoing_edge() -> None:
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen")
    add_screen(gm, "s1_settings", "Settings page")
    gm.add_edge(
        "s0_home",
        "s1_settings",
        {"action": "tap"},
        instruction="open settings",
        target_observation="the settings page",
    )

    edges = gm.get_all_edges_from_node("s0_home")

    assert len(edges) == 1
    assert edges[0]["target"] == "s1_settings"
    assert edges[0]["target_description"] == "Settings page"
    assert edges[0]["actions"] == [{"action": "tap"}]
    assert edges[0]["instructions"] == ["open settings"]
    assert edges[0]["target_observations"] == ["the settings page"]
    assert edges[0]["instruction_templates"] == []
    assert edges[0]["visit_count"] == 1
    assert "schema_deltas" not in edges[0]


def test_get_all_edges_from_node_includes_a_non_empty_schema_delta() -> None:
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen")
    gm.add_edge("s0_home", "s0_home", {"action": "tap"}, schema_delta={"sort": {"after": "price"}})

    edges = gm.get_all_edges_from_node("s0_home")

    assert edges[0]["schema_deltas"] == [{"sort": {"after": "price"}}]


def test_get_all_edges_from_node_is_empty_for_a_missing_node() -> None:
    assert make_manager().get_all_edges_from_node("s9_missing") == []


def test_get_explored_actions_from_node_flattens_every_outgoing_edge() -> None:
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen")
    add_screen(gm, "s1_settings", "Settings page")
    add_screen(gm, "s2_results", "Search results")
    gm.add_edge("s0_home", "s1_settings", {"action": "tap", "id": 1})
    gm.add_edge("s0_home", "s1_settings", {"action": "tap", "id": 2})
    gm.add_edge("s0_home", "s2_results", {"action": "swipe"})

    actions = gm.get_explored_actions_from_node("s0_home")

    assert actions == [
        {"action": "tap", "id": 1},
        {"action": "tap", "id": 2},
        {"action": "swipe"},
    ]


# add_edge


def test_add_edge_creates_an_edge_with_defaults() -> None:
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen")
    add_screen(gm, "s1_settings", "Settings page")

    gm.add_edge("s0_home", "s1_settings", {"action": "tap"})

    data = gm.graph["s0_home"]["s1_settings"]
    assert data["actions"] == [{"action": "tap"}]
    assert data["num_steps"] == [1]
    assert data["visit_count"] == 1
    assert "instructions" not in data


def test_add_edge_derives_num_steps_from_a_compound_action() -> None:
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen")
    add_screen(gm, "s1_settings", "Settings page")

    gm.add_edge("s0_home", "s1_settings", [{"action": "tap"}, {"action": "type"}])

    assert gm.graph["s0_home"]["s1_settings"]["num_steps"] == [2]


def test_add_edge_honours_an_explicit_num_steps() -> None:
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen")
    add_screen(gm, "s1_settings", "Settings page")

    gm.add_edge("s0_home", "s1_settings", {"action": "tap"}, num_steps=7)

    assert gm.graph["s0_home"]["s1_settings"]["num_steps"] == [7]


def test_add_edge_appends_a_different_action_with_its_metadata() -> None:
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen")
    add_screen(gm, "s1_settings", "Settings page")
    gm.add_edge("s0_home", "s1_settings", {"action": "tap"}, instruction="open settings")

    gm.add_edge(
        "s0_home",
        "s1_settings",
        {"action": "swipe"},
        instruction="swipe to settings",
        target_observation="the settings page",
        schema_delta={"tab": {"after": "settings"}},
    )

    data = gm.graph["s0_home"]["s1_settings"]
    assert data["actions"] == [{"action": "tap"}, {"action": "swipe"}]
    assert data["num_steps"] == [1, 1]
    assert data["instructions"] == ["open settings", "swipe to settings"]
    assert data["target_observations"] == ["the settings page"]
    assert data["schema_deltas"] == [{"tab": {"after": "settings"}}]
    assert data["visit_count"] == 1


def test_add_edge_records_a_self_loop_with_a_schema_delta() -> None:
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen")

    gm.add_edge(
        "s0_home",
        "s0_home",
        {"action": "tap"},
        instruction="sort by price",
        target_observation="sorted by price",
        schema_delta={"sort": {"after": "price"}},
    )

    data = gm.graph["s0_home"]["s0_home"]
    assert data["instructions"] == ["sort by price"]
    assert data["target_observations"] == ["sorted by price"]
    assert data["schema_deltas"] == [{"sort": {"after": "price"}}]


# merge_nodes


def test_merge_nodes_returns_false_when_a_node_is_missing() -> None:
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen")

    assert gm.merge_nodes("s0_home", "s9_missing") is False
    assert gm.merge_nodes("s9_missing", "s0_home") is False
    assert gm.graph.number_of_nodes() == 1


def test_merge_nodes_unions_activities_schemas_elements_and_visits() -> None:
    gm = make_manager()
    add_screen(
        gm,
        "s0_home",
        "Home screen",
        visit_count=2,
        state_schema={"tab": ["home"]},
        interactable_elements=[{"description": "Search box", "explored": True}],
    )
    add_screen(
        gm,
        "s1_home_two",
        "Home screen again",
        activity=SETTINGS,
        visit_count=3,
        state_schema={"tab": ["deals"], "sort": ["price"]},
        interactable_elements=[{"description": "search BOX"}, {"description": "Menu"}],
    )

    assert gm.merge_nodes("s0_home", "s1_home_two") is True

    node = gm.get_node("s0_home")
    assert node is not None
    assert node["visit_count"] == 5
    assert node["activities"] == [HOME, SETTINGS]
    assert node["state_schema"] == {"tab": ["home", "deals"], "sort": ["price"]}
    assert [e["description"] for e in node["interactable_elements"]] == ["Search box", "Menu"]
    assert node["page_description"] == "Home screen"
    assert "s1_home_two" not in gm.graph


def test_merge_nodes_rewires_incoming_outgoing_and_self_loop_edges() -> None:
    gm = make_manager()
    for node_id, description in (
        ("s0_keep", "Home screen"),
        ("s1_remove", "Home screen again"),
        ("s2_other", "Settings page"),
    ):
        add_screen(gm, node_id, description)
    gm.add_edge("s2_other", "s1_remove", {"action": "in"})
    gm.add_edge("s1_remove", "s2_other", {"action": "out"})
    gm.add_edge("s0_keep", "s1_remove", {"action": "keep-to-remove"})
    gm.add_edge("s1_remove", "s0_keep", {"action": "remove-to-keep"})
    gm.add_edge("s1_remove", "s1_remove", {"action": "loop"})

    assert gm.merge_nodes("s0_keep", "s1_remove") is True

    assert gm.graph.has_edge("s2_other", "s0_keep")
    assert gm.graph.has_edge("s0_keep", "s2_other")
    assert gm.graph.has_edge("s0_keep", "s0_keep")
    self_loop_actions = gm.graph["s0_keep"]["s0_keep"]["actions"]
    assert {"action": "keep-to-remove"} in self_loop_actions
    assert {"action": "remove-to-keep"} in self_loop_actions
    assert {"action": "loop"} in self_loop_actions
    assert gm.graph.number_of_nodes() == 2


def test_merge_nodes_merges_edge_data_into_an_existing_edge() -> None:
    gm = make_manager()
    for node_id, description in (
        ("s0_keep", "Home screen"),
        ("s1_remove", "Home screen again"),
        ("s2_other", "Settings page"),
    ):
        add_screen(gm, node_id, description)
    gm.add_edge("s0_keep", "s2_other", {"action": "keep"}, instruction="from keep", num_steps=1)
    gm.add_edge(
        "s1_remove", "s2_other", {"action": "remove"}, instruction="from remove", num_steps=3
    )
    gm.graph["s1_remove"]["s2_other"]["instruction_templates"] = [{"template": "go to {x}"}]

    assert gm.merge_nodes("s0_keep", "s1_remove") is True

    data = gm.graph["s0_keep"]["s2_other"]
    assert data["actions"] == [{"action": "keep"}, {"action": "remove"}]
    assert data["instructions"] == ["from keep", "from remove"]
    assert data["num_steps"] == [1, 3]
    assert data["visit_count"] == 2
    assert data["instruction_templates"] == [{"template": "go to {x}"}]


# Edge normalization


@dataclass
class FakeNormalizer:
    """Stand-in for vlm_utils.normalize_edge; the client is unused, so it is _client."""

    result: dict[str, Any] = field(default_factory=dict)
    calls: list[dict[str, Any]] = field(default_factory=list)

    def __call__(
        self,
        _client: OpenAI | None,
        instructions: list[str],
        target_observations: list[str] | None = None,
        model: str = "",
    ) -> dict[str, Any]:
        call = {"instructions": list(instructions)}
        call.update(target_observations=target_observations, model=model)
        self.calls.append(call)
        return self.result


TEMPLATE_RESULT: dict[str, Any] = {
    "is_template": True,
    "instruction_template": "search for {query}",
    "observation_template": "results for {query}",
    "param_names": ["query"],
    "examples": ['search for "shoes"', 'search for "hats"'],
}


@pytest.fixture
def normalizer(monkeypatch: pytest.MonkeyPatch) -> FakeNormalizer:
    fake = FakeNormalizer(result=dict(TEMPLATE_RESULT))
    monkeypatch.setattr(gm_module, "normalize_edge", fake)
    return fake


def _search_graph(gm: GraphManager) -> None:
    add_screen(gm, "s0_home", "Home screen")
    add_screen(gm, "s1_results", "Search results")
    gm.add_edge("s0_home", "s1_results", {"type": "shoes"}, instruction='search for "shoes"')
    gm.add_edge("s0_home", "s1_results", {"type": "hats"}, instruction='search for "hats"')


def test_normalize_all_edges_templatizes_and_then_skips(normalizer: FakeNormalizer) -> None:
    gm = make_manager()
    _search_graph(gm)

    gm.normalize_all_edges()

    templates = gm.graph["s0_home"]["s1_results"]["instruction_templates"]
    assert templates[0]["template"] == "search for {query}"
    assert templates[0]["observation_template"] == "results for {query}"
    assert templates[0]["param_names"] == ["query"]
    assert len(normalizer.calls) == 1
    assert normalizer.calls[0]["instructions"] == ['search for "shoes"', 'search for "hats"']
    assert normalizer.calls[0]["target_observations"] is None
    assert normalizer.calls[0]["model"] == gm.page_detail_model

    gm.normalize_all_edges()  # the existing template already covers every instruction

    assert len(normalizer.calls) == 1


def test_normalize_all_edges_passes_target_observations(normalizer: FakeNormalizer) -> None:
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen")
    add_screen(gm, "s1_results", "Search results")
    gm.add_edge(
        "s0_home",
        "s1_results",
        {"type": "shoes"},
        instruction='search for "shoes"',
        target_observation="shoe results",
    )

    gm.normalize_all_edges()

    assert normalizer.calls[0]["target_observations"] == ["shoe results"]


def test_normalize_all_edges_renormalizes_past_a_non_dict_template(
    normalizer: FakeNormalizer,
) -> None:
    gm = make_manager()
    _search_graph(gm)
    gm.graph["s0_home"]["s1_results"]["instruction_templates"] = ["stale string"]

    gm.normalize_all_edges()

    assert len(normalizer.calls) == 1
    assert gm.graph["s0_home"]["s1_results"]["instruction_templates"][0]["template"] == (
        "search for {query}"
    )


def test_normalize_all_edges_clears_templates_when_the_result_is_not_a_template(
    normalizer: FakeNormalizer,
) -> None:
    normalizer.result = {"is_template": False}
    gm = make_manager()
    _search_graph(gm)

    gm.normalize_all_edges()

    assert gm.graph["s0_home"]["s1_results"]["instruction_templates"] == []
    assert len(normalizer.calls) == 1


def test_normalize_all_edges_skips_an_edge_without_instructions(
    normalizer: FakeNormalizer,
) -> None:
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen")
    add_screen(gm, "s1_results", "Search results")
    gm.add_edge("s0_home", "s1_results", {"action": "tap"})

    gm.normalize_all_edges()

    assert normalizer.calls == []


def test_normalize_all_edges_skips_a_short_single_instruction(normalizer: FakeNormalizer) -> None:
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen")
    add_screen(gm, "s1_settings", "Settings page")
    gm.add_edge("s0_home", "s1_settings", {"action": "tap"}, instruction="open settings")

    gm.normalize_all_edges()

    assert normalizer.calls == []
    assert gm.graph["s0_home"]["s1_settings"]["instruction_templates"] == []


def test_normalize_all_edges_normalizes_a_long_single_instruction(
    normalizer: FakeNormalizer,
) -> None:
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen")
    add_screen(gm, "s1_settings", "Settings page")
    gm.add_edge(
        "s0_home",
        "s1_settings",
        {"action": "tap"},
        instruction="open the settings page from the home screen menu",
    )

    gm.normalize_all_edges()

    assert len(normalizer.calls) == 1


def test_normalize_all_edges_without_a_client_does_nothing(
    normalizer: FakeNormalizer,
) -> None:
    gm = GraphManager()
    _search_graph(gm)

    gm.normalize_all_edges()

    assert normalizer.calls == []
    assert "instruction_templates" not in gm.graph["s0_home"]["s1_results"]


def test_normalize_node_edges_counts_templatized_edges(normalizer: FakeNormalizer) -> None:
    gm = make_manager()
    _search_graph(gm)
    add_screen(gm, "s2_settings", "Settings page")
    gm.add_edge("s0_home", "s2_settings", {"action": "tap"}, instruction="open settings")

    assert gm.normalize_node_edges("s0_home") == 1
    assert len(normalizer.calls) == 1


def test_normalize_node_edges_without_a_client_returns_zero(
    normalizer: FakeNormalizer,
) -> None:
    gm = GraphManager()
    _search_graph(gm)

    assert gm.normalize_node_edges("s0_home") == 0
    assert normalizer.calls == []


def test_normalize_node_edges_for_a_missing_node_returns_zero(
    normalizer: FakeNormalizer,
) -> None:
    assert make_manager().normalize_node_edges("s9_missing") == 0
    assert normalizer.calls == []


def test_maybe_normalize_node_edges_runs_once_per_milestone(normalizer: FakeNormalizer) -> None:
    gm = make_manager()
    _search_graph(gm)
    gm.graph.nodes["s0_home"]["visit_count"] = NORMALIZE_EVERY_N_VISITS

    assert gm.maybe_normalize_node_edges("s0_home") is True
    assert gm.graph.nodes["s0_home"]["last_normalized_visit_milestone"] == 1
    assert gm.maybe_normalize_node_edges("s0_home") is False
    assert len(normalizer.calls) == 1


def test_maybe_normalize_node_edges_below_the_milestone_returns_false(
    normalizer: FakeNormalizer,
) -> None:
    gm = make_manager()
    _search_graph(gm)
    gm.graph.nodes["s0_home"]["visit_count"] = NORMALIZE_EVERY_N_VISITS - 1

    assert gm.maybe_normalize_node_edges("s0_home") is False
    assert normalizer.calls == []


def test_maybe_normalize_node_edges_for_a_missing_node_returns_false() -> None:
    assert make_manager().maybe_normalize_node_edges("s9_missing") is False


# Path finding


def _line_graph(gm: GraphManager) -> None:
    """s0 → s1 → s3 in two cheap hops, or s0 → s3 in one expensive hop."""
    for node_id, description in (
        ("s0_home", "Home screen"),
        ("s1_settings", "Settings page"),
        ("s3_results", "Search results"),
    ):
        add_screen(gm, node_id, description)
    gm.add_edge("s0_home", "s1_settings", {"action": "a"}, num_steps=1)
    gm.add_edge("s1_settings", "s3_results", {"action": "b"}, num_steps=1)
    gm.add_edge("s0_home", "s3_results", {"action": "c"}, num_steps=5)


def test_find_path_from_a_node_to_itself_is_a_single_step() -> None:
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen")

    assert gm.find_path("s0_home", "s0_home") == [("s0_home", {})]


def test_find_path_returns_none_for_a_missing_node() -> None:
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen")

    assert gm.find_path("s0_home", "s9_missing") is None
    assert gm.find_path("s9_missing", "s0_home") is None


def test_find_path_returns_none_when_no_path_exists() -> None:
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen")
    add_screen(gm, "s1_settings", "Settings page")

    assert gm.find_path("s0_home", "s1_settings") is None


def test_find_path_prefers_the_fewest_total_steps() -> None:
    gm = make_manager()
    _line_graph(gm)

    path = gm.find_path("s0_home", "s3_results")

    assert path == [
        ("s0_home", {}),
        ("s1_settings", {"action": "a"}),
        ("s3_results", {"action": "b"}),
    ]


def test_find_path_picks_the_action_with_the_fewest_steps() -> None:
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen")
    add_screen(gm, "s1_settings", "Settings page")
    gm.add_edge("s0_home", "s1_settings", {"action": "long"}, num_steps=3)
    gm.add_edge("s0_home", "s1_settings", {"action": "short"}, num_steps=1)

    path = gm.find_path("s0_home", "s1_settings")

    assert path == [("s0_home", {}), ("s1_settings", {"action": "short"})]


def test_find_path_uses_an_empty_action_for_an_edge_without_actions() -> None:
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen")
    add_screen(gm, "s1_settings", "Settings page")
    gm.graph.add_edge("s0_home", "s1_settings")

    assert gm.find_path("s0_home", "s1_settings") == [("s0_home", {}), ("s1_settings", {})]


def test_find_guided_path_navigates_and_runs_a_matching_self_loop() -> None:
    gm = make_manager()
    _line_graph(gm)
    gm.add_edge(
        "s3_results",
        "s3_results",
        {"action": "sort"},
        instruction="sort by price",
        target_observation="sorted by price",
        schema_delta={"sort": {"after": "price"}},
    )
    gm.graph["s0_home"]["s1_settings"]["instructions"] = ["open settings"]
    gm.graph["s0_home"]["s1_settings"]["target_observations"] = ["the settings page"]

    steps = gm.find_guided_path(
        "s0_home", [{"node_id": "s3_results", "required_schema": {"sort": "price"}}]
    )

    assert steps is not None
    assert [step["type"] for step in steps] == ["navigate", "navigate", "self_loop"]
    assert steps[0]["instruction"] == "open settings"
    assert steps[0]["target_observation"] == "the settings page"
    assert steps[1]["instruction"] == ""
    assert steps[2]["schema_delta"] == {"sort": {"after": "price"}}
    assert steps[2]["action"] == {"action": "sort"}
    assert steps[2]["required_schema"] == {"sort": "price"}


def test_find_guided_path_records_a_schema_gap_without_a_self_loop() -> None:
    gm = make_manager()
    _line_graph(gm)

    steps = gm.find_guided_path(
        "s0_home", [{"node_id": "s0_home", "required_schema": {"sort": "price"}}]
    )

    assert steps == [
        {"type": "schema_gap", "node": "s0_home", "required_schema": {"sort": "price"}}
    ]


def test_find_guided_path_records_a_schema_gap_when_no_delta_matches() -> None:
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen")
    gm.add_edge("s0_home", "s0_home", {"action": "sort"}, schema_delta={"other": {"after": 1}})

    steps = gm.find_guided_path(
        "s0_home", [{"node_id": "s0_home", "required_schema": {"sort": "price"}}]
    )

    assert steps is not None
    assert steps[0]["type"] == "schema_gap"


def test_find_guided_path_ignores_a_self_loop_without_schema_deltas() -> None:
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen")
    gm.add_edge("s0_home", "s0_home", {"action": "sort"})

    steps = gm.find_guided_path(
        "s0_home", [{"node_id": "s0_home", "required_schema": {"sort": "price"}}]
    )

    assert steps is not None
    assert steps[0]["type"] == "schema_gap"


def test_find_guided_path_skips_empty_deltas_and_falls_back_for_short_lists() -> None:
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen")
    gm.graph.add_edge(
        "s0_home",
        "s0_home",
        actions=[],
        instructions=[],
        target_observations=[],
        schema_deltas=[{}, {"sort": {"after": "price"}}],
        visit_count=1,
    )

    steps = gm.find_guided_path(
        "s0_home", [{"node_id": "s0_home", "required_schema": {"sort": "price"}}]
    )

    assert steps is not None
    assert steps[0]["type"] == "self_loop"
    assert steps[0]["instruction"] == ""
    assert steps[0]["target_observation"] == ""
    assert steps[0]["action"] == []


def test_find_guided_path_without_a_required_schema_only_navigates() -> None:
    gm = make_manager()
    _line_graph(gm)

    steps = gm.find_guided_path("s0_home", [{"node_id": "s1_settings"}])

    assert steps is not None
    assert [step["type"] for step in steps] == ["navigate"]


def test_find_guided_path_returns_none_when_a_waypoint_is_unreachable() -> None:
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen")
    add_screen(gm, "s1_settings", "Settings page")

    assert gm.find_guided_path("s0_home", [{"node_id": "s1_settings"}]) is None


def test_shortest_path_returns_the_node_ids() -> None:
    gm = make_manager()
    _line_graph(gm)

    assert gm.shortest_path("s0_home", "s3_results") == ["s0_home", "s1_settings", "s3_results"]


def test_shortest_path_raises_for_an_unknown_source() -> None:
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen")

    with pytest.raises(nx.NodeNotFound):
        gm.shortest_path("s9_missing", "s0_home")


def test_shortest_path_returns_none_without_a_path() -> None:
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen")
    add_screen(gm, "s1_settings", "Settings page")

    assert gm.shortest_path("s0_home", "s1_settings") is None


def test_get_path_actions_takes_the_first_action_of_every_hop() -> None:
    gm = make_manager()
    _line_graph(gm)
    gm.graph.add_edge("s3_results", "s0_home")

    actions = gm.get_path_actions(["s0_home", "s1_settings", "s3_results", "s0_home"])

    assert actions == [{"action": "a"}, {"action": "b"}]


# Selection heuristics


def test_get_start_node_returns_none_for_an_empty_graph() -> None:
    assert make_manager().get_start_node() is None


def test_get_start_node_picks_the_lowest_numeric_id_and_skips_external_nodes() -> None:
    gm = make_manager()
    add_screen(gm, "s2_results", "Search results")
    add_screen(gm, "s1_settings", "Settings page")
    add_screen(gm, "ext_browser", "Browser")
    add_screen(gm, "s0_external", "External home", is_external=True)

    assert gm.get_start_node() == "s1_settings"


def test_get_start_node_falls_back_to_sorted_ids() -> None:
    gm = make_manager()
    add_screen(gm, "sx_home", "Home screen")
    add_screen(gm, "alpha", "Alpha screen")

    assert gm.get_start_node() == "alpha"


def test_get_start_node_returns_none_when_every_node_is_external() -> None:
    gm = make_manager()
    add_screen(gm, "ext_browser", "Browser")

    assert gm.get_start_node() is None


def test_get_least_explored_node_returns_none_without_a_start_node() -> None:
    assert make_manager().get_least_explored_node() is None


def test_get_least_explored_node_prefers_the_fewest_outgoing_edges() -> None:
    gm = make_manager()
    for node_id, description in (
        ("s0_home", "Home screen"),
        ("s1_settings", "Settings page"),
        ("s2_results", "Search results"),
    ):
        add_screen(gm, node_id, description)
    add_screen(gm, "s3_unreachable", "Unreachable screen")
    gm.add_edge("s0_home", "s1_settings", {"action": "a"})
    gm.add_edge("s0_home", "s2_results", {"action": "b"})
    gm.add_edge("s2_results", "s0_home", {"action": "c"})

    assert gm.get_least_explored_node() == "s1_settings"


def test_get_least_explored_node_skips_external_nodes() -> None:
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen")
    add_screen(gm, "ext_browser", "Browser")
    add_screen(gm, "s1_external", "External page", is_external=True)
    add_screen(gm, "s2_results", "Search results")
    gm.add_edge("s0_home", "ext_browser", {"action": "a"})
    gm.add_edge("s0_home", "s1_external", {"action": "b"})
    gm.add_edge("s0_home", "s2_results", {"action": "c"})
    gm.add_edge("s2_results", "s0_home", {"action": "d"})

    assert gm.get_least_explored_node() == "s2_results"


def test_get_least_explored_node_filters_by_package() -> None:
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen")
    add_screen(gm, "s1_other", "Other app screen", activity=OTHER_APP)
    add_screen(gm, "s2_results", "Search results")
    gm.add_edge("s0_home", "s1_other", {"action": "a"})
    gm.add_edge("s0_home", "s2_results", {"action": "b"})
    gm.add_edge("s2_results", "s0_home", {"action": "c"})

    assert gm.get_least_explored_node(package_name="com.example.app") == "s2_results"


def test_get_exploration_target_candidates_is_empty_without_a_start_node() -> None:
    assert make_manager().get_exploration_target_candidates() == []


def test_get_exploration_target_candidates_ranks_by_coverage_need() -> None:
    gm = make_manager()
    add_screen(
        gm,
        "s0_home",
        "Home screen",
        visit_count=5,
        interactable_elements=[{"description": "Search box", "explored": True}],
    )
    add_screen(
        gm,
        "s1_settings",
        "Settings page",
        visit_count=1,
        interactable_elements=[
            {"description": "Wi-Fi", "explored": False},
            {"description": "Sound", "explored": False},
        ],
    )
    gm.add_edge("s0_home", "s1_settings", {"action": "a"})

    candidates = gm.get_exploration_target_candidates()

    assert [c["node_id"] for c in candidates] == ["s1_settings", "s0_home"]
    assert candidates[0]["unexplored_elements"] == 2
    assert candidates[0]["total_elements"] == 2
    assert candidates[0]["out_degree"] == 0
    assert candidates[0]["score"] == 11
    assert candidates[0]["unexplored_element_descriptions"] == ["Wi-Fi", "Sound"]
    assert candidates[1]["score"] == 2


def test_get_exploration_target_candidates_honours_top_k() -> None:
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen")
    add_screen(gm, "s1_settings", "Settings page")
    gm.add_edge("s0_home", "s1_settings", {"action": "a"})

    assert len(gm.get_exploration_target_candidates(top_k=1)) == 1


def test_get_exploration_target_candidates_skips_external_and_other_packages() -> None:
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen")
    add_screen(gm, "ext_browser", "Browser")
    add_screen(gm, "s1_external", "External page", is_external=True)
    add_screen(gm, "s2_other", "Other app screen", activity=OTHER_APP)
    for target in ("ext_browser", "s1_external", "s2_other"):
        gm.add_edge("s0_home", target, {"action": target})

    candidates = gm.get_exploration_target_candidates(package_name="com.example.app")

    assert [c["node_id"] for c in candidates] == ["s0_home"]


# Persistence


def test_embeddings_path_is_a_companion_of_the_graph_file() -> None:
    assert GraphManager._embeddings_path(Path("/graphs/app.json")) == Path("/graphs/app.emb.json")


def test_save_writes_dirty_screenshots_once_and_load_restores_them(
    tmp_path: Path, vlm: FakeVlm
) -> None:
    vlm.descriptions.append(("Home screen", {}, []))
    gm = make_manager()
    node_id = gm.identify_state(HOME, SHOT_A)
    graph_path = tmp_path / "nested" / "graph.json"

    gm.save_graph(graph_path)

    png = graph_path.parent / "graph_screenshots" / f"{node_id}.png"
    assert png.read_bytes() == b"screenshot-a"
    assert (graph_path.parent / "graph.emb.json").is_file()

    loaded = GraphManager()
    loaded.load_graph(graph_path)
    restored = loaded.get_node(node_id)
    assert restored is not None
    assert restored["reference_screenshot"] == SHOT_A
    assert restored["description_embedding"] == EMBEDDINGS["Home screen"]

    png.unlink()
    gm.save_graph(graph_path)
    assert not png.exists()  # the dirty set was cleared by the first save


def test_save_graph_persists_nodes_edges_and_counters(tmp_path: Path) -> None:
    gm = make_manager(similarity_threshold=0.5)
    gm.total_steps_completed = 42
    add_screen(gm, "s0_home", "Home screen", visit_count=3)
    add_screen(gm, "s1_settings", "Settings page")
    gm.add_edge(
        "s0_home",
        "s1_settings",
        {"action": "tap"},
        instruction="open settings",
        schema_delta={"tab": {"after": "settings"}},
    )
    path = tmp_path / "graph.json"

    gm.save_graph(path)

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["similarity_threshold"] == 0.5
    assert data["total_steps_completed"] == 42
    assert data["nodes"][0]["activity"] == HOME
    assert data["nodes"][0]["visit_count"] == 3
    assert data["edges"][0]["schema_deltas"] == [{"tab": {"after": "settings"}}]
    embeddings = json.loads((tmp_path / "graph.emb.json").read_text(encoding="utf-8"))
    assert embeddings["s0_home"] == EMBEDDINGS["Home screen"]


def test_save_graph_omits_an_empty_activity_list_and_embedding(tmp_path: Path) -> None:
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen", activities=[], description_embedding=[])
    path = tmp_path / "graph.json"

    gm.save_graph(path)

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["nodes"][0]["activity"] == ""
    assert json.loads((tmp_path / "graph.emb.json").read_text(encoding="utf-8")) == {}


def test_load_graph_falls_back_to_an_inline_embedding(tmp_path: Path) -> None:
    path = tmp_path / "graph.json"
    path.write_text(
        json.dumps(
            {
                "next_id": 7,
                "nodes": [
                    {
                        "id": "s0_home",
                        "activity": HOME,
                        "page_description": "Home screen",
                        "description_embedding": [1.0, 0.0, 0.0],
                    }
                ],
                "edges": [],
            }
        ),
        encoding="utf-8",
    )
    gm = GraphManager()

    gm.load_graph(path)

    node = gm.get_node("s0_home")
    assert node is not None
    assert node["description_embedding"] == [1.0, 0.0, 0.0]
    assert node["activities"] == [HOME]
    assert node["reference_screenshot"] is None
    assert gm.similarity_threshold == DEFAULT_SIMILARITY_THRESHOLD
    assert gm.total_steps_completed == 0


def test_load_graph_replaces_the_current_graph(tmp_path: Path) -> None:
    source = make_manager()
    add_screen(source, "s0_home", "Home screen")
    path = tmp_path / "graph.json"
    source.save_graph(path)

    target = make_manager()
    add_screen(target, "s9_stale", "Stale screen")
    target.load_graph(path)

    assert list(target.graph.nodes) == ["s0_home"]


# Reporting and audit


def test_find_node_by_description_sorts_by_similarity(vlm: FakeVlm) -> None:
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen")
    add_screen(gm, "s1_settings", "Settings page")
    add_screen(gm, "s2_blank", "Blank screen", description_embedding=None)

    results = gm.find_node_by_description("Home screen")

    assert [node_id for node_id, _ in results] == ["s0_home", "s1_settings"]
    assert results[0][1] == pytest.approx(1.0)
    assert results[1][1] == pytest.approx(0.0)
    assert vlm.kinds() == ["embed"]


def test_find_node_by_description_stringifies_non_string_node_ids(vlm: FakeVlm) -> None:
    gm = make_manager()
    gm.graph.add_node(7, description_embedding=[1.0, 0.0, 0.0])

    results = gm.find_node_by_description("Home screen")

    assert results == [("7", pytest.approx(1.0))]
    assert vlm.kinds() == ["embed"]


def test_summary_counts_nodes_edges_and_element_progress() -> None:
    gm = make_manager()
    add_screen(
        gm,
        "s0_home",
        "Home screen",
        visit_count=2,
        state_schema={"tab": ["home"]},
        interactable_elements=[
            {"description": "Search box", "explored": True},
            {"description": "Menu", "explored": False},
        ],
    )
    add_screen(gm, "s1_settings", "Settings page")
    gm.add_edge("s0_home", "s1_settings", {"action": "tap"})

    summary = gm.summary()

    assert summary["num_nodes"] == 2
    assert summary["num_edges"] == 1
    assert sorted(summary["activities"]) == [HOME]
    assert summary["nodes"][0]["schema_keys"] == ["tab"]
    assert summary["nodes"][0]["elements_explored"] == "1/2"
    assert summary["nodes"][0]["visit_count"] == 2


def test_format_for_audit_lists_nodes_edges_and_marks_self_loops() -> None:
    gm = make_manager()
    add_screen(
        gm,
        "s0_home",
        "Home screen",
        visit_count=1,
        state_schema={"tab": ["home"]},
        interactable_elements=[{"description": "Menu", "explored": True}],
    )
    add_screen(gm, "s1_settings", "Settings page")
    gm.add_edge(
        "s0_home",
        "s1_settings",
        {"action": "tap"},
        instruction="open settings",
        target_observation="the settings page",
    )
    gm.add_edge("s0_home", "s0_home", {"action": "sort"}, instruction="sort by price")

    text = gm.format_for_audit()

    assert "## Nodes (2)" in text
    assert "## Edges (2)" in text
    assert "state: [tab]" in text
    assert "elements: 1/1 explored" in text
    assert '- "open settings" → the settings page' in text
    assert "[self-loop]" in text


def test_run_audit_without_a_client_reports_no_issues() -> None:
    result = GraphManager().run_audit(app_name="Example")

    assert result == {"issues": [], "summary": "no client available"}


def test_run_audit_returns_the_audit_result(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeAudit(
        result={
            "summary": "two issues",
            "issues": [
                {"type": "merge_nodes", "severity": "high", "node_a": "s0", "node_b": "s1"},
                {"type": "retry_edge", "severity": "low", "source_node": "s0", "target_node": "s1"},
                {"type": "explore_node", "severity": "low", "node": "s1"},
                {"type": "unknown", "severity": "low"},
            ],
        }
    )
    monkeypatch.setattr(gm_module, "audit_graph", fake)
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen")

    result = gm.run_audit(app_name="Example")

    assert result["summary"] == "two issues"
    assert fake.calls[0]["app_name"] == "Example"
    assert fake.calls[0]["model"] == gm.page_detail_model
    assert fake.calls[0]["graph_summary"].startswith("## Nodes (1)")


def test_run_node_merge_audit_without_a_client_reports_no_merges() -> None:
    result = GraphManager().run_node_merge_audit()

    assert result == {"issues": [], "results": [], "merged_count": 0}


def test_run_node_merge_audit_skips_a_missing_node(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeAudit(result={"issues": [{"node_a": "s0_home", "node_b": "s9_missing"}]})
    monkeypatch.setattr(gm_module, "audit_merge_nodes", fake)
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen")

    result = gm.run_node_merge_audit(app_name="Example")

    assert result["merged_count"] == 0
    assert result["results"][0]["status"] == "skipped"
    assert result["results"][0]["reason"] == "node missing"


def test_run_node_merge_audit_skips_a_missing_screenshot(
    monkeypatch: pytest.MonkeyPatch, vlm: FakeVlm
) -> None:
    fake = FakeAudit(result={"issues": [{"node_a": "s0_home", "node_b": "s1_home_two"}]})
    monkeypatch.setattr(gm_module, "audit_merge_nodes", fake)
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen", reference_screenshot=SHOT_A)
    add_screen(gm, "s1_home_two", "Home screen again")

    result = gm.run_node_merge_audit()

    assert result["results"][0]["reason"] == "missing screenshot"
    assert vlm.kinds() == []


def test_run_node_merge_audit_merges_a_confirmed_duplicate(
    monkeypatch: pytest.MonkeyPatch, vlm: FakeVlm
) -> None:
    fake = FakeAudit(
        result={"issues": [{"node_a": "s2_home_two", "node_b": "s0_home"}], "summary": "one pair"}
    )
    monkeypatch.setattr(gm_module, "audit_merge_nodes", fake)
    vlm.verdicts.append({"same": True, "reason": "same screen"})
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen", reference_screenshot=SHOT_A, visit_count=1)
    add_screen(gm, "s2_home_two", "Home screen again", reference_screenshot=SHOT_B, visit_count=5)

    result = gm.run_node_merge_audit()

    assert result["merged_count"] == 1
    assert result["summary"] == "one pair"
    assert result["results"][0]["kept"] == "s0_home"
    assert result["results"][0]["removed"] == "s2_home_two"
    assert list(gm.graph.nodes) == ["s0_home"]
    # The description of the more-visited node is the verifier's reference.
    assert vlm.calls[0][1]["existing_description"] == "Home screen again"


def test_run_node_merge_audit_orders_non_numeric_ids_last(
    monkeypatch: pytest.MonkeyPatch, vlm: FakeVlm
) -> None:
    fake = FakeAudit(result={"issues": [{"node_a": "ext_browser", "node_b": "s0_home"}]})
    monkeypatch.setattr(gm_module, "audit_merge_nodes", fake)
    vlm.verdicts.append({"same": True, "reason": "same screen"})
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen", reference_screenshot=SHOT_A)
    add_screen(gm, "ext_browser", "Browser", reference_screenshot=SHOT_B)

    result = gm.run_node_merge_audit()

    assert result["results"][0]["kept"] == "s0_home"


def test_run_node_merge_audit_keeps_a_rejected_pair_separate(
    monkeypatch: pytest.MonkeyPatch, vlm: FakeVlm
) -> None:
    fake = FakeAudit(result={"issues": [{"node_a": "s0_home", "node_b": "s1_home_two"}]})
    monkeypatch.setattr(gm_module, "audit_merge_nodes", fake)
    vlm.verdicts.append({"same": False, "reason": "different banner"})
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen", reference_screenshot=SHOT_A)
    add_screen(gm, "s1_home_two", "Home screen again", reference_screenshot=SHOT_B)

    result = gm.run_node_merge_audit()

    assert result["merged_count"] == 0
    assert result["results"][0]["status"] == "kept_separate"
    assert result["results"][0]["reason"] == "different banner"
    assert gm.graph.number_of_nodes() == 2


# Properties

POOL = [f"s{i}_screen" for i in range(5)]

json_scalars = (
    st.none()
    | st.booleans()
    | st.integers(min_value=-1_000_000, max_value=1_000_000)
    | st.floats(allow_nan=False, allow_infinity=False)
    | st.text(max_size=12)
)
actions = st.dictionaries(st.text(min_size=1, max_size=8), json_scalars, max_size=3)
small_dicts = st.lists(st.dictionaries(st.text(max_size=6), json_scalars, max_size=2), max_size=2)
node_attrs = st.fixed_dictionaries(
    {
        "activities": st.lists(st.text(min_size=1, max_size=30), min_size=1, max_size=3),
        "page_description": st.text(max_size=40),
        "state_schema": st.dictionaries(
            st.text(max_size=10), st.lists(json_scalars, max_size=4), max_size=4
        ),
        "last_detail_snapshot": st.dictionaries(st.text(max_size=10), json_scalars, max_size=4),
        "interactable_elements": st.lists(
            st.fixed_dictionaries(
                {
                    "description": st.text(max_size=20),
                    "position": st.text(max_size=10),
                    "explored": st.booleans(),
                }
            ),
            max_size=4,
        ),
        "description_embedding": st.lists(
            st.floats(allow_nan=False, allow_infinity=False), max_size=4
        ),
        "last_normalized_visit_milestone": st.integers(min_value=0, max_value=10),
        "visit_count": st.integers(min_value=0, max_value=100),
    }
)
edge_attrs = st.fixed_dictionaries(
    {
        "actions": st.lists(actions | st.lists(actions, max_size=3), max_size=3),
        "instructions": st.lists(st.text(max_size=12), max_size=3),
        "instruction_templates": small_dicts,
        "target_observations": st.lists(st.text(max_size=12), max_size=3),
        "num_steps": st.lists(st.integers(min_value=1, max_value=5), max_size=3),
        "visit_count": st.integers(min_value=0, max_value=20),
        "schema_deltas": small_dicts,
    }
)
graphs = st.tuples(
    st.dictionaries(st.sampled_from(POOL), node_attrs, min_size=1),
    st.dictionaries(
        st.tuples(st.sampled_from(POOL), st.sampled_from(POOL)), edge_attrs, max_size=8
    ),
)


def canonical(gm: GraphManager) -> dict[str, Any]:
    """The subset of state that save_graph persists and load_graph restores."""
    nodes = {
        node_id: {
            "activities": d.get("activities", [d.get("activity", "")]),
            "page_description": d.get("page_description", ""),
            "state_schema": d.get("state_schema", {}),
            "last_detail_snapshot": d.get("last_detail_snapshot", {}),
            "interactable_elements": d.get("interactable_elements", []),
            "description_embedding": d.get("description_embedding") or [],
            "last_normalized_visit_milestone": d.get("last_normalized_visit_milestone", 0),
            "visit_count": d.get("visit_count", 0),
        }
        for node_id, d in gm.graph.nodes(data=True)
    }
    edges = {
        (u, v): {
            "actions": d.get("actions", []),
            "instructions": d.get("instructions", []),
            "instruction_templates": d.get("instruction_templates", []),
            "target_observations": d.get("target_observations", []),
            "num_steps": d.get("num_steps", []),
            "visit_count": d.get("visit_count", 0),
            "schema_deltas": d.get("schema_deltas") or None,
        }
        for u, v, d in gm.graph.edges(data=True)
    }
    return {"nodes": nodes, "edges": edges}


@given(
    graph=graphs,
    threshold=st.floats(min_value=0.0, max_value=1.0),
    steps=st.integers(min_value=0, max_value=10_000),
)
def test_save_then_load_round_trips(
    tmp_path_factory: pytest.TempPathFactory,
    graph: tuple[dict[str, dict[str, Any]], dict[tuple[str, str], dict[str, Any]]],
    threshold: float,
    steps: int,
) -> None:
    nodes, edges = graph
    gm = GraphManager(similarity_threshold=threshold)
    gm.total_steps_completed = steps
    for node_id, attrs in nodes.items():
        gm.graph.add_node(node_id, activity=attrs["activities"][0], **attrs)
    for (u, v), attrs in edges.items():
        if u in nodes and v in nodes:
            gm.graph.add_edge(u, v, **attrs)
    path = tmp_path_factory.mktemp("roundtrip") / "graph.json"

    gm.save_graph(path)
    loaded = GraphManager()
    loaded.load_graph(path)

    assert canonical(loaded) == canonical(gm)
    assert loaded.similarity_threshold == threshold
    assert loaded.total_steps_completed == steps
    again = tmp_path_factory.mktemp("roundtrip") / "graph.json"
    loaded.save_graph(again)
    assert again.read_text(encoding="utf-8") == path.read_text(encoding="utf-8")


@given(
    action=actions, repeats=st.integers(min_value=1, max_value=6), instruction=st.text(max_size=20)
)
def test_readding_an_identical_edge_is_idempotent(
    action: dict[str, Any], repeats: int, instruction: str
) -> None:
    gm = GraphManager()
    add_screen(gm, "s0_home", "Home screen")
    add_screen(gm, "s1_settings", "Settings page")

    for _ in range(repeats):
        gm.add_edge("s0_home", "s1_settings", action, instruction=instruction)

    assert gm.graph.number_of_edges() == 1
    data = gm.graph["s0_home"]["s1_settings"]
    assert data["actions"] == [action]
    assert data["visit_count"] == repeats
    assert data["num_steps"] == [1]
    assert len(data.get("instructions", [])) <= 1


operations = st.lists(
    st.tuples(
        st.sampled_from(["edge", "merge", "rename"]),
        st.sampled_from(POOL),
        st.sampled_from(POOL),
        st.text(min_size=1, max_size=12),
    ),
    max_size=25,
)


def fake_get_embedding(_client: OpenAI | None, text: str, **_kwargs: Any) -> list[float]:
    """Module-level fake for property tests; `model=` arrives through **_kwargs."""
    return EMBEDDINGS.get(text, UNKNOWN_EMBEDDING)


@given(ops=operations)
def test_every_edge_references_an_existing_node(ops: list[tuple[str, str, str, str]]) -> None:
    gm = make_manager()
    for node_id in POOL:
        add_screen(gm, node_id, f"screen {node_id}")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(gm_module, "get_embedding", fake_get_embedding)
        for kind, a, b, text in ops:
            if kind == "edge" and a in gm.graph and b in gm.graph:
                gm.add_edge(a, b, {"action_type": "tap", "target": text})
            elif kind == "merge" and a != b:  # merge_nodes(x, x) deletes the node: see #48
                merged = gm.merge_nodes(a, b)
                if merged:
                    assert a in gm.graph
                    assert b not in gm.graph
            elif kind == "rename":
                gm.rename_node(a, text)
            for u, v in gm.graph.edges():
                assert u in gm.graph
                assert v in gm.graph
                assert "page_description" in gm.graph.nodes[u]
                assert "page_description" in gm.graph.nodes[v]
