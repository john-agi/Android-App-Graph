"""Tests for android_app_graph.utils.graph_manager.

Most VLM and embedding calls are replaced by scripted fakes patched onto the
graph_manager module namespace; a few tests run the real SDK over the offline mock
transport. No test needs a device, the network or keys.
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import io
import itertools
import json
import logging
import os
import socket
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, TypedDict, cast, override

import httpx
import networkx as nx
import pytest
from hypothesis import given
from hypothesis import strategies as st
from openrouter import OpenRouter
from openrouter.components.chatusage import ChatUsage
from PIL import Image, ImageDraw

from android_app_graph.graph_files import PER_ACTION_LISTS, save_embeddings, text_embeddings_path
from android_app_graph.step_log import Observation, StepRecorder, save_image_embeddings
from android_app_graph.utils import ModelSettings, vlm_utils
from android_app_graph.utils import graph_manager as gm_module
from android_app_graph.utils.graph_manager import (
    EXTERNAL_WEB_NODE,
    MAX_SETUP_ACTIONS,
    NORMALIZE_EVERY_N_VISITS,
    SHORT_LIST_SOURCES,
    SHORT_LIST_VERIFICATIONS,
    EdgeDetails,
    GraphManager,
    PathStep,
    PathToStart,
    ScreenQuery,
    SetupTrail,
    _merge_elements,
    _merge_into_schema,
)
from android_app_graph.utils.vlm_utils import MAX_REQUESTS_IN_FLIGHT, TokenTracker

PACKAGE = "com.example.app"
HOME = "com.example.app.HomeActivity"
SETTINGS = "com.example.app.SettingsActivity"
OTHER_PACKAGE = "org.other.tool"
OTHER_APP = "org.other.tool.MainActivity"

# Activity class names as `dumpsys window` reports them, for packages of two and five dot-segments.
# Messages and Gmail share their first three segments but are separate packages.
WHATSAPP = "com.whatsapp"
WHATSAPP_HOME = "com.whatsapp.home.ui.HomeActivity"
WHATSAPP_GROUP_INFO = "com.whatsapp.chatinfo.group.GroupChatInfoActivity"
MESSAGES = "com.google.android.apps.messaging"
MESSAGES_LIST = "com.google.android.apps.messaging.ui.ConversationListActivity"
MESSAGES_CONVERSATION = "com.google.android.apps.messaging.ui.conversation.ConversationActivity"
GMAIL = "com.google.android.gm"
GMAIL_LIST = "com.google.android.gm.ConversationListActivityGmail"

any_package_length = pytest.mark.parametrize(
    ("package", "activity", "other_activity"),
    [
        (WHATSAPP, WHATSAPP_HOME, WHATSAPP_GROUP_INFO),
        (MESSAGES, MESSAGES_LIST, MESSAGES_CONVERSATION),
    ],
    ids=["two-segments", "five-segments"],
)

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

FAKE_CLIENT = cast("OpenRouter", object())  # never used: every consumer is patched
PAGE_DETAIL = ModelSettings("openai/gpt-6.1-sol")
EMBEDDING = ModelSettings("google/gemini-embedding-2", request_timeout=4.0)
IMAGE_EMBEDDING = ModelSettings("voyageai/voyage-multimodal-3.5", request_timeout=5.0)
# Every screenshot FakeVlm has no vector for looks alike.
ANY_IMAGE = [1.0, 0.0, 0.0]


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


SHOT_A = b64(b"screenshot-a")
SHOT_B = b64(b"screenshot-b")
SHOT_C = b64(b"screenshot-c")


def make_manager(steps_dir: Path | None = None, client: OpenRouter = FAKE_CLIENT) -> GraphManager:
    """A manager whose client is a sentinel unless given; every caller of the sentinel is
    patched."""
    gm = GraphManager(
        client=client,
        page_detail=PAGE_DETAIL,
        embedding=EMBEDDING,
        image_embedding=IMAGE_EMBEDDING,
    )
    gm.steps_dir = steps_dir
    return gm


def write_observation_image(steps_dir: Path, image: bytes) -> str:
    """Write an observation image into ``steps_dir`` under the next free step ID, as a
    StepRecorder would, and return the ID."""
    steps_dir.mkdir(parents=True, exist_ok=True)
    observation_id = f"{len(list(steps_dir.glob('*.webp'))):06d}"
    (steps_dir / f"{observation_id}.webp").write_bytes(image)
    return observation_id


class Membership(TypedDict):
    observations: list[str]
    representative_observation: str


def write_member_record(steps_dir: Path, observation_id: str) -> None:
    """Write the record exploration leaves for a member observation in ``steps_dir``: its
    pixel hash and describe output."""
    steps_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "id": observation_id,
        "kind": "observation",
        "pixel_hash": f"member-{observation_id}",
        "describe": {"page_description": "", "state": {}, "elements": [], "is_external_web": False},
    }
    (steps_dir / f"{observation_id}.json").write_text(json.dumps(record), "utf-8")


def members(steps_dir: Path, *images: bytes) -> Membership:
    """The membership fields of a node whose observations, ``b"screenshot-a"`` unless
    given, are members in ``steps_dir``; the first is the representative."""
    observation_ids = []
    for image in images or (b"screenshot-a",):
        observation_id = write_observation_image(steps_dir, image)
        write_member_record(steps_dir, observation_id)
        observation_ids.append(observation_id)
    return {"observations": observation_ids, "representative_observation": observation_ids[0]}


def fake_pixel_hash(screenshot: str) -> str:
    """Stands in for the hash of a screenshot's pixels: equal screenshots hash alike."""
    return hashlib.sha256(screenshot.encode("ascii")).hexdigest()


@dataclass(frozen=True)
class Screens:
    """Hands out observations as a StepRecorder would, with their images and records in
    ``steps_dir``."""

    steps_dir: Path

    def __call__(
        self, activity: str = HOME, screenshot: str = SHOT_A, package: str = PACKAGE
    ) -> Observation:
        observation_id = write_observation_image(self.steps_dir, base64.b64decode(screenshot))
        pixel_hash = fake_pixel_hash(screenshot)
        record = {
            "id": observation_id,
            "kind": "observation",
            "package": package,
            "pixel_hash": pixel_hash,
        }
        (self.steps_dir / f"{observation_id}.json").write_text(json.dumps(record), "utf-8")
        return Observation(observation_id, screenshot, activity, package, pixel_hash, black=False)

    def record(self, observation: Observation) -> dict[str, Any]:
        path = self.steps_dir / f"{observation.observation_id}.json"
        return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def screen(tmp_path: Path) -> Screens:
    return Screens(tmp_path / "steps")


def add_screen(
    gm: GraphManager,
    node_id: str,
    description: str,
    activity: str = HOME,
    **overrides: object,
) -> None:
    """Insert a node with the attribute set identify_state would create."""
    attrs: dict[str, object] = {
        "package": PACKAGE,
        "activity": activity,
        "activities": [activity],
        "page_description": description,
        "state_schema": {},
        "last_detail_snapshot": {},
        "interactable_elements": [],
        "last_normalized_visit_milestone": 0,
        "visit_count": 0,
        "replay_misses": 0,
    }
    attrs.update(overrides)
    gm.graph.add_node(node_id, **attrs)


def save_with_text_vectors(gm: GraphManager, path: Path) -> None:
    """Save ``gm`` to ``path`` with a text vector for each member, the embedding of its
    node's description, as exploration gives each screen it describes."""
    gm.save_graph(path)
    vectors = {
        member: EMBEDDINGS.get(data["page_description"], UNKNOWN_EMBEDDING)
        for _, data in gm.graph.nodes(data=True)
        for member in data.get("observations", [])
    }
    save_embeddings(text_embeddings_path(path), vectors, model=EMBEDDING.model)


def write_text_vectors(graph_path: Path, *memberships: Membership) -> None:
    """Write beside the graph file ``graph_path`` a text vector for each member of
    ``memberships``, as exploration keeps one for each screen it describes."""
    vectors = {
        member: UNKNOWN_EMBEDDING
        for membership in memberships
        for member in membership["observations"]
    }
    save_embeddings(text_embeddings_path(graph_path), vectors, model=EMBEDDING.model)


def with_text_vectors(gm: GraphManager, steps_dir: Path) -> GraphManager:
    """Return ``gm`` loaded back from the graph file beside ``steps_dir`` that
    ``save_with_text_vectors`` writes."""
    path = steps_dir.parent / "graph.json"
    save_with_text_vectors(gm, path)
    loaded = make_manager()
    loaded.load_graph(path)
    return loaded


@dataclass
class FakeVlm:
    """Scripted vlm_utils stand-ins; every parameter is recorded, so none is unused."""

    descriptions: list[tuple[str, dict[str, Any], list[dict[str, str]]]] = field(
        default_factory=list
    )
    external_web: set[str] = field(default_factory=set)
    verdicts: list[dict[str, Any]] = field(default_factory=list)
    images: dict[str, list[float]] = field(default_factory=dict)
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def get_image_embedding(
        self,
        client: OpenRouter | None,
        screenshot_b64: str,
        model: str = "",
        request_timeout: float = 0.0,
    ) -> list[float]:
        call = {"client": client, "screenshot": screenshot_b64, "model": model}
        call["timeout"] = request_timeout
        self.calls.append(("image", call))
        return self.images.get(screenshot_b64, ANY_IMAGE)

    def describe_page_and_state(
        self,
        client: OpenRouter | None,
        screenshot_b64: str,
        existing_nodes: list[str] | None = None,
        existing_keys: list[str] | None = None,
        settings: ModelSettings | None = None,
    ) -> tuple[str, dict[str, Any], list[dict[str, str]], bool]:
        call = {"client": client, "screenshot": screenshot_b64, "settings": settings}
        call.update(existing_nodes=existing_nodes, existing_keys=existing_keys)
        self.calls.append(("describe", call))
        description, state, elements = self.descriptions.pop(0)
        return description, state, elements, description in self.external_web

    def verify_same_node(
        self,
        client: OpenRouter | None,
        screenshot_new_b64: str,
        screenshot_existing_b64: str,
        existing_description: str,
        settings: ModelSettings | None = None,
    ) -> dict[str, Any]:
        call = {"client": client, "new": screenshot_new_b64, "existing": screenshot_existing_b64}
        call.update(existing_description=existing_description, settings=settings)
        self.calls.append(("verify", call))
        return self.verdicts.pop(0)

    def get_embedding(
        self, client: OpenRouter | None, text: str, model: str = "", request_timeout: float = 0.0
    ) -> list[float]:
        call = {"client": client, "text": text, "model": model, "timeout": request_timeout}
        self.calls.append(("embed", call))
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
        client: OpenRouter | None,
        graph_summary: str,
        app_name: str = "",
        settings: ModelSettings | None = None,
    ) -> dict[str, Any]:
        call = {"client": client, "graph_summary": graph_summary}
        call.update(app_name=app_name, settings=settings)
        self.calls.append(call)
        return self.result


@pytest.fixture
def vlm(monkeypatch: pytest.MonkeyPatch) -> FakeVlm:
    fake = FakeVlm()
    monkeypatch.setattr(gm_module, "describe_page_and_state", fake.describe_page_and_state)
    monkeypatch.setattr(gm_module, "verify_same_node", fake.verify_same_node)
    monkeypatch.setattr(gm_module, "get_embedding", fake.get_embedding)
    monkeypatch.setattr(gm_module, "get_image_embedding", fake.get_image_embedding)
    return fake


def test_network_guard_blocks_outbound_connections() -> None:
    with pytest.raises(RuntimeError, match="network access is disabled"):
        socket.create_connection(("127.0.0.1", 9), timeout=0.1)


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


def test_construction_defaults() -> None:
    gm = GraphManager()
    assert gm.graph.number_of_nodes() == 0
    assert gm.client is None
    assert gm.page_detail is None
    assert gm.embedding is None
    assert gm.image_embedding is None
    assert gm.total_steps_completed == 0


def test_identify_state_without_page_detail_settings_raises(screen: Screens) -> None:
    gm = GraphManager(embedding=EMBEDDING)
    with pytest.raises(RuntimeError, match="needs a client and page_detail settings"):
        gm.identify_state(screen(HOME, SHOT_A, PACKAGE))


def test_identify_state_without_embedding_settings_raises(vlm: FakeVlm, screen: Screens) -> None:
    vlm.descriptions.append(("Home screen", {}, []))
    gm = GraphManager(client=FAKE_CLIENT, page_detail=PAGE_DETAIL)
    with pytest.raises(RuntimeError, match="needs a client and embedding settings"):
        gm.identify_state(screen(HOME, SHOT_A, PACKAGE))


def test_identify_state_without_image_embedding_settings_raises(screen: Screens) -> None:
    gm = GraphManager(client=FAKE_CLIENT, page_detail=PAGE_DETAIL, embedding=EMBEDDING)
    gm.steps_dir = screen.steps_dir
    with pytest.raises(RuntimeError, match="needs a client and image_embedding settings"):
        gm.identify_state(screen(HOME, SHOT_A, PACKAGE))


def test_identify_state_without_a_step_log_raises(screen: Screens) -> None:
    with pytest.raises(RuntimeError, match="needs a steps_dir"):
        make_manager().identify_state(screen(HOME, SHOT_A, PACKAGE))


def test_identify_state_creates_the_first_node(vlm: FakeVlm, screen: Screens) -> None:
    vlm.descriptions.append(
        ("Home screen", {"tab": "home"}, [{"description": "Search box", "position": "10,20"}])
    )
    gm = make_manager(screen.steps_dir)
    observation = screen(HOME, SHOT_A, PACKAGE)

    node_id = gm.identify_state(observation)

    assert node_id == "s0_home_screen"
    assert vlm.kinds() == ["image", "describe", "embed"]
    node = gm.get_node(node_id)
    assert node is not None
    assert node["visit_count"] == 1
    assert node["state_schema"] == {"tab": ["home"]}
    assert node["observations"] == [observation.observation_id]
    assert node["representative_observation"] == observation.observation_id
    assert node["interactable_elements"] == [
        {"description": "Search box", "position": "10,20", "explored": False}
    ]
    assert gm.image_vector(observation.observation_id) == ANY_IMAGE


def test_identify_state_records_the_describe_output_and_the_decision(
    vlm: FakeVlm, screen: Screens
) -> None:
    elements = [{"description": "Search box", "position": "10,20"}]
    vlm.descriptions.append(("Home screen", {"tab": "home"}, elements))
    gm = make_manager(screen.steps_dir)
    observation = screen(HOME, SHOT_A, PACKAGE)

    gm.identify_state(observation)

    record = screen.record(observation)
    assert (record["id"], record["kind"]) == (observation.observation_id, "observation")
    assert record["describe"] == {
        "page_description": "Home screen",
        "state": {"tab": "home"},
        "elements": elements,
        "is_external_web": False,
    }
    assert record["identify"] == {
        "stage": "new",
        "expected_node": None,
        "exact_match": None,
        "candidates": [],
        "verifications": [],
        "refinement": None,
        "node": "s0_home_screen",
    }


def test_identify_state_passes_the_configured_models_and_clients(
    vlm: FakeVlm, screen: Screens
) -> None:
    vlm.descriptions.append(("Home screen", {}, []))
    gm = make_manager(screen.steps_dir)

    gm.identify_state(screen(HOME, SHOT_A, PACKAGE))

    image, describe, embed = (call for _, call in vlm.calls)
    assert (image["client"], image["screenshot"]) == (FAKE_CLIENT, SHOT_A)
    assert (image["model"], image["timeout"]) == (
        IMAGE_EMBEDDING.model,
        IMAGE_EMBEDDING.request_timeout,
    )
    assert (describe["client"], describe["settings"]) == (FAKE_CLIENT, PAGE_DETAIL)
    assert (embed["model"], embed["timeout"]) == (EMBEDDING.model, EMBEDDING.request_timeout)


def test_an_exact_match_reuses_the_member_s_describe_output_without_describing_or_verifying(
    vlm: FakeVlm, screen: Screens
) -> None:
    elements = [{"description": "Search box", "position": "10,20"}]
    vlm.descriptions.append(("Home screen", {"tab": "home"}, elements))
    gm = make_manager(screen.steps_dir)
    first_observation = screen(HOME, SHOT_A, PACKAGE)
    first = gm.identify_state(first_observation)
    second_observation = screen(SETTINGS, SHOT_A, PACKAGE)

    second = gm.identify_state(second_observation, expected_node="s9_elsewhere")

    assert second == first
    assert vlm.kinds() == ["image", "describe", "embed", "image"]
    node = gm.graph.nodes[first]
    assert node["observations"] == [
        first_observation.observation_id,
        second_observation.observation_id,
    ]
    assert node["representative_observation"] == first_observation.observation_id
    assert node["visit_count"] == 2
    assert node["activities"] == [HOME, SETTINGS]
    assert node["last_detail_snapshot"] == {"tab": "home"}
    record = screen.record(second_observation)
    assert record["describe"] == {
        "page_description": "Home screen",
        "state": {"tab": "home"},
        "elements": elements,
        "is_external_web": False,
        "copied_from": first_observation.observation_id,
    }
    assert record["identify"] == {
        "stage": "exact",
        "expected_node": "s9_elsewhere",
        "exact_match": first_observation.observation_id,
        "candidates": [],
        "verifications": [],
        "refinement": None,
        "node": first,
    }
    assert gm.image_vector(second_observation.observation_id) == ANY_IMAGE


def test_an_exact_match_copies_from_the_latest_member_with_the_pixels(
    vlm: FakeVlm, screen: Screens
) -> None:
    vlm.descriptions.append(("Home screen", {}, []))
    gm = make_manager(screen.steps_dir)
    gm.identify_state(screen(HOME, SHOT_A, PACKAGE))
    copy = screen(HOME, SHOT_A, PACKAGE)
    gm.identify_state(copy)

    third = screen(HOME, SHOT_A, PACKAGE)
    gm.identify_state(third)

    assert screen.record(third)["describe"]["copied_from"] == copy.observation_id


def test_an_observation_without_a_pixel_hash_is_never_an_exact_match(
    vlm: FakeVlm, screen: Screens
) -> None:
    """A state read past the recorder has no hash, which must not match another such."""
    vlm.descriptions.extend([("Home screen", {}, [])] * 2)
    vlm.verdicts.append({"same": True, "reason": "home"})
    gm = make_manager(screen.steps_dir)
    for _ in range(2):
        unhashed = dataclasses.replace(screen(HOME, SHOT_A, PACKAGE), pixel_hash="")
        gm.identify_state(unhashed)

    assert vlm.kinds()[3:] == ["image", "describe", "embed", "verify"]


def test_a_pixel_hash_shared_by_members_of_two_nodes_does_not_decide(
    vlm: FakeVlm, screen: Screens
) -> None:
    """A merge can leave identical pixels in two nodes once the audit splits them again."""
    vlm.descriptions.extend([("Home screen", {}, []), ("Settings page", {}, [])] * 2)
    vlm.verdicts.extend(
        [{"same": False, "reason": "another page"}, {"same": True, "reason": "the same page"}]
    )
    gm = make_manager(screen.steps_dir)
    home = gm.identify_state(screen(HOME, SHOT_A, PACKAGE))
    settings = gm.identify_state(screen(SETTINGS, SHOT_B, PACKAGE))
    gm.graph.nodes[home]["observations"].extend(gm.graph.nodes[settings]["observations"])
    vlm.calls.clear()
    observation = screen(SETTINGS, SHOT_B, PACKAGE)

    gm.identify_state(observation)

    assert vlm.kinds() == ["image", "describe", "embed", "verify"]
    identify = screen.record(observation)["identify"]
    assert (identify["stage"], identify["exact_match"]) == ("short_list", None)


def test_identify_state_reports_external_web_pages_as_one_shared_node(
    vlm: FakeVlm, screen: Screens
) -> None:
    vlm.descriptions.extend([("News article", {"title": "x"}, []), ("Sign-in form", {}, [])])
    vlm.external_web.update({"News article", "Sign-in form"})
    gm = make_manager(screen.steps_dir)
    article = screen(HOME, SHOT_A, PACKAGE)
    sign_in = screen(SETTINGS, SHOT_B, PACKAGE)

    first = gm.identify_state(article)
    second = gm.identify_state(sign_in)

    assert first == second == EXTERNAL_WEB_NODE
    assert list(gm.graph) == [EXTERNAL_WEB_NODE]
    assert gm.get_node(EXTERNAL_WEB_NODE) == {
        "page_description": "[external web page]",
        "replay_misses": 0,
        "visit_count": 2,
        "observations": [article.observation_id, sign_in.observation_id],
        "representative_observation": article.observation_id,
    }
    assert vlm.kinds() == ["image", "describe", "image", "describe"]
    assert gm.get_start_node() is None
    assert screen.record(article)["describe"]["is_external_web"] is True
    assert screen.record(article)["identify"]["stage"] == "external_web"
    assert screen.record(article)["identify"]["node"] == EXTERNAL_WEB_NODE
    assert gm.image_vector(article.observation_id) == ANY_IMAGE


def test_identify_state_matches_the_screen_behind_an_external_web_page_exactly(
    vlm: FakeVlm,
    screen: Screens,
) -> None:
    vlm.descriptions.extend([("Home screen", {}, []), ("News article", {}, [])])
    vlm.external_web.add("News article")
    gm = make_manager(screen.steps_dir)
    home = gm.identify_state(screen(HOME, SHOT_A, PACKAGE))
    gm.identify_state(screen(HOME, SHOT_B, PACKAGE))

    assert gm.identify_state(screen(HOME, SHOT_A, PACKAGE)) == home
    assert vlm.kinds() == ["image", "describe", "embed", "image", "describe", "image"]


def test_identify_state_verified_match_merges_into_existing_node(
    vlm: FakeVlm, screen: Screens
) -> None:
    vlm.descriptions.extend(
        [("Home screen", {"tab": "home"}, []), ("Home screen with banner", {"tab": "deals"}, [])]
    )
    vlm.verdicts.append({"same": True, "reason": "identical layout"})
    gm = make_manager(screen.steps_dir)
    home = screen(HOME, SHOT_A, PACKAGE)
    first = gm.identify_state(home)
    banner = screen(SETTINGS, SHOT_B, PACKAGE)

    second = gm.identify_state(banner)

    assert second == first
    assert vlm.kinds() == ["image", "describe", "embed", "image", "describe", "embed", "verify"]
    assert vlm.calls[4][1]["existing_nodes"] == ["Home screen"]
    assert vlm.calls[4][1]["existing_keys"] == ["tab"]
    node = gm.get_node(first)
    assert node is not None
    assert node["activities"] == [HOME, SETTINGS]
    assert node["visit_count"] == 2
    assert node["state_schema"] == {"tab": ["home", "deals"]}
    assert node["observations"] == [home.observation_id, banner.observation_id]
    assert node["representative_observation"] == home.observation_id
    identify = screen.record(banner)["identify"]
    assert identify["stage"] == "short_list"
    assert identify["candidates"] == [
        {
            "node": first,
            "sources": {"image": pytest.approx(1.0), "text": pytest.approx(1.0)},
            "closest_member": home.observation_id,
            "image_score": pytest.approx(1.0),
        }
    ]
    assert identify["verifications"] == [
        {
            "node": first,
            "member": home.observation_id,
            "same": True,
            "answered": True,
            "reason": "identical layout",
        }
    ]
    assert identify["node"] == first


def test_identify_state_verifies_against_the_member_image_in_the_step_log(
    vlm: FakeVlm, screen: Screens
) -> None:
    """The verifier compares the new screen with the member's image as the step log holds
    it, not with any screenshot identify_state was given."""
    vlm.descriptions.extend([("Home screen", {}, []), ("Home screen with banner", {}, [])])
    vlm.verdicts.append({"same": True, "reason": "identical layout"})
    gm = make_manager(screen.steps_dir)
    home = screen(HOME, SHOT_A, PACKAGE)
    gm.identify_state(home)
    (screen.steps_dir / f"{home.observation_id}.webp").write_bytes(b"member-image")

    gm.identify_state(screen(HOME, SHOT_B, PACKAGE))

    assert vlm.calls[6][1]["existing"] == b64(b"member-image")
    assert vlm.calls[6][1]["new"] == SHOT_B


def test_identify_state_verified_match_does_not_repeat_a_known_activity(
    vlm: FakeVlm, screen: Screens
) -> None:
    vlm.descriptions.extend([("Home screen", {}, []), ("Home screen with banner", {}, [])])
    vlm.verdicts.append({"same": True, "reason": "identical layout"})
    gm = make_manager(screen.steps_dir)
    first = gm.identify_state(screen(HOME, SHOT_A, PACKAGE))

    gm.identify_state(screen(HOME, SHOT_B, PACKAGE))

    node = gm.get_node(first)
    assert node is not None
    assert node["activities"] == [HOME]


def test_identify_state_rejected_match_renames_existing_and_creates_new(
    vlm: FakeVlm, screen: Screens
) -> None:
    vlm.descriptions.extend([("Home screen", {}, []), ("Home screen with banner", {}, [])])
    vlm.verdicts.append(
        {
            "same": False,
            "reason": "overlay",
            "existing_description": "Home feed",
            "new_description": "Promo overlay",
        }
    )
    gm = make_manager(screen.steps_dir)
    home = screen(HOME, SHOT_A, PACKAGE)
    first = gm.identify_state(home)
    overlay = screen(HOME, SHOT_B, PACKAGE)

    second = gm.identify_state(overlay)

    assert first not in gm.graph
    assert "s0_home_feed" in gm.graph
    assert second == "s1_promo_overlay"
    assert vlm.kinds() == [
        *("image", "describe", "embed"),
        *("image", "describe", "embed", "verify", "embed"),
    ]
    renamed = gm.get_node("s0_home_feed")
    assert renamed is not None
    assert renamed["page_description"] == "Home feed"
    assert renamed["observations"] == [home.observation_id]
    assert renamed["representative_observation"] == home.observation_id
    split = gm.graph.nodes[second]
    assert split["observations"] == [overlay.observation_id]
    assert split["representative_observation"] == overlay.observation_id
    identify = screen.record(overlay)["identify"]
    assert (identify["stage"], identify["node"]) == ("new", second)
    assert identify["refinement"] == {
        "node": first,
        "renamed_to": "s0_home_feed",
        "existing_description": "Home feed",
        "new_description": "Promo overlay",
    }


def test_identify_state_rejected_match_without_refinements_creates_new(
    vlm: FakeVlm, screen: Screens
) -> None:
    vlm.descriptions.extend([("Home screen", {}, []), ("Home screen with banner", {}, [])])
    vlm.verdicts.append(
        {
            "same": False,
            "reason": "different",
            "existing_description": "Home screen",
            "new_description": "",
        }
    )
    gm = make_manager(screen.steps_dir)
    first = gm.identify_state(screen(HOME, SHOT_A, PACKAGE))
    banner = screen(HOME, SHOT_B, PACKAGE)

    second = gm.identify_state(banner)

    assert first in gm.graph  # no rename: the verifier repeated the existing description
    assert second == "s1_home_screen_with_banner"
    assert vlm.kinds() == ["image", "describe", "embed", "image", "describe", "embed", "verify"]
    assert screen.record(banner)["identify"]["refinement"] is None


def test_a_node_without_members_is_never_a_candidate(vlm: FakeVlm, screen: Screens) -> None:
    """A node with no member has no image for the verifier to compare with."""
    vlm.descriptions.append(("Home screen", {}, []))
    gm = make_manager(screen.steps_dir)
    add_screen(gm, "s0_home", "Home screen")

    node_id = gm.identify_state(screen(HOME, SHOT_A, PACKAGE))

    assert node_id == "s0_home_screen"
    assert vlm.kinds() == ["image", "describe", "embed"]


def test_identify_state_ignores_nodes_of_another_package(vlm: FakeVlm, screen: Screens) -> None:
    vlm.descriptions.append(("Home screen", {}, []))
    gm = make_manager(screen.steps_dir)
    add_screen(
        gm,
        "s0_other",
        "Home screen",
        activity=OTHER_APP,
        package=OTHER_PACKAGE,
        **members(screen.steps_dir),
    )

    node_id = gm.identify_state(screen(HOME, SHOT_A, PACKAGE))

    assert node_id == "s0_home_screen"
    assert vlm.calls[1][1]["existing_nodes"] is None
    assert vlm.calls[1][1]["existing_keys"] is None
    assert "verify" not in vlm.kinds()


def test_identify_state_lists_a_state_key_shared_by_two_nodes_once(
    vlm: FakeVlm, screen: Screens
) -> None:
    vlm.descriptions.append(("Search results", {}, []))
    gm = make_manager(screen.steps_dir)
    add_screen(gm, "s0_home", "Home screen", state_schema={"tab": ["home"], "sort": ["price"]})
    add_screen(
        gm, "s1_settings", "Settings page", state_schema={"tab": ["deals"], "filter": ["new"]}
    )

    gm.identify_state(screen(HOME, SHOT_A, PACKAGE))

    assert vlm.calls[1][1]["existing_keys"] == ["tab", "sort", "filter"]


def test_the_text_source_skips_text_vectors_of_another_dimension_with_one_warning(
    vlm: FakeVlm, caplog: pytest.LogCaptureFixture, screen: Screens, tmp_path: Path
) -> None:
    """A vector tagged with the configured embedding model but of another dimension is
    never scored against the screen's (see cosine_similarity's ValueError)."""
    gm = make_manager(screen.steps_dir)
    add_screen(gm, "s0_stale", "Home screen", **members(screen.steps_dir, b"stale", b"stale-2"))
    add_screen(gm, "s1_home", "Home feed", **members(screen.steps_dir))
    vlm.images.update({b64(b"stale"): FAR, b64(b"stale-2"): FAR})
    gm.embed_missing_members()
    path = tmp_path / "graph.json"
    gm.save_graph(path)
    (home,) = gm.graph.nodes["s1_home"]["observations"]
    vectors = {member: [1.0, 0.0] for member in gm.graph.nodes["s0_stale"]["observations"]}
    vectors[home] = EMBEDDINGS["Home feed"]
    save_embeddings(text_embeddings_path(path), vectors, model=EMBEDDING.model)
    loaded = make_manager()
    loaded.load_graph(path)
    vlm.descriptions.append(("Home screen", {}, []))
    vlm.verdicts.append({"same": True, "reason": "same page"})

    with caplog.at_level("WARNING"):
        node_id = loaded.identify_state(screen(HOME, SHOT_A, PACKAGE))

    assert node_id == "s1_home"
    (stale_warning,) = [
        m for m in caplog.messages if "description-embedding: embedding cache is stale" in m
    ]
    assert "stale for 2 vector(s): query dim=3, cached dim(s)=[2]" in stale_warning


def test_a_node_whose_description_scores_low_is_verified(vlm: FakeVlm, screen: Screens) -> None:
    """A short-listed node is verified whatever its scores; a rejection makes a new node."""
    vlm.descriptions.append(("Settings page", {}, []))
    vlm.verdicts.append({"same": False, "reason": "another page"})
    gm = make_manager(screen.steps_dir)
    add_screen(gm, "s0_home", "Home screen", **members(screen.steps_dir, b"screenshot-b"))
    vlm.images[SHOT_B] = FAR
    gm.embed_missing_members()
    vlm.calls.clear()

    node_id = gm.identify_state(screen(HOME, SHOT_A, PACKAGE))

    assert node_id == "s0_settings_page"
    assert vlm.kinds() == ["image", "describe", "embed", "verify"]


# Unit image vectors: one per node, so a screen is like exactly the nodes it shares one with.
NEAR = [0.0, 1.0, 0.0]
FAR = [0.0, 0.0, 1.0]


def verified(vlm: FakeVlm) -> list[str]:
    """The existing images the verifier was shown, in order."""
    return [call["existing"] for kind, call in vlm.calls if kind == "verify"]


def _rejection(node: str) -> dict[str, Any]:
    """A rejection that refines both descriptions; the screen's keeps its embedding."""
    return {
        "same": False,
        "reason": f"not {node}",
        "existing_description": f"{node} refined",
        "new_description": "Search results",
    }


def test_a_node_whose_member_image_ranks_first_is_verified_against_that_member(
    vlm: FakeVlm, screen: Screens
) -> None:
    """Its description scores 0 and its representative looks unlike the screen; its other
    member is the screen's look-alike."""
    explored = make_manager(screen.steps_dir)
    add_screen(
        explored, "s0_settings", "Settings page", **members(screen.steps_dir, b"rep", b"twin")
    )
    vlm.images.update({b64(b"rep"): FAR, b64(b"twin"): NEAR, SHOT_A: NEAR})
    explored.embed_missing_members()
    gm = with_text_vectors(explored, screen.steps_dir)
    vlm.calls.clear()
    vlm.descriptions.append(("Search results", {}, []))
    vlm.verdicts.append({"same": True, "reason": "a scrolled list"})
    observation = screen(HOME, SHOT_A, PACKAGE)

    assert gm.identify_state(observation) == "s0_settings"

    assert verified(vlm) == [b64(b"twin")]
    (candidate,) = screen.record(observation)["identify"]["candidates"]
    assert candidate["closest_member"] == gm.graph.nodes["s0_settings"]["observations"][1]
    assert candidate["sources"] == {"image": pytest.approx(1.0), "text": pytest.approx(0.0)}


def _three_image_matches_and_a_text_match(vlm: FakeVlm, steps_dir: Path) -> GraphManager:
    """s0-s2 look like the screen and are described otherwise; s3 is described as it is
    and looks unlike it."""
    gm = make_manager(steps_dir)
    for number, image in enumerate((b"a", b"b", b"c")):
        add_screen(gm, f"s{number}_lookalike", "Settings page", **members(steps_dir, image))
        vlm.images[b64(image)] = NEAR
    add_screen(gm, "s3_results", "Search results", **members(steps_dir, b"d"))
    vlm.images[b64(b"d")] = FAR
    vlm.images[SHOT_A] = NEAR
    gm.embed_missing_members()
    vlm.calls.clear()
    vlm.descriptions.append(("Search results", {}, []))
    return with_text_vectors(gm, steps_dir)


def test_the_best_text_match_is_verified_within_the_cap_behind_three_image_matches(
    vlm: FakeVlm, screen: Screens
) -> None:
    gm = _three_image_matches_and_a_text_match(vlm, screen.steps_dir)
    vlm.verdicts.extend([_rejection("s0"), {"same": True, "reason": "the results"}])
    observation = screen(HOME, SHOT_A, PACKAGE)

    assert gm.identify_state(observation) == "s3_results"

    assert verified(vlm) == [b64(b"a"), b64(b"d")]
    candidates = screen.record(observation)["identify"]["candidates"]
    assert [(c["node"], sorted(c["sources"])) for c in candidates] == [
        ("s0_lookalike", ["image", "text"]),
        ("s3_results", ["text"]),
        ("s1_lookalike", ["image"]),
        ("s2_lookalike", ["image"]),
    ]


def test_verification_stops_at_the_first_confirmation(vlm: FakeVlm, screen: Screens) -> None:
    gm = _three_image_matches_and_a_text_match(vlm, screen.steps_dir)
    vlm.verdicts.append({"same": True, "reason": "the same list"})

    assert gm.identify_state(screen(HOME, SHOT_A, PACKAGE)) == "s0_lookalike"

    assert vlm.kinds() == ["image", "describe", "embed", "verify"]


def test_the_short_list_makes_at_most_three_verifier_calls_and_refines_only_once(
    vlm: FakeVlm, screen: Screens
) -> None:
    gm = _three_image_matches_and_a_text_match(vlm, screen.steps_dir)
    vlm.verdicts.extend(_rejection(node) for node in ("s0", "s3", "s1"))
    observation = screen(HOME, SHOT_A, PACKAGE)

    node_id = gm.identify_state(observation)

    assert len(verified(vlm)) == SHORT_LIST_VERIFICATIONS == 3
    assert node_id == "s0_search_results"
    assert sorted(gm.graph) == [
        "s0_s0_refined",
        "s0_search_results",
        "s1_lookalike",
        "s2_lookalike",
        "s3_results",
    ]
    identify = screen.record(observation)["identify"]
    assert [v["node"] for v in identify["verifications"]] == [
        "s0_lookalike",
        "s3_results",
        "s1_lookalike",
    ]
    assert identify["refinement"] == {
        "node": "s0_lookalike",
        "renamed_to": "s0_s0_refined",
        "existing_description": "s0 refined",
        "new_description": "Search results",
    }


def test_an_expected_node_is_verified_right_after_the_describe_call_and_skips_the_short_list(
    vlm: FakeVlm, screen: Screens
) -> None:
    gm = _three_image_matches_and_a_text_match(vlm, screen.steps_dir)
    vlm.verdicts.append({"same": True, "reason": "the page the replay leads to"})
    observation = screen(HOME, SHOT_A, PACKAGE)

    assert gm.identify_state(observation, expected_node="s2_lookalike") == "s2_lookalike"

    assert vlm.kinds() == ["image", "describe", "embed", "verify"]
    assert verified(vlm) == [b64(b"c")]
    identify = screen.record(observation)["identify"]
    assert (identify["stage"], identify["expected_node"]) == ("expected", "s2_lookalike")
    assert identify["candidates"] == []


def test_a_rejected_expected_node_stays_off_the_short_list(vlm: FakeVlm, screen: Screens) -> None:
    """The order of model calls: image embedding, describe, text embedding, the expected
    node, the short list."""
    gm = _three_image_matches_and_a_text_match(vlm, screen.steps_dir)
    vlm.verdicts.extend([_rejection("s0"), {"same": True, "reason": "the same list"}])
    observation = screen(HOME, SHOT_A, PACKAGE)

    assert gm.identify_state(observation, expected_node="s0_lookalike") == "s1_lookalike"

    assert vlm.kinds() == ["image", "describe", "embed", "verify", "embed", "verify"]
    assert verified(vlm) == [b64(b"a"), b64(b"b")]
    identify = screen.record(observation)["identify"]
    assert [c["node"] for c in identify["candidates"]] == [
        "s1_lookalike",
        "s3_results",
        "s2_lookalike",
    ]
    assert identify["stage"] == "short_list"
    assert identify["refinement"]["renamed_to"] == "s0_s0_refined"


def test_an_expected_node_adds_one_verifier_call_to_the_cap(vlm: FakeVlm, screen: Screens) -> None:
    gm = _three_image_matches_and_a_text_match(vlm, screen.steps_dir)
    vlm.verdicts.extend(_rejection(node) for node in ("s2", "s0", "s3", "s1"))

    gm.identify_state(screen(HOME, SHOT_A, PACKAGE), expected_node="s2_lookalike")

    assert verified(vlm) == [b64(image) for image in (b"c", b"a", b"d", b"b")]


@pytest.mark.parametrize(
    "expected_node",
    [
        pytest.param("s9_missing", id="not in the graph"),
        pytest.param("s1_other_app", id="of another package"),
        pytest.param("s2_no_members", id="without members"),
        pytest.param("ext_org.other.tool", id="external"),
    ],
)
def test_an_expected_node_that_cannot_be_compared_is_not_checked(
    expected_node: str, vlm: FakeVlm, screen: Screens
) -> None:
    gm = make_manager(screen.steps_dir)
    add_screen(gm, "s0_home", "Home screen", **members(screen.steps_dir))
    add_screen(
        gm, "s1_other_app", "Home screen", package=OTHER_PACKAGE, **members(screen.steps_dir)
    )
    add_screen(gm, "s2_no_members", "Home screen")
    gm.record_external_app_visit(screen(OTHER_APP, SHOT_B, OTHER_PACKAGE))
    gm.embed_missing_members()
    vlm.calls.clear()
    vlm.descriptions.append(("Home screen", {}, []))
    vlm.verdicts.append({"same": True, "reason": "home"})

    assert gm.identify_state(screen(HOME, SHOT_A, PACKAGE), expected_node=expected_node) == (
        "s0_home"
    )

    assert vlm.kinds() == ["image", "describe", "embed", "verify"]


def test_an_expected_node_is_followed_through_its_renames(vlm: FakeVlm, screen: Screens) -> None:
    gm = make_manager(screen.steps_dir)
    add_screen(gm, "s0_home", "Home screen", **members(screen.steps_dir))
    gm.embed_missing_members()
    renamed = gm.rename_node("s0_home", "Home feed")
    vlm.calls.clear()
    vlm.descriptions.append(("Home screen", {}, []))
    vlm.verdicts.append({"same": True, "reason": "home"})
    observation = screen(HOME, SHOT_A, PACKAGE)

    assert gm.identify_state(observation, expected_node="s0_home") == renamed

    assert screen.record(observation)["identify"]["stage"] == "expected"


def test_a_verifier_that_never_answers_is_asked_twice_and_the_screen_starts_a_new_node(
    vlm: FakeVlm,
    screen: Screens,
    timing_out_client: OpenRouter,
    timed_out_requests: list[httpx.Request],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The verifier times out, is asked once more and times out again on a node whose
    member looks and is described exactly like the screen; the check counts as a
    rejection, recorded as no answer."""
    monkeypatch.setattr(gm_module, "verify_same_node", vlm_utils.verify_same_node)
    vlm.descriptions.extend([("Home screen", {}, []), ("Home screen", {}, [])])
    gm = make_manager(screen.steps_dir, client=timing_out_client)
    home = screen(HOME, SHOT_A, PACKAGE)
    first = gm.identify_state(home)
    again = screen(HOME, SHOT_B, PACKAGE)

    with caplog.at_level(logging.INFO):
        node_id = gm.identify_state(again)

    assert node_id == "s1_home_screen"
    assert sorted(gm.graph) == [first, node_id]
    assert caplog.text.count("node_verify: timed out after retries") == 2
    assert len(timed_out_requests) == 4
    assert (
        f"Verifier gave no answer on node {first} against member {home.observation_id} "
        "(timed out); it counts as a rejection"
    ) in caplog.text
    identify = screen.record(again)["identify"]
    assert identify["stage"] == "new"
    assert identify["verifications"] == [
        {
            "node": first,
            "member": home.observation_id,
            "same": False,
            "answered": False,
            "reason": "timed out",
        }
    ]


def _png(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return b64(buffer.getvalue())


class _StatusBarDevice:
    """A device that shows the screens it is given in turn, under a 10-row status bar."""

    w = 64
    h = 48

    def __init__(self, *screens: Image.Image) -> None:
        self.config: dict[str, Any] = {}
        self._screens = list(screens)

    def get_state(self) -> dict[str, Any]:
        return {
            "screenshot": _png(self._screens.pop(0)),
            "activity": HOME,
            "package": PACKAGE,
            "status_bar": [0, 0, 64, 10],
        }

    def exe_action(self, action: dict[str, Any]) -> None:
        msg = f"No device action expected: {action}"
        raise AssertionError(msg)


def _screen_with_clock(minute: int) -> Image.Image:
    image = Image.linear_gradient("L").resize((64, 48)).convert("RGB")
    ImageDraw.Draw(image).rectangle((2, 2, 2 + minute, 6), fill=(255, 0, 0))
    return image


@pytest.mark.parametrize(("changed_row", "exact"), [(9, True), (10, False)])
def test_a_screen_that_differs_only_inside_the_status_bar_is_an_exact_match(
    changed_row: int, exact: bool, vlm: FakeVlm, tmp_path: Path
) -> None:
    later = _screen_with_clock(9)
    later.putpixel((40, changed_row), (0, 0, 255))
    recorder = StepRecorder(
        _StatusBarDevice(_screen_with_clock(1), later),
        tmp_path / "steps",
        run_kind="explore",
        source="main_loop",
    )
    gm = make_manager(recorder.steps)
    vlm.descriptions.extend([("Home screen", {}, [])] * 2)
    vlm.verdicts.append({"same": True, "reason": "home"})
    home = gm.identify_state(Observation.from_state(recorder.get_state()))

    assert gm.identify_state(Observation.from_state(recorder.get_state())) == home

    assert ("describe" not in vlm.kinds()[3:]) is exact
    assert ("verify" in vlm.kinds()) is not exact


def test_a_reloaded_graph_matches_its_members_exactly(vlm: FakeVlm, tmp_path: Path) -> None:
    steps_dir = tmp_path / "steps"
    recorder = StepRecorder(
        _StatusBarDevice(*[Image.new("RGB", (64, 48), "white")] * 2),
        steps_dir,
        run_kind="explore",
        source="main_loop",
    )
    gm = make_manager(steps_dir)
    vlm.descriptions.append(("Home screen", {"tab": "home"}, []))
    first = Observation.from_state(recorder.get_state())
    node_id = gm.identify_state(first)
    gm.save_graph(tmp_path / "graph.json")
    loaded = make_manager()
    loaded.load_graph(tmp_path / "graph.json")
    vlm.calls.clear()

    second = Observation.from_state(recorder.get_state())

    assert loaded.identify_state(second) == node_id
    assert vlm.kinds() == ["image"]
    assert loaded.graph.nodes[node_id]["last_detail_snapshot"] == {"tab": "home"}
    record = json.loads((steps_dir / f"{second.observation_id}.json").read_text("utf-8"))
    assert record["describe"]["copied_from"] == first.observation_id


def test_a_described_screen_s_text_vector_is_saved_beside_the_graph_and_loaded_back(
    vlm: FakeVlm, screen: Screens, tmp_path: Path
) -> None:
    vlm.descriptions.append(("Home screen", {}, []))
    gm = make_manager(screen.steps_dir)
    observation = screen(HOME, SHOT_A, PACKAGE)
    gm.identify_state(observation)
    path = tmp_path / "graph.json"

    gm.save_graph(path)
    loaded = make_manager()
    loaded.load_graph(path)

    assert json.loads((tmp_path / "graph_text_emb.json").read_text("utf-8")) == {
        "model": EMBEDDING.model,
        "embeddings": {observation.observation_id: EMBEDDINGS["Home screen"]},
    }
    assert loaded.text_vector(observation.observation_id) == EMBEDDINGS["Home screen"]


def test_a_graph_without_the_text_vector_of_a_described_member_fails_to_load(
    tmp_path: Path,
) -> None:
    steps_dir = tmp_path / "steps"
    member = members(steps_dir, b"kept", b"lost")
    kept, lost = member["observations"]
    describe = {"page_description": "Home screen", "state": {}, "elements": []}
    for observation_id in (kept, lost):
        record = {"pixel_hash": f"hash-{observation_id}", "describe": describe}
        (steps_dir / f"{observation_id}.json").write_text(json.dumps(record), "utf-8")
    path = tmp_path / "graph.json"
    path.write_text(
        json.dumps(
            {
                "nodes": [{"id": "s0_home", "package": PACKAGE, "replay_misses": 0, **member}],
                "edges": [],
            }
        ),
        "utf-8",
    )
    vectors = {kept: EMBEDDINGS["Home screen"]}
    save_embeddings(text_embeddings_path(path), vectors, model=EMBEDDING.model)

    with pytest.raises(ValueError, match=r"graph_text_emb\.json has no vector") as raised:
        make_manager().load_graph(path)

    assert str(raised.value).endswith(
        f"for observation(s) {lost}. Explore the app again into a new graph directory."
    )


def test_an_exact_match_records_its_member_s_text_vector_without_a_request(
    vlm: FakeVlm, screen: Screens, tmp_path: Path
) -> None:
    vlm.descriptions.append(("Home screen", {}, []))
    gm = make_manager(screen.steps_dir)
    member = screen(HOME, SHOT_A, PACKAGE)
    gm.identify_state(member)
    copy = screen(HOME, SHOT_A, PACKAGE)

    gm.identify_state(copy)
    gm.save_graph(tmp_path / "graph.json")

    assert vlm.kinds() == ["image", "describe", "embed", "image"]
    stored = json.loads((tmp_path / "graph_text_emb.json").read_text("utf-8"))
    assert stored["embeddings"] == {
        member.observation_id: EMBEDDINGS["Home screen"],
        copy.observation_id: EMBEDDINGS["Home screen"],
    }


def test_a_screen_keeps_the_vector_of_the_description_its_first_rejection_refined(
    vlm: FakeVlm, screen: Screens
) -> None:
    vlm.descriptions.extend([("Home screen", {}, []), ("Home screen with banner", {}, [])])
    vlm.verdicts.append(
        {
            "same": False,
            "reason": "an overlay",
            "existing_description": "",
            "new_description": "Promo overlay",
        }
    )
    gm = make_manager(screen.steps_dir)
    gm.identify_state(screen(HOME, SHOT_A, PACKAGE))
    overlay = screen(HOME, SHOT_B, PACKAGE)

    gm.identify_state(overlay)

    assert gm.text_vector(overlay.observation_id) == EMBEDDINGS["Promo overlay"]


DESCRIBED = {"page_description": "Home", "state": {}, "elements": [], "is_external_web": False}


@pytest.mark.parametrize(
    "record",
    [
        pytest.param(None, id="no-record"),
        pytest.param({"describe": DESCRIBED}, id="no-pixel-hash"),
        pytest.param({"pixel_hash": "abc"}, id="no-describe"),
        pytest.param({"pixel_hash": "abc", "black": False}, id="no-describe-not-black"),
        pytest.param({"black": True, "activity": "com.example.app.Home"}, id="black-no-pixel-hash"),
        pytest.param(
            {"pixel_hash": "abc", "describe": {**DESCRIBED, "page_description": 3}},
            id="description",
        ),
        pytest.param({"pixel_hash": "abc", "describe": {**DESCRIBED, "state": []}}, id="state"),
        pytest.param(
            {"pixel_hash": "abc", "describe": {**DESCRIBED, "elements": {}}}, id="elements"
        ),
    ],
)
def test_load_graph_rejects_an_in_app_member_without_a_pixel_hash_and_describe_output(
    record: dict[str, object] | None, tmp_path: Path
) -> None:
    """Exact matching and member checks read both from every member of an in-app node,
    except a black one, whose description comes from its activity; a member of an
    ``ext_`` node needs neither."""
    steps_dir = tmp_path / "steps"
    home = members(steps_dir, b"home", b"home-2")
    undescribed = home["observations"][1]
    step_record = steps_dir / f"{undescribed}.json"
    if record is None:
        step_record.unlink()
    else:
        step_record.write_text(json.dumps(record), "utf-8")
    browser = write_observation_image(steps_dir, b"browser")
    path = tmp_path / "graph.json"
    write_graph(
        path,
        [
            {"id": "s0_home", "replay_misses": 0, **home},
            {
                "id": EXTERNAL_WEB_NODE,
                "replay_misses": 0,
                "observations": [browser],
                "representative_observation": browser,
            },
        ],
    )

    with pytest.raises(
        ValueError, match="no pixel hash, or no describe output and is not of a black screenshot"
    ) as excinfo:
        make_manager().load_graph(path)

    message = str(excinfo.value)
    assert f"[{undescribed!r}]" in message
    assert str(path) in message
    assert "Explore the app again into a new graph directory" in message


SET_PATTERN = "com.example.app.password.ChooseLockPattern"
SET_PIN = "com.example.app.password.ChooseLockPassword"
PATTERN_NODE = "s0_secure_screen:_chooselockpattern"
GOOGLE_SIGN_IN = "com.google.android.gms.auth.api.signin.internal.SignInHubActivity"


class _ActivityDevice:
    """A device that shows the screens it is given in turn: each an activity, its image
    and its package."""

    w = 64
    h = 48

    def __init__(self, *screens: tuple[str, Image.Image, str]) -> None:
        self.config: dict[str, Any] = {}
        self._screens = list(screens)

    def get_state(self) -> dict[str, Any]:
        activity, image, package = self._screens.pop(0)
        return {"screenshot": _png(image), "activity": activity, "package": package}

    def exe_action(self, action: dict[str, Any]) -> None:
        msg = f"No device action expected: {action}"
        raise AssertionError(msg)


def _black() -> Image.Image:
    return Image.new("RGB", (64, 48), "black")


def _secure(activity: str, package: str = PACKAGE) -> tuple[str, Image.Image, str]:
    """A screen of ``activity`` that captures as black, as a secure window does."""
    return activity, _black(), package


def _captures(steps_dir: Path, *screens: tuple[str, Image.Image, str]) -> list[Observation]:
    """Observations of ``screens`` as a StepRecorder on ``steps_dir`` captures them."""
    recorder = StepRecorder(
        _ActivityDevice(*screens), steps_dir, run_kind="explore", source="main_loop"
    )
    return [Observation.from_state(recorder.get_state()) for _ in screens]


def test_black_screens_of_two_activities_get_a_node_each_with_no_model_call_but_the_image_embedding(
    vlm: FakeVlm, tmp_path: Path
) -> None:
    """Every black capture has the same pixels, and the description model calls each
    one a blank black screen."""
    vlm.descriptions.extend([("Blank black screen", {}, [])] * 2)
    steps_dir = tmp_path / "steps"
    pattern, pin = _captures(steps_dir, _secure(SET_PATTERN), _secure(SET_PIN))
    gm = make_manager(steps_dir)

    nodes = [gm.identify_state(pattern), gm.identify_state(pin)]

    assert pattern.pixel_hash == pin.pixel_hash
    assert nodes == [PATTERN_NODE, "s1_secure_screen:_chooselockpassword"]
    assert vlm.kinds() == ["image", "image"]
    assert gm.image_vector(pattern.observation_id) == ANY_IMAGE
    assert gm.image_vector(pin.observation_id) == ANY_IMAGE


def test_black_screens_of_one_activity_share_its_node(vlm: FakeVlm, tmp_path: Path) -> None:
    steps_dir = tmp_path / "steps"
    first, second = _captures(steps_dir, _secure(SET_PATTERN), _secure(SET_PATTERN))
    gm = make_manager(steps_dir)

    assert gm.identify_state(first) == gm.identify_state(second) == PATTERN_NODE

    assert vlm.kinds() == ["image", "image"]
    node = gm.graph.nodes[PATTERN_NODE]
    assert node["observations"] == [first.observation_id, second.observation_id]
    assert node["representative_observation"] == first.observation_id
    assert node["visit_count"] == 2
    assert node["activities"] == [SET_PATTERN]


@pytest.mark.usefixtures("vlm")
def test_a_black_screen_is_described_from_its_activity_and_recorded_as_matched_by_it(
    tmp_path: Path,
) -> None:
    steps_dir = tmp_path / "steps"
    (pattern,) = _captures(steps_dir, _secure(SET_PATTERN))
    gm = make_manager(steps_dir)

    gm.identify_state(pattern, expected_node="s9_elsewhere")

    node = gm.graph.nodes[PATTERN_NODE]
    assert node["package"] == PACKAGE
    assert node["page_description"] == "secure screen: ChooseLockPattern"
    assert node["interactable_elements"] == []
    assert node["state_schema"] == {}
    assert gm.text_vector(pattern.observation_id) is None
    record = json.loads((steps_dir / f"{pattern.observation_id}.json").read_text("utf-8"))
    assert record["black"] is True
    assert "describe" not in record
    assert record["identify"] == {
        "stage": "black",
        "expected_node": "s9_elsewhere",
        "exact_match": None,
        "candidates": [],
        "verifications": [],
        "refinement": None,
        "node": PATTERN_NODE,
    }


@pytest.mark.usefixtures("vlm")
def test_the_log_names_the_activity_of_each_black_screen(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    steps_dir = tmp_path / "steps"
    first, second = _captures(steps_dir, _secure(SET_PATTERN), _secure(SET_PATTERN))
    gm = make_manager(steps_dir)

    with caplog.at_level(logging.INFO, logger=gm_module.logger.name):
        gm.identify_state(first)
        gm.identify_state(second)

    assert [m for m in caplog.messages if m.startswith("Black screenshot")] == [
        (
            f"Black screenshot of {SET_PATTERN} starts {PATTERN_NODE}: no node has a black "
            "member of that activity"
        ),
        (
            f"Black screenshot of {SET_PATTERN} joins {PATTERN_NODE}, which has a black member "
            "of that activity"
        ),
    ]


def test_a_black_screen_never_joins_a_node_of_its_activity_without_a_black_member(
    vlm: FakeVlm, tmp_path: Path
) -> None:
    vlm.descriptions.append(("Choose a pattern", {}, []))
    steps_dir = tmp_path / "steps"
    shown, secure = _captures(
        steps_dir, (SET_PATTERN, _screen_with_clock(1), PACKAGE), _secure(SET_PATTERN)
    )
    gm = make_manager(steps_dir)
    shown_node = gm.identify_state(shown)

    assert gm.identify_state(secure) != shown_node

    assert vlm.kinds() == ["image", "describe", "embed", "image"]


def test_a_black_screen_of_another_package_starts_its_own_node(
    vlm: FakeVlm, tmp_path: Path
) -> None:
    """A library activity runs in the package of each app that declares it."""
    steps_dir = tmp_path / "steps"
    in_app, other_app = _captures(
        steps_dir, _secure(GOOGLE_SIGN_IN), _secure(GOOGLE_SIGN_IN, OTHER_PACKAGE)
    )
    gm = make_manager(steps_dir)

    assert gm.identify_state(in_app) != gm.identify_state(other_app)

    assert vlm.kinds() == ["image", "image"]


@pytest.mark.parametrize(("level", "calls"), [(8, ["image"]), (9, ["image", "describe", "embed"])])
def test_a_screen_is_black_while_no_channel_of_a_pixel_exceeds_the_black_level(
    level: int, calls: list[str], vlm: FakeVlm, tmp_path: Path
) -> None:
    vlm.descriptions.append(("Dim screen", {}, []))
    dim = _black()
    dim.putpixel((40, 30), (0, level, 0))
    steps_dir = tmp_path / "steps"
    (observation,) = _captures(steps_dir, (SET_PATTERN, dim, PACKAGE))

    make_manager(steps_dir).identify_state(observation)

    assert vlm.kinds() == calls


def test_a_screen_black_below_the_status_bar_alone_is_described_not_copied_from_a_black_member(
    vlm: FakeVlm, tmp_path: Path
) -> None:
    """Its status bar has pixels above the black level, so it is not black, and its
    pixels below the bar are those of a black member, whose record holds no describe
    output to copy. The black member has no description vector either, so only the
    image source proposes its node."""
    vlm.descriptions.append(("Dark screen", {}, []))
    vlm.verdicts.append({"same": False, "reason": "a clock on a dark screen"})
    clocked = _black()
    ImageDraw.Draw(clocked).rectangle((2, 2, 8, 6), fill=(255, 255, 255))
    recorder = StepRecorder(
        _StatusBarDevice(_black(), clocked),
        tmp_path / "steps",
        run_kind="explore",
        source="main_loop",
    )
    gm = make_manager(recorder.steps)
    black_screen = Observation.from_state(recorder.get_state())
    black_node = gm.identify_state(black_screen)
    observation = Observation.from_state(recorder.get_state())

    assert gm.identify_state(observation) != black_node

    assert not observation.black
    assert observation.pixel_hash == black_screen.pixel_hash
    assert vlm.kinds() == ["image", "image", "describe", "embed", "verify"]
    record = json.loads((recorder.steps / f"{observation.observation_id}.json").read_text("utf-8"))
    assert record["identify"]["stage"] == "new"
    assert [(c["node"], sorted(c["sources"])) for c in record["identify"]["candidates"]] == [
        (black_node, ["image"])
    ]


def test_a_reloaded_graph_joins_a_black_screen_to_the_node_of_its_activity(
    vlm: FakeVlm, tmp_path: Path
) -> None:
    steps_dir = tmp_path / "steps"
    pattern, again, pin = _captures(
        steps_dir, _secure(SET_PATTERN), _secure(SET_PATTERN), _secure(SET_PIN)
    )
    gm = make_manager(steps_dir)
    node_id = gm.identify_state(pattern)
    gm.save_graph(tmp_path / "graph.json")
    loaded = make_manager()
    loaded.load_graph(tmp_path / "graph.json")
    vlm.calls.clear()

    assert loaded.identify_state(again) == node_id
    assert loaded.identify_state(pin) != node_id

    assert vlm.kinds() == ["image", "image"]


def test_a_graph_loads_with_black_members_that_have_no_describe_output_or_description_vector(
    vlm: FakeVlm, tmp_path: Path
) -> None:
    """Loading requires a describe output and a description vector of every member of an
    in-app node but a black one, whose description comes from its activity."""
    vlm.descriptions.append(("Home screen", {}, []))
    steps_dir = tmp_path / "steps"
    home, pattern = _captures(
        steps_dir, (HOME, _screen_with_clock(1), PACKAGE), _secure(SET_PATTERN)
    )
    gm = make_manager(steps_dir)
    gm.identify_state(home)
    pattern_node = gm.identify_state(pattern)
    gm.save_graph(tmp_path / "graph.json")
    loaded = make_manager()

    loaded.load_graph(tmp_path / "graph.json")

    assert loaded.text_vector(home.observation_id) == EMBEDDINGS["Home screen"]
    assert loaded.text_vector(pattern.observation_id) is None
    assert loaded.graph.nodes[pattern_node]["observations"] == [pattern.observation_id]


def _merge_audit_proposes(monkeypatch: pytest.MonkeyPatch, node_a: str, node_b: str) -> None:
    proposal = FakeAudit(result={"issues": [{"node_a": node_a, "node_b": node_b}]})
    monkeypatch.setattr(gm_module, "audit_merge_nodes", proposal)


def test_the_merge_audit_keeps_black_nodes_of_different_activities_apart_unverified(
    vlm: FakeVlm, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The verifier would compare two identical black images, and call them the same."""
    vlm.verdicts.append({"same": True, "reason": "both screens are black"})
    steps_dir = tmp_path / "steps"
    pattern, pin = _captures(steps_dir, _secure(SET_PATTERN), _secure(SET_PIN))
    gm = make_manager(steps_dir)
    pattern_node, pin_node = gm.identify_state(pattern), gm.identify_state(pin)
    _merge_audit_proposes(monkeypatch, pattern_node, pin_node)

    with caplog.at_level(logging.WARNING, logger=gm_module.logger.name):
        result = gm.run_node_merge_audit()

    assert result["merged_count"] == 0
    assert result["results"][0]["status"] == "skipped"
    assert result["results"][0]["reason"] == "black screens of different activities"
    assert {pattern_node, pin_node} <= set(gm.graph)
    assert "verify" not in vlm.kinds()
    assert (
        f"Skipping merge {pattern_node} + {pin_node} — black screens of different activities"
    ) in caplog.messages


@pytest.mark.parametrize("pin_node_holds", ["a shown screen", "a black screen of both activities"])
def test_the_merge_audit_verifies_a_black_node_and_one_not_all_black_or_sharing_its_activity(
    pin_node_holds: str, vlm: FakeVlm, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A screen with pixels to compare, or black members of one activity left in two nodes
    by a merge and a split, leave the verifier something to decide."""
    vlm.descriptions.append(("Choose a PIN", {}, []))
    vlm.verdicts.extend([{"same": False, "reason": "different screens"}] * 2)
    shown = pin_node_holds == "a shown screen"
    pin_screen = (SET_PIN, _screen_with_clock(1), PACKAGE) if shown else _secure(SET_PIN)
    steps_dir = tmp_path / "steps"
    pattern, pin = _captures(steps_dir, _secure(SET_PATTERN), pin_screen)
    gm = make_manager(steps_dir)
    pattern_node, pin_node = gm.identify_state(pattern), gm.identify_state(pin)
    if not shown:
        gm.graph.nodes[pin_node]["observations"].append(pattern.observation_id)
    _merge_audit_proposes(monkeypatch, pattern_node, pin_node)
    vlm.calls.clear()

    result = gm.run_node_merge_audit()

    assert result["results"][0]["status"] == "kept_separate"
    assert vlm.kinds() == ["verify"]


def test_embed_missing_members_embeds_each_in_app_member_once_and_saves_the_vectors(
    vlm: FakeVlm, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    steps_dir = tmp_path / "steps"
    gm = make_manager(steps_dir)
    add_screen(gm, "s0_home", "Home screen", **members(steps_dir, b"home", b"home-2"))
    gm.record_external_app_visit(Screens(steps_dir)(OTHER_APP, SHOT_B, OTHER_PACKAGE))
    vlm.images.update({b64(b"home"): NEAR, b64(b"home-2"): FAR})

    with caplog.at_level(logging.INFO, logger=gm_module.logger.name):
        assert gm.embed_missing_members() == 2
        assert gm.embed_missing_members() == 0

    assert [call["screenshot"] for _, call in vlm.calls] == [b64(b"home"), b64(b"home-2")]
    stored = json.loads((steps_dir / "image_emb.json").read_text(encoding="utf-8"))
    assert stored["model"] == IMAGE_EMBEDDING.model
    home, home_2 = gm.graph.nodes["s0_home"]["observations"]
    assert stored["embeddings"] == {home: NEAR, home_2: FAR}
    assert "Embedding the images of 2 member observation(s)" in caplog.text
    assert "Embedded 2 member image(s) for $0.000000" in caplog.text


def test_embed_missing_members_keeps_the_vectors_it_paid_for_when_one_fails(
    vlm: FakeVlm, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    steps_dir = tmp_path / "steps"
    gm = make_manager(steps_dir)
    add_screen(gm, "s0_home", "Home screen", **members(steps_dir, b"home", b"home-2"))
    answers = iter([NEAR])

    def embed_or_fail(*_args: object, **_kwargs: object) -> list[float]:
        return next(answers)

    monkeypatch.setattr(gm_module, "get_image_embedding", embed_or_fail)
    del vlm

    with pytest.raises(StopIteration):
        gm.embed_missing_members()

    stored = json.loads((steps_dir / "image_emb.json").read_text(encoding="utf-8"))
    assert list(stored["embeddings"].values()) == [NEAR]


def test_embedding_missing_members_needs_image_embedding_settings(tmp_path: Path) -> None:
    gm = GraphManager(client=FAKE_CLIENT, page_detail=PAGE_DETAIL, embedding=EMBEDDING)
    gm.steps_dir = tmp_path
    with pytest.raises(RuntimeError, match="image_embedding settings"):
        gm.embed_missing_members()


def test_a_manager_without_image_embedding_settings_neither_loads_nor_saves_vectors(
    tmp_path: Path, write_image_vectors: Callable[..., None]
) -> None:
    path = tmp_path / "graph.json"
    member = members(tmp_path / "steps")
    path.write_text(
        json.dumps({"nodes": [{"id": "s0_home", "replay_misses": 0, **member}], "edges": []}),
        "utf-8",
    )
    write_image_vectors(tmp_path / "steps", member["observations"])
    gm = GraphManager()

    gm.load_graph(path)
    gm.save_graph(path)

    assert gm.image_vector(member["representative_observation"]) is None
    assert json.loads((tmp_path / "steps" / "image_emb.json").read_text("utf-8"))["embeddings"] == {
        member["representative_observation"]: [1.0, 0.0]
    }


@pytest.mark.parametrize(
    ("vectors", "unusable"),
    [
        pytest.param({}, [0, 1, 2], id="no member has one"),
        pytest.param({0: NEAR, 2: FAR}, [1], id="one member has none"),
        pytest.param({0: NEAR, 1: [1.0, 0.0], 2: FAR}, [1], id="another dimension"),
    ],
)
def test_closest_member_raises_naming_the_node_and_every_member_without_a_usable_image_vector(
    vectors: dict[int, list[float]], unusable: list[int], tmp_path: Path
) -> None:
    steps_dir = tmp_path / "steps"
    member = members(steps_dir, b"home", b"home-2", b"home-3")
    observations = member["observations"]
    save_image_embeddings(
        steps_dir,
        {observations[index]: vector for index, vector in vectors.items()},
        model=IMAGE_EMBEDDING.model,
    )
    path = tmp_path / "graph.json"
    path.write_text(
        json.dumps(
            {
                "nodes": [{"id": "s0_home", "package": PACKAGE, "replay_misses": 0, **member}],
                "edges": [],
            }
        ),
        "utf-8",
    )
    write_text_vectors(path, member)
    gm = make_manager()
    gm.load_graph(path)

    with pytest.raises(ValueError, match="'s0_home'") as caught:
        gm.closest_member("s0_home", NEAR)

    named = [observation for observation in observations if observation in str(caught.value)]
    assert named == [observations[index] for index in unusable]


def test_closest_member_raises_for_a_node_without_members() -> None:
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen")

    with pytest.raises(ValueError, match="'s0_home' has no members"):
        gm.closest_member("s0_home", NEAR)


def test_the_text_source_ranks_a_node_by_its_member_closest_to_the_screen(
    vlm: FakeVlm, screen: Screens
) -> None:
    """The settings node's first member was described as a settings page and its second
    as the search results the screen shows; the banner's only member is described less
    like them."""
    vlm.descriptions.extend(
        [("Settings page", {}, []), ("Search results", {}, []), ("Promo banner", {}, [])]
    )
    vlm.verdicts.extend(
        [{"same": True, "reason": "a scrolled list"}, {"same": False, "reason": "a banner"}]
    )
    gm = make_manager(screen.steps_dir)
    settings = gm.identify_state(screen(HOME, SHOT_A, PACKAGE))
    assert gm.identify_state(screen(HOME, SHOT_B, PACKAGE)) == settings
    banner = gm.identify_state(screen(HOME, b64(b"banner"), PACKAGE))
    (text,) = (source for source in SHORT_LIST_SOURCES if source.name == "text")
    query = ScreenQuery(image_vector=ANY_IMAGE, description_embedding=EMBEDDINGS["Search results"])

    ranked = sorted(text.scores(gm, query, [banner, settings]), key=lambda kv: kv[1], reverse=True)

    assert ranked == [(settings, pytest.approx(1.0)), (banner, pytest.approx(3**-0.5))]


def test_the_short_list_orders_ties_in_rank_by_image_score(vlm: FakeVlm, tmp_path: Path) -> None:
    steps_dir = tmp_path / "steps"
    gm = make_manager(steps_dir)
    add_screen(gm, "s0_text", "Search results", **members(steps_dir, b"far"))
    add_screen(gm, "s1_image", "Settings page", **members(steps_dir, b"near"))
    vlm.images.update({b64(b"far"): FAR, b64(b"near"): NEAR})
    gm.embed_missing_members()

    query = ScreenQuery(image_vector=NEAR, description_embedding=EMBEDDINGS["Search results"])
    ranked = with_text_vectors(gm, steps_dir).short_list(query, ["s0_text", "s1_image"])

    assert [(c.node_id, c.best_rank, c.sources) for c in ranked] == [
        ("s1_image", 0, {"image": pytest.approx(1.0), "text": pytest.approx(0.0)}),
        ("s0_text", 0, {"image": pytest.approx(0.0), "text": pytest.approx(1.0)}),
    ]


def test_identify_state_sanitizes_and_truncates_the_node_id(vlm: FakeVlm, screen: Screens) -> None:
    description = "Home / dashboard screen showing every available promotion"
    vlm.descriptions.append((description, {}, []))
    gm = make_manager(screen.steps_dir)

    node_id = gm.identify_state(screen(HOME, SHOT_A, PACKAGE))

    assert node_id == "s0_home___dashboard_screen_showing_every_av"
    assert len(node_id.split("_", 1)[1]) == 40


def test_rename_node_returns_a_missing_id_unchanged() -> None:
    gm = make_manager()
    assert gm.rename_node("s9_missing", "Anything") == "s9_missing"


def test_a_rename_keeps_the_members_text_vectors_without_a_request(
    vlm: FakeVlm, screen: Screens
) -> None:
    vlm.descriptions.append(("Home screen", {}, []))
    gm = make_manager(screen.steps_dir)
    observation = screen(HOME, SHOT_A, PACKAGE)
    node_id = gm.identify_state(observation)
    vlm.calls.clear()

    gm.rename_node(node_id, "Settings page")

    assert vlm.calls == []
    assert gm.text_vector(observation.observation_id) == EMBEDDINGS["Home screen"]


def test_rename_node_keeps_the_numeric_prefix_and_relabels_edges() -> None:
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen")
    add_screen(gm, "s1_settings", "Settings page")
    gm.add_edge("s0_home", "s1_settings", {"action": "tap"})

    new_id = gm.rename_node("s0_home", "Home feed")

    assert new_id == "s0_home_feed"
    assert gm.graph.has_edge("s0_home_feed", "s1_settings")
    assert gm.graph.nodes[new_id]["page_description"] == "Home feed"


def test_rename_node_suffixes_the_id_on_collision() -> None:
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen")
    add_screen(gm, "s0_settings_page", "Settings page")
    add_screen(gm, "s0_settings_page_2", "Settings page")

    assert gm.rename_node("s0_home", "Settings page") == "s0_settings_page_3"


def test_rename_node_without_an_underscore_uses_the_whole_id_as_prefix() -> None:
    gm = make_manager()
    add_screen(gm, "start", "Home screen")

    assert gm.rename_node("start", "Home feed") == "start_home_feed"


def test_rename_node_to_the_same_id_records_no_alias() -> None:
    gm = make_manager()
    add_screen(
        gm,
        "s0_home_screen",
        "Home screen",
        observations=["000000"],
        representative_observation="000000",
    )

    assert gm.rename_node("s0_home_screen", "Home screen") == "s0_home_screen"

    assert gm.resolve_node_id("s0_home_screen") == "s0_home_screen"
    assert gm.graph.nodes["s0_home_screen"]["observations"] == ["000000"]
    assert gm.graph.nodes["s0_home_screen"]["representative_observation"] == "000000"


def test_rename_node_keeps_the_members_and_representative_under_the_new_id() -> None:
    gm = make_manager()
    add_screen(
        gm,
        "s0_home_screen",
        "Home screen",
        observations=["000000", "000004"],
        representative_observation="000000",
    )

    new_id = gm.rename_node("s0_home_screen", "Home feed")

    assert new_id == "s0_home_feed"
    assert gm.graph.nodes[new_id]["observations"] == ["000000", "000004"]
    assert gm.graph.nodes[new_id]["representative_observation"] == "000000"


def test_an_exact_match_joins_the_node_under_its_new_id(vlm: FakeVlm, screen: Screens) -> None:
    vlm.descriptions.append(("Home screen", {}, []))
    gm = make_manager(screen.steps_dir)
    gm.identify_state(screen(HOME, SHOT_A, PACKAGE))

    new_id = gm.rename_node("s0_home_screen", "Home feed")

    assert gm.identify_state(screen(HOME, SHOT_A, PACKAGE)) == new_id
    assert vlm.kinds() == ["image", "describe", "embed", "image"]


@pytest.mark.usefixtures("vlm")
def test_resolve_node_id_follows_a_chain_of_renames() -> None:
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen")
    feed = gm.rename_node("s0_home", "Home feed")

    settings = gm.rename_node(feed, "Settings page")

    assert gm.resolve_node_id("s0_home") == settings
    assert gm.resolve_node_id(settings) == settings


@pytest.mark.usefixtures("vlm")
def test_resolve_node_id_after_renaming_back_to_an_earlier_id() -> None:
    gm = make_manager()
    add_screen(gm, "s0_home_screen", "Home screen")
    feed = gm.rename_node("s0_home_screen", "Home feed")

    assert gm.rename_node(feed, "Home screen") == "s0_home_screen"

    assert gm.resolve_node_id("s0_home_screen") == "s0_home_screen"
    assert gm.resolve_node_id(feed) == "s0_home_screen"


@pytest.mark.usefixtures("vlm")
def test_resolve_node_id_follows_renames_and_merges_together() -> None:
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen")
    add_screen(gm, "s1_home", "Home screen")
    renamed = gm.rename_node("s1_home", "Home feed")
    assert gm.merge_nodes("s0_home", renamed) is True

    kept = gm.rename_node("s0_home", "Settings page")

    assert {gm.resolve_node_id(node) for node in ("s1_home", renamed, "s0_home")} == {kept}


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


def _add_home_with_two_unexplored_elements(gm: GraphManager) -> None:
    add_screen(
        gm,
        "s0_home",
        "Home screen",
        interactable_elements=[
            {"description": "search box", "explored": True},
            {"description": "Turn on dark mode toggle", "explored": False},
            {"description": "Compose FAB", "explored": False},
        ],
    )


def test_mark_element_explored_counts_from_the_first_unexplored_element() -> None:
    gm = make_manager()
    _add_home_with_two_unexplored_elements(gm)

    gm.mark_element_explored("s0_home", 2)

    assert gm.get_unexplored_elements("s0_home") == [
        {"description": "Turn on dark mode toggle", "explored": False}
    ]


@pytest.mark.parametrize("element", [0, 3, -1])
def test_mark_element_explored_outside_the_unexplored_list_marks_nothing_and_warns(
    element: int, caplog: pytest.LogCaptureFixture
) -> None:
    gm = make_manager()
    _add_home_with_two_unexplored_elements(gm)

    with caplog.at_level("WARNING"):
        gm.mark_element_explored("s0_home", element)

    assert len(gm.get_unexplored_elements("s0_home")) == 2
    assert f"Element {element} is not among the 2 unexplored element(s) on s0_home" in caplog.text


def test_get_all_edges_from_node_reports_one_dict_per_outgoing_edge() -> None:
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen")
    add_screen(gm, "s1_settings", "Settings page")
    gm.add_edge(
        "s0_home",
        "s1_settings",
        {"action": "tap"},
        details=EdgeDetails(instruction="open settings", target_observation="the settings page"),
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
    assert edges[0]["schema_deltas"] == []


def test_get_all_edges_from_node_skips_the_placeholders_of_unrecorded_values() -> None:
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen")
    gm.add_edge("s0_home", "s0_home", {"action": "tap"})
    gm.add_edge(
        "s0_home",
        "s0_home",
        {"action": "swipe"},
        details=EdgeDetails(
            instruction="scroll the feed",
            target_observation="more posts",
            schema_delta={"first_post": {"before": "a", "after": "b"}},
        ),
    )

    edges = gm.get_all_edges_from_node("s0_home")

    assert edges[0]["actions"] == [{"action": "tap"}, {"action": "swipe"}]
    assert edges[0]["instructions"] == ["scroll the feed"]
    assert edges[0]["target_observations"] == ["more posts"]
    assert edges[0]["schema_deltas"] == [{"first_post": {"before": "a", "after": "b"}}]


def test_get_all_edges_from_node_includes_a_non_empty_schema_delta() -> None:
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen")
    gm.add_edge(
        "s0_home",
        "s0_home",
        {"action": "tap"},
        details=EdgeDetails(schema_delta={"sort": {"after": "price"}}),
    )

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


def test_add_edge_creates_an_edge_with_defaults() -> None:
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen")
    add_screen(gm, "s1_settings", "Settings page")

    gm.add_edge("s0_home", "s1_settings", {"action": "tap"})

    data = gm.graph["s0_home"]["s1_settings"]
    assert data["actions"] == [{"action": "tap"}]
    assert data["num_steps"] == [1]
    assert data["visit_count"] == 1
    assert data["instructions"] == [""]
    assert data["target_observations"] == [""]
    assert data["schema_deltas"] == [None]


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

    gm.add_edge("s0_home", "s1_settings", {"action": "tap"}, details=EdgeDetails(num_steps=7))

    assert gm.graph["s0_home"]["s1_settings"]["num_steps"] == [7]


def test_add_edge_appends_a_different_action_with_its_metadata() -> None:
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen")
    add_screen(gm, "s1_settings", "Settings page")
    gm.add_edge(
        "s0_home",
        "s1_settings",
        {"action": "tap"},
        details=EdgeDetails(instruction="open settings"),
    )

    gm.add_edge(
        "s0_home",
        "s1_settings",
        {"action": "swipe"},
        details=EdgeDetails(
            instruction="swipe to settings",
            target_observation="the settings page",
            schema_delta={"tab": {"after": "settings"}},
        ),
    )

    data = gm.graph["s0_home"]["s1_settings"]
    assert data["actions"] == [{"action": "tap"}, {"action": "swipe"}]
    assert data["num_steps"] == [1, 1]
    assert data["instructions"] == ["open settings", "swipe to settings"]
    assert data["target_observations"] == ["", "the settings page"]
    assert data["schema_deltas"] == [None, {"tab": {"after": "settings"}}]
    assert data["visit_count"] == 1


def test_add_edge_records_a_self_loop_with_a_schema_delta() -> None:
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen")

    gm.add_edge(
        "s0_home",
        "s0_home",
        {"action": "tap"},
        details=EdgeDetails(
            instruction="sort by price",
            target_observation="sorted by price",
            schema_delta={"sort": {"after": "price"}},
        ),
    )

    data = gm.graph["s0_home"]["s0_home"]
    assert data["instructions"] == ["sort by price"]
    assert data["target_observations"] == ["sorted by price"]
    assert data["schema_deltas"] == [{"sort": {"after": "price"}}]


_SCROLL = {"action": "swipe", "x1": 5, "y1": 90, "x2": 5, "y2": 10}
_TAP = {"action": "tap", "x": 5, "y": 50}


def test_add_edge_records_each_action_s_setup_and_none_without_one() -> None:
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen")
    add_screen(gm, "s1_settings", "Settings page")

    gm.add_edge("s0_home", "s1_settings", {"action": "tap"})
    gm.add_edge("s0_home", "s1_settings", {"action": "row"}, details=EdgeDetails(setup=[]))
    gm.add_edge(
        "s0_home", "s1_settings", {"action": "far row"}, details=EdgeDetails(setup=[_SCROLL])
    )

    assert gm.graph["s0_home"]["s1_settings"]["setups"] == [None, [], [_SCROLL]]


@pytest.mark.parametrize("extra", [0, 1])
def test_add_edge_drops_a_setup_longer_than_the_cap_and_logs_it(
    extra: int, caplog: pytest.LogCaptureFixture
) -> None:
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen")
    add_screen(gm, "s1_settings", "Settings page")
    setup = [_SCROLL] * (MAX_SETUP_ACTIONS + extra)

    with caplog.at_level(logging.INFO, logger=gm_module.__name__):
        gm.add_edge("s0_home", "s1_settings", _TAP, details=EdgeDetails(setup=setup))

    assert gm.graph["s0_home"]["s1_settings"]["setups"] == [None if extra else setup]
    dropped = "Dropped the setup of an action on s0_home -> s1_settings: 9 actions, more than 8"
    messages = [record.getMessage() for record in caplog.records]
    assert [m for m in messages if m.startswith("Dropped")] == ([dropped] if extra else [])


@pytest.mark.parametrize(
    ("first", "again", "kept"),
    [
        pytest.param(None, [_SCROLL], [_SCROLL], id="known replaces unknown"),
        pytest.param([_SCROLL, _SCROLL], [_SCROLL], [_SCROLL], id="shorter replaces longer"),
        pytest.param([], [_SCROLL], [], id="longer keeps shorter"),
        pytest.param([_SCROLL], None, [_SCROLL], id="unknown keeps known"),
    ],
)
def test_a_repeated_action_keeps_its_shorter_known_setup(
    first: list[dict[str, Any]] | None,
    again: list[dict[str, Any]] | None,
    kept: list[dict[str, Any]],
) -> None:
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen")
    add_screen(gm, "s1_settings", "Settings page")
    gm.add_edge("s0_home", "s1_settings", _TAP, details=EdgeDetails(setup=first))

    gm.add_edge("s0_home", "s1_settings", _TAP, details=EdgeDetails(setup=again))

    data = gm.graph["s0_home"]["s1_settings"]
    assert (data["actions"], data["setups"], data["visit_count"]) == ([_TAP], [kept], 2)


def _identity(node: str) -> str:
    return node


_TOP = {"first_visible_item": "Google"}
_SCROLLED = {"first_visible_item": "Default apps"}
_BACK = {"action": "back"}


def test_a_setup_trail_starts_an_arrival_s_setup_empty_and_extends_it_with_self_loops() -> None:
    trail = SetupTrail(_identity)
    trail.begin("s0_home", _TOP)
    assert trail.setup("s0_home") == []

    trail.step([_SCROLL], "s0_home", _SCROLLED)
    assert trail.setup("s0_home") == [_SCROLL]

    trail.step([_TAP], "s1_list", {})
    assert trail.setup("s1_list") == []


def test_a_round_trip_joins_the_setup_of_the_node_it_returns_to() -> None:
    """A round trip through a dialog that set a form's field joins the form's setup, as
    the menu then opened on the form does. The nodes the round trip passed through are
    left behind: arriving at one of them again starts its setup afresh."""
    open_security, choose_enterprise, open_menu = (
        {"action": "tap", "x": 1},
        {"action": "tap", "x": 2},
        {"action": "tap", "x": 3},
    )
    enterprise = {"security": "WPA/WPA2-Enterprise", "open_menu": None}
    trail = SetupTrail(_identity)
    trail.begin("s0_home", {})
    trail.step([_TAP], "s1_form", {"security": "None", "open_menu": None})
    trail.step([open_security], "s2_security_dialog", {"security": "None"})
    trail.step([choose_enterprise], "s1_form", enterprise)
    trail.step([open_menu], "s1_form", {**enterprise, "open_menu": "CA certificate dropdown"})
    assert trail.setup("s1_form") == [open_security, choose_enterprise, open_menu]

    trail.step([_BACK], "s1_form", enterprise)
    trail.step([open_security], "s2_security_dialog", {"security": "WPA/WPA2-Enterprise"})
    assert trail.setup("s2_security_dialog") == []


def test_a_setup_leaves_out_a_field_typed_and_cleared_and_keeps_the_scroll_after_it() -> None:
    """States compare key by key, and a key one of them lacks counts as null, so clearing
    the field brings the list back to the state it was in on arrival."""
    type_query, clear_query = {"action": "type", "text": "wifi"}, {"action": "tap", "x": 9}
    trail = SetupTrail(_identity)
    trail.begin("s1_list", _TOP)
    trail.step([type_query], "s1_list", {**_TOP, "search_query": "wifi"})
    trail.step([clear_query], "s1_list", {**_TOP, "search_query": None})
    assert trail.setup("s1_list") == []

    trail.step([_SCROLL], "s1_list", {**_SCROLLED, "search_query": None})
    assert trail.setup("s1_list") == [_SCROLL]


def test_a_setup_trail_begun_without_an_arrival_knows_no_setup_until_it_arrives_elsewhere() -> None:
    trail = SetupTrail(_identity)
    trail.begin("s0_home", None)
    trail.step([_SCROLL], "s0_home", _SCROLLED)
    assert trail.setup("s0_home") is None

    trail.step([_TAP], "s1_list", {})
    assert trail.setup("s1_list") == []

    trail.step([_BACK], "s0_home", _SCROLLED)
    assert trail.setup("s0_home") is None
    assert trail.setup("s9_never_reached") is None


def test_a_setup_trail_matches_a_node_merged_away_while_it_holds_it() -> None:
    gm = make_manager()
    for node_id in ("s0_home", "s1_list", "s2_detail", "s5_list"):
        add_screen(gm, node_id, f"screen {node_id}")
    trail = SetupTrail(gm.resolve_node_id)
    trail.begin("s0_home", {})
    trail.step([_TAP], "s1_list", _TOP)
    trail.step([_SCROLL], "s1_list", _SCROLLED)

    assert gm.merge_nodes("s5_list", "s1_list") is True
    assert trail.setup("s5_list") == [_SCROLL]

    open_row = {"action": "tap", "x": 7}
    trail.step([open_row], "s2_detail", {})
    trail.step([_BACK], "s5_list", _SCROLLED)
    trail.step([_TAP], "s5_list", {**_SCROLLED, "open_menu": "overflow menu"})
    assert trail.setup("s5_list") == [_SCROLL, _TAP]


@pytest.mark.parametrize(
    "key", ["num_steps", "instructions", "target_observations", "schema_deltas", "setups"]
)
@pytest.mark.parametrize("entries", [None, []], ids=["absent", "short"])
def test_load_graph_rejects_an_edge_without_one_entry_per_action_in_a_list(
    key: str, entries: list[object] | None, tmp_path: Path
) -> None:
    steps_dir = tmp_path / "steps"
    complete = {
        "actions": [[_TAP]],
        "num_steps": [1],
        "instructions": ["open settings"],
        "target_observations": ["Settings"],
        "schema_deltas": [None],
        "setups": [[]],
    }
    lacking = {"source": "s0_home", "target": "s1_settings", **complete}
    if entries is None:
        del lacking[key]
    else:
        lacking[key] = entries
    path = tmp_path / "graph.json"
    path.write_text(
        json.dumps(
            {
                "nodes": [
                    {"id": "s0_home", **members(steps_dir)},
                    {"id": "s1_settings", **members(steps_dir)},
                ],
                "edges": [lacking, {"source": "s1_settings", "target": "s0_home", **complete}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="one entry per action") as excinfo:
        make_manager().load_graph(path)

    message = str(excinfo.value)
    assert "['s0_home -> s1_settings']" in message
    assert str(path) in message
    assert "Explore the app again into a new graph directory" in message


@pytest.mark.parametrize("actions", [None, []], ids=["absent", "empty"])
def test_load_graph_rejects_an_edge_without_actions(
    actions: list[object] | None, tmp_path: Path
) -> None:
    steps_dir = tmp_path / "steps"
    lists: dict[str, list[object]] = {key: [] for key in PER_ACTION_LISTS}
    edge = {"source": "s0_home", "target": "s1_settings", **lists}
    if actions is not None:
        edge["actions"] = actions
    path = tmp_path / "graph.json"
    path.write_text(
        json.dumps(
            {
                "nodes": [
                    {"id": "s0_home", **members(steps_dir)},
                    {"id": "s1_settings", **members(steps_dir)},
                ],
                "edges": [edge],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"edge\(s\) without actions") as excinfo:
        make_manager().load_graph(path)

    message = str(excinfo.value)
    assert "['s0_home -> s1_settings']" in message
    assert "one entry per action" not in message
    assert str(path) in message
    assert "Explore the app again into a new graph directory" in message


def test_add_edge_ignores_an_unknown_endpoint(caplog: pytest.LogCaptureFixture) -> None:
    """A typo'd endpoint adds no node."""
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen")

    with caplog.at_level(logging.WARNING, logger=gm_module.__name__):
        gm.add_edge("s0_home", "s0_hmoe", {"action_type": "tap", "target": "Settings"})
        gm.add_edge("s9_missing", "s0_home", {"action_type": "tap", "target": "Back"})

    assert list(gm.graph.nodes) == ["s0_home"]
    assert gm.graph.number_of_edges() == 0
    assert caplog.text.count("add_edge: missing node(s)") == 2


@pytest.mark.parametrize("external", [EXTERNAL_WEB_NODE, "ext_com.android.chrome"])
def test_add_edge_refuses_an_edge_out_of_an_external_node(
    external: str, caplog: pytest.LogCaptureFixture
) -> None:
    """Every outside web page shares one node, so an edge out of it joined unrelated
    screens into a path that find_path replayed."""
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen")
    add_screen(gm, external, "Outside page")

    with caplog.at_level(logging.WARNING, logger=gm_module.__name__):
        gm.add_edge(external, "s0_home", {"action": "tap"})

    assert gm.graph.number_of_edges() == 0
    assert f"add_edge: refusing an edge out of external node {external}" in caplog.text


def test_add_edge_ignores_a_self_loop_on_an_unknown_node() -> None:
    """The self-loop branch must not create the node it loops on either."""
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen")

    gm.add_edge("s9_missing", "s9_missing", {"action_type": "tap", "target": "Sort"})

    assert list(gm.graph.nodes) == ["s0_home"]
    assert gm.graph.number_of_edges() == 0


def test_merge_nodes_returns_false_when_a_node_is_missing() -> None:
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen")

    assert gm.merge_nodes("s0_home", "s9_missing") is False
    assert gm.merge_nodes("s9_missing", "s0_home") is False
    assert gm.graph.number_of_nodes() == 1


def test_merge_nodes_is_a_no_op_when_both_ids_name_the_same_node() -> None:
    """A self-merge keeps the node with its visit_count unchanged."""
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen", visit_count=3, state_schema={"tab": ["home"]})
    add_screen(gm, "s1_search", "Search screen")
    gm.add_edge("s1_search", "s0_home", {"action_type": "tap", "target": "Back"})

    assert gm.merge_nodes("s0_home", "s0_home") is False

    node = gm.get_node("s0_home")
    assert node is not None
    assert node["visit_count"] == 3
    assert node["state_schema"] == {"tab": ["home"]}
    assert "s0_home" in gm.graph
    assert gm.graph.has_edge("s1_search", "s0_home")


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


def test_merge_nodes_unites_the_members_in_step_order_and_keeps_the_representative() -> None:
    """Step IDs are zero-padded integers, so a longer ID sorts after every shorter one."""
    gm = make_manager()
    add_screen(
        gm,
        "s0_home",
        "Home screen",
        observations=["000003", "000010", "1000000"],
        representative_observation="000010",
    )
    add_screen(
        gm,
        "s1_home_two",
        "Home screen again",
        observations=["000001", "000010", "000012"],
        representative_observation="000001",
    )

    assert gm.merge_nodes("s0_home", "s1_home_two") is True

    kept = gm.graph.nodes["s0_home"]
    assert kept["observations"] == ["000001", "000003", "000010", "000012", "1000000"]
    assert kept["representative_observation"] == "000010"


def test_record_external_app_visit_adds_each_screen_to_one_node_per_app(
    screen: Screens,
) -> None:
    gm = make_manager(screen.steps_dir)
    first = screen(OTHER_APP, SHOT_A, f" {OTHER_PACKAGE} ")
    second = screen(OTHER_APP, SHOT_B, OTHER_PACKAGE)

    assert gm.record_external_app_visit(first) == f"ext_{OTHER_PACKAGE}"
    assert gm.record_external_app_visit(second) == f"ext_{OTHER_PACKAGE}"

    assert list(gm.graph) == [f"ext_{OTHER_PACKAGE}"]
    assert gm.get_node(f"ext_{OTHER_PACKAGE}") == {
        "activity": OTHER_APP,
        "page_description": f"[external app: {OTHER_PACKAGE}]",
        "state_schema": {},
        "visit_count": 2,
        "replay_misses": 0,
        "is_external": True,
        "observations": [first.observation_id, second.observation_id],
        "representative_observation": first.observation_id,
    }


def test_merge_nodes_adds_only_the_schema_values_the_kept_node_lacks() -> None:
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen", state_schema={"tab": ["home"]})
    add_screen(gm, "s1_home_two", "Home screen again", state_schema={"tab": ["home", "deals"]})

    assert gm.merge_nodes("s0_home", "s1_home_two") is True

    node = gm.get_node("s0_home")
    assert node is not None
    assert node["state_schema"] == {"tab": ["home", "deals"]}


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
    gm.add_edge(
        "s0_keep",
        "s2_other",
        {"action": "keep"},
        details=EdgeDetails(instruction="from keep", num_steps=1),
    )
    gm.add_edge(
        "s1_remove",
        "s2_other",
        {"action": "remove"},
        details=EdgeDetails(instruction="from remove", num_steps=3),
    )
    gm.graph["s1_remove"]["s2_other"]["instruction_templates"] = [{"template": "go to {x}"}]

    assert gm.merge_nodes("s0_keep", "s1_remove") is True

    data = gm.graph["s0_keep"]["s2_other"]
    assert data["actions"] == [{"action": "keep"}, {"action": "remove"}]
    assert data["instructions"] == ["from keep", "from remove"]
    assert data["num_steps"] == [1, 3]
    assert data["visit_count"] == 2
    assert data["instruction_templates"] == [{"template": "go to {x}"}]


def edge_rows(gm: GraphManager, source: str, target: str) -> list[tuple[Any, ...]]:
    """One (action, num_steps, instruction, observation, schema_delta, setup) row per action."""
    data = gm.graph[source][target]
    return list(zip(*(data[key] for key in ("actions", *PER_ACTION_LISTS)), strict=True))


def test_move_edge_action_takes_its_entry_from_every_per_action_list_to_the_other_edge() -> None:
    gm = make_manager()
    for node_id, description in (
        ("s0_home", "Home screen"),
        ("s1_settings", "Settings page"),
        ("s2_search", "Search results"),
    ):
        add_screen(gm, node_id, description)
    search = [{"action": "tap"}, {"action": "type", "text": "milk"}]
    gm.add_edge(
        "s0_home",
        "s1_settings",
        {"action": "tap"},
        details=EdgeDetails(instruction="open settings"),
    )
    gm.add_edge(
        "s0_home",
        "s1_settings",
        search,
        details=EdgeDetails(
            instruction="search for milk",
            target_observation="Settings page",
            schema_delta={"query": {"after": "milk"}},
            setup=[_SCROLL],
        ),
    )
    gm.add_edge("s0_home", "s1_settings", {"action": "swipe"}, details=EdgeDetails(num_steps=3))
    gm.graph["s0_home"]["s1_settings"]["instruction_templates"] = [{"template": "{verb} for x"}]
    gm.add_edge(
        "s0_home",
        "s2_search",
        {"action": "enter"},
        details=EdgeDetails(instruction="submit the search"),
    )

    assert gm.move_edge_action("s0_home", "s1_settings", 1, "s2_search") is True

    assert edge_rows(gm, "s0_home", "s1_settings") == [
        ({"action": "tap"}, 1, "open settings", "", None, None),
        ({"action": "swipe"}, 3, "", "", None, None),
    ]
    assert edge_rows(gm, "s0_home", "s2_search") == [
        ({"action": "enter"}, 1, "submit the search", "", None, None),
        (
            search,
            2,
            "search for milk",
            "Search results",
            {"query": {"after": "milk"}},
            [_SCROLL],
        ),
    ]
    assert gm.graph["s0_home"]["s1_settings"]["instruction_templates"] == []
    assert gm.graph["s0_home"]["s2_search"]["visit_count"] == 2


def test_move_edge_action_removes_the_edge_it_empties_and_creates_the_edge_it_needs() -> None:
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen")
    add_screen(gm, "s1_settings", "Settings page")
    gm.add_edge(
        "s0_home",
        "s1_settings",
        {"action": "tap"},
        details=EdgeDetails(instruction="open settings"),
    )

    assert gm.move_edge_action("s0_home", "s1_settings", 0, "s0_home") is True

    assert not gm.graph.has_edge("s0_home", "s1_settings")
    assert edge_rows(gm, "s0_home", "s0_home") == [
        ({"action": "tap"}, 1, "open settings", "Home screen", None, None)
    ]
    assert gm.graph["s0_home"]["s0_home"]["visit_count"] == 1


def test_move_edge_action_refuses_a_destination_missing_from_the_graph(
    caplog: pytest.LogCaptureFixture,
) -> None:
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen")
    add_screen(gm, "s1_settings", "Settings page")
    gm.add_edge(
        "s0_home",
        "s1_settings",
        {"action": "tap"},
        details=EdgeDetails(instruction="open settings"),
    )

    with caplog.at_level(logging.WARNING, logger=gm_module.__name__):
        assert gm.move_edge_action("s0_home", "s1_settings", 0, "ext_com.other") is False

    assert list(gm.graph.nodes) == ["s0_home", "s1_settings"]
    assert edge_rows(gm, "s0_home", "s1_settings") == [
        ({"action": "tap"}, 1, "open settings", "", None, None)
    ]
    assert "move_edge_action: missing node ext_com.other" in caplog.text


def test_move_edge_action_moves_into_an_external_node_but_never_out_of_one(
    caplog: pytest.LogCaptureFixture,
) -> None:
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen")
    add_screen(gm, "s1_settings", "Settings page")
    add_screen(gm, EXTERNAL_WEB_NODE, "Outside page")
    gm.add_edge(
        "s0_home",
        "s1_settings",
        {"action": "tap"},
        details=EdgeDetails(instruction="open settings"),
    )
    # add_edge refuses an edge out of an ext_ node, but a loaded graph file can hold one.
    gm.graph.add_edge(EXTERNAL_WEB_NODE, "s0_home", actions=[{"action": "back"}])

    assert gm.move_edge_action("s0_home", "s1_settings", 0, EXTERNAL_WEB_NODE) is True
    with caplog.at_level(logging.WARNING, logger=gm_module.__name__):
        assert gm.move_edge_action(EXTERNAL_WEB_NODE, "s0_home", 0, EXTERNAL_WEB_NODE) is False

    assert edge_rows(gm, "s0_home", EXTERNAL_WEB_NODE) == [
        ({"action": "tap"}, 1, "open settings", "Outside page", None, None)
    ]
    assert not gm.graph.has_edge(EXTERNAL_WEB_NODE, EXTERNAL_WEB_NODE)
    assert gm.graph[EXTERNAL_WEB_NODE]["s0_home"]["actions"] == [{"action": "back"}]
    assert f"move_edge_action: refusing an edge out of external node {EXTERNAL_WEB_NODE}" in (
        caplog.text
    )


def test_relabel_edge_action_replaces_one_instruction_and_drops_the_edge_templates() -> None:
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen")
    add_screen(gm, "s1_settings", "Settings page")
    gm.add_edge(
        "s0_home", "s1_settings", {"action": "tap"}, details=EdgeDetails(instruction="open search")
    )
    gm.add_edge(
        "s0_home",
        "s1_settings",
        {"action": "swipe"},
        details=EdgeDetails(instruction="open settings"),
    )
    gm.graph["s0_home"]["s1_settings"]["instruction_templates"] = [{"template": "open {page}"}]

    gm.relabel_edge_action("s0_home", "s1_settings", 0, "Tap the gear icon")

    data = gm.graph["s0_home"]["s1_settings"]
    assert data["instructions"] == ["Tap the gear icon", "open settings"]
    assert data["actions"] == [{"action": "tap"}, {"action": "swipe"}]
    assert data["instruction_templates"] == []


def test_find_edge_action_matches_nothing_for_an_action_without_an_instruction() -> None:
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen")
    add_screen(gm, "s1_settings", "Settings page")
    gm.add_edge("s0_home", "s1_settings", {"action": "swipe"})
    gm.add_edge(
        "s0_home",
        "s1_settings",
        {"action": "tap"},
        details=EdgeDetails(instruction="open settings"),
    )

    assert gm.find_edge_action("s0_home", "s1_settings", "open settings") == (1, {"action": "tap"})
    assert gm.find_edge_action("s0_home", "s1_settings", "") is None


@dataclass
class FakeNormalizer:
    """Stand-in for vlm_utils.normalize_edge; the client is unused, so it is _client."""

    result: dict[str, Any] = field(default_factory=dict)
    calls: list[dict[str, Any]] = field(default_factory=list)

    def __call__(
        self,
        _client: OpenRouter | None,
        instructions: list[str],
        target_observations: list[str] | None = None,
        settings: ModelSettings | None = None,
    ) -> dict[str, Any]:
        call = {"instructions": list(instructions)}
        call.update(target_observations=target_observations, settings=settings)
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
    gm.add_edge(
        "s0_home",
        "s1_results",
        {"type": "shoes"},
        details=EdgeDetails(instruction='search for "shoes"'),
    )
    gm.add_edge(
        "s0_home",
        "s1_results",
        {"type": "hats"},
        details=EdgeDetails(instruction='search for "hats"'),
    )


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
    assert normalizer.calls[0]["settings"] == PAGE_DETAIL

    gm.normalize_all_edges()  # nothing changed since the last normalization

    assert len(normalizer.calls) == 1


@pytest.mark.parametrize(
    ("result", "templated"),
    [({**TEMPLATE_RESULT, "examples": []}, 1), ({"is_template": False}, 0)],
    ids=["template", "not-a-template"],
)
def test_normalize_all_edges_does_not_send_a_normalized_edge_again(
    normalizer: FakeNormalizer, result: dict[str, Any], templated: int
) -> None:
    """Whatever the model answered, and however many examples its template lists."""
    normalizer.result = result
    gm = make_manager()
    _search_graph(gm)

    assert gm.normalize_all_edges() == templated
    assert gm.normalize_all_edges() == templated

    assert len(normalizer.calls) == 1


def test_normalize_all_edges_sends_an_edge_again_once_it_gains_an_instruction(
    normalizer: FakeNormalizer,
) -> None:
    """However many examples its template lists: here two, as many as the instructions it ends
    with."""
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen")
    add_screen(gm, "s1_results", "Search results")
    gm.add_edge(
        "s0_home",
        "s1_results",
        {"type": "shoes"},
        details=EdgeDetails(instruction='search for "shoes"'),
    )
    gm.normalize_all_edges()
    gm.normalize_all_edges()
    gm.add_edge(
        "s0_home",
        "s1_results",
        {"type": "hats"},
        details=EdgeDetails(instruction='search for "hats"'),
    )

    gm.normalize_all_edges()

    assert [call["instructions"] for call in normalizer.calls] == [
        ['search for "shoes"'],
        ['search for "shoes"', 'search for "hats"'],
    ]


def test_normalize_all_edges_sends_a_relabelled_edge_again_even_under_the_same_instruction(
    normalizer: FakeNormalizer,
) -> None:
    """Relabelling drops the edge's templates, so the next normalization rebuilds them."""
    gm = make_manager()
    _search_graph(gm)
    gm.normalize_all_edges()

    gm.relabel_edge_action("s0_home", "s1_results", 1, 'search for "hats"')
    gm.normalize_all_edges()

    assert len(normalizer.calls) == 2
    assert gm.graph["s0_home"]["s1_results"]["instruction_templates"][0]["template"] == (
        "search for {query}"
    )


def test_a_reloaded_graph_keeps_what_each_edge_last_normalization_covered(
    normalizer: FakeNormalizer, tmp_path: Path
) -> None:
    normalizer.result = {**TEMPLATE_RESULT, "examples": []}
    steps_dir = tmp_path / "steps"
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen", **members(steps_dir))
    add_screen(gm, "s1_results", "Search results", **members(steps_dir))
    for query in ("shoes", "hats"):
        gm.add_edge(
            "s0_home",
            "s1_results",
            {"type": query},
            details=EdgeDetails(instruction=f'search for "{query}"'),
        )
    gm.normalize_all_edges()
    path = tmp_path / "graph.json"
    save_with_text_vectors(gm, path)
    loaded = make_manager()
    loaded.load_graph(path)

    loaded.normalize_all_edges()

    assert len(normalizer.calls) == 1


def test_normalize_all_edges_passes_target_observations(normalizer: FakeNormalizer) -> None:
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen")
    add_screen(gm, "s1_results", "Search results")
    gm.add_edge(
        "s0_home",
        "s1_results",
        {"type": "shoes"},
        details=EdgeDetails(instruction='search for "shoes"', target_observation="shoe results"),
    )

    gm.normalize_all_edges()

    assert normalizer.calls[0]["target_observations"] == ["shoe results"]


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


def test_normalize_all_edges_counts_only_the_recorded_instructions(
    normalizer: FakeNormalizer,
) -> None:
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen")
    add_screen(gm, "s1_results", "Search results")
    gm.add_edge("s0_home", "s1_results", {"action": "tap"})
    gm.add_edge(
        "s0_home",
        "s1_results",
        {"type": "shoes"},
        details=EdgeDetails(instruction='search for "shoes"'),
    )
    gm.add_edge(
        "s0_home",
        "s1_results",
        {"type": "hats"},
        details=EdgeDetails(instruction='search for "hats"'),
    )

    gm.normalize_all_edges()
    gm.normalize_all_edges()  # nothing changed since the last normalization

    assert len(normalizer.calls) == 1
    assert normalizer.calls[0]["instructions"] == [
        "",
        'search for "shoes"',
        'search for "hats"',
    ]
    assert normalizer.calls[0]["target_observations"] is None


def test_normalize_all_edges_applies_the_single_instruction_heuristic_past_placeholders(
    normalizer: FakeNormalizer,
) -> None:
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen")
    add_screen(gm, "s1_results", "Search results")
    gm.add_edge("s0_home", "s1_results", {"action": "tap"})
    gm.add_edge(
        "s0_home", "s1_results", {"action": "click"}, details=EdgeDetails(instruction="open search")
    )

    gm.normalize_all_edges()

    assert normalizer.calls == []
    assert gm.graph["s0_home"]["s1_results"]["instruction_templates"] == []


def test_normalize_all_edges_skips_a_short_single_instruction(normalizer: FakeNormalizer) -> None:
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen")
    add_screen(gm, "s1_settings", "Settings page")
    gm.add_edge(
        "s0_home",
        "s1_settings",
        {"action": "tap"},
        details=EdgeDetails(instruction="open settings"),
    )

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
        details=EdgeDetails(instruction="open the settings page from the home screen menu"),
    )

    gm.normalize_all_edges()

    assert len(normalizer.calls) == 1


def test_normalize_all_edges_without_a_client_does_nothing(
    normalizer: FakeNormalizer,
) -> None:
    gm = GraphManager(embedding=ModelSettings("test/embedding"))
    _search_graph(gm)

    gm.normalize_all_edges()

    assert normalizer.calls == []
    assert "instruction_templates" not in gm.graph["s0_home"]["s1_results"]


def test_normalize_node_edges_counts_templatized_edges(normalizer: FakeNormalizer) -> None:
    gm = make_manager()
    _search_graph(gm)
    add_screen(gm, "s2_settings", "Settings page")
    gm.add_edge(
        "s0_home",
        "s2_settings",
        {"action": "tap"},
        details=EdgeDetails(instruction="open settings"),
    )

    assert gm.normalize_node_edges("s0_home") == 1
    assert len(normalizer.calls) == 1


def test_normalize_node_edges_without_a_client_returns_zero(
    normalizer: FakeNormalizer,
) -> None:
    gm = GraphManager(embedding=ModelSettings("test/embedding"))
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


# How long a call waits for the other calls of its batch before it fails the test instead
# of hanging.
MEET_TIMEOUT = 10.0


def _edges_needing_a_call(gm: GraphManager, count: int) -> list[str]:
    """Give s0_home *count* outgoing edges, each with one instruction that names a value, so
    every edge needs a normalize_edge call. Return the instructions in edge order."""
    add_screen(gm, "s0_home", "Home screen")
    instructions = []
    for index in range(1, count + 1):
        instruction = f'open item "{index}"'
        add_screen(gm, f"s{index}_item", f"Item {index}")
        gm.add_edge(
            "s0_home",
            f"s{index}_item",
            {"index": index},
            details=EdgeDetails(instruction=instruction),
        )
        instructions.append(instruction)
    return instructions


def _template_of(instructions: list[str]) -> dict[str, Any]:
    return {**TEMPLATE_RESULT, "instruction_template": f"template of {instructions[0]}"}


def _home_templates(gm: GraphManager) -> list[str]:
    """Each s0_home edge's template in edge order, "" for an edge without one."""
    return [
        (data.get("instruction_templates") or [{"template": ""}])[0]["template"]
        for _, _, data in gm.graph.out_edges("s0_home", data=True)
    ]


@dataclass
class InFlight:
    """Counts the calls inside it, and the most there were at once."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    current: int = 0
    peak: int = 0

    def __enter__(self) -> None:
        with self.lock:
            self.current += 1
            self.peak = max(self.peak, self.current)

    def __exit__(self, *_exc_info: object) -> None:
        with self.lock:
            self.current -= 1


def test_normalize_all_edges_keeps_the_maximum_number_of_calls_in_flight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gm = make_manager()
    instructions = _edges_needing_a_call(gm, 2 * MAX_REQUESTS_IN_FLIGHT)
    batch = threading.Barrier(MAX_REQUESTS_IN_FLIGHT, timeout=MEET_TIMEOUT)
    in_flight = InFlight()

    def normalize(
        _client: OpenRouter, edge_instructions: list[str], **_kwargs: object
    ) -> dict[str, Any]:
        with in_flight:
            batch.wait()
        return _template_of(edge_instructions)

    monkeypatch.setattr(gm_module, "normalize_edge", normalize)

    gm.normalize_all_edges()

    assert in_flight.peak == MAX_REQUESTS_IN_FLIGHT
    assert _home_templates(gm) == [f"template of {i}" for i in instructions]


def test_normalize_all_edges_applies_and_logs_the_results_in_edge_order(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The last edge's call returns first, and every other one only after the next edge's."""
    gm = make_manager()
    instructions = _edges_needing_a_call(gm, MAX_REQUESTS_IN_FLIGHT)
    returned = {instruction: threading.Event() for instruction in instructions}
    next_edge = dict(itertools.pairwise(instructions))

    def normalize(
        _client: OpenRouter, edge_instructions: list[str], **_kwargs: object
    ) -> dict[str, Any]:
        (instruction,) = edge_instructions
        if instruction in next_edge:
            assert returned[next_edge[instruction]].wait(MEET_TIMEOUT)
        returned[instruction].set()
        return _template_of(edge_instructions)

    monkeypatch.setattr(gm_module, "normalize_edge", normalize)

    with caplog.at_level(logging.INFO, logger=gm_module.__name__):
        gm.normalize_all_edges()

    assert _home_templates(gm) == [f"template of {i}" for i in instructions]
    assert [r.getMessage() for r in caplog.records if "Edge " in r.getMessage()] == [
        f"  Edge s0_home→s{index}_item: 'template of {instruction}' (2 examples)"
        for index, instruction in enumerate(instructions, start=1)
    ]


class YieldingUsage(ChatUsage):
    """Usage whose every field read lets other threads run, so totals that several threads
    update without a lock lose some of the updates."""

    @override
    def __getattribute__(self, name: str) -> object:
        if not name.startswith("_"):
            time.sleep(0.0001)
        return super().__getattribute__(name)


def test_normalize_all_edges_counts_the_usage_of_every_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    usage = YieldingUsage.model_validate(
        {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10, "cost": 0.25}
    )
    concurrent, expected = TokenTracker(), TokenTracker()
    gm = make_manager()
    instructions = _edges_needing_a_call(gm, 2 * MAX_REQUESTS_IN_FLIGHT)
    batch = threading.Barrier(MAX_REQUESTS_IN_FLIGHT, timeout=MEET_TIMEOUT)

    def normalize(
        _client: OpenRouter, edge_instructions: list[str], **_kwargs: object
    ) -> dict[str, Any]:
        batch.wait()
        concurrent.record("normalize_edge", PAGE_DETAIL.model, usage)
        return _template_of(edge_instructions)

    monkeypatch.setattr(gm_module, "normalize_edge", normalize)

    gm.normalize_all_edges()
    for _ in instructions:
        expected.record("normalize_edge", PAGE_DETAIL.model, usage)

    assert concurrent.summary() == expected.summary()


class CreditsExhaustedError(RuntimeError):
    """Stands in for a request the provider refuses, such as one past the account's credit."""


def test_normalize_all_edges_applies_the_returned_templates_and_then_raises_a_failed_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gm = make_manager()
    instructions = _edges_needing_a_call(gm, MAX_REQUESTS_IN_FLIGHT)
    failing = instructions[1]
    others_returned = threading.Semaphore(0)

    def normalize(
        _client: OpenRouter, edge_instructions: list[str], **_kwargs: object
    ) -> dict[str, Any]:
        if edge_instructions == [failing]:
            for _ in instructions[1:]:
                assert others_returned.acquire(timeout=MEET_TIMEOUT)
            raise CreditsExhaustedError
        others_returned.release()
        return _template_of(edge_instructions)

    monkeypatch.setattr(gm_module, "normalize_edge", normalize)

    with pytest.raises(CreditsExhaustedError):
        gm.normalize_all_edges()

    assert _home_templates(gm) == [
        "" if instruction == failing else f"template of {instruction}"
        for instruction in instructions
    ]


def _line_graph(gm: GraphManager) -> None:
    """s0 → s1 → s3 in two cheap hops, or s0 → s3 in one expensive hop."""
    for node_id, description in (
        ("s0_home", "Home screen"),
        ("s1_settings", "Settings page"),
        ("s3_results", "Search results"),
    ):
        add_screen(gm, node_id, description)
    gm.add_edge("s0_home", "s1_settings", {"action": "a"}, details=EdgeDetails(num_steps=1))
    gm.add_edge("s1_settings", "s3_results", {"action": "b"}, details=EdgeDetails(num_steps=1))
    gm.add_edge("s0_home", "s3_results", {"action": "c"}, details=EdgeDetails(num_steps=5))


def test_find_path_from_a_node_to_itself_is_a_single_step() -> None:
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen")

    assert gm.find_path("s0_home", "s0_home") == [PathStep("s0_home", {}, ())]


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
        PathStep("s0_home", {}, ()),
        PathStep("s1_settings", {"action": "a"}, ()),
        PathStep("s3_results", {"action": "b"}, ()),
    ]


def test_find_path_picks_the_action_with_the_fewest_steps() -> None:
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen")
    add_screen(gm, "s1_settings", "Settings page")
    gm.add_edge("s0_home", "s1_settings", {"action": "long"}, details=EdgeDetails(num_steps=3))
    gm.add_edge("s0_home", "s1_settings", {"action": "short"}, details=EdgeDetails(num_steps=1))

    path = gm.find_path("s0_home", "s1_settings")

    assert path == [PathStep("s0_home", {}, ()), PathStep("s1_settings", {"action": "short"}, ())]


def test_find_path_returns_the_setup_of_the_action_it_takes() -> None:
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen")
    add_screen(gm, "s1_settings", "Settings page")
    gm.add_edge("s0_home", "s1_settings", {"action": "row"}, details=EdgeDetails(setup=[_SCROLL]))

    assert gm.find_path("s0_home", "s1_settings") == [
        PathStep("s0_home", {}, ()),
        PathStep("s1_settings", {"action": "row"}, (_SCROLL,)),
    ]


@pytest.mark.parametrize("setups", [[None], [{"action": "swipe"}], [[_SCROLL, "swipe"]]])
def test_find_path_sends_no_setup_for_an_entry_that_is_not_a_list_of_actions(
    setups: list[object],
) -> None:
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen")
    add_screen(gm, "s1_settings", "Settings page")
    gm.graph.add_edge("s0_home", "s1_settings", actions=[{"action": "row"}], setups=setups)

    path = gm.find_path("s0_home", "s1_settings")

    assert path is not None
    assert path[1] == PathStep("s1_settings", {"action": "row"}, ())


def test_find_path_weighs_an_action_by_its_setup_and_its_own_steps() -> None:
    """A one-step action whose setup takes two more loses to a path of two one-step
    actions without one, and on one edge an action without a setup beats one with it."""
    gm = make_manager()
    _line_graph(gm)
    gm.add_edge(
        "s0_home",
        "s3_results",
        {"action": "row"},
        details=EdgeDetails(num_steps=1, setup=[_SCROLL, _SCROLL]),
    )
    gm.add_edge(
        "s0_home", "s1_settings", {"action": "scrolled a"}, details=EdgeDetails(setup=[_SCROLL])
    )

    path = gm.find_path("s0_home", "s3_results")

    assert path == [
        PathStep("s0_home", {}, ()),
        PathStep("s1_settings", {"action": "a"}, ()),
        PathStep("s3_results", {"action": "b"}, ()),
    ]


def test_find_guided_path_navigates_and_runs_a_matching_self_loop() -> None:
    gm = make_manager()
    _line_graph(gm)
    gm.add_edge(
        "s3_results",
        "s3_results",
        {"action": "sort"},
        details=EdgeDetails(
            instruction="sort by price",
            target_observation="sorted by price",
            schema_delta={"sort": {"after": "price"}},
        ),
    )
    gm.graph["s0_home"]["s1_settings"]["instructions"] = ["open settings"]
    gm.graph["s0_home"]["s1_settings"]["target_observations"] = ["the settings page"]
    gm.graph["s1_settings"]["s3_results"]["setups"] = [[_SCROLL]]

    steps = gm.find_guided_path(
        "s0_home", [{"node_id": "s3_results", "required_schema": {"sort": "price"}}]
    )

    assert steps is not None
    assert [step["type"] for step in steps] == ["navigate", "navigate", "self_loop"]
    assert steps[0]["instruction"] == "open settings"
    assert steps[0]["target_observation"] == "the settings page"
    assert (steps[0]["setup"], steps[1]["setup"]) == ([], [_SCROLL])
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
    gm.add_edge(
        "s0_home",
        "s0_home",
        {"action": "sort"},
        details=EdgeDetails(schema_delta={"other": {"after": 1}}),
    )

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


def test_find_guided_path_skips_empty_deltas() -> None:
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen")
    gm.add_edge("s0_home", "s0_home", {"action": "tap"}, details=EdgeDetails(schema_delta={}))
    gm.add_edge(
        "s0_home",
        "s0_home",
        {"action": "sort"},
        details=EdgeDetails(
            instruction="sort by price",
            target_observation="sorted by price",
            schema_delta={"sort": {"after": "price"}},
        ),
    )

    steps = gm.find_guided_path(
        "s0_home", [{"node_id": "s0_home", "required_schema": {"sort": "price"}}]
    )

    assert steps is not None
    assert steps[0]["type"] == "self_loop"
    assert steps[0]["instruction"] == "sort by price"
    assert steps[0]["target_observation"] == "sorted by price"
    assert steps[0]["action"] == {"action": "sort"}


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
    gm.add_edge("s3_results", "s0_home", {"action": "d"})
    gm.add_edge("s3_results", "s0_home", {"action": "e"})

    actions = gm.get_path_actions(["s0_home", "s1_settings", "s3_results", "s0_home"])

    assert actions == [{"action": "a"}, {"action": "b"}, {"action": "d"}]


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


def _home_and_list_with_a_dialog_and_two_external_screens() -> GraphManager:
    """Home and List lead to each other; Dialog and the two external screens lead nowhere."""
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen")
    add_screen(gm, "s1_list", "List page")
    add_screen(gm, "s2_dialog", "Dialog")
    add_screen(gm, "s3_picker", "Picker of another app", is_external=True)
    add_screen(gm, "ext_browser", "Browser")
    gm.add_edge("s0_home", "s1_list", {"action": "open list"})
    gm.add_edge("s1_list", "s0_home", {"action": "back"})
    gm.add_edge("s1_list", "s2_dialog", {"action": "open dialog"})
    gm.add_edge("s0_home", "ext_browser", {"action": "open link"})
    gm.add_edge("s1_list", "s3_picker", {"action": "open picker"})
    return gm


def test_nodes_without_path_to_start_lists_external_nodes_apart_from_in_app_ones() -> None:
    gm = _home_and_list_with_a_dialog_and_two_external_screens()

    assert gm.nodes_without_path_to_start() == PathToStart(
        start_node="s0_home", in_app=["s2_dialog"], external=["ext_browser", "s3_picker"]
    )


def test_nodes_without_path_to_start_lists_every_node_of_a_cycle_with_no_way_out() -> None:
    gm = _home_and_list_with_a_dialog_and_two_external_screens()
    add_screen(gm, "s5_form_keyboard", "Form with keyboard")
    add_screen(gm, "s4_form", "Form")
    gm.add_edge("s1_list", "s4_form", {"action": "open form"})
    gm.add_edge("s4_form", "s5_form_keyboard", {"action": "focus field"})
    gm.add_edge("s5_form_keyboard", "s4_form", {"action": "hide keyboard"})

    assert gm.nodes_without_path_to_start().in_app == ["s2_dialog", "s4_form", "s5_form_keyboard"]


def test_nodes_without_path_to_start_never_lists_the_start_node() -> None:
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen")
    add_screen(gm, "s1_list", "List page")
    gm.add_edge("s0_home", "s1_list", {"action": "open list"})

    assert gm.nodes_without_path_to_start() == PathToStart("s0_home", ["s1_list"], [])


def test_nodes_without_path_to_start_is_unchanged_by_a_self_loop_on_a_dead_end() -> None:
    gm = _home_and_list_with_a_dialog_and_two_external_screens()
    gm.add_edge("s2_dialog", "s2_dialog", {"action": "toggle option"})

    assert gm.nodes_without_path_to_start() == PathToStart(
        start_node="s0_home", in_app=["s2_dialog"], external=["ext_browser", "s3_picker"]
    )


def test_nodes_without_path_to_start_of_an_empty_graph_lists_nothing() -> None:
    assert make_manager().nodes_without_path_to_start() == PathToStart(None, [], [])


def test_nodes_without_path_to_start_of_a_graph_of_external_nodes_lists_nothing() -> None:
    gm = make_manager()
    add_screen(gm, "ext_browser", "Browser")
    add_screen(gm, "s0_picker", "Picker of another app", is_external=True)
    gm.add_edge("s0_picker", "ext_browser", {"action": "open link"})

    assert gm.nodes_without_path_to_start() == PathToStart(None, [], [])


def _unexplored(*descriptions: str) -> list[dict[str, object]]:
    return [{"description": description, "explored": False} for description in descriptions]


def test_get_resume_target_returns_none_without_a_start_node() -> None:
    assert make_manager().get_resume_target() is None


def test_get_resume_target_is_the_top_ranked_exploration_target() -> None:
    """The node with unexplored elements wins over the one with the fewest out-edges,
    and a node no path from the start node reaches is never a target."""
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen")
    add_screen(gm, "s1_settings", "Settings page")
    add_screen(gm, "s2_results", "Search results", interactable_elements=_unexplored("Filter"))
    add_screen(
        gm, "s3_unreachable", "Unreachable screen", interactable_elements=_unexplored("A", "B")
    )
    gm.add_edge("s0_home", "s1_settings", {"action": "a"})
    gm.add_edge("s0_home", "s2_results", {"action": "b"})
    gm.add_edge("s2_results", "s0_home", {"action": "c"})

    assert gm.get_resume_target() == "s2_results"
    assert gm.get_exploration_target_candidates(top_k=1)[0]["node_id"] == "s2_results"


def test_get_resume_target_skips_external_nodes() -> None:
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen")
    add_screen(gm, "ext_browser", "Browser")
    add_screen(gm, "s1_external", "External page", is_external=True)
    add_screen(gm, "s2_results", "Search results")
    gm.add_edge("s0_home", "ext_browser", {"action": "a"})
    gm.add_edge("s0_home", "s1_external", {"action": "b"})
    gm.add_edge("s0_home", "s2_results", {"action": "c"})
    gm.add_edge("s2_results", "s0_home", {"action": "d"})

    assert gm.get_resume_target() == "s2_results"


def test_get_resume_target_filters_by_package() -> None:
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen")
    add_screen(gm, "s1_other", "Other app screen", activity=OTHER_APP, package=OTHER_PACKAGE)
    add_screen(gm, "s2_results", "Search results")
    gm.add_edge("s0_home", "s1_other", {"action": "a"})
    gm.add_edge("s0_home", "s2_results", {"action": "b"})
    gm.add_edge("s2_results", "s0_home", {"action": "c"})

    assert gm.get_resume_target(package_name="com.example.app") == "s2_results"


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
    add_screen(gm, "s2_other", "Other app screen", activity=OTHER_APP, package=OTHER_PACKAGE)
    for target in ("ext_browser", "s1_external", "s2_other"):
        gm.add_edge("s0_home", target, {"action": target})

    candidates = gm.get_exploration_target_candidates(package_name="com.example.app")

    assert [c["node_id"] for c in candidates] == ["s0_home"]


def _start_screen_ranked_below_three_forms(*, sections_explored: bool) -> GraphManager:
    """A start node with two sections that ranks below three forms of five unexplored fields."""
    gm = make_manager()
    sections = [
        {"description": description, "explored": sections_explored}
        for description in ("Display", "Storage")
    ]
    add_screen(gm, "s0_home", "Home screen", visit_count=5, interactable_elements=sections)
    for node_id in ("s1_form", "s2_form", "s3_form"):
        fields = _unexplored("Name", "Address", "Port", "Username", "Password")
        add_screen(gm, node_id, "Form", interactable_elements=fields)
        gm.add_edge("s0_home", node_id, {"action": node_id})
    return gm


def test_get_exploration_target_candidates_leaves_out_a_start_node_ranked_below_top_k() -> None:
    gm = _start_screen_ranked_below_three_forms(sections_explored=False)

    candidates = gm.get_exploration_target_candidates(top_k=2)

    assert len(candidates) == 2
    assert "s0_home" not in [c["node_id"] for c in candidates]


def test_always_offer_start_puts_a_start_node_with_unexplored_elements_last() -> None:
    """The start node takes the place of the last of the ``top_k``, so the list keeps its
    length."""
    gm = _start_screen_ranked_below_three_forms(sections_explored=False)

    candidates = gm.get_exploration_target_candidates(top_k=2, always_offer_start=True)

    assert len(candidates) == 2
    assert candidates[0]["node_id"] in {"s1_form", "s2_form", "s3_form"}
    assert candidates[0]["is_start_node"] is False
    assert candidates[-1]["node_id"] == "s0_home"
    assert candidates[-1]["is_start_node"] is True


def test_always_offer_start_leaves_out_a_start_node_without_unexplored_elements() -> None:
    gm = _start_screen_ranked_below_three_forms(sections_explored=True)

    candidates = gm.get_exploration_target_candidates(top_k=2, always_offer_start=True)

    assert len(candidates) == 2
    assert "s0_home" not in [c["node_id"] for c in candidates]


def test_always_offer_start_keeps_a_start_node_ranked_within_top_k_in_its_place() -> None:
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen", interactable_elements=_unexplored("Display"))
    add_screen(gm, "s1_about", "About page", visit_count=5)
    gm.add_edge("s0_home", "s1_about", {"action": "a"})

    candidates = gm.get_exploration_target_candidates(top_k=2, always_offer_start=True)

    assert [c["node_id"] for c in candidates] == ["s0_home", "s1_about"]


def test_always_offer_start_offers_nothing_when_top_k_is_zero() -> None:
    gm = _start_screen_ranked_below_three_forms(sections_explored=False)

    assert gm.get_exploration_target_candidates(top_k=0, always_offer_start=True) == []


def _candidate_ids(gm: GraphManager) -> list[str]:
    return [c["node_id"] for c in gm.get_exploration_target_candidates()]


def _form_replay_missed(times: int) -> GraphManager:
    """A start screen and a form that outranks it, which replay has missed ``times``
    times in a row."""
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen", interactable_elements=_unexplored("Display"))
    add_screen(gm, "s1_form", "Form", interactable_elements=_unexplored("Name", "Port"))
    gm.add_edge("s0_home", "s1_form", {"action": "a"})
    for _ in range(times):
        gm.record_replay_miss("s1_form")
    return gm


def test_a_node_replay_missed_once_stays_the_top_candidate() -> None:
    gm = _form_replay_missed(1)

    assert gm.get_resume_target() == "s1_form"
    assert _candidate_ids(gm) == ["s1_form", "s0_home"]


def test_a_node_replay_missed_twice_in_a_row_leaves_the_ranking() -> None:
    gm = _form_replay_missed(2)

    assert gm.get_resume_target() == "s0_home"
    assert _candidate_ids(gm) == ["s0_home"]


def test_replay_misses_never_take_the_start_node_out_of_the_candidates() -> None:
    gm = _form_replay_missed(0)
    for _ in range(2):
        gm.record_replay_miss("s0_home")

    assert _candidate_ids(gm) == ["s1_form", "s0_home"]


@pytest.mark.usefixtures("vlm")
def test_a_replay_miss_counts_on_the_node_its_target_was_renamed_to() -> None:
    """Identifying the screen a replay landed on can rename the node it replayed to."""
    gm = _form_replay_missed(1)
    renamed = gm.rename_node("s1_form", "Settings page")

    gm.record_replay_miss("s1_form")

    assert renamed != "s1_form"
    assert _candidate_ids(gm) == ["s0_home"]


def _settings_replay_missed_twice(vlm: FakeVlm, screen: Screens) -> GraphManager:
    """A home screen and a settings page reached from it, each identified from one
    screenshot, which look unlike each other; replay has missed the settings page twice
    in a row."""
    vlm.images.update({SHOT_A: FAR, SHOT_B: NEAR, SHOT_C: NEAR})
    vlm.descriptions.extend([("Home screen", {}, []), ("Settings page", {}, [])])
    vlm.verdicts.append({"same": False, "reason": "another page"})
    gm = make_manager(screen.steps_dir)
    home = gm.identify_state(screen(HOME, SHOT_A, PACKAGE))
    settings = gm.identify_state(screen(SETTINGS, SHOT_B, PACKAGE))
    assert (home, settings) == ("s0_home_screen", "s1_settings_page")
    gm.add_edge(home, settings, {"action": "open settings"})
    for _ in range(2):
        gm.record_replay_miss(settings)
    return gm


@pytest.mark.parametrize(
    ("screenshot", "expected_node", "stage"),
    [
        pytest.param(SHOT_B, None, "exact", id="exact"),
        pytest.param(SHOT_C, "s1_settings_page", "expected", id="expected"),
        pytest.param(SHOT_C, None, "short_list", id="short-list"),
    ],
)
def test_a_screen_identified_as_a_node_replay_missed_clears_its_misses(
    screenshot: str, expected_node: str | None, stage: str, vlm: FakeVlm, screen: Screens
) -> None:
    """The node is a candidate again, and one more miss leaves it one."""
    gm = _settings_replay_missed_twice(vlm, screen)
    assert "s1_settings_page" not in _candidate_ids(gm)
    vlm.descriptions.append(("Settings page", {}, []))
    vlm.verdicts.append({"same": True, "reason": "the settings page"})
    observation = screen(SETTINGS, screenshot, PACKAGE)

    assert gm.identify_state(observation, expected_node=expected_node) == "s1_settings_page"

    assert screen.record(observation)["identify"]["stage"] == stage
    assert "s1_settings_page" in _candidate_ids(gm)
    gm.record_replay_miss("s1_settings_page")
    assert "s1_settings_page" in _candidate_ids(gm)


def _app_with_a_gmail_screen(package: str, activity: str, other_activity: str) -> GraphManager:
    gm = make_manager()
    add_screen(gm, "s0_list", "Home screen", activity=activity, package=package)
    add_screen(gm, "s1_detail", "Settings page", activity=other_activity, package=package)
    add_screen(gm, "s2_mail", "Search results", activity=GMAIL_LIST, package=GMAIL)
    gm.add_edge("s0_list", "s1_detail", {"action": "a"})
    gm.add_edge("s0_list", "s2_mail", {"action": "b"})
    gm.add_edge("s1_detail", "s0_list", {"action": "c"})
    return gm


@any_package_length
def test_get_resume_target_finds_the_nodes_of_a_package_of_any_length(
    package: str, activity: str, other_activity: str
) -> None:
    gm = _app_with_a_gmail_screen(package, activity, other_activity)

    assert gm.get_resume_target(package_name=package) == "s1_detail"


@any_package_length
def test_get_exploration_target_candidates_are_the_nodes_of_a_package_of_any_length(
    package: str, activity: str, other_activity: str
) -> None:
    gm = _app_with_a_gmail_screen(package, activity, other_activity)

    candidates = gm.get_exploration_target_candidates(package_name=package)

    assert sorted(c["node_id"] for c in candidates) == ["s0_list", "s1_detail"]


@any_package_length
def test_identify_state_matches_a_node_of_the_package_under_another_activity(
    vlm: FakeVlm, package: str, activity: str, other_activity: str, screen: Screens
) -> None:
    vlm.descriptions.append(("Home screen", {}, []))
    vlm.verdicts.append({"same": True, "reason": "same page"})
    gm = make_manager(screen.steps_dir)
    add_screen(
        gm,
        "s0_home",
        "Home screen",
        activity=other_activity,
        package=package,
        **members(screen.steps_dir),
    )
    gm.embed_missing_members()

    assert gm.identify_state(screen(activity, SHOT_A, package)) == "s0_home"
    assert gm.graph.nodes["s0_home"]["activities"] == [other_activity, activity]


def test_identify_state_keeps_apart_packages_that_share_three_segments(
    vlm: FakeVlm, screen: Screens
) -> None:
    vlm.descriptions.append(("Home screen", {}, []))
    gm = make_manager(screen.steps_dir)
    add_screen(gm, "s0_mail", "Home screen", activity=GMAIL_LIST, package=GMAIL)

    node_id = gm.identify_state(screen(MESSAGES_LIST, SHOT_A, MESSAGES))

    assert node_id == "s0_home_screen"
    assert gm.graph.nodes[node_id]["package"] == MESSAGES
    assert vlm.calls[1][1]["existing_nodes"] is None


def test_save_graph_writes_members_but_no_image_and_load_graph_restores_them(
    tmp_path: Path, vlm: FakeVlm
) -> None:
    vlm.descriptions.append(("Home screen", {}, []))
    vlm.images[SHOT_A] = [0.0, 0.6, 0.8]
    graph_path = tmp_path / "nested" / "graph.json"
    steps_dir = graph_path.parent / "steps"
    gm = make_manager(steps_dir)
    observation = Screens(steps_dir)(HOME, SHOT_A, PACKAGE)
    node_id = gm.identify_state(observation)

    gm.save_graph(graph_path)

    assert sorted(path.name for path in graph_path.parent.iterdir()) == [
        "graph.json",
        "graph_text_emb.json",
        "steps",
    ]
    assert sorted(path.name for path in steps_dir.iterdir()) == [
        f"{observation.observation_id}.json",
        f"{observation.observation_id}.webp",
        "image_emb.json",
    ]
    assert json.loads((steps_dir / "image_emb.json").read_text(encoding="utf-8")) == {
        "model": IMAGE_EMBEDDING.model,
        "embeddings": {observation.observation_id: [0.0, 0.6, 0.8]},
    }
    saved = json.loads(graph_path.read_text(encoding="utf-8"))["nodes"][0]
    assert saved["observations"] == [observation.observation_id]
    assert saved["representative_observation"] == observation.observation_id

    loaded = make_manager()
    loaded.load_graph(graph_path)
    restored = loaded.get_node(node_id)
    assert restored is not None
    assert restored["observations"] == [observation.observation_id]
    assert restored["representative_observation"] == observation.observation_id
    assert loaded.steps_dir == steps_dir
    assert loaded.image_vector(observation.observation_id) == [0.0, 0.6, 0.8]


def test_node_aliases_survive_save_and_load(tmp_path: Path) -> None:
    """A node ID recorded before a rename and a merge still resolves after a reload."""
    steps_dir = tmp_path / "steps"
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen", **members(steps_dir))
    add_screen(gm, "s1_home", "Home screen", **members(steps_dir))
    renamed = gm.rename_node("s1_home", "Home feed")
    assert gm.merge_nodes("s0_home", renamed) is True
    kept = gm.rename_node("s0_home", "Settings page")
    path = tmp_path / "graph.json"

    save_with_text_vectors(gm, path)
    loaded = make_manager()
    loaded.load_graph(path)

    assert json.loads(path.read_text(encoding="utf-8"))["node_aliases"] == {
        "s1_home": renamed,
        renamed: "s0_home",
        "s0_home": kept,
    }
    assert {loaded.resolve_node_id(node) for node in ("s1_home", renamed, "s0_home")} == {kept}


def test_save_graph_persists_nodes_edges_and_counters(tmp_path: Path) -> None:
    gm = make_manager()
    gm.total_steps_completed = 42
    add_screen(
        gm,
        "s0_home",
        "Home screen",
        visit_count=3,
        observations=["000000", "000002"],
        representative_observation="000000",
    )
    add_screen(gm, "s1_settings", "Settings page")
    gm.add_edge(
        "s0_home",
        "s1_settings",
        {"action": "tap"},
        details=EdgeDetails(
            instruction="open settings", schema_delta={"tab": {"after": "settings"}}
        ),
    )
    path = tmp_path / "graph.json"

    gm.save_graph(path)

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["total_steps_completed"] == 42
    assert data["nodes"][0]["activity"] == HOME
    assert data["nodes"][0]["visit_count"] == 3
    assert data["nodes"][0]["observations"] == ["000000", "000002"]
    assert data["nodes"][0]["representative_observation"] == "000000"
    assert data["node_aliases"] == {}
    assert data["edges"][0]["schema_deltas"] == [{"tab": {"after": "settings"}}]


def test_save_graph_omits_an_empty_activity_list_and_writes_no_empty_text_vectors(
    tmp_path: Path,
) -> None:
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen", activities=[])
    path = tmp_path / "graph.json"

    gm.save_graph(path)

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["nodes"][0]["activity"] == ""
    assert sorted(p.name for p in tmp_path.iterdir()) == ["graph.json"]


def test_save_graph_leaves_the_previous_files_intact_on_a_failed_dump(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, vlm: FakeVlm, screen: Screens
) -> None:
    """A crash partway through json.dump must leave the previous complete graph
    file and its text vectors in place, never a truncated mix of the old and new
    content, and must not leave a stray temp file behind: both files are written
    through graph_files.write_json_atomically.
    """
    vlm.descriptions.append(("Home screen", {}, []))
    gm = make_manager(screen.steps_dir)
    gm.identify_state(screen(HOME, SHOT_A, PACKAGE))
    path = tmp_path / "graph.json"
    gm.save_graph(path)
    emb_path = tmp_path / "graph_text_emb.json"
    original_graph = path.read_text(encoding="utf-8")
    original_emb = emb_path.read_text(encoding="utf-8")

    def _dump_then_blow_up(_obj: object, fp: IO[str], **_kwargs: object) -> None:
        fp.write('{"broken"')  # a partial write, as a real crash mid-dump would leave
        msg = "boom"
        raise ValueError(msg)

    monkeypatch.setattr(gm_module.json, "dump", _dump_then_blow_up)

    with pytest.raises(ValueError, match="boom"):
        gm.save_graph(path)

    assert path.read_text(encoding="utf-8") == original_graph
    assert emb_path.read_text(encoding="utf-8") == original_emb
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "graph.json",
        "graph_text_emb.json",
        "steps",
    ]


@pytest.mark.parametrize(
    ("failing_file", "copy_vector"),
    [
        pytest.param("graph.json", EMBEDDINGS["Home screen"], id="graph file"),
        pytest.param("graph_text_emb.json", None, id="text vectors"),
    ],
)
def test_a_save_that_stops_at_either_file_leaves_a_graph_that_loads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    vlm: FakeVlm,
    screen: Screens,
    failing_file: str,
    copy_vector: list[float] | None,
) -> None:
    """The text vectors are written before the graph, so a save that stops at the graph
    file leaves the new member's vector beside the previous graph, and one that stops at
    the vectors leaves both files as they were: the graph on disk never lists a member
    without a vector."""
    vlm.descriptions.append(("Home screen", {}, []))
    gm = make_manager(screen.steps_dir)
    member = screen(HOME, SHOT_A, PACKAGE)
    node_id = gm.identify_state(member)
    path = tmp_path / "graph.json"
    gm.save_graph(path)
    copy = screen(HOME, SHOT_A, PACKAGE)
    gm.identify_state(copy)
    original_replace = os.replace

    def _replace_unless_failing(src: Path, dst: Path) -> None:
        if dst.name == failing_file:
            msg = "No space left on device"
            raise OSError(msg)
        original_replace(src, dst)

    monkeypatch.setattr(os, "replace", _replace_unless_failing)

    with pytest.raises(OSError, match="No space left on device"):
        gm.save_graph(path)
    loaded = make_manager()
    loaded.load_graph(path)

    assert loaded.graph.nodes[node_id]["observations"] == [member.observation_id]
    assert loaded.text_vector(copy.observation_id) == copy_vector


def test_save_graph_and_load_graph_round_trip_each_node_package(tmp_path: Path) -> None:
    source = make_manager()
    steps_dir = tmp_path / "steps"
    add_screen(
        source,
        "s0_list",
        "Home screen",
        activity=MESSAGES_LIST,
        package=MESSAGES,
        **members(steps_dir),
    )
    add_screen(
        source, "s1_mail", "Settings page", activity=GMAIL_LIST, package=GMAIL, **members(steps_dir)
    )
    path = tmp_path / "graph.json"
    save_with_text_vectors(source, path)

    target = make_manager()
    target.load_graph(path)

    assert dict(target.graph.nodes(data="package")) == {"s0_list": MESSAGES, "s1_mail": GMAIL}


def test_load_graph_replaces_the_current_graph(tmp_path: Path) -> None:
    source = make_manager()
    add_screen(source, "s0_home", "Home screen", **members(tmp_path / "steps"))
    path = tmp_path / "graph.json"
    save_with_text_vectors(source, path)

    target = make_manager()
    add_screen(target, "s9_stale", "Stale screen")
    target.load_graph(path)

    assert list(target.graph.nodes) == ["s0_home"]


def test_load_graph_rejects_a_file_without_an_edges_list(tmp_path: Path) -> None:
    path = tmp_path / "graph.json"
    path.write_text(
        json.dumps({"nodes": [{"id": "s0_home", **members(tmp_path / "steps")}]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="list fields 'nodes' and 'edges'") as excinfo:
        make_manager().load_graph(path)

    assert str(path) in str(excinfo.value)
    assert "Explore the app again into a new graph directory" in str(excinfo.value)


def test_an_audited_graph_shares_the_step_log_of_the_graph_it_was_derived_from(
    tmp_path: Path,
) -> None:
    app_dir = tmp_path / "demo"
    membership = members(app_dir / "steps", b"home-image")
    path = app_dir / "demo_audited.json"
    path.write_text(
        json.dumps(
            {
                "nodes": [{"id": "s0_home", "activity": HOME, "replay_misses": 0, **membership}],
                "edges": [],
            }
        ),
        encoding="utf-8",
    )
    write_text_vectors(path, membership)
    gm = make_manager()

    gm.load_graph(path)

    assert gm.steps_dir == app_dir / "steps"


def write_graph(path: Path, nodes: list[dict[str, object]]) -> None:
    path.write_text(json.dumps({"nodes": nodes, "edges": []}), encoding="utf-8")


@pytest.mark.parametrize(
    "membership",
    [
        {},
        {"observations": [], "representative_observation": None},
        {"observations": "000000", "representative_observation": "000000"},
        {"observations": ["000000", 1], "representative_observation": "000000"},
        {"observations": ["000000"]},
        {"observations": ["000000"], "representative_observation": "000001"},
    ],
    ids=[
        "no-members",
        "empty-members",
        "members-not-a-list",
        "a-member-not-a-string",
        "no-representative",
        "representative-not-a-member",
    ],
)
def test_load_graph_rejects_a_node_without_members_naming_it(
    tmp_path: Path, membership: dict[str, object]
) -> None:
    write_observation_image(tmp_path / "steps", b"image")
    write_observation_image(tmp_path / "steps", b"image")
    path = tmp_path / "graph.json"
    write_graph(
        path,
        [
            {"id": "s0_home", "replay_misses": 0, **members(tmp_path / "steps")},
            {"id": "s1_settings", "replay_misses": 0, **membership},
        ],
    )
    gm = make_manager()

    with pytest.raises(ValueError, match="no member observations") as excinfo:
        gm.load_graph(path)

    message = str(excinfo.value)
    assert "['s1_settings']" in message
    assert "s0_home" not in message
    assert str(path) in message
    assert "Explore the app again into a new graph directory" in message


def test_load_graph_rejects_a_representative_without_an_image_naming_the_node(
    tmp_path: Path,
) -> None:
    path = tmp_path / "graph.json"
    write_graph(
        path,
        [
            {"id": "s0_home", **members(tmp_path / "steps")},
            {
                "id": "s1_settings",
                "observations": ["000007"],
                "representative_observation": "000007",
            },
        ],
    )
    gm = make_manager()

    with pytest.raises(ValueError, match="without an image") as excinfo:
        gm.load_graph(path)

    assert "['s1_settings']" in str(excinfo.value)
    assert str(tmp_path / "steps") in str(excinfo.value)
    assert "no member observations" not in str(excinfo.value)


@pytest.mark.parametrize(
    "count", [{}, {"replay_misses": "0"}], ids=["no-count", "count-not-an-int"]
)
def test_load_graph_rejects_a_node_without_a_replay_misses_count_naming_it(
    tmp_path: Path, count: dict[str, object]
) -> None:
    path = tmp_path / "graph.json"
    write_graph(
        path,
        [
            {"id": "s0_home", "replay_misses": 0, **members(tmp_path / "steps")},
            {"id": "s1_settings", **count, **members(tmp_path / "steps")},
        ],
    )
    gm = make_manager()
    add_screen(gm, "s9_stale", "Stale screen")

    with pytest.raises(ValueError, match="without a replay_misses count") as excinfo:
        gm.load_graph(path)

    message = str(excinfo.value)
    assert "['s1_settings']" in message
    assert "s0_home" not in message
    assert str(path) in message
    assert "Explore the app again into a new graph directory." in message
    assert list(gm.graph.nodes) == ["s9_stale"]


def test_load_graph_names_every_node_it_rejects_and_keeps_the_loaded_graph(
    tmp_path: Path,
) -> None:
    previous = tmp_path / "previous" / "graph.json"
    previous.parent.mkdir()
    membership = members(previous.parent / "steps")
    write_graph(previous, [{"id": "s9_stale", "replay_misses": 0, **membership}])
    write_text_vectors(previous, membership)
    path = tmp_path / "graph.json"
    write_graph(
        path,
        [
            {"id": "s0_home"},
            {
                "id": "s1_settings",
                "observations": ["000007"],
                "representative_observation": "000007",
            },
        ],
    )
    gm = make_manager()
    gm.load_graph(previous)

    with pytest.raises(ValueError, match=r"\['s0_home'\].*\['s1_settings'\]"):
        gm.load_graph(path)

    assert list(gm.graph.nodes) == ["s9_stale"]
    assert gm.steps_dir == previous.parent / "steps"


def test_load_graph_rejects_an_edge_whose_endpoint_is_not_in_the_file(tmp_path: Path) -> None:
    """Such an edge fails the load instead of growing a phantom node."""
    path = tmp_path / "graph.json"
    path.write_text(
        json.dumps(
            {
                "nodes": [{"id": "s0_home", "activity": HOME, "page_description": "Home screen"}],
                "edges": [{"source": "s0_home", "target": "s1_missing", "actions": []}],
            }
        ),
        encoding="utf-8",
    )
    gm = make_manager()
    add_screen(gm, "s9_stale", "Stale screen")

    with pytest.raises(ValueError, match="s1_missing"):
        gm.load_graph(path)

    assert list(gm.graph.nodes) == ["s9_stale"]


def test_load_graph_rejects_a_node_without_an_id_naming_the_path(tmp_path: Path) -> None:
    """A node missing "id" must surface as a path-bearing error, not a bare
    KeyError('id') from indexing the node dict, and must be caught before
    ``self.graph.clear()`` or any counter is overwritten, so a bad file leaves
    the previously loaded graph and its counters untouched rather than wiping
    them partway through.
    """
    path = tmp_path / "graph.json"
    path.write_text(
        json.dumps({"next_id": 99, "nodes": [{"page_description": "x"}], "edges": []}),
        encoding="utf-8",
    )
    previous = tmp_path / "previous.json"
    membership = members(tmp_path / "steps")
    previous.write_text(
        json.dumps(
            {
                "next_id": 7,
                "nodes": [{"id": "s9_stale", "replay_misses": 0, **membership}],
                "edges": [],
            }
        ),
        encoding="utf-8",
    )
    write_text_vectors(previous, membership)
    gm = make_manager()
    gm.load_graph(previous)

    with pytest.raises(ValueError, match="node id must be a string") as excinfo:
        gm.load_graph(path)
    assert str(path) in str(excinfo.value)

    kept = tmp_path / "kept.json"
    gm.save_graph(kept)
    saved = json.loads(kept.read_text(encoding="utf-8"))
    assert saved["next_id"] == 7
    assert [node["id"] for node in saved["nodes"]] == ["s9_stale"]


def test_load_graph_rejects_a_null_nodes_field_naming_the_path(tmp_path: Path) -> None:
    """``"nodes": null`` raises the ``ValueError`` every load failure raises, naming the
    path, from ``load_graph``, which has no shape check of its own and relies entirely on
    the shared validator.
    """
    path = tmp_path / "graph.json"
    path.write_text(json.dumps({"nodes": None, "edges": []}), encoding="utf-8")
    gm = make_manager()

    with pytest.raises(ValueError, match="must contain list fields 'nodes' and 'edges'") as excinfo:
        gm.load_graph(path)
    assert str(path) in str(excinfo.value)
    assert "Explore the app again into a new graph directory" in str(excinfo.value)


def test_load_graph_rejects_an_int_node_id_naming_the_path(tmp_path: Path) -> None:
    """An int id would otherwise sail through ``require_known_edge_endpoints``,
    which compares ``str()``-normalised ids, and let ``add_edge`` grow a
    phantom node with a different (string) id -- the same phantom node
    ``require_known_edge_endpoints`` keeps a genuinely undefined endpoint from growing.
    """
    path = tmp_path / "graph.json"
    path.write_text(json.dumps({"nodes": [{"id": 5}], "edges": []}), encoding="utf-8")
    gm = make_manager()

    with pytest.raises(ValueError, match="node id must be a string") as excinfo:
        gm.load_graph(path)
    assert str(path) in str(excinfo.value)


def test_load_graph_rejects_an_int_edge_endpoint_naming_the_path(tmp_path: Path) -> None:
    path = tmp_path / "graph.json"
    path.write_text(
        json.dumps(
            {
                "nodes": [{"id": "n1"}],
                "edges": [{"source": 5, "target": "n1"}],
            }
        ),
        encoding="utf-8",
    )
    gm = make_manager()

    with pytest.raises(ValueError, match="edge endpoint must be a string") as excinfo:
        gm.load_graph(path)
    assert str(path) in str(excinfo.value)


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
        details=EdgeDetails(instruction="open settings", target_observation="the settings page"),
    )
    gm.add_edge(
        "s0_home", "s0_home", {"action": "sort"}, details=EdgeDetails(instruction="sort by price")
    )

    text = gm.format_for_audit()

    assert "## Nodes (2)" in text
    assert "## Edges (2)" in text
    assert "state: [tab]" in text
    assert "elements: 1/1 explored" in text
    assert '- "open settings" → the settings page' in text
    assert "[self-loop]" in text


def test_format_for_audit_pairs_each_instruction_with_its_own_observation() -> None:
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen")
    add_screen(gm, "s1_settings", "Settings page")
    gm.add_edge(
        "s0_home",
        "s1_settings",
        {"action": "tap"},
        details=EdgeDetails(target_observation="first only"),
    )
    gm.add_edge(
        "s0_home",
        "s1_settings",
        {"action": "swipe"},
        details=EdgeDetails(instruction="swipe to settings"),
    )

    edge_lines = gm.format_for_audit().split("## Edges (1)\n", 1)[1].splitlines()

    assert edge_lines[1:] == ['    - "swipe to settings"']


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
    assert fake.calls[0]["settings"] == PAGE_DETAIL._replace(request_timeout=300.0)
    assert fake.calls[0]["graph_summary"].startswith("## Nodes (1)")


@pytest.mark.parametrize("operation", ["audit", "merge_audit"])
def test_audits_send_their_own_timeout(
    operation: str,
    router_client: OpenRouter,
    router_requests: list[httpx.Request],
    router_responses: list[httpx.Response],
    chat_response: Callable[..., httpx.Response],
) -> None:
    manager = GraphManager(
        client=router_client, page_detail=PAGE_DETAIL, embedding=EMBEDDING, audit_timeout=120
    )
    router_responses.append(chat_response(content='{"issues": [], "summary": "fine"}'))
    if operation == "audit":
        assert manager.run_audit()["summary"] == "fine"
    else:
        assert manager.run_node_merge_audit()["merged_count"] == 0
    assert router_requests[0].extensions["timeout"]["read"] == 120


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


def test_run_node_merge_audit_skips_a_self_pair(
    monkeypatch: pytest.MonkeyPatch, vlm: FakeVlm
) -> None:
    """An auditor naming one node twice keeps the node."""
    fake = FakeAudit(result={"issues": [{"node_a": "s0_home", "node_b": "s0_home"}]})
    monkeypatch.setattr(gm_module, "audit_merge_nodes", fake)
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen", visit_count=3)

    result = gm.run_node_merge_audit()

    assert result["merged_count"] == 0
    assert result["results"][0]["status"] == "skipped"
    assert result["results"][0]["reason"] == "same node"
    assert list(gm.graph.nodes) == ["s0_home"]
    assert gm.graph.nodes["s0_home"]["visit_count"] == 3
    assert vlm.kinds() == []


@pytest.mark.parametrize("without_representative", ["s0_home", "s1_home_two"])
def test_run_node_merge_audit_skips_a_node_without_a_representative(
    monkeypatch: pytest.MonkeyPatch, vlm: FakeVlm, tmp_path: Path, without_representative: str
) -> None:
    fake = FakeAudit(result={"issues": [{"node_a": "s0_home", "node_b": "s1_home_two"}]})
    monkeypatch.setattr(gm_module, "audit_merge_nodes", fake)
    gm = make_manager(steps_dir=tmp_path / "steps")
    add_screen(gm, "s0_home", "Home screen", **members(tmp_path / "steps"))
    add_screen(gm, "s1_home_two", "Home screen again", **members(tmp_path / "steps"))
    del gm.graph.nodes[without_representative]["representative_observation"]

    result = gm.run_node_merge_audit()

    assert result["results"][0]["reason"] == "missing screenshot"
    assert vlm.kinds() == []


@pytest.mark.parametrize("external_side", ["node_a", "node_b"])
def test_run_node_merge_audit_never_verifies_an_external_node(
    monkeypatch: pytest.MonkeyPatch, vlm: FakeVlm, tmp_path: Path, external_side: str
) -> None:
    """An ``ext_`` node stands for every screen of another app or the web, so its
    representative is no screen of the app to compare with."""
    pair = {"node_a": "s0_home", "node_b": "s1_home_two", external_side: EXTERNAL_WEB_NODE}
    fake = FakeAudit(result={"issues": [pair]})
    monkeypatch.setattr(gm_module, "audit_merge_nodes", fake)
    gm = make_manager(steps_dir=tmp_path / "steps")
    for node_id in pair.values():
        add_screen(gm, node_id, "Home screen", **members(tmp_path / "steps"))

    result = gm.run_node_merge_audit()

    assert result["results"][0]["status"] == "skipped"
    assert result["results"][0]["reason"] == "external node"
    assert result["merged_count"] == 0
    assert vlm.kinds() == []


def test_run_node_merge_audit_merges_a_confirmed_duplicate(
    monkeypatch: pytest.MonkeyPatch, vlm: FakeVlm, tmp_path: Path
) -> None:
    fake = FakeAudit(
        result={"issues": [{"node_a": "s2_home_two", "node_b": "s0_home"}], "summary": "one pair"}
    )
    monkeypatch.setattr(gm_module, "audit_merge_nodes", fake)
    vlm.verdicts.append({"same": True, "reason": "same screen"})
    steps_dir = tmp_path / "steps"
    gm = make_manager(steps_dir=steps_dir)
    add_screen(gm, "s0_home", "Home screen", visit_count=1, **members(steps_dir, b"home-image"))
    add_screen(
        gm,
        "s2_home_two",
        "Home screen again",
        visit_count=5,
        **members(steps_dir, b"home-two-image"),
    )
    removed_members = gm.graph.nodes["s2_home_two"]["observations"]

    result = gm.run_node_merge_audit()

    assert result["merged_count"] == 1
    assert result["summary"] == "one pair"
    assert result["results"][0]["kept"] == "s0_home"
    assert list(gm.graph.nodes) == ["s0_home"]
    assert gm.graph.nodes["s0_home"]["observations"][1:] == removed_members
    # The description of the more-visited node is the verifier's reference.
    assert vlm.calls[0][1]["existing_description"] == "Home screen again"
    # Each side's representative image comes from the step log.
    assert vlm.calls[0][1]["existing"] == b64(b"home-two-image")
    assert vlm.calls[0][1]["new"] == b64(b"home-image")


@pytest.mark.parametrize("other", ["search_results", "browser"])
def test_run_node_merge_audit_orders_non_numeric_ids_last(
    monkeypatch: pytest.MonkeyPatch, vlm: FakeVlm, tmp_path: Path, other: str
) -> None:
    fake = FakeAudit(result={"issues": [{"node_a": other, "node_b": "s0_home"}]})
    monkeypatch.setattr(gm_module, "audit_merge_nodes", fake)
    vlm.verdicts.append({"same": True, "reason": "same screen"})
    gm = make_manager(steps_dir=tmp_path / "steps")
    add_screen(gm, "s0_home", "Home screen", **members(tmp_path / "steps"))
    add_screen(gm, other, "Search results", **members(tmp_path / "steps"))

    result = gm.run_node_merge_audit()

    assert result["results"][0]["kept"] == "s0_home"


def test_run_node_merge_audit_keeps_a_rejected_pair_separate(
    monkeypatch: pytest.MonkeyPatch, vlm: FakeVlm, tmp_path: Path
) -> None:
    fake = FakeAudit(result={"issues": [{"node_a": "s0_home", "node_b": "s1_home_two"}]})
    monkeypatch.setattr(gm_module, "audit_merge_nodes", fake)
    vlm.verdicts.append({"same": False, "reason": "different banner"})
    gm = make_manager(steps_dir=tmp_path / "steps")
    add_screen(gm, "s0_home", "Home screen", **members(tmp_path / "steps"))
    add_screen(gm, "s1_home_two", "Home screen again", **members(tmp_path / "steps"))

    result = gm.run_node_merge_audit()

    assert result["merged_count"] == 0
    assert result["results"][0]["status"] == "kept_separate"
    assert result["results"][0]["reason"] == "different banner"
    assert gm.graph.number_of_nodes() == 2


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
setup_entries = st.none() | st.lists(actions, max_size=3)
step_ids = st.lists(
    st.integers(min_value=0, max_value=10_000_000), min_size=1, max_size=4, unique=True
).map(lambda numbers: [f"{number:06d}" for number in numbers])
memberships = step_ids.flatmap(
    lambda ids: st.fixed_dictionaries(
        {"observations": st.just(ids), "representative_observation": st.sampled_from(ids)}
    )
)
screen_attrs = st.fixed_dictionaries(
    {
        "package": st.text(min_size=1, max_size=30),
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
        "last_normalized_visit_milestone": st.integers(min_value=0, max_value=10),
        "visit_count": st.integers(min_value=0, max_value=100),
        "replay_misses": st.integers(min_value=0, max_value=10),
    }
)
node_attrs = st.tuples(screen_attrs, memberships).map(lambda parts: {**parts[0], **parts[1]})
# One tuple per action: the action, then its entry in each of PER_ACTION_LISTS.
recorded_actions = st.lists(
    st.tuples(
        actions | st.lists(actions, max_size=3),
        st.integers(min_value=1, max_value=5),
        st.text(max_size=12),
        st.text(max_size=12),
        st.none() | st.dictionaries(st.text(max_size=6), json_scalars, max_size=2),
        setup_entries,
    ),
    min_size=1,
    max_size=3,
)


def edge_from(
    recorded: list[tuple[object, ...]],
    templates: list[dict[str, Any]],
    visit_count: int,
    instructions_hash: str | None,
) -> dict[str, Any]:
    """An edge's attributes with one entry per recorded action in each per-action list."""
    keys = ("actions", *PER_ACTION_LISTS)
    return {
        **{key: [entry[index] for entry in recorded] for index, key in enumerate(keys)},
        "instruction_templates": templates,
        "visit_count": visit_count,
        "normalized_instructions_hash": instructions_hash,
    }


edge_attrs = st.builds(
    edge_from,
    recorded_actions,
    small_dicts,
    st.integers(min_value=0, max_value=20),
    st.none() | st.text(max_size=12),
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
            "package": d.get("package", ""),
            "activities": d.get("activities", []),
            "page_description": d.get("page_description", ""),
            "state_schema": d.get("state_schema", {}),
            "last_detail_snapshot": d.get("last_detail_snapshot", {}),
            "interactable_elements": d.get("interactable_elements", []),
            "last_normalized_visit_milestone": d.get("last_normalized_visit_milestone", 0),
            "visit_count": d.get("visit_count", 0),
            "replay_misses": d["replay_misses"],
            "observations": d.get("observations", []),
            "representative_observation": d.get("representative_observation"),
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
            "schema_deltas": d.get("schema_deltas", []),
            "setups": d.get("setups", []),
            "normalized_instructions_hash": d.get("normalized_instructions_hash") or None,
        }
        for u, v, d in gm.graph.edges(data=True)
    }
    return {"nodes": nodes, "edges": edges}


@given(graph=graphs, steps=st.integers(min_value=0, max_value=10_000))
def test_save_then_load_round_trips(
    tmp_path_factory: pytest.TempPathFactory,
    graph: tuple[dict[str, dict[str, Any]], dict[tuple[str, str], dict[str, Any]]],
    steps: int,
) -> None:
    nodes, edges = graph
    gm = GraphManager()
    gm.total_steps_completed = steps
    for node_id, attrs in nodes.items():
        gm.graph.add_node(node_id, activity=attrs["activities"][0], **attrs)
    for (u, v), attrs in edges.items():
        if u in nodes and v in nodes:
            gm.graph.add_edge(u, v, **attrs)
    path = tmp_path_factory.mktemp("roundtrip") / "graph.json"
    for attrs in nodes.values():
        for observation_id in attrs["observations"]:
            write_member_record(path.parent / "steps", observation_id)
        (path.parent / "steps" / f"{attrs['representative_observation']}.webp").write_bytes(b"x")

    gm.save_graph(path)
    loaded = GraphManager()
    loaded.load_graph(path)

    assert canonical(loaded) == canonical(gm)
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
    gm = GraphManager(embedding=ModelSettings("test/embedding"))
    add_screen(gm, "s0_home", "Home screen")
    add_screen(gm, "s1_settings", "Settings page")

    for _ in range(repeats):
        gm.add_edge("s0_home", "s1_settings", action, details=EdgeDetails(instruction=instruction))

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


def fake_get_embedding(
    _client: OpenRouter | None, text: str, *_args: object, **_kwargs: object
) -> list[float]:
    """Module-level fake for property tests; the model and timeout arrive through the rest."""
    return EMBEDDINGS.get(text, UNKNOWN_EMBEDDING)


@given(ops=operations)
def test_every_edge_references_an_existing_node(ops: list[tuple[str, str, str, str]]) -> None:
    gm = make_manager()
    for node_id in POOL:
        add_screen(gm, node_id, f"screen {node_id}")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(gm_module, "get_embedding", fake_get_embedding)
        for kind, a, b, text in ops:
            if kind == "edge":
                gm.add_edge(a, b, {"action_type": "tap", "target": text})
            elif kind == "merge":
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


edge_and_merge_operations = st.lists(
    st.tuples(
        st.sampled_from(["edge", "merge"]),
        st.sampled_from(POOL),
        st.sampled_from(POOL),
        actions | st.lists(actions, max_size=3),
        st.none() | st.text(max_size=12),
        st.none() | st.text(max_size=12),
        st.none() | st.dictionaries(st.text(max_size=6), json_scalars, max_size=2),
        setup_entries,
    ),
    max_size=25,
)


def per_action_lists(gm: GraphManager) -> dict[tuple[str, str], dict[str, list[Any]]]:
    return {
        (u, v): {key: d.get(key, []) for key in ("actions", *PER_ACTION_LISTS)}
        for u, v, d in gm.graph.edges(data=True)
    }


def misaligned_lists(gm: GraphManager) -> list[tuple[str, str, str]]:
    return [
        (u, v, key)
        for (u, v), lists in per_action_lists(gm).items()
        for key in PER_ACTION_LISTS
        if len(lists[key]) != len(lists["actions"])
    ]


@given(ops=edge_and_merge_operations)
def test_every_per_action_list_has_one_entry_per_action(
    tmp_path_factory: pytest.TempPathFactory,
    ops: list[
        tuple[
            str,
            str,
            str,
            dict[str, Any] | list[dict[str, Any]],
            str | None,
            str | None,
            dict[str, Any] | None,
            list[dict[str, Any]] | None,
        ]
    ],
) -> None:
    path = tmp_path_factory.mktemp("aligned") / "graph.json"
    gm = make_manager()
    for node_id in POOL:
        add_screen(gm, node_id, f"screen {node_id}", **members(path.parent / "steps"))
    for kind, a, b, action, instruction, observation, delta, setup in ops:
        if kind == "edge":
            gm.add_edge(
                a,
                b,
                action,
                details=EdgeDetails(
                    instruction=instruction,
                    target_observation=observation,
                    schema_delta=delta,
                    setup=setup,
                ),
            )
        else:
            gm.merge_nodes(a, b)
        assert misaligned_lists(gm) == []

    gm.save_graph(path)
    loaded = GraphManager()
    loaded.load_graph(path)

    assert misaligned_lists(loaded) == []
    assert per_action_lists(loaded) == per_action_lists(gm)


def test_load_graph_rejects_a_non_object_document_naming_the_path(tmp_path: Path) -> None:
    path = tmp_path / "graph.json"
    path.write_text("[]", encoding="utf-8")
    gm = make_manager()
    with pytest.raises(ValueError, match="must be an object") as excinfo:
        gm.load_graph(path)
    assert str(path) in str(excinfo.value)
