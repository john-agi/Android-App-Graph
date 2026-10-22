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

from android_app_graph.payloads import as_object_dict
from android_app_graph.step_log import Observation, StepRecorder, save_image_embeddings
from android_app_graph.utils import ModelSettings
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
# A library activity runs in the package of the app that declares it.
GOOGLE_SIGN_IN = "com.google.android.gms.auth.api.signin.internal.SignInHubActivity"

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


def make_manager(steps_dir: Path | None = None) -> GraphManager:
    """A manager whose client is a sentinel; every caller of it is patched."""
    gm = GraphManager(
        client=FAKE_CLIENT,
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


def members(steps_dir: Path, *images: bytes) -> Membership:
    """The membership fields of a node whose observations' images, ``b"screenshot-a"``
    unless given, are in ``steps_dir``; the first is the representative."""
    observation_ids = [
        write_observation_image(steps_dir, image) for image in images or (b"screenshot-a",)
    ]
    return {"observations": observation_ids, "representative_observation": observation_ids[0]}


def describe_members(steps_dir: Path, observations: list[str], description: str) -> None:
    """Give each of ``observations`` the ``describe`` output identify_state records for a
    screen it described as ``description``."""
    for observation_id in observations:
        path = steps_dir / f"{observation_id}.json"
        record = (
            as_object_dict(json.loads(path.read_text("utf-8")))
            if path.exists()
            else {"id": observation_id}
        )
        record["describe"] = {
            "page_description": description,
            "state": {},
            "elements": [],
            "is_external_web": False,
        }
        path.write_text(json.dumps(record), "utf-8")


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
        record = {"id": observation_id, "kind": "observation", "package": package}
        (self.steps_dir / f"{observation_id}.json").write_text(json.dumps(record), "utf-8")
        return Observation(
            observation_id, screenshot, activity, package, fake_pixel_hash(screenshot)
        )

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
    """Insert a node with the attribute set identify_state would create. In a manager
    with a step log, each member's record gets the describe output identify_state
    records, so ``embed_missing_members`` embeds ``description`` for it."""
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
    }
    attrs.update(overrides)
    gm.graph.add_node(node_id, **attrs)
    observations = attrs.get("observations")
    if gm.steps_dir is not None and isinstance(observations, list):
        describe_members(gm.steps_dir, [str(member) for member in observations], description)


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


def test_a_manager_without_embedding_settings_rejects_reading_its_model() -> None:
    gm = GraphManager(client=FAKE_CLIENT)
    with pytest.raises(RuntimeError, match="needs embedding settings"):
        _ = gm.embedding_model


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
        {"node": first, "member": home.observation_id, "same": True, "reason": "identical layout"}
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


def test_the_text_source_skips_a_member_without_a_text_vector_without_a_stale_warning(
    vlm: FakeVlm, caplog: pytest.LogCaptureFixture, screen: Screens
) -> None:
    """A member whose record holds no describe output has no description to embed."""
    vlm.descriptions.append(("Home screen", {}, []))
    vlm.verdicts.append({"same": True, "reason": "same page"})
    gm = make_manager()
    # Without a step log, add_screen leaves the member's record without a describe output.
    add_screen(gm, "s0_undescribed", "Home screen", **members(screen.steps_dir, b"undescribed"))
    gm.steps_dir = screen.steps_dir
    add_screen(gm, "s1_home", "Home feed", **members(screen.steps_dir))
    vlm.images[b64(b"undescribed")] = FAR
    gm.embed_missing_members()
    observation = screen(HOME, SHOT_A, PACKAGE)

    with caplog.at_level("WARNING"):
        node_id = gm.identify_state(observation)

    assert node_id == "s1_home"
    candidates = screen.record(observation)["identify"]["candidates"]
    assert [(c["node"], sorted(c["sources"])) for c in candidates] == [
        ("s1_home", ["image", "text"]),
        ("s0_undescribed", ["image"]),
    ]
    assert "stale" not in caplog.text


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
    text_vectors = tmp_path / "graph_text_emb.json"
    stored = json.loads(text_vectors.read_text("utf-8"))
    stored["embeddings"].update(
        {member: [1.0, 0.0] for member in gm.graph.nodes["s0_stale"]["observations"]}
    )
    text_vectors.write_text(json.dumps(stored), "utf-8")
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
    gm = make_manager(screen.steps_dir)
    add_screen(gm, "s0_settings", "Settings page", **members(screen.steps_dir, b"rep", b"twin"))
    vlm.images.update({b64(b"rep"): FAR, b64(b"twin"): NEAR, SHOT_A: NEAR})
    gm.embed_missing_members()
    vlm.calls.clear()
    vlm.descriptions.append(("Search results", {}, []))
    vlm.verdicts.append({"same": True, "reason": "a scrolled list"})
    observation = screen(HOME, SHOT_A, PACKAGE)

    assert gm.identify_state(observation) == "s0_settings"

    assert verified(vlm) == [b64(b"twin")]
    (candidate,) = screen.record(observation)["identify"]["candidates"]
    assert candidate["closest_member"] == gm.graph.nodes["s0_settings"]["observations"][1]
    assert candidate["sources"] == {"image": pytest.approx(1.0), "text": pytest.approx(0.0)}


def _three_image_matches_and_a_text_match(gm: GraphManager, vlm: FakeVlm, steps_dir: Path) -> None:
    """s0-s2 look like the screen and are described otherwise; s3 is described as it is
    and looks unlike it."""
    for number, image in enumerate((b"a", b"b", b"c")):
        add_screen(gm, f"s{number}_lookalike", "Settings page", **members(steps_dir, image))
        vlm.images[b64(image)] = NEAR
    add_screen(gm, "s3_results", "Search results", **members(steps_dir, b"d"))
    vlm.images[b64(b"d")] = FAR
    vlm.images[SHOT_A] = NEAR
    gm.embed_missing_members()
    vlm.calls.clear()
    vlm.descriptions.append(("Search results", {}, []))


def test_the_best_text_match_is_verified_within_the_cap_behind_three_image_matches(
    vlm: FakeVlm, screen: Screens
) -> None:
    gm = make_manager(screen.steps_dir)
    _three_image_matches_and_a_text_match(gm, vlm, screen.steps_dir)
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
    gm = make_manager(screen.steps_dir)
    _three_image_matches_and_a_text_match(gm, vlm, screen.steps_dir)
    vlm.verdicts.append({"same": True, "reason": "the same list"})

    assert gm.identify_state(screen(HOME, SHOT_A, PACKAGE)) == "s0_lookalike"

    assert vlm.kinds() == ["image", "describe", "embed", "verify"]


def test_the_short_list_makes_at_most_three_verifier_calls_and_refines_only_once(
    vlm: FakeVlm, screen: Screens
) -> None:
    gm = make_manager(screen.steps_dir)
    _three_image_matches_and_a_text_match(gm, vlm, screen.steps_dir)
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
    gm = make_manager(screen.steps_dir)
    _three_image_matches_and_a_text_match(gm, vlm, screen.steps_dir)
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
    gm = make_manager(screen.steps_dir)
    _three_image_matches_and_a_text_match(gm, vlm, screen.steps_dir)
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
    gm = make_manager(screen.steps_dir)
    _three_image_matches_and_a_text_match(gm, vlm, screen.steps_dir)
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


def test_an_exact_match_of_a_member_without_a_text_vector_records_none(
    vlm: FakeVlm, tmp_path: Path
) -> None:
    """A loaded graph's members have none until ``embed_missing_members`` embeds them."""
    steps_dir = tmp_path / "steps"
    member = members(steps_dir)
    (observation_id,) = member["observations"]
    record = {"pixel_hash": fake_pixel_hash(SHOT_A)}
    (steps_dir / f"{observation_id}.json").write_text(json.dumps(record), "utf-8")
    describe_members(steps_dir, [observation_id], "Home screen")
    path = tmp_path / "graph.json"
    path.write_text(
        json.dumps({"package_name": PACKAGE, "nodes": [{"id": "s0_home", **member}]}), "utf-8"
    )
    gm = make_manager()
    gm.load_graph(path)
    copy = Screens(steps_dir)(HOME, SHOT_A, PACKAGE)

    assert gm.identify_state(copy) == "s0_home"

    assert vlm.kinds() == ["image"]
    assert gm.text_vector(copy.observation_id) is None


@pytest.mark.parametrize(
    "describe",
    [
        pytest.param(None, id="none"),
        pytest.param({"page_description": 3, "state": {}, "elements": []}, id="description"),
        pytest.param({"page_description": "Home", "state": [], "elements": []}, id="state"),
        pytest.param({"page_description": "Home", "state": {}, "elements": {}}, id="elements"),
    ],
)
def test_a_member_whose_record_holds_no_usable_describe_output_is_never_an_exact_match(
    describe: dict[str, object] | None, vlm: FakeVlm, tmp_path: Path
) -> None:
    steps_dir = tmp_path / "steps"
    member = members(steps_dir)
    record: dict[str, object] = {"pixel_hash": fake_pixel_hash(SHOT_A)}
    if describe is not None:
        record["describe"] = describe
    (steps_dir / f"{member['representative_observation']}.json").write_text(
        json.dumps(record), "utf-8"
    )
    path = tmp_path / "graph.json"
    path.write_text(
        json.dumps(
            {
                "package_name": PACKAGE,
                "nodes": [{"id": "s0_home", "page_description": "Home screen", **member}],
            }
        ),
        "utf-8",
    )
    gm = make_manager()
    gm.load_graph(path)
    gm.embed_missing_members()
    vlm.calls.clear()
    vlm.descriptions.append(("Home screen", {}, []))
    vlm.verdicts.append({"same": True, "reason": "home"})

    gm.identify_state(Screens(steps_dir)(HOME, SHOT_A, PACKAGE))

    assert vlm.kinds()[:2] == ["image", "describe"]


def test_embed_missing_members_embeds_each_in_app_member_once_and_saves_the_vectors(
    vlm: FakeVlm, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    steps_dir = tmp_path / "steps"
    gm = make_manager(steps_dir)
    add_screen(gm, "s0_home", "Home screen", **members(steps_dir, b"home", b"home-2"))
    gm.record_external_app_visit(Screens(steps_dir)(OTHER_APP, SHOT_B, OTHER_PACKAGE))
    vlm.images.update({b64(b"home"): NEAR, b64(b"home-2"): FAR})

    with caplog.at_level(logging.INFO, logger=gm_module.logger.name):
        gm.embed_missing_members()
        gm.embed_missing_members()

    assert [call["screenshot"] for kind, call in vlm.calls if kind == "image"] == [
        b64(b"home"),
        b64(b"home-2"),
    ]
    stored = json.loads((steps_dir / "image_emb.json").read_text(encoding="utf-8"))
    assert stored["model"] == IMAGE_EMBEDDING.model
    home, home_2 = gm.graph.nodes["s0_home"]["observations"]
    assert stored["embeddings"] == {home: NEAR, home_2: FAR}
    assert "Embedding the images of 2 member observation(s)" in caplog.text
    assert "Embedded 2 member image(s) for $0.000000" in caplog.text


def test_described_members_without_a_text_vector_are_embedded_once_per_description(
    vlm: FakeVlm, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    steps_dir = tmp_path / "steps"
    member = members(steps_dir, b"home", b"home-2", b"feed")
    home, home_2, feed = member["observations"]
    describe_members(steps_dir, [home, home_2], "Home screen")
    describe_members(steps_dir, [feed], "Home feed")
    path = tmp_path / "graph.json"
    path.write_text(
        json.dumps({"package_name": PACKAGE, "nodes": [{"id": "s0_home", **member}]}), "utf-8"
    )
    gm = make_manager()
    gm.load_graph(path)

    with caplog.at_level(logging.INFO, logger=gm_module.logger.name):
        gm.embed_missing_members()
        gm.embed_missing_members()
    gm.save_graph(path)
    reloaded = make_manager()
    reloaded.load_graph(path)
    reloaded.embed_missing_members()

    assert [call["text"] for kind, call in vlm.calls if kind == "embed"] == [
        "Home screen",
        "Home feed",
    ]
    assert json.loads((tmp_path / "graph_text_emb.json").read_text("utf-8")) == {
        "model": EMBEDDING.model,
        "embeddings": {
            home: EMBEDDINGS["Home screen"],
            home_2: EMBEDDINGS["Home screen"],
            feed: EMBEDDINGS["Home feed"],
        },
    }
    assert "Embedding the descriptions of 3 member observation(s)" in caplog.text
    assert "Embedded 2 description(s) for $0.000000" in caplog.text


@pytest.mark.parametrize(
    ("copied_from", "copy_ended_with"),
    [
        pytest.param("refined", "Promo overlay", id="the refined member"),
        pytest.param("copy", "Home screen", id="itself"),
        pytest.param("000099", "Home screen", id="a member without a record"),
    ],
)
@pytest.mark.usefixtures("vlm")
def test_the_backfill_embeds_the_description_each_member_ended_with(
    copied_from: str, copy_ended_with: str, tmp_path: Path
) -> None:
    """The step's first rejection refined one member's description; the other is an
    exact copy, which ends with what the member it names ended with."""
    steps_dir = tmp_path / "steps"
    member = members(steps_dir, b"refined", b"copy")
    refined, copy = member["observations"]
    named = {"refined": refined, "copy": copy}.get(copied_from, copied_from)
    describe = {"page_description": "Home screen", "state": {}, "elements": []}
    records = {
        refined: {
            "describe": describe,
            "identify": {"refinement": {"new_description": "Promo overlay"}},
        },
        copy: {"describe": {**describe, "copied_from": named}, "identify": {"refinement": None}},
    }
    for observation_id, record in records.items():
        (steps_dir / f"{observation_id}.json").write_text(json.dumps(record), "utf-8")
    path = tmp_path / "graph.json"
    path.write_text(
        json.dumps({"package_name": PACKAGE, "nodes": [{"id": "s0_home", **member}]}), "utf-8"
    )
    gm = make_manager()
    gm.load_graph(path)

    gm.embed_missing_members()

    assert gm.text_vector(refined) == EMBEDDINGS["Promo overlay"]
    assert gm.text_vector(copy) == EMBEDDINGS[copy_ended_with]


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
    path.write_text(json.dumps({"nodes": [{"id": "s0_home", **member}]}), "utf-8")
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
        json.dumps({"package_name": PACKAGE, "nodes": [{"id": "s0_home", **member}]}), "utf-8"
    )
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
    ranked = gm.short_list(query, ["s0_text", "s1_image"])

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


def test_add_edge_pads_an_edge_whose_lists_are_missing_before_appending() -> None:
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen")
    gm.graph.add_edge("s0_home", "s0_home", actions=[{"action": "tap"}, ["a", "b"]])

    gm.add_edge(
        "s0_home", "s0_home", {"action": "swipe"}, details=EdgeDetails(instruction="scroll")
    )

    data = gm.graph["s0_home"]["s0_home"]
    assert data["num_steps"] == [1, 2, 1]
    assert data["instructions"] == ["", "", "scroll"]
    assert data["target_observations"] == ["", "", ""]
    assert data["schema_deltas"] == [None, None, None]


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
    if first is None:
        gm.graph.add_edge("s0_home", "s1_settings", actions=[_TAP], visit_count=1)
    else:
        gm.add_edge("s0_home", "s1_settings", _TAP, details=EdgeDetails(setup=first))

    gm.add_edge("s0_home", "s1_settings", _TAP, details=EdgeDetails(setup=again))

    data = gm.graph["s0_home"]["s1_settings"]
    assert (data["actions"], data["setups"], data["visit_count"]) == ([_TAP], [kept], 2)


def _identity(node: str) -> str:
    return node


def test_a_setup_trail_starts_an_arrival_s_setup_empty_and_extends_it_with_self_loops() -> None:
    trail = SetupTrail(_identity)
    trail.begin("s0_home", arrived=True)
    assert trail.setup("s0_home") == []

    trail.step([_SCROLL], "s0_home")
    assert trail.setup("s0_home") == [_SCROLL]

    trail.step([_TAP], "s1_list")
    assert trail.setup("s1_list") == []


def test_a_round_trip_joins_the_setup_of_the_node_it_returns_to() -> None:
    """The nodes the round trip passed through are left behind: arriving at one of them
    again starts its setup afresh."""
    open_menu, choose, home = {"action": "tap", "x": 1}, {"action": "tap", "x": 2}, _TAP
    trail = SetupTrail(_identity)
    trail.begin("s0_home", arrived=True)
    trail.step([_TAP], "s1_form")
    trail.step([open_menu], "s2_menu")
    trail.step([choose], "s1_form")
    assert trail.setup("s1_form") == [open_menu, choose]

    trail.step([open_menu], "s2_menu")
    assert trail.setup("s2_menu") == []

    trail.step([home], "s0_home")
    assert trail.setup("s0_home") == [_TAP, open_menu, choose, open_menu, home]


def test_a_setup_trail_begun_without_an_arrival_knows_no_setup_until_it_arrives_elsewhere() -> None:
    trail = SetupTrail(_identity)
    trail.begin("s0_home", arrived=False)
    trail.step([_SCROLL], "s0_home")
    assert trail.setup("s0_home") is None

    trail.step([_TAP], "s1_list")
    assert trail.setup("s1_list") == []

    trail.step([{"action": "back"}], "s0_home")
    assert trail.setup("s0_home") is None
    assert trail.setup("s9_never_reached") is None


def test_a_setup_trail_matches_a_node_merged_away_while_it_holds_it() -> None:
    gm = make_manager()
    for node_id in ("s0_home", "s1_list", "s2_detail", "s5_list"):
        add_screen(gm, node_id, f"screen {node_id}")
    trail = SetupTrail(gm.resolve_node_id)
    trail.begin("s0_home", arrived=True)
    trail.step([_TAP], "s1_list")
    trail.step([_SCROLL], "s1_list")

    assert gm.merge_nodes("s5_list", "s1_list") is True
    assert trail.setup("s5_list") == [_SCROLL]

    trail.step([_TAP], "s2_detail")
    trail.step([{"action": "back"}], "s5_list")
    assert trail.setup("s5_list") == [_SCROLL, _TAP, {"action": "back"}]


def test_a_graph_file_without_setups_loads_replays_without_them_and_saves_none(
    tmp_path: Path, write_graph_file: Callable[[Path, dict[str, Any]], None]
) -> None:
    path = tmp_path / "explored.json"
    write_graph_file(
        path,
        {
            "nodes": [{"id": "s0_home"}, {"id": "s1_settings"}],
            "edges": [
                {
                    "source": "s0_home",
                    "target": "s1_settings",
                    "actions": [[_TAP]],
                    "num_steps": [1],
                }
            ],
        },
    )
    gm = GraphManager()
    gm.load_graph(path)

    assert gm.find_path("s0_home", "s1_settings") == [
        PathStep("s0_home", {}, ()),
        PathStep("s1_settings", [_TAP], ()),
    ]
    again = tmp_path / "saved" / "explored.json"
    gm.save_graph(again)
    assert "setups" not in json.loads(again.read_text(encoding="utf-8"))["edges"][0]


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


@pytest.mark.parametrize("bare_side", ["kept", "removed"])
def test_merge_nodes_pads_the_side_whose_edge_lacks_a_list(bare_side: str) -> None:
    gm = make_manager()
    add_screen(gm, "s0_keep", "Home screen")
    add_screen(gm, "s1_remove", "Home screen again")
    bare, full = ("s0_keep", "s1_remove") if bare_side == "kept" else ("s1_remove", "s0_keep")
    gm.graph.add_edge(bare, bare, actions=[["tap", "type"]], visit_count=1)
    gm.add_edge(
        full,
        full,
        {"action": "sort"},
        details=EdgeDetails(instruction="sort", schema_delta={"sort": "price"}, setup=[]),
    )

    assert gm.merge_nodes("s0_keep", "s1_remove") is True

    data = gm.graph["s0_keep"]["s0_keep"]
    keys = ("actions", *PER_ACTION_LISTS)
    padded = (["tap", "type"], 2, "", "", None, None)
    recorded = ({"action": "sort"}, 1, "sort", "", {"sort": "price"}, [])
    rows = list(zip(*(data[key] for key in keys), strict=True))
    assert rows == ([padded, recorded] if bare_side == "kept" else [recorded, padded])


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
    """A graph loaded from JSON can hold more actions than instructions: the actions
    past the last instruction match nothing, and the length mismatch is not an error.
    """
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
        details=EdgeDetails(instruction="open search"),
    )
    gm.graph["s0_home"]["s1_settings"]["instructions"] = ["open settings"]

    assert gm.find_edge_action("s0_home", "s1_settings", "open settings") == (0, {"action": "tap"})
    assert gm.find_edge_action("s0_home", "s1_settings", "open search") is None


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
        details=EdgeDetails(instruction='search for "shoes"', target_observation="shoe results"),
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
    gm.normalize_all_edges()  # two examples cover the two recorded instructions

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


def test_find_path_uses_an_empty_action_for_an_edge_without_actions() -> None:
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen")
    add_screen(gm, "s1_settings", "Settings page")
    gm.graph.add_edge("s0_home", "s1_settings")

    assert gm.find_path("s0_home", "s1_settings") == [
        PathStep("s0_home", {}, ()),
        PathStep("s1_settings", {}, ()),
    ]


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


def test_find_guided_path_navigates_an_edge_from_a_file_without_text_lists() -> None:
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen")
    add_screen(gm, "s1_settings", "Settings page")
    gm.graph.add_edge("s0_home", "s1_settings", actions=[{"action": "tap"}], num_steps=[1])

    steps = gm.find_guided_path("s0_home", [{"node_id": "s1_settings"}])

    assert steps is not None
    assert (steps[0]["instruction"], steps[0]["target_observation"]) == ("", "")


@pytest.mark.parametrize("source", ["add_edge", "a file without the list"])
def test_find_guided_path_ignores_a_self_loop_without_schema_deltas(source: str) -> None:
    gm = make_manager()
    add_screen(gm, "s0_home", "Home screen")
    if source == "add_edge":
        gm.add_edge("s0_home", "s0_home", {"action": "sort"})
    else:
        gm.graph.add_edge("s0_home", "s0_home", actions=[{"action": "sort"}])

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

    gm.save_graph(path)
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


def test_load_graph_defaults_the_activities_and_the_step_count(tmp_path: Path) -> None:
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
                        **members(tmp_path / "steps"),
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
    assert node["activities"] == [HOME]
    assert gm.total_steps_completed == 0


def test_save_graph_and_load_graph_round_trip_the_package_name(tmp_path: Path) -> None:
    source = make_manager()
    source.package_name = "com.example.app"
    path = tmp_path / "graph.json"
    source.save_graph(path)

    target = make_manager()
    target.load_graph(path)

    assert json.loads(path.read_text(encoding="utf-8"))["package_name"] == "com.example.app"
    assert target.package_name == "com.example.app"


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
    source.save_graph(path)

    target = make_manager()
    target.load_graph(path)

    assert dict(target.graph.nodes(data="package")) == {"s0_list": MESSAGES, "s1_mail": GMAIL}


@pytest.mark.parametrize(
    ("saved_package_name", "configured_package_name"),
    [(WHATSAPP, PACKAGE), (None, WHATSAPP)],
    ids=["saved", "configured"],
)
def test_load_graph_gives_the_app_package_to_each_screen_saved_without_one(
    tmp_path: Path, saved_package_name: str | None, configured_package_name: str
) -> None:
    """Every node saved without a package but the ``ext_`` placeholders gets the app's
    package, including one whose activity class lives outside the package."""
    steps_dir = tmp_path / "steps"
    document: dict[str, Any] = {
        "nodes": [
            {"id": "s0_home", "activity": WHATSAPP_HOME, **members(steps_dir)},
            {"id": "s1_group_info", "activity": WHATSAPP_GROUP_INFO, **members(steps_dir)},
            {"id": "s2_sign_in", "activity": GOOGLE_SIGN_IN, **members(steps_dir)},
            {"id": "ext_com.google.android.gm", "activity": GMAIL_LIST, **members(steps_dir)},
            {"id": EXTERNAL_WEB_NODE, **members(steps_dir)},
        ],
        "edges": [
            *(
                {"source": "s0_home", "target": target}
                for target in ("s1_group_info", "s2_sign_in", "ext_com.google.android.gm")
            ),
            {"source": "s2_sign_in", "target": "s0_home"},
        ],
    }
    if saved_package_name is not None:
        document["package_name"] = saved_package_name
    path = tmp_path / "graph.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    gm = make_manager()

    gm.load_graph(path, default_package_name=configured_package_name)

    assert dict(gm.graph.nodes(data="package")) == {
        "s0_home": WHATSAPP,
        "s1_group_info": WHATSAPP,
        "s2_sign_in": WHATSAPP,
        "ext_com.google.android.gm": "",
        EXTERNAL_WEB_NODE: "",
    }
    assert gm.get_resume_target(package_name=WHATSAPP) == "s1_group_info"


def test_load_graph_replaces_the_current_graph(tmp_path: Path) -> None:
    source = make_manager()
    add_screen(source, "s0_home", "Home screen", **members(tmp_path / "steps"))
    path = tmp_path / "graph.json"
    source.save_graph(path)

    target = make_manager()
    add_screen(target, "s9_stale", "Stale screen")
    target.load_graph(path)

    assert list(target.graph.nodes) == ["s0_home"]


def test_load_graph_accepts_a_file_with_only_nodes(tmp_path: Path) -> None:
    """An absent "nodes" or "edges" key is an empty list, so a file without an
    "edges" key loads; only a present non-list value is rejected.
    """
    path = tmp_path / "graph.json"
    node = {
        "id": "s0_home",
        "activity": HOME,
        "page_description": "Home screen",
        **members(tmp_path / "steps"),
    }
    path.write_text(json.dumps({"nodes": [node]}), encoding="utf-8")
    gm = make_manager()

    gm.load_graph(path)

    assert list(gm.graph.nodes) == ["s0_home"]
    assert gm.graph.number_of_edges() == 0


def test_an_audited_graph_shares_the_step_log_of_the_graph_it_was_derived_from(
    tmp_path: Path,
) -> None:
    app_dir = tmp_path / "demo"
    node = {"id": "s0_home", "activity": HOME, **members(app_dir / "steps", b"home-image")}
    path = app_dir / "demo_audited.json"
    path.write_text(json.dumps({"nodes": [node], "edges": []}), encoding="utf-8")
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
            {"id": "s0_home", **members(tmp_path / "steps")},
            {"id": "s1_settings", **membership},
        ],
    )
    gm = make_manager()

    with pytest.raises(ValueError, match="no member observations") as excinfo:
        gm.load_graph(path)

    message = str(excinfo.value)
    assert "['s1_settings']" in message
    assert "s0_home" not in message
    assert str(path) in message
    assert "explore the app again" in message


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


def test_load_graph_names_every_node_it_rejects_and_keeps_the_loaded_graph(
    tmp_path: Path,
) -> None:
    previous = tmp_path / "previous" / "graph.json"
    previous.parent.mkdir()
    write_graph(previous, [{"id": "s9_stale", **members(previous.parent / "steps")}])
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
    previous.write_text(
        json.dumps(
            {
                "next_id": 7,
                "nodes": [{"id": "s9_stale", **members(tmp_path / "steps")}],
                "edges": [],
            }
        ),
        encoding="utf-8",
    )
    gm = make_manager()
    gm.load_graph(previous)

    with pytest.raises(TypeError, match="node id must be a string") as excinfo:
        gm.load_graph(path)
    assert str(path) in str(excinfo.value)

    kept = tmp_path / "kept.json"
    gm.save_graph(kept)
    saved = json.loads(kept.read_text(encoding="utf-8"))
    assert saved["next_id"] == 7
    assert [node["id"] for node in saved["nodes"]] == ["s9_stale"]


def test_load_graph_rejects_a_null_nodes_field_naming_the_path(tmp_path: Path) -> None:
    """``"nodes": null`` raises a ``TypeError`` naming the path from ``load_graph``, which
    has no shape check of its own and relies entirely on the shared validator.
    """
    path = tmp_path / "graph.json"
    path.write_text(json.dumps({"nodes": None, "edges": []}), encoding="utf-8")
    gm = make_manager()

    with pytest.raises(TypeError, match="must contain list fields 'nodes' and 'edges'") as excinfo:
        gm.load_graph(path)
    assert str(path) in str(excinfo.value)


def test_load_graph_rejects_an_int_node_id_naming_the_path(tmp_path: Path) -> None:
    """An int id would otherwise sail through ``require_known_edge_endpoints``,
    which compares ``str()``-normalised ids, and let ``add_edge`` grow a
    phantom node with a different (string) id -- the same phantom node
    ``require_known_edge_endpoints`` keeps a genuinely undefined endpoint from growing.
    """
    path = tmp_path / "graph.json"
    path.write_text(json.dumps({"nodes": [{"id": 5}], "edges": []}), encoding="utf-8")
    gm = make_manager()

    with pytest.raises(TypeError, match="node id must be a string") as excinfo:
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

    with pytest.raises(TypeError, match="edge endpoint must be a string") as excinfo:
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
    assert result["results"][0]["removed"] == "s2_home_two"
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
    }
)
node_attrs = st.tuples(screen_attrs, memberships).map(lambda parts: {**parts[0], **parts[1]})
edge_attrs = st.fixed_dictionaries(
    {
        "actions": st.lists(actions | st.lists(actions, max_size=3), max_size=3),
        "instructions": st.lists(st.text(max_size=12), max_size=3),
        "instruction_templates": small_dicts,
        "target_observations": st.lists(st.text(max_size=12), max_size=3),
        "num_steps": st.lists(st.integers(min_value=1, max_value=5), max_size=3),
        "visit_count": st.integers(min_value=0, max_value=20),
        "schema_deltas": small_dicts,
        "setups": st.lists(setup_entries, max_size=3),
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
            "last_normalized_visit_milestone": d.get("last_normalized_visit_milestone", 0),
            "visit_count": d.get("visit_count", 0),
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
            "schema_deltas": d.get("schema_deltas") or None,
            "setups": d.get("setups") or None,
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
    (path.parent / "steps").mkdir()
    for attrs in nodes.values():
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


PER_ACTION_LISTS = ("num_steps", "instructions", "target_observations", "schema_deltas", "setups")

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
    with pytest.raises(TypeError, match="must be an object") as excinfo:
        gm.load_graph(path)
    assert str(path) in str(excinfo.value)


def test_the_description_backfill_logs_what_its_requests_cost(
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    embedding_body: dict[str, object],
    write_image_vectors: Callable[..., None],
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    steps_dir = tmp_path / "steps"
    member = members(steps_dir)
    describe_members(steps_dir, member["observations"], "Home screen")
    write_image_vectors(steps_dir, member["observations"], IMAGE_EMBEDDING.model)
    path = tmp_path / "graph.json"
    path.write_text(
        json.dumps({"package_name": PACKAGE, "nodes": [{"id": "s0_home", **member}]}), "utf-8"
    )
    manager = GraphManager(
        client=router_client,
        page_detail=PAGE_DETAIL,
        embedding=EMBEDDING,
        image_embedding=IMAGE_EMBEDDING,
    )
    manager.load_graph(path)
    router_responses.append(httpx.Response(200, json=embedding_body))

    with caplog.at_level(logging.INFO, logger=gm_module.logger.name):
        manager.embed_missing_members()

    assert manager.text_vector(member["representative_observation"]) == [0.25, 0.75]
    assert "Embedded 1 description(s) for $0.000010" in caplog.text


def test_text_vectors_of_another_embedding_model_are_embedded_again(
    vlm: FakeVlm, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    steps_dir = tmp_path / "steps"
    gm = make_manager(steps_dir)
    add_screen(gm, "s0_home", "Home screen", **members(steps_dir))
    gm.embed_missing_members()
    path = tmp_path / "graph.json"
    gm.save_graph(path)
    vlm.calls.clear()
    other_model = ModelSettings("provider/other-embedding")
    reloaded = GraphManager(
        client=FAKE_CLIENT,
        page_detail=PAGE_DETAIL,
        embedding=other_model,
        image_embedding=IMAGE_EMBEDDING,
    )

    with caplog.at_level(logging.WARNING):
        reloaded.load_graph(path)
    reloaded.embed_missing_members()

    assert [(kind, call["model"]) for kind, call in vlm.calls] == [("embed", other_model.model)]
    assert f"was written by model {EMBEDDING.model!r}" in caplog.text
    assert "graph_text_emb.json" in caplog.text
