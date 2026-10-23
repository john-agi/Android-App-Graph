"""Shared pytest configuration: Hypothesis profiles, credential and network guards,
and fixtures that answer OpenRouter SDK requests from an offline mock transport."""

from __future__ import annotations

import base64
import io
import json
import os
import socket
import subprocess
import time
from collections.abc import Callable, Generator, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, NoReturn

import httpx
import pytest
from hypothesis import settings
from openrouter import OpenRouter
from PIL import Image

from android_app_graph import cli, device
from android_app_graph.commands import audit
from android_app_graph.device import DeviceController
from android_app_graph.graph_files import (
    append_json_line,
    graph_steps_dir,
    identify_file_path,
    observation_image_path,
    save_embeddings,
    text_embeddings_path,
)
from android_app_graph.screen_image import encode_screenshot
from android_app_graph.step_log import save_image_embeddings
from android_app_graph.utils import DEFAULT_MATCHING, open_client
from android_app_graph.utils.graph_manager import GraphManager

# Hypothesis's built-in "ci" profile, its own choice when CI is set, is deterministic,
# keeps no example database and has no deadline; CI runs it with more examples.
# Local runs have no deadline either: how long one example takes depends on how busy
# the machine is, not on the code under test.
settings.register_profile("ci", settings.get_profile("ci"), max_examples=500)
settings.register_profile("dev", settings.get_profile("default"), deadline=None)
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "ci" if os.getenv("CI") else "dev"))

# The OpenRouter SDK also reads OPENROUTER_HTTP_REFERER, OPENROUTER_X_OPEN_ROUTER_* and
# OPENROUTER_DEBUG, so the whole prefix is cleared, not only the key.
_CREDENTIAL_PREFIXES = ("APP_GRAPH_", "OPENROUTER_")
_CREDENTIAL_NAMES = frozenset({"GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"})


@pytest.fixture(autouse=True)
def _clear_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove provider credentials and SDK settings so tests never use real keys or endpoints."""
    for name in list(os.environ):
        if name.startswith(_CREDENTIAL_PREFIXES) or name in _CREDENTIAL_NAMES:
            monkeypatch.delenv(name)


def _deny_network(*_args: object, **_kwargs: object) -> NoReturn:
    msg = "network access is disabled in the test suite"
    raise RuntimeError(msg)


@pytest.fixture(autouse=True, scope="session")
def _no_network() -> Iterator[None]:
    """Run the whole session with outbound networking disabled."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(socket, "getaddrinfo", _deny_network)
        mp.setattr(socket, "create_connection", _deny_network)
        mp.setattr(socket.socket, "connect", _deny_network)
        mp.setattr(socket.socket, "connect_ex", _deny_network)
        yield


@pytest.fixture
def model_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit model choices for tests of behavior other than missing configuration."""
    for role, model in (
        ("PAGE_DETAIL", "openai/gpt-6.1-sol"),
        ("EMBEDDING", "google/gemini-embedding-2"),
        ("INSTRUCTION", "openai/gpt-6.1-sol"),
        ("ACTION", "qwen/qwen3.8-flash"),
    ):
        monkeypatch.setenv(f"APP_GRAPH_{role}_MODEL", model)


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record sleeps instead of waiting for them.

    The OpenRouter SDK's retry back-off and the explorer's settling pauses call
    ``time.sleep`` on the shared stdlib module, so patching it here covers them
    all. A test must never let the SDK keep failing on a real clock: it retries
    5XX responses and timeouts until twice the request's timeout has passed on
    ``time.time``, which no patched sleep moves on.
    """
    delays: list[float] = []
    monkeypatch.setattr(time, "sleep", delays.append)
    return delays


@pytest.fixture
def adb_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve adb to the bare name ``adb``, so a test that replaces ``subprocess.run``
    neither needs adb installed nor depends on where it is.
    """
    monkeypatch.setattr(device, "adb_executable", lambda: "adb")


def _fake_adb_stdout(command: list[str]) -> str:
    """Name ``<package>/.Home`` as the launcher activity ``pm resolve-activity`` finds, in
    the layout of its ``--brief`` output; every other command prints nothing."""
    if "resolve-activity" not in command:
        return ""
    return (
        "priority=0 preferredOrder=0 match=0x108000 specificIndex=-1 isDefault=true\n"
        f"{command[-1]}/.Home\n"
    )


@pytest.fixture
def device_log(monkeypatch: pytest.MonkeyPatch, adb_on_path: None) -> list[object]:
    """Fake controllers append their actions here too, so a test sees adb commands
    and device actions in the order the device would have received them.

    Every adb command succeeds; the app's launcher activity is its ``.Home``. Every
    command must name a timeout, as a command without one blocks for as long as the
    device does not answer.
    """
    del adb_on_path  # requested for its side effect on device.adb_executable
    log: list[object] = []

    def run(
        command: list[str], *, timeout: float | None = None, **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        assert timeout is not None, f"adb call without a timeout: {command}"
        log.append(command)
        return subprocess.CompletedProcess(command, 0, stdout=_fake_adb_stdout(command), stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    return log


class _ScreensByLog:
    """A DeviceController that shows the next screen after each entry in ``log``.

    Its actions share the log with the adb commands ``device_log`` captures, so an app
    restart moves the device on too.
    """

    w = 100
    h = 200

    def __init__(self, log: list[object], *screens: dict[str, Any]) -> None:
        self.config: dict[str, Any] = {"device": {"udid": "test-device"}}
        self._log = log
        self._screens = screens

    def get_state(self) -> dict[str, Any]:
        return self._screens[min(len(self._log), len(self._screens) - 1)]

    def exe_action(self, action: dict[str, Any]) -> None:
        self._log.append(action)


@pytest.fixture
def screens_by_log(device_log: list[object]) -> Callable[..., DeviceController]:
    """Build a DeviceController that shows the given screens in turn, one more for each
    entry in ``device_log``, and stays on the last."""

    def build(*screens: dict[str, Any]) -> DeviceController:
        return _ScreensByLog(device_log, *screens)

    return build


@pytest.fixture
def router_requests() -> list[httpx.Request]:
    return []


@pytest.fixture
def router_responses() -> list[httpx.Response]:
    return []


@pytest.fixture
def router_http(
    router_requests: list[httpx.Request], router_responses: list[httpx.Response]
) -> Iterator[httpx.Client]:
    def respond(request: httpx.Request) -> httpx.Response:
        router_requests.append(request)
        return router_responses.pop(0)

    with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:
        yield http_client


@pytest.fixture
def router_client(router_http: httpx.Client) -> Iterator[OpenRouter]:
    with open_client("test-router-key", http_client=router_http) as client:
        yield client


@pytest.fixture
def opened_clients(router_http: httpx.Client, monkeypatch: pytest.MonkeyPatch) -> list[OpenRouter]:
    """Every client a command opens, each over the mock transport."""
    opened: list[OpenRouter] = []

    @contextmanager
    def open_routed(api_key: str) -> Generator[OpenRouter]:
        with open_client(api_key, http_client=router_http) as client:
            opened.append(client)
            yield client

    for module in (cli, audit):
        monkeypatch.setattr(module, "open_client", open_routed)
    return opened


@pytest.fixture
def timed_out_requests() -> list[httpx.Request]:
    return []


@pytest.fixture
def timing_out_client(
    monkeypatch: pytest.MonkeyPatch, no_sleep: list[float], timed_out_requests: list[httpx.Request]
) -> Iterator[OpenRouter]:
    """A client whose every attempt times out, a second past the default 60 s timeout, and
    lands in ``timed_out_requests``.

    Time passes only on the clock the SDK's retry loop reads, so it gives up after
    two attempts instead of retrying without end on a real clock.
    """
    del no_sleep  # requested for its side effect on time.sleep

    def time_out(request: httpx.Request) -> httpx.Response:
        timed_out_requests.append(request)
        msg = "no complete response within 60 s"
        raise httpx.ReadTimeout(msg, request=request)

    monkeypatch.setattr(time, "time", lambda: 61.0 * len(timed_out_requests))
    with (
        httpx.Client(transport=httpx.MockTransport(time_out)) as http_client,
        open_client("test-key", http_client=http_client) as client,
    ):
        yield client


@pytest.fixture
def chat_body() -> dict[str, object]:
    return {
        "id": "chat-test",
        "created": 0,
        "model": "test-model",
        "object": "chat.completion",
        "system_fingerprint": None,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "test reply",
                },
            }
        ],
        "usage": {
            "prompt_tokens": 7,
            "completion_tokens": 3,
            "total_tokens": 10,
            "cost": 0.00042,
            "prompt_tokens_details": {"cached_tokens": 2},
            "completion_tokens_details": {"reasoning_tokens": 1},
        },
    }


@pytest.fixture
def chat_response(chat_body: dict[str, object]) -> Callable[..., httpx.Response]:
    """Build ``chat_body`` with one assistant message made of the given fields.

    ``tool_arguments`` adds a native tool call carrying that JSON; ``content``
    left out is omitted from the message rather than sent as null.
    """

    def build(
        *, tool_arguments: str | None = None, finish_reason: str = "stop", **message: object
    ) -> httpx.Response:
        if tool_arguments is not None:
            message["tool_calls"] = [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "mobile_use", "arguments": tool_arguments},
                }
            ]
        choice = {
            "index": 0,
            "finish_reason": finish_reason,
            "message": {"role": "assistant", **message},
        }
        return httpx.Response(200, json={**chat_body, "choices": [choice]})

    return build


@pytest.fixture
def chat_usage_totals() -> dict[str, float]:
    """What one ``chat_body`` reply adds to a TokenTracker."""
    return {
        "calls": 1,
        "prompt_tokens": 7,
        "cached_prompt_tokens": 2,
        "completion_tokens": 3,
        "reasoning_tokens": 1,
        "total_tokens": 10,
        "cost": 0.00042,
    }


@pytest.fixture
def embedding_usage_totals() -> dict[str, float]:
    """What one ``embedding_body`` reply adds to a TokenTracker."""
    return {
        "calls": 1,
        "prompt_tokens": 5,
        "cached_prompt_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 5,
        "cost": 0.00001,
    }


@pytest.fixture
def embedding_body() -> dict[str, object]:
    return {
        "object": "list",
        "model": "google/gemini-embedding-2",
        "data": [{"object": "embedding", "index": 0, "embedding": [0.25, 0.75]}],
        "usage": {"prompt_tokens": 5, "total_tokens": 5, "cost": 0.00001},
    }


def _png_b64(color: str) -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), color).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


@pytest.fixture
def png_screenshot() -> str:
    return _png_b64("white")


@pytest.fixture
def write_image_vectors() -> Callable[..., None]:
    """Give observations of a step log image vectors from ``model``, by default the one
    ``model_environment`` names, so loading the graph embeds none of them.

    The vectors are orthogonal, so no two of the screens look alike to the audit.
    """

    def write(
        steps_dir: Path, observation_ids: list[str], model: str = "google/gemini-embedding-2"
    ) -> None:
        size = max(2, len(observation_ids))
        vectors = {
            observation_id: [float(index == position) for position in range(size)]
            for index, observation_id in enumerate(observation_ids)
        }
        save_image_embeddings(steps_dir, vectors, model=model)

    return write


@pytest.fixture
def add_member(png_screenshot: str) -> Callable[..., None]:
    """Give a node of a manager one member observation, with an image and a record in the
    manager's step log and a describe output in its identify file; ``image`` defaults to
    ``png_screenshot`` as WebP.

    The observation ID is not a step number, so a StepRecorder on the same step log
    never reuses it.
    """

    def add(graph: GraphManager, node_id: str, image: bytes | None = None) -> None:
        steps_dir = graph.steps_dir
        assert steps_dir is not None, "set the manager's steps_dir first"
        assert graph.identify_file is not None, "set the manager's identify_file first"
        observation_id = f"{node_id}-observation"
        steps_dir.mkdir(parents=True, exist_ok=True)
        observation_image_path(steps_dir, observation_id).write_bytes(
            encode_screenshot(png_screenshot) if image is None else image
        )
        _write_member_record(steps_dir, graph.identify_file, observation_id)
        graph.graph.nodes[node_id]["observations"] = [observation_id]

    return add


def _write_member_record(steps_dir: Path, identify_file: Path, observation_id: str) -> None:
    """Write what exploration leaves for a member observation: its record with its pixel
    hash in ``steps_dir``, and its describe output in ``identify_file``, which describes it
    as ``Screen <observation_id>``."""
    record = {"id": observation_id, "kind": "observation", "pixel_hash": f"hash-{observation_id}"}
    (steps_dir / f"{observation_id}.json").write_text(json.dumps(record), "utf-8")
    describe = {"page_description": f"Screen {observation_id}", "state": {}, "elements": []}
    line = {"observation": observation_id, "describe": {**describe, "is_external_web": False}}
    append_json_line(identify_file, line)


def _saved_edge(edge: dict[str, Any]) -> dict[str, Any]:
    """Return ``edge`` with the per-action lists ``add_edge`` records for its actions, for
    each list it does not hold."""
    actions = edge.get("actions", [])
    recorded = {
        "num_steps": [len(action) if isinstance(action, list) else 1 for action in actions],
        "instructions": [""] * len(actions),
        "target_observations": [""] * len(actions),
        "schema_deltas": [None] * len(actions),
        "setups": [None] * len(actions),
    }
    return {"actions": actions, **recorded, **edge}


@pytest.fixture
def write_graph_file(png_screenshot: str) -> Callable[..., None]:
    """Write ``document`` as a graph file that ``load_graph`` accepts, with what exploration
    records where ``document`` gives none: the default matching settings, every node but an
    ``ext_`` one gets the package ``com.demo``, every node gets a ``replay_misses`` of 0
    and one member observation, with an image and a record holding a pixel hash in the
    step log beside it and a describe output in the identify file beside the graph, every
    member of a node but an ``ext_`` one gets a text vector in the file beside the graph,
    tagged with ``embedding_model``, and every edge gets the per-action lists ``add_edge``
    records."""

    def write(
        path: Path, document: dict[str, Any], embedding_model: str = "google/gemini-embedding-2"
    ) -> None:
        steps_dir = graph_steps_dir(path)
        steps_dir.mkdir(parents=True, exist_ok=True)
        nodes: list[dict[str, Any]] = []
        text_vectors: dict[str, list[float]] = {}
        for node in document.get("nodes", []):
            observation_id = f"{node['id']}-observation"
            observation_image_path(steps_dir, observation_id).write_bytes(
                encode_screenshot(png_screenshot)
            )
            _write_member_record(steps_dir, identify_file_path(path), observation_id)
            in_app = not node["id"].startswith("ext_")
            if in_app:
                text_vectors[observation_id] = [1.0, 0.0]
            recorded = {
                "package": "com.demo" if in_app else "",
                "replay_misses": 0,
                "observations": [observation_id],
            }
            nodes.append(recorded | node)
        edges = [_saved_edge(edge) for edge in document.get("edges", [])]
        save_embeddings(text_embeddings_path(path), text_vectors, model=embedding_model)
        saved = {"matching": DEFAULT_MATCHING._asdict(), **document, "nodes": nodes, "edges": edges}
        path.write_text(json.dumps(saved), "utf-8")

    return write


type Members = dict[str, list[tuple[str, list[float]]]]


@pytest.fixture
def write_member_graph() -> Callable[..., None]:
    """Write a graph file of ``com.demo`` nodes, made under the default matching settings,
    whose members have an image and an image vector in the step log beside it and a
    stored describe output and a text vector in the files beside the graph, and the
    exploration transitions ``(before, after, action, from_node, to_node)``.

    ``nodes`` maps each node to its members, ``(observation_id, vector)`` in step order;
    the vector is the member's image vector and its text vector, unless ``texts`` gives
    the member a text vector of its own. Member ``m`` is described as ``Screen m``, or as
    ``described`` gives.
    ``refined`` maps a member to the description its step's first rejection refined it
    to, and ``copied`` maps a member to the member an exact match copied its describe
    output from. ``black`` maps each member whose screenshot is black to its activity;
    as screen matching records a black screenshot, its record holds no describe output
    and it has no text vector. Each image holds its observation ID, so a fake verifier
    can tell which members it was shown. The vectors are tagged with the model
    ``model_environment`` names.
    """

    def write(
        path: Path,
        nodes: Members,
        *,
        edges: tuple[dict[str, object], ...] = (),
        transitions: tuple[tuple[str, str, object, str, str], ...] = (),
        black: Mapping[str, str] | None = None,
        described: Mapping[str, str] | None = None,
        refined: Mapping[str, str] | None = None,
        copied: Mapping[str, str] | None = None,
        texts: Mapping[str, list[float]] | None = None,
    ) -> None:
        black_members = dict(black or {})
        text_of = dict(texts or {})
        descriptions = dict(described or {})
        refinements = dict(refined or {})
        copies = dict(copied or {})
        steps_dir = graph_steps_dir(path)
        steps_dir.mkdir(parents=True, exist_ok=True)
        image_vectors: dict[str, list[float]] = {}
        text_vectors: dict[str, list[float]] = {}
        for members in nodes.values():
            for observation_id, vector in members:
                image_vectors[observation_id] = vector
                observation_image_path(steps_dir, observation_id).write_bytes(
                    observation_id.encode("ascii")
                )
                black_activity = black_members.get(observation_id)
                record: dict[str, object] = {
                    "id": observation_id,
                    "kind": "observation",
                    "run_kind": "explore",
                    "activity": black_activity or f"com.demo.Activity{observation_id}",
                    "package": "com.demo",
                    "pixel_hash": f"hash-{observation_id}",
                    "black": black_activity is not None,
                }
                if black_activity is None:
                    source = copies.get(observation_id, observation_id)
                    describe: dict[str, object] = {
                        "page_description": descriptions.get(source, f"Screen {source}"),
                        "state": {"seen_in": observation_id},
                        "elements": [{"description": f"Button {observation_id}"}],
                        "is_external_web": False,
                    }
                    if observation_id in copies:
                        describe["copied_from"] = source
                    line: dict[str, object] = {"observation": observation_id, "describe": describe}
                    if observation_id in refinements:
                        refinement = {"new_description": refinements[observation_id]}
                        line["identify"] = {"stage": "new", "refinement": refinement}
                    append_json_line(identify_file_path(path), line)
                    text_vectors[observation_id] = text_of.get(observation_id, vector)
                (steps_dir / f"{observation_id}.json").write_text(json.dumps(record), "utf-8")
        for number, (before, after, action, from_node, to_node) in enumerate(transitions):
            record = {
                "id": f"{900000 + number}",
                "kind": "transition",
                "run_kind": "explore",
                "before": before,
                "after": after,
                "action": action,
                "from_node": from_node,
                "to_node": to_node,
            }
            (steps_dir / f"{900000 + number}.json").write_text(json.dumps(record), "utf-8")
        save_image_embeddings(steps_dir, image_vectors, model="google/gemini-embedding-2")
        save_embeddings(text_embeddings_path(path), text_vectors, model="google/gemini-embedding-2")
        document = {
            "next_id": len(nodes),
            "matching": DEFAULT_MATCHING._asdict(),
            "nodes": [
                {
                    "id": node_id,
                    "package": "com.demo",
                    "activity": "com.demo.Main",
                    "visit_count": len(members),
                    "replay_misses": 0,
                    "observations": [observation_id for observation_id, _ in members],
                }
                for node_id, members in nodes.items()
            ],
            "edges": list(edges),
        }
        path.write_text(json.dumps(document), encoding="utf-8")

    return write


@pytest.fixture
def shown_descriptions() -> Callable[[httpx.Request], list[str]]:
    """Read the text before each screenshot of a verifier request, which names the
    screenshot and gives its description."""

    def read(request: httpx.Request) -> list[str]:
        content = json.loads(request.content)["messages"][0]["content"]
        return [
            content[index - 1]["text"]
            for index, part in enumerate(content)
            if part["type"] == "image_url"
        ]

    return read


@pytest.fixture
def step_records() -> Callable[[Path], list[dict[str, Any]]]:
    """Read every record of a step log, in step order."""

    def read(steps_dir: Path) -> list[dict[str, Any]]:
        return [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(steps_dir.glob("[0-9]*.json"))
        ]

    return read


class _ScreenSequence:
    w = 1080
    h = 1920

    def __init__(self, *screens: dict[str, Any]) -> None:
        self.config: dict[str, Any] = {"device": {"udid": "test-device"}}
        self._screens = list(screens)

    def get_state(self) -> dict[str, Any]:
        return self._screens[0]

    def exe_action(self, action: dict[str, Any]) -> None:
        assert len(self._screens) > 1, f"No screen scripted after {action}"
        self._screens.pop(0)


@pytest.fixture
def two_screen_device(png_screenshot: str) -> DeviceController:
    # Two screens of one package whose pixels differ, so identify_state compares the
    # second with the first instead of deciding on an exact match.
    search = {
        "activity": "com.demo.app.Search",
        "screenshot": png_screenshot,
        "package": "com.demo.app",
    }
    results = {**search, "activity": "com.demo.app.Results", "screenshot": _png_b64("gray")}
    return _ScreenSequence(search, results)
