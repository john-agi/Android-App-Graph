"""Replay matches the observed opener before recording a modal's identity."""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path
from typing import Literal, override

import pytest
from openrouter import OpenRouter
from PIL import Image

from android_app_graph.commands import audit
from android_app_graph.device import DeviceController
from android_app_graph.kobe import ExplorerConfig, Kobe
from android_app_graph.modal_context import MatchInput, ModalContext, Surface
from android_app_graph.step_log import Observation
from android_app_graph.utils import ModelSettings
from android_app_graph.utils.graph_manager import EdgeDetails


class _RouteDevice:
    w = 100
    h = 200

    def __init__(self, opener: str) -> None:
        self.config: dict[str, object] = {"device": {"udid": "test-device"}}
        self.activities = ("Home", opener, "Modal")
        self.actions: list[dict[str, object]] = []

    def get_state(self, **_kwargs: float | None) -> dict[str, object]:
        image = io.BytesIO()
        Image.new("RGB", (self.w, self.h), (len(self.actions), 0, 0)).save(image, "PNG")
        return {
            "activity": self.activities[min(len(self.actions), len(self.activities) - 1)],
            "package": "com.demo",
            "screenshot": base64.b64encode(image.getvalue()).decode("ascii"),
        }

    def exe_action(self, action: dict[str, object]) -> None:
        self.actions.append(action)


def _explorer(device: DeviceController, tmp_path: Path, client: OpenRouter) -> Kobe:
    return Kobe(
        device,
        ExplorerConfig(
            app_name="demo",
            package_name="com.demo",
            graph_dir=tmp_path,
            vlm_config={
                role: {"model": f"provider/{role}"}
                for role in ("page_detail", "embedding", "instruction", "action")
            },
        ),
        client=client,
        logger=logging.getLogger("test"),
    )


def _navigate(boundary: str, explorer: Kobe, device: DeviceController, client: OpenRouter) -> str:
    if boundary == "explorer":
        return explorer.navigate_to_node("Modal")
    issue: dict[str, object] = {
        "type": "explore_node",
        "node": "Modal",
        "execution": {"effect": "inspection", "reason": "Inspect the modal."},
    }
    if boundary == "audit_retry":
        issue.update(
            type="retry_edge", source_node="Home", target_node="Modal", instruction="Inspect modal"
        )
    (result,) = audit.re_explore_issues(
        explorer.graph,
        [issue],
        target=audit.ReExploreTarget(
            app_name="demo", package_name="com.demo", graph_path=explorer.graph_path
        ),
        session=audit.ReExploreSession(
            device,
            ModelSettings("instruction"),
            ModelSettings("action"),
            30,
            steps_per_issue=0,
        ),
        client=client,
    )
    reached = result["reached" if boundary == "audit_retry" else "navigated_to"]
    assert isinstance(reached, str)
    return reached


@pytest.mark.parametrize("boundary", ["explorer", "audit"])
@pytest.mark.parametrize("shape", ["edges", "compound", "setup"])
@pytest.mark.parametrize("opener", ["Settings", "UnexpectedPage"])
@pytest.mark.usefixtures("no_sleep", "device_log")
def test_replay_modal_source_is_the_observed_immediate_opener(
    boundary: Literal["explorer", "audit"],
    shape: Literal["edges", "compound", "setup"],
    opener: str,
    tmp_path: Path,
    router_client: OpenRouter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = _RouteDevice(opener)
    explorer = _explorer(device, tmp_path, router_client)
    graph = explorer.graph
    for node in ("Home", "Settings", "UnexpectedPage", "Modal"):
        graph.graph.add_node(node, observations=[f"stored-{node}"], replay_misses=0)
    open_settings: dict[str, object] = {"action": "tap", "x": 10, "y": 20}
    open_modal: dict[str, object] = {"action": "tap", "x": 30, "y": 40}
    if shape == "edges":
        graph.add_edge("Home", "Settings", [open_settings])
        graph.add_edge("Settings", "Modal", [open_modal])
    elif shape == "compound":
        graph.add_edge("Home", "Modal", [open_settings, open_modal])
    else:
        graph.add_edge("Home", "Modal", [open_modal], details=EdgeDetails(setup=[open_settings]))
    observations: dict[str, str] = {}
    inputs: list[MatchInput] = []

    def identify(
        observation: Observation,
        *,
        match_input: MatchInput,
        expected_observation: str | None = None,
    ) -> str:
        del expected_observation
        observations[observation.observation_id] = observation.activity
        inputs.append(match_input)
        return observation.activity

    monkeypatch.setattr(graph, "identify_state", identify)
    monkeypatch.setattr(graph, "latest_description", lambda node: node)
    reached = _navigate(boundary, explorer, device, router_client)

    assert reached == "Modal"
    assert list(observations.values()) == ["Home", opener, "Modal"]
    source = inputs[-1]
    assert source.previous is not None
    assert observations[source.previous] == opener
    assert source.actions == (open_modal,)
    context = ModalContext().advance("modal", source, Surface("modal", "open"))
    assert context.origin_known
    assert context.frames[-1].source == source
    assert device.actions == [open_settings, open_modal]


@pytest.mark.parametrize("boundary", ["explorer", "audit"])
@pytest.mark.usefixtures("no_sleep", "device_log")
def test_a_failed_intermediate_capture_prevents_the_next_replay_input(
    boundary: str,
    tmp_path: Path,
    router_client: OpenRouter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = _RouteDevice("Settings")
    explorer = _explorer(device, tmp_path, router_client)
    first: dict[str, object] = {"action": "tap", "x": 10, "y": 20}
    second: dict[str, object] = {"action": "tap", "x": 30, "y": 40}
    explorer.graph.graph.add_nodes_from(["Home", "Modal"], replay_misses=0)
    explorer.graph.add_edge("Home", "Modal", [first, second])
    monkeypatch.setattr(explorer.graph, "identify_state", lambda *_args, **_kwargs: "Home")
    monkeypatch.setattr(explorer.graph, "latest_description", lambda node: node)
    capture = device.get_state

    def read(**_kwargs: float | None) -> dict[str, object]:
        if device.actions:
            message = "Intermediate screenshot unavailable"
            raise OSError(message)
        return capture()

    monkeypatch.setattr(device, "get_state", read)
    with pytest.raises(OSError, match="Intermediate screenshot unavailable"):
        _navigate(boundary, explorer, device, router_client)
    assert device.actions == [first]


class _RecoveringDevice(_RouteDevice):
    def __init__(self) -> None:
        super().__init__("Shade")
        self.page = "Home"

    @override
    def get_state(self, **_kwargs: float | None) -> dict[str, object]:
        state = {**super().get_state(), "activity": self.page}
        if self.page == "Shade":
            state.update(package="com.android.systemui", system_surface="notification_shade")
        return state

    @override
    def exe_action(self, action: dict[str, object]) -> None:
        super().exe_action(action)
        self.page = (
            "RecoveredPage"
            if action.get("action") == "back"
            else "Shade"
            if action.get("x") == 10
            else "WrongModal"
        )


@pytest.mark.parametrize(
    ("boundary", "shape"),
    [
        ("explorer", "edges"),
        ("explorer", "compound"),
        ("explorer", "setup"),
        ("audit", "edges"),
        ("audit", "compound"),
        ("audit", "setup"),
        ("audit_retry", "compound"),
        ("audit_retry", "setup"),
    ],
)
@pytest.mark.usefixtures("no_sleep", "device_log")
def test_system_recovery_ends_the_pending_replay_sequence(
    boundary: str,
    shape: str,
    tmp_path: Path,
    router_client: OpenRouter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = _RecoveringDevice()
    explorer = _explorer(device, tmp_path, router_client)
    graph = explorer.graph
    graph.graph.add_nodes_from(["Home", "Settings", "Modal", "RecoveredPage"], replay_misses=0)
    first: dict[str, object] = {"action": "tap", "x": 10, "y": 20}
    pending: dict[str, object] = {"action": "tap", "x": 30, "y": 40}
    if shape == "edges":
        graph.add_edge("Home", "Settings", [first])
        graph.add_edge("Settings", "Modal", [pending])
    elif shape == "compound":
        graph.add_edge(
            "Home", "Modal", [first, pending], details=EdgeDetails(instruction="Inspect modal")
        )
    else:
        graph.add_edge(
            "Home",
            "Modal",
            [pending],
            details=EdgeDetails(instruction="Inspect modal", setup=[first]),
        )
    monkeypatch.setattr(
        graph, "identify_state", lambda observation, **_kwargs: observation.activity
    )
    monkeypatch.setattr(graph, "latest_description", lambda node: node)

    reached = _navigate(boundary, explorer, device, router_client)

    assert device.actions == [first, {"action": "back"}]
    assert reached == "RecoveredPage"
