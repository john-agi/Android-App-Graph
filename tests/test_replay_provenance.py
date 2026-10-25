"""Replay matches the observed opener before recording a modal's identity."""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path
from typing import Literal

import pytest
from openrouter import OpenRouter
from PIL import Image

from android_app_graph.commands import audit
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
            "activity": self.activities[len(self.actions)],
            "package": "com.demo",
            "screenshot": base64.b64encode(image.getvalue()).decode("ascii"),
        }

    def exe_action(self, action: dict[str, object]) -> None:
        self.actions.append(action)


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
    explorer = Kobe(
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
        client=router_client,
        logger=logging.getLogger("test"),
    )
    graph = explorer.graph
    for node in ("Home", "Settings", "UnexpectedPage", "Modal"):
        graph.graph.add_node(node, observations=[f"stored-{node}"], replay_misses=0)
    open_settings: dict[str, object] = {"action": "tap", "x": 10, "y": 20}
    open_modal: dict[str, object] = {"action": "tap", "x": 30, "y": 40}
    if shape == "edges":
        graph.add_edge("Home", "Settings", open_settings)
        graph.add_edge("Settings", "Modal", open_modal)
    elif shape == "compound":
        graph.add_edge("Home", "Modal", [open_settings, open_modal])
    else:
        graph.add_edge("Home", "Modal", open_modal, details=EdgeDetails(setup=[open_settings]))
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
    if boundary == "explorer":
        reached = explorer.navigate_to_node("Modal")
    else:
        (result,) = audit.re_explore_issues(
            graph,
            [
                {
                    "type": "explore_node",
                    "node": "Modal",
                    "execution": {"effect": "inspection", "reason": "Inspect the modal."},
                }
            ],
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
            client=router_client,
        )
        reached = result["navigated_to"]

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
