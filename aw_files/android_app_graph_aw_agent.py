"""Android World adapter for the Android-App-Graph v2 runtime agent."""

from __future__ import annotations

import base64
import io
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from android_world.agents import base_agent
from android_world.env import adb_utils, interface, json_action
from PIL import Image

from android_app_graph.adapters.aitk_translator import UIKobeV2Translator

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger("android_app_graph.android_world")


def _pixels_to_png_b64(pixels: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _component_package(activity: str) -> str:
    """Return the package half of an ``activity`` component, before its "/".

    This is not ``android_packages.package_from_activity``: that one truncates
    to the standard 3-segment package convention, while this keeps every
    dot-segment before the "/" as-is. Android World's ``adb_utils`` always
    reports the full ``package/component`` form, so no truncation is needed
    here; the two functions are named differently so they are not confused.
    """
    return activity.split("/", 1)[0] if activity else ""


def _swipe_direction_from_aitk(action: dict[str, Any]) -> str:
    dx = action["x2"] - action["x1"]
    dy = action["y2"] - action["y1"]
    if abs(dx) >= abs(dy):
        return "right" if dx > 0 else "left"
    return "down" if dy > 0 else "up"


def _aitk_to_android_world_action(
    aitk_action: dict[str, Any],
) -> json_action.JSONAction | None:
    action = aitk_action.get("action")
    if action == "tap":
        return json_action.JSONAction(
            action_type=json_action.CLICK,
            x=aitk_action["x"],
            y=aitk_action["y"],
        )
    if action == "long_press":
        return json_action.JSONAction(
            action_type=json_action.LONG_PRESS,
            x=aitk_action["x"],
            y=aitk_action["y"],
        )
    if action == "swipe":
        return json_action.JSONAction(
            action_type=json_action.SWIPE,
            direction=_swipe_direction_from_aitk(aitk_action),
        )
    if action == "back":
        return json_action.JSONAction(action_type=json_action.NAVIGATE_BACK)
    if action == "home":
        return json_action.JSONAction(action_type=json_action.NAVIGATE_HOME)
    if action == "enter":
        return json_action.JSONAction(action_type=json_action.KEYBOARD_ENTER)
    if action == "wait":
        return json_action.JSONAction(action_type=json_action.WAIT)
    if action == "open":
        return json_action.JSONAction(
            action_type=json_action.OPEN_APP,
            app_name=aitk_action["app"],
        )
    if action == "end":
        return json_action.JSONAction(
            action_type=json_action.STATUS,
            goal_status="complete",
        )
    if action == "type":
        return None
    raise ValueError(f"Unsupported AITK action for Android World: {aitk_action}")


def load_agent_settings(config_path: Path) -> tuple[str, dict]:
    """Load graph_dir and vlm_config from either AITK or explorer YAML."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    translator_args = config.get("translator_args") or {}
    graph_dir = translator_args.get("graph_dir") or (
        (config.get("experiment") or {}).get("graph_dir", "graphs")
    )
    vlm_config = translator_args.get("vlm_config") or config.get("vlm") or {}
    return graph_dir, vlm_config


class UIKobeAndroidWorldAgent(base_agent.EnvironmentInteractingAgent):
    """Runs the Android-App-Graph v2 runtime loop inside Android World."""

    def __init__(
        self,
        env: interface.AsyncEnv,
        *,
        graph_dir: str = "graphs",
        vlm_config: dict | None = None,
        name: str = "Android-App-Graph",
        transition_pause: float | None = 1.0,
    ) -> None:
        super().__init__(env, name=name, transition_pause=transition_pause)
        self._runtime = UIKobeV2Translator(
            graph_dir=graph_dir,
            vlm_config=vlm_config or {},
        )
        self._history_actions: list[str] = []

    @classmethod
    def from_config(
        cls,
        env: interface.AsyncEnv,
        config_path: str | Path,
        **kwargs: Any,
    ) -> UIKobeAndroidWorldAgent:
        graph_dir, vlm_config = load_agent_settings(Path(config_path))
        return cls(env, graph_dir=graph_dir, vlm_config=vlm_config, **kwargs)

    def reset(self, go_home: bool = False) -> None:
        super().reset(go_home=go_home)
        self.env.hide_automation_ui()
        self._runtime._reset_task_state()
        self._history_actions = []

    def _execute_action(
        self, aitk_action: dict[str, Any]
    ) -> tuple[bool, json_action.JSONAction | None]:
        action = aitk_action.get("action")
        if action == "type":
            adb_utils.type_text(aitk_action.get("text", ""), self.env.controller, timeout_sec=10)
            return False, None

        android_world_action = _aitk_to_android_world_action(aitk_action)
        if android_world_action is None:
            return False, None

        if android_world_action.action_type == json_action.STATUS:
            answer = aitk_action.get("answer")
            if answer:
                self.env.execute_action(
                    json_action.JSONAction(
                        action_type=json_action.ANSWER,
                        text=answer,
                    )
                )
            return True, android_world_action

        self.env.execute_action(android_world_action)
        return False, android_world_action

    def step(self, goal: str) -> base_agent.AgentInteractionResult:
        state = self.get_post_transition_state()
        screenshot_b64 = _pixels_to_png_b64(state.pixels)
        activity = self.env.foreground_activity_name
        package = _component_package(activity)

        response_text = self._runtime._step(
            goal,
            {
                "screenshot": screenshot_b64,
                "activity": activity,
                "package": package,
            },
            {"actions": list(self._history_actions)},
        )
        response = json.loads(response_text)
        message = response.get("message", "")
        aitk_action = response.get("aitk_action", {"action": "wait", "time": 1})
        done, aw_action = self._execute_action(aitk_action)
        self._history_actions.append(message)

        step_data = {
            "raw_screenshot": state.pixels.copy(),
            "ui_elements": state.ui_elements,
            "activity": activity,
            "package": package,
            "android_app_graph_message": message,
            "aitk_action": aitk_action,
            "android_world_action": aw_action,
        }
        return base_agent.AgentInteractionResult(done=done, data=step_data)
