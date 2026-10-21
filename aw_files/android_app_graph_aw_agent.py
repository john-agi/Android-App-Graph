"""Android World adapter for the Android-App-Graph v2 runtime agent."""

from __future__ import annotations

import base64
import io
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from android_world.agents import base_agent
from android_world.env import adb_utils, interface, json_action
from PIL import Image

from android_app_graph.adapters.aitk_translator import UIKobeV2Translator
from android_app_graph.android_packages import package_from_activity

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger("android_app_graph.android_world")


def _pixels_to_png_b64(pixels: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _swipe_direction_from_aitk(action: dict[str, Any]) -> str:
    dx = action["x2"] - action["x1"]
    dy = action["y2"] - action["y1"]
    if abs(dx) >= abs(dy):
        return "right" if dx > 0 else "left"
    return "down" if dy > 0 else "up"


_ANDROID_WORLD_ACTIONS: dict[str, Callable[[dict[str, Any]], json_action.JSONAction | None]] = {
    "tap": lambda aitk_action: json_action.JSONAction(
        action_type=json_action.CLICK, x=aitk_action["x"], y=aitk_action["y"]
    ),
    "long_press": lambda aitk_action: json_action.JSONAction(
        action_type=json_action.LONG_PRESS, x=aitk_action["x"], y=aitk_action["y"]
    ),
    "swipe": lambda aitk_action: json_action.JSONAction(
        action_type=json_action.SWIPE, direction=_swipe_direction_from_aitk(aitk_action)
    ),
    "back": lambda _aitk_action: json_action.JSONAction(action_type=json_action.NAVIGATE_BACK),
    "home": lambda _aitk_action: json_action.JSONAction(action_type=json_action.NAVIGATE_HOME),
    "enter": lambda _aitk_action: json_action.JSONAction(action_type=json_action.KEYBOARD_ENTER),
    "wait": lambda _aitk_action: json_action.JSONAction(action_type=json_action.WAIT),
    "open": lambda aitk_action: json_action.JSONAction(
        action_type=json_action.OPEN_APP, app_name=aitk_action["app"]
    ),
    "end": lambda _aitk_action: json_action.JSONAction(
        action_type=json_action.STATUS, goal_status="complete"
    ),
    "type": lambda _aitk_action: None,
}


def _aitk_to_android_world_action(
    aitk_action: dict[str, Any],
) -> json_action.JSONAction | None:
    action = aitk_action.get("action")
    # A JSON list as the action is unhashable, and must stay a ValueError.
    build = _ANDROID_WORLD_ACTIONS.get(action) if isinstance(action, str) else None
    if build is None:
        msg = f"Unsupported AITK action for Android World: {aitk_action}"
        raise ValueError(msg)
    return build(aitk_action)


def load_agent_settings(config_path: Path) -> tuple[str, dict]:
    """Load graph_dir and vlm_config from either AITK or explorer YAML."""
    with Path(config_path).open(encoding="utf-8") as f:
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
        *,
        name: str = "Android-App-Graph",
        transition_pause: float | None = 1.0,
    ) -> UIKobeAndroidWorldAgent:
        graph_dir, vlm_config = load_agent_settings(Path(config_path))
        return cls(
            env,
            graph_dir=graph_dir,
            vlm_config=vlm_config,
            name=name,
            transition_pause=transition_pause,
        )

    def reset(self, go_home: bool = False) -> None:
        super().reset(go_home=go_home)
        self.env.hide_automation_ui()
        self._runtime.reset_task_state()
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
        package = package_from_activity(activity)

        response_text = self._runtime.to_agent(
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
