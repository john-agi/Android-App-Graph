"""Exercise the copy-in adapter with a stub host and the real runtime client."""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Protocol, runtime_checkable

import httpx
import pytest
from openrouter import OpenRouter

from android_app_graph.adapters import aitk_translator
from android_app_graph.payloads import as_object_dict
from android_app_graph.utils import vlm_utils


@runtime_checkable
class _SteppingAgent(Protocol):
    def step(self, _goal: str) -> object: ...


_JSON_ACTION = {
    "JSONAction": SimpleNamespace,
    "CLICK": "click",
    "LONG_PRESS": "long_press",
    "SWIPE": "swipe",
    "NAVIGATE_BACK": "navigate_back",
    "NAVIGATE_HOME": "navigate_home",
    "KEYBOARD_ENTER": "keyboard_enter",
    "WAIT": "wait",
    "OPEN_APP": "open_app",
    "STATUS": "status",
    "ANSWER": "answer",
}
_CONTROLLER = "adb controller"


@pytest.fixture
def typed_text() -> list[tuple[str, object, float]]:
    """Each text the stub ``adb_utils.type_text`` typed, with its controller and timeout."""
    return []


@pytest.fixture
def adapter(
    monkeypatch: pytest.MonkeyPatch, typed_text: list[tuple[str, object, float]]
) -> ModuleType:
    """Load the adapter against stub ``android_world`` modules, which cannot install on 3.14."""

    def type_text(text: str, controller: object, *, timeout_sec: float) -> None:
        typed_text.append((text, controller, timeout_sec))

    for name, attributes in (
        ("android_world", {}),
        (
            "android_world.agents",
            {
                "base_agent": SimpleNamespace(
                    EnvironmentInteractingAgent=object,
                    AgentInteractionResult=SimpleNamespace,
                )
            },
        ),
        (
            "android_world.env",
            {
                "adb_utils": _host_module("adb_utils", {"type_text": type_text}),
                "interface": ModuleType("interface"),
                "json_action": _host_module("json_action", _JSON_ACTION),
            },
        ),
    ):
        monkeypatch.setitem(sys.modules, name, _host_module(name, attributes))

    path = Path(__file__).resolve().parents[1] / "aw_files" / "android_app_graph_aw_agent.py"
    spec = importlib.util.spec_from_file_location("android_world_adapter", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _host_module(name: str, attributes: Mapping[str, object]) -> ModuleType:
    host = ModuleType(name)
    host.__dict__.update(attributes)
    return host


@pytest.mark.usefixtures("model_environment")
def test_android_world_steps_share_one_client_and_preserve_task_memory(
    adapter: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    opened_clients: list[OpenRouter],
    router_http: httpx.Client,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    chat_body: dict[str, object],
    png_screenshot: str,
    no_sleep: list[float],
) -> None:
    monkeypatch.setattr(adapter, "_pixels_to_png_b64", lambda _pixels: png_screenshot)
    agent_type: object = adapter.UIKobeAndroidWorldAgent
    assert isinstance(agent_type, type)
    agent = object.__new__(agent_type)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    runtime = aitk_translator.UIKobeV2Translator(graph_dir=str(tmp_path))
    monkeypatch.setattr(agent, "_runtime", runtime, raising=False)
    monkeypatch.setattr(agent, "_history_actions", [], raising=False)
    monkeypatch.setattr(
        agent, "env", SimpleNamespace(foreground_activity_name="com.demo/.Main"), raising=False
    )
    monkeypatch.setattr(
        agent,
        "get_post_transition_state",
        lambda: SimpleNamespace(pixels=bytearray(b"pixels"), ui_elements=[]),
        raising=False,
    )
    monkeypatch.setattr(agent, "_execute_action", lambda _action: (False, None))
    router_responses.extend(httpx.Response(200, json=chat_body) for _ in range(2))

    def runtime_step(_task: str, state: dict[str, object], history: dict[str, object]) -> str:
        assert state["screenshot"] == png_screenshot
        assert state["package"] == "com.demo"
        assert history["actions"] == ["acted"] * len(runtime.memory.actions)
        response = vlm_utils.send_chat(
            runtime.client,
            call_type="v2_chat",
            settings=runtime.action,
            messages=[{"role": "user", "content": "inspect"}],
        )
        runtime.memory.add_action(vlm_utils.message_text(response))
        return json.dumps({"message": "acted", "aitk_action": {"action": "wait", "time": 1}})

    monkeypatch.setattr(runtime, "_step", runtime_step)
    # Only to_agent clears what an earlier task left; an adapter calling _step would keep it.
    runtime.memory.add_action("earlier task")
    assert isinstance(agent, _SteppingAgent)
    for _ in range(2):
        result = agent.step("Inspect the screen")
        assert isinstance(result, SimpleNamespace)
        assert result.done is False
        assert as_object_dict(result.data)["aitk_action"] == {"action": "wait", "time": 1}
        assert runtime.client.sdk_configuration.client is router_http
    assert opened_clients == [runtime.client]
    assert len(runtime.memory.actions) == 2
    assert len(router_requests) == 2
    assert no_sleep == []


def _agent_replying(
    adapter: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    aitk_action: object,
    executed: list[object],
) -> _SteppingAgent:
    """An agent whose runtime replies with ``aitk_action`` and whose env records what it runs."""
    monkeypatch.setattr(adapter, "_pixels_to_png_b64", lambda _pixels: "screenshot")
    agent_type: object = adapter.UIKobeAndroidWorldAgent
    assert isinstance(agent_type, type)
    agent = object.__new__(agent_type)
    reply = json.dumps({"message": "acted", "aitk_action": aitk_action})
    runtime = SimpleNamespace(to_agent=lambda _goal, _state, _history: reply)
    env = SimpleNamespace(
        foreground_activity_name="com.demo/.Main",
        controller=_CONTROLLER,
        execute_action=executed.append,
    )
    monkeypatch.setattr(agent, "_runtime", runtime, raising=False)
    monkeypatch.setattr(agent, "_history_actions", [], raising=False)
    monkeypatch.setattr(agent, "env", env, raising=False)
    monkeypatch.setattr(
        agent,
        "get_post_transition_state",
        lambda: SimpleNamespace(pixels=bytearray(b"pixels"), ui_elements=[]),
        raising=False,
    )
    assert isinstance(agent, _SteppingAgent)
    return agent


def _swipe(x2: int, y2: int) -> dict[str, object]:
    return {"action": "swipe", "x1": 50, "y1": 50, "x2": x2, "y2": y2}


@pytest.mark.parametrize(
    ("aitk_action", "android_world_action"),
    [
        pytest.param(
            {"action": "tap", "x": 5, "y": 6},
            SimpleNamespace(action_type="click", x=5, y=6),
            id="tap",
        ),
        pytest.param(
            {"action": "long_press", "x": 5, "y": 6, "duration": 1000},
            SimpleNamespace(action_type="long_press", x=5, y=6),
            id="long_press",
        ),
        pytest.param(
            _swipe(50, 10), SimpleNamespace(action_type="swipe", direction="up"), id="swipe-up"
        ),
        pytest.param(
            _swipe(50, 90), SimpleNamespace(action_type="swipe", direction="down"), id="swipe-down"
        ),
        pytest.param(
            _swipe(10, 50), SimpleNamespace(action_type="swipe", direction="left"), id="swipe-left"
        ),
        pytest.param(
            _swipe(90, 50),
            SimpleNamespace(action_type="swipe", direction="right"),
            id="swipe-right",
        ),
        pytest.param({"action": "back"}, SimpleNamespace(action_type="navigate_back"), id="back"),
        pytest.param({"action": "home"}, SimpleNamespace(action_type="navigate_home"), id="home"),
        pytest.param(
            {"action": "enter"}, SimpleNamespace(action_type="keyboard_enter"), id="enter"
        ),
        pytest.param({"action": "wait", "time": 3}, SimpleNamespace(action_type="wait"), id="wait"),
        pytest.param(
            {"action": "open", "app": "Clock"},
            SimpleNamespace(action_type="open_app", app_name="Clock"),
            id="open",
        ),
    ],
)
def test_a_step_executes_its_aitk_action_as_an_android_world_action(
    aitk_action: dict[str, object],
    android_world_action: SimpleNamespace,
    adapter: ModuleType,
    typed_text: list[tuple[str, object, float]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executed: list[object] = []
    agent = _agent_replying(adapter, monkeypatch, aitk_action, executed)

    result = agent.step("Act on the screen")

    assert isinstance(result, SimpleNamespace)
    assert result.done is False
    assert as_object_dict(result.data)["android_world_action"] == android_world_action
    assert executed == [android_world_action]
    assert typed_text == []


@pytest.mark.parametrize(
    ("aitk_action", "done", "android_world_action", "executed_actions", "typed"),
    [
        pytest.param(
            {"action": "end"},
            True,
            SimpleNamespace(action_type="status", goal_status="complete"),
            [],
            [],
            id="end",
        ),
        pytest.param(
            {"action": "end", "answer": "9 EUR"},
            True,
            SimpleNamespace(action_type="status", goal_status="complete"),
            [SimpleNamespace(action_type="answer", text="9 EUR")],
            [],
            id="end-with-answer",
        ),
        pytest.param(
            {"action": "type", "text": "milk"},
            False,
            None,
            [],
            [("milk", _CONTROLLER, 10)],
            id="type",
        ),
    ],
)
def test_an_end_or_type_step_is_handled_outside_the_action_mapping(
    aitk_action: dict[str, object],
    done: bool,
    android_world_action: SimpleNamespace | None,
    executed_actions: list[object],
    typed: list[tuple[str, object, float]],
    adapter: ModuleType,
    typed_text: list[tuple[str, object, float]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executed: list[object] = []
    agent = _agent_replying(adapter, monkeypatch, aitk_action, executed)

    result = agent.step("Finish the task")

    assert isinstance(result, SimpleNamespace)
    assert result.done is done
    assert as_object_dict(result.data)["android_world_action"] == android_world_action
    assert executed == executed_actions
    assert typed_text == typed


@pytest.mark.parametrize("aitk_action", [{"action": "pinch"}, {}, {"action": ["tap"]}])
def test_a_step_rejects_an_aitk_action_android_world_cannot_run(
    aitk_action: dict[str, object], adapter: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    executed: list[object] = []
    agent = _agent_replying(adapter, monkeypatch, aitk_action, executed)

    with pytest.raises(ValueError, match="Unsupported AITK action for Android World"):
        agent.step("Act on the screen")
    assert executed == []
