"""Exercise the copy-in adapter with a stub host and the real runtime client."""

from __future__ import annotations

import importlib.util
import json
import sys
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


@pytest.mark.usefixtures("model_environment")
def test_android_world_steps_share_one_client_and_preserve_task_memory(
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
                "adb_utils": ModuleType("adb_utils"),
                "interface": ModuleType("interface"),
                "json_action": ModuleType("json_action"),
            },
        ),
    ):
        host = ModuleType(name)
        host.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, host)

    path = Path(__file__).resolve().parents[1] / "aw_files" / "android_app_graph_aw_agent.py"
    spec = importlib.util.spec_from_file_location("android_world_adapter", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "_pixels_to_png_b64", lambda _pixels: png_screenshot)
    agent_type: object = module.UIKobeAndroidWorldAgent
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
        assert history["actions"] == ["acted"] * len(runtime._memory.actions)
        response = vlm_utils.send_chat(
            runtime.client,
            call_type="v2_chat",
            settings=runtime.action,
            messages=[{"role": "user", "content": "inspect"}],
        )
        runtime._memory.add_action(vlm_utils.message_text(response))
        return json.dumps({"message": "acted", "aitk_action": {"action": "wait", "time": 1}})

    monkeypatch.setattr(runtime, "_step", runtime_step)
    assert isinstance(agent, _SteppingAgent)
    for _ in range(2):
        result = agent.step("Inspect the screen")
        assert isinstance(result, SimpleNamespace)
        assert result.done is False
        assert as_object_dict(result.data)["aitk_action"] == {"action": "wait", "time": 1}
        assert runtime.client.sdk_configuration.client is router_http
    assert opened_clients == [runtime.client]
    assert len(runtime._memory.actions) == 2
    assert len(router_requests) == 2
    assert no_sleep == []
