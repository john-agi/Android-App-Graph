"""Startup recovers device phases before counting the first exploration action."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import pytest
from openrouter import OpenRouter

from android_app_graph import cli, device
from android_app_graph.graph_files import (
    graph_steps_dir,
    identify_file_path,
    save_embeddings,
    text_embeddings_path,
)
from android_app_graph.kobe import ExplorerConfig, Kobe, _Position
from android_app_graph.utils import DEFAULT_MATCHING
from android_app_graph.utils.graph_manager import GraphManager


def _startup_config(tmp_path: Path, phase: str) -> tuple[Path, Path, int]:
    graph_path = tmp_path / "graphs" / "demo" / "demo.json"
    saved_steps = 4 if phase == "resume" else 0
    config = tmp_path / "startup.yaml"
    config.write_text(
        json.dumps(
            {
                "device": {"udid": "test-device", "unreachable_limit": 45},
                "apps": [{"name": "demo", "package_name": "com.demo"}],
                "experiment": {"graph_dir": str(tmp_path / "graphs"), "max_steps": saved_steps + 1},
            }
        ),
        encoding="utf-8",
    )
    return config, graph_path, saved_steps


def _seed_startup_graph(path: Path, add_member: Callable[..., None]) -> None:
    graph = GraphManager()
    graph.matching = DEFAULT_MATCHING
    graph.steps_dir = graph_steps_dir(path)
    graph.identify_file = identify_file_path(path)
    graph.graph.add_node("s0_home", package="com.demo", replay_misses=2)
    add_member(graph, "s0_home")
    graph.total_steps_completed = 4
    graph.save_graph(path)
    save_embeddings(
        text_embeddings_path(path),
        {"s0_home-observation": [1.0]},
        model="google/gemini-embedding-2",
    )


def _startup_adb(
    phase: str, interrupt: Callable[[str], None]
) -> Callable[..., subprocess.CompletedProcess[str]]:
    def adb(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        if "grant" in args:
            interrupt("grant")
            if phase == "refusal":
                return subprocess.CompletedProcess(args, 1, stdout="", stderr="Permission denied")
        output = ""
        if "package" in args:
            output = "  requested permissions:\n    android.permission.CAMERA\n"
        if "resolve-activity" in args:
            output = "com.demo/.Home"
        if "start" in args:
            interrupt("launch")
        return subprocess.CompletedProcess(args, 0, stdout=output, stderr="")

    return adb


def _install_startup_device(
    monkeypatch: pytest.MonkeyPatch, phase: str, events: list[str], screenshot: str
) -> None:
    interrupted: set[str] = set()
    state: dict[str, object] = {
        "package": "com.demo",
        "activity": "com.demo.Home",
        "screenshot": screenshot,
    }

    def interrupt(name: str) -> None:
        events.append(name)
        if phase == name and name not in interrupted:
            interrupted.add(name)
            raise subprocess.TimeoutExpired(["adb", "-s", "test-device", name], 30)

    class Device:
        w = 100
        h = 200

        def __init__(
            self, config: dict[str, object], _logger: logging.Logger, **_kwargs: object
        ) -> None:
            self.config = config

        def terminate_all_apps(self, *_args: object, **_kwargs: object) -> None:
            events.append("recents")

        def get_state(self, **_kwargs: float | None) -> dict[str, object]:
            interrupt("capture")
            return state

        def exe_action(self, _action: dict[str, object]) -> None:
            interrupt("resume")

    monkeypatch.setattr(cli, "AdbController", Device)
    monkeypatch.setattr(device, "run_adb", _startup_adb(phase, interrupt))


@pytest.mark.usefixtures("model_environment", "opened_clients", "no_sleep")
@pytest.mark.parametrize("phase", ["grant", "capture", "resume", "launch", "refusal"])
def test_cli_startup_reaches_one_valid_step_after_each_interruption(
    phase: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    png_screenshot: str,
    add_member: Callable[..., None],
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setattr(cli, "setup_logging", lambda **_kwargs: None)
    config, graph_path, saved_steps = _startup_config(tmp_path, phase)
    if saved_steps:
        _seed_startup_graph(graph_path, add_member)
    events: list[str] = []
    observations: list[int] = []
    steps: list[int] = []
    _install_startup_device(monkeypatch, phase, events, png_screenshot)

    def identify(explorer: Kobe, observed: dict[str, object]) -> tuple[str, dict[str, object]]:
        observations.append(explorer.graph.total_steps_completed)
        if "s0_home" not in explorer.graph.graph:
            explorer.graph.graph.add_node("s0_home", package="com.demo", replay_misses=0)
            add_member(explorer.graph, "s0_home")
        return "s0_home", observed

    def step(explorer: Kobe, position: _Position, _max_steps: int) -> _Position:
        steps.append(explorer.graph.total_steps_completed)
        explorer.graph.total_steps_completed += 1
        return position

    def navigate(explorer: Kobe, _node: str) -> str:
        explorer.controller.exe_action({"action": "tap", "x": 1, "y": 2})
        return "s0_home"

    monkeypatch.setattr(cli, "check_avd", lambda *_args: None)
    monkeypatch.setattr(Kobe, "_identify", identify)
    monkeypatch.setattr(Kobe, "_explore_step", step)
    monkeypatch.setattr(Kobe, "navigate_to_node", navigate)
    monkeypatch.setattr(GraphManager, "normalize_all_edges", lambda _self: 0)
    monkeypatch.setattr(GraphManager, "embed_missing_members", lambda _self: None)
    args = ["app-graph", "-c", str(config)]
    if saved_steps:
        args.extend(["--resume-from", "s0_home"])
    monkeypatch.setattr(sys, "argv", args)
    caplog.set_level(logging.INFO)

    cli.main()

    saved = json.loads(graph_path.read_text(encoding="utf-8"))
    assert steps == [saved_steps]
    assert observations == [saved_steps]
    assert saved["total_steps_completed"] == saved_steps + 1
    assert [node["id"] for node in saved["nodes"]] == ["s0_home"]
    assert saved["edges"] == []
    assert events.count("recents") == 1
    assert events.count("resume") == (1 if saved_steps else 0)
    assert events.count("launch") == 1
    assert events.count("grant") == (2 if phase == "grant" else 1)
    if phase == "refusal":
        assert any("Could not grant" in message for message in caplog.messages)
        assert not any("Waiting" in message for message in caplog.messages)
    else:
        assert any("test-device" in message and "startup" in message for message in caplog.messages)


@pytest.mark.usefixtures("adb_on_path", "model_environment")
@pytest.mark.parametrize("resumed", [False, True])
def test_startup_capture_limit_saves_the_graph_without_counting_a_step(
    resumed: bool,
    tmp_path: Path,
    router_client: OpenRouter,
    add_member: Callable[..., None],
    no_sleep: list[float],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(time, "monotonic", lambda: sum(no_sleep))
    monkeypatch.setattr(
        device,
        "run_adb",
        lambda args, **_kwargs: subprocess.CompletedProcess(args, 0, stdout="", stderr=""),
    )

    class Device:
        w = 100
        h = 200

        def __init__(self) -> None:
            self.config: dict[str, object] = {"device": {"udid": "test-device"}}

        def get_state(self, **_kwargs: float | None) -> dict[str, object]:
            raise subprocess.TimeoutExpired(["adb", "-s", "test-device", "capture"], 30)

        def exe_action(self, action: dict[str, object]) -> None:
            del action
            pytest.fail("startup capture sends no action")

    explorer = Kobe(
        Device(),
        ExplorerConfig(
            app_name="demo", package_name="com.demo", graph_dir=tmp_path, unreachable_limit=45
        ),
        client=router_client,
        logger=logging.getLogger("test"),
    )
    if resumed:
        explorer.graph.graph.add_node("s0_home", package="com.demo", replay_misses=2)
        add_member(explorer.graph, "s0_home")
        explorer.graph.total_steps_completed = 4
        explorer.save_graph()
    expected = explorer.graph.total_steps_completed

    with pytest.raises(
        device.DeviceUnreachableError, match=r"test-device unreachable for 0\.75 minutes"
    ):
        explorer.explore(max_steps=expected + 1)

    saved = json.loads(explorer.graph_path.read_text(encoding="utf-8"))
    assert saved["total_steps_completed"] == expected
    assert [node["id"] for node in saved["nodes"]] == (["s0_home"] if resumed else [])
    assert saved["edges"] == []
    assert no_sleep == [30, 15]
