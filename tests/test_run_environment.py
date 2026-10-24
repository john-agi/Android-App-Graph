"""The operator lifecycle preserves display settings and disposable-data evidence."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from openrouter import OpenRouter

from android_app_graph import cli, device


class _Script:
    def __init__(self, root: Path, failure: str | None) -> None:
        self.failure = failure
        self.values: dict[str, str | None] = {"font_scale": None}
        self.graph = root / "graphs" / "demo" / "demo.json"

    def mutate(self, stage: str) -> None:
        self.values["font_scale"] = "1.5"
        if self.failure == "interrupt":
            raise KeyboardInterrupt
        if self.failure == stage:
            msg = f"{stage} failure"
            raise RuntimeError(msg)

    def adb(
        self,
        args: list[str],
        *,
        output: str,
        check: bool,
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        del output, check
        command = args[2:]
        result = self._output(command)
        if command[:3] == ["shell", "settings", "put"]:
            self.values[command[-2]] = command[-1]
        if command[:3] == ["shell", "settings", "delete"]:
            self.values[command[-1]] = None
        return subprocess.CompletedProcess(args, 0, result, "")

    def _output(self, command: list[str]) -> str:
        outputs = {
            ("shell", "wm", "size"): "Physical size: 720x1280\n",
            ("shell", "wm", "density"): "Physical density: 320\n",
            ("shell", "cmd", "uimode", "night"): "Night mode: no\n",
            (
                "shell",
                "dumpsys",
                "uimode",
            ): "mOverrideOn/Off=false/false customStart=22:00 customEnd06:00",
        }
        if command[:3] == ["shell", "settings", "get"]:
            return self.values.get(command[-1]) or "null"
        return outputs.get(tuple(command), "")


class _Controller:
    w = 720
    h = 1280

    def __init__(self, script: _Script) -> None:
        self.config: dict[str, object] = {"device": {"udid": "fake"}}
        self.script = script

    def terminate_all_apps(self) -> None:
        if self.script.failure == "startup":
            self.script.mutate("startup")

    def get_state(self, *, deadline: float | None = None) -> dict[str, object]:
        del deadline
        return {}

    def exe_action(self, _action: dict[str, object]) -> None:
        pass


class _Explorer:
    def __init__(self, script: _Script) -> None:
        self.script = script

    def explore(self, *, resume_from: str | None) -> None:
        del resume_from
        self.script.mutate("error")

    def save_graph(self) -> None:
        self.script.graph.parent.mkdir(parents=True, exist_ok=True)
        self.script.graph.write_text('{"total_steps_completed": 1}')


def _configure_cli(monkeypatch: pytest.MonkeyPatch, root: Path, script: _Script) -> None:
    config = root / "config.json"
    config.write_text(
        json.dumps(
            {
                "device": {"udid": "fake", "avd_name": "fake"},
                "apps": [{"name": "demo", "package_name": "com.demo"}],
                "vlm": {
                    role: {"model": "fake/model"}
                    for role in ["page_detail", "embedding", "instruction", "action"]
                },
                "experiment": {"graph_dir": str(root / "graphs")},
            }
        )
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")
    monkeypatch.setattr(sys, "argv", ["app-graph", "-c", str(config)])
    monkeypatch.setattr(cli, "check_avd", lambda *_args: None)
    monkeypatch.setattr(cli, "EmulatorManager", lambda *_args: object())
    monkeypatch.setattr(cli, "grant_runtime_permissions", lambda *_args: None)
    monkeypatch.setattr(device, "run_adb", script.adb)
    monkeypatch.setattr(cli, "AdbController", lambda *_args: _Controller(script))
    monkeypatch.setattr(cli, "Kobe", lambda *_args, **_kwargs: _Explorer(script))
    monkeypatch.setattr(cli, "launch_app", lambda *_args: "com.demo/.Main")


@pytest.mark.parametrize("failure", [None, "interrupt", "error", "startup"])
def test_cli_restores_original_display_values_on_each_exit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    opened_clients: list[OpenRouter],
    failure: str | None,
) -> None:
    """Cleanup preserves the graph and original exception while restoring captured values."""
    del opened_clients
    script = _Script(tmp_path, failure)
    _configure_cli(monkeypatch, tmp_path, script)
    if failure:
        expected = KeyboardInterrupt if failure == "interrupt" else RuntimeError
        with pytest.raises(expected):
            cli.main()
    else:
        cli.main()
    assert script.values["font_scale"] is None
    report = json.loads(script.graph.with_name("demo_run.json").read_text())
    assert report["baseline"]["font_scale"] is None
    assert report["cleanup"][-1]["status"] == "restored"
    if failure != "startup":
        assert script.graph.is_file()
