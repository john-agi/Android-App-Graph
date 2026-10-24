"""The operator lifecycle preserves display settings and disposable-data evidence."""

from __future__ import annotations

import base64
import io
import json
import logging
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import NoReturn, override

import pytest
from openrouter import OpenRouter
from PIL import Image

from android_app_graph import cli, device, run_environment
from android_app_graph.display_state import NIGHT_KEYS, DisplayState
from android_app_graph.run_environment import RunEnvironment, UnsafeReplayError, mutation_config
from android_app_graph.step_log import StepRecorder, StepSource


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

    def terminate_all_apps(self, _recovery: device.StartupRecovery) -> None:
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
    monkeypatch.setattr(cli, "AdbController", lambda *_args, **_kwargs: _Controller(script))
    monkeypatch.setattr(cli, "Kobe", lambda *_args, **_kwargs: _Explorer(script))
    monkeypatch.setattr(cli, "launch_app", lambda *_args: "com.demo/.Main")


@pytest.mark.usefixtures("no_sleep")
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
    assert report["cleanup"][-1]["before"]["font_scale"] == "1.5"
    if failure != "startup":
        assert script.graph.is_file()


class _Access:
    def __init__(self) -> None:
        self.original = DisplayState(
            (720, 1280),
            None,
            320,
            None,
            None,
            "no",
            "22:00",
            "06:00",
            dict.fromkeys(NIGHT_KEYS),
            (False, False),
        )
        self.current = self.original
        self.commands: list[tuple[str, ...]] = []
        self.unreachable = False
        self.hashes: dict[str, str] = {"/sdcard/seed/a.txt": "a" * 64}
        self.seed_reads: list[str] = []

    def capture(self, *, deadline: float | None = None) -> DisplayState:
        del deadline
        if self.unreachable:
            msg = "device is offline"
            raise OSError(msg)
        return self.current

    def execute(self, command: tuple[str, ...], *, deadline: float) -> None:
        del deadline
        self.commands.append(command)
        if self.unreachable:
            msg = "device is offline"
            raise OSError(msg)
        self.current = self.original

    def seed_hashes(self, root: str, *, deadline: float | None = None) -> dict[str, str]:
        del deadline
        self.seed_reads.append(root)
        if self.unreachable:
            msg = "device is offline"
            raise OSError(msg)
        return dict(self.hashes)


class _ScreenDevice:
    w = 720
    h = 1280

    def __init__(self, image: str) -> None:
        self.config: dict[str, object] = {"device": {"udid": "fake"}}
        self.image = image
        self.actions: list[dict[str, object]] = []

    def get_state(self, *, deadline: float | None = None) -> dict[str, object]:
        del deadline
        return {
            "screenshot": self.image,
            "xml": "<hierarchy/>",
            "package": "com.demo",
            "activity": "Main",
        }

    def exe_action(self, action: dict[str, object]) -> None:
        self.actions.append(action)


def _environment(
    tmp_path: Path, png_screenshot: str, access: _Access, **mutation: object
) -> RunEnvironment:
    return RunEnvironment(
        _ScreenDevice(png_screenshot),
        access,
        tmp_path / "demo.json",
        mutation=mutation,
        logger=logging.getLogger("test"),
    )


def test_original_baseline_survives_resume_and_default_value_changes(
    tmp_path: Path, png_screenshot: str
) -> None:
    access = _Access()
    first = _environment(tmp_path, png_screenshot, access)
    with first:
        access.current = replace(access.current, font_scale="1.25", density_override=400)
    access.current = replace(access.current, font_scale="1.5", size_override=(1080, 1920))
    resumed = _environment(tmp_path, png_screenshot, access)
    with resumed:
        assert json.loads(resumed.path.read_text())["baseline"]["font_scale"] is None
    assert access.current == access.original
    assert len(json.loads(resumed.path.read_text())["cleanup"]) == 2
    assert ("settings", "delete", "system", "font_scale") in access.commands
    assert ("wm", "size", "reset") in access.commands


@pytest.mark.parametrize("source", ["navigation_replay", "edge_retry"])
@pytest.mark.parametrize("change", ["density", "font", "size", "native"])
def test_recorded_coordinates_are_rejected_for_display_drift(
    tmp_path: Path,
    png_screenshot: str,
    source: StepSource,
    change: str,
) -> None:
    access = _Access()
    controller = _ScreenDevice(png_screenshot)
    environment = RunEnvironment(
        controller, access, tmp_path / "demo.json", mutation={}, logger=logging.getLogger("test")
    )
    with environment:
        recorder = StepRecorder(
            environment, tmp_path / "steps", run_kind="explore", source="main_loop"
        )
        recorder.get_state()
        action: dict[str, object] = {"action": "tap", "x": 1, "y": 2}
        recorder.exe_action(action)
        recorded = json.loads(json.dumps(action))
        if change == "density":
            access.current = replace(access.current, density_override=400)
        elif change == "font":
            access.current = replace(access.current, font_scale="1.3")
        elif change == "size":
            access.current = replace(access.current, size_override=(1080, 1920))
        else:
            recorded["recorded_display"]["native_size"] = [64, 64]
        with (
            recorder.source(source),
            pytest.raises(UnsafeReplayError, match="recorded display state"),
        ):
            recorder.exe_action(recorded)
        assert len(controller.actions) == 1
        assert list(tmp_path.joinpath("steps").glob("*.json"))
    report = json.loads(environment.path.read_text())
    assert len(report["unsafe_replays"]) == 1


def test_fresh_inputs_and_display_changes_have_observation_attribution(
    tmp_path: Path, png_screenshot: str
) -> None:
    access = _Access()
    environment = _environment(tmp_path, png_screenshot, access)
    with environment:
        recorder = StepRecorder(
            environment, tmp_path / "steps", run_kind="explore", source="main_loop"
        )
        before = recorder.get_state()
        access.current = replace(access.current, font_scale="1.5")
        after = recorder.get_state()
        action: dict[str, object] = {"action": "tap", "x": 1, "y": 2}
        recorder.exe_action(action)
        recorder.exe_action({"action": "back"})
        assert json.loads(json.dumps(action))["recorded_display"]["settings"]["font_scale"] == "1.5"
    report = json.loads(environment.path.read_text())
    assert [r["observation"] for r in report["changes"]] == [
        before["observation_id"],
        after["observation_id"],
    ]
    assert all(r["source"] == "main_loop" for r in report["changes"])


def test_missing_replay_evidence_never_dispatches_a_coordinate(
    tmp_path: Path, png_screenshot: str
) -> None:
    access = _Access()
    controller = _ScreenDevice(png_screenshot)
    environment = RunEnvironment(
        controller, access, tmp_path / "demo.json", mutation={}, logger=logging.getLogger("test")
    )
    with environment:
        recorder = StepRecorder(
            environment, tmp_path / "steps", run_kind="audit", source="edge_retry"
        )
        with pytest.raises(UnsafeReplayError, match="recorded display evidence"):
            recorder.exe_action({"action": "swipe", "x1": 1, "y1": 2, "x2": 3, "y2": 4})
        record = json.loads((tmp_path / "steps" / "000000.json").read_text())
        assert record["kind"] == "action_rejected"
        assert not controller.actions


def test_pending_restoration_keeps_original_error_and_graph(
    tmp_path: Path, png_screenshot: str
) -> None:
    access = _Access()
    environment = _environment(tmp_path, png_screenshot, access)
    message = "run error"
    original = RuntimeError(message)
    graph = tmp_path / "demo.json"

    def fail() -> None:
        with environment:
            graph.write_text('{"total_steps_completed":7}')
            access.current = replace(access.current, font_scale="1.5")
            access.unreachable = True
            raise original

    with pytest.raises(RuntimeError) as caught:
        fail()
    assert caught.value is original
    assert json.loads(graph.read_text())["total_steps_completed"] == 7
    cleanup = json.loads(environment.path.read_text())["cleanup"][-1]
    assert cleanup["status"] == "pending"
    assert cleanup["original_error"] == "RuntimeError"
    assert cleanup["pending_commands"]


def test_seed_manifest_reports_only_the_explicit_public_scope(
    tmp_path: Path, png_screenshot: str
) -> None:
    access = _Access()
    specification = tmp_path / "seed.json"
    specification.write_text(json.dumps({"root": "/sdcard/seed", "files": access.hashes}))
    environment = _environment(
        tmp_path,
        png_screenshot,
        access,
        seed_manifest=str(specification),
        disposable_device=True,
        cleanup_recipe="Reset the disposable device after copying the graph",
    )
    with environment:
        access.hashes = {"/sdcard/seed/b.txt": "b" * 64}
    report = json.loads(environment.path.read_text())
    assert report["seed"]["provided"] == {"/sdcard/seed/a.txt": "a" * 64}
    assert report["cleanup"][-1]["seed_changes"] == {
        "added": ["/sdcard/seed/b.txt"],
        "removed": ["/sdcard/seed/a.txt"],
        "changed": [],
    }
    assert report["contract"]["disposable_device"] is True


@pytest.mark.parametrize(
    "root", ["/data/user/0/private", "/sdcard/a/../private", "/sdcard/a\nextra"]
)
def test_seed_roots_refuse_private_or_ambiguous_paths(
    tmp_path: Path, png_screenshot: str, root: str
) -> None:
    access = _Access()
    specification = tmp_path / "seed.json"
    specification.write_text(json.dumps({"root": root, "files": {}}))
    with (
        pytest.raises(ValueError, match="public-storage"),
        _environment(tmp_path, png_screenshot, access, seed_manifest=str(specification)),
    ):
        pytest.fail("invalid seed root entered the run")


def test_an_existing_graph_requires_its_original_sidecar(
    tmp_path: Path, png_screenshot: str
) -> None:
    (tmp_path / "demo.json").write_text("{}")
    with (
        pytest.raises(ValueError, match="original run baseline"),
        _environment(tmp_path, png_screenshot, _Access()),
    ):
        pytest.fail("a baseline was invented for an existing graph")


@pytest.mark.parametrize(
    "value", [[], {"disposable_device": "yes"}, {"cleanup_recipe": ""}, {"seed_manifest": 1}]
)
def test_mutation_contract_requires_explicit_typed_values(value: object) -> None:
    with pytest.raises((ValueError, TypeError), match=r"experiment\.mutation"):
        mutation_config(value)


@pytest.mark.parametrize("field", ["graph", "device", "changes"])
def test_resume_validates_graph_device_and_report_shape(
    tmp_path: Path, png_screenshot: str, field: str
) -> None:
    access = _Access()
    environment = _environment(tmp_path, png_screenshot, access)
    with environment:
        pass
    report = json.loads(environment.path.read_text())
    report[field] = None
    environment.path.write_text(json.dumps(report))
    with pytest.raises((ValueError, TypeError), match=r"another|must be a list"), environment:
        pytest.fail("incompatible run identity was accepted")


def test_controller_geometry_is_forwarded_and_stable_captures_stay_sparse(
    tmp_path: Path, png_screenshot: str
) -> None:
    access = _Access()
    controller = _ScreenDevice(png_screenshot)
    environment = RunEnvironment(
        controller, access, tmp_path / "demo.json", mutation={}, logger=logging.getLogger("test")
    )
    with environment:
        environment.w, environment.h = 1080, 1920
        assert (environment.w, environment.h) == (controller.w, controller.h) == (1080, 1920)
        environment.exe_action({"action": "long_press", "x": 1, "y": 2})
        environment.get_state()
        environment.get_state()
        assert controller.actions == [{"action": "long_press", "x": 1, "y": 2}]
    assert len(json.loads(environment.path.read_text())["changes"]) == 1


@pytest.mark.parametrize("original_failure", [False, True])
def test_report_write_failure_keeps_the_persisted_pending_baseline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    png_screenshot: str,
    original_failure: bool,
) -> None:
    access = _Access()
    environment = _environment(tmp_path, png_screenshot, access)
    message = "run failure"
    original = RuntimeError(message)

    def write_failure(*_args: object, **_kwargs: object) -> NoReturn:
        message = "report storage unavailable"
        raise OSError(message)

    def run() -> None:
        with environment:
            monkeypatch.setattr(run_environment, "write_json_atomically", write_failure)
            if original_failure:
                raise original

    expected = RuntimeError if original_failure else OSError
    with pytest.raises(expected):
        run()
    persisted = json.loads(environment.path.read_text())
    assert persisted["lifecycle"] == "active"
    assert persisted["pending_restoration"]
    assert persisted["baseline"] == access.original.to_dict()


def test_seed_checksum_changes_and_unreachable_reporting_are_explicit(
    tmp_path: Path, png_screenshot: str
) -> None:
    access = _Access()
    specification = tmp_path / "seed.json"
    specification.write_text(json.dumps({"root": "/sdcard/seed", "files": access.hashes}))
    environment = _environment(tmp_path, png_screenshot, access, seed_manifest=str(specification))
    with environment:
        access.hashes["/sdcard/seed/a.txt"] = "b" * 64
    assert json.loads(environment.path.read_text())["cleanup"][-1]["seed_changes"]["changed"] == [
        "/sdcard/seed/a.txt"
    ]
    with environment:
        access.unreachable = True
    assert "seed_error" in json.loads(environment.path.read_text())["cleanup"][-1]


@pytest.mark.parametrize("root", ["/sdcard/", "/storage/emulated/0/"])
def test_manifest_scopes_a_subdirectory(tmp_path: Path, png_screenshot: str, root: str) -> None:
    specification = tmp_path / "seed.json"
    specification.write_text(json.dumps({"root": root, "files": {}}))
    environment = _environment(
        tmp_path, png_screenshot, _Access(), seed_manifest=str(specification)
    )
    with pytest.raises(ValueError, match="subdirectory"), environment:
        pytest.fail("an entire shared-storage tree was accepted as fake seed scope")


def test_matching_coordinate_replay_preserves_inputs_and_stable_observation_history(
    tmp_path: Path, png_screenshot: str
) -> None:
    access = _Access()
    controller = _ScreenDevice(png_screenshot)
    environment = RunEnvironment(
        controller, access, tmp_path / "demo.json", mutation={}, logger=logging.getLogger("test")
    )
    with environment:
        recorder = StepRecorder(
            environment, tmp_path / "steps", run_kind="explore", source="main_loop"
        )
        recorder.get_state()
        recorder.get_state()
        action: dict[str, object] = {"action": "tap", "x": 1, "y": 2}
        recorder.exe_action(action)
        with recorder.source("navigation_replay"):
            recorder.exe_action(json.loads(json.dumps(action)))
        assert controller.actions == [{"action": "tap", "x": 1, "y": 2}] * 2
    report = json.loads(environment.path.read_text())
    assert len(report["changes"]) == 1
    assert report["unsafe_replays"] == []


def test_native_frame_rotation_is_attributed_without_a_settings_override(
    tmp_path: Path, png_screenshot: str
) -> None:
    access = _Access()
    controller = _ScreenDevice(png_screenshot)
    environment = RunEnvironment(
        controller, access, tmp_path / "demo.json", mutation={}, logger=logging.getLogger("test")
    )
    with environment:
        recorder = StepRecorder(
            environment, tmp_path / "steps", run_kind="explore", source="main_loop"
        )
        recorder.get_state()
        buffer = io.BytesIO()
        Image.new("RGB", (64, 32), "white").save(buffer, format="PNG")
        controller.image = base64.b64encode(buffer.getvalue()).decode()
        after = recorder.get_state()
    changes = json.loads(environment.path.read_text())["changes"]
    assert changes[-1]["observation"] == after["observation_id"]
    assert changes[-1]["native_size"] == [64, 32]
    assert changes[0]["display"] == changes[1]["display"]


class _LateFontWrite(_Access):
    def __init__(self) -> None:
        super().__init__()
        self.restoring = False
        self.late_write = False

    @override
    def execute(self, command: tuple[str, ...], *, deadline: float) -> None:
        super().execute(command, deadline=deadline)
        self.restoring = True

    @override
    def capture(self, *, deadline: float | None = None) -> DisplayState:
        if self.restoring and not self.late_write:
            self.current = replace(self.current, font_scale="1.0")
            self.late_write = True
        return super().capture(deadline=deadline)


def test_cleanup_restores_absence_after_a_late_configuration_write(
    tmp_path: Path, png_screenshot: str
) -> None:
    access = _LateFontWrite()
    environment = _environment(tmp_path, png_screenshot, access)
    with environment:
        access.current = replace(
            access.current, size_override=(768, 1280), density_override=360, font_scale="1.15"
        )
    assert access.current == access.original
    report = json.loads(environment.path.read_text())
    assert report["cleanup"][-1]["status"] == "restored"


class _RetryFailure(_LateFontWrite):
    def __init__(self, fail_verification: bool) -> None:
        super().__init__()
        self.fail_verification = fail_verification

    @override
    def execute(self, command: tuple[str, ...], *, deadline: float) -> None:
        if self.late_write:
            message = "retry input unavailable"
            raise OSError(message)
        super().execute(command, deadline=deadline)

    @override
    def capture(self, *, deadline: float | None = None) -> DisplayState:
        if self.late_write and self.fail_verification:
            message = "retry verification unavailable"
            raise OSError(message)
        return super().capture(deadline=deadline)


@pytest.mark.parametrize("fail_verification", [False, True])
def test_retry_failure_keeps_pending_commands(
    tmp_path: Path, png_screenshot: str, fail_verification: bool
) -> None:
    access = _RetryFailure(fail_verification)
    environment = _environment(tmp_path, png_screenshot, access)
    with environment:
        access.current = replace(access.current, font_scale="1.25")
    cleanup = json.loads(environment.path.read_text())["cleanup"][-1]
    assert cleanup["status"] == "pending"
    assert cleanup["pending_commands"]
    assert any(failure.get("attempt") == 2 for failure in cleanup["failures"])


class _UnrestorableFlags(_Access):
    @override
    def capture(self, *, deadline: float | None = None) -> DisplayState:
        state = super().capture(deadline=deadline)
        return replace(state, night_overrides=(True, False)) if self.commands else state


def test_unsupported_temporary_activation_remains_pending(
    tmp_path: Path, png_screenshot: str
) -> None:
    access = _UnrestorableFlags()
    environment = _environment(tmp_path, png_screenshot, access)
    with environment:
        pass
    cleanup = json.loads(environment.path.read_text())["cleanup"][-1]
    assert cleanup["status"] == "pending"
    assert not any(operation.get("attempt") == 2 for operation in cleanup["operations"])
    assert cleanup["verification"]["night_overrides"]["status"] == "pending"


def test_resumed_seed_reporting_keeps_the_public_scope_boundary(
    tmp_path: Path, png_screenshot: str
) -> None:
    access = _Access()
    specification = tmp_path / "seed.json"
    specification.write_text(json.dumps({"root": "/sdcard/seed", "files": access.hashes}))
    environment = _environment(tmp_path, png_screenshot, access, seed_manifest=str(specification))
    with environment:
        pass
    report = json.loads(environment.path.read_text())
    report["seed"]["root"] = "/data/user/0/private"
    environment.path.write_text(json.dumps(report))
    with environment:
        pass
    assert all(root == "/sdcard/seed" for root in access.seed_reads)
    assert "public-storage" in json.loads(environment.path.read_text())["cleanup"][-1]["seed_error"]


@pytest.mark.usefixtures("no_sleep", "opened_clients")
def test_cli_reports_a_missing_baseline_as_an_operator_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    script = _Script(tmp_path, None)
    _configure_cli(monkeypatch, tmp_path, script)
    script.graph.parent.mkdir(parents=True)
    script.graph.write_text("{}")
    with pytest.raises(SystemExit, match=r"app-graph: error:.*original run baseline"):
        cli.main()
    assert not script.graph.with_name("demo_run.json").exists()
    assert script.graph.read_text() == "{}"
