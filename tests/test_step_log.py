"""Tests for android_app_graph.step_log: the recorder writes every screenshot and
action of the controller it wraps into the step log, and callers add transitions."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from android_app_graph.device import DeviceController
from android_app_graph.screen_image import encode_screenshot
from android_app_graph.step_log import Observation, StepRecorder, Transition

HOME = {"activity": "com.demo.Home", "package": "com.demo"}


def _png(size: tuple[int, int], color: str) -> str:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class _Device:
    """A device that shows one screen and logs the actions it is sent."""

    w = 1080
    h = 1920

    def __init__(self, screenshot: str) -> None:
        self.config: dict[str, Any] = {"device": {"udid": "test-device"}}
        self.screenshot = screenshot
        self.actions: list[dict[str, Any]] = []

    def get_state(self) -> dict[str, Any]:
        return {**HOME, "xml": "<hierarchy/>", "screenshot": self.screenshot}

    def exe_action(self, action: dict[str, Any]) -> None:
        self.actions.append(action)


@pytest.fixture
def device() -> _Device:
    return _Device(_png((40, 30), "white"))


@pytest.fixture
def steps(tmp_path: Path) -> Path:
    return tmp_path / "demo" / "steps"


def _recorder(device: DeviceController, steps: Path) -> StepRecorder:
    return StepRecorder(device, steps, run_kind="explore", source="main_loop")


def _record(steps: Path, step_id: str) -> dict[str, Any]:
    return json.loads((steps / f"{step_id}.json").read_text(encoding="utf-8"))


def _files(steps: Path) -> list[str]:
    return sorted(path.name for path in steps.iterdir())


def test_the_recorder_stands_in_for_the_controller_it_wraps(device: _Device, steps: Path) -> None:
    recorder: DeviceController = _recorder(device, steps)

    assert (recorder.w, recorder.h) == (1080, 1920)
    assert recorder.config is device.config


def test_get_state_records_one_webp_image_and_one_observation(device: _Device, steps: Path) -> None:
    recorder = _recorder(device, steps)

    state = recorder.get_state()

    webp = encode_screenshot(device.screenshot)
    assert _files(steps) == ["000000.json", "000000.webp"]
    assert (steps / "000000.webp").read_bytes() == webp
    assert state == {
        **HOME,
        "xml": "<hierarchy/>",
        "screenshot": base64.b64encode(webp).decode("ascii"),
        "observation_id": "000000",
    }
    record = _record(steps, "000000")
    timestamp = record.pop("timestamp")
    assert timestamp.endswith("+00:00")
    assert record == {
        "id": "000000",
        "kind": "observation",
        "run_id": recorder.run_id,
        "run_kind": "explore",
        "source": "main_loop",
        **HOME,
        "width": 40,
        "height": 30,
    }


def test_exe_action_records_the_action_with_the_observation_it_was_taken_from(
    device: _Device, steps: Path
) -> None:
    recorder = _recorder(device, steps)
    recorder.get_state()
    observation_id = recorder.get_state()["observation_id"]

    recorder.exe_action({"action": "back"})

    assert device.actions == [{"action": "back"}]
    record = _record(steps, "000002")
    assert record["kind"] == "action"
    assert record["observation"] == observation_id == "000001"
    assert record["action"] == {"action": "back"}
    assert (record["activity"], record["package"]) == (HOME["activity"], HOME["package"])
    assert not (steps / "000002.webp").exists()


def test_an_action_before_any_observation_names_none(device: _Device, steps: Path) -> None:
    """A replay starts acting right after an app restart, before the run has looked."""
    _recorder(device, steps).exe_action({"action": "tap", "x": 1, "y": 2})

    record = _record(steps, "000000")
    assert (record["observation"], record["activity"], record["package"]) == (None, None, None)
    assert device.actions == [{"action": "tap", "x": 1, "y": 2}]


def test_a_second_recorder_continues_the_numbering_as_another_run(
    device: _Device, steps: Path
) -> None:
    first = _recorder(device, steps)
    first.get_state()
    first.exe_action({"action": "back"})

    second = StepRecorder(device, steps, run_kind="audit", source="re_explore")
    state = second.get_state()

    assert state["observation_id"] == "000002"
    record = _record(steps, "000002")
    assert record["run_id"] == second.run_id != first.run_id
    assert (record["run_kind"], record["source"]) == ("audit", "re_explore")


def test_an_image_without_a_record_is_ignored_and_its_number_reused(
    device: _Device, steps: Path
) -> None:
    """A crash between publishing an image and its record leaves the image alone; the
    next run takes that number again and replaces the image."""
    _recorder(device, steps).get_state()
    (steps / "000001.webp").write_bytes(b"left by a crash")

    state = _recorder(device, steps).get_state()

    assert state["observation_id"] == "000001"
    assert (steps / "000001.webp").read_bytes() == encode_screenshot(device.screenshot)
    assert _record(steps, "000001")["kind"] == "observation"


def test_only_numbered_records_count_towards_the_next_number(device: _Device, steps: Path) -> None:
    steps.mkdir(parents=True)
    (steps / "000004.json").write_text("{}", encoding="utf-8")
    (steps / "notes.json").write_text("{}", encoding="utf-8")
    (steps / "000009.webp").write_bytes(b"image without a record")

    assert _recorder(device, steps).get_state()["observation_id"] == "000005"


def test_source_tags_the_records_inside_the_block_and_restores_the_outer_one(
    device: _Device, steps: Path
) -> None:
    recorder = _recorder(device, steps)

    with recorder.source("navigation_replay"):
        recorder.exe_action({"action": "back"})
        with recorder.source("external_web_recovery"):
            recorder.get_state()
        recorder.get_state()
    recorder.get_state()

    sources = [_record(steps, f"{n:06d}")["source"] for n in range(4)]
    assert sources == [
        "navigation_replay",
        "external_web_recovery",
        "navigation_replay",
        "main_loop",
    ]


def test_source_restores_the_outer_source_when_the_block_raises(
    device: _Device, steps: Path
) -> None:
    recorder = _recorder(device, steps)
    error = "replay failed"

    with pytest.raises(RuntimeError, match=error), recorder.source("navigation_replay"):
        raise RuntimeError(error)
    recorder.get_state()

    assert _record(steps, "000000")["source"] == "main_loop"


def test_record_transition_names_both_screens_the_plan_and_the_nodes(
    device: _Device, steps: Path
) -> None:
    recorder = _recorder(device, steps)
    before = Observation.from_state(recorder.get_state())
    recorder.exe_action({"action": "tap", "x": 5, "y": 6})
    after = Observation.from_state(recorder.get_state())

    recorder.record_transition(
        Transition(
            before=before,
            after=after,
            instruction="Open search",
            element=2,
            action={"action": "tap", "x": 5, "y": 6},
            from_node="s0_home",
            to_node="s1_search",
        )
    )

    record = _record(steps, "000003")
    record.pop("timestamp")
    assert record == {
        "id": "000003",
        "kind": "transition",
        "run_id": recorder.run_id,
        "run_kind": "explore",
        "source": "main_loop",
        **HOME,
        "before": "000000",
        "after": "000002",
        "instruction": "Open search",
        "element": 2,
        "action": {"action": "tap", "x": 5, "y": 6},
        "from_node": "s0_home",
        "to_node": "s1_search",
    }


def test_observation_from_state_reads_a_recorded_state(device: _Device, steps: Path) -> None:
    state = _recorder(device, steps).get_state()

    assert Observation.from_state(state) == Observation(
        observation_id="000000",
        screenshot_b64=state["screenshot"],
        activity=HOME["activity"],
        package=HOME["package"],
    )


def test_observation_from_state_rejects_a_state_the_recorder_did_not_return(
    device: _Device,
) -> None:
    with pytest.raises(ValueError, match="no observation ID"):
        Observation.from_state(device.get_state())
