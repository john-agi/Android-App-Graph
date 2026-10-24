"""Tests for android_app_graph.step_log: the recorder writes every screenshot and
action of the controller it wraps into the step log, callers add match requests and
transitions, and the step log keeps its observations' image vectors."""

from __future__ import annotations

import base64
import io
import json
import logging
from pathlib import Path
from typing import Any

import pytest
from PIL import Image, ImageDraw

from android_app_graph.device import DeviceController, InputRegions, InputRejectedError
from android_app_graph.screen_image import encode_screenshot, pixel_hash
from android_app_graph.step_log import (
    Observation,
    StepRecorder,
    Transition,
    TransitionRecord,
    load_image_embeddings,
    read_exploration_transitions,
    read_observation_record,
    save_image_embeddings,
)

HOME = {"activity": "com.demo.Home", "package": "com.demo"}


def _encoded(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _png(size: tuple[int, int], color: str) -> str:
    return _encoded(Image.new("RGB", size, color))


class _Device:
    """A device that shows one screen and logs the actions it is sent."""

    w = 1080
    h = 1920

    def __init__(self, screenshot: str, status_bar: object = None) -> None:
        self.config: dict[str, Any] = {"device": {"udid": "test-device"}}
        self.screenshot = screenshot
        self.status_bar = status_bar
        self.actions: list[dict[str, Any]] = []
        self.deadlines: list[float | None] = []

    def get_state(self, *, deadline: float | None = None) -> dict[str, Any]:
        self.deadlines.append(deadline)
        state = {**HOME, "xml": "<hierarchy/>", "screenshot": self.screenshot}
        return state if self.status_bar is None else {**state, "status_bar": self.status_bar}

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


def test_action_grounding_uses_the_current_native_screenshot_dimensions(steps: Path) -> None:
    controller = _Device(_png((720, 1280), "white"))
    recorder = _recorder(controller, steps)
    recorder.get_state()
    assert (recorder.w, recorder.h) == (720, 1280)

    controller.screenshot = _png((1280, 720), "white")
    recorder.get_state()
    assert (recorder.w, recorder.h) == (1280, 720)


def test_rejected_input_records_its_request_geometry_and_reason(
    device: _Device, steps: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = _recorder(device, steps)
    recorder.get_state()
    regions = InputRegions((0, 0, 720, 1280), ((0, 1184, 720, 1280),), ())
    reason = "App input targets a system touch region"

    def reject(_action: dict[str, object]) -> None:
        raise InputRejectedError(reason, regions)

    monkeypatch.setattr(device, "exe_action", reject)
    action = {"action": "tap", "x": 257, "y": 1190}
    with pytest.raises(InputRejectedError, match=reason):
        recorder.exe_action(action)
    assert device.actions == []
    assert _record(steps, "000001")["action"] == action
    rejected = _record(steps, "000002")
    assert (rejected["kind"], rejected["action_record"], rejected["action"]) == (
        "action_rejected",
        "000001",
        action,
    )
    assert (rejected["reason"], rejected["input_regions"]) == (reason, regions.as_dict())


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
        "pixel_hash": pixel_hash(webp, 0),
        "black": False,
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
        "status_bar": None,
        "status_bar_rows": 0,
        "pixel_hash": pixel_hash(webp, 0),
        "black": False,
    }


def test_a_black_screenshot_is_recorded_as_black(steps: Path) -> None:
    state = _recorder(_Device(_png((40, 30), "black")), steps).get_state()

    assert state["black"] is True
    assert _record(steps, "000000")["black"] is True
    assert Observation.from_state(state).black is True


def _screen(size: tuple[int, int]) -> Image.Image:
    """A screen whose every channel varies from pixel to pixel, so no change hides."""
    width, height = size
    return Image.merge(
        "RGB",
        (
            Image.effect_noise(size, 64),
            Image.linear_gradient("L").resize(size),
            Image.linear_gradient("L").rotate(90).resize((width, height)),
        ),
    )


def _with_clock(screen: Image.Image, minute: int, *, clock_row: int) -> str:
    """``screen`` with a "clock", ``minute`` pixels wide, whose last row is ``clock_row``."""
    clocked = screen.copy()
    ImageDraw.Draw(clocked).rectangle((8, clock_row - 6, 8 + minute, clock_row), fill="red")
    return _encoded(clocked)


def test_a_screenshot_over_a_million_pixels_drops_the_band_and_the_rows_lanczos_blended(
    steps: Path,
) -> None:
    """Two captures that differ only in the status bar's last row hash alike once
    encoding has shrunk them: 1250x1000 becomes 1118x894, and a 60-row bar becomes
    ceil(60 * 894 / 1000) + 3 = 57 rows."""
    screen, bar = _screen((1250, 1000)), [0, 0, 1250, 60]
    hashes = [
        _recorder(_Device(_with_clock(screen, minute, clock_row=59), bar), steps).get_state()[
            "pixel_hash"
        ]
        for minute in (1, 30)
    ]

    assert hashes[0] == hashes[1]
    assert (steps / "000000.webp").read_bytes() != (steps / "000001.webp").read_bytes()
    record = _record(steps, "000000")
    assert (record["status_bar"], record["status_bar_rows"], record["height"]) == (bar, 57, 894)
    assert record["pixel_hash"] == hashes[0]


def test_a_screenshot_that_differs_below_the_band_hashes_apart(steps: Path) -> None:
    screen, bar = _screen((400, 300)), [0, 0, 400, 20]
    hashes = [
        _recorder(_Device(_with_clock(screen, minute, clock_row=24), bar), steps).get_state()[
            "pixel_hash"
        ]
        for minute in (1, 2)
    ]

    assert hashes[0] != hashes[1]
    assert _record(steps, "000000")["status_bar_rows"] == 20


@pytest.mark.parametrize("status_bar", [None, [0, 0, 400], "0,0,400,20"])
def test_a_controller_without_a_usable_status_bar_has_the_whole_image_hashed(
    status_bar: object, steps: Path
) -> None:
    device = _Device(_with_clock(_screen((400, 300)), 1, clock_row=10), status_bar)

    state = _recorder(device, steps).get_state()

    webp = (steps / "000000.webp").read_bytes()
    assert state["pixel_hash"] == pixel_hash(webp, 0)
    assert (
        _record(steps, "000000")["status_bar"],
        _record(steps, "000000")["status_bar_rows"],
    ) == (
        None,
        0,
    )


@pytest.mark.parametrize("action", [{"action": "back"}, {"action": "menu"}])
def test_exe_action_records_the_action_with_the_observation_it_was_taken_from(
    device: _Device, steps: Path, action: dict[str, str]
) -> None:
    recorder = _recorder(device, steps)
    recorder.get_state()
    observation_id = recorder.get_state()["observation_id"]

    recorder.exe_action(action)

    assert device.actions == [action]
    record = _record(steps, "000002")
    assert record["kind"] == "action"
    assert record["observation"] == observation_id == "000001"
    assert record["action"] == action
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


def _transition(recorder: StepRecorder, action: dict[str, Any], to_node: str) -> None:
    before = Observation.from_state(recorder.get_state())
    recorder.exe_action(action)
    after = Observation.from_state(recorder.get_state())
    recorder.record_transition(
        Transition(
            before=before,
            after=after,
            instruction="",
            element=None,
            action=action,
            from_node="s0_home",
            to_node=to_node,
        )
    )


def test_the_exploration_transitions_are_read_in_step_order(device: _Device, steps: Path) -> None:
    """The audit's re-exploration records transitions against the audited copy, so
    only exploration runs' transitions are read."""
    explore = _recorder(device, steps)
    for number in range(10):
        _transition(explore, {"action": "tap", "x": number, "y": 0}, f"s{number}_page")
    _transition(
        StepRecorder(device, steps, run_kind="audit", source="re_explore"),
        {"action": "back"},
        "s0_home",
    )

    transitions = read_exploration_transitions(steps)

    assert [t.to_node for t in transitions] == [f"s{number}_page" for number in range(10)]
    assert transitions[0] == TransitionRecord(
        before="000000",
        after="000002",
        action={"action": "tap", "x": 0, "y": 0},
        from_node="s0_home",
        to_node="s0_page",
    )


def test_a_step_log_that_does_not_exist_has_no_transitions(steps: Path) -> None:
    assert read_exploration_transitions(steps) == []


def test_observation_from_state_reads_a_recorded_state(device: _Device, steps: Path) -> None:
    state = _recorder(device, steps).get_state()

    assert Observation.from_state(state) == Observation(
        observation_id="000000",
        screenshot_b64=state["screenshot"],
        activity=HOME["activity"],
        package=HOME["package"],
        pixel_hash=state["pixel_hash"],
        black=False,
    )


def test_observation_from_state_rejects_a_state_the_recorder_did_not_return(
    device: _Device,
) -> None:
    with pytest.raises(ValueError, match="no observation ID"):
        Observation.from_state(device.get_state())


def test_record_match_names_the_screenshot_asked_for_and_the_one_expected(
    device: _Device, steps: Path
) -> None:
    recorder = _recorder(device, steps)
    observation = Observation.from_state(recorder.get_state())
    assert recorder.last_match is None

    with recorder.source("navigation_replay"):
        recorder.record_match(observation, "000042")

    record = _record(steps, "000001")
    record.pop("timestamp")
    assert record == {
        "id": "000001",
        "kind": "match",
        "run_id": recorder.run_id,
        "run_kind": "explore",
        "source": "navigation_replay",
        **HOME,
        "observation": "000000",
        "expected_observation": "000042",
    }
    assert recorder.last_match == "000000"


def test_a_match_without_an_expectation_names_none(device: _Device, steps: Path) -> None:
    recorder = _recorder(device, steps)

    recorder.record_match(Observation.from_state(recorder.get_state()), None)

    assert _record(steps, "000001")["expected_observation"] is None


def test_read_observation_record_reads_a_record_or_none(device: _Device, steps: Path) -> None:
    _recorder(device, steps).get_state()

    assert read_observation_record(steps, "000000") == _record(steps, "000000")
    assert read_observation_record(steps, "000001") is None


MODEL = "voyageai/voyage-multimodal-3.5"


def test_image_vectors_round_trip_with_their_model_tag(steps: Path) -> None:
    steps.mkdir(parents=True)
    assert load_image_embeddings(steps, model=MODEL) == {}

    save_image_embeddings(steps, {"000000": [0.5, 0.25]}, model=MODEL)

    assert json.loads((steps / "image_emb.json").read_text(encoding="utf-8")) == {
        "model": MODEL,
        "embeddings": {"000000": [0.5, 0.25]},
    }
    assert load_image_embeddings(steps, model=MODEL) == {"000000": [0.5, 0.25]}


def test_a_vector_file_from_another_model_is_empty_after_one_warning(
    steps: Path, caplog: pytest.LogCaptureFixture
) -> None:
    steps.mkdir(parents=True)
    save_image_embeddings(steps, {"000000": [0.5]}, model="other/model")

    with caplog.at_level(logging.WARNING):
        assert load_image_embeddings(steps, model=MODEL) == {}

    (warning,) = caplog.messages
    assert "image_emb.json was written by model 'other/model'" in warning


@pytest.mark.parametrize(
    ("content", "damage"),
    [
        pytest.param(json.dumps({"000000": [0.5]}), "is not tagged", id="no tag"),
        pytest.param(
            json.dumps({"model": [1.0], "embeddings": {"000000": [0.5]}}),
            "is not tagged",
            id="bad tag",
        ),
        pytest.param("null", "is not tagged", id="not an object"),
        pytest.param(json.dumps({"model": MODEL}), "has no embeddings object", id="no embeddings"),
        pytest.param(
            json.dumps({"model": MODEL, "embeddings": [[0.5]]}),
            "has no embeddings object",
            id="embeddings not an object",
        ),
        pytest.param("{not json", "cannot be read or parsed", id="corrupt"),
        pytest.param(b'{"model": "x"}\xff', "cannot be read or parsed", id="not utf-8"),
    ],
)
def test_a_damaged_vector_file_raises_naming_the_file_and_the_damage(
    content: str | bytes, damage: str, steps: Path
) -> None:
    steps.mkdir(parents=True)
    path = steps / "image_emb.json"
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match=r"image_emb\.json") as raised:
        load_image_embeddings(steps, model=MODEL)

    assert f"Image embedding file {path} {damage}" in str(raised.value)
    assert str(raised.value).endswith(". Delete it to embed every screenshot again.")


def test_a_malformed_vector_raises_naming_every_such_observation(steps: Path) -> None:
    steps.mkdir(parents=True)
    path = steps / "image_emb.json"
    vectors = {"000000": [1.0, 2.0], "000001": "not a vector", "000002": [], "000003": [1, "x"]}
    path.write_text(
        json.dumps({"model": MODEL, "embeddings": vectors}).replace('[1, "x"]', "[1, NaN]"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"image_emb\.json") as raised:
        load_image_embeddings(steps, model=MODEL)

    assert str(raised.value) == (
        f"Image embedding file {path} has a malformed or non-finite vector for observation(s) "
        "000001, 000002, 000003. Delete it to embed every screenshot again."
    )


def test_observation_deadline_reaches_the_wrapped_device(device: _Device, steps: Path) -> None:
    _recorder(device, steps).get_state(deadline=123)
    assert device.deadlines == [123]
