"""Tests for android_app_graph.step_log: the recorder writes every screenshot, tree
and action of the controller it wraps into the step log, callers add transitions and
annotate observations, and the step log keeps its observations' image vectors."""

from __future__ import annotations

import base64
import io
import json
import logging
from pathlib import Path
from typing import Any

import pytest
from PIL import Image, ImageDraw

from android_app_graph import step_log
from android_app_graph.device import DeviceController
from android_app_graph.screen_image import encode_screenshot, pixel_hash
from android_app_graph.step_log import (
    ActionEffect,
    ActionHit,
    MisTap,
    Observation,
    StepRecorder,
    Transition,
    action_effect,
    annotate_observation,
    load_image_embeddings,
    read_observation_record,
    save_image_embeddings,
)
from android_app_graph.ui_tree import (
    IDLE_FAILURE,
    Anchor,
    Bounds,
    Locator,
    UiTree,
    actionable_elements,
    parse_tree,
    xml_document,
)

HOME = {"activity": "com.demo.Home", "package": "com.demo"}
# The tree AdbController keeps for a settled capture: a trimmed dump of Settings' home.
DOCUMENT = xml_document(
    (Path(__file__).parent / "data" / "uiautomator_windows_plain_list.txt").read_text(
        encoding="utf-8"
    )
)
CAPTURED: dict[str, object] = {
    "xml": DOCUMENT,
    "ui_tree_error": None,
    "ui_tree_retakes": 0,
    "ui_tree_settled": True,
}


def _encoded(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _png(size: tuple[int, int], color: str) -> str:
    return _encoded(Image.new("RGB", size, color))


class _Device:
    """A device that shows one screen, captured as ``capture`` says, and logs the actions
    it is sent."""

    w = 1080
    h = 1920

    def __init__(
        self,
        screenshot: str,
        status_bar: object = None,
        capture: dict[str, object] | None = None,
    ) -> None:
        self.config: dict[str, Any] = {"device": {"udid": "test-device"}}
        self.screenshot = screenshot
        self.status_bar = status_bar
        self.capture = CAPTURED if capture is None else capture
        self.actions: list[dict[str, Any]] = []

    def get_state(self) -> dict[str, Any]:
        state = {**HOME, **self.capture, "screenshot": self.screenshot}
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


def test_get_state_records_one_webp_image_one_tree_and_one_observation(
    device: _Device, steps: Path
) -> None:
    recorder = _recorder(device, steps)

    state = recorder.get_state()

    webp = encode_screenshot(device.screenshot)
    assert _files(steps) == ["000000.json", "000000.webp", "000000.xml"]
    assert (steps / "000000.webp").read_bytes() == webp
    assert (steps / "000000.xml").read_text(encoding="utf-8") == DOCUMENT
    assert state == {
        **HOME,
        "ui_tree_error": None,
        "ui_tree_retakes": 0,
        "ui_tree_settled": True,
        "screenshot": base64.b64encode(webp).decode("ascii"),
        "observation_id": "000000",
        "pixel_hash": pixel_hash(webp, 0),
        "screen_size": [40, 30],
        "ui_tree": parse_tree(DOCUMENT),
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
        "ui_tree_error": None,
        "ui_tree_retakes": 0,
        "ui_tree_settled": True,
    }


def test_get_state_writes_the_image_the_tree_and_the_record_in_that_order(
    device: _Device, steps: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every record names files that exist, even after a crash between two writes."""
    written: list[str] = []
    write_bytes, write_json = step_log.write_bytes_atomically, step_log.write_json_atomically

    def bytes_then_log(path: Path, data: bytes) -> None:
        write_bytes(path, data)
        written.append(path.name)

    def json_then_log(
        path: Path, payload: object, *, indent: int | None = None, ensure_ascii: bool = True
    ) -> None:
        write_json(path, payload, indent=indent, ensure_ascii=ensure_ascii)
        written.append(path.name)

    monkeypatch.setattr(step_log, "write_bytes_atomically", bytes_then_log)
    monkeypatch.setattr(step_log, "write_json_atomically", json_then_log)

    _recorder(device, steps).get_state()

    assert written == ["000000.webp", "000000.xml", "000000.json"]


def test_a_retaken_capture_records_its_retakes_and_whether_it_settled(steps: Path) -> None:
    capture: dict[str, object] = {**CAPTURED, "ui_tree_retakes": 2, "ui_tree_settled": False}

    _recorder(_Device(_png((40, 30), "white"), capture=capture), steps).get_state()

    record = _record(steps, "000000")
    assert (record["ui_tree_error"], record["ui_tree_retakes"], record["ui_tree_settled"]) == (
        None,
        2,
        False,
    )


@pytest.mark.parametrize(
    ("capture", "reason"),
    [
        pytest.param(
            {"xml": None, "ui_tree_error": IDLE_FAILURE, "ui_tree_retakes": 0},
            IDLE_FAILURE,
            id="the controller's reason",
        ),
        pytest.param(
            {"xml": "<hierarchy/>"}, "root is <hierarchy>, not <displays>", id="unusable xml"
        ),
        pytest.param({}, "the device state has no tree", id="no xml"),
    ],
)
def test_a_capture_without_a_tree_writes_no_tree_and_records_why(
    capture: dict[str, object], reason: str, steps: Path
) -> None:
    state = _recorder(_Device(_png((40, 30), "white"), capture=capture), steps).get_state()

    assert _files(steps) == ["000000.json", "000000.webp"]
    assert (state["ui_tree"], state["ui_tree_error"]) == (None, reason)
    assert "xml" not in state
    record = _record(steps, "000000")
    assert (record["ui_tree_error"], record["ui_tree_retakes"], record["ui_tree_settled"]) == (
        reason,
        0,
        False,
    )


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


def test_an_image_and_a_tree_without_a_record_are_ignored_and_their_number_reused(
    device: _Device, steps: Path
) -> None:
    """A crash between publishing an image or a tree and its record leaves them alone;
    the next run takes that number again and replaces them."""
    _recorder(device, steps).get_state()
    (steps / "000001.webp").write_bytes(b"left by a crash")
    (steps / "000001.xml").write_bytes(b"left by a crash")
    (steps / "000002.xml").write_bytes(b"left by a crash")

    state = _recorder(device, steps).get_state()

    assert state["observation_id"] == "000001"
    assert (steps / "000001.webp").read_bytes() == encode_screenshot(device.screenshot)
    assert (steps / "000001.xml").read_text(encoding="utf-8") == DOCUMENT
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


_SEARCH = Locator("com.demo:id/search", "android.widget.Button", "Search", ("android:id/content",))
_MENU = Locator("com.demo:id/menu", "android.widget.ImageButton", "More", ("android:id/content",))


@pytest.mark.parametrize(
    ("element_locator", "recorded"),
    [
        pytest.param(
            _SEARCH,
            {
                "resource_id": "com.demo:id/search",
                "class": "android.widget.Button",
                "label": "Search",
                "ancestor_ids": ["android:id/content"],
            },
            id="a tree element",
        ),
        pytest.param(None, None, id="an element the model found"),
    ],
)
def test_record_transition_names_both_screens_the_plan_and_the_nodes(
    element_locator: Locator | None,
    recorded: dict[str, object] | None,
    device: _Device,
    steps: Path,
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
            element_locator=element_locator,
            action={"action": "tap", "x": 5, "y": 6},
            effect=ActionEffect(
                (ActionHit("element", Anchor(_MENU, Bounds(0, 0, 10, 12))),),
                MisTap(_SEARCH, _MENU),
                no_visible_effect=False,
            ),
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
        "element_locator": recorded,
        "action": {"action": "tap", "x": 5, "y": 6},
        "anchors": [{"locator": _MENU.as_record(), "bounds": [0, 0, 10, 12]}],
        "hits": ["element"],
        "action_flags": [{"mis_tap": {"planned": _SEARCH.as_record(), "hit": _MENU.as_record()}}],
        "from_node": "s0_home",
        "to_node": "s1_search",
    }


@pytest.mark.parametrize(
    ("capture", "tree", "reason", "settled"),
    [
        pytest.param(CAPTURED, parse_tree(DOCUMENT), None, True, id="a tree"),
        pytest.param(
            {"xml": None, "ui_tree_error": IDLE_FAILURE}, None, IDLE_FAILURE, False, id="none"
        ),
    ],
)
def test_observation_from_state_reads_a_recorded_state_with_its_tree_or_the_reason(
    capture: dict[str, object],
    tree: UiTree | None,
    reason: str | None,
    settled: bool,
    steps: Path,
) -> None:
    state = _recorder(_Device(_png((40, 30), "white"), capture=capture), steps).get_state()

    assert Observation.from_state(state) == Observation(
        observation_id="000000",
        screenshot_b64=state["screenshot"],
        activity=HOME["activity"],
        package=HOME["package"],
        pixel_hash=state["pixel_hash"],
        screen_size=(40, 30),
        ui_tree=tree,
        ui_tree_error=reason,
        ui_tree_settled=settled,
    )


def test_the_screen_size_is_the_screenshot_s_before_it_is_shrunk(steps: Path) -> None:
    """The tree's bounds are in the device's pixels, which the WebP image no longer has."""
    state = _recorder(_Device(_png((1344, 2992), "white")), steps).get_state()

    assert state["screen_size"] == [1344, 2992]
    assert Observation.from_state(state).screen_size == (1344, 2992)
    assert _record(steps, "000000")["height"] < 2992


@pytest.mark.parametrize("size", [None, [1344], ["wide", "tall"]])
def test_a_state_without_a_screen_size_reads_as_no_area(size: object, steps: Path) -> None:
    state = _recorder(_Device(_png((40, 30), "white")), steps).get_state()
    state["screen_size"] = size

    assert Observation.from_state(state).screen_size == (0, 0)


def test_observation_from_state_rejects_a_state_the_recorder_did_not_return(
    device: _Device,
) -> None:
    with pytest.raises(ValueError, match="no observation ID"):
        Observation.from_state(device.get_state())


def test_annotate_observation_adds_fields_and_keeps_the_record(
    device: _Device, steps: Path
) -> None:
    recorder = _recorder(device, steps)
    recorder.get_state()
    before = _record(steps, "000000")

    annotate_observation(steps, "000000", {"describe": {"page_description": "Home"}})

    assert _record(steps, "000000") == {**before, "describe": {"page_description": "Home"}}
    assert read_observation_record(steps, "000000") == _record(steps, "000000")


def test_annotating_an_observation_without_a_record_raises(steps: Path) -> None:
    steps.mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        annotate_observation(steps, "000000", {"identify": {}})
    assert read_observation_record(steps, "000000") is None


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


@pytest.mark.parametrize(
    ("content", "warning"),
    [
        pytest.param(
            json.dumps({"model": "other/model", "embeddings": {"000000": [0.5]}}),
            "written by model 'other/model'",
            id="another model",
        ),
        pytest.param(json.dumps({"000000": [0.5]}), "not tagged", id="no tag"),
        pytest.param(
            json.dumps({"model": [1.0], "embeddings": {"000000": [0.5]}}),
            "not tagged",
            id="bad tag",
        ),
        pytest.param("null", "not tagged", id="not an object"),
        pytest.param("{not json", "Corrupt", id="corrupt"),
        pytest.param(b'{"model": "x"}\xff', "Corrupt", id="not utf-8"),
    ],
)
def test_an_unusable_vector_file_is_empty_after_one_warning(
    content: str | bytes, warning: str, steps: Path, caplog: pytest.LogCaptureFixture
) -> None:
    steps.mkdir(parents=True)
    path = steps / "image_emb.json"
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        assert load_image_embeddings(steps, model=MODEL) == {}

    assert len(caplog.records) == 1
    assert warning in caplog.text
    assert "image_emb.json" in caplog.text


def test_a_malformed_vector_is_dropped_with_a_warning(
    steps: Path, caplog: pytest.LogCaptureFixture
) -> None:
    steps.mkdir(parents=True)
    vectors = {"000000": [1.0, 2.0], "000001": "not a vector", "000002": [], "000003": [1, "x"]}
    (steps / "image_emb.json").write_text(
        json.dumps({"model": MODEL, "embeddings": vectors}).replace('[1, "x"]', "[1, NaN]"),
        encoding="utf-8",
    )

    with caplog.at_level(logging.WARNING):
        assert load_image_embeddings(steps, model=MODEL) == {"000000": [1.0, 2.0]}

    dropped = [message.split("observation ")[1].split()[0] for message in caplog.messages]
    assert dropped == ["000001", "000002", "000003"]


def _fixture(name: str) -> str:
    return (Path(__file__).parent / "data" / f"uiautomator_{name}.txt").read_text("utf-8")


_SETTINGS = "com.android.settings"
_SEARCH_APP = "com.google.android.settings.intelligence"
# The screen of the uiautomator fixtures, a Pixel 8 Pro emulator's.
_FIXTURE_SIZE = (1344, 2992)


def _observed(
    dump: str | None = "windows_plain_list",
    *,
    settled: bool = True,
    pixels: str = "pixels-before",
    package: str = _SETTINGS,
) -> Observation:
    """An observation of a fixture's screen, or of a screen without a tree."""
    return Observation(
        observation_id="000000",
        screenshot_b64="",
        activity="",
        package=package,
        pixel_hash=pixels,
        screen_size=_FIXTURE_SIZE,
        ui_tree=parse_tree(_fixture(dump)) if dump is not None else None,
        ui_tree_error=None if dump is not None else IDLE_FAILURE,
        ui_tree_settled=settled,
    )


def _settings_row(label: str) -> Locator:
    """The locator of a row of Settings' home fixture."""
    tree = parse_tree(_fixture("windows_plain_list"))
    return next(
        element.locator
        for element in actionable_elements(tree, _SETTINGS, Bounds(0, 0, *_FIXTURE_SIZE))
        if element.label == label
    )


_NETWORK_TITLE: dict[str, object] = {"action": "tap", "x": 400, "y": 780}
_AFTER = _observed(pixels="pixels-after")


def _tap_effect(
    before: Observation, action: dict[str, object], planned: Locator | None
) -> ActionEffect:
    return action_effect(before, _AFTER, [action], planned=planned, self_loop=False)


@pytest.mark.parametrize("kind", ["tap", "long_press"])
def test_a_tap_on_the_planned_element_records_its_anchor_and_no_flag(kind: str) -> None:
    network = _settings_row("Network & internet")

    effect = _tap_effect(_observed(), {**_NETWORK_TITLE, "action": kind}, network)

    assert effect == ActionEffect(
        (ActionHit("element", Anchor(network, Bounds(0, 699, 1344, 917))),),
        mis_tap=None,
        no_visible_effect=False,
    )
    assert effect.anchor_records() == [
        {"locator": network.as_record(), "bounds": [0, 699, 1344, 917]}
    ]
    assert effect.flag_records() == [None]


def test_a_tap_on_another_element_than_the_planned_one_is_a_mis_tap_naming_both() -> None:
    apps, network = _settings_row("Apps"), _settings_row("Network & internet")

    effect = _tap_effect(_observed(), _NETWORK_TITLE, apps)

    assert effect.mis_tap == MisTap(apps, network)
    assert effect.flag_records() == [
        {"mis_tap": {"planned": apps.as_record(), "hit": network.as_record()}}
    ]


@pytest.mark.parametrize(
    ("dump", "point", "hit"),
    [
        pytest.param("windows_dialog_compose", (672, 600), "outside_dialog", id="outside a dialog"),
        pytest.param("windows_keyboard", (672, 2500), "keyboard", id="the keyboard"),
        pytest.param("windows_plain_list", (600, 50), "system", id="the status bar"),
    ],
)
def test_a_tap_that_lands_on_no_app_element_is_a_mis_tap_naming_what_it_hit(
    dump: str, point: tuple[int, int], hit: str
) -> None:
    planned = _settings_row("Apps")

    effect = _tap_effect(_observed(dump), {"action": "tap", "x": point[0], "y": point[1]}, planned)

    assert effect.hits == (ActionHit(hit),)
    assert effect.flag_records() == [{"mis_tap": {"planned": planned.as_record(), "hit": hit}}]


def test_a_tap_on_a_row_of_the_planned_list_item_is_no_mis_tap() -> None:
    network = _settings_row("Network & internet")
    rows = Locator(network.resource_id, network.class_name, None, network.ancestor_ids)

    assert _tap_effect(_observed(), _NETWORK_TITLE, rows).mis_tap is None


def test_a_tap_on_an_element_inside_the_planned_one_is_no_mis_tap() -> None:
    """Tapping a row of the settings list the planner chose acts on that list."""
    settings_list = _settings_row("")
    assert settings_list.class_name == "android.widget.ScrollView"

    effect = _tap_effect(_observed(), _NETWORK_TITLE, settings_list)

    (hit,) = effect.hits
    assert hit.anchor is not None
    assert hit.anchor.locator == _settings_row("Network & internet")
    assert effect.mis_tap is None


def test_an_action_without_sub_actions_acted_on_nothing() -> None:
    effect = action_effect(_observed(), _observed(), [], planned=None, self_loop=True)

    assert effect == ActionEffect((), mis_tap=None, no_visible_effect=True)
    assert (effect.anchor_records(), effect.flag_records()) == ([], [])


@pytest.mark.parametrize(
    ("before", "outcome"),
    [
        pytest.param(_observed(settled=False), "unsettled_tree", id="a tree that never settled"),
        pytest.param(_observed(None), "no_tree", id="no tree"),
    ],
)
def test_a_screen_whose_tree_never_settled_or_is_missing_gives_no_anchor_and_no_mis_tap(
    before: Observation, outcome: str
) -> None:
    effect = _tap_effect(before, _NETWORK_TITLE, _settings_row("Apps"))

    assert effect == ActionEffect((ActionHit(outcome),), mis_tap=None, no_visible_effect=False)
    assert effect.anchor_records() == [None]


def test_typing_is_anchored_to_the_focused_text_field_and_is_never_a_mis_tap() -> None:
    before = _observed("windows_keyboard", package=_SEARCH_APP)

    effect = _tap_effect(before, {"action": "type", "text": "wifi"}, _settings_row("Apps"))

    (hit,) = effect.hits
    assert hit.outcome == "element"
    assert hit.anchor is not None
    assert hit.anchor.locator.label == "Search settings"
    assert effect.mis_tap is None


def test_typing_with_no_focused_text_field_gives_no_anchor() -> None:
    effect = _tap_effect(_observed(), {"action": "type", "text": "wifi"}, None)

    assert effect.hits == (ActionHit("no_focused_field"),)


@pytest.mark.parametrize(
    "action",
    [
        {"action": "swipe", "x1": 1, "y1": 2, "x2": 3, "y2": 4, "duration": 300},
        {"action": "back"},
        {"action": "enter"},
        {"action": "tap"},
    ],
    ids=["swipe", "back", "enter", "tap without a point"],
)
def test_an_action_on_no_element_has_no_anchor_and_is_never_a_mis_tap(
    action: dict[str, object],
) -> None:
    effect = _tap_effect(_observed(), action, _settings_row("Apps"))

    assert effect == ActionEffect((ActionHit(None),), mis_tap=None, no_visible_effect=False)


def test_only_the_first_sub_action_of_a_compound_action_is_anchored() -> None:
    """The screen a later sub-action acts on is one no tree shows."""
    effect = action_effect(
        _observed(),
        _AFTER,
        [_NETWORK_TITLE, {"action": "type", "text": "x"}],
        planned=None,
        self_loop=True,
    )

    assert [hit.outcome for hit in effect.hits] == ["element", None]
    assert effect.anchor_records()[1] is None


def _self_loop(before: Observation, after: Observation) -> bool:
    return action_effect(
        before, after, [_NETWORK_TITLE], planned=None, self_loop=True
    ).no_visible_effect


def test_a_self_loop_whose_screens_have_one_pixel_hash_had_no_visible_effect() -> None:
    """The pixel hash decides alone, whatever the trees are."""
    assert _self_loop(_observed(None), _observed(None))
    assert _self_loop(_observed(settled=False), _observed("windows_dialog_compose"))


def test_a_self_loop_whose_settled_trees_show_the_same_screen_had_no_visible_effect() -> None:
    """A blinking cursor or an animation changes pixels and not the tree."""
    assert _self_loop(_observed(), _AFTER)


@pytest.mark.parametrize(
    ("before", "after"),
    [
        pytest.param(
            _observed(), _observed(settled=False, pixels="later"), id="after never settled"
        ),
        pytest.param(_observed(settled=False), _AFTER, id="before never settled"),
        pytest.param(_observed(), _observed(None, pixels="later"), id="no tree after"),
        pytest.param(
            _observed("windows_web_view", package="com.example.app"),
            _observed("windows_web_view", package="com.example.app", pixels="later"),
            id="thin trees",
        ),
        pytest.param(
            _observed(), _observed("windows_keyboard", pixels="later"), id="another screen"
        ),
    ],
)
def test_a_self_loop_with_different_pixels_and_no_settled_equal_full_trees_changed_something(
    before: Observation, after: Observation
) -> None:
    assert not _self_loop(before, after)


def test_an_action_to_another_node_is_never_flagged_as_having_no_visible_effect() -> None:
    effect = action_effect(
        _observed(), _observed(), [_NETWORK_TITLE], planned=None, self_loop=False
    )

    assert not effect.no_visible_effect


def test_a_mis_tap_that_changed_nothing_carries_both_flags() -> None:
    apps, network = _settings_row("Apps"), _settings_row("Network & internet")

    effect = action_effect(_observed(), _observed(), [_NETWORK_TITLE], planned=apps, self_loop=True)

    assert effect.flag_records() == [
        {
            "mis_tap": {"planned": apps.as_record(), "hit": network.as_record()},
            "no_visible_effect": True,
        }
    ]
