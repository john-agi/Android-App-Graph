"""Tests for android_app_graph.replay: a replayed tap or long press goes to the current
centre of the element it recorded, or to its recorded coordinates with the reason
logged, and every other action replays as recorded."""

from __future__ import annotations

import base64
import dataclasses
import io
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from android_app_graph.replay import replay_action
from android_app_graph.step_log import Observation, StepRecorder
from android_app_graph.ui_tree import Anchor, Bounds, actionable_elements, parse_tree, xml_document

_SETTINGS = "com.android.settings"
_W, _H = 1344, 2992
# A trimmed dump of Settings' home on a Pixel 8 Pro emulator: the search bar, the list,
# and the "Network & internet" row at [0,699][1344,917] above the "Apps" row.
_HOME = xml_document(
    (Path(__file__).parent / "data" / "uiautomator_windows_plain_list.txt").read_text("utf-8")
)
_NETWORK = "Network &amp; internet"
# The keyboard over the lower part of the screen, with a key over the Apps row.
_KEYBOARD = (
    '<window index="0" title="" bounds="[0,1100][1344,2992]" active="false" focused="false" '
    'accessibility-focused="false" id="90" layer="2" type="TYPE_INPUT_METHOD">'
    '<hierarchy rotation="0"><node index="0" text="" resource-id="" '
    'class="android.widget.FrameLayout" package="com.google.android.inputmethod.latin" '
    'content-desc="" checkable="false" checked="false" clickable="false" enabled="true" '
    'focusable="false" focused="false" scrollable="false" long-clickable="false" '
    'password="false" selected="false" bounds="[0,1100][1344,2992]" drawing-order="0" '
    'hint=""><node index="0" text="" resource-id="" class="android.widget.FrameLayout" '
    'package="com.google.android.inputmethod.latin" content-desc="q" checkable="false" '
    'checked="false" clickable="true" enabled="true" focusable="true" focused="false" '
    'scrollable="false" long-clickable="true" password="false" selected="false" '
    'bounds="[0,1200][1344,1400]" drawing-order="1" hint="" /></node></hierarchy></window>'
)
_LOGGER = logging.getLogger("test.replay")


def _png() -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (_W, _H), "white").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


_SCREENSHOT = _png()


def _screen(xml: str | None = _HOME) -> dict[str, Any]:
    """A settled capture of Settings' home with ``xml`` as its tree, or none."""
    return {
        "activity": "com.android.settings.Settings",
        "screenshot": _SCREENSHOT,
        "package": _SETTINGS,
        "xml": xml,
        "ui_tree_error": None if xml is not None else "adb exited with status 1",
        "ui_tree_settled": True,
    }


class _Device:
    """A DeviceController on Settings' home that counts its screen reads and keeps every
    action it is sent."""

    w = _W
    h = _H

    def __init__(self, screen: dict[str, Any]) -> None:
        self.config: dict[str, Any] = {"device": {"udid": "test-device"}}
        self._screen = screen
        self.reads = 0
        self.actions: list[dict[str, Any]] = []

    def get_state(self) -> dict[str, Any]:
        self.reads += 1
        return self._screen

    def exe_action(self, action: dict[str, Any]) -> None:
        self.actions.append(action)


def _recording(device: _Device, steps: Path) -> StepRecorder:
    return StepRecorder(device, steps, run_kind="explore", source="navigation_replay")


def _anchor(label: str) -> Anchor:
    """The anchor a tap on the row labelled ``label`` recorded while the row was at the
    top of the list, above where Settings' home shows it."""
    (element,) = [
        element
        for element in actionable_elements(parse_tree(_HOME), _SETTINGS, Bounds(0, 0, _W, _H))
        if element.label == label
    ]
    return Anchor(element.locator, Bounds(0, 415, 1344, 633))


# Where a tap on the Network & internet row was recorded, and the centre of that row now.
_RECORDED_TAP = {"action": "tap", "x": 672, "y": 524}
_NETWORK_CENTRE = (672, 808)


@pytest.mark.usefixtures("no_sleep")
def test_an_anchored_tap_reads_the_screen_and_goes_to_its_element_s_current_centre(
    tmp_path: Path,
    step_records: Callable[[Path], list[dict[str, Any]]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    device = _Device(_screen())

    with caplog.at_level(logging.INFO):
        sent = replay_action(
            _recording(device, tmp_path),
            [_RECORDED_TAP],
            [_anchor("Network & internet")],
            logger=_LOGGER,
        )

    x, y = _NETWORK_CENTRE
    assert device.actions == sent == [{"action": "tap", "x": x, "y": y}]
    records = step_records(tmp_path)
    assert [(record["kind"], record["source"]) for record in records] == [
        ("observation", "navigation_replay"),
        ("action", "navigation_replay"),
    ]
    assert records[1]["observation"] == records[0]["id"]
    assert 'tap at (672, 808): anchor "Network & internet", recorded at (672, 524)' in caplog.text


_APPS = _anchor("Apps")
# A row the screen does not show.
_DISPLAY = dataclasses.replace(_APPS, locator=dataclasses.replace(_APPS.locator, label="Display"))


@pytest.mark.parametrize(
    ("screen", "anchor", "reason"),
    [
        pytest.param(_screen(None), _anchor("Network & internet"), "no_tree", id="no tree"),
        pytest.param(_screen(), _DISPLAY, "no_match", id="no match"),
        pytest.param(
            _screen(_HOME.replace('text="Apps"', f'text="{_NETWORK}"')),
            _anchor("Network & internet"),
            "ambiguous",
            id="two rows of that name",
        ),
        pytest.param(
            _screen(_HOME.replace('<window index="1"', f'{_KEYBOARD}<window index="1"')),
            _APPS,
            "covered",
            id="under the keyboard",
        ),
    ],
)
@pytest.mark.usefixtures("no_sleep")
def test_a_tap_without_one_reachable_element_goes_to_the_recorded_coordinates(
    screen: dict[str, Any],
    anchor: Anchor,
    reason: str,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    device = _Device(screen)

    with caplog.at_level(logging.INFO):
        sent = replay_action(
            _recording(device, tmp_path), [_RECORDED_TAP], [anchor], logger=_LOGGER
        )

    assert device.actions == sent == [_RECORDED_TAP]
    assert device.reads == 1
    assert f"tap at (672, 524): recorded coordinates: {reason}" in caplog.text


@pytest.mark.usefixtures("no_sleep")
def test_a_tap_without_an_anchor_goes_to_the_recorded_coordinates_without_reading_the_screen(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    device = _Device(_screen())

    with caplog.at_level(logging.INFO):
        replay_action(_recording(device, tmp_path), [_RECORDED_TAP], [None], logger=_LOGGER)

    assert (device.actions, device.reads) == ([_RECORDED_TAP], 0)
    assert "tap at (672, 524): recorded coordinates: no_anchor" in caplog.text


@pytest.mark.usefixtures("no_sleep")
def test_an_anchored_long_press_keeps_its_duration_at_its_element_s_current_centre(
    tmp_path: Path,
) -> None:
    device = _Device(_screen())
    long_press = {"action": "long_press", "x": 672, "y": 524, "duration": 1500}

    replay_action(
        _recording(device, tmp_path), [long_press], [_anchor("Network & internet")], logger=_LOGGER
    )

    x, y = _NETWORK_CENTRE
    assert device.actions == [{"action": "long_press", "x": x, "y": y, "duration": 1500}]


@pytest.mark.parametrize(
    "action",
    [
        pytest.param({"action": "type", "text": "wifi"}, id="type"),
        pytest.param(
            {"action": "swipe", "x1": 672, "y1": 2000, "x2": 672, "y2": 800, "duration": 400},
            id="swipe",
        ),
        pytest.param({"action": "back"}, id="back"),
        pytest.param({"action": "home"}, id="home"),
        pytest.param({"action": "enter"}, id="enter"),
        pytest.param({"action": "wait", "duration": 2}, id="wait"),
    ],
)
@pytest.mark.usefixtures("no_sleep")
def test_any_other_action_replays_exactly_as_recorded_without_reading_the_screen(
    action: dict[str, Any], tmp_path: Path
) -> None:
    """Typing has an anchor too, the focused field, and still replays as recorded."""
    device = _Device(_screen())

    sent = replay_action(
        _recording(device, tmp_path), [action], [_anchor("Network & internet")], logger=_LOGGER
    )

    assert (device.actions, sent, device.reads) == ([action], [action], 0)


@pytest.mark.usefixtures("no_sleep")
def test_the_screen_the_caller_just_read_serves_the_first_sub_action_only(tmp_path: Path) -> None:
    device = _Device(_screen())
    recorder = _recording(device, tmp_path)
    screen = Observation.from_state(recorder.get_state())
    second = {"action": "tap", "x": 672, "y": 1250}

    replay_action(
        recorder,
        [_RECORDED_TAP, second],
        [_anchor("Network & internet"), _anchor("Apps")],
        logger=_LOGGER,
        screen=screen,
    )

    assert device.reads == 2
    assert [(action["x"], action["y"]) for action in device.actions] == [
        _NETWORK_CENTRE,
        (672, 1298),
    ]


@pytest.mark.parametrize(
    ("action", "anchors", "sent", "sleeps"),
    [
        pytest.param({"action": "back"}, [None], [{"action": "back"}], [1], id="one action"),
        pytest.param(
            [{"action": "back"}, {"action": "back"}],
            [None, None],
            [{"action": "back"}, {"action": "back"}],
            [0.5, 0.5, 1],
            id="a compound action",
        ),
        pytest.param({}, [], [], [1], id="an edge without actions"),
    ],
)
def test_sub_actions_are_half_a_second_apart_and_the_screen_settles_a_second_after(
    action: dict[str, Any] | list[dict[str, Any]],
    anchors: list[Anchor | None],
    sent: list[dict[str, Any]],
    sleeps: list[float],
    tmp_path: Path,
    no_sleep: list[float],
) -> None:
    device = _Device(_screen())

    replay_action(_recording(device, tmp_path), action, anchors, logger=_LOGGER)

    assert (device.actions, no_sleep) == (sent, sleeps)
