"""System input geometry separates app taps from system gesture starts."""

from __future__ import annotations

import pytest

from android_app_graph.device import InputRegions, input_region_hint, input_regions


@pytest.mark.parametrize(
    "window",
    [
        "",
        "InsetsSource type=tappableElement frame=[0,0][720,48] visible=true",
        "WindowInsetsStateController\nInsetsState",
        "WindowInsetsStateController\nInsetsState\nmDisplayFrame=Rect(0, 0 - 720, 1280)\n",
        (
            "WindowInsetsStateController\nInsetsState\nmDisplayFrame=Rect(0, 0 - 720, 1280)\n"
            "InsetsSource type=tappableElement frame=unknown visible=true\n"
        ),
    ],
)
def test_missing_authoritative_display_geometry_is_explicit(window: str) -> None:
    assert input_regions(window) is None


def test_only_visible_nonempty_authoritative_sources_define_input_regions() -> None:
    window = (
        "WindowInsetsStateController\nInsetsState\n"
        "mDisplayFrame=Rect(0, 0 - 720, 1280)\n"
        "InsetsSource type=tappableElement frame=[0,1184][720,1280] visible=true\n"
        "InsetsSource type=tappableElement frame=[0,1184][720,1280] visible=true\n"
        "InsetsSource type=tappableElement frame=[0,0][720,48] visible=false\n"
        "InsetsSource type=systemGestures frame=[0,0][0,0] visible=true\n"
        "InsetsSource type=mandatorySystemGestures frame=[0,1184][720,1280] visible=true\n"
        "InsetsSource type=systemGestures frame=[0,0][720,0] visible=true\n"
        "InsetsSourceProviders:\n"
        "InsetsSource type=tappableElement frame=[0,0][720,1280] visible=true\n"
    )
    regions = input_regions(window)
    assert regions is not None
    assert regions == InputRegions(
        (0, 0, 720, 1280), ((0, 1184, 720, 1280),), ((0, 1184, 720, 1280),)
    )
    assert regions.as_dict() == {
        "display": [0, 0, 720, 1280],
        "tap": [[0, 1184, 720, 1280]],
        "gesture": [[0, 1184, 720, 1280]],
    }


def test_known_hidden_input_sources_preserve_immersive_app_content() -> None:
    assert input_regions(
        "WindowInsetsStateController\nInsetsState\nmDisplayFrame=Rect(0, 0 - 720, 1280)\n"
        "InsetsSource type=tappableElement frame=[0,1184][720,1280] visible=false\n"
    ) == InputRegions((0, 0, 720, 1280), (), ())


@pytest.mark.parametrize(
    "action",
    [
        {"action": "tap", "x": -1, "y": 100},
        {"action": "long_press", "x": 720, "y": 100},
        {"action": "swipe", "x1": 100, "y1": 100, "x2": 100, "y2": 1280},
    ],
)
def test_all_coordinate_points_stay_within_the_display(action: dict[str, object]) -> None:
    regions = InputRegions((0, 0, 720, 1280), (), ())
    assert regions.reject_reason(action) == "App input is outside the display"


def test_a_visual_gesture_navigation_bar_allows_content_taps() -> None:
    regions = input_regions(
        "WindowInsetsStateController\nInsetsState\n"
        "mDisplayFrame=Rect(0, 0 - 1080, 2424)\n"
        "InsetsSource type=navigationBars frame=[0,2361][1080,2424] visible=true\n"
        "InsetsSource type=tappableElement frame=[0,0][0,0] visible=true\n"
        "InsetsSource type=mandatorySystemGestures frame=[0,2340][1080,2424] visible=true\n"
    )
    assert regions is not None
    assert regions.reject_reason({"action": "tap", "x": 500, "y": 2400}) is None
    assert "system gesture region" in str(
        regions.reject_reason({"action": "swipe", "x1": 500, "y1": 2400, "x2": 500, "y2": 700})
    )


def test_planning_uses_observed_geometry_and_same_control_reachability() -> None:
    assert input_region_hint({}) == ""
    hint = input_region_hint(
        {
            "input_regions": {
                "display": [0, 0, 720, 1280],
                "tap": [[0, 1184, 720, 1280]],
                "gesture": [],
            }
        }
    )
    assert "[0, 1184, 720, 1280]" in hint
    assert "same control" in hint
    assert "wait" in hint


@pytest.mark.parametrize("coordinate", [None, True, "invalid"])
def test_invalid_coordinate_payloads_have_an_explicit_reason(coordinate: object) -> None:
    regions = InputRegions((0, 0, 720, 1280), (), ())
    assert (
        regions.reject_reason({"action": "tap", "x": coordinate, "y": 20})
        == "App input has invalid coordinates"
    )
