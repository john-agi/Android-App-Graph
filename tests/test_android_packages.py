"""Tests for android_app_graph.android_packages.package_from_activity.

The join key between a runtime screen (an Android activity) and a graph node,
shared by utils.graph_manager and adapters.aitk_translator; this module owns
its tests so the two consumers do not each pin their own copy.
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from android_app_graph.android_packages import package_from_activity


@pytest.mark.parametrize(
    ("activity", "expected"),
    [
        ("com.example.app/.MainActivity", "com.example.app"),
        ("com.google.android.apps.maps/com.google.Main", "com.google.android"),
        ("com.example.app", "com.example.app"),
        ("com.citymapper.app.home.HomeActivity2", "com.citymapper.app"),
        ("com.citymapper.app/com.citymapper.app.MainActivity", "com.citymapper.app"),
        ("two.parts", "two.parts"),
        ("single", "single"),
        ("", ""),
    ],
)
def test_package_from_activity(activity: str, expected: str) -> None:
    assert package_from_activity(activity) == expected


@given(st.text())
def test_package_from_activity_keeps_at_most_three_components(activity: str) -> None:
    """The package is a dotted prefix of the activity's component, never longer."""
    package = package_from_activity(activity)
    component = activity.split("/", maxsplit=1)[0]
    assert component.startswith(package)
    assert len(package.split(".")) <= max(3, len(component.split(".")))
