"""Tests for android_app_graph.android_packages.

Both graph loaders, utils.graph_manager and adapters.aitk_translator, read a
node's package through this module; it owns the tests so the two consumers do
not each pin their own copy.
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from android_app_graph.android_packages import package_from_activity, saved_node_package


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


@pytest.mark.parametrize(
    ("node", "expected"),
    [
        ({"id": "s0_mail", "package": "com.google.android.gm"}, "com.google.android.gm"),
        ({"id": "s0_home", "activity": "org.chromium.chrome.browser.Main"}, "com.android.chrome"),
        ({"id": "s0_home", "package": ""}, "com.android.chrome"),
        ({"id": "s0_home", "package": None}, "com.android.chrome"),
        ({"id": "ext_com.google.android.gm", "activity": "com.google.android.gm.Main"}, ""),
        ({"id": "ext_web"}, ""),
    ],
)
def test_saved_node_package(node: dict[str, object], expected: str) -> None:
    assert saved_node_package(node, "com.android.chrome") == expected
