"""Tests for android_app_graph.android_packages.

Both graph loaders, utils.graph_manager and adapters.aitk_translator, read a
node's package through this module, and the Android World adapter a screen's;
it owns the tests so its consumers do not each pin their own copy.
"""

from __future__ import annotations

import pytest

from android_app_graph.android_packages import package_from_activity, saved_node_package


@pytest.mark.parametrize(
    ("activity", "expected"),
    [
        ("com.example.app/.MainActivity", "com.example.app"),
        ("com.google.android.apps.maps/com.google.Main", "com.google.android.apps.maps"),
        ("com.whatsapp/com.whatsapp.home.ui.HomeActivity", "com.whatsapp"),
        ("com.whatsapp.home.ui.HomeActivity", ""),
        ("", ""),
    ],
)
def test_package_from_activity(activity: str, expected: str) -> None:
    assert package_from_activity(activity) == expected


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
