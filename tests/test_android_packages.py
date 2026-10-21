"""Tests for android_app_graph.android_packages."""

from __future__ import annotations

import pytest

from android_app_graph.android_packages import saved_node_package


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
