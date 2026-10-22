"""Tests for android_app_graph.ui_tree: parsing ``uiautomator dump --windows`` output
and deciding whether two trees show the same screen."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from android_app_graph.ui_tree import (
    IDLE_FAILURE,
    Bounds,
    UiNode,
    UiTree,
    UiWindow,
    UnusableDumpError,
    parse_tree,
    same_screen,
    xml_document,
)

# Trimmed from `adb exec-out uiautomator dump [--windows] /dev/stdout` on a Pixel 8 Pro
# emulator (API 37, 1344x2992): every window is kept, and in each hierarchy only a few
# labelled nodes and their ancestors. Each file is the command's stdout, trailing line
# included; never_idle is a running stopwatch's.
_DATA = Path(__file__).parent / "data"
_TRAILER = "UI hierchary dumped to: /dev/stdout"


def _dump(name: str) -> str:
    return (_DATA / f"uiautomator_{name}.txt").read_text(encoding="utf-8")


def _texts(tree: UiTree, window: int) -> list[str]:
    return [node.text for node in tree.nodes if node.window == window and node.text]


def _one_node(attributes: dict[str, str]) -> str:
    """A dump with one application window holding one node with ``attributes``, with
    bounds and a drawing order unless given."""
    node = {"bounds": "[0,0][10,10]", "drawing-order": "0"} | attributes
    rendered = " ".join(f'{name}="{value}"' for name, value in node.items())
    return (
        "<?xml version='1.0' encoding='UTF-8' standalone='yes' ?><displays>"
        '<display id="0"><window index="0" title="" bounds="[0,0][100,200]" active="true" '
        'focused="true" accessibility-focused="false" id="7" layer="0" type="TYPE_APPLICATION">'
        f'<hierarchy rotation="0"><node {rendered} /></hierarchy></window></display>'
        f"</displays>{_TRAILER}"
    )


def test_a_plain_screen_has_the_status_bar_and_the_app_window() -> None:
    tree = parse_tree(_dump("windows_plain_list"))

    assert tree.windows == (
        UiWindow(
            window_id=95,
            display_id=0,
            window_type="TYPE_SYSTEM",
            title="",
            layer=1,
            bounds=Bounds(0, 0, 1344, 151),
            active=False,
            focused=False,
        ),
        UiWindow(
            window_id=96,
            display_id=0,
            window_type="TYPE_APPLICATION",
            title="Settings",
            layer=0,
            bounds=Bounds(0, 0, 1344, 2992),
            active=True,
            focused=True,
        ),
    )
    assert _texts(tree, 0) == ["1:33"]
    assert _texts(tree, 1) == [
        "Search Settings",
        "Network & internet",
        "Mobile, Wi\u2011Fi, hotspot",
        "Apps",
    ]


def test_the_nodes_nest_as_in_the_dump() -> None:
    tree = parse_tree(_dump("windows_plain_list"))

    title = next(node for node in tree.nodes if node.text == "Network & internet")
    assert title.parent is not None
    frame = tree.nodes[title.parent]
    assert frame.resource_id == "com.android.settings:id/text_frame"
    assert [tree.nodes[child].text for child in frame.children] == [
        "Network & internet",
        "Mobile, Wi\u2011Fi, hotspot",
    ]
    assert frame.parent is not None
    assert tree.nodes[frame.parent].clickable
    roots = [index for index, node in enumerate(tree.nodes) if node.parent is None]
    assert [tree.nodes[root].window for root in roots] == [0, 1]
    for index, node in enumerate(tree.nodes):
        for child in node.children:
            assert (tree.nodes[child].parent, tree.nodes[child].window) == (index, node.window)


def test_a_dialog_is_its_own_window_over_the_status_bar() -> None:
    """A dialog that takes focus hides the screen under it in the dump too."""
    tree = parse_tree(_dump("windows_dialog_compose"))

    assert [(window.window_type, window.bounds, window.active) for window in tree.windows] == [
        ("TYPE_SYSTEM", Bounds(0, 0, 1344, 151), False),
        ("TYPE_APPLICATION", Bounds(0, 1205, 1344, 1865), True),
    ]
    assert _texts(tree, 1) == ["Disable app", "Cancel"]


def test_the_keyboard_is_a_window_that_is_neither_active_nor_focused() -> None:
    tree = parse_tree(_dump("windows_keyboard"))

    assert [
        (window.window_type, window.title, window.layer, window.active, window.focused)
        for window in tree.windows
    ] == [
        ("TYPE_SYSTEM", "", 2, False, False),
        ("TYPE_INPUT_METHOD", "English (US) (QWERTY)", 1, False, False),
        ("TYPE_APPLICATION", "Settings Services", 0, True, True),
    ]
    keys = [node.content_description for node in tree.nodes if node.window == 1]
    assert keys[-1] == "q"
    field = next(node for node in tree.nodes if node.class_name == "android.widget.EditText")
    assert (field.window, field.focused, field.hint) == (2, True, "Search settings")


def test_a_child_window_listed_again_inside_its_parent_is_read_once() -> None:
    tree = parse_tree(_dump("windows_child_popup_window"))

    assert [(window.title, window.layer) for window in tree.windows] == [
        ("", 2),
        ("Popup Window", 1),
        ("Clock", 0),
    ]
    popup = "You can find the privacy policy here"
    assert [node.text for node in tree.nodes].count(popup) == 1
    assert _texts(tree, 1) == [popup]
    assert popup not in _texts(tree, 2)


def test_a_window_without_a_hierarchy_has_no_nodes() -> None:
    """The dump leaves out the hierarchy of a window whose root it cannot read."""
    tree = parse_tree(re.sub(r"<hierarchy.*</hierarchy>", "", _one_node({})))

    assert [window.window_id for window in tree.windows] == [7]
    assert tree.nodes == ()


def test_a_node_reads_each_attribute_into_its_own_field() -> None:
    tree = parse_tree(
        _one_node(
            {
                "text": "Wi-Fi",
                "resource-id": "android:id/title",
                "class": "android.widget.TextView",
                "package": "com.android.settings",
                "content-desc": "Wi-Fi settings",
                "hint": "Network name",
                "bounds": "[1,2][30,40]",
                "drawing-order": "3",
                "NAF": "true",
            }
        )
    )

    assert tree.nodes == (
        UiNode(
            window=0,
            parent=None,
            children=(),
            resource_id="android:id/title",
            class_name="android.widget.TextView",
            package="com.android.settings",
            text="Wi-Fi",
            content_description="Wi-Fi settings",
            hint="Network name",
            bounds=Bounds(1, 2, 30, 40),
            drawing_order=3,
            checkable=False,
            checked=False,
            clickable=False,
            enabled=False,
            focusable=False,
            focused=False,
            long_clickable=False,
            password=False,
            scrollable=False,
            selected=False,
        ),
    )


_FLAGS = {
    "checkable": "checkable",
    "checked": "checked",
    "clickable": "clickable",
    "enabled": "enabled",
    "focusable": "focusable",
    "focused": "focused",
    "long-clickable": "long_clickable",
    "password": "password",
    "scrollable": "scrollable",
    "selected": "selected",
}


@pytest.mark.parametrize(("attribute", "field"), _FLAGS.items())
def test_each_flag_is_read_from_its_own_attribute(attribute: str, field: str) -> None:
    node = parse_tree(_one_node({attribute: "true"})).nodes[0]

    assert [name for name in _FLAGS.values() if getattr(node, name)] == [field]


def test_the_trailing_line_is_optional_and_the_document_is_xml_without_it() -> None:
    dump = _dump("windows_plain_list")
    document = xml_document(dump)

    assert dump.endswith(_TRAILER)
    assert _TRAILER not in document
    assert document.startswith("<?xml")
    assert document.endswith("</displays>")
    assert parse_tree(document) == parse_tree(dump)


def test_a_tree_deeper_than_the_recursion_limit_parses() -> None:
    depth = 3000
    dump = _one_node({}).replace(
        '<node bounds="[0,0][10,10]" drawing-order="0" />',
        '<node bounds="[0,0][10,10]" drawing-order="0">' * depth + "</node>" * depth,
    )

    nodes = parse_tree(dump).nodes

    assert len(nodes) == depth
    assert nodes[-1].parent == depth - 2


_PLAIN_LIST = _dump("windows_plain_list")
_CUT_SHORT = _PLAIN_LIST.index("</displays>")


@pytest.mark.parametrize(
    ("dump", "reason"),
    [
        pytest.param("", "empty output", id="empty"),
        pytest.param(_TRAILER, "empty output", id="only the trailing line"),
        pytest.param(_dump("windows_never_idle"), IDLE_FAILURE, id="idle failure"),
        pytest.param(
            "ERROR: null root node returned by UiTestAutomationBridge.\n",
            "ERROR: null root node returned by UiTestAutomationBridge.",
            id="another error",
        ),
        pytest.param(
            _dump("default_plain_list"), "root is <hierarchy>, not <displays>", id="default dump"
        ),
        pytest.param(
            _PLAIN_LIST[:_CUT_SHORT],
            f"XML does not parse: no element found: line 1, column {_CUT_SHORT}",
            id="truncated",
        ),
        pytest.param(
            _one_node({"bounds": "[0,0][10]"}),
            "<node> bounds='[0,0][10]' do not parse",
            id="bad bounds",
        ),
        pytest.param(
            _one_node({"drawing-order": "first"}),
            "<node> drawing-order='first' is not a number",
            id="bad number",
        ),
    ],
)
def test_an_unusable_dump_is_rejected_with_its_reason(dump: str, reason: str) -> None:
    with pytest.raises(UnusableDumpError) as excinfo:
        parse_tree(dump)

    assert excinfo.value.reason == reason
    assert str(excinfo.value) == reason


def _with_attribute(dump: str, label: str, name: str, value: str) -> str:
    """``dump`` with attribute ``name`` of the first node whose text is ``label`` set to
    ``value``."""
    start = dump.index(f'text="{label}"')
    end = dump.index(">", start)
    tag = re.sub(rf'(?<![\w-]){name}="[^"]*"', f'{name}="{value}"', dump[start:end], count=1)
    return dump[:start] + tag + dump[end:]


def test_two_trees_that_differ_only_in_the_status_bar_show_the_same_screen() -> None:
    dump = _dump("windows_plain_list")
    next_minute = _with_attribute(dump, "1:33", "text", "1:34").replace('id="95"', 'id="1817"')

    assert parse_tree(next_minute) != parse_tree(dump)
    assert same_screen(parse_tree(dump), parse_tree(next_minute))


def test_window_ids_take_no_part_in_the_comparison() -> None:
    """Window IDs change from one dump to the next."""
    dump = _dump("windows_plain_list")

    assert same_screen(parse_tree(dump), parse_tree(dump.replace('id="96"', 'id="1818"')))


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("text", "Network"),
        ("bounds", "[252,750][648,815]"),
        ("checked", "true"),
        ("resource-id", "android:id/summary"),
        ("class", "android.widget.Button"),
        ("content-desc", "Network"),
        ("selected", "true"),
        ("focused", "true"),
        ("enabled", "false"),
    ],
)
def test_a_change_in_an_app_window_node_shows_another_screen(name: str, value: str) -> None:
    dump = _dump("windows_plain_list")
    changed = _with_attribute(dump, "Network &amp; internet", name, value)

    assert changed != dump
    assert not same_screen(parse_tree(dump), parse_tree(changed))


def test_a_node_attribute_the_comparison_leaves_out_shows_the_same_screen() -> None:
    dump = _dump("windows_plain_list")
    changed = _with_attribute(dump, "Network &amp; internet", "drawing-order", "5")

    assert same_screen(parse_tree(dump), parse_tree(changed))


def test_the_same_nodes_nested_differently_show_another_screen() -> None:
    """The title and the summary are siblings; nesting the summary inside the title keeps
    every node and their order."""
    dump = _dump("windows_plain_list")
    title_end = dump.index("/>", dump.index('text="Network &amp; internet"'))
    summary_end = dump.index("/>", title_end + 2)
    nested = dump[:title_end] + ">" + dump[title_end + 2 : summary_end + 2] + "</node>"
    nested += dump[summary_end + 2 :]

    first, second = parse_tree(dump), parse_tree(nested)

    assert [node.text for node in first.nodes] == [node.text for node in second.nodes]
    assert not same_screen(first, second)


def test_the_keyboard_appearing_shows_another_screen() -> None:
    dump = _dump("windows_keyboard")
    start = dump.index('<window index="1"')
    without_keyboard = dump[:start] + dump[dump.index("</window>", start) + len("</window>") :]

    assert len(parse_tree(without_keyboard).windows) == 2
    assert not same_screen(parse_tree(without_keyboard), parse_tree(dump))


@pytest.mark.parametrize(
    ("name", "value"),
    [("layer", "3"), ("bounds", "[0,1000][1344,2992]"), ("type", "TYPE_ACCESSIBILITY_OVERLAY")],
)
def test_a_change_in_the_window_list_shows_another_screen(name: str, value: str) -> None:
    dump = _dump("windows_keyboard")
    start = dump.index('<window index="1"')
    end = dump.index(">", start)
    window = re.sub(rf'(?<![\w-]){name}="[^"]*"', f'{name}="{value}"', dump[start:end], count=1)

    assert not same_screen(parse_tree(dump), parse_tree(dump[:start] + window + dump[end:]))
