"""Tests for android_app_graph.ui_tree: parsing ``uiautomator dump --windows`` output,
deciding whether two trees show the same screen, and the layout signature, stable
layout and layout similarity."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from android_app_graph.ui_tree import (
    IDLE_FAILURE,
    MIN_LAYOUT_ENTRIES,
    Bounds,
    LayoutEntry,
    UiNode,
    UiTree,
    UiWindow,
    UnusableDumpError,
    layout_signature,
    layout_similarity,
    parse_tree,
    same_screen,
    stable_layout,
    window_roles,
    xml_document,
)

# Trimmed from `adb exec-out uiautomator dump [--windows] /dev/stdout` on a Pixel 8 Pro
# emulator (API 37, 1344x2992): every window is kept, and in each hierarchy only a few
# labelled nodes and their ancestors. Each file is the command's stdout, trailing line
# included; never_idle is a running stopwatch's. The settings_* files are trimmed the
# same way from trees a Settings exploration kept in its step log on a Pixel 8a
# emulator (API 37, 1080x2400): a page, a dialog, and that dialog with the keyboard.
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


def _signature(name: str) -> frozenset[LayoutEntry]:
    return layout_signature(parse_tree(_dump(name)))


def _action(role: str, resource_id: str, class_name: str) -> LayoutEntry:
    return LayoutEntry("action", role, resource_id, class_name, "")


def _window_with(nodes: str, *, title: str = "Page", bounds: str = "[0,0][100,200]") -> str:
    """A dump with the status bar and one application window holding ``nodes``."""
    return (
        "<?xml version='1.0' encoding='UTF-8' standalone='yes' ?><displays><display id=\"0\">"
        '<window index="0" title="" bounds="[0,0][100,10]" active="false" focused="false" '
        'accessibility-focused="false" id="1" layer="1" type="TYPE_SYSTEM"><hierarchy '
        'rotation="0"><node text="1:33" bounds="[0,0][10,10]" drawing-order="0" />'
        f'</hierarchy></window><window index="1" title="{title}" bounds="{bounds}" '
        'active="true" focused="true" accessibility-focused="false" id="2" layer="0" '
        f'type="TYPE_APPLICATION"><hierarchy rotation="0">{nodes}</hierarchy></window>'
        f"</display></displays>{_TRAILER}"
    )


def _node(attributes: str, children: str = "") -> str:
    return f'<node drawing-order="0" bounds="[0,20][100,40]" {attributes}>{children}</node>'


def test_a_page_s_signature_is_its_title_and_its_controls() -> None:
    """Settings draws the page's title and its rows inside the scrolling container, so
    the window's title is the page's only text entry."""
    tree = parse_tree(_dump("windows_settings_page"))

    assert layout_signature(tree) == {
        LayoutEntry("title", "base", "", "", "Data warning & limit"),
        _action("base", "", "android.widget.ImageButton"),
        _action("base", "", "android.widget.LinearLayout"),
        _action("base", "com.android.settings:id/content_parent", "android.widget.ScrollView"),
        _action("base", "com.android.settings:id/switchWidget", "android.widget.Switch"),
    }
    texts = {node.text or node.content_description for node in tree.nodes if node.window == 1}
    assert {"Data warning & limit", "Set data warning", "2.00 GB"} <= texts


def test_a_dialog_s_entries_name_the_dialog_role() -> None:
    assert _signature("windows_settings_dialog") == {
        LayoutEntry("title", "dialog", "", "", "Set data usage warning"),
        _action("dialog", "android:id/button1", "android.widget.Button"),
        _action("dialog", "com.android.settings:id/bytes", "android.widget.EditText"),
        _action("dialog", "com.android.settings:id/size_spinner", "android.widget.Spinner"),
        LayoutEntry(
            "text",
            "",
            "com.android.settings:id/alertTitle",
            "android.widget.TextView",
            "Set data usage warning",
        ),
    }


def test_the_keyboard_and_the_status_bar_are_left_out() -> None:
    """The keyboard moved the dialog up; neither window takes part in the signature."""
    tree = parse_tree(_dump("windows_settings_dialog_keyboard"))

    assert [window.window_type for window in tree.windows] == [
        "TYPE_SYSTEM",
        "TYPE_INPUT_METHOD",
        "TYPE_APPLICATION",
    ]
    assert layout_signature(tree) == _signature("windows_settings_dialog")


@pytest.mark.parametrize(
    ("name", "roles"),
    [
        pytest.param("windows_plain_list", {1: "base"}, id="a page"),
        pytest.param("windows_keyboard", {2: "base"}, id="a page under the keyboard"),
        pytest.param("windows_settings_dialog", {1: "dialog"}, id="a dialog with margins"),
        pytest.param("windows_dialog_compose", {1: "dialog"}, id="a dialog the screen's width"),
        pytest.param("windows_child_popup_window", {1: "popup", 2: "base"}, id="a popup"),
    ],
)
def test_each_app_window_has_a_role_and_no_other_window_does(
    name: str, roles: dict[int, str]
) -> None:
    assert window_roles(parse_tree(_dump(name))) == roles


def test_a_tree_without_windows_has_no_role_and_no_signature() -> None:
    tree = parse_tree("<displays />")

    assert (window_roles(tree), layout_signature(tree)) == ({}, frozenset())


def test_a_window_narrower_than_the_screen_or_below_its_top_is_a_dialog() -> None:
    narrower = parse_tree(_window_with("", bounds="[5,0][95,200]"))
    lower = parse_tree(_window_with("", bounds="[0,50][100,200]"))

    assert window_roles(narrower) == window_roles(lower) == {1: "dialog"}


def test_each_action_flag_makes_a_node_actionable() -> None:
    flags = ("clickable", "long-clickable", "checkable", "scrollable")
    nodes = "".join(
        _node(f'resource-id="id/{flag}" class="View" {flag}="true"')
        for flag in (*flags, "focusable")
    )

    assert layout_signature(parse_tree(_window_with(nodes, title=""))) == {
        _action("base", f"id/{flag}", "View") for flag in flags
    }


def test_static_text_inside_a_scrolling_container_is_left_out() -> None:
    nodes = _node('text="Header" class="TextView"') + _node(
        'class="RecyclerView" scrollable="true"',
        _node('class="LinearLayout"', _node('text="Row" class="TextView"')),
    )

    assert layout_signature(parse_tree(_window_with(nodes, title=""))) == {
        LayoutEntry("text", "", "", "TextView", "Header"),
        _action("base", "", "RecyclerView"),
    }


def test_a_static_text_without_text_is_labelled_by_its_content_description() -> None:
    nodes = _node('text="Wi-Fi" content-desc="Wireless" class="TextView"') + _node(
        'content-desc="Signal full" class="ImageView"'
    )

    assert layout_signature(parse_tree(_window_with(nodes, title=""))) == {
        LayoutEntry("text", "", "", "TextView", "Wi-Fi"),
        LayoutEntry("text", "", "", "ImageView", "Signal full"),
    }


def test_an_invisible_or_unlabelled_node_is_left_out() -> None:
    nodes = (
        '<node drawing-order="0" bounds="[0,20][0,40]" clickable="true" class="Button" />'
        '<node drawing-order="0" bounds="[0,20][100,20]" text="Hidden" class="TextView" />'
        + _node('class="FrameLayout"')
    )

    assert layout_signature(parse_tree(_window_with(nodes))) == {
        LayoutEntry("title", "base", "", "", "Page")
    }


_A, _B, _C, _D, _E = (_action("base", f"id/{name}", "Button") for name in "abcde")


def test_a_stable_layout_keeps_the_entries_on_most_signatures_with_their_share() -> None:
    signatures = [
        frozenset({_A, _B, _C}),
        frozenset({_A, _B, _C}),
        frozenset({_A, _B, _C}),
        frozenset({_A, _B, _D}),
        frozenset({_A, _D, _E}),
    ]

    assert stable_layout(signatures) == {_A: 1.0, _B: 0.8}
    assert stable_layout(signatures, share=0.6) == {_A: 1.0, _B: 0.8, _C: 0.6}


def test_a_stable_layout_of_no_signature_is_empty() -> None:
    assert stable_layout([]) == {}


def test_layout_similarity_is_the_weighted_jaccard_of_two_layouts() -> None:
    screen = dict.fromkeys((_A, _B, _C, _D), 1.0)
    node = {_A: 1.0, _B: 0.8, _E: 0.9}

    assert layout_similarity(screen, node) == pytest.approx((1.0 + 0.8) / (1 + 1 + 1 + 1 + 0.9))
    assert layout_similarity(screen, screen) == 1.0
    assert layout_similarity(screen, dict.fromkeys((_E, _B, _C), 1.0)) == pytest.approx(2 / 5)


def test_a_layout_with_too_few_entries_scores_nothing() -> None:
    small = dict.fromkeys((_A, _B), 1.0)
    large = dict.fromkeys((_A, _B, _C), 1.0)

    assert len(large) == MIN_LAYOUT_ENTRIES
    assert layout_similarity(small, large) is None
    assert layout_similarity(large, small) is None
    assert layout_similarity(large, large) == 1.0
