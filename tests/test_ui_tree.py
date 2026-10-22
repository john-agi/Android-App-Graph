"""Tests for android_app_graph.ui_tree: parsing ``uiautomator dump --windows`` output,
deciding whether two trees show the same screen, listing and locating the elements a
user can act on, and finding what a tap lands on."""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path

import pytest

from android_app_graph.ui_tree import (
    IDLE_FAILURE,
    Anchor,
    Bounds,
    Hit,
    Locator,
    TreeElement,
    UiNode,
    UiTree,
    UiWindow,
    UnusableDumpError,
    acting_windows,
    actionable_elements,
    find_nodes,
    focused_text_field,
    hit_test,
    keyboard_shown,
    locator,
    parse_tree,
    part_of,
    region,
    same_screen,
    thin_tree_reason,
    xml_document,
)

# Trimmed from `adb exec-out uiautomator dump [--windows] /dev/stdout` on a Pixel 8 Pro
# emulator (API 37, 1344x2992): every window is kept, and in each hierarchy only a few
# labelled nodes and their ancestors. Each file is the command's stdout, trailing line
# included; never_idle is a running stopwatch's, popup_menu Clock's overflow menu.
# web_view is written by hand: one web view filling an app's window.
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


_APP = "com.example.app"
_SCREEN = Bounds(0, 0, 1000, 2000)
_FIXTURE_SCREEN = Bounds(0, 0, 1344, 2992)


def _node(attributes: dict[str, str] | None = None, *children: str) -> str:
    """One ``<node>`` of the app, enabled, visible and of ``android.view.View`` unless
    ``attributes`` say otherwise, holding ``children``."""
    node = {
        "class": "android.view.View",
        "package": _APP,
        "enabled": "true",
        "bounds": "[0,0][1000,2000]",
        "drawing-order": "0",
    } | (attributes or {})
    rendered = " ".join(f'{name}="{value}"' for name, value in node.items())
    if not children:
        return f"<node {rendered} />"
    return f"<node {rendered}>{''.join(children)}</node>"


def _window(
    *roots: str,
    kind: str = "TYPE_APPLICATION",
    layer: int = 0,
    active: bool = True,
    bounds: str = "[0,0][1000,2000]",
) -> str:
    flag = "true" if active else "false"
    return (
        f'<window index="0" title="" bounds="{bounds}" active="{flag}" '
        f'focused="{flag}" accessibility-focused="false" id="{layer + 1}" layer="{layer}" '
        f'type="{kind}"><hierarchy rotation="0">{"".join(roots)}</hierarchy></window>'
    )


def _screen(*roots: str) -> UiTree:
    """A tree of one active application window, the whole screen, holding ``roots``."""
    return parse_tree(f'<displays><display id="0">{_window(*roots)}</display></displays>')


def _labels(elements: list[TreeElement]) -> list[str]:
    return [element.label for element in elements]


def _row(label: str, attributes: dict[str, str] | None = None) -> str:
    """A clickable row whose title is ``label``, above a summary."""
    return _node(
        {"class": "android.widget.LinearLayout", "clickable": "true"} | (attributes or {}),
        _node({"class": "android.widget.TextView", "text": label}),
        _node({"class": "android.widget.TextView", "text": f"{label} summary"}),
    )


def test_a_plain_screen_lists_the_app_s_elements_with_their_locators_and_regions() -> None:
    tree = parse_tree(_dump("windows_plain_list"))

    elements = actionable_elements(tree, "com.android.settings", _FIXTURE_SCREEN)

    assert [element.summary() for element in elements] == [
        'LinearLayout "Search Settings" (top)',
        "ScrollView, scrollable (center)",
        'LinearLayout "Network & internet" (top)',
        'LinearLayout "Apps" (center)',
    ]
    assert elements[2].locator == Locator(
        resource_id="",
        class_name="android.widget.LinearLayout",
        label="Network & internet",
        ancestor_ids=(
            "android:id/content",
            "com.android.settings:id/settings_homepage_container",
            "com.android.settings:id/main_content_scrollable_container",
            "com.android.settings:id/homepage_container",
            "com.android.settings:id/main_content",
            "com.android.settings:id/container_material",
            "android:id/list_container",
            "com.android.settings:id/recycler_view",
        ),
    )
    assert elements[2].bounds == Bounds(0, 699, 1344, 917)
    assert (elements[2].rows, elements[2].scrollable, elements[2].text_field) == (1, False, False)


def test_with_a_dialog_open_only_the_dialog_s_nodes_become_elements() -> None:
    tree = parse_tree(_dump("windows_dialog_compose"))

    elements = actionable_elements(tree, "com.android.settings", _FIXTURE_SCREEN)

    assert _labels(elements) == ["Disable app", "Cancel"]
    windows = {
        tree.nodes[index].window
        for element in elements
        for index in find_nodes(tree, element.locator)
    }
    assert [tree.windows[window].window_type for window in windows] == ["TYPE_APPLICATION"]


def test_keyboard_and_system_window_nodes_never_become_elements() -> None:
    tree = parse_tree(_dump("windows_keyboard"))
    package = "com.google.android.settings.intelligence"

    elements = actionable_elements(tree, package, _FIXTURE_SCREEN)

    assert [element.summary() for element in elements] == [
        'EditText "Search settings", text field (top)'
    ]
    assert acting_windows(tree, package) == [2]


def test_clickable_nodes_of_the_keyboard_and_the_status_bar_are_left_out() -> None:
    button = _node({"clickable": "true", "text": "Tap"})
    tree = parse_tree(
        '<displays><display id="0">'
        + _window(button, kind="TYPE_SYSTEM", layer=2, active=False)
        + _window(button, kind="TYPE_INPUT_METHOD", layer=1, active=False)
        + _window(_node({"clickable": "true", "text": "App"}))
        + "</display></displays>"
    )

    assert _labels(actionable_elements(tree, _APP, _SCREEN)) == ["App"]


@pytest.mark.parametrize(
    ("package", "labels"),
    [
        ("com.google.android.deskclock", ["You can find the privacy policy here", "More options"]),
        ("com.example.other", ["More options"]),
    ],
)
def test_a_popup_of_the_app_above_the_active_window_lists_its_elements_too(
    package: str, labels: list[str]
) -> None:
    """A popup that does not take focus leaves the window under it in the dump."""
    tree = parse_tree(_dump("windows_child_popup_window"))

    assert _labels(actionable_elements(tree, package, _FIXTURE_SCREEN)) == labels


def test_a_popup_menu_s_rows_are_elements_although_the_list_handles_their_taps() -> None:
    """No row of Clock's overflow menu is clickable: its ListView takes the taps."""
    tree = parse_tree(_dump("windows_popup_menu"))

    elements = actionable_elements(tree, "com.google.android.deskclock", _FIXTURE_SCREEN)

    assert _labels(elements) == [
        "Screen saver",
        "Settings",
        "Privacy policy",
        "Send feedback",
        "Help",
    ]
    assert [element.region for element in elements] == ["top right"] * 4 + ["right"]


def test_a_screen_without_an_active_application_window_has_no_elements() -> None:
    tree = parse_tree(
        '<displays><display id="0">'
        + _window(_node({"clickable": "true", "text": "Tap"}), active=False)
        + "</display></displays>"
    )

    assert acting_windows(tree, _APP) == []
    assert actionable_elements(tree, _APP, _SCREEN) == []


def test_a_scrollable_list_of_ten_rows_sharing_a_resource_id_is_one_list_item() -> None:
    rows = [_row(f"Item {number}", {"resource-id": f"{_APP}:id/row"}) for number in range(10)]
    results = _node(
        {
            "class": "androidx.recyclerview.widget.RecyclerView",
            "resource-id": f"{_APP}:id/results",
            "scrollable": "true",
        },
        *rows,
    )
    tree = _screen(results)

    elements = actionable_elements(tree, _APP, _SCREEN)

    assert [element.summary() for element in elements] == [
        "RecyclerView, scrollable (center)",
        'LinearLayout "Item 0", list item, one of 10 rows (center)',
    ]
    item = elements[1].locator
    assert item == Locator(
        f"{_APP}:id/row", "android.widget.LinearLayout", None, (f"{_APP}:id/results",)
    )
    assert len(find_nodes(tree, item)) == 10


def test_rows_that_share_a_resource_id_outside_a_scrollable_container_stay_apart() -> None:
    rows = [_row(f"Item {number}", {"resource-id": f"{_APP}:id/row"}) for number in range(3)]

    elements = actionable_elements(_screen(_node({}, *rows)), _APP, _SCREEN)

    assert _labels(elements) == ["Item 0", "Item 1", "Item 2"]
    assert [element.rows for element in elements] == [1, 1, 1]


def test_rows_without_a_resource_id_in_a_scrollable_list_stay_apart() -> None:
    """Settings' rows have no resource ID, and each opens its own page."""
    settings_list = _node({"scrollable": "true"}, _row("Wi-Fi"), _row("Bluetooth"))

    elements = actionable_elements(_screen(settings_list), _APP, _SCREEN)

    assert _labels(elements) == ["", "Wi-Fi", "Bluetooth"]


def test_nodes_that_share_a_locator_are_one_element() -> None:
    same = _node({"clickable": "true", "text": "More"})

    assert _labels(actionable_elements(_screen(_node({}, same, same)), _APP, _SCREEN)) == ["More"]


def test_a_toggle_inside_a_clickable_row_is_part_of_the_row() -> None:
    switch = _node(
        {
            "class": "android.widget.Switch",
            "checkable": "true",
            "resource-id": "android:id/switch_widget",
        }
    )
    row = _node(
        {"class": "android.widget.LinearLayout", "clickable": "true"},
        _node({"class": "android.widget.TextView", "text": "Use this SIM"}),
        switch,
    )

    assert _labels(actionable_elements(_screen(row), _APP, _SCREEN)) == ["Use this SIM"]
    assert _labels(actionable_elements(_screen(_node({}, switch)), _APP, _SCREEN)) == [""]


@pytest.mark.parametrize(
    ("label", "labels"), [("Forget", ["Forget"]), ("Share", ["Forget", "Share"])]
)
def test_a_clickable_node_with_the_label_of_the_clickable_node_it_is_in_is_part_of_it(
    label: str, labels: list[str]
) -> None:
    button = _node({"class": "android.widget.Button", "clickable": "true", "text": label})
    action = _node(
        {"class": "android.widget.LinearLayout", "clickable": "true", "content-desc": "Forget"},
        button,
    )

    assert _labels(actionable_elements(_screen(action), _APP, _SCREEN)) == labels


@pytest.mark.parametrize(
    "attributes",
    [
        pytest.param({"clickable": "true", "enabled": "false"}, id="disabled"),
        pytest.param({"clickable": "true", "bounds": "[10,10][10,50]"}, id="no area"),
        pytest.param({"focusable": "true", "selected": "true"}, id="not actionable"),
    ],
)
def test_a_node_the_user_cannot_act_on_is_no_element(attributes: dict[str, str]) -> None:
    assert actionable_elements(_screen(_node(attributes)), _APP, _SCREEN) == []


@pytest.mark.parametrize(
    ("attributes", "summary"),
    [
        pytest.param(
            {"long-clickable": "true", "text": "Photo"}, 'View "Photo" (center)', id="long-click"
        ),
        pytest.param(
            {"checkable": "true", "content-desc": "Star"}, 'View "Star" (center)', id="checkable"
        ),
        pytest.param(
            {"scrollable": "true", "text": "Feed"},
            'View "Feed", scrollable (center)',
            id="scrollable",
        ),
        pytest.param(
            {"class": "android.widget.EditText", "hint": "Name", "text": "Ada"},
            'EditText "Name", text field (center)',
            id="text field",
        ),
        pytest.param(
            {"class": "android.widget.AutoCompleteTextView", "content-desc": "To"},
            'AutoCompleteTextView "To", text field (center)',
            id="text field subclass",
        ),
    ],
)
def test_each_kind_of_actionable_node_is_an_element(
    attributes: dict[str, str], summary: str
) -> None:
    elements = actionable_elements(_screen(_node(attributes)), _APP, _SCREEN)

    assert [element.summary() for element in elements] == [summary]


@pytest.mark.parametrize(
    ("attributes", "children", "label"),
    [
        pytest.param({"text": "  Wi-Fi\n calling "}, (), "Wi-Fi calling", id="whitespace"),
        pytest.param({"text": "Wi-Fi", "content-desc": "Wireless"}, (), "Wi-Fi", id="text"),
        pytest.param({"content-desc": "Wireless"}, ("Wi-Fi",), "Wireless", id="description"),
        pytest.param({}, ("", "Wi-Fi", "Connected"), "Wi-Fi", id="first descendant text"),
        pytest.param({"scrollable": "true"}, ("Wi-Fi",), "", id="scrollable"),
        pytest.param({"class": "android.widget.EditText", "text": "Ada"}, (), "", id="typed"),
    ],
)
def test_a_locator_s_label(
    attributes: dict[str, str], children: tuple[str, ...], label: str
) -> None:
    texts = [_node({"class": "android.widget.TextView", "text": text}) for text in children]
    tree = _screen(_node({"clickable": "true"} | attributes, *texts))

    assert locator(tree, 0).label == label


def test_find_nodes_returns_every_node_a_locator_names_in_document_order() -> None:
    row = _row("Wi-Fi", {"resource-id": f"{_APP}:id/row"})
    other = _row("Bluetooth", {"resource-id": f"{_APP}:id/row"})
    tree = _screen(_node({}, row, other, row))
    wanted = locator(tree, 1)

    assert find_nodes(tree, wanted) == [1, 7]
    assert find_nodes(tree, dataclasses.replace(wanted, label=None)) == [1, 4, 7]
    assert find_nodes(tree, dataclasses.replace(wanted, ancestor_ids=("a:id/x",))) == []
    assert find_nodes(tree, dataclasses.replace(wanted, class_name="android.view.View")) == []


def test_a_locator_reads_back_what_it_writes() -> None:
    written = Locator("a:id/row", "android.widget.LinearLayout", None, ("a:id/list",))

    assert written.as_record() == {
        "resource_id": "a:id/row",
        "class": "android.widget.LinearLayout",
        "label": None,
        "ancestor_ids": ["a:id/list"],
    }
    assert Locator.from_record(json.loads(json.dumps(written.as_record()))) == written


@pytest.mark.parametrize(
    "record",
    [
        pytest.param(None, id="none"),
        pytest.param([], id="not an object"),
        pytest.param({"class": "View", "label": "", "ancestor_ids": []}, id="no resource ID"),
        pytest.param({"resource_id": "", "label": "", "ancestor_ids": []}, id="no class"),
        pytest.param(
            {"resource_id": "", "class": "View", "label": 3, "ancestor_ids": []}, id="label"
        ),
        pytest.param({"resource_id": "", "class": "View", "label": ""}, id="no ancestors"),
        pytest.param(
            {"resource_id": "", "class": "View", "label": "", "ancestor_ids": [1]}, id="ancestor"
        ),
    ],
)
def test_a_malformed_locator_record_reads_as_none(record: object) -> None:
    assert Locator.from_record(record) is None


@pytest.mark.parametrize(
    ("bounds", "where"),
    [
        (Bounds(0, 0, 100, 100), "top left"),
        (Bounds(0, 0, 1000, 100), "top"),
        (Bounds(900, 0, 1000, 100), "top right"),
        (Bounds(0, 900, 100, 1100), "left"),
        (Bounds(0, 0, 1000, 2000), "center"),
        (Bounds(900, 900, 1000, 1100), "right"),
        (Bounds(0, 1900, 100, 2000), "bottom left"),
        (Bounds(0, 1900, 1000, 2000), "bottom"),
        (Bounds(900, 1900, 1000, 2000), "bottom right"),
    ],
)
def test_a_region_is_where_the_centre_lies_in_thirds_of_the_screen(
    bounds: Bounds, where: str
) -> None:
    assert region(bounds, _SCREEN) == where


def test_a_screen_without_area_has_no_regions() -> None:
    assert region(Bounds(0, 0, 10, 10), Bounds(0, 0, 0, 0)) == ""


def test_a_summary_shortens_a_long_label() -> None:
    tree = _screen(_node({"clickable": "true", "text": "x" * 80}))

    assert actionable_elements(tree, _APP, _SCREEN)[0].summary() == f'View "{"x" * 59}…" (center)'


def test_a_summary_names_no_region_on_a_screen_without_area() -> None:
    tree = _screen(_node({"clickable": "true", "text": "Save"}))

    assert actionable_elements(tree, _APP, Bounds(0, 0, 0, 0))[0].summary() == 'View "Save"'


_MENU = _node({"clickable": "true", "text": "Menu", "bounds": "[0,0][100,100]"})


@pytest.mark.parametrize(
    ("roots", "reason"),
    [
        pytest.param((), "0 actionable element(s), fewer than 1", id="no element"),
        pytest.param(
            (_node({"class": "android.webkit.WebView"}, _MENU),),
            "a WebView covers most of the window",
            id="web view",
        ),
        pytest.param(
            (
                _node(
                    {},
                    _MENU,
                    _node({"class": "android.view.SurfaceView", "bounds": "[0,100][1000,2000]"}),
                ),
            ),
            "a SurfaceView covers most of the window",
            id="canvas",
        ),
    ],
)
def test_a_thin_tree_says_why(roots: tuple[str, ...], reason: str) -> None:
    tree = _screen(*roots)

    assert thin_tree_reason(tree, _APP, actionable_elements(tree, _APP, _SCREEN)) == reason


def test_a_screen_that_one_web_view_fills_is_thin() -> None:
    tree = parse_tree(_dump("windows_web_view"))
    elements = actionable_elements(tree, "com.example.app", _FIXTURE_SCREEN)

    assert [element.summary() for element in elements] == ["WebView, scrollable (center)"]
    assert thin_tree_reason(tree, "com.example.app", elements) == (
        "a WebView covers most of the window"
    )


@pytest.mark.parametrize(
    "roots",
    [
        pytest.param(
            (
                _node(
                    {},
                    _MENU,
                    _node({"class": "android.webkit.WebView", "bounds": "[0,0][1000,500]"}),
                ),
            ),
            id="a small web view",
        ),
        pytest.param((_node({}, _MENU),), id="a view with children"),
    ],
)
def test_a_tree_with_elements_and_no_large_drawn_node_is_not_thin(roots: tuple[str, ...]) -> None:
    tree = _screen(*roots)

    assert thin_tree_reason(tree, _APP, actionable_elements(tree, _APP, _SCREEN)) is None


def _hit(dump: str, point: tuple[int, int]) -> Hit:
    return hit_test(parse_tree(_dump(dump)), point, _FIXTURE_SCREEN)


def _element_at(tree: UiTree, package: str, label: str) -> TreeElement:
    return next(
        element
        for element in actionable_elements(tree, package, _FIXTURE_SCREEN)
        if element.label == label
    )


def test_a_tap_in_a_button_s_padding_resolves_to_the_button() -> None:
    """The point is inside the "Disable app" button and outside its text."""
    tree = parse_tree(_dump("windows_dialog_compose"))
    button = _element_at(tree, "com.android.settings", "Disable app")

    # The Compose views around the button share its locator, and none of them is clickable.
    (node,) = [index for index in find_nodes(tree, button.locator) if tree.nodes[index].clickable]
    assert hit_test(tree, (820, 1660), _FIXTURE_SCREEN) == Hit(
        "element", Anchor(button.locator, Bounds(799, 1649, 1171, 1793)), node
    )


def test_a_tap_on_the_text_inside_a_clickable_row_resolves_to_the_row() -> None:
    tree = parse_tree(_dump("windows_plain_list"))
    row = _element_at(tree, "com.android.settings", "Network & internet")

    hit = hit_test(tree, (400, 780), _FIXTURE_SCREEN)

    (node,) = find_nodes(tree, row.locator)
    assert hit == Hit("element", Anchor(row.locator, Bounds(0, 699, 1344, 917)), node)
    assert hit.anchor is not None
    assert hit.anchor.as_record() == {
        "locator": row.locator.as_record(),
        "bounds": [0, 699, 1344, 917],
    }


@pytest.mark.parametrize(
    ("dump", "point"),
    [
        pytest.param("windows_dialog_compose", (672, 600), id="a dialog"),
        pytest.param("windows_popup_menu", (100, 2000), id="a popup menu"),
    ],
)
def test_a_tap_outside_an_open_dialog_resolves_to_outside_dialog(
    dump: str, point: tuple[int, int]
) -> None:
    """A dialog or menu that takes focus hides the screen under it from the dump."""
    assert _hit(dump, point) == Hit("outside_dialog")


def test_a_tap_on_the_keyboard_resolves_to_the_keyboard_not_an_app_element() -> None:
    """The app's window spans the whole screen under the keyboard's."""
    assert _hit("windows_keyboard", (672, 2500)) == Hit("keyboard")


def test_a_tap_on_the_status_bar_resolves_to_a_system_window() -> None:
    assert _hit("windows_plain_list", (600, 50)) == Hit("system")


def _overlapping(first: dict[str, str], second: dict[str, str]) -> UiTree:
    """Two clickable siblings labelled "First" and "Second" that overlap at 300, 300."""
    return _screen(
        _node(
            {},
            _node({"clickable": "true", "text": "First", "bounds": "[0,0][500,500]"} | first),
            _node({"clickable": "true", "text": "Second", "bounds": "[200,200][700,700]"} | second),
        )
    )


@pytest.mark.parametrize(
    ("first", "second", "label"),
    [
        pytest.param({"drawing-order": "2"}, {"drawing-order": "1"}, "First", id="drawn last"),
        pytest.param({"drawing-order": "1"}, {"drawing-order": "1"}, "Second", id="later in dump"),
        pytest.param({}, {"drawing-order": "3"}, "Second", id="drawn last and later"),
    ],
)
def test_of_two_overlapping_siblings_the_one_drawn_last_wins(
    first: dict[str, str], second: dict[str, str], label: str
) -> None:
    hit = hit_test(_overlapping(first, second), (300, 300), _SCREEN)

    assert hit.anchor is not None
    assert hit.anchor.locator.label == label


def test_a_tap_the_sibling_drawn_last_does_not_take_reaches_the_one_under_it() -> None:
    """Android offers the touch to the next sibling when the one drawn last keeps none."""
    tree = _overlapping({"drawing-order": "1"}, {"drawing-order": "2", "clickable": "false"})

    hit = hit_test(tree, (300, 300), _SCREEN)

    assert hit.anchor is not None
    assert hit.anchor.locator.label == "First"


def test_a_tap_on_nothing_that_takes_taps_resolves_to_the_deepest_node_under_it() -> None:
    """Between two rows of Settings' list, the list itself is under the point."""
    tree = parse_tree(_dump("windows_plain_list"))

    hit = hit_test(tree, (700, 1000), _FIXTURE_SCREEN)

    assert hit.kind == "element"
    assert hit.anchor is not None
    assert hit.anchor.locator.resource_id == "com.android.settings:id/recycler_view"


def test_a_tap_on_a_list_row_the_list_takes_resolves_to_the_row() -> None:
    """No row of Clock's overflow menu is clickable; its ListView takes their taps."""
    tree = parse_tree(_dump("windows_popup_menu"))
    row = _element_at(tree, "com.google.android.deskclock", "Settings")
    bounds = row.bounds
    centre = ((bounds.left + bounds.right) // 2, (bounds.top + bounds.bottom) // 2)

    (node,) = find_nodes(tree, row.locator)
    assert hit_test(tree, centre, _FIXTURE_SCREEN) == Hit(
        "element", Anchor(row.locator, bounds), node
    )


@pytest.mark.parametrize(
    ("point", "label"),
    [
        pytest.param((700, 250), "You can find the privacy policy here", id="on the popup"),
        pytest.param((1280, 250), "More options", id="beside it"),
    ],
)
def test_a_tap_beside_a_popup_that_does_not_take_focus_reaches_the_window_below(
    point: tuple[int, int], label: str
) -> None:
    hit = _hit("windows_child_popup_window", point)

    assert hit.anchor is not None
    assert hit.anchor.locator.label == label


def test_a_tap_on_a_clickable_node_inside_one_with_its_label_resolves_to_the_outer_one() -> None:
    """The anchor is the element ``actionable_elements`` lists, not a part of it."""
    tree = _screen(
        _node(
            {"class": "android.widget.LinearLayout", "clickable": "true", "content-desc": "Forget"},
            _node({"class": "android.widget.Button", "clickable": "true", "text": "Forget"}),
        )
    )
    (element,) = actionable_elements(tree, _APP, _SCREEN)

    hit = hit_test(tree, (500, 500), _SCREEN)

    assert hit.anchor is not None
    assert hit.anchor.locator == element.locator


@pytest.mark.parametrize(
    ("tree", "point"),
    [
        pytest.param(_screen(_node({"clickable": "true"})), (1000, 10), id="off the screen"),
        pytest.param(parse_tree("<displays />"), (10, 10), id="no window"),
        pytest.param(
            _screen(_node({"clickable": "true", "bounds": "[0,0][10,10]"})),
            (500, 500),
            id="no node under it",
        ),
        pytest.param(
            parse_tree(
                '<displays><display id="0">'
                + _window(
                    _node({"clickable": "true", "bounds": "[0,0][10,10]"}),
                    active=False,
                    bounds="[0,0][10,10]",
                )
                + "</display></displays>"
            ),
            (500, 500),
            id="no window under it and none took focus",
        ),
    ],
)
def test_a_tap_under_no_node_resolves_to_nothing(tree: UiTree, point: tuple[int, int]) -> None:
    assert hit_test(tree, point, _SCREEN) == Hit("nothing")


def test_the_focused_text_field_is_the_one_with_input_focus_in_the_app() -> None:
    tree = parse_tree(_dump("windows_keyboard"))

    field = focused_text_field(tree)

    assert field is not None
    assert field.locator.label == "Search settings"
    assert field.bounds == Bounds(144, 151, 1344, 367)


@pytest.mark.parametrize(
    "tree",
    [
        pytest.param(parse_tree(_dump("windows_plain_list")), id="nothing focused"),
        pytest.param(
            _screen(_node({"class": "android.widget.Button", "focused": "true"})),
            id="a focused button",
        ),
    ],
)
def test_a_screen_without_a_focused_text_field_has_none(tree: UiTree) -> None:
    assert focused_text_field(tree) is None


@pytest.mark.parametrize(
    ("dump", "shown"), [("windows_keyboard", True), ("windows_plain_list", False)]
)
def test_the_keyboard_is_shown_when_its_window_is_in_the_tree(dump: str, shown: bool) -> None:
    assert keyboard_shown(parse_tree(_dump(dump))) is shown


def test_a_list_item_s_locator_matches_each_row_and_a_row_s_only_itself() -> None:
    row = Locator("a:id/row", "android.widget.LinearLayout", "Wi-Fi", ("a:id/list",))
    item = dataclasses.replace(row, label=None)
    other = dataclasses.replace(row, label="Bluetooth")

    assert item.matches(row)
    assert item.matches(other)
    assert row.matches(row)
    assert not row.matches(other)
    assert not row.matches(item)
    assert not item.matches(dataclasses.replace(row, ancestor_ids=()))


def test_a_node_is_part_of_the_element_it_is_or_lies_inside() -> None:
    """The edit field of a date picker lies inside the picker."""
    picker = _node(
        {"class": "android.widget.NumberPicker", "scrollable": "true", "resource-id": "a:id/day"},
        _node({"class": "android.widget.EditText", "clickable": "true", "text": "21"}),
    )
    tree = _screen(_node({}, picker), _node({"clickable": "true", "text": "Set"}))
    day, field, done = locator(tree, 1), locator(tree, 2), locator(tree, 3)

    assert part_of(tree, 1, day)
    assert part_of(tree, 2, day)
    assert part_of(tree, 2, field)
    assert not part_of(tree, 1, field)
    assert not part_of(tree, 3, day)
    assert part_of(tree, 3, done)


def test_a_button_is_not_part_of_the_layout_that_shares_another_button_s_locator() -> None:
    """In the Compose dialog the views around both buttons take the label of the first
    text inside them, "Disable app", so they share that button's locator."""
    tree = parse_tree(_dump("windows_dialog_compose"))
    disable = _element_at(tree, "com.android.settings", "Disable app").locator
    cancel = _element_at(tree, "com.android.settings", "Cancel").locator
    (cancel_node,) = [i for i in find_nodes(tree, cancel) if tree.nodes[i].clickable]

    assert len(find_nodes(tree, disable)) > 1
    assert not part_of(tree, cancel_node, disable)
    assert part_of(tree, cancel_node, cancel)
