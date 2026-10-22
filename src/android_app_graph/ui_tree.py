"""The accessibility tree of a screen, as ``uiautomator dump --windows`` prints it.

The dump lists every window of every display, top to bottom:
``<displays><display id><window …><hierarchy rotation><node …>``, each ``<node>``
nesting its children, and then prints ``UI hierchary dumped to: <file>``, Android's
spelling, right after the XML. A window that is a child of another, such as a popup
anchored to the app's window, is listed among the display's windows and again inside
its parent's ``<window>`` element; only the display's own windows are read, so each
window is read once.

Window IDs change from one dump to the next, and ``layer`` is only the windows'
order, so two dumps are compared by window type, layer and bounds.

A screen's layout signature is the part of its tree that names its structure rather
than its content; a node's stable layout is the part of its members' signatures that
most of them share. Layout is evidence for screen matching, never a decision.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Literal, NamedTuple

IDLE_FAILURE = "ERROR: could not get idle state."
SYSTEM_WINDOW = "TYPE_SYSTEM"
APPLICATION_WINDOW = "TYPE_APPLICATION"
# Android's accessibility title for every PopupWindow: menus, dropdowns and tooltips.
POPUP_TITLE = "Popup Window"
STABLE_LAYOUT_SHARE = 0.8
# A layout of fewer entries says too little to compare: one or two controls match
# every screen that has them.
MIN_LAYOUT_ENTRIES = 3

_TRAILER = "UI hierchary dumped to:"
_BOUNDS = re.compile(r"\[(-?\d+),(-?\d+)\]\[(-?\d+),(-?\d+)\]")


class UnusableDumpError(ValueError):
    """A dump with no usable tree; ``reason`` says why, as observation records keep it."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class Bounds:
    """A rectangle in screen pixels."""

    left: int
    top: int
    right: int
    bottom: int


@dataclass(frozen=True, slots=True)
class UiWindow:
    """One window of the screen.

    ``window_type`` is how ``AccessibilityWindowInfo.typeToString`` names its type,
    such as ``TYPE_APPLICATION``, ``TYPE_INPUT_METHOD`` for the keyboard or
    ``TYPE_SYSTEM`` for the status bar; a window with a greater ``layer`` is on top.
    """

    window_id: int
    display_id: int
    window_type: str
    title: str
    layer: int
    bounds: Bounds
    active: bool
    focused: bool


@dataclass(frozen=True, slots=True)
class UiNode:
    """One node of a window's hierarchy.

    ``window`` is its window's index in ``UiTree.windows``; ``parent`` and
    ``children`` are indices in ``UiTree.nodes``, and a window's root has no parent.
    ``bounds`` is the part of the node visible on the screen.
    """

    window: int
    parent: int | None
    children: tuple[int, ...]
    resource_id: str
    class_name: str
    package: str
    text: str
    content_description: str
    hint: str
    bounds: Bounds
    drawing_order: int
    checkable: bool
    checked: bool
    clickable: bool
    enabled: bool
    focusable: bool
    focused: bool
    long_clickable: bool
    password: bool
    scrollable: bool
    selected: bool


@dataclass(frozen=True, slots=True)
class UiTree:
    """Every window of a screen, top to bottom, and their nodes: each window's in
    document order, a parent before its children, and the windows' in window order."""

    windows: tuple[UiWindow, ...]
    nodes: tuple[UiNode, ...]


def xml_document(dump: str) -> str:
    """Return ``dump``, what ``uiautomator dump`` printed, without the line it prints
    after the XML."""
    end = dump.rfind(_TRAILER)
    return (dump if end < 0 else dump[:end]).strip()


def parse_tree(dump: str) -> UiTree:
    """Return the tree in ``dump``, what ``uiautomator dump --windows`` printed, with
    or without the line it prints after the XML.

    Raises ``UnusableDumpError`` for empty output, an ``ERROR: …`` message such as
    ``IDLE_FAILURE``, XML that does not parse, a root other than ``<displays>``, which
    a dump without ``--windows`` has, and a number or bounds that do not parse.
    """
    document = xml_document(dump)
    if not document:
        msg = "empty output"
        raise UnusableDumpError(msg)
    if document.startswith("ERROR:"):
        raise UnusableDumpError(document.splitlines()[0])
    try:
        # The XML comes from the device under test. ElementTree expands no external
        # entity, and Expat 2.4.1 and later resist billion laughs and quadratic blowup:
        # https://docs.python.org/3/library/xml.html#xml-vulnerabilities
        root = ET.fromstring(document)  # noqa: S314  # safe with Expat >= 2.4.1, as the comment above says
    except ET.ParseError as exc:
        msg = f"XML does not parse: {exc}"
        raise UnusableDumpError(msg) from None
    if root.tag != "displays":
        msg = f"root is <{root.tag}>, not <displays>"
        raise UnusableDumpError(msg)
    windows: list[UiWindow] = []
    nodes: list[UiNode] = []
    for display in root.iterfind("display"):
        display_id = _number(display, "id")
        for element in display.iterfind("window"):
            windows.append(_window(element, display_id))
            hierarchy = element.find("hierarchy")
            if hierarchy is not None:
                nodes.extend(_window_nodes(hierarchy, len(windows) - 1, first=len(nodes)))
    return UiTree(tuple(windows), tuple(nodes))


def same_screen(first: UiTree, second: UiTree) -> bool:
    """Return whether ``first`` and ``second`` show the same screen.

    They do when their windows other than ``TYPE_SYSTEM`` ones, whose clock changes
    every minute, have the same type, layer and bounds, in the same order, and the
    same nodes, in the same places, by class, resource ID, text, content description,
    bounds, and checked, selected, focused and enabled flags.
    """
    return _screen_key(first) == _screen_key(second)


type _NodeKey = tuple[str, str, str, str, Bounds, bool, bool, bool, bool, int]
type _WindowKey = tuple[tuple[str, int, Bounds], list[_NodeKey]]


def _screen_key(tree: UiTree) -> list[_WindowKey]:
    keys: list[_WindowKey] = []
    for index, window in enumerate(tree.windows):
        if window.window_type == SYSTEM_WINDOW:
            continue
        # A preorder walk with each node's number of children determines the nesting.
        node_keys = [
            (
                node.class_name,
                node.resource_id,
                node.text,
                node.content_description,
                node.bounds,
                node.checked,
                node.selected,
                node.focused,
                node.enabled,
                len(node.children),
            )
            for node in tree.nodes
            if node.window == index
        ]
        keys.append(((window.window_type, window.layer, window.bounds), node_keys))
    return keys


type WindowRole = Literal["base", "dialog", "popup"]


class LayoutEntry(NamedTuple):
    """One entry of a layout signature, of one of three kinds:

    - ``title``: an app window's role and its title;
    - ``action``: the window role, resource ID and class of a visible actionable node;
    - ``text``: the resource ID, class and label of a visible static text outside
      scrolling containers.

    The fields a kind does not use are empty.
    """

    kind: Literal["title", "action", "text"]
    window_role: str
    resource_id: str
    class_name: str
    label: str


def window_roles(tree: UiTree) -> dict[int, WindowRole]:
    """Return the role of each app window of ``tree``, by its index in ``tree.windows``.

    A ``TYPE_APPLICATION`` window titled ``POPUP_TITLE`` is a ``popup``; one that spans
    the width of the widest window and starts at the top of the highest one is the
    ``base`` screen; any other is a ``dialog``. Every other window type, the keyboard
    and the system bars among them, has no role.
    """
    if not tree.windows:
        return {}
    left = min(window.bounds.left for window in tree.windows)
    right = max(window.bounds.right for window in tree.windows)
    top = min(window.bounds.top for window in tree.windows)
    roles: dict[int, WindowRole] = {}
    for index, window in enumerate(tree.windows):
        if window.window_type != APPLICATION_WINDOW:
            continue
        bounds = window.bounds
        if window.title == POPUP_TITLE:
            roles[index] = "popup"
        elif bounds.left <= left and bounds.right >= right and bounds.top <= top:
            roles[index] = "base"
        else:
            roles[index] = "dialog"
    return roles


def layout_signature(tree: UiTree) -> frozenset[LayoutEntry]:
    """Return the layout signature of the screen ``tree`` shows: the entries of its app
    windows that ``LayoutEntry`` describes.

    A node is visible when its bounds have an area, actionable when it is clickable,
    long-clickable, checkable or scrollable, and static text when it is neither and has
    a text or, without one, a content description, its label. A scrolling container is
    a scrollable ancestor: what scrolls is content, not structure.
    """
    roles = window_roles(tree)
    entries = {
        LayoutEntry("title", role, "", "", tree.windows[index].title)
        for index, role in roles.items()
        if tree.windows[index].title
    }
    for index, node in enumerate(tree.nodes):
        role = roles.get(node.window)
        if role is None or not _visible(node.bounds):
            continue
        if node.clickable or node.long_clickable or node.checkable or node.scrollable:
            entries.add(LayoutEntry("action", role, node.resource_id, node.class_name, ""))
            continue
        label = node.text or node.content_description
        if label and not _in_scrolling_container(tree, index):
            entries.add(LayoutEntry("text", "", node.resource_id, node.class_name, label))
    return frozenset(entries)


def stable_layout(
    signatures: Collection[frozenset[LayoutEntry]], *, share: float = STABLE_LAYOUT_SHARE
) -> dict[LayoutEntry, float]:
    """Return the entries on at least ``share`` of ``signatures``, each weighted by the
    share it is on; empty for no signature."""
    if not signatures:
        return {}
    counts = Counter(entry for signature in signatures for entry in signature)
    shares = {entry: count / len(signatures) for entry, count in counts.items()}
    return {entry: weight for entry, weight in shares.items() if weight >= share}


def layout_similarity(
    first: Mapping[LayoutEntry, float], second: Mapping[LayoutEntry, float]
) -> float | None:
    """Return the weighted Jaccard similarity of two weighted layouts: a screen's
    signature weighs each of its entries 1, a node's stable layout each by its share.

    ``None`` when either has fewer than ``MIN_LAYOUT_ENTRIES`` entries.
    """
    if len(first) < MIN_LAYOUT_ENTRIES or len(second) < MIN_LAYOUT_ENTRIES:
        return None
    entries = first.keys() | second.keys()
    shared = sum(min(first.get(entry, 0.0), second.get(entry, 0.0)) for entry in entries)
    either = sum(max(first.get(entry, 0.0), second.get(entry, 0.0)) for entry in entries)
    return shared / either


def _visible(bounds: Bounds) -> bool:
    return bounds.right > bounds.left and bounds.bottom > bounds.top


def _in_scrolling_container(tree: UiTree, index: int) -> bool:
    parent = tree.nodes[index].parent
    while parent is not None:
        if tree.nodes[parent].scrollable:
            return True
        parent = tree.nodes[parent].parent
    return False


def _window(element: ET.Element, display_id: int) -> UiWindow:
    return UiWindow(
        window_id=_number(element, "id"),
        display_id=display_id,
        window_type=element.get("type", ""),
        title=element.get("title", ""),
        layer=_number(element, "layer"),
        bounds=_bounds(element),
        active=_flag(element, "active"),
        focused=_flag(element, "focused"),
    )


def _window_nodes(hierarchy: ET.Element, window: int, *, first: int) -> list[UiNode]:
    """Return the nodes under ``hierarchy`` in document order, numbered from ``first``.

    The walk keeps its own stack, so a deep tree from the device cannot exhaust
    Python's recursion limit.
    """
    ordered: list[tuple[ET.Element, int | None]] = []
    stack: list[tuple[ET.Element, int | None]] = [
        (child, None) for child in reversed(hierarchy.findall("node"))
    ]
    while stack:
        element, parent = stack.pop()
        index = first + len(ordered)
        ordered.append((element, parent))
        stack.extend((child, index) for child in reversed(element.findall("node")))
    children: dict[int, list[int]] = {}
    for offset, (_element, parent) in enumerate(ordered):
        if parent is not None:
            children.setdefault(parent, []).append(first + offset)
    return [
        _node(element, window, parent, tuple(children.get(first + offset, ())))
        for offset, (element, parent) in enumerate(ordered)
    ]


def _node(
    element: ET.Element, window: int, parent: int | None, children: tuple[int, ...]
) -> UiNode:
    return UiNode(
        window=window,
        parent=parent,
        children=children,
        resource_id=element.get("resource-id", ""),
        class_name=element.get("class", ""),
        package=element.get("package", ""),
        text=element.get("text", ""),
        content_description=element.get("content-desc", ""),
        hint=element.get("hint", ""),
        bounds=_bounds(element),
        drawing_order=_number(element, "drawing-order"),
        checkable=_flag(element, "checkable"),
        checked=_flag(element, "checked"),
        clickable=_flag(element, "clickable"),
        enabled=_flag(element, "enabled"),
        focusable=_flag(element, "focusable"),
        focused=_flag(element, "focused"),
        long_clickable=_flag(element, "long-clickable"),
        password=_flag(element, "password"),
        scrollable=_flag(element, "scrollable"),
        selected=_flag(element, "selected"),
    )


def _flag(element: ET.Element, name: str) -> bool:
    return element.get(name) == "true"


def _number(element: ET.Element, name: str) -> int:
    value = element.get(name, "")
    if not re.fullmatch(r"-?[0-9]+", value):
        msg = f"<{element.tag}> {name}={value!r} is not a number"
        raise UnusableDumpError(msg)
    return int(value)


def _bounds(element: ET.Element) -> Bounds:
    value = element.get("bounds", "")
    match = _BOUNDS.fullmatch(value)
    if match is None:
        msg = f"<{element.tag}> bounds={value!r} do not parse"
        raise UnusableDumpError(msg)
    left, top, right, bottom = (int(number) for number in match.groups())
    return Bounds(left, top, right, bottom)
