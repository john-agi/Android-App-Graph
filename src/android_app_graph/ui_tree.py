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

The elements a user can act on are read from the windows the user can act on, and
each has a ``Locator`` that finds it again in another tree of the same screen. A
tap's ``hit_test`` says which element, or which other window, the tap lands on, and
``resolve_anchor`` where that element is in another tree.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

IDLE_FAILURE = "ERROR: could not get idle state."
SYSTEM_WINDOW = "TYPE_SYSTEM"
APPLICATION_WINDOW = "TYPE_APPLICATION"
INPUT_METHOD_WINDOW = "TYPE_INPUT_METHOD"
# Chosen from the 69 trees of a 60-step Settings exploration: its smallest screens,
# dropdown menus of one row, list that one element correctly, so only a tree that
# lists none is too thin by count.
MIN_ACTIONABLE_ELEMENTS = 1

_TRAILER = "UI hierchary dumped to:"
_BOUNDS = re.compile(r"\[(-?\d+),(-?\d+)\]\[(-?\d+),(-?\d+)\]")
# Subclasses that keep EditText's accessibility class name report it, Compose text
# fields included; the dump has no attribute that says a node takes text.
_TEXT_FIELD_SUFFIXES = ("EditText", "AutoCompleteTextView")
# An AdapterView handles its rows' taps itself, so a popup menu's rows are not clickable.
_ADAPTER_LISTS = frozenset(
    {
        "android.widget.ListView",
        "android.widget.GridView",
        "android.widget.ExpandableListView",
    }
)
_WEB_VIEW = "android.webkit.WebView"
# A leaf of one of these classes draws its content itself, which the tree cannot show.
_CANVAS_CLASSES = frozenset(
    {
        "android.view.View",
        "android.view.SurfaceView",
        "android.view.TextureView",
        "android.opengl.GLSurfaceView",
    }
)
_MOST_OF_WINDOW = 0.5
_LABEL_LIMIT = 60
_BOUNDS_COORDINATES = 4


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


@dataclass(frozen=True, slots=True)
class Locator:
    """What finds a node again in another tree: its resource ID, class, label and the
    resource IDs of its ancestors, outermost first, leaving out ancestors without one.

    A node's label is its text, else its content description, else the first text
    among its descendants, which in a list row is its title and not the summary that
    changes with the setting. A scrollable node's label is its own text or content
    description only, since scrolling changes its descendants, and a text field's is
    its hint, else its content description, since its text is what was typed.
    ``label`` is ``None`` for a list item, which stands for every row that differs only
    in label.
    """

    resource_id: str
    class_name: str
    label: str | None
    ancestor_ids: tuple[str, ...]

    def as_record(self) -> dict[str, object]:
        return {
            "resource_id": self.resource_id,
            "class": self.class_name,
            "label": self.label,
            "ancestor_ids": list(self.ancestor_ids),
        }

    @classmethod
    def from_record(cls, record: object) -> Locator | None:
        """Return the locator ``as_record`` wrote, or ``None`` for anything else."""
        if not isinstance(record, Mapping):
            return None
        resource_id = record.get("resource_id")
        class_name = record.get("class")
        label = record.get("label")
        ancestor_ids = record.get("ancestor_ids")
        if not isinstance(resource_id, str) or not isinstance(class_name, str):
            return None
        if label is not None and not isinstance(label, str):
            return None
        if not isinstance(ancestor_ids, list):
            return None
        ids = [ancestor for ancestor in ancestor_ids if isinstance(ancestor, str)]
        if len(ids) != len(ancestor_ids):
            return None
        return cls(resource_id, class_name, label, tuple(ids))

    def matches(self, found: Locator) -> bool:
        """Return whether ``found``, a node's locator, names what this locator does: the
        same node, or for a list item one of its rows."""
        return (
            found.resource_id == self.resource_id
            and found.class_name == self.class_name
            and found.ancestor_ids == self.ancestor_ids
            and (self.label is None or found.label == self.label)
        )


@dataclass(frozen=True, slots=True)
class Anchor:
    """The element an action acted on, as the tree it was planned from shows it: its
    locator and its bounds then."""

    locator: Locator
    bounds: Bounds

    def as_record(self) -> dict[str, object]:
        bounds = self.bounds
        return {
            "locator": self.locator.as_record(),
            "bounds": [bounds.left, bounds.top, bounds.right, bounds.bottom],
        }

    @classmethod
    def from_record(cls, record: object) -> Anchor | None:
        """Return the anchor ``as_record`` wrote, or ``None`` for anything else."""
        if not isinstance(record, Mapping):
            return None
        found = Locator.from_record(record.get("locator"))
        bounds = record.get("bounds")
        if found is None or not isinstance(bounds, list) or len(bounds) != _BOUNDS_COORDINATES:
            return None
        coordinates = [
            value for value in bounds if isinstance(value, int) and not isinstance(value, bool)
        ]
        if len(coordinates) != _BOUNDS_COORDINATES:
            return None
        left, top, right, bottom = coordinates
        return cls(found, Bounds(left, top, right, bottom))


type HitKind = Literal["element", "outside_dialog", "keyboard", "system", "nothing"]
type AnchorMiss = Literal["no_match", "ambiguous", "covered"]


@dataclass(frozen=True, slots=True)
class Hit:
    """What a tap at one point of a screen lands on.

    ``kind`` is ``element`` for a node of an application window, ``anchor`` the element
    that node is part of and ``node`` that element's index in the tree;
    ``outside_dialog`` for a point outside a dialog or popup that took focus, where a
    tap dismisses it; ``keyboard`` for the keyboard where it draws; ``system`` for any other
    window that is not an application's, such as the status bar; and ``nothing`` for a
    point off the screen or under no node.
    """

    kind: HitKind
    anchor: Anchor | None = None
    node: int | None = None


@dataclass(frozen=True, slots=True)
class TreeElement:
    """An element the user can act on: its locator, its label (a list item's is its
    first row's), its bounds and coarse ``region`` on the screen, and ``rows``, the
    number of rows a list item stands for, or 1."""

    locator: Locator
    label: str
    bounds: Bounds
    region: str
    scrollable: bool
    text_field: bool
    rows: int

    def summary(self) -> str:
        """Return the element in a few words: its short class, its label and traits,
        and its region, as ``Button "Save" (bottom right)``."""
        words = [self.locator.class_name.rsplit(".", 1)[-1] or "node"]
        if self.label:
            shortened = (
                self.label
                if len(self.label) <= _LABEL_LIMIT
                else f"{self.label[: _LABEL_LIMIT - 1]}…"
            )
            words.append(f'"{shortened}"')
        traits = []
        if self.rows > 1:
            traits.append(f"list item, one of {self.rows} rows")
        if self.text_field:
            traits.append("text field")
        if self.scrollable:
            traits.append("scrollable")
        described = " ".join(words) + "".join(f", {trait}" for trait in traits)
        return f"{described} ({self.region})" if self.region else described


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


def acting_windows(tree: UiTree, package: str) -> list[int]:
    """Return the indices of the windows the user can act on, top to bottom: the
    active application window and the application windows of ``package`` above it on
    its display, such as a popup that does not take focus. The keyboard and system
    windows are never among them, and a window that takes focus hides the windows
    below it from the dump."""
    active = _active_application_window(tree)
    if active is None:
        return []
    base = tree.windows[active]
    packages: dict[int, str] = {}
    for node in tree.nodes:
        if node.parent is None:
            packages.setdefault(node.window, node.package)
    return [
        index
        for index, window in enumerate(tree.windows)
        if index == active
        or (
            window.window_type == APPLICATION_WINDOW
            and window.display_id == base.display_id
            and window.layer > base.layer
            and packages.get(index) == package
        )
    ]


def locator(tree: UiTree, index: int) -> Locator:
    """Return the locator of node ``index`` of ``tree``."""
    node = tree.nodes[index]
    ancestor_ids = [
        tree.nodes[ancestor].resource_id
        for ancestor in _ancestors(tree, index)
        if tree.nodes[ancestor].resource_id
    ]
    return Locator(
        node.resource_id, node.class_name, _label(tree, index), tuple(reversed(ancestor_ids))
    )


def find_nodes(tree: UiTree, wanted: Locator) -> list[int]:
    """Return the indices of every node of ``tree`` that ``wanted`` locates, in document
    order; a list item's locator matches each of its rows. The caller decides what one
    match, none or several mean."""
    return [
        index
        for index, node in enumerate(tree.nodes)
        if node.resource_id == wanted.resource_id
        and node.class_name == wanted.class_name
        and wanted.matches(locator(tree, index))
    ]


def hit_test(tree: UiTree, point: tuple[int, int], screen: Bounds) -> Hit:
    """Return what a tap at ``point``, in the pixels of ``screen``, lands on in ``tree``.

    The windows of the active application window's display are walked from the highest
    layer down, and the first that covers the point takes the tap: a window covers its
    bounds, and the keyboard's window only the rectangle around what the keyboard
    draws, so a tap beside a floating keyboard reaches the window below. In an
    application window the tap goes where Android dispatches a touch: down through the
    nodes under the point, the one drawn last first, to the deepest node that takes
    taps (clickable, long-clickable, or a row of a list that takes its rows' taps), or
    the deepest node under the point when none does. The anchor is the element that
    node is part of, as ``actionable_elements`` lists it. A point on the screen that no
    window covers lies outside the dialog or popup that took focus when an application
    window took focus, since such a window hides the windows below it from the dump.
    """
    x, y = point
    if not _holds(screen, x, y) or not tree.windows:
        return Hit("nothing")
    active = _active_application_window(tree)
    display = tree.windows[0 if active is None else active].display_id
    on_display = [i for i, window in enumerate(tree.windows) if window.display_id == display]
    for index in sorted(on_display, key=lambda i: -tree.windows[i].layer):
        window = tree.windows[index]
        if not _covers(tree, index, x, y):
            continue
        if window.window_type == INPUT_METHOD_WINDOW:
            return Hit("keyboard")
        if window.window_type != APPLICATION_WINDOW:
            return Hit("system")
        tapped = _tapped_node(tree, index, x, y)
        if tapped is None:
            return Hit("nothing")
        element = _element_node(tree, tapped)
        return Hit("element", Anchor(locator(tree, element), tree.nodes[element].bounds), element)
    return Hit("nothing" if active is None else "outside_dialog")


def resolve_anchor(
    tree: UiTree, anchor: Anchor, package: str, screen: Bounds
) -> tuple[int, int] | AnchorMiss:
    """Return where a tap on ``anchor`` goes in ``tree``, a screen of ``package`` in the
    pixels of ``screen``: the centre of the current bounds of the one element there that
    the anchor's locator finds, or why there is none.

    The element is a visible, enabled node of the ``acting_windows`` that the locator
    finds. When some of those nodes take taps, only they count: in a Compose screen the
    layout views around a button take their label from its text and share its locator.
    ``no_match`` is no such node and ``ambiguous`` more than one. ``covered`` is a node
    that a tap at its centre, as ``hit_test`` dispatches it, does not reach, because a
    higher window, such as a dialog or the keyboard where it draws, or another element
    lies over it.
    """
    windows = set(acting_windows(tree, package))
    found = [
        index
        for index in find_nodes(tree, anchor.locator)
        if tree.nodes[index].window in windows
        and tree.nodes[index].enabled
        and _area(tree.nodes[index].bounds)
    ]
    candidates = [index for index in found if _tappable(tree, index)] or found
    if not candidates:
        return "no_match"
    if len(candidates) > 1:
        return "ambiguous"
    (index,) = candidates
    bounds = tree.nodes[index].bounds
    centre = ((bounds.left + bounds.right) // 2, (bounds.top + bounds.bottom) // 2)
    if hit_test(tree, centre, screen).node != index:
        return "covered"
    return centre


def part_of(tree: UiTree, index: int, wanted: Locator) -> bool:
    """Return whether node ``index`` of ``tree`` is a node ``wanted`` names, or lies
    inside an element ``wanted`` names.

    Only ancestors that are elements count: a layout container takes its label from
    the first text inside it, so in a screen without resource IDs it can share the
    locator of the element that holds that text, and every node in the container
    would otherwise lie inside that element.
    """
    if wanted.matches(locator(tree, index)):
        return True
    return any(
        _acts(tree, ancestor) and wanted.matches(locator(tree, ancestor))
        for ancestor in _ancestors(tree, index)
    )


def focused_text_field(tree: UiTree) -> Anchor | None:
    """Return the anchor of the text field of an application window that has input
    focus, where typed text goes, or ``None`` when there is none."""
    for index, node in enumerate(tree.nodes):
        window = tree.windows[node.window]
        if node.focused and _is_text_field(node) and window.window_type == APPLICATION_WINDOW:
            return Anchor(locator(tree, index), node.bounds)
    return None


def keyboard_shown(tree: UiTree) -> bool:
    """Return whether the soft keyboard is on the screen: it is a window of its own."""
    return any(window.window_type == INPUT_METHOD_WINDOW for window in tree.windows)


def actionable_elements(tree: UiTree, package: str, screen: Bounds) -> list[TreeElement]:
    """Return the elements of ``tree`` the user can act on, in document order, with
    their regions on ``screen``.

    They are the visible, enabled nodes of the ``acting_windows`` that are clickable,
    long-clickable, checkable or scrollable, the text fields, and the rows of a list
    that handles its rows' taps. A checkable node inside a clickable one, and a
    clickable node with the same label as the clickable one it is in, are part of
    that one. Nodes that share a locator are one element, and the nodes inside one
    scrollable container that share a resource ID and differ only in label are one
    list item.
    """
    windows = set(acting_windows(tree, package))
    groups: dict[tuple[object, ...], list[tuple[int, Locator]]] = {}
    for index, node in enumerate(tree.nodes):
        if node.window not in windows or not _acts(tree, index):
            continue
        found = locator(tree, index)
        container = None if node.scrollable else _scrollable_container(tree, index)
        key: tuple[object, ...] = (
            (container, found.resource_id, found.class_name, found.ancestor_ids)
            if found.resource_id and container is not None
            else (found,)
        )
        groups.setdefault(key, []).append((index, found))
    elements: dict[Locator, TreeElement] = {}
    for members in groups.values():
        index, found = members[0]
        is_list = len({member.label for _, member in members}) > 1
        element_locator = (
            Locator(found.resource_id, found.class_name, None, found.ancestor_ids)
            if is_list
            else found
        )
        node = tree.nodes[index]
        elements.setdefault(
            element_locator,
            TreeElement(
                locator=element_locator,
                label=found.label or "",
                bounds=node.bounds,
                region=region(node.bounds, screen),
                scrollable=node.scrollable,
                text_field=_is_text_field(node),
                rows=len(members) if is_list else 1,
            ),
        )
    return list(elements.values())


def thin_tree_reason(tree: UiTree, package: str, elements: Sequence[TreeElement]) -> str | None:
    """Return why ``tree``, whose ``actionable_elements`` are ``elements``, is too thin to
    list a screen's elements, or ``None`` when it is not.

    It is when it has fewer than ``MIN_ACTIONABLE_ELEMENTS``, or when one web view, or
    one node that draws its content itself, covers most of a window the user can act
    on: such screens expose few of the elements they show.
    """
    if len(elements) < MIN_ACTIONABLE_ELEMENTS:
        return f"{len(elements)} actionable element(s), fewer than {MIN_ACTIONABLE_ELEMENTS}"
    windows = set(acting_windows(tree, package))
    for node in tree.nodes:
        if node.window not in windows:
            continue
        drawn = node.class_name == _WEB_VIEW or (
            not node.children and node.class_name in _CANVAS_CLASSES
        )
        window_area = _area(tree.windows[node.window].bounds)
        if drawn and window_area and _area(node.bounds) > _MOST_OF_WINDOW * window_area:
            short_class = node.class_name.rsplit(".", 1)[-1]
            return f"a {short_class} covers most of the window"
    return None


def region(bounds: Bounds, screen: Bounds) -> str:
    """Return where the centre of ``bounds`` lies on ``screen`` in thirds, as ``top
    left``, ``top``, ``center`` or ``bottom right``; ``""`` for a screen without area."""
    width = screen.right - screen.left
    height = screen.bottom - screen.top
    if width <= 0 or height <= 0:
        return ""
    x = ((bounds.left + bounds.right) / 2 - screen.left) / width
    y = ((bounds.top + bounds.bottom) / 2 - screen.top) / height
    vertical = "top" if y < 1 / 3 else "bottom" if y >= 2 / 3 else ""
    horizontal = "left" if x < 1 / 3 else "right" if x >= 2 / 3 else ""
    return " ".join(word for word in (vertical, horizontal) if word) or "center"


def _holds(bounds: Bounds, x: int, y: int) -> bool:
    return bounds.left <= x < bounds.right and bounds.top <= y < bounds.bottom


def _covers(tree: UiTree, window: int, x: int, y: int) -> bool:
    """Return whether window ``window`` of ``tree`` takes a touch at ``x``, ``y``.

    A window takes the touches inside its bounds, except the keyboard's: those bounds
    are its frame, and a floating keyboard draws a small toolbar in a frame that spans
    the lower screen, so a touch beside the toolbar reaches the window below. The
    keyboard takes the touches inside the smallest rectangle around what it draws,
    ``_keyboard_drawn``: its toolbar when it floats, and when it is docked its keys
    with the gaps between them, which it takes too.
    """
    bounds = tree.windows[window].bounds
    if not _holds(bounds, x, y):
        return False
    if tree.windows[window].window_type != INPUT_METHOD_WINDOW:
        return True
    drawn = _keyboard_drawn(tree, window)
    return drawn is not None and _holds(drawn, x, y)


def _keyboard_drawn(tree: UiTree, window: int) -> Bounds | None:
    """Return the smallest rectangle around the visible nodes of keyboard window
    ``window`` that take taps (clickable, long-clickable or focusable) or have a label
    (text or content description), or ``None`` when it has none.

    A container as large as the window is left out: it spans the frame whatever the
    keyboard draws.
    """
    frame = tree.windows[window].bounds
    drawn = [
        node.bounds
        for node in tree.nodes
        if node.window == window
        and _area(node.bounds)
        and (node.clickable or node.long_clickable or node.focusable or _own_label(node))
        and not _contains(node.bounds, frame)
    ]
    if not drawn:
        return None
    return Bounds(
        min(bounds.left for bounds in drawn),
        min(bounds.top for bounds in drawn),
        max(bounds.right for bounds in drawn),
        max(bounds.bottom for bounds in drawn),
    )


def _contains(outer: Bounds, inner: Bounds) -> bool:
    return (
        outer.left <= inner.left
        and outer.top <= inner.top
        and outer.right >= inner.right
        and outer.bottom >= inner.bottom
    )


def _active_application_window(tree: UiTree) -> int | None:
    return next(
        (
            index
            for index, window in enumerate(tree.windows)
            if window.active and window.window_type == APPLICATION_WINDOW
        ),
        None,
    )


def _tapped_node(tree: UiTree, window: int, x: int, y: int) -> int | None:
    """Return the node of ``window`` a touch at ``x``, ``y`` is dispatched to.

    A view group offers a touch to its children under the point, the one drawn last
    first, and keeps it when none of them takes it; so the first node that takes taps
    in a post-order walk of the nodes under the point, children drawn last first, is
    the one the touch reaches. The walk keeps its own stack, as the parser does.
    """
    roots = [
        i for i, node in enumerate(tree.nodes) if node.window == window and node.parent is None
    ]
    deepest: int | None = None
    stack = [(root, False) for root in reversed(_under_point(tree, roots, x, y))]
    while stack:
        index, visited = stack.pop()
        if not visited:
            stack.append((index, True))
            children = _under_point(tree, tree.nodes[index].children, x, y)
            stack.extend((child, False) for child in reversed(children))
            continue
        if deepest is None:
            deepest = index
        if _tappable(tree, index):
            return index
    return deepest


def _under_point(tree: UiTree, indices: Sequence[int], x: int, y: int) -> list[int]:
    """Return the nodes of ``indices`` whose bounds hold the point, drawn last first; of
    two with the same drawing order, the later one in the dump is drawn later."""
    held = [index for index in indices if _holds(tree.nodes[index].bounds, x, y)]
    return sorted(held, key=lambda index: (tree.nodes[index].drawing_order, index), reverse=True)


def _element_node(tree: UiTree, index: int) -> int:
    """Return the node ``actionable_elements`` lists node ``index`` as part of: the
    clickable node it is in while the two have the same label, else itself."""
    while _tappable(tree, index):
        outer = next(
            (
                ancestor
                for ancestor in _ancestors(tree, index)
                if _tappable(tree, ancestor) and tree.nodes[ancestor].enabled
            ),
            None,
        )
        if outer is None or _label(tree, outer) != _label(tree, index):
            break
        index = outer
    return index


def _area(bounds: Bounds) -> int:
    return max(0, bounds.right - bounds.left) * max(0, bounds.bottom - bounds.top)


def _is_text_field(node: UiNode) -> bool:
    return node.class_name.endswith(_TEXT_FIELD_SUFFIXES)


def _label(tree: UiTree, index: int) -> str:
    node = tree.nodes[index]
    if _is_text_field(node):
        return _words(node.hint) or _words(node.content_description)
    own = _own_label(node)
    if own or node.scrollable:
        return own
    stack = list(reversed(node.children))
    while stack:
        descendant = tree.nodes[stack.pop()]
        if text := _words(descendant.text):
            return text
        stack.extend(reversed(descendant.children))
    return ""


def _own_label(node: UiNode) -> str:
    return _words(node.text) or _words(node.content_description)


def _words(text: str) -> str:
    return " ".join(text.split())


def _tappable(tree: UiTree, index: int) -> bool:
    node = tree.nodes[index]
    in_adapter_list = node.parent is not None and (
        tree.nodes[node.parent].class_name in _ADAPTER_LISTS
    )
    return node.clickable or node.long_clickable or in_adapter_list


def _acts(tree: UiTree, index: int) -> bool:
    node = tree.nodes[index]
    if not node.enabled or not _area(node.bounds):
        return False
    if node.scrollable or _is_text_field(node):
        return True
    tappable = _tappable(tree, index)
    if not tappable and not node.checkable:
        return False
    outer = next(
        (
            ancestor
            for ancestor in _ancestors(tree, index)
            if _tappable(tree, ancestor) and tree.nodes[ancestor].enabled
        ),
        None,
    )
    if outer is None:
        return True
    return tappable and _label(tree, outer) != _label(tree, index)


def _scrollable_container(tree: UiTree, index: int) -> int | None:
    return next(
        (ancestor for ancestor in _ancestors(tree, index) if tree.nodes[ancestor].scrollable),
        None,
    )


def _ancestors(tree: UiTree, index: int) -> Iterator[int]:
    """Yield the indices of the ancestors of node ``index``, innermost first."""
    parent = _parent(tree, index)
    while parent is not None:
        yield parent
        parent = _parent(tree, parent)


def _parent(tree: UiTree, index: int) -> int | None:
    return tree.nodes[index].parent


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
