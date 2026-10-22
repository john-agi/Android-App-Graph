"""The step log: every screenshot exploration and the audit take, and every action
they send, kept in an app's ``steps/`` directory.

Each step has a zero-padded number. Its record is ``<id>.json``; an observation's
image, the screenshot as ``screen_image`` encodes it, is ``<id>.webp`` beside it,
and its accessibility tree, when the capture has one, is ``<id>.xml``. The image
and the tree are published before the record, so every record names files that
exist, and an image or a tree a crash left without a record is ignored and its
number reused. ``image_emb.json`` beside them holds the observations' image vectors.

A transition also records what its action did, read from the screens around it:
the element each tap or typing acted on, a tap that missed the element the planner
chose, and an action after which nothing visible changed.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import secrets
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from PIL import Image

from android_app_graph.graph_files import (
    image_embeddings_path,
    observation_image_path,
    observation_tree_path,
    step_record_path,
    write_bytes_atomically,
    write_json_atomically,
)
from android_app_graph.payloads import (
    as_float_list,
    as_int,
    as_int_list,
    as_object_dict,
    as_str,
)
from android_app_graph.screen_image import encode_screenshot, pixel_hash, status_bar_rows
from android_app_graph.ui_tree import (
    Anchor,
    Bounds,
    Hit,
    Locator,
    TreeElement,
    UiTree,
    UnusableDumpError,
    actionable_elements,
    focused_text_field,
    hit_test,
    parse_tree,
    part_of,
    same_screen,
    thin_tree_reason,
)

if TYPE_CHECKING:
    from pathlib import Path

    from android_app_graph.device import DeviceController

logger = logging.getLogger(__name__)

_FRAME_COORDINATES = 4
_SIZE_COORDINATES = 2
_NO_TREE = "the device state has no tree"
_ANCHORED = frozenset({"tap", "long_press", "type"})

type RunKind = Literal["explore", "audit"]
type StepSource = Literal[
    "main_loop",
    "navigation_replay",
    "coverage_checkpoint",
    "external_app_recovery",
    "external_web_recovery",
    "re_explore",
    "edge_retry",
]


@dataclass(frozen=True, slots=True)
class Observation:
    """One screen as ``StepRecorder.get_state`` captured it, with its screenshot as
    base64 WebP, the ``pixel_hash`` of that image below the status bar, and its
    accessibility tree, or ``None`` with the reason it has none in ``ui_tree_error``;
    ``ui_tree_settled`` says whether the two dumps around the screenshot agreed.

    ``screen_size`` is the width and height of the screenshot as the device took it,
    in the screen pixels the tree's bounds are in.
    """

    observation_id: str
    screenshot_b64: str
    activity: str
    package: str
    pixel_hash: str
    screen_size: tuple[int, int]
    ui_tree: UiTree | None
    ui_tree_error: str | None
    ui_tree_settled: bool

    @classmethod
    def from_state(cls, state: Mapping[str, object]) -> Observation:
        """Return the observation in ``state``, a state ``StepRecorder.get_state``
        returned; raise ``ValueError`` for a state without an observation ID.
        """
        observation_id = as_str(state.get("observation_id"), "")
        if not observation_id:
            msg = f"Device state has no observation ID; read it through a StepRecorder: {sorted(state)}"
            raise ValueError(msg)
        return cls(
            observation_id=observation_id,
            screenshot_b64=as_str(state.get("screenshot"), ""),
            activity=as_str(state.get("activity"), ""),
            package=as_str(state.get("package"), ""),
            pixel_hash=as_str(state.get("pixel_hash"), ""),
            screen_size=_screen_size(state),
            ui_tree=tree if isinstance(tree := state.get("ui_tree"), UiTree) else None,
            ui_tree_error=as_str(state.get("ui_tree_error"), "") or None,
            ui_tree_settled=state.get("ui_tree_settled") is True,
        )


def tree_elements(observation: Observation) -> tuple[list[TreeElement], str | None]:
    """Return the elements of the observation's tree, and why the tree is too thin to
    list them, or ``None`` when it is not."""
    tree = observation.ui_tree
    if tree is None:
        return [], f"no tree: {observation.ui_tree_error}"
    width, height = observation.screen_size
    elements = actionable_elements(tree, observation.package, Bounds(0, 0, width, height))
    return elements, thin_tree_reason(tree, observation.package, elements)


@dataclass(frozen=True, slots=True)
class ActionHit:
    """What one sub-action acted on, as the tree of the screen it was planned from
    shows it.

    For a tap or long press, ``outcome`` is the ``Hit`` kind of its point; for typing,
    ``element`` or ``no_focused_field``. ``no_tree`` and ``unsettled_tree`` say why the
    screen gave no anchor, and ``None`` is a swipe, a button press, a wait, or a
    sub-action after the first, whose screen no tree shows. ``anchor`` is the element
    when the outcome is ``element``.
    """

    outcome: str | None
    anchor: Anchor | None = None


@dataclass(frozen=True, slots=True)
class MisTap:
    """A tap that did not land on the element the planner chose: ``hit`` is the locator
    of the element it landed on, or the ``Hit`` kind when it landed on none."""

    planned: Locator
    hit: Locator | str

    def as_record(self) -> dict[str, object]:
        hit = self.hit if isinstance(self.hit, str) else self.hit.as_record()
        return {"planned": self.planned.as_record(), "hit": hit}


@dataclass(frozen=True, slots=True)
class ActionEffect:
    """What an executed action did: ``hits``, one per sub-action, the ``mis_tap`` when
    its tap missed the planned element, and whether it had ``no_visible_effect``."""

    hits: tuple[ActionHit, ...]
    mis_tap: MisTap | None
    no_visible_effect: bool

    def anchor_records(self) -> list[dict[str, object] | None]:
        """Return each sub-action's anchor as an edge stores it, ``None`` for one without."""
        return [hit.anchor.as_record() if hit.anchor is not None else None for hit in self.hits]

    def flag_records(self) -> list[dict[str, object] | None]:
        """Return each sub-action's flag as an edge stores it: the mis-tap on the first,
        no visible effect on the last, after which the screen was taken, and ``None``
        for a sub-action with neither."""
        flags: list[dict[str, object] | None] = [None] * len(self.hits)
        if self.mis_tap is not None and flags:
            flags[0] = {"mis_tap": self.mis_tap.as_record()}
        if self.no_visible_effect and flags:
            flags[-1] = {**(flags[-1] or {}), "no_visible_effect": True}
        return flags


def action_effect(
    before: Observation,
    after: Observation,
    actions: Sequence[Mapping[str, object]],
    *,
    planned: Locator | None,
    self_loop: bool,
) -> ActionEffect:
    """Return what ``actions``, planned from ``before`` and leading to ``after``, did.

    The first sub-action's anchor comes from ``before``'s tree, when it settled: for a
    tap or long press, the element under its point; for typing, the focused text
    field. A tap is a mis-tap when the element it hit is neither ``planned``, the
    planner's element, nor inside it, or when it hit no element of the app. A
    ``self_loop`` had no visible effect when the two screens have the same pixel hash,
    or when both trees settled, neither is too thin to list the screen's elements, and
    the trees show the same screen.
    """
    if not actions:
        return ActionEffect((), None, self_loop and _no_visible_effect(before, after))
    first, tapped = _action_hit(before, actions[0])
    hits = (first, *(ActionHit(None) for _ in actions[1:]))
    mis_tap = _mis_tap(before, tapped, planned) if planned is not None else None
    return ActionEffect(hits, mis_tap, self_loop and _no_visible_effect(before, after))


def _action_hit(before: Observation, action: Mapping[str, object]) -> tuple[ActionHit, Hit | None]:
    """Return what ``action`` acted on, and the hit test of a tap's point when it ran."""
    kind = action.get("action")
    if kind not in _ANCHORED:
        return ActionHit(None), None
    tree = before.ui_tree
    if tree is None:
        return ActionHit("no_tree"), None
    if not before.ui_tree_settled:
        return ActionHit("unsettled_tree"), None
    if kind == "type":
        field = focused_text_field(tree)
        found = ActionHit("no_focused_field") if field is None else ActionHit("element", field)
        return found, None
    x, y = as_int(action.get("x")), as_int(action.get("y"))
    if x is None or y is None:
        return ActionHit(None), None
    width, height = before.screen_size
    hit = hit_test(tree, (x, y), Bounds(0, 0, width, height))
    return ActionHit(hit.kind, hit.anchor), hit


def _mis_tap(before: Observation, tapped: Hit | None, planned: Locator) -> MisTap | None:
    if tapped is None or before.ui_tree is None:
        return None
    if tapped.anchor is None or tapped.node is None:
        return MisTap(planned, tapped.kind)
    if part_of(before.ui_tree, tapped.node, planned):
        return None
    return MisTap(planned, tapped.anchor.locator)


def _no_visible_effect(before: Observation, after: Observation) -> bool:
    if before.pixel_hash and before.pixel_hash == after.pixel_hash:
        return True
    first, second = before.ui_tree, after.ui_tree
    if first is None or second is None:
        return False
    if not (before.ui_tree_settled and after.ui_tree_settled):
        return False
    if tree_elements(before)[1] is not None or tree_elements(after)[1] is not None:
        return False
    return same_screen(first, second)


@dataclass(frozen=True, kw_only=True, slots=True)
class Transition:
    """One recorded edge: the screens before and after an action, what the planner
    asked for, and the nodes both screens were assigned to at that moment.

    ``instruction`` and ``element`` are the planner's, ``element`` the number it gave the
    element in its list, and ``element_locator`` that element's locator, ``None`` for an
    element the model found on a screen whose tree was too thin; ``action`` is what ran
    on the device, and ``effect`` what it did. A retried edge has no planner, so its
    ``instruction`` is the one the edge recorded and its ``element`` and
    ``element_locator`` are ``None``.
    """

    before: Observation
    after: Observation
    instruction: str
    element: int | None
    element_locator: Locator | None
    action: dict[str, Any] | list[dict[str, Any]]
    effect: ActionEffect
    from_node: str
    to_node: str


def read_observation_record(steps: Path, observation_id: str) -> dict[str, object] | None:
    """Return the record of observation ``observation_id`` in the step log ``steps``, or
    ``None`` when it has none."""
    try:
        text = step_record_path(steps, observation_id).read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    return as_object_dict(json.loads(text))


def annotate_observation(steps: Path, observation_id: str, fields: Mapping[str, object]) -> None:
    """Add ``fields`` to the record of observation ``observation_id`` in the step log
    ``steps``, replacing the record atomically.

    A missing record raises ``FileNotFoundError``: ``StepRecorder.get_state`` writes one
    for every observation, so a missing one means the step log was changed by hand.
    """
    path = step_record_path(steps, observation_id)
    record = as_object_dict(json.loads(path.read_text(encoding="utf-8")))
    write_json_atomically(path, {**record, **fields}, ensure_ascii=False)


def _damaged_image_embeddings(path: Path, damage: str) -> ValueError:
    return ValueError(
        f"Image embedding file {path} {damage}. Delete it to embed every screenshot again."
    )


def load_image_embeddings(steps: Path, *, model: str) -> dict[str, list[float]]:
    """Return the image vectors of the step log ``steps`` by observation ID, when
    ``model`` computed them.

    The file is tagged with the model that wrote it, since a matching dimension
    cannot prove two vectors came from one model. A step log without the file has no
    vectors yet, and a file written by another model is empty after one warning, so
    every observation is embedded again.

    A damaged file raises ``ValueError`` naming it and the damage, so damage to the
    step log is reported instead of repaired by paying for every vector again: a file
    that cannot be read or parsed, is not tagged, has no ``embeddings`` object, or
    holds a malformed or non-finite vector, which the message names by observation.
    """
    path = image_embeddings_path(steps)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        raise _damaged_image_embeddings(path, f"cannot be read or parsed: {exc}") from exc
    stored = as_object_dict(raw)
    tag = stored.get("model")
    if not isinstance(tag, str):
        raise _damaged_image_embeddings(path, "is not tagged with the model that wrote it")
    if not isinstance(stored.get("embeddings"), dict):
        raise _damaged_image_embeddings(path, "has no embeddings object")
    if tag != model:
        logger.warning(
            "Image embedding file %s was written by model %r, not the configured %r; treating "
            "it as empty, so every observation is embedded again",
            path,
            tag,
            model,
        )
        return {}
    embeddings = {
        observation_id: as_float_list(vector)
        for observation_id, vector in as_object_dict(stored.get("embeddings")).items()
    }
    malformed = [observation_id for observation_id, numbers in embeddings.items() if not numbers]
    if malformed:
        raise _damaged_image_embeddings(
            path,
            f"has a malformed or non-finite vector for observation(s) {', '.join(malformed)}",
        )
    return embeddings


def save_image_embeddings(
    steps: Path, embeddings: Mapping[str, list[float]], *, model: str
) -> None:
    """Write ``embeddings``, image vectors by observation ID that ``model`` computed, to
    the step log ``steps``, replacing the file atomically."""
    write_json_atomically(
        image_embeddings_path(steps), {"model": model, "embeddings": dict(embeddings)}
    )


def _status_bar(state: Mapping[str, object]) -> list[int] | None:
    frame = as_int_list(state.get("status_bar"))
    return frame if len(frame) == _FRAME_COORDINATES else None


def _image_size(image: bytes) -> tuple[int, int]:
    with Image.open(io.BytesIO(image)) as opened:
        return opened.size


def _screen_size(state: Mapping[str, object]) -> tuple[int, int]:
    size = as_int_list(state.get("screen_size"))
    return (size[0], size[1]) if len(size) == _SIZE_COORDINATES else (0, 0)


def _state_tree(state: Mapping[str, object]) -> tuple[str, UiTree]:
    """Return the XML document of the tree in ``state``, a controller's state, and the
    tree; raise ``UnusableDumpError`` with the reason the state has none.
    """
    xml = state.get("xml")
    if not isinstance(xml, str):
        raise UnusableDumpError(as_str(state.get("ui_tree_error"), "") or _NO_TREE)
    return xml, parse_tree(xml)


def _first_free_number(steps: Path) -> int:
    numbers = [
        int(path.stem)
        for path in steps.glob("*.json")
        if path.stem.isascii() and path.stem.isdigit()
    ]
    return max(numbers, default=-1) + 1


class StepRecorder:
    """A ``DeviceController`` that records every screenshot and action of the one it
    wraps in the step log ``steps``.

    One recorder is one run: its records share a ``run_id`` and its ``run_kind``.
    Each record also names its ``source``, the part of the run that took it, which
    callers set with ``with recorder.source(...)``.
    """

    def __init__(
        self,
        controller: DeviceController,
        steps: Path,
        *,
        run_kind: RunKind,
        source: StepSource,
    ) -> None:
        self._controller = controller
        self.steps = steps
        self.w = controller.w
        self.h = controller.h
        self.config = controller.config
        self.run_id = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{secrets.token_hex(4)}"
        self._run_kind = run_kind
        self._source = source
        self._last_observation: Observation | None = None
        steps.mkdir(parents=True, exist_ok=True)
        self._next_number = _first_free_number(steps)

    @contextmanager
    def source(self, source: StepSource) -> Generator[None]:
        """Name ``source`` in every record written inside the block."""
        outer = self._source
        self._source = source
        try:
            yield
        finally:
            self._source = outer

    def get_state(self) -> dict[str, Any]:
        """Return the wrapped controller's state with its screenshot encoded as WebP,
        the ``observation_id`` it was recorded under, its ``pixel_hash``, the
        ``screen_size`` of the screenshot it took, and its ``"xml"`` parsed into
        ``"ui_tree"``, or ``None`` with the reason in ``"ui_tree_error"``.

        The hash leaves out the rows of the controller's ``status_bar`` frame; a
        controller that reports none has the whole image hashed.
        """
        state = self._controller.get_state()
        try:
            document, tree = _state_tree(state)
        except UnusableDumpError as exc:
            document, tree, tree_error = None, None, exc.reason
        else:
            tree_error = None
        png = base64.b64decode(state["screenshot"], validate=True)
        webp = encode_screenshot(png)
        width, height = _image_size(webp)
        screen_size = _image_size(png)
        status_bar = _status_bar(state)
        band_rows = status_bar_rows(
            status_bar, screenshot_height=screen_size[1], image_height=height
        )
        observation = Observation(
            observation_id=self._take_step_id(),
            screenshot_b64=base64.b64encode(webp).decode("ascii"),
            activity=as_str(state.get("activity"), ""),
            package=as_str(state.get("package"), ""),
            pixel_hash=pixel_hash(webp, band_rows),
            screen_size=screen_size,
            ui_tree=tree,
            ui_tree_error=tree_error,
            ui_tree_settled=state.get("ui_tree_settled") is True,
        )
        write_bytes_atomically(observation_image_path(self.steps, observation.observation_id), webp)
        if document is not None:
            write_bytes_atomically(
                observation_tree_path(self.steps, observation.observation_id),
                document.encode("utf-8"),
            )
        self._write_record(
            observation.observation_id,
            "observation",
            observation,
            {
                "width": width,
                "height": height,
                "status_bar": status_bar,
                "status_bar_rows": band_rows,
                "pixel_hash": observation.pixel_hash,
                "ui_tree_error": tree_error,
                "ui_tree_retakes": as_int(state.get("ui_tree_retakes")) or 0,
                "ui_tree_settled": observation.ui_tree_settled,
            },
        )
        self._last_observation = observation
        return {
            **{key: value for key, value in state.items() if key != "xml"},
            "screenshot": observation.screenshot_b64,
            "observation_id": observation.observation_id,
            "pixel_hash": observation.pixel_hash,
            "screen_size": list(screen_size),
            "ui_tree": tree,
            "ui_tree_error": tree_error,
        }

    def exe_action(self, action: dict[str, Any]) -> None:
        """Record ``action`` with the latest observation of this run, then send it.

        The latest observation is the screen the action was taken from, except in a
        replay, which acts without looking; ``source`` says which.
        """
        last = self._last_observation
        self._write_record(
            self._take_step_id(),
            "action",
            last,
            {"observation": last.observation_id if last else None, "action": action},
        )
        self._controller.exe_action(action)

    def record_transition(self, transition: Transition) -> None:
        """Record ``transition``, an edge the graph was given."""
        self._write_record(
            self._take_step_id(),
            "transition",
            transition.before,
            {
                "before": transition.before.observation_id,
                "after": transition.after.observation_id,
                "instruction": transition.instruction,
                "element": transition.element,
                "element_locator": (
                    transition.element_locator.as_record()
                    if transition.element_locator is not None
                    else None
                ),
                "action": transition.action,
                "anchors": transition.effect.anchor_records(),
                "hits": [hit.outcome for hit in transition.effect.hits],
                "action_flags": transition.effect.flag_records(),
                "from_node": transition.from_node,
                "to_node": transition.to_node,
            },
        )

    def _take_step_id(self) -> str:
        step_id = f"{self._next_number:06d}"
        self._next_number += 1
        return step_id

    def _write_record(
        self,
        step_id: str,
        kind: Literal["observation", "action", "transition"],
        screen: Observation | None,
        fields: Mapping[str, object],
    ) -> None:
        record = {
            "id": step_id,
            "kind": kind,
            "run_id": self.run_id,
            "run_kind": self._run_kind,
            "source": self._source,
            "timestamp": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "activity": screen.activity if screen else None,
            "package": screen.package if screen else None,
            **fields,
        }
        write_json_atomically(step_record_path(self.steps, step_id), record, ensure_ascii=False)
