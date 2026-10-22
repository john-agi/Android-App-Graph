"""The step log: every screenshot exploration and the audit take, and every action
they send, kept in an app's ``steps/`` directory.

Each step has a zero-padded number. Its record is ``<id>.json``; an observation's
image, the screenshot as ``screen_image`` encodes it, is ``<id>.webp`` beside it,
and its accessibility tree, when the capture has one, is ``<id>.xml``. The image
and the tree are published before the record, so every record names files that
exist, and an image or a tree a crash left without a record is ignored and its
number reused. ``image_emb.json`` beside them holds the observations' image vectors.
"""

from __future__ import annotations

import base64
import io
import json
import secrets
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from PIL import Image

from android_app_graph.graph_files import (
    image_embeddings_path,
    load_embeddings,
    observation_image_path,
    observation_tree_path,
    save_embeddings,
    step_record_path,
    write_bytes_atomically,
    write_json_atomically,
)
from android_app_graph.payloads import as_int, as_int_list, as_object_dict, as_str
from android_app_graph.screen_image import encode_screenshot, pixel_hash, status_bar_rows
from android_app_graph.ui_tree import Locator, UiTree, UnusableDumpError, parse_tree

if TYPE_CHECKING:
    from pathlib import Path

    from android_app_graph.device import DeviceController


_FRAME_COORDINATES = 4
_SIZE_COORDINATES = 2
_NO_TREE = "the device state has no tree"

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
    accessibility tree, or ``None`` with the reason it has none in ``ui_tree_error``.

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
            screen_size=screen_size_of(state),
            ui_tree=tree if isinstance(tree := state.get("ui_tree"), UiTree) else None,
            ui_tree_error=as_str(state.get("ui_tree_error"), "") or None,
        )


@dataclass(frozen=True, kw_only=True, slots=True)
class Transition:
    """One recorded edge: the screens before and after an action, what the planner
    asked for, and the nodes both screens were assigned to at that moment.

    ``instruction`` and ``element`` are the planner's, ``element`` the number it gave the
    element in its list, and ``element_locator`` that element's locator, ``None`` for an
    element the model found on a screen whose tree was too thin; ``action`` is what ran
    on the device. A retried edge has no planner, so its ``instruction`` is the one the
    edge recorded and its ``element`` and ``element_locator`` are ``None``.
    """

    before: Observation
    after: Observation
    instruction: str
    element: int | None
    element_locator: Locator | None
    action: dict[str, Any] | list[dict[str, Any]]
    from_node: str
    to_node: str


@dataclass(frozen=True, slots=True)
class TransitionRecord:
    """A transition as its record in the step log keeps it: the observations before
    and after the action, the action, and the nodes both were assigned to then."""

    before: str
    after: str
    action: object
    from_node: str
    to_node: str


def read_exploration_transitions(steps: Path) -> list[TransitionRecord]:
    """Return the transition records exploration runs wrote to the step log ``steps``,
    in step order.

    An explored graph holds the edges of its exploration runs only: the audit's
    re-exploration records its transitions against the audited copy.
    """
    paths = [path for path in steps.glob("*.json") if path.stem.isascii() and path.stem.isdigit()]
    transitions: list[TransitionRecord] = []
    for path in sorted(paths, key=lambda path: (len(path.stem), path.stem)):
        record = as_object_dict(json.loads(path.read_text(encoding="utf-8")))
        if record.get("kind") == "transition" and record.get("run_kind") == "explore":
            transitions.append(
                TransitionRecord(
                    before=as_str(record.get("before"), ""),
                    after=as_str(record.get("after"), ""),
                    action=record.get("action"),
                    from_node=as_str(record.get("from_node"), ""),
                    to_node=as_str(record.get("to_node"), ""),
                )
            )
    return transitions


def read_observation_record(steps: Path, observation_id: str) -> dict[str, object] | None:
    """Return the record of observation ``observation_id`` in the step log ``steps``, or
    ``None`` when it has none."""
    try:
        text = step_record_path(steps, observation_id).read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    return as_object_dict(json.loads(text))


def read_observation_tree(steps: Path, observation_id: str) -> tuple[UiTree | None, str | None]:
    """Return the tree of an observation in the step log ``steps`` and no reason, or no
    tree and the reason its record gives for having none.

    A tree missing without a reason raises ``FileNotFoundError``: ``StepRecorder.get_state``
    records the tree or the reason for every observation, so a missing tree means the
    step log was changed by hand.
    """
    path = observation_tree_path(steps, observation_id)
    if not path.is_file():
        record = read_observation_record(steps, observation_id) or {}
        if reason := as_str(record.get("ui_tree_error"), ""):
            return None, reason
    return parse_tree(path.read_text(encoding="utf-8")), None


def annotate_observation(steps: Path, observation_id: str, fields: Mapping[str, object]) -> None:
    """Add ``fields`` to the record of observation ``observation_id`` in the step log
    ``steps``, replacing the record atomically.

    A missing record raises ``FileNotFoundError``: ``StepRecorder.get_state`` writes one
    for every observation, so a missing one means the step log was changed by hand.
    """
    path = step_record_path(steps, observation_id)
    record = as_object_dict(json.loads(path.read_text(encoding="utf-8")))
    write_json_atomically(path, {**record, **fields}, ensure_ascii=False)


def load_image_embeddings(steps: Path, *, model: str) -> dict[str, list[float]]:
    """Return the image vectors of the step log ``steps`` by observation ID, when
    ``model`` computed them, by the rules of ``load_embeddings``."""
    return load_embeddings(image_embeddings_path(steps), model=model, kind="image")


def save_image_embeddings(
    steps: Path, embeddings: Mapping[str, list[float]], *, model: str
) -> None:
    """Write ``embeddings``, image vectors by observation ID that ``model`` computed, to
    the step log ``steps``, replacing the file atomically."""
    save_embeddings(image_embeddings_path(steps), embeddings, model=model)


def _status_bar(state: Mapping[str, object]) -> list[int] | None:
    frame = as_int_list(state.get("status_bar"))
    return frame if len(frame) == _FRAME_COORDINATES else None


def _image_size(image: bytes) -> tuple[int, int]:
    with Image.open(io.BytesIO(image)) as opened:
        return opened.size


def screen_size_of(fields: Mapping[str, object]) -> tuple[int, int]:
    """Return the ``screen_size`` of a device state or an observation record, ``(0, 0)``
    when it holds none."""
    size = as_int_list(fields.get("screen_size"))
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
                "screen_size": list(screen_size),
                "ui_tree_error": tree_error,
                "ui_tree_retakes": as_int(state.get("ui_tree_retakes")) or 0,
                "ui_tree_settled": state.get("ui_tree_settled") is True,
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
