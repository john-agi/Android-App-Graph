"""The step log: every screenshot exploration and the audit take, every action they
send, every screenshot they ask a node for and every edge they record, kept in an app's
``steps/`` directory.

Each step has a zero-padded number. Its record is ``<id>.json``; an observation's
image, the screenshot as ``screen_image`` encodes it, is ``<id>.webp`` beside it.
The image is published before its record, so every record names an image that
exists, and an image a crash left without a record is ignored and its number
reused. Once written, a record never changes. ``image_emb.json`` beside them holds
the observations' image vectors.
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
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

from PIL import Image

from android_app_graph.device import InputRejectedError
from android_app_graph.graph_files import (
    image_embeddings_path,
    load_embeddings,
    observation_image_b64,
    observation_image_path,
    save_embeddings,
    step_record_path,
    write_bytes_atomically,
    write_json_atomically,
)
from android_app_graph.modal_context import MAX_OPENING_ACTIONS, MatchInput
from android_app_graph.payloads import as_int_list, as_object_dict, as_str
from android_app_graph.run_environment import RunRecording, UnsafeReplayError
from android_app_graph.screen_image import (
    encode_screenshot,
    is_black,
    mask_regions,
    pixel_hash,
    status_bar_rows,
)
from android_app_graph.system_surfaces import (
    Region,
    SystemOverlayActionError,
    UnresolvedSystemOverlayError,
    check_overlay_action,
    overlay_regions,
)

if TYPE_CHECKING:
    from pathlib import Path

    from android_app_graph.device import DeviceController


_FRAME_COORDINATES = 4
_SIZE_COORDINATES = 2
_OWNERSHIP_FIELDS = frozenset(
    {
        "native_size",
        "system_surface",
        "system_overlays",
        "underlying_package",
        "permission_requester",
        "system_overlay_error",
    }
)
_SYSTEM_SURFACES = frozenset({"notification_shade", "permission", "ime_settings"})

type RunKind = Literal["explore", "audit"]
type StepSource = Literal[
    "main_loop",
    "navigation_replay",
    "coverage_checkpoint",
    "external_app_recovery",
    "external_web_recovery",
    "device_recovery",
    "system_surface_recovery",
    "re_explore",
    "edge_retry",
]


@dataclass(frozen=True, slots=True)
class Observation:
    """One screen as ``StepRecorder.get_state`` captured it, with its screenshot as
    base64 WebP, the ``pixel_hash`` of that image below the status bar, and whether the
    image is ``black``, as a secure window captures."""

    observation_id: str
    screenshot_b64: str
    activity: str
    package: str
    pixel_hash: str
    black: bool
    system_surface: str | None = None
    system_overlays: tuple[Region, ...] = ()
    native_size: tuple[int, int] | None = None

    @property
    def app_image(self) -> str:
        """The app view for model input, with observed system feedback unavailable."""
        return (
            mask_regions(self.screenshot_b64, self.system_overlays, self.native_size)
            if self.native_size is not None
            else self.screenshot_b64
        )

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
            black=state.get("black") is True,
            system_surface=as_str(state.get("system_surface"), "") or None,
            system_overlays=overlay_regions(state.get("system_overlays")),
            native_size=_native_size(state.get("native_size")),
        )


@dataclass(frozen=True, kw_only=True, slots=True)
class Transition:
    """One recorded edge: the screens before and after an action, what the planner
    asked for, and the nodes both screens were assigned to at that moment.

    ``instruction`` and ``element`` are the planner's; ``action`` is what ran on the
    device. A retried edge has no planner, so its ``instruction`` is the one the edge
    recorded and its ``element`` is ``None``.
    """

    before: Observation
    after: Observation
    instruction: str
    element: int | None
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


def step_order(step_id: str) -> tuple[int, str]:
    """Sort key of a step ID: zero-padded integers, so a longer ID is a later step."""
    return len(step_id), step_id


def read_records(steps: Path) -> list[dict[str, object]]:
    """Return every record of the step log ``steps``, in step order."""
    paths = [path for path in steps.glob("*.json") if path.stem.isascii() and path.stem.isdigit()]
    return [
        as_object_dict(json.loads(path.read_text(encoding="utf-8")))
        for path in sorted(paths, key=lambda path: step_order(path.stem))
    ]


class ObservationOwnership(NamedTuple):
    """Validated ownership and native geometry of a persisted observation."""

    surface: str | None
    overlays: tuple[Region, ...]
    native_size: tuple[int, int] | None
    error: str | None


def observation_ownership(record: Mapping[str, object]) -> ObservationOwnership:
    """Require explicit ownership fields before persisted evidence enters graph matching."""
    observation_id = record.get("id")
    missing = _OWNERSHIP_FIELDS.difference(record)
    if missing:
        msg = f"Observation {observation_id!r} lacks ownership metadata: {sorted(missing)}"
        raise ValueError(msg)
    for name in (
        "system_surface",
        "underlying_package",
        "permission_requester",
        "system_overlay_error",
    ):
        value = record[name]
        if value is not None and (not isinstance(value, str) or not value.strip()):
            msg = f"Observation {observation_id!r} has malformed {name}: expected nonempty text or null"
            raise ValueError(msg)
    surface = as_str(record["system_surface"], "") or None
    if surface is not None and surface not in _SYSTEM_SURFACES:
        msg = f"Observation {observation_id!r} has unsupported system_surface {surface!r}"
        raise ValueError(msg)
    raw_regions = record["system_overlays"]
    regions = overlay_regions(raw_regions)
    if (
        not isinstance(raw_regions, list)
        or len(regions) != len(raw_regions)
        or any(
            not isinstance(region, list) or any(type(value) is not int for value in region)
            for region in raw_regions
        )
    ):
        msg = f"Observation {observation_id!r} has malformed system_overlays: expected native rectangles"
        raise ValueError(msg)
    raw_size = record["native_size"]
    size = _native_size(raw_size)
    if raw_size is not None and (
        size is None
        or not isinstance(raw_size, list)
        or any(type(value) is not int for value in raw_size)
    ):
        msg = f"Observation {observation_id!r} has malformed native_size: expected positive width and height or null"
        raise ValueError(msg)
    if regions and size is None:
        msg = f"Observation {observation_id!r} has system_overlays without native_size"
        raise ValueError(msg)
    return ObservationOwnership(
        surface, regions, size, as_str(record["system_overlay_error"], "") or None
    )


def recorded_observation(steps: Path, record: Mapping[str, object]) -> Observation:
    """Return the observation the record ``record`` of the step log ``steps`` keeps, with
    its image as base64."""
    ownership = observation_ownership(record)
    if ownership.error is not None:
        raise UnresolvedSystemOverlayError(ownership.error)
    observation_id = as_str(record.get("id"), "")
    return Observation(
        observation_id=observation_id,
        screenshot_b64=observation_image_b64(steps, observation_id),
        activity=as_str(record.get("activity"), ""),
        package=as_str(record.get("package"), ""),
        pixel_hash=as_str(record.get("pixel_hash"), ""),
        black=record.get("black") is True,
        system_surface=ownership.surface,
        system_overlays=ownership.overlays,
        native_size=ownership.native_size,
    )


def read_exploration_transitions(steps: Path) -> list[TransitionRecord]:
    """Return the transition records exploration runs wrote to the step log ``steps``,
    in step order.

    An explored graph holds the edges of its exploration runs only: the audit's
    re-exploration records its transitions against the audited copy.
    """
    return [
        TransitionRecord(
            before=as_str(record.get("before"), ""),
            after=as_str(record.get("after"), ""),
            action=record.get("action"),
            from_node=as_str(record.get("from_node"), ""),
            to_node=as_str(record.get("to_node"), ""),
        )
        for record in read_records(steps)
        if record.get("kind") == "transition" and record.get("run_kind") == "explore"
    ]


def read_observation_record(steps: Path, observation_id: str) -> dict[str, object] | None:
    """Return the record of observation ``observation_id`` in the step log ``steps``, or
    ``None`` when it has none."""
    try:
        text = step_record_path(steps, observation_id).read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    return as_object_dict(json.loads(text))


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


def _native_size(value: object) -> tuple[int, int] | None:
    values = as_int_list(value)
    if len(values) == _SIZE_COORDINATES and all(v > 0 for v in values):
        return values[0], values[1]
    return None


def _status_bar(state: Mapping[str, object]) -> list[int] | None:
    frame = as_int_list(state.get("status_bar"))
    return frame if len(frame) == _FRAME_COORDINATES else None


def _image_height(image: bytes) -> int:
    with Image.open(io.BytesIO(image)) as opened:
        return opened.height


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
    callers set with ``with recorder.source(...)``. ``last_match`` is the observation
    of the run's latest match record, ``None`` before the first.
    """

    def __init__(
        self,
        controller: DeviceController,
        steps: Path,
        *,
        run_kind: RunKind,
        source: StepSource,
        target_package: str,
    ) -> None:
        self._controller = controller
        self._target_package = target_package
        self.steps = steps
        self.w = controller.w
        self.h = controller.h
        self.config = controller.config
        self.run_id = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{secrets.token_hex(4)}"
        self._run_kind = run_kind
        self._source = source
        self._last_observation: Observation | None = None
        self._last_input_regions: object = None
        self.input_rejection_hint: str = ""
        self.last_match: str | None = None
        self._match_actions: list[dict[str, object]] | None = []
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

    def get_state(self, *, deadline: float | None = None) -> dict[str, Any]:
        """Return the wrapped controller's state with its screenshot encoded as WebP,
        the ``observation_id`` it was recorded under, its ``pixel_hash`` and whether it
        is ``black``.

        The hash leaves out the rows of the controller's ``status_bar`` frame; a
        controller that reports none has the whole image hashed.
        """
        state = self._controller.get_state(deadline=deadline)
        if state.get("package") == self._target_package and state.get("system_surface") in {
            "notification_shade",
            "permission",
            "ime_settings",
        }:
            state = state.copy()
            state["system_surface"] = None
        png = base64.b64decode(state["screenshot"], validate=True)
        with Image.open(io.BytesIO(png)) as native_image:
            self.w, self.h = native_image.size
        webp = encode_screenshot(png)
        with Image.open(io.BytesIO(webp)) as image:
            width, height = image.size
        status_bar = _status_bar(state)
        band_rows = status_bar_rows(
            status_bar, screenshot_height=_image_height(png), image_height=height
        )
        native_size = (self.w, self.h)
        observation = Observation.from_state(
            {
                **state,
                "observation_id": self._take_step_id(),
                "screenshot": base64.b64encode(webp).decode("ascii"),
                "pixel_hash": pixel_hash(webp, band_rows),
                "black": is_black(webp),
                "native_size": list(native_size),
            }
        )
        write_bytes_atomically(observation_image_path(self.steps, observation.observation_id), webp)
        ownership_fields: dict[str, object] = {
            "native_size": list(native_size),
            "system_surface": state.get("system_surface"),
            "system_overlays": state.get("system_overlays", []),
            "underlying_package": state.get("underlying_package"),
            "permission_requester": state.get("permission_requester"),
            "system_overlay_error": state.get("system_overlay_error"),
        }
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
                "black": observation.black,
                "display_state": state.get("display_state"),
                **ownership_fields,
                **({"input_regions": state["input_regions"]} if "input_regions" in state else {}),
            },
        )
        if isinstance(self._controller, RunRecording):
            self._controller.record_observation(observation.observation_id, self._source)
        last = self._last_observation
        if (
            last is None
            or (last.package, last.activity, last.pixel_hash)
            != (observation.package, observation.activity, observation.pixel_hash)
            or self._last_input_regions != state.get("input_regions")
        ):
            self.input_rejection_hint = ""
        self._last_observation = observation
        ownership = observation_ownership({"id": observation.observation_id, **ownership_fields})
        if ownership.error is not None:
            raise UnresolvedSystemOverlayError(ownership.error)
        self._last_input_regions = state.get("input_regions")
        return {
            **state,
            "screenshot": observation.screenshot_b64,
            "observation_id": observation.observation_id,
            "pixel_hash": observation.pixel_hash,
            "black": observation.black,
            "native_size": list(native_size),
        }

    def exe_action(self, action: dict[str, Any]) -> None:
        """Record the action request and forward it within the observed ownership policy.

        The latest observation identifies the request's screen; ``source`` distinguishes
        replay from planning. A rejected request records the policy decision and its reason.
        """
        last = self._last_observation
        if isinstance(self._controller, RunRecording):
            try:
                self._controller.prepare_action(action, source=self._source)
            except UnsafeReplayError as error:
                self._write_record(
                    self._take_step_id(),
                    "action_rejected",
                    last,
                    {"action": action, "reason": str(error)},
                )
                raise
        action_id = self._take_step_id()
        self._write_record(
            action_id,
            "action",
            last,
            {"observation": last.observation_id if last else None, "action": action},
        )
        preceding_actions = self._match_actions
        self._match_actions = None
        try:
            check_overlay_action(action, last.system_overlays if last else ())
            self._controller.exe_action(action)
        except SystemOverlayActionError as exc:
            self.record_system_event(
                {"status": "action_skipped", "action": action, "reason": str(exc)}
            )
            raise
        except InputRejectedError as error:
            self.input_rejection_hint = (
                f"\nPrevious {action['action']} was rejected: {error}. "
                "Choose another reachable action or reveal the obscured control before trying it."
            )
            self._write_record(
                self._take_step_id(),
                "action_rejected",
                last,
                {
                    "action_record": action_id,
                    "action": action,
                    "reason": str(error),
                    "input_regions": error.regions,
                },
            )
            raise
        self.input_rejection_hint = ""
        self._match_actions = preceding_actions
        if self._match_actions is not None:
            self._match_actions.append(dict(action))
            if len(self._match_actions) > MAX_OPENING_ACTIONS:
                self._match_actions = None

    def record_system_event(self, fields: Mapping[str, object]) -> None:
        """Keep ownership recovery and skipped-input evidence outside app transitions."""
        self._match_actions = None
        last = self._last_observation
        self._write_record(
            self._take_step_id(),
            "system_event",
            last,
            {"observation": last.observation_id if last else None, **fields},
        )

    def record_match(
        self, observation: Observation, expected_observation: str | None
    ) -> MatchInput:
        """Record that the run asks for the node of ``observation``, expecting the node
        ``expected_observation`` belongs to when it names one."""
        context = MatchInput(
            self.last_match,
            tuple(self._match_actions) if self._match_actions is not None else None,
        )
        self._write_record(
            self._take_step_id(),
            "match",
            observation,
            {
                "observation": observation.observation_id,
                "expected_observation": expected_observation,
                "match_input": context.as_record(),
            },
        )
        self.last_match = observation.observation_id
        self._match_actions = []
        return context

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
        kind: Literal[
            "observation", "action", "action_rejected", "match", "transition", "system_event"
        ],
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
