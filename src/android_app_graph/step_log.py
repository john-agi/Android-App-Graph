"""The step log: every screenshot exploration and the audit take, and every action
they send, kept in an app's ``steps/`` directory.

Each step has a zero-padded number. Its record is ``<id>.json``; an observation's
image, the screenshot as ``screen_image`` encodes it, is ``<id>.webp`` beside it.
The image is published before its record, so every record names an image that
exists, and an image a crash left without a record is ignored and its number
reused. ``image_emb.json`` beside them holds the observations' image vectors.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import secrets
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from PIL import Image

from android_app_graph.graph_files import (
    image_embeddings_path,
    observation_image_path,
    step_record_path,
    write_bytes_atomically,
    write_json_atomically,
)
from android_app_graph.payloads import as_float_list, as_int_list, as_object_dict, as_str
from android_app_graph.screen_image import encode_screenshot, pixel_hash, status_bar_rows

if TYPE_CHECKING:
    from pathlib import Path

    from android_app_graph.device import DeviceController

logger = logging.getLogger(__name__)

_FRAME_COORDINATES = 4

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
    base64 WebP and the ``pixel_hash`` of that image below the status bar."""

    observation_id: str
    screenshot_b64: str
    activity: str
    package: str
    pixel_hash: str

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
        the ``observation_id`` it was recorded under and its ``pixel_hash``.

        The hash leaves out the rows of the controller's ``status_bar`` frame; a
        controller that reports none has the whole image hashed.
        """
        state = self._controller.get_state()
        png = base64.b64decode(state["screenshot"], validate=True)
        webp = encode_screenshot(png)
        with Image.open(io.BytesIO(webp)) as image:
            width, height = image.size
        status_bar = _status_bar(state)
        band_rows = status_bar_rows(
            status_bar, screenshot_height=_image_height(png), image_height=height
        )
        observation = Observation(
            observation_id=self._take_step_id(),
            screenshot_b64=base64.b64encode(webp).decode("ascii"),
            activity=as_str(state.get("activity"), ""),
            package=as_str(state.get("package"), ""),
            pixel_hash=pixel_hash(webp, band_rows),
        )
        write_bytes_atomically(observation_image_path(self.steps, observation.observation_id), webp)
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
            },
        )
        self._last_observation = observation
        return {
            **state,
            "screenshot": observation.screenshot_b64,
            "observation_id": observation.observation_id,
            "pixel_hash": observation.pixel_hash,
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
