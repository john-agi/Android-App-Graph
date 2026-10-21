"""The step log: every screenshot exploration and the audit take, and every action
they send, kept in an app's ``steps/`` directory.

Each step has a zero-padded number. Its record is ``<id>.json``; an observation's
image, the screenshot as ``screen_image`` encodes it, is ``<id>.webp`` beside it.
The image is published before its record, so every record names an image that
exists, and an image a crash left without a record is ignored and its number
reused.
"""

from __future__ import annotations

import base64
import io
import secrets
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from PIL import Image

from android_app_graph.graph_files import (
    observation_image_path,
    step_record_path,
    write_bytes_atomically,
    write_json_atomically,
)
from android_app_graph.payloads import as_str
from android_app_graph.screen_image import encode_screenshot

if TYPE_CHECKING:
    from pathlib import Path

    from android_app_graph.device import DeviceController

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
    base64 WebP."""

    observation_id: str
    screenshot_b64: str
    activity: str
    package: str

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
        """Return the wrapped controller's state with its screenshot encoded as WebP
        and the ``observation_id`` it was recorded under."""
        state = self._controller.get_state()
        webp = encode_screenshot(state["screenshot"])
        with Image.open(io.BytesIO(webp)) as image:
            width, height = image.size
        observation = Observation(
            observation_id=self._take_step_id(),
            screenshot_b64=base64.b64encode(webp).decode("ascii"),
            activity=as_str(state.get("activity"), ""),
            package=as_str(state.get("package"), ""),
        )
        write_bytes_atomically(observation_image_path(self.steps, observation.observation_id), webp)
        self._write_record(
            observation.observation_id,
            "observation",
            observation,
            {"width": width, "height": height},
        )
        self._last_observation = observation
        return {
            **state,
            "screenshot": observation.screenshot_b64,
            "observation_id": observation.observation_id,
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
