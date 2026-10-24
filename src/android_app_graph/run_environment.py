"""Persisted display ownership, mutation evidence and bounded run cleanup."""

from __future__ import annotations

import base64
import io
import json
import logging
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from types import TracebackType
from typing import Protocol, Self, runtime_checkable

from PIL import Image

from android_app_graph.device import DeviceController
from android_app_graph.display_state import DisplayState
from android_app_graph.graph_files import write_json_atomically
from android_app_graph.payloads import as_object_dict, as_str

_CLEANUP_SECONDS = 15
_COORDINATE_ACTIONS = ("tap", "long_press", "swipe")


class DisplayAccess(Protocol):
    def capture(self, *, deadline: float | None = None) -> DisplayState: ...
    def execute(self, command: tuple[str, ...], *, deadline: float) -> None: ...
    def seed_hashes(self, root: str, *, deadline: float | None = None) -> dict[str, str]: ...


@runtime_checkable
class RunRecording(Protocol):
    def prepare_action(self, action: dict[str, object], *, source: str = "main_loop") -> None: ...
    def record_observation(self, observation_id: str, source: str) -> None: ...


class UnsafeReplayError(ValueError):
    """The current display differs from a coordinate action's recorded display."""


def _native_size(state: Mapping[str, object]) -> list[int]:
    data = base64.b64decode(as_str(state.get("screenshot"), ""))
    with Image.open(io.BytesIO(data)) as image:
        return list(image.size)


def _seed_root(value: object) -> str:
    root = as_str(value, "")
    path = PurePosixPath(root)
    if not (
        root.startswith(("/sdcard/", "/storage/emulated/0/"))
        and ".." not in path.parts
        and "\n" not in root
        and "\x00" not in root
    ):
        msg = "A seed root must be an explicit public-storage subdirectory"
        raise ValueError(msg)
    if path in (PurePosixPath("/sdcard"), PurePosixPath("/storage/emulated/0")):
        msg = "A seed root must name a public-storage subdirectory"
        raise ValueError(msg)
    return str(path)


def mutation_config(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        msg = "experiment.mutation must be an object"
        raise TypeError(msg)
    mutation = as_object_dict(value)
    disposable = mutation.get("disposable_device", False)
    if type(disposable) is not bool:
        msg = "experiment.mutation.disposable_device must be a boolean"
        raise TypeError(msg)
    for key in ("cleanup_recipe", "seed_manifest"):
        item = mutation.get(key)
        if item is not None and (not isinstance(item, str) or not item.strip()):
            msg = f"experiment.mutation.{key} must be nonempty text"
            raise ValueError(msg)
    return mutation


class RunEnvironment:
    """A controller with a durable baseline and display-aware coordinate actions."""

    def __init__(
        self,
        controller: DeviceController,
        access: DisplayAccess,
        graph_path: Path,
        *,
        mutation: Mapping[str, object],
        logger: logging.Logger,
    ) -> None:
        self._controller = controller
        self._access = access
        self._graph = graph_path
        self.path = graph_path.with_name(f"{graph_path.stem}_run.json")
        self.config: dict[str, object] = controller.config
        self._mutation = mutation_config(dict(mutation))
        self._device = as_str(as_object_dict(self.config["device"])["udid"], "")
        self._logger = logger
        self._report: dict[str, object] = {}
        self._last_display: DisplayState | None = None
        self._last_native: list[int] | None = None
        self._observation: str | None = None
        self._counter = 0
        self._pending_change: dict[str, object] | None = None
        self._prepared_action: dict[str, object] | None = None

    @property
    def w(self) -> int:
        return self._controller.w

    @w.setter
    def w(self, value: int) -> None:
        self._controller.w = value

    @property
    def h(self) -> int:
        return self._controller.h

    @h.setter
    def h(self, value: int) -> None:
        self._controller.h = value

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomically(self.path, self._report, ensure_ascii=False)

    def _items(self, name: str) -> list[object]:
        items = self._report[name]
        if not isinstance(items, list):
            msg = f"Run {name} must be a list"
            raise TypeError(msg)
        result = list[object](items)
        self._report[name] = result
        return result

    def __enter__(self) -> Self:
        if self.path.exists():
            self._report = as_object_dict(json.loads(self.path.read_text()))
            if self._report["graph"] != str(self._graph.resolve()):
                msg = "The run baseline belongs to another graph path"
                raise ValueError(msg)
            if self._report["device"] != self._device:
                msg = "The run baseline belongs to another device"
                raise ValueError(msg)
            DisplayState.from_dict(self._report["baseline"])
            for name in ("changes", "cleanup", "unsafe_replays"):
                self._items(name)
        else:
            if self._graph.exists():
                msg = f"The existing graph requires its original run baseline: {self.path}"
                raise ValueError(msg)
            baseline = self._access.capture()
            self._report = {
                "graph": str(self._graph.resolve()),
                "device": self._device,
                "baseline": baseline.to_dict(),
                "captured_at": datetime.now(UTC).isoformat(),
                "contract": {
                    "disposable_device": self._mutation.get("disposable_device", False),
                    "cleanup_recipe": self._mutation.get("cleanup_recipe"),
                },
                "changes": [],
                "cleanup": [],
                "unsafe_replays": [],
            }
            manifest = self._mutation.get("seed_manifest")
            if manifest is not None:
                specification = as_object_dict(json.loads(Path(as_str(manifest, "")).read_text()))
                root = _seed_root(specification["root"])
                self._report["seed"] = {
                    "root": root,
                    "provided": specification["files"],
                    "baseline": self._access.seed_hashes(root),
                }
        self._report["lifecycle"] = "active"
        self._report["pending_restoration"] = [
            list(command)
            for command in DisplayState.from_dict(self._report["baseline"]).restoration_commands()
        ]
        self._save()
        return self

    def get_state(self, *, deadline: float | None = None) -> dict[str, object]:
        state = as_object_dict(self._controller.get_state(deadline=deadline))
        display = self._access.capture(deadline=deadline)
        native = _native_size(state)
        self._counter += 1
        self._observation = f"capture-{self._counter}"
        if self._last_display != display or self._last_native != native:
            changes = self._items("changes")
            self._pending_change = {
                "timestamp": datetime.now(UTC).isoformat(),
                "capture": self._observation,
                "display": display.to_dict(),
                "native_size": native,
            }
            changes.append(self._pending_change)
            self._save()
        self._last_display, self._last_native = display, native
        return {**state, "display_state": display.to_dict()}

    def record_observation(self, observation_id: str, source: str) -> None:
        self._observation = observation_id
        if self._pending_change is not None:
            self._pending_change.update(observation=observation_id, source=source)
            self._save()
            self._pending_change = None

    def prepare_action(self, action: dict[str, object], *, source: str = "main_loop") -> None:
        self._prepared_action = None
        if action.get("action") not in _COORDINATE_ACTIONS:
            self._prepared_action = action
            return
        recorded = action.get("recorded_display")
        if source in ("navigation_replay", "edge_retry") and recorded is None:
            self._items("unsafe_replays").append(
                {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "capture": self._observation,
                    "action": action,
                    "reason": "missing recorded display evidence",
                }
            )
            self._save()
            msg = "Coordinate replay requires recorded display evidence"
            raise UnsafeReplayError(msg)
        if recorded is not None:
            current = self._access.capture()
            native = _native_size(self._controller.get_state())
            expected = as_object_dict(recorded)
            if expected != {"settings": current.geometry(), "native_size": native}:
                self._items("unsafe_replays").append(
                    {
                        "timestamp": datetime.now(UTC).isoformat(),
                        "capture": self._observation,
                        "expected": expected,
                        "actual": {"settings": current.geometry(), "native_size": native},
                        "action": action,
                    }
                )
                self._save()
                msg = "Coordinate replay requires its recorded display state or fresh grounding"
                raise UnsafeReplayError(msg)
        else:
            action["recorded_display"] = self._display_stamp()
        self._prepared_action = action

    def _display_stamp(self) -> dict[str, object]:
        if self._last_display is not None and self._last_native is not None:
            return {"settings": self._last_display.geometry(), "native_size": self._last_native}
        state = self.get_state()
        return {
            "settings": DisplayState.from_dict(state["display_state"]).geometry(),
            "native_size": _native_size(state),
        }

    def exe_action(self, action: dict[str, object]) -> None:
        if action is not self._prepared_action:
            self.prepare_action(action)
        self._prepared_action = None
        self._controller.exe_action(
            {key: value for key, value in action.items() if key != "recorded_display"}
        )

    def __exit__(
        self,
        _kind: type[BaseException] | None,
        error: BaseException | None,
        _trace: TracebackType | None,
    ) -> None:
        try:
            self._cleanup(error)
        except Exception:
            self._logger.exception("Run finalization remains pending at %s", self.path)
            if error is None:
                raise

    def _cleanup(self, error: BaseException | None) -> None:
        deadline = time.monotonic() + _CLEANUP_SECONDS
        baseline = DisplayState.from_dict(self._report["baseline"])
        commands = baseline.restoration_commands()
        failures: list[dict[str, object]] = []
        operations: list[dict[str, object]] = []
        for command in commands:
            try:
                self._access.execute(command, deadline=deadline)
                operations.append({"command": list(command), "status": "applied"})
            except Exception as exc:
                self._logger.exception("Display cleanup command failed: %s", command)
                failure: dict[str, object] = {"command": list(command), "error": str(exc)}
                failures.append(failure)
                operations.append({**failure, "status": "failed"})
        final: DisplayState | None = None
        try:
            final = self._access.capture(deadline=deadline)
        except Exception as exc:
            self._logger.exception("Display cleanup verification failed")
            failures.append({"verification_error": str(exc)})
        cleanup: dict[str, object] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "status": "restored" if final == baseline else "pending",
            "original_error": None if error is None else type(error).__name__,
            "scope": "supported_display_settings",
            "app_data_cleanup": "operator_recipe_or_authorized_device_reset",
            "operations": operations,
            "failures": failures,
            "pending_commands": [] if final == baseline else [list(c) for c in commands],
            "actual": None if final is None else final.to_dict(),
        }
        seed = self._report.get("seed")
        if isinstance(seed, dict):
            self._report_seed(as_object_dict(seed), cleanup, deadline)
        self._items("cleanup").append(cleanup)
        self._report["lifecycle"] = cleanup["status"]
        self._report["pending_restoration"] = cleanup["pending_commands"]
        self._save()

    def _report_seed(
        self, seed: dict[str, object], cleanup: dict[str, object], deadline: float
    ) -> None:
        try:
            final = self._access.seed_hashes(as_str(seed["root"], ""), deadline=deadline)
            before = as_object_dict(seed["baseline"])
            cleanup["seed_changes"] = {
                "added": sorted(set(final) - set(before)),
                "removed": sorted(set(before) - set(final)),
                "changed": sorted(
                    key for key in set(before) & set(final) if before[key] != final[key]
                ),
            }
        except Exception as exc:
            self._logger.exception("Seed cleanup report remains pending")
            cleanup["seed_error"] = str(exc)
