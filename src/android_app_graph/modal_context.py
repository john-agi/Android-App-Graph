"""Bounded, recorded evidence for modal opening destinations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from android_app_graph.payloads import as_object_dict

MAX_MODAL_DEPTH = 4
MAX_OPENING_ACTIONS = 8


def _reference(value: object) -> str | None:
    if value is None or (isinstance(value, str) and value):
        return value
    msg = "Expected an observation reference or null"
    raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class MatchInput:
    """Observed predecessor and input, with null input marking a broken capture chain."""

    previous: str | None = None
    actions: tuple[dict[str, object], ...] | None = None

    def as_record(self) -> dict[str, object]:
        return {"previous": self.previous, "actions": self.actions}

    @classmethod
    def from_record(cls, value: object) -> MatchInput:
        record = as_object_dict(value)
        if "previous" not in record or "actions" not in record:
            msg = "Match input requires previous and actions"
            raise ValueError(msg)
        actions = record["actions"]
        if actions is not None and (
            not isinstance(actions, list)
            or len(actions) > MAX_OPENING_ACTIONS
            or any(not isinstance(action, dict) for action in actions)
        ):
            msg = "Match input requires a bounded action list or null"
            raise ValueError(msg)
        return cls(
            _reference(record["previous"]),
            tuple(as_object_dict(action) for action in actions) if actions is not None else None,
        )


@dataclass(frozen=True, slots=True)
class Surface:
    """The described foreground and its relation to the supplied active modal stack."""

    kind: Literal["page", "modal"] = "page"
    relation: Literal["open", "continue", "return", "unknown"] = "unknown"
    return_depth: int = 0

    def as_record(self) -> dict[str, object]:
        return {"kind": self.kind, "relation": self.relation, "return_depth": self.return_depth}

    @classmethod
    def from_record(cls, value: object) -> Surface:
        record = as_object_dict(value)
        kind, relation, depth = (
            record.get("kind"),
            record.get("relation"),
            record.get("return_depth"),
        )
        if kind not in {"page", "modal"} or relation not in {
            "open",
            "continue",
            "return",
            "unknown",
        }:
            msg = "Expected a page/modal surface and modal relation"
            raise ValueError(msg)
        if type(depth) is not int or not 0 <= depth <= MAX_MODAL_DEPTH:
            msg = "Expected a bounded modal return depth"
            raise ValueError(msg)
        if (relation == "return") != (depth > 0):
            msg = "Only a modal return names a positive depth"
            raise ValueError(msg)
        return cls(
            "modal" if kind == "modal" else "page",
            "open"
            if relation == "open"
            else "continue"
            if relation == "continue"
            else "return"
            if relation == "return"
            else "unknown",
            depth,
        )


@dataclass(frozen=True, slots=True)
class ModalFrame:
    """The captured opening evidence; references identify evidence, not screen identity."""

    opened: str
    source: MatchInput

    def as_record(self) -> dict[str, object]:
        return {"opened": self.opened, "source": self.source.as_record()}


@dataclass(frozen=True, slots=True)
class ModalContext:
    """Active modal openings, from outermost to foreground; a page has no frames."""

    frames: tuple[ModalFrame, ...] = ()

    @property
    def origin_known(self) -> bool:
        return bool(self.frames) and all(frame.source.previous is not None for frame in self.frames)

    def as_record(self) -> list[dict[str, object]]:
        return [frame.as_record() for frame in self.frames]

    @classmethod
    def from_record(cls, value: object) -> ModalContext:
        if not isinstance(value, list) or len(value) > MAX_MODAL_DEPTH:
            msg = "Expected a bounded modal context list"
            raise ValueError(msg)
        frames: list[ModalFrame] = []
        for value_frame in value:
            frame = as_object_dict(value_frame)
            opened = _reference(frame.get("opened"))
            if opened is None:
                msg = "A modal frame requires its opening capture"
                raise ValueError(msg)
            frames.append(ModalFrame(opened, MatchInput.from_record(frame.get("source"))))
        return cls(tuple(frames))

    def advance(self, observation: str, source: MatchInput, surface: Surface) -> ModalContext:
        if surface.kind == "page":
            return ModalContext()
        if surface.relation == "continue" and self.frames:
            return self
        if surface.relation == "return" and 0 < surface.return_depth <= len(self.frames):
            return ModalContext(self.frames[: surface.return_depth])
        if (
            surface.relation == "open"
            and source.previous is not None
            and source.actions
            and len(self.frames) < MAX_MODAL_DEPTH
        ):
            return ModalContext((*self.frames, ModalFrame(observation, source)))
        # Missing predecessors and a truncated nesting chain cannot establish an origin.
        return ModalContext((ModalFrame(observation, MatchInput()),))
