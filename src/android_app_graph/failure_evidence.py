"""Bounded Android failure snapshots and conservative evidence attribution."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FailureEvidence:
    kind: str
    fingerprint: str
    timestamp: str
    process: str | None
    pid: int | None
    source_pid: int | None
    source_uid: str | None
    attribution: str
    reason: str
    process_instance: str = "unresolved_historical"


def crash_evidence(text: str, package: str) -> list[FailureEvidence]:
    raise NotImplementedError


def exit_evidence(text: str, package: str) -> list[FailureEvidence]:
    raise NotImplementedError


@dataclass(frozen=True)
class Coverage:
    relation: str
    new_records: int | None = None


class SnapshotCursor:
    def observe(
        self,
        records: tuple[str, ...],
        boot: str | None,
        device_time: float | None,
        host_time: float,
        *,
        readable: bool,
    ) -> Coverage:
        raise NotImplementedError
