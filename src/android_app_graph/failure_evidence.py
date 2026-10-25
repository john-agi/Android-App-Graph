"""Bounded Android failure snapshots and conservative evidence attribution."""

import hashlib
import re
from dataclasses import dataclass

_CLOCK_TOLERANCE = 5


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
    user: int | None = None
    package_uid: int | None = None


@dataclass(frozen=True)
class _LogRecord:
    timestamp: str
    uid: str | None
    pid: int
    tid: int
    tag: str
    message: str
    raw: str


def fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _attribution(process: str | None, package: str) -> str:
    if process is None:
        return "unresolved_process"
    if process == package or process.startswith(f"{package}:"):
        return "target_process_name"
    return "unrelated_process"


def _records(text: str) -> list[_LogRecord]:
    pattern = r"^\s*(\d+\.\d+)\s+(?:(\S+)\s+)?(\d+)\s+(\d+)\s+[VDIWEFAS]\s+([^:]+): ?(.*)$"
    records = []
    for line in text.splitlines():
        match = re.match(pattern, line)
        if match:
            timestamp, uid, pid, tid, tag, message = match.groups()
            records.append(
                _LogRecord(timestamp, uid, int(pid), int(tid), tag.strip(), message, line)
            )
    return records


def _java_process(records: list[_LogRecord], head: _LogRecord) -> tuple[str | None, int | None]:
    for record in records:
        if (record.uid, record.pid, record.tid, record.tag, record.timestamp) != (
            head.uid,
            head.pid,
            head.tid,
            head.tag,
            head.timestamp,
        ):
            continue
        if record.message.startswith("FATAL EXCEPTION:"):
            break
        match = re.fullmatch(r"Process: (\S+), PID: (\d+)", record.message)
        if match and int(match.group(2)) == head.pid:
            return str(match.group(1)), int(match.group(2))
    return None, None


def crash_evidence(text: str, package: str) -> list[FailureEvidence]:
    records = _records(text)
    result = []
    for index, head in enumerate(records):
        process, pid = None, None
        if head.tag == "AndroidRuntime" and head.message.startswith("FATAL EXCEPTION:"):
            kind = "java_fatal"
            process, pid = _java_process(records[index + 1 : index + 65], head)
        elif head.tag == "DEBUG" and re.match(r"signal \d+ \(", head.message):
            kind = "native_signal"
            for record in reversed(records[max(0, index - 64) : index]):
                if (record.uid, record.pid, record.tag) != (head.uid, head.pid, head.tag):
                    continue
                if record.message.startswith("*** ***"):
                    break
                match = re.match(r"pid: (\d+), tid: \d+, name: .* >>> (\S+) <<<$", record.message)
                if match:
                    pid, process = int(match.group(1)), match.group(2)
                    break
        else:
            continue
        result.append(
            FailureEvidence(
                kind,
                fingerprint(head.raw),
                head.timestamp,
                process,
                pid,
                head.pid,
                head.uid,
                _attribution(process, package),
                head.message,
            )
        )
    return result


def exit_evidence(text: str, package: str) -> list[FailureEvidence]:
    result = []
    for block in re.split(r"ApplicationExitInfo [^:\n]*:\n", text)[1:]:
        identity = re.search(
            r"timestamp=(.*?) pid=(\d+) realUid=(\d+) packageUid=(\d+) definingUid=\d+ user=(\d+)",
            block,
        )
        reason = re.search(r"process=(\S+) reason=(.*?) subreason=", block)
        if identity is None or reason is None:
            continue
        timestamp, pid, uid, package_uid, user = identity.groups()
        process = reason.group(1)
        result.append(
            FailureEvidence(
                "recorded_exit",
                fingerprint(identity.group(0) + reason.group(0)),
                timestamp,
                process,
                int(pid),
                None,
                uid,
                _attribution(process, package),
                reason.group(2),
                user=int(user),
                package_uid=int(package_uid),
            )
        )
    return result


@dataclass(frozen=True)
class Coverage:
    relation: str
    new_records: int | None = None


class SnapshotCursor:
    def __init__(self) -> None:
        self._previous: tuple[str, ...] | None = None
        self._boot: str | None = None
        self._clock: float | None = None
        self._host = 0.0

    def observe(
        self,
        records: tuple[str, ...],
        boot: str | None,
        device_time: float | None,
        host_time: float,
        *,
        readable: bool,
    ) -> Coverage:
        result = self._compare(records, boot, device_time, host_time, readable=readable)
        self._previous = records if readable and boot and device_time is not None else None
        self._boot, self._clock, self._host = boot, device_time, host_time
        return result

    def _compare(
        self,
        records: tuple[str, ...],
        boot: str | None,
        clock: float | None,
        host: float,
        *,
        readable: bool,
    ) -> Coverage:
        invalid = self._invalid_source(boot, clock, host, readable=readable)
        if invalid:
            return Coverage(invalid)
        return self._overlap(records)

    def _invalid_source(
        self, boot: str | None, clock: float | None, host: float, *, readable: bool
    ) -> str | None:
        if not readable:
            return "unavailable"
        if not boot:
            return "identity_unavailable"
        if clock is None:
            return "clock_unavailable"
        if self._previous is not None and boot != self._boot:
            return "boot_changed"
        if (
            self._clock is not None
            and abs((clock - self._clock) - (host - self._host)) > _CLOCK_TOLERANCE
        ):
            return "clock_discontinuity"
        return None

    def _overlap(self, records: tuple[str, ...]) -> Coverage:
        if self._previous is None:
            return Coverage("baseline")
        if len(set(records)) != len(records) or len(set(self._previous)) != len(self._previous):
            return Coverage("ambiguous_overlap")
        if not records or not self._previous:
            return Coverage("unanchored")
        if records == self._previous:
            return Coverage("repeated", 0)
        overlaps = [
            size
            for size in range(1, min(len(records), len(self._previous)) + 1)
            if self._previous[-size:] == records[:size]
        ]
        if len(overlaps) == 1:
            return Coverage("overlap", len(records) - overlaps[0])
        return Coverage("missing_overlap")
