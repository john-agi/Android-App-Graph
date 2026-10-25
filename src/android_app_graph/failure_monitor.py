"""Read-only Android failure snapshots with bounded collection and explicit gaps."""

import hashlib
import json
import logging
import re
import secrets
import shlex
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Protocol, Self

from android_app_graph.device import (
    AdbServerUnreachableError,
    DeviceUnreachableError,
    adb_executable,
)
from android_app_graph.failure_evidence import SnapshotCursor, crash_evidence, exit_evidence

CAPTURE_SECONDS = 8.0
CAPTURE_INTERVAL = 60.0
CAPTURE_LIMIT = 1024
_SOURCE_LIMITS = (("metadata", 32 * 1024), ("crash", 256 * 1024), ("exits", 64 * 1024))


@dataclass(frozen=True)
class SourceRead:
    data: bytes
    status: str
    detail: str = ""

    @property
    def text(self) -> str:
        return self.data.decode("utf-8", errors="replace")


class FailureReader(Protocol):
    def capture(self) -> dict[str, SourceRead]: ...


class AdbFailureReader:
    def __init__(self, serial: str, package: str) -> None:
        if re.fullmatch(r"[A-Za-z0-9_.]+", package) is None:
            msg = "Failure collection requires an Android package name"
            raise ValueError(msg)
        self.serial = serial
        self.package = package

    def capture(self) -> dict[str, SourceRead]:
        deadline = time.monotonic() + CAPTURE_SECONDS
        package = shlex.quote(self.package)
        metadata = (
            "printf 'boot='; cat /proc/sys/kernel/random/boot_id; "
            "printf 'epoch='; date +%s; printf 'uptime='; cat /proc/uptime; "
            "printf 'user='; am get-current-user; "
            f"cmd package list packages -U {package}; "
            "printf 'process_samples\n'; "
            f"ps -A -o PID,UID,NAME | awk -v p={package} "
            "'$3 == p || index($3, p \":\") == 1 {print}'; "
            f"for pid in $(ps -A -o PID,NAME | awk -v p={package} "
            "'$2 == p || index($2, p \":\") == 1 {print $1}'); do "
            'printf \'proc=%s\n\' "$pid"; cat /proc/"$pid"/stat /proc/"$pid"/status; done; '
            "logcat -b crash -g"
        )
        commands = {
            "metadata": metadata,
            "crash": "logcat -b crash -d -v threadtime,epoch,uid -t 512",
            "exits": f"dumpsys activity exit-info {package}",
        }
        return {name: self._read(commands[name], limit, deadline) for name, limit in _SOURCE_LIMITS}

    def _read(self, command: str, limit: int, deadline: float) -> SourceRead:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return SourceRead(b"", "unavailable", "capture deadline")
        marker = f"APP_GRAPH_END_{secrets.token_hex(8)}"
        script = f"{{ {command}; printf '\\n{marker}:%s\\n' \"$?\"; }} 2>&1 | head -c {limit + 1}"
        try:
            argv = [
                adb_executable(),
                "-s",
                self.serial,
                "exec-out",
                "sh",
                "-c",
                script,
            ]
            result = subprocess.run(  # noqa: S603  # adb argv is fixed and the remote script quotes the validated package
                argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=remaining,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return SourceRead(b"", "unavailable", type(exc).__name__)
        if len(result.stdout) > limit:
            return SourceRead(result.stdout[:limit], "truncated", "source byte limit")
        match = re.search(rb"\n" + marker.encode() + rb":(\d+)\r?\n$", result.stdout)
        if match is None:
            return SourceRead(result.stdout, "unavailable", "missing command completion marker")
        data = result.stdout[: match.start()]
        if result.returncode or match.group(1) != b"0":
            return SourceRead(data, "unavailable", "source command failed")
        return SourceRead(data, "ok")


def _clock_identity(source: SourceRead) -> tuple[str | None, float | None]:
    if source.status != "ok":
        return None, None
    boot = re.search(r"^boot=([a-f0-9-]{36})$", source.text, re.MULTILINE)
    epoch = re.search(r"^epoch=(\d+)$", source.text, re.MULTILINE)
    return str(boot.group(1)) if boot else None, float(epoch.group(1)) if epoch else None


def _outcome(exc: BaseException | None) -> str:
    if exc is None:
        return "completed"
    if isinstance(exc, (KeyboardInterrupt, SystemExit)):
        return "operator_stop"
    if isinstance(exc, (DeviceUnreachableError, AdbServerUnreachableError)):
        return "device_outage"
    return "host_failure"


class FailureMonitor:
    """One monitor session retains independent snapshots, including baseline history."""

    def __init__(
        self,
        reader: FailureReader,
        graph: Path,
        package: str,
        serial: str,
        logger: logging.Logger,
    ) -> None:
        self.reader = reader
        self.package = package
        self.serial = serial
        self.logger = logger
        self.session_id = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{secrets.token_hex(4)}"
        self.directory = graph.parent / "failures" / self.session_id
        self._graph = str(graph)
        self._started = datetime.now(UTC).isoformat()
        self._cursor = SnapshotCursor()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._captures = 0
        self._error: str | None = None
        self._latest_counts: dict[str, int] = {}

    def __enter__(self) -> Self:
        if self._safe_capture("baseline"):
            self._thread = threading.Thread(
                target=self._poll, daemon=True, name="app-failure-evidence"
            )
            self._thread.start()
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(CAPTURE_SECONDS + 1)
            if self._thread.is_alive():
                self.logger.error(
                    "Failure collector shutdown is incomplete; coverage remains unknown"
                )
                return
        if self._error is None:
            self._safe_capture("final")
        try:
            self._report(_outcome(exc), type(exc).__name__ if exc else None)
        except OSError:
            self.logger.exception("Cannot persist the failure collection report")

    def _poll(self) -> None:
        while not self._stop.wait(CAPTURE_INTERVAL):
            if self._captures >= CAPTURE_LIMIT - 1 or not self._safe_capture("periodic"):
                return

    def _safe_capture(self, phase: str) -> bool:
        try:
            self.capture(phase)
        except Exception as exc:  # Adapter boundary: diagnostic failures preserve exploration.
            self._error = type(exc).__name__
            self.logger.exception("Failure evidence collection is incomplete")
            return False
        return True

    def capture(self, phase: str) -> None:
        if self._captures >= CAPTURE_LIMIT:
            return
        started = datetime.now(UTC).isoformat()
        sources = self.reader.capture()
        boot, clock = _clock_identity(sources["metadata"])
        crash = sources["crash"]
        records = tuple(
            line for line in crash.text.splitlines() if not line.startswith("---------")
        )
        coverage = self._cursor.observe(
            records, boot, clock, time.monotonic(), readable=crash.status == "ok"
        )
        events = crash_evidence(crash.text, self.package) + exit_evidence(
            sources["exits"].text, self.package
        )
        self._latest_counts = {
            kind: sum(
                event.kind == kind and event.attribution == "target_process_name"
                for event in events
            )
            for kind in ("java_fatal", "native_signal", "recorded_exit")
        }
        self.directory.mkdir(parents=True, exist_ok=True)
        sequence = self._captures
        raw: dict[str, object] = {}
        for name, source in sources.items():
            filename = f"{sequence:04d}-{name}.txt"
            (self.directory / filename).write_bytes(source.data)
            raw[name] = {
                "file": filename,
                "sha256": hashlib.sha256(source.data).hexdigest(),
                "bytes": len(source.data),
                "status": source.status,
                "detail": source.detail,
            }
        capture = {
            "session_id": self.session_id,
            "phase": phase,
            "sequence": sequence,
            "host_started_utc": started,
            "host_finished_utc": datetime.now(UTC).isoformat(),
            "boot_id": boot,
            "device_epoch_seconds": clock,
            "coverage": asdict(coverage),
            "sources": raw,
            "evidence": [asdict(event) for event in events],
            "target_process_name_counts": self._latest_counts,
            "evidence_scope": "retained_snapshot; occurrences can overlap other captures",
            "process_samples": "current non-atomic metadata; historical instances remain unresolved",
        }
        (self.directory / f"{sequence:04d}.json").write_text(
            json.dumps(capture, indent=2) + "\n", encoding="utf-8"
        )
        self._captures += 1
        self._report("running", None)

    def _report(self, outcome: str, exception: str | None) -> None:
        report = {
            "session_id": self.session_id,
            "package": self.package,
            "serial": self.serial,
            "graph": self._graph,
            "host_started_utc": self._started,
            "host_updated_utc": datetime.now(UTC).isoformat(),
            "outcome": outcome,
            "host_exception": exception,
            "captures": self._captures,
            "collection_error": self._error,
            "collection_limit_reached": self._captures >= CAPTURE_LIMIT,
            "run_failure_census": "unknown",
            "latest_snapshot_target_process_name_counts": self._latest_counts,
            "interpretation": "Java/native records are failure evidence. Only exit-info records describe recorded exits. Empty snapshots do not establish zero crashes.",
            "action_association": "Compare capture UTC windows with graph step records; temporal proximity does not establish cause.",
            "bounds": {
                "interval_seconds": CAPTURE_INTERVAL,
                "capture_seconds": CAPTURE_SECONDS,
                "capture_limit": CAPTURE_LIMIT,
                "source_bytes": dict(_SOURCE_LIMITS),
            },
        }
        path = self.directory / "report.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
