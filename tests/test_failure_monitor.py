"""Failure collection remains bounded and separate from graph execution."""

import json
import logging
import subprocess
import threading
from collections.abc import Callable
from pathlib import Path
from typing import override

import pytest

from android_app_graph import failure_monitor as monitor
from android_app_graph.device import AdbServerUnreachableError, DeviceUnreachableError
from android_app_graph.failure_monitor import AdbFailureReader, FailureMonitor, SourceRead


class Reader:
    def __init__(self) -> None:
        self.calls = 0

    def capture(self) -> dict[str, SourceRead]:
        self.calls += 1
        return {
            "metadata": SourceRead(
                b"boot=12345678-1234-1234-1234-123456789abc\nepoch=100\nuser=0\npackage:com.example uid:1001\npackage:com.example.other uid:1002\n",
                "ok",
            ),
            "crash": SourceRead(b"", "ok"),
            "exits": SourceRead(b"", "ok"),
        }


def collector(tmp_path: Path, reader: Reader | None = None) -> FailureMonitor:
    return FailureMonitor(
        reader or Reader(),
        tmp_path / "app.json",
        "com.example",
        "serial",
        logging.getLogger("test"),
    )


def test_context_retains_baseline_final_and_an_unknown_census(tmp_path: Path) -> None:
    reader = Reader()
    with collector(tmp_path, reader) as current:
        assert reader.calls == 1
    report = json.loads((current.directory / "report.json").read_text())
    assert report["outcome"] == "completed"
    assert report["captures"] == reader.calls == 2
    assert report["run_failure_census"] == "unknown"
    assert report["latest_snapshot_target_process_name_counts"] == {
        "java_fatal": 0,
        "native_signal": 0,
        "recorded_exit": 0,
    }
    first = json.loads((current.directory / "0000.json").read_text())
    final = json.loads((current.directory / "0001.json").read_text())
    assert first["phase"] == "baseline"
    assert first["boot_id"] == "12345678-1234-1234-1234-123456789abc"
    assert first["coverage"]["relation"] == "baseline"
    assert first["current_owner"] == {"android_user": 0, "package_uid": 1001}
    assert final["phase"] == "final"
    assert final["coverage"]["relation"] == "unanchored"
    assert first["sources"]["crash"]["bytes"] == 0
    assert (current.directory / "0000-crash.txt").read_bytes() == b""


@pytest.mark.parametrize(
    ("error", "outcome"),
    [
        (KeyboardInterrupt(), "operator_stop"),
        (SystemExit(1), "operator_stop"),
        (ValueError("explorer"), "host_failure"),
        (DeviceUnreachableError("serial"), "device_outage"),
        (AdbServerUnreachableError("server"), "device_outage"),
    ],
)
def test_terminal_outcomes_preserve_the_original_exception(
    tmp_path: Path, error: BaseException, outcome: str
) -> None:
    current = collector(tmp_path)
    with pytest.raises(type(error)), current:
        raise error
    report = json.loads((current.directory / "report.json").read_text())
    assert report["outcome"] == outcome
    assert report["host_exception"] == type(error).__name__


def test_resume_has_a_fresh_baseline_and_preserves_previous_artifacts(tmp_path: Path) -> None:
    with collector(tmp_path) as first:
        pass
    before = (first.directory / "report.json").read_bytes()
    with collector(tmp_path) as second:
        pass
    assert first.directory != second.directory
    assert (first.directory / "report.json").read_bytes() == before
    assert json.loads((second.directory / "0000.json").read_text())["phase"] == "baseline"


def test_collection_error_is_logged_without_replacing_exploration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    reader = Reader()

    def failed() -> dict[str, SourceRead]:
        msg = "reader failed"
        raise OSError(msg)

    monkeypatch.setattr(reader, "capture", failed)
    with collector(tmp_path, reader):
        pass
    assert "Failure evidence collection is incomplete" in caplog.text
    assert "Cannot persist the failure collection report" in caplog.text


def test_capture_limit_reserves_final_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(monitor, "CAPTURE_LIMIT", 3)
    monkeypatch.setattr(monitor, "CAPTURE_INTERVAL", 0.001)
    polled = threading.Event()

    class PollReader(Reader):
        @override
        def capture(self) -> dict[str, SourceRead]:
            result = super().capture()
            if self.calls == 2:
                polled.set()
            return result

    with collector(tmp_path, PollReader()) as current:
        assert polled.wait(2)
    current.capture("periodic")
    report = json.loads((current.directory / "report.json").read_text())
    assert report["captures"] == 3
    assert report["collection_limit_reached"] is True
    assert json.loads((current.directory / "0001.json").read_text())["phase"] == "periodic"
    assert json.loads((current.directory / "0002.json").read_text())["phase"] == "final"


def test_failed_poll_stops_collection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(monitor, "CAPTURE_INTERVAL", 0.001)
    failed = threading.Event()

    class PollReader(Reader):
        @override
        def capture(self) -> dict[str, SourceRead]:
            if self.calls:
                failed.set()
                msg = "periodic read failed"
                raise OSError(msg)
            return super().capture()

    with collector(tmp_path, PollReader()) as current:
        assert failed.wait(2)
    report = json.loads((current.directory / "report.json").read_text())
    assert report["captures"] == 1
    assert report["collection_error"] == "OSError"


def test_slow_worker_does_not_block_shutdown_indefinitely(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    current = collector(tmp_path)
    joins = []

    class Worker:
        def join(self, seconds: float) -> None:
            joins.append(seconds)

        def is_alive(self) -> bool:
            return True

    monkeypatch.setattr(current, "_thread", Worker())
    current.__exit__(None, None, None)
    assert joins == [8.0]
    assert "shutdown is incomplete" in caplog.text


def test_unavailable_metadata_leaves_identity_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reader = Reader()
    sources = reader.capture()
    sources["metadata"] = SourceRead(b"denied", "unavailable")
    sources["crash"] = SourceRead(b"partial", "truncated")
    monkeypatch.setattr(reader, "capture", lambda: sources)
    current = collector(tmp_path, reader)
    current.capture("baseline")
    capture = json.loads((current.directory / "0000.json").read_text())
    assert capture["boot_id"] is None
    assert capture["device_epoch_seconds"] is None
    assert capture["current_owner"] == {"android_user": None, "package_uid": None}
    assert capture["coverage"]["relation"] == "unavailable"
    assert capture["sources"]["crash"]["status"] == "truncated"


def test_reader_quotes_only_validated_package_and_bounds_each_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []
    timeouts: list[float] = []
    monkeypatch.setattr(monitor, "adb_executable", lambda: "/adb")
    monkeypatch.setattr(monitor.secrets, "token_hex", lambda _n: "token")

    def run(
        argv: list[str], *, timeout: float, **_kwargs: object
    ) -> subprocess.CompletedProcess[bytes]:
        commands.append(argv)
        timeouts.append(timeout)
        return subprocess.CompletedProcess(argv, 0, b"sample\nAPP_GRAPH_END_token:0\n")

    monkeypatch.setattr(monitor.subprocess, "run", run)
    sources = AdbFailureReader("serial", "com.example").capture()
    assert all(source.data == b"sample" and source.status == "ok" for source in sources.values())
    assert [command[:6] for command in commands] == [
        ["/adb", "-s", "serial", "exec-out", "sh", "-c"]
    ] * 3
    assert commands[0][-1].startswith("{ printf 'boot='")
    assert "-t 512" in commands[1][-1]
    assert [command[-1].rsplit("head -c ", 1)[1] for command in commands] == [
        "32769",
        "262145",
        "65537",
    ]
    assert all(0 < value <= 8 for value in timeouts)
    with pytest.raises(ValueError, match="package name"):
        AdbFailureReader("serial", "com.example; touch /data/example")


@pytest.mark.parametrize(
    ("data", "returncode", "limit", "status", "detail"),
    [
        (b"123456", 0, 5, "truncated", "source byte limit"),
        (b"denied", 1, 64, "unavailable", "missing command completion marker"),
        (b"denied\nAPP_GRAPH_END_token:1\n", 0, 64, "unavailable", "source command failed"),
        (b"data\nAPP_GRAPH_END_token:0\n", 1, 64, "unavailable", "source command failed"),
    ],
)
def test_reader_retains_partial_and_failed_sources(
    monkeypatch: pytest.MonkeyPatch,
    data: bytes,
    returncode: int,
    limit: int,
    status: str,
    detail: str,
) -> None:
    monkeypatch.setattr(monitor, "adb_executable", lambda: "/adb")
    monkeypatch.setattr(monitor.secrets, "token_hex", lambda _n: "token")
    monkeypatch.setattr(
        monitor.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], returncode, data),
    )
    monkeypatch.setattr(monitor, "_SOURCE_LIMITS", (("crash", limit),))
    result = AdbFailureReader("serial", "com.example").capture()["crash"]
    assert result.status == status
    assert result.detail == detail
    assert len(result.data) <= limit


@pytest.mark.parametrize("error", [OSError("adb"), subprocess.TimeoutExpired("adb", 1)])
def test_reader_reports_command_errors(monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    monkeypatch.setattr(monitor, "adb_executable", lambda: "/adb")

    def failed(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        raise error

    monkeypatch.setattr(monitor.subprocess, "run", failed)
    result = AdbFailureReader("serial", "com.example").capture()
    assert all(source.status == "unavailable" for source in result.values())


def test_exhausted_capture_deadline_performs_no_adb_call(monkeypatch: pytest.MonkeyPatch) -> None:
    times = iter([0.0, 9.0, 10.0, 11.0])
    monkeypatch.setattr(monitor.time, "monotonic", lambda: next(times))
    sources = AdbFailureReader("serial", "com.example").capture()
    assert all(source.detail == "capture deadline" for source in sources.values())


@pytest.mark.parametrize(
    "blocked_stage", ["reader", "0001-metadata.txt", "0001.json", "report.json"]
)
def test_closing_a_stalled_capture_prevents_late_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, blocked_stage: str
) -> None:
    monkeypatch.setattr(monitor, "CAPTURE_INTERVAL", 0.001)
    monkeypatch.setattr(monitor, "CAPTURE_SECONDS", 0.01)
    entered = threading.Event()
    release = threading.Event()
    workers: list[threading.Thread] = []
    original_thread = threading.Thread
    original_write = Path.write_bytes

    def thread(*, target: Callable[[], None], daemon: bool, name: str) -> threading.Thread:
        result = original_thread(target=target, daemon=daemon, name=name)
        workers.append(result)
        return result

    class SlowReader(Reader):
        @override
        def capture(self) -> dict[str, SourceRead]:
            if self.calls == 1 and blocked_stage == "reader":
                entered.set()
                assert release.wait(2)
            return super().capture()

    reader = SlowReader()

    def write(path: Path, data: bytes) -> int:
        result = original_write(path, data)
        if reader.calls == 2 and path.name == f".{blocked_stage}.partial":
            entered.set()
            assert release.wait(2)
        return result

    monkeypatch.setattr(monitor.threading, "Thread", thread)
    monkeypatch.setattr(Path, "write_bytes", write)
    with collector(tmp_path, reader) as current:
        assert entered.wait(2)
    report_before_release = (current.directory / "report.json").read_bytes()
    release.set()
    for worker in workers:
        worker.join(2)
        assert not worker.is_alive()
    assert (current.directory / "report.json").read_bytes() == report_before_release
    assert json.loads(report_before_release)["outcome"] == "running"
    if blocked_stage != "report.json":
        assert not (current.directory / "0001.json").exists()
    calls = reader.calls
    current.capture("periodic")
    assert reader.calls == calls


def test_failed_capture_keeps_counts_of_the_last_published_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reader = Reader()
    original_capture = reader.capture
    original_write = Path.write_bytes

    def capture() -> dict[str, SourceRead]:
        sources = original_capture()
        if reader.calls > 1:
            sources["crash"] = SourceRead(
                b"123.000 1000 42 42 E AndroidRuntime: FATAL EXCEPTION: main\n"
                b"123.000 1000 42 42 E AndroidRuntime: Process: com.example, PID: 42\n",
                "ok",
            )
        return sources

    def write(path: Path, data: bytes) -> int:
        if path.name == ".0001-metadata.txt.partial":
            msg = "controlled persistence failure"
            raise OSError(msg)
        return original_write(path, data)

    monkeypatch.setattr(reader, "capture", capture)
    monkeypatch.setattr(Path, "write_bytes", write)
    with collector(tmp_path, reader) as current:
        pass
    report = json.loads((current.directory / "report.json").read_text())
    assert report["captures"] == 1
    assert report["collection_error"] == "OSError"
    assert report["latest_snapshot_target_process_name_counts"]["java_fatal"] == 0
    assert not (current.directory / "0001.json").exists()


def test_package_identity_query_uses_the_captured_android_user(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(monitor, "adb_executable", lambda: "/adb")
    monkeypatch.setattr(monitor.secrets, "token_hex", lambda _n: "token")

    def run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        commands.append(argv)
        return subprocess.CompletedProcess(argv, 0, b"\nAPP_GRAPH_END_token:0\n")

    monkeypatch.setattr(monitor.subprocess, "run", run)
    AdbFailureReader("serial", "com.example").capture()
    assert "app_graph_user=$(am get-current-user)" in commands[0][-1]
    assert 'list packages -U --user "$app_graph_user" com.example' in commands[0][-1]
