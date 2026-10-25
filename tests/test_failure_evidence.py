"""Retained Android records carry evidence and explicit coverage uncertainty."""

import pytest

from android_app_graph.failure_evidence import SnapshotCursor, crash_evidence, exit_evidence


def log(message: str, *, pid: int = 42, tag: str = "AndroidRuntime", time: str = "123.000") -> str:
    return f"{time} 1000 {pid} 43 E {tag}: {message}\n"


def java(process: str = "com.example") -> str:
    return log("FATAL EXCEPTION: main") + log(f"Process: {process}, PID: 42")


def test_java_header_is_evidence_with_unresolved_instance() -> None:
    (event,) = crash_evidence(java(), "com.example")
    assert event.kind == "java_fatal"
    assert event.process == "com.example"
    assert event.pid == event.source_pid == 42
    assert event.source_uid == "1000"
    assert event.attribution == "target_process_name"
    assert event.process_instance == "unresolved_historical"
    assert event.timestamp == "123.000"
    assert event.reason == "FATAL EXCEPTION: main"


@pytest.mark.parametrize(
    ("text", "attribution"),
    [
        (java("com.example:worker"), "target_process_name"),
        (java("com.example.other"), "unrelated_process"),
        (log("FATAL EXCEPTION: main"), "unresolved_process"),
        (
            log("FATAL EXCEPTION: main") + log("Process: com.example, PID: 43", pid=43),
            "unresolved_process",
        ),
    ],
)
def test_process_attribution_requires_matching_record_source(text: str, attribution: str) -> None:
    assert crash_evidence(text, "com.example")[0].attribution == attribution


def test_late_process_line_refines_a_stable_header_fingerprint() -> None:
    first = crash_evidence(log("FATAL EXCEPTION: main"), "com.example")[0]
    complete = crash_evidence(java(), "com.example")[0]
    assert first.fingerprint == complete.fingerprint
    assert first.process is None
    assert complete.process == "com.example"


def test_native_signal_requires_its_own_process_record() -> None:
    text = log("pid: 51, tid: 52, name: worker  >>> com.example:worker <<<", tag="DEBUG")
    text += log("signal 11 (SIGSEGV), code 1 (SEGV_MAPERR), fault addr 0x0", tag="DEBUG")
    (event,) = crash_evidence(text, "com.example")
    assert event.kind == "native_signal"
    assert event.pid == 51
    assert event.source_pid == 42
    assert event.attribution == "target_process_name"


def test_exit_history_is_separate_framework_evidence() -> None:
    text = """ApplicationExitInfo #0:
      timestamp=2026-10-25 02:00:00.001 pid=42 realUid=1000 packageUid=1000 definingUid=0 user=0
      process=com.example reason=4 (CRASH) subreason=0 (UNKNOWN) status=0
      importance=100 pss=0 rss=0 description=crash state=empty trace=null
"""
    (event,) = exit_evidence(text, "com.example")
    assert event.kind == "recorded_exit"
    assert event.reason == "4 (CRASH)"
    assert event.source_uid == "1000"
    assert event.process_instance == "unresolved_historical"
    assert (
        event.fingerprint == exit_evidence(text.replace("#0", "#7"), "com.example")[0].fingerprint
    )


def test_overlaps_use_order_and_preserve_identical_timestamps() -> None:
    cursor = SnapshotCursor()
    assert cursor.observe(("a", "b"), "boot", 1, 1, readable=True).relation == "baseline"
    assert cursor.observe(("a", "b"), "boot", 2, 2, readable=True).relation == "repeated"
    result = cursor.observe(("b", "c", "d"), "boot", 3, 3, readable=True)
    assert result.relation == "overlap"
    assert result.new_records == 2


@pytest.mark.parametrize(
    ("records", "boot", "clock", "readable", "relation"),
    [
        (("z",), "boot", 2, True, "missing_overlap"),
        (("a",), "other", 2, True, "boot_changed"),
        (("a",), "boot", 99, True, "clock_discontinuity"),
        (("a",), "boot", 2, False, "unavailable"),
        (("a",), None, 2, True, "identity_unavailable"),
        (("a",), "boot", None, True, "clock_unavailable"),
    ],
)
def test_discontinuities_leave_coverage_unknown(
    records: tuple[str, ...], boot: str | None, clock: float | None, readable: bool, relation: str
) -> None:
    cursor = SnapshotCursor()
    cursor.observe(("a",), "boot", 1, 1, readable=True)
    result = cursor.observe(records, boot, clock, 2, readable=readable)
    assert result.relation == relation
    assert result.new_records is None


def test_empty_and_identical_records_do_not_prove_a_complete_interval() -> None:
    cursor = SnapshotCursor()
    cursor.observe((), "boot", 1, 1, readable=True)
    assert cursor.observe(("a",), "boot", 2, 2, readable=True).relation == "unanchored"
    cursor.observe(("a", "a"), "boot", 3, 3, readable=True)
    assert cursor.observe(("a", "a"), "boot", 4, 4, readable=True).relation == "ambiguous_overlap"
