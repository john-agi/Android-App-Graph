"""Inspection decisions over labelled audit advice and malformed assessments."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from android_app_graph.commands.audit_policy import execution_decision, inspection_block_reason
from android_app_graph.payloads import as_object_dict


def _labelled_cases() -> list[dict[str, object]]:
    path = Path(__file__).parent / "fixtures" / "audit_inspection_cases.json"
    return [as_object_dict(case) for case in json.loads(path.read_text())]


@pytest.mark.parametrize("case", _labelled_cases())
def test_labelled_audit_operations_obey_the_inspection_boundary(case: dict[str, object]) -> None:
    decision = execution_decision(as_object_dict(case["issue"]))
    assert decision.executable is case["allowed"], case["id"]
    assert decision.reason
    assert decision.as_record()["policy"] == "inspection"


@pytest.mark.parametrize(
    "assessment",
    [
        None,
        True,
        "inspection",
        {},
        {"effect": "inspection"},
        {"effect": "inspection", "reason": " "},
    ],
)
def test_execution_requires_a_reasoned_inspection_assessment(assessment: object) -> None:
    assert not execution_decision({"execution": assessment}).executable


@pytest.mark.parametrize("pages", ["Send", None, ["Open settings", 1]])
def test_malformed_expected_pages_are_non_executable(pages: object) -> None:
    decision = execution_decision(
        {"execution": {"effect": "inspection", "reason": "Read settings."}, "expected_pages": pages}
    )
    assert not decision.executable
    assert "expected pages must be text" in decision.reason


@pytest.mark.parametrize("instruction", [None, "", " ", 123])
def test_retry_requires_its_recorded_instruction(instruction: object) -> None:
    decision = execution_decision(
        {
            "type": "retry_edge",
            "instruction": instruction,
            "execution": {"effect": "inspection", "reason": "Inspect the form."},
            "execution_policy": {"executable": True},
        }
    )
    assert not decision.executable
    assert "recorded instruction" in decision.reason


@pytest.mark.parametrize(
    ("instruction", "context", "blocked"),
    [
        ("Do not send the message", "", False),
        ("Press OK", "Delete this file confirmation dialog", True),
        ("Tap Yes", "Confirm purchase dialog", True),
        ("Press Cancel", "Delete this file confirmation dialog", False),
        ("Press OK", "Theme chooser dialog", False),
        ("Tap the Confirm button", "", True),
        ("Enable dark theme", "", False),
        ("Open settings then send the message", "", True),
        ("Open settings then disable security permissions", "", True),
    ],
)
def test_planned_instructions_distinguish_inspection_and_commitment(
    instruction: str, context: str, *, blocked: bool
) -> None:
    assert (inspection_block_reason(instruction, context=context) is not None) is blocked


@pytest.mark.parametrize(
    ("instruction", "blocked"),
    [
        ('Tap the green "Call" button at the bottom center of the dialpad.', True),
        ('Tap the blue "Send" icon at the right end of the message field.', True),
        ('Tap the red "Delete" button at the bottom right of the dialog.', True),
        ('Tap "Reset settings" at the bottom of the screen.', True),
        ('Tap "Reset apps" at the bottom right of the "Reset app preferences?" dialog.', True),
        ("Tap the X clear search button at the right end of the query.", False),
        ("Tap the backspace-shaped Delete icon at the right of the numeric keypad.", False),
        ("Tap outside the Remove menu to dismiss it", False),
        ("Tap the overflow icon to the right of the delete icon", False),
        ('Tap the purple "OK" button at the bottom of the "Delete contact?" dialog.', True),
    ],
)
def test_control_descriptions_preserve_the_inspection_boundary(
    instruction: str, *, blocked: bool
) -> None:
    assert (inspection_block_reason(instruction) is not None) is blocked
