"""The inspection boundary for unattended audit suggestions and recorded instructions."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

from android_app_graph.payloads import as_object_dict, as_str

INSPECTION_HINT = (
    "Inspection only: open pages, editable forms or choosers, read settings, and inspect "
    "confirmation dialogs followed by Cancel. Keep calls, actual sends, destructive "
    "confirmations, purchases, account enrollment and security changes unexecuted. "
    "Choose another inspection action or wait when an effect is uncertain."
)


@dataclass(frozen=True, slots=True)
class ExecutionDecision:
    """A local decision, distinct from the auditor's assessment of the operation."""

    executable: bool
    reason: str

    def as_record(self) -> dict[str, object]:
        return {"policy": "inspection", "executable": self.executable, "reason": self.reason}


def inspection_block_reason(instruction: str, *, context: str = "") -> str | None:
    """Recognize explicit commitments even when a model calls them inspection.

    This is a conservative language check, not proof of an action's real effect.
    Opening a form, chooser or confirmation and cancelling remain inspection.
    """
    text = " ".join(instruction.lower().split())
    clauses = re.split(r"[.;\n]|\b(?:and then|then|and)\b", text)
    for raw_clause in clauses:
        clause = raw_clause.strip(" ,:-")
        if re.match(r"(?:do not|don't|never|avoid)\b", clause):
            continue
        if re.fullmatch(
            r"(?:(?:tap|press|click)\s+)?(?:clear|remove|delete)\s+(?:the\s+)?"
            r"(?:search (?:filter|query|text|input)|filter|selection|field text)",
            clause,
        ):
            continue
        if re.search(
            r"\b(?:place|make|start|initiate|test)\s+(?:an?\s+|the\s+)?(?:phone\s+)?call\b"
            r"(?!\s+(?:history|settings)\b)|\bdial\s+(?:the\s+)?(?:number|voicemail|\d)"
            r"|\bcall initiation\b|\baction_(?:call|send|sendto)\b"
            r"|\b(?:outgoing call|call-placement|voicemail dialing|cleared call history|"
            r"after (?:deletion|confirmation|purchase|payment))\b",
            clause,
        ):
            return "inspection policy excludes call or transmission commitments"
        if re.search(
            r"^(?:please\s+)?(?:permanently\s+)?(?:call(?!\s+(?:history|settings)\b)|"
            r"send|delete|erase|remove|clear|purchase|buy|pay|enroll|"
            r"register|subscribe|sign in|sign up|sync)\b",
            clause,
        ):
            return "inspection policy excludes consequential operations"
        commits = re.search(
            r"\b(?:confirm|commit|complete|finalize)\b.*\b(?:delet\w*|remov\w*|eras\w*|"
            r"clear\w*|purchas\w*|payment|send\w*|call|enroll\w*|registration|subscription)\b",
            clause,
        )
        consequential_control = re.search(
            r"\b(?:tap|press|click|select|choose|activate)\s+(?:the\s+)?(?:final\s+)?"
            r"[\"']?(?:call(?!\s+(?:history|settings)\b)|"
            r"send|delete|erase|remove|clear|buy|pay|purchase|confirm)\b",
            clause,
        )
        opens_confirmation = re.search(
            r"\bto (?:open|inspect|view)\b.*\bconfirmation (?:dialog|sheet)\b", clause
        )
        if commits or (consequential_control and not opens_confirmation):
            return "inspection policy excludes committing a consequential control"
        if re.search(
            r"\b(?:tap|press|click|choose)\s+(?:the\s+)?[\"']?(?:ok|yes|continue)\b", clause
        ) and (
            re.search(r"\b(?:delet\w*|clear\w*|send\w*|purchas\w*|payment)\b", context.lower())
            and re.search(r"\b(?:dialog|confirmation)\b", context.lower())
        ):
            return "inspection policy excludes a positive consequential confirmation"
        if re.search(
            r"\b(?:enable|disable|change|reset|grant|revoke|turn on|turn off)\b.*"
            r"\b(?:password|pin|passcode|security|permission|admin|account|factory)\b",
            clause,
        ):
            return "inspection policy excludes account or security changes"
    return None


def execution_decision(issue: Mapping[str, object]) -> ExecutionDecision:
    """Require an explicit inspection assessment and check every executable hint."""
    assessment = as_object_dict(issue.get("execution"))
    effect = assessment.get("effect")
    reason = as_str(assessment.get("reason"), "").strip()
    if effect != "inspection" or not reason:
        detail = reason or "an explicit inspection assessment with a reason is required"
        return ExecutionDecision(executable=False, reason=f"inspection policy: {detail}")
    instruction = as_str(issue.get("instruction"), "")
    if issue.get("type") == "retry_edge" and not instruction.strip():
        return ExecutionDecision(
            executable=False, reason="inspection policy: a retry requires its recorded instruction"
        )
    expected = issue.get("expected_pages", [])
    if not isinstance(expected, list):
        return ExecutionDecision(
            executable=False, reason="inspection policy: expected pages must be text"
        )
    texts = [instruction, as_str(issue.get("description"), "")]
    for page in expected:
        if not isinstance(page, str):
            return ExecutionDecision(
                executable=False, reason="inspection policy: expected pages must be text"
            )
        texts.append(page)
    for text in texts:
        blocked = inspection_block_reason(text)
        if blocked is not None:
            return ExecutionDecision(executable=False, reason=blocked)
    return ExecutionDecision(
        executable=True,
        reason="inspection assessment and local commitment checks permit this proposal",
    )
