"""Audit an explored graph and optionally re-explore flagged issues on device.

Usage:
    # Audit only (no device needed) — logs issues and saves report
    uv run app-graph-audit -c configs/explore.yaml

    # Audit + re-explore flagged issues on device
    uv run app-graph-audit -c configs/explore.yaml --re-explore

    # Re-explore with custom step budget per issue
    uv run app-graph-audit -c configs/explore.yaml --re-explore --steps-per-issue 10

The auditor finds two types of issues:

1. retry_edge — An edge whose instruction doesn't match its destination.
   Re-explore: replay the flagged action from the source node, move it to the
   edge toward the node it reaches or relabel it, then explore from the source.

2. explore_node — A node that's missing expected outgoing edges.
   Re-explore: navigate to the node and explore with hints about what's missing.
   The planner is told what actions to try.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace
from itertools import pairwise
from pathlib import Path
from typing import Any, NamedTuple

import yaml
from openrouter import OpenRouter

from android_app_graph.commands.audit_policy import (
    INSPECTION_HINT,
    execution_decision,
    inspection_block_reason,
)
from android_app_graph.device import (
    AdbController,
    AppLaunch,
    DeviceController,
    DeviceUnreachableError,
    StartupRecovery,
    UnresolvedPermissionError,
    grant_runtime_permissions,
    is_outside_app,
    keyboard_status,
    launch_app,
    resolve_permission_prompt,
    restart_app,
    return_to_app,
    soft_keyboard_hint,
)
from android_app_graph.graph_files import graph_steps_dir
from android_app_graph.member_groups import (
    IMAGE_GROUPING_FLOOR,
    TEXT_GROUPING_FLOOR,
    MemberCheck,
)
from android_app_graph.step_log import (
    Observation,
    StepRecorder,
    Transition,
    TransitionRecord,
    read_exploration_transitions,
)
from android_app_graph.system_surfaces import SystemOverlayActionError, UnresolvedSystemOverlayError
from android_app_graph.utils import (
    MatchingSettings,
    ModelSettings,
    check_config_keys,
    open_client,
    resolve_api_key,
    resolve_audit_timeout,
    resolve_matching_settings,
    resolve_model_settings,
    resolve_unreachable_limit,
)
from android_app_graph.utils.graph_manager import (
    EXTERNAL_WEB_BACK_PRESSES,
    EXTERNAL_WEB_NODE,
    EdgeDetails,
    GraphManager,
    MemberQuestion,
    MemberVerdict,
    MemberVerdicts,
    MergeProposal,
    PathStep,
    SetupTrail,
    open_menu_in,
)
from android_app_graph.utils.logging import setup_logging
from android_app_graph.utils.vlm_utils import (
    ActionRequest,
    PlanRequest,
    describe_executed_action,
    plan_next_action,
    predict_next_action,
    text_input_status,
    token_tracker,
)

logger = logging.getLogger(__name__)

AUDIT_ROLES = ("page_detail", "embedding", "image_embedding")
RE_EXPLORE_ROLES = (*AUDIT_ROLES, "instruction", "action")


def run_audit(graph: GraphManager, app_name: str, report_path: Path) -> dict[str, Any]:
    """Run the LLM audit and save the report."""
    result = graph.run_audit(app_name=app_name)
    for issue in result.get("issues", []):
        if issue.get("type") in {"retry_edge", "explore_node"}:
            issue["execution_policy"] = execution_decision(issue).as_record()

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info("Audit report saved to %s", report_path)

    return result


def normalize_and_save_audited_graph(graph: GraphManager, audited_graph_path: Path) -> int:
    """Normalize, in one batch, the templates of every edge whose instructions changed since
    its last normalization, then save audited graph.

    This runs after audit merges and optional re-exploration, so templates are
    generated from the final graph topology rather than stale pre-merge edges.
    """
    logger.info(
        "Normalizing audited graph edge templates for %d nodes...",
        graph.graph.number_of_nodes(),
    )
    normalized_count = graph.normalize_all_edges()

    graph.save_graph(audited_graph_path)
    logger.info(
        "Audited graph saved to %s after edge-template normalization (%d templated edge(s))",
        audited_graph_path,
        normalized_count,
    )
    return normalized_count


def select_apps_for_audit(
    config: dict[str, Any],
    graph_dir: Path,
    *,
    app_name: str | None = None,
    re_explore: bool = False,
) -> list[dict[str, Any]]:
    """Return app entries to audit.

    In basic audit mode, ``--app`` selects a graph folder by name and does not
    require the app to be present in config["apps"]. Re-exploration still needs
    package_name/device metadata from the config, so app filtering is disabled
    there.
    """
    if not app_name:
        return list(config.get("apps", []))

    if re_explore:
        msg = "--app can only be used for basic audit without --re-explore"
        raise ValueError(msg)

    graph_path = graph_dir / app_name / f"{app_name}.json"
    if not graph_path.exists():
        msg = f"No graph found at {graph_path}"
        raise FileNotFoundError(msg)

    return [{"name": app_name}]


@dataclass(frozen=True, slots=True)
class MemberReport:
    """Every member check the audit made, with the change each confirmed."""

    checks: list[dict[str, object]]

    @property
    def changes(self) -> int:
        return sum(1 for check in self.checks if "change" in check)

    def as_record(self) -> dict[str, object]:
        return {
            "image_grouping_floor": IMAGE_GROUPING_FLOOR,
            "text_grouping_floor": TEXT_GROUPING_FLOOR,
            "checks": self.checks,
        }


def check_members(
    graph: GraphManager, verdicts: MemberVerdicts, transitions: Sequence[TransitionRecord]
) -> MemberReport:
    """Check every group of the members of every in-app node against its largest group;
    README.md#member-checks.

    Every split check of every node is asked at once and applied before the move
    checks are asked, so a moved member can go to a node a split made. ``verdicts``
    holds the run's verifier answers, which the checks reuse and add to: two merged
    nodes whose members are linked by neither images nor descriptions are two groups
    of the merged node, and their split check is the pair the merge was verified on,
    so the merged node stays whole without a second verifier call. ``transitions`` are
    the exploration's transition records, by which edge actions follow the members
    that move.
    """
    checks = graph.member_checks()
    splits = [(node, check) for node, check in checks if len(check.group) > 1]
    moves = [(node, check) for node, check in checks if len(check.group) == 1]
    results = _check_member_groups(graph, splits, verdicts, transitions)
    results += _check_member_groups(graph, moves, verdicts, transitions)
    return MemberReport(checks=results)


def _check_member_groups(
    graph: GraphManager,
    checks: Sequence[tuple[str, MemberCheck]],
    verdicts: MemberVerdicts,
    transitions: Sequence[TransitionRecord],
) -> list[dict[str, object]]:
    """Ask the verifier about every check's closest pair at once, each member shown with
    its own description, then, in order, split off or move each group it tells apart from
    its node's home. Without an answer the node stays whole: a change rests on an answer,
    and node verification has already asked twice."""
    questions = [MemberQuestion(check.pair.first, check.pair.second) for _, check in checks]
    results: list[dict[str, object]] = []
    answers = graph.verify_member_pairs(questions, verdicts)
    for (node_id, check), verdict in zip(checks, answers, strict=True):
        result = _member_check_record(node_id, check, verdict)
        if verdict.answered and not verdict.same:
            result["change"] = _separate_group(graph, node_id, check, verdicts, transitions)
        results.append(result)
    return results


def _member_check_record(
    node_id: str, check: MemberCheck, verdict: MemberVerdict
) -> dict[str, object]:
    kind = "split" if len(check.group) > 1 else "move"
    home_member, member = check.pair.first, check.pair.second
    if not verdict.answered:
        outcome = "no answer, so the node stays whole"
    else:
        outcome = "same" if verdict.same else "different"
    logger.info(
        "%s check of %s on members %s and %s (%.3f): %s",
        kind.capitalize(),
        node_id,
        home_member,
        member,
        check.pair.score,
        outcome,
    )
    return {
        "kind": kind,
        "node": node_id,
        "group": check.group,
        "members": [home_member, member],
        "image_score": check.pair.score,
        "same": verdict.same if verdict.answered else None,
        "reason": verdict.reason,
        "reused": verdict.reused,
    }


def _separate_group(
    graph: GraphManager,
    node_id: str,
    check: MemberCheck,
    verdicts: MemberVerdicts,
    transitions: Sequence[TransitionRecord],
) -> dict[str, object]:
    """Split ``check``'s group off its node, or move its lone member, and return the
    change."""
    if len(check.group) > 1:
        return graph.split_members(node_id, check.group, transitions)
    return graph.move_member(node_id, check.pair.second, verdicts=verdicts, transitions=transitions)


@dataclass(frozen=True, kw_only=True)
class ReExploreTarget:
    """The app ``re_explore_issues`` works on and the graph file it saves after each step."""

    app_name: str
    package_name: str
    graph_path: Path
    launch_activity: str | None = None


@dataclass(frozen=True, slots=True)
class _ReExploreRun:
    """The graph, app, device session and client that one ``re_explore_issues`` call works
    with, the recorder every device call goes through and the launch every restart reuses."""

    graph: GraphManager
    target: ReExploreTarget
    app_launch: AppLaunch
    session: ReExploreSession
    recorder: StepRecorder
    client: OpenRouter
    start_node: str


@dataclass(frozen=True, slots=True)
class _Step:
    """One planned and executed action.

    ``instruction`` is what the edge records; ``planned_instruction`` is what the planner
    wrote, which differs when an ``end`` is recorded as a back press. ``setup`` is what
    ``SetupTrail.setup`` read for the action, ``None`` when unknown, and ``open_menu`` the
    menu open over the source as the action was sent, ``None`` when none was. ``start``
    is the screenshot the run last matched before the step, which shows the screen the
    step starts from.
    """

    start: str | None
    before: Observation
    instruction: str
    planned_instruction: str
    element: int | None
    action: dict[str, Any]
    setup: list[dict[str, Any]] | None = None
    open_menu: str | None = None


@dataclass(frozen=True, slots=True)
class _SkippedAction:
    instruction: str
    reason: str


class _InspectionBlockedError(ValueError):
    """A recorded navigation instruction crosses the audit's inspection boundary."""


def _check_instruction_policy(run: _ReExploreRun, source: str, instruction: str) -> None:
    blocked = inspection_block_reason(instruction, context=run.graph.latest_description(source))
    if blocked is not None:
        raise _InspectionBlockedError(blocked)


def _check_setup_policy(run: _ReExploreRun, source: str, setup: Sequence[object]) -> None:
    for edge in run.graph.get_all_edges_from_node(source):
        for actions, instruction in zip(edge["actions"], edge["instructions"], strict=True):
            if any(action in actions for action in setup):
                _check_instruction_policy(run, source, instruction)


def _check_navigation_policy(run: _ReExploreRun, path: list[PathStep]) -> None:
    for source, step in pairwise(path):
        _check_setup_policy(run, source.node, step.setup)
        edge = run.graph.graph.get_edge_data(source.node, step.node, {})
        for action, instruction in zip(
            edge.get("actions", []), edge.get("instructions", []), strict=True
        ):
            if action == step.action:
                _check_instruction_policy(run, source.node, instruction)


def _identify(
    run: _ReExploreRun, device_state: dict[str, Any], *, expected_observation: str | None = None
) -> str:
    """Record a match of the screen in ``device_state``, which the caller expects to show
    the screen of ``expected_observation`` when it names one, and return its node."""
    if device_state.get("system_surface") is not None:
        run.graph.record_external_app_visit(Observation.from_state(device_state))
        device_state = _recover_system_surface(run, device_state)
    observation = Observation.from_state(device_state)
    run.recorder.record_match(observation, expected_observation)
    return run.graph.identify_state(observation, expected_observation=expected_observation)


def _recover_system_surface(run: _ReExploreRun, state: dict[str, Any]) -> dict[str, Any]:
    with run.recorder.source("system_surface_recovery"):
        run.recorder.record_system_event(
            {"status": "boundary_recovery", "surface": state.get("system_surface")}
        )
        recovered = (
            resolve_permission_prompt(
                run.recorder,
                run.app_launch,
                state,
                run.recorder.record_system_event,
                StartupRecovery(run.recorder, run.session.unreachable_limit, logger),
            )
            if state.get("system_surface") == "permission"
            else return_to_app(run.recorder, run.app_launch, logger)
        )
        run.recorder.record_system_event(
            {"status": "boundary_resolved", "package": recovered.get("package")}
        )
        return recovered


def _left_app(run: _ReExploreRun, device_state: dict[str, Any]) -> bool:
    """Use the observation's owner to distinguish app state from another destination."""
    return is_outside_app(device_state.get("package", "").strip(), run.target.package_name)


def _execute(controller: DeviceController, action: dict[str, Any] | list[dict[str, Any]]) -> None:
    if isinstance(action, list):
        for sub_action in action:
            controller.exe_action(sub_action)
            time.sleep(0.5)
    else:
        controller.exe_action(action)
    time.sleep(1)


def _navigate_to(run: _ReExploreRun, target_node: str) -> str | None:
    """Navigate the app, just restarted, to target node by replaying the shortest path from
    the node of the screen it opened on, when one leads from it, and otherwise from the
    start node, each action after its setup. Returns actual node reached."""
    start = run.graph.resolve_node_id(run.start_node)
    path = run.graph.find_path(start, target_node)
    if path is None:
        logger.warning("No path from %s to %s", start, target_node)
        return None

    opened_on = _identify(
        run, run.recorder.get_state(), expected_observation=run.graph.earliest_member(start)
    )
    start, target_node = run.graph.resolve_node_id(start), run.graph.resolve_node_id(target_node)
    if opened_on != start:
        logger.info(
            "%s opened on %s, not on the start node %s", run.target.app_name, opened_on, start
        )
        path = run.graph.find_path(opened_on, target_node) or path

    _check_navigation_policy(run, path)

    actual = opened_on
    if len(path) > 1:
        try:
            for step in path[1:]:
                for setup_action in step.setup:
                    _execute(run.recorder, setup_action)
                _execute(run.recorder, step.action)
        except SystemOverlayActionError as exc:
            logger.info("Audit replay stops at system feedback: %s", exc)
        actual = _identify(
            run,
            run.recorder.get_state(),
            expected_observation=run.graph.earliest_member(target_node),
        )
    if actual != target_node:
        logger.warning(
            "Expected to reach %s but arrived at %s",
            target_node,
            actual,
        )
    return actual


def _leave_external_web(run: _ReExploreRun, expected_observation: str | None) -> str:
    """Press back until the screen is no longer an outside web page, restarting the
    app if a few presses do not leave it, and return the screen's node.

    A back press is expected to return to the screen of ``expected_observation``, the
    screenshot of the screen whose step opened the page, and a restart to the start
    node. A step from the shared ``ext_web`` node would act on a page outside the app,
    and ``add_edge`` would refuse its edge.
    """
    with run.recorder.source("external_web_recovery"):
        for _ in range(EXTERNAL_WEB_BACK_PRESSES):
            run.recorder.exe_action({"action": "back"})
            time.sleep(1)
            node = _identify(
                run, run.recorder.get_state(), expected_observation=expected_observation
            )
            if node != EXTERNAL_WEB_NODE:
                return node
        logger.warning(
            "Still on an external web page after %d back presses. Restarting %s.",
            EXTERNAL_WEB_BACK_PRESSES,
            run.target.package_name,
        )
        restart_app(run.recorder, run.app_launch, logger)
        return _identify(
            run,
            run.recorder.get_state(),
            expected_observation=run.graph.earliest_member(run.start_node),
        )


def _plan_and_act(
    run: _ReExploreRun, node: str, device_state: dict[str, Any], page_description: str
) -> _Step | _SkippedAction | None:
    """Return an executed step, a policy skip, or ``None`` for an action-model wait."""
    before = Observation.from_state(device_state)
    explored_edges = run.graph.get_all_edges_from_node(node)

    # Check before planning so the planner can choose type vs tap.
    keyboard_hint = soft_keyboard_hint(run.recorder.config["device"]["udid"])
    input_status = text_input_status(
        keyboard_status(keyboard_hint), run.graph.observation_state(before)
    )

    planned_instruction, element = plan_next_action(
        client=run.client,
        request=PlanRequest(
            screenshot_b64=before.app_image,
            page_description=page_description,
            node=node,
            explored_edges=explored_edges,
            app_name=run.target.app_name,
            input_status=input_status,
            back_leaves_app=run.graph.back_leaves_app(node),
            system_feedback=bool(before.system_overlays),
        ),
        settings=run.session.instruction,
    )
    context = f"{run.graph.latest_description(node)} {run.graph.observation_state(before)}"
    blocked = inspection_block_reason(planned_instruction, context=context)
    if blocked is not None:
        return _SkippedAction(planned_instruction, blocked)

    action, _history_entry = predict_next_action(
        client=run.client,
        request=ActionRequest(
            screenshot_b64=before.app_image,
            instruction=f"{planned_instruction}\n{input_status}",
            screen_w=run.recorder.w,
            screen_h=run.recorder.h,
            action_history=[],
            system_feedback=bool(before.system_overlays),
        ),
        settings=run.session.action,
    )

    instruction = planned_instruction
    if action.get("action") == "end":
        logger.info(
            '  Action agent said end for: "%s", recording as back',
            instruction,
        )
        action = {"action": "back"}
        instruction = "Press back"
    if action.get("action") == "wait":
        time.sleep(1)
        return None
    try:
        run.recorder.exe_action(action)
    except SystemOverlayActionError as exc:
        logger.info("Audit action waits for system feedback: %s", exc)
        return None
    return _Step(
        start=run.recorder.last_match,
        before=before,
        instruction=instruction,
        planned_instruction=planned_instruction,
        element=element,
        action=action,
    )


def _record_step(
    run: _ReExploreRun,
    source: str,
    step: _Step,
    device_state: dict[str, Any],
    step_number: int,
) -> tuple[str, dict[str, Any]]:
    """Record the edge *step* took from *source* to the screen *device_state* shows, with
    that screen's description as its target observation, and save the graph. Returns the
    node the next step starts from, with the step's result.
    """
    instruction, action = step.instruction, step.action
    after = Observation.from_state(device_state)
    new_node = _identify(run, device_state)
    source = run.graph.resolve_node_id(source)
    target_obs = run.graph.latest_description(new_node)

    run.graph.add_edge(
        source,
        new_node,
        action=[action],
        details=EdgeDetails(
            instruction=instruction,
            target_observation=target_obs,
            num_steps=1,
            setup=step.setup,
            open_menu=step.open_menu,
        ),
    )
    run.recorder.record_transition(
        Transition(
            before=step.before,
            after=after,
            instruction=step.planned_instruction,
            element=step.element,
            action=action,
            from_node=source,
            to_node=new_node,
        )
    )

    step_result: dict[str, Any] = {
        "step": step_number,
        "from": source,
        "to": new_node,
        "instruction": instruction,
        "action": action,
    }

    logger.info(
        "  Re-explore step %d: %s --[%s]--> %s (instruction: '%s')",
        step_number,
        source,
        action.get("action", "?"),
        new_node,
        instruction,
    )

    run.graph.save_graph(run.target.graph_path)
    if new_node == EXTERNAL_WEB_NODE:
        logger.warning("Opened an external web page during re-explore, pressing back")
        return _leave_external_web(run, step.start), step_result
    return new_node, step_result


def _record_system_step(
    run: _ReExploreRun, source: str, step: _Step, state: dict[str, Any], number: int
) -> tuple[str, dict[str, Any]]:
    """Keep the executed app input's boundary and its separately recovered app screen."""
    after = Observation.from_state(state)
    boundary = run.graph.record_external_app_visit(after)
    run.graph.add_edge(
        source,
        boundary,
        action=[step.action],
        details=EdgeDetails(
            instruction=step.instruction,
            target_observation=f"[system surface: {after.system_surface}]",
        ),
    )
    run.recorder.record_transition(
        Transition(
            before=step.before,
            after=after,
            instruction=step.planned_instruction,
            element=step.element,
            action=step.action,
            from_node=source,
            to_node=boundary,
        )
    )
    recovered = _identify(run, _recover_system_surface(run, state), expected_observation=step.start)
    run.graph.save_graph(run.target.graph_path)
    return recovered, {
        "step": number,
        "from": source,
        "to": boundary,
        "instruction": step.instruction,
        "action": step.action,
        "status": "system_boundary",
        "system_surface": after.system_surface,
        "recovered_to": recovered,
    }


def _last_state(run: _ReExploreRun, node: str) -> dict[str, Any]:
    """Return the state the latest visit to ``node`` recorded."""
    return run.graph.graph.nodes[node].get("last_detail_snapshot", {})


def _explore_steps(
    run: _ReExploreRun,
    start_from: str,
    num_steps: int,
    hint: str,
    *,
    arrived: bool,
) -> list[dict[str, Any]]:
    """Run exploration steps from a node with a hint for the planner, which plans each
    from the description of the screen the device was last identified on. ``arrived`` says
    the replay before them arrived at the node, so the setup of the first step is known."""
    current_node = start_from
    excluded: dict[str, list[str]] = {}
    step_results: list[dict[str, Any]] = []
    trail = SetupTrail(run.graph.resolve_node_id)
    trail.begin(start_from, _last_state(run, start_from) if arrived else None)

    for step_i in range(num_steps):
        device_state = run.recorder.get_state()
        if run.graph.get_node(current_node) is None:
            break

        page_desc = (
            f"{run.graph.latest_description(current_node)}\n[AUDIT HINT: {hint}]\n{INSPECTION_HINT}"
        )
        if skipped_instructions := excluded.get(current_node):
            page_desc += "\nExcluded by the inspection policy:\n" + "\n".join(skipped_instructions)
        step = _plan_and_act(run, current_node, device_state, page_desc)
        if step is None:
            continue
        if isinstance(step, _SkippedAction):
            instructions = excluded.setdefault(current_node, [])
            if step.instruction not in instructions:
                instructions.append(step.instruction)
            step_results.append(
                {
                    "step": step_i + 1,
                    "status": "skipped",
                    "instruction": step.instruction,
                    "reason": step.reason,
                }
            )
            continue

        time.sleep(0.5)
        device_state = run.recorder.get_state()

        if device_state.get("system_surface") is not None:
            current_node, outcome = _record_system_step(
                run, current_node, step, device_state, step_i + 1
            )
            step_results.append(outcome)
            trail.begin(current_node, None)
            continue
        if _left_app(run, device_state):
            logger.warning("Left app during re-explore, pressing back")
            with run.recorder.source("external_app_recovery"):
                device_state = return_to_app(run.recorder, run.app_launch, logger)
            current_node = _identify(run, device_state, expected_observation=step.start)
            trail.begin(current_node, None)
            continue

        step = replace(
            step,
            setup=trail.setup(current_node),
            open_menu=open_menu_in(_last_state(run, current_node)),
        )
        current_node, step_result = _record_step(run, current_node, step, device_state, step_i + 1)
        step_results.append(step_result)
        if step_result["to"] == EXTERNAL_WEB_NODE:
            trail.begin(current_node, None)
        else:
            trail.step([step.action], current_node, _last_state(run, current_node))

    return step_results


def _replay_to(run: _ReExploreRun, node: str) -> str | None:
    path = run.graph.find_path(run.graph.resolve_node_id(run.start_node), node)
    if path is not None:
        _check_navigation_policy(run, path)
    with run.recorder.source("navigation_replay"):
        restart_app(run.recorder, run.app_launch, logger)
        return _navigate_to(run, node)


def _correct_flagged_edge(
    run: _ReExploreRun, source: str, target: str, instruction: str
) -> dict[str, Any]:
    """Execute the flagged action once from *source*, after its setup, and correct its
    edge by where it leads. Returns the outcome fields of the issue's result.
    """
    found = run.graph.find_edge_action(source, target, instruction)
    if found is None:
        logger.warning("Instruction '%s' is not on %s → %s", instruction, source, target)
        return {"status": "instruction_not_found"}
    index, action = found

    setup = run.graph.action_setup(source, target, index)
    if setup is None:
        logger.warning(
            "No recorded action opens the menu that '%s' on %s → %s is sent in",
            instruction,
            source,
            target,
        )
        return {"status": "unverified", "reached": source}
    _check_setup_policy(run, source, setup)
    for setup_action in setup:
        _execute(run.recorder, setup_action)
    before = run.recorder.get_state()
    _execute(run.recorder, action)
    after = run.recorder.get_state()
    reached = (
        f"ext_{after['package'].strip()}"
        if _left_app(run, after)
        else _identify(run, after, expected_observation=run.graph.earliest_member(target))
    )
    source, target = run.graph.resolve_node_id(source), run.graph.resolve_node_id(target)
    run.recorder.record_transition(
        Transition(
            before=Observation.from_state(before),
            after=Observation.from_state(after),
            instruction=instruction,
            element=None,
            action=action,
            from_node=source,
            to_node=reached,
        )
    )

    if after.get("system_surface") is not None:
        run.graph.record_external_app_visit(Observation.from_state(after))
        _recover_system_surface(run, after)
    if reached != target:
        if not run.graph.move_edge_action(source, target, index, reached):
            return {"status": "unverified", "reached": reached}
        return {"status": "moved", "moved_to": reached}

    label = describe_executed_action(
        client=run.client,
        screenshot_b64=Observation.from_state(before).app_image,
        action=action,
        screen_size=(run.recorder.w, run.recorder.h),
        settings=run.session.instruction,
    )
    if label is None:
        return {"status": "relabel_failed"}
    run.graph.relabel_edge_action(source, target, index, label)
    return {"status": "relabelled", "instruction": label}


def _retry_edge(run: _ReExploreRun, issue: dict[str, Any]) -> dict[str, Any]:
    source = run.graph.resolve_node_id(issue.get("source_node", ""))
    target = run.graph.resolve_node_id(issue.get("target_node", ""))
    instruction = issue.get("instruction", "")

    if source not in run.graph.graph:
        logger.warning("Source node %s not found, skipping", source)
        return {"issue": issue, "status": "skipped", "reason": "source not found"}
    # An action from an ext_ node runs outside the app, and the graph keeps no edge out of one.
    if source.startswith("ext_"):
        logger.warning("Source node %s is external, skipping", source)
        return {"issue": issue, "status": "skipped", "reason": "external source"}

    _check_instruction_policy(run, source, instruction)
    logger.info("Retrying edge: %s → %s (instruction: '%s')", source, target, instruction)
    reached = _replay_to(run, source)
    source, target = run.graph.resolve_node_id(source), run.graph.resolve_node_id(target)
    if reached != source:
        return {"issue": issue, "status": "unverified", "reached": reached}

    try:
        with run.recorder.source("edge_retry"):
            outcome = _correct_flagged_edge(run, source, target, instruction)
    except SystemOverlayActionError as exc:
        outcome = {"status": "skipped", "reason": str(exc)}
    logger.info("Flagged edge %s → %s: %s", source, target, outcome)
    if outcome["status"] in {"unverified", "skipped"}:
        return {"issue": issue, **outcome}
    source = run.graph.resolve_node_id(source)
    # Only an executed flagged action has moved the device off the source.
    executed = outcome["status"] != "instruction_not_found"
    if executed:
        try:
            reached = _replay_to(run, source)
        except _InspectionBlockedError as exc:
            outcome["reason"] = str(exc)
            reached = None
        if reached != run.graph.resolve_node_id(source):
            logger.warning("Replay after the correction missed %s; not exploring", source)
            return {"issue": issue, **outcome, "steps": []}

    target = run.graph.resolve_node_id(target)
    target_desc = run.graph.node_descriptions(target) if target in run.graph.graph else target
    hint = (
        f"A previous attempt to reach {target_desc} from this screen may have "
        f"gone to the wrong place. Check if any elements on this screen could lead "
        f"to {target_desc}."
    )
    steps = _explore_steps(
        run, run.graph.resolve_node_id(source), run.session.steps_per_issue, hint=hint, arrived=True
    )
    return {"issue": issue, **outcome, "steps": steps}


def _explore_node(run: _ReExploreRun, issue: dict[str, Any]) -> dict[str, Any]:
    node = run.graph.resolve_node_id(issue.get("node", ""))
    expected = issue.get("expected_pages", [])

    if not node or node not in run.graph.graph:
        logger.warning("Node %s not found, skipping", node)
        return {"issue": issue, "status": "skipped", "reason": "node not found"}

    logger.info(
        "Exploring node: %s (expected actions: %s)",
        node,
        expected,
    )

    actual = _replay_to(run, node)
    if actual is None:
        return {"issue": issue, "status": "skipped", "reason": "unreachable"}

    expected_pages = ", ".join(f"'{a}'" for a in expected)
    hint = (
        f"This screen might be able to lead to pages like {expected_pages}. "
        f"Check if any elements on this screen should be interacted with."
    )
    step_results = _explore_steps(
        run,
        actual,
        run.session.steps_per_issue,
        hint=hint,
        arrived=actual == run.graph.resolve_node_id(node),
    )

    return {
        "issue": issue,
        "status": "explored",
        "navigated_to": actual,
        "steps": step_results,
    }


def re_explore_issues(
    graph: GraphManager,
    issues: list[dict[str, Any]],
    *,
    target: ReExploreTarget,
    session: ReExploreSession,
    client: OpenRouter,
) -> list[dict[str, Any]]:
    """Re-explore flagged issues on device; README.md#correcting-flagged-edges.

    For retry_edge: replays the flagged action from the source node, corrects
    its edge by where it leads, then explores from the source with a hint.

    For explore_node: navigates to the node and explores with hints about
    expected missing actions.

    Every screenshot and action goes into the step log beside ``target.graph_path``.
    """
    results: list[dict[str, Any]] = []
    start_node = graph.get_start_node()
    if start_node is None:
        logger.warning("Graph is empty — nothing to re-explore")
        return results

    recorder = StepRecorder(
        session.controller,
        graph_steps_dir(target.graph_path),
        run_kind="audit",
        source="re_explore",
        target_package=target.package_name,
    )
    run = _ReExploreRun(
        graph=graph,
        target=target,
        app_launch=AppLaunch(target.app_name, target.package_name, target.launch_activity),
        session=session,
        recorder=recorder,
        client=client,
        start_node=start_node,
    )
    try:
        for issue in issues:
            itype = issue.get("type", "")
            if itype not in {"retry_edge", "explore_node"}:
                continue
            decision = execution_decision(issue)
            if not decision.executable:
                results.append(
                    {
                        "issue": issue,
                        "status": "skipped",
                        "reason": decision.reason,
                        "execution_policy": decision.as_record(),
                    }
                )
                continue

            try:
                handler = _retry_edge if itype == "retry_edge" else _explore_node
                results.append(handler(run, issue))
            except _InspectionBlockedError as exc:
                results.append({"issue": issue, "status": "skipped", "reason": str(exc)})
    except UnresolvedPermissionError, UnresolvedSystemOverlayError:
        graph.save_graph(target.graph_path)
        raise

    return results


class ReExploreSession(NamedTuple):
    """Device controller, role settings and limits needed by ``--re-explore``.

    Built as a unit so ``re_explore_issues`` takes non-optional arguments: the
    audit-only path never constructs one and never touches a device.
    """

    controller: DeviceController
    instruction: ModelSettings
    action: ModelSettings
    unreachable_limit: float
    steps_per_issue: int = 5


def _open_re_explore_session(
    config: dict[str, Any],
    settings: dict[str, ModelSettings],
    steps_per_issue: int,
    *,
    unreachable_limit: float,
) -> ReExploreSession:
    """Build the device session for ``--re-explore``."""
    return ReExploreSession(
        controller=AdbController(config, logger, unreachable_limit=unreachable_limit),
        instruction=settings["instruction"],
        action=settings["action"],
        unreachable_limit=unreachable_limit,
        steps_per_issue=steps_per_issue,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="app-graph-audit",
        description="Audit an explored graph for anomalies.",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=Path("configs/explore.yaml"),
        help="Path to the YAML config file (default: configs/explore.yaml).",
    )
    parser.add_argument(
        "--app",
        type=str,
        default=None,
        help=(
            "Basic audit only: audit one app graph folder under graph_dir by "
            "folder/app name, e.g. --app citymapper."
        ),
    )
    parser.add_argument(
        "--re-explore",
        action="store_true",
        help="Navigate to flagged nodes/edges on device and re-explore.",
    )
    parser.add_argument(
        "--steps-per-issue",
        type=int,
        default=5,
        help="Number of exploration steps per flagged issue (default: 5).",
    )
    return parser


@dataclass(frozen=True, slots=True)
class _AuditRun:
    """The config, client, model and matching settings and device session every audited
    app shares."""

    config: dict[str, Any]
    vlm_config: dict[str, Any]
    graph_dir: Path
    client: OpenRouter
    settings: dict[str, ModelSettings]
    matching: MatchingSettings
    session: ReExploreSession | None


def _app_file(graph_dir: Path, app_name: str, suffix: str = "") -> Path:
    return graph_dir / app_name / f"{app_name}{suffix}.json"


def _write_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _save_audited_graph(run: _AuditRun, graph: GraphManager, app_name: str, changes: str) -> None:
    # A separate file, so the pre-audit graph stays intact.
    audited_graph_path = _app_file(run.graph_dir, app_name, "_audited")
    graph.save_graph(audited_graph_path)
    logger.info("Audited graph saved to %s (%s)", audited_graph_path, changes)


def _verify_merges(
    run: _AuditRun,
    graph: GraphManager,
    app_name: str,
    proposals: list[MergeProposal],
    verdicts: MemberVerdicts,
) -> None:
    by_source = Counter(source for proposal in proposals for source in proposal.sources)
    logger.info(
        "Verifying %d merge proposal(s), by source: %s",
        len(proposals),
        ", ".join(f"{source} {count}" for source, count in sorted(by_source.items())),
    )
    report = graph.verify_merge_proposals(proposals, verdicts)
    merge_path = _app_file(run.graph_dir, app_name, "_merge")
    _write_json(merge_path, report.as_record())
    logger.info(
        "Merge results saved to %s: %d round(s), %d pair(s) asked, %d deduced, %d node(s) merged",
        merge_path,
        report.rounds,
        report.asked,
        report.deduced,
        report.merged_count,
    )
    if report.merged_count:
        _save_audited_graph(run, graph, app_name, f"{report.merged_count} node(s) merged")


def _check_members(
    run: _AuditRun, graph: GraphManager, app_name: str, verdicts: MemberVerdicts
) -> None:
    transitions = read_exploration_transitions(graph_steps_dir(_app_file(run.graph_dir, app_name)))
    report = check_members(graph, verdicts, transitions)
    members_path = _app_file(run.graph_dir, app_name, "_members")
    _write_json(members_path, report.as_record())
    logger.info(
        "Member checks saved to %s: %d check(s), %d change(s)",
        members_path,
        len(report.checks),
        report.changes,
    )
    if report.changes:
        _save_audited_graph(run, graph, app_name, f"{report.changes} member change(s)")


def _re_explore_app(
    run: _AuditRun,
    session: ReExploreSession,
    graph: GraphManager,
    app: dict[str, Any],
    issues: list[dict[str, Any]],
) -> None:
    app_name = app["name"]
    package_name = app["package_name"]
    logger.info("Starting re-exploration for %d issues...", len(issues))

    launch_activity = None
    if any(execution_decision(issue).executable for issue in issues):
        recovery = StartupRecovery(session.controller, session.unreachable_limit, logger)
        grant_runtime_permissions(run.config["device"]["udid"], package_name, recovery)
        launch_activity = launch_app(run.config, app, logger, recovery)
        time.sleep(4)

    target = ReExploreTarget(
        app_name=app_name,
        package_name=package_name,
        graph_path=_app_file(run.graph_dir, app_name, "_audited"),
        launch_activity=launch_activity,
    )
    re_results = re_explore_issues(graph, issues, target=target, session=session, client=run.client)

    results_path = _app_file(run.graph_dir, app_name, "_re_explore")
    _write_json(results_path, re_results)
    logger.info("Re-exploration results saved to %s", results_path)


def _report_no_path_to_start(
    graph: GraphManager, audit_result: dict[str, Any], report_path: Path, audited_graph_path: Path
) -> None:
    """Add the audited graph's nodes with no path back to the start node to the report.

    Called on the final graph because merges and re-exploration add and move edges;
    the lists therefore name the audited graph's IDs, not the pre-merge IDs of the
    model's issues.
    """
    found = graph.nodes_without_path_to_start()
    audit_result["no_path_to_start"] = {"graph": audited_graph_path.name, **found._asdict()}
    _write_json(report_path, audit_result)
    logger.log(
        logging.WARNING if found.in_app else logging.INFO,
        "%d in-app and %d external node(s) of %s have no path back to %s; listed in %s",
        len(found.in_app),
        len(found.external),
        audited_graph_path,
        found.start_node,
        report_path,
    )


def _audit_app(run: _AuditRun, app: dict[str, Any]) -> None:
    app_name = app["name"]

    graph = GraphManager(
        client=run.client,
        page_detail=run.settings["page_detail"],
        embedding=run.settings["embedding"],
        image_embedding=run.settings["image_embedding"],
        audit_timeout=resolve_audit_timeout(run.vlm_config),
    )
    graph.matching = run.matching

    graph_path = _app_file(run.graph_dir, app_name)
    if not graph_path.exists():
        logger.warning("No graph found at %s, skipping %s", graph_path, app_name)
        return

    graph.load_graph(graph_path)
    logger.info(
        "Loaded graph for %s: %d nodes, %d edges",
        app_name,
        graph.graph.number_of_nodes(),
        graph.graph.number_of_edges(),
    )
    audited_graph_path = _app_file(run.graph_dir, app_name, "_audited")
    graph.copy_identify_file(audited_graph_path)
    graph.embed_missing_members()

    report_path = _app_file(run.graph_dir, app_name, "_audit")
    audit_result = run_audit(graph, app_name, report_path)

    issues = audit_result.get("issues", [])
    if not issues:
        logger.info("No issues found for %s", app_name)
    else:
        logger.info("Found %d issues for %s", len(issues), app_name)

    merge_issues = [i for i in issues if i.get("type") == "merge_nodes"]
    other_issues = [i for i in issues if i.get("type") != "merge_nodes"]

    verdicts: MemberVerdicts = {}
    proposals = graph.merge_proposals(merge_issues)
    if proposals:
        _verify_merges(run, graph, app_name, proposals, verdicts)
    _check_members(run, graph, app_name, verdicts)

    if run.session is not None and other_issues:
        _re_explore_app(run, run.session, graph, app, other_issues)

    normalize_and_save_audited_graph(graph, audited_graph_path)
    _report_no_path_to_start(graph, audit_result, report_path, audited_graph_path)


def main(argv: list[str] | None = None) -> int:
    try:
        return _main(argv)
    except DeviceUnreachableError as exc:
        sys.stderr.write(f"app-graph-audit: error: {exc}\n")
        return 1


def _main(argv: list[str] | None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.config.is_file():
        parser.error(f"config file not found: {args.config}")

    with args.config.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    exp_config = config.get("experiment", {})
    vlm_config = config.get("vlm", {})
    graph_dir = Path(exp_config.get("graph_dir", "graphs"))

    try:
        check_config_keys(config)
        selected_apps = select_apps_for_audit(
            config,
            graph_dir,
            app_name=args.app,
            re_explore=args.re_explore,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    try:
        settings = resolve_model_settings(
            vlm_config, roles=RE_EXPLORE_ROLES if args.re_explore else AUDIT_ROLES
        )
        matching = resolve_matching_settings(vlm_config)
        unreachable_limit = resolve_unreachable_limit(config) if args.re_explore else None
        api_key = resolve_api_key()
    except ValueError as exc:
        parser.error(str(exc))

    # After argument validation, so --help and the parser.error paths above never
    # touch the root logger.
    setup_logging(level=logging.INFO)

    session = (
        _open_re_explore_session(
            config, settings, args.steps_per_issue, unreachable_limit=unreachable_limit
        )
        if unreachable_limit is not None
        else None
    )

    with open_client(api_key) as client:
        run = _AuditRun(
            config=config,
            vlm_config=vlm_config,
            graph_dir=graph_dir,
            client=client,
            settings=settings,
            matching=matching,
            session=session,
        )
        for app in selected_apps:
            _audit_app(run, app)

    token_tracker.print_summary()
    logger.info("Audit complete.")
    return 0
