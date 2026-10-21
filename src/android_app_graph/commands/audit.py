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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import yaml
from openrouter import OpenRouter

from android_app_graph.device import (
    DeviceController,
    grant_runtime_permissions,
    is_system_overlay,
    keyboard_status,
    launch_app,
    restart_app,
    soft_keyboard_hint,
)
from android_app_graph.utils import (
    ModelSettings,
    open_client,
    resolve_api_key,
    resolve_audit_timeout,
    resolve_model_settings,
)
from android_app_graph.utils.graph_manager import (
    EXTERNAL_WEB_BACK_PRESSES,
    EXTERNAL_WEB_NODE,
    GraphManager,
)
from android_app_graph.utils.logging import setup_logging
from android_app_graph.utils.vlm_utils import (
    describe_executed_action,
    plan_next_action,
    predict_next_action,
    token_tracker,
    verify_same_node,
)

logger = logging.getLogger(__name__)

AUDIT_ROLES = ("page_detail", "embedding")
RE_EXPLORE_ROLES = (*AUDIT_ROLES, "instruction", "action")


def run_audit(graph: GraphManager, app_name: str, report_path: Path) -> dict[str, Any]:
    """Run the LLM audit and save the report."""
    result = graph.run_audit(app_name=app_name)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info("Audit report saved to %s", report_path)

    return result


def normalize_and_save_audited_graph(graph: GraphManager, audited_graph_path: Path) -> int:
    """Normalize outgoing edge templates for every node, then save audited graph.

    This runs after audit merges and optional re-exploration, so templates are
    generated from the final graph topology rather than stale pre-merge edges.
    """
    logger.info(
        "Normalizing audited graph edge templates for %d nodes...",
        graph.graph.number_of_nodes(),
    )
    normalized_count = 0
    for node_id in list(graph.graph.nodes()):
        normalized_count += graph.normalize_node_edges(node_id)

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

    return [{"name": app_name, "package_name": ""}]


def verify_and_merge_nodes(
    graph: GraphManager,
    merge_issues: list[dict[str, Any]],
    client: OpenRouter,
    page_detail: ModelSettings,
) -> list[dict[str, Any]]:
    """Verify merge candidates using screenshot comparison, then merge confirmed pairs.

    For each merge_nodes issue, the verifier compares the reference screenshots
    of the two candidate nodes. If the verifier says they are the same screen,
    the nodes are merged (the lower-numbered node is kept).

    Returns a list of result dicts with status "merged" or "kept_separate".
    """
    results: list[dict[str, Any]] = []
    for issue in merge_issues:
        node_a = issue.get("node_a", "")
        node_b = issue.get("node_b", "")

        # An earlier merge in this loop may already have removed one of the pair.
        if node_a not in graph.graph or node_b not in graph.graph:
            logger.info("Skipping merge %s + %s — node already removed", node_a, node_b)
            results.append({"issue": issue, "status": "skipped", "reason": "node removed"})
            continue

        # An auditor can name one node twice, and it would verify as "same".
        if node_a == node_b:
            logger.warning("Skipping merge %s + itself — same node", node_a)
            results.append({"issue": issue, "status": "skipped", "reason": "same node"})
            continue

        data_a = graph.graph.nodes[node_a]
        data_b = graph.graph.nodes[node_b]
        screenshot_a = data_a.get("reference_screenshot")
        screenshot_b = data_b.get("reference_screenshot")

        if not screenshot_a or not screenshot_b:
            logger.warning("Skipping merge %s + %s — missing screenshot(s)", node_a, node_b)
            results.append({"issue": issue, "status": "skipped", "reason": "missing screenshot"})
            continue

        # The more-visited node has the better-established description.
        desc_a = data_a.get("page_description", "")
        desc_b = data_b.get("page_description", "")
        visits_a = data_a.get("visit_count", 0)
        visits_b = data_b.get("visit_count", 0)
        ref_desc = desc_a if visits_a >= visits_b else desc_b

        logger.info(
            "Verifying merge: %s ('%s') + %s ('%s')",
            node_a,
            desc_a,
            node_b,
            desc_b,
        )

        verify_result = verify_same_node(
            client=client,
            screenshot_new_b64=screenshot_b,
            screenshot_existing_b64=screenshot_a,
            existing_description=ref_desc,
            settings=page_detail,
        )

        if verify_result.get("same", False):
            # The lower-numbered node was discovered first.
            keep, remove = (node_a, node_b) if node_a < node_b else (node_b, node_a)
            graph.merge_nodes(keep, remove)
            logger.info(
                "Merged %s into %s — reason: %s",
                remove,
                keep,
                verify_result.get("reason", ""),
            )
            results.append(
                {
                    "issue": issue,
                    "status": "merged",
                    "kept": keep,
                    "removed": remove,
                    "reason": verify_result.get("reason", ""),
                }
            )
        else:
            logger.info(
                "Kept separate: %s + %s — reason: %s",
                node_a,
                node_b,
                verify_result.get("reason", ""),
            )
            results.append(
                {
                    "issue": issue,
                    "status": "kept_separate",
                    "reason": verify_result.get("reason", ""),
                }
            )

    return results


@dataclass(frozen=True, slots=True)
class _ReExploreRun:
    """The graph, device, app and models that one ``re_explore_issues`` call works with."""

    graph: GraphManager
    controller: DeviceController
    app_name: str
    package_name: str
    graph_path: Path
    client: OpenRouter
    instruction_settings: ModelSettings
    action_settings: ModelSettings
    launch_activity: str | None
    start_node: str


def _left_app(run: _ReExploreRun, device_state: dict[str, Any]) -> bool:
    """A keyboard or system dialog on top of the app does not count as leaving it."""
    package = device_state.get("package", "").strip()
    return bool(package and package != run.package_name and not is_system_overlay(package))


def _execute(controller: DeviceController, action: dict[str, Any] | list[dict[str, Any]]) -> None:
    if isinstance(action, list):
        for sub_action in action:
            controller.exe_action(sub_action)
            time.sleep(0.5)
    elif action:
        controller.exe_action(action)
    time.sleep(1)


def _navigate_to(run: _ReExploreRun, target_node: str) -> str | None:
    """Navigate to target node by replaying shortest path. Returns actual node reached."""
    start = run.graph.resolve_node_id(run.start_node)
    path = run.graph.find_path(start, target_node)
    if path is None:
        logger.warning("No path from %s to %s", start, target_node)
        return None

    for _node_id, action in path[1:]:
        _execute(run.controller, action)

    device_state = run.controller.get_state()
    actual = run.graph.identify_state(
        device_state["activity"],
        device_state["screenshot"],
        device_state["package"],
    )
    if actual != target_node:
        logger.warning(
            "Expected to reach %s but arrived at %s",
            target_node,
            actual,
        )
    return actual


def _leave_external_web(run: _ReExploreRun) -> str:
    """Press back until the screen is no longer an outside web page, restarting the
    app if a few presses do not leave it, and return the screen's node.

    A step from the shared ``ext_web`` node would act on a page outside the app, and
    ``add_edge`` would refuse its edge.
    """
    for _ in range(EXTERNAL_WEB_BACK_PRESSES):
        run.controller.exe_action({"action": "back"})
        time.sleep(1)
        device_state = run.controller.get_state()
        node = run.graph.identify_state(
            device_state["activity"], device_state["screenshot"], device_state["package"]
        )
        if node != EXTERNAL_WEB_NODE:
            return node
    logger.warning(
        "Still on an external web page after %d back presses. Restarting %s.",
        EXTERNAL_WEB_BACK_PRESSES,
        run.package_name,
    )
    restart_app(run.controller, run.app_name, run.package_name, run.launch_activity, logger)
    device_state = run.controller.get_state()
    return run.graph.identify_state(
        device_state["activity"], device_state["screenshot"], device_state["package"]
    )


def _plan_and_act(
    run: _ReExploreRun, node: str, screenshot: str, page_description: str
) -> tuple[str, dict[str, Any]] | None:
    """Plan a step from *node*, execute its action, and return the instruction and the
    action executed, or ``None`` when the action model waits.
    """
    explored_edges = run.graph.get_all_edges_from_node(node)

    # Check before planning so the planner can choose type vs tap.
    keyboard_hint = soft_keyboard_hint()
    input_status = keyboard_status(keyboard_hint)

    instruction, _planned_element = plan_next_action(
        client=run.client,
        screenshot_b64=screenshot,
        page_description=page_description,
        explored_edges=explored_edges,
        app_name=run.app_name,
        input_status=input_status,
        settings=run.instruction_settings,
    )

    action, _history_entry = predict_next_action(
        client=run.client,
        screenshot_b64=screenshot,
        instruction=instruction + keyboard_hint,
        screen_w=run.controller.w,
        screen_h=run.controller.h,
        action_history=[],
        settings=run.action_settings,
    )

    if action.get("action") == "end":
        logger.info(
            '  Action agent said end for: "%s", recording as back',
            instruction,
        )
        action = {"action": "back"}
        instruction = "Press back"
        run.controller.exe_action(action)
    elif action.get("action") == "wait":
        time.sleep(1)
        return None
    else:
        run.controller.exe_action(action)
    return instruction, action


def _record_step(
    run: _ReExploreRun,
    source: str,
    taken: tuple[str, dict[str, Any]],
    device_state: dict[str, Any],
    step_number: int,
) -> tuple[str, dict[str, Any]]:
    """Record the edge *taken* from *source* to the screen *device_state* shows and save
    the graph. Returns the node the next step starts from, with the step's result.
    """
    instruction, action = taken
    new_node = run.graph.identify_state(
        device_state["activity"],
        device_state["screenshot"],
        device_state["package"],
    )
    source = run.graph.resolve_node_id(source)

    new_node_data = run.graph.get_node(new_node)
    target_obs = new_node_data.get("page_description", "") if new_node_data else ""

    run.graph.add_edge(
        source,
        new_node,
        action=[action],
        instruction=instruction,
        target_observation=target_obs,
        num_steps=1,
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

    run.graph.save_graph(run.graph_path)
    if new_node == EXTERNAL_WEB_NODE:
        logger.warning("Opened an external web page during re-explore, pressing back")
        return _leave_external_web(run), step_result
    return new_node, step_result


def _explore_steps(
    run: _ReExploreRun,
    start_from: str,
    num_steps: int,
    hint: str,
) -> list[dict[str, Any]]:
    """Run exploration steps from a node with a hint for the planner."""
    current_node = start_from
    step_results: list[dict[str, Any]] = []

    for step_i in range(num_steps):
        device_state = run.controller.get_state()
        screenshot = device_state["screenshot"]
        node_data = run.graph.get_node(current_node)
        if node_data is None:
            break

        page_desc = f"{node_data['page_description']}\n[AUDIT HINT: {hint}]"
        taken = _plan_and_act(run, current_node, screenshot, page_desc)
        if taken is None:
            continue

        time.sleep(0.5)
        device_state = run.controller.get_state()

        if _left_app(run, device_state):
            logger.warning("Left app during re-explore, pressing back")
            run.controller.exe_action({"action": "back"})
            time.sleep(1)
            device_state = run.controller.get_state()
            current_node = run.graph.identify_state(
                device_state["activity"],
                device_state["screenshot"],
                device_state["package"],
            )
            continue

        current_node, step_result = _record_step(run, current_node, taken, device_state, step_i + 1)
        step_results.append(step_result)

    return step_results


def _replay_to(run: _ReExploreRun, node: str) -> str | None:
    restart_app(run.controller, run.app_name, run.package_name, run.launch_activity, logger)
    return _navigate_to(run, node)


def _correct_flagged_edge(
    run: _ReExploreRun, source: str, target: str, instruction: str
) -> dict[str, Any]:
    """Execute the flagged action once from *source* and correct its edge by where it
    leads. Returns the outcome fields of the issue's result.
    """
    found = run.graph.find_edge_action(source, target, instruction)
    if found is None:
        logger.warning("Instruction '%s' is not on %s → %s", instruction, source, target)
        return {"status": "instruction_not_found"}
    index, action = found

    before = run.controller.get_state()
    _execute(run.controller, action)
    after = run.controller.get_state()
    if _left_app(run, after):
        reached = f"ext_{after['package'].strip()}"
    else:
        reached = run.graph.identify_state(after["activity"], after["screenshot"], after["package"])
    source, target = run.graph.resolve_node_id(source), run.graph.resolve_node_id(target)

    if reached != target:
        if not run.graph.move_edge_action(source, target, index, reached):
            return {"status": "unverified", "reached": reached}
        return {"status": "moved", "moved_to": reached}

    label = describe_executed_action(
        client=run.client,
        screenshot_b64=before["screenshot"],
        action=action,
        screen_w=run.controller.w,
        screen_h=run.controller.h,
        settings=run.instruction_settings,
    )
    if label is None:
        return {"status": "relabel_failed"}
    run.graph.relabel_edge_action(source, target, index, label)
    return {"status": "relabelled", "instruction": label}


def _retry_edge(run: _ReExploreRun, issue: dict[str, Any], steps_per_issue: int) -> dict[str, Any]:
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

    logger.info("Retrying edge: %s → %s (instruction: '%s')", source, target, instruction)
    reached = _replay_to(run, source)
    source, target = run.graph.resolve_node_id(source), run.graph.resolve_node_id(target)
    if reached != source:
        return {"issue": issue, "status": "unverified", "reached": reached}

    outcome = _correct_flagged_edge(run, source, target, instruction)
    logger.info("Flagged edge %s → %s: %s", source, target, outcome)
    if outcome["status"] == "unverified":
        return {"issue": issue, **outcome}
    source = run.graph.resolve_node_id(source)
    # Only an executed flagged action has moved the device off the source.
    executed = outcome["status"] != "instruction_not_found"
    if executed and _replay_to(run, source) != run.graph.resolve_node_id(source):
        logger.warning("Replay after the correction missed %s; not exploring", source)
        return {"issue": issue, **outcome, "steps": []}

    target = run.graph.resolve_node_id(target)
    target_desc = (
        run.graph.graph.nodes[target].get("page_description", target)
        if target in run.graph.graph
        else target
    )
    hint = (
        f"A previous attempt to reach '{target_desc}' from this screen may have "
        f"gone to the wrong place. Check if any elements on this screen could lead "
        f"to '{target_desc}'."
    )
    steps = _explore_steps(run, run.graph.resolve_node_id(source), steps_per_issue, hint=hint)
    return {"issue": issue, **outcome, "steps": steps}


def _explore_node(
    run: _ReExploreRun, issue: dict[str, Any], steps_per_issue: int
) -> dict[str, Any]:
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
    step_results = _explore_steps(run, actual, steps_per_issue, hint=hint)

    return {
        "issue": issue,
        "status": "explored",
        "navigated_to": actual,
        "steps": step_results,
    }


def re_explore_issues(
    graph: GraphManager,
    issues: list[dict[str, Any]],
    controller: DeviceController,
    app_name: str,
    package_name: str,
    graph_path: Path,
    client: OpenRouter,
    instruction_settings: ModelSettings,
    action_settings: ModelSettings,
    steps_per_issue: int = 5,
    launch_activity: str | None = None,
) -> list[dict[str, Any]]:
    """Re-explore flagged issues on device; README.md#correcting-flagged-edges.

    For retry_edge: replays the flagged action from the source node, corrects
    its edge by where it leads, then explores from the source with a hint.

    For explore_node: navigates to the node and explores with hints about
    expected missing actions.
    """
    results: list[dict[str, Any]] = []
    start_node = graph.get_start_node()
    if start_node is None:
        logger.warning("Graph is empty — nothing to re-explore")
        return results

    run = _ReExploreRun(
        graph=graph,
        controller=controller,
        app_name=app_name,
        package_name=package_name,
        graph_path=graph_path,
        client=client,
        instruction_settings=instruction_settings,
        action_settings=action_settings,
        launch_activity=launch_activity,
        start_node=start_node,
    )
    for issue in issues:
        itype = issue.get("type", "")

        if itype == "retry_edge":
            results.append(_retry_edge(run, issue, steps_per_issue))
        elif itype == "explore_node":
            results.append(_explore_node(run, issue, steps_per_issue))

    return results


class ReExploreSession(NamedTuple):
    """Device controller and role settings needed by ``--re-explore``.

    Built as a unit so ``re_explore_issues`` takes non-optional arguments: the
    audit-only path never constructs one and never imports ``aitk``.
    """

    controller: DeviceController
    instruction: ModelSettings
    action: ModelSettings


def _open_re_explore_session(
    parser: argparse.ArgumentParser,
    config: dict[str, Any],
    settings: dict[str, ModelSettings],
) -> ReExploreSession:
    """Build the device session for ``--re-explore``, or exit with guidance.

    The ``aitk`` import stays inside this function so a plain audit runs without
    the device dependency installed.
    """
    try:
        from aitk.utils.adb_controller import ADBController
    except ImportError:
        parser.exit(
            1,
            "app-graph-audit --re-explore requires the 'aitk' dependency. "
            "Install it first, then rerun the command.\n",
        )

    return ReExploreSession(
        controller=ADBController(config, logger),
        instruction=settings["instruction"],
        action=settings["action"],
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
    """The config, client, model settings and device session every audited app shares."""

    config: dict[str, Any]
    vlm_config: dict[str, Any]
    graph_dir: Path
    client: OpenRouter
    settings: dict[str, ModelSettings]
    session: ReExploreSession | None
    steps_per_issue: int


def _app_file(graph_dir: Path, app_name: str, suffix: str = "") -> Path:
    return graph_dir / app_name / f"{app_name}{suffix}.json"


def _write_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _verify_merges(
    run: _AuditRun, graph: GraphManager, app_name: str, merge_issues: list[dict[str, Any]]
) -> None:
    logger.info("Verifying %d merge candidates...", len(merge_issues))
    merge_results = verify_and_merge_nodes(
        graph, merge_issues, run.client, run.settings["page_detail"]
    )
    merge_path = _app_file(run.graph_dir, app_name, "_merge")
    _write_json(merge_path, merge_results)
    logger.info("Merge results saved to %s", merge_path)

    # A separate file, so the pre-audit graph stays intact.
    merged_count = sum(1 for r in merge_results if r["status"] == "merged")
    if merged_count:
        audited_graph_path = _app_file(run.graph_dir, app_name, "_audited")
        graph.save_graph(audited_graph_path)
        logger.info("Audited graph saved to %s (%d merge(s))", audited_graph_path, merged_count)


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

    grant_runtime_permissions(run.config["device"]["udid"], package_name)
    launch_app(run.config, app, logger)
    time.sleep(4)

    re_results = re_explore_issues(
        graph,
        issues,
        session.controller,
        app_name,
        package_name,
        _app_file(run.graph_dir, app_name, "_audited"),
        run.client,
        session.instruction,
        session.action,
        steps_per_issue=run.steps_per_issue,
        launch_activity=app.get("launch_activity"),
    )

    results_path = _app_file(run.graph_dir, app_name, "_re_explore")
    _write_json(results_path, re_results)
    logger.info("Re-exploration results saved to %s", results_path)


def _audit_app(run: _AuditRun, app: dict[str, Any]) -> None:
    app_name = app["name"]
    package_name = app["package_name"]

    graph = GraphManager(
        client=run.client,
        page_detail=run.settings["page_detail"],
        embedding=run.settings["embedding"],
        similarity_threshold=run.vlm_config.get("similarity_threshold", 0.85),
        audit_timeout=resolve_audit_timeout(run.vlm_config),
    )

    graph_path = _app_file(run.graph_dir, app_name)
    if not graph_path.exists():
        logger.warning("No graph found at %s, skipping %s", graph_path, app_name)
        return

    graph.load_graph(graph_path, default_package_name=package_name)
    logger.info(
        "Loaded graph for %s: %d nodes, %d edges",
        app_name,
        graph.graph.number_of_nodes(),
        graph.graph.number_of_edges(),
    )

    report_path = _app_file(run.graph_dir, app_name, "_audit")
    audit_result = run_audit(graph, app_name, report_path)

    issues = audit_result.get("issues", [])
    if not issues:
        logger.info("No issues found for %s", app_name)
    else:
        logger.info("Found %d issues for %s", len(issues), app_name)

    merge_issues = [i for i in issues if i.get("type") == "merge_nodes"]
    other_issues = [i for i in issues if i.get("type") != "merge_nodes"]

    if merge_issues:
        _verify_merges(run, graph, app_name, merge_issues)

    if run.session is not None and other_issues:
        _re_explore_app(run, run.session, graph, app, other_issues)

    normalize_and_save_audited_graph(graph, _app_file(run.graph_dir, app_name, "_audited"))


def main(argv: list[str] | None = None) -> int:
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
        api_key = resolve_api_key()
    except ValueError as exc:
        parser.error(str(exc))

    # After argument validation, so --help and the parser.error paths above never
    # touch the root logger; before the controller, which logs while it connects.
    setup_logging(level=logging.INFO)

    session = _open_re_explore_session(parser, config, settings) if args.re_explore else None

    with open_client(api_key) as client:
        run = _AuditRun(
            config=config,
            vlm_config=vlm_config,
            graph_dir=graph_dir,
            client=client,
            settings=settings,
            session=session,
            steps_per_issue=args.steps_per_issue,
        )
        for app in selected_apps:
            _audit_app(run, app)

    token_tracker.print_summary()
    logger.info("Audit complete.")
    return 0
