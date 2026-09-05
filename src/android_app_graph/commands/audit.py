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
   Re-explore: navigate to the source node and re-run the exploration step.
   The planner is told which instruction to retry.

2. explore_node — A node that's missing expected outgoing edges.
   Re-explore: navigate to the node and explore with hints about what's missing.
   The planner is told what actions to try.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Any, NamedTuple

import yaml
from openai import OpenAI

from android_app_graph.cli import launch_app
from android_app_graph.device import DeviceController
from android_app_graph.utils import make_client
from android_app_graph.utils.graph_manager import GraphManager
from android_app_graph.utils.logging import setup_logging
from android_app_graph.utils.vlm_utils import (
    plan_next_action,
    predict_next_action,
    token_tracker,
    verify_same_node,
)

logger = logging.getLogger(__name__)


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
    page_detail_client: OpenAI,
    page_detail_model: str,
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

        # The auditor can name the same node twice.  Verifying it against its
        # own screenshot would always say "same", so skip the pair before
        # spending a VLM call on it.
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
            client=page_detail_client,
            screenshot_new_b64=screenshot_b,
            screenshot_existing_b64=screenshot_a,
            existing_description=ref_desc,
            model=page_detail_model,
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


def re_explore_issues(
    graph: GraphManager,
    issues: list[dict[str, Any]],
    controller: DeviceController,
    app_name: str,
    package_name: str,
    graph_path: Path,
    instruction_client: OpenAI,
    instruction_model: str,
    action_client: OpenAI,
    action_model: str,
    steps_per_issue: int = 5,
) -> list[dict[str, Any]]:
    """Re-explore flagged issues on device.

    For retry_edge: navigates to source node, re-runs the exploration step
    with a hint to retry the specific instruction.

    For explore_node: navigates to the node and explores with hints about
    expected missing actions.
    """
    results: list[dict[str, Any]] = []
    start_node = graph.get_start_node()
    if start_node is None:
        logger.warning("Graph is empty — nothing to re-explore")
        return results

    def _relaunch_app() -> None:
        relaunch = subprocess.run(
            [
                "adb",
                "-s",
                controller.config.get("device", controller.config).get("udid", "emulator-5554"),
                "shell",
                "monkey",
                "-p",
                package_name,
                "-c",
                "android.intent.category.LAUNCHER",
                "1",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if relaunch.returncode != 0:
            logger.warning(
                "Relaunch of %s exited with %d; continuing, the next state check will confirm",
                package_name,
                relaunch.returncode,
            )
        time.sleep(4)

    def _navigate_to(target_node: str) -> str | None:
        """Navigate to target node by replaying shortest path. Returns actual node reached."""
        path = graph.find_path(start_node, target_node)
        if path is None:
            logger.warning("No path from %s to %s", start_node, target_node)
            return None

        for i, (node_id, action) in enumerate(path):
            if i == 0:
                continue
            if isinstance(action, list):
                for sub_action in action:
                    controller.exe_action(sub_action)
                    time.sleep(0.5)
            elif action:
                controller.exe_action(action)
            time.sleep(1)

        device_state = controller.get_state()
        actual = graph.identify_state(
            device_state["activity"],
            device_state["screenshot"],
        )
        if actual != target_node:
            logger.warning(
                "Expected to reach %s but arrived at %s",
                target_node,
                actual,
            )
        return actual

    def _explore_steps(
        start_from: str,
        num_steps: int,
        hint: str = "",
    ) -> list[dict[str, Any]]:
        """Run exploration steps from a node with an optional hint for the planner."""
        current_node = start_from
        step_results: list[dict[str, Any]] = []

        for step_i in range(num_steps):
            device_state = controller.get_state()
            screenshot = device_state["screenshot"]
            node_data = graph.get_node(current_node)
            if node_data is None:
                break

            explored_edges = graph.get_all_edges_from_node(current_node)

            page_desc = node_data["page_description"]
            if hint:
                page_desc = f"{page_desc}\n[AUDIT HINT: {hint}]"

            # Check before planning so the planner can choose type vs tap.
            keyboard_hint = ""
            input_status = (
                "No OS keyboard signal detected. Still inspect the screenshot: "
                "a bottom input/keyboard bar can mean a text field is active."
            )
            try:
                kb_check = subprocess.run(
                    ["adb", "shell", "dumpsys", "input_method"],
                    capture_output=True,
                    text=True,
                    timeout=3,
                    check=False,
                )
                if "mInputShown=true" in kb_check.stdout:
                    keyboard_hint = " (Note: the soft keyboard is currently visible.)"
                    input_status = (
                        "Soft keyboard is visible; a text field is focused and ready for typing."
                    )
            except (OSError, subprocess.SubprocessError) as exc:
                logger.debug("Soft keyboard probe failed: %s", exc)

            instruction = plan_next_action(
                client=instruction_client,
                screenshot_b64=screenshot,
                page_description=page_desc,
                explored_edges=explored_edges,
                app_name=app_name,
                input_status=input_status,
                model=instruction_model,
            )

            action, _history_entry = predict_next_action(
                client=action_client,
                screenshot_b64=screenshot,
                instruction=instruction + keyboard_hint,
                screen_w=controller.w,
                screen_h=controller.h,
                action_history=[],
                model=action_model,
            )

            if action.get("action") == "end":
                action = {"action": "back"}
                controller.exe_action(action)
            elif action.get("action") == "wait":
                time.sleep(1)
                continue
            else:
                controller.exe_action(action)

            time.sleep(0.5)
            device_state = controller.get_state()

            current_package = device_state.get("package", "").strip()
            if current_package and current_package != package_name:
                logger.warning("Left app during re-explore, pressing back")
                controller.exe_action({"action": "back"})
                time.sleep(1)
                device_state = controller.get_state()
                current_node = graph.identify_state(
                    device_state["activity"],
                    device_state["screenshot"],
                )
                continue

            new_node = graph.identify_state(
                device_state["activity"],
                device_state["screenshot"],
            )

            new_node_data = graph.get_node(new_node)
            target_obs = new_node_data.get("page_description", "") if new_node_data else ""

            graph.add_edge(
                current_node,
                new_node,
                action=[action],
                instruction=instruction,
                target_observation=target_obs,
                num_steps=1,
            )

            step_results.append(
                {
                    "step": step_i + 1,
                    "from": current_node,
                    "to": new_node,
                    "instruction": instruction,
                    "action": action,
                }
            )

            logger.info(
                "  Re-explore step %d: %s --[%s]--> %s (instruction: '%s')",
                step_i + 1,
                current_node,
                action.get("action", "?"),
                new_node,
                instruction,
            )

            graph.save_graph(graph_path)
            current_node = new_node

        return step_results

    for issue in issues:
        itype = issue.get("type", "")

        if itype == "retry_edge":
            source = issue.get("source_node", "")
            target = issue.get("target_node", "")
            instr = issue.get("instruction", "")

            if not source or source not in graph.graph:
                logger.warning("Source node %s not found, skipping", source)
                results.append({"issue": issue, "status": "skipped", "reason": "source not found"})
                continue

            logger.info(
                "Retrying edge: %s → %s (instruction: '%s')",
                source,
                target,
                instr,
            )

            _relaunch_app()
            actual = _navigate_to(source)
            if actual is None:
                results.append({"issue": issue, "status": "skipped", "reason": "unreachable"})
                continue

            target_desc = (
                graph.graph.nodes[target].get("page_description", target)
                if target in graph.graph
                else target
            )
            hint = (
                f"A previous attempt to reach '{target_desc}' from this screen may have "
                f"gone to the wrong place. Check if any elements on this screen could lead "
                f"to '{target_desc}'."
            )
            step_results = _explore_steps(actual, steps_per_issue, hint=hint)

            results.append(
                {
                    "issue": issue,
                    "status": "retried",
                    "navigated_to": actual,
                    "steps": step_results,
                }
            )

        elif itype == "explore_node":
            node = issue.get("node", "")
            expected = issue.get("expected_pages", [])

            if not node or node not in graph.graph:
                logger.warning("Node %s not found, skipping", node)
                results.append({"issue": issue, "status": "skipped", "reason": "node not found"})
                continue

            logger.info(
                "Exploring node: %s (expected actions: %s)",
                node,
                expected,
            )

            _relaunch_app()
            actual = _navigate_to(node)
            if actual is None:
                results.append({"issue": issue, "status": "skipped", "reason": "unreachable"})
                continue

            expected_pages = ", ".join(f"'{a}'" for a in expected)
            hint = (
                f"This screen might be able to lead to pages like {expected_pages}. "
                f"Check if any elements on this screen should be interacted with."
            )
            step_results = _explore_steps(actual, steps_per_issue, hint=hint)

            results.append(
                {
                    "issue": issue,
                    "status": "explored",
                    "navigated_to": actual,
                    "steps": step_results,
                }
            )

    return results


class ReExploreSession(NamedTuple):
    """Device controller and VLM clients needed by ``--re-explore``.

    Built as a unit so ``re_explore_issues`` takes non-optional arguments: the
    audit-only path never constructs one and never imports ``aitk``.
    """

    controller: DeviceController
    instruction_client: OpenAI
    instruction_model: str
    action_client: OpenAI
    action_model: str


def _open_re_explore_session(
    parser: argparse.ArgumentParser,
    config: dict[str, Any],
    vlm_config: dict[str, Any],
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

    instruction_client, instruction_model = make_client(vlm_config.get("instruction"))
    action_client, action_model = make_client(vlm_config.get("action"))
    return ReExploreSession(
        controller=ADBController(config, logger),
        instruction_client=instruction_client,
        instruction_model=instruction_model,
        action_client=action_client,
        action_model=action_model,
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

    # After argument validation, so --help and the parser.error paths above never
    # touch the root logger; before the controller, which logs while it connects.
    setup_logging(level=logging.INFO)

    session = _open_re_explore_session(parser, config, vlm_config) if args.re_explore else None

    page_detail_client, page_detail_model = make_client(vlm_config.get("page_detail"))
    embedding_client, embedding_model = make_client(vlm_config.get("embedding"))

    for app in selected_apps:
        app_name = app["name"]
        package_name = app["package_name"]

        graph = GraphManager(
            page_detail_client=page_detail_client,
            page_detail_model=page_detail_model,
            embedding_client=embedding_client,
            embedding_model=embedding_model,
            similarity_threshold=vlm_config.get("similarity_threshold", 0.85),
        )

        graph_path = graph_dir / app_name / f"{app_name}.json"
        audited_graph_path = graph_dir / app_name / f"{app_name}_audited.json"
        if not graph_path.exists():
            logger.warning("No graph found at %s, skipping %s", graph_path, app_name)
            continue

        graph.load_graph(graph_path)
        logger.info(
            "Loaded graph for %s: %d nodes, %d edges",
            app_name,
            graph.graph.number_of_nodes(),
            graph.graph.number_of_edges(),
        )

        report_path = graph_path.parent / f"{app_name}_audit.json"
        audit_result = run_audit(graph, app_name, report_path)

        issues = audit_result.get("issues", [])
        if not issues:
            logger.info("No issues found for %s", app_name)
        else:
            logger.info("Found %d issues for %s", len(issues), app_name)

        merge_issues = [i for i in issues if i.get("type") == "merge_nodes"]
        other_issues = [i for i in issues if i.get("type") != "merge_nodes"]

        if merge_issues:
            logger.info("Verifying %d merge candidates...", len(merge_issues))
            merge_results = verify_and_merge_nodes(
                graph,
                merge_issues,
                page_detail_client,
                page_detail_model,
            )
            merge_path = graph_path.parent / f"{app_name}_merge.json"
            with merge_path.open("w", encoding="utf-8") as f:
                json.dump(merge_results, f, indent=2, ensure_ascii=False)
            logger.info("Merge results saved to %s", merge_path)

            # A separate file, so the pre-audit graph stays intact.
            merged_count = sum(1 for r in merge_results if r["status"] == "merged")
            if merged_count:
                graph.save_graph(audited_graph_path)
                logger.info(
                    "Audited graph saved to %s (%d merge(s))", audited_graph_path, merged_count
                )

        if session is not None and other_issues:
            logger.info("Starting re-exploration for %d issues...", len(other_issues))

            launch_app(config, app, logger)
            time.sleep(4)

            re_results = re_explore_issues(
                graph,
                other_issues,
                session.controller,
                app_name,
                package_name,
                audited_graph_path,
                session.instruction_client,
                session.instruction_model,
                session.action_client,
                session.action_model,
                steps_per_issue=args.steps_per_issue,
            )

            results_path = graph_path.parent / f"{app_name}_re_explore.json"
            with results_path.open("w", encoding="utf-8") as f:
                json.dump(re_results, f, indent=2, ensure_ascii=False)
            logger.info("Re-exploration results saved to %s", results_path)

        # Runs even when the auditor reported no structural issues.
        normalize_and_save_audited_graph(graph, audited_graph_path)

    token_tracker.print_summary()
    logger.info("Audit complete.")
    return 0
