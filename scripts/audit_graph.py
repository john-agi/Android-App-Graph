"""Audit an explored graph and optionally re-explore flagged issues on device.

Usage:
    # Audit only (no device needed) — prints issues and saves report
    uv run python scripts/audit_graph.py -c configs/explore.yaml

    # Audit + re-explore flagged issues on device
    uv run python scripts/audit_graph.py -c configs/explore.yaml --re-explore

    # Re-explore with custom step budget per issue
    uv run python scripts/audit_graph.py -c configs/explore.yaml --re-explore --steps-per-issue 10

The auditor finds two types of issues:

1. retry_edge — An edge whose instruction doesn't match its destination.
   Re-explore: navigate to the source node and re-run the exploration step.
   The planner is told which instruction to retry.

2. explore_node — A node that's missing expected outgoing edges.
   Re-explore: navigate to the node and explore with hints about what's missing.
   The planner is told what actions to try.
"""

import argparse
import json
import logging
import subprocess
import time
from pathlib import Path

import yaml

from ui_kobe.utils import make_client
from ui_kobe.utils.graph_manager import GraphManager
from ui_kobe.utils.logging import setup_logging
from ui_kobe.utils.vlm_utils import (
    plan_next_action,
    predict_next_action,
    token_tracker,
    verify_same_node,
)

logger = logging.getLogger("ui_kobe.audit")


def run_audit(graph: GraphManager, app_name: str, report_path: Path) -> dict:
    """Run the LLM audit and save the report."""
    result = graph.run_audit(app_name=app_name)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
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


def launch_app(config: dict, app: dict) -> None:
    """Launch an app using an explicit activity when configured."""
    udid = config["device"]["udid"]
    app_name = app["name"]
    package_name = app["package_name"]
    launch_activity = app.get("launch_activity")

    if launch_activity:
        logger.info("Launching app %s via activity %s ...", app_name, launch_activity)
        result = subprocess.run(
            [
                "adb",
                "-s",
                udid,
                "shell",
                "am",
                "start",
                "-n",
                launch_activity,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return
        logger.warning(
            "Activity launch failed for %s; falling back to package launch. error=%s",
            app_name,
            (result.stderr or result.stdout or "").strip(),
        )

    logger.info("Launching app %s (%s) ...", app_name, package_name)
    subprocess.run(
        [
            "adb",
            "-s",
            udid,
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
    )


def select_apps_for_audit(
    config: dict,
    graph_dir: Path,
    app_name: str | None = None,
    re_explore: bool = False,
) -> list[dict]:
    """Return app entries to audit.

    In basic audit mode, ``--app`` selects a graph folder by name and does not
    require the app to be present in config["apps"]. Re-exploration still needs
    package_name/device metadata from the config, so app filtering is disabled
    there.
    """
    if not app_name:
        return list(config.get("apps", []))

    if re_explore:
        raise ValueError("--app can only be used for basic audit without --re-explore")

    graph_path = graph_dir / app_name / f"{app_name}.json"
    if not graph_path.exists():
        raise FileNotFoundError(f"No graph found at {graph_path}")

    return [{"name": app_name, "package_name": ""}]


def verify_and_merge_nodes(
    graph: GraphManager,
    merge_issues: list[dict],
    page_detail_client,
    page_detail_model: str,
) -> list[dict]:
    """Verify merge candidates using screenshot comparison, then merge confirmed pairs.

    For each merge_nodes issue, the verifier compares the reference screenshots
    of the two candidate nodes. If the verifier says they are the same screen,
    the nodes are merged (the lower-numbered node is kept).

    Returns a list of result dicts with status "merged" or "kept_separate".
    """
    results = []
    for issue in merge_issues:
        node_a = issue.get("node_a", "")
        node_b = issue.get("node_b", "")

        # Check both nodes still exist (earlier merge may have removed one)
        if node_a not in graph.graph or node_b not in graph.graph:
            logger.info("Skipping merge %s + %s — node already removed", node_a, node_b)
            results.append({"issue": issue, "status": "skipped", "reason": "node removed"})
            continue

        data_a = graph.graph.nodes[node_a]
        data_b = graph.graph.nodes[node_b]
        screenshot_a = data_a.get("reference_screenshot")
        screenshot_b = data_b.get("reference_screenshot")

        if not screenshot_a or not screenshot_b:
            logger.warning(
                "Skipping merge %s + %s — missing screenshot(s)", node_a, node_b
            )
            results.append({"issue": issue, "status": "skipped", "reason": "missing screenshot"})
            continue

        # Use the description of the node with more visits as the reference
        desc_a = data_a.get("page_description", "")
        desc_b = data_b.get("page_description", "")
        visits_a = data_a.get("visit_count", 0)
        visits_b = data_b.get("visit_count", 0)
        ref_desc = desc_a if visits_a >= visits_b else desc_b

        logger.info(
            "Verifying merge: %s ('%s') + %s ('%s')",
            node_a, desc_a, node_b, desc_b,
        )

        verify_result = verify_same_node(
            client=page_detail_client,
            screenshot_new_b64=screenshot_b,
            screenshot_existing_b64=screenshot_a,
            existing_description=ref_desc,
            model=page_detail_model,
        )

        if verify_result.get("same", False):
            # Keep the lower-numbered node (earlier discovered)
            keep, remove = (node_a, node_b) if node_a < node_b else (node_b, node_a)
            graph.merge_nodes(keep, remove)
            logger.info(
                "Merged %s into %s — reason: %s",
                remove, keep, verify_result.get("reason", ""),
            )
            results.append({
                "issue": issue,
                "status": "merged",
                "kept": keep,
                "removed": remove,
                "reason": verify_result.get("reason", ""),
            })
        else:
            logger.info(
                "Kept separate: %s + %s — reason: %s",
                node_a, node_b, verify_result.get("reason", ""),
            )
            results.append({
                "issue": issue,
                "status": "kept_separate",
                "reason": verify_result.get("reason", ""),
            })

    return results


def re_explore_issues(
    graph: GraphManager,
    issues: list[dict],
    controller,
    app_name: str,
    package_name: str,
    graph_path: Path,
    instruction_client,
    instruction_model: str,
    action_client,
    action_model: str,
    steps_per_issue: int = 5,
) -> list[dict]:
    """Re-explore flagged issues on device.

    For retry_edge: navigates to source node, re-runs the exploration step
    with a hint to retry the specific instruction.

    For explore_node: navigates to the node and explores with hints about
    expected missing actions.
    """
    results = []
    start_node = graph.get_start_node()
    if start_node is None:
        logger.warning("Graph is empty — nothing to re-explore")
        return results

    def _relaunch_app():
        subprocess.run(
            ["adb", "-s", controller.config.get("device", controller.config).get("udid", "emulator-5554"),
             "shell", "monkey", "-p", package_name,
             "-c", "android.intent.category.LAUNCHER", "1"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
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

        # Verify we arrived
        device_state = controller.get_state()
        actual = graph.identify_state(
            device_state["activity"], device_state["screenshot"],
            app_name=app_name,
        )
        if actual != target_node:
            logger.warning(
                "Expected to reach %s but arrived at %s",
                target_node, actual,
            )
        return actual

    def _explore_steps(
        start_from: str,
        num_steps: int,
        hint: str = "",
    ) -> list[dict]:
        """Run exploration steps from a node with an optional hint for the planner."""
        current_node = start_from
        step_results = []

        for step_i in range(num_steps):
            device_state = controller.get_state()
            screenshot = device_state["screenshot"]
            node_data = graph.get_node(current_node)
            if node_data is None:
                break

            explored_edges = graph.get_all_edges_from_node(current_node)

            # Inject hint into page description for the planner
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
                    capture_output=True, text=True, timeout=3,
                )
                if "mInputShown=true" in kb_check.stdout:
                    keyboard_hint = " (Note: the soft keyboard is currently visible.)"
                    input_status = (
                        "Soft keyboard is visible; a text field is focused "
                        "and ready for typing."
                    )
            except Exception:
                pass

            instruction = plan_next_action(
                client=instruction_client,
                screenshot_b64=screenshot,
                page_description=page_desc,
                explored_edges=explored_edges,
                app_name=app_name,
                input_status=input_status,
                model=instruction_model,
            )

            action, history_entry = predict_next_action(
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

            # Check if left app
            current_package = device_state.get("package", "").strip()
            if current_package and current_package != package_name:
                logger.warning("Left app during re-explore, pressing back")
                controller.exe_action({"action": "back"})
                time.sleep(1)
                device_state = controller.get_state()
                current_node = graph.identify_state(
                    device_state["activity"], device_state["screenshot"],
                    app_name=app_name,
                )
                continue

            new_node = graph.identify_state(
                device_state["activity"], device_state["screenshot"],
                app_name=app_name,
            )

            # Build target observation
            new_node_data = graph.get_node(new_node)
            target_obs = new_node_data.get("page_description", "") if new_node_data else ""

            graph.add_edge(
                current_node, new_node,
                action=[action], instruction=instruction,
                target_observation=target_obs,
                num_steps=1,
            )

            step_results.append({
                "step": step_i + 1,
                "from": current_node,
                "to": new_node,
                "instruction": instruction,
                "action": action,
            })

            logger.info(
                "  Re-explore step %d: %s --[%s]--> %s (instruction: '%s')",
                step_i + 1, current_node,
                action.get("action", "?"), new_node, instruction,
            )

            graph.save_graph(graph_path)
            current_node = new_node

        return step_results

    # Process each issue
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
                source, target, instr,
            )

            _relaunch_app()
            actual = _navigate_to(source)
            if actual is None:
                results.append({"issue": issue, "status": "skipped", "reason": "unreachable"})
                continue

            # Get the target node's description for a vague hint
            target_desc = graph.graph.nodes[target].get("page_description", target) if target in graph.graph else target
            hint = (
                f"A previous attempt to reach '{target_desc}' from this screen may have "
                f"gone to the wrong place. Check if any elements on this screen could lead "
                f"to '{target_desc}'."
            )
            step_results = _explore_steps(actual, steps_per_issue, hint=hint)

            results.append({
                "issue": issue,
                "status": "retried",
                "navigated_to": actual,
                "steps": step_results,
            })

        elif itype == "explore_node":
            node = issue.get("node", "")
            expected = issue.get("expected_pages", [])

            if not node or node not in graph.graph:
                logger.warning("Node %s not found, skipping", node)
                results.append({"issue": issue, "status": "skipped", "reason": "node not found"})
                continue

            logger.info(
                "Exploring node: %s (expected actions: %s)",
                node, expected,
            )

            _relaunch_app()
            actual = _navigate_to(node)
            if actual is None:
                results.append({"issue": issue, "status": "skipped", "reason": "unreachable"})
                continue

            # Build vague hints about what pages might be reachable
            expected_pages = ", ".join(f"'{a}'" for a in expected)
            hint = (
                f"This screen might be able to lead to pages like {expected_pages}. "
                f"Check if any elements on this screen should be interacted with."
            )
            step_results = _explore_steps(actual, steps_per_issue, hint=hint)

            results.append({
                "issue": issue,
                "status": "explored",
                "navigated_to": actual,
                "steps": step_results,
            })

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit an explored graph for anomalies")
    parser.add_argument("--config", "-c", type=str, default="configs/explore.yaml")
    parser.add_argument(
        "--app",
        type=str,
        default=None,
        help=(
            "Basic audit only: audit one app graph folder under graph_dir by "
            "folder/app name, e.g. --app citymapper"
        ),
    )
    parser.add_argument(
        "--re-explore", action="store_true",
        help="Navigate to flagged nodes/edges on device and re-explore",
    )
    parser.add_argument(
        "--steps-per-issue", type=int, default=5,
        help="Number of exploration steps per flagged issue (default: 5)",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    setup_logging(level=logging.INFO)

    exp_config = config.get("experiment", {})
    vlm_config = config.get("vlm", {})
    graph_dir = exp_config.get("graph_dir", "graphs")
    selected_apps = select_apps_for_audit(
        config,
        Path(graph_dir),
        app_name=args.app,
        re_explore=args.re_explore,
    )

    page_detail_client, page_detail_model = make_client(vlm_config.get("page_detail"))
    embedding_client, embedding_model = make_client(vlm_config.get("embedding"))

    # Only needed for --re-explore
    controller = None
    instruction_client = instruction_model = action_client = action_model = None
    if args.re_explore:
        from aitk.utils.adb_controller import ADBController
        controller = ADBController(config, logger)
        instruction_client, instruction_model = make_client(vlm_config.get("instruction"))
        action_client, action_model = make_client(vlm_config.get("action"))

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

        graph_path = Path(graph_dir) / app_name / f"{app_name}.json"
        audited_graph_path = Path(graph_dir) / app_name / f"{app_name}_audited.json"
        if not graph_path.exists():
            logger.warning("No graph found at %s, skipping %s", graph_path, app_name)
            continue

        graph.load_graph(graph_path)
        logger.info(
            "Loaded graph for %s: %d nodes, %d edges",
            app_name, graph.graph.number_of_nodes(), graph.graph.number_of_edges(),
        )

        # Phase 1: LLM audit
        report_path = graph_path.parent / f"{app_name}_audit.json"
        audit_result = run_audit(graph, app_name, report_path)

        issues = audit_result.get("issues", [])
        if not issues:
            logger.info("No issues found for %s", app_name)
        else:
            logger.info("Found %d issues for %s", len(issues), app_name)

        # Phase 2: Verify and merge duplicate nodes (no device needed)
        merge_issues = [i for i in issues if i.get("type") == "merge_nodes"]
        other_issues = [i for i in issues if i.get("type") != "merge_nodes"]

        if merge_issues:
            logger.info("Verifying %d merge candidates...", len(merge_issues))
            merge_results = verify_and_merge_nodes(
                graph, merge_issues,
                page_detail_client, page_detail_model,
            )
            # Save merge results
            merge_path = graph_path.parent / f"{app_name}_merge.json"
            with open(merge_path, "w", encoding="utf-8") as f:
                json.dump(merge_results, f, indent=2, ensure_ascii=False)
            logger.info("Merge results saved to %s", merge_path)

            # Save updated graph after merges to a NEW file (preserve original)
            merged_count = sum(1 for r in merge_results if r["status"] == "merged")
            if merged_count:
                graph.save_graph(audited_graph_path)
                logger.info("Audited graph saved to %s (%d merge(s))", audited_graph_path, merged_count)

        # Phase 3: Re-explore (optional, requires device)
        if args.re_explore and controller and other_issues:
            logger.info("Starting re-exploration for %d issues...", len(other_issues))

            # Launch the app
            launch_app(config, app)
            time.sleep(4)

            re_results = re_explore_issues(
                graph, other_issues, controller,
                app_name, package_name, audited_graph_path,
                instruction_client, instruction_model,
                action_client, action_model,
                steps_per_issue=args.steps_per_issue,
            )

            # Save results
            results_path = graph_path.parent / f"{app_name}_re_explore.json"
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(re_results, f, indent=2, ensure_ascii=False)
            logger.info("Re-exploration results saved to %s", results_path)

        # Phase 4: Final edge-template normalization.
        # This runs after audit merge/re-explore modifications and also runs
        # when the auditor reports no structural issues.
        normalize_and_save_audited_graph(graph, audited_graph_path)

    token_tracker.print_summary()
    logger.info("Audit complete.")
