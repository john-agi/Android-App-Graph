"""UI-KOBE: Knowledge Ontology Builder and Explorer for mobile GUI Agents."""

from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING

from ui_kobe.utils import make_client
from ui_kobe.utils.graph_manager import GraphManager
from ui_kobe.utils.vlm_utils import (
    plan_next_action,
    predict_next_action,
    select_exploration_target,
    token_tracker,
)

if TYPE_CHECKING:
    from aitk.utils.adb_controller import ADBController


def _compute_schema_delta(before: dict, after: dict) -> dict | None:
    """Compute the schema delta between two detail snapshots.

    Returns a dict of changed keys: {"key": {"before": old, "after": new}}
    Returns None if nothing changed.
    """
    delta = {}
    all_keys = set(before.keys()) | set(after.keys())
    for key in all_keys:
        val_before = before.get(key)
        val_after = after.get(key)
        if val_before != val_after:
            delta[key] = {"before": val_before, "after": val_after}
    return delta if delta else None


class Kobe:
    """Explores an Android app and builds a state-transition graph.

    Each node is a unique app state identified by:
      - Activity name (from Android)
      - Page description (from VLM)
      - Detailed screen contents JSON (from VLM)

    Each edge is an action (in AITK format) that transitions between states.
    """

    def __init__(
        self,
        controller: ADBController,
        app_name: str,
        package_name: str,
        logger: logging.Logger,
        vlm_config: dict | None = None,
        graph_dir: str | Path | None = None,
        max_steps: int = 20,
        coverage_checkpoint_steps: int = 50,
        coverage_checkpoint_top_k: int = 15,
    ):
        self.controller = controller
        self.app_name = app_name
        self.package_name = package_name
        logger_name = getattr(logger, "name", "")
        if logger_name.startswith("ui_kobe"):
            self.logger = logging.getLogger("ui_kobe.kobe")
        else:
            self.logger = logging.getLogger(
                f"{logger_name}.kobe" if logger_name else "ui_kobe.kobe"
            )
        self.max_steps = max_steps
        self.coverage_checkpoint_steps = coverage_checkpoint_steps
        self.coverage_checkpoint_top_k = coverage_checkpoint_top_k

        vlm_config = vlm_config or {}

        # Create per-call clients
        self.page_detail_client, self.page_detail_model = make_client(
            vlm_config.get("page_detail")
        )
        self.embedding_client, self.embedding_model = make_client(
            vlm_config.get("embedding")
        )
        self.instruction_client, self.instruction_model = make_client(
            vlm_config.get("instruction")
        )
        self.action_client, self.action_model = make_client(vlm_config.get("action"))

        similarity_threshold = vlm_config.get("similarity_threshold", 0.85)

        # Graph manager with embedding-based state matching
        self.graph = GraphManager(
            page_detail_client=self.page_detail_client,
            page_detail_model=self.page_detail_model,
            embedding_client=self.embedding_client,
            embedding_model=self.embedding_model,
            similarity_threshold=similarity_threshold,
        )

        # Graph persistence path: graphs/{app_name}/{app_name}.json
        base_graph_dir = (
            Path(graph_dir) if graph_dir is not None else Path.cwd() / "graphs"
        )
        self.graph_path = base_graph_dir / app_name / f"{app_name}.json"

        if self.graph_path.exists():
            self.graph.load_graph(self.graph_path)
            self.logger.info("Graph loaded from %s", self.graph_path)
        else:
            self.logger.info(
                "No existing graph at %s. Starting fresh.", self.graph_path
            )

    def _maybe_continue_from_coverage_checkpoint(
        self,
        step: int,
        max_steps: int,
        current_node: str,
    ) -> str | None:
        """Periodically jump to an under-explored reachable node and continue."""
        if self.coverage_checkpoint_steps <= 0:
            return None
        if step <= 0 or step >= max_steps:
            return None
        if step % self.coverage_checkpoint_steps != 0:
            return None

        self.logger.info(
            "Coverage checkpoint at step %d: evaluating under-explored continuation targets.",
            step,
        )
        merge_result = self.graph.run_node_merge_audit(app_name=self.app_name)
        merged_count = merge_result.get("merged_count", 0)
        if merged_count:
            self.logger.info(
                "Coverage checkpoint at step %d: merged %d duplicate node(s).",
                step,
                merged_count,
            )
            self.save_graph()

            if current_node not in self.graph.graph:
                current_node = self.graph.get_start_node() or current_node

        candidates = self.graph.get_exploration_target_candidates(
            package_name=self.package_name,
            top_k=self.coverage_checkpoint_top_k,
        )
        candidates = [c for c in candidates if c.get("node_id") != current_node]
        if not candidates:
            self.logger.info(
                "Coverage checkpoint at step %d: no alternate under-explored candidates.",
                step,
            )
            return None

        selected = select_exploration_target(
            self.page_detail_client,
            app_name=self.app_name,
            candidates=candidates,
            model=self.page_detail_model,
        )
        if not selected:
            selected = candidates[0]["node_id"]
            self.logger.info(
                "Coverage checkpoint at step %d: model did not select a valid node; using top candidate %s.",
                step,
                selected,
            )

        if selected == current_node:
            return None

        self.logger.info(
            "Coverage checkpoint at step %d: continuing exploration from %s.",
            step,
            selected,
        )
        try:
            actual_node = self.navigate_to_node(selected)
        except Exception as exc:
            self.logger.warning(
                "Coverage checkpoint navigation to %s failed: %s",
                selected,
                exc,
            )
            return None
        return actual_node

    def _apply_coverage_checkpoint_after_step(
        self,
        step: int,
        max_steps: int,
        current_node: str,
        device_state: dict,
    ) -> tuple[str, dict]:
        """Apply periodic coverage continuation after any completed step."""
        checkpoint_node = self._maybe_continue_from_coverage_checkpoint(
            step,
            max_steps,
            current_node,
        )
        if not checkpoint_node:
            return current_node, device_state

        return checkpoint_node, self.controller.get_state()

    def navigate_to_node(self, target_node: str) -> str:
        """Navigate to a target node by replaying saved edges from the start node.

        This executes the shortest path of already-known actions from the
        app's entry point to the target node — no VLM calls needed.

        Args:
            target_node: The node ID to navigate to.

        Returns:
            The node ID actually reached (verified via identify_state).
        """
        start_node = self.graph.get_start_node()
        if start_node is None:
            raise ValueError("Graph is empty — cannot navigate.")

        path = self.graph.find_path(start_node, target_node)
        if path is None:
            raise ValueError(
                f"No path from {start_node} to {target_node} in the graph."
            )

        if len(path) <= 1:
            self.logger.info("Already at target node %s", target_node)
            device_state = self.controller.get_state()
            return self.graph.identify_state(
                device_state["activity"], device_state["screenshot"],
                app_name=self.app_name,
            )

        self.logger.info(
            "Replaying %d actions to reach node %s ...",
            len(path) - 1,
            target_node,
        )

        for i, (node_id, action) in enumerate(path):
            if i == 0:
                continue  # skip the start node (no action)

            # action can be a single dict or a list of dicts (compound)
            if isinstance(action, list):
                self.logger.info(
                    "  Replay %d/%d: %s --> %s via %d compound actions",
                    i,
                    len(path) - 1,
                    path[i - 1][0],
                    node_id,
                    len(action),
                )
                for sub_action in action:
                    self.controller.exe_action(sub_action)
                    time.sleep(0.5)
            else:
                self.logger.info(
                    "  Replay %d/%d: %s --> %s via %s",
                    i,
                    len(path) - 1,
                    path[i - 1][0],
                    node_id,
                    action.get("action", "?"),
                )
                self.controller.exe_action(action)
            time.sleep(1)  # brief pause for the UI to settle

        # Verify we reached the expected state
        device_state = self.controller.get_state()
        actual_node = self.graph.identify_state(
            device_state["activity"], device_state["screenshot"],
            app_name=self.app_name,
        )
        if actual_node != target_node:
            self.logger.warning(
                "Expected to reach %s but arrived at %s. Continuing from %s.",
                target_node,
                actual_node,
                actual_node,
            )
        else:
            self.logger.info("Successfully reached target node %s", target_node)

        return actual_node

    def explore(
        self,
        max_steps: int | None = None,
        resume_from: str | None = None,
    ) -> None:
        """Run the exploration loop for up to max_steps total actions.

        Args:
            max_steps: Maximum total exploration steps. If the graph already
                has some steps completed, exploration continues from there.
            resume_from: Node ID to resume from. If provided, the explorer
                replays saved edges to reach this node first (no VLM calls),
                then continues exploring from there.
                Special values:
                  - "auto": picks the least-explored node automatically.

        The graph is auto-saved after every step so crashes lose at most one step.
        """
        max_steps = max_steps if max_steps is not None else self.max_steps
        start_step = self.graph.total_steps_completed

        if start_step >= max_steps:
            self.logger.info(
                "Already completed %d/%d steps. Nothing to do. "
                "Running final graph normalization; increase max_steps to continue exploring.",
                start_step,
                max_steps,
            )
            self.graph.normalize_all_edges()
            self.save_graph()
            return

        if resume_from is not None and self.graph.graph.number_of_nodes() > 0:
            # Resolve "auto" to least-explored node
            if resume_from == "auto":
                resume_from = self.graph.get_least_explored_node(package_name=self.package_name)
                self.logger.info("Auto-selected least-explored node: %s", resume_from)

            if resume_from and resume_from in self.graph.graph:
                current_node = self.navigate_to_node(resume_from)
                device_state = self.controller.get_state()
            else:
                self.logger.warning(
                    "Node %s not found in graph. Starting from current screen.",
                    resume_from,
                )
                device_state = self.controller.get_state()
                current_node = self.graph.identify_state(
                    device_state["activity"], device_state["screenshot"],
                    app_name=self.app_name,
                )
        elif start_step > 0:
            self.logger.info(
                "Resuming exploration from step %d (target: %d steps total)",
                start_step,
                max_steps,
            )
            device_state = self.controller.get_state()
            current_node = self.graph.identify_state(
                device_state["activity"], device_state["screenshot"],
                app_name=self.app_name,
            )
        else:
            device_state = self.controller.get_state()
            current_node = self.graph.identify_state(
                device_state["activity"], device_state["screenshot"],
                app_name=self.app_name,
            )

        step = start_step


        try:
            while step < max_steps:
                # Snapshot token counts before this step
                tokens_before = token_tracker.snapshot_by_type()

                screenshot = device_state["screenshot"]
                node_data = self.graph.get_node(current_node)

                # Capture schema snapshot before action (for delta computation)
                detail_before = dict(node_data.get("last_detail_snapshot", {}))

                # Get edges already explored from this node (with instructions)
                explored_edges = self.graph.get_all_edges_from_node(current_node)

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
                        keyboard_hint = " (Note: the soft keyboard is currently visible — a text field is focused and ready for typing.)"
                        input_status = (
                            "Soft keyboard is visible; a text field is focused "
                            "and ready for typing."
                        )
                except Exception:
                    pass

                # Step 1: Plan — decide WHAT to do (natural language)
                unexplored_elements = self.graph.get_unexplored_elements(current_node)
                instruction = plan_next_action(
                    client=self.instruction_client,
                    screenshot_b64=screenshot,
                    page_description=node_data["page_description"],
                    explored_edges=explored_edges,
                    app_name=self.app_name,
                    unexplored_elements=unexplored_elements,
                    input_status=input_status,
                    model=self.instruction_model,
                )

                # Step 2: Action agent executes ONE action per exploration step
                left_app = False

                action, history_entry = predict_next_action(
                    client=self.action_client,
                    screenshot_b64=screenshot,
                    instruction=instruction + keyboard_hint,
                    screen_w=self.controller.w,
                    screen_h=self.controller.h,
                    action_history=[],
                    model=self.action_model,
                )

                # Agent says "end" — no action to take
                if action.get("action") == "end":
                    self.logger.info(
                        '  Action agent said end for: "%s", recording as back',
                        instruction,
                    )
                    action = {"action": "back"}
                    self.controller.exe_action(action)
                elif action.get("action") == "wait":
                    self.logger.info("  Action agent requested wait")
                    time.sleep(1)
                    device_state = self.controller.get_state()
                    continue
                else:
                    self.controller.exe_action(action)

                action_sequence = [action]

                # Get final device state
                time.sleep(0.5)
                device_state = self.controller.get_state()

                # Check if we left the app
                if not left_app:
                    current_package = device_state.get("package", "").strip()
                    if current_package and current_package != self.package_name:
                        left_app = True

                # Handle external app
                if left_app:
                    new_node = self._handle_external_app(
                        current_node, action_sequence, current_package, device_state,
                        instruction=instruction,
                    )
                    device_state = self.controller.get_state()
                    step += 1
                    self.graph.total_steps_completed = step
                    self.graph.maybe_normalize_node_edges(current_node)
                    self.save_graph()
                    current_node = self.graph.identify_state(
                        device_state["activity"], device_state["screenshot"],
                        app_name=self.app_name,
                    )
                    current_node, device_state = self._apply_coverage_checkpoint_after_step(
                        step,
                        max_steps,
                        current_node,
                        device_state,
                    )
                    continue

                # Identify the new state node
                new_node = self.graph.identify_state(
                    device_state["activity"], device_state["screenshot"],
                    app_name=self.app_name,
                )

                # Check for in-app WebView showing external content
                new_node_data = self.graph.get_node(new_node)
                if self._is_external_web(
                    device_state.get("activity", ""),
                    new_node_data.get("page_description", ""),
                ):
                    self.logger.warning(
                        "In-app external web detected at node %s. Pressing back.",
                        new_node,
                    )
                    self.controller.exe_action({"action": "back"})
                    time.sleep(1)
                    device_state = self.controller.get_state()
                    step += 1
                    self.graph.total_steps_completed = step
                    self.save_graph()
                    current_node = self.graph.identify_state(
                        device_state["activity"], device_state["screenshot"],
                        app_name=self.app_name,
                    )
                    current_node, device_state = self._apply_coverage_checkpoint_after_step(
                        step,
                        max_steps,
                        current_node,
                        device_state,
                    )
                    continue

                # Compute schema delta for self-loop edges
                schema_delta = None
                if new_node == current_node:
                    new_data = self.graph.get_node(new_node)
                    detail_after = new_data.get("last_detail_snapshot", {})
                    schema_delta = _compute_schema_delta(detail_before, detail_after)
                    if schema_delta:
                        self.logger.info(
                            "  Schema delta: %s",
                            {k: v for k, v in schema_delta.items()},
                        )

                # Build target observation from the post-action state
                target_obs = new_node_data.get("page_description", "")
                post_detail = new_node_data.get("last_detail_snapshot", {})
                if post_detail:
                    detail_str = ", ".join(
                        f"{k}={v}" for k, v in post_detail.items() if v is not None
                    )
                    if detail_str:
                        target_obs = f"{target_obs} ({detail_str})"

                # Record the transition — store the full action sequence as the edge
                self.graph.add_edge(
                    current_node,
                    new_node,
                    action=action_sequence,
                    instruction=instruction,
                    target_observation=target_obs,
                    schema_delta=schema_delta,
                    num_steps=len(action_sequence),
                )
                # Mark the element that was interacted with as explored
                self.graph.mark_element_explored(current_node, instruction)
                self.graph.maybe_normalize_node_edges(current_node)
                step += 1
                self.graph.total_steps_completed = step

                # Per-step token usage by agent
                tokens_after = token_tracker.snapshot_by_type()
                agent_names = [
                    "page_describe_and_state",
                    "node_verify",
                    "embedding",
                    "instruction",
                    "action",
                ]
                step_tokens = {
                    a: tokens_after.get(a, 0) - tokens_before.get(a, 0)
                    for a in agent_names
                }
                step_total = sum(step_tokens.values())
                cumulative = sum(tokens_after.values())

                token_parts = " | ".join(
                    f"{a}={step_tokens[a]}" for a in agent_names if step_tokens[a] > 0
                )

                action_summary = "+".join(a.get("action", "?") for a in action_sequence)
                total_actions = sum(
                    len(self.graph.graph[s][t].get("actions", []))
                    for s, t in self.graph.graph.edges()
                )
                self.logger.info(
                    "Step %d/%d: %s --[%s]--> %s | graph: %d nodes, %d edges, %d total actions",
                    step,
                    max_steps,
                    current_node,
                    action_summary,
                    new_node,
                    self.graph.graph.number_of_nodes(),
                    self.graph.graph.number_of_edges(),
                    total_actions,
                )
                self.logger.info(
                    "  Tokens this step: %s | step_total=%d, cumulative=%d",
                    token_parts or "0",
                    step_total,
                    cumulative,
                )

                # Auto-save after every step
                self.save_graph()

                current_node, device_state = self._apply_coverage_checkpoint_after_step(
                    step,
                    max_steps,
                    new_node,
                    device_state,
                )
        except KeyboardInterrupt:
            self.logger.warning(
                "Exploration interrupted at step %d. Saving graph before exit.",
                step,
            )
            self.save_graph()
            token_tracker.print_summary()
            raise
        except Exception:
            self.logger.error(
                "Exploration crashed at step %d. Graph saved. Re-run to resume.",
                step,
            )
            self.save_graph()
            token_tracker.print_summary()
            raise

        self.logger.info("Running final graph normalization...")
        self.graph.normalize_all_edges()
        self.save_graph()
        self.logger.info(
            "Exploration finished after %d steps. Graph: %d nodes, %d edges.",
            step,
            self.graph.graph.number_of_nodes(),
            self.graph.graph.number_of_edges(),
        )
        token_tracker.print_summary()

    # Activity names that indicate an in-app browser / WebView.
    _WEBVIEW_ACTIVITY_PATTERNS = (
        "webview",
        "customtab",
        "browser",
        "chromeclient",
        "inappbrowser",
    )

    # Keywords in the page description that indicate external web content.
    _WEB_DESCRIPTION_KEYWORDS = (
        "web page",
        "webpage",
        "website",
        "browser",
        "external link",
        "url bar",
        "address bar",
        "loading a web",
        "cookie consent",
        "cookie policy",
        "privacy policy",
        "terms of service",
        "terms and conditions",
    )

    def _is_external_web(self, activity: str, page_description: str) -> bool:
        """Detect if the current screen is an in-app WebView showing external content.

        Uses two signals:
        1. Activity name contains WebView/browser patterns.
        2. Page description mentions web/browser keywords.
        """
        activity_lower = activity.lower()
        for pattern in self._WEBVIEW_ACTIVITY_PATTERNS:
            if pattern in activity_lower:
                self.logger.info(
                    "External web detected via activity: %s", activity,
                )
                return True

        desc_lower = page_description.lower()
        for keyword in self._WEB_DESCRIPTION_KEYWORDS:
            if keyword in desc_lower:
                self.logger.info(
                    'External web detected via description keyword "%s": %s',
                    keyword,
                    page_description,
                )
                return True

        return False

    # Packages that are system overlays (keyboard, permissions, system UI).
    # These sit on top of the app but don't represent real navigation away.
    _SYSTEM_OVERLAY_PACKAGES = frozenset(
        {
            # Input methods / keyboards
            "com.google.android.inputmethod.latin",  # Gboard
            "com.android.inputmethod.latin",  # AOSP keyboard
            "com.samsung.android.honeyboard",  # Samsung keyboard
            "com.swiftkey.swiftkey",  # SwiftKey
            "com.touchtype.swiftkey",  # SwiftKey (alt)
            # System UI / overlays
            "com.android.systemui",  # Status bar, notifications
            "com.android.permissioncontroller",  # Permission dialogs
            "com.google.android.permissioncontroller",  # Permission dialogs (Google)
            "com.android.packageinstaller",  # Install prompts
            "android",  # System dialogs
        }
    )

    def _is_system_overlay(self, package: str) -> bool:
        """Check if a package is a system overlay (keyboard, dialog, etc.)."""
        if package in self._SYSTEM_OVERLAY_PACKAGES:
            return True
        # Heuristic: input method packages usually contain "inputmethod" or "keyboard"
        lower = package.lower()
        if "inputmethod" in lower or "keyboard" in lower or "ime" in lower:
            return True
        return False

    def _handle_external_app(
        self,
        source_node: str,
        action: dict | list[dict],
        external_package: str,
        device_state: dict,
        instruction: str | None = None,
    ) -> str:
        """Handle navigation to an external app.

        Creates a special node for the external app (no VLM calls), records the
        edge, presses back, and if that doesn't return to our app, relaunches it.

        Returns:
            The external app node ID.
        """
        ext_activity = device_state.get("activity", "unknown")
        ext_node_id = f"ext_{external_package}"

        self.logger.warning(
            "Left target app! Current package: %s (expected: %s). "
            "Creating external node and pressing back.",
            external_package,
            self.package_name,
        )

        # Create external app node if it doesn't exist (no VLM calls)
        if ext_node_id not in self.graph.graph:
            self.graph.graph.add_node(
                ext_node_id,
                activity=ext_activity,
                page_description=f"[external app: {external_package}]",
                state_schema={},
                visit_count=0,
                is_external=True,
            )

        self.graph.graph.nodes[ext_node_id]["visit_count"] = (
            self.graph.graph.nodes[ext_node_id].get("visit_count", 0) + 1
        )

        # Record the edge that led to the external app
        self.graph.add_edge(
            source_node, ext_node_id, action=action, instruction=instruction,
            target_observation=f"[left app to {external_package}]",
        )

        if isinstance(action, list):
            action_summary = "+".join(a.get("action", "?") for a in action)
        else:
            action_summary = action.get("action", "?")
        self.logger.info(
            "Step: %s --[%s]--> %s (external)",
            source_node,
            action_summary,
            ext_node_id,
        )

        # Try pressing back
        self.controller.exe_action({"action": "back"})
        time.sleep(1)

        # Check if we're back in the correct app
        state = self.controller.get_state()
        if state.get("package", "") != self.package_name:
            self.logger.warning(
                "Back button didn't return to %s (now at %s). Relaunching...",
                self.package_name,
                state.get("package", ""),
            )
            ctrl_config = getattr(self.controller, "config", {})
            udid = ctrl_config.get("device", ctrl_config).get("udid", "emulator-5554")
            subprocess.run(
                [
                    "adb",
                    "-s",
                    udid,
                    "shell",
                    "monkey",
                    "-p",
                    self.package_name,
                    "-c",
                    "android.intent.category.LAUNCHER",
                    "1",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(3)

        return ext_node_id

    def save_graph(self) -> None:
        """Persist the graph to disk."""
        self.graph.save_graph(self.graph_path)
