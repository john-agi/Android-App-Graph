"""Android-App-Graph: Knowledge Ontology Builder and Explorer for mobile GUI Agents."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from openrouter import OpenRouter

from android_app_graph.device import (
    AppLaunch,
    DeviceController,
    ForceStopRefusedError,
    is_outside_app,
    keyboard_status,
    restart_app,
    restart_into_app,
    return_to_app,
    soft_keyboard_hint,
)
from android_app_graph.graph_files import app_graph_path, graph_steps_dir
from android_app_graph.step_log import Observation, StepRecorder, Transition
from android_app_graph.utils import resolve_audit_timeout, resolve_model_settings
from android_app_graph.utils.graph_manager import (
    EXTERNAL_WEB_BACK_PRESSES,
    EXTERNAL_WEB_NODE,
    EdgeDetails,
    GraphManager,
)
from android_app_graph.utils.vlm_utils import (
    ActionRequest,
    PlanRequest,
    plan_next_action,
    predict_next_action,
    select_exploration_target,
    token_tracker,
)

EXPLORATION_ROLES = ("page_detail", "embedding", "image_embedding", "instruction", "action")
_STEP_AGENT_NAMES = (
    "image_embedding",
    "page_describe_and_state",
    "node_verify",
    "embedding",
    "instruction",
    "action",
)


class NoPathError(ValueError):
    """The graph records no path from its start node to the target, so no replay began."""


def _compute_schema_delta(
    before: dict[str, Any], after: dict[str, Any]
) -> dict[str, dict[str, Any]] | None:
    """Compute the schema delta between two detail snapshots.

    Returns a dict of changed keys: {"key": {"before": old, "after": new}}
    Returns None if nothing changed.
    """
    delta: dict[str, dict[str, Any]] = {}
    all_keys = set(before.keys()) | set(after.keys())
    for key in all_keys:
        val_before = before.get(key)
        val_after = after.get(key)
        if val_before != val_after:
            delta[key] = {"before": val_before, "after": val_after}
    return delta or None


def _target_observation(page_description: str, detail_snapshot: dict[str, Any]) -> str:
    """Return ``"<description> (key=value, ...)"`` for the details that have a value, or
    the bare description when none has."""
    details = (
        ", ".join(f"{k}={v}" for k, v in detail_snapshot.items() if v is not None)
        if detail_snapshot
        else ""
    )
    return f"{page_description} ({details})" if details else page_description


@dataclass(frozen=True, slots=True)
class _Position:
    """Where the next exploration step starts."""

    node: str
    device_state: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _Move:
    """One planned action, and what the step read before it that recording it needs.

    ``instruction`` is what the edge records; ``planned_instruction`` is what the planner
    wrote, which differs when an ``end`` is recorded as a back press.
    """

    source: str
    before: Observation
    action: dict[str, Any]
    instruction: str
    planned_instruction: str
    planned_element: int | None
    detail_before: dict[str, Any]
    tokens_before: dict[str, int]

    @property
    def action_sequence(self) -> list[dict[str, Any]]:
        """An edge records each of its actions as a sequence, which can be compound."""
        return [self.action]


@dataclass(frozen=True, kw_only=True)
class ExplorerConfig:
    """The app ``Kobe`` explores, its model settings, where its graph is saved and its budgets."""

    app_name: str
    package_name: str
    vlm_config: dict[str, Any] | None = None
    graph_dir: str | Path | None = None
    max_steps: int = 20
    coverage_checkpoint_steps: int = 50
    coverage_checkpoint_top_k: int = 15
    launch_activity: str | None = None


class Kobe:
    """Explores an Android app and builds a state-transition graph.

    Each node is a unique app state identified by:
      - Activity name (from Android)
      - Page description (from VLM)
      - Detailed screen contents JSON (from VLM)

    Each edge is a device action that transitions between states.
    """

    def __init__(
        self,
        controller: DeviceController,
        config: ExplorerConfig,
        *,
        client: OpenRouter,
        logger: logging.Logger,
    ) -> None:
        self.app_name = config.app_name
        self.package_name = config.package_name
        self.app_launch = AppLaunch(config.app_name, config.package_name, config.launch_activity)
        logger_name = getattr(logger, "name", "")
        if logger_name.startswith("android_app_graph"):
            self.logger = logging.getLogger("android_app_graph.kobe")
        else:
            self.logger = logging.getLogger(
                f"{logger_name}.kobe" if logger_name else "android_app_graph.kobe"
            )
        self.max_steps = config.max_steps
        self.coverage_checkpoint_steps = config.coverage_checkpoint_steps
        self.coverage_checkpoint_top_k = config.coverage_checkpoint_top_k

        vlm_config = config.vlm_config or {}
        settings = resolve_model_settings(vlm_config, roles=EXPLORATION_ROLES)
        self.client = client
        self.page_detail = settings["page_detail"]
        self.instruction = settings["instruction"]
        self.action = settings["action"]

        graph_dir = config.graph_dir
        base_graph_dir = Path(graph_dir) if graph_dir is not None else Path.cwd() / "graphs"
        self.graph_path = app_graph_path(base_graph_dir, self.app_name)
        steps_dir = graph_steps_dir(self.graph_path)

        self.graph = GraphManager(
            client=client,
            page_detail=self.page_detail,
            embedding=settings["embedding"],
            image_embedding=settings["image_embedding"],
            audit_timeout=resolve_audit_timeout(vlm_config),
        )
        self.graph.steps_dir = steps_dir

        if self.graph_path.exists():
            self.graph.load_graph(self.graph_path, default_package_name=self.package_name)
            self.logger.info("Graph loaded from %s", self.graph_path)
            self.graph.embed_missing_members()
        else:
            self.logger.info("No existing graph at %s. Starting fresh.", self.graph_path)
        self.graph.package_name = self.package_name
        self.controller = StepRecorder(
            controller, steps_dir, run_kind="explore", source="main_loop"
        )

    def maybe_continue_from_coverage_checkpoint(
        self,
        step: int,
        max_steps: int,
        current_node: str,
    ) -> str | None:
        """Periodically jump to an under-explored reachable node and continue."""
        if not self._is_coverage_checkpoint(step, max_steps):
            return None

        self.logger.info(
            "Coverage checkpoint at step %d: evaluating under-explored continuation targets.",
            step,
        )
        current_node = self._merge_duplicate_nodes(step, current_node)
        selected = self._choose_continuation_target(current_node)
        if selected is None:
            self.logger.info(
                "Coverage checkpoint at step %d: no alternate under-explored candidates.",
                step,
            )
            return None

        if selected == current_node:
            return None

        self.logger.info(
            "Coverage checkpoint at step %d: continuing exploration from %s.",
            step,
            selected,
        )
        return self._navigate_to_continuation(selected, current_node)

    def _is_coverage_checkpoint(self, step: int, max_steps: int) -> bool:
        return (
            self.coverage_checkpoint_steps > 0
            and 0 < step < max_steps
            and step % self.coverage_checkpoint_steps == 0
        )

    def _merge_duplicate_nodes(self, step: int, current_node: str) -> str:
        """Run the checkpoint's merge audit and return the ID ``current_node`` has after it."""
        merge_result = self.graph.run_node_merge_audit(app_name=self.app_name)
        merged_count = merge_result.get("merged_count", 0)
        if merged_count:
            self.logger.info(
                "Coverage checkpoint at step %d: merged %d duplicate node(s).",
                step,
                merged_count,
            )
            self.save_graph()
            current_node = self.graph.resolve_node_id(current_node)
        return current_node

    def _choose_continuation_target(self, current_node: str) -> str | None:
        """Return the node the page-detail model chooses among the ranked candidates other
        than ``current_node``, the top-ranked one when it chooses none, or ``None`` when
        no other node is ranked."""
        candidates = self._continuation_candidates(current_node)
        if not candidates:
            return None
        selected = select_exploration_target(
            self.client,
            app_name=self.app_name,
            candidates=candidates,
            settings=self.page_detail,
        )
        if selected:
            return selected
        top_candidate = str(candidates[0]["node_id"])
        self.logger.info(
            "The model did not select a valid continuation target; using top candidate %s.",
            top_candidate,
        )
        return top_candidate

    def _continuation_candidates(self, current_node: str) -> list[dict[str, Any]]:
        candidates = self.graph.get_exploration_target_candidates(
            package_name=self.package_name,
            top_k=self.coverage_checkpoint_top_k,
            always_offer_start=True,
        )
        return [c for c in candidates if c.get("node_id") != current_node]

    def _navigate_to_continuation(self, selected: str, current_node: str) -> str | None:
        """Return the node exploration continues from after navigating to ``selected``, or
        ``None`` when the navigation failed before the app restart and the device is still
        on ``current_node``."""
        try:
            actual_node = self.navigate_to_node(selected)
        except (NoPathError, ForceStopRefusedError) as exc:
            self.logger.warning(
                "Navigation to continuation target %s failed before the app restart; "
                "continuing from %s.",
                selected,
                current_node,
                exc_info=exc,
            )
            return None
        except Exception as exc:
            self.logger.warning(
                "Navigation to continuation target %s failed after the app restart; "
                "continuing from the screen the device is on.",
                selected,
                exc_info=exc,
            )
            return self._identify_screen_after_failed_navigation()
        return actual_node

    def _identify_screen_after_failed_navigation(self) -> str:
        """Return the node of the screen a navigation left when it failed after the restart.

        A screen of another app, such as the launcher a failed launch leaves, gets
        one more restart first, and an outside web page is left with back presses,
        so exploration continues inside the app.
        """
        node, _ = self._identify(self.controller.get_state())
        if node == EXTERNAL_WEB_NODE:
            node, _ = self._leave_external_web(expected_node=None)
        return node

    def _require_node(self, node_id: str) -> dict[str, Any]:
        """Return a node's attributes.

        ``identify_state`` either matches an existing node or creates one, so a
        missing node means the graph and the exploration state have diverged.
        """
        node_data = self.graph.get_node(node_id)
        if node_data is None:
            msg = f"Node {node_id!r} is missing from the graph"
            raise RuntimeError(msg)
        return node_data

    def _apply_coverage_checkpoint_after_step(
        self,
        step: int,
        max_steps: int,
        current_node: str,
        device_state: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        """Apply periodic coverage continuation after any completed step."""
        with self.controller.source("coverage_checkpoint"):
            checkpoint_node = self.maybe_continue_from_coverage_checkpoint(
                step,
                max_steps,
                current_node,
            )
            if not checkpoint_node:
                # The checkpoint's merge audit can merge the caller's node away without navigating.
                return self.graph.resolve_node_id(current_node), device_state

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
        with self.controller.source("navigation_replay"):
            return self._replay_to(target_node)

    def _replay_to(self, target_node: str) -> str:
        start_node = self.graph.get_start_node()
        if start_node is None:
            msg = "Graph is empty — cannot navigate."
            raise NoPathError(msg)

        path = self.graph.find_path(start_node, target_node)
        if path is None:
            msg = f"No path from {start_node} to {target_node} in the graph."
            raise NoPathError(msg)

        restart_app(self.controller, self.app_launch, self.logger)

        if len(path) <= 1:
            self.logger.info("Already at target node %s", target_node)
            node, _ = self._identify(self.controller.get_state(), expected_node=target_node)
            return node

        self.logger.info(
            "Replaying %d actions to reach node %s ...",
            len(path) - 1,
            target_node,
        )

        for i, (node_id, action) in enumerate(path):
            if i == 0:
                continue  # the first entry is the start node and carries no action

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
                # find_path yields {} for an edge a loaded graph file holds without actions.
                if action:
                    self.controller.exe_action(action)
            time.sleep(1)  # brief pause for the UI to settle

        actual_node, _ = self._identify(self.controller.get_state(), expected_node=target_node)
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
                  - "auto": the top-ranked exploration target, chosen without a model call.

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

        position = self._starting_position(resume_from, start_step, max_steps)

        try:
            while self.graph.total_steps_completed < max_steps:
                position = self._explore_step(position, max_steps)
        except KeyboardInterrupt:
            self.logger.warning(
                "Exploration interrupted at step %d. Saving graph before exit.",
                self.graph.total_steps_completed,
            )
            self.save_graph()
            token_tracker.print_summary()
            raise
        except Exception:
            self.logger.exception(
                "Exploration crashed at step %d. Graph saved. Re-run to resume.",
                self.graph.total_steps_completed,
            )
            self.save_graph()
            token_tracker.print_summary()
            raise

        self.logger.info("Running final graph normalization...")
        self.graph.normalize_all_edges()
        self.save_graph()
        self.logger.info(
            "Exploration finished after %d steps. Graph: %d nodes, %d edges.",
            self.graph.total_steps_completed,
            self.graph.graph.number_of_nodes(),
            self.graph.graph.number_of_edges(),
        )
        token_tracker.print_summary()

    def _starting_position(
        self, resume_from: str | None, start_step: int, max_steps: int
    ) -> _Position:
        if resume_from is not None and self.graph.graph.number_of_nodes() > 0:
            if resume_from == "auto":
                resume_from = self.graph.get_resume_target(package_name=self.package_name)
                self.logger.info("Auto-selected resume target: %s", resume_from)

            if resume_from and resume_from in self.graph.graph:
                node = self.navigate_to_node(resume_from)
                return _Position(node, self.controller.get_state())
            self.logger.warning(
                "Node %s not found in graph. Starting from current screen.",
                resume_from,
            )
        elif start_step > 0:
            self.logger.info(
                "Resuming exploration from step %d (target: %d steps total)",
                start_step,
                max_steps,
            )
        node, device_state = self._identify(self.controller.get_state())
        return _Position(node, device_state)

    def _explore_step(self, position: _Position, max_steps: int) -> _Position:
        """Take one step from ``position``, record it, and return where the next one starts."""
        tokens_before = token_tracker.snapshot_by_type()
        move = self._plan_move(position, tokens_before)
        if move is None:
            time.sleep(1)
            device_state = self.controller.get_state()
            self.graph.total_steps_completed += 1
            self.save_graph()
            return _Position(position.node, device_state)

        if move.action.get("action") == "back" and self._back_leaves_app(position.node):
            continued = self._continue_from_ranked_target(position, max_steps)
            if continued is not None:
                return continued
        self.controller.exe_action(move.action)
        time.sleep(0.5)
        device_state = self.controller.get_state()
        package = device_state.get("package", "").strip()
        if is_outside_app(package, self.package_name):
            return self._record_external_app_step(move, device_state, max_steps)

        new_node, device_state = self._identify(device_state)
        move = replace(move, source=self.graph.resolve_node_id(move.source))
        new_node_data = self._require_node(new_node)
        if new_node == EXTERNAL_WEB_NODE:
            return self._record_external_web_step(move, new_node_data, device_state, max_steps)
        return self._record_in_app_step(move, new_node, device_state, max_steps)

    def _back_leaves_app(self, node: str) -> bool:
        """Return whether Back on ``node`` is taken to leave the app, so that a step that
        plans it there continues from a ranked target instead.

        The start node is the screen the app is launched on, the root activity of its
        task, and Back there returns to the home screen
        (https://developer.android.com/guide/components/activities/tasks-and-back-stack).
        """
        return node == self.graph.get_start_node()

    def _continue_from_ranked_target(self, position: _Position, max_steps: int) -> _Position | None:
        """Take the step that planned Back on ``position`` by continuing from a ranked
        target, chosen and reached as at a coverage checkpoint but without its merge audit,
        and return where the next step starts; ``None``, having done nothing, when no other
        node is ranked, so that the caller presses Back.

        A navigation that fails before the app restart leaves the device on ``position``,
        where the next step starts.
        """
        with self.controller.source("coverage_checkpoint"):
            target = self._choose_continuation_target(position.node)
            if target is None:
                self.logger.info(
                    "  Back on %s leaves the app, but no other node is ranked; pressing it.",
                    position.node,
                )
                return None
            self.logger.info(
                "Step %d/%d: Back on %s leaves the app; continuing from ranked target %s instead.",
                self.graph.total_steps_completed + 1,
                max_steps,
                position.node,
                target,
            )
            reached = self._navigate_to_continuation(target, position.node)
            if reached is not None:
                position = _Position(reached, self.controller.get_state())
        self.graph.total_steps_completed += 1
        self.save_graph()
        return self._continue_after(position.node, position.device_state, max_steps)

    def _plan_move(self, position: _Position, tokens_before: dict[str, int]) -> _Move | None:
        """Plan an action from ``position`` for the caller to execute; ``None`` when the action
        model asks to wait, which executes nothing."""
        screenshot = position.device_state["screenshot"]
        node_data = self._require_node(position.node)

        detail_before = dict(node_data.get("last_detail_snapshot", {}))

        explored_edges = self.graph.get_all_edges_from_node(position.node)

        # Check before planning so the planner can choose type vs tap.
        keyboard_hint = soft_keyboard_hint(self.controller.config["device"]["udid"])
        input_status = keyboard_status(keyboard_hint)

        unexplored_elements = self.graph.get_unexplored_elements(position.node)
        instruction, planned_element = plan_next_action(
            client=self.client,
            request=PlanRequest(
                screenshot_b64=screenshot,
                page_description=node_data["page_description"],
                explored_edges=explored_edges,
                app_name=self.app_name,
                unexplored_elements=unexplored_elements,
                input_status=input_status,
            ),
            settings=self.instruction,
        )

        action, _history_entry = predict_next_action(
            client=self.client,
            request=ActionRequest(
                screenshot_b64=screenshot,
                instruction=instruction + keyboard_hint,
                screen_w=self.controller.w,
                screen_h=self.controller.h,
                action_history=[],
            ),
            settings=self.action,
        )

        recorded_instruction = instruction
        if action.get("action") == "end":
            self.logger.info(
                '  Action agent said end for: "%s", recording as back',
                instruction,
            )
            action = {"action": "back"}
            recorded_instruction = "Press back"
        elif action.get("action") == "wait":
            self.logger.info("  Action agent requested wait")
            return None
        return _Move(
            source=position.node,
            before=Observation.from_state(position.device_state),
            action=action,
            instruction=recorded_instruction,
            planned_instruction=instruction,
            planned_element=planned_element,
            detail_before=detail_before,
            tokens_before=tokens_before,
        )

    def _record_edge(
        self, move: _Move, target: str, after: dict[str, Any], details: EdgeDetails
    ) -> None:
        """Add the edge ``move`` took to ``target`` and its transition to the step log."""
        self.graph.add_edge(move.source, target, action=move.action_sequence, details=details)
        self.controller.record_transition(
            Transition(
                before=move.before,
                after=Observation.from_state(after),
                instruction=move.planned_instruction,
                element=move.planned_element,
                action=move.action,
                from_node=move.source,
                to_node=target,
            )
        )

    def _record_external_app_step(
        self, move: _Move, device_state: dict[str, Any], max_steps: int
    ) -> _Position:
        device_state = self._handle_external_app(move, device_state)
        self.graph.total_steps_completed += 1
        self.graph.maybe_normalize_node_edges(move.source)
        self.save_graph()
        node, device_state = self._identify(device_state, expected_node=move.source)
        return self._continue_after(node, device_state, max_steps)

    def _record_external_web_step(
        self,
        move: _Move,
        web_node_data: dict[str, Any],
        device_state: dict[str, Any],
        max_steps: int,
    ) -> _Position:
        self.logger.warning(
            "External web page opened from %s. Recording the edge and pressing back.",
            move.source,
        )
        self._record_edge(
            move,
            EXTERNAL_WEB_NODE,
            device_state,
            EdgeDetails(
                instruction=move.instruction,
                target_observation=web_node_data.get("page_description", ""),
            ),
        )
        self.graph.total_steps_completed += 1
        self.save_graph()
        node, device_state = self._leave_external_web(expected_node=move.source)
        return self._continue_after(node, device_state, max_steps)

    def _record_in_app_step(
        self, move: _Move, new_node: str, device_state: dict[str, Any], max_steps: int
    ) -> _Position:
        new_node_data = self._require_node(new_node)
        detail_after = new_node_data.get("last_detail_snapshot", {})
        schema_delta = None
        if new_node == move.source:
            schema_delta = _compute_schema_delta(move.detail_before, detail_after)
            if schema_delta:
                self.logger.info("  Schema delta: %s", dict(schema_delta))

        self._record_edge(
            move,
            new_node,
            device_state,
            EdgeDetails(
                instruction=move.instruction,
                target_observation=_target_observation(
                    new_node_data.get("page_description", ""), detail_after
                ),
                schema_delta=schema_delta,
                num_steps=len(move.action_sequence),
            ),
        )
        self.graph.mark_element_explored(move.source, move.planned_element)
        self.graph.maybe_normalize_node_edges(move.source)
        self.graph.total_steps_completed += 1
        self._log_step(move, new_node, max_steps)
        self.save_graph()
        return self._continue_after(new_node, device_state, max_steps)

    def _log_step(self, move: _Move, new_node: str, max_steps: int) -> None:
        tokens_after = token_tracker.snapshot_by_type()
        step_tokens = {
            a: tokens_after.get(a, 0) - move.tokens_before.get(a, 0) for a in _STEP_AGENT_NAMES
        }
        token_parts = " | ".join(
            f"{a}={step_tokens[a]}" for a in _STEP_AGENT_NAMES if step_tokens[a] > 0
        )
        action_summary = "+".join(a.get("action", "?") for a in move.action_sequence)
        total_actions = sum(
            len(self.graph.graph[s][t].get("actions", [])) for s, t in self.graph.graph.edges()
        )
        self.logger.info(
            "Step %d/%d: %s --[%s]--> %s | graph: %d nodes, %d edges, %d total actions",
            self.graph.total_steps_completed,
            max_steps,
            move.source,
            action_summary,
            new_node,
            self.graph.graph.number_of_nodes(),
            self.graph.graph.number_of_edges(),
            total_actions,
        )
        self.logger.info(
            "  Tokens this step: %s | step_total=%d, cumulative=%d",
            token_parts or "0",
            sum(step_tokens.values()),
            sum(tokens_after.values()),
        )

    def _identify(
        self, device_state: dict[str, Any], *, expected_node: str | None = None
    ) -> tuple[str, dict[str, Any]]:
        """Return the node of the screen in ``device_state``, which the caller expects
        to be ``expected_node`` when it names one, and that state.

        Another app's screen gets no node: with another app in front, the app is
        restarted first, and the node and state returned are those of the screen the
        restart shows, which is expected to be the start node.
        """
        package = device_state.get("package", "")
        if is_outside_app(package, self.package_name):
            self.logger.warning(
                "%s is in front of %s. Restarting %s.", package, self.package_name, self.app_name
            )
            with self.controller.source("external_app_recovery"):
                device_state = restart_into_app(self.controller, self.app_launch, self.logger)
            expected_node = self.graph.get_start_node()
        node = self.graph.identify_state(
            Observation.from_state(device_state), expected_node=expected_node
        )
        return node, device_state

    def _continue_after(self, node: str, device_state: dict[str, Any], max_steps: int) -> _Position:
        node, device_state = self._apply_coverage_checkpoint_after_step(
            self.graph.total_steps_completed,
            max_steps,
            node,
            device_state,
        )
        return _Position(node, device_state)

    def _leave_external_web(self, *, expected_node: str | None) -> tuple[str, dict[str, Any]]:
        """Press back until the screen is no longer an outside web page, restarting the
        app if a few presses do not leave it, and return the screen's node and state.

        A back press is expected to return to ``expected_node``, the screen that opened
        the page, when the caller knows it, and a restart to the start node. A step from
        the shared ``ext_web`` node would act on a page outside the app, and
        ``add_edge`` would refuse its edge.
        """
        with self.controller.source("external_web_recovery"):
            return self._press_back_off_external_web(expected_node)

    def _press_back_off_external_web(self, expected_node: str | None) -> tuple[str, dict[str, Any]]:
        for _ in range(EXTERNAL_WEB_BACK_PRESSES):
            self.controller.exe_action({"action": "back"})
            time.sleep(1)
            node, device_state = self._identify(
                self.controller.get_state(), expected_node=expected_node
            )
            if node != EXTERNAL_WEB_NODE:
                return node, device_state
        self.logger.warning(
            "Still on an external web page after %d back presses. Restarting %s.",
            EXTERNAL_WEB_BACK_PRESSES,
            self.package_name,
        )
        restart_app(self.controller, self.app_launch, self.logger)
        return self._identify(
            self.controller.get_state(), expected_node=self.graph.get_start_node()
        )

    def _handle_external_app(self, move: _Move, device_state: dict[str, Any]) -> dict[str, Any]:
        """Handle navigation to an external app.

        Records the screen in the external app's node (no VLM calls) and the edge, and
        returns to the app with ``return_to_app``.

        Returns:
            The state of the app's screen that ``return_to_app`` returned to.
        """
        observation = Observation.from_state(device_state)
        external_package = observation.package.strip()
        self.logger.warning(
            "Left target app! Current package: %s (expected: %s). "
            "Creating external node and pressing back.",
            external_package,
            self.package_name,
        )
        ext_node_id = self.graph.record_external_app_visit(observation)
        self._record_edge(
            move,
            ext_node_id,
            device_state,
            EdgeDetails(
                instruction=move.instruction,
                target_observation=f"[left app to {external_package}]",
            ),
        )
        self.logger.info(
            "Step: %s --[%s]--> %s (external)",
            move.source,
            move.action.get("action", "?"),
            ext_node_id,
        )
        with self.controller.source("external_app_recovery"):
            return return_to_app(self.controller, self.app_launch, self.logger)

    def save_graph(self) -> None:
        """Persist the graph to disk."""
        self.graph.save_graph(self.graph_path)
