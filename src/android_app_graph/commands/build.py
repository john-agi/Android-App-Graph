"""Build a graph from a recorded step log, without a device, under the config's matching
settings; README.md#build-a-graph-from-a-step-log.

Usage:
    uv run app-graph-build -c configs/explore.yaml --app settings --steps STEPS --out NEW_DIR
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from android_app_graph.device import is_outside_app
from android_app_graph.graph_files import (
    identify_file_path,
    image_embeddings_path,
    read_json_lines,
)
from android_app_graph.payloads import as_object_dict, as_str
from android_app_graph.step_log import (
    load_image_embeddings,
    observation_ownership,
    read_records,
    recorded_observation,
    step_order,
)
from android_app_graph.utils import (
    check_config_keys,
    open_client,
    resolve_api_key,
    resolve_audit_timeout,
    resolve_matching_settings,
    resolve_model_settings,
)
from android_app_graph.utils.graph_manager import (
    EXTERNAL_WEB_NODE,
    EdgeDetails,
    GraphManager,
    SetupTrail,
    open_menu_in,
    schema_delta,
    target_observation,
)
from android_app_graph.utils.logging import setup_logging
from android_app_graph.utils.vlm_utils import token_tracker

logger = logging.getLogger(__name__)

BUILD_ROLES = ("page_detail", "embedding", "image_embedding")


@dataclass(frozen=True, slots=True)
class RecordedLog:
    """What a build replays from a step log: the match and transition records of the runs
    it takes, each at the step of the screenshot it matches or at its own step, in that
    order, and every observation record by ID."""

    steps: Path
    events: list[dict[str, object]]
    observations: dict[str, dict[str, object]]

    @property
    def matched(self) -> list[str]:
        """The screenshots the match records name, in order."""
        return [
            as_str(event.get("observation"), "")
            for event in self.events
            if event.get("kind") == "match"
        ]


def _event_step(record: Mapping[str, object]) -> tuple[int, str]:
    step = record.get("observation") if record.get("kind") == "match" else record.get("id")
    return step_order(as_str(step, ""))


def read_recorded_log(steps: Path, run_kinds: Collection[str]) -> RecordedLog:
    """Return the match and transition records of the runs of ``run_kinds`` in the step
    log ``steps``, ordered as a build replays them, with every observation record.

    A step log that does not exist raises ``FileNotFoundError``, and a match record that
    names an observation without a record raises ``ValueError`` naming each such one.
    """
    if not steps.is_dir():
        msg = f"No step log at {steps}"
        raise FileNotFoundError(msg)
    records = read_records(steps)
    observations = {
        as_str(record.get("id"), ""): record
        for record in records
        if record.get("kind") == "observation"
    }
    events = sorted(
        (
            record
            for record in records
            if record.get("kind") in {"match", "transition"} and record.get("run_kind") in run_kinds
        ),
        key=_event_step,
    )
    log = RecordedLog(steps, events, observations)
    unrecorded = [observation for observation in log.matched if observation not in observations]
    if unrecorded:
        msg = f"{steps} has match records of observations without a record: {unrecorded}"
        raise ValueError(msg)
    for record in observations.values():
        observation_ownership(record)
    return log


def require_image_vectors(log: RecordedLog, model: str) -> dict[str, list[float]]:
    """Return the image vectors ``model`` made of the screenshots ``log`` matches, from the
    step log's ``image_emb.json``, which a build never writes; raise ``ValueError`` naming
    how many of them have none."""
    vectors = load_image_embeddings(log.steps, model=model)
    matched = [
        observation
        for observation in log.matched
        if observation_ownership(log.observations[observation]).surface is None
    ]
    missing = [observation for observation in matched if observation not in vectors]
    if missing:
        msg = (
            f"{len(missing)} of the {len(matched)} app screenshot(s) the log matches have no "
            f"image vector of {model} in {image_embeddings_path(log.steps)}. Configure the "
            "image-embedding model that embedded the log."
        )
        raise ValueError(msg)
    return {observation: vectors[observation] for observation in matched}


@dataclass(frozen=True, kw_only=True)
class BuildTarget:
    """The app a build places screenshots for, the graph file it writes, and the
    coverage-checkpoint interval at which it runs the merge audit."""

    app_name: str
    package_name: str
    graph_path: Path
    checkpoint_steps: int


@dataclass(slots=True)
class _Run:
    """What a build knows of one recorded run as it replays it: the screenshots it matched,
    in order, and its setup trail, last moved on by the step that ended on ``trail_head``."""

    trail: SetupTrail
    matched: list[str] = field(default_factory=list)
    trail_head: str | None = None

    def start_of(self, before: str) -> str | None:
        """Return the screenshot the run last matched at or before ``before``, which shows
        the screen a step from ``before`` starts on."""
        step = step_order(before)
        return next((m for m in reversed(self.matched) if step_order(m) <= step), None)


class _Replay:
    """One build: the recorded log's matches and transitions applied to ``graph`` in
    order, as exploration applied them."""

    def __init__(self, graph: GraphManager, log: RecordedLog, target: BuildTarget) -> None:
        self.graph = graph
        self.log = log
        self.target = target
        self._runs: dict[str, _Run] = {}
        # By screenshot, the state its node recorded when the build matched it.
        self._states: dict[str, dict[str, Any]] = {}
        # The screenshots a replay matched on the node it expected, where it arrived, so the
        # setup of the next action there starts empty.
        self._arrivals: set[str] = set()

    def _run(self, record: Mapping[str, object]) -> _Run:
        run_id = as_str(record.get("run_id"), "")
        if run_id not in self._runs:
            self._runs[run_id] = _Run(SetupTrail(self.graph.resolve_node_id))
        return self._runs[run_id]

    def run(self) -> None:
        explored = sum(
            1
            for event in self.log.events
            if event.get("kind") == "transition" and event.get("run_kind") == "explore"
        )
        steps = 0
        for event in self.log.events:
            if event.get("kind") == "match":
                self._match(event)
                continue
            self._transition(event)
            if event.get("run_kind") == "explore":
                steps += 1
                if self._is_checkpoint(steps, explored):
                    logger.info("Coverage checkpoint at step %d: merge audit", steps)
                    self.graph.run_node_merge_audit(app_name=self.target.app_name)
        self.graph.total_steps_completed = steps
        self.graph.normalize_all_edges()
        self.graph.save_graph(self.target.graph_path)

    def _is_checkpoint(self, step: int, steps: int) -> bool:
        interval = self.target.checkpoint_steps
        return interval > 0 and 0 < step < steps and step % interval == 0

    def _match(self, record: Mapping[str, object]) -> None:
        observation_id = as_str(record.get("observation"), "")
        observation = recorded_observation(self.log.steps, self.log.observations[observation_id])
        expected = record.get("expected_observation")
        expected_observation = expected if isinstance(expected, str) else None
        node = self.graph.identify_state(observation, expected_observation=expected_observation)
        state = self.graph.graph.nodes[node].get("last_detail_snapshot", {})
        self._states[observation_id] = dict(state)
        in_replay = record.get("source") == "navigation_replay"
        if in_replay and node == self.graph.node_of(expected_observation):
            self._arrivals.add(observation_id)
        self._run(record).matched.append(observation_id)

    def _transition(self, record: Mapping[str, object]) -> None:
        run = self._run(record)
        step = as_str(record.get("id"), "")
        start = run.start_of(as_str(record.get("before"), ""))
        source = self.graph.node_of(start)
        if start is None or source is None:
            logger.warning("Step %s starts on no screenshot the build matched; skipping it", step)
            return
        action = record.get("action")
        actions = [as_object_dict(a) for a in (action if isinstance(action, list) else [action])]
        after = as_str(record.get("after"), "")
        instruction = as_str(record.get("instruction"), "")
        placed = self._target(after, instruction, (source, start), len(actions))
        if placed is None:
            logger.warning("Step %s ends on a screen of the app that was not matched", step)
            return
        target, details = placed
        if run.trail_head != start:
            arrival = self._states[start] if start in self._arrivals else None
            run.trail.begin(source, arrival)
        self.graph.add_edge(
            source,
            target,
            actions,
            details=replace(
                details, setup=run.trail.setup(source), open_menu=open_menu_in(self._states[start])
            ),
        )
        element = record.get("element")
        self.graph.mark_element_explored(source, element if isinstance(element, int) else None)
        if target.startswith("ext_"):
            run.trail_head = None
        else:
            run.trail.step(actions, target, self._states[after])
            run.trail_head = after
        logger.info("Step %s: %s --[%s]--> %s", step, source, instruction, target)

    def _target(
        self, after: str, instruction: str, start: tuple[str, str], num_steps: int
    ) -> tuple[str, EdgeDetails] | None:
        """Return the node the screenshot ``after`` belongs to, or the ``ext_<package>``
        node of another app's screen, with the details its edge from ``start``, a node and
        the screenshot it was matched on, records; ``None`` for an unmatched screen of the
        app."""
        source, start_screenshot = start
        node = self.graph.node_of(after) if after in self._states else None
        if node == EXTERNAL_WEB_NODE:
            description = self.graph.latest_description(node)
            return node, EdgeDetails(instruction=instruction, target_observation=description)
        if node is not None:
            state = self._states[after]
            delta = schema_delta(self._states[start_screenshot], state) if node == source else None
            return node, EdgeDetails(
                instruction=instruction,
                target_observation=target_observation(self.graph.latest_description(node), state),
                schema_delta=delta,
                num_steps=num_steps,
            )
        record = self.log.observations.get(after, {})
        package = as_str(record.get("package"), "").strip()
        if not record or not is_outside_app(package, self.target.package_name):
            return None
        other_app = self.graph.record_external_app_visit(
            recorded_observation(self.log.steps, record)
        )
        return other_app, EdgeDetails(
            instruction=instruction, target_observation=f"[left app to {package}]"
        )


def build_graph(graph: GraphManager, log: RecordedLog, target: BuildTarget) -> None:
    """Replay ``log`` into ``graph``, a new graph, and save it at ``target.graph_path``:
    match the screenshot of each match record, add each transition as an edge between the
    nodes of its screenshots, run the merge audit at the coverage-checkpoint interval and
    edge normalization at the end; README.md#build-a-graph-from-a-step-log."""
    _Replay(graph, log, target).run()
    _log_summary(graph, target.graph_path)


def _log_summary(graph: GraphManager, graph_path: Path) -> None:
    decisions = [
        as_object_dict(line.get("identify"))
        for line in read_json_lines(identify_file_path(graph_path))
    ]
    stages = Counter(as_str(decision.get("stage"), "") for decision in decisions)
    verifications = sum(
        len(calls) for d in decisions if isinstance(calls := d.get("verifications"), list)
    )
    in_app = [node for node in graph.graph if not str(node).startswith("ext_")]
    logger.info(
        "Built %s: %d in-app nodes, %d edges; decisions by stage: %s; %d verifier call(s) "
        "in screen matching",
        graph_path,
        len(in_app),
        graph.graph.number_of_edges(),
        ", ".join(f"{stage} {count}" for stage, count in sorted(stages.items())),
        verifications,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="app-graph-build",
        description="Build a graph from a recorded step log, without a device.",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=Path("configs/explore.yaml"),
        help="Path to the YAML config file (default: configs/explore.yaml).",
    )
    parser.add_argument(
        "--app", required=True, help="The app of the config whose step log is built."
    )
    parser.add_argument("--steps", type=Path, required=True, help="The step log to build from.")
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="A new directory for the graph; its steps is a link to the step log.",
    )
    parser.add_argument(
        "--audit-runs",
        action="store_true",
        help="Also replay the audit's re-exploration runs, after the exploration runs.",
    )
    return parser


def _config_app(config: Mapping[str, object], app_name: str) -> tuple[str, str]:
    """Return the name and package of the app ``app_name`` in the config's ``apps``."""
    apps = config.get("apps")
    for app in apps if isinstance(apps, list) else []:
        entry = as_object_dict(app)
        package = as_str(entry.get("package_name"), "")
        if entry.get("name") == app_name and package:
            return app_name, package
    msg = f"The config has no app named {app_name!r} with a package_name"
    raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class _Inputs:
    """Everything a build reads, checked before any request or file is written."""

    target: BuildTarget
    log: RecordedLog
    image_vectors: dict[str, list[float]]
    graph: GraphManager
    api_key: str


def _read_inputs(args: argparse.Namespace, config: dict[str, Any]) -> _Inputs:
    check_config_keys(config)
    vlm_config = as_object_dict(config.get("vlm"))
    experiment = as_object_dict(config.get("experiment"))
    app_name, package_name = _config_app(config, args.app)
    settings = resolve_model_settings(vlm_config, roles=BUILD_ROLES)
    graph = GraphManager(
        page_detail=settings["page_detail"],
        embedding=settings["embedding"],
        audit_timeout=resolve_audit_timeout(vlm_config),
    )
    graph.matching = resolve_matching_settings(vlm_config)
    api_key = resolve_api_key()
    out_dir = Path(args.out)
    if out_dir.exists():
        msg = f"{out_dir} exists; a build writes into a new directory"
        raise ValueError(msg)
    log = read_recorded_log(args.steps, ("explore", "audit") if args.audit_runs else ("explore",))
    image_vectors = require_image_vectors(log, settings["image_embedding"].model)
    checkpoint_steps = experiment.get("coverage_checkpoint_steps", 50)
    target = BuildTarget(
        app_name=app_name,
        package_name=package_name,
        graph_path=out_dir / f"{app_name}.json",
        checkpoint_steps=checkpoint_steps if isinstance(checkpoint_steps, int) else 50,
    )
    return _Inputs(target, log, image_vectors, graph, api_key)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.config.is_file():
        parser.error(f"config file not found: {args.config}")
    with args.config.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    try:
        inputs = _read_inputs(args, config)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    # After argument validation, so --help and the parser.error paths above never touch the
    # root logger.
    setup_logging(level=logging.INFO)
    out_dir = inputs.target.graph_path.parent
    out_dir.mkdir(parents=True)
    steps_link = out_dir / "steps"
    steps_link.symlink_to(inputs.log.steps.resolve(), target_is_directory=True)
    graph = inputs.graph
    graph.steps_dir = steps_link
    graph.identify_file = identify_file_path(inputs.target.graph_path)
    graph.use_image_vectors(inputs.image_vectors)
    logger.info(
        "Building %s from %d match and transition record(s) of %s",
        inputs.target.graph_path,
        len(inputs.log.events),
        inputs.log.steps,
    )
    with open_client(inputs.api_key) as client:
        graph.client = client
        build_graph(graph, inputs.log, inputs.target)
    token_tracker.print_summary()
    return 0
