"""Checkpoint evidence bounds and reuse across an owned graph's checkpoints."""

from __future__ import annotations

import itertools
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from openrouter import OpenRouter

from android_app_graph.utils import ModelSettings, graph_manager, vlm_utils
from android_app_graph.utils.graph_manager import GraphManager
from android_app_graph.utils.vlm_utils import DescribedScreenshot


@dataclass
class CheckpointModels:
    summaries: list[str] = field(default_factory=list)
    comparisons: list[tuple[DescribedScreenshot, DescribedScreenshot]] = field(default_factory=list)
    unanswered: int = 0

    def propose(self, _client: object, graph_summary: str, **_kwargs: object) -> dict[str, object]:
        self.summaries.append(graph_summary)
        return {"issues": [{"node_a": "n0", "node_b": "n1"}], "summary": "one candidate"}

    def verify(
        self,
        _client: object,
        *,
        existing: DescribedScreenshot,
        new: DescribedScreenshot,
        **_kwargs: object,
    ) -> dict[str, object]:
        self.comparisons.append((existing, new))
        answered = self.unanswered == 0
        self.unanswered = max(0, self.unanswered - 1)
        return {
            "same": False,
            "answered": answered,
            "reason": "different" if answered else "timeout",
        }


@pytest.fixture
def checkpoint_models(monkeypatch: pytest.MonkeyPatch) -> CheckpointModels:
    models = CheckpointModels()
    monkeypatch.setattr(graph_manager, "audit_merge_nodes", models.propose)
    monkeypatch.setattr(graph_manager, "verify_same_node", models.verify)
    return models


def _load(path: Path, client: OpenRouter) -> GraphManager:
    settings = ModelSettings("test/checkpoint")
    graph = GraphManager(
        client=client,
        page_detail=settings,
        embedding=ModelSettings("google/gemini-embedding-2"),
        image_embedding=ModelSettings("google/gemini-embedding-2"),
    )
    graph.load_graph(path)
    return graph


@pytest.fixture
def checkpoint_graph(
    tmp_path: Path, write_member_graph: Callable[..., None], router_client: OpenRouter
) -> GraphManager:
    path = tmp_path / "checkpoint" / "checkpoint.json"
    write_member_graph(
        path,
        {"n0": [("000001", [1.0, 0.0])], "n1": [("000002", [0.0, 1.0])]},
    )
    return _load(path, router_client)


def test_unchanged_checkpoint_evidence_reuses_its_answer(
    checkpoint_graph: GraphManager, checkpoint_models: CheckpointModels
) -> None:
    checkpoint_graph.run_node_merge_audit()
    checkpoint_graph.run_node_merge_audit()

    assert len(checkpoint_models.comparisons) == 1


@pytest.mark.parametrize("change", ["description", "image", "model", "rule"])
def test_changed_question_evidence_gets_a_fresh_answer(
    change: str,
    checkpoint_graph: GraphManager,
    checkpoint_models: CheckpointModels,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_graph.run_node_merge_audit()
    checkpoint_graph.run_node_merge_audit()
    if change == "description":
        monkeypatch.setattr(
            checkpoint_graph,
            "_descriptions",
            {"000001": "Another screen description", "000002": "Screen 000002"},
        )
    elif change == "image":
        assert checkpoint_graph.steps_dir is not None
        (checkpoint_graph.steps_dir / "000001.webp").write_bytes(b"different image bytes")
    elif change == "model":
        checkpoint_graph.page_detail = ModelSettings("test/another-model")
    else:
        monkeypatch.setattr(vlm_utils, "NODE_VERIFY_PROMPT", "A changed verification rule")
    checkpoint_graph.run_node_merge_audit()

    assert len(checkpoint_models.comparisons) == 2


def test_unanswered_checkpoint_question_is_eligible_again(
    checkpoint_graph: GraphManager, checkpoint_models: CheckpointModels
) -> None:
    checkpoint_models.unanswered = 1

    for _ in range(3):
        checkpoint_graph.run_node_merge_audit()

    assert len(checkpoint_models.comparisons) == 2


def test_edge_action_growth_preserves_duplicate_proposal_input(
    checkpoint_graph: GraphManager, checkpoint_models: CheckpointModels
) -> None:
    checkpoint_graph.graph.add_edge(
        "n0", "n1", instructions=["Tap next"], target_observations=["Destination"]
    )
    checkpoint_graph.run_node_merge_audit(app_name="First")
    edge = checkpoint_graph.graph.edges["n0", "n1"]
    edge["instructions"].extend(["Repeat this action " * 200] * 100)
    edge["target_observations"].extend(["Still the destination"] * 100)
    checkpoint_graph.run_node_merge_audit(app_name="Second")

    assert checkpoint_models.summaries[0] == checkpoint_models.summaries[1]
    assert "Tap next" not in checkpoint_models.summaries[0]


def test_each_graph_owns_its_checkpoint_answers(
    checkpoint_graph: GraphManager,
    checkpoint_models: CheckpointModels,
    tmp_path: Path,
    write_member_graph: Callable[..., None],
    router_client: OpenRouter,
) -> None:
    path = tmp_path / "other" / "other.json"
    write_member_graph(path, {"n0": [("000001", [1.0, 0.0])], "n1": [("000002", [0.0, 1.0])]})
    other = _load(path, router_client)

    for graph in (checkpoint_graph, other):
        graph.run_node_merge_audit()
        graph.run_node_merge_audit()

    assert len(checkpoint_models.comparisons) == 2


def test_checkpoint_requests_bound_input_and_cover_cross_batch_pairs(
    tmp_path: Path,
    write_member_graph: Callable[..., None],
    router_client: OpenRouter,
    checkpoint_models: CheckpointModels,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    count = 65
    path = tmp_path / "large" / "large.json"
    nodes = {f"s{n}": [(f"{n + 1:06d}", [1.0, float(n)])] for n in range(count)}
    descriptions = {
        f"{n + 1:06d}": f"purpose-{n} "
        + ("home-alpha " if n == 0 else "home-omega " if n == count - 1 else "")
        + "long description " * 200
        for n in range(count)
    }
    write_member_graph(path, nodes, described=descriptions)
    graph = _load(path, router_client)

    def propose(_client: object, graph_summary: str, **_kwargs: object) -> dict[str, object]:
        checkpoint_models.summaries.append(graph_summary)
        found: dict[str, str] = {}
        for line in graph_summary.splitlines():
            match = re.match(r"\s*(\w+):", line)
            if match:
                for label in ("home-alpha", "home-omega"):
                    if label in line:
                        found[label] = match.group(1)
        issues = (
            [{"node_a": found["home-alpha"], "node_b": found["home-omega"]}]
            if len(found) == 2
            else []
        )
        return {"issues": issues, "summary": "cross-batch candidate"}

    monkeypatch.setattr(graph_manager, "audit_merge_nodes", propose)
    result = graph.run_node_merge_audit()

    assert len(checkpoint_models.summaries) > 1
    assert all(len(text.encode("utf-8")) <= 131_072 for text in checkpoint_models.summaries)
    considered = set().union(
        *(
            set(
                itertools.combinations(
                    sorted({int(n) for n in re.findall(r"purpose-(\d+)", text)}), 2
                )
            )
            for text in checkpoint_models.summaries
        )
    )
    assert considered == set(itertools.combinations(range(count), 2))
    assert [(pair["node_a"], pair["node_b"]) for pair in result["results"]] == [("s0", "s64")]
