"""Checkpoint evidence bounds and reuse across an owned graph's checkpoints."""

from __future__ import annotations

import itertools
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import pytest
from openrouter import OpenRouter

from android_app_graph.checkpoint import CHECKPOINT_INPUT_BYTES, IdentityBatch, IdentityNode
from android_app_graph.utils import ModelSettings, graph_manager, vlm_utils
from android_app_graph.utils.graph_manager import GraphManager
from android_app_graph.utils.vlm_utils import DescribedScreenshot


@dataclass
class CheckpointModels:
    summaries: list[str] = field(default_factory=list)
    comparisons: list[tuple[DescribedScreenshot, DescribedScreenshot]] = field(default_factory=list)
    unanswered: int = 0
    proposal_unanswered: int = 0
    same: bool = False
    proposal_reply: dict[str, object] | None = None
    verifier_reply: dict[str, object] | None = None

    def propose(self, _client: object, graph_summary: str, **_kwargs: object) -> dict[str, object]:
        self.summaries.append(graph_summary)
        if self.proposal_reply is not None:
            return self.proposal_reply
        if self.proposal_unanswered:
            self.proposal_unanswered -= 1
            return {"issues": [], "summary": "timeout", "answered": False}
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
        if self.verifier_reply is not None:
            return self.verifier_reply
        answered = self.unanswered == 0
        self.unanswered = max(0, self.unanswered - 1)
        return {
            "same": self.same,
            "answered": answered,
            "reason": "scripted" if answered else "timeout",
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


@pytest.mark.parametrize(
    "change", ["description", "image", "model", "rule", "temperature", "timeout", "schema"]
)
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
    elif change == "temperature":
        checkpoint_graph.page_detail = ModelSettings("test/checkpoint", temperature=0.4)
    elif change == "timeout":
        checkpoint_graph.page_detail = ModelSettings("test/checkpoint", request_timeout=12)
    elif change == "schema":
        monkeypatch.setattr(vlm_utils, "_NODE_VERIFICATION_SCHEMA", {"type": "object"})
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


def test_saving_preserves_checkpoint_caches_and_loading_starts_their_lifecycle(
    checkpoint_graph: GraphManager, checkpoint_models: CheckpointModels
) -> None:
    assert checkpoint_graph.steps_dir is not None
    path = checkpoint_graph.steps_dir.parent / "checkpoint.json"
    checkpoint_graph.run_node_merge_audit()
    checkpoint_graph.save_graph(path)
    reused = checkpoint_graph.run_node_merge_audit()
    assert len(checkpoint_models.summaries) == len(checkpoint_models.comparisons) == 1
    assert (reused["proposal_reused"], reused["verifier_reused"]) == (1, 1)

    checkpoint_graph.load_graph(path)
    fresh = checkpoint_graph.run_node_merge_audit()

    assert len(checkpoint_models.summaries) == len(checkpoint_models.comparisons) == 2
    assert (fresh["proposal_requests"], fresh["verifier_requests"]) == (1, 1)


def test_proposal_timeout_is_retried_and_a_complete_reply_is_reused(
    checkpoint_graph: GraphManager, checkpoint_models: CheckpointModels
) -> None:
    checkpoint_models.proposal_unanswered = 1

    first = checkpoint_graph.run_node_merge_audit()
    checkpoint_graph.run_node_merge_audit()
    third = checkpoint_graph.run_node_merge_audit()

    assert (first["proposal_unanswered"], first["verifier_requests"]) == (1, 0)
    assert len(checkpoint_models.summaries) == 2
    assert len(checkpoint_models.comparisons) == 1
    assert (third["proposal_reused"], third["verifier_reused"]) == (1, 1)


@pytest.mark.parametrize(
    "reply",
    [{}, {"issues": False}, {"issues": [{"node_a": "n0", "node_b": 5}]}],
)
def test_malformed_proposals_are_saved_as_unanswered_and_remain_eligible(
    reply: dict[str, object], checkpoint_graph: GraphManager, checkpoint_models: CheckpointModels
) -> None:
    checkpoint_models.proposal_reply = reply

    first = checkpoint_graph.run_node_merge_audit()
    checkpoint_models.proposal_reply = None
    checkpoint_graph.run_node_merge_audit()
    checkpoint_graph.run_node_merge_audit()

    assert (first["proposal_unanswered"], first["verifier_requests"]) == (1, 0)
    assert len(checkpoint_models.summaries) == 2
    assert len(checkpoint_models.comparisons) == 1


def test_cross_batch_scope_violations_remain_eligible_for_a_fresh_proposal(
    tmp_path: Path,
    write_member_graph: Callable[..., None],
    router_client: OpenRouter,
    checkpoint_models: CheckpointModels,
) -> None:
    path = tmp_path / "many" / "many.json"
    write_member_graph(
        path,
        {f"s{n}": [(f"{n + 1:06d}", [1.0, float(n)])] for n in range(65)},
    )
    graph = _load(path, router_client)

    first = graph.run_node_merge_audit()
    second = graph.run_node_merge_audit()

    assert first["candidate_pairs"] == 65 * 64 // 2
    assert (
        first["proposal_requests"],
        first["skipped_proposals"],
        first["proposal_unanswered"],
    ) == (2, 1, 1)
    assert (second["proposal_requests"], second["proposal_reused"]) == (1, 1)
    assert len(checkpoint_models.summaries) == 3
    assert len(checkpoint_models.comparisons) == 1


def test_changed_closest_member_is_a_distinct_checkpoint_question(
    tmp_path: Path,
    write_member_graph: Callable[..., None],
    router_client: OpenRouter,
    checkpoint_models: CheckpointModels,
) -> None:
    path = tmp_path / "members" / "members.json"
    write_member_graph(
        path,
        {
            "n0": [("000001", [1.0, 0.0])],
            "n1": [("000002", [0.0, 1.0])],
            "n2": [("000003", [0.0, 1.0])],
        },
    )
    graph = _load(path, router_client)
    graph.run_node_merge_audit()
    graph.graph.nodes["n0"]["observations"].append("000003")
    graph.graph.remove_node("n2")

    graph.run_node_merge_audit()

    assert len(checkpoint_models.comparisons) == 2
    assert checkpoint_models.comparisons[0][0] != checkpoint_models.comparisons[1][0]


def test_checkpoint_counts_describe_the_input_scope_even_when_a_merge_changes_it(
    checkpoint_graph: GraphManager, checkpoint_models: CheckpointModels
) -> None:
    checkpoint_models.same = True

    result = checkpoint_graph.run_node_merge_audit()

    assert result["merged_count"] == 1
    assert (result["candidate_nodes"], result["candidate_pairs"], result["excluded_nodes"]) == (
        2,
        1,
        0,
    )


def test_unicode_and_json_escaping_obey_the_full_request_bound(
    tmp_path: Path,
    write_member_graph: Callable[..., None],
    router_client: OpenRouter,
    router_requests: list[httpx.Request],
    router_responses: list[httpx.Response],
    chat_response: Callable[..., httpx.Response],
) -> None:
    path = tmp_path / "unicode" / "unicode.json"
    nodes = {f"s{n}-" + "🧭" * 500: [(f"{n + 1:06d}", [1.0, float(n)])] for n in range(128)}
    write_member_graph(
        path,
        nodes,
        described={f"{n + 1:06d}": '\\"\0🧭' * 500 for n in range(128)},
    )
    graph = _load(path, router_client)
    router_responses.extend(
        chat_response(content='{"issues":[],"summary":"distinct"}') for _ in range(3)
    )

    result = graph.run_node_merge_audit(app_name='🧭\\"' * 500)

    assert len(router_requests) == 3
    for request in router_requests:
        assert len(request.content) <= result["proposal_request_byte_bound"]
    for batch in graph.checkpoint_batches():
        assert len(batch.text.encode("utf-8")) <= CHECKPOINT_INPUT_BYTES


def test_an_oversized_manually_constructed_batch_is_rejected() -> None:
    node = IdentityNode(
        "id", "package", (), ("x" * 1000,) * 3, ("y" * 1000,) * 2, ("z" * 1000,) * 2, "evidence"
    )
    batch = IdentityBatch((node,) * 300, 0)

    with pytest.raises(ValueError, match="input byte bound"):
        _ = batch.text


@pytest.mark.parametrize(
    "alias", ["", "s0", "n-1", "n01", "n\N{ARABIC-INDIC DIGIT ONE}", "n1000", "n" + "9" * 5000]
)
def test_only_declared_bounded_ascii_aliases_resolve(alias: str) -> None:
    node = IdentityNode("original-node", "package", (), ("Home",), (), (), "evidence")
    batch = IdentityBatch((node,), 0)

    assert batch.node(alias) is None
    assert not batch.permits(alias, "n0")


@pytest.mark.parametrize(
    ("boundary", "reason"),
    [
        ("missing", "node missing"),
        ("self", "same node"),
        ("external", "external node"),
        ("memberless", "missing screenshot"),
    ],
)
def test_full_merge_verification_retains_its_boundary_checks(
    boundary: str,
    reason: str,
    checkpoint_graph: GraphManager,
    checkpoint_models: CheckpointModels,
) -> None:
    first, second = "n0", "n1"
    if boundary == "missing":
        second = "missing"
    elif boundary == "self":
        second = first
    elif boundary == "external":
        checkpoint_graph.graph.nodes[second]["is_external"] = True
    else:
        checkpoint_graph.graph.nodes[second]["observations"] = []
    proposals = checkpoint_graph.merge_proposals([{"node_a": first, "node_b": second}], sources=())

    result = checkpoint_graph.verify_merge_proposals(proposals, {})

    assert result.pairs[0]["reason"] == reason
    assert result.pairs[0]["status"] == "skipped"
    assert not checkpoint_models.comparisons


def test_a_cached_answer_cannot_hide_a_missing_member_description(
    checkpoint_graph: GraphManager,
    checkpoint_models: CheckpointModels,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_graph.run_node_merge_audit()
    monkeypatch.setattr(checkpoint_graph, "_descriptions", {"000001": "Screen 000001"})

    with pytest.raises(ValueError, match="description"):
        checkpoint_graph.run_node_merge_audit()

    assert len(checkpoint_models.comparisons) == 1


@pytest.mark.parametrize("reply", [{}, {"same": None}, {"same": "false"}])
def test_a_missing_boolean_answer_is_unanswered_and_remains_eligible(
    reply: dict[str, object], checkpoint_graph: GraphManager, checkpoint_models: CheckpointModels
) -> None:
    checkpoint_models.verifier_reply = reply
    first = checkpoint_graph.run_node_merge_audit()
    checkpoint_models.verifier_reply = None
    checkpoint_graph.run_node_merge_audit()
    third = checkpoint_graph.run_node_merge_audit()

    assert first["results"][0]["same"] is None
    assert len(checkpoint_models.comparisons) == 2
    assert third["verifier_reused"] == 1
