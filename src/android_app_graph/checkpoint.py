"""Bounded identity evidence and complete deterministic checkpoint pair batches."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import NotRequired, TypedDict

IDENTITY_BLOCK_SIZE = 64
CHECKPOINT_INPUT_BYTES = 131_072
APP_NAME_BYTES = 128
_DESCRIPTION_BYTES = 128
_CONTEXT_BYTES = 64
_DESCRIPTIONS = 3
_NEIGHBORS = 2


class CheckpointReport(TypedDict):
    issues: list[dict[str, object]]
    results: list[dict[str, object]]
    merged_count: int
    summary: NotRequired[str]
    candidate_pairs: NotRequired[int]
    candidate_nodes: NotRequired[int]
    excluded_nodes: NotRequired[int]
    excluded_pairs: NotRequired[int]
    proposal_requests: NotRequired[int]
    proposal_reused: NotRequired[int]
    proposal_unanswered: NotRequired[int]
    proposal_input_bytes: NotRequired[int]
    largest_proposal_input_bytes: NotRequired[int]
    proposal_request_byte_bound: NotRequired[int]
    verifier_requests: NotRequired[int]
    verifier_reused: NotRequired[int]
    skipped_proposals: NotRequired[int]


def evidence_key(value: object) -> str:
    """Hash structured evidence, preserving field and sequence boundaries."""
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def clipped_text(value: str, limit: int) -> str:
    """Fit one line of text in ``limit`` UTF-8 bytes."""
    return " ".join(value.split()).encode("utf-8")[:limit].decode("utf-8", errors="ignore")


@dataclass(frozen=True, slots=True)
class IdentityNode:
    node_id: str
    package: str
    activities: tuple[str, ...]
    descriptions: tuple[str, ...]
    incoming: tuple[str, ...]
    outgoing: tuple[str, ...]

    def line(self, alias: str) -> str:
        descriptions = "; ".join(
            clipped_text(text, _DESCRIPTION_BYTES) for text in self.descriptions[:_DESCRIPTIONS]
        )
        incoming = "; ".join(
            clipped_text(text, _CONTEXT_BYTES) for text in self.incoming[:_NEIGHBORS]
        )
        outgoing = "; ".join(
            clipped_text(text, _CONTEXT_BYTES) for text in self.outgoing[:_NEIGHBORS]
        )
        return (
            f"  {alias}: package={clipped_text(self.package, _CONTEXT_BYTES)}"
            f" | activities={clipped_text(' / '.join(self.activities[:2]), _CONTEXT_BYTES)}"
            f" | descriptions={descriptions} ({min(len(self.descriptions), _DESCRIPTIONS)}"
            f" of {len(self.descriptions)} distinct descriptions)"
            f" | incoming={incoming} | outgoing={outgoing}"
        )


@dataclass(frozen=True, slots=True)
class IdentityBatch:
    """One within-block or cross-block request, with local aliases for bounded IDs."""

    nodes: tuple[IdentityNode, ...]
    left_size: int

    @property
    def pair_count(self) -> int:
        count = len(self.nodes)
        return (
            self.left_size * (count - self.left_size)
            if self.left_size
            else count * (count - 1) // 2
        )

    @property
    def text(self) -> str:
        aliases = [f"n{index}" for index in range(len(self.nodes))]
        scope = (
            f"Compare only pairs across Left={aliases[: self.left_size]} and "
            f"Right={aliases[self.left_size :]}."
            if self.left_size
            else "Compare every unordered pair of distinct nodes below."
        )
        lines = [scope, "Use the local node IDs below in every proposal."]
        lines.extend(node.line(alias) for node, alias in zip(self.nodes, aliases, strict=True))
        text = "\n".join(lines)
        if len(text.encode("utf-8")) > CHECKPOINT_INPUT_BYTES:
            message = "Checkpoint identity batch exceeds its input byte bound"
            raise ValueError(message)
        return text

    @property
    def evidence(self) -> str:
        return evidence_key([[node.node_id for node in self.nodes], self.text])

    def node(self, alias: str) -> str | None:
        matched = re.fullmatch(r"n(0|[1-9][0-9]{0,2})", alias)
        if matched is None:
            return None
        index = int(matched[1])
        return self.nodes[index].node_id if index < len(self.nodes) else None

    def permits(self, first: str, second: str) -> bool:
        if first == second or self.node(first) is None or self.node(second) is None:
            return False
        return not self.left_size or (
            (int(first[1:]) < self.left_size) != (int(second[1:]) < self.left_size)
        )


def identity_batches(nodes: Sequence[IdentityNode]) -> Iterator[IdentityBatch]:
    """Cover every unordered pair once, with at most two 64-node blocks per request."""
    blocks = [
        tuple(nodes[start : start + IDENTITY_BLOCK_SIZE])
        for start in range(0, len(nodes), IDENTITY_BLOCK_SIZE)
    ]
    for index, block in enumerate(blocks):
        if len(block) > 1:
            yield IdentityBatch(block, 0)
        for other in blocks[index + 1 :]:
            yield IdentityBatch(block + other, len(block))


@dataclass(frozen=True, slots=True)
class MergeHint:
    first: str
    second: str
    severity: str
    description: str

    def as_record(self, batch: IdentityBatch) -> dict[str, object]:
        return {
            "type": "merge_nodes",
            "node_a": batch.node(self.first) or self.first,
            "node_b": batch.node(self.second) or self.second,
            "severity": self.severity,
            "description": self.description,
        }


@dataclass(frozen=True, slots=True)
class ProposalReply:
    hints: tuple[MergeHint, ...]
    summary: str
    answered: bool
