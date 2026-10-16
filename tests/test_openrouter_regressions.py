"""Offline regressions for the OpenRouter migration's recovery and configuration."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
from openrouter import OpenRouter

from android_app_graph.utils.graph_manager import GraphManager

_CLICK = '{"action":"click","coordinate":[500,500]}'


@pytest.mark.parametrize("storage", ["inline", "sidecar"])
@pytest.mark.parametrize("model", [None, "provider/old-same-dimensions"])
@pytest.mark.parametrize("operation", ["find", "identify"])
def test_text_embedding_unknown_or_mismatched_provenance_blocks_comparison(
    storage: str,
    model: str | None,
    operation: str,
    tmp_path: Path,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    embedding_body: dict[str, object],
    png_screenshot: str,
    chat_response: Callable[..., httpx.Response],
) -> None:
    path = tmp_path / "demo.json"
    node: dict[str, object] = {
        "id": "home",
        "activity": "com.demo.Home",
        "page_description": "home",
    }
    if storage == "inline":
        node["description_embedding"] = [0.25, 0.75]
    if model is not None:
        node["description_embedding_model"] = model
    path.write_text(json.dumps({"nodes": [node], "edges": []}), encoding="utf-8")
    if storage == "sidecar":
        path.with_suffix(".emb.json").write_text(
            json.dumps(
                {
                    "home": [0.25, 0.75]
                    if model is None
                    else {"model": model, "embedding": [0.25, 0.75]}
                }
            ),
            encoding="utf-8",
        )
    manager = GraphManager(
        page_detail_client=router_client,
        embedding_client=router_client,
        page_detail_model="openai/gpt-6.1-sol",
        embedding_model="google/gemini-embedding-2",
    )
    manager.load_graph(path)
    router_responses.extend(
        [
            chat_response(content='{"page_description":"home","state":{},"elements":[]}'),
            httpx.Response(200, json=embedding_body),
        ]
        if operation == "identify"
        else [httpx.Response(200, json=embedding_body)]
    )

    def compare() -> None:
        if operation == "find":
            manager.find_node_by_description("home")
        else:
            manager.identify_state("com.demo.Home", png_screenshot)

    for _ in range(2):
        with pytest.raises(ValueError, match=r"Text embedding.*Rebuild or re-explore"):
            compare()
        assert router_requests == []


def test_text_embedding_model_survives_save_load(
    tmp_path: Path,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    embedding_body: dict[str, object],
) -> None:
    manager = GraphManager(
        embedding_client=router_client, embedding_model="google/gemini-embedding-2"
    )
    manager.graph.add_node(
        "home",
        description_embedding=[0.25, 0.75],
        description_embedding_model=manager.embedding_model,
    )
    path = tmp_path / "demo.json"
    manager.save_graph(path)
    saved = json.loads(path.with_suffix(".emb.json").read_text(encoding="utf-8"))
    assert saved["home"]["model"] == "google/gemini-embedding-2"
    manager.load_graph(path)
    router_responses.append(httpx.Response(200, json=embedding_body))
    assert manager.find_node_by_description("home") == [("home", pytest.approx(1.0))]


@pytest.mark.parametrize("exceptional", [False, True])
@pytest.mark.usefixtures("model_environment")
def test_audit_cli_closes_owned_clients_on_exit(
    exceptional: bool,
    tmp_path: Path,
    owned_router_clients: list[httpx.Client],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from android_app_graph.commands import audit

    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    graph_dir = tmp_path / "graphs"
    app_dir = graph_dir / "demo"
    app_dir.mkdir(parents=True)
    (app_dir / "demo.json").write_text('{"nodes":[],"edges":[]}')
    config = tmp_path / "config.yaml"
    config.write_text(
        f"experiment:\n  graph_dir: {graph_dir}\napps:\n  - name: demo\n    package_name: com.demo\n"
    )

    def audit_graph(_graph: GraphManager, _app: str, _path: Path) -> dict[str, object]:
        if exceptional:
            raise RuntimeError("audit failed")
        return {"issues": []}

    monkeypatch.setattr(audit, "run_audit", audit_graph)
    monkeypatch.setattr(audit, "normalize_and_save_audited_graph", lambda *_args: 0)
    if exceptional:
        with pytest.raises(RuntimeError, match="audit failed"):
            audit.main(["-c", str(config)])
    else:
        assert audit.main(["-c", str(config)]) == 0
    assert len(owned_router_clients) == 2
    assert all(client.is_closed for client in owned_router_clients)
