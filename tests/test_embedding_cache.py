"""Tests for android_app_graph.embedding_cache.

The single shared implementation of graph-file discovery and the image-embedding
sidecar cache used by both adapters.aitk_translator (runtime) and commands.embed
(offline precomputation).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from android_app_graph import embedding_cache


def test_image_embeddings_path_is_a_sidecar() -> None:
    assert embedding_cache.image_embeddings_path(Path("graphs/demo/demo.json")) == Path(
        "graphs/demo/demo.image_emb.json"
    )


def test_save_and_load_image_embeddings_round_trip(tmp_path: Path) -> None:
    graph_path = tmp_path / "demo.json"
    assert embedding_cache.load_image_embeddings(graph_path) == {}
    embedding_cache.save_image_embeddings(graph_path, {"n1": [0.5, 0.25]})
    assert embedding_cache.load_image_embeddings(graph_path) == {"n1": [0.5, 0.25]}


def test_load_image_embeddings_without_a_sidecar(tmp_path: Path) -> None:
    assert embedding_cache.load_image_embeddings(tmp_path / "demo.json") == {}


def test_load_image_embeddings_drops_malformed_entries(tmp_path: Path) -> None:
    graph_path = tmp_path / "demo.json"
    embedding_cache.image_embeddings_path(graph_path).write_text(
        json.dumps({"n1": [1.0, 2.0], "n2": "not a vector", "n3": [], "n4": [1.0, "two"]}),
        encoding="utf-8",
    )
    assert embedding_cache.load_image_embeddings(graph_path) == {"n1": [1.0, 2.0]}


def test_load_image_embeddings_warns_about_each_dropped_entry(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    graph_path = tmp_path / "demo.json"
    embedding_cache.image_embeddings_path(graph_path).write_text(
        json.dumps({"n1": [1.0, 2.0], "n2": "not a vector"}), encoding="utf-8"
    )
    with caplog.at_level("WARNING"):
        embedding_cache.load_image_embeddings(graph_path)
    assert "n2" in caplog.text


def test_load_image_embeddings_treats_a_corrupt_sidecar_as_no_cache(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    graph_path = tmp_path / "demo.json"
    embedding_cache.image_embeddings_path(graph_path).write_text("{not json", encoding="utf-8")
    with caplog.at_level("WARNING"):
        assert embedding_cache.load_image_embeddings(graph_path) == {}
    assert "demo.image_emb.json" in caplog.text


def test_iter_graph_files_without_a_graph_dir(tmp_path: Path) -> None:
    assert embedding_cache.iter_graph_files(tmp_path / "absent") == []


def test_iter_graph_files_prefers_the_audited_graph(tmp_path: Path) -> None:
    app_dir = tmp_path / "eboox"
    app_dir.mkdir()
    (app_dir / "eboox.json").write_text("{}", encoding="utf-8")
    audited = app_dir / "eboox_audited.json"
    audited.write_text("{}", encoding="utf-8")
    assert embedding_cache.iter_graph_files(tmp_path) == [("eboox", audited)]


def test_iter_graph_files_sorts_apps_and_skips_side_files(tmp_path: Path) -> None:
    for app in ("zebra", "alpha"):
        app_dir = tmp_path / app
        app_dir.mkdir()
        (app_dir / f"{app}.json").write_text("{}", encoding="utf-8")
    (tmp_path / "alpha" / "alpha_audit_report.json").write_text("{}", encoding="utf-8")
    (tmp_path / "alpha" / "alpha.image_emb.json").write_text("{}", encoding="utf-8")
    (tmp_path / "loose.json").write_text("{}", encoding="utf-8")
    (tmp_path / "empty").mkdir()

    assert embedding_cache.iter_graph_files(tmp_path) == [
        ("alpha", tmp_path / "alpha" / "alpha.json"),
        ("zebra", tmp_path / "zebra" / "zebra.json"),
    ]


def test_iter_graph_files_can_select_one_app(tmp_path: Path) -> None:
    for app in ("demo", "other"):
        app_dir = tmp_path / app
        app_dir.mkdir()
        (app_dir / f"{app}.json").write_text("{}", encoding="utf-8")
    assert embedding_cache.iter_graph_files(tmp_path, "demo") == [
        ("demo", tmp_path / "demo" / "demo.json")
    ]


def test_iter_graph_files_skips_an_unknown_app(tmp_path: Path) -> None:
    assert embedding_cache.iter_graph_files(tmp_path, "absent") == []
