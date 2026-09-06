"""Tests for android_app_graph.embedding_cache.

The single shared implementation of graph-file discovery and the image-embedding
sidecar cache used by both adapters.aitk_translator (runtime) and commands.embed
(offline precomputation).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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


def test_load_image_embeddings_drops_a_nan_entry(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A cached vector containing the literal ``NaN`` is rejected, not loaded.

    json.load happily parses ``NaN``; a non-finite value must not reach
    cosine_similarity, where a naive clamp would score it as the best match
    everywhere.
    """
    graph_path = tmp_path / "demo.json"
    embedding_cache.image_embeddings_path(graph_path).write_text(
        json.dumps({"n1": [1.0, 2.0], "n2": [1.0, float("nan")]}),
        encoding="utf-8",
    )
    with caplog.at_level("WARNING"):
        embeddings = embedding_cache.load_image_embeddings(graph_path)
    assert embeddings == {"n1": [1.0, 2.0]}
    assert "n2" in caplog.text


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


def test_load_image_embeddings_treats_invalid_utf8_as_no_cache(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Invalid UTF-8 in the sidecar is an empty cache, like malformed JSON.

    UnicodeDecodeError is a ValueError like JSONDecodeError and must never be a
    reason for the per-app ``except Exception`` in ``_load_all_graphs`` to drop
    the whole graph.
    """
    graph_path = tmp_path / "demo.json"
    embedding_cache.image_embeddings_path(graph_path).write_bytes(b'{"n1": [1.0]}\xff\xfe')
    with caplog.at_level("WARNING"):
        assert embedding_cache.load_image_embeddings(graph_path) == {}
    assert "demo.image_emb.json" in caplog.text


def test_load_image_embeddings_treats_an_unreadable_file_as_no_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    graph_path = tmp_path / "demo.json"
    embedding_cache.image_embeddings_path(graph_path).write_text("{}", encoding="utf-8")

    def _raise_permission_error(_self: Path, *_args: object, **_kwargs: object) -> None:
        msg = "Permission denied"
        raise OSError(msg)

    monkeypatch.setattr(Path, "open", _raise_permission_error)
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


# ---------------------------------------------------------------------------
# compute_embedding_with_retry
# ---------------------------------------------------------------------------


def _retry(api_key: str, screenshot_b64: str) -> list[float]:
    return embedding_cache.compute_embedding_with_retry(
        api_key,
        screenshot_b64,
        model="gemini-embedding-2",
        base_url="https://generativelanguage.googleapis.com/v1beta",
        app_name="demo",
        node_id="s0_home",
    )


@pytest.mark.usefixtures("no_sleep")
def test_compute_embedding_with_retry_returns_on_first_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []

    def fake_embedding(api_key: str, screenshot_b64: str, **_kwargs: Any) -> list[float]:
        calls.append((api_key, screenshot_b64))
        return [1.0, 2.0]

    monkeypatch.setattr(embedding_cache, "get_gemini_native_image_embedding", fake_embedding)

    assert _retry("key", "shot") == [1.0, 2.0]
    assert calls == [("key", "shot")]


@pytest.mark.usefixtures("no_sleep")
def test_compute_embedding_with_retry_recovers_after_a_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts: list[int] = []

    def flaky_embedding(*_args: Any, **_kwargs: Any) -> list[float]:
        attempts.append(len(attempts))
        if len(attempts) == 1:
            msg = "429 rate limited"
            raise RuntimeError(msg)
        return [0.25]

    monkeypatch.setattr(embedding_cache, "get_gemini_native_image_embedding", flaky_embedding)

    assert _retry("key", "shot") == [0.25]
    assert len(attempts) == 2


@pytest.mark.usefixtures("no_sleep")
def test_compute_embedding_with_retry_reraises_after_the_last_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts: list[int] = []

    def always_failing(*_args: Any, **_kwargs: Any) -> list[float]:
        attempts.append(len(attempts))
        msg = "500 upstream error"
        raise RuntimeError(msg)

    monkeypatch.setattr(embedding_cache, "get_gemini_native_image_embedding", always_failing)

    with pytest.raises(RuntimeError, match="500 upstream error"):
        _retry("key", "shot")
    assert len(attempts) == embedding_cache.IMAGE_EMBEDDING_RETRIES + 1
