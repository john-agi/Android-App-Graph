"""Tests for android_app_graph.embedding_cache.

The single shared implementation of graph-file discovery and the image-embedding
sidecar cache used by both adapters.aitk_translator (runtime) and commands.embed
(offline precomputation).
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import IO, Any

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


def test_save_image_embeddings_is_atomic_on_a_failed_dump(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash mid-dump must leave the sidecar as either the previous complete
    file or the new complete file, never a truncated mix of the two, and must
    not leave a stray temporary file behind.
    """
    graph_path = tmp_path / "demo.json"
    embedding_cache.save_image_embeddings(graph_path, {"n1": [1.0, 2.0]})
    sidecar = embedding_cache.image_embeddings_path(graph_path)
    original = sidecar.read_text(encoding="utf-8")

    def _dump_then_blow_up(_obj: object, fp: IO[str], **_kwargs: object) -> None:
        fp.write('{"n9": [0.0')  # a partial write, as a real crash mid-dump would leave
        msg = "boom"
        raise ValueError(msg)

    monkeypatch.setattr(embedding_cache.json, "dump", _dump_then_blow_up)

    with pytest.raises(ValueError, match="boom"):
        embedding_cache.save_image_embeddings(graph_path, {"n1": [9.9]})

    assert sidecar.read_text(encoding="utf-8") == original
    assert list(tmp_path.iterdir()) == [sidecar]


def test_save_image_embeddings_unlinks_the_temp_file_when_the_replace_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed ``os.replace`` (EPERM, EBUSY, target is a directory, ...) must not
    orphan the temp file in the graph directory: an unguarded replace leaves a
    stray ``demo.image_emb.json.<random>.tmp`` behind, and every later run adds
    another one.
    """
    graph_path = tmp_path / "demo.json"
    embedding_cache.save_image_embeddings(graph_path, {"n1": [1.0, 2.0]})
    sidecar = embedding_cache.image_embeddings_path(graph_path)
    original = sidecar.read_text(encoding="utf-8")

    def _raise_replace(_src: object, _dst: object) -> None:
        msg = "Device or resource busy"
        raise OSError(msg)

    monkeypatch.setattr(embedding_cache.os, "replace", _raise_replace)

    with pytest.raises(OSError, match="Device or resource busy"):
        embedding_cache.save_image_embeddings(graph_path, {"n1": [9.9]})

    assert sidecar.read_text(encoding="utf-8") == original
    assert list(tmp_path.iterdir()) == [sidecar]


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


def test_load_image_embeddings_drops_an_element_too_large_for_a_float(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A 400-digit JSON integer raises OverflowError out of float(), not ValueError:
    it must still be dropped as a malformed entry, not escape ``(OSError, ValueError)``
    and take the whole app graph down with it.
    """
    graph_path = tmp_path / "demo.json"
    embedding_cache.image_embeddings_path(graph_path).write_text(
        json.dumps({"n1": [1.0, 2.0], "n2": [10**400]}),
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


def test_load_image_embeddings_warns_when_the_sidecar_is_not_an_object(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """``null`` parses cleanly but is not a mapping; ``as_str_dict`` would silently
    turn it into ``{}`` like every other case, unlike a corrupt or unreadable
    sidecar, which is warned about by name.
    """
    graph_path = tmp_path / "demo.json"
    embedding_cache.image_embeddings_path(graph_path).write_text("null", encoding="utf-8")
    with caplog.at_level("WARNING"):
        assert embedding_cache.load_image_embeddings(graph_path) == {}
    assert "demo.image_emb.json" in caplog.text


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


def test_reference_screenshot_path_finds_the_node_in_the_app_directory(tmp_path: Path) -> None:
    """Only the app directory exists (a plain, never-audited graph)."""
    app_dir = tmp_path / "demo"
    app_dir.mkdir()
    app_screenshots = app_dir / "demo_screenshots"
    app_screenshots.mkdir()
    (app_screenshots / "n1.png").write_bytes(b"shot")
    graph_path = app_dir / "demo.json"
    assert embedding_cache.reference_screenshot_path(graph_path, "n1") == app_screenshots / "n1.png"


def test_reference_screenshot_path_prefers_the_stem_directory_for_a_reexplored_node(
    tmp_path: Path,
) -> None:
    """``GraphManager.save_graph`` writes a re-explored node's screenshot into
    ``<stem>_screenshots``; when the node is there it must win over the older
    copy that may still sit under the app-name directory.
    """
    app_dir = tmp_path / "demo"
    app_dir.mkdir()
    app_screenshots = app_dir / "demo_screenshots"
    app_screenshots.mkdir()
    (app_screenshots / "n1.png").write_bytes(b"stale")
    stem_screenshots = app_dir / "demo_audited_screenshots"
    stem_screenshots.mkdir()
    (stem_screenshots / "n1.png").write_bytes(b"fresh")
    graph_path = app_dir / "demo_audited.json"
    assert (
        embedding_cache.reference_screenshot_path(graph_path, "n1") == stem_screenshots / "n1.png"
    )


def test_reference_screenshot_path_falls_back_when_the_node_is_only_in_the_app_directory(
    tmp_path: Path,
) -> None:
    """Both directories exist (some nodes were re-explored, this one was not),
    so a node missing from ``<stem>_screenshots`` must still resolve to its
    screenshot under the app-name directory rather than being reported missing.
    """
    app_dir = tmp_path / "demo"
    app_dir.mkdir()
    app_screenshots = app_dir / "demo_screenshots"
    app_screenshots.mkdir()
    (app_screenshots / "n2.png").write_bytes(b"shot")
    (app_dir / "demo_audited_screenshots").mkdir()
    graph_path = app_dir / "demo_audited.json"
    assert embedding_cache.reference_screenshot_path(graph_path, "n2") == app_screenshots / "n2.png"


def test_reference_screenshot_path_does_not_fall_back_for_a_sibling_graph(
    tmp_path: Path,
) -> None:
    """A sibling graph in the same app directory (an operator's ``demo_v1.json``
    kept next to ``demo.json``) must never borrow another graph's screenshot for
    a colliding node id. Only the plain graph and its audited pair produce the
    split layout the app-directory fallback exists for.
    """
    app_dir = tmp_path / "demo"
    app_dir.mkdir()
    app_screenshots = app_dir / "demo_screenshots"
    app_screenshots.mkdir()
    (app_screenshots / "s0.png").write_bytes(b"shot")
    graph_path = app_dir / "demo_v1.json"
    assert embedding_cache.reference_screenshot_path(graph_path, "s0") is None


def test_reference_screenshot_path_returns_none_when_neither_directory_has_the_node(
    tmp_path: Path,
) -> None:
    app_dir = tmp_path / "demo"
    app_dir.mkdir()
    graph_path = app_dir / "demo_audited.json"
    assert embedding_cache.reference_screenshot_path(graph_path, "n1") is None


def test_reference_screenshot_b64_reads_and_encodes_the_file(tmp_path: Path) -> None:
    app_dir = tmp_path / "demo"
    app_dir.mkdir()
    screenshots = app_dir / "demo_screenshots"
    screenshots.mkdir()
    (screenshots / "n1.png").write_bytes(b"shot-bytes")
    graph_path = app_dir / "demo.json"
    assert embedding_cache.reference_screenshot_b64(graph_path, "n1") == base64.b64encode(
        b"shot-bytes"
    ).decode("ascii")


def test_reference_screenshot_b64_is_none_without_a_screenshot(tmp_path: Path) -> None:
    app_dir = tmp_path / "demo"
    app_dir.mkdir()
    graph_path = app_dir / "demo.json"
    assert embedding_cache.reference_screenshot_b64(graph_path, "n1") is None


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


def test_resolve_image_embedding_settings_defaults() -> None:
    settings = embedding_cache.resolve_image_embedding_settings({})
    assert settings.model == "gemini-embedding-2"
    assert settings.api_key is None
    assert settings.base_url == "https://generativelanguage.googleapis.com/v1beta"


def test_resolve_image_embedding_settings_reads_config_values() -> None:
    settings = embedding_cache.resolve_image_embedding_settings(
        {
            "image_embedding": {
                "model": "config-model",
                "api_key": "config-key",
                "native_base_url": "https://config.googleapis.com/v1",
            }
        }
    )
    assert settings == embedding_cache.ImageEmbeddingSettings(
        api_key="config-key", model="config-model", base_url="https://config.googleapis.com/v1"
    )


def test_resolve_image_embedding_settings_base_url_defaults_to_google() -> None:
    """A base URL that is not a googleapis.com host falls back to the native
    default, since get_gemini_native_image_embedding only speaks that API.
    """
    settings = embedding_cache.resolve_image_embedding_settings(
        {"image_embedding": {"base_url": "http://localhost:9000/v1"}}
    )
    assert settings.base_url == "https://generativelanguage.googleapis.com/v1beta"


def test_resolve_image_embedding_settings_keeps_a_google_base_url_override() -> None:
    settings = embedding_cache.resolve_image_embedding_settings(
        {"image_embedding": {"native_base_url": "https://eu.googleapis.com/v1beta"}}
    )
    assert settings.base_url == "https://eu.googleapis.com/v1beta"


def test_resolve_image_embedding_settings_reads_the_api_key_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "from-env")
    settings = embedding_cache.resolve_image_embedding_settings({})
    assert settings.api_key == "from-env"


def test_resolve_image_embedding_settings_overrides_win_over_config() -> None:
    settings = embedding_cache.resolve_image_embedding_settings(
        {
            "image_embedding": {
                "model": "config-model",
                "api_key": "config-key",
                "native_base_url": "https://config.googleapis.com/v1",
            }
        },
        model_override="cli-model",
        api_key_override="cli-key",
        base_url_override="https://cli.example.com/v1",
    )
    # A base URL override that is not a googleapis.com host still falls back to
    # the native default, same as one read from config.
    assert settings == embedding_cache.ImageEmbeddingSettings(
        api_key="cli-key",
        model="cli-model",
        base_url="https://generativelanguage.googleapis.com/v1beta",
    )
