"""Tests for android_app_graph.embedding_cache.

The image-embedding sidecar cache and its compute loop, shared by
adapters.aitk_translator (runtime) and commands.embed (offline
precomputation) so they cannot drift.
"""

from __future__ import annotations

import json
import logging
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
    assert (
        embedding_cache.load_image_embeddings(
            graph_path, model="gemini-embedding-2", node_ids=set()
        )
        == {}
    )
    embedding_cache.save_image_embeddings(
        graph_path, {"n1": [0.5, 0.25]}, model="gemini-embedding-2"
    )
    sidecar = json.loads(
        embedding_cache.image_embeddings_path(graph_path).read_text(encoding="utf-8")
    )
    assert sidecar == {"model": "gemini-embedding-2", "embeddings": {"n1": [0.5, 0.25]}}
    assert embedding_cache.load_image_embeddings(
        graph_path, model="gemini-embedding-2", node_ids={"n1"}
    ) == {"n1": [0.5, 0.25]}


def test_load_image_embeddings_drops_a_vector_for_a_node_not_in_node_ids(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Both callers (the runtime translator and offline precomputation) write
    this dict straight back to the sidecar, so an entry for a node id that
    vanished from the graph since it was cached must not survive being loaded
    back in -- pruning to the caller's current node ids happens once, here,
    so the two cannot drift on what the rewritten sidecar ends up holding.

    Logged at DEBUG, not INFO: a load stays a read, so this must never be the
    line an operator sees repeat on every start of a fully cached graph --
    only the compute loop's own eventual write is worth an INFO line.
    """
    graph_path = tmp_path / "demo.json"
    embedding_cache.save_image_embeddings(
        graph_path, {"n1": [0.1, 0.2], "gone": [0.3, 0.4]}, model="gemini-embedding-2"
    )

    with caplog.at_level("DEBUG"):
        result = embedding_cache.load_image_embeddings(
            graph_path, model="gemini-embedding-2", node_ids={"n1"}
        )

    assert result == {"n1": [0.1, 0.2]}
    assert "dropping 1 vector" in caplog.text
    assert not [r for r in caplog.records if r.levelno >= logging.INFO]


def test_load_image_embeddings_returns_empty_for_a_different_model(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A dimension match cannot prove two vectors came from the same model, so a
    tagged sidecar written by a different model must be treated as empty --
    not scored, not silently reused -- so every node is recomputed as missing.
    """
    graph_path = tmp_path / "demo.json"
    embedding_cache.save_image_embeddings(graph_path, {"n1": [0.5, 0.25]}, model="model-a")

    with caplog.at_level("WARNING"):
        result = embedding_cache.load_image_embeddings(graph_path, model="model-b", node_ids={"n1"})

    assert result == {}
    assert "model-a" in caplog.text
    assert "model-b" in caplog.text
    assert len([r for r in caplog.records if r.levelname == "WARNING"]) == 1


def test_load_image_embeddings_discards_a_legacy_bare_dict(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A sidecar written before model tagging existed has no "model" key at all,
    so its vectors carry no record of which model produced them -- accepting
    them would assert a provenance they never had. They are recomputed under
    the current model instead, the same as a sidecar tagged with a different
    model, not silently reused.
    """
    graph_path = tmp_path / "demo.json"
    embedding_cache.image_embeddings_path(graph_path).write_text(
        json.dumps({"n1": [0.5, 0.25]}), encoding="utf-8"
    )

    with caplog.at_level("WARNING"):
        result = embedding_cache.load_image_embeddings(
            graph_path, model="gemini-embedding-2", node_ids={"n1"}
        )

    assert result == {}
    assert "predates model tagging" in caplog.text


def test_save_image_embeddings_goes_through_write_json_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """save_image_embeddings must delegate the write to
    graph_files.write_json_atomically rather than duplicating its mechanism --
    that mechanism's own behaviour (temp-file-and-replace, mode preservation,
    cleanup on failure) is covered once, in tests/test_graph_files.py.
    """
    graph_path = tmp_path / "demo.json"
    calls: list[tuple[Path, object, bool]] = []
    original = embedding_cache.write_json_atomically

    def _tracking(
        path: Path, payload: object, *, indent: int | None = None, ensure_ascii: bool = True
    ) -> None:
        calls.append((path, payload, ensure_ascii))
        original(path, payload, indent=indent, ensure_ascii=ensure_ascii)

    monkeypatch.setattr(embedding_cache, "write_json_atomically", _tracking)

    embedding_cache.save_image_embeddings(graph_path, {"n1": [0.5]}, model="gemini-embedding-2")

    assert calls == [
        (
            embedding_cache.image_embeddings_path(graph_path),
            {"model": "gemini-embedding-2", "embeddings": {"n1": [0.5]}},
            False,
        )
    ]
    assert embedding_cache.load_image_embeddings(
        graph_path, model="gemini-embedding-2", node_ids={"n1"}
    ) == {"n1": [0.5]}


def test_load_image_embeddings_without_a_sidecar(tmp_path: Path) -> None:
    assert (
        embedding_cache.load_image_embeddings(
            tmp_path / "demo.json", model="gemini-embedding-2", node_ids=set()
        )
        == {}
    )


def test_load_image_embeddings_drops_malformed_entries(tmp_path: Path) -> None:
    graph_path = tmp_path / "demo.json"
    embedding_cache.image_embeddings_path(graph_path).write_text(
        json.dumps(
            {
                "model": "gemini-embedding-2",
                "embeddings": {
                    "n1": [1.0, 2.0],
                    "n2": "not a vector",
                    "n3": [],
                    "n4": [1.0, "two"],
                },
            }
        ),
        encoding="utf-8",
    )
    assert embedding_cache.load_image_embeddings(
        graph_path, model="gemini-embedding-2", node_ids={"n1", "n2", "n3", "n4"}
    ) == {"n1": [1.0, 2.0]}


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
        json.dumps(
            {
                "model": "gemini-embedding-2",
                "embeddings": {"n1": [1.0, 2.0], "n2": [1.0, float("nan")]},
            }
        ),
        encoding="utf-8",
    )
    with caplog.at_level("WARNING"):
        embeddings = embedding_cache.load_image_embeddings(
            graph_path, model="gemini-embedding-2", node_ids={"n1", "n2"}
        )
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
        json.dumps(
            {"model": "gemini-embedding-2", "embeddings": {"n1": [1.0, 2.0], "n2": [10**400]}}
        ),
        encoding="utf-8",
    )
    with caplog.at_level("WARNING"):
        embeddings = embedding_cache.load_image_embeddings(
            graph_path, model="gemini-embedding-2", node_ids={"n1", "n2"}
        )
    assert embeddings == {"n1": [1.0, 2.0]}
    assert "n2" in caplog.text


def test_load_image_embeddings_warns_about_each_dropped_entry(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    graph_path = tmp_path / "demo.json"
    embedding_cache.image_embeddings_path(graph_path).write_text(
        json.dumps(
            {"model": "gemini-embedding-2", "embeddings": {"n1": [1.0, 2.0], "n2": "not a vector"}}
        ),
        encoding="utf-8",
    )
    with caplog.at_level("WARNING"):
        embedding_cache.load_image_embeddings(
            graph_path, model="gemini-embedding-2", node_ids={"n1", "n2"}
        )
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
        assert (
            embedding_cache.load_image_embeddings(
                graph_path, model="gemini-embedding-2", node_ids=set()
            )
            == {}
        )
    assert "demo.image_emb.json" in caplog.text


def test_load_image_embeddings_treats_a_corrupt_sidecar_as_no_cache(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    graph_path = tmp_path / "demo.json"
    embedding_cache.image_embeddings_path(graph_path).write_text("{not json", encoding="utf-8")
    with caplog.at_level("WARNING"):
        assert (
            embedding_cache.load_image_embeddings(
                graph_path, model="gemini-embedding-2", node_ids=set()
            )
            == {}
        )
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
        assert (
            embedding_cache.load_image_embeddings(
                graph_path, model="gemini-embedding-2", node_ids={"n1"}
            )
            == {}
        )
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
        assert (
            embedding_cache.load_image_embeddings(
                graph_path, model="gemini-embedding-2", node_ids=set()
            )
            == {}
        )
    assert "demo.image_emb.json" in caplog.text


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


def test_resolve_image_embedding_settings_resolves_an_env_reference_base_url_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--base-url '${GEMINI_BASE_URL}'`` must resolve through the environment
    like ``model_override`` and ``api_key_override`` already do, not keep the
    literal ``${...}`` string (which fails the googleapis.com check and
    silently falls back to the default endpoint).
    """
    monkeypatch.setenv("GEMINI_BASE_URL", "https://eu.googleapis.com/v1beta")
    settings = embedding_cache.resolve_image_embedding_settings(
        {}, base_url_override="${GEMINI_BASE_URL}"
    )
    assert settings.base_url == "https://eu.googleapis.com/v1beta"


def test_resolve_image_embedding_settings_falls_back_for_an_unresolved_or_non_google_env_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GEMINI_BASE_URL", raising=False)
    settings = embedding_cache.resolve_image_embedding_settings(
        {}, base_url_override="${GEMINI_BASE_URL}"
    )
    assert settings.base_url == "https://generativelanguage.googleapis.com/v1beta"

    monkeypatch.setenv("GEMINI_BASE_URL", "https://not-google.example.com/v1")
    settings = embedding_cache.resolve_image_embedding_settings(
        {}, base_url_override="${GEMINI_BASE_URL}"
    )
    assert settings.base_url == "https://generativelanguage.googleapis.com/v1beta"


def test_load_image_embeddings_treats_a_legacy_node_named_model_as_untagged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The tagged form is recognised by a string tag plus an embeddings object,
    not by the presence of a "model" key alone: a legacy bare dict may hold a
    node called "model" whose value is a vector, not a model-name string. That
    shape is still the untagged legacy form -- discarded like any other
    untagged sidecar, not misread as a mismatched tag.
    """
    graph_path = tmp_path / "demo.json"
    embedding_cache.image_embeddings_path(graph_path).write_text(
        json.dumps({"model": [1.0, 0.0], "n1": [0.0, 1.0]}), encoding="utf-8"
    )
    with caplog.at_level("WARNING"):
        loaded = embedding_cache.load_image_embeddings(
            graph_path, model="img-model", node_ids={"model", "n1"}
        )
    assert loaded == {}
    assert "predates model tagging" in caplog.text
