"""Tests for android_app_graph.embedding_cache.

The image-embedding sidecar cache and its compute loop, shared by
adapters.aitk_translator (runtime) and commands.embed (offline
precomputation) so they cannot drift.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from openrouter import OpenRouter
from openrouter.errors import OpenRouterError, ResponseValidationError

from android_app_graph import embedding_cache
from android_app_graph.utils import vlm_utils


def test_image_embeddings_path_is_a_sidecar() -> None:
    assert embedding_cache.image_embeddings_path(Path("graphs/demo/demo.json")) == Path(
        "graphs/demo/demo.image_emb.json"
    )


def test_save_and_load_image_embeddings_round_trip(tmp_path: Path) -> None:
    graph_path = tmp_path / "demo.json"
    assert (
        embedding_cache.load_image_embeddings(
            graph_path, model="google/gemini-embedding-2", node_ids=set()
        )
        == {}
    )
    embedding_cache.save_image_embeddings(
        graph_path, {"n1": [0.5, 0.25]}, model="google/gemini-embedding-2"
    )
    sidecar = json.loads(
        embedding_cache.image_embeddings_path(graph_path).read_text(encoding="utf-8")
    )
    assert sidecar == {"model": "google/gemini-embedding-2", "embeddings": {"n1": [0.5, 0.25]}}
    assert embedding_cache.load_image_embeddings(
        graph_path, model="google/gemini-embedding-2", node_ids={"n1"}
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
        graph_path, {"n1": [0.1, 0.2], "gone": [0.3, 0.4]}, model="google/gemini-embedding-2"
    )

    with caplog.at_level("DEBUG"):
        result = embedding_cache.load_image_embeddings(
            graph_path, model="google/gemini-embedding-2", node_ids={"n1"}
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
            graph_path, model="google/gemini-embedding-2", node_ids={"n1"}
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

    embedding_cache.save_image_embeddings(
        graph_path, {"n1": [0.5]}, model="google/gemini-embedding-2"
    )

    assert calls == [
        (
            embedding_cache.image_embeddings_path(graph_path),
            {"model": "google/gemini-embedding-2", "embeddings": {"n1": [0.5]}},
            False,
        )
    ]
    assert embedding_cache.load_image_embeddings(
        graph_path, model="google/gemini-embedding-2", node_ids={"n1"}
    ) == {"n1": [0.5]}


def test_load_image_embeddings_without_a_sidecar(tmp_path: Path) -> None:
    assert (
        embedding_cache.load_image_embeddings(
            tmp_path / "demo.json", model="google/gemini-embedding-2", node_ids=set()
        )
        == {}
    )


def test_load_image_embeddings_drops_malformed_entries(tmp_path: Path) -> None:
    graph_path = tmp_path / "demo.json"
    embedding_cache.image_embeddings_path(graph_path).write_text(
        json.dumps(
            {
                "model": "google/gemini-embedding-2",
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
        graph_path, model="google/gemini-embedding-2", node_ids={"n1", "n2", "n3", "n4"}
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
                "model": "google/gemini-embedding-2",
                "embeddings": {"n1": [1.0, 2.0], "n2": [1.0, float("nan")]},
            }
        ),
        encoding="utf-8",
    )
    with caplog.at_level("WARNING"):
        embeddings = embedding_cache.load_image_embeddings(
            graph_path, model="google/gemini-embedding-2", node_ids={"n1", "n2"}
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
            {
                "model": "google/gemini-embedding-2",
                "embeddings": {"n1": [1.0, 2.0], "n2": [10**400]},
            }
        ),
        encoding="utf-8",
    )
    with caplog.at_level("WARNING"):
        embeddings = embedding_cache.load_image_embeddings(
            graph_path, model="google/gemini-embedding-2", node_ids={"n1", "n2"}
        )
    assert embeddings == {"n1": [1.0, 2.0]}
    assert "n2" in caplog.text


def test_load_image_embeddings_warns_about_each_dropped_entry(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    graph_path = tmp_path / "demo.json"
    embedding_cache.image_embeddings_path(graph_path).write_text(
        json.dumps(
            {
                "model": "google/gemini-embedding-2",
                "embeddings": {"n1": [1.0, 2.0], "n2": "not a vector"},
            }
        ),
        encoding="utf-8",
    )
    with caplog.at_level("WARNING"):
        embedding_cache.load_image_embeddings(
            graph_path, model="google/gemini-embedding-2", node_ids={"n1", "n2"}
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
                graph_path, model="google/gemini-embedding-2", node_ids=set()
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
                graph_path, model="google/gemini-embedding-2", node_ids=set()
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
                graph_path, model="google/gemini-embedding-2", node_ids={"n1"}
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
                graph_path, model="google/gemini-embedding-2", node_ids=set()
            )
            == {}
        )
    assert "demo.image_emb.json" in caplog.text


IMAGE_SETTINGS = embedding_cache.ImageEmbeddingSettings("google/gemini-embedding-2")


def _retry(client: OpenRouter, screenshot_b64: str) -> list[float]:
    return embedding_cache.compute_embedding_with_retry(
        client, screenshot_b64, settings=IMAGE_SETTINGS, app_name="demo", node_id="s0_home"
    )


@pytest.mark.usefixtures("no_sleep")
def test_compute_embedding_with_retry_returns_on_first_success(
    router_client: OpenRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[OpenRouter, str, str]] = []

    def fake_embedding(
        client: OpenRouter, screenshot_b64: str, model: str, **_kwargs: Any
    ) -> list[float]:
        calls.append((client, screenshot_b64, model))
        return [1.0, 2.0]

    monkeypatch.setattr(embedding_cache, "get_image_embedding", fake_embedding)

    assert _retry(router_client, "shot") == [1.0, 2.0]
    assert calls == [(router_client, "shot", "google/gemini-embedding-2")]


@pytest.mark.usefixtures("model_environment")
def test_resolve_image_embedding_settings_from_environment() -> None:
    assert embedding_cache.resolve_image_embedding_settings(
        {}
    ) == embedding_cache.ImageEmbeddingSettings(model="google/gemini-embedding-2")


def test_resolve_image_embedding_settings_reads_config_values() -> None:
    settings = embedding_cache.resolve_image_embedding_settings(
        {"image_embedding": {"model": "config-model", "request_timeout": 5}}
    )
    assert settings == embedding_cache.ImageEmbeddingSettings(
        model="config-model", request_timeout=5.0
    )


def test_resolve_image_embedding_settings_overrides_win_over_config() -> None:
    settings = embedding_cache.resolve_image_embedding_settings(
        {"image_embedding": {"model": "config-model"}}, model_override="cli-model"
    )
    assert settings == embedding_cache.ImageEmbeddingSettings(model="cli-model")


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


def test_image_settings_inherit_the_embedding_model_until_they_set_their_own() -> None:
    config: dict[str, object] = {"embedding": {"model": "google/gemini-embedding-2"}}
    assert embedding_cache.resolve_image_embedding_settings(
        config
    ) == embedding_cache.ImageEmbeddingSettings("google/gemini-embedding-2")
    config["image_embedding"] = {"model": "provider/image-model"}
    assert embedding_cache.resolve_image_embedding_settings(
        config
    ) == embedding_cache.ImageEmbeddingSettings("provider/image-model")


def test_image_cache_retry_uses_real_sdk_and_counts_usage(
    router_client: OpenRouter,
    router_requests: list[httpx.Request],
    router_responses: list[httpx.Response],
    embedding_body: dict[str, object],
    no_sleep: list[float],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracker = vlm_utils.TokenTracker()
    monkeypatch.setattr(vlm_utils, "token_tracker", tracker)
    router_responses.extend(
        [
            httpx.Response(429, json={"error": {"code": 429, "message": "rate limited"}}),
            httpx.Response(200, json=embedding_body),
        ]
    )
    assert _retry(router_client, "shot") == [0.25, 0.75]
    assert len(router_requests) == 2
    assert no_sleep == [2.0]
    assert tracker.snapshot() == {
        "prompt_tokens": 5,
        "completion_tokens": 0,
        "total_tokens": 5,
        "calls": 1,
    }


@pytest.mark.parametrize(("status", "attempts"), [(429, 3), (400, 1)])
def test_image_cache_sdk_errors_are_bounded(
    status: int,
    attempts: int,
    router_client: OpenRouter,
    router_requests: list[httpx.Request],
    router_responses: list[httpx.Response],
    no_sleep: list[float],
) -> None:
    router_responses.extend(
        httpx.Response(status, json={"error": {"code": status, "message": "provider failure"}})
        for _ in range(attempts)
    )
    with pytest.raises(OpenRouterError, match="provider failure"):
        _retry(router_client, "shot")
    assert len(router_requests) == attempts
    assert no_sleep == ([2.0, 4.0] if status == 429 else [])


@pytest.mark.parametrize("recover", [False, True])
def test_sdk_validation_uses_bounded_application_recovery(
    recover: bool,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    embedding_body: dict[str, object],
    png_screenshot: str,
    no_sleep: list[float],
) -> None:
    bad = {key: value for key, value in embedding_body.items() if key != "model"}
    router_responses.extend(httpx.Response(200, json=bad) for _ in range(1 if recover else 3))
    if recover:
        router_responses.append(httpx.Response(200, json=embedding_body))

    def request() -> list[float]:
        return _retry(router_client, png_screenshot)

    if recover:
        assert request() == [0.25, 0.75]
    else:
        with pytest.raises(ResponseValidationError) as caught:
            request()
        assert caught.value.status_code == 200
    assert len(router_requests) == (2 if recover else 3)
    assert no_sleep == ([2.0] if recover else [2.0, 4.0])


@pytest.mark.parametrize("interrupted", [False, True])
def test_screenshot_batch_uses_the_callers_client_and_keeps_partial_progress(
    interrupted: bool,
    tmp_path: Path,
    router_client: OpenRouter,
    router_http: httpx.Client,
    router_requests: list[httpx.Request],
    router_responses: list[httpx.Response],
    embedding_body: dict[str, object],
    no_sleep: list[float],
) -> None:
    graph = tmp_path / "demo.json"
    cache: dict[str, list[float]] = {}
    router_responses.extend(
        [
            httpx.Response(503, json={"error": {"code": 503, "message": "busy"}}),
            httpx.Response(200, json=embedding_body),
            httpx.Response(200, json=embedding_body),
        ]
    )

    def candidates() -> Iterator[tuple[str, str]]:
        yield "n1", "shot-1"
        if interrupted:
            raise KeyboardInterrupt
        yield "n2", "shot-2"

    def compute() -> None:
        embedding_cache.compute_missing_image_embeddings(
            graph,
            cache,
            candidates(),
            client=router_client,
            settings=embedding_cache.ImageEmbeddingSettings("model"),
            app_name="demo",
        )

    if interrupted:
        with pytest.raises(KeyboardInterrupt):
            compute()
    else:
        compute()
    assert len(router_requests) == (2 if interrupted else 3)
    assert no_sleep == [2.0]
    assert router_client.sdk_configuration.client is router_http
    assert (
        embedding_cache.load_image_embeddings(graph, model="model", node_ids={"n1", "n2"}) == cache
    )
    assert set(cache) == ({"n1"} if interrupted else {"n1", "n2"})


def test_image_batches_send_their_own_model_and_timeout(
    tmp_path: Path,
    router_client: OpenRouter,
    router_responses: list[httpx.Response],
    router_requests: list[httpx.Request],
    embedding_body: dict[str, object],
) -> None:
    router_responses.extend(httpx.Response(200, json=embedding_body) for _ in range(4))
    for index in range(2):
        result = embedding_cache.compute_missing_image_embeddings(
            tmp_path / f"graph-{index}.json",
            {},
            [("n1", "one"), ("n2", "two")],
            client=router_client,
            settings=embedding_cache.ImageEmbeddingSettings(f"model-{index}", index + 1.5),
            app_name="demo",
        )
        assert result.computed == 2
    for index, request in enumerate(router_requests):
        assert request.headers["authorization"] == "Bearer test-router-key"
        assert request.extensions["timeout"]["read"] == index // 2 + 1.5
        assert str(request.url) == "https://openrouter.ai/api/v1/embeddings"
        assert json.loads(request.content)["model"] == f"model-{index // 2}"


def test_image_embedding_requires_a_model() -> None:
    with pytest.raises(ValueError, match=r"image_embedding.*model"):
        embedding_cache.resolve_image_embedding_settings({})


@pytest.mark.parametrize("value", ["", "  ", "${APP_GRAPH_UNSET_MODEL}"])
def test_explicit_invalid_image_model_does_not_inherit(value: str) -> None:
    with pytest.raises(ValueError, match=r"image_embedding.*model"):
        embedding_cache.resolve_image_embedding_settings(
            {"embedding": {"model": "provider/text"}, "image_embedding": {"model": value}}
        )


@pytest.mark.parametrize("value", ["", "  ", "${APP_GRAPH_UNSET_MODEL}"])
def test_invalid_cli_image_model_cannot_fall_back_to_configuration(value: str) -> None:
    with pytest.raises(ValueError, match=r"image_embedding.*model"):
        embedding_cache.resolve_image_embedding_settings(
            {"image_embedding": {"model": "provider/image"}}, model_override=value
        )


@pytest.mark.parametrize(
    ("config", "timeout"),
    [
        ({}, 60.0),
        ({"embedding": {"request_timeout": 7.5}, "image_embedding": {}}, 7.5),
        (
            {"embedding": {"request_timeout": 7.5}, "image_embedding": {"request_timeout": 2.25}},
            2.25,
        ),
        ({"embedding": {"request_timeout": 7.5}, "image_embedding": {"timeout": 2.25}}, 2.25),
    ],
)
@pytest.mark.usefixtures("model_environment")
def test_image_settings_resolve_their_request_timeout(
    config: dict[str, object], timeout: float
) -> None:
    assert embedding_cache.resolve_image_embedding_settings(config).request_timeout == timeout


@pytest.mark.parametrize("key", ["api_key", "base_url", "native_base_url"])
@pytest.mark.parametrize("role", ["embedding", "image_embedding"])
@pytest.mark.usefixtures("model_environment")
def test_resolver_rejects_removed_endpoint_and_credential_keys(
    role: str, key: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "dummy-google-secret-never-log"
    monkeypatch.setenv("GEMINI_API_KEY", secret)
    legacy = {key: "${GEMINI_API_KEY}", "model": "gemini-embedding-2"}
    with pytest.raises(
        ValueError, match=rf"^Invalid vlm configuration: {role}\.{key} was removed"
    ) as caught:
        embedding_cache.resolve_image_embedding_settings({role: legacy})
    assert secret not in str(caught.value)
