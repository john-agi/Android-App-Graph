"""app-graph-embed settings, retry and end-to-end helpers, exercised without network or keys."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import pytest

from android_app_graph import embedding_cache
from android_app_graph.commands import embed
from android_app_graph.embedding_cache import image_embeddings_path

_SCREENSHOT = b"not-really-a-png"


def _write_graph_tree(tmp_path: Path, *, app: str = "demo", audited: bool = False) -> Path:
    """Create ``<tmp_path>/<app>/`` with one graph file and one node screenshot."""
    app_dir = tmp_path / app
    app_dir.mkdir(parents=True)
    stem = f"{app}_audited" if audited else app
    (app_dir / f"{stem}.json").write_text(
        json.dumps({"nodes": [{"id": "s0_home"}, {"id": "s1_detail"}, {"id": ""}]}),
        encoding="utf-8",
    )
    screenshots = app_dir / f"{app}_screenshots"
    screenshots.mkdir()
    (screenshots / "s0_home.png").write_bytes(_SCREENSHOT)
    return app_dir / f"{stem}.json"


def test_embed_parser_defaults() -> None:
    args = embed.build_parser().parse_args([])
    assert args.config == Path("configs/explore.yaml")
    assert args.graphs is None
    assert args.app is None
    assert args.model is None
    assert args.base_url is None
    assert args.api_key is None


def test_embed_main_rejects_missing_config(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        embed.main(["-c", "does-not-exist.yaml"])
    assert excinfo.value.code == 2
    assert "config file not found" in capsys.readouterr().err


def test_embed_main_rejects_missing_graph_root(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "explore.yaml"
    config.write_text("experiment:\n  graph_dir: nowhere\n", encoding="utf-8")
    with pytest.raises(SystemExit) as excinfo:
        embed.main(["-c", str(config), "--graphs", str(tmp_path / "absent")])
    assert excinfo.value.code == 2
    assert "graph root does not exist" in capsys.readouterr().err


def test_embed_main_requires_an_api_key(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = tmp_path / "explore.yaml"
    config.write_text(f"experiment:\n  graph_dir: {tmp_path}\n", encoding="utf-8")
    with pytest.raises(SystemExit) as excinfo:
        embed.main(["-c", str(config)])
    assert excinfo.value.code == 2
    assert "missing API key" in capsys.readouterr().err


def test_embed_settings_from_config(tmp_path: Path) -> None:
    config = tmp_path / "explore.yaml"
    config.write_text("experiment:\n  graph_dir: my_graphs\n", encoding="utf-8")
    graph_dir, api_key, model, base_url = embed.load_image_embedding_settings(
        config,
        graphs_override=None,
        api_key_override=None,
        model_override=None,
        base_url_override=None,
    )
    assert graph_dir == Path("my_graphs")
    assert api_key is None
    assert model == "gemini-embedding-2"
    assert base_url == "https://generativelanguage.googleapis.com/v1beta"


def test_embed_settings_prefer_translator_args_and_overrides(tmp_path: Path) -> None:
    config = tmp_path / "controller.yaml"
    config.write_text(
        "translator_args:\n"
        "  graph_dir: from_translator\n"
        "  vlm_config:\n"
        "    image_embedding:\n"
        "      model: config-model\n"
        "      native_base_url: https://config.googleapis.com/v1\n"
        "      api_key: config-key\n",
        encoding="utf-8",
    )
    graph_dir, api_key, model, base_url = embed.load_image_embedding_settings(
        config,
        graphs_override=None,
        api_key_override=None,
        model_override=None,
        base_url_override=None,
    )
    assert graph_dir == Path("from_translator")
    assert api_key == "config-key"
    assert model == "config-model"
    assert base_url == "https://config.googleapis.com/v1"

    overridden = embed.load_image_embedding_settings(
        config,
        graphs_override=Path("cli_graphs"),
        api_key_override="cli-key",
        model_override="cli-model",
        base_url_override="https://cli.example.com/v1",
    )
    # A base URL that is not a googleapis.com host falls back to the native default.
    assert overridden == (
        Path("cli_graphs"),
        "cli-key",
        "cli-model",
        "https://generativelanguage.googleapis.com/v1beta",
    )


def test_embed_settings_read_the_api_key_from_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "from-env")
    config = tmp_path / "explore.yaml"
    config.write_text("{}\n", encoding="utf-8")
    _graph_dir, api_key, _model, _base_url = embed.load_image_embedding_settings(
        config,
        graphs_override=None,
        api_key_override=None,
        model_override=None,
        base_url_override=None,
    )
    assert api_key == "from-env"


def test_load_graph_json_rejects_a_non_object(tmp_path: Path) -> None:
    graph_path = tmp_path / "demo.json"
    graph_path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(TypeError, match="Graph JSON must be an object"):
        embed.load_graph_json(graph_path)


def test_reference_screenshot_b64(tmp_path: Path) -> None:
    graph_path = _write_graph_tree(tmp_path)
    assert embed.reference_screenshot_b64(graph_path, "s0_home") == base64.b64encode(
        _SCREENSHOT
    ).decode("ascii")
    assert embed.reference_screenshot_b64(graph_path, "s1_detail") is None


def _precompute(tmp_path: Path, app_name: str | None = None) -> dict[str, int]:
    return embed.precompute_graph_image_embeddings(
        tmp_path,
        api_key="key",
        model="gemini-embedding-2",
        base_url="https://generativelanguage.googleapis.com/v1beta",
        app_name=app_name,
    )


@pytest.mark.usefixtures("no_sleep")
def test_precompute_computes_caches_and_skips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph_path = _write_graph_tree(tmp_path)
    monkeypatch.setattr(
        embedding_cache, "get_gemini_native_image_embedding", lambda *_a, **_kw: [0.1, 0.2]
    )

    summary = _precompute(tmp_path)
    assert summary == {
        "graphs": 1,
        "reference_screenshots": 1,
        "already_cached": 0,
        "computed": 1,
        "skipped_missing_screenshot": 1,
        "skipped_failed": 0,
    }
    assert embed.load_image_embeddings(graph_path) == {"s0_home": [0.1, 0.2]}

    assert _precompute(tmp_path)["already_cached"] == 1


@pytest.mark.usefixtures("no_sleep")
def test_precompute_keeps_going_when_a_node_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph_path = _write_graph_tree(tmp_path)

    def always_failing(*_args: Any, **_kwargs: Any) -> list[float]:
        msg = "503 unavailable"
        raise RuntimeError(msg)

    monkeypatch.setattr(embedding_cache, "get_gemini_native_image_embedding", always_failing)

    summary = _precompute(tmp_path)
    assert summary["skipped_failed"] == 1
    assert summary["computed"] == 0
    assert embed.load_image_embeddings(graph_path) == {}


@pytest.mark.usefixtures("no_sleep")
def test_precompute_recomputes_a_node_whose_cached_entry_was_malformed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dropped (malformed) cache entry must not be mistaken for "already cached"."""
    graph_path = _write_graph_tree(tmp_path)
    image_embeddings_path(graph_path).write_text(
        json.dumps({"s0_home": "not-a-vector"}), encoding="utf-8"
    )
    monkeypatch.setattr(
        embedding_cache, "get_gemini_native_image_embedding", lambda *_a, **_kw: [0.3, 0.4]
    )

    summary = _precompute(tmp_path)
    assert summary["already_cached"] == 0
    assert summary["computed"] == 1
    assert embed.load_image_embeddings(graph_path) == {"s0_home": [0.3, 0.4]}


@pytest.mark.usefixtures("no_sleep")
def test_precompute_recomputes_a_node_whose_cached_entry_overflowed_a_float(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cached vector element too large for float() must not crash the run;
    it is dropped like any other malformed entry and the node is recomputed.
    """
    graph_path = _write_graph_tree(tmp_path)
    image_embeddings_path(graph_path).write_text(
        json.dumps({"s0_home": [10**400]}), encoding="utf-8"
    )
    monkeypatch.setattr(
        embedding_cache, "get_gemini_native_image_embedding", lambda *_a, **_kw: [0.3, 0.4]
    )

    summary = _precompute(tmp_path)
    assert summary["already_cached"] == 0
    assert summary["computed"] == 1
    assert embed.load_image_embeddings(graph_path) == {"s0_home": [0.3, 0.4]}
