"""app-graph-embed settings, retry and end-to-end helpers, exercised without network or keys."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import pytest

from android_app_graph.commands import embed

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


# Argument parsing and settings


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


def test_embed_main_requires_an_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)  # conftest removes GEMINI_API_KEY
    config = tmp_path / "explore.yaml"
    config.write_text(f"experiment:\n  graph_dir: {tmp_path}\n", encoding="utf-8")
    with pytest.raises(SystemExit) as excinfo:
        embed.main(["-c", str(config)])
    assert excinfo.value.code == 2
    assert "missing API key" in capsys.readouterr().err


def test_embed_settings_from_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)  # conftest removes GEMINI_API_KEY
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


# Pure file helpers


def test_image_embeddings_path() -> None:
    assert embed.image_embeddings_path(Path("graphs/demo/demo.json")) == Path(
        "graphs/demo/demo.image_emb.json"
    )


def test_save_and_load_image_embeddings_round_trip(tmp_path: Path) -> None:
    graph_path = tmp_path / "demo.json"
    assert embed.load_image_embeddings(graph_path) == {}
    embed.save_image_embeddings(graph_path, {"s0_home": [0.5, 1.0]})
    assert embed.load_image_embeddings(graph_path) == {"s0_home": [0.5, 1.0]}


def test_load_image_embeddings_drops_malformed_entries(tmp_path: Path) -> None:
    graph_path = tmp_path / "demo.json"
    embed.image_embeddings_path(graph_path).write_text(
        json.dumps({"s0_home": [1, 2], "s1_bad": "not-a-vector"}), encoding="utf-8"
    )
    assert embed.load_image_embeddings(graph_path) == {"s0_home": [1.0, 2.0], "s1_bad": []}


def test_iter_graph_files_prefers_the_audited_graph(tmp_path: Path) -> None:
    _write_graph_tree(tmp_path, app="demo")
    audited = _write_graph_tree(tmp_path, app="other", audited=True)
    (tmp_path / "other" / "other.json").write_text("{}", encoding="utf-8")
    (tmp_path / "loose_file.txt").write_text("ignored", encoding="utf-8")

    assert embed.iter_graph_files(tmp_path) == [
        ("demo", tmp_path / "demo" / "demo.json"),
        ("other", audited),
    ]


def test_iter_graph_files_can_select_one_app(tmp_path: Path) -> None:
    graph_path = _write_graph_tree(tmp_path, app="demo")
    _write_graph_tree(tmp_path, app="other")
    assert embed.iter_graph_files(tmp_path, "demo") == [("demo", graph_path)]


def test_iter_graph_files_skips_an_unknown_app(tmp_path: Path) -> None:
    assert embed.iter_graph_files(tmp_path, "absent") == []


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


# Retry behaviour


@pytest.fixture
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(embed.time, "sleep", lambda _seconds: None)


@pytest.mark.usefixtures("_no_sleep")
def test_compute_embedding_with_retry_returns_on_first_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []

    def fake_embedding(api_key: str, screenshot_b64: str, **_kwargs: Any) -> list[float]:
        calls.append((api_key, screenshot_b64))
        return [1.0, 2.0]

    monkeypatch.setattr(embed, "get_gemini_native_image_embedding", fake_embedding)

    assert _retry("key", "shot") == [1.0, 2.0]
    assert calls == [("key", "shot")]


@pytest.mark.usefixtures("_no_sleep")
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

    monkeypatch.setattr(embed, "get_gemini_native_image_embedding", flaky_embedding)

    assert _retry("key", "shot") == [0.25]
    assert len(attempts) == 2


@pytest.mark.usefixtures("_no_sleep")
def test_compute_embedding_with_retry_reraises_after_the_last_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts: list[int] = []

    def always_failing(*_args: Any, **_kwargs: Any) -> list[float]:
        attempts.append(len(attempts))
        msg = "500 upstream error"
        raise RuntimeError(msg)

    monkeypatch.setattr(embed, "get_gemini_native_image_embedding", always_failing)

    with pytest.raises(RuntimeError, match="500 upstream error"):
        _retry("key", "shot")
    assert len(attempts) == embed.IMAGE_EMBEDDING_RETRIES + 1


def _retry(api_key: str, screenshot_b64: str) -> list[float]:
    return embed.compute_embedding_with_retry(
        api_key,
        screenshot_b64,
        model="gemini-embedding-2",
        base_url="https://generativelanguage.googleapis.com/v1beta",
        app_name="demo",
        node_id="s0_home",
    )


# precompute_graph_image_embeddings, end to end on a temporary graph tree


def _precompute(tmp_path: Path, app_name: str | None = None) -> dict[str, int]:
    return embed.precompute_graph_image_embeddings(
        tmp_path,
        api_key="key",
        model="gemini-embedding-2",
        base_url="https://generativelanguage.googleapis.com/v1beta",
        app_name=app_name,
    )


@pytest.mark.usefixtures("_no_sleep")
def test_precompute_computes_caches_and_skips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph_path = _write_graph_tree(tmp_path)
    monkeypatch.setattr(embed, "get_gemini_native_image_embedding", lambda *_a, **_kw: [0.1, 0.2])

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


@pytest.mark.usefixtures("_no_sleep")
def test_precompute_keeps_going_when_a_node_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph_path = _write_graph_tree(tmp_path)

    def always_failing(*_args: Any, **_kwargs: Any) -> list[float]:
        msg = "503 unavailable"
        raise RuntimeError(msg)

    monkeypatch.setattr(embed, "get_gemini_native_image_embedding", always_failing)

    summary = _precompute(tmp_path)
    assert summary["skipped_failed"] == 1
    assert summary["computed"] == 0
    assert embed.load_image_embeddings(graph_path) == {}
