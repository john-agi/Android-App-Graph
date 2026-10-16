"""Required model configuration fails before device work or provider requests."""

from __future__ import annotations

import builtins
import sys
from pathlib import Path

import httpx
import pytest

from android_app_graph import cli
from android_app_graph.commands import audit, embed, plot
from android_app_graph.embedding_cache import resolve_image_embedding_settings


def test_image_embedding_requires_a_model() -> None:
    with pytest.raises(ValueError, match=r"image_embedding.*model"):
        resolve_image_embedding_settings({})


@pytest.mark.parametrize("value", ["", "  ", "${APP_GRAPH_UNSET_MODEL}"])
def test_explicit_invalid_image_model_does_not_inherit(value: str) -> None:
    with pytest.raises(ValueError, match=r"image_embedding.*model"):
        resolve_image_embedding_settings(
            {"embedding": {"model": "provider/text"}, "image_embedding": {"model": value}}
        )


@pytest.mark.parametrize("value", ["", "  ", "${APP_GRAPH_UNSET_MODEL}"])
def test_invalid_cli_image_model_cannot_fall_back_to_configuration(value: str) -> None:
    with pytest.raises(ValueError, match=r"image_embedding.*model"):
        resolve_image_embedding_settings(
            {"image_embedding": {"model": "provider/image"}}, model_override=value
        )


@pytest.mark.parametrize("re_explore", [False, True])
def test_audit_requires_only_the_models_it_uses(
    re_explore: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    owned_router_clients: list[httpx.Client],
) -> None:
    config = tmp_path / "audit.yaml"
    config.write_text("apps: []\n", encoding="utf-8")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("APP_GRAPH_PAGE_DETAIL_MODEL", "provider/detail")
    monkeypatch.setenv("APP_GRAPH_EMBEDDING_MODEL", "provider/embedding")
    args = ["--config", str(config)]
    if re_explore:
        with pytest.raises(SystemExit) as caught:
            audit.main([*args, "--re-explore"])
        assert caught.value.code == 2
        error = capsys.readouterr().err
        assert "APP_GRAPH_INSTRUCTION_MODEL" in error
        assert "APP_GRAPH_ACTION_MODEL" in error
        assert owned_router_clients == []
    else:
        assert audit.main(args) == 0
        assert len(owned_router_clients) == 2
        assert all(client.is_closed for client in owned_router_clients)


@pytest.mark.parametrize("command", ["audit", "embed", "plot"])
def test_command_help_needs_no_model_configuration(command: str) -> None:
    main = {"audit": audit.main, "embed": embed.main, "plot": plot.main}[command]
    with pytest.raises(SystemExit) as caught:
        main(["--help"])
    assert caught.value.code == 0


def test_explorer_reports_all_missing_models_before_importing_device_tools(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "explore.yaml"
    config.write_text("vlm: {}\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["app-graph", "--config", str(config)])
    original_import = builtins.__import__

    def guarded_import(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        assert not name.startswith("aitk"), "configuration must be checked before device setup"
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2
    error = capsys.readouterr().err
    for role in ("PAGE_DETAIL", "EMBEDDING", "INSTRUCTION", "ACTION"):
        assert f"APP_GRAPH_{role}_MODEL" in error
    assert "Traceback" not in error
