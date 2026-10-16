"""Required model configuration fails before device work or provider requests."""

from __future__ import annotations

import builtins
import sys
from pathlib import Path

import httpx
import pytest

from android_app_graph import cli
from android_app_graph.adapters.aitk_translator import UIKobeV2Translator
from android_app_graph.commands import audit, embed, plot
from android_app_graph.embedding_cache import resolve_image_embedding_settings
from android_app_graph.utils import make_client


@pytest.mark.parametrize("role", ["page_detail", "embedding", "instruction", "action"])
@pytest.mark.parametrize("value", [None, "", " \t ", "${APP_GRAPH_UNSET_MODEL}", 123, False])
def test_client_rejects_missing_model(
    role: str, value: object, owned_router_clients: list[httpx.Client]
) -> None:
    with pytest.raises(ValueError, match=rf"{role}.*model"):
        make_client({"api_key": "test-key", "model": value}, role=role)
    assert owned_router_clients == []


@pytest.mark.parametrize("value", ["", " \t ", "${APP_GRAPH_UNSET_MODEL}"])
def test_blank_environment_model_is_rejected(value: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_GRAPH_ACTION_MODEL", value)
    with pytest.raises(ValueError, match="APP_GRAPH_ACTION_MODEL"):
        make_client({"api_key": "test-key"}, role="action")


@pytest.mark.parametrize("value", ["", " \t ", "${APP_GRAPH_UNSET_MODEL}"])
def test_invalid_explicit_model_cannot_fall_back_to_environment(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("APP_GRAPH_ACTION_MODEL", "provider/action")
    with pytest.raises(ValueError, match=r"action.model"):
        make_client({"api_key": "test-key", "model": value}, role="action")


@pytest.mark.parametrize("role", ["page_detail", "embedding", "instruction", "action"])
def test_client_reads_required_model_from_environment(
    role: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(f"APP_GRAPH_{role.upper()}_MODEL", f"provider/{role}")
    with_model, model = make_client({"api_key": "test-key"}, role=role)
    with with_model:
        assert model == f"provider/{role}"
    assert not (tmp_path / ".env").exists()


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


def test_runtime_reports_missing_models_before_creating_clients(
    tmp_path: Path, owned_router_clients: list[httpx.Client]
) -> None:
    with pytest.raises(ValueError, match="Invalid model configuration") as caught:
        UIKobeV2Translator(graph_dir=str(tmp_path))
    for role in ("PAGE_DETAIL", "EMBEDDING", "INSTRUCTION", "ACTION"):
        assert f"APP_GRAPH_{role}_MODEL" in str(caught.value)
    assert owned_router_clients == []


def test_runtime_keeps_caller_environment_references(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    config: dict[str, object] = {}
    for role in ("page_detail", "embedding", "instruction", "action"):
        variable = f"APP_GRAPH_{role.upper()}_MODEL"
        monkeypatch.setenv(variable, f"provider/{role}")
        config[role] = {"model": f"${{{variable}}}"}
    first = UIKobeV2Translator(graph_dir=str(tmp_path), vlm_config=config)
    monkeypatch.setenv("APP_GRAPH_ACTION_MODEL", "provider/new-action")
    second = UIKobeV2Translator(graph_dir=str(tmp_path), vlm_config=config)
    assert first.model_name == "provider/action"
    assert second.model_name == "provider/new-action"
    assert config["action"] == {"model": "${APP_GRAPH_ACTION_MODEL}"}
    assert "image_embedding" not in config


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
