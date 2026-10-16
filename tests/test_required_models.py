"""Required model configuration fails before device work or provider requests."""

from __future__ import annotations

import builtins
import sys
from pathlib import Path

import pytest

from android_app_graph import cli
from android_app_graph.commands import audit, embed, plot


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
