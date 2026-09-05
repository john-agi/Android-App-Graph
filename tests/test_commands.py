"""Help and packaging tests shared by the kobe-* console scripts."""

from __future__ import annotations

from collections.abc import Callable
from importlib.metadata import entry_points

import pytest

from ui_kobe.commands import audit, embed, plot

Main = Callable[[list[str] | None], int]

COMMANDS: dict[str, Main] = {
    "kobe-audit": audit.main,
    "kobe-embed": embed.main,
    "kobe-plot": plot.main,
}


@pytest.mark.parametrize("script", sorted(COMMANDS))
def test_help_exits_zero(script: str, capsys: pytest.CaptureFixture[str]) -> None:
    """--help is handled inside parse_args(), before any work or logging setup."""
    with pytest.raises(SystemExit) as excinfo:
        COMMANDS[script](["--help"])
    assert excinfo.value.code == 0
    assert capsys.readouterr().out.startswith(f"usage: {script}")


@pytest.mark.parametrize("script", sorted(COMMANDS))
def test_console_script_entry_point_resolves(script: str) -> None:
    """The [project.scripts] entry point points at the main() this suite drives."""
    (entry_point,) = entry_points(group="console_scripts", name=script)
    assert entry_point.load() is COMMANDS[script]
