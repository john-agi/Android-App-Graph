"""app-graph argument parsing, exercised without AITK, adb or an emulator."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from android_app_graph.cli import build_parser, main


def test_help_exits_zero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """argparse handles --help inside parse_args() and exits 0 before aitk is imported."""
    monkeypatch.setattr(sys, "argv", ["app-graph", "--help"])
    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 0
    assert "--config" in capsys.readouterr().out


def test_parser_defaults() -> None:
    args = build_parser().parse_args([])
    assert args.config == Path("configs/explore.yaml")
    assert args.max_steps is None
    assert args.resume_from is None


def test_parser_overrides() -> None:
    args = build_parser().parse_args(["-c", "custom.yaml", "--max-steps", "5", "-r", "auto"])
    assert args.config == Path("custom.yaml")
    assert args.max_steps == 5
    assert args.resume_from == "auto"
