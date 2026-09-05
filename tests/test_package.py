"""Package-level invariants."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

import android_app_graph


def test_version_matches_project_metadata() -> None:
    """__version__ is read from the installed metadata; it must match [project].version."""
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject.open("rb") as f:
        expected = tomllib.load(f)["project"]["version"]
    assert android_app_graph.__version__ == expected


def test_no_credentials_in_environment() -> None:
    """The autouse fixture in conftest.py must have removed every credential variable."""
    leaked = [n for n in os.environ if n.startswith("APP_GRAPH_")]
    assert leaked == []
    assert "GEMINI_API_KEY" not in os.environ
    assert "OPENAI_API_KEY" not in os.environ
