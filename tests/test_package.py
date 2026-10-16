"""Package-level invariants."""

from __future__ import annotations

import os
import sys
import tomllib
from pathlib import Path

import android_app_graph


def test_python_pin_matches_test_interpreter() -> None:
    """Local interpreter overrides must not hide drift in CI's Python pin."""
    python_version = Path(__file__).resolve().parents[1] / ".python-version"
    pinned_minor = tuple(int(part) for part in python_version.read_text().strip().split(".")[:2])
    assert sys.version_info[:2] == pinned_minor


def test_version_matches_project_metadata() -> None:
    """__version__ is read from the installed metadata; it must match [project].version."""
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject.open("rb") as f:
        expected = tomllib.load(f)["project"]["version"]
    assert android_app_graph.__version__ == expected


def test_no_credentials_in_environment() -> None:
    """The autouse fixture in conftest.py must have removed every credential variable."""
    leaked = [n for n in os.environ if n.startswith(("APP_GRAPH_", "OPENROUTER_"))]
    assert leaked == []
    assert "GEMINI_API_KEY" not in os.environ
    assert "GOOGLE_API_KEY" not in os.environ
    assert "OPENAI_API_KEY" not in os.environ
