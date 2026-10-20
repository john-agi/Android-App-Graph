"""Package-level invariants."""

from __future__ import annotations

import os
import subprocess
import sys
import tomllib
from pathlib import Path

import android_app_graph

_PRINT_LOADED_DEADLINE = (
    "import runpy, sys; from hypothesis import settings; "
    "runpy.run_path(sys.argv[1]); print(settings().deadline)"
)


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


def test_local_hypothesis_profile_has_no_deadline() -> None:
    """Outside CI no example fails for running slowly on a busy machine.

    conftest.py picks the profile while it is imported, so a fresh interpreter
    imports it with neither CI nor HYPOTHESIS_PROFILE set.
    """
    local = {k: v for k, v in os.environ.items() if k not in {"CI", "HYPOTHESIS_PROFILE"}}
    conftest = Path(__file__).with_name("conftest.py")
    loaded = subprocess.run(
        [sys.executable, "-I", "-c", _PRINT_LOADED_DEADLINE, str(conftest)],
        env=local,
        capture_output=True,
        text=True,
        check=False,
    )
    assert loaded.stdout == "None\n", loaded.stderr
