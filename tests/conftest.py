"""Shared pytest configuration: Hypothesis profiles and a credential guard."""

from __future__ import annotations

import os

import pytest
from hypothesis import settings

# The built-in "default" profile runs 100 examples with a 200 ms deadline. The
# built-in "ci" profile is deterministic, has no deadline and no example database,
# prints reproduction blobs, and is selected automatically when the CI environment
# variable is set; it is re-registered here with more examples.
settings.register_profile("ci", settings.get_profile("ci"), max_examples=500)
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "ci" if os.getenv("CI") else "default"))

_CREDENTIAL_PREFIXES = ("UI_KOBE_",)
_CREDENTIAL_NAMES = frozenset({"GEMINI_API_KEY", "OPENAI_API_KEY"})


@pytest.fixture(autouse=True)
def _clear_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove provider credentials so tests can never use real keys or endpoints."""
    for name in list(os.environ):
        if name.startswith(_CREDENTIAL_PREFIXES) or name in _CREDENTIAL_NAMES:
            monkeypatch.delenv(name)
