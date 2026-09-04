"""Property and example tests for ui_kobe.utils.resolve_env."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ui_kobe.utils import resolve_env


def test_none_passes_through() -> None:
    assert resolve_env(None) is None


@given(st.text(alphabet=st.characters(exclude_characters="$")))
def test_strings_without_dollar_are_returned_unchanged(value: str) -> None:
    """Only a full ${NAME} match is resolved; anything else is returned as given."""
    assert resolve_env(value) == value


def test_reference_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UI_KOBE_TEST_VALUE", "hello")
    assert resolve_env("${UI_KOBE_TEST_VALUE}") == "hello"


def test_reference_to_unset_variable_is_none() -> None:
    assert resolve_env("${UI_KOBE_TEST_UNSET}") is None
