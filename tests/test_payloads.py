"""Tests for android_app_graph.payloads.as_float_list.

A non-finite (NaN/inf) element must reject the whole vector rather than being
kept or silently dropped: dropping one element would change the vector's
dimension out from under a caller that expects a fixed-length embedding.
"""

from __future__ import annotations

import math

from hypothesis import given
from hypothesis import strategies as st

from android_app_graph.payloads import as_float_list


def test_as_float_list_of_ints_and_floats() -> None:
    assert as_float_list([1, 2.5, 3]) == [1.0, 2.5, 3.0]


def test_as_float_list_rejects_a_non_list() -> None:
    assert as_float_list("not a list") == []


def test_as_float_list_rejects_a_bool_element() -> None:
    assert as_float_list([1.0, True]) == []


def test_as_float_list_rejects_a_non_numeric_element() -> None:
    assert as_float_list([1.0, "two"]) == []


def test_as_float_list_rejects_a_vector_containing_nan() -> None:
    assert as_float_list([1.0, math.nan]) == []


def test_as_float_list_rejects_a_vector_containing_infinity() -> None:
    assert as_float_list([1.0, math.inf]) == []
    assert as_float_list([1.0, -math.inf]) == []


@given(
    st.lists(
        st.floats(allow_nan=False, allow_infinity=False, width=32),
        max_size=8,
    )
)
def test_as_float_list_keeps_every_finite_element(values: list[float]) -> None:
    assert as_float_list(list(values)) == [float(v) for v in values]
