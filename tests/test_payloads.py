"""Tests for android_app_graph.payloads's numeric narrowing helpers.

A non-finite (NaN/inf) element must reject the whole vector rather than being
kept or silently dropped: dropping one element would change the vector's
dimension out from under a caller that expects a fixed-length embedding. An
element ``float()`` cannot even represent — a JSON integer too large for a
float's exponent range — is malformed the same way, not an uncaught
``OverflowError`` escaping past callers whose guard is ``(OSError, ValueError)``.
"""

from __future__ import annotations

import math

from hypothesis import given
from hypothesis import strategies as st

from android_app_graph.payloads import as_float_list, as_int, as_int_list

# One digit past a float's ~1.8e308 max: float()/int(float()) cannot represent it.
_UNREPRESENTABLE_AS_FLOAT = 10**400


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


def test_as_float_list_rejects_an_int_too_large_for_a_float() -> None:
    """A 400-digit JSON integer raises OverflowError out of float(); as_float_list
    must treat that as malformed like any other unusable element, not propagate it.
    """
    assert as_float_list([_UNREPRESENTABLE_AS_FLOAT]) == []


@given(
    st.lists(
        st.floats(allow_nan=False, allow_infinity=False, width=32),
        max_size=8,
    )
)
def test_as_float_list_keeps_every_finite_element(values: list[float]) -> None:
    assert as_float_list(list(values)) == [float(v) for v in values]


def test_as_int_of_numbers_and_numeric_strings() -> None:
    assert as_int(3) == 3
    assert as_int(3.7) == 3
    assert as_int("42") == 42
    assert as_int("3.9") == 3


def test_as_int_rejects_a_bool() -> None:
    assert as_int(True) is None


def test_as_int_rejects_a_non_numeric_value() -> None:
    assert as_int("not a number") is None
    assert as_int(None) is None
    assert as_int([1]) is None


def test_as_int_rejects_a_float_too_large_to_convert() -> None:
    """A float ``int()`` cannot represent (here, infinity) raises OverflowError,
    not ValueError; as_int must turn it into None like any other bad value.
    """
    assert as_int(math.inf) is None
    assert as_int(-math.inf) is None
    assert as_int(math.nan) is None


def test_as_int_rejects_a_string_that_parses_to_infinity() -> None:
    """``float("1e400")`` silently overflows to ``inf`` (no ValueError from float()
    itself); the ``int(inf)`` that follows must not raise OverflowError past as_int.
    """
    assert as_int("1e400") is None


@given(st.integers())
def test_as_int_keeps_every_integer_including_ones_too_large_for_a_float(value: int) -> None:
    """An arbitrary-precision JSON int is never converted through float, so it
    is kept exactly even far outside a float's representable range.
    """
    assert as_int(value) == value


def test_as_int_list_rejects_a_string_element_that_parses_to_infinity() -> None:
    assert as_int_list([1, "1e400", 3]) == []
