"""Tests for android_app_graph.utils.vlm_utils.cosine_similarity.

The single shared implementation used by both android_app_graph.utils.graph_manager
and android_app_graph.adapters.aitk_translator; both modules' node-retrieval tests
exercise it end to end, so this module owns its unit tests.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from android_app_graph.utils.vlm_utils import cosine_similarity


def test_cosine_similarity_of_identical_vectors_is_one() -> None:
    assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_cosine_similarity_of_orthogonal_vectors_is_zero() -> None:
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_of_opposite_vectors_is_minus_one() -> None:
    assert cosine_similarity([1.0, 1.0], [-1.0, -1.0]) == pytest.approx(-1.0)


@pytest.mark.parametrize(("a", "b"), [([0.0, 0.0], [1.0, 1.0]), ([1.0, 1.0], [0.0, 0.0])])
def test_cosine_similarity_with_a_zero_vector_is_zero(a: list[float], b: list[float]) -> None:
    """A zero norm has no direction, so the similarity is defined as 0.0 rather than NaN."""
    assert cosine_similarity(a, b) == 0.0


def test_cosine_similarity_rejects_vectors_of_different_length() -> None:
    """Two vectors of different dimension are never scored against each other.

    ``zip`` would otherwise silently truncate to the shorter vector, scoring a
    cached embedding from a different model's space against the overlapping
    prefix and ranking it as garbage with no signal that anything went wrong.
    """
    with pytest.raises(ValueError, match=r"2 vs 1|1 vs 2"):
        cosine_similarity([1.0, 0.0], [1.0])


@pytest.mark.parametrize(
    ("a", "b"),
    [
        # Subnormal floats break Cauchy-Schwarz under naive floating-point rounding:
        # the unclamped ratio can land slightly outside [-1.0, 1.0].
        ([2.5e-162], [3.4e-162]),
        ([1.0], [8.30800819665626e-160]),
    ],
)
def test_cosine_similarity_clamps_subnormal_rounding_artifacts(
    a: list[float], b: list[float]
) -> None:
    assert -1.0 <= cosine_similarity(a, b) <= 1.0


_finite_float = st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)
# Same length for both vectors: cosine_similarity rejects a mismatch outright,
# and this property is about the score, not that guard.
_same_length_vector_pair = st.integers(min_value=1, max_value=8).flatmap(
    lambda n: st.tuples(
        st.lists(_finite_float, min_size=n, max_size=n),
        st.lists(_finite_float, min_size=n, max_size=n),
    )
)


@given(vectors=_same_length_vector_pair)
def test_cosine_similarity_is_bounded_and_symmetric(
    vectors: tuple[list[float], list[float]],
) -> None:
    a, b = vectors
    similarity = cosine_similarity(a, b)
    assert -1.0 <= similarity <= 1.0
    assert similarity == pytest.approx(cosine_similarity(b, a))
    assert not math.isnan(similarity)


def test_cosine_similarity_with_a_nan_element_is_zero() -> None:
    """A NaN component scores 0.0 and never wins a similarity search.

    A naive clamp (``max(-1.0, min(1.0, nan))``) would return 1.0 instead.
    """
    assert cosine_similarity([math.nan, 1.0], [1.0, 1.0]) == 0.0


def test_cosine_similarity_never_returns_nan() -> None:
    assert not math.isnan(cosine_similarity([math.nan], [1.0]))
    assert not math.isnan(cosine_similarity([math.inf], [math.inf]))
