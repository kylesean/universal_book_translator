"""chrF: character n-gram F-score used as the Phase-2 no-degradation lower bound.

The metric is deliberately small and self-contained. The contracts pinned here:
whitespace-normalized, word-boundary-padded n-grams; ``n <= 0`` coerced to 1;
two empty strings score a perfect 1.0 while one empty scores 0.0; the score is
bounded in ``[0, 1]``, symmetric at ``beta == 1``, and weights recall more as
``beta`` grows (so a hypothesis with precision above recall *drops* when beta
increases).
"""

from __future__ import annotations

from collections import Counter

import pytest

from ubt.core.qe.chrf import _char_ngrams, chrf

pytestmark = pytest.mark.fast


# --------------------------------------------------------------------------- #
# _char_ngrams
# --------------------------------------------------------------------------- #


def test_ngrams_pad_with_word_boundaries() -> None:
    assert _char_ngrams("ab", 2) == Counter({" a": 1, "ab": 1, "b ": 1})


def test_ngrams_normalize_whitespace() -> None:
    assert _char_ngrams("a   b", 3) == _char_ngrams("a b", 3)


def test_ngrams_empty_text_is_empty() -> None:
    assert _char_ngrams("", 6) == Counter()
    assert _char_ngrams("   ", 6) == Counter()


def test_ngrams_shorter_than_order_keeps_the_padded_whole() -> None:
    assert _char_ngrams("a b", 6) == Counter({" a b ": 1})


def test_ngrams_non_positive_order_is_one() -> None:
    assert _char_ngrams("ab", 0) == Counter({" ": 2, "a": 1, "b": 1})


# --------------------------------------------------------------------------- #
# chrf
# --------------------------------------------------------------------------- #


def test_two_empty_strings_are_a_perfect_match() -> None:
    assert chrf("", "") == 1.0


def test_one_empty_string_scores_zero() -> None:
    assert chrf("a", "") == 0.0
    assert chrf("", "a") == 0.0


def test_identical_text_scores_one() -> None:
    assert chrf("hello world", "hello world") == 1.0


def test_disjoint_text_scores_zero() -> None:
    assert chrf("abc", "xyz") == 0.0


def test_partial_overlap_scores_between_zero_and_one() -> None:
    assert chrf("the quick brown fox", "the quick brown cat") == pytest.approx(0.75)


def test_score_is_bounded() -> None:
    score = chrf("a b c", "a b d", n=2)
    assert 0.0 <= score <= 1.0


def test_score_is_symmetric_at_beta_one() -> None:
    forward = chrf("ab", "ab cd ef", n=3, beta=1.0)
    backward = chrf("ab cd ef", "ab", n=3, beta=1.0)
    assert forward == backward == pytest.approx(0.4)


def test_higher_beta_weights_recall() -> None:
    # Precision (1.0) exceeds recall (0.25) here, so weighting recall more
    # lowers the score: beta=2 < beta=1. The exact values pin beta**2.
    beta_one = chrf("ab", "ab cd ef", n=3, beta=1.0)
    beta_two = chrf("ab", "ab cd ef", n=3, beta=2.0)
    assert beta_one == pytest.approx(0.4)
    assert beta_two == pytest.approx(0.29411764705882354)
    assert beta_two < beta_one


def test_more_overlap_scores_higher() -> None:
    close = chrf("the quick brown fox", "the quick brown fox jumps", n=3)
    far = chrf("the quick brown fox", "a totally different sentence", n=3)
    assert close > far
