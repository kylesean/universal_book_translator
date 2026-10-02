"""Deterministic terminology metrics (TP/TF) for the quality report.

Computed at export time with zero LLM cost, over the *rendering fidelity* of the
blocks that actually carry a term (source or alias). Pinned contracts:

* ``TermDrift.drift_rate`` is ``1 - exact/occurrences`` rounded to 4 places, and
  ``0.0`` for a zero-occurrence term;
* ``summarize_drift`` aggregates per-hit results per source, keeps only terms
  that actually drifted, bounds the retained block ids without touching the
  counts, and sorts worst-first (drift rate, then occurrences, then source);
* ``_fuzzy_match`` is exact-only below 3 characters and otherwise uses a
  partial-ratio bar (WMT's 80%);
* ``evaluate_terms`` skips blocks without a translation, counts expected
  occurrences, distinct rendered terms, and returns an all-zero result for an
  empty glossary or no matching block.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from ubt.core.qe.term_metrics import (
    TermDrift,
    TermHit,
    TermMetrics,
    _fuzzy_match,
    evaluate_terms,
    summarize_drift,
)

pytestmark = pytest.mark.fast


@dataclass
class _Block:
    id: str
    source_text: str
    target_text: str | None


# --------------------------------------------------------------------------- #
# TermDrift.drift_rate
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("occurrences", "exact", "expected"),
    [
        (0, 0, 0.0),
        (4, 0, 1.0),
        (4, 1, 0.75),
        (3, 1, 0.6667),
        (4, 4, 0.0),
    ],
)
def test_drift_rate(occurrences: int, exact: int, expected: float) -> None:
    drift = TermDrift("s", "e", occurrences, exact)
    assert drift.drift_rate == expected


# --------------------------------------------------------------------------- #
# _fuzzy_match
# --------------------------------------------------------------------------- #


def test_fuzzy_match_is_disabled_below_three_characters() -> None:
    assert _fuzzy_match("xxabxx", "ab", 0.8) is False


def test_fuzzy_match_accepts_a_substring() -> None:
    assert _fuzzy_match("xxabcxx", "abc", 0.8) is True


def test_fuzzy_match_rejects_unrelated_text() -> None:
    assert _fuzzy_match("zzzzz", "abc", 0.8) is False


def test_fuzzy_match_honours_the_threshold() -> None:
    # partial_ratio("color", "colour") ~= 88.9.
    assert _fuzzy_match("color", "colour", 0.8) is True
    assert _fuzzy_match("color", "colour", 0.9) is False


# --------------------------------------------------------------------------- #
# summarize_drift
# --------------------------------------------------------------------------- #


def _metrics(*hits: TermHit) -> TermMetrics:
    return TermMetrics(
        terms_expected=0,
        terms_rendered=0,
        term_precision=0.0,
        fuzzy_term_precision=0.0,
        term_recall=0.0,
        per_hit=hits,
    )


def test_summarize_drift_aggregates_per_source() -> None:
    metrics = _metrics(
        TermHit("A", "EA", "b1", exact=False, fuzzy=False),
        TermHit("A", "EA", "b2", exact=False, fuzzy=False),
        TermHit("A", "EA", "b3", exact=True, fuzzy=True),
        TermHit("B", "EB", "b1", exact=True, fuzzy=True),
    )
    (drift,) = summarize_drift(metrics)
    assert drift.source == "A"
    assert drift.occurrences == 3
    assert drift.exact_renderings == 1
    assert drift.drifted_block_ids == ("b1", "b2")
    assert drift.drift_rate == 0.6667


def test_summarize_drift_omits_terms_without_drift() -> None:
    metrics = _metrics(TermHit("B", "EB", "b1", exact=True, fuzzy=True))
    assert summarize_drift(metrics) == ()


def test_summarize_drift_bounds_retained_block_ids() -> None:
    metrics = _metrics(
        TermHit("A", "EA", "b1", exact=False, fuzzy=False),
        TermHit("A", "EA", "b2", exact=False, fuzzy=False),
        TermHit("A", "EA", "b3", exact=False, fuzzy=False),
    )
    (drift,) = summarize_drift(metrics, max_blocks_per_term=1)
    assert drift.drifted_block_ids == ("b1",)
    assert drift.occurrences == 3


def test_summarize_drift_sorts_worst_first_then_by_occurrences() -> None:
    metrics = _metrics(
        TermHit("Mild", "E", "b1", exact=True, fuzzy=True),
        TermHit("Mild", "E", "b2", exact=False, fuzzy=False),  # rate 0.5
        TermHit("Bad", "E", "b1", exact=False, fuzzy=False),
        TermHit("Bad", "E", "b2", exact=False, fuzzy=False),  # rate 1.0
    )
    assert [d.source for d in summarize_drift(metrics)] == ["Bad", "Mild"]


# --------------------------------------------------------------------------- #
# evaluate_terms
# --------------------------------------------------------------------------- #


_GLOSSARY = [{"source": "FinFET", "translation": "鳍式场效应晶体管"}]


def test_empty_glossary_is_all_zero() -> None:
    metrics = evaluate_terms([], [])
    assert metrics.terms_expected == 0
    assert metrics.term_precision == 0.0
    assert metrics.term_recall == 0.0
    assert metrics.per_hit == ()


def test_blocks_without_a_translation_are_skipped() -> None:
    blocks: list[Any] = [
        _Block("b1", "a FinFET", ""),
        _Block("b2", "a FinFET", None),
    ]
    assert evaluate_terms(blocks, _GLOSSARY).per_hit == ()


def test_no_matching_block_yields_zero() -> None:
    metrics = evaluate_terms([_Block("b1", "nothing", "无关")], _GLOSSARY)
    assert metrics.terms_expected == 0
    assert metrics.per_hit == ()


def test_exact_renderings_score_full_precision_and_recall() -> None:
    metrics = evaluate_terms([_Block("b1", "The FinFET device", "鳍式场效应晶体管器件")], _GLOSSARY)
    assert metrics.terms_expected == 1
    assert metrics.terms_rendered == 1
    assert metrics.term_precision == 1.0
    assert metrics.term_recall == 1.0


def test_drift_lowers_precision_but_keeps_recall() -> None:
    metrics = evaluate_terms(
        [
            _Block("b1", "The FinFET device", "鳍式场效应晶体管器件"),
            _Block("b2", "a FinFET", "missing rendering"),
        ],
        _GLOSSARY,
    )
    assert metrics.terms_expected == 1
    assert metrics.terms_rendered == 1
    assert metrics.term_precision == 0.5
    assert metrics.fuzzy_term_precision == 0.5
    assert metrics.term_recall == 1.0
    assert [(h.block_id, h.exact) for h in metrics.per_hit] == [("b1", True), ("b2", False)]


def test_fuzzy_precision_accepts_a_near_rendering() -> None:
    metrics = evaluate_terms(
        [_Block("b1", "analyse this", "analyze this")],
        [{"source": "analyse", "translation": "analyse"}],
    )
    assert metrics.term_precision == 0.0
    assert metrics.fuzzy_term_precision == 1.0
    assert metrics.term_recall == 0.0


def test_recall_counts_distinct_terms() -> None:
    glossary = [
        {"source": "Alpha", "translation": "甲"},
        {"source": "Beta", "translation": "乙"},
    ]
    metrics = evaluate_terms([_Block("b1", "Alpha and Beta", "甲 and 乙")], glossary)
    assert metrics.terms_expected == 2
    assert metrics.terms_rendered == 2
    assert metrics.term_recall == 1.0
