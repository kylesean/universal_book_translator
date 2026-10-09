"""Deterministic terminology metrics (TP/TF) for the quality report.

WMT-style terminology precision, computed at export time with zero LLM cost.
A glossary entry is evaluated only where its source surfaces — ``source`` *and
its aliases* — actually occur in a block (boundary-aware, outside
math/HTML/URL/masker spans, folding case on both sides), via the shared
detection primitive :func:`ubt.core.qe.term_drift.detect_term_drift`, so the
metric measures **rendering fidelity**, not glossary coverage.

- ``term_precision`` (TP): exact canonical renderings / expected occurrences.
- ``fuzzy_term_precision`` (TF): allows a fuzzy (>= threshold) rendering match,
  mirroring the WMT practice of an 80% bar.
- ``term_recall``: distinct terms with >= 1 exact rendering / distinct terms
  expected anywhere in the job.

Because this scan feeds the consistency stage's repair planner, its source-side
caliber must equal the quality gate's (``GlossaryConsistencyValidator``): both
ask ``detect_term_drift``. Searching for ``source`` alone would leave a block
whose source carried a term solely as an alias unplanned for repair while the
gate already flagged it as drift.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from rapidfuzz import fuzz

from ubt.core.qe.term_drift import detect_term_drift, normalize_glossary
from ubt.core.validators.glossary_enforcer import extract_protected_spans

# Fuzzy matching is meaningless for very short surfaces (a 1-2 char CJK term
# fuzzy-matches almost anything), so those are exact-only.
_MIN_FUZZY_SURFACE_LEN = 3


@dataclass(frozen=True, slots=True)
class TermHit:
    """One evaluated (term, block) occurrence pair."""

    source: str
    expected: str
    block_id: str
    exact: bool
    fuzzy: bool


@dataclass(frozen=True, slots=True)
class TermMetrics:
    """Aggregate terminology precision/recall over a job's finalized blocks."""

    terms_expected: int
    terms_rendered: int
    term_precision: float
    fuzzy_term_precision: float
    term_recall: float
    per_hit: tuple[TermHit, ...] = ()


@dataclass(frozen=True, slots=True)
class TermDrift:
    """Document-level drift for one source term across all its occurrences."""

    source: str
    expected: str
    occurrences: int
    exact_renderings: int
    drifted_block_ids: tuple[str, ...] = ()

    @property
    def drift_rate(self) -> float:
        if self.occurrences <= 0:
            return 0.0
        return round(1.0 - (self.exact_renderings / self.occurrences), 4)


def summarize_drift(metrics: TermMetrics, max_blocks_per_term: int = 50) -> tuple[TermDrift, ...]:
    """Aggregate per-hit results into per-term document drift, worst first.

    A "drift" occurrence is a block where one of the term's source surfaces
    (``source`` *or* an alias — the caliber shared with the quality gate)
    occurs but its canonical rendering is absent. ``max_blocks_per_term``
    bounds the retained block ids so a systematically mistranslated term cannot
    bloat the report; the counts are always complete.
    """
    acc: dict[str, dict[str, Any]] = {}
    for hit in metrics.per_hit:
        rec = acc.setdefault(
            hit.source, {"expected": hit.expected, "occ": 0, "exact": 0, "drift": []}
        )
        rec["occ"] += 1
        if hit.exact:
            rec["exact"] += 1
        elif len(rec["drift"]) < max_blocks_per_term:
            rec["drift"].append(hit.block_id)
    drifts = [
        TermDrift(
            source=source,
            expected=rec["expected"],
            occurrences=rec["occ"],
            exact_renderings=rec["exact"],
            drifted_block_ids=tuple(rec["drift"]),
        )
        for source, rec in acc.items()
        if rec["occ"] > rec["exact"]
    ]
    drifts.sort(key=lambda d: (-d.drift_rate, -d.occurrences, d.source))
    return tuple(drifts)


def _fuzzy_match(target_text: str, expected: str, fuzzy_threshold: float) -> bool:
    """Whether ``target`` fuzzy-carries ``expected`` (>= threshold, WMT's 80% bar).

    The exact verdict is deliberately NOT recomputed here: it comes from
    :func:`ubt.core.qe.term_drift.detect_term_drift`, so "exact" means the
    same boundary-aware, case-folded, protected-span-aware rendering check
    the quality gate makes.
    """
    if len(expected) < _MIN_FUZZY_SURFACE_LEN:
        return False
    return fuzz.partial_ratio(expected, target_text) >= fuzzy_threshold * 100


def evaluate_terms(
    blocks: Sequence[Any],
    glossary: Sequence[dict[str, Any]],
    fuzzy_threshold: float = 0.8,
) -> TermMetrics:
    """Compute terminology metrics over ``blocks`` for ``glossary``.

    ``blocks`` are ``IRBlock``-like objects exposing ``source_text``,
    ``target_text`` and ``id``. Entries missing a source or translation are
    skipped; an empty glossary or no matching block yields an all-zero result
    with ``terms_expected == 0``.

    Detection is delegated to :func:`ubt.core.qe.term_drift.detect_term_drift`
    — one normalization of the glossary for the whole job, then one drift
    scan per block whose structural spans are derived once per side.
    """
    terms = normalize_glossary(glossary)
    expected_count = 0
    exact_hits = 0
    fuzzy_hits = 0
    expected_terms: set[str] = set()
    rendered_terms: set[str] = set()
    per_hit: list[TermHit] = []

    for block in blocks:
        target = getattr(block, "target_text", "") or ""
        if not target:
            continue
        source = getattr(block, "source_text", "") or ""
        block_id = str(getattr(block, "id", ""))
        # Structural spans of a text are the same for every term asked about it,
        # so derive them once per block instead of once per (block, term).
        source_spans = extract_protected_spans(source)
        target_spans = extract_protected_spans(target)
        for finding in detect_term_drift(
            source,
            target,
            terms,
            source_protected=source_spans,
            target_protected=target_spans,
        ):
            if not finding.occurs_in_source:
                continue
            expected_count += 1
            expected_terms.add(finding.source)
            exact = finding.rendered
            fuzzy = exact or _fuzzy_match(target, finding.expected, fuzzy_threshold)
            exact_hits += int(exact)
            fuzzy_hits += int(fuzzy)
            if exact:
                rendered_terms.add(finding.source)
            per_hit.append(TermHit(finding.source, finding.expected, block_id, exact, fuzzy))

    distinct_expected = len(expected_terms)
    return TermMetrics(
        terms_expected=distinct_expected,
        terms_rendered=len(rendered_terms),
        term_precision=round(exact_hits / expected_count, 4) if expected_count else 0.0,
        fuzzy_term_precision=round(fuzzy_hits / expected_count, 4) if expected_count else 0.0,
        term_recall=round(len(rendered_terms) / distinct_expected, 4) if distinct_expected else 0.0,
        per_hit=tuple(per_hit),
    )
