"""Shared policy: which blocks count as *QE-scored* for averages.

The quality report's ``score_metrics`` excludes two populations —
verbatim skips (``skip_translate``) and TM exact hits (``tm_hit``) — because
both carry a pass stamp rather than a measured score. Mixing them in skews
aggregate metrics toward false perfection (one 0.30 defect among 499 skips
reads 0.9994).

Exclusion is by *provenance*, not by a magic score value: a neural/LLM engine
can legitimately return a perfect 1.0, so keying on ``mtqe_score == 1.0``
silently dropped every genuine perfect score from the average. The ledger
records ``skip_translate`` and ``tm_hit``, and :class:`IRBlock` now exposes
both, so the predicate is unambiguous.

Both the status surface and quality report use unified scoring filters:
the report filters blocks via :func:`is_qe_scored`, and the ledger filters
score rows with :data:`QE_SCORED_SQL`.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

#: Pass stamp written on blocks that never met the QE gate: verbatim ships
#: (``skip_translate``) and TM exact hits (``tm_hit``). It is kept as the stamped
#: value for continuity, but it is not how those blocks are recognised —
#: exclusion is provenance-based (see :func:`is_qe_scored`), so a real 1.0 from
#: a neural/LLM engine is counted rather than mistaken for this placeholder.
PLACEHOLDER_MTQE_SCORE = 1.0

#: SQL predicate mirroring :func:`is_qe_scored` for score sets the ledger
#: selects directly (``get_job_stats``'s ``AVG`` over the ``blocks`` table).
#: ``skip_translate``/``tm_hit`` are stored as 0/1 INTEGER; ``mtqe_score`` is
#: REAL/NULL.
QE_SCORED_SQL = "mtqe_score IS NOT NULL AND skip_translate = 0 AND COALESCE(tm_hit, 0) = 0"


def is_qe_scored(block: Any) -> bool:
    """Whether ``block.mtqe_score`` is a real QE score.

    A block is excluded when it was never measured: ``mtqe_score is None``
    (e.g. FastPass-cleared drafts), a verbatim skip (``skip_translate``), or an
    exact TM hit (``tm_hit``). A perfect ``1.0`` from a real scorer is *not*
    excluded — only its provenance can distinguish it from the pass stamp.
    """
    if block.mtqe_score is None or block.skip_translate:
        return False
    return not getattr(block, "tm_hit", False)


def qe_scored_values(blocks: Iterable[Any]) -> list[float]:
    """Ascending real QE scores over ``blocks`` (placeholders and skips out)."""
    values = [float(b.mtqe_score) for b in blocks if is_qe_scored(b)]
    values.sort()
    return values


def average_qe_scored(blocks: Iterable[Any]) -> float:
    """Mean of the QE-scored population, rounded like the report's ``avg_qe``.

    Returns 0.0 for an empty population — the report's "nothing scored"
    reading, never a fabricated score.
    """
    values = qe_scored_values(blocks)
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)


__all__ = [
    "PLACEHOLDER_MTQE_SCORE",
    "QE_SCORED_SQL",
    "average_qe_scored",
    "is_qe_scored",
    "qe_scored_values",
]


# --------------------------------------------------------------------------- #
# What "shippable" means, in one place
#
# ``max(0.5, qe_threshold)`` appeared four times — triage's auto-pass, the
# repair stage's standalone finalizer, and two circuit-breaker exits in the
# repair loop — each with its own comment explaining the floor. Four copies of a
# rule is four chances to change one and forget the others, and "shippable" is
# the single most consequential threshold in the pipeline.
# --------------------------------------------------------------------------- #

#: Lowest bar a block may be released at, whatever the operator configures.
#: A threshold below this would ship drafts the deterministic gates rejected:
#: the twelve defect classes sit at or below 0.70, so 0.5 keeps a real failure
#: out of the delivery even when someone sets ``qe_threshold = 0.2``.
SHIPPABLE_FLOOR = 0.5


def shippable_threshold(qe_threshold: float) -> float:
    """The score a block must reach to be released without human review.

    Tracks the operator's ``qe_threshold`` upward so repair and triage never
    disagree about the bar, and never drops below :data:`SHIPPABLE_FLOOR` so a
    deliberately low setting cannot promote a barely-passable draft.
    """
    return max(SHIPPABLE_FLOOR, qe_threshold)


def is_shippable(score: float | None, qe_threshold: float, flags: list[str]) -> bool:
    """Whether a block may be released as machine output.

    Two conditions, and both are required: the score clears the shippable bar
    *and* no defect flag remains. The flag half is what keeps a structurally
    broken draft from shipping on a favourable score — the score is a routing
    signal, not evidence that the defect was fixed.
    """
    if flags:
        return False
    return (score or 0.0) >= shippable_threshold(qe_threshold)
