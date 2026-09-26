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
#: (``skip_translate``) and TM exact hits (``tm_hit``). Kept as the stamped
#: value for continuity, but it is no longer how those blocks are recognised —
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
