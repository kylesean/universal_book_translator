"""Shared policy: which blocks count as *QE-scored* for averages.

The quality report's ``score_metrics`` excludes two populations —
verbatim skips (``skip_translate``) and TM exact hits — because both carry a
1.0 placeholder stamped by ingest/draft/lnds rather than a measured score
(the heuristic engine tops out at ``QE_SCORE_PASS`` = 0.92, so 1.0 indicates
"not scored"). Mixing them in skews aggregate metrics toward false perfection.

Both the status surface and quality report use unified scoring filters:
the report filters blocks via :func:`is_qe_scored`, and the ledger filters
score rows with :data:`QE_SCORED_SQL`.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

#: Placeholder stamped on blocks that never met the QE gate: verbatim ships
#: (``skip_translate``) and TM exact hits. The heuristic engine tops out at
#: ``QE_SCORE_PASS`` (0.92), so 1.0 here means "not scored", never "perfect".
PLACEHOLDER_MTQE_SCORE = 1.0

#: SQL predicate mirroring :func:`is_qe_scored` for score sets the ledger
#: selects directly (``get_job_stats``'s ``AVG`` over the ``blocks`` table).
#: ``skip_translate`` is stored as 0/1 INTEGER; ``mtqe_score`` is REAL/NULL.
QE_SCORED_SQL = "mtqe_score IS NOT NULL AND skip_translate = 0 AND mtqe_score != 1.0"


def is_qe_scored(block: Any) -> bool:
    """Whether ``block.mtqe_score`` is a real QE score.

    Averages must exclude the 1.0 placeholders on skips/TM hits: mixing them
    in makes a single real defect among hundreds of placeholders read as a
    near-perfect score (one 0.30 defect among 499 skips reads 0.9994).
    Unscored blocks (``mtqe_score is None``, e.g. FastPass-cleared drafts) are
    already excluded by the ``is not None`` filter; this drops the placeholder
    population too.
    """
    score = block.mtqe_score
    return score is not None and not block.skip_translate and score != PLACEHOLDER_MTQE_SCORE


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
