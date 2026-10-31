"""QE scoring policy: which blocks count toward aggregate scores.

Exclusion is by *provenance*, not by a magic score value: verbatim skips
(``skip_translate``) and exact TM hits (``tm_hit``) carry a pass stamp, not a
measurement, so they must be kept out of averages — while a genuine ``1.0`` from
a real scorer must stay in. The SQL predicate mirrors the Python predicate for
the ledger's direct ``AVG``.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from ubt.core.qe.score_policy import (
    QE_SCORED_SQL,
    is_qe_scored,
    qe_scored_values,
)

pytestmark = pytest.mark.fast


@dataclass
class _B:
    mtqe_score: float | None = None
    skip_translate: bool = False
    tm_hit: bool = False


# --------------------------------------------------------------------------- #
# is_qe_scored
# --------------------------------------------------------------------------- #


def test_an_unscored_block_is_not_qe_scored() -> None:
    assert is_qe_scored(_B(mtqe_score=None)) is False


def test_a_verbatim_skip_is_not_qe_scored() -> None:
    assert is_qe_scored(_B(mtqe_score=1.0, skip_translate=True)) is False


def test_a_tm_hit_is_not_qe_scored() -> None:
    assert is_qe_scored(_B(mtqe_score=1.0, tm_hit=True)) is False


def test_a_real_perfect_score_is_qe_scored() -> None:
    assert is_qe_scored(_B(mtqe_score=1.0)) is True


def test_an_ordinary_score_is_qe_scored() -> None:
    assert is_qe_scored(_B(mtqe_score=0.3)) is True


def test_a_block_without_a_tm_hit_attribute_is_scored() -> None:
    block: Any = SimpleNamespace(mtqe_score=0.5, skip_translate=False)
    assert is_qe_scored(block) is True


# --------------------------------------------------------------------------- #
# qe_scored_values
# --------------------------------------------------------------------------- #


def test_values_are_ascending_and_filtered() -> None:
    blocks = [
        _B(mtqe_score=0.9),
        _B(mtqe_score=None),
        _B(mtqe_score=0.2),
        _B(mtqe_score=1.0, skip_translate=True),
        _B(mtqe_score=1.0, tm_hit=True),
        _B(mtqe_score=0.5),
    ]
    assert qe_scored_values(blocks) == [0.2, 0.5, 0.9]


def test_no_scored_blocks_yields_no_values() -> None:
    assert qe_scored_values([_B(), _B(mtqe_score=1.0, tm_hit=True)]) == []


def test_the_sql_predicate_matches_the_python_predicate_shape() -> None:
    assert "mtqe_score IS NOT NULL" in QE_SCORED_SQL
    assert "skip_translate = 0" in QE_SCORED_SQL
    assert "COALESCE(tm_hit, 0) = 0" in QE_SCORED_SQL
