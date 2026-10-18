"""Rewriting stored flag text: only the sentence, never the verdict.

The defect being repaired: ``FastPassFilter`` printed the per-row column list and
called it "columns". ``markdown_grid_shape`` guarantees every row shares one
column count, so a shape difference is always a ROW count -- the old text told a
reviewer the column count oscillated when a row had been dropped.

These pin the two properties that make the rewrite safe to run on a real ledger:
it changes nothing but the message, and it is idempotent and conservative (an
unrecognised flag is left untouched rather than guessed at).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from ubt.core.engine.flag_rewrite import apply, plan, rewrite_flag

pytestmark = pytest.mark.fast

_OLD_MISMATCH = (
    "Table grid mismatch: the source rows hold [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5] "
    "columns, the target holds [5, 5, 5, 5, 5, 5, 5, 5, 5, 5] — keep the row and "
    "column count identical and translate only the text inside each cell"
)
_OLD_DROPPED = (
    "Table dropped: the source held a [14, 14, 14, 14, 14, 14, 14, 14, 14]"
    "-row/column grid but the target has none — translate the table in place "
    "instead of flattening it to prose"
)


def test_a_mismatch_is_restated_as_rows_and_columns() -> None:
    fixed = rewrite_flag(_OLD_MISMATCH)
    assert fixed is not None
    assert "the source is a 11-row x 5-column grid" in fixed
    assert "the target is a 10-row x 5-column grid" in fixed
    # The instruction half of the message is preserved verbatim.
    assert fixed.endswith("translate only the text inside each cell")
    # And the misleading phrasing is gone.
    assert "rows hold" not in fixed
    assert "[5, 5" not in fixed


def test_a_dropped_table_is_restated_too() -> None:
    fixed = rewrite_flag(_OLD_DROPPED)
    assert fixed is not None
    assert "the source held a 9-row x 14-column grid" in fixed
    assert "row/column grid" not in fixed


def test_a_repair_prefixed_flag_keeps_its_prefix() -> None:
    """The repair prompt puts the reason in error_flags verbatim under a label."""
    prefixed = f"repair_structural_failure: {_OLD_MISMATCH}"
    fixed = rewrite_flag(prefixed)
    assert fixed is not None
    assert fixed.startswith("repair_structural_failure: ")
    assert "11-row x 5-column" in fixed


def test_a_ragged_shape_keeps_its_raw_widths() -> None:
    """A misaligned table must not be flattened into a uniform description.

    History recorded ``[2, 3, 2]`` for a genuinely ragged target. Saying "a
    3-row x 2-column grid" would erase the row that made it wrong, which is the
    same class of error this rewrite exists to fix.
    """
    ragged = (
        "Table grid mismatch: the source rows hold [2, 2, 2] columns, the target "
        "holds [2, 3, 2] — keep the row and column count identical"
    )
    fixed = rewrite_flag(ragged)
    assert fixed is not None
    assert "the source is a 3-row x 2-column grid" in fixed
    assert "ragged" in fixed
    assert "[2, 3, 2]" in fixed


@pytest.mark.parametrize(
    "flag",
    [
        "Glossary term violation: 三角 -> 三角形",
        "Numeric fidelity failure: 12 -> 21",
        "Table grid mismatch: something else entirely",
        "",
    ],
)
def test_an_unrecognised_flag_is_left_alone(flag: str) -> None:
    """Conservative by construction: no match means no change, not a guess."""
    assert rewrite_flag(flag) is None


def test_rewriting_twice_changes_nothing_the_second_time(tmp_path: Path) -> None:
    """Idempotent, so an operator can re-run it without checking first."""
    db = _ledger(tmp_path, {"b1": [_OLD_MISMATCH, "Glossary term violation: A -> B"]})
    first = plan(tmp_path)
    assert len(first) == 1
    assert apply(first) == 1

    assert plan(tmp_path) == []
    # The untouched flag on the same block is byte-identical.
    assert _flags(db, "b1")[1] == "Glossary term violation: A -> B"


def test_a_rewrite_does_not_touch_anything_but_the_flags(tmp_path: Path) -> None:
    """The property that makes this safe on a delivered job's ledger."""
    db = _ledger(
        tmp_path,
        {"b1": [_OLD_MISMATCH]},
        extra={"status": "needs_human", "mqm_severity": "critical", "mtqe_score": 0.42},
    )
    before = _row(db, "b1")
    apply(plan(tmp_path))
    after = _row(db, "b1")

    for field in ("status", "mqm_severity", "mtqe_score", "repair_rounds", "target_text"):
        assert before[field] == after[field], f"{field} must not change"
    assert before["error_flags_json"] != after["error_flags_json"]


def test_a_database_without_a_blocks_table_is_skipped(tmp_path: Path) -> None:
    """``tm.sqlite`` lives beside the ledgers and has no blocks table."""
    conn = sqlite3.connect(tmp_path / "tm.sqlite")
    conn.execute("CREATE TABLE tm (id TEXT)")
    conn.commit()
    conn.close()
    assert plan(tmp_path) == []


def _ledger(
    root: Path,
    flags: dict[str, list[str]],
    *,
    extra: dict[str, object] | None = None,
) -> Path:
    db = root / "job_test.sqlite"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE blocks (block_id TEXT PRIMARY KEY, status TEXT, mqm_severity TEXT, "
        "mtqe_score REAL, repair_rounds INTEGER, target_text TEXT, error_flags_json TEXT)"
    )
    fields: dict[str, object] = {
        "status": "drafted",
        "mqm_severity": None,
        "mtqe_score": None,
        "repair_rounds": 0,
        "target_text": "译文",
    }
    fields.update(extra or {})
    for block_id, block_flags in flags.items():
        conn.execute(
            "INSERT INTO blocks VALUES (?,?,?,?,?,?,?)",
            (
                block_id,
                fields["status"],
                fields["mqm_severity"],
                fields["mtqe_score"],
                fields["repair_rounds"],
                fields["target_text"],
                json.dumps(block_flags, ensure_ascii=False),
            ),
        )
    conn.commit()
    conn.close()
    return db


def _flags(db: Path, block_id: str) -> list[str]:
    raw = _row(db, block_id)["error_flags_json"]
    parsed = json.loads(str(raw))
    return [str(item) for item in parsed]


def _row(db: Path, block_id: str) -> dict[str, object]:
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM blocks WHERE block_id = ?", (block_id,)).fetchone()
    conn.close()
    return dict(row)
