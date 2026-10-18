"""Rewrite stored flag texts that an older build phrased in a misleading way.

A ledger is not an archive of text: it is live input. The quality report, the
XLIFF companion a reviewer opens and every later reader take their wording from
it. So when a gate's message is found to be wrong, fixing the gate is only half
the repair -- every flag already written still carries the old sentence, and it
keeps misleading people until the block is re-run (which may be never).

The specific defect this rewrites: ``FastPassFilter`` compared markdown grid
shapes with a message that printed the per-row column list and called it
"columns". ``markdown_grid_shape`` guarantees every row has the same column
count, so a shape difference is always a ROW count -- the old text told a
reviewer the column count oscillated when a row had actually been dropped.

Only the sentence changes. Status, severity, score, repair budget and the set of
flags are untouched: a rewrite that could silently change a verdict would be
worse than the wording it fixes. A flag that does not match the exact old form is
left alone rather than guessed at, so a re-run of this tool is a no-op.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

#: The old mismatch text, capturing both raw per-row lists.
_MISMATCH = re.compile(
    r"the source rows hold (?P<src>\[[0-9, ]*\]) columns, "
    r"the target holds (?P<tgt>\[[0-9, ]*\])"
)

#: The old dropped-table text, capturing the source's per-row list.
_DROPPED = re.compile(r"the source held a (?P<src>\[[0-9, ]*\])-row/column grid")

#: Repair prefixes that may wrap either sentence (the repair prompt puts the
#: reason in ``error_flags`` verbatim, sometimes under a structural-failure label).
_PREFIXES = ("repair_structural_failure: ",)

#: ``(table, column)`` pairs that carry a flags JSON array. The block ledger and
#: the quality-report rewrite both read the same shape.
_LIKE_CLAUSES = "error_flags_json LIKE '%rows hold%' OR error_flags_json LIKE '%row/column grid%'"


def _parse(shape: str) -> list[int]:
    """'[5, 5, 5]' -> [5, 5, 5]."""
    inner = shape.strip()[1:-1].strip()
    if not inner:
        return []
    return [int(part) for part in inner.split(",")]


def _label(shape: str) -> str:
    """'[5, 5, 5]' -> 'a 3-row x 5-column grid'.

    A ragged shape cannot be described this way without losing information, so
    it keeps its raw form rather than being flattened: history recorded
    ``[2, 3, 2]`` for a genuinely misaligned table, and "a 3-row x 2-column grid"
    would erase the row that made it wrong.
    """
    counts = _parse(shape)
    if not counts:
        return "a 0-row x 0-column grid"
    if len(set(counts)) != 1:
        return f"a ragged grid with {len(counts)} row(s) of widths {shape}"
    return f"a {len(counts)}-row x {counts[0]}-column grid"


def rewrite_flag(flag: str) -> str | None:
    """The corrected flag, or ``None`` when this flag needs no rewrite."""
    for prefix in _PREFIXES:
        if flag.startswith(prefix):
            body = flag[len(prefix) :]
            fixed = _rewrite_body(body)
            return None if fixed is None else prefix + fixed
    return _rewrite_body(flag)


def _rewrite_body(body: str) -> str | None:
    mismatch = _MISMATCH.search(body)
    if mismatch:
        corrected = (
            f"the source is {_label(mismatch.group('src'))}, "
            f"the target is {_label(mismatch.group('tgt'))}"
        )
        return body[: mismatch.start()] + corrected + body[mismatch.end() :]
    dropped = _DROPPED.search(body)
    if dropped:
        corrected = f"the source held {_label(dropped.group('src'))}"
        return body[: dropped.start()] + corrected + body[dropped.end() :]
    return None


@dataclass(frozen=True)
class RewriteHit:
    """One block whose stored flags would change."""

    db: Path
    block_id: str
    old_flags: tuple[str, ...]
    new_flags: tuple[str, ...]

    @property
    def changed(self) -> int:
        return sum(1 for a, b in zip(self.old_flags, self.new_flags, strict=True) if a != b)


def _blocks_with_flags(db: Path) -> Iterator[tuple[str, list[str]]]:
    """``(block_id, flags)`` for every block carrying a table-shape flag."""
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    except sqlite3.Error:
        return
    try:
        conn.row_factory = sqlite3.Row
        # A database that is not a job ledger (``tm.sqlite``) has no blocks table;
        # it is not an error, just nothing to do.
        try:
            rows = conn.execute(
                f"SELECT block_id, error_flags_json FROM blocks WHERE {_LIKE_CLAUSES}"
            ).fetchall()
        except sqlite3.Error:
            return
        for row in rows:
            try:
                flags = json.loads(row["error_flags_json"] or "[]")
            except (TypeError, ValueError):
                continue
            if isinstance(flags, list):
                yield str(row["block_id"]), [str(flag) for flag in flags]
    finally:
        conn.close()


def plan(root: Path) -> list[RewriteHit]:
    """Every block under *root* whose stored flags the current wording would change."""
    hits: list[RewriteHit] = []
    for db in sorted(root.glob("*.sqlite")):
        for block_id, flags in _blocks_with_flags(db):
            new_flags = [rewrite_flag(flag) or flag for flag in flags]
            if new_flags != flags:
                hits.append(RewriteHit(db, block_id, tuple(flags), tuple(new_flags)))
    return hits


def apply(hits: list[RewriteHit]) -> int:
    """Write *hits* back, returning the number of flags changed.

    Grouped per database so each file takes one transaction: a half-applied
    ledger is worse than an unapplied one, and a reviewer reading mid-rewrite
    would see a mixed vocabulary.
    """
    changed = 0
    by_db: dict[Path, list[RewriteHit]] = {}
    for hit in hits:
        by_db.setdefault(hit.db, []).append(hit)
    for db, db_hits in by_db.items():
        conn = sqlite3.connect(db)
        try:
            conn.executemany(
                "UPDATE blocks SET error_flags_json = ? WHERE block_id = ?",
                [
                    (json.dumps(list(hit.new_flags), ensure_ascii=False), hit.block_id)
                    for hit in db_hits
                ],
            )
            conn.commit()
        finally:
            conn.close()
        changed += sum(hit.changed for hit in db_hits)
    return changed


__all__ = ["RewriteHit", "apply", "plan", "rewrite_flag"]
