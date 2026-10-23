"""Upgrade stored free-text flags to the structured ``<category>:<severity>:<detail>`` form.

A ledger is live input, not an archive: the quality report, the review
workbench, the XLIFF companion and every later reader classify blocks from the
flags stored in it. Legacy flags are classified by substring tables, which is
why a rename in one stage can silently change a verdict in another. The
structured form removes that coupling by carrying the category and severity in
the flag itself.

This module performs the **storage** half of that migration: it rewrites the
legacy spellings it can classify with certainty, and leaves everything else
byte-for-byte alone. Two properties are non-negotiable, and both are tested:

* **No verdict may change.** The rewrite is applied only when the legacy flag
  and its structured replacement classify identically — same category, same
  severity, same routing outcome under ``classify_defect``. A flag whose
  severity the tables cannot state with certainty is skipped rather than
  guessed, because a wrong severity is worse than an old spelling.
* **It is idempotent.** A flag already in structured form is left alone, so a
  second run changes nothing.

Deliberately separate from :mod:`ubt.core.engine.flag_rewrite`, which fixes
*wording* while preserving the flag's encoding. The two have different
preconditions and different blast radii; sharing one pass would make it
impossible to run either without the other.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from ubt.core.qe.defect_taxonomy import (
    classify_defect,
    format_defect_flag,
    has_critical_defect,
    has_structural_defect,
    is_format_only,
    parse_defect_flag,
)

#: ``(category, detail-prefix)`` for legacy spellings whose severity is settled.
#:
#: Only markers whose severity the taxonomy states *unambiguously* are listed.
#: A structural marker that is not critical is ``major`` by the triage contract;
#: a critical one is ``critical``. Anything the tables cannot place is absent on
#: purpose — see :func:`upgrade_flag`.
_CATEGORY_BY_MARKER: dict[str, str] = {
    "Empty target text": "empty_target",
    "Numeric fidelity": "numeric_fidelity",
    "Omission suspected": "omission",
    "Table dropped": "table_dropped",
    "Table grid mismatch": "table_grid",
    "Heading label dropped": "heading_label",
    "Added reference": "added_reference",
    "Prompt scaffold": "prompt_scaffold",
    "Prompt template XML artifacts": "prompt_leak",
    "Repetitive loop hallucination": "repetition",
    "Math span mismatch": "math_span",
    "Undelimited math": "undelimited_math",
    "Hallucinated LaTeX": "hallucinated_latex",
    "Glossary term violation": "terminology_consistency",
    "html_structure_mismatch": "html_structure_mismatch",
    "HTML delta failure": "html_formatting_drift",
    "email_token_corrupt": "email_token_corrupt",
    "soup_token_corrupt": "soup_token_corrupt",
    "math_token_corrupt": "math_token_corrupt",
    "cite_token_corrupt": "cite_token_corrupt",
    "code_token_corrupt": "code_token_corrupt",
    "Visual witness discrepancy": "visual_witness",
    "Target text suspiciously truncated": "length_truncated",
    "Target text suspiciously inflated": "length_inflated",
    "script density": "script_density",
    "Foreign-script text": "foreign_script",
}


def _category_for(flag: str) -> str | None:
    """The structured category a legacy flag belongs to, if unambiguous."""
    # Longest marker first: "Repetitive loop hallucination" must win over
    # "repetition", and "Table dropped" over any shorter prefix of itself.
    for marker in sorted(_CATEGORY_BY_MARKER, key=len, reverse=True):
        if marker in flag:
            return _CATEGORY_BY_MARKER[marker]
    return None


def _severity_for(flag: str) -> str:
    """The severity the taxonomy assigns this flag, by its own rules.

    Format-only classes come first. They are also in the structural table (a
    dropped markup tag is still a real defect), so checking structural first
    labelled ``HTML delta failure`` as ``major`` — and a format-only flag must
    stay minor, or the repair loop stops taking its cheap path for exactly the
    defects that path exists for.
    """
    if is_format_only([flag]):
        return "minor"
    if has_critical_defect([flag]):
        return "critical"
    if has_structural_defect([flag]):
        return "major"
    return "minor"


def _detail(flag: str, category: str) -> str:
    """The flag's own text, with the marker's category removed when it prefixes it.

    ``Numeric fidelity failure: dropped 42`` becomes
    ``numeric_fidelity:critical:failure: dropped 42`` rather than repeating the
    category inside the detail — the structured form already carries it, and a
    duplicated prefix is noise in the review workbench.
    """
    for marker in sorted(_CATEGORY_BY_MARKER, key=len, reverse=True):
        if _CATEGORY_BY_MARKER[marker] != category:
            continue
        if flag.startswith(marker):
            rest = flag[len(marker) :].lstrip(": ").strip()
            return rest or marker
    return flag


def upgrade_flag(flag: str) -> str | None:
    """The structured form of *flag*, or ``None`` when it must be left alone.

    Returns ``None`` for:

    * a flag already in structured form (idempotence);
    * a lifecycle note, which is not a quality defect and whose readers match
      by prefix (``Drafting error:`` and friends are in
      ``TRANSIENT_FAILURE_PREFIXES``);
    * a flag no table places, whose severity would have to be guessed.

    Every rewrite is verified against the routing rule before it is returned:
    the legacy and structured forms must classify identically. That check is
    the whole safety argument for touching a live ledger.
    """
    if parse_defect_flag(flag) is not None:
        return None
    if _is_lifecycle(flag):
        return None
    category = _category_for(flag)
    if category is None:
        return None
    severity = _severity_for(flag)
    upgraded = format_defect_flag(category, severity, _detail(flag, category))

    # The safety check: same verdict, same severity, same format-only answer.
    if has_critical_defect([flag]) != has_critical_defect([upgraded]):
        return None
    if has_structural_defect([flag]) != has_structural_defect([upgraded]):
        return None
    if is_format_only([flag]) != is_format_only([upgraded]):
        return None
    if classify_defect([flag], score=0.9, threshold=0.75) != classify_defect(
        [upgraded], score=0.9, threshold=0.75
    ):
        return None
    return upgraded


def _is_lifecycle(flag: str) -> bool:
    from ubt.core.qe.defect_taxonomy import (
        DRAFTING_ERROR_PREFIX,
        NON_RETRYABLE_DRAFT_PREFIX,
        REPAIR_ERROR_PREFIX,
        UNTRANSLATED_PREFIX,
    )

    lifecycle = (
        DRAFTING_ERROR_PREFIX,
        NON_RETRYABLE_DRAFT_PREFIX,
        REPAIR_ERROR_PREFIX,
        UNTRANSLATED_PREFIX,
    )
    return flag.startswith(lifecycle)


@dataclass(frozen=True)
class UpgradeHit:
    """One block whose stored flags the upgrade would rewrite."""

    db: Path
    block_id: str
    old_flags: tuple[str, ...]
    new_flags: tuple[str, ...]

    @property
    def changed(self) -> int:
        return sum(1 for a, b in zip(self.old_flags, self.new_flags, strict=True) if a != b)


def _blocks(db: Path) -> Iterator[tuple[str, list[str]]]:
    """``(block_id, flags)`` for every block with at least one flag."""
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    except sqlite3.Error:
        return
    try:
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT block_id, error_flags_json FROM blocks "
                "WHERE error_flags_json IS NOT NULL AND error_flags_json != '[]'"
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


def plan(root: Path) -> list[UpgradeHit]:
    """Every block under *root* the upgrade would change."""
    hits: list[UpgradeHit] = []
    for db in sorted(root.glob("*.sqlite")):
        for block_id, flags in _blocks(db):
            upgraded = [upgrade_flag(flag) or flag for flag in flags]
            if upgraded != flags:
                hits.append(UpgradeHit(db, block_id, tuple(flags), tuple(upgraded)))
    return hits


def apply(hits: list[UpgradeHit]) -> int:
    """Write *hits* back; returns the number of flags changed.

    One transaction per database: a half-upgraded ledger would mix encodings
    within a single job, which is exactly the state the readers must not see.
    """
    changed = 0
    by_db: dict[Path, list[UpgradeHit]] = {}
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


__all__ = ["UpgradeHit", "apply", "plan", "upgrade_flag"]
