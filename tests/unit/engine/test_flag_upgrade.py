"""The storage migration must not change a single verdict.

``flag_upgrade`` rewrites legacy free-text flags into
``<category>:<severity>:<detail>``. The ledger is live input — the quality
report, the review workbench and every later reader classify blocks from it —
so the only acceptable migration is one that is **verdict-preserving**. That is
not a property a reviewer can check by reading a diff of the mapping table; it
is checked here, exhaustively, against every flag the pipeline emits.

Three properties, in order of importance:

1. **Same routing.** For every legacy flag the package emits, the legacy and
   upgraded forms must agree on ``classify_defect``, ``has_structural_defect``,
   ``has_critical_defect`` and ``is_format_only``. Where they cannot agree the
   migration must decline the rewrite, and this suite asserts that it does.
2. **Idempotence.** Upgrading an already-upgraded flag is a no-op.
3. **Containment.** Lifecycle notes are never rewritten — their readers match
   by prefix and a rewrite would break resume.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from ubt.core.engine.flag_upgrade import UpgradeHit, apply, plan, upgrade_flag
from ubt.core.qe.defect_taxonomy import (
    FORMAT_ONLY_CATEGORIES,
    STRUCTURAL_CATEGORIES,
    classify_defect,
    has_critical_defect,
    has_structural_defect,
    is_format_only,
    parse_defect_flag,
)

pytestmark = pytest.mark.fast

#: Legacy flags the pipeline really emits, one per family. Kept here rather
#: than imported from the producer modules so a producer's own rename cannot
#: silently shrink the corpus this migration is checked against.
_LEGACY_CORPUS: tuple[str, ...] = (
    "Empty target text",
    "Numeric fidelity failure: dropped 42",
    "Omission suspected: target has 3 sentence(s) vs 5 in source",
    "Table dropped: the source held a 3-row grid",
    "Table grid mismatch: the source is a 3-row grid",
    "Heading label dropped: the source heading is labelled '2.1'",
    "Added reference(s): the target cites [12]",
    "Prompt scaffold leaked into the target: the output opens markdown heading(s)",
    "Prompt template XML artifacts leaked into target text",
    "Repetitive loop hallucination detected",
    "Math span mismatch: source carries 2 math span(s)",
    "Undelimited math: source carries math symbols but the target has no $...$ span",
    "Hallucinated LaTeX: target introduces LaTeX control sequence(s)",
    "Glossary term violation: ALU",
    "html_structure_mismatch: expected <p> got <div>",
    "HTML delta failure: <b> -> <strong>",
    "Target text suspiciously truncated (ratio=0.30)",
    "Target text suspiciously inflated (ratio=1.90)",
    "email_token_corrupt missing=1 mismatched=0",
    "soup_token_corrupt missing=0 mismatched=1",
    "math_token_corrupt missing=1 mismatched=0",
    "cite_token_corrupt missing=1 mismatched=0",
    "code_token_corrupt missing=1 mismatched=0",
    "Visual witness discrepancy",
    "Insufficient zh script density (ratio=0.30)",
    "Foreign-script text in a Chinese target: '기' is not Chinese",
    "Target identical to source: the passage was not translated",
    "Target reads as source-language residue: the target's common words are still English",
)


def _verdict(flag: str) -> tuple[bool, bool, bool, str]:
    return (
        has_structural_defect([flag]),
        has_critical_defect([flag]),
        is_format_only([flag]),
        classify_defect([flag], score=0.90, threshold=0.75),
    )


# --------------------------------------------------------------------------- #
# 1. Verdict preservation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("legacy", _LEGACY_CORPUS)
def test_an_upgraded_flag_keeps_its_verdict(legacy: str) -> None:
    """The core safety property, one flag at a time."""
    upgraded = upgrade_flag(legacy)
    if upgraded is None:
        pytest.skip("the migration declines this flag (covered by the next test)")
    assert _verdict(legacy) == _verdict(upgraded), (
        f"{legacy!r} -> {upgraded!r} changed its routing verdict"
    )


def test_every_declined_flag_is_declined_for_a_reason() -> None:
    """A skip is only acceptable for a flag the tables cannot place.

    The failure mode this prevents: a mapping entry whose severity guess
    flips a verdict, silently dropped from the migration so the corpus test
    above passes by skipping everything.
    """
    from ubt.core.qe.defect_taxonomy import ECHO_MARKERS

    declined = [flag for flag in _LEGACY_CORPUS if upgrade_flag(flag) is None]
    for flag in declined:
        assert (
            parse_defect_flag(flag) is not None
            or _is_lifecycle(flag)
            or any(marker in flag for marker in ECHO_MARKERS)
        ), f"{flag!r} was declined but is neither structured, lifecycle, nor an echo"


def _is_lifecycle(flag: str) -> bool:
    from ubt.core.qe.defect_taxonomy import (
        DRAFTING_ERROR_PREFIX,
        NON_RETRYABLE_DRAFT_PREFIX,
        REPAIR_ERROR_PREFIX,
        RETRYABLE_CONFIG_DRAFT_PREFIX,
        UNTRANSLATED_PREFIX,
    )

    return flag.startswith(
        (
            DRAFTING_ERROR_PREFIX,
            NON_RETRYABLE_DRAFT_PREFIX,
            REPAIR_ERROR_PREFIX,
            RETRYABLE_CONFIG_DRAFT_PREFIX,
            UNTRANSLATED_PREFIX,
        )
    )


def test_the_corpus_is_not_mostly_declined() -> None:
    """Guard against the migration becoming a no-op by refusing everything.

    Two of the 28 are echoes, which are deliberately never rewritten: their
    reason text doubles as the repair instruction, so changing its encoding
    would change the prompt the model is sent.
    """
    upgraded = [flag for flag in _LEGACY_CORPUS if upgrade_flag(flag) is not None]
    assert len(upgraded) >= len(_LEGACY_CORPUS) - 2, (
        f"only {len(upgraded)}/{len(_LEGACY_CORPUS)} legacy flags were upgraded; "
        "the mapping table has lost coverage"
    )


# --------------------------------------------------------------------------- #
# 2. Idempotence
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("legacy", _LEGACY_CORPUS)
def test_upgrading_twice_changes_nothing(legacy: str) -> None:
    once = upgrade_flag(legacy)
    if once is None:
        pytest.skip("declined")
    assert upgrade_flag(once) is None, f"{once!r} was rewritten a second time"


# --------------------------------------------------------------------------- #
# 3. Containment
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "lifecycle",
    [
        "Drafting error: provider 500",
        "Drafting config error: missing model",
        "Drafting unrecoverable: 401",
        "Repair error: timeout",
        "untranslated: stale block swept by export",
    ],
)
def test_a_lifecycle_note_is_never_rewritten(lifecycle: str) -> None:
    """``reset_transient_failures`` matches these by prefix."""
    assert upgrade_flag(lifecycle) is None


def test_an_unclassifiable_flag_is_left_alone() -> None:
    assert upgrade_flag("some_vendor_note: 42") is None


# --------------------------------------------------------------------------- #
# The category tables must stay coherent
# --------------------------------------------------------------------------- #


def test_every_format_only_category_is_also_structural() -> None:
    """The legacy tables say a format defect is both; the categories must too.

    This invariant is what makes a format-only upgrade verdict-preserving: the
    flag stays structural (never auto-passed) *and* stays cheap to fix.
    """
    missing = [c for c in FORMAT_ONLY_CATEGORIES if c not in STRUCTURAL_CATEGORIES]
    assert missing == [], f"format-only categories that are not structural: {missing}"


def test_the_upgrade_only_emits_categories_the_taxonomy_knows() -> None:
    """Every category a rewrite produces must be classifiable."""
    from ubt.core.engine import flag_upgrade as module

    known = set(module._CATEGORY_BY_MARKER.values())
    unknown = sorted(known - set(STRUCTURAL_CATEGORIES))
    assert unknown == [], f"categories the upgrade emits but the taxonomy ignores: {unknown}"


def test_the_upgrade_covers_every_structural_marker_it_claims_to() -> None:
    """A marker with no category is a legacy flag that can never be migrated.

    Three exclusions, each deliberate:

    * echo phrasings — the reason doubles as the repair instruction;
    * lifecycle prefixes — not quality defects;
    * ``repetition`` / ``loop hallucination`` — substrings of
      ``Repetitive loop hallucination``, which *is* mapped, so a flag matching
      them is migrated through the longer marker;
    * ``terminology_consistency:`` — already structured; rewriting it is a
      no-op by definition.
    """
    from ubt.core.engine import flag_upgrade as module
    from ubt.core.qe.defect_taxonomy import STRUCTURAL_DEFECT_MARKERS

    covered_elsewhere = {
        "repetition",
        "loop hallucination",
        "terminology_consistency:",
        "Drafting error",
        "Repair error",
        "repair_structural_failure",
    }
    unmapped = [
        marker
        for marker in STRUCTURAL_DEFECT_MARKERS
        if marker not in module._CATEGORY_BY_MARKER
        and marker not in _ECHO_MARKERS
        and marker not in covered_elsewhere
    ]
    assert unmapped == [], f"structural markers the upgrade cannot migrate: {unmapped}"


def _echo_markers() -> tuple[str, ...]:
    from ubt.core.qe.defect_taxonomy import ECHO_MARKERS

    return ECHO_MARKERS


_ECHO_MARKERS = _echo_markers()


# --------------------------------------------------------------------------- #
# The database half
# --------------------------------------------------------------------------- #


def _ledger(tmp_path: Path, rows: dict[str, list[str]]) -> Path:
    db = tmp_path / "job_test.sqlite"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE blocks (block_id TEXT PRIMARY KEY, error_flags_json TEXT)")
    conn.executemany(
        "INSERT INTO blocks VALUES (?, ?)",
        [(bid, json.dumps(flags)) for bid, flags in rows.items()],
    )
    conn.commit()
    conn.close()
    return db


def test_plan_and_apply_rewrite_a_real_ledger(tmp_path: Path) -> None:
    db = _ledger(
        tmp_path,
        {
            "b1": ["Numeric fidelity failure: dropped 42"],
            "b2": ["Drafting error: 500"],
            "b3": [],
        },
    )
    hits = plan(tmp_path)
    assert [h.block_id for h in hits] == ["b1"]

    changed = apply(hits)
    assert changed == 1

    conn = sqlite3.connect(db)
    rows = dict(conn.execute("SELECT block_id, error_flags_json FROM blocks"))
    conn.close()
    assert json.loads(rows["b1"]) == ["numeric_fidelity:critical:failure: dropped 42"]
    # Untouched rows are byte-identical.
    assert json.loads(rows["b2"]) == ["Drafting error: 500"]
    assert json.loads(rows["b3"]) == []


def test_applying_twice_is_a_no_op(tmp_path: Path) -> None:
    _ledger(tmp_path, {"b1": ["Numeric fidelity failure: dropped 42"]})
    apply(plan(tmp_path))
    assert plan(tmp_path) == []


def test_a_non_ledger_database_is_skipped(tmp_path: Path) -> None:
    """``tm.sqlite`` has no blocks table; that is not an error."""
    conn = sqlite3.connect(tmp_path / "tm.sqlite")
    conn.execute("CREATE TABLE other (x INTEGER)")
    conn.commit()
    conn.close()
    assert plan(tmp_path) == []


def test_the_upgrade_and_the_wording_rewrite_are_separate(tmp_path: Path) -> None:
    """``flag_rewrite`` fixes wording; ``flag_upgrade`` fixes encoding.

    Sharing one pass would make it impossible to run either alone, and their
    preconditions differ (one preserves the encoding, the other changes it).
    """
    from ubt.core.engine import flag_rewrite, flag_upgrade

    assert flag_rewrite is not flag_upgrade
    assert UpgradeHit is flag_upgrade.UpgradeHit


# --------------------------------------------------------------------------- #
# Phase 4 gate: what must be true before the substring tables can be deleted
# --------------------------------------------------------------------------- #


def test_the_tables_cannot_be_deleted_while_lifecycle_flags_need_them() -> None:
    """A precondition for deleting the marker tables, asserted rather than noted.

    Deleting ``STRUCTURAL_DEFECT_MARKERS`` is only safe once every flag family
    it classifies has a structured equivalent that preserves the routing
    verdict. Lifecycle notes do not: they carry no severity, because the defect
    is about the provider *call*, not the text. Removing the table today would
    flip ``Drafting error: 500`` from ``DETERMINISTIC_DEFECT`` to ``PASS`` — a
    block whose draft never arrived would be recorded as passing.

    This test exists so the blocker is executable. When the last family gains a
    structured equivalent, it fails and says the deletion is now unblocked.
    """
    from ubt.core.qe.defect_taxonomy import classify_defect, parse_defect_flag

    lifecycle_flags = [
        "Drafting error: provider 500",
        "Repair error: timeout",
        "repair_structural_failure: the rewrite dropped content",
    ]
    unrepresentable = [
        flag
        for flag in lifecycle_flags
        # No structured equivalent exists for any of these...
        if parse_defect_flag(flag) is None
        # ...yet the tables keep them out of the auto-pass path.
        and classify_defect([flag], score=0.95, threshold=0.75) != "PASS"
    ]
    assert unrepresentable, (
        "every lifecycle flag now has a structured equivalent — "
        "STRUCTURAL_DEFECT_MARKERS may be deletable; verify with a real-ledger "
        "replay before doing it"
    )


def test_an_echo_flag_is_deliberately_never_upgraded() -> None:
    """The reason text doubles as the repair instruction."""
    from ubt.core.qe.defect_taxonomy import ECHO_MARKERS

    for marker in ECHO_MARKERS:
        assert upgrade_flag(f"{marker}: the passage was not translated") is None
