"""The structured-flag layer must never *weaken* a verdict.

Two defects, both introduced by the incremental P2 landing and both live:

1. ``triage._flags_severity`` returned the highest **structured** severity
   whenever any structured flag was present, without consulting the legacy
   tables at all. A batch holding one structured ``minor`` note and one legacy
   ``Numeric fidelity failure`` therefore resolved to ``minor`` — a critical
   block downgraded to a minor one, released by the minor auto-pass branch.

2. ``classify_defect`` ignored structured flags entirely. A writer that
   declared ``critical`` produced a flag the routing rule could not see, so the
   block classified as ``PASS`` at any score above the line. The declaration is
   the whole point of the format; a layer that can only *add* severity must
   never be able to *remove* it.

Both are pinned here against the real functions.
"""

from __future__ import annotations

import pytest

from ubt.core.qe.defect_taxonomy import (
    classify_defect,
    format_defect_flag,
    structured_severity,
)

pytestmark = pytest.mark.fast

_LEGACY_CRITICAL = "Numeric fidelity failure: dropped 42"


def test_structured_severity_reads_only_structured_flags() -> None:
    """The helper's contract: it is the *declared* half, not the whole answer."""
    assert structured_severity([format_defect_flag("x", "minor", "d")]) == "minor"
    assert structured_severity([_LEGACY_CRITICAL]) is None


def test_triage_never_lets_a_structured_minor_downgrade_a_legacy_critical() -> None:
    """The regression: max() must span both encodings, not just the new one."""
    from ubt.core.engine.stages.triage import _flags_severity

    mixed = [
        format_defect_flag("advisory", "minor", "word order differs"),
        _LEGACY_CRITICAL,
    ]
    assert _flags_severity(mixed) == "critical"


def test_triage_takes_the_worse_of_the_two_encodings() -> None:
    from ubt.core.engine.stages.triage import _flags_severity

    # Structured critical + legacy minor -> critical.
    assert (
        _flags_severity(
            [format_defect_flag("terminology_consistency", "critical", "x"), "Low legibility font"]
        )
        == "critical"
    )
    # Structured minor alone stays minor.
    assert _flags_severity([format_defect_flag("advisory", "minor", "x")]) == "minor"


def test_a_structured_critical_flag_is_a_deterministic_defect() -> None:
    """A declared critical defect must route to repair, not PASS on a high score."""
    flag = format_defect_flag("some_new_defect", "critical", "detail")
    assert classify_defect([flag], score=0.95, threshold=0.75) == "DETERMINISTIC_DEFECT"


def test_a_structured_major_flag_is_a_deterministic_defect() -> None:
    flag = format_defect_flag("some_new_defect", "major", "detail")
    assert classify_defect([flag], score=0.95, threshold=0.75) == "DETERMINISTIC_DEFECT"


def test_a_structured_minor_flag_does_not_block_a_good_score() -> None:
    """Minor is advisory by definition; it must not force repair on its own."""
    flag = format_defect_flag("advisory", "minor", "word order differs")
    assert classify_defect([flag], score=0.95, threshold=0.75) == "PASS"


def test_a_structured_minor_flag_still_routes_a_low_score_to_doubt() -> None:
    flag = format_defect_flag("advisory", "minor", "word order differs")
    assert classify_defect([flag], score=0.50, threshold=0.75) == "SEMANTIC_DOUBT"


def test_a_structured_critical_outranks_a_structured_minor() -> None:
    flags = [
        format_defect_flag("advisory", "minor", "a"),
        format_defect_flag("terminology_consistency", "critical", "b"),
    ]
    assert classify_defect(flags, score=0.95, threshold=0.75) == "DETERMINISTIC_DEFECT"
