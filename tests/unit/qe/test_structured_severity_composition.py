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
    assert structured_severity([format_defect_flag("x", "minor", "d")]) == "minor"
    assert structured_severity([_LEGACY_CRITICAL]) is None


def test_triage_never_lets_a_structured_minor_downgrade_a_legacy_critical() -> None:
    from ubt.core.engine.stages.triage import _flags_severity

    mixed = [
        format_defect_flag("advisory", "minor", "word order differs"),
        _LEGACY_CRITICAL,
    ]
    assert _flags_severity(mixed) == "critical"


def test_triage_takes_the_worse_of_the_two_encodings() -> None:
    from ubt.core.engine.stages.triage import _flags_severity

    assert (
        _flags_severity(
            [format_defect_flag("terminology_consistency", "critical", "x"), "Low legibility font"]
        )
        == "critical"
    )
    assert _flags_severity([format_defect_flag("advisory", "minor", "x")]) == "minor"


def test_a_structured_critical_flag_is_a_deterministic_defect() -> None:
    flag = format_defect_flag("some_new_defect", "critical", "detail")
    assert classify_defect([flag], score=0.95, threshold=0.75) == "DETERMINISTIC_DEFECT"


def test_a_structured_major_flag_is_a_deterministic_defect() -> None:
    flag = format_defect_flag("some_new_defect", "major", "detail")
    assert classify_defect([flag], score=0.95, threshold=0.75) == "DETERMINISTIC_DEFECT"


def test_a_structured_minor_flag_does_not_block_a_good_score() -> None:
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
