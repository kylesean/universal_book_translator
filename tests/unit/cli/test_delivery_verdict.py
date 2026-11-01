from __future__ import annotations

import pytest

from ubt.cli.commands.translate import _delivery_verdict

pytestmark = pytest.mark.fast


def test_a_clean_run_is_passed() -> None:
    state, message = _delivery_verdict({"completed_blocks": 310})
    assert state == "passed"
    assert message == ""


def test_quarantined_blocks_are_blocked_not_passed() -> None:
    state, message = _delivery_verdict(
        {"completed_blocks": 310, "blocked_human_blocks": 20, "needs_human_blocks": 20}
    )
    assert state == "blocked"
    assert "20" in message


def test_needs_human_alone_is_a_warning() -> None:
    state, _ = _delivery_verdict({"completed_blocks": 310, "needs_human_blocks": 5})
    assert state == "warning"


def test_failed_blocks_are_blocked() -> None:
    state, _ = _delivery_verdict({"completed_blocks": 300, "failed_blocks": 3})
    assert state == "blocked"


def test_zero_blocks_through_qa_is_blocked() -> None:
    state, _ = _delivery_verdict({"completed_blocks": 0, "needs_human_blocks": 12})
    assert state == "blocked"


def test_an_empty_summary_does_not_crash() -> None:
    state, _ = _delivery_verdict({})
    assert state == "blocked"


def test_qe_measured_flag_is_read_from_the_report() -> None:
    from ubt.cli.commands.translate import _qe_measured_from_report

    assert _qe_measured_from_report({}) is False
    assert _qe_measured_from_report({"score_metrics": {}}) is False
    assert _qe_measured_from_report({"score_metrics": {"qe_measured": False}}) is False
    assert _qe_measured_from_report({"score_metrics": {"qe_measured": True}}) is True
    assert _qe_measured_from_report({"score_metrics": "garbage"}) is False
