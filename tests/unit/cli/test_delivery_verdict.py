"""The three-state delivery verdict behind the CLI success banner.

``ubt translate`` used to print ``✓ Translation Completed Successfully!`` in
green whenever it exported a file -- even with 20 blocks quarantined and shipped
as untranslated English (arXiv 2610.02185). The verdict now reads the report
summary and distinguishes PASSED / WARNING / BLOCKED, matching the operator
console vocabulary. The artifact is delivered in all three cases; the state only
tells the reader whether they will meet untranslated source. ``--strict``
remains the opt-in hard stop (non-zero exit).
"""

from __future__ import annotations

import pytest

from ubt.cli.commands.translate import _delivery_verdict

pytestmark = pytest.mark.fast


def test_a_clean_run_is_passed() -> None:
    state, message = _delivery_verdict({"completed_blocks": 310})
    assert state == "passed"
    assert message == ""


def test_quarantined_blocks_are_blocked_not_passed() -> None:
    """The exact shape of the 2610.02185 run: 310 done, 20 quarantined."""
    state, message = _delivery_verdict(
        {"completed_blocks": 310, "blocked_human_blocks": 20, "needs_human_blocks": 20}
    )
    assert state == "blocked"
    assert "20" in message


def test_needs_human_alone_is_a_warning() -> None:
    """A block queued for review still ships its translation -- not BLOCKED."""
    state, _ = _delivery_verdict({"completed_blocks": 310, "needs_human_blocks": 5})
    assert state == "warning"


def test_failed_blocks_are_blocked() -> None:
    state, _ = _delivery_verdict({"completed_blocks": 300, "failed_blocks": 3})
    assert state == "blocked"


def test_zero_blocks_through_qa_is_blocked() -> None:
    state, _ = _delivery_verdict({"completed_blocks": 0, "needs_human_blocks": 12})
    assert state == "blocked"


def test_an_empty_summary_does_not_crash() -> None:
    # The CLI only calls the verdict when the summary is non-empty; guard the
    # shape anyway so a partial report cannot raise inside the success path.
    state, _ = _delivery_verdict({})
    assert state == "blocked"
