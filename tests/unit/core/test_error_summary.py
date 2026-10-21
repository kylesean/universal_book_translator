"""The failure notice inside a block must fit in a pane and name a cause.

``repair.py`` wraps a failed draft in ``<mark class="ubt-failed-draft"
title="...">`` so the text is kept. The title received ``str(exception)``
verbatim — a whole escaped JSON envelope, ~700 characters on the reference job.
The review workbench renders ``target_text`` as plain text, so the attribute text
became content: an 828px line inside a 446px column, and the page scrolled
sideways. The envelope is also useless to a human; the ledger already has it.
"""

from __future__ import annotations

import pytest

from ubt.core.qe.defect_taxonomy import TRANSIENT_FAILURE_PREFIXES
from ubt.core.qe.error_summary import summarize_provider_error

pytestmark = pytest.mark.fast

_LLAMASWAP = (
    'Model API error (500): {"src":"llama-swap","error":{"message":'
    '"unspecific error: upstream command exited prematurely"}}'
)


def test_the_summary_is_short_and_names_the_cause() -> None:
    summary = summarize_provider_error(_LLAMASWAP)
    assert len(summary) < 100, f"still too long for a title attribute: {summary!r}"
    assert "upstream command exited prematurely" in summary
    assert "llama-swap" not in summary, "the envelope belongs on the ledger"
    assert "500" in summary


def test_the_retry_prefix_is_preserved_exactly() -> None:
    """Load-bearing: resume decides retryability with ``str.startswith``.

    Inserting the status code before the colon (``"Repair error 500: ..."``)
    silently made every provider failure non-retryable, because
    ``TRANSIENT_FAILURE_PREFIXES`` matches on ``"Repair error:"``.
    """
    for raw in (_LLAMASWAP, "Connection refused", "timeout after 30s"):
        summary = summarize_provider_error(raw)
        assert any(summary.startswith(prefix) for prefix in TRANSIENT_FAILURE_PREFIXES), (
            f"{summary!r} would not be retried on resume"
        )


def test_a_long_trailing_explanatory_text_is_capped() -> None:
    summary = summarize_provider_error("Connection reset by peer " + "x" * 500)
    assert len(summary) <= 160
    assert summary.endswith("…"), "a truncated summary must say so"


def test_an_empty_error_still_yields_a_readable_notice() -> None:
    for value in ("", "   "):
        assert summarize_provider_error(value) == "Repair error"


def test_a_malformed_envelope_does_not_raise() -> None:
    """This runs on the error path; a summarizer that raises replaces a block."""
    for raw in ('{"unterminated":', "{not json}", "Error: }{", "\x00"):
        summarize_provider_error(raw)


def _block(**overrides):
    """A minimal IRBlock: only the fields the checkpoint builder reads."""
    from ubt.core.ir.models import BlockStatus, BlockType, FlowID, RegionKind, make_element

    element = make_element(
        id="b1",
        spine_index=0,
        block_type=BlockType.NARRATIVE,
        flow_id=FlowID.MAIN_STORY,
        region=RegionKind.BODY,
        source_text="Original English prose",
        bbox=None,
    )
    from ubt.core.ir.models import IRBlock

    fields = dict(
        element=element,
        target_text="译文正文",
        status=BlockStatus.REPAIRED,
        mtqe_score=0.42,
        repair_rounds=2,
        error_flags=[],
    )
    fields.update(overrides)
    return IRBlock(**fields)


def test_the_checkpoint_keeps_the_translation_and_a_short_reason() -> None:
    """Behaviour, not just the string: this is what reaches ``target_text``.

    The closure that wrote this used to hold ``f"{REPAIR_ERROR_PREFIX} {res}"``,
    so every unit test above still passed while the pane received 620 characters
    of escaped JSON. Testing the builder closes that gap.
    """
    from ubt.core.engine.stages.repair import failed_draft_checkpoint

    result = failed_draft_checkpoint(
        _block(), RuntimeError("Model API error (500): " + _LLAMASWAP.split(": ", 1)[1])
    )
    target = result["target_text"]

    # The translation survives, inside the marker, so nothing is lost.
    assert target.startswith('<mark class="ubt-failed-draft" title="')
    assert target.endswith(">译文正文</mark>")
    # The title is a summary, not the envelope.
    assert "upstream command exited prematurely" in target
    assert len(target) < 300, f"mark still carries the whole envelope: {len(target)} chars"
    assert "&quot;src&quot;" not in target
    # The full error stays on the ledger, where it belongs.
    assert any("llama-swap" in flag for flag in result["error_flags"])


def test_the_checkpoint_falls_back_to_the_source_when_nothing_was_translated() -> None:
    from ubt.core.engine.stages.repair import failed_draft_checkpoint

    result = failed_draft_checkpoint(_block(target_text="", repair_rounds=0), RuntimeError("boom"))
    assert result["target_text"] == "Original English prose"
