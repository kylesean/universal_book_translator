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

from ubt.core.exceptions import (
    IntegrityViolationError,
    ProviderUnreachableError,
    public_job_error,
)
from ubt.core.ir.models import (
    BlockStatus,
    BlockType,
    FlowID,
    IRBlock,
    make_element,
)
from ubt.core.qe.defect_taxonomy import TRANSIENT_FAILURE_PREFIXES
from ubt.core.qe.error_summary import summarize_provider_error
from ubt.model.ast import RegionKind

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


def _block(*, target_text: str = "译文正文", repair_rounds: int = 2) -> IRBlock:
    """A block carrying only the fields the checkpoint builder reads.

    Constructed explicitly rather than from a ``dict`` splat: ``mypy --strict``
    cannot check a ``**dict`` against ``IRBlock``'s fields, and it reported eleven
    ``arg-type`` errors for a helper whose whole job is to be a valid block.
    """
    return IRBlock(
        element=make_element(
            id="b1",
            spine_index=0,
            block_type=BlockType.NARRATIVE,
            flow_id=FlowID.MAIN_STORY,
            region=RegionKind.BODY,
            source_text="Original English prose",
            bbox=None,
        ),
        target_text=target_text,
        status=BlockStatus.REPAIRED,
        mtqe_score=0.42,
        repair_rounds=repair_rounds,
        error_flags=[],
    )


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


def test_an_integrity_gate_explains_how_to_recover() -> None:
    """The completion floor is a deliberate gate, not an opaque crash.

    ``IntegrityViolationError`` messages name the missing translations and the
    recovery action ("resume" or lower the floor). Masking them as
    ``(see server logs)`` left the operator unable to tell a partial book from
    a corrupt ledger.
    """
    error = IntegrityViolationError(
        "Export blocked: only 0% of 218 blocks carry a translation. Resume the job to retry them."
    )

    message = public_job_error(error, "job-1")

    assert message.startswith("IntegrityViolationError: ")
    assert "Resume the job" in message
    assert "see server logs" not in message


def test_generic_errors_do_not_expose_provider_bodies_or_host_paths() -> None:
    error = RuntimeError('Model API error: {"body":"secret manuscript"} at /home/operator/book.pdf')

    message = public_job_error(error, "job-1")

    assert message == "RuntimeError (see server logs; job_id=job-1)"
    assert "secret manuscript" not in message
    assert "/home/operator" not in message


def test_a_tripped_fail_fast_circuit_names_the_dead_endpoint() -> None:
    """The breaker aborts a dead provider, so the console must be able to map it.

    The breaker's message carries the original ``transport_unreachable``
    detail; folding it into "(see server logs)" left the failure banner
    generic exactly when the fix is specific (start the gateway, fix the URL).
    """
    error = ProviderUnreachableError(
        "Draft aborted after 30 consecutive non-retryable provider failures "
        "(fail-fast circuit). Last error: HTTP connection failed: All connection "
        "attempts failed [details={'fail_fast': True, 'reason': 'transport_unreachable'}]"
    )

    message = public_job_error(error, "job-1")

    assert message.startswith("ProviderUnreachableError: ")
    assert "transport_unreachable" in message
    assert "see server logs" not in message
