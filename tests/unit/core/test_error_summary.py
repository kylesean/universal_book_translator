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


def test_the_repair_stage_actually_uses_the_summary() -> None:
    """The unit tests above pass even if ``repair.py`` ignores this module.

    Reverting the call site to ``f"{REPAIR_ERROR_PREFIX} {res}"`` kept every test
    green, because they only exercised the function. The behaviour that matters is
    which string reaches the ``<mark title=...>``, so assert on the source of the
    stage as well.
    """
    from pathlib import Path

    stage = Path(__file__).resolve().parents[3] / "ubt" / "core" / "engine" / "stages" / "repair.py"
    source = stage.read_text(encoding="utf-8")
    assert "summarize_provider_error(res)" in source, (
        "repair.py must summarize the provider error before putting it in a title"
    )
    # The raw exception must not be interpolated into the attribute again.
    assert 'title="{REPAIR_ERROR_PREFIX} {html.escape(str(res))}"' not in source
