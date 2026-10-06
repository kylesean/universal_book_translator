"""The P2 end-to-end smoke: a rehearsal run delivers a book with zero tokens.

This is the pipeline's own smoke gate (P2 of ``docs/guides/TESTING_STRATEGY.md``):
``create_dry_run_orchestrator`` assembles the *real* orchestrator -- the echo
provider, mock QE, and the real MarkdownAdapter -- and the whole staged plan
runs against a synthetic two-chapter book on disk. Nothing is stubbed inside the
pipeline itself; only the LLM/QE hop is a double.

Pinned contracts (the manual E2E spec's automated core):

- **Reaches a delivered artifact**: the final event is ``EXPORT_COMPLETED``
  carrying the requested output path; the job finalizes ``completed`` in the
  ledger; every block lands in a terminal state with no failed/human-blocked
  rows -- the rehearsal proves the *plumbing*, not translation quality.
- **The delivered book resembles the book**: headings keep their level,
  every prose paragraph carries the rehearsal echo, the code fence is
  byte-identical (``skip_translate``), and masked spans (inline math, the
  citation) are restored to their source form -- masking + restore is lossless
  through the whole chain, not just in unit isolation.
- **The audit trail is complete**: the ``*_quality_report.json`` sidecar
  exists, parses, and reports full completion for this job.
- **The XLIFF companion leaves with the delivery** (emit_xliff_companion): it
  exists beside the artifact, parses as XLIFF, and carries segments whose
  target is the rehearsal echo.
"""

from __future__ import annotations

import json
from pathlib import Path

from ubt.core.config import UBTConfig
from ubt.core.engine.dry_run import create_dry_run_orchestrator
from ubt.core.engine.events import EventType, TranslationProgressEvent
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.ir.models import TERMINAL_STATUSES
from ubt.core.job_options import companion_path, sidecar_path
from ubt.core.qe.fast_pass import REHEARSAL_MARKER
from ubt.segment.xliff import from_xliff

_CODE_LINE = "def train(model): return model"
_MATH = "$e^{i\\pi} + 1 = 0$"

_BOOK = f"""# The Attention Machine

The machine relies on attention. Each layer runs `self.forward(x)` over the batch.

Euler wrote {_MATH} linking five constants of analysis.

Following Smith et al. (2020), the score reached 98.45% on every benchmark.

```python
{_CODE_LINE}
```

# The Descendants

The second chapter widens the lens to every descendant of the first model.
"""


def _config(tmp_path: Path) -> UBTConfig:
    """A hermetic config: ledgers and memory under tmp, no page-image egress."""
    return UBTConfig().model_copy(
        update={
            "db_dir": tmp_path / "ledgers",
            "ocr_mode": "off",
            "allow_page_upload": False,
            "visual_judge_enabled": False,
        }
    )


async def _run_rehearsal(tmp_path: Path) -> tuple[list[TranslationProgressEvent], Path, Path, str]:
    book = tmp_path / "book.md"
    book.write_text(_BOOK, encoding="utf-8")
    output = tmp_path / "book_bilingual.md"
    job_id = "smoke-rehearsal"

    orchestrator = create_dry_run_orchestrator(_config(tmp_path))
    events: list[TranslationProgressEvent] = []
    async for event in orchestrator.run(book, output, job_id=job_id):
        events.append(event)
    return events, book, output, job_id


async def test_the_rehearsal_reaches_a_delivered_artifact(tmp_path: Path) -> None:
    events, _, output, job_id = await _run_rehearsal(tmp_path)

    assert events, "the pipeline yielded nothing"
    assert events[0].event_type is EventType.JOB_STARTED
    terminal = events[-1]
    assert terminal.event_type is EventType.EXPORT_COMPLETED
    assert terminal.artifact_path == str(output)
    assert not any(e.event_type is EventType.PIPELINE_FAILED for e in events)

    db_path = tmp_path / "ledgers" / f"{job_id}.sqlite"
    ledger = SQLiteJobLedger(db_path)
    try:
        assert ledger.get_job_status(job_id) == "completed"
        blocks = ledger.get_all_blocks(job_id)
        assert blocks, "ingest recorded no blocks"
        assert all(block.status in TERMINAL_STATUSES for block in blocks)
        bad = [b.id for b in blocks if b.status.value in ("failed", "needs_human", "blocked_human")]
        assert bad == []
        # Every translatable block was drafted and carries the rehearsal echo.
        prose = [b for b in blocks if not b.skip_translate and b.source_text.strip()]
        assert prose
        assert all(b.target_text and REHEARSAL_MARKER in b.target_text for b in prose)
        # A deliberately kept block (the code listing, a citation-classified
        # sentence) ships verbatim: target == source, never a rehearsal echo.
        kept = [b for b in blocks if b.skip_translate]
        assert kept and all(b.target_text == b.source_text for b in kept)
        code = [b for b in kept if _CODE_LINE in b.source_text]
        assert code
    finally:
        ledger.close()


async def test_the_delivered_book_resembles_the_book(tmp_path: Path) -> None:
    _, _, output, _ = await _run_rehearsal(tmp_path)

    assert output.exists()
    delivered = output.read_text(encoding="utf-8")
    # Chapter titles survive the echo.
    for title in ("The Attention Machine", "The Descendants"):
        echoed = [line for line in delivered.splitlines() if title in line]
        assert echoed, f"chapter title lost: {title}"
        assert any(REHEARSAL_MARKER in line for line in echoed)
    # Every translated prose paragraph came back as the rehearsal echo of itself.
    prose_fragments = (
        "The machine relies on attention",
        "Euler wrote",
        "The second chapter widens the lens",
    )
    for fragment in prose_fragments:
        echoed = [line for line in delivered.splitlines() if fragment in line]
        assert echoed, f"prose lost: {fragment}"
        # Inline dual mode pairs the source line with its echo; exactly one of
        # them carries the rehearsal marker.
        assert any(REHEARSAL_MARKER in line for line in echoed)
        assert any(REHEARSAL_MARKER not in line for line in echoed)
    # Masked spans are restored to their source form, kept blocks ship verbatim,
    # and the code fence is byte-identical.
    assert _MATH in delivered
    assert "Smith et al. (2020)" in delivered
    assert "98.45%" in delivered
    assert _CODE_LINE in delivered


async def test_the_audit_trail_is_complete(tmp_path: Path) -> None:
    events, _, output, job_id = await _run_rehearsal(tmp_path)
    total = events[-1].total_blocks

    report_path = sidecar_path(output, "quality_report.json")
    assert report_path.exists(), "no quality report beside the artifact"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["job_id"] == job_id
    summary = report["summary"]
    assert summary["total_blocks"] == total > 0
    assert summary["completed_blocks"] == summary["total_blocks"]


async def test_the_xliff_companion_leaves_with_the_delivery(tmp_path: Path) -> None:
    _, _, output, _ = await _run_rehearsal(tmp_path)

    companion = companion_path(output, ".xliff")
    assert companion.exists(), "no XLIFF companion beside the artifact"
    parsed = from_xliff(companion.read_text(encoding="utf-8"))
    assert parsed.segments, "the companion carries no segments"
    assert parsed.src_lang == "en"
    assert parsed.trg_lang == "zh"
    assert all(segment.source for segment in parsed.segments)
    # The delivered side carries the rehearsal echo.
    assert any(REHEARSAL_MARKER in (segment.target or "") for segment in parsed.segments)
    # Protected spans survive as inline codes: the code call and the inline math
    # round-trip with token, kind and original intact.
    placeholders = [p for s in parsed.segments for p in s.placeholders]
    kinds = {(p.kind, p.original) for p in placeholders}
    assert ("code", "`self.forward(x)`") in kinds
    assert ("math", _MATH) in kinds
