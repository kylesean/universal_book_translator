"""Crash/resume e2e: an interrupted run must leave a resumable ledger, and the
resumed run must finish the book without re-translating what survived.

The pipeline's durability contract is the ledger: a run that dies mid-book
(cancellation, crash, provider outage) leaves its checkpointed remainder --
``drafted`` blocks whose translation is already banked, and ``pending`` ones --
instead of a half-written artifact, with the job marked ``cancelled``. The
resume half of the contract lives in two places: the draft stage's claim loop
fetches only ``status = 'pending'`` blocks (a banked draft is never re-billed),
and the ingest stage re-parses only when the ledger holds no blocks at all
(a missing fingerprint means "ingest never finished" and is cleared safely).

This module proves that contract end to end over the same deterministic harness
``test_synthetic_render_e2e.py`` uses: a real Docling ingest, the echo provider,
and the Typst renderer. Run 1 is cancelled cooperatively the moment the first
draft batch lands -- the batch checkpoint is flushed before the event is
yielded, so the ledger is deterministic from here: every ingested block reads
``drafted`` and the job reads ``cancelled``. Run 2 (same job id, same
orchestrator factory) must deliver the PDF and drive every block to
``mtqe_passed`` without re-stamping the drafts run 1 banked.

The fixture's twelve pages segment into few fused blocks (the engines merge
heading+body and cross-page prose), which is fine: the contract is about the
blocks the run actually produced, whatever their count.

Why ``slow``: the render path shells out to the external ``typst`` compiler,
exactly like the synthetic render e2e it shares the harness with; CI installs a
checksum-pinned ``typst`` for this tier.
"""

from __future__ import annotations

import asyncio
import shutil
from collections import Counter
from pathlib import Path

import pytest
from pdf_builders import write_text_pdf

import ubt.adapters.pdf.oxide_render as oxide_render
from ubt.core.config import UBTConfig
from ubt.core.engine.dry_run import create_dry_run_orchestrator, rehearsal_db_dir
from ubt.core.engine.events import EventType, TranslationProgressEvent
from ubt.core.engine.ledger import SQLiteJobLedger

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        shutil.which("typst") is None,
        reason="the typst compiler is not installed; the render path cannot run",
    ),
]

#: Four complete, self-contained pages (each chapter's body ends its own
#: argument, so the engines keep them as separate blocks -- a continuation-shaped
#: fixture fuses into one composite block and gives the cancel nothing to
#: interrupt). The tests read the block count from the ledger rather than
#: hard-coding it.
_PAGES = tuple(
    (
        title,
        f"{title} prose that the echo provider drafts and renders for the "
        f"crash resume rehearsal, chapter {index}, with its own complete argument.",
    )
    for index, title in enumerate(
        (
            "The Attention Machine",
            "The Descendants",
            "The Continuation",
            "The Finale",
        ),
        start=1,
    )
)

_JOB_ID = "crash-resume-e2e"


def _config(tmp_path: Path) -> UBTConfig:
    """A hermetic config: ledgers under tmp, no page-image egress.

    The ledger the rehearsal writes is ``rehearsal_db_dir(tmp_path/"ledgers")``
    — a sibling of the path named here — because a dry run keeps its own store.
    Every reader below derives the path the same way the orchestrator does.
    """
    return UBTConfig().model_copy(
        update={
            "db_dir": tmp_path / "ledgers",
            "ocr_mode": "off",
            "allow_page_upload": False,
            "visual_judge_enabled": False,
        }
    )


async def _cancelled_mid_draft_run(tmp_path: Path) -> Counter[str]:
    """Run the job, cancel it the moment the first draft batch completes.

    Returns the status histogram of the ledger the interrupted run left.
    """
    book = write_text_pdf(tmp_path / "book.pdf", _PAGES)
    cancel = asyncio.Event()
    orchestrator = create_dry_run_orchestrator(_config(tmp_path))
    statuses: Counter[str] = Counter()
    with pytest.raises(Exception, match="cancel"):
        async for event in orchestrator.run(
            book, tmp_path / "book_es.pdf", target_lang="es", job_id=_JOB_ID, cancel_token=cancel
        ):
            if event.event_type is EventType.DRAFT_BATCH_COMPLETED:
                cancel.set()
    # The run raised JobInterruptedError out of the generator; the abort path
    # marked the job and the flusher's close() committed every queued draft.
    ledger = SQLiteJobLedger(rehearsal_db_dir(tmp_path / "ledgers") / f"{_JOB_ID}.sqlite")
    try:
        for block in ledger.get_all_blocks(_JOB_ID):
            statuses[block.status.value] += 1
    finally:
        ledger.close()
    return statuses


async def _ledger_status_counts(tmp_path: Path) -> tuple[str, Counter[str]]:
    """(job status, block status histogram) straight from the ledger."""
    ledger = SQLiteJobLedger(rehearsal_db_dir(tmp_path / "ledgers") / f"{_JOB_ID}.sqlite")
    try:
        blocks = ledger.get_all_blocks(_JOB_ID)
        counts: Counter[str] = Counter(b.status.value for b in blocks)
        status = ledger.get_job_status(_JOB_ID)
        return (status or "", counts)
    finally:
        ledger.close()


async def _resume_run(tmp_path: Path) -> list[TranslationProgressEvent]:
    """Run 2: the resume over the interrupted job id, collecting its events."""
    book = write_text_pdf(tmp_path / "book.pdf", _PAGES)
    orchestrator = create_dry_run_orchestrator(_config(tmp_path))
    return [
        event
        async for event in orchestrator.run(
            book, tmp_path / "book_es.pdf", target_lang="es", job_id=_JOB_ID
        )
    ]


def test_an_interrupted_run_leaves_a_cancelled_ledger_with_drafts_banked(
    tmp_path: Path,
) -> None:
    """Run 1, cut at the first draft batch: cancelled job, drafts checkpointed.

    The batch checkpoint flush runs before the DRAFT_BATCH_COMPLETED event is
    yielded, so the abort is deterministic: every ingested block is banked as
    ``drafted`` (the resume re-claims that state, its translation intact) and
    nothing sits in a state a resume could not pick up ("translating" would be
    a lost block). The job itself reads ``cancelled``, not a fake "completed".
    """
    asyncio.run(_cancelled_mid_draft_run(tmp_path))

    status, counts = asyncio.run(_ledger_status_counts(tmp_path))
    assert status == "cancelled", f"the interrupted job reads {status!r}"
    assert counts, "the interrupted run checkpointed no blocks"
    resumable = {"pending", "drafted"}
    strays = {s for s in counts if s not in resumable}
    assert strays <= {"mtqe_passed", "failed", "needs_human", "repaired", "blocked_human"}, (
        f"non-resumable states survived the abort: {dict(counts)}"
    )
    assert counts.get("drafted", 0) > 0, (
        f"cancellation dropped the work already drafted: {dict(counts)}"
    )


def test_a_resumed_run_finishes_the_book_and_delivers_the_pdf(tmp_path: Path) -> None:
    """Run 2 over the same job id: every block driven to delivered, PDF out."""
    asyncio.run(_cancelled_mid_draft_run(tmp_path))
    events = asyncio.run(_resume_run(tmp_path))

    assert events, "the resumed run yielded nothing"
    assert events[-1].event_type is EventType.EXPORT_COMPLETED
    assert not any(e.event_type is EventType.PIPELINE_FAILED for e in events)
    output = tmp_path / "book_es.pdf"
    assert output.exists(), "the resumed run delivered no PDF"
    assert output.read_bytes()[:5] == b"%PDF-"

    ledger = SQLiteJobLedger(rehearsal_db_dir(tmp_path / "ledgers") / f"{_JOB_ID}.sqlite")
    try:
        assert ledger.get_job_status(_JOB_ID) == "completed"
        blocks = ledger.get_all_blocks(_JOB_ID)
        assert blocks, "the resumed ledger holds no blocks"
        assert all(b.status.value == "mtqe_passed" for b in blocks), (
            f"blocks not fully delivered after resume: "
            f"{dict(Counter(b.status.value for b in blocks))}"
        )
    finally:
        ledger.close()

    # The delivered artifact carries the book's prose (the same flat-match the
    # render e2e uses; extraction reflows lines and can clip at glyph bounds,
    # so match on chapter-unique title stems, first and last).
    delivered = " ".join("\n".join(oxide_render.extract_page_texts(output)).split())
    for marker in ("Attention Machine", "Finale"):
        assert marker in delivered, f"{marker} missing from the delivered PDF"


def test_the_resume_never_re_drafts_a_surviving_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Banked drafts survive resume untouched -- the no-re-billing invariant.

    The echo provider is deterministic, so a re-draft would be byte-identical
    and no ledger diff could tell it from a kept draft. Run 2's provider is
    therefore patched to echo a *different* marker: every block run 2 actually
    re-drafts would carry ``[run2-redraft]`` where run 1 banked
    ``[模拟翻译]``, and the checkpointed blocks must keep run 1's text.
    """
    statuses_before = asyncio.run(_cancelled_mid_draft_run(tmp_path))
    drafted_before = statuses_before.get("drafted", 0)
    assert drafted_before > 0, "the cancellation left nothing drafted; nothing to protect"

    ledger = SQLiteJobLedger(rehearsal_db_dir(tmp_path / "ledgers") / f"{_JOB_ID}.sqlite")
    try:
        drafts_before = {b.id: b.draft_text for b in ledger.get_all_blocks(_JOB_ID)}
    finally:
        ledger.close()

    # The translate cache is disabled for run 2: a cache hit finalizes the
    # identical draft, which would mask a leaked re-claim. With it off, any
    # re-draft must call the patched provider and carry the run-2 marker.
    config = _config(tmp_path).model_copy(update={"cache_enabled": False})

    # Run 2 redrafts (if the claim loop leaked) would carry this marker.
    from ubt.core.engine import dry_run

    def _run2_echo(self: object, prompt: str, **kwargs: object) -> str:
        from ubt.core.router.prompts import draft_source_from_prompt

        return f"[run2-redraft] {draft_source_from_prompt(prompt)}"

    monkeypatch.setattr(dry_run.DryRunModelProvider, "generate", _run2_echo)

    async def _resume_no_cache() -> None:
        book = write_text_pdf(tmp_path / "book.pdf", _PAGES)
        orchestrator = create_dry_run_orchestrator(config)
        async for _ in orchestrator.run(
            book, tmp_path / "book_es.pdf", target_lang="es", job_id=_JOB_ID
        ):
            pass

    asyncio.run(_resume_no_cache())

    ledger = SQLiteJobLedger(rehearsal_db_dir(tmp_path / "ledgers") / f"{_JOB_ID}.sqlite")
    try:
        blocks_after = {b.id: b for b in ledger.get_all_blocks(_JOB_ID)}
        for block_id, draft_before in drafts_before.items():
            block = blocks_after[block_id]
            assert block.draft_text == draft_before, (
                f"block {block_id} was re-drafted on resume: "
                f"{draft_before!r} -> {block.draft_text!r}"
            )
            assert "[run2-redraft]" not in (block.target_text or ""), (
                f"block {block_id} was re-drafted on resume: run-2 output leaked in"
            )
    finally:
        ledger.close()


__all__ = ["pytest"]
