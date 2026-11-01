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
    return UBTConfig().model_copy(
        update={
            "db_dir": tmp_path / "ledgers",
            "ocr_mode": "off",
            "allow_page_upload": False,
            "visual_judge_enabled": False,
        }
    )


async def _cancelled_mid_draft_run(tmp_path: Path) -> Counter[str]:
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
    ledger = SQLiteJobLedger(rehearsal_db_dir(tmp_path / "ledgers") / f"{_JOB_ID}.sqlite")
    try:
        for block in ledger.get_all_blocks(_JOB_ID):
            statuses[block.status.value] += 1
    finally:
        ledger.close()
    return statuses


async def _ledger_status_counts(tmp_path: Path) -> tuple[str, Counter[str]]:
    ledger = SQLiteJobLedger(rehearsal_db_dir(tmp_path / "ledgers") / f"{_JOB_ID}.sqlite")
    try:
        blocks = ledger.get_all_blocks(_JOB_ID)
        counts: Counter[str] = Counter(b.status.value for b in blocks)
        status = ledger.get_job_status(_JOB_ID)
        return (status or "", counts)
    finally:
        ledger.close()


async def _resume_run(tmp_path: Path) -> list[TranslationProgressEvent]:
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

    delivered = " ".join("\n".join(oxide_render.extract_page_texts(output)).split())
    for marker in ("Attention Machine", "Finale"):
        assert marker in delivered, f"{marker} missing from the delivered PDF"


def test_the_resume_never_re_drafts_a_surviving_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    statuses_before = asyncio.run(_cancelled_mid_draft_run(tmp_path))
    drafted_before = statuses_before.get("drafted", 0)
    assert drafted_before > 0, "the cancellation left nothing drafted; nothing to protect"

    ledger = SQLiteJobLedger(rehearsal_db_dir(tmp_path / "ledgers") / f"{_JOB_ID}.sqlite")
    try:
        drafts_before = {b.id: b.draft_text for b in ledger.get_all_blocks(_JOB_ID)}
    finally:
        ledger.close()

    config = _config(tmp_path).model_copy(update={"cache_enabled": False})

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
