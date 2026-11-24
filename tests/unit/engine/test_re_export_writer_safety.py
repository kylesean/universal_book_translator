from __future__ import annotations

from pathlib import Path

import pytest

from ubt.core.config import UBTConfig
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.engine.re_export import ReExportError, re_export_job
from ubt.core.engine.writer_lock import LedgerWriterLock
from ubt.core.exceptions import (
    IntegrityViolationError,
    LedgerWriterLockConflictError,
    UnsupportedDocumentFormatError,
)
from ubt.core.ir.models import (
    BlockStatus,
    BlockType,
    BookManifest,
    BoundingBox,
    ChapterIR,
    IRBlock,
    make_element,
)

pytestmark = pytest.mark.fast

_JOB = "reexport-race-job"


def _block(index: int, source: str, target: str, status: BlockStatus) -> IRBlock:
    element = make_element(
        id=f"b{index:03d}",
        spine_index=index,
        block_type=BlockType.NARRATIVE,
        source_text=source,
        bbox=BoundingBox(page=1, x0=50.0, y0=700.0 - index * 20, x1=350.0, y1=712.0 - index * 20),
    )
    block = IRBlock(element=element)
    block.target_text = target
    block.status = status
    return block


def _seed(
    tmp_path: Path, blocks: list[IRBlock], *, job_status: str = "running"
) -> tuple[UBTConfig, Path]:
    config = UBTConfig(db_dir=tmp_path / "db", allowed_dirs=str(tmp_path))
    config.db_dir.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "in.md"
    source.write_text("# t\n\nDone sentence.\n", encoding="utf-8")
    db_path = config.db_dir / f"{_JOB}.sqlite"
    with SQLiteJobLedger(db_path) as ledger:
        ledger.init_job_from_manifest(
            _JOB, BookManifest(doc_id=_JOB, title="t", source_path=str(source))
        )
        ledger.set_job_metadata_value(_JOB, "source_lang", "en")
        ledger.append_chapter(
            _JOB,
            ChapterIR(doc_id=_JOB, chapter_id="c1", title="C1", spine_index=0, blocks=blocks),
        )
        if job_status != "initialized":
            ledger.finalize_job(_JOB, status=job_status)
    return config, db_path


async def test_re_export_refuses_non_terminal_blocks_instead_of_failing_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    blocks = [
        _block(1, "Done sentence.", "已完成。", BlockStatus.MTQE_PASSED),
        _block(2, "Still drafting here.", "", BlockStatus.DRAFTED),
    ]
    config, db_path = _seed(tmp_path, blocks, job_status="failed")
    with SQLiteJobLedger(db_path) as ledger:
        ledger.set_job_metadata_value(_JOB, "output_file", str(tmp_path / "out.md"))
    import ubt.core.engine.re_export as re_export_module

    dummy_router, dummy_qe = object(), object()
    monkeypatch.setattr(
        re_export_module,
        "_default_collaborators",
        lambda config, router, qe: (dummy_router, dummy_qe),
    )
    with pytest.raises(IntegrityViolationError, match="non-terminal"):
        await re_export_job(db_path=db_path, job_id=_JOB, config=config)
    with SQLiteJobLedger(db_path, read_only=True) as ledger:
        row = ledger.get_block("b002", _JOB)
        assert row is not None
        assert row.status is BlockStatus.DRAFTED
        assert row.error_flags == []
        assert ledger.get_job_status(_JOB) == "failed"


async def test_re_export_refuses_contradicting_output_suffix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ubt.core.engine.re_export as re_export_module

    blocks = [_block(1, "Done sentence.", "已完成。", BlockStatus.MTQE_PASSED)]
    config, db_path = _seed(tmp_path, blocks, job_status="failed")
    with SQLiteJobLedger(db_path) as ledger:
        ledger.set_job_metadata_value(_JOB, "output_file", str(tmp_path / "out.docx"))
    monkeypatch.setattr(
        re_export_module,
        "_default_collaborators",
        lambda config, router, qe: (object(), object()),
    )
    with pytest.raises(UnsupportedDocumentFormatError, match="would not be a readable docx"):
        await re_export_job(db_path=db_path, job_id=_JOB, config=config)


async def test_re_export_holds_the_writer_lock_against_a_live_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    blocks = [_block(1, "Done sentence.", "已完成。", BlockStatus.MTQE_PASSED)]
    config, db_path = _seed(tmp_path, blocks)
    lock = LedgerWriterLock(db_path, _JOB)
    lock.acquire()
    try:
        with pytest.raises(LedgerWriterLockConflictError, match="already being written"):
            await re_export_job(db_path=db_path, job_id=_JOB, config=config)
    finally:
        lock.release()


async def test_re_export_still_reports_missing_output_after_the_terminal_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    blocks = [_block(1, "Done sentence.", "已完成。", BlockStatus.MTQE_PASSED)]
    config, db_path = _seed(tmp_path, blocks)
    with pytest.raises(ReExportError, match="no recorded output path"):
        await re_export_job(db_path=db_path, job_id=_JOB, config=config)


async def test_re_export_refuses_modified_source_fingerprint(tmp_path: Path) -> None:
    blocks = [_block(1, "Done sentence.", "已完成。", BlockStatus.MTQE_PASSED)]
    config, db_path = _seed(tmp_path, blocks, job_status="completed")
    with SQLiteJobLedger(db_path) as ledger:
        ledger.set_job_metadata_value(_JOB, "output_file", str(tmp_path / "out.md"))
        ledger.set_job_metadata_value(_JOB, "source_fingerprint", "0" * 64)
    (tmp_path / "in.md").write_text("# modified\n\nChanged text.\n", encoding="utf-8")

    with pytest.raises(ReExportError, match="source document changed"):
        await re_export_job(db_path=db_path, job_id=_JOB, config=config)


async def test_re_export_persists_delivery_override_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ubt.core.engine.re_export as re_export_module
    from ubt.core.engine.events import EventType, TranslationProgressEvent

    blocks = [_block(1, "Done sentence.", "已完成。", BlockStatus.MTQE_PASSED)]
    config, db_path = _seed(tmp_path, blocks, job_status="completed")
    out_md = tmp_path / "out.md"
    with SQLiteJobLedger(db_path) as ledger:
        ledger.set_job_metadata_value(_JOB, "output_file", str(out_md))

    monkeypatch.setattr(
        re_export_module,
        "_default_collaborators",
        lambda config, router, qe: (object(), object()),
    )

    from collections.abc import AsyncIterator
    from typing import Any

    async def _fake_export(
        ctx: Any,
        _services: Any,
        _render_plan: Any,
        _term: Any,
        *,
        claims_finality: bool = True,
    ) -> AsyncIterator[TranslationProgressEvent]:
        del claims_finality
        yield TranslationProgressEvent(
            event_type=EventType.EXPORT_COMPLETED,
            job_id=ctx.job_id,
            total_blocks=1,
            completed_blocks=1,
            artifact_path=str(out_md),
        )

    monkeypatch.setattr(re_export_module, "run_export_stage", _fake_export)

    event = await re_export_job(db_path=db_path, job_id=_JOB, config=config, delivery="readable")
    assert event.event_type is EventType.EXPORT_COMPLETED
    with SQLiteJobLedger(db_path, read_only=True) as ledger:
        assert ledger.get_job_metadata_value(_JOB, "delivery_mode") == "readable"
        assert ledger.get_job_metadata_value(_JOB, "delivery_mode_effective") == "readable"
        assert ledger.get_job_metadata_value(_JOB, "delivery_mode_requested") == "readable"
        plan = ledger.get_job_metadata_value(_JOB, "delivery_plan")
        assert isinstance(plan, dict) and plan.get("contract") == "readable"
