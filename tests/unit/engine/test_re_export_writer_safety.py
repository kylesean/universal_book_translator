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
        ledger.set_job_metadata_value(_JOB, "output_file", str(tmp_path / "out.pdf"))
    monkeypatch.setattr(
        re_export_module,
        "_default_collaborators",
        lambda config, router, qe: (object(), object()),
    )
    with pytest.raises(UnsupportedDocumentFormatError, match="would not be a readable pdf"):
        await re_export_job(db_path=db_path, job_id=_JOB, config=config)


async def test_re_export_closes_the_paid_visual_judge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ubt.core.engine.re_export as re_export_module

    blocks = [_block(1, "Done sentence.", "已完成。", BlockStatus.MTQE_PASSED)]
    config, db_path = _seed(tmp_path, blocks, job_status="failed")
    config.visual_judge_enabled = True
    with SQLiteJobLedger(db_path) as ledger:
        ledger.set_job_metadata_value(_JOB, "output_file", str(tmp_path / "out.md"))
    monkeypatch.setattr(
        re_export_module,
        "_default_collaborators",
        lambda config, router, qe: (object(), object()),
    )

    seen: dict[str, object] = {}

    async def _capture_ctx(ctx: object, *args: object, **kwargs: object) -> object:
        seen["visual_judge_enabled"] = ctx.config.visual_judge_enabled  # type: ignore[attr-defined]
        return
        yield  # pragma: no cover - never iterated

    monkeypatch.setattr(re_export_module, "run_export_stage", _capture_ctx)
    with pytest.raises(ReExportError, match="produced no artifact"):
        await re_export_job(db_path=db_path, job_id=_JOB, config=config)
    assert seen["visual_judge_enabled"] is False


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
