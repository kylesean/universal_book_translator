from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest
from pdf_builders import write_text_pdf

import ubt.adapters.pdf.oxide_render as oxide_render
from ubt.api.review import apply_human_edit
from ubt.core.config import UBTConfig
from ubt.core.engine.dry_run import create_dry_run_orchestrator, rehearsal_db_dir
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.engine.progress import persist_progress_metadata
from ubt.core.engine.re_export import re_export_job

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        shutil.which("typst") is None,
        reason="the typst compiler is not installed; the render path cannot run",
    ),
]

_PAGE = (
    "The Attention Machine",
    "The machine relies on attention. Each layer runs a forward pass.",
)

_EDITED = "HUMAN-EDITED-SENTINEL-7f3a"


def _config(tmp_path: Path) -> UBTConfig:
    return UBTConfig().model_copy(
        update={
            "db_dir": tmp_path / "ledgers",
            "ocr_mode": "off",
            "allow_page_upload": False,
            "visual_judge_enabled": False,
        }
    )


async def _run_then_edit(tmp_path: Path) -> tuple[Path, str, str, Any]:
    book = write_text_pdf(tmp_path / "book.pdf", [_PAGE])
    output = tmp_path / "book_es.pdf"
    job_id = "reexport-job"
    config = _config(tmp_path)
    orchestrator = create_dry_run_orchestrator(config)
    terminal = None
    async for event in orchestrator.run(book, output, target_lang="es", job_id=job_id):
        terminal = event
    assert terminal is not None
    db_path = rehearsal_db_dir(config.db_dir) / f"{job_id}.sqlite"
    with SQLiteJobLedger(db_path) as ledger:
        persist_progress_metadata(terminal, ledger, job_id)

    with SQLiteJobLedger(db_path, read_only=True) as ledger:
        blocks = ledger.get_all_blocks(job_id)
    target = next(b for b in blocks if b.source_text.strip() and not b.skip_translate)
    apply_human_edit(db_path, job_id, target.id, _EDITED)
    return output, job_id, target.id, orchestrator


async def test_a_human_edit_reaches_the_delivered_file(tmp_path: Path) -> None:
    output, job_id, _block_id, orchestrator = await _run_then_edit(tmp_path)
    config = _config(tmp_path)
    db_path = rehearsal_db_dir(config.db_dir) / f"{job_id}.sqlite"

    before = _flat("\n".join(oxide_render.extract_page_texts(output)))
    assert _EDITED not in before, "the sentinel leaked into the pre-edit render"

    event = await re_export_job(
        db_path=db_path,
        job_id=job_id,
        config=config,
        router=orchestrator.router,
        qe_runner=orchestrator.qe_runner,
    )

    assert event.artifact_path == str(output)
    after = _flat("\n".join(oxide_render.extract_page_texts(output)))
    assert _EDITED in after, "the human edit did not reach the delivered PDF"


async def test_a_re_export_writes_the_original_output_path_by_default(tmp_path: Path) -> None:
    output, job_id, _, orchestrator = await _run_then_edit(tmp_path)
    config = _config(tmp_path)
    db_path = rehearsal_db_dir(config.db_dir) / f"{job_id}.sqlite"

    event = await re_export_job(
        db_path=db_path,
        job_id=job_id,
        config=config,
        router=orchestrator.router,
        qe_runner=orchestrator.qe_runner,
    )

    assert Path(event.artifact_path or "") == output
    assert output.exists()


def _flat(text: str) -> str:
    return " ".join(text.split())


async def test_a_re_export_never_rewrites_a_block(tmp_path: Path) -> None:
    output, job_id, block_id, orchestrator = await _run_then_edit(tmp_path)
    config = _config(tmp_path)
    db_path = rehearsal_db_dir(config.db_dir) / f"{job_id}.sqlite"

    with SQLiteJobLedger(db_path, read_only=True) as ledger:
        before = {
            b.id: (b.target_text, b.status, tuple(b.error_flags))
            for b in ledger.get_all_blocks(job_id)
        }

    await re_export_job(
        db_path=db_path,
        job_id=job_id,
        config=config,
        router=orchestrator.router,
        qe_runner=orchestrator.qe_runner,
    )

    with SQLiteJobLedger(db_path, read_only=True) as ledger:
        after = {
            b.id: (b.target_text, b.status, tuple(b.error_flags))
            for b in ledger.get_all_blocks(job_id)
        }

    changed = {bid for bid in before if before[bid] != after.get(bid)}
    assert not changed, f"re-export rewrote block(s) {sorted(changed)}"
    assert _EDITED in _flat("\n".join(oxide_render.extract_page_texts(output)))
