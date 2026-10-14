"""Human edits must reach the delivered file (P1-C6).

A reviewer edits a segment in the L3 workbench; the edit lands in the ledger
(``apply_human_edit``). Before this, nothing re-rendered the deliverable, so the
reviewer saw the correction in the single-page preview while the customer kept
downloading the stale machine draft. ``re_export_job`` closes that gap: it
rebuilds the run's inputs from the ledger and re-runs the export stage over the
current text.

Needs the external ``typst`` compiler (the render path shells out to it), so it
lives in the slow tier and skips when typst is absent.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest
from pdf_builders import write_text_pdf

import ubt.adapters.pdf.oxide_render as oxide_render
from ubt.api.review import apply_human_edit
from ubt.core.config import UBTConfig
from ubt.core.engine.dry_run import create_dry_run_orchestrator
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

#: A marker no rehearsal echo would ever produce, so finding it in the delivered
#: PDF proves the human edit reached the file rather than the machine draft.
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
    # The API manager and the queue worker persist the artifact triple from the
    # terminal event (``persist_progress_metadata``); a bare orchestrator has no
    # completion hook, so do it here exactly as they do — otherwise a re-export
    # has no recorded output path to overwrite.
    assert terminal is not None
    db_path = config.db_dir / f"{job_id}.sqlite"
    with SQLiteJobLedger(db_path) as ledger:
        persist_progress_metadata(terminal, ledger, job_id)

    # Pick a translatable block and revise it.
    with SQLiteJobLedger(db_path, read_only=True) as ledger:
        blocks = ledger.get_all_blocks(job_id)
    target = next(b for b in blocks if b.source_text.strip() and not b.skip_translate)
    apply_human_edit(db_path, job_id, target.id, _EDITED)
    # Re-export spends nothing (every block is terminal), so the dry-run
    # orchestrator's echo router is the right collaborator to hand in: no live
    # provider is contacted.
    return output, job_id, target.id, orchestrator


async def test_a_human_edit_reaches_the_delivered_file(tmp_path: Path) -> None:
    output, job_id, _block_id, orchestrator = await _run_then_edit(tmp_path)
    config = _config(tmp_path)
    db_path = config.db_dir / f"{job_id}.sqlite"

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
    """With no explicit path, the re-export overwrites the deliverable in place."""
    output, job_id, _, orchestrator = await _run_then_edit(tmp_path)
    config = _config(tmp_path)
    db_path = config.db_dir / f"{job_id}.sqlite"

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
