from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from pdf_builders import write_text_pdf

import ubt.adapters.pdf.oxide_render as oxide_render
from ubt.core.config import UBTConfig
from ubt.core.content.verify import contract_path_for_artifact, load_contract_file
from ubt.core.engine.dry_run import create_dry_run_orchestrator, rehearsal_db_dir
from ubt.core.engine.events import EventType, TranslationProgressEvent
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.ir.models import TERMINAL_STATUSES
from ubt.core.job_options import companion_path

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        shutil.which("typst") is None,
        reason="the typst compiler is not installed; the render path cannot run",
    ),
]

_PAGE_ONE = (
    "The Attention Machine",
    "The machine relies on attention. Each layer runs a forward pass over the batch.",
    "Euler wrote a formula linking five constants of analysis.",
    "Following Smith et al. the score reached 98.45 percent on every benchmark.",
)
_PAGE_TWO = (
    "The Descendants",
    "The second chapter widens the lens to every descendant of the first model.",
)

_PARAGRAPHS = (
    "The machine relies on attention. Each layer runs a forward pass over the batch.",
    "Euler wrote a formula linking five constants of analysis.",
    "Following Smith et al. the score reached 98.45 percent on every benchmark.",
    "The second chapter widens the lens to every descendant of the first model.",
)


def _config(tmp_path: Path) -> UBTConfig:
    return UBTConfig().model_copy(
        update={
            "db_dir": tmp_path / "ledgers",
            "ocr_mode": "off",
            "allow_page_upload": False,
            "visual_judge_enabled": False,
        }
    )


async def _run(tmp_path: Path) -> tuple[list[TranslationProgressEvent], Path, str]:
    book = write_text_pdf(tmp_path / "book.pdf", [_PAGE_ONE, _PAGE_TWO])
    output = tmp_path / "book_es.pdf"
    job_id = "synthetic-render"
    orchestrator = create_dry_run_orchestrator(_config(tmp_path))
    events = []
    async for event in orchestrator.run(book, output, target_lang="es", job_id=job_id):
        events.append(event)
    return events, output, job_id


def _flat(text: str) -> str:
    return " ".join(text.split())


async def test_the_synthetic_pdf_reaches_a_delivered_pdf(tmp_path: Path) -> None:
    events, output, _ = await _run(tmp_path)

    assert events, "the pipeline yielded nothing"
    assert events[0].event_type is EventType.JOB_STARTED
    terminal = events[-1]
    assert terminal.event_type is EventType.EXPORT_COMPLETED
    assert terminal.artifact_path == str(output)
    assert not any(e.event_type is EventType.PIPELINE_FAILED for e in events)

    assert output.exists(), "no PDF was delivered"
    assert output.read_bytes()[:5] == b"%PDF-", "the artifact is not a PDF"


async def test_the_delivered_pdf_carries_every_paragraph(tmp_path: Path) -> None:
    _, output, _ = await _run(tmp_path)

    delivered = _flat("\n".join(oxide_render.extract_page_texts(output)))
    for title in ("The Attention Machine", "The Descendants"):
        assert _flat(title) in delivered, f"title lost: {title}"
    for paragraph in _PARAGRAPHS:
        assert _flat(paragraph) in delivered, f"paragraph lost: {paragraph}"


async def test_the_delivered_contract_is_balanced(tmp_path: Path) -> None:
    _, output, _ = await _run(tmp_path)

    contract = load_contract_file(contract_path_for_artifact(output))
    assert not contract.errors, contract.errors
    assert contract.missing_assets == 0
    assert contract.source_kept_text == 0
    assert contract.total_text > 0
    assert contract.delivered_text == contract.total_text


async def test_the_ledger_records_a_completed_job(tmp_path: Path) -> None:
    _, _, job_id = await _run(tmp_path)

    ledger = SQLiteJobLedger(rehearsal_db_dir(tmp_path / "ledgers") / f"{job_id}.sqlite")
    try:
        assert ledger.get_job_status(job_id) == "completed"
        blocks = ledger.get_all_blocks(job_id)
        assert blocks, "ingest recorded no blocks"
        assert all(block.status in TERMINAL_STATUSES for block in blocks)
        bad = [b.id for b in blocks if b.status.value in ("failed", "needs_human", "blocked_human")]
        assert bad == [], f"blocks not cleanly terminal: {bad}"
    finally:
        ledger.close()


async def test_the_render_wires_the_run_target_language_into_the_typesetter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ubt.render.typst import document_renderer as dr

    seen: list[str | None] = []
    original = dr.TypstDocumentRenderer.build_source

    def _recording_build_source(
        self: object, blocks: object, build_dir: object, target_lang: str = "zh"
    ) -> object:
        seen.append(target_lang)
        return original(self, blocks, build_dir, target_lang)  # type: ignore[arg-type]

    monkeypatch.setattr(dr.TypstDocumentRenderer, "build_source", _recording_build_source)
    await _run(tmp_path)
    assert seen, "the render never built a Typst document"
    assert all(lang == "es" for lang in seen), seen


async def test_the_attestation_artifact_check_finds_every_realization(tmp_path: Path) -> None:
    _, output, _ = await _run(tmp_path)

    shadow = companion_path(output, "_attestations.json")
    assert shadow.exists(), "no attestation companion beside the artifact"
    payload = json.loads(shadow.read_text(encoding="utf-8"))
    artifact = payload["artifact"]
    assert artifact["total"] > 0
    assert artifact["missing"] == [], artifact["summary"]
