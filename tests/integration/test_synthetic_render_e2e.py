"""Synthetic-PDF render e2e: a dry run carries a PDF all the way to a PDF.

``test_pipeline_smoke.py`` proves the stage plan over Markdown; this proves the
*PDF* half of the same plan -- the real pdfium adapter, the IR/AST bridge, the
translation stages, and the Typst renderer -- on a PDF this test builds itself
(:mod:`tests.pdf_builders`), so it needs neither the gitignored corpus nor an API
key. It is the synthetic companion to the real-document corpus gate
(``tests/integration/test_corpus_acceptance.py``): the corpus gate covers real
pagination and geometry, this covers the same code path deterministically in CI.

Why ``slow`` and not ``fast``: every PDF render path shells out to the external
``typst`` compiler -- a heavy subprocess, per
``docs/guides/TESTING_STRATEGY.md`` -- so it cannot be in the inner loop. CI
installs a checksum-pinned ``typst`` and runs the slow tier (``pytest -m slow``);
locally the module skips when ``typst`` is absent. The pure-Python ingest half
runs in the ``fast`` tier instead (``tests/unit/adapters/test_pdf_adapter_e2e.py``).

The target language is ``es`` (Latin) on purpose: it keeps the render off any
CJK font the runner may not have, and the rehearsal echo (English) still satisfies
the target-script check.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from pdf_builders import write_text_pdf

import ubt.adapters.pdf.oxide_render as oxide_render
from ubt.core.config import UBTConfig
from ubt.core.content.verify import contract_path_for_artifact, load_contract_file
from ubt.core.engine.dry_run import create_dry_run_orchestrator
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

#: Paragraphs that must survive extraction from the delivered PDF (titles too).
_PARAGRAPHS = (
    "The machine relies on attention. Each layer runs a forward pass over the batch.",
    "Euler wrote a formula linking five constants of analysis.",
    "Following Smith et al. the score reached 98.45 percent on every benchmark.",
    "The second chapter widens the lens to every descendant of the first model.",
)


def _config(tmp_path: Path) -> UBTConfig:
    """A hermetic config: ledgers under tmp, no page-image egress."""
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
    """Whitespace-insensitive form: PDF extraction reflows lines, so match flat."""
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
    # The rehearsal echo delivers every translatable text node.
    assert contract.delivered_text == contract.total_text


async def test_the_ledger_records_a_completed_job(tmp_path: Path) -> None:
    _, _, job_id = await _run(tmp_path)

    ledger = SQLiteJobLedger(tmp_path / "ledgers" / f"{job_id}.sqlite")
    try:
        assert ledger.get_job_status(job_id) == "completed"
        blocks = ledger.get_all_blocks(job_id)
        assert blocks, "ingest recorded no blocks"
        assert all(block.status in TERMINAL_STATUSES for block in blocks)
        bad = [b.id for b in blocks if b.status.value in ("failed", "needs_human", "blocked_human")]
        assert bad == [], f"blocks not cleanly terminal: {bad}"
    finally:
        ledger.close()


async def test_the_attestation_artifact_check_finds_every_realization(tmp_path: Path) -> None:
    # Regression: the renderer strips the rehearsal prefix, so the artifact check
    # must compare against the rendered text. When only one side knew the prefix,
    # every rehearsal reported short elements (headings) as "missing".
    _, output, _ = await _run(tmp_path)

    shadow = companion_path(output, "_attestations.json")
    assert shadow.exists(), "no attestation companion beside the artifact"
    payload = json.loads(shadow.read_text(encoding="utf-8"))
    artifact = payload["artifact"]
    assert artifact["total"] > 0
    assert artifact["missing"] == [], artifact["summary"]
