"""The overlay engine end to end (LayerCompositor path).

Drives the real dry-run pipeline over a
synthetic PDF and pins the delivery: the three-layer composition keeps source
page geometry, strips+replaces the reconstructed regions, passes the visual gate
(no occluded/overlapped text, so no block is quarantined), and reconciles a clean
contract. Needs Typst (the micro-fragment typesetter), hence the ``slow`` tier.
"""

from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

import pytest
from pdf_builders import write_text_pdf, write_two_column_pdf

from ubt.adapters.pdf import oxide_render, pdf_struct
from ubt.core.config import UBTConfig
from ubt.core.content.verify import contract_path_for_artifact, load_contract_file
from ubt.core.engine.dry_run import create_dry_run_orchestrator
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.job_options import sidecar_path

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        shutil.which("typst") is None,
        reason="the composite engine needs the typst fragment typesetter",
    ),
]

_PAGE = (
    "The Attention Machine",
    "The machine relies on attention and runs a forward pass over the batch.",
)


def _config(tmp_path: Path) -> UBTConfig:
    return UBTConfig().model_copy(
        update={
            "db_dir": tmp_path / "ledgers",
            "ocr_mode": "off",
            "allow_page_upload": False,
            "visual_judge_enabled": False,
            # These tests pin the monolingual composition; the bilingual tests
            # below opt back in with an explicit dual_mode.
            "dual_mode": "monolingual",
        }
    )


async def _run(tmp_path: Path) -> tuple[Path, str]:
    source = write_text_pdf(tmp_path / "book.pdf", [_PAGE])
    output = tmp_path / "book_es.pdf"
    job_id = "composite-e2e"
    orchestrator = create_dry_run_orchestrator(_config(tmp_path))
    async for _ in orchestrator.run(source, output, target_lang="es", job_id=job_id):
        pass
    return output, job_id


async def test_the_composite_engine_delivers_a_geometry_preserving_pdf(tmp_path: Path) -> None:
    output, _ = await _run(tmp_path)
    source = tmp_path / "book.pdf"

    assert output.exists() and output.read_bytes()[:5] == b"%PDF-"
    assert pdf_struct.page_sizes(output) == pdf_struct.page_sizes(source), (
        "geometry must be preserved"
    )


async def test_the_composite_engine_places_the_translation(tmp_path: Path) -> None:
    output, _ = await _run(tmp_path)

    delivered = " ".join("\n".join(oxide_render.extract_page_texts(output)).split())
    # The rehearsal echo is the target; the source text under it was stripped, so
    # the delivered page carries the target, not a doubled source+target.
    assert "The Attention Machine" in delivered
    assert "The machine relies on attention" in delivered


async def test_the_composite_engine_passes_the_visual_gate_and_reconciles(tmp_path: Path) -> None:
    output, job_id = await _run(tmp_path)

    visual = json.loads(sidecar_path(output, "visual_report.json").read_text(encoding="utf-8"))
    assert visual["passed"], visual["findings"]

    contract = load_contract_file(contract_path_for_artifact(output))
    assert not contract.errors, contract.errors
    assert contract.delivered_text == contract.total_text > 0

    ledger = SQLiteJobLedger(tmp_path / "ledgers" / f"{job_id}.sqlite")
    try:
        assert ledger.get_job_status(job_id) == "completed"
        statuses = Counter(block.status.value for block in ledger.get_all_blocks(job_id))
        # A passing visual gate must not quarantine anything.
        assert "needs_human" not in statuses, statuses
    finally:
        ledger.close()


async def test_the_composite_engine_interleaves_a_page_bilingual_mode(tmp_path: Path) -> None:
    # A page-pairing bilingual mode (alternating/facing) is served by zipping the
    # source pages with the page-aligned composition, doubling the page count.
    source = write_text_pdf(tmp_path / "book.pdf", [_PAGE, _PAGE])
    output = tmp_path / "book_es.pdf"
    config = _config(tmp_path).model_copy(update={"dual_mode": "alternating"})
    orchestrator = create_dry_run_orchestrator(config)
    async for _ in orchestrator.run(source, output, target_lang="es", job_id="composite-bi"):
        pass

    assert len(pdf_struct.page_sizes(output)) == 2 * len(pdf_struct.page_sizes(source))
    source_sizes = list(pdf_struct.page_sizes(source).values())
    out_sizes = list(pdf_struct.page_sizes(output).values())
    # Pages alternate source, translated, source, translated.
    assert out_sizes[0] == source_sizes[0]
    assert out_sizes[2] == source_sizes[1]


async def test_the_composite_engine_draws_in_place_bilingual(tmp_path: Path) -> None:
    # The default bilingual mode is in-place: target over source in the same box,
    # so a one-page source stays one page (unlike the doubling page zip). Both
    # languages reach the artifact, and the source is redrawn, not masked away.
    source = write_text_pdf(tmp_path / "book.pdf", [_PAGE])
    output = tmp_path / "book_es.pdf"
    config = _config(tmp_path).model_copy(update={"dual_mode": "inline"})
    orchestrator = create_dry_run_orchestrator(config)
    async for _ in orchestrator.run(source, output, target_lang="es", job_id="composite-ip"):
        pass

    assert len(pdf_struct.page_sizes(output)) == len(pdf_struct.page_sizes(source))
    text = " ".join("\n".join(oxide_render.extract_page_texts(output)).split())
    assert (
        text.count("The machine relies on attention and runs a forward pass over the batch.") >= 2
    )


async def test_the_composite_engine_flows_a_cross_page_continuation(tmp_path: Path) -> None:
    # A sentence split by a page break: one overlay flowed across two boxes, drawn
    # on its two source pages, instead of one overlay per half-block.
    source = write_text_pdf(
        tmp_path / "book.pdf",
        [["The machine relies on attention and"], ["runs a forward pass over the batch."]],
    )
    output = tmp_path / "book_es.pdf"
    orchestrator = create_dry_run_orchestrator(_config(tmp_path))
    async for _ in orchestrator.run(source, output, target_lang="es", job_id="composite-flow"):
        pass

    pages = oxide_render.extract_page_texts(output)
    assert len(pages) == 2
    assert all(page.strip() for page in pages)
    assert pdf_struct.page_sizes(output) == pdf_struct.page_sizes(source)
    contract = load_contract_file(contract_path_for_artifact(output))
    assert not contract.errors, contract.errors


async def test_the_composite_engine_flows_a_cross_column_continuation(tmp_path: Path) -> None:
    # Two columns on one page: the paragraph ends at the bottom of the left
    # column and resumes at the top of the right; one overlay flows across the
    # gutter instead of one overlay per half-block.
    left = [f"left column filler line number {i} continues here" for i in range(1, 9)]
    right = ["and yet it runs in seconds on", "modern hardware today."] + [
        f"right column filler line {i}" for i in range(3, 9)
    ]
    source = write_two_column_pdf(tmp_path / "book.pdf", [(left, right)])
    output = tmp_path / "book_es.pdf"
    orchestrator = create_dry_run_orchestrator(_config(tmp_path))
    async for _ in orchestrator.run(source, output, target_lang="es", job_id="composite-cols"):
        pass

    pages = oxide_render.extract_page_texts(output)
    assert len(pages) == 1
    assert pages[0].strip()
    assert pdf_struct.page_sizes(output) == pdf_struct.page_sizes(source)
    contract = load_contract_file(contract_path_for_artifact(output))
    assert not contract.errors, contract.errors
