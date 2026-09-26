"""Formula engine matrix: one real 26-page book, three math backends.

Runs the full pipeline with a mock provider (no API cost) once per backend.
Corpus selection is two-tier, because the copyright sweep (7b4c456) took the
Elsevier chapter out of the repository while leaving its structural twin behind:

* **Generated corpus** (default): ``tests/fixtures/synthetic-duo.pdf``, generated from
  typst source with the same 540x665.972pt MediaBox, printed equation tags,
  IEEE reference lists and vector figure drawings. Everything this file asserts
  that is a property of the *engines* runs on it — in CI, on every clone.
* **Real corpus** (opt-in): ``UBT_TEST_REAL_PDF=~/Documents/chapter-3.pdf`` adds
  the recorded OCR-damage inventory of the withdrawn chapter (three findings:
  two witness aspect flags — 4.17x, and 4.31x with components 36 vs 10 — plus
  one render error, "Extra close brace"), each falling back to the source
  graphic. Those numbers are damage *this* scan has; asserting them against a
  generated file would be asserting the absence of a fixture.

``image`` renders every display formula as the source crop, so nothing degrades;
``typst`` + ``witness`` is the legacy converter plus the deterministic witness
(structural flags, the pre-engine baseline). Translation runs once (first
backend); the two resume runs reuse the same ledger, so the PDF is parsed once
per test. Skipped when docling or Node is unavailable, and when docling's
formula enrichment yields no LaTeX (a degraded CPU-only environment would
otherwise assert on empty input).
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from tests.mock_providers import TokenEchoMockProvider
from ubt.adapters.pdf.docling_adapter import DoclingPDFAdapter
from ubt.adapters.pdf.math_renderer import MathjaxRenderer
from ubt.core.config import UBTConfig
from ubt.core.engine.pipeline import PipelineOrchestrator
from ubt.core.job_options import sidecar_path
from ubt.core.qe.comet_runner import MockQERunner
from ubt.core.router.router import ModelRouter

pytestmark = pytest.mark.slow  # full chapter pipeline incl. real docling parse

REPO_ROOT = Path(__file__).resolve().parents[2]
# Opt-in real corpus; the committed synthetic is the default so CI runs the
# engine invariants instead of skipping them.
REAL_PDF = (
    Path(os.environ.get("UBT_TEST_REAL_PDF", "")).expanduser()
    if os.environ.get("UBT_TEST_REAL_PDF")
    else None
)
SOURCE_PDF = REAL_PDF or REPO_ROOT / "tests" / "fixtures" / "synthetic-duo.pdf"
JOB_ID = "formula-engine-matrix"
MOCK_TRANSLATION = "这是用于公式引擎矩阵测试的中文模拟译文。"

requires_docling = pytest.mark.skipif(
    not DoclingPDFAdapter().is_docling_installed(),
    reason="docling not installed (uv sync --extra pdf)",
)
requires_node = pytest.mark.skipif(
    not MathjaxRenderer().available(), reason="Node + scripts/mathjax deps not installed"
)
requires_fixture = pytest.mark.skipif(
    not SOURCE_PDF.is_file(),
    reason=(
        f"UBT_TEST_REAL_PDF points at a missing file: {SOURCE_PDF}"
        if REAL_PDF is not None
        else f"{SOURCE_PDF.relative_to(REPO_ROOT)} corpus missing "
        "(regenerate with scripts/make_sample_corpus.py)"
    ),
)
# The recorded damage inventory belongs to one specific scanned document; the
# test asserts it only when that document is the input (see ``REAL_PDF``).


def _orchestrator(db_dir: Path, *, math_backend: str, formula_render: str) -> PipelineOrchestrator:
    config = UBTConfig(
        db_dir=db_dir,
        draft_model="mock-draft",
        repair_model="mock-repair",
        rate_limit_rpm=600,
        math_backend=math_backend,  # type: ignore[arg-type]
        formula_render=formula_render,  # type: ignore[arg-type]
    )
    provider = TokenEchoMockProvider(default_response=MOCK_TRANSLATION)
    router = ModelRouter(provider=provider, draft_model="mock-draft", repair_model="mock-repair")
    return PipelineOrchestrator(config=config, router=router, qe_runner=MockQERunner(0.9))


async def _run_once(
    db_dir: Path, output: Path, *, math_backend: str, formula_render: str
) -> dict[str, Any]:
    orchestrator = _orchestrator(db_dir, math_backend=math_backend, formula_render=formula_render)
    async for _event in orchestrator.run(
        input_path=SOURCE_PDF,
        output_path=output,
        target_lang="zh",
        job_id=JOB_ID,
    ):
        pass
    report_path = sidecar_path(output, "quality_report.json")
    report: dict[str, Any] = json.loads(report_path.read_text(encoding="utf-8"))
    return report


def _finding_ids(report: dict[str, Any], field: str) -> set[str]:
    return {entry.split(":", 1)[0].strip() for entry in report.get(field) or []}


def _findings(report: dict[str, Any], field: str) -> list[str]:
    return list(report.get(field) or [])


def _page_count(pdf_path: Path) -> int:
    import pypdfium2 as pdfium

    document = pdfium.PdfDocument(str(pdf_path))
    try:
        return len(document)
    finally:
        document.close()


def _formula_block_count(db_dir: Path) -> int:
    ledger = db_dir / f"{JOB_ID}.sqlite"
    with sqlite3.connect(ledger) as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM blocks WHERE block_type = 'formula' AND source_text LIKE '%\\%'"
        ).fetchone()
    return int(row[0]) if row else 0


@requires_docling
@requires_node
@requires_fixture
@pytest.mark.asyncio
async def test_formula_engine_matrix(tmp_path: Path) -> None:
    db_dir = tmp_path / "ledgers"
    out_dir = tmp_path / "out"

    mathjax_report = await _run_once(
        db_dir, out_dir / "book-mathjax.pdf", math_backend="mathjax", formula_render="witness"
    )
    if _formula_block_count(db_dir) < 10:
        pytest.skip("docling formula enrichment produced no LaTeX in this environment")

    assert _finding_ids(mathjax_report, "syntax_fallbacks") == set(), (
        f"mathjax syntax fallbacks: {mathjax_report.get('syntax_fallbacks')}"
    )
    mathjax_reasons = _findings(mathjax_report, "formula_witness_fallbacks")
    if REAL_PDF is not None:
        # The withdrawn Elsevier chapter's recorded inventory: exactly these
        # three degradations, no more, no less.
        assert len(mathjax_reasons) == 3, f"mathjax findings: {mathjax_reasons}"
        assert sum("engine witness aspect 4.17x" in r for r in mathjax_reasons) == 1
        assert sum("engine render failed (Extra close brace" in r for r in mathjax_reasons) == 1
        merged = [r for r in mathjax_reasons if "aspect 4.31x" in r and "components 36 vs 10" in r]
        assert len(merged) == 1, f"merged-equation witness missing: {mathjax_reasons}"
    else:
        # On the committed corpus the invariant is that the witness *reports*
        # rather than dies: every finding must name a block id and a cause.
        for finding in mathjax_reasons:
            assert ":" in finding, f"unattributed witness finding: {finding}"
    mathjax_pages = _page_count(out_dir / "book-mathjax.pdf")

    image_report = await _run_once(
        db_dir, out_dir / "book-image.pdf", math_backend="image", formula_render="witness"
    )
    assert _findings(image_report, "formula_witness_fallbacks") == [], (
        f"image findings: {image_report.get('formula_witness_fallbacks')}"
    )
    assert _finding_ids(image_report, "syntax_fallbacks") == set(), (
        f"image syntax fallbacks: {image_report.get('syntax_fallbacks')}"
    )
    image_pages = _page_count(out_dir / "book-image.pdf")

    typst_report = await _run_once(
        db_dir, out_dir / "book-typst.pdf", math_backend="typst", formula_render="witness"
    )
    assert _finding_ids(typst_report, "syntax_fallbacks") == set(), (
        f"typst syntax fallbacks: {typst_report.get('syntax_fallbacks')}"
    )
    typst_reasons = _findings(typst_report, "formula_witness_fallbacks")
    if REAL_PDF is not None:
        # The withdrawn chapter's pre-engine baseline: the legacy converter
        # degrades at least two structural classes, more than the MathJax route.
        # Measured against the committed corpus it yields exactly one finding,
        # so the count is a property of that scan, not of the engine.
        assert len(typst_reasons) >= 2, f"typst findings: {typst_reasons}"
        assert any("aspect" in r for r in typst_reasons)
        assert any("components" in r for r in typst_reasons)
    for finding in typst_reasons:
        assert ":" in finding, f"unattributed witness finding: {finding}"
    typst_pages = _page_count(out_dir / "book-typst.pdf")

    # All three artifact variants stay inside a tight page band around the
    # engine output (formula rendering must not repaginate the book).
    assert abs(image_pages - mathjax_pages) <= 2
    assert abs(typst_pages - mathjax_pages) <= 2
