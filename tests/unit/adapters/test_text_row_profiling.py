"""``n_text_rows`` must count rows, not raw pdfium rect fragments.

Regression (arXiv 2608.25512, measured 2026-10-25): the pre-flight
``PATHOLOGICAL_PAGE_RISK`` gate read ``PageFacts.n_rect_rows`` -- pdfium's
raw rect-fragment count. A Typst table of contents dots each leader line,
so the paper's two TOC pages carried 2,289 / 2,100 "rows" while the geometry
engine regrouped them into 35 / 32 rows and ordered each page in 0.02s.
Every dotted TOC therefore warned of quadratic slowness that never came.
The profiler now reports ``n_text_rows`` (fragments regrouped into rows by
vertical overlap) and the gate reads that instead; these tests pin the
profiler half.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pdf_builders import write_dotted_toc_pdf, write_text_pdf

from ubt.adapters.pdf.page_profiler import collect_page_facts

pytestmark = pytest.mark.fast


def test_a_dotted_toc_groups_back_into_its_rows(tmp_path: Path) -> None:
    """Thousands of leader-dot runs, a few dozen rows: the old gate's input
    versus the geometry engine's real one."""
    pdf = write_dotted_toc_pdf(tmp_path / "toc.pdf", rows=30, dots_per_row=70)
    (facts,) = collect_page_facts(pdf)
    # 30 rows x (title + 70 dots + page number) runs: enough raw fragments
    # that the old raw-count gate (>= 2000) fired on exactly this shape.
    assert facts.n_rect_rows >= 2000
    assert facts.n_text_rows == 30


def test_a_plain_prose_page_counts_one_row_per_line(tmp_path: Path) -> None:
    pdf = write_text_pdf(
        tmp_path / "prose.pdf", [[f"plain body line number {i}" for i in range(20)]]
    )
    (facts,) = collect_page_facts(pdf)
    assert facts.n_rect_rows == 20
    assert facts.n_text_rows == 20
