from __future__ import annotations

from pathlib import Path

import pytest
from pdf_builders import write_dotted_toc_pdf, write_text_pdf

from ubt.adapters.pdf.page_profiler import collect_page_facts

pytestmark = pytest.mark.fast


def test_a_dotted_toc_still_reports_its_fragment_count(tmp_path: Path) -> None:
    pdf = write_dotted_toc_pdf(tmp_path / "toc.pdf", rows=30, dots_per_row=70)
    (facts,) = collect_page_facts(pdf)
    assert facts.n_rect_rows >= 2000
    assert facts.n_rect_rows * facts.n_chars < 100_000_000


def test_a_plain_prose_page_counts_one_rect_per_line(tmp_path: Path) -> None:
    pdf = write_text_pdf(
        tmp_path / "prose.pdf", [[f"plain body line number {i}" for i in range(20)]]
    )
    (facts,) = collect_page_facts(pdf)
    assert facts.n_rect_rows == 20
    assert facts.n_chars > 0
