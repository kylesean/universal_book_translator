"""The pre-flight slowness gate reads a page's extraction *cost*, not a proxy.

Regression history (measured 2026-10-25):

* The gate first read ``PageFacts.n_rect_rows``, pdfium's raw fragment count.
  A Typst table of contents dots each leader line, so an ordinary TOC page
  carried 2,289 "rows" and warned of quadratic slowness — while the geometry
  engine grouped it into 35 rows and ordered it in 0.02s.
* It then read ``n_text_rows`` (fragments regrouped by vertical overlap).
  That killed the TOC false positive but measured the wrong quantity: across
  844 corpus pages, ``corr(extract_lines time, n_text_rows)`` is -0.02, while
  ``corr(time, rects * chars)`` is 1.000. The one genuinely slow page
  (arXiv 2609.20519 p1, 8.5s vs 0.14s for the next worst) groups to 26 rows —
  *fewer* than an ordinary page — so the gate fired on 0 of 844 pages and the
  slowness it existed to report was invisible.
* It now reads ``rects * chars``, which is the work ``extract_lines`` actually
  does: one ``get_text_bounded`` + ``_probe_rect_font_style`` per fragment, each
  walking the page's characters.

These tests pin the profiler half: the facts the gate multiplies must be the
raw fragment count and the real character count.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pdf_builders import write_dotted_toc_pdf, write_text_pdf

from ubt.adapters.pdf.page_profiler import collect_page_facts

pytestmark = pytest.mark.fast


def test_a_dotted_toc_still_reports_its_fragment_count(tmp_path: Path) -> None:
    """A TOC page is many fragments over modest text — a *low* cost page.

    This is the shape that must NOT warn: 30 rows of leader dots are ~2,000+
    fragments, but the page's own text is a few thousand characters (6,949 in
    this fixture, against 104,456 on the pathological page), so
    ``rects * chars`` lands at ~1.5e7 — two orders under the bar. The gate's
    predecessor (raw fragment count >= 2000) fired on exactly this page.
    """
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
