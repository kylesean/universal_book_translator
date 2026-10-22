"""3 first-page heuristic engine selection for born-digital PDF routing.

Signals (cheap pypdfium2 probe of sampled pages):

- extractable text volume — scanned / image-only PDFs must go to Docling
  (OCR / vision pipeline); pypdfium2 would extract nothing;
- right-column edge share — multi-column layouts need Docling's
  reading-order reconstruction; pdfium's rect rows would interleave the
  columns and scramble the translation stream;
- formula-fragment density — technical books whose body pages carry
  math/table symbol debris (isolated single-letter tokens from shattered
  equations) need Docling's FORMULA / TABLE / LIST_ITEM classification;
  pdfium's heading-vs-narrative split mangles them into phantom headings,
  which then flow into translation + QE + repair and fail as untranslatable
  blocks. The cover page alone must NOT decide: sampling covers the front,
  middle and back of the document.

``UBT_PDF_ENGINE`` overrides the probe entirely: any explicit engine name
(built-in or custom) is honored as-is; the probe only runs for ``auto``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ubt.adapters.pdf import pdf_struct
from ubt.adapters.pdf.page_profiler import column_right_share, probe_pdfium_page
from ubt.adapters.pdf.pdfium_gate import pdfium_serialized
from ubt.cache.dirs import cache_root
from ubt.core.policy.layout_policy import (
    PDF_PATH_OPS,
    PROBE_COLUMN_SHARE,
    PROBE_FORMULA_SHARE,
    PROBE_MIN_CHARS,
    PROBE_MIN_ROWS,
    PROBE_MIN_SIGNAL_PAGES,
    PROBE_SAMPLE_FRACTIONS,
    PROFILE_VECTOR_PATH_OPS,
    formula_debris_share,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PDFRoutePlan:
    """Decoupled engine routing plan across document layout and page components."""

    primary_engine: str
    has_scan: bool
    has_vector_diagrams: bool
    has_formulas: bool
    has_multicolumn: bool


# Where the *per-page* profile cache lives (:func:`profile_pdf` keys it by
# profiler version + path + size + mtime). Under the shared cache root:
# CWD-relative paths silently forked the cache per working directory. The engine
# probe (:func:`inspect_pdf_route_plan`) samples a few pages and is not cached.
PROFILE_CACHE_DIR = cache_root() / "profile_cache"


def _scan_detected_from_probe(char_counts: Sequence[int]) -> bool:
    """Scanned/image-only signal: at least half the sampled pages lack text.

    A single page is not enough — a text cover (or a scanned cover with a
    digital title) would misroute the whole document. Sampling the front,
    middle and back and requiring a majority of low-text pages is the cheap
    fail-safe: a born-digital book with one image page still routes to
    pdfium, an image-only book routes to Docling.
    """
    if not char_counts:
        return False
    low = sum(1 for count in char_counts if count < PROBE_MIN_CHARS)
    return low * 2 >= len(char_counts)


def _probe_page_vector_ops(page: Any) -> int:
    """Safely count vector path operations for a pikepdf page.

    ``pdf_struct.count_ops`` reproduces pypdf's raw per-page operator count
    exactly (no XObject recursion), so ``PROFILE_VECTOR_PATH_OPS`` stays
    calibrated across the engine swap. A probe failure returns -1 so the
    caller can take the conservative route (docling) instead of steering
    toward the faster engine on missing data.
    """
    try:
        return pdf_struct.count_ops(page, PDF_PATH_OPS)
    except Exception:
        return -1


@pdfium_serialized
def inspect_pdf_route_plan(
    pdf_path: Path,
) -> PDFRoutePlan:
    """Sample pages for the assess signals (scan / formula / multicolumn / vector).

    This is NOT a routing decision anymore: ``auto`` takes the lossless docling
    engine unconditionally (see the long comment at the return — measured on the
    reference corpus, every document carried structure the pdfium reader cannot
    represent, and the cheap signals read zero exactly on vector-drawn
    figures). The probe's consumers are the assess report, which shows the
    operator what the document contains, and the explicit-``pdfium`` escape
    hatch's warning in the adapter factory. Keeping the two statements apart is
    deliberate: the sample is real evidence about the document, the engine
    choice above it is a fixed policy with its reasoning attached.
    """
    default_docling = PDFRoutePlan(
        primary_engine="docling",
        has_scan=False,
        has_vector_diagrams=False,
        has_formulas=False,
        has_multicolumn=False,
    )
    try:
        import pypdfium2 as pdfium
    except ImportError:
        return default_docling

    try:
        pdf = pdfium.PdfDocument(str(pdf_path))
        try:
            if len(pdf) == 0:
                return default_docling
            n_pages: int = len(pdf)
            page_indices = sorted(
                {min(int(f * n_pages), n_pages - 1) for f in PROBE_SAMPLE_FRACTIONS}
            )

            has_formulas = False
            has_multicolumn = False
            has_vector_diagrams = False
            vector_flagged_pages = 0
            sampled_char_counts: list[int] = []
            # Per-signal page counts. One page must not veto the whole document:
            # a single reference page scored as formula debris, or one figure
            # page read as two columns, flipped an entire book to Docling
            # (2026-10-10). ``PROBE_MIN_SIGNAL_PAGES`` is the bar — two pages is
            # a signal, one is noise, and a strict majority is too high for a
            # five-page sample on a long maths book (see the constant).
            formula_pages = 0
            multicolumn_pages = 0

            for idx in page_indices:
                page = pdf[idx]
                try:
                    probe = probe_pdfium_page(page)
                    sampled_char_counts.append(probe.n_chars)
                    if probe.n_chars < PROBE_MIN_CHARS:
                        continue
                    if formula_debris_share(probe.text) >= PROBE_FORMULA_SHARE:
                        formula_pages += 1
                    if (
                        probe.n_rect_rows >= PROBE_MIN_ROWS
                        and column_right_share(list(probe.rects), probe.unrotated_width_pt)
                        > PROBE_COLUMN_SHARE
                    ):
                        multicolumn_pages += 1
                finally:
                    page.close()

            scanned_detected = _scan_detected_from_probe(sampled_char_counts)
            has_formulas = formula_pages >= PROBE_MIN_SIGNAL_PAGES
            has_multicolumn = multicolumn_pages >= PROBE_MIN_SIGNAL_PAGES
            if formula_pages and not has_formulas:
                logger.debug(
                    "PDF probe: %d/%d sampled pages show formula debris in '%s' "
                    "(below the signal bar; assess reports the document, routing "
                    "does not use it)",
                    formula_pages,
                    len(page_indices),
                    pdf_path.name,
                )
            if multicolumn_pages and not has_multicolumn:
                logger.debug(
                    "PDF probe: %d/%d sampled pages show a right-column edge in '%s' "
                    "(below the signal bar; assess reports the document, routing "
                    "does not use it)",
                    multicolumn_pages,
                    len(page_indices),
                    pdf_path.name,
                )

            # Inspect vector diagrams via pikepdf (raw content-stream ops).
            # The signal aggregates over the sampled pages with the same
            # >=half majority as the scan tell: a single cover image in a
            # born-digital book must not force the docling mainline on the
            # whole document.
            try:
                with pdf_struct.open_pdf(pdf_path) as pike_doc:
                    for idx in page_indices:
                        if idx >= len(pike_doc.pages):
                            continue
                        pg = pike_doc.pages[idx]
                        n_path_ops = _probe_page_vector_ops(pg)
                        if n_path_ops < 0:
                            # Probe failure is conservative: assume heavy.
                            vector_flagged_pages = len(page_indices)
                            break
                        if (
                            n_path_ops >= PROFILE_VECTOR_PATH_OPS
                            or pdf_struct.resource_image_count(pg) >= 1
                        ):
                            vector_flagged_pages += 1
            except Exception as exc:
                logger.debug("PDF vector probe skipped on '%s': %s", pdf_path.name, exc)
            has_vector_diagrams = vector_flagged_pages * 2 >= len(page_indices)

            # A PDF target is *rebuilt* from the IR, so a block the extractor
            # cannot represent is content the reader loses. The pdfium path
            # (``ubt/adapters/pdf/reader_pdf``) emits only headings, paragraphs, list
            # items and code — no tables, no figures, no formulas. A source-layer
            # canvas would have kept an unextracted table visible in the original
            # language; a rebuilt document has no canvas, so the same omission is
            # a hole in the delivered book.
            #
            # The probe cannot rule that out. Measured on the reference corpus
            # every document carried structure, and one (``mamba``) had 19
            # figures and 16 tables with *every* cheap signal reading zero — its
            # figures are vector drawings, which ``resource_image_count`` does
            # not see. Sampling five pages also misses structure spread thin
            # (``llama2``: 90 structural blocks across 55 pages, 4 on a sampled
            # page). A full-page scan costs 0.05–0.10s on a 75-page paper, but
            # even that finds no signal for a vector-drawn figure.
            #
            # The cost argument for pdfium also expired: ingestion finishes
            # before drafting starts (``ubt/core/engine/plan.py``), so there is
            # no VRAM contention with the LLM, and docling's converter is
            # cleaned up deterministically after each document. On a 75-page
            # paper docling costs 4.2s against pdfium's 0.8s — under 1% of a run
            # that then spends minutes translating 811 blocks.
            #
            # So ``auto`` takes the lossless engine — unconditionally. The
            # sampled signals above are evidence ABOUT the document (the
            # assess report shows them), not a vote on the engine. ``pdfium``
            # remains a real opt-in (``--pdf-engine pdfium``, the
            # ``preview``/``fast`` presets) for a document known to be prose
            # only; the factory warns about what it would miss.
            return PDFRoutePlan(
                primary_engine="docling",
                has_scan=scanned_detected,
                has_vector_diagrams=has_vector_diagrams,
                has_formulas=has_formulas,
                has_multicolumn=has_multicolumn,
            )
        finally:
            pdf.close()
    except Exception as exc:
        logger.debug(
            "PDF probe failed on '%s'; assess signals unavailable: %s",
            pdf_path.name,
            exc,
        )
        return default_docling
