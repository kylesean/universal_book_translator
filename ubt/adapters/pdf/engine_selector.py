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
from ubt.adapters.pdf.page_profiler import PageKind, column_right_share, probe_pdfium_page
from ubt.adapters.pdf.pdfium_gate import pdfium_serialized
from ubt.cache.dirs import cache_root
from ubt.core.policy.layout_policy import (
    PDF_PATH_OPS,
    PROBE_COLUMN_SHARE,
    PROBE_FORMULA_SHARE,
    PROBE_MIN_CHARS,
    PROBE_MIN_ROWS,
    PROBE_SAMPLE_FRACTIONS,
    PROFILE_VECTOR_PATH_OPS,
    formula_debris_share,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PageIngestPlan:
    """Per-page ingest plan derived from page profile facts.

    Deliberately minimal: per-page fields are only worth carrying when a
    consumer reads them — ``parser_backend`` / ``extract_diagram_vector``
    style provenance without a reader is paper manifest data. The
    document-level engine routing (the decision that actually executes)
    lives in :func:`inspect_pdf_route_plan`. Add a per-page field back the
    day a reader appears.
    """

    page_number: int
    kind: PageKind


@dataclass(frozen=True)
class PDFRoutePlan:
    """Decoupled engine routing plan across document layout and page components."""

    primary_engine: str
    has_vector_diagrams: bool
    has_formulas: bool
    has_multicolumn: bool
    page_plans: tuple[PageIngestPlan, ...] = ()


# Where cheap page-profile facts are memoized. The cache key is
# (resolved path, size, mtime), so the engine probe, the ingest-plan builder and
# the parser all hit one entry per file instead of each re-scanning the document.
# Under the shared cache root: CWD-relative paths silently forked the cache per
# working directory.
PROFILE_CACHE_DIR = cache_root() / "profile_cache"


def build_page_ingest_plans(pdf_path: Path, cache_dir: Path | None = None) -> list[PageIngestPlan]:
    """Construct per-page ingest plans from cheap page profiler facts."""
    try:
        from ubt.adapters.pdf.page_profiler import profile_pdf

        profiles = profile_pdf(pdf_path, cache_dir=cache_dir)
    except Exception as exc:
        logger.debug("Failed to profile PDF '%s' for ingest plans: %s", pdf_path.name, exc)
        return []

    plans: list[PageIngestPlan] = []
    for p in profiles:
        plans.append(PageIngestPlan(page_number=p.facts.page, kind=p.kind))
    return plans


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
    cache_dir: Path | None = None,
    *,
    include_page_plans: bool = False,
) -> PDFRoutePlan:
    """Inspect sampled pages to build a decoupled layout and engine routing plan.

    Decouples narrative prose routing from specialized component routing:
    - Pure single-column born-digital text routes to the fast pdfium backend;
    - Formula-dense pages or multi-column layouts route to docling;
    - Vector-heavy diagram pages are reported via ``has_vector_diagrams``.

    ``page_plans`` is opt-in because building it profiles *every* page, while the
    probe above samples only a handful — and nothing in production reads the
    field (the parser that wants per-page plans calls
    :func:`build_page_ingest_plans` itself, with its own cached call).
    """
    default_docling = PDFRoutePlan(
        primary_engine="docling",
        has_vector_diagrams=False,
        has_formulas=False,
        has_multicolumn=False,
        page_plans=(),
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

            for idx in page_indices:
                page = pdf[idx]
                try:
                    probe = probe_pdfium_page(page)
                    sampled_char_counts.append(probe.n_chars)
                    if probe.n_chars < PROBE_MIN_CHARS:
                        continue
                    if formula_debris_share(probe.text) >= PROBE_FORMULA_SHARE:
                        logger.debug(
                            "PDF engine probe: formula-dense page %d in '%s'; using docling",
                            idx + 1,
                            pdf_path.name,
                        )
                        has_formulas = True
                    if (
                        probe.n_rect_rows >= PROBE_MIN_ROWS
                        and column_right_share(list(probe.rects), probe.width_pt)
                        > PROBE_COLUMN_SHARE
                    ):
                        has_multicolumn = True
                finally:
                    page.close()

            scanned_detected = _scan_detected_from_probe(sampled_char_counts)

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

            needs_docling = (
                scanned_detected or has_formulas or has_multicolumn or has_vector_diagrams
            )
            page_plans = (
                tuple(build_page_ingest_plans(pdf_path, cache_dir=cache_dir))
                if include_page_plans
                else ()
            )
            return PDFRoutePlan(
                primary_engine="docling" if needs_docling else "pdfium",
                has_vector_diagrams=has_vector_diagrams,
                has_formulas=has_formulas,
                has_multicolumn=has_multicolumn,
                page_plans=page_plans,
            )
        finally:
            pdf.close()
    except Exception as exc:
        logger.debug(
            "PDF engine probe failed on '%s'; defaulting to docling: %s",
            pdf_path.name,
            exc,
        )
        return default_docling


def select_pdf_engine(pdf_path: Path) -> str:
    """Return ``'pdfium'`` for clean single-column born-digital PDFs, else ``'docling'``.

    Any probe failure (unreadable file, missing pypdfium2) conservatively
    defaults to the Docling mainline engine. This samples a handful of pages
    only: building the full per-page ingest plan here would profile the
    whole document — twice, since the adapter factory calls this from both
    its resolution paths — for data the caller does not consume.
    """
    return inspect_pdf_route_plan(pdf_path).primary_engine
