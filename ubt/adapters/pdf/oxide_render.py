"""Optional ``pdf_oxide`` backend — render and text legs.

pdf_oxide (Rust/PyO3, zero Python deps, MIT/Apache) renders a page to PNG
bytes and extracts page text in-process, replacing the poppler ``pdftoppm``
subprocess (a system binary plus a Popen attack surface) and the retired
runtime ``pypdf`` text leg. Every entry point here is
non-raising: render ``None`` means "no image", text ``""`` means "no text",
so a faulty engine degrades the visual gates instead of crashing the run.
Structural reads (geometry, content-stream ops, resources) deliberately do
NOT live here — see :mod:`ubt.adapters.pdf.pdf_struct` for why they stay on
pikepdf.

Threading stance: pdf_oxide's PyO3 methods take ``&mut self`` (a cell
borrow), so sharing one ``PdfDocument`` across threads raises
``BorrowMutError`` — the opposite failure mode from libpdfium's heap
corruption that forced :mod:`ubt.adapters.pdf.pdfium_gate` to serialize the
world behind ``PDFIUM_LOCK``. There is no global init state here to guard,
so the whole mitigation is *open the document per call*; both callers run
under ``asyncio.to_thread`` and each renders one page per document open.

DPI parity: ``render_page(dpi=N)`` is assumed to mean the same scale as
``pdftoppm -r N`` (both N dots per 72pt inch). Frozen as verified by the
size criterion in ``scripts/oxide_render_ab.py``; ``svg_diagram``'s
``bbox * dpi/72`` crop math silently depends on it.
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def is_available() -> bool:
    """True when pdf_oxide is importable (it is a base dependency since
    v3.1; the probe survives for diagnostics and degraded-install tests).
    Not cached, so tests can flip availability by patching
    :func:`importlib.util.find_spec`."""
    return importlib.util.find_spec("pdf_oxide") is not None


def render_page_png(pdf_path: Path, page: int, dpi: int) -> bytes | None:
    """Render a 1-based page to PNG bytes; ``None`` on any failure.

    ``page`` is 1-based (UBT convention); pdf_oxide is 0-based — the ``-1``
    mapping is deliberate.
    """
    try:
        from pdf_oxide import PdfDocument
    except ImportError:
        return None
    try:
        doc = PdfDocument(str(pdf_path))  # open-per-call; see module docstring
        if page < 1 or page > doc.page_count:
            return None
        return bytes(doc.render_page(page - 1, dpi=dpi))
    except Exception as exc:  # any renderer fault must fall back, not crash
        logger.debug("oxide render p%d: %s", page, exc)
        return None


def write_page_png(pdf_path: Path, page: int, dpi: int, out_dir: Path) -> Path | None:
    """Render and write ``out_dir/p<page>.png``; ``None`` on any failure."""
    data = render_page_png(pdf_path, page, dpi)
    if data is None:
        return None
    target = out_dir / f"p{page}.png"
    try:
        target.write_bytes(data)
    except OSError as exc:
        logger.debug("oxide render: cannot write %s: %s", target, exc)
        return None
    return target


def extract_page_texts(pdf_path: Path) -> list[str]:
    """Per-page text (1-based order) via one document open; empty list on failure.

    Replaces the retired pypdf text-extraction legs (blank-page candidates,
    short-doc char census, plain-text sampling, the docling-parser fallback
    extractor). Never raises: ``[]`` tells the caller "no text signal".
    """
    try:
        from pdf_oxide import PdfDocument
    except ImportError:
        return []
    try:
        doc = PdfDocument(str(pdf_path))
        return [str(doc.extract_text(i)) for i in range(int(doc.page_count))]
    except Exception as exc:  # text probing must never fail a run
        logger.debug("oxide text extraction failed for %s: %s", pdf_path, exc)
        return []
