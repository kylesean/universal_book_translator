"""Short-document fast-lane probe.

Decides whether a PDF may take the fast lane (seed-only bible, same stage
graph otherwise): short (<= QA_FULL_GATE_MAX_PAGES) AND born-digital
(extractable text above FAST_LANE_MIN_TEXT_CHARS). Scan-like docs always
take the full path. Only permissive libraries: pdf_oxide (MIT/Apache), lazily
imported so the probe never hardens the import graph.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ubt.adapters.pdf.pdfium_gate import pdfium_serialized
from ubt.core.policy.layout_policy import (
    FAST_LANE_MIN_TEXT_CHARS,
    QA_FULL_GATE_MAX_PAGES,
)

logger = logging.getLogger(__name__)


@pdfium_serialized
def probe_pdf_pages(pdf_path: Path) -> tuple[int, int]:
    """Return (page_count, extractable_chars) via pypdfium2 or pdf_oxide; (0, 0) on failure."""
    try:
        import pypdfium2 as pdfium

        doc = pdfium.PdfDocument(str(pdf_path))
        # ``PdfDocument`` holds a native handle; a corrupt page mid-loop would
        # leak it for the life of the process without the ``finally``.
        try:
            n_pages = len(doc)
            chars = 0
            for i in range(n_pages):
                page = doc[i]
                try:
                    tp = page.get_textpage()
                    try:
                        chars += tp.count_chars()
                    finally:
                        tp.close()
                finally:
                    page.close()
            return n_pages, chars
        finally:
            doc.close()
    except Exception:
        pass

    try:
        from ubt.adapters.pdf import oxide_render

        texts = oxide_render.extract_page_texts(Path(pdf_path))
        if not texts:
            return 0, 0
        return len(texts), sum(len(t) for t in texts)
    except Exception as exc:
        logger.debug("fast-lane probe: cannot read '%s': %s", Path(pdf_path).name, exc)
        return 0, 0


def is_fast_lane_eligible(pdf_path: Path | str) -> bool:
    """True when the PDF qualifies for the fast lane (pure decision)."""
    pages, chars = probe_pdf_pages(Path(pdf_path))
    if pages <= 0 or pages > QA_FULL_GATE_MAX_PAGES:
        return False
    return chars >= FAST_LANE_MIN_TEXT_CHARS
