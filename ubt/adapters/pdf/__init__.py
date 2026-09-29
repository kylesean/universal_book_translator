"""PDF document engine adapters (Docling + Typst + pdf_oxide).

100% permissively licensed (MIT / Apache-2.0 / BSD-3) production-grade PDF processing.
"""

from ubt.adapters.pdf.alternator import BilingualAlternator
from ubt.adapters.pdf.diagram_localizer import DiagramLocalizer
from ubt.adapters.pdf.docling_adapter import DoclingPDFAdapter
from ubt.adapters.pdf.typst_reconstructor import TypstReconstructor

__all__ = [
    "DiagramLocalizer",
    "DoclingPDFAdapter",
    "BilingualAlternator",
    "TypstReconstructor",
]
