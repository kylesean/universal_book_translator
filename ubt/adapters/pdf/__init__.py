"""PDF document engine adapters (Docling + Typst + pdf_oxide).

100% permissively licensed (MIT / Apache-2.0 / BSD-3) production-grade PDF processing.
"""

from ubt.adapters.pdf.alternator import BilingualAlternator
from ubt.adapters.pdf.docling_adapter import DoclingPDFAdapter
from ubt.adapters.pdf.page_chunking import compute_pdf_page_chunks, get_pdf_page_chunk_size

__all__ = [
    "DoclingPDFAdapter",
    "BilingualAlternator",
    "compute_pdf_page_chunks",
    "get_pdf_page_chunk_size",
]
