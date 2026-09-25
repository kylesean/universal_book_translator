"""Document Adapters and the in-process adapter SPI."""

from ubt.adapters.base import BaseDocumentAdapter, BasePDFEngineAdapter
from ubt.adapters.docx.adapter import DOCXAdapter
from ubt.adapters.epub.adapter import EPUBAdapter
from ubt.adapters.factory import (
    get_adapter_for_path,
    is_pdf_engine_registered,
    register_adapter,
    register_pdf_engine,
    supported_suffixes,
)
from ubt.adapters.html.adapter import HTMLAdapter
from ubt.adapters.markdown.adapter import MarkdownAdapter
from ubt.adapters.pdf.docling_adapter import DoclingPDFAdapter
from ubt.adapters.pdf.pdfium_adapter import PDFiumAdapter

__all__ = [
    "BaseDocumentAdapter",
    "BasePDFEngineAdapter",
    "DOCXAdapter",
    "DoclingPDFAdapter",
    "EPUBAdapter",
    "HTMLAdapter",
    "MarkdownAdapter",
    "PDFiumAdapter",
    "get_adapter_for_path",
    "is_pdf_engine_registered",
    "register_adapter",
    "register_pdf_engine",
    "supported_suffixes",
]
