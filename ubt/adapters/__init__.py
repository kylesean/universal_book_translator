"""Document Adapters and the in-process adapter SPI."""

from ubt.adapters.base import BaseDocumentAdapter, BasePDFEngineAdapter
from ubt.adapters.docx.adapter import DOCXInspectionAdapter
from ubt.adapters.epub.adapter import EPUBInspectionAdapter
from ubt.adapters.factory import (
    get_adapter_for_path,
    get_inspection_adapter,
    is_pdf_engine_registered,
    register_adapter,
    register_pdf_engine,
    supported_suffixes,
)
from ubt.adapters.html.adapter import HTMLInspectionAdapter
from ubt.adapters.markdown.adapter import MarkdownInspectionAdapter
from ubt.adapters.pdf.docling_adapter import ScanPDFToTypstCodec

__all__ = [
    "BaseDocumentAdapter",
    "BasePDFEngineAdapter",
    "DOCXInspectionAdapter",
    "ScanPDFToTypstCodec",
    "EPUBInspectionAdapter",
    "HTMLInspectionAdapter",
    "MarkdownInspectionAdapter",
    "get_adapter_for_path",
    "get_inspection_adapter",
    "is_pdf_engine_registered",
    "register_adapter",
    "register_pdf_engine",
    "supported_suffixes",
]
