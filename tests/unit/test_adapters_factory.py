"""Tests for adapter factory resolution."""

from pathlib import Path

import pytest

from ubt.adapters import (
    DoclingPDFAdapter,
    DOCXAdapter,
    EPUBAdapter,
    HTMLAdapter,
    MarkdownAdapter,
    get_adapter_for_path,
)
from ubt.core.exceptions import UnsupportedDocumentFormatError


def test_get_adapter_for_path_resolves_epub() -> None:
    adapter = get_adapter_for_path(Path("book.epub"))
    assert isinstance(adapter, EPUBAdapter)


def test_get_adapter_for_path_resolves_markdown() -> None:
    adapter1 = get_adapter_for_path("document.md")
    adapter2 = get_adapter_for_path(Path("notes.MARKDOWN"))
    assert isinstance(adapter1, MarkdownAdapter)
    assert isinstance(adapter2, MarkdownAdapter)


def test_get_adapter_for_path_resolves_txt_to_markdown_adapter() -> None:
    """Flat text reuses the Markdown adapter (blank-line paragraph split)."""
    adapter = get_adapter_for_path("notes.txt")
    assert isinstance(adapter, MarkdownAdapter)


def test_get_adapter_for_path_resolves_html() -> None:
    adapter1 = get_adapter_for_path("page.html")
    adapter2 = get_adapter_for_path(Path("PAGE.HTM"))
    assert isinstance(adapter1, HTMLAdapter)
    assert isinstance(adapter2, HTMLAdapter)


def test_get_adapter_for_path_resolves_docx() -> None:
    adapter = get_adapter_for_path(Path("contract.docx"))
    assert isinstance(adapter, DOCXAdapter)


def test_get_adapter_for_path_resolves_pdf_default() -> None:
    adapter = get_adapter_for_path("paper.pdf")
    assert isinstance(adapter, DoclingPDFAdapter)
    assert adapter.engine_name == "docling"


def test_get_adapter_for_path_rejects_deprecated_babeldoc() -> None:
    with pytest.raises(UnsupportedDocumentFormatError, match="BabelDOC engine is deprecated"):
        get_adapter_for_path("paper.pdf", pdf_engine="babeldoc")


def test_get_adapter_for_path_unsupported_format() -> None:
    with pytest.raises(UnsupportedDocumentFormatError, match="No adapter registered"):
        get_adapter_for_path("novel.xyz")


def test_get_adapter_for_path_unsupported_pdf_engine() -> None:
    with pytest.raises(
        UnsupportedDocumentFormatError, match="Unsupported or unregistered PDF engine"
    ):
        get_adapter_for_path("paper.pdf", pdf_engine="unknown_engine")


def test_forced_pdfium_on_probe_docling_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Gate 1b: forcing pdfium where the probe wants docling warns loudly."""
    import ubt.adapters.factory as factory_mod
    from ubt.adapters.pdf.pdfium_adapter import PDFiumAdapter

    monkeypatch.setattr(factory_mod, "select_pdf_engine", lambda _p: "docling")
    pdf = tmp_path / "handbook.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    with caplog.at_level("WARNING", logger="ubt.adapters.factory"):
        adapter = get_adapter_for_path(pdf, pdf_engine="pdfium")
    assert isinstance(adapter, PDFiumAdapter)
    assert "recommends 'docling'" in caplog.text


def test_forced_pdfium_on_probe_pdfium_quiet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import ubt.adapters.factory as factory_mod
    from ubt.adapters.pdf.pdfium_adapter import PDFiumAdapter

    monkeypatch.setattr(factory_mod, "select_pdf_engine", lambda _p: "pdfium")
    pdf = tmp_path / "clean.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    with caplog.at_level("WARNING", logger="ubt.adapters.factory"):
        adapter = get_adapter_for_path(pdf, pdf_engine="pdfium")
    assert isinstance(adapter, PDFiumAdapter)
    assert "recommends 'docling'" not in caplog.text


def test_forced_pdfium_missing_path_no_probe_crash() -> None:
    from ubt.adapters.pdf.pdfium_adapter import PDFiumAdapter

    adapter = get_adapter_for_path("ghost.pdf", pdf_engine="pdfium")
    assert isinstance(adapter, PDFiumAdapter)
