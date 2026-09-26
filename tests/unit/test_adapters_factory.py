"""Tests for adapter factory resolution."""

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from ubt.adapters import (
    DoclingPDFAdapter,
    DOCXAdapter,
    EPUBAdapter,
    HTMLAdapter,
    MarkdownAdapter,
)
from ubt.adapters.base import BaseDocumentAdapter
from ubt.adapters.factory import (
    _ADAPTER_REGISTRY,
    _PDF_ENGINE_REGISTRY,
    get_adapter_for_path,
    register_adapter,
    register_pdf_engine,
)
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.exceptions import UnsupportedDocumentFormatError
from ubt.core.ir.models import BookManifest, ChapterIR


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


def test_adapters_declare_the_suffixes_they_write() -> None:
    from ubt.adapters.base import BaseDocumentAdapter, BasePDFEngineAdapter
    from ubt.adapters.epub.adapter import EPUBAdapter
    from ubt.adapters.markdown.adapter import MarkdownAdapter

    assert BaseDocumentAdapter.output_suffixes == frozenset()
    assert BasePDFEngineAdapter.output_suffixes == frozenset({".pdf"})
    assert MarkdownAdapter.output_suffixes == frozenset({".md", ".markdown", ".txt"})
    assert EPUBAdapter.output_suffixes == frozenset({".epub"})


def test_adapter_registry_extensibility() -> None:
    """Verify that new adapters and PDF engines can be registered without modifying core factory."""

    class CustomXYZAdapter(BaseDocumentAdapter):
        async def extract_manifest(self, input_path: Path) -> BookManifest:
            return BookManifest(
                doc_id="xyz_doc",
                title="XYZ",
                source_path=str(input_path),
                source_lang="en",
                target_lang="zh",
                chapters=[],
            )

        async def parse_stream(
            self, input_path: Path, pages: set[int] | None = None
        ) -> AsyncIterator[ChapterIR]:
            yield ChapterIR(
                doc_id="xyz_doc", chapter_id="xyz_ch1", title="XYZ", spine_index=1, blocks=[]
            )

        async def render_output(
            self,
            manifest: BookManifest,
            ledger: SQLiteJobLedger,
            target_lang: str,
            output_path: Path,
            job_id: str | None = None,
            bilingual_mode: str | None = None,
            **kwargs: Any,
        ) -> Path:
            return output_path

            # Register a custom mock adapter factory

    @register_adapter([".xyz", ".zyx"])
    def make_xyz_adapter(pdf_engine: str, path: Path) -> BaseDocumentAdapter:
        return CustomXYZAdapter()

        # Verify lookup succeeds for both extensions

    adapter_xyz = get_adapter_for_path(Path("sample.xyz"))
    assert isinstance(adapter_xyz, CustomXYZAdapter)

    adapter_zyx = get_adapter_for_path(Path("sample.zyx"))
    assert isinstance(adapter_zyx, CustomXYZAdapter)

    # Register a custom PDF engine
    @register_pdf_engine(["custom_pdf_engine"])
    def create_custom_engine() -> Any:
        return CustomXYZAdapter()

    custom_pdf = get_adapter_for_path(Path("sample.pdf"), pdf_engine="custom_pdf_engine")
    assert isinstance(custom_pdf, CustomXYZAdapter)

    # Verify unsupported extension raises UnsupportedDocumentFormatError
    with pytest.raises(UnsupportedDocumentFormatError) as exc:
        get_adapter_for_path(Path("file.unknown_ext"))
    assert "No adapter registered" in str(exc.value)

    # Verify unsupported PDF engine raises UnsupportedDocumentFormatError
    with pytest.raises(UnsupportedDocumentFormatError) as exc:
        get_adapter_for_path(Path("file.pdf"), pdf_engine="nonexistent_engine")
    assert "Unsupported or unregistered PDF engine" in str(exc.value)

    # Clean up registries
    _ADAPTER_REGISTRY.pop(".xyz", None)
    _ADAPTER_REGISTRY.pop(".zyx", None)
    _PDF_ENGINE_REGISTRY.pop("custom_pdf_engine", None)
