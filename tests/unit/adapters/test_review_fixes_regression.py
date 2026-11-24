"""Regression tests for second-round audit fixes across adapters, codecs, and app layer."""

from __future__ import annotations

from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pikepdf
import pytest

from ubt.adapters.pdf.artifact_gate import run_artifact_gate
from ubt.adapters.pdf.docling_adapter import ScanPDFToTypstCodec
from ubt.adapters.pdf.docling_parser import extract_with_docling, vlm_fallback_missing_pages
from ubt.adapters.pdf.page_chunking import compute_pdf_page_chunks
from ubt.app.document_services import AdapterDocumentServices
from ubt.app.spec import JobSpec
from ubt.codecs.protocol import DeliveryMode
from ubt.core.exceptions import DocumentParseError
from ubt.core.ir.models import BlockType, BoundingBox, FlowID, IRBlock, make_element
from ubt.render.typst.fonts import DEFAULT_FONT_DIR

pytestmark = pytest.mark.fast


def _pdf_with_text(path: Path, text: str, *, font_name: str = "/Helvetica") -> Path:
    pdf = pikepdf.new()
    font = pdf.make_indirect(
        pikepdf.Dictionary(
            {
                "/Type": pikepdf.Name("/Font"),
                "/Subtype": pikepdf.Name("/Type1"),
                "/BaseFont": pikepdf.Name(font_name),
            }
        )
    )
    page = pdf.add_blank_page(page_size=(400, 300))
    page["/Resources"] = pikepdf.Dictionary({"/Font": pikepdf.Dictionary({"/F1": font})})
    page["/Contents"] = pikepdf.Stream(
        pdf, f"BT /F1 12 Tf 40 200 Tm ({text}) Tj ET".encode("latin1")
    )
    pdf.save(path)
    pdf.close()
    return path


def test_artifact_gate_accepts_korean_hangul_and_skips_vector_figure_blank_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf_path = tmp_path / "ko_with_figure.pdf"
    pdf = pikepdf.new()
    p1 = pdf.add_blank_page(page_size=(400, 300))
    p1["/Contents"] = pikepdf.Stream(pdf, b"BT /F1 12 Tf (x) Tj ET")
    p2 = pdf.add_blank_page(page_size=(400, 300))
    # Page 2 is a pure vector figure page (path paint op), not a broken blank page.
    p2["/Contents"] = pikepdf.Stream(pdf, b"10 10 100 100 re S")
    pdf.save(pdf_path)
    pdf.close()

    monkeypatch.setattr(
        "ubt.adapters.pdf.artifact_gate._page_texts",
        lambda _p: ["한국어 번역 테스트 문장입니다 충분히 긴 텍스트입니다.", ""],
    )
    res = run_artifact_gate(pdf_path, target_lang="ko")
    codes = {f.code for f in res.findings}
    assert "target_language_absent" not in codes
    assert "blank_page" not in codes
    assert res.passed is True


def test_artifact_gate_reports_per_page_unmapped_font_even_when_other_pages_have_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf_path = _pdf_with_text(tmp_path / "mixed_tounicode.pdf", "Normal page text here")
    monkeypatch.setattr(
        "ubt.adapters.pdf.artifact_gate._page_texts",
        lambda _p: ["这是第一页正常的中文译文内容超过二十个字符。", ""],
    )
    # Page 1 actually has text, wait: single-page PDF here — build 2 pages so page 2 has unmapped font
    pdf = pikepdf.new()
    f_ok = pdf.make_indirect(
        pikepdf.Dictionary(
            {
                "/Type": pikepdf.Name("/Font"),
                "/Subtype": pikepdf.Name("/Type1"),
                "/BaseFont": pikepdf.Name("/Helvetica"),
                "/ToUnicode": pdf.make_stream(b""),
            }
        )
    )
    f_bad = pdf.make_indirect(
        pikepdf.Dictionary(
            {
                "/Type": pikepdf.Name("/Font"),
                "/Subtype": pikepdf.Name("/Type1"),
                "/BaseFont": pikepdf.Name("/Custom"),
            }
        )
    )
    p1 = pdf.add_blank_page(page_size=(400, 300))
    p1["/Resources"] = pikepdf.Dictionary({"/Font": pikepdf.Dictionary({"/F1": f_ok})})
    p2 = pdf.add_blank_page(page_size=(400, 300))
    p2["/Resources"] = pikepdf.Dictionary({"/Font": pikepdf.Dictionary({"/F2": f_bad})})
    pdf.save(pdf_path)
    pdf.close()

    res = run_artifact_gate(pdf_path, target_lang="zh")
    assert any(f.code == "font_missing_tounicode" and f.page == 2 for f in res.findings)


@pytest.mark.asyncio
async def test_scan_reflow_codec_does_not_mutate_cache_or_collide_empty_pages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    codec = ScanPDFToTypstCodec()

    b1 = IRBlock(
        element=make_element(
            id="b1",
            spine_index=0,
            block_type=BlockType.NARRATIVE,
            source_text="Source 1",
            bbox=BoundingBox(page=1, x0=10, y0=10, x1=100, y1=30),
        )
    )
    b2 = IRBlock(
        element=make_element(
            id="b2",
            spine_index=1,
            block_type=BlockType.NARRATIVE,
            source_text="Source 2",
            bbox=BoundingBox(page=2, x0=10, y0=10, x1=100, y1=30),
        )
    )

    from collections.abc import AsyncIterator

    async def _fake_manifest(_p: Path) -> object:
        from ubt.core.ir.models import BookManifest

        return BookManifest(
            doc_id="d1", title="T", source_path=str(pdf), metadata={"page_count": 2}
        )

    async def _fake_parse(_p: Path, pages: set[int] | None = None) -> AsyncIterator[object]:
        from ubt.core.ir.models import ChapterIR

        selected = [b for b in (b1, b2) if pages is None or (b.bbox and b.bbox.page in pages)]
        yield ChapterIR(doc_id="d1", chapter_id="c1", title="C1", spine_index=0, blocks=selected)

    rendered_targets: list[dict[str, str | None]] = []

    async def _fake_render(
        _manifest: object,
        blocks: list[IRBlock],
        _lang: str,
        out_path: Path,
        **_kwargs: object,
    ) -> Path:
        rendered_targets.append({b.id: b.target_text for b in blocks})
        out_path.write_bytes(b"%PDF-1.4\n")
        return out_path

    monkeypatch.setattr(codec, "extract_manifest", _fake_manifest)
    monkeypatch.setattr(codec, "parse_stream", _fake_parse)
    monkeypatch.setattr(codec, "render_blocks", _fake_render)

    _, empty_units = await codec.extract(pdf, pages=set())
    _, full_units = await codec.extract(pdf, pages=None)
    assert empty_units == []
    assert [u.unit_id for u in full_units] == ["b1", "b2"]

    rep1 = await codec.apply(
        pdf,
        full_units,
        {"b1": "译文1", "b2": "译文2"},
        target_lang="zh",
        output_path=tmp_path / "o1.pdf",
    )
    rep2 = await codec.apply(
        pdf, [full_units[0]], {"b1": "仅译1"}, target_lang="zh", output_path=tmp_path / "o2.pdf"
    )
    assert rep1.total_units == 2 and rep2.total_units == 2
    assert rendered_targets[0] == {"b1": "译文1", "b2": "译文2"}
    assert rendered_targets[1] == {"b1": "仅译1", "b2": None}


def test_docling_isolated_worker_empty_read_refuses_in_process_fallback(tmp_path: Path) -> None:
    pdf = tmp_path / "empty_scan.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    mock_pool = MagicMock()
    mock_pool.convert.return_value = "cache-key-empty"

    with (
        patch("ubt.adapters.pdf.docling_worker.get_docling_worker_pool", return_value=mock_pool),
        patch("ubt.adapters.pdf.docling_cache.read_docling_document", return_value=None),
        patch("ubt.adapters.pdf.docling_parser.convert_docling_document") as mock_inproc,
    ):
        with pytest.raises(DocumentParseError, match="refusing to load the Docling model"):
            extract_with_docling(path=pdf, page_range=None, enrich=False, isolate_process=True)
        mock_inproc.assert_not_called()

    # Keep BrokenProcessPool coverage sanity check intact.
    assert BrokenProcessPool is not None


def test_page_chunking_clamps_out_of_range_pages_and_all_mode_respects_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert compute_pdf_page_chunks(total_pages=5, pages={1, 3, 99}, chunk_size=10) == [[1, 3]]

    class _FakePdf:
        def __init__(self, _path: str) -> None:
            self._pages = [object(), object(), object()]

        def __len__(self) -> int:
            return len(self._pages)

        def __getitem__(self, index: int) -> object:
            return self._pages[index]

        def close(self) -> None:
            return

    b1 = IRBlock(
        element=make_element(
            id="b1",
            spine_index=1,
            block_type=BlockType.NARRATIVE,
            flow_id=FlowID.MAIN_STORY,
            source_text="Page 1 text",
            bbox=BoundingBox(page=1, x0=10, y0=10, x1=100, y1=20),
        )
    )
    b2 = IRBlock(
        element=make_element(
            id="b2",
            spine_index=2,
            block_type=BlockType.NARRATIVE,
            flow_id=FlowID.MAIN_STORY,
            source_text="Page 2 text",
            bbox=BoundingBox(page=2, x0=10, y0=10, x1=100, y1=20),
        )
    )
    b3 = IRBlock(
        element=make_element(
            id="b3",
            spine_index=3,
            block_type=BlockType.NARRATIVE,
            flow_id=FlowID.MAIN_STORY,
            source_text="Page 3 text",
            bbox=BoundingBox(page=3, x0=10, y0=10, x1=100, y1=20),
        )
    )

    import pypdfium2

    import ubt.adapters.pdf.page_profiler as page_profiler
    import ubt.adapters.pdf.vlm.registry as registry

    monkeypatch.setattr(pypdfium2, "PdfDocument", _FakePdf)
    monkeypatch.setattr(
        page_profiler,
        "profile_pdf",
        lambda *_args, **_kwargs: [
            SimpleNamespace(
                facts=SimpleNamespace(page=1), kind=SimpleNamespace(value="editable_text")
            ),
            SimpleNamespace(facts=SimpleNamespace(page=2), kind=SimpleNamespace(value="figure")),
            SimpleNamespace(
                facts=SimpleNamespace(page=3), kind=SimpleNamespace(value="editable_text")
            ),
        ],
    )
    probed: list[object] = []

    def _probe(**kwargs: object) -> tuple[None, None]:
        probed.append(kwargs)
        return None, None

    monkeypatch.setattr(registry, "probe_effective_driver", _probe)

    result = vlm_fallback_missing_pages(
        Path("scope.pdf"),
        [b1, b2, b3],
        ocr_fallback="all",
        page_scope=[1, 3],
        allow_page_upload=True,
        ocr_mode="local",
    )
    # Page 2 is inside [1..3] but NOT in page_scope {1, 3}, so ALL-mode proofread must skip it.
    assert probed == []
    assert result == [b1, b2, b3]


def test_job_spec_layout_mode_string_and_services_defaults() -> None:
    spec = JobSpec(input_path="book.pdf", layout_mode="interleaved")
    assert spec.codec_mode == DeliveryMode.INTERLEAVED
    assert spec.resolved_output_path().name == "book_zh_interleaved.pdf"

    masked = AdapterDocumentServices().masked_source(text="hello")
    assert masked.placeholders == ()
    assert DEFAULT_FONT_DIR.name == "fonts"


def test_mcp_server_installs_document_services() -> None:
    import importlib
    import sys

    import ubt.app.bootstrap as bootstrap
    import ubt.core.engine._document_services_registry as doc_registry

    bootstrap.reset_services_for_testing()
    doc_registry.provider = None
    if "ubt.mcp.server" in sys.modules:
        importlib.reload(sys.modules["ubt.mcp.server"])
    else:
        import ubt.mcp.server  # noqa: F401
    try:
        assert bootstrap._installed is True
        assert isinstance(doc_registry.provider, AdapterDocumentServices)
    finally:
        bootstrap.install_services()
