"""Integration: textless-page VLM fallback through the real seam.

Builds an image-only PDF (no text layer) from docs/synthetic-mono.pdf at test time —
no committed image fixtures. Requires rapidocr (bundled ONNX models); skipped
otherwise. Exercises ``DoclingPDFAdapter.parse_stream`` end to end with
``UBT_VLM_SCAN_FALLBACK=1``: Docling (do_ocr=False) yields zero blocks for
the image page, the vlm/ core transcribes it, blocks come back with
``vlm:`` provenance. With the switch off, ingest fails loudly instead of
exporting an empty book.
"""

import asyncio
import importlib.util
from pathlib import Path

import pytest

from ubt.adapters.pdf.docling_adapter import DoclingPDFAdapter
from ubt.adapters.pdf.vlm.transcribe import FALLBACK_ENV_VAR
from ubt.core.exceptions import DocumentParseError

pytestmark = [
    pytest.mark.slow,  # live VLM backend calls
    pytest.mark.skipif(
        importlib.util.find_spec("rapidocr") is None
        or importlib.util.find_spec("onnxruntime") is None,
        reason="rapidocr (or its onnxruntime backend) not installed",
    ),
]

# NOTE: tests/fixtures/book3-*.pdf was removed; any real PDF works as the raster source.
BOOK3 = Path("docs/synthetic-mono.pdf")


def _image_only_pdf(dst: Path) -> Path:
    import pypdfium2 as pdfium

    src = pdfium.PdfDocument(str(BOOK3))
    try:
        img = src[0].render(scale=1.5).to_pil().convert("RGB")
    finally:
        src.close()
    out = dst / "textless.pdf"
    img.save(out, "PDF")
    probe = pdfium.PdfDocument(str(out))
    try:
        chars = len(probe[0].get_textpage().get_text_range(0, -1) or "")
    finally:
        probe.close()
    assert chars < 50, f"fixture is not textless: {chars} chars"
    return out


def test_fallback_off_fail_loud_on_textless_book(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Old contract was ``blocks == []`` (the silent-empty-book defect, §10.5-A4)."""
    monkeypatch.delenv(FALLBACK_ENV_VAR, raising=False)
    pdf = _image_only_pdf(tmp_path)
    with pytest.raises(DocumentParseError, match="No content could be parsed"):
        DoclingPDFAdapter._vlm_fallback_missing_pages(pdf, [])


def test_fallback_on_transcribes_textless_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(FALLBACK_ENV_VAR, "1")
    pdf = _image_only_pdf(tmp_path)
    blocks = DoclingPDFAdapter._vlm_fallback_missing_pages(pdf, [])
    assert len(blocks) > 5
    assert all(b.bbox is not None and b.bbox.page == 1 for b in blocks)
    assert all(str(b.provenance.get("parser", "")).startswith("vlm:") for b in blocks)
    assert any(
        "MATHEMATICS" in (b.source_text or "") or len(b.source_text or "") > 20 for b in blocks
    )
    spine = [b.spine_index for b in blocks]
    assert spine == sorted(spine) and len(set(spine)) == len(spine)


def test_parse_stream_seam_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(FALLBACK_ENV_VAR, "1")
    pdf = _image_only_pdf(tmp_path)

    async def _run() -> int:
        adapter = DoclingPDFAdapter()
        n = 0
        async for chapter in adapter.parse_stream(pdf):
            n += len(chapter.blocks)
        return n

    assert asyncio.run(_run()) > 5
