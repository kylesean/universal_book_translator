"""Unit tests for the F-img SVG vector diagram route (poppler backend)."""

from __future__ import annotations

import io
import os
import shutil
from pathlib import Path

import pytest

from ubt.adapters.pdf import svg_diagram
from ubt.adapters.pdf.svg_diagram import (
    LocalizedSpan,
    detect_diagram_regions,
    export_page_svg,
    is_docling_direct_image,
    is_svg_backend_available,
    is_svg_rendering_supported,
    localize_diagram_svg,
    render_diagram_svg,
)

KV_PDF = Path(
    os.environ.get(
        "UBT_TEST_KV_PDF",
        str(Path(__file__).parents[2] / "tests" / "fixtures" / "kv-cache-handbook.pdf"),
    )
)
DOCS_PDF = (
    Path(os.environ["UBT_TEST_REAL_PDF"]).expanduser()
    if os.environ.get("UBT_TEST_REAL_PDF")
    else Path(__file__).parents[2] / "docs" / "chapter-3.pdf"
)
SAMPLE_PDF = KV_PDF if KV_PDF.exists() else DOCS_PDF

# One reason, whichever condition is actually missing: "poppler or test PDF
# missing" blamed the wrong component on a machine with poppler installed and a
# Typst build that cannot render SVG at all.
_SVG_SKIP_REASON = next(
    (
        reason
        for reason, missing in (
            ("poppler (pdftocairo) unavailable", not is_svg_backend_available()),
            (
                "this Typst build does not render SVG shapes (vector diagrams fall back to PNG)",
                not is_svg_rendering_supported(),
            ),
            (
                f"{SAMPLE_PDF} fixture missing (set UBT_TEST_REAL_PDF)",
                not SAMPLE_PDF.exists(),
            ),
        )
        if missing
    ),
    "",
)

NEEDS_POPPLER = pytest.mark.skipif(bool(_SVG_SKIP_REASON), reason=_SVG_SKIP_REASON)
# Named like this so a skip says which input is gone: "test PDF missing" sent
# nobody to the right place after docs/chapter-3.pdf was withdrawn.
NEEDS_SAMPLE_PDF = pytest.mark.skipif(
    not SAMPLE_PDF.exists(),
    reason=f"{SAMPLE_PDF} fixture missing (set UBT_TEST_REAL_PDF to a scanned chapter PDF)",
)

_PAGE_SVG = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="595pt" height="842pt" viewBox="0 0 595 842">
<g id="page1"><path d="M 10 10 L 100 10"/></g>
</svg>
"""


def _write_page_svg(path: Path) -> Path:
    path.write_text(_PAGE_SVG, encoding="utf-8")
    return path


def test_docling_direct_image_id_gate() -> None:
    assert is_docling_direct_image("pdf_main#img_0007")
    assert is_docling_direct_image("pdf_main#img_42")
    assert not is_docling_direct_image("pdf_main#img_pic_p6_3")
    assert not is_docling_direct_image("ch01#p001")


def test_localize_crops_viewbox_and_backfills_labels(tmp_path: Path) -> None:
    page_svg = _write_page_svg(tmp_path / "page_1.svg")
    out = localize_diagram_svg(
        page_svg,
        bbox_topdown=(100.0, 200.0, 300.0, 400.0),
        spans=[
            LocalizedSpan(text="Layer 1:", translated="第 1 层:", x0=110, y0=210, x1=170, y1=224),
            LocalizedSpan(text="K", translated="K", x0=180, y0=210, x1=188, y1=224),
        ],
        out_path=tmp_path / "diagram.svg",
    )
    text = out.read_text(encoding="utf-8")
    # Cropped to bbox + 2pt pad.
    assert 'viewBox="98 198 204 204"' in text
    assert 'width="204pt"' in text
    # Translated label patched; untranslated single letter untouched.
    assert "第 1 层:" in text
    assert "<text" in text
    assert text.count("<rect") == 1
    assert "white" in text


def test_localize_escapes_xml_specials(tmp_path: Path) -> None:
    page_svg = _write_page_svg(tmp_path / "page_1.svg")
    out = localize_diagram_svg(
        page_svg,
        bbox_topdown=(0.0, 0.0, 200.0, 200.0),
        spans=[LocalizedSpan(text="a", translated="x < y & z", x0=10, y0=10, x1=60, y1=24)],
        out_path=tmp_path / "esc.svg",
    )
    text = out.read_text(encoding="utf-8")
    assert "x &lt; y &amp; z" in text


def test_localize_rejects_tiny_bbox(tmp_path: Path) -> None:
    from ubt.core.exceptions import DocumentParseError

    page_svg = _write_page_svg(tmp_path / "page_1.svg")
    with pytest.raises((ValueError, DocumentParseError), match="too small"):
        localize_diagram_svg(
            page_svg,
            bbox_topdown=(0.0, 0.0, 5.0, 5.0),
            spans=[],
            out_path=tmp_path / "tiny.svg",
        )


def test_render_returns_none_without_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    assert not svg_diagram.is_svg_backend_available()
    from ubt.adapters.pdf.diagram_localizer import DiagramLocalizer

    assert (
        render_diagram_svg(
            KV_PDF,
            page_no=6,
            bbox_bottomup=(50.0, 50.0, 400.0, 400.0),
            page_height=720.0,
            localizer=DiagramLocalizer(),
            out_path=tmp_path / "x.svg",
        )
        is None
    )


def test_get_diagram_backend_status_diagnostics(monkeypatch: pytest.MonkeyPatch) -> None:
    from ubt.adapters.pdf.svg_diagram import get_diagram_backend_status

    status = get_diagram_backend_status()
    assert isinstance(status, dict)
    assert "preferred_mode" in status
    assert status["preferred_mode"] in {"vector_svg", "raster_png", "none"}
    assert "oxide" in status and "raster_backend" in status
    # pdf_oxide is a base dependency: raster attribution is always oxide.
    assert status["raster_backend"] == "oxide"

    # Simulate missing poppler SVG tools: raster still preferred via oxide.
    monkeypatch.setattr(shutil, "which", lambda name: None)
    fallback_status = get_diagram_backend_status()
    assert fallback_status["preferred_mode"] == "raster_png"
    assert not fallback_status["pdftocairo"]


@NEEDS_POPPLER
def test_export_page_svg_real_pdf(tmp_path: Path) -> None:
    page_no = 6 if KV_PDF.exists() else 1
    out = export_page_svg(SAMPLE_PDF, page_no, tmp_path)
    assert out is not None and out.exists()
    assert "<svg" in out.read_text(encoding="utf-8")[:500]


@NEEDS_POPPLER
def test_render_diagram_svg_end_to_end(tmp_path: Path) -> None:
    """Full vector path on real PDF: export, span locate, backfill."""
    from ubt.adapters.pdf.diagram_localizer import DiagramLocalizer

    localizer = DiagramLocalizer()
    page_no = 6 if KV_PDF.exists() else 1
    bbox = (0.0, 0.0, 514.8, 720.0) if KV_PDF.exists() else (83.679, 586.261, 540.0, 617.159)
    page_height = 720.0 if KV_PDF.exists() else 666.0
    out = render_diagram_svg(
        SAMPLE_PDF,
        page_no=page_no,
        bbox_bottomup=bbox,
        page_height=page_height,
        localizer=localizer,
        external_translator=lambda s: f"译:{s}",
        work_dir=tmp_path / "work",
        out_path=tmp_path / f"diagram_p{page_no}.svg",
    )
    assert out is not None and out.exists()
    text = out.read_text(encoding="utf-8")
    assert "译:" in text
    assert "<text" in text


# ---------------------------------------------------------------------------
# Content-stream vector region detection (pikepdf-based, no poppler needed)
# ---------------------------------------------------------------------------


def _build_vector_pdf(path: Path, pages_ops: list[list[str]]) -> Path:
    """Minimal PDF whose pages contain raw vector ops (re/m/l/S)."""
    streams: list[bytes] = []
    for ops in pages_ops:
        body = "q\n" + "\n".join(ops) + "\nQ\n"
        streams.append(body.encode("latin-1"))
    page_ids = [4 + 2 * i for i in range(len(streams))]
    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {len(streams)} >>".encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for pid, stream in zip(page_ids, streams, strict=True):
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
            f"/Contents {pid + 1} 0 R /Resources << /Font << /F1 3 0 R >> >> >>".encode()
        )
        objects.append(
            b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream"
        )
    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets: list[int] = []
    for i, obj_body in enumerate(objects, start=1):
        offsets.append(out.tell())
        out.write(f"{i} 0 obj\n".encode() + obj_body + b"\nendobj\n")
    xref_pos = out.tell()
    out.write(f"xref\n0 {len(objects) + 1}\n".encode())
    out.write(b"0000000000 65535 f \n")
    for off in offsets:
        out.write(f"{off:010d} 00000 n \n".encode())
    out.write(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF".encode()
    )
    path.write_bytes(out.getvalue())
    return path


def test_detect_merges_nearby_rects(tmp_path: Path) -> None:
    pdf = _build_vector_pdf(
        tmp_path / "vec.pdf",
        [
            [
                "100 100 60 40 re S",  # cluster A (gap 10 < 14 merges)
                "170 100 60 40 re S",
                "400 400 80 60 re S",  # cluster B, far away
            ]
        ],
    )
    regions = detect_diagram_regions(pdf, 1)
    assert len(regions) == 2
    boxes = sorted(regions)
    assert boxes[0] == pytest.approx((100, 100, 230, 140))
    assert boxes[1] == pytest.approx((400, 400, 480, 460))


def test_detect_drops_hairlines_and_noise(tmp_path: Path) -> None:
    pdf = _build_vector_pdf(
        tmp_path / "noise.pdf",
        [
            [
                "0 800 595 1 re S",  # full-bleed hairline rule
                "50 50 5 5 re S",  # tiny speck
                "300 300 100 80 re S",
                "310 310 60 50 re S",
            ]
        ],
    )
    regions = detect_diagram_regions(pdf, 1)
    assert len(regions) == 1
    assert regions[0] == pytest.approx((300, 300, 400, 380))


def test_detect_honours_table_exclusion(tmp_path: Path) -> None:
    pdf = _build_vector_pdf(
        tmp_path / "table.pdf",
        [["100 100 200 120 re S", "110 110 50 40 re S", "170 130 40 30 re S"]],
    )
    assert len(detect_diagram_regions(pdf, 1)) == 1
    assert detect_diagram_regions(pdf, 1, exclude=[(90.0, 90.0, 320.0, 230.0)]) == []


def test_detect_clamps_to_mediabox(tmp_path: Path) -> None:
    pdf = _build_vector_pdf(
        tmp_path / "spill.pdf",
        [["500 700 150 150 re S", "510 710 60 60 re S", "520 720 40 40 re S"]],
    )
    regions = detect_diagram_regions(pdf, 1)
    assert len(regions) == 1
    x0, y0, x1, y1 = regions[0]
    assert x1 <= 595 and y1 <= 842
    # Mostly-off-page artefacts are dropped, not clamped.
    pdf2 = _build_vector_pdf(
        tmp_path / "spill2.pdf",
        [["600 850 200 200 re S", "610 860 60 60 re S", "620 870 40 40 re S"]],
    )
    assert detect_diagram_regions(pdf2, 1) == []


@NEEDS_SAMPLE_PDF
def test_detect_real_handbook_diagram_pages() -> None:
    """Pages known to carry vector diagrams are found; coords stay on-page."""
    pages = (6, 8, 13, 16, 21) if KV_PDF.exists() else (1, 3, 4, 8, 9)
    found = {p: detect_diagram_regions(SAMPLE_PDF, p) for p in pages}
    assert all(len(regions) >= 1 for regions in found.values()), found
    for regions in found.values():
        for x0, y0, x1, y1 in regions:
            assert 0 <= x0 < x1 <= 600.0
            assert 0 <= y0 < y1 <= 850.0


def _w(x0: float, y0: float, text: str) -> tuple[float, float, float, float, str]:
    return (x0, y0, x0 + len(text) * 5.0, y0 + 8.0, text)


def test_text_panel_rejects_prose_with_trivial_ink() -> None:
    from ubt.adapters.pdf.svg_diagram import is_text_panel

    words = [
        _w(60, 110, "Primary"),
        _w(110, 110, "papers"),
        _w(160, 110, "and"),
        _w(60, 125, "Vaswani"),
        _w(120, 125, "Attention"),
        _w(180, 125, "NeurIPS"),
    ]
    assert is_text_panel(1, words) is True
    assert is_text_panel(2, words) is True


def test_text_panel_keeps_real_diagrams() -> None:
    from ubt.adapters.pdf.svg_diagram import is_text_panel

    # Rich ink always keeps, even with prose around.
    assert is_text_panel(5, [_w(60 + i * 30, 110, f"word{i}") for i in range(6)]) is False
    # Single chip with one label row.
    assert is_text_panel(1, [_w(100, 110, "Decode")]) is False
    # Symbol annotations only.
    assert is_text_panel(1, [_w(100, 110, "K"), _w(120, 110, "V")]) is False
    # No words at all.
    assert is_text_panel(1, []) is False


@NEEDS_SAMPLE_PDF
def test_detect_drops_p33_title_panel() -> None:
    if KV_PDF.exists():
        regions = detect_diagram_regions(KV_PDF, 33)
    else:
        # docs/synthetic-duo.pdf page 2 has background fill and text but no diagrams
        regions = detect_diagram_regions(SAMPLE_PDF, 2)
    assert regions == []


def test_row_bands_merge_same_row_chips() -> None:
    from ubt.adapters.pdf.svg_diagram import _merge_row_bands

    chips = [
        (88.0, 414.0, 250.0, 437.0),
        (276.0, 414.0, 338.0, 437.0),
        (364.0, 414.0, 427.0, 437.0),
    ]
    merged = _merge_row_bands(chips)
    assert len(merged) == 1
    assert merged[0] == pytest.approx((88.0, 414.0, 427.0, 437.0))


def test_row_bands_keep_vertical_stacks() -> None:
    from ubt.adapters.pdf.svg_diagram import _merge_row_bands

    stack = [
        (198.0, 438.0, 317.0, 461.0),
        (198.0, 511.0, 317.0, 534.0),
        (198.0, 549.0, 317.0, 571.0),
    ]
    assert len(_merge_row_bands(stack)) == 3


@NEEDS_SAMPLE_PDF
def test_detect_p3_single_row_region() -> None:
    if KV_PDF.exists():
        regions = detect_diagram_regions(KV_PDF, 3)
        assert len(regions) == 1
        x0, y0, x1, y1 = regions[0]
        assert x0 < 150 and x1 > 350  # spans the whole Prompt→Decode row
    else:
        regions = detect_diagram_regions(SAMPLE_PDF, 1)
        assert len(regions) == 1
        x0, y0, x1, y1 = regions[0]
        assert x0 < 100 and x1 > 500


def test_dereference_svg_uses_inlines_glyphs() -> None:
    """Poppler <use> glyph refs become plain translated geometry."""
    import xml.etree.ElementTree as ET

    from ubt.adapters.pdf.svg_diagram import dereference_svg_uses

    root = ET.fromstring(
        '<svg xmlns="http://www.w3.org/2000/svg">'
        '<defs><g id="g0"><path d="M 0 0 L 1 1" /></g></defs>'
        '<g fill="black"><use href="#g0" x="10" y="20" /></g>'
        "</svg>"
    )
    assert dereference_svg_uses(root) == 1
    serialized = ET.tostring(root, encoding="unicode")
    assert "<use" not in serialized
    assert 'transform="translate(10,20)"' in serialized
    assert ":path" in serialized or "<path" in serialized


def test_dereference_svg_uses_skips_dangling() -> None:
    """Dangling references are left alone (never crash the stager)."""
    import xml.etree.ElementTree as ET

    from ubt.adapters.pdf.svg_diagram import dereference_svg_uses

    root = ET.fromstring(
        '<svg xmlns="http://www.w3.org/2000/svg"><g><use href="#missing" x="1" y="2" /></g></svg>'
    )
    assert dereference_svg_uses(root) == 0


def test_svg_rendering_probe_reports_this_typst() -> None:
    """Capability probe reflects the real binary (bool, cached, stable)."""
    import shutil

    import pytest

    from ubt.adapters.pdf.svg_diagram import is_svg_rendering_supported

    if shutil.which("typst") is None:
        pytest.skip("typst binary unavailable")
    first = is_svg_rendering_supported()
    assert isinstance(first, bool)
    assert is_svg_rendering_supported() == first


def test_render_diagram_png_crops_region(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """In-process pdf_oxide page raster cropped to the diagram bbox."""
    import pytest

    pytest.importorskip("PIL")

    from PIL import Image

    from ubt.adapters.pdf.svg_diagram import render_diagram_png

    src = DOCS_PDF
    if not src.exists():
        pytest.skip(f"{src} missing (set UBT_TEST_REAL_PDF to a scanned chapter PDF)")
    out = tmp_path / "ras_p1_0.png"
    got = render_diagram_png(
        src,
        page_no=1,
        bbox_bottomup=(80.0, 400.0, 500.0, 600.0),
        page_height=792.0,
        out_path=out,
    )
    assert got is not None and out.exists()
    with Image.open(out) as im:
        assert 1500 < im.width < 1900
        assert 700 < im.height < 950


def test_render_diagram_png_rejects_tiny_bbox(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from ubt.adapters.pdf.svg_diagram import render_diagram_png

    assert (
        render_diagram_png(
            "whatever.pdf",
            page_no=1,
            bbox_bottomup=(0, 0, 5, 5),
            page_height=800.0,
            out_path=tmp_path / "x.png",
        )
        is None
    )


def test_probe_disk_cache_hit_skips_subprocess(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    import json

    from ubt.adapters.pdf import svg_diagram

    monkeypatch.setenv("UBT_CACHE_DIR", str(tmp_path))
    binary = "/fake/typst"
    disk = svg_diagram._probe_disk_path(binary)
    assert disk is not None
    disk.parent.mkdir(parents=True, exist_ok=True)
    disk.write_text(
        json.dumps({"binary": binary, "mtime_ns": 123, "result": True}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        svg_diagram,
        "_probe_rendering_cached",
        lambda b, m: (_ for _ in ()).throw(AssertionError("must not probe")),
    )
    assert svg_diagram._probe_disk_hit(binary, 123, disk) is True


def test_probe_disk_cache_stale_mtime_misses(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    import json

    from ubt.adapters.pdf import svg_diagram

    monkeypatch.setenv("UBT_CACHE_DIR", str(tmp_path))
    binary = "/fake/typst"
    disk = svg_diagram._probe_disk_path(binary)
    assert disk is not None
    disk.parent.mkdir(parents=True, exist_ok=True)
    disk.write_text(
        json.dumps({"binary": binary, "mtime_ns": 123, "result": True}),
        encoding="utf-8",
    )
    assert svg_diagram._probe_disk_hit(binary, 999, disk) is None


def test_detect_rejects_page_background_fill() -> None:
    from ubt.adapters.pdf.svg_diagram import detect_diagram_regions

    src = DOCS_PDF
    if not src.exists():
        pytest.skip(f"{src} missing (set UBT_TEST_REAL_PDF to a scanned chapter PDF)")
    # Source p2 paints `0 0 540 665.972 re` (page background): it must not
    # come back as a near-full-page "diagram" thumbnail.
    assert detect_diagram_regions(src, 2) == []
