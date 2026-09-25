"""Unit tests for the deterministic formula witness and its modes."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from ubt.adapters.pdf import formula_witness as fw
from ubt.adapters.pdf.typst_reconstructor import TypstReconstructor
from ubt.core.ir.models import BlockType, BoundingBox, IRBlock


def _img(width: int = 200, height: int = 60) -> Image.Image:
    return Image.new("L", (width, height), 255)


def _strokes(img: Image.Image, count: int = 8) -> Image.Image:
    draw = ImageDraw.Draw(img)
    for i in range(count):
        x = 10 + i * 20
        draw.line((x, 15, x, 45), fill=0, width=2)
    return img


def _blobs(img: Image.Image, count: int, size: int = 6) -> Image.Image:
    draw = ImageDraw.Draw(img)
    for i in range(count):
        x = 8 + i * 14
        draw.rectangle((x, 20, x + size, 20 + size), fill=0)
    return img


class TestCompareStructure:
    def test_similar_structures_pass(self) -> None:
        left = _strokes(_img())
        right = _strokes(_img())
        assert fw.compare_structure(left, right) == []

    def test_component_drop_is_reported(self) -> None:
        solid = _img()
        ImageDraw.Draw(solid).rectangle((10, 10, 120, 50), fill=0)
        sparse = _strokes(_img())
        findings = fw.compare_structure(solid, sparse)
        assert any("components" in f for f in findings)

    def test_aspect_mismatch_is_reported(self) -> None:
        wide = _img(300, 40)
        ImageDraw.Draw(wide).rectangle((5, 12, 295, 28), fill=0)
        tall = _img(60, 200)
        ImageDraw.Draw(tall).rectangle((15, 5, 45, 195), fill=0)
        assert any("aspect" in f for f in fw.compare_structure(wide, tall))

    def test_component_count_mismatch_is_reported(self) -> None:
        few = _blobs(_img(400, 60), count=3)
        many = _blobs(_img(400, 60), count=24)
        assert any("components" in f for f in fw.compare_structure(few, many))

    def test_blank_source_is_unwitnessable(self) -> None:
        assert fw.compare_structure(_strokes(_img()), _img()) == ["unwitnessable"]

    def test_blank_rendered_fails(self) -> None:
        findings = fw.compare_structure(_img(), _strokes(_img()))
        assert findings and "no measurable ink" in findings[0]


class TestTrimFormulaCrop:
    def _crop(self) -> Image.Image:
        return Image.new("L", (200, 120), 255)

    def test_bottom_neighbour_sliver_is_dropped(self) -> None:
        img = self._crop()
        draw = ImageDraw.Draw(img)
        draw.rectangle((10, 10, 190, 70), fill=0)
        draw.rectangle((10, 116, 190, 119), fill=0)
        out = fw.trim_formula_crop(img)
        assert out.height < img.height
        assert out.height <= 75

    def test_top_neighbour_sliver_is_dropped(self) -> None:
        img = self._crop()
        draw = ImageDraw.Draw(img)
        draw.rectangle((10, 0, 190, 3), fill=0)
        draw.rectangle((10, 40, 190, 100), fill=0)
        out = fw.trim_formula_crop(img)
        assert out.height < img.height
        assert out.height <= 65

    def test_internal_rows_are_preserved(self) -> None:
        img = self._crop()
        draw = ImageDraw.Draw(img)
        draw.rectangle((10, 20, 190, 50), fill=0)
        draw.rectangle((10, 60, 190, 80), fill=0)
        assert fw.trim_formula_crop(img).size == img.size

    def test_edge_band_close_to_body_is_preserved(self) -> None:
        img = self._crop()
        draw = ImageDraw.Draw(img)
        draw.rectangle((10, 5, 190, 55), fill=0)
        draw.rectangle((10, 58, 190, 80), fill=0)
        assert fw.trim_formula_crop(img).size == img.size

    def test_single_band_is_untouched(self) -> None:
        img = _strokes(_img())
        assert fw.trim_formula_crop(img).size == img.size


class _FakeBBox:
    def __init__(self, page: int = 1) -> None:
        self.page = page


def _default_bbox() -> BoundingBox:
    return BoundingBox(page=1, x0=1, y0=1, x1=50, y1=10)


def _no_bbox_block() -> IRBlock:
    return _block(bbox=BoundingBox(page=0, x0=0, y0=0, x1=0, y1=0))


def _block(bbox: BoundingBox | None = None) -> IRBlock:
    if bbox is None:
        bbox = _default_bbox()
    return IRBlock(
        id="pdf_main#b1",
        spine_index=1,
        block_type=BlockType.FORMULA,
        source_text=r"x = y",
        bbox=bbox,
    )


class TestWitnessFormula:
    def test_missing_bbox_is_unwitnessable(self) -> None:
        result = fw.witness_formula("$ x = y $", _no_bbox_block(), "src.pdf", "typst")
        assert result.status == "unwitnessable"

    def test_crop_failure_is_unwitnessable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import ubt.adapters.pdf.visual_scalpel as scalpel

        def _boom(*_a: object, **_k: object) -> object:
            raise RuntimeError("no pdf")

        monkeypatch.setattr(scalpel, "crop_block_pil", _boom)
        result = fw.witness_formula("$ x = y $", _block(), "src.pdf", "typst")
        assert result.status == "unwitnessable"

    def test_rasterizer_crash_is_unwitnessable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A PDF-backend blowup inside rasterize must not escape."""
        import ubt.adapters.pdf.visual_scalpel as scalpel

        monkeypatch.setattr(scalpel, "crop_block_pil", lambda *a, **k: _strokes(_img()))

        def _boom(*_a: object, **_k: object) -> object:
            raise RuntimeError("pdfium backend exploded")

        monkeypatch.setattr(fw, "rasterize_math", _boom)
        result = fw.witness_formula("$ x = y $", _block(), "src.pdf", "typst")
        assert result.status == "unwitnessable"

    def test_compare_crash_is_unwitnessable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A metric-backend blowup inside compare must not escape either."""
        import ubt.adapters.pdf.visual_scalpel as scalpel

        monkeypatch.setattr(scalpel, "crop_block_pil", lambda *a, **k: _strokes(_img()))
        monkeypatch.setattr(fw, "rasterize_math", lambda *a, **k: _strokes(_img()))

        def _boom(*_a: object, **_k: object) -> object:
            raise RuntimeError("metric backend exploded")

        monkeypatch.setattr(fw, "compare_structure", _boom)
        result = fw.witness_formula("$ x = y $", _block(), "src.pdf", "typst")
        assert result.status == "unwitnessable"


class TestReconstructorModes:
    def _recon_with_crop(self, monkeypatch: pytest.MonkeyPatch) -> TypstReconstructor:
        rec = TypstReconstructor()
        monkeypatch.setattr(
            rec, "_crop_formula_fallback", lambda block: f'#image("{block.id}.png")'
        )
        return rec

    def test_image_mode_replaces_display_formulas_only(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rec = self._recon_with_crop(monkeypatch)
        block = _block()
        lines = [
            "这里的行内公式 $x = y$ 不动。",
            f"$ x = y $  // [formula {block.id}]",
            "正文段落。",
        ]
        swapped = rec._substitute_formulas_with_source_graphic(lines, [block])
        assert swapped == 1
        assert lines[0] == "这里的行内公式 $x = y$ 不动。"
        assert lines[1] == f'#image("{block.id}.png")'
        assert lines[2] == "正文段落。"

    def test_witness_mode_swaps_failed_formula(self, monkeypatch: pytest.MonkeyPatch) -> None:
        rec = self._recon_with_crop(monkeypatch)
        rec.source_pdf = Path("/tmp/source.pdf")
        block = _block()

        def _fail(*_a: object, **_k: object) -> fw.WitnessResult:
            return fw.WitnessResult("fail", ["ink density 0.1 vs 0.4"])

        monkeypatch.setattr("ubt.adapters.pdf.formula_witness.witness_formula", _fail)
        lines = [f"$ x = y $  // [formula {block.id}]"]
        rec._witness_math_lines(lines, [block])
        assert lines[0] == f'#image("{block.id}.png")'
        assert rec.last_witness_findings and "ink density" in rec.last_witness_findings[0]

    def test_witness_mode_keeps_passing_formula(self, monkeypatch: pytest.MonkeyPatch) -> None:
        rec = self._recon_with_crop(monkeypatch)
        rec.source_pdf = Path("/tmp/source.pdf")
        block = _block()
        monkeypatch.setattr(
            "ubt.adapters.pdf.formula_witness.witness_formula",
            lambda *_a, **_k: fw.WitnessResult("pass"),
        )
        lines = [f"$ x = y $  // [formula {block.id}]"]
        assert rec._witness_math_lines(lines, [block]) == 0
        assert lines[0].startswith("$")
        assert rec.last_witness_findings == []

    def test_witness_mode_without_source_pdf_is_noop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        rec = TypstReconstructor()
        assert rec.source_pdf is None
        block = _block()
        lines = [f"$ x = y $  // [formula {block.id}]"]
        assert rec._witness_math_lines(lines, [block]) == 0
        assert lines[0].startswith("$")


def test_record_witness_findings_into_manifest() -> None:
    from ubt.adapters.pdf.docling_adapter import DoclingPDFAdapter
    from ubt.adapters.pdf.docling_render import record_witness_findings
    from ubt.core.ir.models import BookManifest

    adapter = DoclingPDFAdapter()
    adapter.reconstructor.last_witness_findings = ["pdf_main#b1: aspect 3.00x of source"]
    manifest = BookManifest(doc_id="d", title="t", source_path="s.pdf")
    record_witness_findings(adapter.reconstructor, manifest)
    assert manifest.metadata["formula_witness_findings"] == ["pdf_main#b1: aspect 3.00x of source"]
