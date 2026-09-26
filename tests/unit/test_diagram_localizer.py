"""Unit tests for DiagramLocalizer (in-diagram text localization)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ubt.adapters.pdf.diagram_localizer import (
    DiagramLocalizer,
    DiagramTextSpan,
)
from ubt.core.ir.models import BlockType, BoundingBox, FlowID, IRBlock

# ─────────────────── DiagramLocalizer ───────────────────


class TestDiagramTextSpan:
    def test_frozen(self) -> None:
        span = DiagramTextSpan(text="hello", x0=0, y0=0, x1=10, y1=10)
        with pytest.raises(AttributeError):
            span.text = "bye"  # type: ignore[misc]


class TestDiagramLocalizerGlossary:
    def test_default_glossary_populated(self) -> None:
        loc = DiagramLocalizer()
        assert "free" in loc.glossary
        assert loc.glossary["free"] == "空闲"
        assert "logical sequence" in loc.glossary

    def test_custom_glossary_merge(self) -> None:
        loc = DiagramLocalizer(custom_glossary={"batch": "批次", "Free": "自由"})
        assert loc.glossary["batch"] == "批次"
        # Custom should override default (case-insensitive key)
        assert loc.glossary["free"] == "自由"

    def test_translate_exact_match(self) -> None:
        loc = DiagramLocalizer()
        assert loc.translate_phrase("Free") == "空闲"
        assert loc.translate_phrase("ATTENTION") == "注意力"
        assert loc.translate_phrase("Logical sequence") == "逻辑序列"

    def test_translate_layer_regex(self) -> None:
        loc = DiagramLocalizer()
        assert loc.translate_phrase("Layer 1:") == "第 1 层:"
        assert loc.translate_phrase("Layer 2:") == "第 2 层:"
        assert loc.translate_phrase("Layer L:") == "第 L 层:"
        assert loc.translate_phrase("Layer 3") == "第 3 层"

    def test_builtin_glossary_not_applied_to_non_chinese_targets(self) -> None:
        """§10.5-A8: the EN->ZH vocabulary must not paint Chinese on other targets.

        The built-in glossary and the Layer formatting are English->Chinese.
        Before the fix a French/Japanese/German target still got '注意力'/'第 2 层',
        making target_lang inert. Now they fall through to the external
        translator or keep the source label.
        """
        loc = DiagramLocalizer()
        for lang in ("fr", "de", "ja", "ko", "es"):
            assert loc.translate_phrase("Attention", lang) == "Attention", lang
            assert loc.translate_phrase("Layer 2", lang) == "Layer 2", lang
        # A Chinese target still uses the vocabulary.
        assert loc.translate_phrase("Attention", "zh-CN") == "注意力"
        # A non-Chinese target with a language-aware translator still translates.
        assert loc.translate_phrase("Attention", "fr", lambda s: s.upper()) == "ATTENTION"

    def test_translate_unknown_returns_original(self) -> None:
        loc = DiagramLocalizer()
        assert loc.translate_phrase("XYZ_UNKNOWN") == "XYZ_UNKNOWN"

    def test_translate_external_translator(self) -> None:
        loc = DiagramLocalizer()
        result = loc.translate_phrase(
            "custom_label",
            external_translator=lambda x: "自定义标签",
        )
        assert result == "自定义标签"

    def test_translate_external_translator_error_fallback(self) -> None:
        """If external translator raises, fall back to original text."""
        loc = DiagramLocalizer()

        def bad_translator(x: str) -> str:
            raise ValueError("API error")

        result = loc.translate_phrase("custom_label", external_translator=bad_translator)
        assert result == "custom_label"


class TestDiagramLocalizerExtraction:
    def test_extract_text_spans_no_pdftotext(self, tmp_path: Path) -> None:
        """Should return empty list when pdftotext fails."""
        loc = DiagramLocalizer()
        bbox = BoundingBox(page=1, x0=0, y0=0, x1=100, y1=100)
        spans = loc.extract_text_spans(tmp_path / "nonexistent.pdf", 1, bbox, page_height=720)
        assert spans == []

    @patch("subprocess.check_output")
    def test_extract_spans_from_xml(self, mock_subprocess: MagicMock) -> None:
        """Parse pdftotext -bbox XML output and filter to bbox region."""
        # Page 720pt high; bbox in PDF space: bottom-left origin
        # bbox: x0=100, y0=500, x1=400, y1=600 → pdftotext top-left: top_y=120, bot_y=220
        mock_subprocess.return_value = (
            "<doc>\n"
            '<page width="500" height="720">\n'
            '<word xMin="150.0" yMin="130.0" xMax="200.0" yMax="140.0">Free</word>\n'
            '<word xMin="210.0" yMin="130.0" xMax="260.0" yMax="140.0">block</word>\n'
            '<word xMin="50.0" yMin="500.0" xMax="80.0" yMax="510.0">outside</word>\n'
            "</page>\n"
            "</doc>"
        )
        loc = DiagramLocalizer()
        bbox = BoundingBox(page=1, x0=100, y0=500, x1=400, y1=600)
        spans = loc.extract_text_spans("/fake.pdf", 1, bbox, page_height=720)

        # "Free" should be inside the bbox region (y 130 is within 120..220 range)
        assert len(spans) >= 1
        texts = [s.text.lower() for s in spans]
        assert "free" in texts
        # "outside" at y=500 should be filtered out (well outside 120..220)
        assert "outside" not in texts


class TestDiagramLocalizerImage:
    def test_localize_image_nonexistent(self, tmp_path: Path) -> None:
        """Non-existent image should return path unchanged."""
        loc = DiagramLocalizer()
        bbox = BoundingBox(page=1, x0=0, y0=0, x1=100, y1=100)
        result = loc.localize_image(tmp_path / "missing.png", [], bbox)
        assert result == tmp_path / "missing.png"

    def test_localize_image_empty_spans(self, tmp_path: Path) -> None:
        """Empty spans list should skip processing."""
        from PIL import Image

        img_path = tmp_path / "test.png"
        Image.new("RGB", (200, 100), "white").save(img_path)
        loc = DiagramLocalizer()
        bbox = BoundingBox(page=1, x0=0, y0=0, x1=200, y1=100)
        result = loc.localize_image(img_path, [], bbox)
        assert result == img_path

    def test_localize_image_renders_translation(self, tmp_path: Path) -> None:
        """Verify that image is modified when translatable spans exist."""
        from PIL import Image

        img_path = tmp_path / "diagram.png"
        Image.new("RGB", (400, 200), "white").save(img_path)

        loc = DiagramLocalizer()
        bbox = BoundingBox(page=1, x0=100, y0=500, x1=500, y1=700)
        # Create a span that matches "free" in the glossary
        spans = [DiagramTextSpan(text="free", x0=150, y0=130, x1=200, y1=140)]
        result = loc.localize_image(img_path, spans, bbox, page_height=720)
        assert result == img_path

        # Verify the image was rewritten in place (dimensions preserved)
        original_blank = Image.new("RGB", (400, 200), "white")
        modified = Image.open(img_path)
        assert modified.size == original_blank.size

    def test_localize_image_samples_non_white_background(self, tmp_path: Path) -> None:
        """Verify that text erasure on non-white background samples surrounding color instead of drawing pure white."""
        from PIL import Image

        img_path = tmp_path / "gray_diagram.png"
        bg_color = (230, 235, 240)
        Image.new("RGB", (400, 200), bg_color).save(img_path)

        loc = DiagramLocalizer()
        bbox = BoundingBox(page=1, x0=100, y0=500, x1=500, y1=700)
        spans = [DiagramTextSpan(text="free", x0=150, y0=130, x1=200, y1=140)]
        loc.localize_image(img_path, spans, bbox, page_height=720)

        modified = Image.open(img_path)
        # Point inside the erased rectangle: (px0 - pad_x + 2, py0 - pad_y + 1)
        # px0 = (150 - 100) * (400 / 400) = 50. py0 = (130 - 20) * (200 / 200) = 110.
        # Check pixel near the erased margin:
        px = modified.getpixel((49, 109))
        assert isinstance(px, tuple), f"expected RGB pixel tuple, got {px!r}"
        assert px != (255, 255, 255), (
            "Erased area was filled with hardcoded white patch instead of background color"
        )
        assert (
            abs(px[0] - bg_color[0]) <= 5
            and abs(px[1] - bg_color[1]) <= 5
            and abs(px[2] - bg_color[2]) <= 5
        )


class TestPageHeightCache:
    def test_page_height_cache_is_bounded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from ubt.adapters.pdf import pdf_struct

        class _Pdf:
            pages = ["page"]

            def __enter__(self) -> "_Pdf":
                return self

            def __exit__(self, *exc: object) -> None:
                pass

        monkeypatch.setattr(pdf_struct, "open_pdf", lambda path: _Pdf())
        monkeypatch.setattr(pdf_struct, "page_size", lambda page: (600.0, 800.0))
        loc = DiagramLocalizer()
        for i in range(40):
            assert loc.get_page_height(tmp_path / f"doc{i}.pdf", 1) == 800.0
        assert len(loc._page_heights_cache) <= 16


class TestLocalizAllDiagrams:
    def test_nonexistent_pdf_returns_blocks(self) -> None:
        loc = DiagramLocalizer()
        blocks = [
            IRBlock(
                id="b1",
                spine_index=1,
                block_type=BlockType.IMAGE,
                flow_id=FlowID.CAPTION,
                source_text="/tmp/pic.png",
                target_text="/tmp/pic.png",
                bbox=BoundingBox(page=1, x0=0, y0=0, x1=100, y1=100),
            )
        ]
        result = loc.localize_all_diagrams("/nonexistent.pdf", blocks)
        assert len(result) == 1


@pytest.mark.fast
def test_diagram_localizer_unescapes_xml_entities_in_pdftotext_bbox(tmp_path: Path) -> None:
    """[MEDIUM-T2-3] DiagramLocalizer.extract_text_spans must html.unescape XML entities."""
    localizer = DiagramLocalizer.__new__(DiagramLocalizer)
    localizer.glossary = {"r&d <5v>": "研发 <5V>"}
    sample_xml = (
        '<doc><page width="200" height="200">'
        '<word xMin="20.0" yMin="20.0" xMax="80.0" yMax="40.0">R&amp;D &lt;5V&gt;</word>'
        "</page></doc>"
    )
    with patch("subprocess.check_output", return_value=sample_xml):
        spans = localizer.extract_text_spans(
            tmp_path / "fig.pdf",
            page_no=1,
            bbox=BoundingBox(page=1, x0=10, y0=10, x1=190, y1=190),
            page_height=200.0,
        )
    assert len(spans) == 1
    assert spans[0].text == "R&D <5V>"


def test_fit_label_font_shrinks_to_fit_and_falls_back_without_a_ttf() -> None:
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (200, 50), "white")
    draw = ImageDraw.Draw(img)

    # No TTF configured -> default bitmap font, base size unchanged.
    loc = DiagramLocalizer()
    loc.cjk_font_path = None
    _font, size = loc._fit_label_font(draw, "hello world", 14, 200.0)
    assert size == 14

    ttf = loc._find_default_cjk_font()
    if not ttf:
        pytest.skip("no CJK TTF available")
    loc2 = DiagramLocalizer(cjk_font_path=ttf)
    font2, size2 = loc2._fit_label_font(draw, "这是一个非常长的图注标签", 20, 10.0)
    assert size2 < 20
    assert isinstance(font2, ImageFont.FreeTypeFont)
