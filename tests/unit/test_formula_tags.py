"""Unit tests for source-PDF equation tag recovery (formula_tags)."""

from pathlib import Path

import pytest

from ubt.adapters.pdf.formula_tags import match_tag_in_page, recover_formula_tag


def _boxes_for(
    text: str, x_start: float, y: float, per_char: float = 6.0
) -> list[tuple[float, float, float, float]]:
    """Synthetic character boxes laid out left to right on one row band."""
    boxes = []
    x = x_start
    for _ch in text:
        boxes.append((x, y, x + per_char, y + 9.0))
        x += per_char
    return boxes


def test_match_prefers_tag_nearest_the_formula_right_edge() -> None:
    """Nearest wins, not rightmost: the other column's tag is far away."""
    text = "(A.1)(B.2)"
    boxes = _boxes_for("(A.1)", 420.0, 500.0) + _boxes_for("(B.2)", 700.0, 500.0)
    assert match_tag_in_page(text, boxes, 495.0, 520.0, 456.0) == "A.1"


def test_match_ignores_tags_in_other_vertical_bands() -> None:
    text = "(A.1)"
    boxes = _boxes_for(text, 430.0, 500.0)
    assert match_tag_in_page(text, boxes, 100.0, 130.0, 460.0) is None


def test_match_ignores_oversized_parentheses() -> None:
    """A display-formula height pair is not an equation number."""
    text = "(3.1)"
    boxes = [(430.0, 480.0, 455.0, 520.0)] * len(text)
    assert match_tag_in_page(text, boxes, 480.0, 520.0, 456.0) is None


def test_match_normalizes_spaced_ocr_tags() -> None:
    text = "( A . 12a )"
    boxes = _boxes_for(text, 430.0, 500.0)
    assert match_tag_in_page(text, boxes, 495.0, 520.0, 456.0) == "A.12a"


def test_match_returns_none_without_candidates() -> None:
    text = "no tags here ( 2 r - 1 )"
    boxes = _boxes_for(text, 120.0, 500.0)
    assert match_tag_in_page(text, boxes, 495.0, 520.0, 456.0) is None


@pytest.mark.skipif(
    not Path("tests/fixtures/synthetic-duo.pdf").exists(),
    reason="tests/fixtures/synthetic-duo.pdf not present",
)
def test_recover_synthetic_appendix_and_body_tags() -> None:
    """Golden bands from the synthetic corpus, calibrated against pdfium's
    text-page space (bboxes mirror what the ledger would carry for the
    formula blocks; see scripts/make_sample_corpus.py)."""

    class _BBox:
        def __init__(self, page: int, y0: float, y1: float, x1: float) -> None:
            self.page = page
            self.y0 = y0
            self.y1 = y1
            self.x1 = x1

    # Appendix identity (A.6) on PDF page 3.
    assert (
        recover_formula_tag("tests/fixtures/synthetic-duo.pdf", 3, _BBox(3, 232.0, 246.0, 240.0))
        == "A.6"
    )
    # Body formula (3.2) on PDF page 1 (bands calibrated against pdfium).
    assert (
        recover_formula_tag("tests/fixtures/synthetic-duo.pdf", 1, _BBox(1, 302.0, 312.0, 456.0))
        == "3.2"
    )
    # A tag-free band yields None instead of guessing.
    assert (
        recover_formula_tag("tests/fixtures/synthetic-duo.pdf", 18, _BBox(18, 10.0, 20.0, 456.0))
        is None
    )


@pytest.mark.skipif(
    not Path("tests/fixtures/synthetic-duo.pdf").exists(),
    reason="tests/fixtures/synthetic-duo.pdf not present",
)
def test_renderer_uses_recovered_tag_in_generated_typst() -> None:
    """A formula whose OCR text has no tag still renders the author's number."""
    from ubt.adapters.pdf.typst_reconstructor import TypstReconstructor
    from ubt.core.ir.models import BlockType, BoundingBox, IRBlock

    recon = TypstReconstructor(target_lang="zh-cn")
    recon.source_pdf = Path("tests/fixtures/synthetic-duo.pdf")
    recon.math_backend = "typst"
    block = IRBlock(
        id="eq-recover",
        spine_index=1,
        block_type=BlockType.FORMULA,
        source_text=r"\beta = e^{F}",
        target_text=r"\beta = e^{F}",
        bbox=BoundingBox(page=1, x0=256.99, y0=302.0, x1=456.29, y1=312.0),
    )
    source = recon.generate_typst_source([block], bilingual=False, page_strict=False)

    assert 'numbering: _ => "(3.2)"' in source
