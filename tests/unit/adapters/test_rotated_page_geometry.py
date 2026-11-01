from __future__ import annotations

from pathlib import Path

import pypdfium2 as pdfium
import pytest
from pdf_builders import write_text_pdf
from PIL import ImageOps

from ubt.adapters.pdf.coordinate_resolver import (
    rotate_rect_clockwise,
    undo_page_rotation,
)
from ubt.adapters.pdf.vlm.anchor import anchor_transcript
from ubt.adapters.pdf.vlm.transcribe import transcribe_page_to_blocks
from ubt.adapters.pdf.vlm.types import PageTranscript, VlmLine

pytestmark = pytest.mark.fast

_PAGE_W = 300.0
_PAGE_H = 500.0
_LINE = ("ABCDEF",)


def _rotated(tmp_path: Path, rotation: int) -> Path:
    return write_text_pdf(
        tmp_path / f"rot{rotation}.pdf", [_LINE], width=_PAGE_W, height=_PAGE_H, rotation=rotation
    )


@pytest.mark.parametrize(("page_rotation", "expected"), [(0, 0), (90, 270), (180, 180), (270, 90)])
def test_undoing_a_page_rotation_is_its_complement(page_rotation: int, expected: int) -> None:
    assert undo_page_rotation(page_rotation) == expected


def test_an_unrotated_box_is_unchanged() -> None:
    box = (10.0, 20.0, 30.0, 40.0)
    assert rotate_rect_clockwise(box, undo_page_rotation(0), _PAGE_W, _PAGE_H) == box


def test_a_box_read_off_the_rendered_page_lands_on_the_text(tmp_path: Path) -> None:
    source = _rotated(tmp_path, 90)
    pdf = pdfium.PdfDocument(str(source))
    try:
        page = pdf[0]
        rotation = int(page.get_rotation())
        assert rotation == 90
        width, height = float(page.get_width()), float(page.get_height())
        assert (width, height) == (_PAGE_H, _PAGE_W)
        image = page.render(scale=1.0).to_pil().convert("RGB")
        textpage = page.get_textpage()
        textpage.count_rects(0, -1)
        text_rect = tuple(float(v) for v in textpage.get_rect(0))
        textpage.close()
        page.close()
    finally:
        pdf.close()

    ink = ImageOps.invert(image.convert("L")).getbbox()
    assert ink is not None
    measured_on_the_bitmap = (
        float(ink[0]),
        height - float(ink[3]),
        float(ink[2]),
        height - float(ink[1]),
    )

    mapped = rotate_rect_clockwise(
        measured_on_the_bitmap, undo_page_rotation(rotation), width, height
    )

    assert mapped == pytest.approx(text_rect, abs=2.0)
    assert not all(abs(a - b) < 2.0 for a, b in zip(measured_on_the_bitmap, text_rect, strict=True))


class _RecordingDriver:
    name = "recording"
    measured_boxes = False

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[float, float], float, int]] = []

    def recognize(
        self,
        image: object,
        page_size_pt: tuple[float, float],
        scale: float,
        rotation: int = 0,
        origin_pt: tuple[float, float] = (0.0, 0.0),
    ) -> PageTranscript:
        self.calls.append((page_size_pt, scale, rotation))
        return PageTranscript(lines=(VlmLine("ABCDEF"),), engine=self.name, measured_boxes=False)


def test_transcription_tells_the_driver_the_page_is_rotated(tmp_path: Path) -> None:
    source = _rotated(tmp_path, 90)
    driver = _RecordingDriver()

    transcribe_page_to_blocks(source, 1, driver=driver)

    (page_size_pt, _scale, rotation) = driver.calls[0]
    assert rotation == 90
    assert page_size_pt == (_PAGE_H, _PAGE_W)


class _DetectorDriver:
    name = "detector"
    measured_boxes = True

    def recognize(
        self,
        image: object,
        page_size_pt: tuple[float, float],
        scale: float,
        rotation: int = 0,
        origin_pt: tuple[float, float] = (0.0, 0.0),
    ) -> PageTranscript:
        width, height = page_size_pt
        ink = ImageOps.invert(image.convert("L")).getbbox()  # type: ignore[attr-defined]
        assert ink is not None
        box = (
            ink[0] / scale,
            height - ink[3] / scale,
            ink[2] / scale,
            height - ink[1] / scale,
        )
        measured = rotate_rect_clockwise(box, undo_page_rotation(rotation), width, height)
        return PageTranscript(
            lines=(VlmLine("ABCDEF", measured_box=measured),),
            engine=self.name,
            measured_boxes=True,
        )


def test_a_recognized_scan_page_gets_blocks_where_the_reader_sees_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("ubt.adapters.pdf.vlm.transcribe._harvest_text_lines", lambda _textpage: [])
    source = _rotated(tmp_path, 90)

    blocks, stats = transcribe_page_to_blocks(source, 1, driver=_DetectorDriver(), scale=1.0)

    (block,) = blocks
    assert stats.vlm_only == 1
    box = block.bbox
    assert box is not None
    assert (box.x0, box.y0, box.x1, box.y1) == pytest.approx((54.0, 445.0, 102.0, 456.0), abs=2.5)
    assert box.x1 <= _PAGE_W and box.y1 <= _PAGE_H


def _unplaceable_transcript() -> PageTranscript:
    return PageTranscript(
        lines=(VlmLine("unplaced", measured_box=None),),
        engine="stub",
        measured_boxes=True,
    )


def test_the_fallback_box_is_the_page_in_user_space() -> None:
    anchored, _stats = anchor_transcript(
        [], _unplaceable_transcript(), (_PAGE_H, _PAGE_W), rotation=90
    )

    assert [line.box for line in anchored] == [(0.0, 0.0, _PAGE_W, _PAGE_H)]
    assert anchored[0].needs_review


def test_the_fallback_box_is_unchanged_on_an_unrotated_page() -> None:
    anchored, _stats = anchor_transcript(
        [], _unplaceable_transcript(), (_PAGE_W, _PAGE_H), rotation=0
    )

    assert [line.box for line in anchored] == [(0.0, 0.0, _PAGE_W, _PAGE_H)]


def test_an_unsupported_rotation_is_reported_not_guessed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("WARNING", logger="ubt.adapters.pdf.coordinate_resolver"):
        box = rotate_rect_clockwise((1.0, 2.0, 3.0, 4.0), 45, _PAGE_W, _PAGE_H)

    assert box == (1.0, 2.0, 3.0, 4.0)
    assert "unsupported page rotation" in caplog.text
