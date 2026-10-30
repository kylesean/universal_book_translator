"""Typst MeasurementService tests: real compile, real line boxes, real cache."""

from __future__ import annotations

import pytest

from ubt.adapters.pdf.typst_compile import typst_available
from ubt.adapters.pdf.typst_measurement import TypstMeasurementService
from ubt.delivery.measurement import (
    MeasurementError,
    MeasurementRequest,
    MeasurementStyle,
    measurement_cache_key,
)

pytestmark = [
    pytest.mark.fast,
    pytest.mark.skipif(not typst_available(), reason="typst compiler is not installed"),
]

_FONT = "Noto Serif CJK SC"


def _style(*, horizontal_scale: float = 1.0) -> MeasurementStyle:
    return MeasurementStyle(
        font_family=_FONT,
        font_version="test",
        size_pt=10.0,
        leading_pt=13.0,
        language="zh",
        horizontal_scale=horizontal_scale,
    )


def test_measures_real_lines_and_height() -> None:
    service = TypstMeasurementService()
    text = (
        "这是一段用于测量真实排版高度的中文段落，宽度固定，观察自动换行与页面高度。"
        "This is a second run of Latin text in the same paragraph."
    )
    measured = service.measure_block(text, _style(), 180.0)

    assert measured.line_count >= 2
    assert measured.height_pt > _style().size_pt
    assert len(measured.line_boxes) == measured.line_count
    assert measured.measurement_id == measurement_cache_key(text, _style(), 180.0)


def test_narrower_width_increases_line_count() -> None:
    service = TypstMeasurementService()
    text = "这是一段用于测量真实排版高度的中文段落，宽度固定，观察自动换行与页面高度。"
    wide = service.measure_block(text, _style(), 240.0)
    narrow = service.measure_block(text, _style(), 90.0)

    assert narrow.line_count > wide.line_count


def test_cache_reuses_one_compile() -> None:
    service = TypstMeasurementService()
    text = "短段落。"
    first = service.measure_block(text, _style(), 120.0)
    second = service.measure_block(text, _style(), 120.0)

    assert first == second
    assert service.cache_size == 1


def test_empty_text_is_zero_height_without_compiling() -> None:
    service = TypstMeasurementService()
    measured = service.measure_block("   ", _style(), 120.0)

    assert measured.line_count == 0
    assert measured.height_pt == 0.0
    assert measured.line_boxes == ()


def test_non_unity_horizontal_scale_is_refused() -> None:
    service = TypstMeasurementService()
    with pytest.raises(MeasurementError, match="horizontal_scale"):
        service.measure_block("水平缩放", _style(horizontal_scale=0.95), 120.0)


def test_measures_inline_math_with_real_typst_shaping() -> None:
    service = TypstMeasurementService()
    math_text = (
        "其中 $V_{tm} = k_B T / q$ 且 $\\psi_B = V_{tm} \\ln(N_{ch} / n_i)$ 是热电势与费米电势。"
    )
    measured = service.measure_block(math_text, _style(), 140.0)
    assert measured.line_count >= 1
    assert measured.height_pt > 0.0


def test_measure_many_batches_and_populates_cache() -> None:
    service = TypstMeasurementService()
    style = _style()
    reqs = [
        MeasurementRequest(
            text="第一段测量文本，包含内联公式 $x_1 + x_2 = y$。", style=style, width_pt=160.0
        ),
        MeasurementRequest(
            text="第二段较长的测量文本，用于验证批量编译与分页提取行高的一致性。",
            style=style,
            width_pt=120.0,
        ),
    ]
    batch_res = service.measure_many(reqs)
    assert len(batch_res) == 2
    assert service.cache_size == 2
    assert service.measure_block(reqs[0].text, style, 160.0) == batch_res[0]
    assert service.measure_block(reqs[1].text, style, 120.0) == batch_res[1]
