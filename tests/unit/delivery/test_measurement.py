"""MeasurementService cache scaffold tests: real delegate, stable keys."""

from __future__ import annotations

import pytest

from ubt.delivery.measurement import (
    CachedMeasurementService,
    MeasuredBlock,
    MeasurementStyle,
    measurement_cache_key,
)

pytestmark = pytest.mark.fast


def _style(*, font_version: str = "2026.10") -> MeasurementStyle:
    return MeasurementStyle(
        font_family="Noto Serif CJK SC",
        font_version=font_version,
        size_pt=10.0,
        leading_pt=13.0,
    )


def test_cache_reuses_one_real_measurement() -> None:
    calls: list[tuple[str, MeasurementStyle, float]] = []

    def delegate(text: str, style: MeasurementStyle, width_pt: float) -> MeasuredBlock:
        calls.append((text, style, width_pt))
        return MeasuredBlock(height_pt=12.0, line_count=1, measurement_id="delegate")

    service = CachedMeasurementService(delegate)
    first = service.measure_block("hello", _style(), 120.0)
    second = service.measure_block("hello", _style(), 120.0)

    assert first == second
    assert len(calls) == 1
    assert service.cache_size == 1
    assert first.measurement_id == measurement_cache_key("hello", _style(), 120.0)


def test_cache_key_changes_with_width() -> None:
    calls: list[tuple[str, MeasurementStyle, float]] = []

    def delegate(text: str, style: MeasurementStyle, width_pt: float) -> MeasuredBlock:
        calls.append((text, style, width_pt))
        return MeasuredBlock(height_pt=12.0, line_count=1, measurement_id="delegate")

    service = CachedMeasurementService(delegate)
    service.measure_block("hello", _style(), 120.0)
    service.measure_block("hello", _style(), 180.0)

    assert len(calls) == 2
    assert service.cache_size == 2


def test_cache_key_changes_with_font_version() -> None:
    calls: list[tuple[str, MeasurementStyle, float]] = []

    def delegate(text: str, style: MeasurementStyle, width_pt: float) -> MeasuredBlock:
        calls.append((text, style, width_pt))
        return MeasuredBlock(height_pt=12.0, line_count=1, measurement_id="delegate")

    service = CachedMeasurementService(delegate)
    service.measure_block("hello", _style(), 120.0)
    service.measure_block("hello", _style(font_version="2027.01"), 120.0)

    assert len(calls) == 2
    assert service.cache_size == 2


def test_non_positive_width_is_refused() -> None:
    def delegate(text: str, style: MeasurementStyle, width_pt: float) -> MeasuredBlock:
        return MeasuredBlock(height_pt=12.0, line_count=1, measurement_id="delegate")

    service = CachedMeasurementService(delegate)
    with pytest.raises(ValueError, match="width_pt must be positive"):
        service.measure_block("hello", _style(), 0.0)
