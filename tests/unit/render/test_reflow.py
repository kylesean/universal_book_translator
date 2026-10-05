"""Band-reflow unit tests: the pure repacking transform over Overlays."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import pytest

from ubt.model.span import PhysicalBox
from ubt.render.outputs import Overlay
from ubt.render.reflow import reflow_overlays

_Item = tuple[str, float, float, float | None, str, bool]


def _ov(eid: str, page: int, bbox: tuple[float, float, float, float], text: str = "x") -> Overlay:
    return Overlay(eid, page, bbox, text, font_size=11.0)


def _measure(constant: float) -> Callable[[Sequence[_Item]], list[float]]:
    def _many(items: Sequence[_Item]) -> list[float]:
        return [constant for _ in items]

    return _many


def _cap(kind: str, font_size: float | None) -> float:
    return (font_size or 10.0) * 1.05


def test_two_stacked_paragraphs_are_repacked_into_the_band() -> None:
    # A is above B with a small source gap; the pass stacks them top-down and
    # spends the band's slack on the paragraph gap.
    a = _ov("a", 1, (10.0, 100.0, 200.0, 140.0))
    b = _ov("b", 1, (10.0, 50.0, 200.0, 90.0))

    out = reflow_overlays([a, b], [], measure_many=_measure(20.0), cap_size=_cap)
    by_id = {ov.element_id: ov for ov in out}

    # Both moved to a fixed box, and each keeps its original box to mask.
    assert by_id["a"].fixed_box and by_id["b"].fixed_box
    assert by_id["a"].mask_boxes == (PhysicalBox.of(1, (10.0, 100.0, 200.0, 140.0)),)
    # A sits at the band top (y1 == 140) and is 20 + eps tall.
    assert by_id["a"].bbox[3] == pytest.approx(140.0)
    assert by_id["a"].bbox[1] == pytest.approx(140.0 - 20.5)
    # B is placed below A with the same gap the band could afford.
    gap = by_id["a"].bbox[1] - by_id["b"].bbox[3]
    assert gap > 0.0
    assert by_id["b"].bbox[3] < by_id["a"].bbox[1]


def test_gaps_between_paragraphs_are_uniform() -> None:
    band = [
        _ov("a", 1, (10.0, 160.0, 200.0, 200.0)),
        _ov("b", 1, (10.0, 110.0, 200.0, 150.0)),
        _ov("c", 1, (10.0, 60.0, 200.0, 100.0)),
    ]
    out = reflow_overlays(band, [], measure_many=_measure(20.0), cap_size=_cap)
    by_id = {ov.element_id: ov for ov in out}
    gap_ab = by_id["a"].bbox[1] - by_id["b"].bbox[3]
    gap_bc = by_id["b"].bbox[1] - by_id["c"].bbox[3]
    assert gap_ab == pytest.approx(gap_bc)


def test_a_band_whose_target_overflows_is_left_on_the_source_boxes() -> None:
    a = _ov("a", 1, (10.0, 100.0, 200.0, 140.0))
    b = _ov("b", 1, (10.0, 50.0, 200.0, 90.0))

    out = reflow_overlays([a, b], [], measure_many=_measure(60.0), cap_size=_cap)

    assert out == (a, b)


def test_an_obstacle_between_paragraphs_splits_the_band() -> None:
    a = _ov("a", 1, (10.0, 160.0, 200.0, 200.0))
    b = _ov("b", 1, (10.0, 60.0, 200.0, 100.0))
    figure = PhysicalBox.of(1, (10.0, 100.0, 200.0, 150.0))

    out = reflow_overlays([a, b], [figure], measure_many=_measure(20.0), cap_size=_cap)

    assert out == (a, b)


def test_two_columns_are_not_merged() -> None:
    left = _ov("a", 1, (10.0, 100.0, 200.0, 140.0))
    right = _ov("b", 1, (300.0, 90.0, 490.0, 130.0))

    out = reflow_overlays([left, right], [], measure_many=_measure(20.0), cap_size=_cap)

    assert out == (left, right)


def test_a_lone_paragraph_is_untouched() -> None:
    only = _ov("a", 1, (10.0, 100.0, 200.0, 140.0))
    out = reflow_overlays([only], [], measure_many=_measure(20.0), cap_size=_cap)
    assert out == (only,)


def test_non_text_overlays_are_anchors_not_moved() -> None:
    heading = Overlay("h", 1, (10.0, 200.0, 200.0, 220.0), "Title", kind="heading")
    a = _ov("a", 1, (10.0, 100.0, 200.0, 140.0))
    b = _ov("b", 1, (10.0, 50.0, 200.0, 90.0))

    out = reflow_overlays([heading, a, b], [], measure_many=_measure(20.0), cap_size=_cap)
    by_id = {ov.element_id: ov for ov in out}

    assert by_id["h"] == heading
    assert by_id["a"].fixed_box and by_id["b"].fixed_box


def test_the_class_size_is_the_document_minimum_cap() -> None:
    # Two same-class paragraphs whose caps differ (a caller-supplied cap) must be
    # measured and drawn at one shared size, matching the compositor's uniform rule.
    seen: list[float] = []

    def measure(
        items: Sequence[tuple[str, float, float, float | None, str, bool]],
    ) -> list[float]:
        seen.extend(item[2] for item in items)
        return [20.0 for _ in items]

    a = Overlay("a", 1, (10.0, 100.0, 200.0, 140.0), "x", font_size=12.0)
    b = Overlay("b", 1, (10.0, 50.0, 200.0, 90.0), "y", font_size=9.0)

    reflow_overlays([a, b], [], measure_many=measure, cap_size=_cap)

    # Different font sizes are different style classes, so each keeps its own cap.
    assert seen == [pytest.approx(12.0 * 1.05), pytest.approx(9.0 * 1.05)]
