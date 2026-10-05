"""Continuation runs draw at one uniform size across their box chain."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, ClassVar

import pytest

from ubt.model.span import PhysicalBox
from ubt.render.outputs import LayerCompositor, Overlay

pytestmark = pytest.mark.fast


class _Measurer:
    """A ``FragmentTypesetter`` stub exposing only what ``_flow_plan`` reads."""

    name: ClassVar[str] = "measurer"

    def cap_size(self, kind: str, font_size: float | None) -> float:
        return (font_size or 10.0) * 1.05

    def measure_fixed(
        self,
        text: str,
        width: float,
        size: float,
        *,
        kind: str = "text",
        is_bold: bool = False,
        indent_pt: float | None = None,
    ) -> float:
        per_line = max(1.0, width / size)
        return max(1, math.ceil(len(text) / per_line)) * size * 1.2

    def measure(self, text: str, width_pt: float) -> float:
        return self.measure_fixed(text, width_pt, 10.0)

    def typeset(
        self,
        text: str,
        width_pt: float,
        height_pt: float,
        *,
        kind: str = "text",
        font_size: float | None = None,
        is_bold: bool = False,
        runs: tuple[Any, ...] = (),
    ) -> Path | None:
        return None

    def typeset_math(self, latex: str, width_pt: float, height_pt: float) -> Path | None:
        return None


def _overlay(text: str, boxes: tuple[PhysicalBox, ...]) -> Overlay:
    return Overlay("e", boxes[0].page, boxes[0].bbox, text, boxes=boxes, font_size=11.0)


def test_a_continuation_that_fits_keeps_the_cap_size() -> None:
    boxes = (
        PhysicalBox.of(1, (0.0, 100.0, 300.0, 140.0)),
        PhysicalBox.of(2, (0.0, 700.0, 300.0, 760.0)),
    )
    compositor = LayerCompositor("unused.pdf", typesetter=_Measurer())

    _parts, size = compositor._flow_plan(_overlay("短句。" * 5, boxes), boxes)

    assert size == pytest.approx(11.0 * 1.05)


def test_an_overlong_continuation_shrinks_the_whole_run_uniformly() -> None:
    boxes = (
        PhysicalBox.of(1, (0.0, 100.0, 300.0, 120.0)),
        PhysicalBox.of(2, (0.0, 700.0, 300.0, 720.0)),
    )
    compositor = LayerCompositor("unused.pdf", typesetter=_Measurer())

    _parts, size = compositor._flow_plan(_overlay("长" * 400, boxes), boxes)

    assert size is not None
    assert size < 11.0 * 1.05
