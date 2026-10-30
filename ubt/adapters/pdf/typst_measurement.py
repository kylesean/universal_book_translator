"""Typst-backed MeasurementService: real line boxes, not a height estimate.

The probe lays the text out on an auto-height page with the requested style,
compiles it with the same Typst binary the renderer uses, and reads the PDF's
own text lines. It is intentionally a separate adapter: ``ubt.delivery`` owns
the protocol/cache, this module owns the compiler dependency.

Limitations are explicit rather than silent:

* ``horizontal_scale`` is refused until a shaping path (HarfBuzz/rustybuzz)
  can change advance widths. Typst's ``text(stretch:)`` did not change static
  font widths in the measured version, so reporting a measurement for it would
  be a fake. CELS Stage 3A can add that path later.
* A compiler/font/probe failure raises :class:`MeasurementError`; it never
  returns a zero-height "fits" result.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

from ubt.adapters.pdf.textgeom import extract_lines
from ubt.adapters.pdf.typst_compile import typst_available, typst_compile, typst_version
from ubt.delivery.measurement import (
    CachedMeasurementService,
    LineBox,
    MeasuredBlock,
    MeasurementError,
    MeasurementStyle,
)

logger = logging.getLogger(__name__)

__all__ = ["TypstMeasurementService"]

#: A line may exceed the requested measure by this much because Typst's
#: justification/overhang can paint hanging punctuation into the margin. The
#: allowance grows with the font size (half an em), so a real unbreakable word
#: still trips the overflow flag while a hanging comma does not.
_OVERFLOW_TOLERANCE_PT = 1.0
_OVERFLOW_OVERHANG_EM = 0.5


class TypstMeasurementService:
    """Measure a block by compiling the smallest Typst document that holds it."""

    def __init__(self, *, binary: str = "typst", timeout: float = 30.0) -> None:
        self._binary = binary
        self._timeout = timeout
        self._cache = CachedMeasurementService(self._measure_uncached)

    @property
    def cache_size(self) -> int:
        return self._cache.cache_size

    @property
    def typst_version(self) -> str | None:
        return typst_version(self._binary)

    def measure_block(
        self,
        text: str,
        style: MeasurementStyle,
        width_pt: float,
    ) -> MeasuredBlock:
        return self._cache.measure_block(text, style, width_pt)

    def _measure_uncached(
        self, text: str, style: MeasurementStyle, width_pt: float
    ) -> MeasuredBlock:
        if abs(style.horizontal_scale - 1.0) > 1e-6:
            raise MeasurementError(
                "horizontal_scale is not supported by the Typst probe; "
                "use 1.0 until a shaping-backed measurement lands"
            )
        if not text.strip():
            return MeasuredBlock(height_pt=0.0, line_count=0, measurement_id="")
        if not typst_available(self._binary):
            raise MeasurementError(
                f"Typst compiler binary {self._binary!r} is not available; "
                "cannot produce a real measurement"
            )
        source = self._probe_source(text, style, width_pt)
        with tempfile.TemporaryDirectory(prefix="ubt_measure_") as tmp:
            typ_path = Path(tmp) / "probe.typ"
            pdf_path = Path(tmp) / "probe.pdf"
            typ_path.write_text(source, encoding="utf-8")
            ok, stderr = typst_compile(
                str(typ_path),
                str(pdf_path),
                binary=self._binary,
                timeout=self._timeout,
                target_lang=style.language,
            )
            if not ok:
                raise MeasurementError(f"Typst measurement compile failed: {stderr}")
            lines, _page_size = extract_lines(pdf_path, 1)
        if not lines:
            raise MeasurementError("Typst measurement produced no text lines")
        top = max(line.rect[3] for line in lines)
        bottom = min(line.rect[1] for line in lines)
        height = top - bottom
        if height <= 0.0:
            raise MeasurementError("Typst measurement produced a non-positive height")
        boxes = tuple(
            LineBox(
                x0=float(line.rect[0]),
                y0=float(top - line.rect[3]),
                x1=float(line.rect[2]),
                y1=float(top - line.rect[1]),
            )
            for line in lines
        )
        overflow_tolerance = max(_OVERFLOW_TOLERANCE_PT, style.size_pt * _OVERFLOW_OVERHANG_EM)
        overflow = any(
            float(line.rect[2]) > width_pt + overflow_tolerance
            or float(line.rect[0]) < -overflow_tolerance
            for line in lines
        )
        return MeasuredBlock(
            height_pt=round(float(height), 2),
            line_count=len(lines),
            line_boxes=boxes,
            overflow=overflow,
            measurement_id="",
        )

    @staticmethod
    def _probe_source(text: str, style: MeasurementStyle, width_pt: float) -> str:
        """Minimal Typst source for one block at the requested style."""
        pitch_ratio = style.leading_pt / style.size_pt
        gap_em = max(0.0, pitch_ratio - style.font_line_height_em)
        family = json.dumps(style.font_family, ensure_ascii=False)
        body = json.dumps(text, ensure_ascii=False)
        justify = "true" if style.justify else "false"
        return (
            f"#set page(width: {width_pt:g}pt, height: auto, margin: 0pt)\n"
            f"#set text(font: ({family},), size: {style.size_pt:g}pt, "
            f'lang: "{style.language}", tracking: {style.tracking_em:g}em, '
            "overhang: true, cjk-latin-spacing: auto)\n"
            f"#set par(leading: {gap_em:.4f}em, justify: {justify}, "
            'linebreaks: "optimized")\n'
            f"#text({body})\n"
        )
