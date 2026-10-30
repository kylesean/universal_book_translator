#!/usr/bin/env python3
"""Measure one block with the real Typst MeasurementService.

This is the G2 calibration tool: it compiles the requested text through the
same Typst binary the renderer uses and reports the measured height, line
count, line boxes and overflow. It spends no model tokens.

    python scripts/calibrate_measurement.py --text "..." --width 200
    python scripts/calibrate_measurement.py --json --width 120 --size 9
"""

from __future__ import annotations

import argparse
import json
import sys

from ubt.adapters.pdf.typst_measurement import TypstMeasurementService
from ubt.delivery.measurement import MeasurementError, MeasurementStyle

_DEFAULT_TEXT = (
    "这是一段用于测量真实排版高度的中文段落，宽度固定，观察自动换行与页面高度。"
    "Inline Latin text is included to exercise CJK-Latin spacing."
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default=_DEFAULT_TEXT, help="block text to measure")
    parser.add_argument("--width", type=float, required=True, help="measure in points")
    parser.add_argument("--size", type=float, default=10.0, help="font size in points")
    parser.add_argument("--leading", type=float, default=13.0, help="target baseline pitch (pt)")
    parser.add_argument("--tracking", type=float, default=0.0, help="tracking in em")
    parser.add_argument("--font", default="Noto Serif CJK SC", help="font family")
    parser.add_argument("--language", default="zh", help="Typst language tag")
    parser.add_argument("--no-justify", action="store_true", help="measure ragged-right text")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    service = TypstMeasurementService()
    style = MeasurementStyle(
        font_family=args.font,
        font_version=service.typst_version or "",
        size_pt=args.size,
        leading_pt=args.leading,
        tracking_em=args.tracking,
        language=args.language,
        justify=not args.no_justify,
    )
    try:
        measured = service.measure_block(args.text, style, args.width)
    except MeasurementError as exc:
        print(f"measurement failed: {exc}", file=sys.stderr)
        return 1
    payload = measured.model_dump(mode="json")
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(
            f"height={measured.height_pt:.2f}pt lines={measured.line_count} "
            f"overflow={measured.overflow} id={measured.measurement_id[:12]}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
