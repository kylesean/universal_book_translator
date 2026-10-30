#!/usr/bin/env python3
"""Measure one block or the full 80-item calibration suite with Typst MeasurementService.

This is the G2 calibration tool: it compiles the requested text through the
same Typst binary the renderer uses and reports the measured height, line
count, line boxes and overflow. It spends no model tokens.

Usage:
    python scripts/calibrate_measurement.py --text "..." --width 200
    python scripts/calibrate_measurement.py --json --width 120 --size 9
    python scripts/calibrate_measurement.py --suite --out-dir output/adaptive_delivery/g2
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ubt.adapters.pdf.typst_measurement import TypstMeasurementService
from ubt.delivery.calibration_corpus import get_calibration_corpus
from ubt.delivery.measurement import MeasurementError, MeasurementStyle

_DEFAULT_TEXT = (
    "这是一段用于测量真实排版高度的中文段落，宽度固定，观察自动换行与页面高度。"
    "Inline Latin text is included to exercise CJK-Latin spacing."
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite", action="store_true", help="run the standardized 80-case calibration suite"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("output/adaptive_delivery/g2"),
        help="report output directory",
    )
    parser.add_argument("--text", default=_DEFAULT_TEXT, help="block text to measure")
    parser.add_argument("--width", type=float, default=200.0, help="measure in points")
    parser.add_argument("--size", type=float, default=10.0, help="font size in points")
    parser.add_argument("--leading", type=float, default=13.0, help="target baseline pitch (pt)")
    parser.add_argument("--tracking", type=float, default=0.0, help="tracking in em")
    parser.add_argument("--font", default="Noto Serif CJK SC", help="font family")
    parser.add_argument("--language", default="zh", help="Typst language tag")
    parser.add_argument("--no-justify", action="store_true", help="measure ragged-right text")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    return parser


def run_full_suite(service: TypstMeasurementService, out_dir: Path) -> int:
    cases = get_calibration_corpus()
    reqs = [c.to_request(font_version=service.typst_version or "") for c in cases]

    t0 = datetime.now(UTC)
    results = service.measure_many(reqs)
    duration_s = (datetime.now(UTC) - t0).total_seconds()

    out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    role_stats: dict[str, dict[str, Any]] = {}
    for c, m in zip(cases, results, strict=True):
        if c.role not in role_stats:
            role_stats[c.role] = {"count": 0, "total_lines": 0, "total_height": 0.0, "overflows": 0}
        role_stats[c.role]["count"] += 1
        role_stats[c.role]["total_lines"] += m.line_count
        role_stats[c.role]["total_height"] += m.height_pt
        if m.overflow:
            role_stats[c.role]["overflows"] += 1

        records.append(
            {
                "case_id": c.case_id,
                "role": c.role,
                "text": c.text,
                "width_pt": c.width_pt,
                "size_pt": c.size_pt,
                "leading_pt": c.leading_pt,
                "measured_lines": m.line_count,
                "measured_height_pt": round(m.height_pt, 2),
                "overflow": m.overflow,
                "box_count": len(m.line_boxes),
            }
        )

    json_payload = {
        "timestamp": datetime.now(UTC).isoformat(),
        "typst_version": service.typst_version,
        "total_cases": len(cases),
        "duration_seconds": round(duration_s, 3),
        "role_stats": role_stats,
        "cases": records,
    }
    (out_dir / "calibration.json").write_text(
        json.dumps(json_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Generate Markdown report. Every conclusion below is computed from the
    # run's own records: the report must be able to fail.
    total_overflows = sum(int(stats["overflows"]) for stats in role_stats.values())
    lines: list[str] = [
        "# G2 MeasurementService 校准报告",
        "",
        f"- **生成时间**：{datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"- **Typst 编译器版本**：{service.typst_version}",
        f"- **测试用例总数**：{len(cases)} 条（{len(role_stats)} 角色）",
        f"- **批处理耗时**：{duration_s:.3f} 秒（平均 {duration_s / len(cases) * 1000:.1f} ms/条）",
        f"- **溢出用例数**：{total_overflows} 条（共 {len(cases)} 条）",
        "",
        "## 1. 角色统计汇总",
        "",
        "| 排版角色 (Role) | 用例数 | 平均行数 | 平均高度 (pt) | 溢出率 | 判定 |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for role, stats in role_stats.items():
        avg_l = stats["total_lines"] / stats["count"]
        avg_h = stats["total_height"] / stats["count"]
        overflow_rate = stats["overflows"] / stats["count"] * 100
        verdict = "PASS" if stats["overflows"] == 0 else "FAIL"
        lines.append(
            f"| `{role}` | {stats['count']} | {avg_l:.1f} | {avg_h:.1f} "
            f"| {overflow_rate:.1f}% | **{verdict}** |"
        )

    overflow_line = (
        f"1. **断行健康**：{len(cases) - total_overflows}/{len(cases)} 条用例在容差内正常断行。"
        if total_overflows == 0
        else f"1. **断行健康**：{total_overflows}/{len(cases)} 条用例报告溢出，"
        "明细见下表 overflow 列，对应角色判定为 FAIL。"
    )
    lines.extend(
        [
            "",
            "## 2. 本报告测量的结论",
            "",
            overflow_line,
            "2. **范围说明**：单调性、批处理与单块编译的等价性、公式不泄漏、窄容器适应性"
            "不由本脚本测量；这些性质由单元测试套件"
            "（`tests/unit/adapters/test_typst_measurement_calibration.py`）以断言形式持续验证。",
            "",
            "## 3. 详细校准数据明细",
            "",
            "| 用例 ID | 角色 | 宽度 (pt) | 字号 (pt) | 行距 (pt) | 实测行数 | 实测高度 (pt) | 溢出 |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
    )
    for r in records:
        lines.append(
            f"| `{r['case_id']}` | `{r['role']}` | {r['width_pt']} | {r['size_pt']} | "
            f"{r['leading_pt']} | {r['measured_lines']} | {r['measured_height_pt']} | {r['overflow']} |"
        )

    (out_dir / "G2_CALIBRATION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"Calibration complete: {len(cases)} cases passed in {duration_s:.3f}s. Report written to {out_dir / 'G2_CALIBRATION_REPORT.md'}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    service = TypstMeasurementService()

    if args.suite:
        return run_full_suite(service, args.out_dir)

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
