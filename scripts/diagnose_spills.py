#!/usr/bin/env python3
"""Typst layout spill and physical overflow diagnostic utility.

Deterministic geometry inspection tool for delivered PDF documents.
Identifies margin intrusions, multi-column gutter collisions, formula/table
spills, and type collapses, outputting spatial heatmaps and structured JSON.

Usage:
    uv run python scripts/diagnose_spills.py book.pdf
    uv run python scripts/diagnose_spills.py book.pdf --template academic-dual --tolerance 4.0
    uv run python scripts/diagnose_spills.py book.pdf --pages 1-5 --json
    uv run python scripts/diagnose_spills.py corpus/documents/*.pdf --summary
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console

from ubt.render.typst.layout_diagnostics import (
    DEFAULT_TOLERANCE_PT,
    PageLayoutSpec,
    diagnose_pdf_layout,
    print_rich_layout_report,
)

console = Console()


def _parse_pages(spec_str: str | None) -> list[int] | None:
    if not spec_str:
        return None
    res: set[int] = set()
    for part in spec_str.split(","):
        part = part.strip()
        if "-" in part:
            s, e = part.split("-", 1)
            start = int(s.strip()) if s.strip() else 1
            if e.strip():
                end = int(e.strip())
                res.update(range(start, end + 1))
            else:
                res.add(start)
        elif part.isdigit():
            res.add(int(part))
    return sorted(res) if res else None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose Typst layout spills and physical page overflows in PDF artifacts."
    )
    parser.add_argument("pdfs", nargs="+", type=Path, help="One or more PDF artifacts to audit.")
    parser.add_argument(
        "--template",
        "-t",
        default="auto",
        choices=["auto", "book", "academic", "academic-dual"],
        help="Layout template specification.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE_PT,
        help="Margin overflow tolerance in points (default: 5.0).",
    )
    parser.add_argument(
        "--pages",
        "-p",
        type=str,
        default=None,
        help="Subset of pages to check (e.g. '1-10', '1,3,7').",
    )
    parser.add_argument(
        "--no-heatmap",
        action="store_true",
        help="Disable spatial 2D terminal layout heatmaps.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero on warnings as well as critical errors.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output machine-readable JSON to stdout.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print compact single-line summary per document.",
    )

    args = parser.parse_args()
    target_pages = _parse_pages(args.pages)

    spec: PageLayoutSpec | None = None
    if args.template != "auto":
        spec = PageLayoutSpec.from_name(args.template, tolerance_pt=args.tolerance)

    overall_failed = False
    all_reports: list[dict] = []

    for pdf_path in args.pdfs:
        if not pdf_path.is_file():
            console.print(f"[bold red]File not found:[/] {pdf_path}", file=sys.stderr)
            overall_failed = True
            continue

        try:
            report = diagnose_pdf_layout(
                pdf_path=pdf_path,
                spec=spec,
                pages=target_pages,
                tolerance_pt=args.tolerance,
            )
        except Exception as exc:
            console.print(
                f"[bold red]Failed to diagnose {pdf_path.name}:[/] {exc}", file=sys.stderr
            )
            overall_failed = True
            continue

        if report.error_count > 0 or (args.strict and report.warning_count > 0):
            overall_failed = True

        if args.json:
            all_reports.append(report.to_dict())
        elif args.summary:
            status = (
                "[bold green]PASS[/bold green]" if report.passed else "[bold red]FAIL[/bold red]"
            )
            console.print(
                f"{status} {pdf_path.name}: {report.error_count} err, {report.warning_count} warn "
                f"across {len(report.spill_pages)}/{report.total_pages} pages"
            )
        else:
            active_spec = spec or PageLayoutSpec.book(tolerance_pt=args.tolerance)
            print_rich_layout_report(
                report=report,
                spec=active_spec,
                console=console,
                show_heatmap=not args.no_heatmap,
            )

    if args.json:
        payload = all_reports[0] if len(all_reports) == 1 else all_reports
        print(json.dumps(payload, ensure_ascii=False, indent=2))

    return 1 if overall_failed else 0


if __name__ == "__main__":
    sys.exit(main())
