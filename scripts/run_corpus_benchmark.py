#!/usr/bin/env python3
"""Automated corpus benchmark runner and quality/cost/spill dashboard generator.

Runs evaluation over documents in ``corpus/cases.json`` across:
1. Delivery Contract reconciliation (text delivery ratio, accounted ratio, violations)
2. Typst layout spill diagnostics (margins, gutters, type collapse floor)
3. Quality & Terminology metrics (MTQE scores, placeholder preservation, terminology recall)
4. Financial & Token performance (prompt/completion tokens, cache hit rate, USD cost)

Usage:
    # Dry-run rehearsal across all cases in corpus/cases.json
    uv run python scripts/run_corpus_benchmark.py --mode dry-run

    # Evaluate pre-built delivered artifacts in a directory
    uv run python scripts/run_corpus_benchmark.py --mode eval-only --artifacts-dir ~/Documents/UBT

    # Run with HTML dashboard and Markdown output
    uv run python scripts/run_corpus_benchmark.py --html-out report.html --markdown-out report.md

    # Compare against historical baseline and detect regressions
    uv run python scripts/run_corpus_benchmark.py --baseline corpus/baseline/corpus_benchmark_baseline.json

    # Save current run as new baseline
    uv run python scripts/run_corpus_benchmark.py --save-baseline corpus/baseline/corpus_benchmark_baseline.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ubt.core.benchmark.drift import compute_benchmark_drift  # noqa: E402
from ubt.core.benchmark.models import (  # noqa: E402
    BenchmarkDriftReport,
    BenchmarkMode,
    CorpusBenchmarkReport,
)
from ubt.core.benchmark.reporters import (  # noqa: E402
    render_html_dashboard,
    render_json_report,
    render_markdown_report,
    render_terminal_dashboard,
)
from ubt.core.benchmark.runner import BenchmarkRunner  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Automated corpus benchmark runner and multi-dimensional quality/cost/spill dashboard",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=ROOT / "corpus",
        help="Path to corpus directory containing cases.json (default: corpus/)",
    )
    parser.add_argument(
        "--case",
        "-c",
        dest="cases",
        action="append",
        help="Case ID to evaluate (can be passed multiple times)",
    )
    parser.add_argument(
        "--mode",
        "-m",
        choices=[m.value for m in BenchmarkMode],
        default=BenchmarkMode.DRY_RUN.value,
        help="Execution mode (default: dry-run)",
    )
    parser.add_argument(
        "--artifacts-dir",
        "-a",
        type=Path,
        default=None,
        help="Directory containing delivered artifacts (for eval-only mode or outputs)",
    )
    parser.add_argument(
        "--baseline",
        "-b",
        type=Path,
        default=None,
        help="Path to historical baseline JSON report for drift detection",
    )
    parser.add_argument(
        "--save-baseline",
        type=Path,
        default=None,
        help="Path to save current report as new baseline JSON",
    )
    parser.add_argument(
        "--tolerance-pt",
        type=float,
        default=5.0,
        help="Layout margin spill tolerance in points (default: 5.0)",
    )
    parser.add_argument(
        "--html-out",
        type=Path,
        default=None,
        help="Path to write standalone responsive HTML visual dashboard",
    )
    parser.add_argument(
        "--markdown-out",
        type=Path,
        default=None,
        help="Path to write GitHub Flavored Markdown report",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Path to write structured JSON report",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output machine-readable JSON to stdout instead of terminal dashboard",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.corpus.exists():
        msg = f"Corpus directory not found: {args.corpus}"
        if args.json:
            print(json.dumps({"status": "error", "error": msg}))
        else:
            print(f"Error: {msg}", file=sys.stderr)
        return 2

    runner = BenchmarkRunner(
        corpus_dir=args.corpus,
        mode=args.mode,
        artifacts_dir=args.artifacts_dir,
        tolerance_pt=args.tolerance_pt,
    )

    try:
        report = runner.run_all(case_ids=args.cases)
    except Exception as exc:
        msg = f"Benchmark run failed: {exc}"
        if args.json:
            print(json.dumps({"status": "error", "error": msg}))
        else:
            print(f"Error: {msg}", file=sys.stderr)
        return 2

    drift: BenchmarkDriftReport | None = None
    if args.baseline:
        if not args.baseline.exists():
            msg = f"Baseline file not found: {args.baseline}"
            if args.json:
                print(json.dumps({"status": "error", "error": msg}))
            else:
                print(f"Error: {msg}", file=sys.stderr)
            return 2
        try:
            base_report = CorpusBenchmarkReport.from_json(args.baseline.read_text(encoding="utf-8"))
            drift = compute_benchmark_drift(baseline=base_report, candidate=report)
        except Exception as exc:
            msg = f"Failed to compute baseline drift: {exc}"
            if args.json:
                print(json.dumps({"status": "error", "error": msg}))
            else:
                print(f"Error: {msg}", file=sys.stderr)
            return 2

    if args.save_baseline:
        try:
            report.save(args.save_baseline)
            if not args.json:
                print(f"Baseline saved to {args.save_baseline}")
        except Exception as exc:
            print(f"Error saving baseline: {exc}", file=sys.stderr)

    if args.html_out:
        html_content = render_html_dashboard(report, drift=drift)
        args.html_out.parent.mkdir(parents=True, exist_ok=True)
        args.html_out.write_text(html_content, encoding="utf-8")
        if not args.json:
            print(f"HTML dashboard saved to {args.html_out}")

    if args.markdown_out:
        md_content = render_markdown_report(report, drift=drift)
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(md_content, encoding="utf-8")
        if not args.json:
            print(f"Markdown report saved to {args.markdown_out}")

    if args.json_out:
        json_content = render_json_report(report, drift=drift)
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json_content, encoding="utf-8")
        if not args.json:
            print(f"JSON report saved to {args.json_out}")

    if args.json:
        print(render_json_report(report, drift=drift))
    else:
        render_terminal_dashboard(report, drift=drift)

    success = report.overall_passed
    if drift and not drift.drift_passed:
        success = False

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
