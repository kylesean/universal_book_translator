#!/usr/bin/env python3
"""Fidelity baseline harness — measure rigid-render fidelity over a corpus.

The in-pipeline fidelity ruler (ubt/adapters/pdf/render_fidelity.py) reports a
number per job; this script gives the aggregate baseline the masterplan's M2/M3
decisions are calibrated against. It takes matched ``source`` and ``artifact``
PDF pairs (either two parallel directories of the same filenames, or one
directory where artifacts end in ``.translated.pdf``) and writes a JSON summary
of non-text residual and painted coverage per document plus the corpus mean.

    uv run python scripts/fidelity_baseline.py \
        --source-dir benchmarks/src --artifact-dir benchmarks/out \
        --dpi 300 --pages 8 -o benchmarks/fidelity.baseline.json

This is an offline ops tool (not imported by the package). It uses the public
port bridge so it exercises the exact code path the pipeline uses.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


def pair_sources_artifacts(source_dir: Path, artifact_dir: Path | None) -> list[tuple[Path, Path]]:
    """Resolve matched (source, artifact) PDF pairs.

    Two modes: a parallel ``artifact_dir`` with the same file names, or a single
    directory where each ``X.pdf`` has a sibling ``X.translated.pdf``.
    """
    pairs: list[tuple[Path, Path]] = []
    if artifact_dir is not None:
        for src in sorted(source_dir.glob("*.pdf")):
            art = artifact_dir / src.name
            if art.exists():
                pairs.append((src, art))
        return pairs
    for src in sorted(source_dir.glob("*.pdf")):
        if src.name.endswith(".translated.pdf"):
            continue
        art = src.with_name(src.name[: -len(".pdf")] + ".translated.pdf")
        if art.exists():
            pairs.append((src, art))
    return pairs


def format_summary(results: list[dict[str, object]]) -> dict[str, object]:
    """Aggregate per-document fidelity dicts into a corpus summary."""
    measured = [r for r in results if r.get("pages_measured", 0)]
    residuals = [float(r["non_text_diff_ratio"]) for r in measured]  # type: ignore[arg-type]
    coverages = [float(r["masked_coverage_ratio"]) for r in measured]  # type: ignore[arg-type]
    return {
        "documents": len(results),
        "documents_measured": len(measured),
        "non_text_residual_mean": round(statistics.fmean(residuals), 6) if residuals else None,
        "non_text_residual_max": max(residuals) if residuals else None,
        "painted_coverage_mean": round(statistics.fmean(coverages), 6) if coverages else None,
        "painted_coverage_min": min(coverages) if coverages else None,
        "per_document": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rigid-render fidelity baseline.")
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--pages", type=int, default=8)
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args(argv)

    # Import the pipeline's own module so the harness measures the real path.
    from ubt.adapters.pdf.render_fidelity import compute_render_fidelity

    pairs = pair_sources_artifacts(args.source_dir, args.artifact_dir)
    if not pairs:
        print("no matched (source, artifact) PDF pairs found", file=sys.stderr)
        return 1

    results: list[dict[str, object]] = []
    failures = 0
    for src, art in pairs:
        # No IR blocks to pass: an empty mask set means "compare the whole
        # page", and compute_render_fidelity samples the first ``--pages``
        # common pages. The guard below still fails loudly if a pair yields no
        # measurable page at all, so the harness can never print a perfect
        # 0.0000 for a measurement that never ran.
        stats = compute_render_fidelity(src, art, [], dpi=args.dpi, max_pages=args.pages)
        if not stats.get("pages_measured", 0):
            print(
                f"{src.name}: ERROR — no measurable pages "
                f"({stats.get('skipped_reason') or 'unknown'})",
                file=sys.stderr,
            )
            failures += 1
            continue
        stats["document"] = src.name
        results.append(stats)
        print(
            f"{src.name}: pages={stats.get('pages_measured')} "
            f"residual={stats.get('non_text_diff_ratio'):.4f} "
            f"coverage={stats.get('masked_coverage_ratio'):.4f}"
        )

    summary = format_summary(results)
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(text)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
