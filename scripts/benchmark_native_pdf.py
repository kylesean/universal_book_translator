#!/usr/bin/env python3
"""Measure the native PDF codec's inspect/extract cost on real documents.

The layout pass (144dpi raster + PP-DocLayout ONNX) is the dominant cost; a
whole-book timing is the only honest way to decide whether a caching/parallelism
change is worth its risk. Prints one row per document, and optionally JSON.

Usage:
    uv run python scripts/benchmark_native_pdf.py corpus/documents/*.pdf
    uv run python scripts/benchmark_native_pdf.py --json corpus/documents/*.pdf
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

from ubt.codecs.pdf.codec import NativePDFCodec


def _measure(path: Path) -> dict[str, Any]:
    codec = NativePDFCodec()
    started = time.perf_counter()
    profile = codec.inspect(path)
    inspect_s = time.perf_counter() - started
    started = time.perf_counter()
    document = codec.extract_detailed(path)
    extract_s = time.perf_counter() - started
    pages = profile.page_count or 0
    return {
        "path": str(path),
        "pages": pages,
        "units": len(document.units),
        "inspect_s": round(inspect_s, 3),
        "extract_s": round(extract_s, 3),
        "extract_s_per_page": round(extract_s / pages, 3) if pages else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", nargs="+", type=Path)
    parser.add_argument("--json", action="store_true", help="emit machine-readable rows")
    parser.add_argument(
        "--timing",
        action="store_true",
        help="log per-page analyze/render/detect/fuse breakdown",
    )
    args = parser.parse_args()
    if args.timing:
        logging.basicConfig(level=logging.DEBUG, format="%(message)s")

    rows = [_measure(path) for path in args.pdf if path.is_file()]
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return 0
    if not rows:
        print("no readable PDFs given")
        return 1
    print(f"{'document':40} {'pages':>6} {'units':>7} {'inspect':>9} {'extract':>9} {'s/page':>8}")
    for row in rows:
        print(
            f"{Path(row['path']).name[:40]:40} {row['pages']:>6} {row['units']:>7} "
            f"{row['inspect_s']:>8.3f}s {row['extract_s']:>8.3f}s "
            f"{(row['extract_s_per_page'] or 0):>7.3f}s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
