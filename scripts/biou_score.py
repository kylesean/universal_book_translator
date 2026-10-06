"""BIoU layout-fidelity score: source PDF vs translated PDF (offline calibration).

Follows the BabelDOC ACL 2026 methodology at a pragmatic fidelity level:
layout-element bounding boxes are extracted from each source page and each
translated page *using the same parser* (pypdfium2 text rows here),
coordinates are normalized by page size, elements are matched by reading
order with greedy proximity matching, and the final BIoU is averaged over
matched elements and pages.

Purpose: offline calibration objective for the bilingual-mode advisory
thresholds (``ubt/core/policy/bilingual_advisor.py``) and for comparing
render modes (inline vs alternating vs monolingual) on golden documents.
It is NOT a translation-quality metric: it scores geometry only, so pair it
with human readability review.

Scope notes (v1, documented limitations):

- Compares single-flow reflow outputs (publication/monolingual Typst PDFs)
  against the source. Alternating dual outputs (2x pages) are out of scope:
  compare their ``*_trans_stage.pdf`` instead.
- Text rows are the layout elements; vector figures contribute no rows, so
  figure-heavy pages score lower by construction (intended: figures that
  move ARE layout drift for inline interleave).
- Greedy order matching, not optimal assignment: O(n) per page, adequate
  for calibration sweeps.

Usage:
    uv run python scripts/biou_score.py source.pdf translated.pdf [--json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _page_rows(pdf_path: Path, page_idx: int) -> tuple[list[tuple[float, ...]], float, float]:
    """Extract normalized text-row rects (x0, y0, x1, y1 in 0..1) for one page."""
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(str(pdf_path))
    try:
        if not 0 <= page_idx < len(doc):
            return [], 1.0, 1.0
        page = doc[page_idx]
        try:
            width = float(page.get_width())
            height = float(page.get_height())
            textpage = page.get_textpage()
            try:
                rects = []
                for i in range(textpage.count_rects(0, -1)):
                    x0, y0, x1, y1 = (float(v) for v in textpage.get_rect(i))
                    if x1 > x0 and y1 > y0:
                        rects.append((x0 / width, y0 / height, x1 / width, y1 / height))
                return rects, width, height
            finally:
                textpage.close()
        finally:
            page.close()
    finally:
        doc.close()


def _iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _page_biou(src_rows: list[tuple[float, ...]], tgt_rows: list[tuple[float, ...]]) -> float:
    """Greedy reading-order match (sorted top-to-bottom, left-to-right)."""
    if not src_rows or not tgt_rows:
        return 0.0 if src_rows or tgt_rows else 1.0

    def key(r: tuple[float, ...]) -> tuple[float, float]:
        return (round(r[1], 3), round(r[0], 3))

    src_sorted = sorted(src_rows, key=key)
    tgt_sorted = sorted(tgt_rows, key=key)
    used = [False] * len(tgt_sorted)
    scores = []
    for s in src_sorted:
        best, best_j = 0.0, -1
        for j, t in enumerate(tgt_sorted):
            if used[j]:
                continue
            v = _iou(s, t)
            if v > best:
                best, best_j = v, j
        if best_j >= 0:
            used[best_j] = True
        scores.append(best)
    return sum(scores) / len(scores)


def biou_score(source_pdf: Path, translated_pdf: Path) -> dict:
    """Mean BIoU over the overlapping page range + per-page breakdown."""
    import pypdfium2 as pdfium

    # Context managers, not bare constructors: PdfDocument owns a native handle
    # whose close() is the documented contract, and leaving it to GC timing is
    # fragile (the handle outlives the call on non-refcounting runtimes).
    with (
        pdfium.PdfDocument(str(source_pdf)) as src_doc,
        pdfium.PdfDocument(str(translated_pdf)) as tgt_doc,
    ):
        n_src = len(src_doc)
        n_tgt = len(tgt_doc)
    pages = min(n_src, n_tgt)
    per_page = []
    for i in range(pages):
        src_rows, _, _ = _page_rows(source_pdf, i)
        tgt_rows, _, _ = _page_rows(translated_pdf, i)
        per_page.append(
            {
                "page": i + 1,
                "src_rows": len(src_rows),
                "tgt_rows": len(tgt_rows),
                "biou": round(_page_biou(src_rows, tgt_rows), 4),
            }
        )
    mean = round(sum(p["biou"] for p in per_page) / len(per_page), 4) if per_page else 0.0
    src_rows = sum(p["src_rows"] for p in per_page)
    tgt_rows = sum(p["tgt_rows"] for p in per_page)
    return {
        "source": str(source_pdf),
        "translated": str(translated_pdf),
        "source_pages": n_src,
        "translated_pages": n_tgt,
        "pages_compared": pages,
        "mean_biou": mean,
        # Row-count ratio ≈ interleave density (≈2.0 inline, ≈1.0 mono):
        # advisor-relevant calibration feature alongside BIoU.
        "src_rows_total": src_rows,
        "tgt_rows_total": tgt_rows,
        "row_ratio": round(tgt_rows / src_rows, 3) if src_rows else 0.0,
        "per_page": per_page,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BIoU layout-fidelity calibration score")
    parser.add_argument("source_pdf", type=Path)
    parser.add_argument("translated_pdf", type=Path)
    parser.add_argument("--json", action="store_true", help="emit full per-page JSON")
    args = parser.parse_args(argv)
    if not args.source_pdf.exists() or not args.translated_pdf.exists():
        print("error: input file not found", file=sys.stderr)
        return 2
    result = biou_score(args.source_pdf, args.translated_pdf)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"mean BIoU: {result['mean_biou']} over {result['pages_compared']} page(s)")
        print(f"(source {result['source_pages']}p vs translated {result['translated_pages']}p)")
        print(
            f"rows: source={result['src_rows_total']} translated={result['tgt_rows_total']} "
            f"(ratio {result['row_ratio']}; ~2.0 inline, ~1.0 mono)"
        )
        print(
            "note: absolute BIoU is pagination-sensitive for reflow outputs; "
            "use it for relative mode comparison + regression tracking, "
            "paired with human readability review."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
