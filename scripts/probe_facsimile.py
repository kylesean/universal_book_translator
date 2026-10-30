#!/usr/bin/env python3
"""Cheap feasibility prefilter for the `faithful` delivery contract.

Answers, for a real PDF, one question that gates the `faithful` contract:
**can this page's translation be set at one uniform size inside the source
geometry, and how big can that size be?** It is a prefilter, not the delivery
acceptance criterion: the v3.0 PRD requires artifact-level measurement before
a `faithful` page may ship (docs/design/adaptive_delivery_prd.md §7.3, §8.1).

That question has a clean answer because English→Chinese is *compression*, not
expansion: one Han glyph costs ~3.5 Latin characters of source text but is one
em wide, so the target needs roughly half the source's area. Measured on the
five corpus documents, 91–100% of paragraphs fit their source box at the source
size. This probe is therefore a **prefilter**: it can rule a page out cheaply,
but the production solver must still measure real Typst layout before it
decides `faithful` is deliverable (see
``docs/design/adaptive_delivery_prd.md`` §7.3, §8.1, §10.2).

Two numbers per page come out of the probe; they are prefilter evidence, not
the delivery acceptance criterion. The probe can measure either of two
constraint models (``--mode``), and the difference between them is the measured
value of the faithful engine's "vertical glue chain":

``pinned``  Every paragraph's translation must fit its own source box. Blank
            space between paragraphs is untouchable. This was v1.0's model.
``glue``    A paragraph's top edge is fixed and it may grow downward into the
            gap before the next paragraph, but it may never push that
            paragraph. This is the report's vertical glue chain, and it is
            *stricter* than ``pinned``: it refuses a page whose source
            paragraphs already overlap (a float or footnote intruding on the
            text column), where ``pinned`` happily overflows into it.

``size_pt``      The largest single body size at which *every* paragraph on the
                 page still fits its source box. One size per page is the whole
                 point: per-paragraph scaling is what makes BabelDOC's output
                 look like a ransom note (its own docs admit "No global page
                 information consideration, inconsistent text sizes").
``fits``         False when even the readability floor cannot hold the page;
                 such a page is ineligible for `faithful` and must route to
                 `readable`, never ship at 4pt.

Nothing here renders, translates, or calls a model — it is pure geometry over
the source PDF, so it is safe to run on any document before paying for a job.

Usage:
    uv run python scripts/probe_facsimile.py book.pdf
    uv run python scripts/probe_facsimile.py book.pdf --pages 1-12 --json
    uv run python scripts/probe_facsimile.py corpus/documents/*.pdf --summary

Calibration
-----------
``--en-per-zh`` and ``--en-char-width`` are the two measured constants the
whole estimate rests on. Defaults come from ``chapter-3.pdf`` (a 9pt Elsevier
textbook chapter) cross-checked against three hand-translated EN→ZH samples:

    EN 189 chars → ZH 50 chars   1 Han = 3.78 Latin chars
    EN 158 chars → ZH 52 chars   1 Han = 3.04
    EN 143 chars → ZH 38 chars   1 Han = 3.76
                                 mean 3.53

Re-calibrate them per corpus rather than trusting the default blindly: a
German target, or a Chinese target of a terse English source, moves both.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

#: One Han glyph's cost in Latin characters, measured on real EN→ZH pairs.
#: Used to convert a paragraph's source character count into target glyphs.
DEFAULT_EN_PER_ZH = 3.53
#: Mean advance width of a Latin character at the probe's reference size.
#: ``chapter-3.pdf`` measures 4.07pt at 9pt over 38 body lines.
DEFAULT_EN_CHAR_WIDTH_PT = 4.07
#: Readability floor. ``visual_gate`` fails a page closed below 4pt median
#: (``type_illegible``); 6pt is the size at which a reflowed table stops being
#: legible in this codebase (``_TABLE_MIN_TEXT_SCALE``), so it is the honest
#: floor for a channel that promises the *original* reading experience.
FLOOR_PT = 6.0
#: Ceiling as a multiple of the page's source body size. Chinese only needs
#: ~56% of the source area, so most pages can grow; capping at 1.4x keeps the
#: result recognisably the same book rather than a large-print edition.
CEILING_RATIO = 1.4
#: A line whose box is taller than this is not body prose (it is a rule, a
#: vertical sidebar, a decorative glyph). Mirrors ``textgeom``'s own filter.
MAX_LINE_H_PT = 22.0
MIN_LINE_W_PT = 50.0
#: A right-hand line only makes the page two-column if this share of lines
#: lives there; anything less is a table or a floating caption.
TWO_COLUMN_SHARE = 0.25
#: Line-gap multiple that separates paragraphs. Wider than the naive 1.6x
#: because reviewer line numbers and reference numerals sit *between* body
#: lines and are filtered out as too narrow — the surviving pair then reports
#: the gap across the excluded line, which would split one paragraph into two.
PARA_GAP_FACTOR = 2.2
#: Tolerance for float comparisons on point geometry.
TOL_PT = 0.5
#: The two constraint models. ``pinned`` keeps each paragraph inside its own
#: box; ``glue`` lets it eat the gap before the next paragraph but never push
#: that paragraph. See the module docstring for why glue is the stricter one.
MODES = ("pinned", "glue")


@dataclass(slots=True)
class PageProbe:
    """One page's verdict."""

    page: int
    page_w: float
    page_h: float
    columns: int
    source_size_pt: float
    paragraphs: int
    #: Paragraphs that fit their own box at the *source* size (no shrinking).
    fits_at_source: int
    #: Largest uniform size every paragraph on this page still fits.
    uniform_size_pt: float
    #: False when even ``FLOOR_PT`` overflows: this page must go to reflow.
    fits: bool
    #: Area the target needs over the area the source occupied. < 1 means the
    #: translation is more compact than the original; > 1 means it overflows.
    fill_ratio: float
    #: True when the source page's own paragraphs overlap (a float or footnote
    #: intruding on the text column). No size can reproduce that at 1:1, so the
    #: caller routes the page to reflow.
    source_overlap: bool = False
    #: Which constraint model produced ``uniform_size_pt``.
    mode: str = "glue"

    @property
    def fill_pct(self) -> float:
        return 100.0 * self.fill_ratio


@dataclass(slots=True)
class DocumentProbe:
    path: str
    pages: int
    probed: int
    source_size_pt: float
    uniform_size_pt: float
    uniform_size_min_pt: float
    pages_at_floor: int
    pages_unfittable: int
    #: Pages whose source geometry already overlaps — not a layout failure, an
    #: input the channel cannot serve. Excluded from ``pages_unfittable``.
    pages_source_overlap: int
    paragraphs: int
    fits_at_source_pct: float
    per_page: list[PageProbe]


def _paragraphs(lines: Sequence[Any]) -> list[list[Any]]:
    """Group a column's lines into paragraphs by vertical gap.

    Lines arrive sorted top-down (``extract_lines`` guarantees reading order),
    so a gap larger than the body leading starts a new paragraph. A gap of zero
    or less means two lines share a baseline (a row of a table, a header line
    with its rule) and also starts a new one — fusing those would invent a
    paragraph that does not exist.
    """
    if not lines:
        return []
    ordered = sorted(lines, key=lambda line: -line.rect[1])
    groups: list[list[Any]] = [[ordered[0]]]
    for prev, nxt in zip(ordered, ordered[1:], strict=False):
        gap = prev.rect[1] - nxt.rect[1]
        if gap > PARA_GAP_FACTOR * 12.0 or gap < 0.5:
            groups.append([nxt])
        else:
            groups[-1].append(nxt)
    return groups


def _body_lines(lines: Sequence[Any], page_w: float) -> list[Any]:
    return [
        line
        for line in lines
        if (line.rect[2] - line.rect[0]) > MIN_LINE_W_PT
        and 4.0 < (line.rect[3] - line.rect[1]) < MAX_LINE_H_PT
    ]


def _source_size(lines: Sequence[Any]) -> float:
    """The page's body size, from its most common ink height.

    ``extract_char_styles`` reports ``font_size`` as 1.0 on some producers
    (the reference textbook among them), so the glyph box height is the only
    signal that survives every producer. The mode of the ink heights, scaled by
    the 9/8.9 ratio measured on the reference page, recovers the nominal size.
    """
    heights = [round(line.rect[3] - line.rect[1], 1) for line in lines if line.text.strip()]
    if not heights:
        return 9.0
    ink = float(statistics.mode(heights))
    return max(6.0, min(14.0, ink * 9.0 / 8.9))


def _target_glyphs(paragraph: Sequence[Any], en_per_zh: float) -> float:
    return sum(len(line.text) for line in paragraph) / en_per_zh


def _needed_lines(paragraph: Sequence[Any], size_pt: float, en_per_zh: float) -> int:
    """Target lines this paragraph occupies at ``size_pt``.

    A Han glyph is one em wide, so the line holds ``width / size`` glyphs. The
    estimate is deliberately a plain division rather than a real typesetter
    measurement: it only has to be monotone in ``size_pt`` for the binary search
    to be correct, and it errs conservative because it ignores that justification
    can squeeze an extra glyph into a line.
    """
    width = max(line.rect[2] - line.rect[0] for line in paragraph)
    if width <= 0:
        return len(paragraph)
    per_line = max(1.0, float(width) / size_pt)
    return max(1, math.ceil(_target_glyphs(paragraph, en_per_zh) / per_line))


def _leading(lines: Sequence[Any], source_size: float) -> float:
    """The page's body leading, from its most common line-to-line pitch."""
    ordered = sorted(lines, key=lambda line: -line.rect[1])
    pitches = [
        round(ordered[i].rect[1] - ordered[i + 1].rect[1], 1) for i in range(len(ordered) - 1)
    ]
    plausible = [pitch for pitch in pitches if 6.0 < pitch < 40.0]
    return statistics.mode(plausible) if plausible else source_size * 1.333


def probe_page(
    pdf_path: Path,
    page_no: int,
    *,
    en_per_zh: float = DEFAULT_EN_PER_ZH,
    floor_pt: float = FLOOR_PT,
    ceiling_ratio: float = CEILING_RATIO,
    mode: str = "glue",
) -> PageProbe | None:
    """Probe one page. ``None`` when the page carries no body text.

    ``mode`` selects the constraint model (see the module docstring):
    ``pinned`` bounds each paragraph by its own box, ``glue`` bounds it by the
    next paragraph's top edge instead.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
    from ubt.adapters.pdf.textgeom import extract_lines

    try:
        lines, (page_w, page_h) = extract_lines(pdf_path, page_no)
    except Exception:
        return None

    # Page furniture (running head, folio) is not body text: it keeps its own
    # box in the background layer and must not constrain the body's budget.
    body = [
        line
        for line in _body_lines(lines, page_w)
        if page_h * 0.08 < line.rect[1] and line.rect[3] < page_h * 0.92
    ]
    if len(body) < 5:
        return None

    source_size = _source_size(body)
    source_leading = _leading(body, source_size)
    # Leading tracks the size, so a paragraph's height at size S is
    # ``lines(S) * (leading/size) * S``. Shrinking the leading instead would
    # break the "it looks like the same book" promise.
    em = source_leading / source_size

    right = [line for line in body if line.rect[0] >= page_w * 0.55]
    columns = 2 if len(right) > len(body) * TWO_COLUMN_SHARE else 1
    groups = (
        [[line for line in body if line.rect[0] < page_w * 0.55], right] if columns == 2 else [body]
    )

    # Per column: its paragraphs, and the lowest ink in that column. A column's
    # paragraphs constrain only each other — a two-column page's left column
    # must not be bounded by where the right column happens to start.
    columns_paras: list[tuple[list[dict[str, Any]], float]] = []
    for group in groups:
        if len(group) < 2:
            continue
        paras = _paragraphs(group)
        records = []
        for para in paras:
            top = max(line.rect[3] for line in para)
            ink_bottom = min(line.rect[1] for line in para)
            # The box's bottom is the *line box* bottom, not the last ink
            # pixel's: a single-line paragraph's ink box is ~8pt tall inside a
            # 12pt line, so using the ink bottom as the box bottom would credit
            # the paragraph with 4pt it does not have (measured: chapter-3 p3,
            # where that turned a fittable page into a 6.4pt page).
            line_box_bottom = top - len(para) * source_leading
            records.append(
                {
                    "top": top,
                    "bot": min(ink_bottom, line_box_bottom),
                    "w": max(line.rect[2] - line.rect[0] for line in para),
                    "n": len(para),
                    "glyphs": _target_glyphs(para, en_per_zh),
                }
            )
        columns_paras.append((records, min(line.rect[1] for line in group)))
    if not columns_paras:
        return None

    # Source geometry that already overlaps cannot be reproduced at 1:1 by any
    # size: a float or footnote is sitting on the text column. Report it so the
    # caller routes the page to reflow instead of trusting the size below.
    overlapped = any(
        records[i]["bot"] < records[i + 1]["top"] - TOL_PT
        for records, _ in columns_paras
        for i in range(len(records) - 1)
    )

    def needed(record: dict[str, Any], size: float) -> float:
        width = float(record["w"])
        if width <= 0:
            return float(record["n"]) * em * size
        per_line = max(1.0, width / size)
        lines_needed = max(1, math.ceil(float(record["glyphs"]) / per_line))
        return float(lines_needed) * em * size

    def fits(size: float) -> bool:
        if mode == "pinned":
            return all(
                needed(rec, size) <= rec["n"] * source_leading + TOL_PT
                for records, _ in columns_paras
                for rec in records
            )
        # glue: the top edge is fixed; growth may consume the gap to the next
        # paragraph but must never reach it. The last paragraph of a column is
        # bounded by *its own* line-box bottom, not by the column's lowest ink
        # — a single-line paragraph's ink box stops ~3pt above its line box, so
        # using the ink bottom there would make glue stricter than pinned
        # (measured: chapter-3 p3, 9.4pt under pinned vs 6.4pt under glue).
        for records, _column_bottom in columns_paras:
            for index, rec in enumerate(records):
                limit = rec["bot"]
                for later in records[index + 1 :]:
                    if later["top"] < rec["top"] - TOL_PT:
                        limit = later["top"]
                        break
                if rec["top"] - needed(rec, size) < limit - TOL_PT:
                    return False
        return True

    # Monotone: a smaller size holds more glyphs per line, so it never needs
    # more lines. Binary search is therefore exact, not an approximation.
    ceiling = min(12.0, source_size * ceiling_ratio)
    lo, hi = floor_pt, ceiling
    best = floor_pt
    if fits(hi):
        best = hi
    elif fits(floor_pt):
        for _ in range(26):
            mid = (lo + hi) / 2.0
            if fits(mid):
                best = mid
                lo = mid
            else:
                hi = mid
    else:
        best = floor_pt

    paragraphs = [rec for records, _ in columns_paras for rec in records]
    # Area the target wants over the area the source used, at the source size.
    # This is the "how much room is left over" number the v2.1 design doc
    # quoted. It is a prefilter statistic, not a delivery acceptance criterion.
    src_area = sum(rec["w"] * rec["n"] * source_leading for rec in paragraphs)
    target_area = sum(rec["w"] * needed(rec, source_size) for rec in paragraphs)
    fill = target_area / src_area if src_area > 0 else 0.0

    return PageProbe(
        page=page_no,
        page_w=round(page_w, 1),
        page_h=round(page_h, 1),
        columns=columns,
        source_size_pt=round(source_size, 1),
        paragraphs=len(paragraphs),
        fits_at_source=sum(
            1
            for rec in paragraphs
            if needed(rec, source_size) <= rec["n"] * source_leading + TOL_PT
        ),
        uniform_size_pt=round(best, 1),
        fits=fits(floor_pt),
        fill_ratio=round(fill, 3),
        source_overlap=overlapped,
        mode=mode,
    )


def probe_document(
    pdf_path: Path,
    *,
    pages: Sequence[int] | None = None,
    en_per_zh: float = DEFAULT_EN_PER_ZH,
    en_char_width_pt: float = DEFAULT_EN_CHAR_WIDTH_PT,
    floor_pt: float = FLOOR_PT,
    ceiling_ratio: float = CEILING_RATIO,
    mode: str = "glue",
) -> DocumentProbe:
    from ubt.adapters.pdf import pdf_struct

    total = pdf_struct.page_count(pdf_path)
    wanted = list(pages) if pages else list(range(1, total + 1))
    probed: list[PageProbe] = []
    for page_no in wanted:
        if not 1 <= page_no <= total:
            continue
        result = probe_page(
            pdf_path,
            page_no,
            en_per_zh=en_per_zh,
            floor_pt=floor_pt,
            ceiling_ratio=ceiling_ratio,
            mode=mode,
        )
        if result is not None:
            probed.append(result)
    if not probed:
        return DocumentProbe(
            path=str(pdf_path),
            pages=total,
            probed=0,
            source_size_pt=0.0,
            uniform_size_pt=0.0,
            uniform_size_min_pt=0.0,
            pages_at_floor=0,
            pages_unfittable=0,
            pages_source_overlap=0,
            paragraphs=0,
            fits_at_source_pct=0.0,
            per_page=[],
        )

    sizes = [page.uniform_size_pt for page in probed]
    paragraphs = sum(page.paragraphs for page in probed)
    fits_at_source = sum(page.fits_at_source for page in probed)
    return DocumentProbe(
        path=str(pdf_path),
        pages=total,
        probed=len(probed),
        source_size_pt=round(statistics.median([page.source_size_pt for page in probed]), 1),
        uniform_size_pt=round(statistics.median(sizes), 1),
        uniform_size_min_pt=round(min(sizes), 1),
        pages_at_floor=sum(1 for size in sizes if size <= floor_pt + 0.05),
        pages_unfittable=sum(1 for page in probed if not page.fits and not page.source_overlap),
        pages_source_overlap=sum(1 for page in probed if page.source_overlap),
        paragraphs=paragraphs,
        fits_at_source_pct=round(100.0 * fits_at_source / paragraphs, 1) if paragraphs else 0.0,
        per_page=probed,
    )


def _parse_pages(spec: str) -> list[int]:
    out: set[int] = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            first, last = chunk.split("-", 1)
            out.update(range(int(first), int(last) + 1))
        else:
            out.add(int(chunk))
    return sorted(out)


def _print_page_table(result: DocumentProbe) -> None:
    print(
        f"{'page':>4} {'cols':>4} {'src pt':>7} {'paras':>6} "
        f"{'fit@src':>8} {'uniform':>8} {'fill%':>7} {'verdict':>10}"
    )
    for page in result.per_page:
        if page.source_overlap:
            verdict = "→ reflow*"
        elif page.fits:
            verdict = "facsimile"
        else:
            verdict = "→ reflow"
        print(
            f"{page.page:>4} {page.columns:>4} {page.source_size_pt:>7.1f} "
            f"{page.paragraphs:>6} {page.fits_at_source:>8} "
            f"{page.uniform_size_pt:>8.1f} {page.fill_pct:>6.0f}% {verdict:>10}"
        )
    if any(page.source_overlap for page in result.per_page):
        print("  * source page geometry already overlaps (float/footnote on the text column)")


def _print_summary(result: DocumentProbe) -> None:
    print(f"\n{result.path}")
    print(f"  pages probed          {result.probed} / {result.pages}")
    print(f"  source body size      {result.source_size_pt}pt")
    print(f"  paragraphs            {result.paragraphs}")
    print(f"  fit at source size    {result.fits_at_source_pct}%")
    print(f"  uniform size (median) {result.uniform_size_pt}pt")
    print(f"  uniform size (min)    {result.uniform_size_min_pt}pt")
    print(f"  pages at floor        {result.pages_at_floor}")
    print(f"  pages unfittable      {result.pages_unfittable}")
    print(f"  pages source-overlap  {result.pages_source_overlap}  (cannot be 1:1 by any size)")
    if result.pages_unfittable == 0:
        verdict = "facsimile viable for every page it can serve"
    else:
        verdict = f"{result.pages_unfittable} page(s) must fall back to reflow"
    print(f"  verdict               {verdict}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("pdfs", nargs="+", type=Path, help="source PDF(s) to probe")
    parser.add_argument(
        "--pages",
        default=None,
        help="page selection, e.g. '1-12' or '1,3,7' (default: all pages)",
    )
    parser.add_argument(
        "--en-per-zh",
        type=float,
        default=DEFAULT_EN_PER_ZH,
        help=f"Latin characters per Han glyph (default {DEFAULT_EN_PER_ZH})",
    )
    parser.add_argument(
        "--floor",
        type=float,
        default=FLOOR_PT,
        help=f"readability floor in pt (default {FLOOR_PT})",
    )
    parser.add_argument(
        "--ceiling-ratio",
        type=float,
        default=CEILING_RATIO,
        help=f"max size as a multiple of the source size (default {CEILING_RATIO})",
    )
    parser.add_argument(
        "--mode",
        choices=MODES,
        default="glue",
        help=(
            "constraint model: 'pinned' bounds each paragraph by its own box, "
            "'glue' lets it eat the gap before the next paragraph (default: glue)"
        ),
    )
    parser.add_argument("--json", action="store_true", help="emit JSON instead of tables")
    parser.add_argument(
        "--summary",
        action="store_true",
        help="print only the per-document summary (skip the per-page table)",
    )
    args = parser.parse_args(argv)

    pages = _parse_pages(args.pages) if args.pages else None
    results: list[DocumentProbe] = []
    for path in args.pdfs:
        if not path.exists():
            print(f"probe_facsimile: no such file: {path}", file=sys.stderr)
            return 2
        results.append(
            probe_document(
                path,
                pages=pages,
                en_per_zh=args.en_per_zh,
                floor_pt=args.floor,
                ceiling_ratio=args.ceiling_ratio,
                mode=args.mode,
            )
        )

    if args.json:
        payload: dict[str, Any] = {
            "en_per_zh": args.en_per_zh,
            "floor_pt": args.floor,
            "ceiling_ratio": args.ceiling_ratio,
            "mode": args.mode,
            "documents": [
                {**asdict(result), "per_page": [asdict(page) for page in result.per_page]}
                for result in results
            ],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    for index, result in enumerate(results):
        if not args.summary:
            if index:
                print()
            _print_page_table(result)
        _print_summary(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
