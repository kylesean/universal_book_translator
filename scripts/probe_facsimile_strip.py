#!/usr/bin/env python3
"""Regression probe for the facsimile background-strip plan.

Drives the *production* helpers — :func:`ubt.adapters.pdf.docling_render._facsimile_strip_plan`
and :func:`ubt.adapters.pdf.strip_pdf_text.strip_text_streams` — over a real PDF
parsed by the real adapter, then answers the three questions the background layer
must satisfy. Every question is measured against the source text layer, not
against an assumption:

1. **No English leaks.** No source line belonging to a *translated* block may
   survive the strip. A survivor keeps its ink underneath the translation, so the
   reader sees both languages overlapping.
2. **Nothing kept is lost.** Every source line inside a block the pipeline chose
   *not* to translate (formula, figure, chrome) must still be there.
3. **Nothing unowned is touched.** Text that no IR block covers — a folio, a
   running head Docling discarded, an axis label drawn outside a figure's bbox —
   has no erase rect over it and must survive.

Why this exists: the first strip plan kept a full-width furniture band at
``[0, 0.08·H]`` and ``[0.92·H, H]`` on every page. It looked conservative and was
in fact harmful — body prose routinely reaches into those bands, so the band
vetoed real erase rects and leaked the source line under its translation
(measured: 2 lines per page on ``chapter-3.pdf`` p1–p3). A probe that measures
leaks *and* losses *and* unowned survival catches that class of mistake; a probe
that only counts characters does not, because the totals barely move.

A second, worse variant of the same mistake leaked the *title* of every arXiv
paper: a show operator's pen sits slightly left of the ink it paints, so a
15.9pt title's pen at ``x = 92.2`` fell 0.6pt outside its erase box at
``x0 = 92.9`` and the whole line survived. Both are caught here, and
``--self-test`` re-injects the defect to prove the metric still has teeth.

Usage:
    uv run python scripts/probe_facsimile_strip.py book.pdf
    uv run python scripts/probe_facsimile_strip.py corpus/documents/*.pdf --pages 1-6
    uv run python scripts/probe_facsimile_strip.py book.pdf --strict
    uv run python scripts/probe_facsimile_strip.py book.pdf --self-test

``--strict`` exits non-zero on any leak, loss or unowned erase, for CI.
``--self-test`` exits non-zero if the injected defect produces *no* leak, i.e. if
this probe has gone blind.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import tempfile
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ubt.adapters.pdf import pdf_struct  # noqa: E402
from ubt.adapters.pdf.docling_render import _facsimile_strip_plan  # noqa: E402
from ubt.adapters.pdf.strip_pdf_text import strip_text_streams  # noqa: E402
from ubt.adapters.pdf.textgeom import extract_char_styles, extract_lines  # noqa: E402
from ubt.core.ir.models import BlockType  # noqa: E402


def _key(text: str) -> str:
    """Alphanumeric-only identity for a line.

    Punctuation is dropped because PDF extraction splits the same line
    differently across producers (``(3.1)`` versus ``( 3 . 1 )``); keeping it
    would report a spurious loss. Letters and digits are kept because they are
    what a reader would notice missing.
    """
    return "".join(ch for ch in text if ch.isalnum())


def _chars_in(chars: list[Any], bbox: Any, exclude: Sequence[Any] = ()) -> Counter[str]:
    """Alnum character multiset of every char whose centre falls in ``bbox``.

    Character multisets, not line identities, because the extractor re-segments
    freely: a formula that reads ``∂2ψðx, yÞ`` before the rewrite can come back
    as ``∂ ψðx, yÞ`` after it with every glyph still on the page. Comparing line
    strings would call that a loss on every page of every maths-heavy document
    and bury the losses that matter.

    Measured at *character* granularity, not line granularity. The first version
    of this helper took whole ``extract_lines`` records and tested their
    *centres*, which is stable only while pdfium groups the same glyphs into the
    same line. It does not: on ``2608.25512v1.pdf`` p2 a table-of-contents row
    leaves as ``'4'`` (a 5pt-wide box at x≈519) and comes back folded into a
    leader-dot run spanning x=155.9→524.1, so the centre moved 185pt and every
    such digit was reported as erased. The raw pdfium char box, by contrast, is
    identical before and after (``('4', 519.0, 690.0, 524.1, 697.5)`` both ways)
    — which follows from the stripper's own contract: it blanks a show operator's
    *payload* and never touches its operator, so no glyph can move.

    ``exclude`` drops chars whose centre falls in one of the given rects. The
    probe uses it to carve out the *overlap zone* — the area a keep block and a
    translated block both claim. Inside that zone the reader sees the redrawn
    translation, so neither a surviving source glyph (looks like a leak) nor a
    blanked one (looks like a loss) is attributable to the background layer.
    Outside it, both verdicts are exact.
    """
    out: Counter[str] = Counter()
    for char in chars:
        cx = (char.rect[0] + char.rect[2]) / 2
        cy = (char.rect[1] + char.rect[3]) / 2
        if not (bbox.x0 - 1 <= cx <= bbox.x1 + 1 and bbox.y0 - 1 <= cy <= bbox.y1 + 1):
            continue
        if any(r.x0 - 1 <= cx <= r.x1 + 1 and r.y0 - 1 <= cy <= r.y1 + 1 for r in exclude):
            continue
        out.update(_key(char.text))
    return out


#: Two boxes this close in both axes are the same glyph, re-read. The stripper
#: never moves ink, so anything wider would start excusing real movement.
_SAME_GLYPH_PT: float = 0.5


def _survivors(chars: list[Any]) -> dict[str, list[tuple[float, float]]]:
    """Index surviving glyphs by character, with their box corners.

    Keyed by character rather than by position so a lookup is a dict hit plus a
    short scan of that character's own occurrences (typically a few dozen on a
    page), not a page-wide cross product.
    """
    out: dict[str, list[tuple[float, float]]] = {}
    for char in chars:
        key = _key(char.text)
        if not key:
            continue
        out.setdefault(key, []).append((char.rect[0], char.rect[1]))
    return out


def _survives(index: dict[str, list[tuple[float, float]]], char: Any) -> bool:
    """Whether this source glyph is still on the page, at the same position.

    Position is the identity here, not a convenience: the stripper's guarantee is
    that surviving text does not move by a single point (it clears payloads, it
    never deletes the operators that advance the text matrix). So a glyph that
    reappears 200pt away is not the same glyph, and treating "the character
    exists somewhere" as survival is the defect this probe was rewritten to
    avoid — it reported 41 phantom leaks on ``chapter-3.pdf``.
    """
    key = _key(char.text)
    if not key:
        return True
    x, y = char.rect[0], char.rect[1]
    for ax, ay in index.get(key, ()):
        if abs(ax - x) <= _SAME_GLYPH_PT and abs(ay - y) <= _SAME_GLYPH_PT:
            return True
    return False


def _overlaps_rect(a: Any, b: Any) -> bool:
    """Whether two axis-aligned rects share any area.

    Touching edges do not count: a keep rect authored flush against a translated
    block's box is the normal case for adjacent blocks, and counting a zero-area
    touch as protection would hide every leak along a block boundary.
    """
    return a.x0 < b.x1 and b.x0 < a.x1 and a.y0 < b.y1 and b.y0 < a.y1


def _intersections(rects: Sequence[Any], others: Sequence[Any]) -> list[Any]:
    """The areas two rect sets both claim.

    A keep block and a translated block routinely overlap — a figure's box
    contains the caption that gets redrawn, and on ``chapter-3.pdf`` p1 the
    display ``3`` sits inside both ``img_0002`` and the translated heading
    ``b0003``. The reader sees the *translation* there, so the probe cannot
    attribute a surviving glyph (looks like a leak) or a blanked one (looks like
    a loss) in that zone to the background layer. It carves the zone out of both
    verdicts instead of guessing, which is the difference between a probe that
    reports real defects and one that reports 41 of them.
    """
    out: list[Any] = []
    for a in rects:
        for b in others:
            x0, y0 = max(a.x0, b.x0), max(a.y0, b.y0)
            x1, y1 = min(a.x1, b.x1), min(a.y1, b.y1)
            if x1 > x0 and y1 > y0:
                out.append(SimpleNamespace(x0=x0, y0=y0, x1=x1, y1=y1))
    return out


def _keeps_source(block: Any) -> bool:
    return bool(block.skip_translate) or block.block_type in (
        BlockType.FORMULA,
        BlockType.IMAGE,
    )


def _covered(x: float, y: float, blocks: list[Any]) -> bool:
    """Whether any block's box claims the point ``(x, y)``.

    Ownership is ambiguous where a figure's box overlaps a prose block's box —
    measured on ``chapter-3.pdf`` p1, where the display ``3`` sits inside both
    the figure's box and the translated heading that redraws it. The stripper
    resolves that ambiguity as *keep wins* (see
    :func:`ubt.adapters.pdf.strip_pdf_text._keep_point`), so a glyph in the
    overlap is blanked only when no keep rect covers its pen position.

    That is why this helper answers only *"is this point owned at all"* and is
    used for the unowned check alone. It is deliberately **not** used to decide
    leaks or losses: the stripper classifies by a glyph's pen position, and a
    pen position cannot be reconstructed from the text layer. Measured on
    ``chapter-3.pdf`` p2, 181 show-op positions face 48 textgeom lines, only 16
    pairs agree within 0.6 pt, and 133 show ops have no textgeom line within
    2 pt. A point-in-box ownership test therefore misattributes every run that
    straddles a box boundary — which is most of them.
    """
    for block in blocks:
        bbox = block.bbox
        if bbox is None:
            continue
        if bbox.x0 - 1 <= x <= bbox.x1 + 1 and bbox.y0 - 1 <= y <= bbox.y1 + 1:
            return True
    return False


def _legacy_furniture_bands(
    blocks_by_page: dict[int, list[Any]], pages: list[int]
) -> dict[int, list[Any]]:
    """A keep rect over the largest *translated* block — the rejected design's bug.

    The first strip plan kept a full-width ``[0, 0.08·H]`` / ``[0.92·H, H]`` band
    on every page. The defect that mattered was not the band's geometry but its
    *logic*: a keep rect that covers a block which is being translated vetoes that
    block's erase rect, so the source text survives underneath its own
    translation. This reproduces that logic directly — protect the biggest
    translated block on each page — which is deterministic on any document with
    text, whereas the literal bands only leak on documents whose prose happens to
    reach into the margins (``2606.23344v1.pdf`` does not, so a band-based
    self-test would pass vacuously there).

    If the probe reports no leak with this injected, its leak metric is blind and
    the clean result it reports on the real plan means nothing.
    """
    from ubt.adapters.pdf.strip_pdf_text import BoxRect

    bands: dict[int, list[Any]] = {}
    for pno in pages:
        translated = [b for b in blocks_by_page.get(pno, []) if not _keeps_source(b)]
        if not translated:
            continue
        biggest = max(
            translated,
            key=lambda b: (b.bbox.x1 - b.bbox.x0) * (b.bbox.y1 - b.bbox.y0),
        )
        bands[pno] = [
            BoxRect(
                page_index=pno,
                x0=float(biggest.bbox.x0),
                y0=float(biggest.bbox.y0),
                x1=float(biggest.bbox.x1),
                y1=float(biggest.bbox.y1),
            )
        ]
    return bands


async def _probe(
    path: Path,
    pages: list[int],
    tmpdir: Path,
    *,
    inject_bands: bool = False,
) -> dict[str, Any]:
    from ubt.adapters.pdf.docling_adapter import DoclingPDFAdapter

    total = pdf_struct.page_count(path)
    wanted = [p for p in pages if 1 <= p <= total]

    blocks_by_page: dict[int, list[Any]] = {}
    async for chapter in DoclingPDFAdapter().parse_stream(path, wanted):
        for block in chapter.blocks:
            if block.bbox is not None:
                blocks_by_page.setdefault(block.bbox.page, []).append(block)

    erase, keep = _facsimile_strip_plan(blocks_by_page, total, path)
    if inject_bands:
        for pno, bands in _legacy_furniture_bands(blocks_by_page, wanted).items():
            keep.setdefault(pno, []).extend(bands)
    bg = tmpdir / f"{path.stem}_bg.pdf"
    strip_text_streams(path, bg, pages=wanted, per_page_erase_rects=erase, per_page_keep_rects=keep)

    report: dict[str, Any] = {
        "path": path,
        "pages": [],
        "leaks": 0,
        "losses": 0,
        "unowned_erased": 0,
    }

    for pno in wanted:
        before, _dims = extract_lines(path, pno)
        before_chars = extract_char_styles(path, pno)
        # ``strip_text_streams`` wrote only ``wanted``, in order, so the page's
        # position in the output is its position in ``wanted``.
        after, _ = extract_lines(bg, wanted.index(pno) + 1)
        after_chars = extract_char_styles(bg, wanted.index(pno) + 1)

        blocks = blocks_by_page.get(pno, [])
        page_erase = erase.get(pno, [])
        page_keep = keep.get(pno, [])
        leaks: list[tuple[str, int]] = []
        losses: list[tuple[str, int]] = []
        unowned_erased: list[tuple[str, int]] = []

        # The area both a keep box and an erase box claim. Inside it the
        # stripper's pen-position test can legitimately blank a glyph whose *ink*
        # sits in the keep box — measured on ``chapter-3.pdf`` p1, where the
        # display ``3`` has its origin at ``x = 376.6`` and its ink at
        # ``x = 380.3`` — so a "loss" there is an artifact of origin-vs-ink, not a
        # defect. It is carved out of the **loss** test only. Carving it out of
        # the leak test would be self-defeating: a bogus keep rect (the rejected
        # furniture band) overlaps the very translated block whose text it
        # leaks, so the carve-out would hide exactly the defect being hunted.
        contested = _intersections(page_keep, page_erase)

        # Leaks are judged against the *IR blocks*, never against the rect sets.
        # The rect sets are the stripper's input and may contain a rect with no
        # block behind it; the blocks are the pipeline's actual intent — "this
        # prose gets translated" versus "this formula is the deliverable".
        translated_boxes = [b.bbox for b in blocks if not _keeps_source(b)]
        keep_boxes = [b.bbox for b in blocks if _keeps_source(b)]

        survivors = _survivors(after_chars)

        # 1. Losses: a source glyph the pipeline chose to keep must still be on
        #    the page, at the position it had. Measured per character because
        #    pdfium re-segments lines freely — see ``_chars_in``.
        for block in blocks:
            if not _keeps_source(block):
                continue
            bbox = block.bbox
            lost = _chars_in(before_chars, bbox, exclude=contested) - _chars_in(
                after_chars, bbox, exclude=contested
            )
            if lost:
                losses.append((f"{block.id} ({block.block_type})", sum(lost.values())))

        # 2. Leaks: a surviving source glyph that the pipeline intended to
        #    translate and that no keep block protects. Tested by *position*, so
        #    the verdict is geometric rather than a page-wide character census.
        for char in before_chars:
            key = _key(char.text)
            if not key:
                continue
            cx = (char.rect[0] + char.rect[2]) / 2
            cy = (char.rect[1] + char.rect[3]) / 2
            if not any(
                b.x0 - 1 <= cx <= b.x1 + 1 and b.y0 - 1 <= cy <= b.y1 + 1 for b in translated_boxes
            ):
                continue
            if any(b.x0 - 1 <= cx <= b.x1 + 1 and b.y0 - 1 <= cy <= b.y1 + 1 for b in keep_boxes):
                continue
            if _survives(survivors, char):
                leaks.append((char.text, 1))

        # 3. Unowned text has no erase rect over it and must survive. Anything
        #    missing was clobbered by a neighbouring translated box.
        for char in before_chars:
            key = _key(char.text)
            if not key:
                continue
            cx = (char.rect[0] + char.rect[2]) / 2
            cy = (char.rect[1] + char.rect[3]) / 2
            if _covered(cx, cy, blocks):
                continue
            if not _survives(survivors, char):
                unowned_erased.append((char.text, 1))

        report["pages"].append(
            {
                "page": pno,
                "blocks": len(blocks),
                "erase_rects": len(erase.get(pno, [])),
                "keep_rects": len(keep.get(pno, [])),
                "leaks": leaks,
                "losses": losses,
                "unowned_erased": unowned_erased,
            }
        )
        report["leaks"] += len(leaks)
        report["losses"] += len(losses)
        report["unowned_erased"] += len(unowned_erased)

    return report


def _print_report(report: dict[str, Any], verbose: bool) -> None:
    path: Path = report["path"]
    print(f"\n{'=' * 74}\n{path.name}")
    print(
        f"  leaks={report['leaks']}  losses={report['losses']}  "
        f"unowned_erased={report['unowned_erased']}"
    )
    for page in report["pages"]:
        bad = page["leaks"] or page["losses"] or page["unowned_erased"]
        flag = "FAIL" if bad else "ok  "
        print(
            f"  [{flag}] p{page['page']:<4d} blocks={page['blocks']:>3d} "
            f"erase={page['erase_rects']:>3d} keep={page['keep_rects']:>3d} "
            f"leaks={len(page['leaks'])} losses={len(page['losses'])} "
            f"unowned_erased={len(page['unowned_erased'])}"
        )
        if verbose:
            for text, n in page["leaks"]:
                print(f"        LEAK    ({n} chars) {text[:60]!r}")
            for text, n in page["losses"]:
                print(f"        LOSS    ({n} chars) {text[:60]!r}")
            for text, n in page["unowned_erased"]:
                print(f"        UNOWNED ({n} chars) {text[:60]!r}")


def _parse_pages(spec: str | None, total: int) -> list[int]:
    if not spec:
        return list(range(1, total + 1))
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pdfs", nargs="+", type=Path)
    parser.add_argument("--pages", help="e.g. 1-6 or 1,3,5 (default: all)")
    parser.add_argument("--verbose", action="store_true", help="print every leak/loss")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero on any leak, loss or unowned erase",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help=(
            "inject the rejected design's defect — a keep rect over a block that "
            "is being translated — and require the probe to report leaks, which "
            "proves its leak metric still detects them"
        ),
    )
    args = parser.parse_args(argv)

    tmpdir = Path(tempfile.mkdtemp(prefix="ubt_strip_probe_"))
    bad = 0
    if args.self_test:
        # A probe that reports zero defects is only trustworthy if it also
        # reports the defects it was written to catch. Inject the rejected
        # design's defect — a keep rect over a translated block — and require a
        # non-zero leak count; exit 1 if the metric has gone blind.
        total_leaks = 0
        for path in args.pdfs:
            if not path.exists():
                print(f"probe_facsimile_strip: no such file: {path}", file=sys.stderr)
                return 1
            pages = _parse_pages(args.pages, pdf_struct.page_count(path))
            report = await _probe(path, pages, tmpdir, inject_bands=True)
            _print_report(report, args.verbose)
            total_leaks += report["leaks"]
        if total_leaks == 0:
            print(
                "\nSELF-TEST FAILED: a keep rect over a translated block produced no "
                "leaks, so this probe cannot detect the defect it exists to catch.",
                file=sys.stderr,
            )
            return 1
        print(f"\nself-test passed: defect injected, {total_leaks} leak(s) detected")
        return 0

    for path in args.pdfs:
        if not path.exists():
            print(f"probe_facsimile_strip: no such file: {path}", file=sys.stderr)
            bad += 1
            continue
        pages = _parse_pages(args.pages, pdf_struct.page_count(path))
        report = await _probe(path, pages, tmpdir)
        _print_report(report, args.verbose)
        if report["leaks"] or report["losses"] or report["unowned_erased"]:
            bad += 1

    if args.strict and bad:
        print(f"\n{bad} document(s) with strip defects", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
