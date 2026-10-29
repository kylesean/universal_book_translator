"""The source book's own trim size: measured, and the one thing L3 must honour.

The reflow (L3) channel is the page-agnostic one: it rebuilds the book from the
IR and lays it out on a paper size the *renderer* chooses. Until now that choice
was the literal ``paper: "a4"`` in two templates, so a book measured at
190.5 x 234.9mm was re-laid-out at 210 x 297mm — 39% more page area than the
source, with the reader's copy on a different page size than the one the book
was designed for. "Isomorphic delivery" cannot mean that.

Two facts drive the design here, both measured on this repository's corpus:

* **The corpus is not A4.** Of 25 documents, 19 are US Letter (215.9 x 279.4),
  5 are A4, and 1 is a custom trim (190.5 x 234.9). Hardcoding *any* one of
  those is wrong for the rest, which is why this measures instead of defaulting.
* **A page's *size* is not a readability property; its *measure* is.** Chinese
  body text wants roughly 25-40 characters per line; past that the eye's return
  sweep starts skipping. On A4 with 2.2cm margins the body sets at 44.8 chars
  (10.5pt), which is over budget — and the fix for that is the margin, not the
  paper. See ``TARGET_MEASURE_MM``.

So this module answers exactly one question -- "what trim does this document
declare?" -- and returns ``None`` rather than a guess when the source cannot
answer it (EPUB, Markdown, or a PDF that will not open). Callers fall back
deliberately; the fallback is a product decision and belongs at the call site
where it can be documented, not hidden in a measurement function.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

#: Millimetres per PDF point (1/72 inch).
MM_PER_PT: float = 25.4 / 72.0

#: A trim narrower or shorter than this is not a page a book was set on: it is a
#: rotated page, a poster fold-out, or a mis-read MediaBox. Used to reject a
#: measured size rather than propagate it into the layout.
MIN_PLAUSIBLE_TRIM_MM: float = 90.0
#: ...and taller/wider than this is likewise not a book trim.
MAX_PLAUSIBLE_TRIM_MM: float = 500.0

#: Share of pages that must agree on one size before it counts as the document's
#: trim. A book may carry a few fold-outs or a differently-sized cover; a
#: document where no size reaches even this share of pages has no single trim to
#: honour, and laying it out at the plurality page's size would be inventing one.
TRIM_AGREEMENT_SHARE: float = 0.60

#: The body measure the reflow channel aims for, in millimetres.
#:
#: **This is the readability commitment; the margin is derived from it.** A page
#: setup has three numbers (paper, margin, measure) and any two fix the third, so
#: hardcoding the margin instead of the measure is what made the previous value
#: wrong: a fixed 3.2cm margin gives 39.4 chars on A4 (210mm) but 41.0 on US
#: Letter (215.9mm) -- the same setting lands inside the convention on one paper
#: and outside it on the other.
#:
#: Measured, in chars per line at the 10.5pt Chinese body (3.70mm per character):
#:
#:     A4, margin 2.2cm (previous)  166.0mm -> 44.8 chars   over budget
#:     146mm measure                146.0mm -> 39.4 chars   in budget
#:     185mm 16-mo, margin 2.0cm    145.0mm -> 39.1 chars   in budget
#:
#: 146mm therefore reproduces the *reading experience* of a Chinese trade book
#: while leaving the paper where the reader expects it. 25-40 characters per line
#: is the convention; past 40 the eye's return sweep starts skipping lines.
TARGET_MEASURE_MM: float = 146.0

#: Narrowest margin the reflow channel will emit. Below this the text runs into
#: the trim on a small page: a 130mm 32-mo cannot reach a 146mm measure at all
#: (it would need negative margins), so it keeps this margin and accepts the
#: narrower measure -- 80mm, 21.6 chars -- rather than printing off the edge.
MIN_PAGE_MARGIN_MM: float = 25.0

#: Vertical margin, unchanged from the previous 2.4cm. Deliberately not widened
#: with the horizontal one: the measure is what the eye tracks left-to-right, and
#: cropping the height would cost lines per page without improving either.
DEFAULT_PAGE_MARGIN_Y_MM: float = 24.0

#: A4, as the reflow channel's declared fallback. Named here rather than spelled
#: ``210.0`` at each site so the template and the layout audit cannot disagree
#: about what "no trim measured" means.
A4_WIDTH_MM: float = 210.0
A4_HEIGHT_MM: float = 297.0


#: Typst's default column gutter, as a share of the text block. Measured, not
#: assumed: ``#context layout(size => ...)`` on a two-column page reports
#: ``236.75pt`` per column inside a 493.24pt block — 0.48 of the block each,
#: i.e. a 3.96% gutter — and the same 4% reproduces a 400pt page's 153.6pt
#: column exactly. The renderer's table-fit arithmetic used a fixed 6mm, which is
#: 0.03 of an A4 block, so a two-column table was sized ~3pt too wide per column.
COLUMN_GUTTER_SHARE: float = 0.04


@dataclass(frozen=True, slots=True)
class TrimSize:
    """A measured page trim in millimetres, plus the evidence behind it."""

    width_mm: float
    height_mm: float
    #: Pages measured, and how many agreed on this size. Kept so a report can
    #: say "1 of 26 pages" instead of presenting a plurality as a consensus.
    pages_measured: int
    pages_agreeing: int

    @property
    def agreement(self) -> float:
        """Share of measured pages that agree on this trim, 0.0 when none."""
        if self.pages_measured <= 0:
            return 0.0
        return self.pages_agreeing / self.pages_measured

    @property
    def typst_width_spec(self) -> str:
        """The ``width:`` argument for this trim, e.g. ``190.5mm``."""
        return f"{self.width_mm:.1f}mm"

    @property
    def typst_height_spec(self) -> str:
        """The ``height:`` argument for this trim, e.g. ``234.9mm``."""
        return f"{self.height_mm:.1f}mm"

    def typst_page_spec(self) -> str:
        """The size arguments for ``#set page(...)``.

        Emitted as ``width: Xmm, height: Ymm`` rather than ``paper: (width: ...)``
        because Typst's ``paper`` parameter takes a *name* from a closed list
        (``a4``, ``us-letter``, ``uk-book-a``, ...) and rejects a dictionary with
        ``expected "a4", ... found dictionary``. A measured trim is usually not
        one of those names, and snapping to the nearest would undo the
        measurement this whole path exists to honour — so the size is passed
        positionally instead.

        Values are rounded to 0.1mm, below what any printer resolves, which also
        keeps the emitted markup stable across re-renders of the same source.
        """
        return f"width: {self.typst_width_spec}, height: {self.typst_height_spec}"


def measure_trim(source_path: Path | str | None) -> TrimSize | None:
    """The document's own trim, or ``None`` when the source cannot declare one.

    ``None`` is the honest answer for a source that has no pages (EPUB, Markdown,
    DOCX) and for a PDF whose pages disagree with each other. Returning a
    majority-guess here would let a caller believe a measurement happened.

    Rotation is applied: a landscape page stored as ``/Rotate 90`` over a
    portrait MediaBox is portrait to the reader, and honouring the unrotated box
    would deliver a sideways book.
    """
    if source_path is None:
        return None
    path = Path(source_path)
    if not path.exists() or path.suffix.lower() != ".pdf":
        return None
    try:
        import pikepdf

        from ubt.adapters.pdf.pdf_struct import page_rotation, page_size
    except Exception as exc:  # pragma: no cover - import failure
        logger.debug("Trim measurement unavailable: %s", exc)
        return None

    sizes: Counter[tuple[float, float]] = Counter()
    try:
        with pikepdf.open(path) as pdf:
            total = len(pdf.pages)
            if total == 0:
                return None
            for page in pdf.pages:
                w_pt, h_pt = page_size(page)
                if page_rotation(page) % 180 == 90:
                    w_pt, h_pt = h_pt, w_pt
                w_mm = w_pt * MM_PER_PT
                h_mm = h_pt * MM_PER_PT
                # Round to 0.1mm before counting: the same trim arrives with
                # sub-point jitter across pages of one book (595.276 vs 595.32),
                # and exact float equality would split one size into many.
                w_mm, h_mm = round(w_mm, 1), round(h_mm, 1)
                if not (
                    MIN_PLAUSIBLE_TRIM_MM <= w_mm <= MAX_PLAUSIBLE_TRIM_MM
                    and MIN_PLAUSIBLE_TRIM_MM <= h_mm <= MAX_PLAUSIBLE_TRIM_MM
                ):
                    continue
                sizes[(w_mm, h_mm)] += 1
    except Exception as exc:
        logger.debug("Cannot measure the trim of %s: %s", path, exc)
        return None

    if not sizes:
        return None
    (width_mm, height_mm), agreeing = sizes.most_common(1)[0]
    measured = sum(sizes.values())
    trim = TrimSize(
        width_mm=width_mm,
        height_mm=height_mm,
        pages_measured=measured,
        pages_agreeing=agreeing,
    )
    if trim.agreement < TRIM_AGREEMENT_SHARE:
        logger.debug(
            "No single trim on %s: the most common size (%sx%smm) covers only "
            "%.0f%% of %d measured pages, below the %.0f%% needed to call it the "
            "document's trim.",
            path,
            width_mm,
            height_mm,
            trim.agreement * 100,
            measured,
            TRIM_AGREEMENT_SHARE * 100,
        )
        return None
    return trim


def body_measure_mm(
    width_mm: float,
    *,
    margin_mm: float,
    columns: int = 1,
) -> float:
    """Usable body width in millimetres for a trim and margin.

    Exposed because the *measure* is the thing worth asserting in a test: a
    change that widens the paper without widening the margin, or narrows the
    margin without checking the resulting line length, should fail a test rather
    than ship and be noticed by a reader.

    The gutter follows Typst's own rule (:data:`COLUMN_GUTTER_SHARE`) rather
    than a fixed length, because a fixed length is wrong on every page but the
    one it was measured on.
    """
    block_mm = max(0.0, width_mm - 2 * margin_mm)
    if columns <= 1:
        return block_mm
    return max(0.0, block_mm * (1.0 - COLUMN_GUTTER_SHARE) / columns)


def margin_for_measure(
    width_mm: float,
    *,
    target_measure_mm: float = TARGET_MEASURE_MM,
    min_margin_mm: float = MIN_PAGE_MARGIN_MM,
) -> float:
    """The horizontal margin that yields ``target_measure_mm`` on ``width_mm``.

    Deriving the margin from the target measure -- rather than deriving the
    measure from a fixed margin -- is what makes the readability promise hold
    across papers. A fixed margin silently changes the line length every time the
    trim changes: 39.4 chars on A4, 41.0 on US Letter, from the identical setting.

    Clamped at ``min_margin_mm``. A narrow page (a 130mm 32-mo) cannot reach
    146mm of text at all, and the clamp is what turns that from an impossible
    request into a narrower measure (80mm, 21.6 chars) instead of text running
    off the trim.
    """
    if width_mm <= 0:
        return min_margin_mm
    derived = (width_mm - target_measure_mm) / 2.0
    return max(min_margin_mm, derived)


@dataclass(frozen=True, slots=True)
class PageSetup:
    """The page geometry the reflow channel will emit, in millimetres.

    Carries the *size* as well as the margin because the layout audit has to be
    told the geometry it is auditing: a spec built for A4 declares a safe area
    ~28pt wider per side than a 190.5mm trim has, so every line of a
    correctly-set narrow book would report as clean while a real spill was
    invisible (or, reversed, a correct page would "heal" by shrinking).
    """

    #: Typst markup for the page size — a ``paper:`` name, or ``width:``/
    #: ``height:`` when the trim is not a named paper. ``None`` when no trim was
    #: measured; the caller then keeps A4.
    paper: str | None
    width_mm: float
    height_mm: float
    margin_x_mm: float
    margin_y_mm: float
    #: Usable body width for the **book** channel at this trim, margin, and
    #: column count.
    #:
    #: Named ``book_`` and not ``measure_`` on purpose: the academic template
    #: keeps its own submission margins (``ACADEMIC_SINGLE_MARGIN_X_CM`` and
    #: friends), so its measure is *not* this number even on the same paper.
    #: A caller rendering academic content must read
    #: :meth:`ubt.render.typst.document_renderer.TypstDocumentRenderer
    #: .text_width_pt`, which is the one authority on the emitted block.
    book_measure_mm: float

    @property
    def measured(self) -> bool:
        """Whether this setup came from the source, rather than the A4 fallback."""
        return self.paper is not None

    def typst_page_args(self) -> str:
        """The ``#set page(...)`` size arguments for this setup.

        ``paper: "a4"`` when nothing was measured, else ``width: …, height: …``.
        Returned as one string so a template cannot emit half a size.
        """
        if self.paper is None:
            return 'paper: "a4"'
        return self.paper


def resolved_page_setup(
    trim: TrimSize | None,
    *,
    columns: int = 1,
    target_measure_mm: float = TARGET_MEASURE_MM,
) -> PageSetup:
    """The page setup the reflow channel should emit.

    ``paper`` is ``None`` when the source declared no trim (EPUB, Markdown, or a
    PDF with no agreed page size). The *caller* then falls back to A4 — a product
    decision kept at the call site where it can be documented, not buried in a
    measurement function that would then be reporting a measurement it never made.
    The fallback geometry is returned anyway so the audit and the template agree
    on what "the fallback" means instead of each hardcoding 210x297.

    The measure is returned computed rather than left to the caller so a test can
    assert the readability promise on the exact numbers that get emitted.
    """
    width = trim.width_mm if trim is not None else A4_WIDTH_MM
    height = trim.height_mm if trim is not None else A4_HEIGHT_MM
    margin_x = margin_for_measure(width, target_measure_mm=target_measure_mm)
    return PageSetup(
        paper=trim.typst_page_spec() if trim is not None else None,
        width_mm=width,
        height_mm=height,
        margin_x_mm=margin_x,
        margin_y_mm=DEFAULT_PAGE_MARGIN_Y_MM,
        book_measure_mm=body_measure_mm(width, margin_mm=margin_x, columns=columns),
    )


def chars_per_line(measure_mm: float, size_pt: float) -> float:
    """Approximate CJK characters per line at ``size_pt`` in a given measure.

    A CJK glyph is square, so one character occupies one em -- ``size_pt``
    points wide. Used to assert the 25-40 character convention rather than to
    typeset: the real count depends on line breaking, which Typst owns.
    """
    if measure_mm <= 0 or size_pt <= 0:
        return 0.0
    em_mm = size_pt * MM_PER_PT
    return measure_mm / em_mm


__all__ = [
    "A4_HEIGHT_MM",
    "A4_WIDTH_MM",
    "DEFAULT_PAGE_MARGIN_Y_MM",
    "MAX_PLAUSIBLE_TRIM_MM",
    "MIN_PAGE_MARGIN_MM",
    "MIN_PLAUSIBLE_TRIM_MM",
    "MM_PER_PT",
    "TARGET_MEASURE_MM",
    "TRIM_AGREEMENT_SHARE",
    "PageSetup",
    "TrimSize",
    "body_measure_mm",
    "chars_per_line",
    "margin_for_measure",
    "measure_trim",
    "resolved_page_setup",
]
