"""Zero-AGPL pure-Python PDF page alternator and dimension inspector.

CRITICAL ARCHITECTURE INVARIANT:
This module must NEVER import or reference 'fitz' or 'pymupdf' to maintain 100%
license safety and commercial independence. Page merging is backed by pikepdf
(MPL-2.0); the retired pypdf leg moved here when pdf_oxide replaced it.

SECOND INVARIANT (exit-134 post-mortem): this is where pikepdf
enters the ``ubt.adapters`` import graph, and pikepdf's nanobind extension can
be initialized only ONCE per process. Erasing ``sys.modules["pikepdf..."]`` and
importing again makes nanobind call ``abort()`` — no traceback, no exception, no
test result. So never stub these names through ``sys.modules`` (use a lazy import
seam), and never hand coverage a dotted source under ``ubt.adapters``: it imports
the module to locate it and then un-imports everything it touched.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

import pikepdf

from ubt.adapters.pdf import pdf_struct
from ubt.core.exceptions import DocumentParseError

logger = logging.getLogger(__name__)


class PageDimension(NamedTuple):
    """Page dimension in standard typographic points (1 pt = 1/72 inch)."""

    page_index: int
    width_pt: float
    height_pt: float


class InterleaveResult(NamedTuple):
    """The interleaved artifact plus the blank pages it had to insert.

    ``padding_pages`` holds the 1-based numbers of every page this alternator
    filled with an intentional blank (the facing flyleaf and any page added to
    square up unequal source/target page counts). The visual gate consumes this
    list to exempt those pages from ``blank_page``/``blank_candidate`` findings:
    without it, an interleave-padded but otherwise correct artifact reported
    ``passed=False`` and could be refused by the opt-in blocking gate.
    """

    output_path: Path
    padding_pages: tuple[int, ...] = ()


class BilingualAlternator:
    """Interleaves original and translated PDF pages to produce facing/alternating bilingual editions.

    Eliminates squished in-place 5pt 'ant fonts' by dedicating full pages to reflowed translation.
    """

    def inspect_dimensions(self, pdf_path: Path | str) -> list[PageDimension]:
        """Inspect page dimensions across all pages in the PDF."""
        path = Path(pdf_path)
        if not path.exists():
            raise DocumentParseError(f"PDF file not found: {path}")

        dimensions: list[PageDimension] = []
        try:
            with pdf_struct.open_pdf(path) as pdf:
                for idx, page in enumerate(pdf.pages):
                    width, height = pdf_struct.page_size(page)
                    dimensions.append(
                        PageDimension(page_index=idx, width_pt=width, height_pt=height)
                    )
            return dimensions
        except Exception as err:
            raise DocumentParseError(
                f"Failed to inspect PDF dimensions '{path.name}': {err}"
            ) from err

    def interleave_pages(
        self,
        source_pdf: Path | str,
        translated_pdf: Path | str,
        output_pdf: Path | str,
        facing_spread: bool = False,
    ) -> InterleaveResult:
        """Interleaves pages from source and translated PDFs in a zipper-like sequence.

        If facing_spread is False (default):
            Sequence: [Source Page 1, Translated Page 1, Source Page 2, Translated Page 2, ...].
        If facing_spread is True (book publishing facing-pages spread):
            Inserts a Recto flyleaf at Page 1 so that [Verso (Even, Left), Recto (Odd, Right)]
            perfectly pairs [Source Page N, Translated Page N] on the same double-page spread.

        Returns the output path together with the 1-based numbers of the blank
        pages inserted (flyleaf and page-count padding), so the visual gate can
        exempt them from blank-page defects.
        """
        src_path = Path(source_pdf)
        trans_path = Path(translated_pdf)
        out_path = Path(output_pdf)

        if not src_path.exists():
            raise DocumentParseError(f"Source PDF file not found: {src_path}")
        if not trans_path.exists():
            raise DocumentParseError(f"Translated PDF file not found: {trans_path}")

        try:
            with pdf_struct.open_pdf(src_path) as src, pdf_struct.open_pdf(trans_path) as trans:
                src_len = len(src.pages)
                trans_len = len(trans.pages)
                max_pages = max(src_len, trans_len)

                if src_len != trans_len:
                    logger.warning(
                        "Page count mismatch in bilingual interleaving: "
                        "source PDF has %d pages but translated PDF has %d pages. "
                        "Missing pages will be padded with blank pages. "
                        "Enable page_strict mode in TypstReconstructor to fix alignment.",
                        src_len,
                        trans_len,
                    )

                src_width, src_height = (
                    pdf_struct.page_size(src.pages[0])
                    if src_len
                    else (
                        595.0,
                        842.0,
                    )
                )
                trans_width, trans_height = (
                    pdf_struct.page_size(trans.pages[0])
                    if trans_len
                    else (
                        src_width,
                        src_height,
                    )
                )

                with pikepdf.new() as out:
                    padding_pages: list[int] = []
                    if facing_spread:
                        # In book printing, Page 1 is always Recto (single right page).
                        # Spreads (facing pages) appear at [Verso (Even, Left), Recto (Odd, Right)].
                        # To ensure Source Page N and Translated Page N face each other on the
                        # same spread, add a blank Recto flyleaf (Page 1).
                        out.add_blank_page(page_size=(int(src_width), int(src_height)))
                        padding_pages.append(len(out.pages))

                    for i in range(max_pages):
                        if i < src_len:
                            out.pages.append(src.pages[i])
                        else:
                            out.add_blank_page(page_size=(int(src_width), int(src_height)))
                            padding_pages.append(len(out.pages))
                        if i < trans_len:
                            out.pages.append(trans.pages[i])
                        else:
                            out.add_blank_page(page_size=(int(trans_width), int(trans_height)))
                            padding_pages.append(len(out.pages))

                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out.save(str(out_path))

            return InterleaveResult(output_path=out_path, padding_pages=tuple(padding_pages))
        except Exception as err:
            raise DocumentParseError(f"Failed to interleave PDF documents: {err}") from err

    async def interleave_pages_async(
        self,
        source_pdf: Path | str,
        translated_pdf: Path | str,
        output_pdf: Path | str,
        facing_spread: bool = False,
    ) -> InterleaveResult:
        """Async version of interleave_pages offloading to a thread worker."""
        import asyncio

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self.interleave_pages,
            source_pdf,
            translated_pdf,
            output_pdf,
            facing_spread,
        )
