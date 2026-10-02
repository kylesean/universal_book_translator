"""Extraction witness: catch font-level text damage before it is billed.

The PDF that motivated this had twelve math-bearing fonts
with no ``/ToUnicode`` map. Every extraction library must then guess the
encoding, and they all guess the same MacRoman prior — so ``Nch = 2 × 10¹⁸
cm⁻³`` arrives as ``Nch ¼ 2 % 1018 cm-3`` identically through pdftotext,
pdfium, pypdf and pdfplumber. Cross-channel agreement (the original idea)
cannot see this class at all; on that corpus it scores 0/26 while the damage
sits on 20/26 pages.

Two signals replace it:

- **confirm** — the page's extracted text contains characters that only appear
  when a math glyph was misdecoded into the Latin-1 supplement (¼ for =, ð/Þ
  for parens, …). High precision; a page that hits it *is* damaged.
- **risk** — the page uses a font with no ``/ToUnicode`` and contains non-ASCII
  text. High recall; not proof, but enough to require visual confirmation.

Findings are recorded as ``extraction_witness`` metadata and a
``font_encoding_damage`` error flag on blocks of confirmed pages; the triage
stage and QE already surface flagged blocks to humans. Nothing here raises on
suspect input — silent garbage that reaches the translator is what this
witness exists to end, but a page being *risky* is not a page being wrong.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pikepdf

from ubt.adapters.pdf.pdfium_gate import pdfium_serialized
from ubt.core.ir.models import IRBlock

logger = logging.getLogger(__name__)

# MacRoman-fallback residue: codepoints a misdecoded math glyph lands on that
# correct technical prose almost never contains (¼ as a standalone operator,
# ð/Þ as brackets, ½/Œ/œ as large delimiters).
CONFIRM_CHARS = frozenset("¼ðÞ½Œœ")


@dataclass(frozen=True, slots=True)
class PageVerdict:
    page: int  # 1-based
    confirm_hits: int  # residue characters found in extracted text
    risk_fonts: int  # fonts on this page without /ToUnicode
    non_ascii: int


@pdfium_serialized
def inspect_pdf(
    pdf_path: Path, crop_margins: bool = True, margin_ratio: float = 0.05
) -> list[PageVerdict]:
    """Per-page font-encoding damage verdicts for one PDF."""
    import pypdfium2 as pdfium

    verdicts: list[PageVerdict] = []
    doc = pdfium.PdfDocument(str(pdf_path))
    # ``PdfDocument`` holds a native handle; a parse error mid-loop would leak
    # it for the life of the process without the ``finally``.
    try:
        with pikepdf.open(str(pdf_path)) as pdf:
            for i in range(len(doc)):
                page = doc[i]
                try:
                    try:
                        tp = page.get_textpage()
                        try:
                            if crop_margins:
                                width, height = page.get_size()
                                bottom = height * margin_ratio
                                top = height * (1.0 - margin_ratio)
                                text = tp.get_text_bounded(
                                    left=0, bottom=bottom, right=width, top=top
                                )
                            else:
                                text = tp.get_text_bounded()
                        finally:
                            tp.close()
                    except pdfium.PdfiumError:
                        verdicts.append(PageVerdict(i + 1, 0, 0, 0))
                        continue
                finally:
                    page.close()
                confirms = sum(text.count(ch) for ch in CONFIRM_CHARS)
                non_ascii = sum(1 for ch in text if ord(ch) > 127)
                no_tu = 0
                if i < len(pdf.pages):
                    fonts: dict[str, Any] = {}
                    try:
                        page_obj = pdf.pages[i]
                        if "/Resources" in page_obj and "/Font" in page_obj["/Resources"]:
                            raw_fonts = page_obj["/Resources"]["/Font"]
                            fonts = {str(k): font for k, font in raw_fonts.items()}
                    except (KeyError, TypeError, AttributeError):
                        fonts = {}
                    for font in fonts.values():
                        try:
                            if "/ToUnicode" not in font or not font["/ToUnicode"]:
                                no_tu += 1
                        except Exception:  # malformed font dicts: count, don't crash
                            no_tu += 1
                verdicts.append(PageVerdict(i + 1, confirms, no_tu, non_ascii))
    finally:
        doc.close()
    return verdicts


def dirty_pages(verdicts: list[PageVerdict]) -> set[int]:
    """Pages where the confirmation signal fired — the high-precision set."""
    return {v.page for v in verdicts if v.confirm_hits > 0}


def at_risk_pages(verdicts: list[PageVerdict]) -> set[int]:
    """Pages where damage is possible but unconfirmed (risk prior)."""
    return {
        v.page for v in verdicts if v.confirm_hits == 0 and v.risk_fonts > 0 and v.non_ascii > 0
    }


def summarize(verdicts: list[PageVerdict]) -> dict[str, int]:
    total = len(verdicts)
    dirty = dirty_pages(verdicts)
    risk = at_risk_pages(verdicts)
    return {
        "pages": total,
        "confirmed_pages": len(dirty),
        "at_risk_pages": len(risk),
        "residue_chars": sum(v.confirm_hits for v in verdicts),
    }


def annotate_blocks(blocks: Sequence[IRBlock], verdicts: list[PageVerdict]) -> list[IRBlock]:
    """Flag blocks sitting on confirmed-damaged pages; return the flagged ones.

    ``font_encoding_damage:page=N`` tells triage/QE that the *source* of these
    blocks may already be wrong — the defect predates translation, so repair
    must not trust the source text verbatim either.
    """
    from ubt.model.ast import RegionKind

    dirty = dirty_pages(verdicts)
    if not dirty:
        return []
    flagged: list[IRBlock] = []
    for block in blocks:
        if block.region in (RegionKind.HEADER, RegionKind.FOOTER, RegionKind.PAGE_NUMBER):
            continue
        page = block.bbox.page if block.bbox is not None else None
        if page in dirty:
            flag = f"font_encoding_damage:page={page}"
            if flag not in block.error_flags:
                block.error_flags.append(flag)
                flagged.append(block)
    return flagged


__all__ = [
    "CONFIRM_CHARS",
    "PageVerdict",
    "annotate_blocks",
    "at_risk_pages",
    "dirty_pages",
    "inspect_pdf",
    "summarize",
]
