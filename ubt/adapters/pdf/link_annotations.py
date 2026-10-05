"""Dynamic PDF Link Annotation Relocation.

When an overlay typesetter replaces translated text on a PDF page, the original
text is stripped and the new translation is painted at new coordinates. In the
original PDF, interactive elements like hyperlinks and academic citations
(e.g., [1], Shi et al., 2026) are stored as `/Subtype /Link` annotation
dictionaries with static bounding boxes (`/Rect`).

Because translated words occupy different horizontal and vertical positions,
leaving `/Rect` unchanged renders the links non-clickable ("dead text") at their
new positions, while creating confusing "ghost click" zones at the old coordinates.

This module inspects the source page and the compiled overlay PDF, locates the
new character/glyph bounding boxes of corresponding citations and links, and
dynamically rewrites the `/Rect` coordinates on the output page, while pruning
unmatched ghost links.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path

import pikepdf
import pypdfium2 as pdfium

from ubt.adapters.pdf.pdfium_gate import PDFIUM_LOCK

logger = logging.getLogger(__name__)

Rect = tuple[float, float, float, float]

_CITE_KEY_RE = re.compile(r"cite\.([a-zA-Z]+)(\d{4})?", re.IGNORECASE)
_BRACKET_NUM_RE = re.compile(r"\[(\d{1,4}(?:[-–—]\d{1,4})?)\]")
# A cross-reference's *displayed* number ("Figure 1", "Table 4", "Sec. 3.1").
# The label is translated ("Figure" -> "图") but the number is invariant, so it
# is the token to search for on the translated page. Note the destination key's
# counter is NOT the displayed number (e.g. table.caption.5 shows "Table 1"),
# which is why the number is read from the source fragment instead.
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)*")


def _is_number_token(text: str) -> bool:
    """True for a candidate that ends in a number ("1", "3.1", "表 1")."""
    return bool(text) and text[-1].isdigit()


def _standalone_number(text: str, start: int, end: int) -> bool:
    """True if ``text[start:end]`` ends at a whole number, not a longer one.

    Rejects "1" matching inside "10" and "表 1" inside "表 10".
    """
    if start > 0 and (text[start - 1].isdigit() or text[start - 1] == "."):
        return False
    return not (end < len(text) and (text[end].isdigit() or text[end] == "."))


def _rect_intersects_any(rect: Sequence[float], strip_rects: Sequence[Rect]) -> bool:
    """True if rect [x0, y0, x1, y1] overlaps any rect in strip_rects."""
    rx0, ry0, rx1, ry1 = rect[0], rect[1], rect[2], rect[3]
    for sx0, sy0, sx1, sy1 in strip_rects:
        if not (rx1 < sx0 or rx0 > sx1 or ry1 < sy0 or ry0 > sy1):
            return True
    return False


def _extract_candidates_from_annot(
    annot: pikepdf.Object,
    src_text: str,
    *,
    figure_prefix: str = "",
    table_prefix: str = "",
) -> list[str]:
    """Derive prioritized search tokens for a link annotation.

    ``figure_prefix``/``table_prefix`` are the target language's caption labels
    (e.g. "图"/"表" for Chinese), used to rebuild a cross-reference whose label
    the translation replaced.
    """
    candidates: list[str] = []
    raw = src_text.strip()
    clean_src = raw.strip("(),;[]'\" \t\r\n")

    # 0. The literal text the annotation covers. A multi-citation bracket
    # ("[1,2,3,4]") stores one annotation per number, so this is a *fragment*
    # ("[1,", "2,", "4]") that the translated page keeps verbatim. Trying it
    # first relocates single-digit citations the rules below miss: the bracket
    # regex needs a closing "]", and a lone "1" was never added as a candidate
    # (the length guard), so their links stayed dead at the stripped source rect.
    if len(raw) >= 2:
        candidates.append(raw)

    # 1. Bracket-style citation in source text: "[12]" or "[1-3]"
    bracket_m = _BRACKET_NUM_RE.search(src_text)
    if bracket_m:
        candidates.append(bracket_m.group(0))
        candidates.append(bracket_m.group(1))

    # 2. Check Action / Dest dictionary
    action = annot.get("/A")
    dest = str(action.get("/D")) if action and "/D" in action else ""
    if not dest:
        dest_val = annot.get("/Dest")
        if dest_val is not None:
            dest = str(dest_val)

    uri = str(action.get("/URI")) if action and "/URI" in action else ""

    # Named citation target (LaTeX / hyperref convention: 'cite.shi2026harbor')
    if dest and "cite." in dest.lower():
        m = _CITE_KEY_RE.search(dest)
        if m:
            author, year = m.group(1), m.group(2)
            if author and len(author) >= 2:
                candidates.append(author)
            if year:
                candidates.append(year)

    # Cross-reference target ("figure.caption.3", "table.caption.5",
    # "subsection.3.1"). The destination tells us the *kind*; the displayed
    # number lives in the source fragment ("Figure 1", "Table 4", "Sec. 3.1").
    # Rebuild the localized form ("图 1", "表 4") the translated page carries.
    if dest:
        low_dest = dest.lower()
        number_m = _NUMBER_RE.search(raw)
        if number_m:
            number = number_m.group(0)
            if "figure" in low_dest and figure_prefix:
                candidates.append(f"{figure_prefix} {number}")
                candidates.append(f"{figure_prefix}{number}")
            elif "table" in low_dest and table_prefix:
                candidates.append(f"{table_prefix} {number}")
                candidates.append(f"{table_prefix}{number}")
            elif "section" in low_dest:
                candidates.append(number)

    if uri:
        # For URLs, search for the full URI or domain/path fragment
        candidates.append(uri)
        domain_m = re.search(r"https?://([^/\s]+)", uri)
        if domain_m:
            candidates.append(domain_m.group(1))

    # Clean text from source box (e.g. 'Shi et al.', '2026', 'Table 1'). A lone
    # digit ("[1," -> "1") is a valid last resort: the target keeps the number
    # even when its punctuation was rewritten, and the vertical distance guard
    # below keeps the match near the source line.
    if clean_src not in candidates and (len(clean_src) >= 2 or clean_src.isdigit()):
        candidates.append(clean_src)

    return candidates


def relocate_page_annotations(
    *,
    page: pikepdf.Page,
    page_no: int,
    source_pdf: Path | str,
    overlay_path: str,
    strip_rects: Sequence[Rect],
    overlay_page_no: int = 0,
    figure_prefix: str = "",
    table_prefix: str = "",
) -> int:
    """Relocate PDF `/Subtype /Link` annotations on `page` to match overlay text.

    ``overlay_path`` is a translated page image; ``overlay_page_no`` (0-based)
    selects which page of it holds the translation for ``page_no``. A per-page
    overlay passes 0 (its only page); a whole composed document passes
    ``page_no - 1``. ``figure_prefix``/``table_prefix`` are the target language's
    caption labels, used to rebuild a translated cross-reference.

    Returns the count of modified annotations (relocated + pruned).
    """
    annots = page.get("/Annots")
    if not annots or not strip_rects:
        return 0

    if not Path(overlay_path).is_file():
        return 0

    # Collect indices of link annotations in stripped zones
    link_indices: list[int] = []
    for idx, a in enumerate(annots):
        if a.get("/Subtype") == "/Link" and "/Rect" in a:
            rect = [float(x) for x in a["/Rect"]]
            if _rect_intersects_any(rect, strip_rects):
                link_indices.append(idx)

    if not link_indices:
        return 0

    relocated_count = 0
    kept_count = 0

    with PDFIUM_LOCK:
        try:
            src_doc = pdfium.PdfDocument(str(source_pdf))
            overlay_doc = pdfium.PdfDocument(str(overlay_path))
        except Exception as exc:
            logger.debug("Failed opening PDF for annot relocation on page %d: %s", page_no, exc)
            return 0

        src_page = None
        overlay_page = None
        tp_src = None
        tp_overlay = None
        try:
            if page_no - 1 >= len(src_doc) or len(overlay_doc) <= overlay_page_no:
                return 0

            src_page = src_doc[page_no - 1]
            overlay_page = overlay_doc[overlay_page_no]

            tp_src = src_page.get_textpage()
            tp_overlay = overlay_page.get_textpage()
            overlay_text = tp_overlay.get_text_range()

            for idx in link_indices:
                annot = annots[idx]
                rect = [float(x) for x in annot["/Rect"]]
                rx0, ry0, rx1, ry1 = rect
                orig_y = (ry0 + ry1) / 2.0

                # Expand slightly by 1pt to avoid clipping adjacent punctuation
                src_text = tp_src.get_text_bounded(
                    rx0 - 1.0, ry0 - 1.0, rx1 + 1.0, ry1 + 1.0
                ).strip()
                candidates = _extract_candidates_from_annot(
                    annot,
                    src_text,
                    figure_prefix=figure_prefix,
                    table_prefix=table_prefix,
                )

                best_box: list[float] | None = None
                min_dist = float("inf")

                for cand in candidates:
                    if not cand:
                        continue
                    # Short candidates are unsafe in general, but a lone digit
                    # is a real citation number (see _extract_candidates_from_annot).
                    if len(cand) < 2 and not cand.isdigit():
                        continue
                    is_number = _is_number_token(cand)
                    cand_lower = cand.lower()
                    pos = 0
                    while True:
                        hit_idx = overlay_text.lower().find(cand_lower, pos)
                        if hit_idx == -1:
                            break
                        end_idx = hit_idx + len(cand)
                        # A bare number must be a whole token: "1" must not match
                        # the "1" inside "10" or "3.1".
                        if is_number and not _standalone_number(overlay_text, hit_idx, end_idx):
                            pos = end_idx
                            continue
                        boxes = [tp_overlay.get_charbox(c) for c in range(hit_idx, end_idx)]
                        if boxes:
                            bx0 = min(b[0] for b in boxes)
                            by0 = min(b[1] for b in boxes)
                            bx1 = max(b[2] for b in boxes)
                            by1 = max(b[3] for b in boxes)
                            cand_y = (by0 + by1) / 2.0
                            dist = abs(cand_y - orig_y)
                            if dist < min_dist:
                                min_dist = dist
                                best_box = [bx0, by0, bx1, by1]
                        pos = end_idx

                    # If we found a match within 45pt vertical distance, consider it resolved
                    if best_box is not None and min_dist < 45.0:
                        break

                # Accept matches within reasonable vertical reading context (80pt)
                if best_box is not None and min_dist < 80.0:
                    annot["/Rect"] = pikepdf.Array(
                        [
                            Decimal(f"{best_box[0]:.2f}"),
                            Decimal(f"{best_box[1]:.2f}"),
                            Decimal(f"{best_box[2]:.2f}"),
                            Decimal(f"{best_box[3]:.2f}"),
                        ]
                    )
                    relocated_count += 1
                else:
                    # No translated glyph matched the candidate. Keep the link
                    # where it is: a citation whose click zone sits a few points
                    # off is still a working reference, whereas pruning it makes
                    # the citation dead. Ghost clicks are the lesser failure.
                    kept_count += 1
        except Exception as exc:
            logger.debug("Error during annot relocation on page %d: %s", page_no, exc)
        finally:
            if tp_src is not None:
                tp_src.close()
            if tp_overlay is not None:
                tp_overlay.close()
            if src_page is not None:
                src_page.close()
            if overlay_page is not None:
                overlay_page.close()
            src_doc.close()
            overlay_doc.close()

    if relocated_count > 0:
        logger.info(
            "Relocated %d PDF link annotation(s) on page %d to matching translated glyphs",
            relocated_count,
            page_no,
        )
    if kept_count > 0:
        logger.info(
            "Kept %d link annotation(s) at their source rect on page %d (no translated match)",
            kept_count,
            page_no,
        )

    return relocated_count + kept_count
