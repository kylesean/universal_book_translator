"""PDFium line geometry vs Docling layout IoU cross-check (Defense 1).

In the analyze stage, born-digital PDF pages can suffer from deep-learning
layout model hallucinations (e.g. cross-column spans, phantom bounding boxes,
or misclassified raster regions). This module performs lightweight (~2ms)
cross-checking by extracting physical text lines via PDFium (the ground-truth
glyph stream) and computing the Intersection-over-Union (IoU) with Docling's
bounding boxes.

When a text block's IoU with physical text lines falls below the threshold
(default 0.60), or when a block with text content has no physical lines under
it, the block is gracefully demoted to PRESERVED_OPAQUE:
- Its element confidence is set to Confidence.UNKNOWN.
- Its skip_translate and policy_translate are set to False/True (skip translation).
- Its provenance records the mismatch details.
- LayerCompositor leaves Layer 0 (the original canvas) intact without masking.

The same physical lines are the witness for one text-content repair:
:func:`repair_missing_spaces_with_lines` re-inserts inter-word spaces Docling's
line join drops ("A Sandbox" -> "ASandbox"), which otherwise reaches the model
as a single token and mistranslates.
"""

from __future__ import annotations

import dataclasses
import difflib
import logging
import re
from collections.abc import Sequence
from pathlib import Path

from ubt.adapters.pdf.textgeom import (
    CharStyle,
    LineBox,
    box_area,
    box_iou,
    extract_char_styles,
    extract_lines,
    styled_runs_in_box,
)
from ubt.core.ir.models import BlockType, IRBlock
from ubt.core.policy.layout_policy import CONTROL_RE
from ubt.model.ast import Confidence
from ubt.model.span import CompositeSpan, PhysicalBox

logger = logging.getLogger(__name__)

#: Text block types subject to physical glyph grounding verification.
_VERIFIABLE_TYPES = frozenset(
    {BlockType.NARRATIVE, BlockType.DIALOGUE, BlockType.HEADING, BlockType.LIST_ITEM}
)

#: Default IoU threshold below which a Docling box is considered ungrounded or displaced.
DEFAULT_MIN_IOU = 0.60


def _boxes_intersect(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> bool:
    return max(a[0], b[0]) < min(a[2], b[2]) and max(a[1], b[1]) < min(a[3], b[3])


# ---------------------------------------------------------------------------
# Missing inter-word spaces (Docling's joined line text)
# ---------------------------------------------------------------------------
#: A glued row fragment counts as separated by a real space when the gap between
#: the fragments is at least this many em. Below it the fragments are ONE word
#: the extractor split by run segmentation ("Chr"+"onus" at a 0.3pt gap on a
#: 22pt display label), and adopting the space would corrupt the word.
_WITNESS_SPACE_EM = 0.2
#: A witness whose spaces make up more than this share of its inter-character
#: gaps is tracked/letter-spaced display text ("D a t a"), never prose.
_WITNESS_MAX_SPACE_SHARE = 0.35
#: An unmapped custom-font glyph (pdfium's \x02) sitting at a line's end is the
#: font's hyphen at a line break; ``CONTROL_RE`` strips it for comparison.
_LINE_END_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]\s*$")


def _is_cjk(char: str) -> bool:
    return (
        "\u2e80" <= char <= "\u9fff"
        or "\u3000" <= char <= "\u303f"
        or "\u3040" <= char <= "\u30ff"
        or "\uff00" <= char <= "\uffef"
    )


def _is_word_char(char: str) -> bool:
    return bool(char) and char.isalnum()


def _joins_with_space(left: str, right: str) -> bool:
    """Whether a real inter-word gap separated two characters inside a line."""
    return _is_word_char(left) and _is_word_char(right) and not _is_cjk(left) and not _is_cjk(right)


def _line_break_joins_with_space(left: str, right: str) -> bool:
    """A Latin line break stands in for a space; a CJK one does not."""
    return not _is_cjk(left) and not _is_cjk(right)


def witness_line_text(line: LineBox) -> str:
    """One pdfium line's text, with glued-fragment spaces re-decided by geometry.

    ``textgeom`` glues a row's fragments with a plain space (``_glue_run``)
    because pdfium splits a row by run, not by word. On normally spaced prose
    that is right; on a tracked label it bakes in a space that is not there
    ("Chronus" -> "Chr onus"). The fragments' own rects carry the real gap, so
    a space is only kept where the gap justifies one.
    """
    members: tuple[LineBox, ...] = tuple(getattr(line, "members", ()) or ())
    if len(members) < 2:
        return line.text
    ordered = sorted(members, key=lambda member: member.rect[0])
    font = line.font_size or 10.0
    out = ordered[0].text.strip()
    for previous, current in zip(ordered, ordered[1:], strict=False):
        token = current.text.strip()
        if not token:
            continue
        gap = current.rect[0] - previous.rect[2]
        if not out:
            out = token
        elif gap >= _WITNESS_SPACE_EM * font and _joins_with_space(out[-1], token[0]):
            out += f" {token}"
        else:
            out += token
    return out


def join_witness_lines(lines: Sequence[LineBox]) -> str:
    """A block's physical lines in reading order as one candidate text."""
    ordered = sorted(lines, key=lambda line: (-line.rect[3], line.rect[0]))
    out = ""
    previous_broke_at_hyphen = False
    for line in ordered:
        raw = witness_line_text(line)
        token = CONTROL_RE.sub("", raw).strip()
        if not token:
            continue
        # A written hyphen survives as a character; the custom-font hyphen the
        # extractor leaves as a control glyph is stripped from the token above
        # and only its flag marks the break.
        broke_at_hyphen = bool(_LINE_END_CONTROL_RE.search(raw))
        if not out:
            out = token
        elif out.endswith("-") and not out.endswith("--"):
            out = out[:-1] + token
        elif previous_broke_at_hyphen:
            out += token
        elif _line_break_joins_with_space(out[-1], token[0]):
            out += f" {token}"
        else:
            out += token
        previous_broke_at_hyphen = broke_at_hyphen
    return out


def _witness_spacing_is_usable(witness: str) -> bool:
    """Whether a witness's own spacing looks like laid-out prose, not tracking."""
    gaps = 0
    spaced = 0
    after_space = False
    for char in witness:
        if char.isspace():
            after_space = True
            continue
        gaps += 1
        if after_space:
            spaced += 1
        after_space = False
    return gaps >= 4 and spaced / max(1, gaps - 1) <= _WITNESS_MAX_SPACE_SHARE


def _char_tokens(text: str) -> list[tuple[str, str | None]]:
    """``(run of whitespace before the char, char)`` pairs, plus a tail run."""
    tokens: list[tuple[str, str | None]] = []
    gap = ""
    for char in text:
        if char.isspace():
            gap += char
        else:
            tokens.append((gap, char))
            gap = ""
    tokens.append((gap, None))
    return tokens


def reinsert_missing_spaces(text: str, witness: str) -> str | None:
    """``text`` with the spaces Docling dropped re-inserted from ``witness``.

    Only ever *adds* spaces, and only between two word characters -- never next
    to punctuation, so pdfium's own widened punctuation spacing ("( Gao et al.
    ,2019 )") is never imported. Returns ``None`` when the two texts are not the
    same character stream, so any other witness disagreement (a control glyph,
    a different quote) leaves the text untouched.
    """
    text_chars = [char for char in text if not char.isspace()]
    witness_chars = [char for char in witness if not char.isspace()]
    if not text_chars or text_chars != witness_chars:
        return None
    out: list[str] = []
    previous = ""
    for (gap, char), (witness_gap, _) in zip(
        _char_tokens(text), _char_tokens(witness), strict=True
    ):
        if char is None:
            out.append(gap)
            continue
        if not gap and witness_gap and previous and _joins_with_space(previous, char):
            out.append(" ")
        else:
            out.append(gap)
        out.append(char)
        previous = char
    return "".join(out)


#: Known character corruption mappings where IBM Docling mistranscribes math/TeX glyphs
#: to visually-similar but semantically-corrupt Unicode points, and PDFium physical glyph stream
#: holds the true ASCII / math character.
MATH_SYMBOL_CONFUSIONS: dict[str, set[str]] = {
    "∅": {"=", "≠", "<", ">", "≤", "≥"},  # Docling U+2205 EMPTY SET -> PDFium '='
    "⊕": {"+", "±"},  # Docling U+2295 CIRCLED PLUS -> PDFium '+'
    "ϒ": {"−", "-", "–", "—"},  # Docling U+03D2 GREEK UPSILON WITH HOOK -> PDFium '−' / '-'
    "′": {"·", "*", "×", "•"},  # Docling U+2032 PRIME -> PDFium '·' (middle dot)
    "∩": {"<", "≤"},  # Docling U+2229 INTERSECTION -> PDFium '<'
    "∪": {">", "≥"},  # Docling U+222A UNION -> PDFium '>'
    "ϕ": {"|", "l", "1", "/", "!"},  # Docling U+03D5 GREEK PHI -> PDFium '|'
    "≤": {"×", "*", "·"},  # Docling \le -> PDFium \times
}


def repair_math_symbol_corruptions(text: str, witness: str) -> str:
    """Repair Docling math glyph corruptions using the PDFium physical line witness.

    Aligns the non-whitespace character stream of ``text`` against ``witness``.
    When an alignment difference maps a known Docling corrupted symbol (e.g. '∅', '⊕', 'ϒ')
    to its PDFium ground-truth counterpart (e.g. '=', '+', '−'), the corrupted character
    in ``text`` is replaced in-place, preserving all original whitespace and surrounding layout.
    """
    if not any(c in MATH_SYMBOL_CONFUSIONS for c in text):
        return text

    doc_chars: list[tuple[int, str]] = [(i, c) for i, c in enumerate(text) if not c.isspace()]
    wit_chars: list[str] = [c for c in witness if not c.isspace()]
    if not doc_chars or not wit_chars:
        return text

    sm = difflib.SequenceMatcher(None, [c for _, c in doc_chars], wit_chars, autojunk=False)
    replacements: list[tuple[int, str]] = []

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "replace":
            d_slice = doc_chars[i1:i2]
            w_slice = wit_chars[j1:j2]
            if len(d_slice) == 1 and len(w_slice) == 1:
                orig_idx, d_char = d_slice[0]
                w_char = w_slice[0]
                if d_char in MATH_SYMBOL_CONFUSIONS and w_char in MATH_SYMBOL_CONFUSIONS[d_char]:
                    replacements.append((orig_idx, w_char))
            else:
                # Sub-align multi-char differences (e.g. adjacent formulas or superscript shifts)
                sub_sm = difflib.SequenceMatcher(
                    None, [c for _, c in d_slice], w_slice, autojunk=False
                )
                for s_tag, si1, si2, sj1, sj2 in sub_sm.get_opcodes():
                    if s_tag == "replace" and si2 - si1 == 1 and sj2 - sj1 == 1:
                        orig_idx, d_char = d_slice[si1]
                        w_char = w_slice[sj1]
                        if (
                            d_char in MATH_SYMBOL_CONFUSIONS
                            and w_char in MATH_SYMBOL_CONFUSIONS[d_char]
                        ):
                            replacements.append((orig_idx, w_char))

    if not replacements:
        return text

    chars = list(text)
    for orig_idx, w_char in replacements:
        chars[orig_idx] = w_char
    return "".join(chars)


#: Block types subject to math symbol witness repair
_MATH_REPAIR_TYPES = frozenset(
    {
        BlockType.NARRATIVE,
        BlockType.DIALOGUE,
        BlockType.HEADING,
        BlockType.LIST_ITEM,
        BlockType.CODE,
        BlockType.TABLE,
        BlockType.FORMULA,
    }
)


def repair_math_symbols_with_lines(blocks: Sequence[IRBlock], pdf_path: Path) -> list[IRBlock]:
    """Repair Docling math glyph corruptions using the page's PDFium line witness.

    IBM Docling's PDF parser occasionally maps TeX math font glyphs to look-alike
    Unicode characters (e.g. '=' to '∅', '+' to '⊕', '−' to 'ϒ', '·' to '′').
    PDFium's built-in TeX glyph heuristics preserve the correct character stream.
    This function aligns each block against intersecting PDFium lines and replaces
    the corrupted characters with their ground-truth equivalents.
    """
    path = Path(pdf_path)
    lines_by_page: dict[int, list[LineBox]] = {}

    def _page_lines(page: int) -> list[LineBox]:
        if page not in lines_by_page:
            try:
                lines_by_page[page] = list(extract_lines(path, page)[0])
            except Exception as exc:
                logger.debug(
                    "Math symbol repair: pdfium line extraction failed on p%d: %s", page, exc
                )
                lines_by_page[page] = []
        return lines_by_page[page]

    for block in blocks:
        if block.block_type not in _MATH_REPAIR_TYPES:
            continue
        text = block.source_text
        if not text or not any(c in MATH_SYMBOL_CONFUSIONS for c in text):
            continue

        seen: set[tuple[float, float, float, float, str]] = set()
        witness_lines: list[LineBox] = []
        for pbox in _span_boxes(block):
            for line in _page_lines(pbox.page):
                if not line.text.strip() or not _line_centered_in_box(line.rect, pbox.bbox):
                    continue
                key = (
                    round(line.rect[0], 1),
                    round(line.rect[1], 1),
                    round(line.rect[2], 1),
                    round(line.rect[3], 1),
                    line.text,
                )
                if key in seen:
                    continue
                seen.add(key)
                witness_lines.append(line)

        if not witness_lines:
            continue

        witness = join_witness_lines(witness_lines)
        if not witness:
            continue

        repaired = repair_math_symbol_corruptions(text, witness)
        if repaired != text:
            logger.info(
                "Repaired math symbols in block %s from pdfium lines: %r -> %r",
                block.id,
                text[:40],
                repaired[:40],
            )
            block.set_source_text(repaired)
            block.provenance.math_symbol_repair = "pdfium-line-witness"

    return list(blocks)


def _span_boxes(block: IRBlock) -> tuple[PhysicalBox, ...]:
    """The block's own boxes (its chain when it has one), else its single box."""
    span = block.element.span
    if isinstance(span, CompositeSpan) and span.boxes:
        return span.boxes
    box = block.bbox
    if box is None or box.page <= 0 or box.x1 <= box.x0 or box.y1 <= box.y0:
        return ()
    return (PhysicalBox.of(box.page, (box.x0, box.y0, box.x1, box.y1)),)


def _line_centered_in_box(
    rect: tuple[float, float, float, float], box: tuple[float, float, float, float]
) -> bool:
    center_y = (rect[1] + rect[3]) / 2.0
    if not (box[1] - 1.0 <= center_y <= box[3] + 1.0):
        return False
    return min(rect[2], box[2]) - max(rect[0], box[0]) > 1.0


def repair_missing_spaces_with_lines(blocks: Sequence[IRBlock], pdf_path: Path) -> list[IRBlock]:
    """Re-insert the inter-word spaces Docling's PDF line join drops.

    Docling concatenates a paragraph's lines without the space they were
    separated by when one side is a single letter ("A Sandbox" -> "ASandbox",
    "A single" -> "Asingle") or a run boundary ("We use AppArmor profiles" ->
    "WeuseAppArmorprofiles"). The glued token reaches the model as one word and
    is mistranslated ("ASandbox" reads as "A-type sandbox").

    The page's own pdfium glyph lines are the witness: when they spell exactly
    the same character stream as the block's text, their spacing is adopted for
    the gaps Docling left empty. Anything else (a witness that disagrees at all,
    a layout whose spacing is not prose-like) leaves the block byte-identical.
    """
    path = Path(pdf_path)
    lines_by_page: dict[int, list[LineBox]] = {}

    def _page_lines(page: int) -> list[LineBox]:
        if page not in lines_by_page:
            try:
                lines_by_page[page] = list(extract_lines(path, page)[0])
            except Exception as exc:
                logger.debug("Space repair: pdfium line extraction failed on p%d: %s", page, exc)
                lines_by_page[page] = []
        return lines_by_page[page]

    for block in blocks:
        if block.skip_translate or block.block_type not in _VERIFIABLE_TYPES:
            continue
        text = block.source_text
        if not text.strip():
            continue
        seen: set[tuple[float, float, float, float, str]] = set()
        witness_lines: list[LineBox] = []
        for pbox in _span_boxes(block):
            for line in _page_lines(pbox.page):
                if not line.text.strip() or not _line_centered_in_box(line.rect, pbox.bbox):
                    continue
                key = (
                    round(line.rect[0], 1),
                    round(line.rect[1], 1),
                    round(line.rect[2], 1),
                    round(line.rect[3], 1),
                    line.text,
                )
                if key in seen:
                    continue
                seen.add(key)
                witness_lines.append(line)
        if not witness_lines:
            continue
        witness = join_witness_lines(witness_lines)
        if not _witness_spacing_is_usable(witness):
            continue
        repaired = reinsert_missing_spaces(text, witness)
        if repaired is None or repaired == text:
            continue
        logger.info(
            "Repaired %d missing space(s) in block %s from the page's pdfium lines",
            repaired.count(" ") - text.count(" "),
            block.id,
        )
        block.set_source_text(repaired)
        block.provenance.space_repair = "pdfium-line-witness"

    return list(blocks)


def cross_check_blocks_with_pdfium(
    blocks: Sequence[IRBlock],
    pdf_path: Path,
    *,
    min_iou: float = DEFAULT_MIN_IOU,
) -> list[IRBlock]:
    """Verify Docling text block bounding boxes against PDFium glyph lines.

    For each prose text block on a born-digital page:
    1. Extracts cached PDFium lines for the page.
    2. If the page is textless (scanned image), skips verification (left to OCR/VLM).
    3. Finds PDFium lines intersecting the block's bounding box.
    4. Computes the IoU between the block's bbox and the union of intersecting lines.
    5. If IoU < min_iou (or zero intersecting lines for substantial text), marks the
       block with Confidence.UNKNOWN and skip_translate=True so it sinks to
       PRESERVED_OPAQUE on Layer 0.
    """
    path = Path(pdf_path)
    output: list[IRBlock] = []
    lines_by_page: dict[int, list[LineBox]] = {}
    chars_by_page: dict[int, list[CharStyle]] = {}
    cache_page = 0

    for block in blocks:
        if block.block_type not in _VERIFIABLE_TYPES or block.skip_translate:
            output.append(block)
            continue
        bbox = block.bbox
        if bbox is None or bbox.page <= 0:
            output.append(block)
            continue

        # Bound the per-page caches to the page in flight. Without this they
        # held every page's lines and char styles for the whole book (hundreds
        # of MB on a long scan). Blocks arrive in reading order, so a forward
        # page step drops what is behind; an out-of-order block simply
        # re-extracts its page.
        if bbox.page > cache_page:
            cache_page = bbox.page
            for cache in (lines_by_page, chars_by_page):
                for stale in [p for p in cache if p < cache_page]:
                    del cache[stale]

        span = block.element.span
        eval_boxes = (
            span.boxes
            if isinstance(span, CompositeSpan)
            else (PhysicalBox.of(bbox.page, (bbox.x0, bbox.y0, bbox.x1, bbox.y1)),)
        )

        failed_reason: str | None = None
        text_len = len(block.source_text.strip())
        all_inter_lines: list[LineBox] = []

        for pbox in eval_boxes:
            p_page = pbox.page
            if p_page not in lines_by_page:
                try:
                    page_lines, _ = extract_lines(path, p_page)
                    lines_by_page[p_page] = page_lines
                except Exception as exc:
                    logger.debug(
                        "PDFium line extraction failed for cross-check page %d: %s",
                        p_page,
                        exc,
                    )
                    lines_by_page[p_page] = []

            page_lines = lines_by_page[p_page]
            if not page_lines:
                # Scanned / raster-only page: no born-digital lines to cross-check against.
                continue

            doc_box = pbox.bbox
            if box_area(*doc_box) <= 0:
                continue

            inter_lines = [line for line in page_lines if _boxes_intersect(doc_box, line.rect)]
            if not inter_lines:
                if text_len > 10:
                    failed_reason = f"no_physical_lines(p{p_page})"
                    break
                continue

            all_inter_lines.extend(inter_lines)
            ux0 = min(line.rect[0] for line in inter_lines)
            uy0 = min(line.rect[1] for line in inter_lines)
            ux1 = max(line.rect[2] for line in inter_lines)
            uy1 = max(line.rect[3] for line in inter_lines)
            iou = box_iou(doc_box, (ux0, uy0, ux1, uy1))

            effective_min_iou = 0.40 if text_len <= 5 else min_iou
            if iou < effective_min_iou:
                failed_reason = f"iou_mismatch(p{p_page}: iou={iou:.2f} < {effective_min_iou:.2f})"
                break

        if failed_reason is not None:
            logger.info(
                "Block %s on page %d failed cross-check (%s); sinking to PRESERVED_OPAQUE",
                block.id,
                bbox.page,
                failed_reason,
            )
            fused_elem = dataclasses.replace(
                block.element, confidence=Confidence.UNKNOWN, skip_translate=True
            )
            demoted = IRBlock(element=fused_elem)
            demoted.skip_translate = True
            demoted.policy_translate = False
            demoted.provenance = block.provenance.model_copy(
                update={"iou_crosscheck": f"opaque: {failed_reason}"}
            )
            demoted.error_flags = list(block.error_flags) + [
                f"skip:preserved_opaque({failed_reason})"
            ]
            output.append(demoted)
            continue

        if all_inter_lines:
            try:
                from ubt.adapters.pdf.textgeom import _aggregate_line_styles
                from ubt.core.ir.models import InlineRun, StyleMeta

                fsz, is_bold, is_italic = _aggregate_line_styles(all_inter_lines)
                if fsz >= 4.5:
                    # Preserve any style already found (e.g. the first-line indent
                    # the parser recorded): a bare StyleMeta would drop it.
                    block.style = (block.style or StyleMeta()).model_copy(update={"font_size": fsz})
                    block.provenance.font_size = fsz
                if is_bold:
                    block.provenance.is_bold = True
                if is_italic:
                    block.provenance.is_italic = True
            except Exception as exc:
                logger.debug("Failed to extract line styles for block %s: %s", block.id, exc)

        # Inline runs: a colour change (a blue citation) or a raised marker (a
        # footnote dagger) lives on individual characters, which the line-level
        # aggregate above cannot see. Probe them so the render can re-apply the
        # style where the span survives translation verbatim.
        probe_page = bbox.page
        if probe_page not in chars_by_page:
            try:
                chars_by_page[probe_page] = extract_char_styles(path, probe_page)
            except Exception as exc:
                logger.debug("PDFium char probe failed for page %d: %s", probe_page, exc)
                chars_by_page[probe_page] = []
        runs = styled_runs_in_box(chars_by_page[probe_page], (bbox.x0, bbox.y0, bbox.x1, bbox.y1))
        if runs:
            inline = tuple(
                InlineRun(text=text, bold=bold, italic=italic, superscript=super_, color_hex=color)
                for text, bold, italic, super_, color in runs
            )
            block.style = (block.style or StyleMeta()).model_copy(update={"inline_runs": inline})

        output.append(block)

    return output


__all__ = [
    "DEFAULT_MIN_IOU",
    "MATH_SYMBOL_CONFUSIONS",
    "cross_check_blocks_with_pdfium",
    "join_witness_lines",
    "reinsert_missing_spaces",
    "repair_math_symbol_corruptions",
    "repair_math_symbols_with_lines",
    "repair_missing_spaces_with_lines",
    "witness_line_text",
]
