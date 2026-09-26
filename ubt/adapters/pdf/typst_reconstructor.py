"""Modern publication-grade Typst document reconstructor.

Converts translated IR blocks into clean Typst markup and compiles them
into high-aesthetic vector PDFs without squished 'ant fonts'.

Fragment split: math conversion lives in :mod:`ubt.adapters.pdf.typst_math`,
table/text/image emission in :mod:`ubt.adapters.pdf.typst_fragments` — this
module keeps the orchestrating :class:`TypstReconstructor` and re-exports
those helpers so existing import paths keep working.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import re
import shutil
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ubt.adapters.pdf.font_probe import resolve_font_stack
from ubt.adapters.pdf.formula_tags import recover_formula_tag
from ubt.adapters.pdf.typst_constants import (
    _FLOWING_STYLE,
    _FORMULA_ID_RE,
    _LIST_MARKERS,
    _MISCLASSIFIED_CAPTION_RE,
    _PAGE_STYLE,
    StyleProfile,
)
from ubt.adapters.pdf.typst_fragments import (
    _cell_to_typst,
    _decouple_inline_box_calls,
    _display_width,
    _escape_typst_markup,
    _footer_display_title,
    _image_size_spec,
    _markdown_table_to_typst,
    _prose_to_typst,
    _reference_numbers,
    _sanitize_image_ref,
    _stage_image_assets,
    _svg_native_size,
    sanitize_lang_tag,
)
from ubt.adapters.pdf.typst_healer import (
    TypstDiagnosticHealer,
    _degrade_failing_line,
    _heal_persistent_comment_error,
    _quote_unknown_var_on_line,
)
from ubt.adapters.pdf.typst_math import (
    _MATH_FIT_GLYPHS,
    _MATH_FIT_GLYPHS_SMALL,
    _MATH_SHRINK_SIZE_PT,
    _TYPST_ROW_SPLIT_RE,
    _balanced_delimiters,
    _clean_ocr_formula,
    _emit_formula_math,
    _latex_math_to_typst,
    _latex_math_to_typst_regex,
    _math_display_len,
    _normalize_docling_math,
    _pandoc_math_to_typst,
)
from ubt.core.ir.models import BlockStatus, BlockType, FlowID, IRBlock, LayoutRole
from ubt.core.language_profile import resolve_font_config

logger = logging.getLogger(__name__)

# Cover-page decision is structural, never heuristic: page 1 is a cover only
# when it carries no body content (headings / images / meta only). Any page
# with narrative, list, table, formula or code text is an interior page, so
# single-page documents and chapter pages with body text always translate
# as content. Explicit "always"/"never" overrides come from cover_mode.
_BODY_BLOCK_TYPES = frozenset(
    {
        BlockType.NARRATIVE,
        BlockType.DIALOGUE,
        BlockType.LIST_ITEM,
        BlockType.TABLE,
        BlockType.FORMULA,
        BlockType.CODE,
    }
)

_VALID_COVER_MODES = frozenset({"auto", "always", "never"})

#: Cover-page subtitle/description cuts, as fractions of the page height. These
#: reproduce the original A4-absolute thresholds (300pt / 200pt on 842pt) while
#: scaling to any page size; see :meth:`TypstReconstructor._emit_cover_page`.
_COVER_SUBTITLE_FRAC = 300.0 / 842.0
_COVER_DESC_FRAC = 200.0 / 842.0


def normalize_cover_mode(value: object) -> str:
    """Coerce a cover_mode value to auto/always/never (unknown -> auto)."""
    if isinstance(value, str) and value.strip().lower() in _VALID_COVER_MODES:
        return value.strip().lower()
    logger.warning("Unknown cover_mode=%r, falling back to 'auto'", value)
    return "auto"


# Raster scale for the formula image fallback. A display equation is small
# type, so the visual-scapel default (150 dpi) reads soft once Typst scales it;
# 300 dpi keeps subscripts legible at roughly 35 KB per equation.
_FORMULA_FALLBACK_DPI = 300

# Upper bound for the fallback graphic's rendered width. A4 text columns run
# ~493pt wide, so 460pt keeps even a pathological bbox inside the margins.
_FORMULA_FALLBACK_MAX_WIDTH_PT = 460.0

# Engine backend: the witness compares the MathJax raster and
# the source crop at the same 150 dpi the deterministic witness calibrated on.
_ENGINE_WITNESS_DPI = 150
# Equation numbers next to engine-rendered formulas; matches the 10.5pt body.
_FORMULA_NUMBER_SIZE_PT = 9.5

# Inline engine formulas: natural size is used (em units from the viewBox), but
# a span wider than this many em is capped to the text width instead of
# overfull-ing the line.
_INLINE_MAX_EM = 28.0
# MathJax reports the descent of an inline SVG as a CSS vertical-align in ex
# units; Typst has no x-height unit, so it is converted with the usual
# approximation 1ex = 0.5em (serif x-heights sit at 0.45-0.55em; a few
# percent of baseline shift is invisible next to the text).
_EX_TO_EM = 0.5
# MathJax drops the unit for an exactly-on-baseline span ("vertical-align:
# 0;"), so the optional ex is part of the contract, not a nicety.
_SVG_VA_RE = re.compile(r'vertical-align:\s*(-?[\d.]+)(?:ex)?[;"]')

# Standalone page chrome: reflowed page numerals / lone digits ("3", "69",
# "- 69 -"). Digits + punctuation only, at most 8 chars — never body prose.
_PAGE_CHROME_RE = re.compile(r"^[\d\W]{1,8}$")


def _is_page_chrome(text: str) -> bool:
    """True for lone page-numeral paragraphs that reflow would strand as body."""
    return bool(_PAGE_CHROME_RE.match(text.strip()))


# StyleProfile / _FLOWING_STYLE / _PAGE_STYLE (the emitter style constants)
# and _LIST_MARKERS / _MISCLASSIFIED_CAPTION_RE (the shared emitter patterns)
# are imported from typst_constants — the single definition site shared with
# the rest of the Typst pipeline. See StyleProfile's drift record there.


@dataclass
class _EmitContext:
    """Per-path state threaded through the shared ``_emit_block`` core.

    ``None`` (the default for direct ``_emit_block`` calls and the flowing
    layout) means "flowing profile, no page context"; ``_emit_interior_page``
    creates one context per page and mutates ``index`` / ``first_heading_open``
    across the page's blocks.
    """

    profile: StyleProfile = _FLOWING_STYLE
    siblings: Sequence[IRBlock] = ()
    index: int = -1
    page_num: int = 0
    first_heading_open: bool = True


_FIG_LABEL_RE = re.compile(r"^\s*(FIG\.?|Fig\.?|FIGURE)\s*([A-Za-z0-9][\w.\-]*)", re.IGNORECASE)
_TABLE_LABEL_RE = re.compile(r"^\s*(TABLE|TAB\.?)\s*([A-Za-z0-9][\w.\-]*)", re.IGNORECASE)
_TILDE_MANGLE_RE = re.compile(r"([A-Za-z\)\]])\s*¼\s*(\d)")
_ARROW_ASCII_RE = re.compile(r"->")
_ECH_MANGLE_RE = re.compile(r"(?<![A-Za-z])ech\b")
_TOX_SPACED_RE = re.compile(r"\b([Tt])\s+ox\b")


def _polish_target_text(s: str, target_lang: str = "zh") -> str:
    """Deterministic post-fixes the small model reliably misses.

    - ``FIG. 3.1`` / ``TABLE 3.1`` caption labels → localized prefix (e.g.
      ``图 3.1`` / ``表 3.1``, ``図 3.1`` / ``表 3.1``, ``Fig. 3.1`` /
      ``Table 3.1``).
      Line-start rigid; already-localized lines pass through untouched.
    - Glyph confusion ``X ¼ 5`` → ``X～5``: certain tokenizers render ``~``
      as ``¼`` between a token and a digit. Scoped to letter/paren-glued
      positions so genuine vulgar fractions (``¼ cup``, ``a ¼ share``)
      never match.
    - ASCII arrows ``->`` → ``→`` (prose only; code/formula branches never
      call this helper, so ``a->b`` member access stays intact).
    - Romanized dielectrics ``ech`` → ``εch``: the model spells out the
      epsilon. Word-boundary guarded so ``tech`` never matches.
    - Spaced oxide ``t ox`` → ``tox`` (model tokenization artifact).
    - Cleans residual XML/Markdown boundary tags, orphan blockquotes, and OCR artifacts.
    """
    lang_fonts = resolve_font_config(target_lang)
    s = _FIG_LABEL_RE.sub(lambda m: f"{lang_fonts.figure_prefix} {m.group(2)}", s, count=1)
    s = _TABLE_LABEL_RE.sub(lambda m: f"{lang_fonts.table_prefix} {m.group(2)}", s, count=1)
    s = _TILDE_MANGLE_RE.sub(r"\1～\2", s)
    s = _ARROW_ASCII_RE.sub("→", s)
    s = _ECH_MANGLE_RE.sub("εch", s)
    s = s.replace("\u2011", "-")
    s = _TOX_SPACED_RE.sub(r"\1ox", s)
    # Strip prompt instruction leakages if present
    s = re.sub(
        r"(?:\n\s*)?Translate only the text under this heading.*$",
        "",
        s,
        flags=re.IGNORECASE | re.DOTALL,
    )
    s = re.sub(
        r"(?:\n\s*)?Provide the direct translation under this heading.*$",
        "",
        s,
        flags=re.IGNORECASE | re.DOTALL,
    )
    # Strip stray LLM tags and orphan blockquote markers
    s = re.sub(r"</?(?:translation|final_translation)[^>]*>", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^\s*>\s*", "", s)
    # Strip leaked mask tokens (preserving genuine Scott semantic brackets like ⟦e⟧)
    s = re.sub(r"⟦(?:UBT:[A-Z]+:|[A-Z]+_MASK_|[A-Z]+_)\d{1,6}(?:-[0-9a-z]{3})?⟧", "", s)
    # Strip mock prefix
    s = re.sub(r"^\[(?:模拟翻译|Mock\s*Translation)\]\s*", "", s, flags=re.IGNORECASE)
    # Strip quarantine badge from publication document if leaked
    s = re.sub(r"【待人工审校\s*\|\s*Human review required】\s*", "", s)
    s = re.sub(r"</?mark\b[^>]*>", "", s)
    return s.strip()


def _resolve_content(block: IRBlock) -> tuple[str, str, str]:
    """Fail-closed (content, target, source) triple for one block.

    Terminally-failed blocks still carry a usable draft (QE-killed, not
    empty): prefer it over raw source so Chinese books never sprout
    full-English paragraphs while discarding real translations.
    """
    text = (block.target_text or "").strip()
    draft = (block.draft_text or "").strip()
    source = (block.source_text or "").strip()

    # Structured blocks (IMAGE, FORMULA, CODE, TABLE) are syntax/assets that must not be wrapped in review markers
    if block.block_type in (BlockType.IMAGE, BlockType.FORMULA, BlockType.CODE, BlockType.TABLE):
        effective = text or draft or source
        return (effective, text, source)

    if block.status == BlockStatus.BLOCKED_HUMAN:
        # Check if draft is merely a mock string like "[模拟翻译] ..." or contains prompt leakage
        is_mock_draft = bool(
            draft and ("[模拟翻译]" in draft or "Translate only the text" in draft)
        )
        if is_mock_draft or not draft:
            effective = f"【待审校: {source}】" if source else ""
        else:
            effective = draft
        return (effective, text, source)
    effective = text or draft
    if not effective and source:
        # Prevent silent raw English leakage for un-drafted failed prose blocks
        effective = f"【待审校: {source}】"
    return (effective, text, source)


_FOOTNOTE_LEADING_MARKER_RE = re.compile(
    r"^[\s\[(（]*(?:(\d{1,4})|[\*†‡#]|注[:：]?|note[:：]?)[\s\])）\.:-]*",
    re.IGNORECASE,
)
_SUPERSCRIPT_MAP = {
    "0": "⁰",
    "1": "¹",
    "2": "²",
    "3": "³",
    "4": "⁴",
    "5": "⁵",
    "6": "⁶",
    "7": "⁷",
    "8": "⁸",
    "9": "⁹",
}
# Tag grammar shared by the reference patterns: an optional uppercase
# appendix prefix, a number and an optional sub-equation letter ("A.12a").
_EQ_TAG = r"([A-Z]?\s*\.?\s*[0-9]+(?:\s*\.\s*[0-9]+)?\s*[a-z]?)"
_EQ_REF_PATTERNS = (
    # Chinese: 式 (1), 式(1), 公式 (1), 公式(1), 式 1, 公式 1, 式 (A.12a)
    re.compile(rf"((?:式|公式)\s*[\(（]{_EQ_TAG}[\)）])"),
    re.compile(rf"((?:式|公式)\s+{_EQ_TAG})\b"),
    # English: Equation (1), Eq. (1), Formula (1), Equation 1, Eq. 1
    re.compile(rf"(\b(?:Eq\.|Equation|Formula)\s*[\(（]?{_EQ_TAG}[\)）]?)", re.IGNORECASE),
)


def _format_footnote_markup(
    clean_tgt: str,
    clean_src: str = "",
    bilingual: bool = False,
) -> str:
    """Format a Typst native #footnote[...] callout."""
    if bilingual and clean_src and clean_tgt and clean_src != clean_tgt:
        esc_src = _prose_to_typst(clean_src)
        esc_tgt = _prose_to_typst(clean_tgt)
        return (
            f'#footnote[#text(size: 8pt, fill: rgb("#94a3b8"))[{esc_src}] \\ '
            f'#text(size: 8.5pt, fill: rgb("#64748b"))[{esc_tgt}]]'
        )
    esc_tgt = _prose_to_typst(clean_tgt)
    return f'#footnote[#text(size: 8.5pt, fill: rgb("#64748b"))[{esc_tgt}]]'


def _parse_footnote_block(
    blk: IRBlock,
    target_lang: str = "zh",
) -> tuple[str | None, str, str]:
    """Extract (marker_num, clean_content, clean_source) from a footnote block."""
    content, text, source = _resolve_content(blk)
    content = _polish_target_text(content, target_lang=target_lang)
    m_content = _FOOTNOTE_LEADING_MARKER_RE.match(content)
    m_source = _FOOTNOTE_LEADING_MARKER_RE.match(source)
    m = m_content or m_source
    marker_num: str | None = None
    clean_content = content
    clean_source = source
    if m:
        marker_num = m.group(1)
        # Each side is sliced by its OWN match: a marker present only in the
        # source (the translator dropped the leading number) must not have its
        # source offset applied to the target, which would eat leading target
        # characters.
        if m_content:
            clean_content = content[m_content.end() :].strip() or content
        if m_source:
            clean_source = source[m_source.end() :].strip() or source
    return marker_num, clean_content, clean_source


def _attach_footnote_to_prose(esc_prose: str, marker_num: str | None, fn_callout: str) -> str:
    """Attach a native footnote callout to escaped prose, replacing anchor if found."""
    if marker_num:
        # 1. Escaped bracketed marker: \[1\]
        pat1 = re.compile(r"\\\[\s*" + re.escape(marker_num) + r"\s*\\\]")
        if pat1.search(esc_prose):
            return pat1.sub(fn_callout, esc_prose, count=1)

        # 2. Parenthesized marker: (1) or （1）
        pat2 = re.compile(r"[\(（]\s*" + re.escape(marker_num) + r"\s*[\)）]")
        if pat2.search(esc_prose):
            return pat2.sub(fn_callout, esc_prose, count=1)

        # 3. Unicode superscript: ¹, ², etc.
        sup_marker = "".join(_SUPERSCRIPT_MAP.get(c, c) for c in marker_num)
        if sup_marker in esc_prose:
            return esc_prose.replace(sup_marker, fn_callout, 1)

    # Fallback: append to the end of the narrative paragraph
    return esc_prose.rstrip() + fn_callout


def _apply_cross_refs(text: str, formula_map: Mapping[str, str] | None) -> str:
    """Rewrite static equation references to dynamic Typst cross-refs if label exists."""
    if not formula_map:
        return text
    result = text
    for pat in _EQ_REF_PATTERNS:

        def _replace_match(m: re.Match[str]) -> str:
            full_span = m.group(1)
            num = re.sub(r"\s+", "", m.group(2))
            if num in formula_map:
                label = formula_map[num]
                # `#ref(<…>)` instead of `@label`: a bare @label followed
                # directly by CJK text absorbs it into the label name and
                # Typst then fails the whole line (chapter-3 appendix refs).
                if full_span.startswith(("式", "公式")):
                    prefix = "公式 " if full_span.startswith("公式") else "式 "
                    return f"{prefix}#ref(<{label}>)"
                else:
                    return f"#ref(<{label}>)"
            return full_span

        result = pat.sub(_replace_match, result)
    return result


def _build_formula_map(blocks: Sequence[IRBlock]) -> dict[str, str]:
    """Map formula sequential numbers, block IDs, and LaTeX tags to Typst labels."""
    f_map: dict[str, str] = {}
    f_idx = 1
    for b in blocks:
        if b.block_type == BlockType.FORMULA:
            clean_id = re.sub(r"[^a-zA-Z0-9_\-]", "-", b.id).strip("-")
            label = f"eq-{clean_id}" if clean_id else f"eq-{f_idx}"
            f_map[str(f_idx)] = label
            f_map[b.id] = label
            raw = (b.target_text or b.draft_text or b.source_text or "").strip()
            tag = _extract_formula_tag(raw)
            if tag:
                f_map[tag] = label
            f_idx += 1
    return f_map


def _formula_label(block: IRBlock) -> str:
    """Dynamic Typst label for a formula block, mirroring _emit_formula_line.

    Image-backed formulas (engine SVG or source crop) keep the same
    ``<eq-…>`` label native formulas carry so in-text ``#ref`` links resolve.
    It must sit directly after the content element: a label after
    ``#counter(...).step()`` attaches to the counter update and Typst then
    rejects every reference with "cannot reference counter-update".
    """
    clean_id = re.sub(r"[^a-zA-Z0-9_\-]", "-", block.id).strip("-")
    return f" <eq-{clean_id}>" if clean_id else ""


def _extract_formula_tag(raw: str) -> str | None:
    """Extract explicit equation tag e.g. \\tag{3.1}, (3.1) or ( A . 12a ).

    Docling OCR separates every token with spaces ("( 3 . 1 4 )"), so a
    second candidate pass normalizes whitespace before validating. Pure
    numeric spaced tags stay on the chapter counter (the body already
    matches the source that way); only labels carrying a letter — the
    appendix's A.12a-style numbering — are promoted to explicit tags.
    """
    if not raw:
        return None
    tag_m = re.search(
        r"(?:\\tag\{\s*|(?:&&?|\\quad|\\qquad)\s*\(\s*|(?<![+\-*=/,<>\^~:])\s+\(\s*)([0-9]+(?:\.[0-9]+)?)\s*(?:[\}]|\)(?:\s*$|(?=\s*\\\\)))",
        raw,
    )
    if tag_m:
        return tag_m.group(1)
    candidate = re.search(
        r"(?:\\tag\{\s*|(?:&&?|\\quad|\\qquad)\s*\(\s*|(?<![+\-*=/,<>\^~:])\s+\(\s*)([A-Za-z0-9][A-Za-z0-9\s.]{0,11}?)\s*(?:[\}]|\)(?:\s*$|(?=\s*\\\\)))",
        raw,
    )
    if candidate is None:
        return None
    tag = re.sub(r"\s+", "", candidate.group(1))
    # Uppercase prefix only: the appendix writes A.12a. A lowercase prefix
    # would also swallow OCR body artifacts such as "( a . 1 )".
    if not re.fullmatch(r"(?:[A-Z]\.)?[0-9]+(?:\.[0-9]+)?[a-z]?", tag):
        return None
    if not any(ch.isalpha() for ch in tag):
        return None
    return tag


_CN_NUM_MAP = {
    "一": 1,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
    "十一": 11,
    "十二": 12,
    "十三": 13,
    "十四": 14,
    "十五": 15,
    "十六": 16,
    "十七": 17,
    "十八": 18,
    "十九": 19,
    "二十": 20,
}


def _extract_chapter_number(text: str) -> int | None:
    """Extract chapter number from Chinese or English heading (e.g. '第三章', 'Chapter 3', '3.1')."""
    if not text:
        return None
    # e.g. "第三章", "第 3 章"
    m = re.search(r"第\s*([0-9]+|[一二三四五六七八九十]+)\s*章", text)
    if m:
        val = m.group(1)
        if val.isdigit():
            return int(val)
        return _CN_NUM_MAP.get(val)
    # e.g. "Chapter 3", "CHAPTER 3"
    m = re.search(r"\bchapter\s*([0-9]+)\b", text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    # e.g. "3. 紧凑建模" or "3 紧凑建模"
    m = re.match(r"^\s*([1-9][0-9]?)\s*[\.、\s]", text)
    if m:
        return int(m.group(1))
    return None


def _is_decorative_chapter_banner(
    blk: IRBlock,
    blocks: Sequence[IRBlock],
    idx: int,
    asset_path: str = "",
) -> bool:
    """Detect ornamental chapter header banners (e.g. blue CHAPTER bar) to prevent English ornament leaks."""
    # 1. Check if source/target text carries an explicit "chapter" label (e.g. Docling OCR on banner)
    raw_text = (blk.source_text or blk.target_text or "").strip().lower()
    # For an IMAGE block this field is the asset *path* (the caller passes the
    # same string as ``asset_path``), so a path containing "chapter" must not be
    # read as a banner label.
    looks_like_path = (
        raw_text.endswith((".png", ".jpg", ".jpeg", ".webp", ".svg", ".gif", ".bmp"))
        or "/" in raw_text
        or "\\" in raw_text
    )
    if (
        raw_text
        and not looks_like_path
        and ("chapter" in raw_text or "chap." in raw_text)
        and len(raw_text) <= 40
    ):
        return True

    # 2. Must be close to a chapter heading (level 1 or matching chapter pattern)
    has_adjacent_heading = False
    for offset in (-2, -1, 1, 2):
        pos = idx + offset
        if 0 <= pos < len(blocks) and blocks[pos].block_type == BlockType.HEADING:
            has_adjacent_heading = True
            break
    if not has_adjacent_heading:
        return False

    # 3. Check if there is an explicit figure caption attached (real content figures have captions)
    for offset in (-1, 1, 2):
        pos = idx + offset
        if 0 <= pos < len(blocks):
            txt = (blocks[pos].source_text or "").strip().lower()
            if txt.startswith(("fig", "figure", "图")):
                return False

    # 4. Check bounding box or image aspect ratio
    if blk.bbox:
        width = blk.bbox.x1 - blk.bbox.x0
        height = blk.bbox.y1 - blk.bbox.y0
        # Aspect-only: a bare ``height <= 90`` dropped every small (but real)
        # figure sitting near a heading, whatever its shape.
        if height > 0 and width / height >= 2.0:
            return True

    if asset_path and Path(asset_path).exists():
        try:
            from PIL import Image

            with Image.open(asset_path) as img:
                w, h = img.size
                if h > 0 and (w / h >= 2.0 or h <= 90):
                    return True
        except Exception as exc:
            # The aspect probe decides decorative-vs-content; an unreadable
            # asset silently falls back to "content". Say so, don't swallow.
            logger.warning(
                "typst_reconstructor: asset aspect probe failed for %s (%s)", asset_path, exc
            )

    return False


def _emit_formula_line(block: IRBlock, content: str, tag_hint: str | None = None) -> str:
    """Convert one formula block; tag ``$`` lines for probe mapping.

    Auto-injects dynamic label `<eq-{clean_id}>` for dynamic Typst cross-referencing.
    When an explicit equation tag (e.g. (3.1)) is present, wraps the equation in
    #math.equation(block: true, numbering: _ => "({tag})")[...] so original numbering is preserved.
    The trailing id comment lets the compile-probe pass map failures back
    to the source block for verbatim degradation.
    """
    raw = (block.target_text or block.draft_text or block.source_text or "").strip()
    tag = _extract_formula_tag(raw) or _extract_formula_tag(content) or tag_hint
    math_line = _emit_formula_math(content, block.id)
    clean_id = re.sub(r"[^a-zA-Z0-9_\-]", "-", block.id).strip("-")
    label = f"<eq-{clean_id}>" if clean_id else ""

    if math_line.lstrip().startswith("$"):
        # Shrink band: fit is decided per ROW (a 2-row equation is as wide as
        # its widest row, not the sum). Rows that miss the body-size budget
        # but fit one step down keep the author's breaks at the smaller size
        # instead of being re-derived into more rows or overflowing.
        stripped = math_line.strip()
        inner = stripped[1:-1] if len(stripped) >= 2 else stripped
        row_widths = [
            _math_display_len(r) for r in _TYPST_ROW_SPLIT_RE.split(inner) if r.strip()
        ] or [0]
        shrink = _MATH_FIT_GLYPHS < max(row_widths) <= _MATH_FIT_GLYPHS_SMALL
        if tag:
            # Preserve original extracted equation numbering (e.g. (3.1))
            math_line = f'#math.equation(block: true, numbering: _ => "({tag})")[{math_line}]'
        if shrink:
            if label:
                math_line = f"{math_line} {label}"
                label = ""
            math_line = (
                "#block[#show math.equation: "
                f"set text(size: {_MATH_SHRINK_SIZE_PT}pt); {math_line}]"
            )
        if label:
            math_line = f"{math_line} {label}  // [formula {block.id}]"
        else:
            math_line = f"{math_line}  // [formula {block.id}]"
    elif math_line.lstrip().startswith("#math.equation"):
        if label:
            math_line = f"{math_line} {label}  // [formula {block.id}]"
        else:
            math_line = f"{math_line}  // [formula {block.id}]"
    elif math_line.lstrip().startswith("`"):
        # Gate 4 refused this formula outright, so it never becomes a `$` line
        # and _verify_math_lines() — which only scans `$` / #math.equation —
        # will not see it. Tag it anyway so the render pass can still swap in
        # the source graphic instead of shipping a wall of raw text
        # (see TypstReconstructor._substitute_verbatim_formulas).
        math_line = f"{math_line}  // [formula {block.id}]"
    return math_line


__all__ = [
    "TypstReconstructor",
    "_apply_cross_refs",
    "_attach_footnote_to_prose",
    "_balanced_delimiters",
    "_build_formula_map",
    "_cell_to_typst",
    "_clean_ocr_formula",
    "_decouple_inline_box_calls",
    "_degrade_failing_line",
    "_display_width",
    "_emit_formula_line",
    "_emit_formula_math",
    "_escape_typst_markup",
    "_footer_display_title",
    "_format_footnote_markup",
    "_heal_persistent_comment_error",
    "_image_size_spec",
    "_latex_math_to_typst",
    "_latex_math_to_typst_regex",
    "_markdown_table_to_typst",
    "_normalize_docling_math",
    "_pandoc_math_to_typst",
    "_parse_footnote_block",
    "_prose_to_typst",
    "_quote_unknown_var_on_line",
    "_reference_numbers",
    "_sanitize_image_ref",
    "_stage_image_assets",
    "_svg_native_size",
]


class TypstReconstructor:
    """Reconstructs translated documents into Typst (.typ) and compiles vector PDFs.

    Features:
    - 10.5pt standard publication font size with CJK serif font priority.
    - Native LaTeX math formula compilation ($ ... $).
    - Multi-column and facing bilingual layout support.
    """

    def __init__(
        self,
        font_size_pt: float = 10.5,
        paper_size: str = "a4",
        typst_binary: str = "typst",
        target_lang: str = "zh",
        leading_em: float = 0.85,
    ) -> None:
        self.font_size_pt = font_size_pt
        self.paper_size = paper_size
        self.typst_binary = shutil.which(typst_binary) or typst_binary
        self.target_lang = target_lang
        self.leading_em = leading_em
        # Operator font override, written by the adapter's ``font_family``
        # setter. Prepended to the language-resolved stack (never replacing
        # it), so an uninstalled name still falls back instead of tofu-ing.
        self.font_family: str | None = None
        # Image/cover assets that could not be staged, as ``(block_id, reason)``.
        # The emitter's only other signal is a ``//`` comment in the Typst
        # source, invisible in the PDF: a reflow render that dropped every
        # figure would still report full render coverage, and ``--strict``
        # would pass it. The rigid engine keeps this ledger already; the
        # render strategy copies it into ``last_render_skips`` so one channel
        # serves both.
        self.last_image_skips: list[tuple[str, str]] = []
        # Optional context for the formula image fallback: when a formula
        # cannot be converted, the caller may hand us the ORIGINAL pdf so the
        # equation can be shown as its source graphic instead of raw text.
        # Left None for non-PDF inputs and for unit tests, which keeps the
        # verbatim-text fallback.
        self.source_pdf: Path | None = None
        # Equation numbering is per chapter in the source books. Section
        # headings (3.1, 3.2 …) also carry the chapter digit, so the Typst
        # equation counter may only be reset when the chapter number actually
        # changes — resetting at every section restarts numbering
        # mid-chapter and desynchronizes the printed numbers from the
        # in-text references.
        self._last_equation_chapter: int | None = None
        # Syntax-fallback audit trail: lines that the
        # self-healing loop commented OUT of the delivered PDF. The translated
        # content they carried is silently gone from the artifact, so every
        # removal is recorded here (reset per compile_pdf run) and surfaced by
        # the callers into manifest.metadata -> quality report.
        self.last_syntax_fallbacks: list[str] = []
        # Formula render mode: "native" keeps the converted
        # Typst math, "image" renders every display formula from its source
        # graphic, "witness" keeps native math unless the deterministic visual
        # witness reports a structural mismatch.
        self.formula_render: str = "native"
        # Witness failures of the last generation (formula id + findings),
        # surfaced by the adapter into manifest.metadata -> quality report.
        self.last_witness_findings: list[str] = []
        # Display-formula backend. "typst" keeps native Typst conversion;
        # the orchestrator sets "mathjax" (engine SVG, default) or "image" (source crops).
        self.math_backend: str = "typst"
        self._math_renderer: object | None = None
        # Body font size of the current generation; engine SVGs are sized in
        # em units from MathJax's viewBox so every formula matches the text.
        self._active_font_pt: float = self.font_size_pt
        self._healer = TypstDiagnosticHealer(typst_binary=self.typst_binary)

    def is_compiler_available(self) -> bool:
        """Check if typst CLI compiler is available on PATH."""
        self._healer.typst_binary = self.typst_binary
        return self._healer.is_compiler_available()

    def close(self) -> None:
        """Release the persistent MathJax node subprocess, if one was started.

        The renderer is lazy and long-lived; without an explicit close a
        long-running server accumulates one node process per job until it exits.
        Idempotent and never raises.
        """
        renderer = self._math_renderer
        self._math_renderer = None
        closer = getattr(renderer, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception:
                logger.debug("MathJax renderer close failed", exc_info=True)

    def compiler_version(self) -> str | None:
        """Report the Typst compiler version (e.g. ``"0.15.1"``)."""
        self._healer.typst_binary = self.typst_binary
        return self._healer.compiler_version()

    def generate_typst_source(
        self,
        blocks: Sequence[IRBlock],
        title: str = "Translated Document",
        bilingual: bool = False,
        columns: int = 1,
        page_strict: bool = False,
        cover_mode: str = "auto",
        pagebreaks: bool = True,
        target_lang: str | None = None,
        page_size: tuple[float, float] | None = None,
        start_page_num: int = 1,
        font_size: float | None = None,
        leading_em: float | None = None,
        source_page_height: float | None = None,
    ) -> str:
        """Generate clean, publication-ready Typst source code from IR blocks.

        Args:
            blocks: Sequence of translated IR blocks.
            title: Document title for the header.
            bilingual: If True, render source and target side-by-side.
            columns: Number of text columns (2 for academic two-column layout).
            page_strict: If True, group blocks by their original page number
                and insert #pagebreak() between page groups to maintain strict
                1:1 page correspondence with the source PDF.
            cover_mode: "auto" (page 1 is a cover only when it has no body
                text), "always" (page 1 is always a cover), "never" (no
                cover page; page 1 renders as interior content).
            pagebreaks: Emit hard #pagebreak() between source page groups.
                Only the alternating (page-zipper) mode needs 1:1 alignment;
                monolingual reflow must flow continuously, otherwise a tall
                figure + translated text overflow strands whole pages
                (output p2 = figure-only, body slips to p3).
            target_lang: Target language for typography and localized labels.
            page_size: Optional (width_pt, height_pt) tuple for exact page dimensions.
            start_page_num: Initial page number counter value (default 1).
            font_size: Optional typography font size in pt (overrides default/instance size).
            leading_em: Optional paragraph line-spacing in em (overrides default/instance leading).
            source_page_height: Optional source page height in pt, used only to
                place cover-page subtitle/description cuts relative to the page
                (does not change the output page size).
        """
        effective_target_lang = target_lang or self.target_lang
        self.target_lang = effective_target_lang
        # Each generation starts a fresh equation-numbering scope.
        self._last_equation_chapter = None
        # Fresh witness audit trail per generation.
        self.last_witness_findings = []
        resolved_font_size = font_size if font_size is not None else self.font_size_pt
        if page_strict:
            # Use slightly smaller font for page-strict academic mode to fit content
            resolved_font_size = min(resolved_font_size, 9.5 if font_size is None else font_size)
        resolved_leading_em = leading_em if leading_em is not None else self.leading_em
        self._active_font_pt = float(resolved_font_size)
        # Per-render ledger: the three emitters below append to it, and the
        # render strategy copies it out once the source is produced.
        self.last_image_skips = []

        col_setting = f", columns: {columns}" if columns > 1 else ""
        # Collapse whitespace so a crafted multi-line title cannot inject
        # Content below the heading, then escape markup specials.
        escaped_title = _escape_typst_markup(" ".join(str(title).split())) if title else ""
        footer_title = _footer_display_title(blocks, title, bilingual)
        footer_block = ""
        if page_strict and footer_title:
            footer_block = f""",
  footer: context {{
    if counter(page).get().first() > 1 {{
      line(length: 100%, stroke: 0.5pt + rgb("#cbd5e1"))
      v(-0.2em)
      grid(
        columns: (1fr, 1fr),
        align(left)[#text(size: 8.5pt, fill: rgb("#64748b"))[{footer_title}]],
        align(right)[#text(size: 8.5pt, fill: rgb("#64748b"))[#counter(page).display()]],
      )
    }}
  }}"""

        margins = (
            "margin: (x: 2.2cm, top: 2.5cm, bottom: 2.3cm)"
            if page_strict
            else "margin: (x: 1.8cm, y: 2cm)"
        )

        font_cfg = resolve_font_config(effective_target_lang)
        font_stack = list(font_cfg.typst_fonts)
        if self.font_family and self.font_family not in font_stack:
            font_stack.insert(0, str(self.font_family))
        # Prune to families the compiler can actually resolve, so a machine
        # without any CJK font surfaces instead of shipping blank squares. An
        # explicit font_family override is protected: the user named it, so
        # dropping it silently would be worse than Typst's own fallback.
        resolved_fonts = resolve_font_stack(
            font_stack,
            target_lang=effective_target_lang,
            typst_binary=self.typst_binary,
            protected=((str(self.font_family),) if self.font_family else ()),
        )
        if resolved_fonts.probed and resolved_fonts.unavailable:
            logger.warning(
                "Font families not installed on this machine, removed from the "
                "rendered stack (an explicit override is kept): %s",
                ", ".join(resolved_fonts.unavailable),
            )
        if resolved_fonts.substituted:
            logger.warning(
                "No requested family can render target %r; substituting %s so the "
                "text is readable rather than tofu. The preset's typeface is not "
                "what ships.",
                effective_target_lang,
                ", ".join(resolved_fonts.substituted),
            )
        if not resolved_fonts.cjk_available:
            logger.warning(
                "No CJK-capable font installed for target %r; Chinese/Japanese/Korean "
                "glyphs will render as tofu. Install a CJK font (e.g. noto-fonts-cjk) "
                "or pass an available family via font_family.",
                effective_target_lang,
            )
        font_tuple_str = resolved_fonts.as_typst_tuple()
        page_dim = (
            f"width: {page_size[0]:.2f}pt, height: {page_size[1]:.2f}pt"
            if page_size is not None
            else f'paper: "{self.paper_size}"'
        )
        counter_line = [f"#counter(page).update({start_page_num})"] if start_page_num > 1 else []

        lang_code = (
            effective_target_lang.split("-")[0].split("_")[0].lower()
            if effective_target_lang
            else ""
        )
        if lang_code in ("zh", "zh-cn", "zh-hans", "zh-tw", "zh-hant", "zh-hk"):
            indent_val = "2em"
        elif lang_code == "ja":
            indent_val = "1em"
        else:
            indent_val = ""
        indent_setting = f", first-line-indent: {indent_val}" if indent_val else ""
        safe_lang = sanitize_lang_tag(lang_code)
        text_lang_setting = f', lang: "{safe_lang}"' if safe_lang else ""

        lines: list[str] = [
            f"#set page({page_dim}, {margins}{col_setting}{footer_block})",
            *counter_line,
            f"#set text(font: {font_tuple_str}, size: {resolved_font_size}pt{text_lang_setting})",
            f"#set par(justify: true, leading: {resolved_leading_em}em{indent_setting})",
            "#set list(spacing: 0.7em, marker: [•])",
            "#show heading.where(level: 1): it => { counter(math.equation).update(0); it }",
            '#set math.equation(numbering: (..nums) => { let ch = counter(heading).get().first(); if ch > 0 { "(" + str(ch) + "." + str(nums.pos().first()) + ")" } else { "(" + str(nums.pos().first()) + ")" } })',
            "",
        ]

        # Image-backed formulas (engine SVG or source crop) cannot carry a
        # referenceable Typst label — only numbered elements are referenceable,
        # and a label on a grid/counter fails at every #ref with "cannot
        # reference grid/counter-update". With the image backends active the
        # in-text references therefore stay literal text: the printed number
        # still matches (engine columns print the OCR tag, crops carry it).
        formula_map = (
            {} if self.math_backend in ("mathjax", "image") else _build_formula_map(blocks)
        )

        if page_strict:
            self._generate_page_strict(
                blocks,
                lines,
                bilingual,
                _reference_numbers(blocks),
                cover_mode,
                pagebreaks,
                formula_map=formula_map,
                source_page_height=source_page_height,
            )
        else:
            if title:
                lines.append(f"= {escaped_title}")
                lines.append("")
            self._generate_flowing(
                blocks,
                lines,
                bilingual,
                _reference_numbers(blocks),
                formula_map=formula_map,
            )

        # Formula fidelity modes (default keeps native math):
        # 'image' replaces every display formula with its source graphic;
        # 'witness' keeps native math unless the deterministic visual witness
        # finds a structural mismatch, then falls back to the same graphic.
        backend = self.math_backend
        engine = self._engine() if backend == "mathjax" else None
        if backend == "mathjax" and engine is None:
            logger.warning(
                "math_backend='mathjax' requested but Node/MathJax is unavailable; "
                "falling back to the typst backend for this render"
            )
            backend = "typst"
        # The compile probe below is only meaningful when Typst itself will
        # typeset the math. The mathjax backend replaces every display formula
        # with engine-rendered SVG (quality guarded by the formula_witness
        # inside the substitution pass) and the image backend swaps in the
        # source graphic — probing either first is pure spend: batch + per-line
        # typst spawns plus one pandoc call per formula, all discarded.
        if backend not in ("mathjax", "image"):
            self._verify_math_lines(lines, blocks)
        # Runs after the probe pass: that pass handles lines which *became*
        # unsound at compile time, this one handles formulas Gate 4 refused
        # outright (they are never `$` lines, so the probe never sees them).
        self._substitute_verbatim_formulas(lines, blocks)
        if backend == "image":
            swapped = self._substitute_formulas_with_source_graphic(lines, blocks)
            if swapped:
                logger.info(
                    "Formula backend 'image': %d display formula(s) rendered "
                    "from their source graphics",
                    swapped,
                )
        elif backend == "mathjax" and engine is not None:
            swapped = self._substitute_formulas_with_engine(lines, blocks, engine)
            if swapped:
                logger.info(
                    "Formula backend 'mathjax': %d display formula(s) typeset "
                    "from the OCR LaTeX by the engine",
                    swapped,
                )
        elif self.formula_render == "image":
            swapped = self._substitute_formulas_with_source_graphic(lines, blocks)
            if swapped:
                logger.info(
                    "Formula render mode 'image': %d display formula(s) rendered "
                    "from their source graphics",
                    swapped,
                )
        elif self.formula_render == "witness":
            self._witness_math_lines(lines, blocks)

        return "\n".join(lines)

    def _substitute_formulas_with_source_graphic(
        self, lines: list[str], blocks: Sequence[IRBlock]
    ) -> int:
        """Render every display formula from its source crop (mode 'image').

        Fidelity by construction: the LaTeX->Typst conversion is skipped for
        display equations, so no converter defect can reach the artifact. Only
        tagged display-formula lines are swapped; inline math inside prose is
        untouched. Lines whose crop is unavailable stay native.
        """
        by_id = {b.id: b for b in blocks}
        swapped = 0
        for i, ln in enumerate(lines):
            if not ln.lstrip().startswith(("$", "#math.equation", "#block[#show math.equation")):
                continue
            m = _FORMULA_ID_RE.search(ln)
            block = by_id.get(m.group(1)) if m else None
            if block is None:
                continue
            image_line = self._crop_formula_fallback(block)
            if image_line is None:
                continue
            lines[i] = image_line
            swapped += 1
        return swapped

    def _engine(self) -> Any:
        """The MathJax renderer when it is installed, else None."""
        from ubt.adapters.pdf.math_renderer import MathjaxRenderer

        if self._math_renderer is None:
            self._math_renderer = MathjaxRenderer()
        renderer = self._math_renderer
        if isinstance(renderer, MathjaxRenderer) and renderer.available():
            return renderer
        return None

    def _prose(self, text: str) -> str:
        """Prose → Typst with inline math routed through the engine."""
        return _prose_to_typst(text, inline_math=self._inline_math_renderer())

    def _inline_math_renderer(self) -> Callable[[str], str | None] | None:
        """Inline replacement callback when the engine backend is active."""
        if self.math_backend != "mathjax":
            return None
        engine = self._engine()
        if engine is None:
            return None

        def _render(latex: str) -> str | None:
            return self._inline_svg(latex, engine)

        return _render

    def _inline_svg(self, latex: str, renderer: Any) -> str | None:
        """Inline formula → baseline-aligned SVG box, or None on failure.

        Inline spans have no per-span source bbox, so the deterministic
        witness cannot run here; a render failure returns None and the caller
        keeps native Typst conversion for that span.
        """
        if not latex.strip():
            return None
        result = renderer.render(latex, display=False)
        if not result.ok or not result.svg or not result.width or not result.height:
            return None
        try:
            digest = hashlib.sha256(result.svg.encode("utf-8")).hexdigest()[:12]
            from ubt.adapters.pdf.math_renderer import MATH_CACHE_DIR
            from ubt.core.fs_perms import restrict_dir_to_owner

            asset_dir = MATH_CACHE_DIR
            restrict_dir_to_owner(asset_dir)
            svg_path = asset_dir / f"inline-{digest}.svg"
            if not svg_path.is_file():
                svg_path.write_text(result.svg, encoding="utf-8")
        except OSError as exc:
            logger.debug("Inline math asset unavailable: %s", exc)
            return None
        height_em = float(result.height) / 1000.0
        width_em = float(result.width) / 1000.0
        size = (
            f"width: {_INLINE_MAX_EM:.1f}em"
            if width_em > _INLINE_MAX_EM
            else f"height: {height_em:.3f}em"
        )
        match = _SVG_VA_RE.search(result.svg)
        if match is None:
            # A bare #image is block-level in Typst and would break the
            # paragraph around it; the box keeps the span inline with the
            # image's bottom edge on the baseline.
            return f'#box[#image("{svg_path}", {size})]'
        baseline_em = -float(match.group(1)) * _EX_TO_EM
        if baseline_em == 0.0:
            baseline_em = 0.0  # collapse -0.0 so the emitted em is never "-0.000"
        return f'#box(baseline: {baseline_em:.3f}em)[#image("{svg_path}", {size})]'

    def _recover_formula_tag(self, block: IRBlock) -> str | None:
        """Printed equation number recovered from the source PDF text layer.

        Docling's VLM formula items usually omit the margin equation number,
        and the chapter counter would print a position-based number (3.40)
        where the book prints the author's tag (A.1). The text layer still
        carries the tag next to the formula bbox, so it is read back
        deterministically (never raises; None leaves the chapter-counter
        numbering in place).
        """
        if self.source_pdf is None or block.bbox is None:
            return None
        return recover_formula_tag(self.source_pdf, block.bbox.page, block.bbox)

    def _substitute_formulas_with_engine(
        self, lines: list[str], blocks: Sequence[IRBlock], renderer: Any
    ) -> int:
        """Typeset every display formula with the engine (backend 'mathjax').

        The OCR LaTeX is rendered by MathJax into an SVG vector embedded in
        the document; the equation number is drawn by Typst from the extracted
        source tag (A.12a) or the chapter-scoped sequence, so numbering stays
        under our control. Every render is verified by the same deterministic
        witness the Typst path uses (rasterized SVG vs source crop); a render
        or witness failure degrades to the source graphic and is recorded in
        ``last_witness_findings``.
        """
        from ubt.adapters.pdf.formula_witness import compare_structure, trim_formula_crop

        by_id = {b.id: b for b in blocks}
        chapter: int | None = None
        seq = 0
        swapped = 0
        for i, ln in enumerate(lines):
            # Chapter scope: the generator drives Typst's heading counter
            # explicitly before each chapter's content, which is exactly the
            # digit the equation numbering function would print.
            heading_update = re.search(r"#counter\(heading\)\.update\((\d+)\)", ln)
            if heading_update:
                # Section headings repeat the chapter digit; only a real
                # chapter transition may restart the equation sequence
                # (mirrors _last_equation_chapter on the emit path).
                new_chapter = int(heading_update.group(1))
                if new_chapter != chapter:
                    chapter = new_chapter
                    seq = 0
            elif re.match(r"^=\s", ln):
                parsed = _extract_chapter_number(ln)
                if parsed is not None:
                    chapter = parsed
                    seq = 0
            if "#counter(math.equation).update(0)" in ln:
                seq = 0
            if "#counter(math.equation).step()" in ln:
                seq += 1
                continue
            if not ln.lstrip().startswith(("$", "#math.equation", "#block[#show math.equation")):
                continue
            m = _FORMULA_ID_RE.search(ln)
            block = by_id.get(m.group(1)) if m else None
            if block is None:
                seq += 1
                continue
            seq += 1
            raw = (block.target_text or block.draft_text or block.source_text or "").strip()
            latex = _clean_ocr_formula(raw)
            tag = (
                _extract_formula_tag(raw)
                or self._recover_formula_tag(block)
                or (f"{chapter}.{seq}" if chapter is not None else str(seq))
            )

            png_width: int | None = None
            source_img = None
            if self.source_pdf is not None and block.bbox is not None:
                width_pt = float(block.bbox.x1) - float(block.bbox.x0)
                png_width = max(32, round(width_pt * _ENGINE_WITNESS_DPI / 72.0))
                try:
                    from ubt.adapters.pdf.visual_scalpel import crop_block_pil

                    source_img = trim_formula_crop(
                        crop_block_pil(
                            self.source_pdf,
                            block.bbox.page,
                            block.bbox,
                            dpi=_ENGINE_WITNESS_DPI,
                            bleed_pt=0.0,
                        ),
                        dpi=_ENGINE_WITNESS_DPI,
                    )
                except Exception as exc:  # witness must never break a render
                    logger.debug("Engine witness crop unavailable for %s: %s", block.id, exc)
                    source_img = None

            result = renderer.render(latex, display=True, png_width=png_width)
            if not result.ok or not result.svg:
                fallback = self._crop_formula_fallback(block)
                if fallback is not None:
                    lines[i] = fallback
                self.last_witness_findings.append(
                    f"{block.id}: engine render failed ({result.error}); source graphic used"
                )
                continue

            findings = self._engine_findings(result, source_img, compare_structure)
            if findings:
                fallback = self._crop_formula_fallback(block)
                if fallback is not None:
                    lines[i] = fallback
                self.last_witness_findings.append(
                    f"{block.id}: engine witness {'; '.join(findings)}; source graphic used"
                )
                continue

            svg_line = self._svg_formula_line(block, result, tag)
            if svg_line is None:
                fallback = self._crop_formula_fallback(block)
                if fallback is not None:
                    lines[i] = fallback
                continue
            lines[i] = svg_line
            swapped += 1
        return swapped

    @staticmethod
    def _engine_findings(result: Any, source_img: Any, compare: Any) -> list[str]:
        """Witness the engine raster against the source crop (empty = pass)."""
        if result.png is None or source_img is None:
            return []
        try:
            import io

            from PIL import Image

            with Image.open(io.BytesIO(result.png)) as raw:
                # The engine raster may carry an alpha channel; flattening it
                # with `convert("L")` alone would paint the background black
                # and every pixel would count as ink, so composite on white.
                rendered = raw.convert("RGBA")
                background = Image.new("RGBA", rendered.size, (255, 255, 255, 255))
                rendered = Image.alpha_composite(background, rendered).convert("L")
            return list(compare(rendered, source_img))
        except Exception as exc:  # never break a render
            logger.debug("Engine witness comparison skipped: %s", exc)
            return []

    def _svg_formula_line(self, block: IRBlock, result: Any, tag: str) -> str | None:
        """Persist the engine SVG and emit its Typst image line with number."""
        svg = result.svg
        if not svg or block.bbox is None:
            return None
        try:
            import hashlib

            digest = hashlib.sha256(svg.encode("utf-8")).hexdigest()[:12]
            from ubt.adapters.pdf.math_renderer import MATH_CACHE_DIR
            from ubt.core.fs_perms import restrict_dir_to_owner

            asset_dir = MATH_CACHE_DIR
            restrict_dir_to_owner(asset_dir)
            svg_path = asset_dir / f"{re.sub(r'[^A-Za-z0-9_.-]', '-', block.id)}-{digest}.svg"
            if not svg_path.is_file():
                svg_path.write_text(svg, encoding="utf-8")
        except OSError as exc:
            logger.warning("Engine SVG asset unavailable for %s: %s", block.id, exc)
            return None

        # Size from MathJax's own em metrics (viewBox units are 1/1000 em) so
        # glyphs match the body text. Scaling to the source bbox would blow up
        # formulas whose box also covers the right-margin equation number.
        if result.width:
            width_pt = (float(result.width) / 1000.0) * self._active_font_pt
        else:
            width_pt = float(block.bbox.x1) - float(block.bbox.x0)
        width_pt = min(max(width_pt, 1.0), _FORMULA_FALLBACK_MAX_WIDTH_PT)
        image = f'#image("{svg_path}", width: {width_pt:.1f}pt)'
        number = f"#text(size: {_FORMULA_NUMBER_SIZE_PT}pt)[({tag})]"
        return (
            f"#v(0.4em)#grid(columns: (1fr, auto), column-gutter: 1em, "
            f"align(center)[{image}], align(right + horizon)[{number}])"
            f"{_formula_label(block)}#counter(math.equation).step()#v(0.4em)"
        )

    def _witness_math_lines(self, lines: list[str], blocks: Sequence[IRBlock]) -> int:
        """Verify every emitted formula against its source pixels (mode 'witness').

        Deterministic L1 witness: compile + raster the emitted
        line, crop the source region, compare ink density, aspect, connected
        components and baseline. A failing formula is swapped for its source
        graphic (lossless) and recorded in ``last_witness_findings``.
        """
        from ubt.adapters.pdf.formula_witness import witness_formula

        if self.source_pdf is None:
            return 0
        by_id = {b.id: b for b in blocks}
        swapped = 0
        for i, ln in enumerate(lines):
            if not ln.lstrip().startswith(("$", "#math.equation", "#block[#show math.equation")):
                continue
            m = _FORMULA_ID_RE.search(ln)
            block = by_id.get(m.group(1)) if m else None
            if block is None:
                continue
            result = witness_formula(ln, block, self.source_pdf, self.typst_binary)
            if result.status != "fail":
                continue
            image_line = self._crop_formula_fallback(block)
            if image_line is None:
                self.last_witness_findings.append(
                    f"{block.id}: witness failed but no source graphic ({'; '.join(result.findings)})"
                )
                continue
            lines[i] = image_line
            swapped += 1
            self.last_witness_findings.append(f"{block.id}: {'; '.join(result.findings)}")
            logger.warning(
                "Formula %s failed visual witness (%s); using its source graphic",
                block.id,
                "; ".join(result.findings),
            )
        return swapped

    def _verify_math_lines(self, lines: list[str], blocks: Sequence[IRBlock]) -> int:
        """Batch compile-probe every ``$...$`` math line; degrade failures.

        Only called when Typst itself will typeset the math (see
        ``generate_typst_source``): the mathjax/image backends replace those
        formulas wholesale, so probing them first is discarded work.
        """
        self._healer.typst_binary = self.typst_binary
        return self._healer.verify_math_lines(lines, blocks, self._degrade_math_line)

    def _substitute_verbatim_formulas(self, lines: list[str], blocks: Sequence[IRBlock]) -> int:
        """Swap Gate-4-rejected formula text spans for their source graphic.

        ``_verify_math_lines`` only scans ``$`` / ``#math.equation`` lines, so a
        formula that Gate 4 refuses at emission time (e.g. Docling LaTeX with
        unbalanced braces) never reaches the degradation
        path and would otherwise ship as a 480-character wall of raw text. Those lines
        carry a ``// [formula <id>]`` tag (see ``_emit_formula_line``); this
        pass turns each into the original equation graphic when the source PDF
        is available, and leaves it alone otherwise.
        """
        by_id = {b.id: b for b in blocks}
        swapped = 0
        for i, ln in enumerate(lines):
            if not ln.lstrip().startswith("`"):
                continue
            m = _FORMULA_ID_RE.search(ln)
            if m is None or m.group(1) not in by_id:
                continue
            image_line = self._crop_formula_fallback(by_id[m.group(1)])
            if image_line is None:
                continue
            lines[i] = image_line
            swapped += 1
        if swapped:
            logger.info(
                "Substituted %d non-typesettable formula(s) with their source graphic", swapped
            )
        return swapped

    def _crop_formula_fallback(self, block: IRBlock) -> str | None:
        """Show a failed formula as its ORIGINAL graphic, not as raw text.

        Gate 4 and the compile probe have both refused to typeset this
        equation, so the honest options left are "print the source text" —
        unreadable for a 480-character formula (it lands as
        a wall of ``cal(E)_("x s")= sqrt( frac( ...``) — or "show the source
        graphic". The graphic is lossless and is what the reader can actually
        use: the L4 visual-scalpel approach applied to the
        deliverable instead of to the repair loop.

        Returns a Typst ``#image(...)`` line, or None when the fallback is not
        possible (no source PDF, no usable bbox, crop failure) — callers then
        keep the verbatim-text fallback, so this is strictly additive.
        """
        if self.source_pdf is None or block.bbox is None:
            return None
        try:
            import io

            from PIL import Image

            from ubt.adapters.pdf.formula_witness import trim_formula_crop
            from ubt.adapters.pdf.visual_scalpel import crop_ir_block_image

            # bleed 0: a glued neighbour prose line inside the bbox would
            # otherwise be pulled into the graphic as a readable fragment.
            b64 = crop_ir_block_image(
                self.source_pdf, block, dpi=_FORMULA_FALLBACK_DPI, bleed_pt=0.0
            )
            if not b64:
                return None

            # Content-addressed inside the owner-only math cache: the same
            # crop bytes map to the same file (re-runs overwrite instead of
            # piling up, and two renders that share a block id but differ in
            # content get different names — no cross-job collision and no
            # per-run token growing the dir forever). MATH_CACHE_DIR is also
            # the one asset root Typst staging accepts for engine formulas.
            from ubt.adapters.pdf.math_renderer import MATH_CACHE_DIR
            from ubt.core.fs_perms import restrict_dir_to_owner

            asset_dir = MATH_CACHE_DIR
            asset_dir.mkdir(parents=True, exist_ok=True)
            restrict_dir_to_owner(asset_dir)
            safe_id = re.sub(r"[^A-Za-z0-9_.-]", "-", block.id)
            raw = base64.b64decode(b64)
            digest = hashlib.sha256(raw).hexdigest()[:12]
            png = asset_dir / f"fallback-{safe_id}-{digest}.png"
            try:
                with Image.open(io.BytesIO(raw)) as crop_img:
                    trimmed = trim_formula_crop(crop_img, dpi=_FORMULA_FALLBACK_DPI)
                    buf = io.BytesIO()
                    trimmed.save(buf, format="PNG")
                png.write_bytes(buf.getvalue())
            except OSError as exc:
                # Fail-open: an undecodable crop keeps its original bytes.
                logger.debug("Formula crop trim skipped for %s: %s", block.id, exc)
                png.write_bytes(raw)
        except Exception as exc:
            logger.warning("Formula image fallback unavailable for %s: %s", block.id, exc)
            return None
        logger.info(
            "Formula %s: conversion failed, falling back to its source graphic",
            block.id,
        )
        # Size the graphic from the source box, not from the generic raster
        # heuristic: _image_size_spec() maps any wide-ish raster to width: 85%,
        # which would blow a 285pt equation up to the full column. The crop is
        # rendered at the block's bbox scale, so its bbox width reproduces the
        # original size. Clamped so a pathological bbox cannot overflow.
        width_pt = float(block.bbox.x1) - float(block.bbox.x0)
        width_pt = min(max(width_pt, 1.0), _FORMULA_FALLBACK_MAX_WIDTH_PT)
        # The graphic carries the ORIGINAL equation number (the crop covers it),
        # but an #image does not advance Typst's equation counter — without the
        # explicit step every following formula would be numbered one too low.
        # The dynamic label rides along: replacing a formula must not orphan
        # the in-text `#ref(<eq-…>)` links that pointed at it.
        return (
            f'#v(0.4em)#align(center)[#image("{png}", width: {width_pt:.1f}pt)]'
            f"{_formula_label(block)}#counter(math.equation).step()#v(0.4em)"
        )

    def _degrade_math_line(self, lines: list[str], li: int, by_id: Mapping[str, IRBlock]) -> None:
        """Replace one math line with its verbatim source span (fail-closed)."""
        m = _FORMULA_ID_RE.search(lines[li])
        if m and m.group(1) in by_id:
            b = by_id[m.group(1)]
            # Prefer the source graphic when we have one: the text fallback
            # below is complete but unreadable for a long equation.
            image_line = self._crop_formula_fallback(b)
            if image_line is not None:
                lines[li] = image_line
                return
            raw = (b.target_text or b.draft_text or b.source_text or "").strip()
            # Same cleaning as the math path: already-handled artifacts stay
            # out while the residue that broke compilation remains visible.
            raw = _clean_ocr_formula(raw)
            raw = raw.replace("`", "'").replace("\n", " ")
            tag = m.group(1)
            lines[li] = f"`{raw}`" if raw else f"// [formula {tag} omitted: empty]"
            return

        # Prose line containing broken inline math: degrade uncompilable $...$ spans to code spans
        def _degrade_span(sm: re.Match[str]) -> str:
            span_content = sm.group(1).replace("`", "'")
            return f"`${span_content}$`"

        lines[li] = re.sub(r"\$([^$\n]+)\$", _degrade_span, lines[li])

    def _collect_footnotes(
        self, blocks: Sequence[IRBlock], bilingual: bool
    ) -> tuple[dict[int, list[tuple[str | None, str]]], set[int]]:
        """Pre-pass shared by both layouts: anchor footnotes to their narrative.

        Returns ``(attached, consumed)``: per-narrative-index callout lists and
        the footnote-block indices the emitters must skip (they render inside
        the anchored paragraph instead).

        Footnotes are structural (``flow_id``), never positional: any
        y-position trigger (e.g. y0<100) fires on EVERY page whose body flows
        to the bottom (i.e. nearly every page), injecting ``#v(1fr)`` springs
        that tear giant gaps and dye body text slate-gray; a substring trigger
        ("References" in prose) misfires the same way. Never hijack list items
        or verbatim-skipped blocks (e.g. bibliography entries): they are list
        content with uniform styling, and re-rendering them as gray footnotes
        mid-list breaks numbering and rhythm.
        """
        attached: dict[int, list[tuple[str | None, str]]] = defaultdict(list)
        consumed: set[int] = set()
        last_narrative_idx: int | None = None
        for idx, blk in enumerate(blocks):
            is_footnote = (
                blk.flow_id == FlowID.FOOTNOTE
                and blk.block_type != BlockType.LIST_ITEM
                and not blk.skip_translate
            )
            if is_footnote:
                marker_num, clean_content, clean_source = _parse_footnote_block(
                    blk, target_lang=self.target_lang
                )
                fn_callout = _format_footnote_markup(
                    clean_content, clean_source, bilingual=bilingual
                )
                if last_narrative_idx is not None:
                    attached[last_narrative_idx].append((marker_num, fn_callout))
                    consumed.add(idx)
            elif blk.block_type in (BlockType.NARRATIVE, BlockType.DIALOGUE):
                last_narrative_idx = idx
        return attached, consumed

    def _generate_flowing(
        self,
        blocks: Sequence[IRBlock],
        lines: list[str],
        bilingual: bool,
        ref_numbers: Mapping[str, str] | None = None,
        formula_map: Mapping[str, str] | None = None,
    ) -> None:
        """Emit blocks in continuous flowing layout with native footnotes and cross-refs."""
        attached_footnotes, consumed_footnotes = self._collect_footnotes(blocks, bilingual)

        prev_block: IRBlock | None = None
        for i, block in enumerate(blocks):
            if i in consumed_footnotes:
                continue
            if block.block_type == BlockType.IMAGE:
                asset_path = _sanitize_image_ref(
                    (block.target_text or block.source_text or "").strip()
                )
                if _is_decorative_chapter_banner(block, blocks, i, asset_path):
                    logger.info(
                        "Dropping decorative chapter banner image in flowing layout: %s", block.id
                    )
                    self.last_image_skips.append((str(block.id), "decorative_banner"))
                    continue
            self._emit_block(
                block,
                lines,
                bilingual,
                prev_block=prev_block,
                ref_numbers=ref_numbers,
                formula_map=formula_map,
                footnote_attachments=attached_footnotes.get(i),
            )
            prev_block = block

    def _generate_page_strict(
        self,
        blocks: Sequence[IRBlock],
        lines: list[str],
        bilingual: bool,
        ref_numbers: Mapping[str, str] | None = None,
        cover_mode: str = "auto",
        pagebreaks: bool = True,
        formula_map: Mapping[str, str] | None = None,
        source_page_height: float | None = None,
    ) -> None:
        """Emit blocks grouped by original page number with strict #pagebreak() boundaries.

        Ensures the Typst output has exactly N pages matching the source PDF's N pages,
        so the BilingualAlternator can interleave them 1:1 without content drift.
        """
        # Group blocks by source page number
        page_groups: dict[int, list[IRBlock]] = {}
        unassigned: list[IRBlock] = []
        for block in blocks:
            page = block.bbox.page if block.bbox and block.bbox.page > 0 else None
            if page is None:
                # A block whose provenance carried no bbox still knows its page
                # (docling_parser records it): honouring that keeps the block on
                # its own page instead of the preceding one.
                recorded = block.provenance.get("source_page")
                if isinstance(recorded, int) and recorded > 0:
                    page = recorded
            if page is not None:
                page_groups.setdefault(page, []).append(block)
            else:
                unassigned.append(block)

        if not page_groups:
            # No page metadata available: fall back to flowing layout
            self._generate_flowing(
                blocks,
                lines,
                bilingual,
                ref_numbers=ref_numbers,
                formula_map=formula_map,
            )
            return

        # A block with no bbox has no page of its own. Place it on the page of
        # the nearest *preceding* block in reading order (the last assigned
        # block with a smaller spine_index), or page 1 when nothing precedes it.
        # The previous code dumped every unassigned block on the LAST page,
        # which corrupted the 1:1 source/target page alignment page-strict mode
        # exists to guarantee.
        if unassigned:
            assigned = sorted(
                (b.spine_index, b.bbox.page)
                for b in blocks
                if b.bbox is not None and b.bbox.page > 0
            )
            for block in unassigned:
                target_page = 1
                for spine_index, page_num in assigned:
                    if spine_index <= block.spine_index:
                        target_page = page_num
                    else:
                        break
                page_groups.setdefault(target_page, []).append(block)

        # Sort each page's blocks by reading order so an unassigned block
        # inserted above lands between its neighbours, not after them.
        for page_block_list in page_groups.values():
            page_block_list.sort(key=lambda b: b.spine_index)

        sorted_pages = sorted(page_groups.keys())
        for page_idx, page_num in enumerate(sorted_pages):
            if page_idx > 0 and pagebreaks:
                lines.append("#pagebreak()")
                lines.append("")

            page_blocks = page_groups[page_num]
            if page_num == 1 and self._is_cover_page(page_blocks, cover_mode):
                self._emit_cover_page(page_blocks, lines, bilingual, source_page_height)
                continue

            self._emit_interior_page(
                page_blocks,
                lines,
                bilingual,
                page_num,
                ref_numbers,
                formula_map=formula_map,
            )

    def _is_eyebrow(self, text: str) -> bool:
        """Detect if heading text functions as a section eyebrow / topic indicator."""
        t = text.strip()
        return (
            bool(re.match(r"^\d+\s*[-–—]", t))
            or t.upper() in ("ORIENTATION", "REFERENCES")
            or t in ("导读", "参考文献", "概述")
        )

    def _emit_chapter_counter(
        self, lines: list[str], heading_content: str, heading_source: str
    ) -> None:
        """Re-open the heading and equation counters when a chapter begins."""
        ch_num = _extract_chapter_number(heading_content) or _extract_chapter_number(heading_source)
        if not ch_num:
            return
        lines.append(f"#counter(heading).update({ch_num})")
        if ch_num != self._last_equation_chapter:
            # Reset only on a real chapter transition: numbered sections
            # (3.1, 3.2 …) repeat the chapter digit and must not restart the
            # equation counter.
            self._last_equation_chapter = ch_num
            lines.append("#counter(math.equation).update(0)")

    def _emit_interior_page(
        self,
        blocks: Sequence[IRBlock],
        lines: list[str],
        bilingual: bool,
        page_num: int = 1,
        ref_numbers: Mapping[str, str] | None = None,
        formula_map: Mapping[str, str] | None = None,
    ) -> None:
        """Render a publication-grade interior page with proper visual hierarchy.

        Thin page-strict shell over the shared :meth:`_emit_block` core: the
        page supplies its ``_PAGE_STYLE`` profile and page context, and the
        core applies the publication treatment —

        - Eyebrow chapter indicator paired with major section title and horizontal rule.
        - Section subheadings with bold emphasis and balanced vertical rhythm.
        - Centered, aspect-ratio-constrained figure/diagram embeddings.
        - Booktabs three-line table rendering without box borders.
        - Clean bulleted list item formatting.
        - Typst native #footnote[...] rendering rigid to body narrative.

        plus every behavioral fix that used to live only in ``_emit_block``
        (page chrome, draft-over-source, caption polish, verbatim formula
        tags, heading-echo trim) — there is no interior-only shadow logic
        left (incident tests: ``test_cover_formula_prompt_gates.py:272`` and
        ``:460``).
        """
        # Reading order is the caller's (``spine_index`` is column-aware;
        # geometry here would interleave the two columns of a page).
        page_blocks = list(blocks)
        attached_footnotes, consumed_footnotes = self._collect_footnotes(page_blocks, bilingual)
        ctx = _EmitContext(profile=_PAGE_STYLE, siblings=page_blocks, page_num=page_num)
        i = 0
        prev_block: IRBlock | None = None
        while i < len(page_blocks):
            if i in consumed_footnotes:
                i += 1
                continue
            blk = page_blocks[i]
            ctx.index = i
            self._emit_block(
                blk,
                lines,
                bilingual,
                prev_block=prev_block,
                ref_numbers=ref_numbers,
                formula_map=formula_map,
                footnote_attachments=attached_footnotes.get(i),
                context=ctx,
            )
            prev_block = blk
            # The core may consume the fused next heading (eyebrow + title):
            # it advances ``ctx.index`` so that block is never emitted twice.
            i = ctx.index + 1

    def _emit_page_heading(
        self,
        lines: list[str],
        bilingual: bool,
        ctx: _EmitContext,
        profile: StyleProfile,
        heading_content: str,
        heading_source: str,
    ) -> None:
        """Page-strict publication headings (``_PAGE_STYLE.custom_headings``).

        Chapter openers get the eyebrow + 17pt title + rule treatment and
        section subheads a bold 12.5pt line, where the flowing path delegates
        to Typst's ``== `` markup. May consume the NEXT sibling (eyebrow/title
        fusion) by advancing ``ctx.index`` so the caller skips it.
        """

        def _chapter_title(title: str, title_src: str) -> None:
            """Bilingual source line (when it differs) + title + rule."""
            if bilingual and title_src and title_src != title:
                lines.append(
                    f'#text(fill: {profile.heading_source_fill}, size: {profile.heading_source_size}, weight: "regular")[{_escape_typst_markup(title_src)}]'
                )
                lines.append("  #v(0.2em)")
            lines.append(
                f'#text(size: {profile.chapter_title_size}, weight: "bold", fill: {profile.chapter_title_fill})[{_escape_typst_markup(title)}]'
            )
            lines.append("  #v(0.3em)")
            lines.append(f"  #line(length: 100%, stroke: {profile.chapter_rule_stroke})")
            lines.append("  #v(0.8em)")
            lines.append("")

        # Numbered sections like 3.1, 3.2, A.1, A.2 are never 17pt chapter titles!
        is_numbered_section = bool(re.match(r"^(?:\d+\.\d+|[A-Z]\.\d+)", heading_content.strip()))
        is_chapter_title = not is_numbered_section and (
            ctx.page_num == 1
            or bool(
                re.match(
                    r"^(?:第\s*[0-9一二三四五六七八九十百]+|[Cc]hapter\s*\d+|[Aa]ppendix\b|附录)",
                    heading_content.strip(),
                )
            )
        )

        if not (ctx.first_heading_open and is_chapter_title):
            if (
                bilingual
                and heading_source
                and heading_content
                and heading_source != heading_content
            ):
                lines.append(
                    f'#v(0.6em)#text(fill: {profile.heading_source_fill}, size: {profile.heading_source_size}, weight: "regular")[{_escape_typst_markup(heading_source)}]'
                )
                lines.append("  #v(0.1em)")
                lines.append(
                    f'#text(size: {profile.section_title_size}, weight: "bold", fill: {profile.section_title_fill})[{_escape_typst_markup(heading_content)}]#v(0.3em)'
                )
            else:
                lines.append(
                    f'#v(0.6em)#text(size: {profile.section_title_size}, weight: "bold", fill: {profile.section_title_fill})[{_escape_typst_markup(heading_content)}]#v(0.3em)'
                )
            lines.append("")
            return

        ctx.first_heading_open = False
        is_eyebrow = self._is_eyebrow(heading_source) or self._is_eyebrow(heading_content)
        next_heading: IRBlock | None = None
        if is_eyebrow and ctx.index + 1 < len(ctx.siblings):
            cand = ctx.siblings[ctx.index + 1]
            if cand.block_type == BlockType.HEADING:
                next_heading = cand

        if is_eyebrow and next_heading is not None:
            eyebrow_text = heading_content
            eyebrow_src = heading_source
            # Advancing ``ctx.index`` below consumes the fused heading without
            # rendering it, so its own counter update has to be emitted here
            # or the chapter keeps the stale number.
            self._emit_chapter_counter(
                lines,
                (
                    next_heading.target_text
                    or next_heading.draft_text
                    or next_heading.source_text
                    or ""
                ).strip(),
                (next_heading.source_text or "").strip(),
            )
            title_text = _polish_target_text(
                (
                    next_heading.target_text
                    or next_heading.draft_text
                    or next_heading.source_text
                    or ""
                ).strip(),
                target_lang=self.target_lang,
            )
            title_src = (next_heading.source_text or "").strip()

            if bilingual and eyebrow_src and eyebrow_src != eyebrow_text:
                lines.append(
                    f'#text(fill: {profile.eyebrow_fill}, size: {profile.eyebrow_size}, weight: "bold")[{_escape_typst_markup(eyebrow_src)}]'
                )
                lines.append("  #v(0.1em)")
            lines.append(
                f'#text(size: {profile.eyebrow_label_size}, fill: {profile.eyebrow_label_fill}, weight: "bold")[{_escape_typst_markup(eyebrow_text)}]'
            )
            lines.append("  #v(0.2em)")

            _chapter_title(title_text, title_src)
            ctx.index += 1
            return

        _chapter_title(heading_content, heading_source)

    def _is_cover_page(self, blocks: Sequence[IRBlock], cover_mode: str = "auto") -> bool:
        """Deterministic cover rule: page 1 is a cover only without body text.

        ``auto`` (default): cover iff at least one block exists and none of
        them is a body block (narrative / dialogue / list / table / formula /
        code) carrying text. ``always`` forces a cover, ``never`` disables it.
        Unknown modes fall back to ``auto``. The rule is structural, never a
        block count: a count-based heuristic (<=10 blocks + any heading)
        misclassifies chapter pages and swallows single-page documents into
        cover-only output.
        """
        mode = normalize_cover_mode(cover_mode)
        if mode == "never":
            return False
        if mode == "always":
            return len(blocks) > 0
        if not blocks:
            return False
        for b in blocks:
            body = (b.target_text or b.draft_text or b.source_text or "").strip()
            if b.block_type in _BODY_BLOCK_TYPES and body:
                return False
        return True

    def _emit_cover_page(
        self,
        blocks: Sequence[IRBlock],
        lines: list[str],
        bilingual: bool,
        page_height: float | None = None,
    ) -> None:
        """Render an elegant, publication-grade cover page.

        Subtitle/description placement keys off vertical position. The old
        absolute cuts (``y0 > 300`` / ``> 200`` pt) are A4-specific — on a
        shorter cover 300pt sits mid-page, so a subtitle could be mistaken for
        a description (or vice versa). When the source page height is known the
        cuts scale with it using the A4-equivalent fractions, preserving the
        same reading order on any page size; without it the absolute fallback
        keeps the previous behaviour.
        """
        if page_height and page_height > 0:
            subtitle_cut = page_height * _COVER_SUBTITLE_FRAC
            desc_cut = page_height * _COVER_DESC_FRAC
        else:
            subtitle_cut = 300.0
            desc_cut = 200.0

        title_block: IRBlock | None = None
        subtitle_block: IRBlock | None = None
        desc_block: IRBlock | None = None
        author_blocks: list[IRBlock] = []
        image_blocks: list[IRBlock] = []

        for b in blocks:
            if b.block_type == BlockType.HEADING and not title_block:
                title_block = b
            elif b.block_type == BlockType.IMAGE:
                # Figures on the cover must render as images, never as text:
                # IMAGE blocks carry the asset path in source/target_text, so
                # letting them fall into author_blocks prints an absolute
                # filesystem path into the document (vec_p*_*.svg leak).
                image_blocks.append(b)
            elif title_block and not subtitle_block and (b.bbox and b.bbox.y0 > subtitle_cut):
                subtitle_block = b
            elif title_block and not desc_block and (b.bbox and b.bbox.y0 > desc_cut):
                desc_block = b
            else:
                author_blocks.append(b)

        title = (
            (
                title_block.target_text or title_block.draft_text or title_block.source_text or ""
            ).strip()
            if title_block
            else ""
        )
        sub = (
            (
                subtitle_block.target_text
                or subtitle_block.draft_text
                or subtitle_block.source_text
                or ""
            ).strip()
            if subtitle_block
            else ""
        )
        desc = (
            (
                desc_block.target_text or desc_block.draft_text or desc_block.source_text or ""
            ).strip()
            if desc_block
            else ""
        )

        lines.append("#align(left)[")
        lines.append("  #v(3.5cm)")
        if title:
            lines.append(
                f'  #text(size: 26pt, weight: "bold", fill: rgb("#111827"))[{_escape_typst_markup(title)}]'
            )
            lines.append("  #v(0.8em)")
        if sub:
            lines.append(
                f'  #text(size: 14pt, weight: "medium", fill: rgb("#374151"))[{_escape_typst_markup(sub)}]'
            )
            lines.append("  #v(1.8em)")
        lines.append('  #line(length: 100%, stroke: 0.75pt + rgb("#cbd5e1"))')
        lines.append("  #v(1.8em)")
        if desc:
            lines.append(
                f'  #text(size: 11pt, style: "italic", fill: rgb("#4b5563"))[{_escape_typst_markup(desc)}]'
            )
            lines.append("  #v(1.5em)")
        for ib in image_blocks:
            asset_path = _sanitize_image_ref(((ib.target_text or ib.source_text) or "").strip())
            if asset_path and Path(asset_path).exists():
                lines.append(f'  #align(center)[#image("{asset_path}", width: 80%)]')
                lines.append("  #v(0.8em)")
            else:
                lines.append(f"  // Cover image asset unavailable: {ib.id}")
                self.last_image_skips.append((str(ib.id), "cover_asset_missing"))

        lines.append("  #v(1fr)")
        for ab in author_blocks:
            atxt = (ab.target_text or ab.draft_text or ab.source_text or "").strip()
            if atxt:
                lines.append(
                    f'  #text(size: 11pt, fill: rgb("#1f2937"))[{_escape_typst_markup(atxt)}]'
                )
                lines.append("  #v(0.3em)")
        lines.append("]")
        lines.append("")

    def _emit_block(
        self,
        block: IRBlock,
        lines: list[str],
        bilingual: bool,
        prev_block: IRBlock | None = None,
        ref_numbers: Mapping[str, str] | None = None,
        formula_map: Mapping[str, str] | None = None,
        footnote_attachments: Sequence[tuple[str | None, str]] | None = None,
        *,
        context: _EmitContext | None = None,
    ) -> None:
        """Emit a single IRBlock as Typst markup — the ONE shared core.

        Both layouts emit through this method: the flowing layout
        (``_generate_flowing``, default context) and the page-strict layout
        (``_emit_interior_page``, which passes its ``_PAGE_STYLE`` page
        context). Behavioral fixes — page-chrome drop, draft preferred over
        source, caption polish, verbatim formula tags, chrome-block drop,
        heading-echo trim — therefore land exactly once and can never exist
        in only one path again (the two incidents pinned by
        ``tests/unit/test_cover_formula_prompt_gates.py``). Style and the
        remaining deliberate page-strict design differences travel in
        ``context.profile`` (see :class:`StyleProfile` for the drift record).
        """
        ctx = context if context is not None else _EmitContext()
        profile = ctx.profile
        content, text, source = _resolve_content(block)
        if not content:
            return

        # Standalone page chrome (page numerals, lone chapter digits like
        # "3" / "69" / "- 69 -") is meaningless as reflowed body and renders
        # as broken lone-digit paragraphs. Only prose blocks qualify: a short
        # formula ("1+1=2"), a one-cell table ("|1|2|") or a code line ("3.1")
        # is real content, not chrome. Headings are exempt (a "3" section
        # marker stays a marker).
        if block.block_type in (BlockType.NARRATIVE, BlockType.DIALOGUE) and _is_page_chrome(
            content
        ):
            logger.debug("Dropping page-chrome block %s (%r)", block.id, content[:16])
            return

        # Drop duplicate heading if model echoed the immediately preceding
        # heading into narrative (shared by both layouts).
        if (
            prev_block
            and prev_block.block_type == BlockType.HEADING
            and block.block_type != BlockType.HEADING
        ):
            prev_txt = (
                prev_block.target_text or prev_block.draft_text or prev_block.source_text or ""
            ).strip()
            for candidate in (prev_txt, re.sub(r"^#+\s*", "", prev_txt).strip()):
                if candidate and content.startswith(candidate):
                    content = content[len(candidate) :].strip()
                    break
        if not content:
            return

        if block.layout_role in (
            LayoutRole.FOOTER,
            LayoutRole.HEADER,
            LayoutRole.PAGE_NUMBER,
        ) or (getattr(block, "policy_reason", None) == "verdict:chrome"):
            logger.debug("Dropping chrome block from reflow: %s", block.id)
            return

        # Structural footnote anchor (never positional — see
        # _collect_footnotes): FOOTNOTE-flow prose renders as its
        # #footnote[...] callout at its own position. ``not
        # footnote_attachments`` keeps a block with queued callouts on the
        # prose path. Configured profiles determine verbatim-skipped block behavior:
        # flowing attaches callouts (anchor_verbatim_footnotes=True), while
        # page-strict keeps them as prose.
        if (
            block.block_type in (BlockType.NARRATIVE, BlockType.DIALOGUE)
            and block.flow_id == FlowID.FOOTNOTE
            and not footnote_attachments
            and (not block.skip_translate or profile.anchor_verbatim_footnotes)
        ):
            marker_num, clean_content, clean_source = _parse_footnote_block(
                block, target_lang=self.target_lang
            )
            lines.append(_format_footnote_markup(clean_content, clean_source, bilingual=bilingual))
            lines.append("")
            return

        prov = getattr(block, "provenance", None) or {}
        if prov.get("toc_entry"):
            toc_content = _polish_target_text(content.strip(), target_lang=self.target_lang)
            toc_source = source.strip()
            toc_page = str(prov.get("toc_page") or "").strip()
            sec_token = (toc_source or toc_content).split(maxsplit=1)[0].rstrip(".")
            depth = sec_token.count(".") if any(ch.isdigit() for ch in sec_token) else 0
            indent = f"{depth * 1.4:.1f}em" if depth > 0 else "0pt"
            weight = '"regular"' if depth > 0 else '"semibold"'
            entry_markup = self._prose(toc_content)
            if bilingual and toc_source and toc_content and toc_source != toc_content:
                entry_markup = (
                    f"{entry_markup} #h(0.35em) "
                    f'#text(fill: {profile.source_fill}, size: 0.88em, weight: "regular")'
                    f"[({self._prose(toc_source)})]"
                )
            if toc_page:
                lines.append(
                    f"#block(above: 0.45em, below: 0.45em, inset: (left: {indent}))["
                    f"#text(weight: {weight})[{entry_markup}] #h(0.35em) "
                    f"#text(fill: luma(140))[#box(width: 1fr, repeat[.])] #h(0.35em) "
                    f"#text(weight: {weight})[{_escape_typst_markup(toc_page)}]]"
                )
            else:
                lines.append(
                    f"#block(above: 0.45em, below: 0.45em, inset: (left: {indent}))["
                    f"#text(weight: {weight})[{entry_markup}]]"
                )
            lines.append("")
            return

        if block.block_type == BlockType.HEADING:
            content = _polish_target_text(content, target_lang=self.target_lang)
            # Demote figure/table captions misclassified as headings.
            if _MISCLASSIFIED_CAPTION_RE.match(content.strip()):
                lines.append(
                    f'#align(center)[#text(size: {profile.caption_size}, fill: {profile.caption_fill}, style: "italic")[{self._prose(content)}]]'
                )
                lines.append("")
                return

            content_lines = [line.strip() for line in content.split("\n") if line.strip()]
            heading_content = content_lines[0] if content_lines else content

            source_lines = [line.strip() for line in source.split("\n") if line.strip()]
            heading_source = source_lines[0] if source_lines else source

            # Only retain extra narrative lines if the source itself had multiple lines
            extra_narrative = content_lines[1:] if len(source_lines) > 1 else []

            self._emit_chapter_counter(lines, heading_content, heading_source)

            if profile.custom_headings:
                self._emit_page_heading(
                    lines, bilingual, ctx, profile, heading_content, heading_source
                )
            else:
                # Flowing path: delegate to Typst ``== `` markup (the heading
                # counter above already ran for both paths).
                if (
                    bilingual
                    and heading_source
                    and heading_content
                    and heading_source != heading_content
                ):
                    lines.append(
                        f'#text(fill: {profile.heading_source_fill}, size: {profile.heading_source_size}, weight: "regular")[{_escape_typst_markup(heading_source)}]'
                    )
                lines.append(f"== {_escape_typst_markup(heading_content)}")
                lines.append("")

            for extra_p in extra_narrative:
                lines.append(self._prose(extra_p))
                lines.append("")
            return

        if block.block_type == BlockType.FORMULA:
            # Clean OCR artifacts before LaTeX-to-Typst conversion; never
            # silently drop (Gate 4 verified rendering). The trailing id
            # comment lets the compile-probe pass map failures back to the
            # source block for verbatim degradation (fail-closed math).
            lines.append(
                _emit_formula_line(block, content, tag_hint=self._recover_formula_tag(block))
            )
            lines.append("")
            return

        if block.block_type == BlockType.CODE:
            max_backticks = 3
            matches = re.findall(r"`+", content)
            if matches:
                max_backticks = max(max_backticks, max(len(m) for m in matches) + 1)
            fence = "`" * max_backticks
            lines.append(fence)
            lines.append(content)
            lines.append(fence)
            lines.append("")
            return

        if block.block_type == BlockType.TABLE:
            lines.append(_markdown_table_to_typst(content))
            lines.append("")
            return

        if block.block_type == BlockType.IMAGE:
            # Asset from target/source, never from draft text: a draft is
            # prose, not a path (the layouts were unified on this rule).
            asset_path = _sanitize_image_ref(text or source)
            if ctx.siblings:
                banner = _is_decorative_chapter_banner(
                    block, ctx.siblings, max(ctx.index, 0), asset_path
                )
            elif prev_block:
                banner = _is_decorative_chapter_banner(block, [prev_block, block], 1, asset_path)
            else:
                banner = False
            if banner:
                logger.info("Dropping decorative chapter banner image: %s", block.id)
                # Booked once per layout: flowing's full-context pre-check
                # books and skips before this core runs; the page layout has
                # no pre-check and books here. The pair-context drop below is
                # the flowing fallback the pre-check missed.
                self.last_image_skips.append((str(block.id), "decorative_banner"))
                return
            if asset_path and Path(asset_path).exists():
                image = f'#align(center)[#image("{asset_path}", {_image_size_spec(asset_path)})]'
                if profile.image_spacer:
                    # Page-strict figure rhythm (0.5em above and below).
                    lines.append(f"{profile.image_spacer}{image}{profile.image_spacer}")
                else:
                    lines.append(image)
            else:
                # Missing asset: comment only. Emitting a #image ref to a
                # nonexistent file fails the whole Typst compile, and
                # printing `content` would leak the absolute asset path.
                # Both flowing and interior layouts emit a uniform comment placeholder.
                lines.append(f"// Image asset unavailable: {block.id}")
                # Booked only on this branch: the ledger feeds the quality
                # report's render_skip flags and --strict, so recording a
                # figure that *did* ship marked every rendered image as lost
                # (and flipped rendered figures into NEEDS_HUMAN under the
                # length policy).
                self.last_image_skips.append((str(block.id), "missing_asset"))
            lines.append("")
            return

        if (
            block.block_type == BlockType.LIST_ITEM
            or content.startswith(_LIST_MARKERS)
            or source.startswith(_LIST_MARKERS)
            or (
                profile.semicolon_list_heuristics
                and (source.endswith(";") or content.endswith(("；", ";")))
            )
        ):
            # Strip the marker from both sides, then polish the target side:
            # order matters, a ``• FIG. 3.1`` label only localizes after the
            # bullet is gone (the flow-pinned order).
            clean_content = _polish_target_text(
                re.sub(r"^(?:[•\-\*·–]|\d+[.)](?!\d))\s*", "", content).strip(),
                target_lang=self.target_lang,
            )
            clean_source = re.sub(r"^(?:[•\-\*·–]|\d+[.)](?!\d))\s*", "", source).strip()
            ref_label = (ref_numbers or {}).get(block.id, "")
            if ref_label:
                # Bibliography entries keep citable [n] labels instead of
                # anonymous bullets (in-text citations resolve here).
                lines.append(
                    f'#block(breakable: false)[#strong("{ref_label}") {self._prose(clean_content)}]'
                )
            elif bilingual and source and text and source != text:
                esc_src = self._prose(clean_source)
                esc_tgt = self._prose(clean_content)
                lines.append(
                    f"- #text(fill: {profile.list_source_fill}, size: {profile.list_source_size})[{esc_src}] \\"
                )
                lines.append(
                    f"  #text(fill: {profile.list_target_fill}, size: {profile.list_target_size})[{esc_tgt}]"
                )
            else:
                lines.append(f"- {self._prose(clean_content)}")
            if profile.list_item_blank:
                # Flowing separates every item with a blank line; page-strict
                # keeps items contiguous so they read as one Typst list.
                lines.append("")
            return

        # NARRATIVE, DIALOGUE, etc.
        # Model-miss polish (FIG labels, ¼ glyph) on target-side text only;
        # source keeps its English for reference.
        content = _polish_target_text(content, target_lang=self.target_lang)
        text = _polish_target_text(text, target_lang=self.target_lang)
        if bilingual and source and text and source != text:
            escaped_source = self._prose(source)
            escaped_text = self._prose(text)
            if formula_map:
                escaped_text = _apply_cross_refs(escaped_text, formula_map)
            if footnote_attachments:
                for marker_num, fn_callout in footnote_attachments:
                    escaped_text = _attach_footnote_to_prose(escaped_text, marker_num, fn_callout)
            lines.append(f"#block(spacing: {profile.block_spacing})[")
            lines.append(
                f"  #text(fill: {profile.source_fill}, size: {profile.source_size})[{escaped_source}]"
            )
            lines.append("  #parbreak()")
            lines.append(
                f"  #text(fill: {profile.target_fill}, size: {profile.target_size})[{escaped_text}]"
            )
            lines.append("]")
            lines.append("")
        else:
            escaped_content = self._prose(content)
            if formula_map:
                escaped_content = _apply_cross_refs(escaped_content, formula_map)
            if footnote_attachments:
                for marker_num, fn_callout in footnote_attachments:
                    escaped_content = _attach_footnote_to_prose(
                        escaped_content, marker_num, fn_callout
                    )
            if (
                block.flow_id == FlowID.CAPTION
                or getattr(block, "layout_role", None) == LayoutRole.CAPTION
            ):
                # Standardized caption styling across layouts.
                lines.append(
                    f'#align(center)[#text(size: {profile.caption_size}, fill: {profile.caption_fill}, style: "italic")[{escaped_content}]]'
                )
            else:
                lines.append(escaped_content)
            lines.append("")

    def compile_pdf(
        self,
        typ_source: str,
        output_pdf_path: Path | str,
    ) -> Path:
        """Compile Typst source code into a target PDF document.

        No ``run_typst_override`` is passed: the healer's own ``run_typst`` is
        the runner (the old reconstructor ``_run_typst`` was a pure
        pass-through to it), and dropping the override lets the healer's
        friendly missing-binary pre-check run instead of being skipped.
        """
        self._healer.typst_binary = self.typst_binary
        out_pdf = self._healer.heal_and_compile(typ_source, output_pdf_path)
        self.last_syntax_fallbacks = list(self._healer.last_syntax_fallbacks)
        return out_pdf

    async def compile_pdf_async(
        self,
        typ_source: str,
        output_pdf_path: Path | str,
    ) -> Path:
        """Non-blocking compilation of Typst source executed in a threadpool executor."""
        self._healer.typst_binary = self.typst_binary
        out_pdf = await self._healer.heal_and_compile_async(typ_source, output_pdf_path)
        self.last_syntax_fallbacks = list(self._healer.last_syntax_fallbacks)
        return out_pdf
