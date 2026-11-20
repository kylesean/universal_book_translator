"""Deterministic HarfBuzz-measured paragraph typesetter inside a source PDF BBox.

Takes a translated string containing lightweight inline anchors (``<b1>...</b1>``,
``<i2>...</i2>``, ``<c3>...</c3>``, ``<f1/>``) and lays out exact lines inside
``ParagraphGeometry``:
- HarfBuzz exact advance widths (via :class:`FontEngine`)
- CJK kinsoku (避头尾禁则) & CJK/Latin micro-spacing
- Mixed inline rigid formula boxes (:class:`InlineFormulaBox`) baseline-aligned
- Left / Center / Right / Justify alignment
- Clean 2-step fit ladder (safe downward borrow -> gentle leading -> font scale)
- Exact :class:`OverflowItem` emission when text still exceeds the font floor
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ubt.codecs.pdf.font_engine import FontEngine, FontSlot
from ubt.codecs.pdf.formula_box import InlineFormulaBox
from ubt.codecs.pdf.layout_fusion import ParagraphGeometry
from ubt.codecs.protocol import InlineToken, OverflowItem, UnitRole

_TAG_TOKEN_RE = re.compile(r"<(/?)([a-z]\d+)(\s*/)?\s*>")

# CJK characters that must not start a line (避头标点).
_NO_LINE_START: frozenset[str] = frozenset("，。、；：？！》」』）】〉》〕〗〙〛…—·.,;:!?)]}%")
# CJK characters that must not end a line (避尾标点).
_NO_LINE_END: frozenset[str] = frozenset("《「『（【〈《〔〖〘〚([{$¥€£")


def _is_cjk(ch: str) -> bool:
    cp = ord(ch)
    return (
        0x4E00 <= cp <= 0x9FFF
        or 0x3400 <= cp <= 0x4DBF
        or 0x3000 <= cp <= 0x30FF
        or 0xAC00 <= cp <= 0xD7AF
        or 0xFF00 <= cp <= 0xFFEF
    )


def _parse_hex_rgb(
    hex_str: str | None, fallback: tuple[float, float, float]
) -> tuple[float, float, float]:
    if not hex_str or not hex_str.startswith("#") or len(hex_str) != 7:
        return fallback
    try:
        return (
            int(hex_str[1:3], 16) / 255.0,
            int(hex_str[3:5], 16) / 255.0,
            int(hex_str[5:7], 16) / 255.0,
        )
    except ValueError:
        return fallback


@dataclass(frozen=True, slots=True)
class _Atom:
    """One atomic layout unit (a CJK char, a Latin word, a space, or a formula box)."""

    text: str
    is_space: bool
    is_cjk_char: bool
    slot: FontSlot
    is_italic: bool
    color_rgb: tuple[float, float, float]
    formula: InlineFormulaBox | None = None


@dataclass(frozen=True, slots=True)
class PositionedRun:
    """One positioned text run or Form XObject invocation ready for PDF emission."""

    x: float
    baseline_y: float
    font_size_pt: float
    slot: FontSlot
    is_italic: bool
    color_rgb: tuple[float, float, float]
    text: str = ""
    formula: InlineFormulaBox | None = None
    extra_char_spacing_pt: float = 0.0


@dataclass(frozen=True, slots=True)
class TypesetParagraph:
    """Exact positioned output for one paragraph plus any measured overflow."""

    unit_id: str
    runs: tuple[PositionedRun, ...]
    used_font_size_pt: float
    used_leading_pt: float
    line_count: int
    total_height_pt: float
    overflow: OverflowItem | None = None


def _tokenize_marked_text(
    marked_text: str,
    tokens: tuple[InlineToken, ...],
    geom: ParagraphGeometry,
    formulas: dict[str, InlineFormulaBox],
    font_engine: FontEngine,
) -> list[_Atom]:
    token_map = {t.tag_id: t for t in tokens}
    atoms: list[_Atom] = []

    bold_stack: list[bool] = [geom.is_bold]
    italic_stack: list[bool] = [False]
    color_stack: list[tuple[float, float, float]] = [geom.color_rgb]
    open_tags: list[str] = []

    def _emit_plain(segment: str) -> None:
        if not segment:
            return
        cur_bold = bold_stack[-1]
        cur_italic = italic_stack[-1]
        cur_color = color_stack[-1]
        slot = font_engine.pick_slot(is_serif=geom.is_serif, is_bold=cur_bold)

        i = 0
        n = len(segment)
        while i < n:
            ch = segment[i]
            if ch in " \t\r\n":
                # Collapse consecutive whitespace into one space atom.
                while i < n and segment[i] in " \t\r\n":
                    i += 1
                atoms.append(
                    _Atom(
                        text=" ",
                        is_space=True,
                        is_cjk_char=False,
                        slot=slot,
                        is_italic=cur_italic,
                        color_rgb=cur_color,
                    )
                )
            elif _is_cjk(ch):
                atoms.append(
                    _Atom(
                        text=ch,
                        is_space=False,
                        is_cjk_char=True,
                        slot=slot,
                        is_italic=cur_italic,
                        color_rgb=cur_color,
                    )
                )
                i += 1
            else:
                j = i + 1
                while j < n and segment[j] not in " \t\r\n" and not _is_cjk(segment[j]):
                    j += 1
                word = segment[i:j]
                atoms.append(
                    _Atom(
                        text=word,
                        is_space=False,
                        is_cjk_char=False,
                        slot=slot,
                        is_italic=cur_italic,
                        color_rgb=cur_color,
                    )
                )
                i = j

    pos = 0
    for m in _TAG_TOKEN_RE.finditer(marked_text):
        _emit_plain(marked_text[pos : m.start()])
        pos = m.end()

        is_close = bool(m.group(1))
        tag_id = m.group(2)
        is_self_close = bool(m.group(3))

        if is_self_close or tag_id in formulas:
            fbox = formulas.get(tag_id)
            if fbox is not None:
                atoms.append(
                    _Atom(
                        text="",
                        is_space=False,
                        is_cjk_char=False,
                        slot=font_engine.pick_slot(is_serif=geom.is_serif, is_bold=bold_stack[-1]),
                        is_italic=False,
                        color_rgb=color_stack[-1],
                        formula=fbox,
                    )
                )
            continue

        if not is_close:
            tok = token_map.get(tag_id)
            if tok is not None and tok.kind == "pair":
                open_tags.append(tag_id)
                bold_stack.append(bool(tok.payload.get("bold", False)) or bold_stack[-1])
                italic_stack.append(bool(tok.payload.get("italic", False)) or italic_stack[-1])
                color_stack.append(_parse_hex_rgb(tok.payload.get("color_hex"), color_stack[-1]))
        else:
            if open_tags and open_tags[-1] == tag_id:
                open_tags.pop()
                if len(bold_stack) > 1:
                    bold_stack.pop()
                if len(italic_stack) > 1:
                    italic_stack.pop()
                if len(color_stack) > 1:
                    color_stack.pop()

    _emit_plain(marked_text[pos:])
    return atoms


def _atom_width(atom: _Atom, font_size: float, font_engine: FontEngine) -> float:
    if atom.formula is not None:
        return atom.formula.width_pt
    return font_engine.measure_text_pt(atom.text, atom.slot, font_size)


def _needs_cjk_latin_gap(prev: _Atom, curr: _Atom) -> bool:
    if prev.is_space or curr.is_space:
        return False
    if prev.formula is not None or curr.formula is not None:
        return True
    return (
        prev.is_cjk_char != curr.is_cjk_char
        and prev.text not in _NO_LINE_END
        and curr.text not in _NO_LINE_START
    )


def _break_into_lines(
    atoms: list[_Atom],
    max_width: float,
    font_size: float,
    first_line_indent: float,
    font_engine: FontEngine,
) -> list[list[tuple[_Atom, float]]]:
    """Greedy HarfBuzz-measured line breaker with kinsoku and long-word splitting."""
    lines: list[list[tuple[_Atom, float]]] = []
    cur: list[tuple[_Atom, float]] = []
    cur_w = 0.0
    micro_gap = font_size * 0.14

    def _limit() -> float:
        return max(font_size * 1.5, max_width - (first_line_indent if not lines else 0.0))

    def _flush() -> None:
        nonlocal cur, cur_w
        while cur and cur[-1][0].is_space:
            cur.pop()
        if cur:
            lines.append(cur)
        cur = []
        cur_w = 0.0

    for atom in atoms:
        if atom.is_space and not cur:
            continue
        w = _atom_width(atom, font_size, font_engine)
        gap = micro_gap if (cur and _needs_cjk_latin_gap(cur[-1][0], atom)) else 0.0

        if cur and (cur_w + gap + w) > _limit() + 0.5:
            # Kinsoku: avoid leaving opening bracket at end of line.
            if len(cur) >= 2 and cur[-1][0].text in _NO_LINE_END:
                carried = cur.pop()
                _flush()
                cur.append(carried)
                cur_w = carried[1]
                gap = micro_gap if _needs_cjk_latin_gap(carried[0], atom) else 0.0
            # Kinsoku: allow a tiny 0.55em overhang for closing punctuation, otherwise pull the
            # preceding atom onto the next line (挤下) so forbidden punctuation never starts a line.
            elif atom.text in _NO_LINE_START:
                if (cur_w + gap + w) <= _limit() + font_size * 0.55:
                    cur.append((atom, w + gap))
                    cur_w += w + gap
                    continue
                if len(cur) >= 2 and not cur[-1][0].is_space:
                    carried = cur.pop()
                    _flush()
                    cur.append(carried)
                    cur_w = carried[1]
                    gap = micro_gap if _needs_cjk_latin_gap(carried[0], atom) else 0.0
                else:
                    _flush()
                    gap = 0.0
            else:
                _flush()
                if atom.is_space:
                    continue
                gap = 0.0

        # Split an overlong non-CJK word (e.g. URL) cleanly across lines.
        if (
            not cur
            and not atom.is_cjk_char
            and atom.formula is None
            and w > _limit()
            and len(atom.text) > 1
        ):
            rem = atom.text
            while rem:
                lim = _limit()
                cut = len(rem)
                while (
                    cut > 1 and font_engine.measure_text_pt(rem[:cut], atom.slot, font_size) > lim
                ):
                    cut -= 1
                piece = _Atom(
                    text=rem[:cut],
                    is_space=False,
                    is_cjk_char=False,
                    slot=atom.slot,
                    is_italic=atom.is_italic,
                    color_rgb=atom.color_rgb,
                )
                pw = font_engine.measure_text_pt(piece.text, piece.slot, font_size)
                if cut < len(rem):
                    lines.append([(piece, pw)])
                else:
                    cur.append((piece, pw))
                    cur_w = pw
                rem = rem[cut:]
            continue

        cur.append((atom, w + gap))
        cur_w += w + gap

    _flush()
    return lines


def _position_lines(
    lines: list[list[tuple[_Atom, float]]],
    geom: ParagraphGeometry,
    font_size: float,
    leading: float,
    first_line_indent: float,
    font_engine: FontEngine,
) -> tuple[list[PositionedRun], float]:
    bx0, _, bx1, by1 = geom.bbox
    box_w = max(1.0, bx1 - bx0)
    ascent_em, descent_em = font_engine.metrics(
        font_engine.pick_slot(is_serif=geom.is_serif, is_bold=geom.is_bold)
    )
    first_ascent = max(font_size * ascent_em, font_size * 0.80)
    last_descent = max(abs(font_size * descent_em), font_size * 0.22)

    runs: list[PositionedRun] = []
    cur_y = by1 - first_ascent

    for ln_idx, line in enumerate(lines):
        # Account for tall inline formulas pushing the baseline slightly down.
        extra_up = 0.0
        extra_down = 0.0
        for atom, _ in line:
            if atom.formula is not None:
                extra_up = max(extra_up, atom.formula.ascent_pt - first_ascent)
                extra_down = max(extra_down, atom.formula.descent_pt - last_descent)
        if extra_up > 0.0:
            cur_y -= extra_up

        indent = first_line_indent if ln_idx == 0 else 0.0
        avail_w = max(1.0, box_w - indent)
        line_w = sum(w for _, w in line)

        x = bx0 + indent
        extra_gap_per_break = 0.0
        is_last = ln_idx == len(lines) - 1

        if geom.alignment == "center":
            x = bx0 + max(0.0, (box_w - line_w) * 0.5)
        elif geom.alignment == "right":
            x = bx1 - line_w
        elif geom.alignment == "justify" and not is_last and len(line) >= 3:
            slack = avail_w - line_w
            if 0.0 < slack < font_size * 4.0:
                extra_gap_per_break = slack / (len(line) - 1)

        # Coalesce adjacent text atoms sharing slot/italic/color when not justifying per-char.
        micro_gap = font_size * 0.14
        prev_atom: _Atom | None = None
        for a_idx, (atom, _measured_w) in enumerate(line):
            if prev_atom is not None and _needs_cjk_latin_gap(prev_atom, atom):
                x += micro_gap
            raw_w = _atom_width(atom, font_size, font_engine)

            if atom.formula is not None:
                runs.append(
                    PositionedRun(
                        x=x,
                        baseline_y=cur_y,
                        font_size_pt=font_size,
                        slot=atom.slot,
                        is_italic=False,
                        color_rgb=atom.color_rgb,
                        formula=atom.formula,
                    )
                )
            elif atom.text:
                # Register fallback-split runs for subsetting & emit.
                sub_runs = font_engine.record_usage(atom.text, atom.slot)
                sub_x = x
                for sub_slot, sub_txt in sub_runs:
                    runs.append(
                        PositionedRun(
                            x=sub_x,
                            baseline_y=cur_y,
                            font_size_pt=font_size,
                            slot=sub_slot,
                            is_italic=atom.is_italic,
                            color_rgb=atom.color_rgb,
                            text=sub_txt,
                        )
                    )
                    sub_x += font_engine.measure_text_pt(sub_txt, sub_slot, font_size)

            x += raw_w + (extra_gap_per_break if a_idx < len(line) - 1 else 0.0)
            prev_atom = atom

        cur_y -= leading + extra_down

    total_height = by1 - (cur_y + leading - last_descent)
    return runs, max(0.0, total_height)


def typeset_paragraph(
    *,
    unit_id: str,
    page_no: int,
    role: UnitRole,
    marked_text: str,
    tokens: tuple[InlineToken, ...],
    geom: ParagraphGeometry,
    formulas: dict[str, InlineFormulaBox],
    font_engine: FontEngine,
    target_lang: str = "zh",
) -> TypesetParagraph:
    """Lay out ``marked_text`` inside ``geom.bbox`` with deterministic 2-step fit."""
    lang = (target_lang or "zh").lower().split("-")[0]
    optical_scale = 0.92 if lang in {"zh", "ja", "ko"} else 1.0
    base_size = max(5.5, geom.dominant_font_size * optical_scale)
    min_floor = 6.0 if role in {UnitRole.CAPTION, UnitRole.FOOTNOTE, UnitRole.TABLE_CELL} else 7.0
    min_size = min(base_size, min_floor)

    bx0, by0, bx1, by1 = geom.bbox
    box_w = max(8.0, bx1 - bx0)
    box_h = max(base_size * 1.1, by1 - by0)
    max_h = box_h + geom.max_borrow_down_pt
    indent = geom.first_line_indent if geom.first_line_indent < box_w * 0.25 else 0.0

    atoms = _tokenize_marked_text(marked_text, tokens, geom, formulas, font_engine)
    if not atoms:
        return TypesetParagraph(
            unit_id=unit_id,
            runs=(),
            used_font_size_pt=base_size,
            used_leading_pt=geom.line_height,
            line_count=0,
            total_height_pt=0.0,
            overflow=None,
        )

    # Candidate ladder: (scale_ratio, leading_ratio)
    # 1) Base optical size with natural leading
    # 2) Base optical size with gently tightened leading
    # 3) Progressive font scaling down to floor
    base_lead_ratio = max(1.20, min(1.42, geom.line_height / max(1.0, base_size)))
    ladder: list[tuple[float, float]] = [
        (1.00, base_lead_ratio),
        (1.00, min(base_lead_ratio, 1.16)),
        (0.95, 1.15),
        (0.90, 1.14),
        (0.85, 1.13),
        (0.80, 1.12),
        (0.75, 1.11),
        (0.70, 1.10),
    ]

    best_lines: list[list[tuple[_Atom, float]]] = []
    best_size = base_size
    best_leading = base_size * base_lead_ratio
    best_scale = 1.0

    for scale, lead_ratio in ladder:
        cand_size = max(min_size, base_size * scale)
        cand_leading = cand_size * lead_ratio
        lines = _break_into_lines(atoms, box_w, cand_size, indent, font_engine)
        est_h = (len(lines) - 1) * cand_leading + cand_size * 1.12 if lines else 0.0
        best_lines = lines
        best_size = cand_size
        best_leading = cand_leading
        best_scale = cand_size / base_size
        if est_h <= max_h + 1.0:
            break
        if cand_size <= min_size + 0.01:
            break

    runs, actual_h = _position_lines(best_lines, geom, best_size, best_leading, indent, font_engine)

    overflow: OverflowItem | None = None
    if actual_h > max_h + 1.5:
        ratio = actual_h / max(1.0, max_h)
        plain_len = len(re.sub(r"</?[a-z]\d+/?\s*>", "", marked_text).strip())
        char_budget = max(8, int(plain_len / ratio * 0.92))
        overflow = OverflowItem(
            unit_id=unit_id,
            page_no=page_no,
            overflow_ratio=round(ratio, 3),
            target_char_budget=char_budget,
            applied_font_scale=round(best_scale, 3),
            clipped=True,
        )

    return TypesetParagraph(
        unit_id=unit_id,
        runs=tuple(runs),
        used_font_size_pt=best_size,
        used_leading_pt=best_leading,
        line_count=len(best_lines),
        total_height_pt=actual_h,
        overflow=overflow,
    )
