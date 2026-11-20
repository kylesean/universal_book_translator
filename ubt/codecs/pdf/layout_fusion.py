"""Macro layout + micro content-stream fusion for native PDF translation.

Responsibilities:
1. Assign each :class:`StreamTextOp` to its enclosing :class:`LayoutRegion`.
2. Protect display formulas, figures, algorithms, and page furniture verbatim.
3. Split table regions into individual cell boxes using vector borders and x/y
   projections so original table rules/shading stay untouched.
4. Detect inline formula islands inside prose paragraphs (via ``INLINE_FORMULA``
   regions, math fonts, math Unicode, and sub/superscript geometry) and encode
   them as self-closing ``<f1/>`` rigid anchors alongside ``<bN>``/``<iN>``/``<cN>``
   inline style spans.
5. Compute per-paragraph geometry including safe downward borrow headroom within
   the same column.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from ubt.codecs.pdf.layout_detector import LayoutRegion, RegionKind
from ubt.codecs.pdf.stream_analyzer import PageStreamAnalysis, StreamTextOp, VectorRule
from ubt.codecs.protocol import InlineToken, TranslationUnit, UnitRole

_PROTECTED_KINDS: frozenset[RegionKind] = frozenset(
    {
        RegionKind.DISPLAY_FORMULA,
        RegionKind.FIGURE,
        RegionKind.ALGORITHM,
        RegionKind.FURNITURE,
    }
)

_REGION_TO_UNIT_ROLE: dict[RegionKind, UnitRole] = {
    RegionKind.PARAGRAPH: UnitRole.PARAGRAPH,
    RegionKind.HEADING: UnitRole.HEADING,
    RegionKind.CAPTION: UnitRole.CAPTION,
    RegionKind.FOOTNOTE: UnitRole.FOOTNOTE,
    RegionKind.TABLE: UnitRole.TABLE_CELL,
}

# Unicode math operators, Greek letters, mathematical alphanumeric symbols.
_MATH_CHAR_RE = re.compile(
    r"[\u0370-\u03FF\u2100-\u214F\u2190-\u21FF\u2200-\u22FF\u27C0-\u27EF\u2980-\u2AFF\U0001D400-\U0001D7FF]"
)
_BRIDGE_MATH_RE = re.compile(r"^[\s=+\-*/<>≤≥≠≈∼∈∉⊂⊆∪∩∧∨±×÷·(),\[\]{}|^_:;0-9.]+$")
_PURE_NUMERIC_RE = re.compile(r"^[\s\d.,%±+\-−/()\[\]:;·×eE]+$")
_PURE_LINK_RE = re.compile(
    r"^(?:https?://\S+|www\.\S+|(?:\{[^\s}]{1,120}\}|[A-Za-z0-9._%+\-]+)@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\.?)$",
    re.IGNORECASE,
)
_SINGLE_VAR_RE = re.compile(r"^[A-Za-z]$")
_SIMPLE_INLINE_MATH_SYMBOLS = frozenset(
    {
        "×",
        "·",
        "±",
        "−",
        "+",
        "=",
        "<",
        ">",
        "≤",
        "≥",
        "≈",
        "≠",
        "%",
        "°",
        ".",
        ",",
        ":",
        ";",
        "(",
        ")",
        "↑",
        "↓",
        "→",
        "←",
    }
)
_AFFIL_MARKER_RE = re.compile(r"^(?:[0-9]{1,2}|[*†‡§¶])(?:[,，]\s*(?:[0-9]{1,2}|[*†‡§¶]))*$")
# TeX math accents emitted in roman font (CMR) right before/over a math variable (\hat{z}, \tilde{x}, \bar{y}).
_TEX_MATH_ACCENTS: frozenset[str] = frozenset(
    "ˆ˜¯˙¨ˇ˘~^`´\u02c6\u02dc\u00af\u02d9\u00a8\u02c7\u02d8"
)
_SHORT_REF_LABEL_RE = re.compile(r"^[A-Za-z]?\.?\d[\d.,A-Za-z\-]*$")
_DATA_TOKEN_RE = re.compile(
    r"^(?:[\d.,%±+\-−–—/()\[\]:;·×eEMBKGT]+|[A-Z0-9][A-Z0-9._+\-]{0,14}|[A-Z]\.\d+(?:\.\d+)?)$"
)
_PROSE_WORD_RE = re.compile(r"\b[A-Za-z]?[a-z]{3,}\b")


def _is_non_prose_data_block(text: str) -> bool:
    """True for benchmark/table data rows made purely of numbers, dashes, and model acronyms."""
    tokens = text.split()
    if not tokens:
        return True
    prose_words = _PROSE_WORD_RE.findall(text)
    if len(prose_words) >= 2:
        return False
    data_tokens = sum(1 for tok in tokens if _DATA_TOKEN_RE.match(tok))
    if data_tokens == len(tokens):
        return True
    return len(tokens) >= 4 and data_tokens / len(tokens) >= 0.75 and len(prose_words) == 0


@dataclass(frozen=True, slots=True)
class ParagraphGeometry:
    """Exact source box & typography metadata for one fused translatable block."""

    bbox: tuple[float, float, float, float]  # logical (x0, y0, x1, y1)
    dominant_font_size: float
    line_height: float
    alignment: str  # "left" | "center" | "right" | "justify"
    first_line_indent: float
    is_serif: bool
    is_bold: bool
    color_rgb: tuple[float, float, float]
    max_borrow_down_pt: float


@dataclass(slots=True)
class FormulaIslandSpec:
    """An inline formula detected inside a paragraph, ready for Form XObject slicing."""

    tag_id: str
    ops: list[StreamTextOp]
    rules: list[VectorRule]
    prose_baseline_y: float


@dataclass(slots=True)
class FusedBlock:
    """Bridge object linking a :class:`TranslationUnit` back to source stream ops."""

    unit: TranslationUnit
    geometry: ParagraphGeometry
    erase_op_indices: list[int]
    erase_rule_op_indices: list[int]
    formula_islands: list[FormulaIslandSpec] = field(default_factory=list)


def _overlap_area(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    ix0 = max(a[0], b[0])
    iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2])
    iy1 = min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    return (ix1 - ix0) * (iy1 - iy0)


def _box_area(b: tuple[float, float, float, float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _center_in(
    op_box: tuple[float, float, float, float], reg: tuple[float, float, float, float]
) -> bool:
    cx = 0.5 * (op_box[0] + op_box[2])
    cy = 0.5 * (op_box[1] + op_box[3])
    return (reg[0] - 2.0) <= cx <= (reg[2] + 2.0) and (reg[1] - 2.0) <= cy <= (reg[3] + 2.0)


def _op_is_inside_region(op: StreamTextOp, region: LayoutRegion) -> float:
    area = _box_area(op.bbox)
    if area <= 1e-4:
        return 1.0 if _center_in(op.bbox, region.bbox) else 0.0
    return _overlap_area(op.bbox, region.bbox) / area


def _is_non_black(rgb: tuple[float, float, float]) -> bool:
    return max(rgb) > 0.18 and (max(rgb) - min(rgb)) > 0.08


def _color_hex(rgb: tuple[float, float, float]) -> str:
    r, g, b = (int(round(max(0.0, min(1.0, c)) * 255)) for c in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def _group_ops_into_lines(ops: list[StreamTextOp]) -> list[list[StreamTextOp]]:
    if not ops:
        return []
    # Sort primarily top-to-bottom by baseline, then left-to-right.
    ordered = sorted(ops, key=lambda o: (-o.baseline_y, o.bbox[0]))
    lines: list[list[StreamTextOp]] = []
    line_bases: list[float] = []
    line_sizes: list[float] = []

    for op in ordered:
        best_idx = -1
        best_dist = 1e9
        for i, base_y in enumerate(line_bases):
            # Allow superscripts/subscripts within ~0.58 * line_font_size to stay on same line.
            tol = max(2.5, line_sizes[i] * 0.58)
            dist = abs(op.baseline_y - base_y)
            if dist <= tol and dist < best_dist:
                best_dist = dist
                best_idx = i
        if best_idx >= 0:
            lines[best_idx].append(op)
            # Re-anchor baseline on larger non-math prose ops if present.
            prose_ops = [
                o
                for o in lines[best_idx]
                if not o.is_math_font and o.font_size_pt >= line_sizes[best_idx] * 0.85
            ]
            if prose_ops:
                line_bases[best_idx] = sum(o.baseline_y for o in prose_ops) / len(prose_ops)
                line_sizes[best_idx] = max(o.font_size_pt for o in prose_ops)
        else:
            lines.append([op])
            line_bases.append(op.baseline_y)
            line_sizes.append(op.font_size_pt)

    for ln in lines:
        ln.sort(key=lambda o: o.bbox[0])
    # Sort lines top-to-bottom
    paired = sorted(zip(line_bases, lines, strict=False), key=lambda pair: -pair[0])
    return [ln for _, ln in paired]


def _split_table_into_cells(
    ops: list[StreamTextOp],
    region_bbox: tuple[float, float, float, float],
    rules: list[VectorRule],
) -> list[tuple[tuple[float, float, float, float], list[StreamTextOp]]]:
    """Partition table ops into per-cell boxes using row lines + column gutters."""
    if not ops:
        return []
    lines = _group_ops_into_lines(ops)
    if not lines:
        return []

    # Find horizontal separators inside the table.
    rx0, ry0, rx1, ry1 = region_bbox
    h_rules = sorted(
        {
            round(0.5 * (r.bbox[1] + r.bbox[3]), 1)
            for r in rules
            if r.is_horizontal
            and r.bbox[0] < rx1
            and r.bbox[2] > rx0
            and ry0 <= 0.5 * (r.bbox[1] + r.bbox[3]) <= ry1
            and (r.bbox[2] - r.bbox[0]) >= (rx1 - rx0) * 0.35
        },
        reverse=True,
    )

    # Split each visual line into horizontal cell clusters separated by wide gaps.
    cells: list[tuple[tuple[float, float, float, float], list[StreamTextOp]]] = []
    for ln in lines:
        if not ln:
            continue
        body_fs = sorted(o.font_size_pt for o in ln)[len(ln) // 2]
        gap_thresh = max(8.0, body_fs * 1.15)
        clusters: list[list[StreamTextOp]] = [[ln[0]]]
        for op in ln[1:]:
            prev_x1 = max(o.bbox[2] for o in clusters[-1])
            if op.bbox[0] - prev_x1 > gap_thresh:
                clusters.append([op])
            else:
                clusters[-1].append(op)
        for cl in clusters:
            cx0 = min(o.bbox[0] for o in cl)
            cy0 = min(o.bbox[1] for o in cl)
            cx1 = max(o.bbox[2] for o in cl)
            cy1 = max(o.bbox[3] for o in cl)
            # Extend cell width slightly toward right neighbor gap (never past table right)
            cells.append(((cx0, cy0, min(rx1, cx1 + 4.0), cy1), cl))

    # Merge vertically stacked lines within the same column & row band (between h_rules).
    def _row_band(cy: float) -> int:
        band = 0
        for hy in h_rules:
            if cy < hy:
                band += 1
        return band

    merged: list[tuple[tuple[float, float, float, float], list[StreamTextOp]]] = []
    for box, cl in cells:
        placed = False
        b_band = _row_band(0.5 * (box[1] + box[3]))
        for idx, (mbox, mops) in enumerate(merged):
            m_band = _row_band(0.5 * (mbox[1] + mbox[3]))
            if b_band != m_band:
                continue
            # Same horizontal column overlap and vertically adjacent (<= 1.4x line height)
            h_ov = min(box[2], mbox[2]) - max(box[0], mbox[0])
            min_w = max(1.0, min(box[2] - box[0], mbox[2] - mbox[0]))
            v_gap = mbox[1] - box[3]
            fs = max(o.font_size_pt for o in mops)
            if h_ov / min_w >= 0.45 and -3.0 <= v_gap <= fs * 0.85:
                new_box = (
                    min(mbox[0], box[0]),
                    min(mbox[1], box[1]),
                    max(mbox[2], box[2]),
                    max(mbox[3], box[3]),
                )
                merged[idx] = (new_box, mops + cl)
                placed = True
                break
        if not placed:
            merged.append((box, cl))

    return merged


def _is_math_op(
    op: StreamTextOp,
    line_base_y: float,
    body_size: float,
    inline_formula_regions: list[LayoutRegion],
) -> bool:
    if any(_op_is_inside_region(op, reg) >= 0.45 for reg in inline_formula_regions):
        return True
    if op.is_math_font:
        return True
    stripped = op.text.strip()
    if not stripped:
        return False
    if _MATH_CHAR_RE.search(stripped):
        return True
    # Subscript or superscript shifted significantly off baseline with smaller font
    if abs(op.baseline_y - line_base_y) > body_size * 0.16 and op.font_size_pt < body_size * 0.82:
        return True
    # Single-letter italic variable (not common English pronoun "I" or "a" standing alone)
    return bool(op.is_italic and _SINGLE_VAR_RE.match(stripped) and stripped not in {"a", "I"})


def _detect_line_formula_flags(
    line_ops: list[StreamTextOp],
    line_base_y: float,
    body_size: float,
    inline_formula_regions: list[LayoutRegion],
) -> list[bool]:
    flags = [_is_math_op(op, line_base_y, body_size, inline_formula_regions) for op in line_ops]
    # Absorb TeX roman-font math accents (\hat, \tilde, \bar, \dot) positioned over/adjacent to
    # a math op so ``\hat{z}^{(k-1)}`` becomes one intact Form XObject instead of ``ˆ<f7/>``.
    n_ops = len(line_ops)
    for i, op in enumerate(line_ops):
        if flags[i]:
            continue
        txt = op.text.strip()
        if not txt or len(txt) > 2 or not all(ch in _TEX_MATH_ACCENTS for ch in txt):
            continue
        adj_right = (
            i + 1 < n_ops
            and flags[i + 1]
            and (line_ops[i + 1].bbox[0] - op.bbox[2]) < body_size * 0.6
        )
        adj_left = (
            i > 0 and flags[i - 1] and (op.bbox[0] - line_ops[i - 1].bbox[2]) < body_size * 0.6
        )
        if adj_right or adj_left:
            flags[i] = True
    # Absorb a short base variable (e.g. ``n``, ``V``, ``k`` in custom-encoded publisher fonts
    # lacking Italic flags) immediately preceding a tight subscript/superscript math op so
    # ``n_i`` / ``V_t`` / ``k_B`` stay a single Form XObject instead of ``n<f1/>``.
    for i, op in enumerate(line_ops):
        if flags[i] or i + 1 >= n_ops or not flags[i + 1]:
            continue
        txt = op.text.strip()
        if not txt or len(txt) > 2 or not (txt.isalpha() or not op.has_tounicode):
            continue
        nxt = line_ops[i + 1]
        is_sub_or_sup = (
            abs(nxt.baseline_y - line_base_y) > body_size * 0.14
            and nxt.font_size_pt < body_size * 0.85
        )
        if is_sub_or_sup and (nxt.bbox[0] - op.bbox[2]) < body_size * 0.35:
            flags[i] = True
    # Bridge operators/parentheses/numbers sandwiched between two math ops on same line.
    changed = True
    while changed:
        changed = False
        for i, op in enumerate(line_ops):
            if flags[i]:
                continue
            txt = op.text.strip()
            if not txt or not _BRIDGE_MATH_RE.match(txt):
                continue
            left_math = (
                i > 0 and flags[i - 1] and (op.bbox[0] - line_ops[i - 1].bbox[2]) < body_size * 1.2
            )
            right_math = (
                i + 1 < len(line_ops)
                and flags[i + 1]
                and (line_ops[i + 1].bbox[0] - op.bbox[2]) < body_size * 1.2
            )
            if left_math and right_math:
                flags[i] = True
                changed = True

    # Unflag isolated single-glyph symbols natively covered by CJK/Latin fallback fonts
    # and standalone superscript affiliation/footnote markers outside inline_formula regions.
    n = len(line_ops)
    for i, op in enumerate(line_ops):
        if not flags[i]:
            continue
        prev_math = i > 0 and flags[i - 1]
        next_math = i + 1 < n and flags[i + 1]
        if prev_math or next_math:
            continue
        if any(_op_is_inside_region(op, reg) >= 0.45 for reg in inline_formula_regions):
            continue
        txt = op.text.strip()
        if txt in _SIMPLE_INLINE_MATH_SYMBOLS:
            flags[i] = False
            continue
        if (
            not op.is_math_font
            and abs(op.baseline_y - line_base_y) > body_size * 0.16
            and op.font_size_pt < body_size * 0.82
            and _AFFIL_MARKER_RE.match(txt)
            and (
                i == 0
                or (op.bbox[0] - line_ops[i - 1].bbox[2]) >= body_size * 0.1
                or line_ops[i - 1].text.endswith((" ", ",", ")"))
            )
        ):
            flags[i] = False
    return flags


def _find_matching_fraction_rules(
    island_ops: list[StreamTextOp],
    vector_rules: list[VectorRule],
) -> list[VectorRule]:
    if not island_ops or not vector_rules:
        return []
    ix0 = min(o.bbox[0] for o in island_ops) - 2.0
    iy0 = min(o.bbox[1] for o in island_ops) - 2.0
    ix1 = max(o.bbox[2] for o in island_ops) + 2.0
    iy1 = max(o.bbox[3] for o in island_ops) + 2.0
    matched: list[VectorRule] = []
    for r in vector_rules:
        if not r.is_horizontal:
            continue
        rw = r.bbox[2] - r.bbox[0]
        if rw > (ix1 - ix0) + 6.0:
            continue
        if r.bbox[0] >= ix0 - 2.0 and r.bbox[2] <= ix1 + 2.0 and iy0 <= r.bbox[1] <= iy1:
            matched.append(r)
    return matched


def _infer_alignment(
    lines: list[list[StreamTextOp]],
    box: tuple[float, float, float, float],
    role: UnitRole,
) -> tuple[str, float]:
    bx0, _, bx1, _ = box
    bw = max(1.0, bx1 - bx0)
    if not lines:
        return "left", 0.0

    first_x0 = min(o.bbox[0] for o in lines[0])
    indent = max(0.0, first_x0 - bx0)
    if len(lines) == 1:
        lx0 = first_x0
        lx1 = max(o.bbox[2] for o in lines[0])
        left_margin = lx0 - bx0
        right_margin = bx1 - lx1
        if (
            role == UnitRole.HEADING
            and abs(left_margin - right_margin) <= 6.0
            and left_margin > 8.0
        ):
            return "center", 0.0
        return "left", indent

    # Multi-line paragraph: check if non-final lines reach close to right edge (justified).
    inner_lines = lines[:-1]
    right_fills = [(max(o.bbox[2] for o in ln) - bx0) / bw for ln in inner_lines]
    if right_fills and sum(1 for f in right_fills if f >= 0.92) >= max(
        1, int(len(right_fills) * 0.65)
    ):
        return "justify", indent
    return "left", indent


def _build_fused_block(
    *,
    unit_id: str,
    chapter_id: str,
    page_no: int,
    role: UnitRole,
    region_bbox: tuple[float, float, float, float],
    ops: list[StreamTextOp],
    inline_formula_regions: list[LayoutRegion],
    vector_rules: list[VectorRule],
    max_borrow_down_pt: float,
) -> FusedBlock | None:
    lines = _group_ops_into_lines(ops)
    if not lines:
        return None

    # Dominant typography across non-math prose ops.
    prose_ops = [o for o in ops if not o.is_math_font and o.text.strip()]
    sample_ops = prose_ops or ops
    sizes = sorted(o.font_size_pt for o in sample_ops)
    dom_size = sizes[len(sizes) // 2]
    dom_serif = sum(len(o.text) for o in sample_ops if o.is_serif) >= sum(
        len(o.text) for o in sample_ops if not o.is_serif
    )
    dom_bold = sum(len(o.text) for o in sample_ops if o.is_bold) > sum(
        len(o.text) for o in sample_ops if not o.is_bold
    )
    color_counts = Counter(
        (round(o.color_rgb[0], 2), round(o.color_rgb[1], 2), round(o.color_rgb[2], 2))
        for o in sample_ops
    )
    dom_color = color_counts.most_common(1)[0][0]

    # Measure empirical line height from consecutive line baselines.
    line_bases = [sum(o.baseline_y for o in ln) / len(ln) for ln in lines]
    if len(line_bases) >= 2:
        diffs = [
            line_bases[i] - line_bases[i + 1]
            for i in range(len(line_bases) - 1)
            if line_bases[i] > line_bases[i + 1]
        ]
        line_height = sorted(diffs)[len(diffs) // 2] if diffs else dom_size * 1.25
    else:
        line_height = dom_size * 1.25

    # Tight union bbox of actual ops, widened to region width for natural reflow.
    op_x0 = min(o.bbox[0] for o in ops)
    op_y0 = min(o.bbox[1] for o in ops)
    op_x1 = max(o.bbox[2] for o in ops)
    op_y1 = max(o.bbox[3] for o in ops)
    box = (
        min(op_x0, region_bbox[0]) if abs(op_x0 - region_bbox[0]) < dom_size * 2.5 else op_x0,
        min(op_y0, region_bbox[1]),
        max(op_x1, region_bbox[2]) if abs(region_bbox[2] - op_x1) < dom_size * 4.0 else op_x1,
        max(op_y1, region_bbox[3]),
    )
    alignment, first_indent = _infer_alignment(lines, box, role)

    marked_parts: list[str] = []
    plain_parts: list[str] = []
    tokens: list[InlineToken] = []
    formula_islands: list[FormulaIslandSpec] = []
    erase_rule_ops: list[int] = []

    formula_seq = 0
    style_seq = 0

    for line_idx, ln in enumerate(lines):
        # Compute prose baseline for this line.
        ln_prose = [o for o in ln if not o.is_math_font and o.font_size_pt >= dom_size * 0.8]
        line_base_y = (
            sum(o.baseline_y for o in ln_prose) / len(ln_prose)
            if ln_prose
            else sum(o.baseline_y for o in ln) / len(ln)
        )
        mflags = _detect_line_formula_flags(ln, line_base_y, dom_size, inline_formula_regions)

        if line_idx > 0 and marked_parts:
            # De-hyphenate wrapped English words at line break.
            prev_s = marked_parts[-1]
            if prev_s.endswith("-") and len(prev_s) >= 2 and prev_s[-2].isalpha():
                marked_parts[-1] = prev_s[:-1]
                if plain_parts and plain_parts[-1].endswith("-"):
                    plain_parts[-1] = plain_parts[-1][:-1]
            else:
                marked_parts.append(" ")
                plain_parts.append(" ")

        i = 0
        n = len(ln)
        while i < n:
            # Inter-op horizontal space detection on same line.
            if i > 0:
                gap = ln[i].bbox[0] - ln[i - 1].bbox[2]
                if gap > dom_size * 0.16 and not (marked_parts and marked_parts[-1].endswith(" ")):
                    marked_parts.append(" ")
                    plain_parts.append(" ")

            if mflags[i]:
                j = i + 1
                while j < n and mflags[j] and (ln[j].bbox[0] - ln[j - 1].bbox[2]) < dom_size * 1.4:
                    j += 1
                island_ops = ln[i:j]
                formula_seq += 1
                tag_id = f"f{formula_seq}"
                matched_rules = _find_matching_fraction_rules(island_ops, vector_rules)
                erase_rule_ops.extend(idx for r in matched_rules for idx in r.op_indices)
                formula_islands.append(
                    FormulaIslandSpec(
                        tag_id=tag_id,
                        ops=island_ops,
                        rules=matched_rules,
                        prose_baseline_y=line_base_y,
                    )
                )
                tokens.append(
                    InlineToken(
                        tag_id=tag_id,
                        kind="rigid",
                        payload={
                            "type": "formula",
                            "raw_text": "".join(o.text for o in island_ops),
                            "op_indices": [o.op_index for o in island_ops],
                        },
                    )
                )
                marked_parts.append(f"<{tag_id}/>")
                plain_parts.append("".join(o.text for o in island_ops))
                i = j
                continue

            # Regular prose op: coalesce contiguous ops sharing identical inline style diff.
            op = ln[i]
            rel_bold = op.is_bold and not dom_bold
            rel_italic = op.is_italic
            rel_color = (
                _color_hex(op.color_rgb)
                if _is_non_black(op.color_rgb) and op.color_rgb != dom_color
                else None
            )
            style_key = (rel_bold, rel_italic, rel_color)

            chunk_texts = [op.text]
            j = i + 1
            while j < n and not mflags[j]:
                nxt = ln[j]
                nxt_key = (
                    nxt.is_bold and not dom_bold,
                    nxt.is_italic,
                    _color_hex(nxt.color_rgb)
                    if _is_non_black(nxt.color_rgb) and nxt.color_rgb != dom_color
                    else None,
                )
                if nxt_key != style_key:
                    break
                gap = nxt.bbox[0] - ln[j - 1].bbox[2]
                if (
                    gap > dom_size * 0.16
                    and not chunk_texts[-1].endswith(" ")
                    and not nxt.text.startswith(" ")
                ):
                    chunk_texts.append(" ")
                chunk_texts.append(nxt.text)
                j += 1

            seg = "".join(chunk_texts)
            plain_parts.append(seg)
            core = seg.strip()
            # Do not wrap color-only hyperref citation/section numbers ([1, 2], Figure 5,
            # Section 4.6) in <cN> tags: PDF /Annots keep links clickable while avoiding
            # breaking CitationMasker / AddedContentGate and confusing smaller LLMs.
            emit_style_tag = bool((rel_bold or rel_italic or rel_color) and core)
            if (
                emit_style_tag
                and rel_color
                and not rel_bold
                and not rel_italic
                and _SHORT_REF_LABEL_RE.match(core)
            ):
                emit_style_tag = False
            if emit_style_tag:
                style_seq += 1
                prefix = "b" if rel_bold else ("i" if rel_italic else "c")
                tag_id = f"{prefix}{style_seq}"
                tokens.append(
                    InlineToken(
                        tag_id=tag_id,
                        kind="pair",
                        payload={
                            "bold": rel_bold,
                            "italic": rel_italic,
                            "color_hex": rel_color,
                        },
                    )
                )
                # Keep surrounding spaces outside the inline tag for cleaner LLM input.
                lead_sp = seg[: len(seg) - len(seg.lstrip())]
                trail_sp = seg[len(seg.rstrip()) :]
                marked_parts.append(f"{lead_sp}<{tag_id}>{core}</{tag_id}>{trail_sp}")
            else:
                marked_parts.append(seg)
            i = j

    source_marked = re.sub(r" {2,}", " ", "".join(marked_parts)).strip()
    source_plain = re.sub(r" {2,}", " ", "".join(plain_parts)).strip()
    if not source_plain:
        return None

    # Skip pure numeric cells, bare links/emails, benchmark data rows, or formula-only blocks.
    non_tag_text = re.sub(r"</?[a-z]\d+/?\s*>", "", source_marked).strip()
    translate = (
        bool(non_tag_text)
        and not _PURE_NUMERIC_RE.match(non_tag_text)
        and not _PURE_LINK_RE.match(non_tag_text)
        and not _is_non_prose_data_block(non_tag_text)
    )

    geom = ParagraphGeometry(
        bbox=box,
        dominant_font_size=dom_size,
        line_height=line_height,
        alignment=alignment,
        first_line_indent=first_indent,
        is_serif=dom_serif,
        is_bold=dom_bold,
        color_rgb=dom_color,
        max_borrow_down_pt=max_borrow_down_pt,
    )
    unit = TranslationUnit(
        unit_id=unit_id,
        chapter_id=chapter_id,
        role=role,
        source_marked=source_marked,
        source_plain=source_plain,
        tokens=tuple(tokens),
        translate=translate,
        page_no=page_no,
        box_hint={
            "width_pt": box[2] - box[0],
            "height_pt": box[3] - box[1],
            "font_size_pt": dom_size,
            "line_height_pt": line_height,
        },
        codec_meta={
            "bbox": list(box),
            "alignment": alignment,
            "is_serif": dom_serif,
            "is_bold": dom_bold,
        },
    )
    return FusedBlock(
        unit=unit,
        geometry=geom,
        erase_op_indices=[o.op_index for o in ops],
        erase_rule_op_indices=sorted(set(erase_rule_ops)),
        formula_islands=formula_islands,
    )


def _compute_borrow_down(
    box: tuple[float, float, float, float],
    all_boxes: list[tuple[float, float, float, float]],
    page_bottom_margin: float = 28.0,
) -> float:
    """Measure safe vertical whitespace directly below ``box`` in the same column."""
    x0, y0, x1, _ = box
    nearest_below = page_bottom_margin
    for ox0, oy0, ox1, oy1 in all_boxes:
        if (ox0, oy0, ox1, oy1) == box:
            continue
        # Horizontal overlap with this column?
        if min(x1, ox1) - max(x0, ox0) <= 4.0:
            continue
        if oy1 <= y0 + 1.0:
            nearest_below = max(nearest_below, oy1)
    # Keep a 4.0pt safety gutter above the next element.
    return max(0.0, y0 - nearest_below - 4.0)


def fuse_page(
    analysis: PageStreamAnalysis,
    regions: list[LayoutRegion],
    *,
    chapter_id: str | None = None,
) -> list[FusedBlock]:
    """Fuse macro layout regions and micro stream ops for one page."""
    chap = chapter_id or f"page_{analysis.page_no:04d}"
    protected = [r for r in regions if r.kind in _PROTECTED_KINDS]
    inline_formulas = [r for r in regions if r.kind == RegionKind.INLINE_FORMULA]
    translatable = sorted(
        [r for r in regions if r.kind in _REGION_TO_UNIT_ROLE],
        key=lambda r: r.reading_order,
    )

    # Assign each text op to either protected (untouched) or best translatable region.
    region_ops: dict[int, list[StreamTextOp]] = {idx: [] for idx in range(len(translatable))}
    for op in analysis.text_ops:
        if not op.text.strip():
            continue
        if any(_op_is_inside_region(op, pr) >= 0.45 for pr in protected):
            continue
        best_idx = -1
        best_score = 0.0
        for idx, tr in enumerate(translatable):
            ov = _op_is_inside_region(op, tr)
            if ov > best_score:
                best_score = ov
                best_idx = idx
        if best_idx >= 0 and best_score >= 0.30:
            region_ops[best_idx].append(op)

    # Collect obstacle boxes for safe downward borrow calculation.
    obstacle_boxes: list[tuple[float, float, float, float]] = [r.bbox for r in regions]

    fused: list[FusedBlock] = []
    unit_seq = 0
    for idx, region in enumerate(translatable):
        ops = region_ops[idx]
        if not ops:
            continue
        if region.kind == RegionKind.TABLE:
            cells = _split_table_into_cells(ops, region.bbox, analysis.vector_rules)
            for cell_box, cell_ops in cells:
                unit_seq += 1
                block = _build_fused_block(
                    unit_id=f"p{analysis.page_no:04d}_u{unit_seq:04d}",
                    chapter_id=chap,
                    page_no=analysis.page_no,
                    role=UnitRole.TABLE_CELL,
                    region_bbox=cell_box,
                    ops=cell_ops,
                    inline_formula_regions=inline_formulas,
                    vector_rules=analysis.vector_rules,
                    max_borrow_down_pt=0.0,  # Never spill outside table cell border
                )
                if block is not None:
                    fused.append(block)
        else:
            unit_seq += 1
            borrow = _compute_borrow_down(region.bbox, obstacle_boxes)
            block = _build_fused_block(
                unit_id=f"p{analysis.page_no:04d}_u{unit_seq:04d}",
                chapter_id=chap,
                page_no=analysis.page_no,
                role=_REGION_TO_UNIT_ROLE[region.kind],
                region_bbox=region.bbox,
                ops=ops,
                inline_formula_regions=inline_formulas,
                vector_rules=analysis.vector_rules,
                max_borrow_down_pt=borrow,
            )
            if block is not None:
                fused.append(block)

    return fused
