"""PDF content-stream analyzer: instruction-indexed text ops, graphics snapshots & rules.

Single-pass parser over a page's ``pikepdf`` content stream that produces:
- Every text-painting instruction (``Tj``, ``TJ``, ``'``, ``"``) indexed by its
  exact position in ``pikepdf.parse_content_stream(page)`` so surgical erasure
  can blank payloads with 0.0 pt drift.
- Full graphics & text state snapshots at each instruction so inline formulas
  can be sliced into 200-byte Form XObjects without copying the entire page.
- Canonical upright logical coordinates ``(0, 0) -> (page_width, page_height)``
  with ``/CropBox`` origin and ``/Rotate`` normalized once at ingest.
- Optional ``pypdfium2`` Unicode recovery for legacy Type1 fonts lacking a
  ``/ToUnicode`` CMap (such as ``chapter-3.pdf``).
"""

from __future__ import annotations

import contextlib
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pikepdf

#: 3x2 affine transform in PDF row-vector convention: ``p' = p * M``.
Matrix = tuple[float, float, float, float, float, float]
IDENTITY_MATRIX: Matrix = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)

_SHOW_OPS: frozenset[str] = frozenset({"Tj", "TJ", "'", '"'})

_MATH_FONT_RE = re.compile(
    r"(?:^|[+\-_])(?:"
    r"CMMI|CMSY|CMEX|CMBSY|MSAM|MSBM|EUFM|EUSM|EURM|EUBM|"
    r"STIX.*Math|CambriaMath|Asana-Math|LatinModernMath|LMMath|"
    r"MTMI|MTSY|MTEX|MathDesign|TXMI|TXSY|PXMI|PXSY|"
    r"Symbol|Mathematical|Mathematica|NewTXMI|NewTXSY|rtxmi|rtxr|zplmr|"
    r"DejaVuMath|FiraMath|LibertinusMath|XITS.*Math"
    r")",
    re.IGNORECASE,
)

_BOLD_FONT_RE = re.compile(
    r"(?:Bold|Black|Heavy|Semibold|Demibold|CMBX|CMSSBX|NimbusSanL-Bold|URWPalladioL-Bold|NimbusRomNo9L-Medi)",
    re.IGNORECASE,
)
_ITALIC_FONT_RE = re.compile(
    r"(?:Italic|Oblique|Slant|CMTI|CMSL|CMMI|NimbusRomNo9L-ReguItal|URWPalladioL-Ital)",
    re.IGNORECASE,
)
_SANS_FONT_RE = re.compile(
    r"(?:Sans|Helvetica|Arial|CMSS|NimbusSan|DejaVuSans|Calibri|Roboto|Inter|Lato|Myriad|Frutiger|Futura|Gill)",
    re.IGNORECASE,
)

_BFCHAR_RE = re.compile(r"beginbfchar\s*(.*?)\s*endbfchar", re.DOTALL)
_BFRANGE_RE = re.compile(r"beginbfrange\s*(.*?)\s*endbfrange", re.DOTALL)
_HEX_RE = re.compile(r"<([0-9A-Fa-f]+)>")


def mat_mul(a: Matrix, b: Matrix) -> Matrix:
    """Multiply two PDF row-vector affine matrices: ``(p * a) * b``."""
    return (
        a[0] * b[0] + a[1] * b[2],
        a[0] * b[1] + a[1] * b[3],
        a[2] * b[0] + a[3] * b[2],
        a[2] * b[1] + a[3] * b[3],
        a[4] * b[0] + a[5] * b[2] + b[4],
        a[4] * b[1] + a[5] * b[3] + b[5],
    )


def mat_apply(m: Matrix, x: float, y: float) -> tuple[float, float]:
    return (x * m[0] + y * m[2] + m[4], x * m[1] + y * m[3] + m[5])


def mat_invert(m: Matrix) -> Matrix:
    a, b, c, d, e, f = m
    det = a * d - b * c
    if abs(det) < 1e-12:
        return IDENTITY_MATRIX
    inv_det = 1.0 / det
    ia = d * inv_det
    ib = -b * inv_det
    ic = -c * inv_det
    id_ = a * inv_det
    ie = -(e * ia + f * ic)
    if_ = -(e * ib + f * id_)
    return (ia, ib, ic, id_, ie, if_)


def _compute_page_transforms(
    page: pikepdf.Page,
) -> tuple[float, float, Matrix, Matrix, tuple[float, float, float, float], int]:
    """Return ``(width, height, stream_to_logical, logical_to_stream, cropbox, rotate)``."""
    raw_box = page.get("/CropBox") or page.get("/MediaBox")
    if raw_box is None or len(raw_box) < 4:
        cx0, cy0, cx1, cy1 = 0.0, 0.0, 612.0, 792.0
    else:
        vals = [float(v) for v in raw_box[:4]]
        cx0, cy0 = min(vals[0], vals[2]), min(vals[1], vals[3])
        cx1, cy1 = max(vals[0], vals[2]), max(vals[1], vals[3])
    w = max(1.0, cx1 - cx0)
    h = max(1.0, cy1 - cy0)
    rotate = int(page.get("/Rotate", 0)) % 360

    # Translate crop origin to (0, 0), then apply clockwise page /Rotate into
    # upright viewer coordinates (origin bottom-left, +x right, +y up).
    t_origin: Matrix = (1.0, 0.0, 0.0, 1.0, -cx0, -cy0)
    if rotate == 90:
        r_mat: Matrix = (0.0, -1.0, 1.0, 0.0, 0.0, w)
        vis_w, vis_h = h, w
    elif rotate == 180:
        r_mat = (-1.0, 0.0, 0.0, -1.0, w, h)
        vis_w, vis_h = w, h
    elif rotate == 270:
        r_mat = (0.0, 1.0, -1.0, 0.0, h, 0.0)
        vis_w, vis_h = h, w
    else:
        r_mat = IDENTITY_MATRIX
        vis_w, vis_h = w, h

    s2l = mat_mul(t_origin, r_mat)
    l2s = mat_invert(s2l)
    return vis_w, vis_h, s2l, l2s, (cx0, cy0, cx1, cy1), rotate


@dataclass(slots=True)
class _ParsedFont:
    resource: str
    base_name: str
    is_two_byte: bool
    is_bold: bool
    is_italic: bool
    is_serif: bool
    is_math_font: bool
    has_tounicode: bool
    to_unicode: dict[int, str]
    widths: dict[int, float]  # glyph code -> width in 1/1000 em
    default_width: float = 500.0
    differences: dict[int, str] = field(default_factory=dict)

    def width_em(self, code: int) -> float:
        return self.widths.get(code, self.default_width) / 1000.0

    def decode_bytes(self, raw: bytes) -> tuple[str, list[int]]:
        codes: list[int] = []
        if self.is_two_byte and len(raw) >= 2 and len(raw) % 2 == 0:
            for i in range(0, len(raw), 2):
                codes.append((raw[i] << 8) | raw[i + 1])
        else:
            codes = list(raw)

        chars: list[str] = []
        for code in codes:
            mapped = self.to_unicode.get(code)
            if mapped is not None:
                chars.append(mapped)
            elif code in self.differences:
                chars.append(self.differences[code])
            elif 32 <= code <= 126:
                chars.append(chr(code))
            elif not self.is_two_byte and code in _WINANSI_EXT:
                chars.append(_WINANSI_EXT[code])
            else:
                chars.append("")
        return "".join(chars), codes


_WINANSI_EXT: dict[int, str] = {
    0x85: "…",
    0x91: "‘",
    0x92: "’",
    0x93: "“",
    0x94: "”",
    0x95: "•",
    0x96: "–",
    0x97: "—",
    0xA0: " ",
    0xB0: "°",
    0xB1: "±",
    0xD7: "×",
    0xF7: "÷",
}

_GLYPH_NAME_FALLBACK: dict[str, str] = {
    "fi": "fi",
    "fl": "fl",
    "ff": "ff",
    "ffi": "ffi",
    "ffl": "ffl",
    "endash": "–",
    "emdash": "—",
    "quotedblleft": "“",
    "quotedblright": "”",
    "quoteleft": "‘",
    "quoteright": "’",
    "bullet": "•",
    "minus": "−",
    "multiply": "×",
    "divide": "÷",
    "plusminus": "±",
    "degree": "°",
    "space": " ",
    "hyphen": "-",
    "period": ".",
    "comma": ",",
    "colon": ":",
    "semicolon": ";",
    "parenleft": "(",
    "parenright": ")",
    "bracketleft": "[",
    "bracketright": "]",
    "braceleft": "{",
    "braceright": "}",
    "slash": "/",
    "backslash": "\\",
    "equal": "=",
    "plus": "+",
    "less": "<",
    "greater": ">",
    "percent": "%",
    "ampersand": "&",
    "at": "@",
    "numbersign": "#",
    "dollar": "$",
    "asterisk": "*",
    "question": "?",
    "exclam": "!",
    "quotesingle": "'",
    "quotedbl": '"',
    "underscore": "_",
    "bar": "|",
    "asciitilde": "~",
    "asciicircum": "^",
}


def _hex_to_str(hex_s: str) -> str:
    raw = bytes.fromhex(hex_s if len(hex_s) % 2 == 0 else "0" + hex_s)
    if len(raw) >= 2 and len(raw) % 2 == 0:
        try:
            return raw.decode("utf-16-be")
        except UnicodeDecodeError:
            pass
    return raw.decode("latin-1", errors="ignore")


def _parse_tounicode_stream(data: bytes) -> dict[int, str]:
    text = data.decode("latin-1", errors="ignore")
    mapping: dict[int, str] = {}
    for block in _BFCHAR_RE.findall(text):
        tokens = _HEX_RE.findall(block)
        for i in range(0, len(tokens) - 1, 2):
            src = int(tokens[i], 16)
            mapping[src] = _hex_to_str(tokens[i + 1])
    for block in _BFRANGE_RE.findall(text):
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        for ln in lines:
            if "[" in ln:
                head, _, rest = ln.partition("[")
                ends = _HEX_RE.findall(head)
                vals = _HEX_RE.findall(rest)
                if len(ends) >= 2:
                    start = int(ends[0], 16)
                    for idx, vh in enumerate(vals):
                        mapping[start + idx] = _hex_to_str(vh)
            else:
                toks = _HEX_RE.findall(ln)
                if len(toks) >= 3:
                    start = int(toks[0], 16)
                    end = int(toks[1], 16)
                    dst_base = int(toks[2], 16)
                    if end - start <= 4096:
                        for offset in range(end - start + 1):
                            mapping[start + offset] = chr(dst_base + offset)
    return mapping


def _parse_cid_widths(w_obj: Any) -> dict[int, float]:
    widths: dict[int, float] = {}
    if not isinstance(w_obj, pikepdf.Array):
        return widths
    i = 0
    n = len(w_obj)
    while i < n:
        start = int(w_obj[i])
        i += 1
        if i >= n:
            break
        nxt = w_obj[i]
        if isinstance(nxt, pikepdf.Array):
            for offset, w in enumerate(nxt):
                widths[start + offset] = float(w)
            i += 1
        else:
            if i + 1 < n:
                end = int(nxt)
                w_val = float(w_obj[i + 1])
                if end - start <= 65536:
                    for code in range(start, end + 1):
                        widths[code] = w_val
                i += 2
            else:
                break
    return widths


def _parse_page_fonts(page: pikepdf.Page) -> dict[str, _ParsedFont]:
    result: dict[str, _ParsedFont] = {}
    resources = page.get("/Resources")
    if not isinstance(resources, pikepdf.Dictionary):
        return result
    fonts_dict = resources.get("/Font")
    if not isinstance(fonts_dict, pikepdf.Dictionary):
        return result

    for raw_key, fobj in fonts_dict.items():
        res_name = str(raw_key)
        if not res_name.startswith("/"):
            res_name = "/" + res_name
        if not isinstance(fobj, pikepdf.Dictionary):
            continue
        base_font = str(fobj.get("/BaseFont", "")).lstrip("/")
        subtype = str(fobj.get("/Subtype", ""))
        is_two_byte = subtype == "/Type0"

        to_unicode: dict[int, str] = {}
        tu_stream = fobj.get("/ToUnicode")
        has_tounicode = False
        if isinstance(tu_stream, pikepdf.Stream):
            try:
                to_unicode = _parse_tounicode_stream(bytes(tu_stream.read_bytes()))
                has_tounicode = bool(to_unicode)
            except Exception:
                has_tounicode = False

        widths: dict[int, float] = {}
        default_w = 1000.0 if is_two_byte else 500.0
        if is_two_byte:
            desc_arr = fobj.get("/DescendantFonts")
            if isinstance(desc_arr, pikepdf.Array) and len(desc_arr) > 0:
                desc0 = desc_arr[0]
                if isinstance(desc0, pikepdf.Dictionary):
                    if "/DW" in desc0:
                        default_w = float(desc0["/DW"])
                    if "/W" in desc0:
                        widths = _parse_cid_widths(desc0["/W"])
        else:
            first_char = int(fobj.get("/FirstChar", 0))
            w_arr = fobj.get("/Widths")
            if isinstance(w_arr, pikepdf.Array):
                for offset, wv in enumerate(w_arr):
                    widths[first_char + offset] = float(wv)
                if widths:
                    default_w = sum(widths.values()) / len(widths)

        differences: dict[int, str] = {}
        enc = fobj.get("/Encoding")
        if isinstance(enc, pikepdf.Dictionary):
            diffs = enc.get("/Differences")
            if isinstance(diffs, pikepdf.Array):
                cur_code = 0
                for item in diffs:
                    if isinstance(item, pikepdf.Name):
                        gname = str(item).lstrip("/")
                        if gname in _GLYPH_NAME_FALLBACK:
                            differences[cur_code] = _GLYPH_NAME_FALLBACK[gname]
                        elif len(gname) == 1:
                            differences[cur_code] = gname
                        cur_code += 1
                    else:
                        with contextlib.suppress(TypeError, ValueError):
                            cur_code = int(item)

        is_math = bool(_MATH_FONT_RE.search(base_font))
        is_bold = bool(_BOLD_FONT_RE.search(base_font))
        is_italic = bool(_ITALIC_FONT_RE.search(base_font))
        is_serif = not bool(_SANS_FONT_RE.search(base_font))
        result[res_name] = _ParsedFont(
            resource=res_name,
            base_name=base_font,
            is_two_byte=is_two_byte,
            is_bold=is_bold,
            is_italic=is_italic,
            is_serif=is_serif,
            is_math_font=is_math,
            has_tounicode=has_tounicode,
            to_unicode=to_unicode,
            widths=widths,
            default_width=default_w,
            differences=differences,
        )
    return result


@dataclass(frozen=True, slots=True)
class OpGraphicsSnapshot:
    """Minimal PDF graphics & text state needed to replay one text op inside a Form XObject."""

    ctm: Matrix
    text_matrix: Matrix
    font_resource: str
    font_size_operand: float
    tc: float
    tw: float
    tz: float
    ts: float
    color_rgb: tuple[float, float, float]


@dataclass(slots=True)
class StreamTextOp:
    """One text-painting instruction from the page's content stream."""

    op_index: int
    operator: str
    text: str
    bbox: tuple[float, float, float, float]  # logical (x0, y0, x1, y1)
    baseline_y: float
    font_resource: str
    font_name: str
    font_size_pt: float
    color_rgb: tuple[float, float, float]
    is_bold: bool
    is_italic: bool
    is_serif: bool
    is_math_font: bool
    has_tounicode: bool
    rise_pt: float
    snapshot: OpGraphicsSnapshot
    raw_operands: tuple[Any, ...]
    text_advance: float = 0.0
    tj_item_range: tuple[int, int] | None = None


@dataclass(frozen=True, slots=True)
class VectorRule:
    """A thin axis-aligned line or rectangle (table border or inline fraction bar)."""

    bbox: tuple[float, float, float, float]  # logical (x0, y0, x1, y1)
    op_indices: tuple[int, ...]
    color_rgb: tuple[float, float, float]
    is_horizontal: bool


@dataclass(slots=True)
class PageStreamAnalysis:
    """Complete instruction-indexed analysis of one PDF page."""

    page_no: int
    page_width: float
    page_height: float
    cropbox: tuple[float, float, float, float]
    rotate: int
    stream_to_logical: Matrix
    logical_to_stream: Matrix
    text_ops: list[StreamTextOp]
    vector_rules: list[VectorRule]
    instruction_count: int
    missing_tounicode_ops: int


def _cmyk_to_rgb(c: float, m: float, y: float, k: float) -> tuple[float, float, float]:
    return (
        max(0.0, min(1.0, (1.0 - c) * (1.0 - k))),
        max(0.0, min(1.0, (1.0 - m) * (1.0 - k))),
        max(0.0, min(1.0, (1.0 - y) * (1.0 - k))),
    )


def _measure_and_decode_show(
    operator: str,
    operands: list[Any],
    font: _ParsedFont | None,
    font_size: float,
    tc: float,
    tw: float,
    tz: float,
) -> tuple[str, float]:
    """Decode text and compute exact text-space horizontal advance."""
    tz_scale = tz / 100.0
    parts: list[str] = []
    advance = 0.0
    pending_space = False

    def _consume_string(raw: bytes) -> None:
        nonlocal advance, pending_space
        if font is None:
            decoded = raw.decode("latin-1", errors="ignore")
            if (
                pending_space
                and decoded
                and (not parts or not parts[-1].endswith((" ", "\t", "\n")))
                and not decoded.startswith((" ", "\t", "\n"))
            ):
                parts.append(" ")
            if decoded:
                pending_space = False
            parts.append(decoded)
            advance += (len(decoded) * 0.5 * font_size + len(decoded) * tc) * tz_scale
            return
        decoded, codes = font.decode_bytes(raw)
        if (
            pending_space
            and decoded
            and (not parts or not parts[-1].endswith((" ", "\t", "\n")))
            and not decoded.startswith((" ", "\t", "\n"))
        ):
            parts.append(" ")
        if decoded:
            pending_space = False
        parts.append(decoded)
        for code in codes:
            w = font.width_em(code) * font_size
            sp = tw if code == 32 else 0.0
            advance += (w + tc + sp) * tz_scale

    if operator == "TJ" and operands and isinstance(operands[0], pikepdf.Array):
        for elem in operands[0]:
            if isinstance(elem, pikepdf.String):
                _consume_string(bytes(elem))
            else:
                try:
                    adj = float(elem)
                    # Negative TJ adjustments >= 0.16 em represent inter-word spaces
                    # in TeX/LaTeX & PDF producers without literal 0x20 bytes,
                    # including leading spaces at font switches like [-333 (word)].
                    if adj <= -160.0:
                        pending_space = True
                    advance -= (adj / 1000.0) * font_size * tz_scale
                except (TypeError, ValueError):
                    pass
        if pending_space and parts and not parts[-1].endswith((" ", "\t", "\n")):
            parts.append(" ")
    elif operator in {"Tj", "'"} and operands and isinstance(operands[0], pikepdf.String):
        _consume_string(bytes(operands[0]))
    elif operator == '"' and len(operands) >= 3 and isinstance(operands[2], pikepdf.String):
        _consume_string(bytes(operands[2]))

    return "".join(parts), advance


# Negative TJ kerning adjustment (in thousandths of an em) large enough to
# indicate an inter-column / tab jump inside a single PDF ``TJ`` operator
# (e.g. pdflatex/dvipdfm booktabs tables emitting an entire row in one ``TJ``).
# Normal TeX word spaces are ~-250..-580; section number gaps are ~-1125;
# table column jumps are typically <= -1300.
_TJ_COLUMN_JUMP_THOUSANDTHS = -1250.0


def _decode_tj_slice(
    items: list[Any],
    font: _ParsedFont | None,
    font_size: float,
    tc: float,
    tw: float,
    tz: float,
) -> tuple[str, float]:
    """Decode a contiguous slice of ``TJ`` array items and compute its advance."""
    return _measure_and_decode_show("TJ", [pikepdf.Array(items)], font, font_size, tc, tw, tz)


def _split_tj_segments(
    arr: pikepdf.Array,
    font: _ParsedFont | None,
    font_size: float,
    tc: float,
    tw: float,
    tz: float,
) -> tuple[list[tuple[str, float, float, tuple[int, int] | None, pikepdf.Array]], float]:
    """Split a ``TJ`` array across large horizontal jumps while preserving exact advance.

    Returns ``(segments, total_advance)`` where each segment is
    ``(decoded_text, offset_adv, seg_adv, tj_item_range_or_None, sub_array)``.
    Leading horizontal jumps (``<= -160``) before the first string are skipped
    into ``offset_adv`` so the segment's bbox starts at actual ink rather than
    preceding column whitespace.
    """
    items = list(arr)
    if not items:
        return [], 0.0
    tz_scale = tz / 100.0

    # Identify string indices and split points on large column jumps.
    raw_spans: list[tuple[int, int]] = []
    cur_start: int | None = None
    last_str_end = -1
    for i, elem in enumerate(items):
        if isinstance(elem, pikepdf.String):
            if cur_start is None:
                cur_start = i
            last_str_end = i + 1
        else:
            try:
                adj = float(elem)
            except (TypeError, ValueError):
                continue
            if cur_start is not None and adj <= _TJ_COLUMN_JUMP_THOUSANDTHS:
                raw_spans.append((cur_start, last_str_end))
                cur_start = None
                last_str_end = -1
    if cur_start is not None and last_str_end > cur_start:
        raw_spans.append((cur_start, last_str_end))

    if not raw_spans:
        _, total_adv = _measure_and_decode_show("TJ", [arr], font, font_size, tc, tw, tz)
        return [], total_adv

    # Check whether any leading jump or multi-segment split occurred.
    has_leading_jump = False
    if raw_spans[0][0] > 0:
        for elem in items[: raw_spans[0][0]]:
            try:
                if float(elem) <= -160.0:
                    has_leading_jump = True
                    break
            except (TypeError, ValueError):
                pass
    is_split = len(raw_spans) > 1 or has_leading_jump

    if not is_split:
        decoded, total_adv = _measure_and_decode_show("TJ", [arr], font, font_size, tc, tw, tz)
        return [(decoded, 0.0, total_adv, None, arr)], total_adv

    # Compute cumulative advance before each item for exact segment offsets.
    item_advs: list[float] = []
    for elem in items:
        if isinstance(elem, pikepdf.String):
            _, adv = _measure_and_decode_show("Tj", [elem], font, font_size, tc, tw, tz)
            item_advs.append(adv)
        else:
            try:
                adj = float(elem)
                item_advs.append(-(adj / 1000.0) * font_size * tz_scale)
            except (TypeError, ValueError):
                item_advs.append(0.0)

    prefix = [0.0]
    for a in item_advs:
        prefix.append(prefix[-1] + a)
    total_adv = prefix[-1]

    segments: list[tuple[str, float, float, tuple[int, int] | None, pikepdf.Array]] = []
    for start, end in raw_spans:
        sub_items = items[start:end]
        decoded, seg_adv = _decode_tj_slice(sub_items, font, font_size, tc, tw, tz)
        offset_adv = prefix[start]
        segments.append((decoded, offset_adv, seg_adv, (start, end), pikepdf.Array(sub_items)))
    return segments, total_adv


def analyze_page_stream(
    page: pikepdf.Page,
    *,
    page_no: int = 1,
    pdfium_fallback_path: Path | None = None,
) -> PageStreamAnalysis:
    """Parse ``page`` content stream into logical-coordinate text ops and vector rules."""
    vis_w, vis_h, s2l, l2s, cropbox, rotate = _compute_page_transforms(page)
    fonts = _parse_page_fonts(page)
    instructions = pikepdf.parse_content_stream(page)

    ctm: Matrix = IDENTITY_MATRIX
    color_rgb: tuple[float, float, float] = (0.0, 0.0, 0.0)
    gstack: list[tuple[Matrix, tuple[float, float, float]]] = []

    in_text = False
    tm: Matrix = IDENTITY_MATRIX
    tlm: Matrix = IDENTITY_MATRIX
    cur_font_res = ""
    cur_font_size = 10.0
    tc = 0.0
    tw = 0.0
    tz = 100.0
    tl = 0.0
    ts = 0.0

    text_ops: list[StreamTextOp] = []
    vector_rules: list[VectorRule] = []
    path_pts: list[tuple[float, float]] = []
    path_ops: list[int] = []
    missing_tu = 0

    for op_idx, inst in enumerate(instructions):
        op = str(inst.operator)
        operands = list(inst.operands)

        if op == "q":
            gstack.append((ctm, color_rgb))
        elif op == "Q":
            if gstack:
                ctm, color_rgb = gstack.pop()
        elif op == "cm" and len(operands) >= 6:
            m = tuple(float(v) for v in operands[:6])
            ctm = mat_mul((m[0], m[1], m[2], m[3], m[4], m[5]), ctm)
        elif op == "rg" and len(operands) >= 3:
            color_rgb = (float(operands[0]), float(operands[1]), float(operands[2]))
        elif op == "g" and len(operands) >= 1:
            gv = float(operands[0])
            color_rgb = (gv, gv, gv)
        elif op == "k" and len(operands) >= 4:
            color_rgb = _cmyk_to_rgb(
                float(operands[0]),
                float(operands[1]),
                float(operands[2]),
                float(operands[3]),
            )
        elif op in {"sc", "scn"}:
            if len(operands) == 1:
                with contextlib.suppress(TypeError, ValueError):
                    gv = float(operands[0])
                    color_rgb = (gv, gv, gv)
            elif len(operands) == 3:
                with contextlib.suppress(TypeError, ValueError):
                    color_rgb = (float(operands[0]), float(operands[1]), float(operands[2]))
            elif len(operands) >= 4:
                with contextlib.suppress(TypeError, ValueError):
                    color_rgb = _cmyk_to_rgb(
                        float(operands[0]),
                        float(operands[1]),
                        float(operands[2]),
                        float(operands[3]),
                    )
        elif op == "BT":
            in_text = True
            tm = IDENTITY_MATRIX
            tlm = IDENTITY_MATRIX
        elif op == "ET":
            in_text = False
        elif op == "Tf" and len(operands) >= 2:
            res = str(operands[0])
            cur_font_res = res if res.startswith("/") else "/" + res
            cur_font_size = float(operands[1])
        elif op == "Tc" and len(operands) >= 1:
            tc = float(operands[0])
        elif op == "Tw" and len(operands) >= 1:
            tw = float(operands[0])
        elif op == "Tz" and len(operands) >= 1:
            tz = float(operands[0])
        elif op == "TL" and len(operands) >= 1:
            tl = float(operands[0])
        elif op == "Ts" and len(operands) >= 1:
            ts = float(operands[0])
        elif op == "Tm" and len(operands) >= 6:
            m = tuple(float(v) for v in operands[:6])
            tm = (m[0], m[1], m[2], m[3], m[4], m[5])
            tlm = tm
        elif op == "Td" and len(operands) >= 2:
            tx, ty = float(operands[0]), float(operands[1])
            tlm = mat_mul((1.0, 0.0, 0.0, 1.0, tx, ty), tlm)
            tm = tlm
        elif op == "TD" and len(operands) >= 2:
            tx, ty = float(operands[0]), float(operands[1])
            tl = -ty
            tlm = mat_mul((1.0, 0.0, 0.0, 1.0, tx, ty), tlm)
            tm = tlm
        elif op == "T*":
            tlm = mat_mul((1.0, 0.0, 0.0, 1.0, 0.0, -tl), tlm)
            tm = tlm
        elif op in _SHOW_OPS and in_text:
            if op == "'":
                tlm = mat_mul((1.0, 0.0, 0.0, 1.0, 0.0, -tl), tlm)
                tm = tlm
            elif op == '"' and len(operands) >= 3:
                tw = float(operands[0])
                tc = float(operands[1])
                tlm = mat_mul((1.0, 0.0, 0.0, 1.0, 0.0, -tl), tlm)
                tm = tlm

            pfont = fonts.get(cur_font_res)
            base_tm = tm
            tj_segments: list[tuple[str, float, float, tuple[int, int] | None, Any]]
            if op == "TJ" and operands and isinstance(operands[0], pikepdf.Array):
                tj_segments, total_adv = _split_tj_segments(
                    operands[0], pfont, cur_font_size, tc, tw, tz
                )
            else:
                decoded_one, total_adv = _measure_and_decode_show(
                    op, operands, pfont, cur_font_size, tc, tw, tz
                )
                tj_segments = [
                    (
                        decoded_one,
                        0.0,
                        total_adv,
                        None,
                        operands[0] if operands else None,
                    )
                ]

            for decoded, offset_adv, seg_adv, item_range, seg_op0 in tj_segments:
                seg_tm = (
                    mat_mul((1.0, 0.0, 0.0, 1.0, offset_adv, 0.0), base_tm)
                    if abs(offset_adv) > 1e-6
                    else base_tm
                )
                snap = OpGraphicsSnapshot(
                    ctm=ctm,
                    text_matrix=seg_tm,
                    font_resource=cur_font_res,
                    font_size_operand=cur_font_size,
                    tc=tc,
                    tw=tw,
                    tz=tz,
                    ts=ts,
                    color_rgb=color_rgb,
                )

                # Compute start/end & ascent/descent corners in logical page space.
                trm = mat_mul(seg_tm, ctm)
                trm_logical = mat_mul(trm, s2l)
                eff_size = abs(cur_font_size) * math.hypot(trm_logical[2], trm_logical[3])
                if eff_size <= 0.05:
                    eff_size = abs(cur_font_size) * math.hypot(trm_logical[0], trm_logical[1])

                x0_t, y0_t = 0.0, ts - 0.22 * abs(cur_font_size)
                x1_t, y1_t = seg_adv, ts + 0.78 * abs(cur_font_size)
                corners = [
                    mat_apply(trm_logical, x0_t, y0_t),
                    mat_apply(trm_logical, x1_t, y0_t),
                    mat_apply(trm_logical, x0_t, y1_t),
                    mat_apply(trm_logical, x1_t, y1_t),
                ]
                bx0 = min(c[0] for c in corners)
                by0 = min(c[1] for c in corners)
                bx1 = max(c[0] for c in corners)
                by1 = max(c[1] for c in corners)
                _, base_y = mat_apply(trm_logical, 0.0, ts)

                if pfont is not None and not pfont.has_tounicode:
                    missing_tu += 1

                seg_raw_operands = (seg_op0,) if item_range is not None else tuple(operands)
                # Record op even if decoded is empty (when ToUnicode is missing) so
                # pdfium recovery or surgical erasure still tracks its op_index.
                if decoded or (pfont is not None and not pfont.has_tounicode):
                    text_ops.append(
                        StreamTextOp(
                            op_index=op_idx,
                            operator=op,
                            text=decoded,
                            bbox=(bx0, by0, bx1, by1),
                            baseline_y=base_y,
                            font_resource=cur_font_res,
                            font_name=pfont.base_name if pfont else "",
                            font_size_pt=eff_size,
                            color_rgb=color_rgb,
                            is_bold=pfont.is_bold if pfont else False,
                            is_italic=pfont.is_italic if pfont else False,
                            is_serif=pfont.is_serif if pfont else True,
                            is_math_font=pfont.is_math_font if pfont else False,
                            has_tounicode=pfont.has_tounicode if pfont else False,
                            rise_pt=ts * math.hypot(trm_logical[2], trm_logical[3]),
                            snapshot=snap,
                            raw_operands=seg_raw_operands,
                            text_advance=seg_adv,
                            tj_item_range=item_range,
                        )
                    )

            # Advance text matrix exactly per PDF spec over the full instruction.
            tm = mat_mul((1.0, 0.0, 0.0, 1.0, total_adv, 0.0), base_tm)
        # Lightweight vector rule tracking (table borders & inline fraction bars).
        elif op == "re" and len(operands) >= 4:
            rx, ry, rw, rh = (float(v) for v in operands[:4])
            c2l = mat_mul(ctm, s2l)
            pts = [
                mat_apply(c2l, rx, ry),
                mat_apply(c2l, rx + rw, ry),
                mat_apply(c2l, rx + rw, ry + rh),
                mat_apply(c2l, rx, ry + rh),
            ]
            path_pts.extend(pts)
            path_ops.append(op_idx)
        elif op in {"m", "l"} and len(operands) >= 2:
            c2l = mat_mul(ctm, s2l)
            path_pts.append(mat_apply(c2l, float(operands[0]), float(operands[1])))
            path_ops.append(op_idx)
        elif op in {"S", "s", "f", "F", "f*", "B", "B*", "b", "b*"}:
            if len(path_pts) >= 2:
                rx0 = min(p[0] for p in path_pts)
                ry0 = min(p[1] for p in path_pts)
                rx1 = max(p[0] for p in path_pts)
                ry1 = max(p[1] for p in path_pts)
                rw, rh = rx1 - rx0, ry1 - ry0
                # Record thin rules (fraction bars, underlines, table borders)
                if (rw >= 3.0 and rh <= 2.5) or (rh >= 3.0 and rw <= 2.5):
                    vector_rules.append(
                        VectorRule(
                            bbox=(rx0, ry0, rx1, ry1),
                            op_indices=tuple(path_ops + [op_idx]),
                            color_rgb=color_rgb,
                            is_horizontal=rw >= rh,
                        )
                    )
            path_pts.clear()
            path_ops.clear()
        elif op == "n":
            path_pts.clear()
            path_ops.clear()

    analysis = PageStreamAnalysis(
        page_no=page_no,
        page_width=vis_w,
        page_height=vis_h,
        cropbox=cropbox,
        rotate=rotate,
        stream_to_logical=s2l,
        logical_to_stream=l2s,
        text_ops=text_ops,
        vector_rules=vector_rules,
        instruction_count=len(instructions),
        missing_tounicode_ops=missing_tu,
    )

    if pdfium_fallback_path is not None and missing_tu > 0:
        _recover_missing_tounicode_via_pdfium(analysis, pdfium_fallback_path)

    return analysis


def _recover_missing_tounicode_via_pdfium(
    analysis: PageStreamAnalysis,
    pdf_path: Path,
) -> None:
    """Fill missing Unicode text on Type1 ops lacking /ToUnicode using pypdfium2."""
    import pypdfium2 as pdfium

    from ubt.adapters.pdf.pdfium_gate import PDFIUM_LOCK

    ops_needing = [op for op in analysis.text_ops if not op.has_tounicode]
    if not ops_needing:
        return

    chars_with_pos: list[tuple[float, float, float, float, str]] = []
    with PDFIUM_LOCK:
        doc = pdfium.PdfDocument(str(pdf_path))
        try:
            if analysis.page_no - 1 >= len(doc):
                return
            page = doc[analysis.page_no - 1]
            try:
                tp = page.get_textpage()
                try:
                    n_chars = tp.count_chars()
                    full_text = tp.get_text_range()
                    for idx in range(min(n_chars, len(full_text))):
                        ch = full_text[idx]
                        if ch in "\r\n":
                            continue
                        left, bottom, right, top = tp.get_charbox(idx)
                        cx = 0.5 * (left + right)
                        cy = 0.5 * (bottom + top)
                        chars_with_pos.append((left, right, cx, cy, ch))
                finally:
                    tp.close()
            finally:
                page.close()
        finally:
            doc.close()

    if not chars_with_pos:
        return

    # Assign each Pdfium character to at most ONE StreamTextOp so adjacent ops never
    # double-capture boundary glyphs such as ``[`` / ``]`` around colored citations
    # (``[[1]]``) or inline math punctuation.
    #
    # Ranking by the character's *center* alone lost the last glyph of a line: the
    # analyzer's advance estimate for a font without /ToUnicode can be a few points
    # short of the glyph's real box, so the final ``s`` in ``The Descendants`` sat
    # outside ``op.bbox`` and was dropped from the recovered text (measured on the
    # synthetic render fixture). Horizontal overlap is the stable signal -- a glyph
    # belongs to the op whose box it covers most, even when its center falls just
    # past the estimated advance width. Each character still goes to exactly one op,
    # so a boundary glyph cannot be recovered twice.
    needing_ids = {id(op) for op in ops_needing}
    assigned: dict[int, list[tuple[float, str]]] = {}

    for left, right, cx, cy, ch in chars_with_pos:
        if ch == "\ufffe":
            continue
        best_op: StreamTextOp | None = None
        best_rank: tuple[int, float, float] | None = None
        for op in analysis.text_ops:
            x0, y0, x1, y1 = op.bbox
            pad_y = max(1.5, op.font_size_pt * 0.35)
            if not ((y0 - pad_y) <= cy <= (y1 + pad_y)):
                continue
            pad_x = max(0.4, min(0.9, op.font_size_pt * 0.08))
            overlap = min(x1, right) - max(x0, left)
            if x0 <= cx <= x1:
                tier = 0
            elif overlap > 0.0:
                # Center just outside the estimated advance, but the glyph still
                # overlaps the op's box: prefer the larger overlap.
                tier = 1
            elif (x0 - pad_x) <= cx <= (x1 + pad_x):
                tier = 2
            else:
                continue
            dy = abs(cy - 0.5 * (y0 + y1))
            rank = (tier, -overlap, dy)
            if best_rank is None or rank < best_rank:
                best_rank = rank
                best_op = op
        if best_op is not None and id(best_op) in needing_ids:
            assigned.setdefault(id(best_op), []).append((cx, ch))

    for op in ops_needing:
        matched = assigned.get(id(op))
        if not matched:
            continue
        matched.sort(key=lambda item: item[0])
        recovered = "".join(ch for _, ch in matched)
        if recovered.strip():
            op.text = recovered
