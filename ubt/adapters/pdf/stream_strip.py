"""2D High-precision content-stream text operator stripping via pikepdf.

Eliminates source English text under overlay covers with 2D coordinate accuracy
while strictly preserving:
1. Protected regions (mathematical formulas, vector diagrams, code blocks, images).
2. Opposite columns in multi-column layouts (no 1D row-band horizontal spillage).
3. Form XObject nested text streams (recursive XObject rewriting).

Zero-AGPL: uses pikepdf (MPL-2.0). Never imports fitz or pymupdf.
"""

from __future__ import annotations

import contextlib
import logging
from bisect import bisect_right
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Any, cast

import pikepdf

logger = logging.getLogger(__name__)

Rect = tuple[float, float, float, float]
Matrix2D = tuple[float, float, float, float, float, float]

IDENTITY_MATRIX: Matrix2D = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
TEXT_SHOW_OPERATORS = frozenset({"Tj", "TJ", "'", '"'})
# Path decoration stripping: a source underline can survive the text strip
# because it is a vector rule, not a text operator. A painted path is dropped
# only when it is FULLY covered by a strip rect (PyMuPDF's
# PDF_REDACT_LINE_ART_REMOVE_IF_COVERED semantics -- intersect-but-not-covered
# keeps table rules that graze the zone padding) AND it is thin/horizontal
# enough to be a text decoration (underline/strikeout/rule) rather than
# diagram geometry.
PATH_PAINT_OPERATORS = frozenset({"S", "s", "f", "F", "f*", "B", "B*", "b", "b*"})
PATH_CONSTRUCTION_OPERATORS = frozenset({"m", "l", "c", "v", "y", "re", "h"})
DECORATION_MAX_THICKNESS_PT = 2.5
DECORATION_MIN_WIDTH_PT = 1.0
PATH_CONTAIN_EPS_PT = 0.5
# Estimated advance width per glyph, in em, used when deriving a show op's
# bbox from character count. It deliberately ignores the font's real /Widths
# table: the estimate is only used to decide whether a text run overlaps a
# strip rect, and the rect already carries generous padding. A wide glyph
# (e.g. a full-width CJK form) can therefore be under-measured; parsing
# /Widths per font resource would tighten this but is not done for this
# fallback estimate.
DEFAULT_GLYPH_WIDTH_EM = 0.5
# Type0 codes are two bytes wide for every predefined CMap a CID font actually
# uses (Identity-H/V, the UniGB/UniJIS/UniKS CJK CMaps, GBK-EUC-H …). Only a
# handful of one-byte simple-font encodings ever appear on a (malformed) Type0
# wrapper; treating anything else as one byte measured CJK books at half their
# real advance and left the tail of long lines unstripped.
_ONE_BYTE_TYPE0_ENCODINGS = frozenset(
    {
        "/WinAnsiEncoding",
        "/MacRomanEncoding",
        "/StandardEncoding",
        "/MacExpertEncoding",
        "/PDFDocEncoding",
    }
)


def mul_matrix(m1: Matrix2D, m2: Matrix2D) -> Matrix2D:
    """Multiply two 3x3 affine transform matrices: M1 x M2.

    [ a1 b1 0 ]   [ a2 b2 0 ]   [ a1*a2 + b1*c2       a1*b2 + b1*d2       0 ]
    [ c1 d1 0 ] x [ c2 d2 0 ] = [ c1*a2 + d1*c2       c1*b2 + d1*d2       0 ]
    [ e1 f1 1 ]   [ e2 f2 1 ]   [ e1*a2 + f1*c2 + e2  e1*b2 + f1*d2 + f2  1 ]
    """
    a1, b1, c1, d1, e1, f1 = m1
    a2, b2, c2, d2, e2, f2 = m2
    return (
        a1 * a2 + b1 * c2,
        a1 * b2 + b1 * d2,
        c1 * a2 + d1 * c2,
        c1 * b2 + d1 * d2,
        e1 * a2 + f1 * c2 + e2,
        e1 * b2 + f1 * d2 + f2,
    )


def transform_point(x: float, y: float, m: Matrix2D) -> tuple[float, float]:
    """Transform point (x, y) by affine matrix m."""
    a, b, c, d, e, f = m
    return (x * a + y * c + e, x * b + y * d + f)


def matrix_from_object(obj: Any) -> Matrix2D | None:
    """Convert pikepdf.Array or python sequence into Matrix2D."""
    if obj is None:
        return None
    try:
        vals = [float(v) for v in obj]
        if len(vals) >= 6:
            return (vals[0], vals[1], vals[2], vals[3], vals[4], vals[5])
    except (TypeError, ValueError):
        pass
    return None


@dataclass(frozen=True)
class Rect2DIndex:
    """Fast spatial index for 2D bounding boxes sorted by y0."""

    rects: tuple[Rect, ...]
    y0_sorted: tuple[float, ...]
    bounds: Rect | None = None

    @classmethod
    def build(cls, rects: Iterable[Rect]) -> Rect2DIndex:
        normalized: list[Rect] = []
        for r in rects:
            try:
                x0, y0, x1, y1 = float(r[0]), float(r[1]), float(r[2]), float(r[3])
                if x0 > x1:
                    x0, x1 = x1, x0
                if y0 > y1:
                    y0, y1 = y1, y0
                if x1 > x0 and y1 > y0:
                    normalized.append((x0, y0, x1, y1))
            except (TypeError, IndexError, ValueError):
                continue

        normalized.sort(key=lambda r: r[1])
        if not normalized:
            return cls(rects=(), y0_sorted=(), bounds=None)

        min_x = min(r[0] for r in normalized)
        min_y = min(r[1] for r in normalized)
        max_x = max(r[2] for r in normalized)
        max_y = max(r[3] for r in normalized)

        return cls(
            rects=tuple(normalized),
            y0_sorted=tuple(r[1] for r in normalized),
            bounds=(min_x, min_y, max_x, max_y),
        )

    def contains_point(self, x: float, y: float) -> bool:
        if self.bounds is None:
            return False
        bx0, by0, bx1, by1 = self.bounds
        if not (bx0 <= x <= bx1 and by0 <= y <= by1):
            return False
        limit = bisect_right(self.y0_sorted, y)
        for i in range(limit):
            x0, y0, x1, y1 = self.rects[i]
            if y1 < y:
                continue
            if x0 <= x <= x1 and y0 <= y <= y1:
                return True
        return False

    def intersects(self, r: Rect) -> bool:
        if self.bounds is None:
            return False
        bx0, by0, bx1, by1 = self.bounds
        rx0, ry0, rx1, ry1 = r
        if rx1 < bx0 or rx0 > bx1 or ry1 < by0 or ry0 > by1:
            return False
        limit = bisect_right(self.y0_sorted, ry1)
        for i in range(limit):
            x0, y0, x1, y1 = self.rects[i]
            if y1 < ry0:
                continue
            if not (rx1 < x0 or rx0 > x1 or ry1 < y0 or ry0 > y1):
                return True
        return False

    def fully_contains(self, r: Rect, eps: float = PATH_CONTAIN_EPS_PT) -> bool:
        """True when some indexed rect covers ``r`` entirely (±eps).

        The path-decoration analogue of PyMuPDF's REMOVE_IF_COVERED: a rule
        that merely crosses the rect border (a table line grazing a caption
        zone) is never dropped by this predicate.
        """
        if self.bounds is None:
            return False
        rx0, ry0, rx1, ry1 = r
        for x0, y0, x1, y1 in self.rects:
            if rx0 >= x0 - eps and rx1 <= x1 + eps and ry0 >= y0 - eps and ry1 <= y1 + eps:
                return True
        return False

    def matches_text_for_removal(self, x: float, y: float, text_rect: Rect) -> bool:
        """Check if text origin or text bounding box matches a strip rect."""
        if self.bounds is None:
            return False
        rx0, ry0, rx1, ry1 = text_rect
        search_max_y = max(y, ry1)
        limit = bisect_right(self.y0_sorted, search_max_y)

        t_w = max(rx1 - rx0, 1e-3)
        t_h = max(ry1 - ry0, 1e-3)
        t_area = t_w * t_h

        for i in range(limit):
            x0, y0, x1, y1 = self.rects[i]
            if y1 < min(y, ry0):
                continue
            # Point check
            if x0 <= x <= x1 and y0 <= y <= y1:
                return True
            # Substantial 2D overlap check
            ix0 = max(rx0, x0)
            iy0 = max(ry0, y0)
            ix1 = min(rx1, x1)
            iy1 = min(ry1, y1)
            if ix1 > ix0 and iy1 > iy0:
                overlap_area = (ix1 - ix0) * (iy1 - iy0)
                overlap_w = (ix1 - ix0) / t_w
                overlap_h = (iy1 - iy0) / t_h
                if (overlap_area / t_area >= 0.25) or (overlap_w >= 0.35 and overlap_h >= 0.3):
                    return True
        return False

    def protects_text_rect(self, x: float, y: float, text_rect: Rect) -> bool:
        """Check if text intersects any protected region."""
        if self.bounds is None:
            return False
        rx0, ry0, rx1, ry1 = text_rect
        search_max_y = max(y, ry1)
        limit = bisect_right(self.y0_sorted, search_max_y)

        t_w = max(rx1 - rx0, 1e-3)
        t_h = max(ry1 - ry0, 1e-3)
        t_area = t_w * t_h

        for i in range(limit):
            x0, y0, x1, y1 = self.rects[i]
            if y1 < min(y, ry0):
                continue
            if x0 <= x <= x1 and y0 <= y <= y1:
                return True
            # If text baseline y is below the protected rect's bottom (y0),
            # this text is located below a formula/image; ascenders grazing the bottom
            # padding margin must not treat the narrative text as protected.
            if y < y0:
                continue
            ix0 = max(rx0, x0)
            iy0 = max(ry0, y0)
            ix1 = min(rx1, x1)
            iy1 = min(ry1, y1)
            if ix1 > ix0 and iy1 > iy0:
                overlap_area = (ix1 - ix0) * (iy1 - iy0)
                if overlap_area > 1.0 and (
                    overlap_area / t_area > 0.15 or (iy1 - iy0) / t_h > 0.25
                ):
                    return True
        return False


@dataclass
class TextState:
    font_size: float = 12.0
    char_spacing: float = 0.0
    word_spacing: float = 0.0
    horizontal_scaling: float = 1.0  # 100% = 1.0
    leading: float = 0.0
    rise: float = 0.0
    render_mode: int = 0


@dataclass
class StreamStripStats:
    """Accounting for 2D stream stripping."""

    page: int
    dropped_ops: int = 0
    kept_ops: int = 0
    dropped_paths: int = 0
    form_ops: int = 0
    forms_changed: int = 0
    #: Forms skipped because they are referenced by more than one page; rewriting
    #: one in place would erase text on every page that draws it.
    shared_forms_skipped: int = 0
    aborted: str | None = None


def _extract_text_metrics(operands: Sequence[Any]) -> tuple[int, int, float]:
    """Returns (char_count, space_count, kerning_adjustment)."""
    if not operands:
        return 0, 0, 0.0
    val = operands[-1]
    if isinstance(val, (str, bytes, pikepdf.String)):
        s = str(val)
        return len(s), s.count(" "), 0.0
    if isinstance(val, (list, pikepdf.Array)):
        chars = 0
        spaces = 0
        adj = 0.0
        for item in val:
            if isinstance(item, (str, bytes, pikepdf.String)):
                s = str(item)
                chars += len(s)
                spaces += s.count(" ")
            elif isinstance(item, (int, float)):
                adj += float(item)
        return chars, spaces, adj
    return 0, 0, 0.0


def _show_glyph_bytes(operands: Sequence[Any]) -> bytes:
    """Concatenate the literal string bytes of a Tj/TJ show operand.

    Numbers in a TJ array are kerning adjustments (handled separately), not
    glyphs; only the string parts carry the advance. Bytes are the raw glyph
    codes the font's /Widths are indexed by.
    """
    if not operands:
        return b""
    val = operands[-1]
    parts: list[bytes] = []
    if isinstance(val, (str, bytes, pikepdf.String)):
        parts.append(_as_bytes(val))
    elif isinstance(val, (list, pikepdf.Array)):
        for item in val:
            if isinstance(item, (str, bytes, pikepdf.String)):
                parts.append(_as_bytes(item))
    return b"".join(parts)


def _as_bytes(val: Any) -> bytes:
    if isinstance(val, bytes):
        return val
    raw = getattr(val, "read_bytes", None)
    if callable(raw):
        try:
            return bytes(raw())
        except Exception:  # fall through to str-encode
            pass
    return str(val).encode("latin-1", "replace")


@dataclass(frozen=True)
class FontAdvance:
    """Per-glyph advance lookup for one font, with its code byte width.

    ``code_bytes`` is 1 for simple fonts and 2 for Type0 CIDFonts (every
    predefined CMap they use is two-byte); ``widths`` maps a glyph code to its
    advance in 1000-em units; ``default`` covers codes absent from the table
    (MissingWidth / DW).
    """

    code_bytes: int
    widths: dict[int, float]
    default: float


def _parse_cid_widths(w_array: Any, widths: dict[int, float]) -> None:
    """Fold a CIDFont /W array into ``widths`` (cid → 1000-em advance).

    /W entries alternate between ``cid`` followed by ``[w w ...]`` (a run of
    consecutive cids) and ``[first last w]`` (a range sharing one width)."""
    items = list(w_array)
    i = 0
    while i < len(items):
        entry = items[i]
        if isinstance(entry, (list, pikepdf.Array)):
            first, last, w = (int(entry[0]), int(entry[1]), float(entry[2]))
            for cid in range(first, last + 1):
                widths[cid] = w
            i += 1
        else:
            start = int(entry)
            i += 1
            if i < len(items) and isinstance(items[i], (list, pikepdf.Array)):
                for off, w in enumerate(items[i]):
                    widths[start + off] = float(w)
                i += 1


def _font_advance(font_obj: Any) -> FontAdvance | None:
    """Build a FontAdvance for a simple or Type0/CID font; None if unresolvable.

    Covers the /Widths simple-font case and the /DescendantFonts /W CID case
    that carry colored citation runs in born-digital academic PDFs; the strip's
    fixed 0.5em/glyph estimate drifts tens of points across a long line and can
    let a trailing citation escape the erase rect. Any font we cannot parse
    returns None so the caller keeps the conservative estimate.
    """
    try:
        subtype = font_obj.get(pikepdf.Name("/Subtype"))
    except Exception:  # unresolvable font object
        return None
    # Compare by value: /Subtype is a Name in real PDFs but a String in some
    # fixtures/tests; str() normalizes both.
    if str(subtype) == "/Type0":
        return _type0_advance(font_obj)
    return _simple_font_advance(font_obj)


def _type0_advance(font_obj: Any) -> FontAdvance | None:
    try:
        descendants = font_obj.get(pikepdf.Name("/DescendantFonts"))
        if not descendants:
            return None
        cid_font = descendants[0]
        encoding = font_obj.get(pikepdf.Name("/Encoding"))
        code_bytes = 1 if str(encoding) in _ONE_BYTE_TYPE0_ENCODINGS else 2
        default = float(cid_font.get(pikepdf.Name("/DW"), 1000) or 1000)
        widths: dict[int, float] = {}
        w_array = cid_font.get(pikepdf.Name("/W"))
        if w_array:
            _parse_cid_widths(w_array, widths)
        if not widths:
            return None
        return FontAdvance(code_bytes=code_bytes, widths=widths, default=default)
    except Exception:  # malformed CID structure
        return None


def _simple_font_advance(font_obj: Any) -> FontAdvance | None:
    try:
        widths_arr = font_obj.get(pikepdf.Name("/Widths"))
        first_char = int(font_obj.get(pikepdf.Name("/FirstChar"), 0))
    except Exception:  # malformed font dict
        return None
    if not widths_arr:
        return None
    try:
        default = 0.0
        descriptor = font_obj.get(pikepdf.Name("/FontDescriptor"))
        if descriptor is not None:
            default = float(descriptor.get(pikepdf.Name("/MissingWidth"), 0) or 0)
        widths: dict[int, float] = {}
        for i, w in enumerate(widths_arr):
            widths[first_char + i] = float(w)
        return FontAdvance(code_bytes=1, widths=widths, default=default)
    except (TypeError, ValueError):
        return None


def _advance_from_widths(data: bytes, fa: FontAdvance) -> float:
    """Sum glyph advances (in em units) for the raw show bytes."""
    total = 0.0
    cb = fa.code_bytes
    if cb == 1:
        for code in data:
            total += fa.widths.get(code, fa.default)
    else:
        for i in range(0, len(data) - cb + 1, cb):
            code = (data[i] << 8) | data[i + 1]
            total += fa.widths.get(code, fa.default)
    return total / 1000.0


def _fonts_resources(resources: Any) -> Any:
    """The /Font resource dictionary of the current scope, or None."""
    if resources is None:
        return None
    try:
        return resources.get(pikepdf.Name("/Font"))
    except Exception:  # resources shape varies
        return None


def _resolve_font_widths(fonts_res: Any, font_name: Any) -> FontAdvance | None:
    """Look up the Tf-selected font and build its advance map."""
    if fonts_res is None or font_name is None:
        return None
    try:
        font_obj = fonts_res.get(font_name)
        if font_obj is None:
            return None
        # A font may be an indirect object; pikepdf resolves on attribute access.
        return _font_advance(font_obj)
    except Exception:  # any resolution miss → conservative estimate
        return None


def _calculate_text_advance(
    text_state: TextState,
    char_count: int,
    space_count: int,
    adj: float,
    glyph_width_em: float | None = None,
) -> float:
    # glyph_width_em is the real summed advance in em units from the font's
    # /Widths; when the font could not be resolved it is None and we fall back
    # to the fixed per-glyph estimate.
    if glyph_width_em is not None:
        glyph_w = glyph_width_em * text_state.font_size
    else:
        glyph_w = char_count * text_state.font_size * DEFAULT_GLYPH_WIDTH_EM
    spacing = char_count * text_state.char_spacing + space_count * text_state.word_spacing
    kerning = -adj * text_state.font_size / 1000.0
    tx = (glyph_w + spacing + kerning) * text_state.horizontal_scaling
    return max(0.0, tx)


def _compute_user_text_geometry(
    ctm: Matrix2D,
    text_matrix: Matrix2D,
    text_state: TextState,
    advance_w: float,
) -> tuple[tuple[float, float], Rect]:
    """Compute baseline origin and 2D bounding box in user coordinates."""
    eff_matrix = mul_matrix(text_matrix, ctm)
    user_x, user_y = transform_point(0.0, text_state.rise, eff_matrix)

    h = max(text_state.font_size, 0.5) if text_state.font_size > 0 else 1.0
    w = max(advance_w, text_state.font_size * DEFAULT_GLYPH_WIDTH_EM, 1.0)

    # 4 corners in text space:
    # baseline is y=0, descent is ~ -0.25*h, ascent is ~ 0.85*h
    p1 = transform_point(0.0, -0.25 * h + text_state.rise, eff_matrix)
    p2 = transform_point(w, -0.25 * h + text_state.rise, eff_matrix)
    p3 = transform_point(w, 0.85 * h + text_state.rise, eff_matrix)
    p4 = transform_point(0.0, 0.85 * h + text_state.rise, eff_matrix)

    xs = [p1[0], p2[0], p3[0], p4[0]]
    ys = [p1[1], p2[1], p3[1], p4[1]]
    rect = (min(xs), min(ys), max(xs), max(ys))
    return (user_x, user_y), rect


def strip_stream_instructions(
    instructions: list[tuple[Sequence[Any], Any]],
    strip_index: Rect2DIndex,
    protected_index: Rect2DIndex,
    stats: StreamStripStats,
    *,
    initial_ctm: Matrix2D = IDENTITY_MATRIX,
    resources: Any = None,
    recurse_forms: bool = True,
    visited_forms: set[str] | None = None,
    shared_forms: set[tuple[int, int]] | None = None,
) -> tuple[list[tuple[Sequence[Any], Any]], int]:
    """Filter content stream instructions, returning (new_instructions, dropped_count).

    ``shared_forms`` holds the ``objgen`` of every Form XObject referenced by
    more than one page (see :func:`shared_form_objgens`). Such a Form is never
    rewritten here: it is one indirect object drawn by several pages, so a
    strip rect belonging to one page would otherwise erase the source text from
    all of them.
    """
    if visited_forms is None:
        visited_forms = set()

    ctm = initial_ctm
    # PDF 32000-1 §9.3: the graphics state includes the text state and the
    # resolved font, so q/Q must save/restore all three. A page that wraps a
    # caption in `q … /F2 Tf … Tj … Q` and then shows body text in the outer
    # font would otherwise keep F2's advance map and mis-measure the body run.
    gstate_stack: list[tuple[Matrix2D, TextState, FontAdvance | None, float]] = []

    text_state = TextState()
    text_matrix = IDENTITY_MATRIX
    line_matrix = IDENTITY_MATRIX

    output: list[tuple[Sequence[Any], Any]] = []
    dropped_count = 0
    # Advance map (code → 1000-em) for the font set by the last Tf, so a long
    # line's trailing run is positioned by real metrics instead of the 0.5em/glyph
    # estimate that drifts the colored citation past the erase rect.
    font_adv: FontAdvance | None = None
    fonts_res = _fonts_resources(resources)

    # Pending path (construction operators not yet painted) for decoration
    # detection: buffer the construction ops, decide at the paint operator.
    path_buf: list[tuple[Sequence[Any], Any]] = []
    path_pts: list[tuple[float, float]] = []
    path_clipped = False
    line_width = 1.0
    cur_pt: tuple[float, float] = (0.0, 0.0)
    subpath_start: tuple[float, float] = (0.0, 0.0)

    def _reset_path() -> None:
        nonlocal path_pts, path_clipped, cur_pt, subpath_start
        path_buf.clear()
        path_pts = []
        path_clipped = False
        cur_pt = (0.0, 0.0)
        subpath_start = (0.0, 0.0)

    def _keep_path() -> None:
        output.extend(path_buf)
        _reset_path()

    for operands, operator in instructions:
        op = str(operator)

        # ---- path decoration tracking -------------------------------------
        if op in PATH_CONSTRUCTION_OPERATORS:
            if op == "m":
                if path_buf:
                    _keep_path()  # previous path unpainted (malformed) — keep it
                try:
                    x, y = float(operands[-2]), float(operands[-1])
                except (ValueError, TypeError, IndexError):
                    output.append((operands, operator))
                    continue
                path_buf.append((operands, operator))
                path_pts = [(x, y)]
                cur_pt = (x, y)
                subpath_start = (x, y)
                continue
            if not path_buf and op != "re":
                # Construction without a preceding m is malformed for
                # stroke ops, but "re" is a self-contained subpath and by far
                # the common spelling of LaTeX underline/rules (`x y w h re f`).
                output.append((operands, operator))
                continue
            path_buf.append((operands, operator))
            try:
                if op == "l":
                    x, y = float(operands[-2]), float(operands[-1])
                    path_pts.append((x, y))
                    cur_pt = (x, y)
                elif op == "c":
                    nums = [float(v) for v in operands[-6:]]
                    path_pts.extend([(nums[0], nums[1]), (nums[2], nums[3]), (nums[4], nums[5])])
                    cur_pt = (nums[4], nums[5])
                elif op == "v":
                    nums = [float(v) for v in operands[-4:]]
                    path_pts.extend(
                        [(cur_pt[0], cur_pt[1]), (nums[0], nums[1]), (nums[2], nums[3])]
                    )
                    cur_pt = (nums[2], nums[3])
                elif op == "y":
                    nums = [float(v) for v in operands[-4:]]
                    path_pts.extend([(nums[0], nums[1]), (nums[2], nums[3]), (nums[2], nums[3])])
                    cur_pt = (nums[2], nums[3])
                elif op == "re":
                    x, y, w_, h_ = (float(v) for v in operands[-4:])
                    path_pts.extend([(x, y), (x + w_, y), (x, y + h_), (x + w_, y + h_)])
                    cur_pt = (x, y)
                    subpath_start = (x, y)
                elif op == "h":
                    path_pts.append(subpath_start)
                    cur_pt = subpath_start
            except (ValueError, TypeError, IndexError):
                path_clipped = True  # unparseable geometry: force-keep
            continue
        if op in ("W", "W*"):
            if path_buf:
                path_buf.append((operands, operator))
                path_clipped = True
            else:
                output.append((operands, operator))
            continue
        if op == "n":
            # Ends the path without painting (clip or discard) — never drop.
            _keep_path()
            output.append((operands, operator))
            continue
        if op in PATH_PAINT_OPERATORS:
            dropped_path = False
            if path_buf and not path_clipped and len(path_pts) >= 2:
                dev = [transform_point(px, py, ctm) for px, py in path_pts]
                bbox: Rect = (
                    min(p[0] for p in dev),
                    min(p[1] for p in dev),
                    max(p[0] for p in dev),
                    max(p[1] for p in dev),
                )
                stroked = op in ("S", "s", "B", "B*", "b", "b*")
                thickness = (bbox[3] - bbox[1]) + (line_width if stroked else 0.0)
                span = bbox[2] - bbox[0]
                if (
                    thickness <= DECORATION_MAX_THICKNESS_PT
                    and span >= DECORATION_MIN_WIDTH_PT
                    and strip_index.fully_contains(bbox)
                    and not protected_index.intersects(bbox)
                ):
                    dropped_count += len(path_buf) + 1
                    stats.dropped_ops += len(path_buf) + 1
                    stats.dropped_paths += 1
                    _reset_path()
                    dropped_path = True
            if not dropped_path:
                _keep_path()
                output.append((operands, operator))
            continue
        if path_buf:
            # Any other operator interrupts the path — conservative keep.
            _keep_path()
        if op == "w" and operands:
            with contextlib.suppress(ValueError, TypeError):
                line_width = max(float(operands[0]), 0.0)
            output.append((operands, operator))
            continue

        # Graphic state stack: text_state is mutated in place, so push a copy.
        # Line width is graphics state too (PDF 32000-1 §8.4.1): without it a
        # ``q 8 w ... S Q`` border left line_width at 8, so a later thin
        # underline measured thickness = bbox_h + 8 and survived the
        # decoration strip under the translated overlay.
        if op == "q":
            gstate_stack.append((ctm, replace(text_state), font_adv, line_width))
            output.append((operands, operator))
            continue
        if op == "Q":
            if gstate_stack:
                ctm, text_state, font_adv, line_width = gstate_stack.pop()
            output.append((operands, operator))
            continue
        if op == "cm":
            m = matrix_from_object(operands)
            if m is not None:
                ctm = mul_matrix(m, ctm)
            output.append((operands, operator))
            continue

        # Text state operators
        if op == "BT":
            text_matrix = IDENTITY_MATRIX
            line_matrix = IDENTITY_MATRIX
            output.append((operands, operator))
            continue
        if op == "ET":
            output.append((operands, operator))
            continue
        if op == "Tf" and len(operands) >= 2:
            with contextlib.suppress(ValueError, TypeError):
                text_state.font_size = max(float(operands[1]), 0.1)
            font_adv = _resolve_font_widths(fonts_res, operands[0])
            output.append((operands, operator))
            continue
        if op == "Tc" and operands:
            with contextlib.suppress(ValueError, TypeError):
                text_state.char_spacing = float(operands[0])
            output.append((operands, operator))
            continue
        if op == "Tw" and operands:
            with contextlib.suppress(ValueError, TypeError):
                text_state.word_spacing = float(operands[0])
            output.append((operands, operator))
            continue
        if op == "Tz" and operands:
            with contextlib.suppress(ValueError, TypeError):
                text_state.horizontal_scaling = max(float(operands[0]) / 100.0, 0.01)
            output.append((operands, operator))
            continue
        if op == "TL" and operands:
            with contextlib.suppress(ValueError, TypeError):
                text_state.leading = float(operands[0])
            output.append((operands, operator))
            continue
        if op == "Ts" and operands:
            with contextlib.suppress(ValueError, TypeError):
                text_state.rise = float(operands[0])
            output.append((operands, operator))
            continue
        if op == "Tm":
            m = matrix_from_object(operands)
            if m is not None:
                text_matrix = m
                line_matrix = m
            output.append((operands, operator))
            continue
        if op in ("Td", "TD") and len(operands) >= 2:
            try:
                tx, ty = float(operands[-2]), float(operands[-1])
                if op == "TD":
                    text_state.leading = -ty
                line_matrix = mul_matrix((1.0, 0.0, 0.0, 1.0, tx, ty), line_matrix)
                text_matrix = line_matrix
            except (ValueError, TypeError):
                pass
            output.append((operands, operator))
            continue
        if op == "T*":
            line_matrix = mul_matrix((1.0, 0.0, 0.0, 1.0, 0.0, -text_state.leading), line_matrix)
            text_matrix = line_matrix
            output.append((operands, operator))
            continue

        # Form XObject recursion
        if op == "Do" and operands and recurse_forms and resources is not None:
            stats.form_ops += 1
            xobj_name = operands[0]
            name_str = str(xobj_name)
            xobjects = None
            try:
                if pikepdf.Name("/XObject") in resources:
                    xobjects = resources[pikepdf.Name("/XObject")]
            except Exception as exc:
                # Silently skipping the form's text here would drop content with
                # no trace; report it so a lost-coverage bug is diagnosable.
                logger.warning("stream_strip: unreadable /XObject resources (%s)", exc)

            if xobjects is not None and name_str not in visited_forms:
                try:
                    target_xobj = xobjects.get(xobj_name)
                    if target_xobj is not None and target_xobj.get(
                        pikepdf.Name("/Subtype")
                    ) == pikepdf.Name("/Form"):
                        if (
                            shared_forms
                            and target_xobj.is_indirect
                            and target_xobj.objgen in shared_forms
                        ):
                            # The form is one indirect object drawn by several
                            # pages; rewriting it in place would delete this
                            # page's text from every other page too. Fail closed:
                            # keep the source text rather than lose it elsewhere.
                            stats.shared_forms_skipped += 1
                            logger.debug(
                                "stream_strip: form %s is shared across pages; "
                                "leaving its text in place",
                                name_str,
                            )
                        else:
                            form_matrix = (
                                matrix_from_object(target_xobj.get(pikepdf.Name("/Matrix")))
                                or IDENTITY_MATRIX
                            )
                            child_ctm = mul_matrix(form_matrix, ctm)
                            child_res = target_xobj.get(pikepdf.Name("/Resources")) or resources
                            form_parsed = pikepdf.parse_content_stream(target_xobj)
                            visited_forms.add(name_str)
                            try:
                                sub_out, sub_dropped = strip_stream_instructions(
                                    cast(list[tuple[Sequence[Any], Any]], list(form_parsed)),
                                    strip_index,
                                    protected_index,
                                    stats,
                                    initial_ctm=child_ctm,
                                    resources=child_res,
                                    recurse_forms=recurse_forms,
                                    visited_forms=visited_forms,
                                    shared_forms=shared_forms,
                                )
                            finally:
                                # A failed recursion must not poison the set:
                                # without this, later same-named Forms are
                                # skipped and their text survives (fail-open).
                                visited_forms.remove(name_str)
                            if sub_dropped > 0:
                                target_xobj.write(pikepdf.unparse_content_stream(sub_out))
                                stats.forms_changed += 1
                                dropped_count += sub_dropped
                except Exception as exc:
                    logger.debug("Error processing Form XObject %s: %s", name_str, exc)

            output.append((operands, operator))
            continue

        # Text show operators: Tj, TJ, ', "
        if op in TEXT_SHOW_OPERATORS:
            if op in ("'", '"'):
                if op == '"' and len(operands) >= 3:
                    try:
                        text_state.word_spacing = float(operands[0])
                        text_state.char_spacing = float(operands[1])
                    except (ValueError, TypeError):
                        pass
                line_matrix = mul_matrix(
                    (1.0, 0.0, 0.0, 1.0, 0.0, -text_state.leading), line_matrix
                )
                text_matrix = line_matrix

            char_count, space_count, adj = _extract_text_metrics(operands)
            glyph_width_em: float | None = None
            if font_adv:
                em = _advance_from_widths(_show_glyph_bytes(operands), font_adv)
                # A zero sum means the codes were not in the width table (wrong
                # encoding or a font we mis-parsed); keep the conservative estimate.
                if em > 0.0:
                    glyph_width_em = em
            advance_w = _calculate_text_advance(
                text_state, char_count, space_count, adj, glyph_width_em
            )
            user_origin, text_rect = _compute_user_text_geometry(
                ctm, text_matrix, text_state, advance_w
            )

            # Advance text matrix horizontally for subsequent shows
            text_matrix = mul_matrix((1.0, 0.0, 0.0, 1.0, advance_w, 0.0), text_matrix)

            # Test 2D strip vs protection
            is_strip_target = strip_index.matches_text_for_removal(
                user_origin[0], user_origin[1], text_rect
            )
            is_protected = protected_index.protects_text_rect(
                user_origin[0], user_origin[1], text_rect
            )

            if is_strip_target and not is_protected:
                dropped_count += 1
                stats.dropped_ops += 1
                continue
            else:
                stats.kept_ops += 1
                output.append((operands, operator))
                continue

        # All other operators
        output.append((operands, operator))

    if path_buf:
        _keep_path()  # stream ended with an unpainted path — never lose it

    return output, dropped_count


def shared_form_objgens(pdf: pikepdf.Pdf) -> set[tuple[int, int]]:
    """Objgen of every Form XObject drawn by more than one page.

    Rewriting such a Form in place while stripping page A would erase page B's
    text too, because both pages draw the same indirect object. Callers pass the
    result to :func:`strip_page_text_pikepdf` so those Forms are treated as
    read-only. A form drawn twice on the *same* page counts once and is not
    shared (rewriting it only affects that page).
    """
    pages_drawing: dict[tuple[int, int], set[int]] = {}
    for page_index, page in enumerate(pdf.pages):
        resources = page.get(pikepdf.Name("/Resources"))
        if resources is None:
            continue
        xobjects = resources.get(pikepdf.Name("/XObject"))
        if xobjects is None:
            continue
        try:
            items = list(xobjects.items())
        except Exception:
            continue
        for _name, val in items:
            try:
                if (
                    val.get(pikepdf.Name("/Subtype")) == pikepdf.Name("/Form")
                    and val.is_indirect
                ):
                    pages_drawing.setdefault(val.objgen, set()).add(page_index)
            except Exception:
                continue
    return {objgen for objgen, pages in pages_drawing.items() if len(pages) > 1}


def strip_page_text_pikepdf(
    page: pikepdf.Page,
    strip_rects: list[Rect],
    protected_rects: list[Rect] | None = None,
    page_no: int = 0,
    recurse_forms: bool = True,
    shared_forms: set[tuple[int, int]] | None = None,
) -> StreamStripStats:
    """Delete source text under strip_rects in a pikepdf Page with 2D precision.

    Leaves protected_rects untouched. Handles Form XObjects recursively, except
    forms shared across pages (``shared_forms``, from :func:`shared_form_objgens`),
    which are left intact so their text is not erased from other pages.
    """
    stats = StreamStripStats(page=page_no)
    if not strip_rects:
        stats.aborted = "no_rects"
        return stats

    strip_index = Rect2DIndex.build(strip_rects)
    protected_index = Rect2DIndex.build(protected_rects or [])

    try:
        if pikepdf.Name("/Contents") not in page:
            stats.aborted = "empty_contents"
            return stats

        page.contents_coalesce()
        parsed = pikepdf.parse_content_stream(page)
        instructions = cast(list[tuple[Sequence[Any], Any]], list(parsed))
        resources = page.get(pikepdf.Name("/Resources"))

        new_ops, dropped = strip_stream_instructions(
            instructions,
            strip_index,
            protected_index,
            stats,
            initial_ctm=IDENTITY_MATRIX,
            resources=resources,
            recurse_forms=recurse_forms,
            shared_forms=shared_forms,
        )

        if dropped > 0:
            unparsed = pikepdf.unparse_content_stream(new_ops)
            page.Contents.write(unparsed)

    except Exception as exc:
        # Keep the abort reason actionable: callers inspect ``aborted`` to
        # decide whether to fall back, so preserve the exception type and a
        # bounded message rather than only the type name.
        stats.aborted = f"error:{type(exc).__name__}:{str(exc)[:200]}"
        logger.debug("pikepdf strip page %s aborted: %s", page_no, exc)
        return stats

    return stats


__all__ = [
    "IDENTITY_MATRIX",
    "Matrix2D",
    "Rect",
    "Rect2DIndex",
    "StreamStripStats",
    "TextState",
    "mul_matrix",
    "shared_form_objgens",
    "strip_page_text_pikepdf",
    "strip_stream_instructions",
    "transform_point",
]
