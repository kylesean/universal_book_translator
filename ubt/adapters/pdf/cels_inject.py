"""CELS Stage 5 spike: rewrite one text line's glyph run in a PDF content stream.

G3 scope (``docs/guides/ADAPTIVE_DELIVERY_IMPLEMENTATION_PLAN.md`` §4.3): prove
that ``Tz`` / ``Tc`` / ``TL`` injection survives text extraction, colours,
clipping and ``q``/``Q`` **before** G4 builds a solver on top of it.

Why the probe is ASCII
----------------------
Measured on ``chapter-3.pdf`` p9/p12, every text font is a single-byte Type1
with a ``MacRomanEncoding`` / ``/Differences`` encoding and the body fonts carry
**no** ``/ToUnicode``. A CJK translation therefore cannot be written with the
source font at all; it needs an embedded, subsetted target font with its own
``ToUnicode`` — which is Stage 5 work for G4, not this spike. Using an ASCII
probe against the source font is what isolates the *mechanism* (metrics
operators, text extraction, graphics state) from the *coverage* question, and
the spike reports coverage separately as its own go/no-go item.

What is exact and what is not
-----------------------------
Widths are computed from the PDF itself: ``/Widths`` per code, plus ``Tc`` per
glyph, ``Tw`` per space, the ``TJ`` number adjustments, all multiplied by ``Tz``
— the spec's displacement formula, so a fit decision never guesses a glyph
advance. What this module does *not* model is a font whose ``/Widths`` is
missing (base-14 without metrics): it fails closed with
:class:`InjectionUnsupported` instead of inventing numbers.

Text space versus device space (measured G3 finding)
-----------------------------------------------------
``Tc``, ``Tw`` and ``TL`` are **text-space** operands: the text matrix scales
them afterwards. ``chapter-3.pdf`` sets ``Tf … 1`` and carries the real size in
``Tm`` (``9.9626``), so ``Tc -0.164`` is a **-1.64pt** device shift per glyph and
``TL 11.45`` moves a line **114pt**, not 11.45. Every device-space delta this
module computes is therefore divided by :attr:`LineSite.text_scale` before it is
written. ``Tz`` is a percentage of the already-scaled advance and needs no
conversion. This is exactly the class of defect G3 exists to find: the operator
sequence looks right, the PDF parses, and the page is silently wrong.

Graphics state is never touched: the rewrite inserts operators and replaces
show operators, but every ``q``/``Q``/``cm``/``re``/``W``/colour/``Do``
instruction stays in place and in order. The spike re-parses the result and
counts them to prove it, rather than asserting it in a docstring.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pikepdf

from ubt.adapters.pdf.strip_pdf_text import cast_instructions
from ubt.core.layout import Box, InlineRigidBox

#: A 3x2 affine transform, PDF's row-vector convention: ``p' = p · M``.
Matrix = tuple[float, float, float, float, float, float]

_IDENTITY: Matrix = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)

#: Operators that paint glyphs.
_SHOW_OPS: frozenset[str] = frozenset({"Tj", "TJ", "'", '"'})

#: Operators that move the *line* matrix (and therefore start a new line).
_LINE_MOVE_OPS: frozenset[str] = frozenset({"Td", "TD", "T*", "Tm", "'", '"'})

#: Vertical tolerance, in points, for "these show operators share a baseline".
#: Subscripts and superscripts inside one line sit 1–3pt off the baseline and
#: have a different ``Tf`` size, so they split off as their own sites; that is
#: the honest model, because a line whose parts have different sizes is not one
#: elastic run.
BASELINE_TOL_PT: float = 1.0

#: Stage 3A's search bounds. The PRD's micro-typography order is *tracking
#: before scaling*: negative ``Tc`` changes spacing without altering the face,
#: while ``Tz`` visibly narrows the glyphs, so ``Tz`` is only spent after the
#: tracking budget is exhausted.
DEFAULT_MAX_TRACKING_EM: float = 0.12
DEFAULT_MAX_SCALE_LOSS: float = 0.12

#: Vertical extent assumed for a line when only its baseline and size are
#: known. Used *only* for the y-overlap guard in Stage 1's gutter test (a
#: 0.02-vs-0.5 comparison), never for a fit decision, which is why an
#: approximation is acceptable here and not acceptable in a width.
_ASCENT_SHARE: float = 0.75
_DESCENT_SHARE: float = 0.25

#: Fixed-point grid for the fit search, in scale units. 0.005 over a 0.12
#: budget is 25 candidates — fine enough that the reported scale is not
#: visibly coarser than the measurement it is matched to.
_SCALE_STEP: float = 0.005


class InjectionUnsupported(RuntimeError):
    """The page/font cannot be rewritten by this spike's simplifications.

    Raised for a composite (``Type0``) font, a font without ``/Widths``, a
    page whose content stream cannot be parsed, and any non-ASCII probe text.
    Each of those needs the G4 machinery this spike deliberately does not have.
    """


@dataclass(frozen=True, slots=True)
class FontMetrics:
    """Advance widths of a simple (single-byte) font, in 1/1000 em."""

    first_char: int
    widths: tuple[float, ...]

    @classmethod
    def from_font(cls, font: Any) -> FontMetrics:
        obj = font.get_object() if hasattr(font, "get_object") else font
        subtype = str(obj.get("/Subtype", ""))
        if subtype == "/Type0":
            raise InjectionUnsupported(
                "composite /Type0 font: needs CMap + CID widths, i.e. the G4 "
                "font pipeline, not a single-byte rewrite"
            )
        widths = obj.get("/Widths")
        first = obj.get("/FirstChar")
        if widths is None or first is None:
            raise InjectionUnsupported(
                f"font {obj.get('/BaseFont')!s} has no /Widths+/FirstChar; base-14 "
                "metrics are not modelled and guessing an advance would corrupt a fit"
            )
        return cls(first_char=int(first), widths=tuple(float(w) for w in widths))

    def advance_em(self, code: int) -> float:
        """Advance of one character code in ems (0 for codes outside the table)."""
        offset = code - self.first_char
        if 0 <= offset < len(self.widths):
            return self.widths[offset] / 1000.0
        return 0.0


def encode_probe(text: str) -> bytes:
    """Encode an ASCII probe for a single-byte font.

    Refuses anything outside printable ASCII: the corpus fonts use different
    single-byte encodings (MacRoman here, WinAnsi elsewhere) and writing a
    Latin-1 byte would silently paint the wrong glyph under one of them.
    """
    try:
        raw = text.encode("ascii")
    except UnicodeEncodeError as exc:
        raise InjectionUnsupported(
            "probe text is not ASCII; a non-Latin target needs an embedded font "
            "with its own /Encoding and /ToUnicode (G4 Stage 5), not the source font"
        ) from exc
    if any(byte < 0x20 or byte > 0x7E for byte in raw):
        raise InjectionUnsupported("probe text must stay inside printable ASCII 0x20–0x7E")
    return raw


def advance_width(
    text: str,
    metrics: FontMetrics,
    size_pt: float,
    *,
    horizontal_scale: float = 1.0,
    char_spacing: float = 0.0,
    word_spacing: float = 0.0,
    rigid_boxes: Sequence[InlineRigidBox] = (),
) -> float:
    """Rendered advance of ``text`` under the PDF displacement formula.

    ``width = Th · (Σ w0·Tfs + n·Tc + n_spaces·Tw) + Σ W_rigid`` in the spec's
    terms, which is why ``Tc`` is inside the ``Tz`` multiplication: horizontal
    scaling shrinks the spacing too, while rigid boxes stay fixed.
    """
    raw = text.encode("ascii")
    total = char_spacing * len(raw)
    for code in raw:
        total += metrics.advance_em(code) * size_pt
        if code == 0x20:
            total += word_spacing
    text_width = total * horizontal_scale
    rigid_width = sum(b.width_pt for b in rigid_boxes)
    return text_width + rigid_width


@dataclass(frozen=True, slots=True)
class HorizontalFit:
    """Stage 3A's horizontal decision for one line."""

    horizontal_scale: float
    char_spacing: float
    width_pt: float
    fits: bool
    #: How far past the available width the line still paints when tracking and
    #: scaling are both at their bound. Positive only when ``fits`` is false, and
    #: that number is Stage 4's (L1 compression) input.
    deficit_pt: float = 0.0


def solve_horizontal(
    text: str,
    metrics: FontMetrics,
    size_pt: float,
    available_pt: float,
    *,
    max_tracking_em: float = DEFAULT_MAX_TRACKING_EM,
    max_scale_loss: float = DEFAULT_MAX_SCALE_LOSS,
    word_spacing: float = 0.0,
    rigid_boxes: Sequence[InlineRigidBox] = (),
) -> HorizontalFit:
    """Fit ``text`` into ``available_pt`` by tracking first, then scaling.

    ``available_pt`` is an **advance** budget, not an ink budget: the last
    glyph's right side bearing paints past its advance, measured at 0.59pt on
    ``chapter-3.pdf`` p12. A caller that must keep ink inside a column therefore
    reserves that pad itself instead of letting the solver assume the bearing
    away.
    """
    rigid_width = sum(b.width_pt for b in rigid_boxes)
    net_available = available_pt - rigid_width
    tracking_floor = -max_tracking_em * size_pt
    count = len(text.encode("ascii"))

    if net_available <= 0.0 and count > 0:
        return HorizontalFit(
            horizontal_scale=round(1.0 - max_scale_loss, 4),
            char_spacing=round(tracking_floor, 4),
            width_pt=round(rigid_width, 3),
            fits=False,
            deficit_pt=round(abs(net_available), 3),
        )

    natural = advance_width(text, metrics, size_pt, word_spacing=word_spacing)
    if count == 0:
        return HorizontalFit(
            1.0,
            0.0,
            round(rigid_width, 3),
            rigid_width <= available_pt,
            max(0.0, round(rigid_width - available_pt, 3)),
        )
    if natural <= net_available:
        return HorizontalFit(1.0, 0.0, round(natural + rigid_width, 3), True, 0.0)

    steps = max(1, int(round(max_scale_loss / _SCALE_STEP)))
    for step in range(steps + 1):
        scale = 1.0 - step * _SCALE_STEP
        needed = (net_available / scale - natural) / count
        if needed < tracking_floor:
            continue
        width = advance_width(
            text,
            metrics,
            size_pt,
            horizontal_scale=scale,
            char_spacing=needed,
            word_spacing=word_spacing,
            rigid_boxes=rigid_boxes,
        )
        return HorizontalFit(round(scale, 4), round(needed, 4), round(width, 3), True)

    scale = 1.0 - max_scale_loss
    tracking = tracking_floor
    width = advance_width(
        text,
        metrics,
        size_pt,
        horizontal_scale=scale,
        char_spacing=tracking,
        word_spacing=word_spacing,
        rigid_boxes=rigid_boxes,
    )
    return HorizontalFit(
        horizontal_scale=round(scale, 4),
        char_spacing=round(tracking, 4),
        width_pt=round(width, 3),
        fits=False,
        deficit_pt=round(width - available_pt, 3),
    )


@dataclass(frozen=True, slots=True)
class ShowSite:
    """One show operator, located in page space and at its instruction index."""

    index: int
    op: str
    text: str
    x: float
    baseline: float
    font_size: float
    #: ``Tf`` size times the text-matrix scale. ``Tc`` and ``TL`` operands are
    #: in *text space* and the matrix multiplies them afterwards, so a device
    #: delta has to be divided by this before it is written into the stream.
    text_scale: float
    font_key: str
    horizontal_scale: float
    char_spacing: float
    word_spacing: float


@dataclass(frozen=True, slots=True)
class LineSite:
    """A run of show operators sharing one baseline, font and size."""

    sites: tuple[ShowSite, ...]
    text: str
    x0: float
    x1: float
    baseline: float
    font_size: float
    text_scale: float
    font_key: str
    horizontal_scale: float
    char_spacing: float
    word_spacing: float

    @property
    def first_index(self) -> int:
        return self.sites[0].index

    @property
    def last_index(self) -> int:
        return self.sites[-1].index

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    def box(self) -> Box:
        """Ink-ish box for Stage 1's y-overlap guard (see ``_ASCENT_SHARE``)."""
        return Box(
            x0=self.x0,
            y0=self.baseline - _DESCENT_SHARE * self.font_size,
            x1=self.x1,
            y1=self.baseline + _ASCENT_SHARE * self.font_size,
        )


def scan_lines(
    page: pikepdf.Page,
    *,
    baseline_tol: float = BASELINE_TOL_PT,
) -> tuple[LineSite, ...]:
    """Group a page's show operators into measured lines.

    Widths are accumulated exactly (``/Widths`` + ``TJ`` adjustments + ``Tc`` /
    ``Tw``, all through ``Tz``), so a fit decision downstream is arithmetic on
    the PDF's own metrics rather than on an average-glyph guess.
    """
    instructions = _instructions(page)
    fonts = _font_resources(page)
    metrics: dict[str, FontMetrics] = {}

    out: list[LineSite] = []
    ctm: Matrix = _IDENTITY
    saved: list[Matrix] = []
    text_matrix: Matrix = _IDENTITY
    line_matrix: Matrix = _IDENTITY
    leading = 0.0
    font_size = 0.0
    font_key = ""
    horizontal_scale = 1.0
    char_spacing = 0.0
    word_spacing = 0.0
    origin_x = 0.0
    origin_y = 0.0
    pending: list[ShowSite] = []
    pending_width = 0.0

    def flush() -> None:
        nonlocal pending, pending_width
        if pending:
            out.append(
                LineSite(
                    sites=tuple(pending),
                    text="".join(site.text for site in pending),
                    x0=pending[0].x,
                    x1=pending[0].x + pending_width,
                    baseline=pending[0].baseline,
                    font_size=pending[0].font_size,
                    text_scale=pending[-1].text_scale,
                    font_key=pending[0].font_key,
                    horizontal_scale=pending[-1].horizontal_scale,
                    char_spacing=pending[-1].char_spacing,
                    word_spacing=pending[-1].word_spacing,
                )
            )
        pending = []
        pending_width = 0.0

    def metrics_for(key: str) -> FontMetrics:
        if key not in metrics:
            font = fonts.get(key)
            if font is None:
                raise InjectionUnsupported(f"font resource {key!r} not found on the page")
            metrics[key] = FontMetrics.from_font(font)
        return metrics[key]

    for index, inst in enumerate(instructions):
        operands, operator = inst[0], inst[1]
        op = str(operator)

        if op == "q":
            saved.append(ctm)
        elif op == "Q":
            if saved:
                ctm = saved.pop()
            flush()
        elif op == "cm" and len(operands) >= 6:
            ctm = _compose(_matrix_from(operands), ctm)
        elif op == "BT":
            text_matrix = line_matrix = _IDENTITY
            origin_x, origin_y = _device_origin(text_matrix, ctm)
            flush()
        elif op == "ET":
            flush()
        elif op == "Tm" and len(operands) >= 6:
            flush()
            text_matrix = line_matrix = _matrix_from(operands)
            origin_x, origin_y = _device_origin(text_matrix, ctm)
        elif op in ("Td", "TD") and len(operands) >= 2:
            flush()
            tx, ty = float(operands[0]), float(operands[1])
            if op == "TD":
                leading = -ty
            line_matrix = _compose(_translate(tx, ty), line_matrix)
            text_matrix = line_matrix
            origin_x, origin_y = _device_origin(text_matrix, ctm)
        elif op == "T*":
            flush()
            line_matrix = _compose(_translate(0.0, -leading), line_matrix)
            text_matrix = line_matrix
            origin_x, origin_y = _device_origin(text_matrix, ctm)
        elif op == "TL" and len(operands) >= 1:
            flush()
            leading = float(operands[0])
        elif op == "Tf" and len(operands) >= 2:
            flush()
            font_key = str(operands[0])
            try:
                font_size = float(operands[1])
            except (TypeError, ValueError) as exc:
                raise InjectionUnsupported(f"unparsable Tf size {operands[1]!r}") from exc
        elif op == "Tz" and len(operands) >= 1:
            flush()
            horizontal_scale = _percent(operands[0])
        elif op == "Tc" and len(operands) >= 1:
            flush()
            char_spacing = float(operands[0])
        elif op == "Tw" and len(operands) >= 1:
            flush()
            word_spacing = float(operands[0])
        elif op in ("'", '"'):
            flush()
            line_matrix = _compose(_translate(0.0, -leading), line_matrix)
            text_matrix = line_matrix
            origin_x, origin_y = _device_origin(text_matrix, ctm)
        elif op in _LINE_MOVE_OPS:
            flush()

        if op not in _SHOW_OPS:
            continue
        if not font_key or font_size <= 0.0:
            # A show operator with no font in force is malformed; leaving it out
            # of the model is safer than inventing a size for it.
            continue
        effective = _effective_font_size(font_size, text_matrix, ctm)
        if pending and (
            abs(origin_y - pending[0].baseline) > baseline_tol
            or font_key != pending[0].font_key
            or abs(effective - pending[0].font_size) > 0.01
        ):
            flush()
        raw = _show_text(operands, op)
        site = ShowSite(
            index=index,
            op=op,
            text=raw.decode("latin-1"),
            x=origin_x,
            baseline=origin_y,
            font_size=effective,
            text_scale=_text_scale(text_matrix, ctm),
            font_key=font_key,
            horizontal_scale=horizontal_scale,
            char_spacing=char_spacing,
            word_spacing=word_spacing,
        )
        pending.append(site)
        pending_width += _show_advance(
            operands,
            op,
            metrics_for(font_key),
            effective,
            horizontal_scale=horizontal_scale,
            char_spacing=char_spacing,
            word_spacing=word_spacing,
        )
    flush()
    return tuple(out)


def inject_line(
    page: pikepdf.Page,
    line: LineSite,
    text: str,
    *,
    horizontal_scale: float = 1.0,
    char_spacing: float = 0.0,
    leading_probe: tuple[str, float] | None = None,
) -> bytes:
    """Replace ``line``'s glyph run with ``text``, preserving the operator order.

    The whole run is replaced by one ``Tj``: a line built from 12 ``Tj``/``TJ``
    operators becomes one show, so the model is "one line, one string" — which
    is what a translation is. ``Tz``/``Tc`` are set before and restored after
    with the values that were already in force, so a following line is
    unaffected.

    ``leading_probe``, when given as ``(probe_text, leading_pt)``, appends
    ``TL`` + ``T*`` + the probe text. Because ``T*`` moves from the **line**
    matrix (still at the line start) and not from the advanced text matrix, the
    probe must land at ``x0`` and ``baseline - leading`` — that is the property
    the spike measures, and the reason ``TL`` is useless for per-line ``Tm``
    positioning.
    """
    if not 0 <= line.first_index <= line.last_index:
        raise InjectionUnsupported("line has no show operators to replace")
    instructions = _instructions(page)
    if line.last_index >= len(instructions):
        raise InjectionUnsupported("line indices are outside the content stream")

    # ``Tc`` and ``TL`` operands are text-space quantities: the text matrix
    # multiplies them afterwards, so a device-space delta must be divided by the
    # matrix scale. ``Tz`` is a percentage and needs no conversion.
    scale = line.text_scale if line.text_scale > 0.0 else 1.0
    tracking_operand = char_spacing / scale
    replacement: list[Any] = [
        ([round(horizontal_scale * 100.0, 4)], pikepdf.Operator("Tz")),
        ([round(tracking_operand, 4)], pikepdf.Operator("Tc")),
        ([pikepdf.String(encode_probe(text))], pikepdf.Operator("Tj")),
    ]
    if leading_probe is not None:
        probe_text, leading = leading_probe
        replacement.extend(
            [
                ([round(leading / scale, 4)], pikepdf.Operator("TL")),
                ([], pikepdf.Operator("T*")),
                ([pikepdf.String(encode_probe(probe_text))], pikepdf.Operator("Tj")),
            ]
        )
    replacement.extend(
        [
            ([round(line.horizontal_scale * 100.0, 4)], pikepdf.Operator("Tz")),
            ([round(line.char_spacing, 4)], pikepdf.Operator("Tc")),
        ]
    )
    merged = instructions[: line.first_index] + replacement + instructions[line.last_index + 1 :]
    return pikepdf.unparse_content_stream(merged)


def replace_stream(page: pikepdf.Page, data: bytes, pdf: pikepdf.Pdf) -> None:
    """Install rewritten stream bytes on a page (the only PDF-mutating step)."""
    page.Contents = pdf.make_stream(data)


def _instructions(page: pikepdf.Page) -> list[Any]:
    try:
        return cast_instructions(pikepdf.parse_content_stream(page))
    except Exception as exc:
        raise InjectionUnsupported(f"content stream does not parse: {exc}") from exc


def _font_resources(page: pikepdf.Page) -> dict[str, Any]:
    resources = page.Resources
    fonts = resources.get("/Font") if resources is not None else None
    if fonts is None:
        return {}
    return {str(key): value for key, value in fonts.items()}


def _show_text(operands: Sequence[Any], op: str) -> bytes:
    """The glyph bytes one show operator paints."""
    if op == "TJ":
        return b"".join(_glyph_bytes(item) for item in operands[0] if _is_string(item))
    if op in ("'", '"'):
        return _glyph_bytes(operands[-1])
    return _glyph_bytes(operands[0])


def _glyph_bytes(item: Any) -> bytes:
    """Raw glyph bytes of a content-stream string operand.

    ``bytes(obj)`` is the only accessor that works on a parsed
    ``pikepdf.objects.Object`` string: attribute access on one is dictionary
    lookup (``obj.bytes`` raises "cannot get key '/bytes'"), and ``str(obj)``
    decodes through PDFDocEncoding, which is not the glyph code space here.
    """
    if isinstance(item, str):
        return item.encode("latin-1", "replace")
    if isinstance(item, bytes):
        return item
    try:
        return bytes(item)
    except (TypeError, ValueError):
        return str(item).encode("latin-1", "replace")


def _is_string(item: Any) -> bool:
    """Whether a ``TJ`` element is a glyph run rather than a kerning number.

    A parsed ``pikepdf`` object exposes no type code from Python and probing an
    attribute on a string raises instead of returning a default, so the
    discriminator is the conversion the operand would need: a kerning number
    parses as ``float``, a glyph run does not.
    """
    if isinstance(item, (str, bytes)):
        return True
    if isinstance(item, pikepdf.Object):
        try:
            float(item)
        except (TypeError, ValueError):
            return True
        return False
    return False


def _show_advance(
    operands: Sequence[Any],
    op: str,
    metrics: FontMetrics,
    size_pt: float,
    *,
    horizontal_scale: float,
    char_spacing: float,
    word_spacing: float,
) -> float:
    """Advance of one show operator, including ``TJ`` kerning adjustments."""
    raw_total = 0.0
    if op == "TJ":
        for item in operands[0]:
            if _is_string(item):
                glyphs = _glyph_bytes(item)
                raw_total += char_spacing * len(glyphs)
                for code in glyphs:
                    raw_total += metrics.advance_em(code) * size_pt
                    if code == 0x20:
                        raw_total += word_spacing
            else:
                try:
                    raw_total += -float(item) / 1000.0 * size_pt
                except (TypeError, ValueError):
                    continue
    else:
        glyphs = _show_text(operands, op)
        raw_total += char_spacing * len(glyphs)
        for code in glyphs:
            raw_total += metrics.advance_em(code) * size_pt
            if code == 0x20:
                raw_total += word_spacing
    return raw_total * horizontal_scale


def _percent(value: Any) -> float:
    try:
        return float(value) / 100.0
    except (TypeError, ValueError) as exc:
        raise InjectionUnsupported(f"unparsable Tz operand {value!r}") from exc


def _compose(first: Matrix, second: Matrix) -> Matrix:
    """``first`` applied, then ``second`` (row-vector convention)."""
    a1, b1, c1, d1, e1, f1 = first
    a2, b2, c2, d2, e2, f2 = second
    return (
        a1 * a2 + b1 * c2,
        a1 * b2 + b1 * d2,
        c1 * a2 + d1 * c2,
        c1 * b2 + d1 * d2,
        e1 * a2 + f1 * c2 + e2,
        e1 * b2 + f1 * d2 + f2,
    )


def _translate(tx: float, ty: float) -> Matrix:
    return (1.0, 0.0, 0.0, 1.0, tx, ty)


def _matrix_from(operands: Sequence[Any]) -> Matrix:
    a, b, c, d, e, f = (float(value) for value in operands[:6])
    return (a, b, c, d, e, f)


def _device_origin(text_matrix: Matrix, ctm: Matrix) -> tuple[float, float]:
    dev = _compose(text_matrix, ctm)
    return dev[4], dev[5]


def _effective_font_size(font_size: float, text_matrix: Matrix, ctm: Matrix) -> float:
    dev = _compose(text_matrix, ctm)
    scale = math.hypot(dev[0], dev[1])
    return font_size * scale if scale > 0.0 else font_size


def _text_scale(text_matrix: Matrix, ctm: Matrix) -> float:
    """Text-space → device-space scale factor (``hypot(a, b)`` of the matrix)."""
    dev = _compose(text_matrix, ctm)
    return math.hypot(dev[0], dev[1])


__all__ = [
    "BASELINE_TOL_PT",
    "DEFAULT_MAX_SCALE_LOSS",
    "DEFAULT_MAX_TRACKING_EM",
    "FontMetrics",
    "HorizontalFit",
    "InjectionUnsupported",
    "LineSite",
    "ShowSite",
    "advance_width",
    "encode_probe",
    "inject_line",
    "replace_stream",
    "scan_lines",
    "solve_horizontal",
]
