"""PDF background canvas text stripper.

Builds the vector background layer for 1:1 facsimile delivery: the source page
with its *translated* text removed and everything else — vector paths, fills,
strokes, shadings, rules, clipping paths, embedded raster images, figure labels
— left intact.

**Granularity is one text line, not one ``BT…ET`` block.** A ``BT…ET`` block is
not a unit of text layout: measured on the corpus, one block routinely carries an
entire page (``2606.23344v1.pdf`` p2 has 2 blocks for 91 text lines;
``2609.22978v1.pdf`` p14 has 1 block for 72 lines; ``2609.20519v1.pdf`` p1 has
6 blocks for 313 lines). Deciding per block therefore collapses every page to
all-or-nothing, which is what the previous implementation did — it stripped
100 % of the text on every page and took the formulas, folios and figure labels
with it.

**Removal blanks the glyph string; it never deletes the operator.** Every show
operator also advances the text matrix, so deleting one shifts every following
run on the page. Replacing its payload with ``""`` keeps the operator sequence
structurally identical, which makes zero position drift a property of the
algorithm rather than something to hope for. Measured on the corpus: survivors
keep their positions to 0.0 pt.

**Form XObjects are not descended into by default.** The measured share of page
text living in Form XObjects is 0–8.8 %, and what lives there is figure content —
chart axis ticks (``68 70 75 80 85 90``) and a legend (``DocLayout-YOLO``,
``MinerU2.5``, ``Dolphin-v2``). Phase 1–4 of the delivery design keeps original
in-figure text, so stripping those is a defect, not a feature.

**The fail-safe direction is "keep".** ``erase_rects=None`` means *strip
nothing*, not *strip everything*. The old default stripped the page clean, and
the renderer called it with no rects at all — a single omission silently deleted
every formula and folio in the book. A caller that genuinely wants a blank page
passes one whole-page rect, which is explicit and greppable.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pikepdf

logger = logging.getLogger(__name__)

#: A 3x2 affine transform, PDF's row-vector convention: ``p' = p · M``.
Matrix = tuple[float, float, float, float, float, float]

_IDENTITY: Matrix = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)

#: Operators that paint glyphs.
_SHOW_OPS: frozenset[str] = frozenset({"Tj", "TJ", "'", '"'})

#: How far left of a show operator's pen its first glyph's ink can start, as a
#: share of the font size — the left side bearing. Erase boxes are grown leftward
#: by this share so a run cannot escape the box it paints into.
#:
#: Two measurements set the coefficient, and the *larger* one wins:
#:
#: * ``2606.23344v1.pdf`` p1: a 15.9pt title's pen at ``x = 92.2`` against an
#:   erase box at ``x0 = 92.9`` — a 0.6pt shortfall, **0.038 em**.
#: * ``2608.25512v1.pdf`` p2: a 11pt table-of-contents entry's pen at
#:   ``x = 87.23`` against ``x0 = 87.89`` — a 0.66pt shortfall, **0.060 em**.
#:
#: The coefficient was 0.05, which covers the first case (0.05 · 15.9 = 0.80pt)
#: but not the second (0.05 · 11 = 0.55pt < 0.66pt), so every TOC entry number
#: on that document survived the strip — 15 glyphs, found by the character-level
#: probe and swept: 0.05 and 0.06 em still leak, 0.07 em and above leak nothing
#: with 0 losses. 0.08 is the measured threshold plus margin.
#:
#: Leftward only, because ink never starts *right* of the pen for left-to-right
#: text; padding the right edge would instead swallow the first glyph of a
#: neighbouring block. Keep rects are tested first and win, so a keep block
#: adjacent to a translated one is unaffected.
PEN_PAD_EM: float = 0.10
#: Floor for :data:`PEN_PAD_EM`. A producer that reports no usable ``Tf`` size
#: (or reports ``1.0``) would otherwise get no padding at all. Also bounds the
#: 9pt body case, where ``0.10 · 9 = 0.90pt`` is padded up to 0.75pt.
MIN_PEN_PAD_PT: float = 0.75


@dataclass(frozen=True, slots=True)
class BoxRect:
    """Axis-aligned rectangle in PDF user coordinates (pt, origin bottom-left)."""

    page_index: int
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def area(self) -> float:
        return self.width * self.height

    def contains_point(self, x: float, y: float, tol: float = 0.5) -> bool:
        return (self.x0 - tol) <= x <= (self.x1 + tol) and (self.y0 - tol) <= y <= (self.y1 + tol)


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
    """The first six operands as a matrix, in PDF operand order.

    ``tuple(float(v) for v in operands[:6])`` is the obvious spelling, but mypy
    widens it to ``tuple[float, ...]`` and every caller then fails against
    :data:`Matrix`. Naming the six components here keeps the one place that
    knows the width a place the checker can verify.
    """
    a, b, c, d, e, f = (float(value) for value in operands[:6])
    return (a, b, c, d, e, f)


def _device_origin(text_matrix: Matrix, ctm: Matrix) -> tuple[float, float]:
    """Page-space position of the text origin: text matrix then CTM."""
    dev = _compose(text_matrix, ctm)
    return dev[4], dev[5]


def _effective_font_size(font_size: float, text_matrix: Matrix, ctm: Matrix) -> float:
    """Page-space font size: ``Tf`` nominal size scaled by ``text_matrix ∘ ctm``."""
    dev = _compose(text_matrix, ctm)
    scale = math.hypot(dev[0], dev[1])
    if font_size > 0.0:
        return font_size * scale if scale > 0.0 else font_size
    return scale if scale > 1.0 else 0.0


def _rects_for_page(
    rects: Sequence[BoxRect] | None,
    page_index: int | None,
) -> Sequence[BoxRect] | None:
    """Restrict a pooled rect list to one page (``None`` stays ``None``)."""
    if rects is None or page_index is None:
        return rects
    return [rect for rect in rects if rect.page_index == page_index]


def _keep_point(
    x: float,
    y: float,
    erase_rects: Sequence[BoxRect] | None,
    keep_rects: Sequence[BoxRect] | None,
    tol: float = 0.5,
    font_size: float = 0.0,
) -> bool:
    """Whether a text line at page position ``(x, y)`` survives the strip.

    ``keep_rects`` wins over ``erase_rects``: an explicit protection is never
    overridden by a translated block's box that happens to overlap it.

    The vertical test is against the line's *baseline*, which sits a couple of
    points above the ink bottom that a block bbox bounds. That offset is inside
    every real block box, so a baseline test classifies each line correctly
    while a test on the ink box would need the line height we do not have.

    The horizontal test is against the *origin*, which is the pen position, not
    the ink. Glyphs extend rightward from it, so an origin can sit a fraction of
    an em outside the ink box it paints: measured on ``chapter-3.pdf`` p1, the
    display ``3`` has its origin at ``x = 376.6`` while its ink starts at
    ``x = 380.3``. That case is *not* a defect — the ``3`` is inside a
    translated block that redraws it — but it is the reason a keep rect should
    be authored from a block's ink box rather than a glyph's pen position.

    The origin-vs-ink gap cuts the other way too, and that *is* a defect: when a
    block's ink box starts a hair to the right of the pen that paints it, the
    pen falls outside the box, the run escapes the erase, and the source text
    survives under its translation. Measured on ``2606.23344v1.pdf`` p1, the
    15.9pt title's pen is at ``x = 92.2`` against an erase box at ``x0 = 92.9``,
    so the title leaked on all three arXiv papers in the corpus. Erase boxes are
    therefore grown leftward by :data:`PEN_PAD_EM` ems, scaled by ``font_size``
    when the caller knows it. The pad is leftward only — see :data:`PEN_PAD_EM`.
    """
    if keep_rects and any(rect.contains_point(x, y, tol) for rect in keep_rects):
        return True
    if erase_rects is None:
        return True
    pad = max(MIN_PEN_PAD_PT, PEN_PAD_EM * font_size) if font_size > 0 else MIN_PEN_PAD_PT
    return not any(
        rect.contains_point(x, y, tol)
        or (rect.x0 - pad <= x <= rect.x1 and rect.y0 - tol <= y <= rect.y1 + tol)
        for rect in erase_rects
    )


def _blank_strings(operands: Any) -> list[Any]:
    """The same operands with every glyph run replaced by an empty string.

    The operator keeps its structure — ``Tj`` still has one operand, ``"`` still
    has its word/char spacing pair, ``TJ`` still has its array with its kerning
    adjustments — so the text matrix advances exactly as it did before and no
    surviving run shifts.
    """

    def _blank(value: Any) -> Any:
        if isinstance(value, pikepdf.String):
            return pikepdf.String("")
        if isinstance(value, (pikepdf.Array, list, tuple)):
            return pikepdf.Array([_blank(item) for item in value])
        return value

    return [_blank(operand) for operand in operands]


def _deref(value: Any) -> Any:
    """Resolve a pikepdf indirect reference to the object it points at.

    ``pikepdf`` exposes ``get_object()`` on ``Object``, but the stub types it as
    an attribute of an opaque ``Object`` rather than a callable, so every
    ``value.get_object()`` site needs one cast rather than a project-wide
    ``# type: ignore``. ``hasattr`` stays: a plain Python value (a list, a str)
    has no ``get_object`` and is already its own object.
    """
    if hasattr(value, "get_object"):
        return cast(Any, value).get_object()
    return value


def _content_bytes(content_obj: pikepdf.Object | pikepdf.Stream | pikepdf.Page) -> bytes:
    """The raw bytes of a page's or stream's content, for the fail-open path.

    A page's ``/Contents`` may be a single stream or an array of them; both are
    concatenated in the order the interpreter would read them.
    """
    try:
        if isinstance(content_obj, pikepdf.Page):
            contents: Any = content_obj.get("/Contents")
        else:
            contents = content_obj
        if contents is None:
            return b""
        contents = _deref(contents)
        if isinstance(contents, pikepdf.Array):
            return b"\n".join(bytes(_deref(item).read_bytes()) for item in contents)
        return bytes(contents.read_bytes())
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Could not read original content bytes: %s", exc)
        return b""


def _rewrite_instructions(
    instructions: Sequence[Any],
    erase: Sequence[BoxRect] | None,
    keep: Sequence[BoxRect] | None,
    *,
    initial_ctm: Matrix = _IDENTITY,
    form_invocations: list[tuple[str, Matrix]] | None = None,
) -> list[Any]:
    """Walk one parsed content stream, blanking the show operators inside ``erase``.

    Split out from :func:`strip_stream_text` so a stream that fails *during*
    iteration — not only at parse time — can still be caught and kept intact.
    ``pikepdf.parse_content_stream`` is lazy in the operator token: a stream with
    a non-UTF-8 operator byte parses fine and then raises ``UnicodeDecodeError``
    on the first ``str(operator)``, which is exactly the case that used to be
    lost.

    ``initial_ctm`` is the CTM in force when this stream begins: identity for a
    page's top-level stream, ``form /Matrix ∘ CTM at Do`` for a Form XObject's
    stream (see :func:`_strip_one_form`). ``form_invocations``, when given,
    collects every ``(xobject name, CTM at Do)`` this stream invokes, so the
    caller can descend into forms with the correct page-space geometry.
    """
    out: list[Any] = []
    ctm: Matrix = initial_ctm
    saved_ctm: list[Matrix] = []
    text_matrix: Matrix = _IDENTITY
    line_matrix: Matrix = _IDENTITY
    leading = 0.0
    #: Font size in force, for the leftward pen pad only (see
    #: :data:`PEN_PAD_EM`). Reset by ``BT``, which is where a text object
    #: establishes its font.
    font_size = 0.0
    #: Page-space origin of the line currently being painted. Updated only by
    #: operators that move the *line* matrix, so a run of show operators on one
    #: line shares one classification.
    line_x = 0.0
    line_y = 0.0

    for inst in instructions:
        operands, operator = inst[0], inst[1]
        op = str(operator)

        if op == "q":
            saved_ctm.append(ctm)
        elif op == "Q":
            if saved_ctm:
                ctm = saved_ctm.pop()
        elif op == "cm" and len(operands) >= 6:
            ctm = _compose(_matrix_from(operands), ctm)
        elif op == "Do" and form_invocations is not None and operands:
            # The named XObject's stream runs in ``form /Matrix ∘ CTM here``;
            # recording the CTM now (Do does not modify it) lets the caller
            # descend with page-space geometry instead of form-local guesses.
            form_invocations.append((str(operands[0]), ctm))
        elif op == "BT":
            text_matrix = line_matrix = _IDENTITY
            line_x, line_y = _device_origin(text_matrix, ctm)
        elif op == "Tm" and len(operands) >= 6:
            text_matrix = line_matrix = _matrix_from(operands)
            line_x, line_y = _device_origin(text_matrix, ctm)
        elif op in ("Td", "TD") and len(operands) >= 2:
            tx, ty = float(operands[0]), float(operands[1])
            if op == "TD":
                leading = -ty
            line_matrix = _compose(_translate(tx, ty), line_matrix)
            text_matrix = line_matrix
            line_x, line_y = _device_origin(text_matrix, ctm)
        elif op == "T*":
            line_matrix = _compose(_translate(0.0, -leading), line_matrix)
            text_matrix = line_matrix
            line_x, line_y = _device_origin(text_matrix, ctm)
        elif op == "TL" and len(operands) >= 1:
            leading = float(operands[0])
        elif op == "Tf" and len(operands) >= 2:
            # The size in force matters only for the leftward pen pad: the
            # left side bearing scales with the font, so a 15.9pt title needs a
            # wider pad than a 9pt body line. A producer that reports a
            # placeholder size is caught by ``_keep_point``'s floor.
            try:
                font_size = float(operands[1])
            except (TypeError, ValueError):
                font_size = 0.0
        elif op in ("'", '"'):
            # These move to the next line and then paint, so the position is the
            # *post-move* one.
            line_matrix = _compose(_translate(0.0, -leading), line_matrix)
            text_matrix = line_matrix
            line_x, line_y = _device_origin(text_matrix, ctm)

        if op in _SHOW_OPS:
            eff_size = _effective_font_size(font_size, text_matrix, ctm)
            if not _keep_point(line_x, line_y, erase, keep, font_size=eff_size):
                out.append((_blank_strings(operands), operator))
                continue
        out.append((operands, operator))

    return out


def strip_stream_text(
    content_obj: pikepdf.Object | pikepdf.Stream | pikepdf.Page,
    *,
    erase_rects: Sequence[BoxRect] | None = None,
    keep_rects: Sequence[BoxRect] | None = None,
    page_index: int | None = None,
    form_invocations: list[tuple[str, Matrix]] | None = None,
) -> bytes:
    """Rewrite one content stream, blanking the text lines inside ``erase_rects``.

    ``erase_rects`` is the set of boxes whose text is being replaced by a
    translation. ``None`` means *strip nothing* (see the module docstring);
    ``keep_rects`` protects regions inside those boxes and always wins.

    A stream that cannot be parsed — or that fails while being walked — is
    returned **unchanged**. Returning ``b""`` instead, the previous behaviour,
    silently blanked the page's vector art along with its text, so one malformed
    stream cost the whole page.
    """
    erase = _rects_for_page(erase_rects, page_index)
    keep = _rects_for_page(keep_rects, page_index)

    try:
        instructions = cast_instructions(pikepdf.parse_content_stream(content_obj))
        out = _rewrite_instructions(instructions, erase, keep, form_invocations=form_invocations)
        return pikepdf.unparse_content_stream(out)
    except Exception as exc:
        logger.warning(
            "Could not rewrite a content stream (%s); keeping it byte-identical "
            "rather than blanking the page",
            exc,
        )
        return _content_bytes(content_obj)


def cast_instructions(parsed: Any) -> list[Any]:
    return list(parsed)


def strip_form_xobjects_recursive(
    resources: pikepdf.Dictionary | None,
    *,
    erase_rects: Sequence[BoxRect] | None = None,
    keep_rects: Sequence[BoxRect] | None = None,
    page_index: int | None = None,
    depth: int = 0,
    max_depth: int = 5,
) -> int:
    """Strip text from nested ``/Subtype /Form`` XObjects. Returns how many it visited.

    **Off by default, and deliberately so.** A Form XObject's content stream runs
    in its own coordinate space (its ``/Matrix`` plus the CTM in force at the
    ``Do`` call), so page-space rects do not describe it. Passing the page's
    ``erase_rects`` here tests every XObject line against the wrong geometry, and
    on this corpus the text that lives in XObjects is figure content — axis
    ticks and legends that the delivery design requires be kept.

    A caller may still opt in when it knows the XObjects share the page's space
    (an identity ``/Matrix`` invoked with an identity CTM). Note that this
    mutates the objects in ``pdf``; only ever call it on a handle opened for a
    throwaway copy.
    """
    if resources is None or depth > max_depth:
        return 0
    xobjects = resources.get("/XObject")
    if not isinstance(xobjects, pikepdf.Dictionary):
        return 0

    visited = 0
    for _key, value in xobjects.items():
        try:
            xobj = _deref(value)
            if not isinstance(xobj, pikepdf.Stream) or xobj.get("/Subtype") != pikepdf.Name(
                "/Form"
            ):
                continue
            visited += 1
            raw = strip_stream_text(
                xobj,
                erase_rects=erase_rects,
                keep_rects=keep_rects,
                page_index=page_index,
            )
            xobj.write(raw)
            child_resources = xobj.get("/Resources")
            if isinstance(child_resources, pikepdf.Dictionary):
                visited += strip_form_xobjects_recursive(
                    child_resources,
                    erase_rects=erase_rects,
                    keep_rects=keep_rects,
                    page_index=page_index,
                    depth=depth + 1,
                    max_depth=max_depth,
                )
        except Exception as exc:
            logger.debug("Error stripping Form XObject at depth %d: %s", depth, exc)
    return visited


_FORM_COPIED_KEYS = ("/Type", "/Subtype", "/BBox", "/Matrix", "/Group", "/Name", "/StructParents")


def _strip_one_form(
    src_pdf: pikepdf.Pdf,
    form: pikepdf.Stream,
    erase: Sequence[BoxRect] | None,
    keep: Sequence[BoxRect] | None,
    *,
    initial_ctm: Matrix,
    page_index: int | None,
    depth: int = 0,
    max_depth: int = 5,
) -> pikepdf.Stream | None:
    """Rewrite a *copy* of one Form XObject's stream under page-space rects.

    The form's text positions are mapped into page space through
    ``form /Matrix ∘ initial_ctm`` (the CTM in force at its ``Do`` call), so the
    same erase/keep rects that classify top-level lines classify the form's
    lines correctly. Text whose page-space pen position falls inside an erase
    rect is blanked — which is exactly the text a translated block owns — while
    figure labels outside every erase rect survive, and keep rects win as
    always.

    Nested forms are descended with their own composed CTMs. Returns a
    replacement Stream when anything in the subtree changed (copy-on-write: the
    original object is never mutated, so a form shared with another page is
    unaffected), else ``None``.
    """
    if depth > max_depth:
        return None
    raw = _content_bytes(form)
    if not raw:
        return None
    matrix_val = form.get("/Matrix")
    form_matrix = _IDENTITY
    if matrix_val is not None:
        try:
            form_matrix = _matrix_from(list(matrix_val))
        except (TypeError, ValueError):
            form_matrix = _IDENTITY
    start_ctm = _compose(form_matrix, initial_ctm)

    try:
        instructions = cast_instructions(pikepdf.parse_content_stream(form))
    except Exception as exc:
        logger.debug("Could not parse a Form XObject (%s); keeping it intact", exc)
        return None

    nested: list[tuple[str, Matrix]] = []
    out = _rewrite_instructions(
        instructions,
        erase,
        keep,
        initial_ctm=start_ctm,
        form_invocations=nested,
    )

    child_resources = form.get("/Resources")
    child_replacements: dict[str, pikepdf.Stream] = {}
    child_xobjects: pikepdf.Dictionary | None = None
    if nested and isinstance(child_resources, pikepdf.Dictionary):
        candidate = child_resources.get("/XObject")
        if isinstance(candidate, pikepdf.Dictionary):
            child_xobjects = candidate
            seen: set[tuple[int, int]] = set()
            for name, ctm_at_do in nested:
                value = candidate.get(name)
                child = _deref(value)
                if not isinstance(child, pikepdf.Stream):
                    continue
                if child.get("/Subtype") != pikepdf.Name("/Form"):
                    continue
                key = child.objgen
                if key in seen:
                    continue
                seen.add(key)
                replacement = _strip_one_form(
                    src_pdf,
                    child,
                    erase,
                    keep,
                    initial_ctm=ctm_at_do,
                    page_index=page_index,
                    depth=depth + 1,
                    max_depth=max_depth,
                )
                if replacement is not None:
                    child_replacements[str(name)] = replacement

    new_bytes = pikepdf.unparse_content_stream(out)
    if new_bytes == raw and not child_replacements:
        return None

    replacement = src_pdf.make_stream(new_bytes)
    for copied_key in _FORM_COPIED_KEYS:
        value = form.get(copied_key)
        if value is not None:
            replacement[copied_key] = value
    if child_replacements and child_xobjects is not None:
        new_xobjects = pikepdf.Dictionary({str(k): v for k, v in child_xobjects.items()})
        for name, stream in child_replacements.items():
            new_xobjects[name] = stream
        res_items = (
            {str(k): v for k, v in child_resources.items()}
            if isinstance(child_resources, pikepdf.Dictionary)
            else {}
        )
        new_resources = pikepdf.Dictionary(res_items)
        new_resources["/XObject"] = new_xobjects
        replacement["/Resources"] = new_resources
    elif child_resources is not None:
        replacement["/Resources"] = child_resources
    return replacement


def _strip_page_forms(
    page: pikepdf.Page,
    src_pdf: pikepdf.Pdf,
    *,
    erase_rects: Sequence[BoxRect] | None,
    keep_rects: Sequence[BoxRect] | None,
    page_index: int | None,
    invocations: Sequence[tuple[str, Matrix]] | None = None,
) -> None:
    """Blank, inside the Form XObjects this page invokes, the text a translation replaces.

    Producers bury real body text in forms (measured on ``atc25-peng-yuke.pdf``
    p1: the entire proceedings notice — "This paper is included …", the dates,
    the ISBN — lives in one ``/Fm0``), and the renderer still places a
    translation over that region: the source survives *under* its translation.
    The old ``descend_forms`` path tested page-space rects against form-local
    coordinates, which is the wrong geometry for any placed form, so descent
    stayed off by default and the text leaked.

    This walks the page stream (or uses ``invocations`` already collected during
    the top-level stream rewrite), records each ``Do``'s CTM, rewrites the invoked
    forms with the composed transform, and installs the replacements through a
    cloned ``/Resources`` — a form shared with other pages keeps its original
    bytes there.
    """
    resources = page.get("/Resources")
    if not isinstance(resources, pikepdf.Dictionary):
        return
    if invocations is None:
        try:
            instructions = cast_instructions(pikepdf.parse_content_stream(page))
        except Exception as exc:
            logger.debug("Could not parse a page stream for form descent (%s)", exc)
            return
        collected: list[tuple[str, Matrix]] = []
        _rewrite_instructions(instructions, None, None, form_invocations=collected)
        invocations = collected
    if not invocations:
        return

    xobjects = resources.get("/XObject")
    if not isinstance(xobjects, pikepdf.Dictionary):
        return

    replacements: dict[str, pikepdf.Stream] = {}
    seen: set[tuple[int, int]] = set()
    for name, ctm_at_do in invocations:
        value = xobjects.get(name)
        form = _deref(value)
        if not isinstance(form, pikepdf.Stream):
            continue
        if form.get("/Subtype") != pikepdf.Name("/Form"):
            continue
        key = form.objgen
        if key in seen:
            continue
        seen.add(key)
        replacement = _strip_one_form(
            src_pdf,
            form,
            _rects_for_page(erase_rects, page_index),
            _rects_for_page(keep_rects, page_index),
            initial_ctm=ctm_at_do,
            page_index=page_index,
        )
        if replacement is not None:
            replacements[name] = replacement

    if not replacements:
        return

    new_xobjects = pikepdf.Dictionary({str(k): v for k, v in xobjects.items()})
    for name, stream in replacements.items():
        new_xobjects[name] = stream
    new_resources = pikepdf.Dictionary({str(k): v for k, v in resources.items()})
    new_resources["/XObject"] = new_xobjects
    page.Resources = new_resources


def strip_text_streams(
    src_pdf: Path | str,
    dst_pdf: Path | str,
    *,
    pages: Sequence[int] | None = None,
    per_page_erase_rects: Mapping[int, Sequence[BoxRect]] | None = None,
    per_page_keep_rects: Mapping[int, Sequence[BoxRect]] | None = None,
    descend_forms: bool = False,
) -> Path:
    """Write ``src_pdf`` to ``dst_pdf`` with the requested text lines blanked.

    ``per_page_erase_rects`` maps a 1-indexed page number to the boxes whose text
    is being replaced. A page absent from the mapping keeps **all** its text —
    the fail-safe direction. ``per_page_keep_rects`` protects regions inside
    those boxes and always wins.

    ``descend_forms`` is off by default for the reason given in
    :func:`strip_form_xobjects_recursive`.
    """
    src_path = Path(src_pdf)
    dst_path = Path(dst_pdf)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    erase_map = per_page_erase_rects or {}
    keep_map = per_page_keep_rects or {}

    with pikepdf.open(src_path) as src:
        out = pikepdf.Pdf.new()
        total_pages = len(src.pages)
        wanted = list(pages) if pages is not None else list(range(1, total_pages + 1))
        for pno in wanted:
            if not (1 <= pno <= total_pages):
                continue
            page = src.pages[pno - 1]
            erase_r = erase_map.get(pno)
            keep_r = keep_map.get(pno)

            invocations: list[tuple[str, Matrix]] | None = [] if descend_forms else None
            raw = strip_stream_text(
                page,
                erase_rects=erase_r,
                keep_rects=keep_r,
                page_index=pno,
                form_invocations=invocations,
            )
            page.Contents = src.make_stream(raw)

            if descend_forms:
                _strip_page_forms(
                    page,
                    src,
                    erase_rects=erase_r,
                    keep_rects=keep_r,
                    page_index=pno,
                    invocations=invocations,
                )

            out.pages.append(page)

        out.save(
            dst_path,
            compress_streams=True,
            object_stream_mode=pikepdf.ObjectStreamMode.generate,
        )
    return dst_path


__all__ = [
    "BoxRect",
    "cast_instructions",
    "strip_form_xobjects_recursive",
    "strip_stream_text",
    "strip_text_streams",
]
