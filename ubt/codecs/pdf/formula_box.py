"""Lightweight instruction-slice Form XObject builder for inline PDF formulas.

Instead of duplicating the entire source page stream per inline formula (which
bloats file size 30x on math-heavy papers), this module emits ONLY the exact
PDF text/vector instructions belonging to the formula island plus their minimal
graphics state snapshots and referenced ``/Font`` entries.

The resulting Form XObject normalizes the island into logical coordinates with
its bottom-left corner at ``(0, 0)``, so placing it at any translated cursor
``(x, y)`` is a single ``q 1 0 0 1 x y cm /UBT_FX_k Do Q`` invocation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pikepdf

from ubt.codecs.pdf.stream_analyzer import (
    Matrix,
    StreamTextOp,
    VectorRule,
    mat_apply,
    mat_mul,
)


@dataclass(frozen=True, slots=True)
class InlineFormulaBox:
    """Rigid inline formula box ready to travel with translated prose."""

    tag_id: str  # e.g. "f1"
    resource_name: str  # e.g. "/UBT_FX_p1_0"
    width_pt: float
    height_pt: float
    ascent_pt: float
    descent_pt: float
    baseline_offset_pt: float  # distance from box bottom y0 to prose baseline
    op_indices: tuple[int, ...]
    rule_op_indices: tuple[int, ...]
    bbox: tuple[float, float, float, float]  # logical (x0, y0, x1, y1)
    xobject: pikepdf.Stream | None = None


def _fmt(v: float) -> str:
    s = f"{v:.4f}".rstrip("0").rstrip(".")
    return "0" if s in {"", "-0"} else s


def _serialize_pdf_operand(obj: Any) -> bytes:
    if isinstance(obj, pikepdf.String):
        raw = bytes(obj)
        return b"<" + raw.hex().upper().encode("ascii") + b">"
    if isinstance(obj, pikepdf.Array):
        inner = b" ".join(_serialize_pdf_operand(item) for item in obj)
        return b"[" + inner + b"]"
    if isinstance(obj, pikepdf.Name):
        s = str(obj)
        return (s if s.startswith("/") else "/" + s).encode("ascii")
    try:
        return _fmt(float(obj)).encode("ascii")
    except (TypeError, ValueError):
        return str(obj).encode("latin-1", errors="ignore")


def _emit_text_op_bytes(op: StreamTextOp) -> bytes:
    snap = op.snapshot
    parts: list[bytes] = [b"q"]
    ctm = snap.ctm
    if ctm != (1.0, 0.0, 0.0, 1.0, 0.0, 0.0):
        parts.append(
            f"{_fmt(ctm[0])} {_fmt(ctm[1])} {_fmt(ctm[2])} {_fmt(ctm[3])} "
            f"{_fmt(ctm[4])} {_fmt(ctm[5])} cm".encode("ascii")
        )
    r, g, b = snap.color_rgb
    parts.append(f"{_fmt(r)} {_fmt(g)} {_fmt(b)} rg".encode("ascii"))
    parts.append(b"BT")
    if snap.font_resource:
        parts.append(f"{snap.font_resource} {_fmt(snap.font_size_operand)} Tf".encode("ascii"))
    if abs(snap.tc) > 1e-6:
        parts.append(f"{_fmt(snap.tc)} Tc".encode("ascii"))
    if abs(snap.tw) > 1e-6:
        parts.append(f"{_fmt(snap.tw)} Tw".encode("ascii"))
    if abs(snap.tz - 100.0) > 1e-4:
        parts.append(f"{_fmt(snap.tz)} Tz".encode("ascii"))
    if abs(snap.ts) > 1e-6:
        parts.append(f"{_fmt(snap.ts)} Ts".encode("ascii"))

    tm = snap.text_matrix
    parts.append(
        f"{_fmt(tm[0])} {_fmt(tm[1])} {_fmt(tm[2])} {_fmt(tm[3])} "
        f"{_fmt(tm[4])} {_fmt(tm[5])} Tm".encode("ascii")
    )

    if op.operator == "TJ" and op.raw_operands:
        parts.append(_serialize_pdf_operand(op.raw_operands[0]) + b" TJ")
    elif op.operator == '"' and len(op.raw_operands) >= 3:
        parts.append(_serialize_pdf_operand(op.raw_operands[2]) + b" Tj")
    elif op.raw_operands:
        parts.append(_serialize_pdf_operand(op.raw_operands[0]) + b" Tj")

    parts.append(b"ET")
    parts.append(b"Q")
    return b"\n".join(parts)


def _emit_rule_bytes(rule: VectorRule, logical_to_stream: Matrix) -> bytes:
    x0, y0, x1, y1 = rule.bbox
    corners = [
        mat_apply(logical_to_stream, x0, y0),
        mat_apply(logical_to_stream, x1, y0),
        mat_apply(logical_to_stream, x1, y1),
        mat_apply(logical_to_stream, x0, y1),
    ]
    sx0 = min(c[0] for c in corners)
    sy0 = min(c[1] for c in corners)
    sx1 = max(c[0] for c in corners)
    sy1 = max(c[1] for c in corners)
    w = max(0.3, sx1 - sx0)
    h = max(0.3, sy1 - sy0)
    r, g, b = rule.color_rgb
    return (
        f"q {_fmt(r)} {_fmt(g)} {_fmt(b)} rg {_fmt(sx0)} {_fmt(sy0)} {_fmt(w)} {_fmt(h)} re f Q"
    ).encode("ascii")


def build_formula_xobject(
    pdf: pikepdf.Pdf,
    source_page: pikepdf.Page,
    ops: Sequence[StreamTextOp],
    *,
    tag_id: str,
    resource_name: str,
    prose_baseline_y: float,
    stream_to_logical: Matrix,
    logical_to_stream: Matrix,
    rules: Sequence[VectorRule] = (),
) -> InlineFormulaBox:
    """Slice ``ops`` (and optional fraction/radical ``rules``) into a Form XObject."""
    if not ops:
        raise ValueError("Cannot build InlineFormulaBox from empty op sequence")

    x0 = min(op.bbox[0] for op in ops)
    y0 = min(op.bbox[1] for op in ops)
    x1 = max(op.bbox[2] for op in ops)
    y1 = max(op.bbox[3] for op in ops)
    for rule in rules:
        x0 = min(x0, rule.bbox[0])
        y0 = min(y0, rule.bbox[1])
        x1 = max(x1, rule.bbox[2])
        y1 = max(y1, rule.bbox[3])

    pad = 0.6
    lx0, ly0, lx1, ly1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
    width = max(1.0, lx1 - lx0)
    height = max(1.0, ly1 - ly0)

    # Stream-space clipping BBox mapped back from logical box with safety margin.
    s_corners = [
        mat_apply(logical_to_stream, lx0 - 1.0, ly0 - 1.0),
        mat_apply(logical_to_stream, lx1 + 1.0, ly0 - 1.0),
        mat_apply(logical_to_stream, lx1 + 1.0, ly1 + 1.0),
        mat_apply(logical_to_stream, lx0 - 1.0, ly1 + 1.0),
    ]
    sbx0 = min(c[0] for c in s_corners)
    sby0 = min(c[1] for c in s_corners)
    sbx1 = max(c[0] for c in s_corners)
    sby1 = max(c[1] for c in s_corners)

    # Form /Matrix maps stream coordinates -> logical coordinates -> origin (0, 0).
    form_matrix = mat_mul(stream_to_logical, (1.0, 0.0, 0.0, 1.0, -lx0, -ly0))

    stream_chunks = [_emit_text_op_bytes(op) for op in ops]
    for rule in rules:
        stream_chunks.append(_emit_rule_bytes(rule, logical_to_stream))
    stream_bytes = b"\n".join(stream_chunks)

    # Copy ONLY the font references actually used by these ops.
    used_fonts = {op.font_resource for op in ops if op.font_resource}
    src_res = source_page.get("/Resources")
    src_fonts = src_res.get("/Font") if isinstance(src_res, pikepdf.Dictionary) else None
    form_fonts = pikepdf.Dictionary()
    if isinstance(src_fonts, pikepdf.Dictionary):
        for fname in sorted(used_fonts):
            key = fname if fname.startswith("/") else "/" + fname
            if key in src_fonts:
                form_fonts[key] = src_fonts[key]

    xobj = pikepdf.Stream(pdf, stream_bytes)
    xobj["/Type"] = pikepdf.Name("/XObject")
    xobj["/Subtype"] = pikepdf.Name("/Form")
    xobj["/FormType"] = 1
    xobj["/BBox"] = pikepdf.Array([sbx0, sby0, sbx1, sby1])
    xobj["/Matrix"] = pikepdf.Array(list(form_matrix))
    xobj["/Resources"] = pikepdf.Dictionary({"/Font": form_fonts})

    res_clean = resource_name if resource_name.startswith("/") else "/" + resource_name
    baseline_offset = prose_baseline_y - ly0
    ascent = max(0.0, ly1 - prose_baseline_y)
    descent = max(0.0, prose_baseline_y - ly0)

    return InlineFormulaBox(
        tag_id=tag_id,
        resource_name=res_clean,
        width_pt=width,
        height_pt=height,
        ascent_pt=ascent,
        descent_pt=descent,
        baseline_offset_pt=baseline_offset,
        op_indices=tuple(op.op_index for op in ops),
        rule_op_indices=tuple(idx for r in rules for idx in r.op_indices),
        bbox=(lx0, ly0, lx1, ly1),
        xobject=xobj,
    )
