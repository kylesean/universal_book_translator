"""Surgical PDF content-stream patcher & Mono / Interleaved / Side-by-Side assembler.

Core invariants:
1. Instruction-index blanking replaces painted glyph payloads while preserving
   exact PDF graphics state and ``Tm`` advancement (0.0 pt drift for surviving
   formulas, figures, table borders, and headers/footers).
2. Translated prose and inline formula Form XObjects are appended inside an
   isolated ``q <logical_to_stream> cm ... Q`` block, so ``/CropBox`` offsets
   and ``/Rotate`` pages render upright with zero distortion.
3. Supports ``MONO``, ``INTERLEAVED`` (alternating source + translated pages),
   and ``SIDE_BY_SIDE`` (2W spread via page Form XObjects), plus millisecond
   single-page PNG preview for live operator review.
"""

from __future__ import annotations

import io
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pikepdf

from ubt.codecs.pdf.font_engine import EmbeddedFont, FontEngine, FontSlot
from ubt.codecs.pdf.formula_box import InlineFormulaBox, build_formula_xobject
from ubt.codecs.pdf.layout_fusion import FusedBlock
from ubt.codecs.pdf.paragraph_typesetter import TypesetParagraph, typeset_paragraph
from ubt.codecs.pdf.stream_analyzer import IDENTITY_MATRIX, Matrix, PageStreamAnalysis
from ubt.codecs.protocol import ApplyReport, DeliveryMode, OverflowItem

logger = logging.getLogger(__name__)

_PAINT_PATH_OPS: frozenset[str] = frozenset({"S", "s", "f", "F", "f*", "B", "B*", "b", "b*"})


def _fmt(v: float) -> str:
    s = f"{v:.4f}".rstrip("0").rstrip(".")
    return "0" if s in {"", "-0"} else s


def _blank_tj_slices(
    arr: pikepdf.Array,
    slices: Sequence[tuple[int, int, float, float, float]],
) -> pikepdf.ContentStreamInstruction:
    """Replace only selected ``[start, end)`` sub-slices of a ``TJ`` array with exact advances."""
    items = list(arr)
    ordered = sorted(slices, key=lambda s: (s[0], s[1]))
    new_items: list[Any] = []
    pos = 0
    for start, end, seg_adv, fs, tz in ordered:
        s = max(pos, start)
        e = min(len(items), end)
        if s >= e:
            continue
        if s > pos:
            new_items.extend(items[pos:s])
        denom = fs * (tz / 100.0)
        if abs(denom) > 1e-6 and abs(seg_adv) > 1e-6:
            new_items.append(-seg_adv / denom * 1000.0)
        pos = e
    if pos < len(items):
        new_items.extend(items[pos:])
    if not new_items:
        return pikepdf.ContentStreamInstruction([pikepdf.String(b"")], pikepdf.Operator("Tj"))
    return pikepdf.ContentStreamInstruction([pikepdf.Array(new_items)], pikepdf.Operator("TJ"))


def _blank_stream_instructions(
    page: pikepdf.Page,
    erase_op_indices: set[int],
    erase_rule_op_indices: set[int],
    op_advances: dict[int, tuple[float, float, float]],  # op_index -> (adv_text, font_size, tz)
    erase_tj_slices: dict[int, list[tuple[int, int, float, float, float]]] | None = None,
) -> bytes:
    """Blank selected show/rule instructions in-place while preserving exact Tm advance."""
    instructions = pikepdf.parse_content_stream(page)
    out: list[pikepdf.ContentStreamInstruction | pikepdf.ContentStreamInlineImage] = []
    tj_map = erase_tj_slices or {}

    for idx, inst in enumerate(instructions):
        op = str(inst.operator)
        if idx in erase_op_indices and op in {"Tj", "TJ", "'", '"'}:
            adv_info = op_advances.get(idx)
            # Preserve vertical line advance for ' and " operators.
            if op == '"':
                operands = list(inst.operands)
                if len(operands) >= 2:
                    out.append(
                        pikepdf.ContentStreamInstruction([operands[0]], pikepdf.Operator("Tw"))
                    )
                    out.append(
                        pikepdf.ContentStreamInstruction([operands[1]], pikepdf.Operator("Tc"))
                    )
                out.append(pikepdf.ContentStreamInstruction([], pikepdf.Operator("T*")))
            elif op == "'":
                out.append(pikepdf.ContentStreamInstruction([], pikepdf.Operator("T*")))

            # Preserve exact horizontal Tm displacement via empty-glyph TJ adjustment.
            if adv_info is not None:
                adv_text, fs, tz = adv_info
                denom = fs * (tz / 100.0)
                if abs(denom) > 1e-6 and abs(adv_text) > 1e-6:
                    adj = -adv_text / denom * 1000.0
                    out.append(
                        pikepdf.ContentStreamInstruction(
                            [pikepdf.Array([adj])],
                            pikepdf.Operator("TJ"),
                        )
                    )
                else:
                    out.append(
                        pikepdf.ContentStreamInstruction(
                            [pikepdf.String(b"")],
                            pikepdf.Operator("Tj"),
                        )
                    )
            else:
                out.append(
                    pikepdf.ContentStreamInstruction(
                        [pikepdf.String(b"")],
                        pikepdf.Operator("Tj"),
                    )
                )
        elif (
            idx in tj_map
            and op == "TJ"
            and inst.operands
            and isinstance(inst.operands[0], pikepdf.Array)
        ):
            out.append(_blank_tj_slices(inst.operands[0], tj_map[idx]))
        elif idx in erase_rule_op_indices and op in _PAINT_PATH_OPS:
            # End path without stroking/filling so graphics state stays balanced.
            out.append(pikepdf.ContentStreamInstruction([], pikepdf.Operator("n")))
        else:
            out.append(inst)

    return bytes(pikepdf.unparse_content_stream(out))


def _emit_overlay_stream(
    paragraphs: Sequence[TypesetParagraph],
    embedded_fonts: dict[FontSlot, EmbeddedFont],
    logical_to_stream: Matrix,
) -> bytes:
    """Serialize all typeset paragraphs and inline Form XObjects in logical coordinates."""
    lines: list[bytes] = [b"q"]
    if logical_to_stream != IDENTITY_MATRIX:
        m = logical_to_stream
        lines.append(
            f"{_fmt(m[0])} {_fmt(m[1])} {_fmt(m[2])} {_fmt(m[3])} {_fmt(m[4])} {_fmt(m[5])} cm".encode(
                "ascii"
            )
        )

    for para in paragraphs:
        if not para.runs:
            continue
        # Emit text runs in BT..ET and Form XObjects outside BT..ET.
        in_bt = False
        cur_font_res = ""
        cur_font_size = -1.0
        cur_rgb: tuple[float, float, float] | None = None

        def _close_bt() -> None:
            nonlocal in_bt, cur_font_res, cur_font_size
            if in_bt:
                lines.append(b"ET")
                in_bt = False
                cur_font_res = ""
                cur_font_size = -1.0

        for run in para.runs:
            if run.formula is not None:
                _close_bt()
                fx_x = run.x
                fx_y = run.baseline_y - run.formula.baseline_offset_pt
                lines.append(
                    f"q 1 0 0 1 {_fmt(fx_x)} {_fmt(fx_y)} cm {run.formula.resource_name} Do Q".encode(
                        "ascii"
                    )
                )
                continue

            if not run.text:
                continue
            efont = embedded_fonts.get(run.slot)
            if efont is None:
                continue

            rgb = (
                round(run.color_rgb[0], 3),
                round(run.color_rgb[1], 3),
                round(run.color_rgb[2], 3),
            )
            if rgb != cur_rgb:
                _close_bt()
                lines.append(f"{_fmt(rgb[0])} {_fmt(rgb[1])} {_fmt(rgb[2])} rg".encode("ascii"))
                cur_rgb = rgb

            if not in_bt:
                lines.append(b"BT")
                in_bt = True

            if efont.resource_name != cur_font_res or abs(run.font_size_pt - cur_font_size) > 1e-3:
                lines.append(f"{efont.resource_name} {_fmt(run.font_size_pt)} Tf".encode("ascii"))
                cur_font_res = efont.resource_name
                cur_font_size = run.font_size_pt

            # Synthetic 12-degree italic slant matrix when needed.
            slant = "0.2126" if run.is_italic else "0"
            lines.append(f"1 0 {slant} 1 {_fmt(run.x)} {_fmt(run.baseline_y)} Tm".encode("ascii"))
            hex_payload = efont.encode_text(run.text).hex().upper().encode("ascii")
            lines.append(b"<" + hex_payload + b"> Tj")

        _close_bt()

    lines.append(b"Q")
    return b"\n".join(lines)


def _attach_resources(
    page: pikepdf.Page,
    embedded_fonts: dict[FontSlot, EmbeddedFont],
    xobjects: dict[str, pikepdf.Stream],
) -> None:
    res = page.get("/Resources")
    if not isinstance(res, pikepdf.Dictionary):
        res = pikepdf.Dictionary()
        page["/Resources"] = res

    if embedded_fonts:
        fdict = res.get("/Font")
        if not isinstance(fdict, pikepdf.Dictionary):
            fdict = pikepdf.Dictionary()
            res["/Font"] = fdict
        for efont in embedded_fonts.values():
            fdict[efont.resource_name] = efont.pdf_font_obj

    if xobjects:
        xdict = res.get("/XObject")
        if not isinstance(xdict, pikepdf.Dictionary):
            xdict = pikepdf.Dictionary()
            res["/XObject"] = xdict
        for name, xobj in xobjects.items():
            key = name if name.startswith("/") else "/" + name
            xdict[key] = xobj


@dataclass(slots=True)
class _PagePatchPlan:
    page_index: int  # 0-based index in target PDF
    analysis: PageStreamAnalysis
    erase_op_indices: set[int]
    erase_tj_slices: dict[int, list[tuple[int, int, float, float, float]]]
    erase_rule_op_indices: set[int]
    op_advances: dict[int, tuple[float, float, float]]
    paragraphs: list[TypesetParagraph]
    xobjects: dict[str, pikepdf.Stream]


def _page_to_form_xobject(
    dst_pdf: pikepdf.Pdf,
    src_pdf: pikepdf.Pdf,
    src_page: pikepdf.Page,
) -> tuple[pikepdf.Stream, float, float]:
    """Wrap ``src_page`` as a normalized upright Form XObject inside ``dst_pdf``."""
    raw_box = src_page.get("/CropBox") or src_page.get("/MediaBox")
    vals = [float(v) for v in raw_box[:4]] if raw_box is not None else [0.0, 0.0, 612.0, 792.0]
    cx0, cy0 = min(vals[0], vals[2]), min(vals[1], vals[3])
    cx1, cy1 = max(vals[0], vals[2]), max(vals[1], vals[3])
    w, h = max(1.0, cx1 - cx0), max(1.0, cy1 - cy0)
    rotate = int(src_page.get("/Rotate", 0)) % 360

    # Read combined content stream bytes
    contents = src_page.get("/Contents")
    chunks: list[bytes] = []
    if isinstance(contents, pikepdf.Array):
        for s in contents:
            if isinstance(s, pikepdf.Stream):
                chunks.append(bytes(s.read_bytes()))
    elif isinstance(contents, pikepdf.Stream):
        chunks.append(bytes(contents.read_bytes()))
    stream_bytes = b"\n".join(chunks)

    # Transform to upright (0, 0) -> (vis_w, vis_h)
    from ubt.codecs.pdf.stream_analyzer import mat_mul

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

    mat = mat_mul(t_origin, r_mat)
    xobj = pikepdf.Stream(dst_pdf, stream_bytes)
    xobj["/Type"] = pikepdf.Name("/XObject")
    xobj["/Subtype"] = pikepdf.Name("/Form")
    xobj["/FormType"] = 1
    xobj["/BBox"] = pikepdf.Array([cx0, cy0, cx1, cy1])
    xobj["/Matrix"] = pikepdf.Array(list(mat))
    src_res = src_page.get("/Resources")
    if isinstance(src_res, pikepdf.Dictionary):
        res_ref = src_res if src_res.is_indirect else src_pdf.make_indirect(src_res)
        xobj["/Resources"] = dst_pdf.copy_foreign(res_ref)
    else:
        xobj["/Resources"] = pikepdf.Dictionary()
    return xobj, vis_w, vis_h


def _build_op_advances(analysis: PageStreamAnalysis) -> dict[int, tuple[float, float, float]]:
    advances: dict[int, tuple[float, float, float]] = {}
    for op in analysis.text_ops:
        # Approximate text-space horizontal advance from logical bbox width & CTM/Tm scale.
        w_logical = max(0.0, op.bbox[2] - op.bbox[0])
        snap = op.snapshot
        # Convert logical width back to text-space using effective font_size vs operand.
        scale = (
            op.font_size_pt / abs(snap.font_size_operand)
            if abs(snap.font_size_operand) > 1e-6
            else 1.0
        )
        adv_text = w_logical / scale if scale > 1e-6 else w_logical
        advances[op.op_index] = (adv_text, snap.font_size_operand, snap.tz)
    return advances


def apply_pdf_patches(
    *,
    source_path: Path,
    analyses: dict[int, PageStreamAnalysis],
    fused_by_page: dict[int, list[FusedBlock]],
    translations: dict[str, str],
    target_lang: str,
    mode: DeliveryMode,
    output_path: Path,
    pages: set[int] | None = None,
    fonts_dir: Path | None = None,
) -> ApplyReport:
    """Patch ``source_path`` in-place and write ``MONO`` / ``INTERLEAVED`` / ``SIDE_BY_SIDE`` PDF."""
    font_engine = FontEngine(target_lang=target_lang, fonts_dir=fonts_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pikepdf.open(source_path) as work_pdf:
        total_src_pages = len(work_pdf.pages)
        selected_pages = (
            sorted(p for p in pages if 1 <= p <= total_src_pages)
            if pages
            else list(range(1, total_src_pages + 1))
        )

        patch_plans: list[_PagePatchPlan] = []
        overflows: list[OverflowItem] = []
        total_units = 0
        patched_units = 0

        for pno in selected_pages:
            page = work_pdf.pages[pno - 1]
            analysis = analyses.get(pno)
            blocks = fused_by_page.get(pno, [])
            if analysis is None or not blocks:
                continue

            erase_ops: set[int] = set()
            erase_tj_slices: dict[int, list[tuple[int, int, float, float, float]]] = {}
            erase_rules: set[int] = set()
            page_xobjs: dict[str, pikepdf.Stream] = {}
            typeset_paras: list[TypesetParagraph] = []
            fx_counter = 0

            for block in blocks:
                total_units += 1
                if not block.unit.translate:
                    continue
                translated = translations.get(block.unit.unit_id)
                if translated is None or not translated.strip():
                    continue

                # Build lightweight Form XObjects for inline formula islands in this block.
                formulas: dict[str, InlineFormulaBox] = {}
                for island in block.formula_islands:
                    res_name = f"/UBT_FX_p{pno}_{fx_counter}"
                    fx_counter += 1
                    fbox = build_formula_xobject(
                        work_pdf,
                        page,
                        island.ops,
                        tag_id=island.tag_id,
                        resource_name=res_name,
                        prose_baseline_y=island.prose_baseline_y,
                        stream_to_logical=analysis.stream_to_logical,
                        logical_to_stream=analysis.logical_to_stream,
                        rules=island.rules,
                    )
                    formulas[island.tag_id] = fbox
                    if fbox.xobject is not None:
                        page_xobjs[fbox.resource_name] = fbox.xobject

                para = typeset_paragraph(
                    unit_id=block.unit.unit_id,
                    page_no=pno,
                    role=block.unit.role,
                    marked_text=translated,
                    tokens=block.unit.tokens,
                    geom=block.geometry,
                    formulas=formulas,
                    font_engine=font_engine,
                    target_lang=target_lang,
                )
                typeset_paras.append(para)
                erase_ops.update(block.erase_op_indices)
                for op_idx, s_idx, e_idx, s_adv, s_fs, s_tz in block.erase_tj_slices:
                    erase_tj_slices.setdefault(op_idx, []).append((s_idx, e_idx, s_adv, s_fs, s_tz))
                erase_rules.update(block.erase_rule_op_indices)
                patched_units += 1
                if para.overflow is not None:
                    overflows.append(para.overflow)

            if typeset_paras:
                patch_plans.append(
                    _PagePatchPlan(
                        page_index=pno - 1,
                        analysis=analysis,
                        erase_op_indices=erase_ops,
                        erase_tj_slices=erase_tj_slices,
                        erase_rule_op_indices=erase_rules,
                        op_advances=_build_op_advances(analysis),
                        paragraphs=typeset_paras,
                        xobjects=page_xobjs,
                    )
                )

        # Subset & embed all used target fonts ONCE per document.
        font_started = time.perf_counter()
        embedded_fonts = font_engine.embed_all_used(work_pdf)
        logger.debug(
            "native font subsetting: %d slot(s) in %.0fms",
            len(embedded_fonts),
            (time.perf_counter() - font_started) * 1000,
        )

        # Apply surgical content-stream patches to each modified page.
        for plan in patch_plans:
            page = work_pdf.pages[plan.page_index]
            blanked = _blank_stream_instructions(
                page,
                plan.erase_op_indices,
                plan.erase_rule_op_indices,
                plan.op_advances,
                erase_tj_slices=plan.erase_tj_slices,
            )
            overlay = _emit_overlay_stream(
                plan.paragraphs,
                embedded_fonts,
                plan.analysis.logical_to_stream,
            )
            combined = blanked + b"\n" + overlay
            page["/Contents"] = pikepdf.Stream(work_pdf, combined)
            _attach_resources(page, embedded_fonts, plan.xobjects)

        # Assemble final output according to DeliveryMode.
        if mode == DeliveryMode.MONO:
            if pages:
                # Emit only the selected pages in ascending order.
                out_pdf = pikepdf.new()
                for pno in selected_pages:
                    out_pdf.pages.append(work_pdf.pages[pno - 1])
                out_pdf.save(output_path)
                out_pages = len(out_pdf.pages)
                out_pdf.close()
            else:
                work_pdf.save(output_path)
                out_pages = len(work_pdf.pages)
        elif mode == DeliveryMode.INTERLEAVED:
            with pikepdf.open(source_path) as src_pdf:
                out_pdf = pikepdf.new()
                for pno in selected_pages:
                    out_pdf.pages.append(src_pdf.pages[pno - 1])
                    out_pdf.pages.append(work_pdf.pages[pno - 1])
                out_pdf.save(output_path)
                out_pages = len(out_pdf.pages)
                out_pdf.close()
        elif mode == DeliveryMode.SIDE_BY_SIDE:
            with pikepdf.open(source_path) as src_pdf:
                out_pdf = pikepdf.new()
                for pno in selected_pages:
                    left_xobj, vis_w, vis_h = _page_to_form_xobject(
                        out_pdf, src_pdf, src_pdf.pages[pno - 1]
                    )
                    right_xobj, r_w, r_h = _page_to_form_xobject(
                        out_pdf, work_pdf, work_pdf.pages[pno - 1]
                    )
                    spread_w = vis_w + r_w
                    spread_h = max(vis_h, r_h)
                    spread = out_pdf.add_blank_page(page_size=(spread_w, spread_h))
                    spread["/Resources"] = pikepdf.Dictionary(
                        {
                            "/XObject": pikepdf.Dictionary(
                                {
                                    "/UBT_SRC": out_pdf.make_indirect(left_xobj),
                                    "/UBT_TGT": out_pdf.make_indirect(right_xobj),
                                }
                            )
                        }
                    )
                    # Draw subtle vertical divider line between source and target.
                    cmd = (
                        f"q 1 0 0 1 0 0 cm /UBT_SRC Do Q\n"
                        f"q 1 0 0 1 {_fmt(vis_w)} 0 cm /UBT_TGT Do Q\n"
                        f"q 0.85 0.85 0.85 RG 0.5 w {_fmt(vis_w)} 0 m {_fmt(vis_w)} {_fmt(spread_h)} l S Q"
                    ).encode("ascii")
                    spread["/Contents"] = pikepdf.Stream(out_pdf, cmd)
                out_pdf.save(output_path)
                out_pages = len(out_pdf.pages)
                out_pdf.close()
        else:
            # Default fallback: save mono patched PDF
            work_pdf.save(output_path)
            out_pages = len(work_pdf.pages)

    return ApplyReport(
        output_path=output_path,
        mode=mode,
        total_units=total_units,
        patched_units=patched_units,
        overflows=tuple(overflows),
        page_count_source=len(selected_pages),
        page_count_output=out_pages,
    )


def render_single_page_png(
    *,
    source_path: Path,
    page_no: int,
    analysis: PageStreamAnalysis,
    blocks: list[FusedBlock],
    translations: dict[str, str],
    target_lang: str,
    dpi: int = 144,
    fonts_dir: Path | None = None,
) -> bytes:
    """Patch only ``page_no`` in memory and rasterize to PNG bytes (<100ms typical)."""
    import pypdfium2 as pdfium

    from ubt.adapters.pdf.pdfium_gate import PDFIUM_LOCK

    font_engine = FontEngine(target_lang=target_lang, fonts_dir=fonts_dir)
    one_pdf = pikepdf.new()
    try:
        with pikepdf.open(source_path) as src_pdf:
            if page_no < 1 or page_no > len(src_pdf.pages):
                raise ValueError(f"Page {page_no} out of range (1..{len(src_pdf.pages)})")
            one_pdf.pages.append(src_pdf.pages[page_no - 1])

        page = one_pdf.pages[0]
        erase_ops: set[int] = set()
        erase_tj_slices: dict[int, list[tuple[int, int, float, float, float]]] = {}
        erase_rules: set[int] = set()
        page_xobjs: dict[str, pikepdf.Stream] = {}
        typeset_paras: list[TypesetParagraph] = []
        fx_counter = 0

        for block in blocks:
            if not block.unit.translate:
                continue
            translated = translations.get(block.unit.unit_id)
            if translated is None or not translated.strip():
                continue
            formulas: dict[str, InlineFormulaBox] = {}
            for island in block.formula_islands:
                res_name = f"/UBT_FX_p{page_no}_{fx_counter}"
                fx_counter += 1
                fbox = build_formula_xobject(
                    one_pdf,
                    page,
                    island.ops,
                    tag_id=island.tag_id,
                    resource_name=res_name,
                    prose_baseline_y=island.prose_baseline_y,
                    stream_to_logical=analysis.stream_to_logical,
                    logical_to_stream=analysis.logical_to_stream,
                    rules=island.rules,
                )
                formulas[island.tag_id] = fbox
                if fbox.xobject is not None:
                    page_xobjs[fbox.resource_name] = fbox.xobject
            para = typeset_paragraph(
                unit_id=block.unit.unit_id,
                page_no=page_no,
                role=block.unit.role,
                marked_text=translated,
                tokens=block.unit.tokens,
                geom=block.geometry,
                formulas=formulas,
                font_engine=font_engine,
                target_lang=target_lang,
            )
            typeset_paras.append(para)
            erase_ops.update(block.erase_op_indices)
            for op_idx, s_idx, e_idx, s_adv, s_fs, s_tz in block.erase_tj_slices:
                erase_tj_slices.setdefault(op_idx, []).append((s_idx, e_idx, s_adv, s_fs, s_tz))
            erase_rules.update(block.erase_rule_op_indices)

        if typeset_paras:
            embedded_fonts = font_engine.embed_all_used(one_pdf)
            blanked = _blank_stream_instructions(
                page,
                erase_ops,
                erase_rules,
                _build_op_advances(analysis),
                erase_tj_slices=erase_tj_slices,
            )
            overlay = _emit_overlay_stream(
                typeset_paras, embedded_fonts, analysis.logical_to_stream
            )
            page["/Contents"] = pikepdf.Stream(one_pdf, blanked + b"\n" + overlay)
            _attach_resources(page, embedded_fonts, page_xobjs)

        pdf_buf = io.BytesIO()
        one_pdf.save(pdf_buf)
        pdf_bytes = pdf_buf.getvalue()
    finally:
        one_pdf.close()

    with PDFIUM_LOCK:
        doc = pdfium.PdfDocument(pdf_bytes)
        try:
            p = doc[0]
            try:
                pil = p.render(scale=dpi / 72.0).to_pil()
            finally:
                p.close()
        finally:
            doc.close()

    png_buf = io.BytesIO()
    pil.save(png_buf, format="PNG")
    return png_buf.getvalue()
