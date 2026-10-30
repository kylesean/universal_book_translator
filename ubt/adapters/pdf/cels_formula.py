"""CELS Stage 5 Formula Follower: InlineRigidBox & Form XObject tracking.

Design: ``docs/design/adaptive_delivery_prd.md`` §7.4.5 & §8.2. Resolves GAP-14.

When technical documents are translated (e.g. EN -> ZH), the sentence structure
and word order change, causing inline formulas to move to new semantic positions.
Rather than redrawing math formulas (which mangles fonts and symbols) or leaving
them at English positions (which causes catastrophic text collisions), this module:
1. Wraps source inline formulas into isolated PDF Form XObjects with normalized
   /Matrix [1 0 0 1 -x0 -y0] and /BBox [x0 y0 x1 y1].
2. Tracks cursor advance across mixed text runs and rigid formula boxes.
3. Injects translated text and invokes the Form XObject at the translated position:
   ``q 1 0 0 1 x_cursor y_baseline cm /<xobject_ref> Do Q``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pikepdf

from ubt.adapters.pdf.cels_font import CidFont, FontPipelineError
from ubt.adapters.pdf.cels_inject import LineSite
from ubt.adapters.pdf.strip_pdf_text import cast_instructions
from ubt.core.layout import Box, InlineRigidBox

__all__ = [
    "create_formula_form_xobject",
    "emit_line_with_rigids",
    "inject_mixed_cid_line",
]


def create_formula_form_xobject(
    pdf: pikepdf.Pdf,
    source_page: pikepdf.Page,
    formula_box: Box,
    *,
    token_id: str,
    resource_name: str,
    baseline_offset_pt: float = 0.0,
) -> InlineRigidBox:
    """Create an isolated Form XObject from a bounding box on source_page.

    The Form XObject encapsulates the source page contents clipped to formula_box,
    with /Matrix [1 0 0 1 -x0 -y0] so that invoking it with CTM [1 0 0 1 x y cm]
    places the formula's bottom-left corner at (x, y).

    Returns the corresponding InlineRigidBox.
    """
    if formula_box.width <= 0 or formula_box.height <= 0:
        raise ValueError(f"formula_box has non-positive dimensions: {formula_box}")

    res_clean = resource_name if resource_name.startswith("/") else f"/{resource_name}"

    # Extract source page stream bytes
    contents_obj = source_page.get("/Contents")
    if contents_obj is None:
        raise ValueError("source_page has no /Contents")

    if isinstance(contents_obj, pikepdf.Array):
        stream_data = b"".join(bytes(stream.read_raw_bytes()) for stream in contents_obj)
    else:
        stream_data = bytes(contents_obj.read_raw_bytes())

    # Create Form XObject stream
    form = pdf.make_stream(stream_data)
    form.Type = pikepdf.Name("/XObject")
    form.Subtype = pikepdf.Name("/Form")
    form.BBox = pikepdf.Array(
        [
            formula_box.x0,
            formula_box.y0,
            formula_box.x1,
            formula_box.y1,
        ]
    )
    form.Matrix = pikepdf.Array(
        [
            1.0,
            0.0,
            0.0,
            1.0,
            -formula_box.x0,
            -formula_box.y0,
        ]
    )

    # Share source resources (fonts, graphic states, colorspaces)
    if hasattr(source_page, "Resources") and source_page.Resources is not None:
        form.Resources = source_page.Resources

    # Register into source page's /Resources /XObject dictionary
    if not hasattr(source_page, "Resources") or source_page.Resources is None:
        source_page.Resources = pdf.make_indirect(pikepdf.Dictionary())
    if "/XObject" not in source_page.Resources:
        source_page.Resources["/XObject"] = pdf.make_indirect(pikepdf.Dictionary())

    source_page.Resources["/XObject"][res_clean] = form

    return InlineRigidBox(
        token_id=token_id,
        width_pt=round(formula_box.width, 3),
        height_pt=round(formula_box.height, 3),
        baseline_offset_pt=round(baseline_offset_pt, 3),
        xobject_ref=res_clean,
    )


def emit_line_with_rigids(
    line: LineSite,
    segments: Sequence[str | InlineRigidBox],
    *,
    font: CidFont,
    size_pt: float,
    horizontal_scale: float = 1.0,
    char_spacing: float = 0.0,
) -> list[Any]:
    """Emit content stream instructions for a line containing text and InlineRigidBoxes."""
    scale = line.text_scale if line.text_scale > 0.0 else 1.0
    previous_size = round(line.font_size / scale, 4) if line.font_size else 1.0

    ops: list[Any] = []
    cursor_x = line.x0
    in_text_object = True

    # Initial text object setup
    ops.append(([round(horizontal_scale * 100.0, 4)], pikepdf.Operator("Tz")))
    ops.append(([round(char_spacing / scale, 4)], pikepdf.Operator("Tc")))
    ops.append(([pikepdf.Name(font.resource), round(size_pt / scale, 4)], pikepdf.Operator("Tf")))

    for seg in segments:
        if isinstance(seg, str):
            if not seg:
                continue
            if not font.supports(seg):
                raise FontPipelineError(f"font cannot encode {font.missing(seg)!r}")

            if not in_text_object:
                ops.append(([], pikepdf.Operator("BT")))
                ops.append(([round(horizontal_scale * 100.0, 4)], pikepdf.Operator("Tz")))
                ops.append(([round(char_spacing / scale, 4)], pikepdf.Operator("Tc")))
                ops.append(
                    (
                        [pikepdf.Name(font.resource), round(size_pt / scale, 4)],
                        pikepdf.Operator("Tf"),
                    )
                )
                ops.append(
                    (
                        [1.0, 0.0, 0.0, 1.0, round(cursor_x, 4), round(line.baseline, 4)],
                        pikepdf.Operator("Tm"),
                    )
                )
                in_text_object = True

            encoded = font.encode(seg)
            ops.append(([pikepdf.String(encoded)], pikepdf.Operator("Tj")))
            seg_advance = font.advance_pt(seg, size_pt) * horizontal_scale + char_spacing * len(seg)
            cursor_x += seg_advance

        elif isinstance(seg, InlineRigidBox):
            if in_text_object:
                ops.append(([], pikepdf.Operator("ET")))
                in_text_object = False

            # Invoke Form XObject positioned at cursor
            ops.append(([], pikepdf.Operator("q")))
            ops.append(
                (
                    [
                        1.0,
                        0.0,
                        0.0,
                        1.0,
                        round(cursor_x, 4),
                        round(line.baseline + seg.baseline_offset_pt, 4),
                    ],
                    pikepdf.Operator("cm"),
                )
            )
            ops.append(([pikepdf.Name(seg.xobject_ref)], pikepdf.Operator("Do")))
            ops.append(([], pikepdf.Operator("Q")))

            cursor_x += seg.width_pt

    # Restore page's original font and state
    if not in_text_object:
        ops.append(([], pikepdf.Operator("BT")))
    ops.append(([pikepdf.Name(line.font_key), previous_size], pikepdf.Operator("Tf")))
    ops.append(([round(line.horizontal_scale * 100.0, 4)], pikepdf.Operator("Tz")))
    ops.append(([round(line.char_spacing, 4)], pikepdf.Operator("Tc")))

    return ops


def inject_mixed_cid_line(
    page: pikepdf.Page,
    line: LineSite,
    segments: Sequence[str | InlineRigidBox],
    *,
    font: CidFont,
    size_pt: float,
    horizontal_scale: float = 1.0,
    char_spacing: float = 0.0,
    pdf: pikepdf.Pdf | None = None,
) -> bytes:
    """Replace line's glyph run with mixed CJK text and InlineRigidBox Form XObjects."""
    instructions = cast_instructions(pikepdf.parse_content_stream(page))
    replacement = emit_line_with_rigids(
        line,
        segments,
        font=font,
        size_pt=size_pt,
        horizontal_scale=horizontal_scale,
        char_spacing=char_spacing,
    )
    merged = instructions[: line.first_index] + replacement + instructions[line.last_index + 1 :]
    unparsed = pikepdf.unparse_content_stream(merged)
    if pdf is not None:
        page.Contents = pdf.make_stream(unparsed)
    return unparsed
