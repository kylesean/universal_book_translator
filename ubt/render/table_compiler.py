"""Table complexity assessment and Typst table compiler.

Determines whether a table block can be typeset in-place via Typst micro-grid
or should be deferred to a dedicated high-fidelity translated table appendix.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ubt.adapters.pdf.overlay_text import typst_escape


@dataclass(frozen=True, slots=True)
class TableComplexity:
    """Assessment of table geometry and structural complexity."""

    is_complex: bool
    reason: str
    rows: int
    cols: int


def parse_markdown_table(markup: str) -> tuple[list[str], list[list[str]]] | None:
    """Parse a Markdown table into (headers, rows).

    Returns None if markup does not look like a recognizable Markdown table.
    """
    lines = [line.strip() for line in markup.strip().splitlines() if line.strip()]
    table_lines = [line for line in lines if line.startswith("|") and line.endswith("|")]
    if len(table_lines) < 2:
        return None

    # Check separator row
    header_line = table_lines[0]
    sep_idx = -1
    for idx in range(1, min(3, len(table_lines))):
        cells = [c.strip() for c in table_lines[idx].split("|")[1:-1]]
        if cells and all(re.match(r"^:?-+:?$", c) for c in cells):
            sep_idx = idx
            break

    if sep_idx == -1:
        # No standard separator line found
        return None

    headers = [typst_escape(c.strip()) for c in header_line.split("|")[1:-1]]
    cols = len(headers)
    if cols == 0:
        return None

    rows: list[list[str]] = []
    for line in table_lines[sep_idx + 1 :]:
        cells = [typst_escape(c.strip()) for c in line.split("|")[1:-1]]
        if len(cells) < cols:
            cells.extend([""] * (cols - len(cells)))
        elif len(cells) > cols:
            cells = cells[:cols]
        rows.append(cells)

    return headers, rows


def probe_table_complexity(markup: str, bbox: Any | None = None) -> TableComplexity:
    """Assess whether a table can be safely overlaid in-place without clipping or overflow."""
    text_lower = markup.lower()
    # 1. Merged cells or complex HTML markers
    if (
        "colspan" in text_lower
        or "rowspan" in text_lower
        or r"\multicolumn" in markup
        or r"\multirow" in markup
    ):
        return TableComplexity(is_complex=True, reason="merged_cells", rows=0, cols=0)

    parsed = parse_markdown_table(markup)
    if parsed is None:
        return TableComplexity(is_complex=True, reason="non_standard_grid", rows=0, cols=0)

    headers, rows = parsed
    cols = len(headers)
    total_rows = len(rows) + 1  # Including header

    # 2. Too many columns
    if cols >= 6:
        return TableComplexity(
            is_complex=True, reason=f"too_many_columns_{cols}", rows=total_rows, cols=cols
        )

    # 3. Geometry density check
    if bbox is not None:
        height = getattr(bbox, "height", None)
        if height is None and hasattr(bbox, "y1") and hasattr(bbox, "y0"):
            height = bbox.y1 - bbox.y0
        width = getattr(bbox, "width", None)
        if width is None and hasattr(bbox, "x1") and hasattr(bbox, "x0"):
            width = bbox.x1 - bbox.x0

        if height is not None and height > 0 and total_rows > 0:
            row_height = height / total_rows
            if row_height < 12.0:
                return TableComplexity(
                    is_complex=True,
                    reason=f"dense_row_height_{row_height:.1f}pt",
                    rows=total_rows,
                    cols=cols,
                )

        if width is not None and 0 < width < 120.0 and cols >= 4:
            return TableComplexity(
                is_complex=True,
                reason=f"too_narrow_{width:.1f}pt",
                rows=total_rows,
                cols=cols,
            )

    # 4. Long prose in table cells
    all_cells = headers + [c for r in rows for c in r]
    if all_cells:
        avg_len = sum(len(c) for c in all_cells) / len(all_cells)
        if avg_len > 100 or any(len(c) > 250 for c in all_cells):
            return TableComplexity(
                is_complex=True, reason="prose_paragraphs_in_cells", rows=total_rows, cols=cols
            )

    return TableComplexity(is_complex=False, reason="simple_grid", rows=total_rows, cols=cols)


def markdown_table_to_typst(
    markup: str,
    *,
    width_pt: float | None = None,
    height_pt: float | None = None,
    font_size_pt: float = 7.5,
    font_line: str = "",
    stroke_color: str = "luma(120)",
    fill_header: bool = True,
) -> str | None:
    """Compile Markdown table markup into Typst code."""
    parsed = parse_markdown_table(markup)
    if parsed is None:
        return None

    headers, rows = parsed
    cols = len(headers)
    if cols == 0:
        return None

    fill_rule = "(col, row) => if row == 0 { luma(240) } else { none }" if fill_header else "none"
    col_spec = f"({cols} * (1fr,))"

    table_lines: list[str] = [
        "#table(",
        f"  columns: {col_spec},",
        "  inset: 2.2pt,",
        f"  stroke: 0.35pt + {stroke_color},",
        f"  fill: {fill_rule},",
        "  align: (col, row) => if row == 0 { center + horizon } else { left + horizon },",
    ]
    # Header cells
    for h in headers:
        table_lines.append(f'  [#text(weight: "bold")[{h}]],')
    # Body cells
    for r in rows:
        for c in r:
            table_lines.append(f"  [{c}],")
    table_lines.append(")")
    table_code = "\n".join(table_lines)

    if width_pt is not None and height_pt is not None:
        return (
            f"#set page(width: {width_pt}pt, height: {height_pt}pt, margin: 1pt)\n"
            f'#set text(size: {font_size_pt}pt{font_line}, top-edge: "ascender", bottom-edge: "descender")\n'
            f"#box(width: {width_pt}pt, height: {height_pt}pt, clip: true)[\n"
            f"  #align(center + horizon)[\n"
            f"    {table_code}\n"
            f"  ]\n"
            f"]\n"
        )

    return f"#set text(size: {font_size_pt}pt{font_line})\n{table_code}\n"


def render_table_appendix_typst(
    tables: list[tuple[int, str, str]],
    *,
    font_line: str = "",
) -> str:
    """Generate a complete standalone Typst document for translated table appendix pages."""
    typst_parts = [
        '#set page(paper: "a4", margin: (x: 2cm, y: 2.5cm))',
        f"#set text(size: 8.5pt{font_line})",
        '#align(center)[#text(size: 16pt, weight: "bold")[附录：核心图表与数据矩阵对照]]',
        "#v(0.6cm)",
        "#align(center)[#text(size: 9pt, fill: luma(100))[（本附录收录正文中因原始版面紧凑而保留原貌的全部复杂图表完整译文）]]",
        "#v(1cm)",
    ]

    for page_num, label, markup in tables:
        rendered = markdown_table_to_typst(
            markup,
            font_size_pt=8.0,
            font_line=font_line,
        )
        if rendered is None:
            # Fallback to plain monospace / text if table parsing failed
            rendered = f"```\n{markup}\n```"

        typst_parts.append(
            f"== 图表对照（原书第 {page_num} 页: {typst_escape(label)}）\n"
            f"#v(2mm)\n"
            f"{rendered}\n"
            f"#v(8mm)\n"
        )

    return "\n".join(typst_parts)


__all__ = [
    "TableComplexity",
    "markdown_table_to_typst",
    "parse_markdown_table",
    "probe_table_complexity",
    "render_table_appendix_typst",
]
