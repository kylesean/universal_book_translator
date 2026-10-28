"""Styling engine for multi-page repeated table headers and continuation notices."""

from __future__ import annotations

import re

from ubt.core.tables.models import TableContinuationConfig


def sanitize_table_uid(raw_id: str) -> str:
    """Sanitize raw block ID into a valid Typst label identifier."""
    clean = re.sub(r"[^A-Za-z0-9_-]+", "-", str(raw_id)).strip("-")
    return clean or "tbl"


def generate_typst_table_continuation_elements(
    columns: int,
    table_uid: str,
    lang: str = "zh",
    config: TableContinuationConfig | None = None,
    indent: str = "    ",
) -> tuple[str, str, str, str]:
    """Generate the four Typst primitives for multi-page table continuation.

    Returns:
        (start_anchor, header_continuation_cell, footer_continuation_block, end_anchor)
    """
    cfg = config or TableContinuationConfig()
    uid = sanitize_table_uid(table_uid)
    cols = max(1, columns)

    header_text = cfg.get_continued_header_text(lang)
    footer_text = cfg.get_next_page_footer_text(lang)

    start_anchor = f'[#metadata("table-start") <ubt-tbl-start-{uid}>]'
    end_anchor = f'[#metadata("table-end") <ubt-tbl-end-{uid}>]'

    header_cell = ""
    if cfg.enable_continued_header:
        header_cell = (
            f"{indent}table.cell(colspan: {cols})[\n"
            f"{indent}  #context {{\n"
            f"{indent}    let matches = query(selector(<ubt-tbl-start-{uid}>).before(here()))\n"
            f"{indent}    if matches.len() > 0 {{\n"
            f"{indent}      let el = matches.last()\n"
            f"{indent}      if here().page() > el.location().page() [\n"
            f'{indent}        #align(right)[#text(size: {cfg.font_size_pt}pt, fill: luma({cfg.text_color_luma}), style: "italic")[{header_text}]]\n'
            f"{indent}      ]\n"
            f"{indent}    }}\n"
            f"{indent}  }}\n"
            f"{indent}],"
        )

    footer_block = ""
    if cfg.enable_next_page_footer:
        footer_block = (
            f"{indent}table.footer(\n"
            f"{indent}  repeat: true,\n"
            f"{indent}  table.cell(colspan: {cols})[\n"
            f"{indent}    #context {{\n"
            f"{indent}      let matches = query(selector(<ubt-tbl-end-{uid}>).after(here()))\n"
            f"{indent}      if matches.len() > 0 {{\n"
            f"{indent}        let el = matches.first()\n"
            f"{indent}        if here().page() < el.location().page() [\n"
            f'{indent}          #align(right)[#text(size: {cfg.font_size_pt}pt, fill: luma({cfg.text_color_luma}), style: "italic")[{footer_text}]]\n'
            f"{indent}        ]\n"
            f"{indent}      }}\n"
            f"{indent}    }}\n"
            f"{indent}  ],\n"
            f"{indent}),"
        )

    return start_anchor, header_cell, footer_block, end_anchor


def generate_html_table_continuation_css() -> str:
    """Generate CSS rules ensuring repeated headers and continuation styling across printed pages."""
    return (
        "<style>\n"
        "@media print {\n"
        "  table.ubt-table {\n"
        "    break-inside: auto;\n"
        "  }\n"
        "  table.ubt-table thead {\n"
        "    display: table-header-group;\n"
        "  }\n"
        "  table.ubt-table tfoot {\n"
        "    display: table-footer-group;\n"
        "  }\n"
        "  .ubt-table-continued-notice {\n"
        "    font-size: 0.8em;\n"
        "    color: #666;\n"
        "    font-style: italic;\n"
        "    text-align: right;\n"
        "  }\n"
        "}\n"
        "</style>"
    )
