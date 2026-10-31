"""Typst/HTML continuation markup for tables that break across pages.

A continuation notice can only be placed by comparing the page it is drawn on
against the page where the table started (or ends), so the Typst markup carries
``<ubt-tbl-start-…>`` / ``<ubt-tbl-end-…>`` metadata labels and finds them with
``query()`` at compile time. ``sanitize_table_uid`` restricts the label suffix to
``[A-Za-z0-9_-]`` because a Typst label is an *identifier*: any other character
is a parse error in the generated source, not a cosmetic difference.
"""

from __future__ import annotations

import re

from ubt.core.tables.models import TableContinuationConfig


def sanitize_table_uid(raw_id: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_-]+", "-", str(raw_id)).strip("-")
    return clean or "tbl"


def generate_typst_table_continuation_elements(
    columns: int,
    table_uid: str,
    lang: str = "zh",
    config: TableContinuationConfig | None = None,
    indent: str = "    ",
) -> tuple[str, str, str, str]:
    """(start_anchor, header_cell, footer_block, end_anchor), in source order.

    The header cell and footer block come back ``""`` when their config switch
    is off, so four returned strings do not mean four things to emit.
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
    """Print CSS: ``*-header-group``/``*-footer-group`` repeat thead/tfoot per page.

    Needs ``break-inside: auto`` on the table itself, or the browser moves the
    whole table to the next page instead of splitting it and repeating headers.
    """
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
