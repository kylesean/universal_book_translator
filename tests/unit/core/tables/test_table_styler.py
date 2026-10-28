"""Unit tests for table continuation configuration and Typst/HTML styling primitives."""

from __future__ import annotations

import pytest

from ubt.core.tables.models import TableContinuationConfig
from ubt.core.tables.styler import (
    generate_html_table_continuation_css,
    generate_typst_table_continuation_elements,
    sanitize_table_uid,
)
from ubt.render.typst.document_renderer import render_table

pytestmark = pytest.mark.fast


def test_table_continuation_config_language_resolution() -> None:
    cfg = TableContinuationConfig()
    # Chinese resolution
    assert cfg.get_continued_header_text("zh") == "（续表）"
    assert cfg.get_continued_header_text("zh-CN") == "（续表）"
    assert cfg.get_next_page_footer_text("zh") == "（续下页）"

    # English resolution
    assert cfg.get_continued_header_text("en") == "(Continued from previous page)"
    assert cfg.get_next_page_footer_text("en") == "(Continued on next page)"
    assert cfg.get_continued_header_text("fr") == "(Continued from previous page)"


def test_sanitize_table_uid() -> None:
    assert sanitize_table_uid("b_001") == "b_001"
    assert sanitize_table_uid("table:item#42") == "table-item-42"
    assert sanitize_table_uid("") == "tbl"


def test_generate_typst_table_continuation_elements() -> None:
    cfg = TableContinuationConfig(
        continued_header_text_zh="[续表]",
        next_page_footer_text_zh="[接下页]",
        font_size_pt=8.0,
        text_color_luma=90,
    )
    start_anc, header_cell, footer_blk, end_anc = generate_typst_table_continuation_elements(
        columns=3,
        table_uid="tab1",
        lang="zh",
        config=cfg,
        indent="  ",
    )

    assert "<ubt-tbl-start-tab1>" in start_anc
    assert "<ubt-tbl-end-tab1>" in end_anc

    # Header cell checks
    assert "table.cell(colspan: 3)" in header_cell
    assert "#context" in header_cell
    assert "query(selector(<ubt-tbl-start-tab1>).before(here()))" in header_cell
    assert "here().page() > el.location().page()" in header_cell
    assert "[续表]" in header_cell

    # Footer block checks
    assert "table.footer(" in footer_blk
    assert "repeat: true" in footer_blk
    assert "query(selector(<ubt-tbl-end-tab1>).after(here()))" in footer_blk
    assert "here().page() < el.location().page()" in footer_blk
    assert "[接下页]" in footer_blk


def test_generate_html_table_continuation_css() -> None:
    css = generate_html_table_continuation_css()
    assert "@media print" in css
    assert "table-header-group" in css
    assert "table-footer-group" in css
    assert ".ubt-table-continued-notice" in css


def test_render_table_with_continuation_enabled() -> None:
    md_table = (
        "| ID | Metric | Score |\n"
        "| --- | --- | --- |\n"
        "| 1 | Latency | 12ms |\n"
        "| 2 | Memory | 450MB |\n"
        "| 3 | Throughput | 8.2k |\n"
        "| 4 | Accuracy | 99.1% |\n"
        "| 5 | F1 | 0.98 |\n"
    )

    cfg = TableContinuationConfig()
    markup = render_table(
        md_table,
        enable_continuation=True,
        table_uid="bench-tab",
        continuation_config=cfg,
    )

    assert "<ubt-tbl-start-bench-tab>" in markup
    assert "<ubt-tbl-end-bench-tab>" in markup
    assert "table.header(" in markup
    assert "（续表）" in markup
    assert "table.footer(" in markup
    assert "（续下页）" in markup
