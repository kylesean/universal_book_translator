"""A heading whose draft appended a paragraph is trimmed to its translation.

Regression (``pdf_main#b0003``): a 96-char paper title was drafted as the
title translation plus an abstract-summary paragraph, which the QE length gate
flagged as "suspiciously inflated" and quarantined. ``trim_heading_expansion``
keeps the leading paragraph so the title still ships translated.
"""

from __future__ import annotations

import pytest

from ubt.core.engine.stages.draft import trim_heading_expansion

pytestmark = pytest.mark.fast

_TITLE = "DeepSeek Elastic Compute (DSec): A Sandbox Infrastructure for Effective Agentic Training at Scale"
_TITLE_ZH = "DeepSeek 弹性计算（DSec）：用于高效大规模智能体训练的沙盒基础设施"
_SUMMARY = "DeepSeek 弹性计算（DSec）是一种专为大规模智能体训练设计的沙盒基础设施。"


def test_an_appended_paragraph_is_dropped() -> None:
    draft = f"{_TITLE_ZH}\n\n{_SUMMARY}"
    assert trim_heading_expansion(_TITLE, draft) == _TITLE_ZH


def test_a_single_paragraph_is_untouched() -> None:
    assert trim_heading_expansion(_TITLE, _TITLE_ZH) == _TITLE_ZH


def test_extra_blank_lines_do_not_keep_empty_paragraphs() -> None:
    draft = f"{_TITLE_ZH}\n\n\n{_SUMMARY}"
    assert trim_heading_expansion(_TITLE, draft) == _TITLE_ZH


def test_a_multi_line_source_is_left_alone() -> None:
    source = "Part I\n\nFoundations"
    draft = "第一部分\n\n基础"
    assert trim_heading_expansion(source, draft) == draft


def test_a_single_newline_in_the_source_is_left_alone() -> None:
    source = "Part I\nFoundations"
    draft = "第一部分\n\n基础"
    assert trim_heading_expansion(source, draft) == draft


def test_an_empty_draft_is_returned_as_is() -> None:
    assert trim_heading_expansion(_TITLE, "") == ""


def test_leading_whitespace_is_trimmed() -> None:
    draft = f"  {_TITLE_ZH}  \n\n{_SUMMARY}"
    assert trim_heading_expansion(_TITLE, draft) == _TITLE_ZH
