"""Unit tests for chapter header decorations and drop caps."""

from __future__ import annotations

import pytest

from ubt.core.decor.dropcap import (
    HTML_DROPCAP_CSS,
    TYPST_DROPCAP_MACRO,
    apply_dropcap_html,
    apply_dropcap_typst,
    extract_dropcap_initial,
)
from ubt.core.decor.header import (
    parse_chapter_heading,
    render_chapter_header_html,
    render_chapter_header_typst,
)
from ubt.core.decor.models import (
    ChapterHeaderConfig,
    DecorSpec,
    DropCapConfig,
    HeaderStyle,
)


@pytest.mark.fast
def test_extract_dropcap_initial() -> None:
    # Plain CJK
    prefix, letter, rem = extract_dropcap_initial("在大规模分布式系统中")
    assert prefix == ""
    assert letter == "在"
    assert rem == "大规模分布式系统中"

    # Plain Latin
    prefix, letter, rem = extract_dropcap_initial("In distributed systems")
    assert prefix == ""
    assert letter == "I"
    assert rem == "n distributed systems"

    # Leading quotation marks
    for q in ["“", "‘", '"', "'", "「", "『", "《", "（", "("]:
        prefix, letter, rem = extract_dropcap_initial(f"{q}自然语言处理")
        assert prefix == q
        assert letter == "自"
        assert rem == "然语言处理"

    # Leading quote with spaces
    prefix, letter, rem = extract_dropcap_initial("“  宇宙的尽头")
    assert prefix == "“"
    assert letter == "宇"
    assert rem == "宙的尽头"

    # Empty or whitespace
    assert extract_dropcap_initial("") == ("", "", "")
    assert extract_dropcap_initial("   ") == ("", "", "")

    # Single character
    assert extract_dropcap_initial("A") == ("", "A", "")
    assert extract_dropcap_initial("“") == ("", "“", "")


@pytest.mark.fast
def test_apply_dropcap_typst() -> None:
    cfg = DropCapConfig(lines=3, color="#1e40af")
    text = "在大规模分布式系统中，高可用性是关键。"
    res = apply_dropcap_typst(text, cfg)
    assert '#ubt-dropcap("在", prefix: "", lines: 3, fill: rgb("#1e40af"))' in res
    assert "大规模分布式系统中，高可用性是关键。" in res

    # With opening quote
    q_text = "“我们在构建系统时应当考量…”"
    res_q = apply_dropcap_typst(q_text, cfg)
    assert '#ubt-dropcap("我", prefix: "“", lines: 3, fill: rgb("#1e40af"))' in res_q

    # Disabled
    cfg_off = DropCapConfig(enabled=False)
    assert apply_dropcap_typst(text, cfg_off) == text

    # Skipped syntax (markdown headers, formulas, code, list items)
    assert apply_dropcap_typst("# Heading", cfg) == "# Heading"
    assert apply_dropcap_typst("$E = mc^2$", cfg) == "$E = mc^2$"
    assert apply_dropcap_typst("```python", cfg) == "```python"
    assert apply_dropcap_typst("- list item", cfg) == "- list item"
    assert apply_dropcap_typst("* bullet", cfg) == "* bullet"


@pytest.mark.fast
def test_apply_dropcap_html() -> None:
    cfg = DropCapConfig(lines=2, color="#0f172a")
    text = "在大规模分布式系统中。"
    res = apply_dropcap_html(text, cfg)
    assert '<span class="ubt-dropcap-wrap"' in res
    assert '<span class="ubt-dropcap-letter">在</span>' in res

    # With prefix
    res_q = apply_dropcap_html("“深度学习模型”", cfg)
    assert '<span class="ubt-dropcap-prefix">“</span>' in res_q
    assert '<span class="ubt-dropcap-letter">深</span>' in res_q

    # Disabled
    cfg_off = DropCapConfig(enabled=False)
    assert apply_dropcap_html(text, cfg_off) == text


@pytest.mark.fast
def test_parse_chapter_heading() -> None:
    # Chinese chapter label
    label, title, sub = parse_chapter_heading("第 1 章：分布式系统导论")
    assert label == "第 1 章"
    assert title == "分布式系统导论"
    assert sub == ""

    # English chapter label with subtitle delimiter
    label, title, sub = parse_chapter_heading("Chapter 4: Paxos Made Simple -- Part 1")
    assert label == "Chapter 4"
    assert title == "Paxos Made Simple"
    assert sub == "Part 1"

    # Chinese em-dash subtitle
    label, title, sub = parse_chapter_heading("第十二篇 机器学习 —— 理论与实践")
    assert label == "第十二篇"
    assert title == "机器学习"
    assert sub == "理论与实践"

    # Numeric dot label
    label, title, sub = parse_chapter_heading("3.2: 存储引擎架构")
    assert label == "3.2"
    assert title == "存储引擎架构"
    assert sub == ""

    # Bare title
    label, title, sub = parse_chapter_heading("前言与致谢")
    assert label == ""
    assert title == "前言与致谢"
    assert sub == ""

    # Empty
    assert parse_chapter_heading("") == ("", "", "")


@pytest.mark.fast
def test_render_chapter_header_typst() -> None:
    heading = "第 3 章：共识算法 -- Raft协议"

    # Classic
    cfg_classic = ChapterHeaderConfig(style=HeaderStyle.CLASSIC)
    typst_classic = render_chapter_header_typst(heading, cfg_classic)
    assert "#align(center)" in typst_classic
    assert "第 3 章" in typst_classic
    assert "共识算法" in typst_classic
    assert "Raft协议" in typst_classic
    assert "❦" in typst_classic

    # Modern
    cfg_modern = ChapterHeaderConfig(style=HeaderStyle.MODERN)
    typst_modern = render_chapter_header_typst(heading, cfg_modern)
    assert "inset: (left: 14pt)" in typst_modern
    assert "第 3 章" in typst_modern

    # Academic
    cfg_acad = ChapterHeaderConfig(style=HeaderStyle.ACADEMIC)
    typst_acad = render_chapter_header_typst(heading, cfg_acad)
    assert "#line(length: 100%" in typst_acad

    # Minimal
    cfg_min = ChapterHeaderConfig(style=HeaderStyle.MINIMAL)
    typst_min = render_chapter_header_typst(heading, cfg_min)
    assert 'weight: "medium"' in typst_min

    # Disabled
    cfg_off = ChapterHeaderConfig(enabled=False)
    typst_off = render_chapter_header_typst(heading, cfg_off)
    assert typst_off == "#heading(level: 1)[第 3 章：共识算法 -- Raft协议]"


@pytest.mark.fast
def test_render_chapter_header_html() -> None:
    heading = "第 1 章：引言"
    cfg_classic = ChapterHeaderConfig(style=HeaderStyle.CLASSIC)
    html_out = render_chapter_header_html(heading, cfg_classic)
    assert '<header class="ubt-chapter-header ubt-header-classic">' in html_out
    assert '<div class="ubt-chapter-label">第 1 章</div>' in html_out
    assert '<h1 class="ubt-chapter-title">引言</h1>' in html_out
    assert "❦" in html_out

    # Disabled
    cfg_off = ChapterHeaderConfig(enabled=False)
    assert render_chapter_header_html(heading, cfg_off) == "<h1>第 1 章：引言</h1>"


@pytest.mark.fast
def test_decor_models_to_dict() -> None:
    spec = DecorSpec()
    d = spec.to_dict()
    assert "header" in d
    assert "drop_cap" in d
    assert d["header"]["style"] == "classic"
    assert d["drop_cap"]["lines"] == 2
    assert "TYPST_DROPCAP_MACRO" in globals() or len(TYPST_DROPCAP_MACRO) > 0
    assert "HTML_DROPCAP_CSS" in globals() or len(HTML_DROPCAP_CSS) > 0
