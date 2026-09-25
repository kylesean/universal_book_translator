"""Typst publishing output: fonts, leading, footnotes and cross-refs per language.

Was regressions/test_publishing_refactor.py -- 21 unrelated tests filed under
the milestone that produced them. The TypstReconstructor rendering cases live
here; adapter monolingual modes went to their own adapter files.
"""

from __future__ import annotations

from ubt.adapters.pdf.typst_reconstructor import TypstReconstructor
from ubt.core.ir.models import BlockStatus, BlockType, FlowID, IRBlock
from ubt.core.language_profile import resolve_font_config


def test_chinese_typst_fonts_prioritize_serif() -> None:
    """Verify that CJK language profiles prioritize Serif (Songti/Mingti) for book typography."""
    for lang in ("zh", "zh-cn", "zh-hans"):
        cfg = resolve_font_config(lang)
        assert cfg.typst_fonts[0] == "Noto Serif CJK SC", f"Failed for {lang}"
        assert cfg.typst_fonts[1] == "Source Han Serif SC"

    for lang in ("zh-tw", "zh-hant"):
        cfg = resolve_font_config(lang)
        assert cfg.typst_fonts[0] == "Noto Serif CJK TC"


def test_typst_reconstructor_preamble_contains_lang_and_indent(
    noto_cjk_installed: None,
) -> None:
    """Verify that Typst reconstructor injects lang: 'zh' and first-line-indent for Chinese books."""
    reconstructor = TypstReconstructor()
    blocks = [
        IRBlock(
            id="b1",
            flow_id=FlowID.MAIN_STORY,
            spine_index=1,
            block_type=BlockType.NARRATIVE,
            source_text="This is a paragraph.",
            target_text="这是一段测试文字。",
        )
    ]
    typst_code = reconstructor.generate_typst_source(
        blocks,
        target_lang="zh",
        page_strict=False,
    )
    assert 'lang: "zh"' in typst_code
    assert "first-line-indent: 2em" in typst_code
    assert '"Noto Serif CJK SC"' in typst_code


def test_typst_flowing_footnotes_native_linking() -> None:
    """Verify that flowing mode converts footnotes to native Typst #footnote[...] attached to narrative."""
    recon = TypstReconstructor()
    blocks = [
        IRBlock(
            id="p1",
            spine_index=1,
            flow_id=FlowID.MAIN_STORY,
            block_type=BlockType.NARRATIVE,
            source_text="Recent studies [1] have demonstrated this effect.",
            target_text="近期研究 [1] 已经证实了这一效应。",
        ),
        IRBlock(
            id="fn1",
            spine_index=2,
            flow_id=FlowID.FOOTNOTE,
            block_type=BlockType.NARRATIVE,
            source_text="[1] Smith et al. (2020), Nature, 500:10-15.",
            target_text="[1] 史密斯等人（2020），《自然》，500:10-15。",
        ),
        IRBlock(
            id="p2",
            spine_index=3,
            flow_id=FlowID.MAIN_STORY,
            block_type=BlockType.NARRATIVE,
            source_text="Furthermore, subsequent analysis corroborated the findings.",
            target_text="此外，后续分析证实了这些发现。",
        ),
    ]

    typ_src = recon.generate_typst_source(blocks, page_strict=False, target_lang="zh")
    # 1. Native footnote callout must be embedded in the narrative paragraph
    assert "#footnote[" in typ_src
    assert "史密斯等人" in typ_src
    # 2. Footnote should NOT be dumped as an inline standalone body paragraph
    lines = [ln.strip() for ln in typ_src.splitlines() if ln.strip()]
    standalone_fn = [ln for ln in lines if ln.startswith("[1] 史密斯等人")]
    assert len(standalone_fn) == 0, f"Footnote was dumped inline: {standalone_fn}"
    # 3. Subsequent paragraph flows without break
    assert "此外，后续分析证实了这些发现。" in typ_src


def test_typst_flowing_footnotes_bilingual() -> None:
    """Verify bilingual footnote formatting inside native #footnote[...]."""
    recon = TypstReconstructor()
    blocks = [
        IRBlock(
            id="p1",
            spine_index=1,
            flow_id=FlowID.MAIN_STORY,
            block_type=BlockType.NARRATIVE,
            source_text="A critical observation [1].",
            target_text="一个关键观察 [1]。",
        ),
        IRBlock(
            id="fn1",
            spine_index=2,
            flow_id=FlowID.FOOTNOTE,
            block_type=BlockType.NARRATIVE,
            source_text="[1] Source note content.",
            target_text="[1] 目标注脚内容。",
        ),
    ]

    typ_src = recon.generate_typst_source(
        blocks, bilingual=True, page_strict=False, target_lang="zh"
    )
    assert "#footnote[" in typ_src
    assert "Source note content." in typ_src
    assert "目标注脚内容。" in typ_src


def test_typst_formula_dynamic_label_and_cross_ref() -> None:
    """Verify formula blocks get dynamic labels <eq-...> and narrative references are rewritten."""
    recon = TypstReconstructor()
    blocks = [
        IRBlock(
            id="ch01#eq01",
            spine_index=1,
            flow_id=FlowID.MAIN_STORY,
            block_type=BlockType.FORMULA,
            source_text="E = m c^2 && (1)",
            target_text="E = m c^2 && (1)",
            skip_translate=True,
        ),
        IRBlock(
            id="p1",
            spine_index=2,
            flow_id=FlowID.MAIN_STORY,
            block_type=BlockType.NARRATIVE,
            source_text="As shown in Eq. (1), energy and mass are equivalent. By contrast, (99) is missing.",
            target_text="正如式 (1) 所示，质能等价。相比之下，式 (99) 不存在。",
        ),
    ]

    typ_src = recon.generate_typst_source(blocks, page_strict=False, target_lang="zh")
    # 1. Preamble enables equation numbering
    assert "#set math.equation(numbering:" in typ_src
    # 2. Formula has dynamic label
    assert "<eq-ch01-eq01>" in typ_src
    # 3. Narrative references to formula 1 are converted to Typst dynamic references
    assert "正如式 #ref(<eq-ch01-eq01>) 所示" in typ_src
    # 4. Non-existent formula 99 remains unchanged (no dangling reference)
    assert "式 (99)" in typ_src
    assert "@eq-99" not in typ_src


def test_typst_reconstructor_font_size_and_leading_overrides() -> None:
    """Verify generate_typst_source honors font_size and leading_em overrides."""
    recon = TypstReconstructor()
    blocks = [
        IRBlock(
            id="b1",
            spine_index=1,
            flow_id=FlowID.MAIN_STORY,
            block_type=BlockType.NARRATIVE,
            source_text="Test source paragraph.",
            target_text="测试正文段落。",
        )
    ]
    typ_src = recon.generate_typst_source(
        blocks=blocks,
        font_size=8.2,
        leading_em=0.62,
        target_lang="zh",
    )
    assert "size: 8.2pt" in typ_src
    assert "leading: 0.62em" in typ_src


def test_japanese_first_line_indent_1em() -> None:
    """Verify Japanese standard JIS X 4051 1em indent vs Chinese GB/T 9851 2em indent."""
    recon = TypstReconstructor()
    blocks = [
        IRBlock(
            id="b1",
            spine_index=1,
            block_type=BlockType.NARRATIVE,
            flow_id=FlowID.MAIN_STORY,
            source_text="Test",
            target_text="日本語のテキスト段落です。",
            status=BlockStatus.MTQE_PASSED,
        )
    ]
    src_ja = recon.generate_typst_source(blocks, title="Test", target_lang="ja")
    assert "first-line-indent: 1em" in src_ja
    assert 'lang: "ja"' in src_ja

    src_zh = recon.generate_typst_source(blocks, title="Test", target_lang="zh")
    assert "first-line-indent: 2em" in src_zh
    assert 'lang: "zh"' in src_zh


def test_typst_reconstructor_preserves_scott_semantic_brackets() -> None:
    """Scott semantic brackets ⟦e⟧ in translated narrative must not be stripped."""
    recon = TypstReconstructor()
    blocks = [
        IRBlock(
            id="b1",
            spine_index=1,
            block_type=BlockType.NARRATIVE,
            flow_id=FlowID.MAIN_STORY,
            source_text="Semantics ⟦e⟧ is defined.",
            target_text="语义 ⟦e⟧ 得到严格定义。",
            status=BlockStatus.MTQE_PASSED,
        )
    ]
    typ_src = recon.generate_typst_source(blocks, title="Test", target_lang="zh")
    assert "⟦e⟧" in typ_src


def test_markdown_table_escaped_pipe_and_math() -> None:
    r"""Markdown table parsing must shield escaped pipes \| and inline math $P(A|B)$."""
    from ubt.adapters.pdf.typst_fragments import _markdown_table_to_typst

    # Table with escaped pipe in cell and inline math with pipe
    md_tbl = (
        "| Operation | Expression |\n"
        "| :--- | :---: |\n"
        "| Norm | $\\|x\\| \\ge 0$ |\n"
        "| Conditional | $P(A|B)$ |\n"
    )
    typst_tbl = _markdown_table_to_typst(md_tbl)
    # Exactly two columns: an unshielded escaped pipe (\|) or the pipe inside
    # $P(A|B)$ would split a cell and yield three or four. Counting the column
    # specs pins the arity without hard-coding content-dependent widths.
    columns_line = next(
        line for line in typst_tbl.splitlines() if line.strip().startswith("columns:")
    )
    column_specs = columns_line.split("(", 1)[1].split(")", 1)[0].split(",")
    assert len(column_specs) == 2, columns_line
    assert "Norm" in typst_tbl
    assert "Conditional" in typst_tbl
    # Alignments must reflect the :---: center specifier
    assert "center" in typst_tbl


def test_untranslated_table_block_preserves_markdown() -> None:
    """An un-drafted BlockType.TABLE block must not be wrapped with 【待审校: ...】."""
    recon = TypstReconstructor()
    table_src = "| Col1 | Col2 |\n|---|---|\n| A | B |"
    blocks = [
        IRBlock(
            id="tbl1",
            spine_index=1,
            block_type=BlockType.TABLE,
            flow_id=FlowID.MAIN_STORY,
            source_text=table_src,
            target_text="",
            status=BlockStatus.PENDING,
        )
    ]
    typ_src = recon.generate_typst_source(blocks, title="Test", target_lang="zh")
    assert "【待审校:" not in typ_src
    assert "#table(" in typ_src


def test_footnote_math_uses_prose_to_typst() -> None:
    r"""Inline math $x > 0$ inside footnotes must not be escaped to literal \$x > 0\$."""
    from ubt.adapters.pdf.typst_reconstructor import _format_footnote_markup

    fn_markup = _format_footnote_markup("See $x > 0$ for details.")
    assert "$x > 0$" in fn_markup
    assert "\\$x > 0\\$" not in fn_markup


def test_extract_formula_tag_does_not_steal_body_constants() -> None:
    """Expressions like f(x) = (0.5) * y must not have (0.5) parsed as equation tag."""
    from ubt.adapters.pdf.typst_reconstructor import _extract_formula_tag

    # Mid-formula parentheses around numbers must not be stolen as tags
    assert _extract_formula_tag("f(x) = (0.5) * y") is None
    assert _extract_formula_tag("y = (3.14) x + (1.2)") is None

    # Genuine tags at the end or via \tag{} must be extracted
    assert _extract_formula_tag(r"E = mc^2 \tag{3.1}") == "3.1"
    assert _extract_formula_tag("E = mc^2 \\qquad (3.1)") == "3.1"
    assert _extract_formula_tag("E = mc^2 & (3.1)") == "3.1"
