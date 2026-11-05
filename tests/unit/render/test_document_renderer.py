from __future__ import annotations

from pathlib import Path

import pytest

from ubt.core.ir.models import BlockType, InlineRun, IRBlock, make_element
from ubt.render.typst.document_renderer import (
    _PENDING_CAPTION_MAX_CHARS,
    ReflowStats,
    TypstDocumentRenderer,
    render_prose,
    render_table,
)

pytestmark = pytest.mark.fast


def _block(
    block_type: BlockType,
    *,
    block_id: str = "b1",
    source: str = "source",
    target: str | None = None,
    draft: str | None = None,
    level: int = 1,
    skip: bool = False,
    policy_translate: bool | None = None,
    asset_id: str = "",
) -> IRBlock:
    element = make_element(
        id=block_id,
        spine_index=1,
        block_type=block_type,
        source_text=source,
        level=level,
        skip_translate=skip,
    )
    if asset_id:
        element = type(element)(
            **{
                **{f: getattr(element, f) for f in ("id", "spine_index", "flow", "region")},
                "asset_id": asset_id,
                "kind": element.kind,
            }
        )
    return IRBlock(
        element=element, target_text=target, draft_text=draft, policy_translate=policy_translate
    )


def _render(blocks: list[IRBlock], tmp_path: Path) -> str:
    renderer = TypstDocumentRenderer()
    source, _stats = renderer.build_source(blocks, tmp_path / "build")
    return source


def _render_with_stats(blocks: list[IRBlock], tmp_path: Path) -> tuple[str, ReflowStats]:
    renderer = TypstDocumentRenderer()
    return renderer.build_source(blocks, tmp_path / "build")


def test_math_is_transpiled_not_escaped(tmp_path: Path) -> None:
    out = render_prose("Let $x_i$ be the input.")
    assert "$x_i$" in out
    assert r"\$" not in out


def test_prose_markup_is_escaped(tmp_path: Path) -> None:
    out = render_prose("Use a_b and #tag here.")
    assert r"a\_b" in out
    assert r"\#tag" in out


def test_a_failed_math_conversion_falls_back_to_escaped_text(tmp_path: Path) -> None:
    out = render_prose("Broken $\\undefinedcmd{x}$ end.")

    assert out.count("$") % 2 == 0
    assert "Broken" in out and "end." in out, "the surrounding prose must survive"
    assert "⟨math⟩" in out, "the span must degrade to a visible marker, not vanish"


def test_display_math_gets_its_own_block(tmp_path: Path) -> None:
    out = render_prose(r"See $$E = mc^2$$ here.")
    assert "\n$ " in out


def test_a_math_body_the_compiler_rejects_degrades_instead_of_failing(tmp_path: Path) -> None:
    rejected = "frac(a, )"

    def gate(body: str) -> bool:
        return body != rejected

    out = render_prose(r"Bad $\frac{a}{}$ here.", is_math_ok=gate)
    assert rejected not in out
    assert out.count("$") % 2 == 0

    ungated = render_prose(r"Bad $\frac{a}{}$ here.")
    assert rejected in ungated


def test_a_renderer_math_gate_reaches_every_prose_call_site(tmp_path: Path) -> None:
    renderer = TypstDocumentRenderer(math_gate=lambda body: body != "frac(a, )")
    blocks = [
        _block(BlockType.HEADING, block_id="h1", source="Head", target=r"Head $\frac{a}{}$"),
        _block(BlockType.NARRATIVE, block_id="p1", source="Body", target=r"Body $\frac{a}{}$"),
    ]
    out, _stats = renderer.build_source(blocks, tmp_path / "build")
    assert "frac(a, )" not in out


def test_the_gate_reaches_the_formula_block_branch(tmp_path: Path) -> None:
    renderer = TypstDocumentRenderer(math_gate=lambda body: body != "frac(a, )")
    blocks = [
        _block(
            BlockType.FORMULA,
            block_id="f1",
            source=r"$$\frac{a}{}$$",
            target="",
        )
    ]
    out, stats = renderer.build_source(blocks, tmp_path / "build")
    assert "frac(a, )" not in out
    assert stats.formulas == 0


def test_an_acceptable_formula_body_still_compiles_under_the_gate(tmp_path: Path) -> None:
    renderer = TypstDocumentRenderer(math_gate=lambda body: body != "frac(a, )")
    blocks = [
        _block(BlockType.FORMULA, block_id="f1", source=r"$$\frac{a}{b}$$", target=""),
    ]
    out, stats = renderer.build_source(blocks, tmp_path / "build")
    assert "frac(a, b)" in out
    assert stats.formulas == 1


def test_a_heading_carries_its_level(tmp_path: Path) -> None:
    blocks = [
        _block(BlockType.HEADING, block_id="h1", source="1 Introduction", target="1 引言", level=1),
        _block(BlockType.HEADING, block_id="h2", source="2.1 Routing", target="2.1 路由", level=2),
    ]
    out = _render(blocks, tmp_path)
    assert "#heading(level: 1)[1 引言]" in out
    assert "#heading(level: 2)[2.1 路由]" in out


def test_an_arxiv_stamp_heading_is_dropped(tmp_path: Path) -> None:
    blocks = [_block(BlockType.HEADING, source="arXiv:2101.03961v3 [cs.LG] 16 Jun 2022")]
    out = _render(blocks, tmp_path)
    assert "arXiv:2101" not in out


def test_a_table_is_rebuilt_with_translated_headers(tmp_path: Path) -> None:
    markup = "| Model | Quality |\n|---|---|\n| T5-Base | -1.73 |"
    out = render_table(markup)
    assert "#table(" in out
    assert "模型" not in out
    assert "[*Model*]" in out


def test_table_columns_are_weighted_by_content() -> None:
    markup = (
        "| Layout Analysis | Params | FPS | RecognitionMethod |\n"
        "|---|---|---|---|\n"
        "| DocLayout-YOLO [14] | 20M | 107.4 | MonkeyOCR [27] MinerU2.5 [5] |"
    )
    out = render_table(markup)
    spec = next(line for line in out.splitlines() if "columns:" in line)
    widths = [
        float(part.strip().removesuffix("pt"))
        for part in spec.split("(")[1].rstrip("),").split(",")
    ]
    assert len(widths) == 4
    assert all(w > 0 for w in widths)
    assert widths[3] > widths[0]
    assert min(widths) >= 10.0
    assert 425.0 <= sum(widths) <= 455.0


def test_a_formula_block_is_typeset_as_math(tmp_path: Path) -> None:
    blocks = [
        _block(
            BlockType.FORMULA,
            source=r"$$ \frac { a } { b } $$",
            skip=True,
        )
    ]
    out = _render(blocks, tmp_path)
    assert "frac(a, b)" in out
    assert r"\frac" not in out


def test_a_formula_uses_its_translated_annotation(tmp_path: Path) -> None:
    blocks = [
        _block(
            BlockType.FORMULA,
            block_id="f1",
            source=r"$$ \mathbf{A} : \text{parameter} $$",
            target=r"$$ \mathbf{A} : \text{参数} $$",
            skip=True,
        )
    ]
    renderer = TypstDocumentRenderer()
    out, stats = renderer.build_source(blocks, tmp_path / "build")
    assert "参数" in out
    assert "parameter" not in out
    assert stats.formulas_from_target == 1


def test_an_untranslated_formula_is_not_counted_as_target_sourced(tmp_path: Path) -> None:
    blocks = [
        _block(BlockType.FORMULA, block_id="f2", source=r"$$ E = mc^2 $$", skip=True),
    ]
    _out, stats = _render_with_stats(blocks, tmp_path)
    assert stats.formulas == 1
    assert stats.formulas_from_target == 0


def test_a_bibliography_entry_is_kept_in_source(tmp_path: Path) -> None:
    blocks = [
        _block(
            BlockType.LIST_ITEM,
            source="Shazeer et al. (2017). Outrageously large neural networks.",
            draft="Shazeer 等人（2017）。",
            skip=True,
        )
    ]
    out = _render(blocks, tmp_path)
    assert "Outrageously large neural networks" in out
    assert "Shazeer 等人" not in out


def test_a_policy_verdict_outranks_the_extractor_default(tmp_path: Path) -> None:
    blocks = [
        _block(
            BlockType.NARRATIVE,
            source="Kept verbatim.",
            target="被翻译了。",
            policy_translate=False,
        )
    ]
    out = _render(blocks, tmp_path)
    assert "Kept verbatim." in out


def test_a_quarantine_placeholder_never_reaches_the_page(tmp_path: Path) -> None:
    blocks = [
        _block(
            BlockType.NARRATIVE,
            source="Why use Switch Transformer?",
            target='<mark class="ubt-blocked-human" title="quarantined">Why use Switch</mark>',
            draft="为何选择 Switch Transformer？",
        )
    ]
    out = _render(blocks, tmp_path)
    assert "<mark" not in out
    assert "ubt-blocked-human" not in out
    assert "为何选择 Switch Transformer" in out


def test_a_block_with_no_translation_ships_its_source(tmp_path: Path) -> None:
    blocks = [_block(BlockType.NARRATIVE, source="Untranslated sentence.")]
    out = _render(blocks, tmp_path)
    assert "Untranslated sentence." in out


def test_a_missing_asset_leaves_a_visible_placeholder(tmp_path: Path) -> None:
    blocks = [_block(BlockType.IMAGE, source="/nonexistent/fig.png")]
    renderer = TypstDocumentRenderer(assets_root=tmp_path)
    out, stats = renderer.build_source(blocks, tmp_path)
    assert "missing figure" in out
    assert stats.assets_missing


def test_a_caption_follows_its_figure(tmp_path: Path) -> None:
    asset = tmp_path / "fig.png"
    asset.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
    blocks = [
        _block(BlockType.IMAGE, block_id="i1", source=str(asset)),
        _block(
            BlockType.NARRATIVE,
            block_id="c1",
            source="Figure 1: A diagram.",
            target="图 1：示意图。",
        ),
    ]
    renderer = TypstDocumentRenderer()
    out, stats = renderer.build_source(blocks, tmp_path / "build")
    assert stats.figures == 1
    assert "caption: [图 1：示意图。]" in out


def test_bilingual_pairs_the_source_with_its_translation(tmp_path: Path) -> None:
    blocks = [
        _block(
            BlockType.NARRATIVE,
            block_id="p1",
            source="A generative model learns the data distribution.",
            target="生成模型学习数据分布。",
        )
    ]
    renderer = TypstDocumentRenderer(bilingual=True)
    out, _stats = renderer.build_source(blocks, tmp_path / "build")

    assert "A generative model learns the data distribution." in out
    assert "生成模型学习数据分布。" in out


def test_monolingual_omits_the_source(tmp_path: Path) -> None:
    blocks = [
        _block(
            BlockType.NARRATIVE,
            block_id="p1",
            source="A generative model learns the data distribution.",
            target="生成模型学习数据分布。",
        )
    ]
    renderer = TypstDocumentRenderer(bilingual=False)
    out, _stats = renderer.build_source(blocks, tmp_path / "build")

    assert "生成模型学习数据分布。" in out
    assert "A generative model learns" not in out


def test_bilingual_does_not_repeat_an_untranslated_block(tmp_path: Path) -> None:
    blocks = [_block(BlockType.NARRATIVE, block_id="p1", source="Untranslated text.")]
    renderer = TypstDocumentRenderer(bilingual=True)
    out, _stats = renderer.build_source(blocks, tmp_path / "build")

    assert out.count("Untranslated text.") == 1


def test_bilingual_does_not_repeat_a_source_kept_block(tmp_path: Path) -> None:
    blocks = [
        _block(
            BlockType.NARRATIVE,
            block_id="ref1",
            source="Robbins, H. Some aspects of the sequential design of experiments.",
            target="Robbins, H. Some aspects of the sequential design of experiments.",
            skip=True,
        )
    ]
    renderer = TypstDocumentRenderer(bilingual=True)
    out, _stats = renderer.build_source(blocks, tmp_path / "build")

    assert out.count("Robbins, H.") == 1


def test_bilingual_pairs_headings_and_tables_too(tmp_path: Path) -> None:
    blocks = [
        _block(BlockType.HEADING, block_id="h1", source="Related Work", target="相关工作"),
        _block(
            BlockType.TABLE,
            block_id="t1",
            source="| A | B |\n| --- | --- |\n| 1 | 2 |",
            target="| 甲 | 乙 |\n| --- | --- |\n| 一 | 二 |",
        ),
    ]
    renderer = TypstDocumentRenderer(bilingual=True)
    out, _stats = renderer.build_source(blocks, tmp_path / "build")

    assert "Related Work" in out
    assert "相关工作" in out
    assert "甲" in out
    assert "| A | B |" not in out
    assert out.count("#table(") == 2


@pytest.mark.parametrize(
    ("text", "run", "expect"),
    [
        ("lo g(4)", InlineRun(text="g", superscript=True), "("),
        (
            "Firecracker microVMs (Agache et al., 2020)",
            InlineRun(text="Firecracker microVMs ", bold=True),
            "(",
        ),
    ],
)
def test_a_styled_run_never_swallows_a_following_paren(
    text: str, run: InlineRun, expect: str
) -> None:
    out = render_prose(text, (run,))
    assert "/**/" in out or "#text(" in out, f"the run's expression must be closed: {out!r}"
    assert expect in out, f"the paren must survive verbatim: {out!r}"


def test_an_unstyled_paren_is_left_alone() -> None:
    assert "/**/" not in render_prose("plain (paren)", ())


@pytest.mark.parametrize(
    ("text", "run", "expect"),
    [
        ("检验∃𝑑.installed𝑚(𝛾) 一致", InlineRun(text="𝑑", superscript=True), "."),
        ("见上标𝑛。后文", InlineRun(text="𝑛", superscript=True), "。"),
    ],
)
def test_a_styled_run_never_absorbs_a_following_dot(text: str, run: InlineRun, expect: str) -> None:
    out = render_prose(text, (run,))
    assert "/**/" in out or "#text(" in out, f"the run's expression must be closed: {out!r}"
    assert expect in out, f"the dot must survive verbatim: {out!r}"


@pytest.mark.parametrize(
    "token",
    [
        "⟦SOUP_MASK_0001-333⟧",
        "⟦MATH_MASK_0002-abc⟧",
        "⟦CITE_MASK_0003-def⟧",
        "⟦CODE_MASK_0004-012⟧",
        "⟦EMAIL_MASK_0005-345⟧",
    ],
)
def test_a_mask_token_inside_a_math_span_is_not_transpiled(token: str) -> None:
    out = render_prose(f"$a_i : {token} b_j$", ())

    assert '"SOUP"' not in out and '"MASK"' not in out, f"code mode leaked: {out!r}"
    assert "#" not in out, f"a bare # would enter code mode: {out!r}"
    assert token.replace("_", "\\_") in out


def test_a_mask_token_in_display_math_is_not_transpiled() -> None:
    out = render_prose("$$x = ⟦MATH_MASK_0002-abc⟧$$", ())
    assert "#" not in out, out
    assert "\\_MASK\\_" in out


def test_real_math_still_transpiles_when_no_token_is_present() -> None:
    out = render_prose("$x^2 + y^2 = z^2$", ())
    assert out.startswith("$") and out.endswith("$"), out
    assert "#" not in out


def test_a_mask_token_in_prose_is_escaped_as_before() -> None:
    out = render_prose("prose ⟦CITE_MASK_0001-abc⟧ text", ())
    assert out == "prose ⟦CITE\\_MASK\\_0001-abc⟧ text"


def test_a_leaked_emphasis_marker_inside_math_does_not_split_the_formula() -> None:
    out = render_prose(r"扫描输入 $(\Delta, ⟦ B ⟧ A ⟦ /B ⟧)$ 结束", ())

    assert out.count("$") == 0, f"a delimiter stayed a math switch: {out!r}"
    assert "⟦" in out and "⟧" in out


def test_a_latex_transcribed_emphasis_marker_keeps_its_backslashes() -> None:
    out = render_prose(r"$(\Delta, \llbracket B \rrbracket A \llbracket /B \rrbracket)$", ())

    assert "llbracket" in out and "rrbracket" in out
    assert "\\llbracket" in out, f"the backslash must survive: {out!r}"
    assert out.count("$") == 0, f"a delimiter stayed a math switch: {out!r}"


def test_the_leaked_sentinel_count_is_reported(tmp_path: Path) -> None:
    blocks = [
        _block(
            BlockType.NARRATIVE,
            block_id="b1",
            target=r"扫描输入 $(\Delta, \llbracket B \rrbracket A)$ 结束",
        ),
        _block(BlockType.NARRATIVE, block_id="b2", target="干净的段落"),
    ]
    _source, stats = _render_with_stats(blocks, tmp_path)

    assert stats.sentinels_leaked == 1, stats


def test_a_clean_document_reports_no_leaked_sentinels(tmp_path: Path) -> None:
    blocks = [_block(BlockType.NARRATIVE, target=r"普通公式 $x^2 + y^2 = z^2$ 与引用 [21]")]
    _source, stats = _render_with_stats(blocks, tmp_path)

    assert stats.sentinels_leaked == 0, stats


def test_a_run_straddling_a_math_boundary_does_not_split_the_formula() -> None:
    text = r"$\{ x_k \} \to 0$是一个序列"
    runs = (InlineRun(text=r"x_k \} \to 0$是", superscript=True),)
    out = render_prose(text, runs)

    assert out.count("$") == 2, f"the formula must keep both delimiters: {out!r}"
    assert "\\$" not in out, f"a delimiter was escaped as literal text: {out!r}"
    assert "#super" not in out, f"the straddling run must be dropped: {out!r}"


def test_a_run_fully_inside_a_formula_is_still_dropped() -> None:
    text = r"$\{ x_k \} \to 0$是一个序列"
    out = render_prose(text, (InlineRun(text="x_k", superscript=True),))
    assert out.count("$") == 2
    assert "#super" not in out


def test_a_run_beside_a_formula_still_styles_its_own_text() -> None:
    out = render_prose(r"前 $x_i$ 后", (InlineRun(text="前", bold=True),))
    assert "#strong[前]" in out
    assert out.count("$") == 2


def test_a_run_ending_exactly_at_the_opening_delimiter_still_styles() -> None:
    out = render_prose(r"前$x_i$", (InlineRun(text="前", bold=True),))
    assert "#strong[前]" in out, out
    assert out.count("$") == 2


def test_a_stray_dollar_does_not_swallow_prose_into_math() -> None:
    text = r"{ x_k } \to 0$ 是一个序列，其中 $x_k \in \Omega$。"
    out = render_prose(text, ())

    assert out.count("\\$") == text.count("$"), f"a delimiter stayed a math switch: {out!r}"
    assert out.replace("\\$", "").count("$") == 0, f"a bare $ remained: {out!r}"
    assert "是一个序列" in out


def test_an_unclosed_trailing_dollar_keeps_the_earlier_pair() -> None:
    text = r"$x$ 和 $y"
    out = render_prose(text, ())
    assert out.startswith("$x$"), f"the balanced pair must stay math: {out!r}"
    assert out.endswith("\\$y")


def test_a_cjk_free_span_is_still_math() -> None:
    assert render_prose(r"a $x_i + y^2$ b", ()) == r"a $x_i + y^2$ b"


def test_a_display_formula_with_cjk_inside_is_prose() -> None:
    out = render_prose("$$这段中文$$", ())
    assert "\\$" in out


def test_a_table_cell_with_math_transpiles_it() -> None:
    out = render_table("| a | b |\n|---|---|\n| 循环 | O($n \\cdot d^2$) |")

    assert "$n dot d^2$" in out, f"the cell's math must stay math: {out!r}"
    assert "\\\\$" not in out, f"a double-escaped delimiter would break Typst: {out!r}"


def test_a_table_cell_without_math_is_escaped_once() -> None:
    out = render_table("| a |\n|---|\n| 100% [1] |")
    assert "\\[1\\]" in out or "\\[1\\]" in out
    assert "\\\\[" not in out, f"a double-escaped bracket: {out!r}"


def test_parse_markdown_table_can_return_raw_cells() -> None:
    from ubt.render.table_compiler import parse_markdown_table

    md = "| a |\n|---|\n| O($n$) |"
    escaped = parse_markdown_table(md)
    raw = parse_markdown_table(md, escape=False)
    assert escaped is not None and raw is not None
    assert escaped[1][0][0] == "O(\\$n\\$)"
    assert raw[1][0][0] == "O($n$)"


def test_a_formula_with_a_chinese_text_annotation_stays_math() -> None:
    out = render_prose(r"$\mathbf{A} : ( D , N ) \text{ )← 参数 } \vdash N \times N$", ())

    assert "bold(A)" in out, out
    assert "\\mathbf" not in out and "\\text" not in out, out
    assert "\\{" not in out, out


def test_a_formula_made_only_of_chinese_words_is_prose() -> None:
    out = render_prose("$ 是一个序列，其中 $", ())
    assert "$" not in out.replace("\\$", ""), out


def test_a_punctuation_only_span_is_not_math() -> None:
    for span in ("$、$", "$, $", "$，$"):
        out = render_prose(span, ())
        assert "$" not in out.replace("\\$", ""), f"{span} entered math mode: {out!r}"


def test_a_run_with_unbalanced_delimiters_is_not_wrapped() -> None:
    text = r"用于 $Delta$、$\$、$C$ 和矩阵 $\llbracket A \rrbracket$ 的投影"
    bad = InlineRun(text=r"$、$C$ 和矩阵 $\llbracket A \rrbracket$ 的投影", bold=True)
    out = render_prose(text, [bad])

    assert "#strong[" not in out, out


def test_a_run_with_balanced_delimiters_is_still_wrapped() -> None:
    out = render_prose("粗体结尾", (InlineRun(text="粗体结尾", bold=True),))
    assert "#strong[粗体结尾]" in out, out


def test_a_missing_figure_still_renders_its_caption(tmp_path: Path) -> None:
    renderer = TypstDocumentRenderer(assets_root=tmp_path / "assets")
    blocks = [
        _block(BlockType.IMAGE, block_id="f1", source="figures/gone.png"),
        _block(
            BlockType.NARRATIVE,
            block_id="c1",
            source="Figure 1: System architecture overview.",
            target="图 1：系统整体架构示意图。",
        ),
    ]
    out, stats = renderer.build_source(blocks, tmp_path / "build")

    assert "missing figure" in out
    assert "系统整体架构示意图" in out
    assert stats.assets_missing == [("f1", "figures/gone.png")]


def test_a_missing_figure_without_a_caption_renders_only_the_placeholder(
    tmp_path: Path,
) -> None:
    renderer = TypstDocumentRenderer(assets_root=tmp_path / "assets")
    out, _stats = renderer.build_source(
        [_block(BlockType.IMAGE, block_id="f1", source="figures/gone.png")],
        tmp_path / "build",
    )
    assert "missing figure" in out
    assert out.count("missing figure") == 1
    assert out.count("fill: luma(60))[") == 1


def test_a_bilingual_table_does_not_paste_its_markdown_source(tmp_path: Path) -> None:
    renderer = TypstDocumentRenderer(bilingual=True)
    blocks = [
        _block(
            BlockType.TABLE,
            block_id="t1",
            source="| Method | BLEU |\n| --- | --- |\n| Transformer | 28.4 |",
            target="| 方法 | BLEU |\n| --- | --- |\n| Transformer | 28.4 |",
        )
    ]
    out, _stats = renderer.build_source(blocks, tmp_path / "build")

    assert "| Method | BLEU |" not in out
    assert out.count("#table(") == 2


def test_unparseable_table_falls_back_to_source_and_is_counted(tmp_path: Path) -> None:
    malformed_target = "模型 | Huginn-0125 |\n|---|---|\n| 前奏 | 测试 |"
    good_source = "| Model | Huginn-0125 |\n|---|---|\n| Intro | Test |"
    block = _block(BlockType.TABLE, block_id="tbl", source=good_source, target=malformed_target)

    source, stats = _render_with_stats([block], tmp_path)

    assert stats.tables == 1
    assert stats.tables_fell_back == 1
    assert stats.tables_dropped == 0
    assert "#table(" in source
    assert "Model" in source


def test_table_unparseable_in_both_grids_is_reported_not_silent(tmp_path: Path) -> None:
    malformed = "模型 | Huginn-0125 |\n|---|---|\n| 前奏 | 测试 |"
    block = _block(BlockType.TABLE, block_id="tbl2", source=malformed, target=malformed)

    renderer = TypstDocumentRenderer()
    _source, stats = renderer.build_source([block], tmp_path / "build")

    assert stats.tables_dropped == 1
    assert stats.tables == 0
    assert (("tbl2", "table_unparseable")) in renderer.last_render_skips


def test_parseable_table_renders_without_fallback(tmp_path: Path) -> None:
    grid = "| Model | Score |\n|---|---|\n| A | 1 |"
    block = _block(BlockType.TABLE, block_id="tbl3", source=grid, target=grid)
    _source, stats = _render_with_stats([block], tmp_path)
    assert stats.tables == 1
    assert stats.tables_fell_back == 0
    assert stats.tables_dropped == 0


def test_a_shape_preserving_table_rebuild_is_verified(tmp_path: Path) -> None:
    source_grid = "| Model | Score |\n|---|---|\n| A | 1 |"
    target_grid = "| 模型 | 分数 |\n|---|---|\n| 甲 | 一 |"
    block = _block(BlockType.TABLE, block_id="tv", source=source_grid, target=target_grid)

    renderer = TypstDocumentRenderer()
    renderer.build_source([block], tmp_path / "build")

    (record,) = renderer.table_reconstructions
    assert record.block_id == "tv"
    assert record.verified is True
    assert record.corrupt is False


def test_a_table_rebuild_that_dropped_a_column_is_corrupt(tmp_path: Path) -> None:
    source_grid = "| Model | Score | Note |\n|---|---|---|\n| A | 1 | x |"
    target_grid = "| 模型 | 分数 |\n|---|---|\n| 甲 | 一 |"
    block = _block(BlockType.TABLE, block_id="tc", source=source_grid, target=target_grid)

    renderer = TypstDocumentRenderer()
    renderer.build_source([block], tmp_path / "build")

    (record,) = renderer.table_reconstructions
    assert record.corrupt is True
    assert record.verified is False
    assert "column_count_mismatch" in record.reason


def test_a_table_row_that_gained_a_cell_ships_merged_and_verified(tmp_path: Path) -> None:
    source_grid = "| A | B | C |\n|---|---|---|\n| 1 | 2 | 3 |"
    target_grid = "| 甲 | 乙 | 丙 |\n|---|---|---|\n| 一 | 二 | 三 | 四 |"
    block = _block(BlockType.TABLE, block_id="tg", source=source_grid, target=target_grid)

    renderer = TypstDocumentRenderer()
    renderer.build_source([block], tmp_path / "build")

    (record,) = renderer.table_reconstructions
    assert record.corrupt is False
    assert record.verified is True


def test_a_table_that_fell_back_to_source_is_unverified_not_corrupt(tmp_path: Path) -> None:
    malformed_target = "模型 | Score |\n|---|---|\n| 甲 | 一 |"
    good_source = "| Model | Score |\n|---|---|\n| A | 1 |"
    block = _block(BlockType.TABLE, block_id="tf", source=good_source, target=malformed_target)

    renderer = TypstDocumentRenderer()
    renderer.build_source([block], tmp_path / "build")

    (record,) = renderer.table_reconstructions
    assert record.verified is False
    assert record.corrupt is False


def test_a_degraded_formula_is_recorded_on_the_render_quality_channel(tmp_path: Path) -> None:
    block = _block(BlockType.FORMULA, block_id="f1", source=r"\$\$\undefinedcmd{x}\$\$")

    renderer = TypstDocumentRenderer()
    _source, stats = renderer.build_source([block], tmp_path / "build")

    assert stats.formulas_degraded == 1, stats
    assert stats.formulas_dropped == 0, stats
    assert ("f1", "formula_degraded") in renderer.last_render_flags
    assert all(block_id != "f1" for block_id, _reason in renderer.last_render_skips)


def test_a_dropped_formula_is_a_fail_closed_render_skip(tmp_path: Path) -> None:
    block = _block(BlockType.FORMULA, block_id="fdrop", source="$$")

    renderer = TypstDocumentRenderer()
    _source, stats = renderer.build_source([block], tmp_path / "build")

    assert stats.formulas_dropped == 1, stats
    assert ("fdrop", "math_unrenderable") in renderer.last_render_skips


def test_a_leaked_sentinel_flags_its_own_block(tmp_path: Path) -> None:
    blocks = [
        _block(
            BlockType.NARRATIVE,
            block_id="leaky",
            target=r"扫描 $(\Delta, \llbracket B \rrbracket A)$ 结束",
        ),
        _block(BlockType.NARRATIVE, block_id="clean", target="干净的段落"),
    ]

    renderer = TypstDocumentRenderer()
    _source, stats = renderer.build_source(blocks, tmp_path / "build")

    assert stats.sentinels_leaked == 1, stats
    assert ("leaky", "sentinel_leaked") in renderer.last_render_flags
    assert ("clean", "sentinel_leaked") not in renderer.last_render_flags


def _block_with(*, target: str | None, draft: str | None = None) -> IRBlock:
    return _block(BlockType.NARRATIVE, target=target, draft=draft)


def test_a_failed_repair_still_renders_its_translation() -> None:
    from ubt.render.typst.document_renderer import _block_text

    block = _block_with(
        target=(
            '<mark class="ubt-failed-draft" title="Repair error: Model API error (500)">'
            "我们提出了帧内核。</mark>"
        ),
        draft="我们提出了 framekernel。",
    )
    rendered = _block_text(block)
    assert rendered == "我们提出了帧内核。"
    assert "<mark" not in rendered
    assert "Repair error" not in rendered
    assert "title=" not in rendered


def test_a_quarantine_marker_falls_back_to_the_draft() -> None:
    from ubt.render.typst.document_renderer import _block_text

    block = _block_with(
        target=(
            '<mark class="ubt-blocked-human" title="MQM Critical unresolved">'
            "【待人工审校 | Human review required】We propose framekernel.</mark>"
        ),
        draft="我们提出了帧内核。",
    )
    assert _block_text(block) == "我们提出了帧内核。"


def test_only_the_quarantine_wrapper_counts_as_a_placeholder() -> None:
    from ubt.render.delivered_text import is_placeholder as _is_placeholder

    assert _is_placeholder('<mark class="ubt-blocked-human" title="x">src</mark>') is True
    assert _is_placeholder('<mark class="ubt-failed-draft" title="x">译文</mark>') is False
    assert _is_placeholder("plain prose") is False
    assert _is_placeholder("use the <mark> element") is False


def test_unwrapping_keeps_a_body_containing_its_own_markup() -> None:
    from ubt.render.delivered_text import unwrap_mark as _unwrap_mark

    wrapped = '<mark class="ubt-failed-draft" title="t">a <b>bold</b> b</mark>'
    assert _unwrap_mark(wrapped) == "a <b>bold</b> b"
    assert _unwrap_mark('<mark class="x">unclosed') == '<mark class="x">unclosed'


def test_a_rendered_table_carries_no_inserted_break_characters() -> None:
    from ubt.render.typst.document_renderer import render_table

    markup = (
        "| | Access | Acquire |\n"
        "|---|---|---|\n"
        "| RedLeaf | // lib/devices/ixgbe: ptr::read_volatile(addr) "
        "| // lib/devices/tpm: MmioAddr::new(base,len) |\n"
    )
    out = render_table(markup)
    assert "MmioAddr::\u200bnew" not in out
    assert "read\u200b_volatile" not in out
    assert "MmioAddr::new(base,len)" in out


def test_a_table_that_cannot_fit_its_box_is_refused_when_fit_is_required() -> None:
    from ubt.render.typst.document_renderer import render_table

    markup = "\n".join(
        ["| Index | Measurement |", "|---|---|"] + [f"| item_{i} | value_{i} |" for i in range(10)]
    )

    out = render_table(
        markup,
        content_height_pt=30.0,
        base_font_size_pt=10.0,
        require_fit=True,
    )

    assert out == ""


def test_a_table_with_a_wide_unbreakable_run_is_typeset_smaller() -> None:
    from ubt.render.typst.document_renderer import _TABLE_TEXT_SCALE, render_table

    out = render_table(
        "| | A | B |\n|---|---|---|\n"
        "| R | // lib/devices/tpm: MmioAddr::new(base,len,flags)::from_parts | x |\n",
        content_width_pt=453.55,
    )
    assert f"#set text(size: {_TABLE_TEXT_SCALE}em)" in out


_WIDE_PAGE_PT = 493.2

_NARROW_PAGE_PT = 219.5


def test_a_table_that_fits_its_page_keeps_the_document_size() -> None:
    from ubt.render.typst.document_renderer import render_table

    markup = (
        "| 系统 | 说明 |\n|---|---|\n"
        "| OpenClaw | agentskills.io 兼容；通过 metadata.openclaw.requires 控制 |\n"
    )
    out = render_table(markup, content_width_pt=_WIDE_PAGE_PT)
    assert "#set text(size:" not in out


def test_a_table_that_already_fits_keeps_the_document_size() -> None:
    from ubt.render.typst.document_renderer import render_table

    for markup in (
        "| a | b |\n|---|---|\n| 1 | 2 |\n",
        "| 模块 | 测试数 |\n|---|---|\n| dma | 12 |\n",
    ):
        assert "#set text(size:" not in render_table(markup)


def test_the_size_override_is_scoped_to_the_table() -> None:
    from ubt.render.typst.document_renderer import render_table

    markup = "| " + " | ".join("x" * 30 for _ in range(3)) + " |\n|---|---|---|\n"
    out = render_table(markup, page_columns=1)
    body = out.strip()
    assert body.startswith("#[")
    assert body.endswith("]")
    assert "#set text(size: " in body
    import re

    emitted = float(re.search(r"#set text\(size: ([0-9.]+)em\)", body).group(1))  # type: ignore[union-attr]
    assert emitted < 0.84, body


def test_a_shrunken_table_does_not_resize_the_prose_after_it() -> None:
    pytest.importorskip("pypdfium2")
    import subprocess
    import tempfile

    from ubt.render.typst.document_renderer import render_table

    markup = "| " + " | ".join("y" * 30 for _ in range(3)) + " |\n|---|---|---|\n"
    table = render_table(markup, page_columns=1)
    source = "\n\n".join(
        [
            "#set page(width: 420pt, height: 900pt, margin: 20pt)",
            "#set text(size: 10pt)",
            "BEFORE the table.",
            table,
            "BETWEEN the tables.",
            table,
            "AFTER the tables.",
        ]
    )
    with tempfile.TemporaryDirectory() as tmp:
        typ = Path(tmp) / "main.typ"
        pdf = Path(tmp) / "main.pdf"
        typ.write_text(source, encoding="utf-8")
        result = subprocess.run(
            ["typst", "compile", str(typ), str(pdf)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            pytest.skip(f"typst unavailable: {result.stderr[:200]}")
        import pypdfium2 as pdfium

        doc = pdfium.PdfDocument(str(pdf))
        page = doc[0]
        textpage = page.get_textpage()
        heights: dict[str, float] = {}
        for index in range(textpage.count_rects(0, -1)):
            rect = textpage.get_rect(index)
            chunk = textpage.get_text_bounded(*rect)
            for label in ("BEFORE", "BETWEEN", "AFTER"):
                if chunk.strip() == label or chunk.startswith(label):
                    heights.setdefault(label, round(rect[3] - rect[1], 2))
        doc.close()
    assert heights.keys() >= {"BEFORE", "BETWEEN", "AFTER"}, heights
    assert len(set(heights.values())) == 1, heights


def test_an_identifier_column_gets_room_for_its_run() -> None:
    from ubt.render.typst.document_renderer import render_table

    markup = (
        "| 系统 | 调用方式 | 规范/加载方式 |\n"
        "|---|---|---|\n"
        "| OpenClaw | 作为 XML 注入系统提示；/skills 斜杠命令列出可用技能 | "
        "agentskills.io 兼容；通过 metadata.openclaw.requires（二进制文件、环境变量、"
        "操作系统）进行控制；受管理的安装（Skill Workshop 批准流程） |\n"
    )
    out = render_table(markup, content_width_pt=_NARROW_PAGE_PT)
    spec = next(line for line in out.splitlines() if "columns:" in line)
    widths = [
        float(part.strip().removesuffix("pt"))
        for part in spec.split("(")[1].rstrip("),").split(",")
    ]
    has_shrink = "#set text(size:" in out
    factor = 0.84 if has_shrink else 1.0
    assert widths[2] >= 148.0 * factor * 0.9, spec


def test_a_prose_column_is_never_promoted_to_auto() -> None:
    from ubt.render.typst.document_renderer import render_table

    long_prose = (
        "12-15 个命名章节。静态前缀（身份、规则、工具）通过哈希缓存；"
        "动态后缀（环境、内存、MCP）每次计算，边界标记将两者分离。"
    )
    markup = f"| 阶段 | Claude Code | Codex |\n|---|---|---|\n| 提示词组装 | {long_prose} | 25-30 个内置工具 |\n"
    out = render_table(markup, content_width_pt=_NARROW_PAGE_PT)
    spec = next(line for line in out.splitlines() if "columns:" in line)
    assert "auto" not in spec, spec


def test_cjk_punctuation_closes_an_unbreakable_run() -> None:
    from ubt.render.typst.document_renderer import render_table

    markup = (
        "| 阶段 | Claude Code | Codex |\n|---|---|---|\n"
        "| 结构 | 前缀缓存分离。BOUNDARY 将两者分开 | 25-30 个工具 |\n"
    )
    out = render_table(markup, content_width_pt=_WIDE_PAGE_PT)
    spec = next(line for line in out.splitlines() if "columns:" in line)
    assert "auto" not in spec, spec


def test_a_narrow_figure_is_sized_relative_to_its_column_not_in_points(
    tmp_path: Path,
) -> None:
    from ubt.render.typst.document_renderer import _Figure

    asset = tmp_path / "wide.png"
    asset.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
    renderer = TypstDocumentRenderer()
    build = tmp_path / "build"
    build.mkdir()

    narrow = renderer._figure_markup(_Figure(asset=str(asset), width_pt=290.0), build)
    assert "width: 290pt" not in narrow, "an absolute width ignores the column"
    assert "width: 61.6%" in narrow

    logo = renderer._figure_markup(_Figure(asset=str(asset), width_pt=121.0), build)
    assert "width: 25.7%" in logo

    unknown = renderer._figure_markup(_Figure(asset=str(asset)), build)
    assert f"width: {renderer.figure_width}" in unknown


def test_body_prose_after_a_figure_is_not_rendered_as_its_caption(tmp_path: Path) -> None:
    asset = tmp_path / "fig.png"
    asset.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
    blocks = [
        _block(BlockType.IMAGE, block_id="i1", source=str(asset)),
        _block(
            BlockType.NARRATIVE,
            block_id="body1",
            source="where psi(x, y) is the electrostatic potential in the channel",
            target="其中 ψ(x, y) 是通道中的静电势。",
        ),
    ]
    out, _ = TypstDocumentRenderer().build_source(blocks, tmp_path / "build")
    assert "caption: [" not in out
    assert "其中 ψ(x, y) 是通道中的静电势。" in out


def test_a_formula_crop_never_absorbs_the_next_paragraph(tmp_path: Path) -> None:
    asset = tmp_path / "eq.png"
    asset.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
    blocks = [
        _block(BlockType.IMAGE, block_id="eq1", source=str(asset)),
        _block(
            BlockType.NARRATIVE,
            block_id="lead",
            source="Gauss's law leads to a second important equation:",
            target="高斯定律可导出第二个重要方程：",
        ),
    ]
    out, _ = TypstDocumentRenderer().build_source(blocks, tmp_path / "build")
    assert "caption: [" not in out
    assert "高斯定律可导出第二个重要方程：" in out


def test_a_long_descriptive_caption_still_binds(tmp_path: Path) -> None:
    from ubt.core.ir.models import FlowID, make_element

    asset = tmp_path / "fig.png"
    asset.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
    target = "DG FinFET 的漏极电流与栅极电压关系曲线，" + "并涵盖三种不同的通道掺杂浓度。" * 22
    caption_element = make_element(
        id="c1",
        spine_index=1,
        block_type=BlockType.NARRATIVE,
        flow_id=FlowID.CAPTION,
        source_text="Drain current versus gate voltage of DG FinFETs",
    )
    blocks = [
        _block(BlockType.IMAGE, block_id="i1", source=str(asset)),
        IRBlock(element=caption_element, target_text=target),
    ]
    assert blocks[1].flow_id is FlowID.CAPTION
    assert len(target) > _PENDING_CAPTION_MAX_CHARS
    out, _ = TypstDocumentRenderer().build_source(blocks, tmp_path / "build")
    assert "caption: [" in out


def test_prose_that_opens_with_a_figure_number_is_not_a_caption(tmp_path: Path) -> None:
    asset = tmp_path / "fig.png"
    asset.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
    blocks = [
        _block(BlockType.IMAGE, block_id="i1", source=str(asset)),
        _block(
            BlockType.NARRATIVE,
            block_id="body1",
            source="Fig. 3.1 shows a two-dimensional cross section of a DG FinFET.",
            target=(
                "图 3.1 展示了双栅 FinFET 的二维横截面，该结构被用作模型推导的参考。"
                "在渐变通道近似（GCA）条件下，假设反型载流子服从玻尔兹曼分布。"
            ),
        ),
    ]
    out, _ = TypstDocumentRenderer().build_source(blocks, tmp_path / "build")
    assert "caption: [" not in out


def test_two_figures_sharing_a_basename_both_survive(tmp_path: Path) -> None:
    from PIL import Image

    from ubt.core.ir.models import BlockType, BoundingBox, FlowID, IRBlock, make_element

    for sub, colour in (("a", "red"), ("b", "blue")):
        (tmp_path / sub).mkdir()
        Image.new("RGB", (300, 100), colour).save(tmp_path / sub / "figure.png")

    def block(path: Path, index: int, y: float) -> IRBlock:
        return IRBlock(
            element=make_element(
                id=f"pdf_main#img_{index:04d}",
                spine_index=index,
                block_type=BlockType.IMAGE,
                flow_id=FlowID.CAPTION,
                source_text=str(path),
                bbox=BoundingBox(page=1, x0=10.0, y0=y, x1=200.0, y1=y + 80.0),
                skip_translate=True,
            ),
            target_text=str(path),
        )

    blocks = [
        block(tmp_path / "a" / "figure.png", 1, 10.0),
        block(tmp_path / "b" / "figure.png", 2, 200.0),
    ]
    build = tmp_path / "build"
    source, _stats = TypstDocumentRenderer().build_source(blocks, build)

    copied = sorted(p.name for p in build.iterdir() if p.suffix == ".png")
    assert len(copied) == 2, f"one picture overwrote the other: {copied}"
    payloads = {(build / name).read_bytes() for name in copied}
    assert len(payloads) == 2, "both figures must carry distinct bytes"
    red = (tmp_path / "a" / "figure.png").read_bytes()
    blue = (tmp_path / "b" / "figure.png").read_bytes()
    assert payloads == {red, blue}
    for name in copied:
        assert f'image("{name}"' in source


def test_render_table_dense_single_col_landscape_contingency() -> None:
    header = "| " + " | ".join(f"Col{i}" for i in range(9)) + " |"
    sep = "| " + " | ".join("---" for _ in range(9)) + " |"
    row = "| " + " | ".join(f"Value{i}" for i in range(9)) + " |"
    md = f"{header}\n{sep}\n{row}\n"

    out_default = render_table(md, page_columns=1, allow_landscape=False)
    assert out_default.strip().startswith("#[")
    assert "#page(flipped: true)" not in out_default

    out_landscape = render_table(md, page_columns=1, allow_landscape=True)
    assert out_landscape.strip().startswith("#page(flipped: true)[")
    assert out_landscape.strip().endswith("]")


def test_typst_document_renderer_dense_table_landscape(tmp_path: Path) -> None:
    from ubt.core.ir.models import BlockType, BoundingBox, FlowID, IRBlock, make_element
    from ubt.render.typst.document_renderer import TypstDocumentRenderer

    header = "| " + " | ".join(f"Col{i}" for i in range(9)) + " |"
    sep = "| " + " | ".join("---" for _ in range(9)) + " |"
    row = "| " + " | ".join(f"Value{i}" for i in range(9)) + " |"
    table_md = f"{header}\n{sep}\n{row}\n"

    table_block = IRBlock(
        element=make_element(
            id="t1",
            spine_index=1,
            block_type=BlockType.TABLE,
            flow_id=FlowID.TABLE_GRID,
            source_text=table_md,
            bbox=BoundingBox(page=1, x0=10.0, y0=10.0, x1=400.0, y1=100.0),
        ),
        target_text=table_md,
    )

    build_dir = tmp_path / "build"

    renderer_default = TypstDocumentRenderer(columns=1)
    source_default, _ = renderer_default.build_source([table_block], build_dir)
    assert "#page(flipped: true)[" in source_default

    renderer_no_ls = TypstDocumentRenderer(columns=1, allow_landscape_tables=False)
    source_no_ls, _ = renderer_no_ls.build_source([table_block], build_dir)
    assert "#page(flipped: true)[" not in source_no_ls

    renderer_2col = TypstDocumentRenderer(columns=2)
    source_2col, _ = renderer_2col.build_source([table_block], build_dir)
    assert "#page(flipped: true)[" not in source_2col

    renderer_bilingual = TypstDocumentRenderer(columns=1, bilingual=True)
    source_bilingual, _ = renderer_bilingual.build_source([table_block], build_dir)
    assert "#page(flipped: true)[" not in source_bilingual

    renderer_academic = TypstDocumentRenderer(columns=1, layout_template="academic")
    source_academic, _ = renderer_academic.build_source([table_block], build_dir)
    assert "#page(flipped: true)[" not in source_academic


def test_prose_with_kinsoku_enabled(tmp_path: Path) -> None:
    from ubt.core.cleaners.kinsoku import WORD_JOINER

    block = _block(
        BlockType.NARRATIVE,
        block_id="p1",
        source="System architecture and core modules.",
        target="技术架构，核心模块。",
    )
    renderer = TypstDocumentRenderer(enable_kinsoku=True)
    source, _ = renderer.build_source([block], tmp_path / "build")
    assert f"构{WORD_JOINER}，" in source
    assert f"块{WORD_JOINER}。" in source


def test_renderer_with_colophon_enabled(tmp_path: Path) -> None:
    block = _block(
        BlockType.HEADING,
        block_id="h1",
        source="Distributed Systems Principles",
        target="分布式系统原理",
    )
    renderer = TypstDocumentRenderer(enable_colophon=True, paper_title="分布式系统原理")
    source, _ = renderer.build_source([block], tmp_path / "build")
    assert "图书在版编目（CIP）数据" in source
    assert "分布式系统原理" in source


def test_renderer_with_index_enabled(tmp_path: Path) -> None:
    blocks = [
        _block(
            BlockType.HEADING,
            block_id="h1",
            source="Foundations",
            target="基础理论",
        ),
        _block(
            BlockType.NARRATIVE,
            block_id="p1",
            source="We discuss Consensus Algorithm and Replication.",
            target="我们讨论共识算法（Consensus Algorithm）以及复制技术。",
        ),
    ]
    renderer = TypstDocumentRenderer(enable_index=True)
    source, _ = renderer.build_source(blocks, tmp_path / "build")
    assert "#heading(level: 1, numbering: none)[索引 (Index)]" in source
    assert "#columns(3" in source
    assert "*Consensus Algorithm* (共识算法)" in source


def test_renderer_with_decor_enabled(tmp_path: Path) -> None:
    blocks = [
        _block(
            BlockType.HEADING,
            block_id="h1",
            source="Chapter 1: Distributed Storage",
            target="第 1 章：分布式存储架构",
        ),
        _block(
            BlockType.NARRATIVE,
            block_id="p1",
            source="In modern computing systems, scalability is paramount.",
            target="在现代计算系统中，可扩展性是核心设计指标。",
        ),
        _block(
            BlockType.NARRATIVE,
            block_id="p2",
            source="Another consideration is data persistence.",
            target="另一个重要考量是数据持久性。",
        ),
        _block(
            BlockType.HEADING,
            block_id="h2",
            source="Chapter 2: Replication Protocols",
            target="第 2 章：副本复制协议",
        ),
        _block(
            BlockType.NARRATIVE,
            block_id="p3",
            source="“Replication guarantees fault tolerance.”",
            target="“副本技术保障了系统的容错能力。”",
        ),
    ]
    from ubt.core.decor.models import (
        ChapterHeaderConfig,
        DecorSpec,
        DropCapConfig,
        HeaderStyle,
    )

    spec = DecorSpec(
        header=ChapterHeaderConfig(style=HeaderStyle.CLASSIC),
        drop_cap=DropCapConfig(lines=2, color="#1e3a8a"),
    )
    renderer = TypstDocumentRenderer(enable_decor=True, decor_spec=spec)
    source, _ = renderer.build_source(blocks, tmp_path / "build")

    assert "#let ubt-dropcap" in source
    assert "第 1 章" in source
    assert "分布式存储架构" in source
    assert "❦" in source
    assert '#ubt-dropcap("在", prefix: "", lines: 2, fill: rgb("#1e3a8a"))' in source
    assert '#ubt-dropcap("另"' not in source
    assert "第 2 章" in source
    assert "副本复制协议" in source
    assert '#ubt-dropcap("副", prefix: "“", lines: 2, fill: rgb("#1e3a8a"))' in source


def test_renderer_with_notes_enabled(tmp_path: Path) -> None:
    from ubt.core.notes.models import NotePlacement, NotesConfig

    blocks = [
        _block(
            BlockType.HEADING,
            block_id="h1",
            source="Chapter 1: Consistency",
            target="第 1 章：强一致性",
        ),
        _block(
            BlockType.NARRATIVE,
            block_id="p1",
            source="Linearizability[^1] ensures atomic visibility.",
            target="线性一致性[^1]保障原子可见性。",
        ),
        _block(
            BlockType.NARRATIVE,
            block_id="fn1",
            source="[^1]: Herlihy & Wing, 1990.",
            target="[^1]: Herlihy 与 Wing 于 1990 年提出。",
        ),
    ]

    renderer_fn = TypstDocumentRenderer(enable_notes=True)
    source_fn, _ = renderer_fn.build_source(blocks, tmp_path / "build_fn")
    assert "#footnote[Herlihy 与 Wing 于 1990 年提出。]" in source_fn
    assert "线性一致性#footnote" in source_fn

    cfg_end = NotesConfig(placement=NotePlacement.CHAPTER_ENDNOTE)
    renderer_end = TypstDocumentRenderer(enable_notes=True, notes_config=cfg_end)
    source_end, _ = renderer_end.build_source(blocks, tmp_path / "build_end")
    assert "#link(<fn-ch1-1>)[#super[1]]" in source_end
    assert "#heading(level: 2)[注释]" in source_end
    assert "Herlihy 与 Wing 于 1990 年提出。" in source_end


def test_renderer_with_backmatter_enabled(tmp_path: Path) -> None:
    from ubt.core.backmatter.models import BackmatterConfig

    blocks = [
        _block(
            BlockType.HEADING,
            block_id="h1",
            source="Chapter 1: Foundations",
            target="第 1 章：基础理论",
        ),
        _block(
            BlockType.NARRATIVE,
            block_id="p1",
            source="Main prose content.",
            target="正文基础理论描述。",
        ),
        _block(
            BlockType.HEADING,
            block_id="app1",
            source="Appendix: Proofs",
            target="附录：定理证明",
        ),
        _block(
            BlockType.NARRATIVE,
            block_id="p_app",
            source="Proof content.",
            target="证明细节内容。",
        ),
        _block(
            BlockType.HEADING,
            block_id="bib",
            source="References",
            target="参考文献",
        ),
        _block(
            BlockType.NARRATIVE,
            block_id="p_bib",
            source="[1] Reference item.",
            target="[1] 参考文献条目。",
        ),
    ]

    cfg = BackmatterConfig(appendix_numbering="A.1")
    renderer = TypstDocumentRenderer(enable_backmatter=True, backmatter_config=cfg)
    source, _ = renderer.build_source(blocks, tmp_path / "build_bm")

    assert "#counter(heading).update(0)" in source
    assert '#set heading(numbering: "A.1")' in source
    assert "附录 A：定理证明" in source
    assert "#heading(level: 1, numbering: none)[参考文献]" in source


def test_renderer_with_grid_figures_enabled(tmp_path: Path) -> None:
    from ubt.core.figures.models import FigureLayoutConfig

    img_dir = tmp_path / "img_assets"
    img_dir.mkdir()
    f1 = img_dir / "panel_a.png"
    f2 = img_dir / "panel_b.png"
    f1.write_bytes(b"PNG_FAKE_DATA_1")
    f2.write_bytes(b"PNG_FAKE_DATA_2")

    blocks = [
        _block(
            BlockType.NARRATIVE,
            block_id="p1",
            source="Here is the experiment comparison:",
            target="以下是实验对比：",
        ),
        _block(
            BlockType.IMAGE,
            block_id="img1",
            source=str(f1),
            target=str(f1),
        ),
        _block(
            BlockType.IMAGE,
            block_id="img2",
            source=str(f2),
            target=str(f2),
        ),
        _block(
            BlockType.NARRATIVE,
            block_id="cap",
            source="Figure 2: Performance analysis. (a) Latency. (b) Memory.",
            target="图 2：性能分析。（a）延迟 （b）内存占用。",
        ),
        _block(
            BlockType.NARRATIVE,
            block_id="p2",
            source="Subsequent discussion text.",
            target="后续正文讨论。",
        ),
    ]

    cfg = FigureLayoutConfig(columns=2, gutter_pt=12.0)
    renderer = TypstDocumentRenderer(
        assets_root=img_dir,
        enable_grid_figures=True,
        figure_config=cfg,
    )
    source, stats = renderer.build_source(blocks, tmp_path / "build_grid")

    assert "grid(" in source
    assert "columns: (1fr, 1fr)" in source
    assert "gutter: 12.0pt" in source
    assert "*(a)*" in source
    assert "*(b)*" in source
    assert "caption: [图 2：性能分析。]" in source
    assert stats.figures == 2


def test_renderer_with_table_continuation_enabled(tmp_path: Path) -> None:
    from ubt.core.tables.models import TableContinuationConfig

    md_table = (
        "| 参数 | 基线 | 优化版 |\n"
        "| --- | --- | --- |\n"
        "| 吞吐量 | 100 req/s | 550 req/s |\n"
        "| P99 延迟 | 45ms | 8ms |\n"
        "| 内存开销 | 2.1GB | 680MB |\n"
        "| 错误率 | 0.05% | 0.001% |\n"
        "| 恢复时间 | 30s | 1.2s |\n"
    )

    blocks = [
        _block(
            BlockType.NARRATIVE,
            block_id="p1",
            source="System performance comparison across versions:",
            target="各版本系统性能对比总结：",
        ),
        _block(
            BlockType.TABLE,
            block_id="tbl1",
            source=md_table,
            target=md_table,
        ),
        _block(
            BlockType.NARRATIVE,
            block_id="p2",
            source="Further analysis follow up.",
            target="后续深度分析正文。",
        ),
    ]

    cfg = TableContinuationConfig(min_rows_for_continuation=4)
    renderer = TypstDocumentRenderer(
        enable_table_continuation=True,
        table_continuation_config=cfg,
    )
    source, stats = renderer.build_source(blocks, tmp_path / "build_tbl")

    assert "<ubt-tbl-start-tgt-tbl1>" in source
    assert "<ubt-tbl-end-tgt-tbl1>" in source
    assert "table.header(" in source
    assert "（续表）" in source
    assert "table.footer(" in source
    assert "（续下页）" in source
    assert stats.tables == 1


def test_renderer_with_epigraph_enabled(tmp_path: Path) -> None:
    from ubt.core.decor.models import DecorSpec, DropCapConfig
    from ubt.core.epigraph.models import EpigraphConfig, EpigraphStyle

    blocks = [
        _block(
            BlockType.HEADING,
            block_id="h1",
            source="Chapter 1: The Essence of Programming",
            target="第 1 章：编程的本质",
            level=1,
        ),
        _block(
            BlockType.NARRATIVE,
            block_id="epi",
            source="Programs must be written for people to read. —— Abelson & Sussman",
            target="代码主要是为了让人类阅读而编写的。 —— 阿贝尔森与萨斯曼",
        ),
        _block(
            BlockType.NARRATIVE,
            block_id="abs",
            source="**Chapter Summary**: Overview of abstraction and functional state.",
            target="**本章导读**：本章探讨抽象屏障与函数式编程状态机制。",
        ),
        _block(
            BlockType.NARRATIVE,
            block_id="p1",
            source="Computer science is a magical discipline.",
            target="计算机科学是一门充满魔力的学科。",
        ),
    ]

    epi_cfg = EpigraphConfig(style=EpigraphStyle.CLASSIC_RIGHT)
    decor_cfg = DecorSpec(drop_cap=DropCapConfig(enabled=True, lines=3))

    renderer = TypstDocumentRenderer(
        enable_epigraph=True,
        epigraph_config=epi_cfg,
        enable_decor=True,
        decor_spec=decor_cfg,
    )
    source, stats = renderer.build_source(blocks, tmp_path / "build_epi")

    assert "#align(right)[" in source
    assert "代码主要是为了让人类阅读而编写的。" in source
    assert "阿贝尔森与萨斯曼" in source

    assert "fill: luma(248)" in source
    assert "本章导读" in source
    assert "抽象屏障与函数式编程" in source

    assert '#ubt-dropcap("计", prefix: "", lines: 3' in source
    assert stats.blocks_rendered >= 3
