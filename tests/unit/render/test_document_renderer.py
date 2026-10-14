"""The semantic reflow renderer: IRBlocks → one Typst document.

The renderer is the piece that makes the RFC's product claims real — a uniform
body size, native tables, correct math — because it rebuilds the document
instead of drawing fragments onto the source page. Its failure modes are
therefore page-level, not local: an escaped ``$`` turns a formula into literal
text, a missing asset leaves a hole, an unhandled block type silently drops
content. These tests pin each of those on the emitted *source*, so they run
without the Typst binary.

The bugs these were written against were all found by measuring the reference
paper, and each is named in the test that pins it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ubt.core.ir.models import BlockType, InlineRun, IRBlock, make_element
from ubt.render.typst.document_renderer import (
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


# --- prose and math --------------------------------------------------------


def test_math_is_transpiled_not_escaped(tmp_path: Path) -> None:
    """`$x_i$` must become real Typst math, not literal ``\\$x_i\\$``."""
    out = render_prose("Let $x_i$ be the input.")
    assert "$x_i$" in out
    assert r"\$" not in out


def test_prose_markup_is_escaped(tmp_path: Path) -> None:
    """A ``#`` or ``_`` in prose is text, not Typst syntax."""
    out = render_prose("Use a_b and #tag here.")
    assert r"a\_b" in out
    assert r"\#tag" in out


def test_a_failed_math_conversion_falls_back_to_escaped_text(tmp_path: Path) -> None:
    """An un-convertible span must not leave an unbalanced ``$``.

    A stray delimiter unbalances the paragraph and Typst then fails the whole
    page, so the span degrades to literal text instead.
    """
    out = render_prose("Broken $\\undefinedcmd{x}$ end.")
    assert out.count("$") % 2 == 0


def test_display_math_gets_its_own_block(tmp_path: Path) -> None:
    out = render_prose(r"See $$E = mc^2$$ here.")
    assert "\n$ " in out


def test_a_math_body_the_compiler_rejects_degrades_instead_of_failing(tmp_path: Path) -> None:
    """A body the transpiler accepts but Typst rejects must not reach the page.

    ``\\frac{a}{}`` converts to ``frac(a, )``, which parses in the transpiler's
    own grammar but is ``error: expected expression`` in Typst. One such body
    fails the *whole* document, so the compile gate degrades the span to
    readable Unicode the same way an unconvertible span does.
    """
    rejected = "frac(a, )"

    def gate(body: str) -> bool:
        return body != rejected

    out = render_prose(r"Bad $\frac{a}{}$ here.", is_math_ok=gate)
    assert rejected not in out
    assert out.count("$") % 2 == 0

    # Without the gate the body is emitted verbatim (the pre-fix behaviour the
    # gate exists to catch).
    ungated = render_prose(r"Bad $\frac{a}{}$ here.")
    assert rejected in ungated


def test_a_renderer_math_gate_reaches_every_prose_call_site(tmp_path: Path) -> None:
    """The renderer threads its gate through headings, prose and captions."""
    renderer = TypstDocumentRenderer(math_gate=lambda body: body != "frac(a, )")
    blocks = [
        _block(BlockType.HEADING, block_id="h1", source="Head", target=r"Head $\frac{a}{}$"),
        _block(BlockType.NARRATIVE, block_id="p1", source="Body", target=r"Body $\frac{a}{}$"),
    ]
    out, _stats = renderer.build_source(blocks, tmp_path / "build")
    assert "frac(a, )" not in out


def test_the_gate_reaches_the_formula_block_branch(tmp_path: Path) -> None:
    """P1-C5: a *formula* block skips the prose transpiler path.

    The formula branch transpiles the block body directly, so the prose-side
    gate never ran on it: a body the transpiler accepts but Typst rejects
    (``\\frac{a}{}`` -> ``frac(a, )``) was emitted verbatim and failed the whole
    document at the single compile. The gate must degrade it to readable Unicode
    here too, exactly as it does inside prose.
    """
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
    # Degraded rather than counted as a successful formula transpile.
    assert stats.formulas == 0


def test_an_acceptable_formula_body_still_compiles_under_the_gate(tmp_path: Path) -> None:
    """The gate must not reject a formula Typst actually accepts."""
    renderer = TypstDocumentRenderer(math_gate=lambda body: body != "frac(a, )")
    blocks = [
        _block(BlockType.FORMULA, block_id="f1", source=r"$$\frac{a}{b}$$", target=""),
    ]
    out, stats = renderer.build_source(blocks, tmp_path / "build")
    assert "frac(a, b)" in out
    assert stats.formulas == 1


# --- block types -----------------------------------------------------------


def test_a_heading_carries_its_level(tmp_path: Path) -> None:
    """Docling flattens every heading to level 1; the outline supplies depth."""
    blocks = [
        _block(BlockType.HEADING, block_id="h1", source="1 Introduction", target="1 引言", level=1),
        _block(BlockType.HEADING, block_id="h2", source="2.1 Routing", target="2.1 路由", level=2),
    ]
    out = _render(blocks, tmp_path)
    assert "#heading(level: 1)[1 引言]" in out
    assert "#heading(level: 2)[2.1 路由]" in out


def test_an_arxiv_stamp_heading_is_dropped(tmp_path: Path) -> None:
    """Running chrome is not body content."""
    blocks = [_block(BlockType.HEADING, source="arXiv:2101.03961v3 [cs.LG] 16 Jun 2022")]
    out = _render(blocks, tmp_path)
    assert "arXiv:2101" not in out


def test_a_table_is_rebuilt_with_translated_headers(tmp_path: Path) -> None:
    markup = "| Model | Quality |\n|---|---|\n| T5-Base | -1.73 |"
    out = render_table(markup)
    assert "#table(" in out
    assert "模型" not in out  # header comes from the caller's markup verbatim
    assert "[*Model*]" in out


def test_a_formula_block_is_typeset_as_math(tmp_path: Path) -> None:
    """A formula block is skip-translate but must NOT be pasted as source."""
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
    """A localized formula annotation must be typeset, not the source one.

    The formula branch read only ``source_text``, so a model that translated an
    annotation inside the formula (``\\text{parameter}`` -> ``\\text{参数}``) had
    its work dropped — the reader got the source-language annotation back with no
    count. It now follows the target→draft→source priority every other block
    uses, and counts the target-sourced renders.
    """
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
    """The common case (no localization) keeps reading the source, uncounted."""
    blocks = [
        _block(BlockType.FORMULA, block_id="f2", source=r"$$ E = mc^2 $$", skip=True),
    ]
    _out, stats = _render_with_stats(blocks, tmp_path)
    assert stats.formulas == 1
    assert stats.formulas_from_target == 0


def test_a_bibliography_entry_is_kept_in_source(tmp_path: Path) -> None:
    """A policy keep must render its source, never a stale draft."""
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


# --- fail-open and placeholders --------------------------------------------


def test_a_quarantine_placeholder_never_reaches_the_page(tmp_path: Path) -> None:
    """The ``<mark class="ubt-blocked-human">`` wrapper is markup, not content.

    Measured on the reference run: rendering it produced 36 occurrences of the
    placeholder text in the artifact.
    """
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
    # The real draft is shipped instead of the placeholder.
    assert "为何选择 Switch Transformer" in out


def test_a_block_with_no_translation_ships_its_source(tmp_path: Path) -> None:
    blocks = [_block(BlockType.NARRATIVE, source="Untranslated sentence.")]
    out = _render(blocks, tmp_path)
    assert "Untranslated sentence." in out


# --- figures ---------------------------------------------------------------


def test_a_missing_asset_leaves_a_visible_placeholder(tmp_path: Path) -> None:
    """A dropped figure must be attributable, not a silent hole."""
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


# --- bilingual pairing -----------------------------------------------------


def test_bilingual_pairs_the_source_with_its_translation(tmp_path: Path) -> None:
    """`publish` + a dual mode must emit the source above the translation.

    This is the behaviour the overlay path had and publish silently lacked: a
    `--render-mode publish --dual-mode facing` run produced a monolingual PDF
    with no warning, so the user paid for a bilingual book and got a mono one.
    """
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
    """With bilingual off the source must not leak into the page."""
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
    """A block with no target renders once, not source-above-source.

    An untranslated paragraph keeps its source as the body, so pairing it would
    print the same sentence twice — the failure mode a naive
    "always emit the source line" implementation would ship.
    """
    blocks = [_block(BlockType.NARRATIVE, block_id="p1", source="Untranslated text.")]
    renderer = TypstDocumentRenderer(bilingual=True)
    out, _stats = renderer.build_source(blocks, tmp_path / "build")

    assert out.count("Untranslated text.") == 1


def test_bilingual_does_not_repeat_a_source_kept_block(tmp_path: Path) -> None:
    """A bibliography entry keeps its source as both sides; pairing would double it."""
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
    """Every translatable block type pairs, not just prose.

    A bilingual book whose headings and table headers stayed monolingual would
    be worse than useless for the reader comparing the two languages.
    """
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
    # The source half of a table pair is *also* a table. Rendering it as prose
    # pasted the Markdown grid onto the page as italic grey text; both halves
    # must now compile as tables, so no bare grid row may survive.
    assert "| A | B |" not in out
    assert out.count("#table(") == 2


# --------------------------------------------------------------------------- #
# Styled runs must not swallow a following parenthesis
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("text", "run", "expect"),
    [
        # A superscript run ends in ``]``; a literal ``(`` right after it parses
        # as a call on the run's result. Measured on the gold corpus (``gan``):
        # ``lo#super[...][g](4)`` failed to compile and the whole block was lost.
        ("lo g(4)", InlineRun(text="g", superscript=True), '#text("(")'),
        # A bold run before a citation's opening paren (arXiv 2609.22978 b0039).
        (
            "Firecracker microVMs (Agache et al., 2020)",
            InlineRun(text="Firecracker microVMs ", bold=True),
            '#text("(")',
        ),
    ],
)
def test_a_styled_run_never_swallows_a_following_paren(
    text: str, run: InlineRun, expect: str
) -> None:
    out = render_prose(text, (run,))
    assert expect in out, f"the paren after a styled run must be escaped: {out!r}"


def test_an_unstyled_paren_is_left_alone() -> None:
    # The guard must be narrow: a plain ``(`` with no styled run before it is
    # ordinary prose and must not gain markup.
    assert "#text(" not in render_prose("plain (paren)", ())


# --------------------------------------------------------------------------- #
# A masker token must never be transpiled as math
# --------------------------------------------------------------------------- #


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
    """Typst math mode turns the token's underscores into code-mode subscripts.

    Measured on the gold corpus: a restore that could not account for a token
    left ``⟦SOUP_MASK_0001-333⟧`` inside a ``$...$`` span, ``typstify_math``
    emitted ``⟦#"SOUP"_#"MASK"_0001-333⟧``, and the compiler failed with
    ``unknown variable: SOUP`` — losing the entire document (alphafold2 and
    attention-is-all-you-need both hit it). The token renders literally instead.
    """
    out = render_prose(f"$a_i : {token} b_j$", ())

    assert '"SOUP"' not in out and '"MASK"' not in out, f"code mode leaked: {out!r}"
    assert "#" not in out, f"a bare # would enter code mode: {out!r}"
    # Escaped as prose, the token's underscores are literal.
    assert token.replace("_", "\\_") in out


def test_a_mask_token_in_display_math_is_not_transpiled() -> None:
    out = render_prose("$$x = ⟦MATH_MASK_0002-abc⟧$$", ())
    assert "#" not in out, out
    assert "\\_MASK\\_" in out


def test_real_math_still_transpiles_when_no_token_is_present() -> None:
    # The guard must be narrow: an ordinary formula keeps its math rendering.
    out = render_prose("$x^2 + y^2 = z^2$", ())
    assert out.startswith("$") and out.endswith("$"), out
    assert "#" not in out


def test_a_mask_token_in_prose_is_escaped_as_before() -> None:
    # Prose already escaped correctly; the fix must not change that path.
    out = render_prose("prose ⟦CITE_MASK_0001-abc⟧ text", ())
    assert out == "prose ⟦CITE\\_MASK\\_0001-abc⟧ text"


def test_a_leaked_emphasis_marker_inside_math_does_not_split_the_formula() -> None:
    """A stray ``⟦/B⟧`` in a formula made Typst read ``/`` as the fraction operator.

    Measured on the gold corpus (mamba ``pdf_main#b0426``, neural-ode): the
    restore left ``⟦ B ⟧ A ⟦ /B ⟧`` inside a ``$...$`` span and the compiler
    failed with ``unexpected slash`` — losing the whole document.
    """
    out = render_prose(r"扫描输入 $(\Delta, ⟦ B ⟧ A ⟦ /B ⟧)$ 结束", ())

    # Nothing entered math mode, so no bare delimiter remains as a switch.
    assert out.count("$") == 0, f"a delimiter stayed a math switch: {out!r}"
    # The marker is visible rather than silently deleted: a reader can attribute
    # the failed restore.
    assert "⟦" in out and "⟧" in out


def test_a_latex_transcribed_emphasis_marker_keeps_its_backslashes() -> None:
    """A model's ``\\llbracket`` must stay visible, not degrade to ``llbracket``.

    Rendering the inner text verbatim (rather than through
    ``clean_math_fallback_text``, which strips backslashes) keeps the token
    honest: a reader sees what the model wrote, and the leak stays attributable.
    """
    out = render_prose(r"$(\Delta, \llbracket B \rrbracket A \llbracket /B \rrbracket)$", ())

    assert "llbracket" in out and "rrbracket" in out
    assert "\\llbracket" in out, f"the backslash must survive: {out!r}"
    assert out.count("$") == 0, f"a delimiter stayed a math switch: {out!r}"


def test_the_leaked_sentinel_count_is_reported(tmp_path: Path) -> None:
    """A leaked sentinel must reach the quality report, not just compile.

    Escaping the marker keeps the document compiling, which is the point — but
    a silent fix would hide the upstream restore defect. The stat makes it
    countable.
    """
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


# --------------------------------------------------------------------------- #
# A styled run must never split a math span
# --------------------------------------------------------------------------- #


def test_a_run_straddling_a_math_boundary_does_not_split_the_formula() -> None:
    """A run that starts inside a formula and ends outside must be dropped.

    Measured on the gold corpus (alphafold2, attention-is-all-you-need): a
    superscript run whose tolerant match reached past the closing ``$`` split the
    span, each ``$`` was escaped to a literal ``\\$``, and the formula shipped as
    backslashes — the document then failed to compile with an unbalanced
    delimiter. Only *full containment* used to be checked, so this case slipped
    through.
    """
    text = r"$\{ x_k \} \to 0$是一个序列"
    runs = (InlineRun(text=r"x_k \} \to 0$是", superscript=True),)
    out = render_prose(text, runs)

    assert out.count("$") == 2, f"the formula must keep both delimiters: {out!r}"
    assert "\\$" not in out, f"a delimiter was escaped as literal text: {out!r}"
    assert "#super" not in out, f"the straddling run must be dropped: {out!r}"


def test_a_run_fully_inside_a_formula_is_still_dropped() -> None:
    # The pre-existing rule, pinned so the widened predicate cannot regress it.
    text = r"$\{ x_k \} \to 0$是一个序列"
    out = render_prose(text, (InlineRun(text="x_k", superscript=True),))
    assert out.count("$") == 2
    assert "#super" not in out


def test_a_run_beside_a_formula_still_styles_its_own_text() -> None:
    # The predicate must stay narrow: a run that does not touch math still works.
    out = render_prose(r"前 $x_i$ 后", (InlineRun(text="前", bold=True),))
    assert "#strong[前]" in out
    assert out.count("$") == 2


def test_a_run_ending_exactly_at_the_opening_delimiter_still_styles() -> None:
    # Touching the boundary from outside is not an overlap: ``前`` ends where
    # ``$`` begins. Half-open interval logic must keep this styled.
    out = render_prose(r"前$x_i$", (InlineRun(text="前", bold=True),))
    assert "#strong[前]" in out, out
    assert out.count("$") == 2


# --------------------------------------------------------------------------- #
# An unbalanced `$` must not swallow prose into math mode
# --------------------------------------------------------------------------- #


def test_a_stray_dollar_does_not_swallow_prose_into_math() -> None:
    """A model's extra ``$`` must not turn a sentence into a formula.

    Measured on the gold corpus (alphafold2 ``pdf_main#b0146``): the model
    returned seven ``$`` for six delimiters. ``_MATH_SPAN_RE`` pairs positionally,
    so the stray one made ``$ 是一个序列，其中 $`` a math span and Typst failed
    with ``unknown variable: 是一个序列`` — losing the whole document.
    """
    text = r"{ x_k } \to 0$ 是一个序列，其中 $x_k \in \Omega$。"
    out = render_prose(text, ())

    # Every delimiter is literal, so nothing entered math mode.
    assert out.count("\\$") == text.count("$"), f"a delimiter stayed a math switch: {out!r}"
    # No bare `$` survives as a math-mode switch (every one is backslash-escaped).
    assert out.replace("\\$", "").count("$") == 0, f"a bare $ remained: {out!r}"
    assert "是一个序列" in out


def test_an_unclosed_trailing_dollar_keeps_the_earlier_pair() -> None:
    text = r"$x$ 和 $y"
    out = render_prose(text, ())
    assert out.startswith("$x$"), f"the balanced pair must stay math: {out!r}"
    assert out.endswith("\\$y")


def test_a_cjk_free_span_is_still_math() -> None:
    # The guard must be narrow: real formulas keep transpiling.
    assert render_prose(r"a $x_i + y^2$ b", ()) == r"a $x_i + y^2$ b"


def test_a_display_formula_with_cjk_inside_is_prose() -> None:
    # A ``$$`` span containing CJK is prose wearing delimiters, not math.
    out = render_prose("$$这段中文$$", ())
    assert "\\$" in out


# --------------------------------------------------------------------------- #
# Table cells: math must transpile, and a double escape must never happen
# --------------------------------------------------------------------------- #


def test_a_table_cell_with_math_transpiles_it() -> None:
    """A cell's ``$...$`` must become real math, not literal ``\\$``.

    Measured on the gold corpus (attention-is-all-you-need ``pdf_main#b0062``,
    the layer-complexity table): cells were pre-escaped by
    ``parse_markdown_table`` and then escaped *again* by ``render_prose``, so
    ``$`` became ``\\\\$``. Inside a ``[...]`` content block ``\\\\`` is a line
    break, so the delimiter never closed and Typst failed with
    ``unclosed delimiter`` — losing the whole document.
    """
    out = render_table("| a | b |\n|---|---|\n| 循环 | O($n \\cdot d^2$) |")

    # The cell is inside a math span (``$...$``) and the LaTeX was transpiled:
    # ``\\cdot`` became Typst's ``dot``.
    assert "$n dot d^2$" in out, f"the cell's math must stay math: {out!r}"
    assert "\\\\$" not in out, f"a double-escaped delimiter would break Typst: {out!r}"


def test_a_table_cell_without_math_is_escaped_once() -> None:
    # Escaping must still happen exactly once for ordinary prose cells.
    out = render_table("| a |\n|---|\n| 100% [1] |")
    assert "\\[1\\]" in out or "\\[1\\]" in out
    assert "\\\\[" not in out, f"a double-escaped bracket: {out!r}"


def test_parse_markdown_table_can_return_raw_cells() -> None:
    """The escape contract is explicit, because two callers need different things.

    ``markdown_table_to_typst`` interpolates cells raw, so it needs escaping.
    The publication renderer runs cells through ``render_prose`` (which owns
    escaping and math), so it needs the raw text.
    """
    from ubt.render.table_compiler import parse_markdown_table

    md = "| a |\n|---|\n| O($n$) |"
    escaped = parse_markdown_table(md)
    raw = parse_markdown_table(md, escape=False)
    assert escaped is not None and raw is not None
    assert escaped[1][0][0] == "O(\\$n\\$)"
    assert raw[1][0][0] == "O($n$)"


# --------------------------------------------------------------------------- #
# A formula that annotates itself in Chinese is still math
# --------------------------------------------------------------------------- #


def test_a_formula_with_a_chinese_text_annotation_stays_math() -> None:
    r"""``$\mathbf{A} : ( D , N ) \text{ 参数 }$`` is math, not prose.

    Rejecting every CJK-bearing span (the fix for the unbalanced ``$`` case)
    was too broad: a translated document routinely annotates formula parts in
    Chinese. Measured on mamba ``pdf_main#b0086``, the whole formula was escaped
    as prose and shipped as literal backslashes
    (``\$\\mathbf\{A\} \text\{ )← 参数 \}\$``).
    """
    out = render_prose(r"$\mathbf{A} : ( D , N ) \text{ )← 参数 } \vdash N \times N$", ())

    # It transpiled — the LaTeX commands became Typst math (``bold``, ``tack.r``)
    # and the delimiters are real math switches. Escaped prose would still carry
    # the backslashes.
    assert "bold(A)" in out, out
    assert "\\mathbf" not in out and "\\text" not in out, out
    assert "\\{" not in out, out


def test_a_formula_made_only_of_chinese_words_is_prose() -> None:
    # The original repair must still hold: prose wearing delimiters is prose.
    out = render_prose("$ 是一个序列，其中 $", ())
    assert "$" not in out.replace("\\$", ""), out


def test_a_punctuation_only_span_is_not_math() -> None:
    r"""A stray ``$`` leaves ``$、$`` behind; treating it as math misaligns the rest.

    Measured on mamba ``pdf_main#b0086``: the model wrote ``$\Delta$、$$、$C$``
    (three delimiters where two were meant), and the punctuation-only ``$、$``
    being accepted as math made ``$ 和矩阵 $`` a span, so the surrounding prose
    was rendered as a formula.
    """
    for span in ("$、$", "$, $", "$，$"):
        out = render_prose(span, ())
        assert "$" not in out.replace("\\$", ""), f"{span} entered math mode: {out!r}"


def test_a_run_with_unbalanced_delimiters_is_not_wrapped() -> None:
    r"""A styled run whose text holds an odd number of ``$`` must lose its style.

    ``#strong[…]`` wraps the slice, so one delimiter would fall outside the
    wrapper and its partner inside and Typst reads an unbalanced span. Measured
    on mamba ``pdf_main#b0086``: the model's bold marker began mid-formula with
    five ``$``, and the render emitted ``\$#strong[\$`` — the formula as literal
    backslashes.
    """
    text = r"用于 $Delta$、$\$、$C$ 和矩阵 $\llbracket A \rrbracket$ 的投影"
    bad = InlineRun(text=r"$、$C$ 和矩阵 $\llbracket A \rrbracket$ 的投影", bold=True)
    out = render_prose(text, [bad])

    assert "#strong[" not in out, out


def test_a_run_with_balanced_delimiters_is_still_wrapped() -> None:
    # The guard must be narrow: an ordinary styled run keeps its markup.
    out = render_prose("粗体结尾", (InlineRun(text="粗体结尾", bold=True),))
    assert "#strong[粗体结尾]" in out, out


# --------------------------------------------------------------------------- #
# A missing asset must not take its caption with it
# --------------------------------------------------------------------------- #


def test_a_missing_figure_still_renders_its_caption(tmp_path: Path) -> None:
    """The caption is the only surviving description of the picture.

    Dropping the asset is unavoidable (the file is gone), but dropping the
    caption as well removed translated prose the reader needs, and left no trace
    on the page of what the figure showed. The placeholder stays; the caption is
    rendered beneath it.
    """
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
    assert stats.assets_missing == ["figures/gone.png"]


def test_a_missing_figure_without_a_caption_renders_only_the_placeholder(
    tmp_path: Path,
) -> None:
    renderer = TypstDocumentRenderer(assets_root=tmp_path / "assets")
    out, _stats = renderer.build_source(
        [_block(BlockType.IMAGE, block_id="f1", source="figures/gone.png")],
        tmp_path / "build",
    )
    assert "missing figure" in out
    # One placeholder, and no caption line beneath it. The preamble's own
    # figure-caption show rule is the one other ``luma(60)`` block; the caption
    # branch adds a second.
    assert out.count("missing figure") == 1
    assert out.count("fill: luma(60))[") == 1


# --------------------------------------------------------------------------- #
# A bilingual table's source half is a table, not prose
# --------------------------------------------------------------------------- #


def test_a_bilingual_table_does_not_paste_its_markdown_source(tmp_path: Path) -> None:
    """The source row must be compiled, not shown as literal grid syntax.

    ``_pair`` renders the source through ``render_prose``, which is right for a
    sentence and wrong for a table: the page showed
    ``| Method | BLEU |`` in italic grey above the real table. Measured on the
    default (``dual_mode=auto`` -> bilingual) path.
    """
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
    # Both halves are tables: the source pair and the translation.
    assert out.count("#table(") == 2


def test_unparseable_table_falls_back_to_source_and_is_counted(tmp_path: Path) -> None:
    """A translated table whose grid does not parse must not vanish silently.

    ``parse_markdown_table`` requires every line to both start and end with
    ``|``. A model that drops the header's leading pipe (``模型 | …``) leaves a
    grid that does not parse; the renderer used to return "" and drop the whole
    block with no flag, no count, and a content contract that read it as
    preserved. The source grid is a valid table, so the fix renders *that* and
    counts the degradation (``tables_fell_back``) so the source-language cells
    are visible rather than a silent success.
    """
    malformed_target = "模型 | Huginn-0125 |\n|---|---|\n| 前奏 | 测试 |"
    good_source = "| Model | Huginn-0125 |\n|---|---|\n| Intro | Test |"
    block = _block(BlockType.TABLE, block_id="tbl", source=good_source, target=malformed_target)

    source, stats = _render_with_stats([block], tmp_path)

    assert stats.tables == 1
    assert stats.tables_fell_back == 1
    assert stats.tables_dropped == 0
    # The source grid is what reached the page (target did not parse).
    assert "#table(" in source
    assert "Model" in source


def test_table_unparseable_in_both_grids_is_reported_not_silent(tmp_path: Path) -> None:
    """When neither grid parses the block is dropped, but the drop is reported.

    The renderer records ``(block_id, "table_unparseable")`` so the export
    stage flags the block through the render-skip channel — the quality report
    and the content contract both see the gap instead of a shorter document
    with no explanation.
    """
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
    """A translated grid with the source's shape is a verified reconstruction."""
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
    """A translated grid that lost a column is a corrupt reconstruction.

    This is the defect the delivery contract's RECONSTRUCTED branch exists to
    catch: the grid still parses and compiles, so nothing downstream would
    notice the data loss — only the shape check against the source grid does.
    """
    source_grid = "| Model | Score | Note |\n|---|---|---|\n| A | 1 | x |"
    target_grid = "| 模型 | 分数 |\n|---|---|\n| 甲 | 一 |"
    block = _block(BlockType.TABLE, block_id="tc", source=source_grid, target=target_grid)

    renderer = TypstDocumentRenderer()
    renderer.build_source([block], tmp_path / "build")

    (record,) = renderer.table_reconstructions
    assert record.corrupt is True
    assert record.verified is False
    assert "column_count_mismatch" in record.reason


def test_a_table_row_that_gained_a_cell_is_corrupt_not_verified(tmp_path: Path) -> None:
    r"""A translated row that OVERFILLED the grid is a corrupt reconstruction.

    The old oracle compared the *parsed* grids, and :func:`parse_markdown_table`
    pads/truncates rows into a rectangle before returning them — so a row the
    model overfilled was silently truncated (``四`` dropped) AND the shape
    check blessed the result ``verified``. The comparison now runs on raw cell
    counts, so the extra cell is visible corruption.
    """
    source_grid = "| A | B | C |\n|---|---|---|\n| 1 | 2 | 3 |"
    target_grid = "| 甲 | 乙 | 丙 |\n|---|---|---|\n| 一 | 二 | 三 | 四 |"
    block = _block(BlockType.TABLE, block_id="tg", source=source_grid, target=target_grid)

    renderer = TypstDocumentRenderer()
    renderer.build_source([block], tmp_path / "build")

    (record,) = renderer.table_reconstructions
    assert record.corrupt is True
    assert record.verified is False
    assert "row_1_cell_count_mismatch:3->4" in record.reason


def test_a_table_that_fell_back_to_source_is_unverified_not_corrupt(tmp_path: Path) -> None:
    """A fallback is a degradation, not a corruption — no false ERROR."""
    malformed_target = "模型 | Score |\n|---|---|\n| 甲 | 一 |"
    good_source = "| Model | Score |\n|---|---|\n| A | 1 |"
    block = _block(BlockType.TABLE, block_id="tf", source=good_source, target=malformed_target)

    renderer = TypstDocumentRenderer()
    renderer.build_source([block], tmp_path / "build")

    (record,) = renderer.table_reconstructions
    assert record.verified is False
    assert record.corrupt is False


# --------------------------------------------------------------------------- #
# Render side channels: a degradation or a leak must be attributable per block
# --------------------------------------------------------------------------- #


def test_a_degraded_formula_is_recorded_on_the_render_quality_channel(tmp_path: Path) -> None:
    """A formula delivered as Unicode, not typeset math, flags its block.

    ``formulas_degraded`` is a document total; the quality report needs the
    *block*, so the renderer also emits it on the render-quality channel
    (``last_render_flags``), which the export stage turns into a plain
    ``formula_degraded`` error flag — never a skip, because the equation ships.
    """
    # ``\undefinedcmd{x}`` cannot be transpiled, so the body degrades to readable
    # Unicode rather than being dropped.
    block = _block(BlockType.FORMULA, block_id="f1", source=r"\$\$\undefinedcmd{x}\$\$")

    renderer = TypstDocumentRenderer()
    _source, stats = renderer.build_source([block], tmp_path / "build")

    assert stats.formulas_degraded == 1, stats
    assert stats.formulas_dropped == 0, stats
    assert ("f1", "formula_degraded") in renderer.last_render_flags
    assert all(block_id != "f1" for block_id, _reason in renderer.last_render_skips)


def test_a_dropped_formula_is_a_fail_closed_render_skip(tmp_path: Path) -> None:
    """A formula with nothing to place is gone from the page — never silent.

    The empty-body case (a bare ``$$``) has no fallback text, so the block is
    dropped entirely and must surface as a ``render_skip:math_unrenderable``
    (a fail-closed family the taxonomy already names) rather than vanishing.
    """
    block = _block(BlockType.FORMULA, block_id="fdrop", source="$$")

    renderer = TypstDocumentRenderer()
    _source, stats = renderer.build_source([block], tmp_path / "build")

    assert stats.formulas_dropped == 1, stats
    assert ("fdrop", "math_unrenderable") in renderer.last_render_skips


def test_a_leaked_sentinel_flags_its_own_block(tmp_path: Path) -> None:
    """The block that leaks the token is named, not just counted.

    ``sentinels_leaked`` is a document total; a review needs to know *which*
    block carries a restore defect, so the renderer also emits
    ``(block_id, "sentinel_leaked")`` on the render-quality channel.
    """
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
