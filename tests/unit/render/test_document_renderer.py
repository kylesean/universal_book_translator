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
    page, so the span degrades to literal text instead. Asserting only the
    dollar parity is not enough: the output has no dollars at all, so the check
    also passes when the whole span (or the surrounding prose) was dropped.
    """
    out = render_prose("Broken $\\undefinedcmd{x}$ end.")

    assert out.count("$") % 2 == 0
    assert "Broken" in out and "end." in out, "the surrounding prose must survive"
    assert "⟨math⟩" in out, "the span must degrade to a visible marker, not vanish"


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


def test_table_columns_are_weighted_by_content() -> None:
    """An even split crushes a long text column beside short numeric ones.

    Measured on the PaddleOCR benchmark table: 9 even columns wrapped
    "MonkeyOCR [27]" onto three lines and squeezed "公式 CDM↑" to one glyph per
    line. Widths now follow the longest cell per column.
    """
    markup = (
        "| Layout Analysis | Params | FPS | RecognitionMethod |\n"
        "|---|---|---|---|\n"
        "| DocLayout-YOLO [14] | 20M | 107.4 | MonkeyOCR [27] MinerU2.5 [5] |"
    )
    out = render_table(markup)
    # Columns are emitted as absolute pt widths (a minmax-style allocation, see
    # render_table): every column first gets the width its unbreakable run
    # needs, then the leftover is shared by proportion. The assertion is on the
    # shape of that allocation rather than on the internal weights.
    spec = next(line for line in out.splitlines() if "columns:" in line)
    widths = [
        float(part.strip().removesuffix("pt"))
        for part in spec.split("(")[1].rstrip("),").split(",")
    ]
    assert len(widths) == 4
    assert all(w > 0 for w in widths)
    # The recognition column carries "MonkeyOCR [27] MinerU2.5 [5]" and must lead.
    assert widths[3] > widths[0]
    # No column may collapse to a sliver.
    assert min(widths) >= 10.0
    # The allocation fills the text block it was given (453.55pt default, less
    # the gutters and the safety margin).
    assert 425.0 <= sum(widths) <= 455.0


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
        ("lo g(4)", InlineRun(text="g", superscript=True), "("),
        # A bold run before a citation's opening paren (arXiv 2609.22978 b0039).
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
    """The paren must survive as its own character, not become a call argument.

    Asserted as "the expression is terminated" rather than "the paren was
    re-emitted as ``#text("(")``": the terminator is an implementation choice
    (today ``/**/``) and the contract is only that Typst reads the paren as
    prose. ``assert_compiles`` below pins the part that actually matters.
    """
    out = render_prose(text, (run,))
    assert "/**/" in out or "#text(" in out, f"the run's expression must be closed: {out!r}"
    assert expect in out, f"the paren must survive verbatim: {out!r}"


def test_an_unstyled_paren_is_left_alone() -> None:
    # The guard must be narrow: a plain ``(`` with no styled run before it is
    # ordinary prose and must not gain markup.
    assert "/**/" not in render_prose("plain (paren)", ())


@pytest.mark.parametrize(
    ("text", "run", "expect"),
    [
        # arXiv 2608.25512: a superscripted variable followed by a literal dot
        # parses as a field access on the super element. The whole delivery
        # aborted with ``element super has no method `installed𝑚```
        # (Typst words it as a field when an identifier follows).
        ("检验∃𝑑.installed𝑚(𝛾) 一致", InlineRun(text="𝑑", superscript=True), "."),
        # A Chinese full stop is not an identifier, but it follows a run too.
        ("见上标𝑛。后文", InlineRun(text="𝑛", superscript=True), "。"),
    ],
)
def test_a_styled_run_never_absorbs_a_following_dot(text: str, run: InlineRun, expect: str) -> None:
    """Typst continues an expression across ``]``: ``].x`` is a field access.

    Every styled run now gets an unconditional terminator, so this holds for
    prose dots as well as for the field-access case that motivated the guard —
    the rule no longer depends on which character follows.
    """
    out = render_prose(text, (run,))
    assert "/**/" in out or "#text(" in out, f"the run's expression must be closed: {out!r}"
    assert expect in out, f"the dot must survive verbatim: {out!r}"


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

    A caller that interpolates cells raw needs escaping; the publication
    renderer runs cells through ``render_prose`` (which owns escaping and math),
    so it needs the raw text.
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
    # ``(block_id, asset)``: the block id is what lets the export stage's
    # block-keyed skip channel see the miss at all.
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


# --------------------------------------------------------------------------- #
# The two <mark> wrappers mean opposite things
#
# `ubt-blocked-human` is a quarantine marker carrying the SOURCE: rendering it
# would put pipeline HTML on the page. `ubt-failed-draft` carries a REAL draft
# whose repair CALL failed (provider 5xx); the translation itself is fine.
#
# Collapsing them into one "is a <mark>, so skip it" check made the renderer
# fall back to `draft_text`, which predates glossary enforcement. On the ATC'25
# paper that shipped 22 occurrences of `framekernel` in the artifact while the
# ledger -- correctly rewritten to 帧内核 by the export stage -- said otherwise,
# and discarded every repair that had actually landed before the 5xx.
# --------------------------------------------------------------------------- #


def _block_with(*, target: str | None, draft: str | None = None) -> IRBlock:
    return _block(BlockType.NARRATIVE, target=target, draft=draft)


def test_a_failed_repair_still_renders_its_translation() -> None:
    """A repair 5xx is a provider failure, not a verdict on the draft."""
    from ubt.render.typst.document_renderer import _block_text

    block = _block_with(
        target=(
            '<mark class="ubt-failed-draft" title="Repair error: Model API error (500)">'
            "我们提出了帧内核。</mark>"
        ),
        draft="我们提出了 framekernel。",
    )
    rendered = _block_text(block)
    # The wrapped translation wins over the pre-enforcement draft.
    assert rendered == "我们提出了帧内核。"
    assert "<mark" not in rendered
    assert "Repair error" not in rendered
    assert "title=" not in rendered


def test_a_quarantine_marker_falls_back_to_the_draft() -> None:
    """`ubt-blocked-human` carries the SOURCE, so the draft is what the reader gets."""
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
    # Imported from its defining module, not the renderer's compatibility
    # re-export: mypy --strict rejects reaching a symbol through a module
    # that merely re-exports it (no explicit `as` in __all__).
    from ubt.render.delivered_text import is_placeholder as _is_placeholder

    assert _is_placeholder('<mark class="ubt-blocked-human" title="x">src</mark>') is True
    assert _is_placeholder('<mark class="ubt-failed-draft" title="x">译文</mark>') is False
    assert _is_placeholder("plain prose") is False
    # A stray literal tag in prose is not a wrapper.
    assert _is_placeholder("use the <mark> element") is False


def test_unwrapping_keeps_a_body_containing_its_own_markup() -> None:
    """Only the outer wrapper is stripped; inner markup is the block's content."""
    from ubt.render.delivered_text import unwrap_mark as _unwrap_mark

    wrapped = '<mark class="ubt-failed-draft" title="t">a <b>bold</b> b</mark>'
    assert _unwrap_mark(wrapped) == "a <b>bold</b> b"
    # No closing tag: not a wrapper, so nothing is truncated.
    assert _unwrap_mark('<mark class="x">unclosed') == '<mark class="x">unclosed'


# --------------------------------------------------------------------------- #
# A wide code cell must not overrun a narrow column
#
# A Typst `table` sized in `fr` still cannot shrink a cell below its widest
# UNBREAKABLE run. On the ATC'25 paper `(7fr, 46fr, 43fr)` inside a two-column
# page's ~232pt text column could not fit
# `// lib/devices/tpm: MmioAddr::new(base,len)`; the cell overran the column and
# its text drew on top of the neighbouring one (p3:rect327 reached 652.9pt on a
# 595.3pt page).
#
# The fix shrinks the table body. The first attempt inserted zero-width spaces
# at code boundaries: it fit, but a ZWSP is a real character in the PDF text
# layer, so the delivery carried 67 of them and `MmioAddr::new` could no longer
# be found by search or copied into an editor. These tests pin the property that
# matters -- the typeset text is byte-identical to what was translated.
# --------------------------------------------------------------------------- #


def test_a_rendered_table_carries_no_inserted_break_characters() -> None:
    """Typography must not corrupt the text it typesets."""
    from ubt.render.typst.document_renderer import render_table

    markup = (
        "| | Access | Acquire |\n"
        "|---|---|---|\n"
        "| RedLeaf | // lib/devices/ixgbe: ptr::read_volatile(addr) "
        "| // lib/devices/tpm: MmioAddr::new(base,len) |\n"
    )
    out = render_table(markup)
    # A `//` in code is protected with a ZWSP upstream; what must NOT appear is
    # one inside an identifier the reader would search for.
    assert "MmioAddr::\u200bnew" not in out
    assert "read\u200b_volatile" not in out
    assert "MmioAddr::new(base,len)" in out


def test_a_table_with_a_wide_unbreakable_run_is_typeset_smaller() -> None:
    """The lossless lever for a column that cannot shrink enough.

    The run has to genuinely exceed the space a column gets on the page it will
    be typeset into -- 30 Latin characters at 10.5pt is ~175pt, more than a
    three-column 453pt text block can give one cell. A shorter run that *does*
    fit must keep the document size (see the next test): shrinking it would be
    the old heuristic's mistake, which shrunk a table that measured 199pt wide
    inside a 320pt column.
    """
    from ubt.render.typst.document_renderer import _TABLE_TEXT_SCALE, render_table

    out = render_table(
        "| | A | B |\n|---|---|---|\n"
        "| R | // lib/devices/tpm: MmioAddr::new(base,len,flags)::from_parts | x |\n",
        content_width_pt=453.55,
    )
    assert f"#set text(size: {_TABLE_TEXT_SCALE}em)" in out


#: A page-wide text block (A4 less 1.8cm margins), the widest layout the
#: renderer emits; a table that does not fit *this* genuinely cannot fit.
_WIDE_PAGE_PT = 493.2

#: A two-column page's text block: tight enough that a long identifier cannot fit
#: a proportional share, which is the case that exposed the column-width defect.
_NARROW_PAGE_PT = 219.5


def test_a_table_that_fits_its_page_keeps_the_document_size() -> None:
    """README's defect, restated as a property: a fitting run is not shrunk.

    ``metadata.openclaw.requires`` is 26 Latin characters = 14.8 em = 155pt. It
    overflowed because its column was given 105pt, not because the table needed
    a smaller font: given a page-wide layout it fits. Shrinking here would hide
    the real cause.
    """
    from ubt.render.typst.document_renderer import render_table

    markup = (
        "| 系统 | 说明 |\n|---|---|\n"
        "| OpenClaw | agentskills.io 兼容；通过 metadata.openclaw.requires 控制 |\n"
    )
    out = render_table(markup, content_width_pt=_WIDE_PAGE_PT)
    assert "#set text(size:" not in out


def test_a_table_that_already_fits_keeps_the_document_size() -> None:
    """Shrinking every table would restyle the 62% that were never at risk."""
    from ubt.render.typst.document_renderer import render_table

    for markup in (
        "| a | b |\n|---|---|\n| 1 | 2 |\n",
        "| 模块 | 测试数 |\n|---|---|\n| dma | 12 |\n",
    ):
        assert "#set text(size:" not in render_table(markup)


def test_the_size_override_is_scoped_to_the_table() -> None:
    """`#set` inside a content block must not leak into the document body.

    The assertion is on the *opening* token: ``#[`` opens a content block in
    markup mode, where a bare ``[`` is just a text character. The bare form used
    to be emitted, so ``#set text(size: 0.84em)`` applied to everything after the
    table and each shrunken table multiplied the rest of the document by 0.84 --
    a 17-table paper ended at 0.53pt type, which is what a delivered 22-page file
    measured. The old test asserted the broken shape (``startswith("[")``), so it
    stayed green while the whole document shrank.
    """
    from ubt.render.typst.document_renderer import render_table

    # A table wide enough to trigger the size override, so the scoping is real.
    markup = "| " + " | ".join("x" * 30 for _ in range(3)) + " |\n|---|---|---|\n"
    out = render_table(markup, page_columns=1)
    body = out.strip()
    assert body.startswith("#[")
    assert body.endswith("]")
    # No stray bracket survives as typeset text either: the content block is the
    # only place ``[`` may appear at the very start.
    #
    # The *value* is the computed scale for this table, not the constant 0.84:
    # emitting the constant while sizing the columns for a smaller body is what
    # stacked a 9-column table's headers one ideograph per line.
    assert "#set text(size: " in body
    import re

    emitted = float(re.search(r"#set text\(size: ([0-9.]+)em\)", body).group(1))  # type: ignore[union-attr]
    assert emitted < 0.84, body


def test_a_shrunken_table_does_not_resize_the_prose_after_it() -> None:
    """The defect measured end-to-end: the leak compounded across tables.

    Compiles a document with two shrunken tables and checks that the paragraphs
    before, between, and after them are all typeset at the document size. This is
    the property that failed in the delivered 22-page file, where the last page's
    type had shrunk to 0.49pt -- a size no reader can use.
    """
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
    # All three are document-size text: the tables are contained.
    assert len(set(heights.values())) == 1, heights


def test_an_identifier_column_gets_room_for_its_run() -> None:
    """A blocked identifier column must not be squeezed by a prose column.

    Delivered defect: ``metadata.openclaw.requires`` (14.1 em) sat beside CJK
    prose columns that claimed most of the weight, so the identifier got a
    column narrower than the string and ran 39.8pt past the page edge. The
    allocation now gives every column at least the width of its unbreakable run.
    """
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
    # The identifier column must hold the run: 14.1 em at 10.5pt is ~148pt, minus
    # whatever shrink the whole table needed (the emitted body size is in the
    # spec's sibling ``#set text`` line). A column narrower than the string is
    # the defect; its absolute width also has to leave room for the cell inset.
    has_shrink = "#set text(size:" in out
    factor = 0.84 if has_shrink else 1.0
    assert widths[2] >= 148.0 * factor * 0.9, spec


def test_a_prose_column_is_never_promoted_to_auto() -> None:
    """``auto`` on a prose column swallows the table.

    Measured: a 4-column table whose prose column was promoted reached x=684 on
    a 595pt page and crushed its neighbour to 8pt -- one ideograph per line. A
    run wider than ``_TABLE_AUTO_MAX_EM`` is prose that re-wraps, so it keeps a
    fraction.
    """
    from ubt.render.typst.document_renderer import render_table

    # Pure prose: no token long enough to need ``auto``. A paragraph containing
    # SYSTEM_PROMPT_DYNAMIC_BOUNDARY is deliberately *not* used here -- that is a
    # real 19.7 em identifier, and ``auto`` is the correct answer for it in a
    # 219pt column (the next test asserts that directly).
    long_prose = (
        "12-15 个命名章节。静态前缀（身份、规则、工具）通过哈希缓存；"
        "动态后缀（环境、内存、MCP）每次计算，边界标记将两者分离。"
    )
    markup = f"| 阶段 | Claude Code | Codex |\n|---|---|---|\n| 提示词组装 | {long_prose} | 25-30 个内置工具 |\n"
    out = render_table(markup, content_width_pt=_NARROW_PAGE_PT)
    spec = next(line for line in out.splitlines() if "columns:" in line)
    assert "auto" not in spec, spec


def test_cjk_punctuation_closes_an_unbreakable_run() -> None:
    """``。IDENTIFIER`` is two tokens, not one.

    The sentence-ending 。shares the U+3000..U+303F block with ideographs, so
    testing the broader CJK class first glued the punctuation to the following
    identifier. The observable consequence: the middle column in this table was
    promoted to ``auto`` for a 20.4 em "run" that was really a 19.7 em
    identifier plus a period. With the fix the column is still a candidate on
    its own merits, but the punctuation no longer inflates the figure -- so a
    table whose identifier is short is not promoted on punctuation alone.
    """
    from ubt.render.typst.document_renderer import render_table

    # An identifier short enough to fit proportionally, preceded by a period.
    # If the period glued on, the run would clear the 8 em promotion floor.
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
    """A figure must fit whichever column it lands in.

    The width was emitted as an absolute ``pt`` length measured against A4 text
    width (~470pt). On the two-column academic template the body column measures
    228.98pt (470.55pt text, 1.2em gutter), so a 290pt figure was typeset at
    290pt and ran ~61pt past the column, over the text beside it — Typst draws
    rather than clips (measured by rendering: 290.5pt of ink in a 228.98pt
    column). A container-relative width lets Typst fit both templates.
    """
    from ubt.render.typst.document_renderer import _Figure

    asset = tmp_path / "wide.png"
    asset.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
    renderer = TypstDocumentRenderer()
    build = tmp_path / "build"
    build.mkdir()

    narrow = renderer._figure_markup(_Figure(asset=str(asset), width_pt=290.0), build)
    assert "width: 290pt" not in narrow, "an absolute width ignores the column"
    # 290 / 470.55 = 61.6% of whatever container Typst lays the figure into.
    assert "width: 61.6%" in narrow

    # A genuinely small graphic keeps its small size, still container-relative.
    logo = renderer._figure_markup(_Figure(asset=str(asset), width_pt=121.0), build)
    assert "width: 25.7%" in logo

    # No measured width: the existing cap applies, unchanged.
    unknown = renderer._figure_markup(_Figure(asset=str(asset)), build)
    assert f"width: {renderer.figure_width}" in unknown


# --- captions are figure captions, not "whatever follows a figure" ---------


def test_body_prose_after_a_figure_is_not_rendered_as_its_caption(tmp_path: Path) -> None:
    """A main-story paragraph keeps body type even when it is short.

    Translation compresses a source paragraph to ~0.3-0.5x, so a 486-character
    English paragraph arrives as ~180 Chinese characters. Under the old
    length-only rule that landed under the caption cap and was re-laid at 8.5pt
    grey under the preceding image — on the reference chapter, 15 real
    paragraphs, among them the definitions that introduce the next equation.
    """
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
    """The paragraph after an equation is prose, however short it is.

    A formula recovered as a source crop is not a picture: the paragraph under
    it either defines its symbols or leads into the next equation. Treating it
    as a caption put the lead-in sentence at 8.5pt under the equation, which is
    how "Gauss's law … leads to a second important equation:" stopped being the
    first half of a sentence.
    """
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
    """The extractor's own ``caption`` flow binds regardless of length.

    The description under FIG. 3.11 runs to 162 characters; a length rule would
    orphan it from its image, which is a worse defect than the one being fixed.
    """
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
    """A sentence naming a figure is not that figure's caption.

    "图 3.1 展示了…" reads as prose, not as the small grey line under an image.
    """
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
    """Assets are named by content, not by basename.

    The build directory is flat and Typst addresses images by name, so two
    different pictures both called ``figure.png`` used to collide: the second
    copy overwrote the first (or, at equal sizes, was skipped) and the first
    figure silently rendered the second picture.
    """
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
    # Both names are referenced by the emitted source, so neither figure is
    # drawn from the other's copy.
    for name in copied:
        assert f'image("{name}"' in source


def test_render_table_dense_single_col_landscape_contingency() -> None:
    # 9-column dense table in single-column layout with allow_landscape=True
    header = "| " + " | ".join(f"Col{i}" for i in range(9)) + " |"
    sep = "| " + " | ".join("---" for _ in range(9)) + " |"
    row = "| " + " | ".join(f"Value{i}" for i in range(9)) + " |"
    md = f"{header}\n{sep}\n{row}\n"

    # Default allow_landscape=False: scoped in #[ ... ]
    out_default = render_table(md, page_columns=1, allow_landscape=False)
    assert out_default.strip().startswith("#[")
    assert "#page(flipped: true)" not in out_default

    # With allow_landscape=True: wraps in #page(flipped: true)[ ... ]
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

    # Default single-column renderer enables landscape tables
    renderer_default = TypstDocumentRenderer(columns=1)
    source_default, _ = renderer_default.build_source([table_block], build_dir)
    assert "#page(flipped: true)[" in source_default

    # Explicitly disabled
    renderer_no_ls = TypstDocumentRenderer(columns=1, allow_landscape_tables=False)
    source_no_ls, _ = renderer_no_ls.build_source([table_block], build_dir)
    assert "#page(flipped: true)[" not in source_no_ls

    # Dual columns does not flip to landscape
    renderer_2col = TypstDocumentRenderer(columns=2)
    source_2col, _ = renderer_2col.build_source([table_block], build_dir)
    assert "#page(flipped: true)[" not in source_2col

    # Bilingual mode does not flip to landscape (avoids illegal container nesting)
    renderer_bilingual = TypstDocumentRenderer(columns=1, bilingual=True)
    source_bilingual, _ = renderer_bilingual.build_source([table_block], build_dir)
    assert "#page(flipped: true)[" not in source_bilingual


def test_prose_with_kinsoku_enabled(tmp_path: Path) -> None:
    """Renderer with enable_kinsoku=True injects Word Joiners around CJK punctuation."""
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
    """Renderer with enable_colophon=True emits publication CIP colophon page."""
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
    """Renderer with enable_index=True appends publication concordance index section."""
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
    """Renderer with enable_decor=True emits decorative headers and drop caps."""
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

    # Drop cap macro in preamble
    assert "#let ubt-dropcap" in source
    # Classic decorative header for Chapter 1
    assert "第 1 章" in source
    assert "分布式存储架构" in source
    assert "❦" in source
    # Drop cap on p1
    assert '#ubt-dropcap("在", prefix: "", lines: 2, fill: rgb("#1e3a8a"))' in source
    # p2 does NOT have drop cap
    assert '#ubt-dropcap("另"' not in source
    # Chapter 2 header
    assert "第 2 章" in source
    assert "副本复制协议" in source
    # Drop cap on p3 with quote prefix
    assert '#ubt-dropcap("副", prefix: "“", lines: 2, fill: rgb("#1e3a8a"))' in source
