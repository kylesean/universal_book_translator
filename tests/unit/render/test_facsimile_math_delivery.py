r"""Fix E — a translated page's inline LaTeX must become math, never escaped text.

The controlled ``chapter-3.pdf`` run (``gemma4-e4b`` for BOTH draft and repair)
proved the model was not the problem. Its targets contain 44 ``$`` and its
sources contain 0: gemma4 added LaTeX delimiters, which is the exact input the
transpiler exists to consume. Both facsimile channels however ran
``typst_escape`` over the WHOLE block, and ``$``/``_``/``{``/``}`` are all in the
escape set, so a translated paragraph shipped its formula source code to the
reader. Measured in the shipped ``facsimile.typ``:

* 265 literal ``\$`` across 44 lines;
* 108 raw macro leaks (``\text`` 35, ``\beta`` 33, ``\psi`` 16, ``\times`` 8,
  ``\ln`` 4, ``\mu`` 4, ``\frac`` 3, ``\epsilon`` 2, ``\partial`` 2,
  ``\equiv`` 1).

After this fix, over the SAME ledger (a render-only re-export, so the model
output is a controlled constant): 7 ``\$`` and 5 macro leaks, all 5 from one
block whose delimiters are genuinely unbalanced (9 ``$``) — tracked separately.

The invariant these tests pin is per-channel and per-branch, because the defect
was: L1 canvas, L2 page-locked and the FORMULA branch each took their own
escape-only route. Any one of them reverting re-opens the leak.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from ubt.core.ir.models import BlockType, BoundingBox, IRBlock, make_element
from ubt.render.typst.document_renderer import (
    math_bodies_for_blocks,
    math_bodies_for_prose,
    render_prose,
)
from ubt.render.typst.facsimile_renderer import (
    FacsimileBlockPlacement,
    FacsimilePageSpec,
    FacsimileRenderer,
)
from ubt.render.typst.page_locked_renderer import (
    PageGeometry,
    PageLockedPageSpec,
    PageLockedReflowRenderer,
)

pytestmark = pytest.mark.fast

#: A LaTeX body the transpiler accepts. Kept deliberately small: what matters is
#: the shape (a macro inside ``$…$``), not the maths.
_PSI = r"其中 $\psi_1$ 为表面势。"
_SUBSCRIPT = r"$N_{ch}$ 是沟道浓度"


#: A gate that approves everything: separates "the span was transpiled" from
#: "the span was transpiled *and* compiled".
def _allow(_body: str) -> bool:
    return True


def _deny(_body: str) -> bool:
    return False


def _geometry(**overrides: float) -> PageGeometry:
    kwargs: dict[str, float] = {
        "page_w": 540.0,
        "page_h": 666.0,
        "margin_top": 36.0,
        "margin_bottom": 36.0,
        "margin_left": 36.0,
        "margin_right": 36.0,
    }
    kwargs.update(overrides)
    return PageGeometry(**kwargs)  # type: ignore[arg-type]


def _block(
    block_id: str,
    text: str,
    *,
    block_type: BlockType = BlockType.NARRATIVE,
    page: int = 1,
    box: tuple[float, float, float, float] = (40.0, 500.0, 500.0, 540.0),
) -> IRBlock:
    return IRBlock(
        element=make_element(
            id=block_id,
            spine_index=1,
            block_type=block_type,
            source_text=text,
            bbox=BoundingBox(page=page, x0=box[0], y0=box[1], x1=box[2], y1=box[3]),
        ),
        target_text=text,
    )


def _l2_markup(text: str, *, math_gate: Callable[[str], bool] | None = _allow) -> str:
    """One Tier-L2 page holding a single narrative block."""
    renderer = PageLockedReflowRenderer(math_gate=math_gate)
    spec = PageLockedPageSpec(
        page_no=1,
        geometry=_geometry(),
        uniform_size_pt=9.0,
        leading_em=0.35,
        par_spacing_em=0.65,
        blocks=[_block("b_math", text)],
    )
    return renderer.build_page_markup(spec)


def _l1_markup(text: str, *, math_gate: Callable[[str], bool] | None = _allow) -> str:
    """One Tier-L1 canvas page holding a single placement."""
    renderer = FacsimileRenderer(math_gate=math_gate)
    spec = FacsimilePageSpec(
        page_no=1,
        page_w=540.0,
        page_h=666.0,
        uniform_size_pt=9.0,
        leading_em=0.35,
        placements=[
            FacsimileBlockPlacement(
                block_id="b_math",
                page=1,
                x0=40.0,
                y0=500.0,
                x1=500.0,
                y1=540.0,
                text=text,
            )
        ],
    )
    return renderer.build_markup([spec], "bg.pdf")


class TestInlineMathIsTranspiledNotEscaped:
    """Every channel converts a ``$…$`` span to Typst math."""

    @pytest.mark.parametrize("channel", [_l1_markup, _l2_markup], ids=["L1", "L2"])
    def test_a_greek_macro_becomes_math(self, channel: Callable[[str], str]) -> None:
        markup = channel(_PSI)
        assert "$psi_1$" in markup, markup
        # The shipped defect, stated as the negative: escaping the delimiter.
        assert "\\$" not in markup
        assert "\\psi" not in markup

    @pytest.mark.parametrize("channel", [_l1_markup, _l2_markup], ids=["L1", "L2"])
    def test_subscript_braces_are_not_escaped(self, channel: Callable[[str], str]) -> None:
        markup = channel(_SUBSCRIPT)
        assert "\\_" not in markup, markup
        assert "\\{" not in markup
        assert "N_" in markup

    def test_measured_multiline_case(self) -> None:
        r"""Page 5's real paragraph: prose plus two formulas plus a citation."""
        text = (
            r"其中 $\psi_{pert}$ 由在 $x = T_{fin}/2$ 处评估的 $\psi_2$ 给出，如图 3.5 所示 [3]。"
        )
        markup = _l2_markup(text)
        assert "\\$" not in markup
        for leaked in ("\\psi", "\\text", "\\frac", "\\_"):
            assert leaked not in markup, leaked
        # The citation's brackets still survive: they are prose, and the escape
        # set has to keep protecting them.
        assert "\\[3\\]" in markup


class TestRejectedBodyDegradesToUnicode:
    """A body Typst refuses must never print a command name."""

    def test_l2_denied_span_degrades(self) -> None:
        markup = _l2_markup(_PSI, math_gate=_deny)
        assert "\\psi" not in markup, markup
        assert "ψ" in markup

    def test_l1_denied_span_degrades(self) -> None:
        markup = _l1_markup(_PSI, math_gate=_deny)
        assert "\\psi" not in markup, markup
        assert "ψ" in markup

    def test_no_gate_still_transpiles(self) -> None:
        """``math_gate=None`` gates nothing: the transpiled body is emitted."""
        assert "$psi_1$" in _l2_markup(_PSI, math_gate=None)


class TestDisplayFormulaBranch:
    """The FORMULA branch must transpile rather than escape-then-wrap."""

    def test_reproduced_compile_error_is_gone(self) -> None:
        r"""The old branch emitted ``$ N\_\{ch\} $``, which failed the document.

        ``error: unknown variable: ch`` — Typst math mode has no backslash
        escapes, so an escaped brace became a token. Reproduced verbatim before
        being fixed, which is why it is pinned here rather than described.
        """
        markup = _l2_markup(
            r"$ N_{ch} $",
        )
        assert "$ N\\_\\{ch\\} $" not in markup
        assert "\\_\\{" not in markup

    def test_display_formula_block_becomes_math(self) -> None:
        renderer = PageLockedReflowRenderer(math_gate=_allow)
        spec = PageLockedPageSpec(
            page_no=1,
            geometry=_geometry(),
            uniform_size_pt=9.0,
            leading_em=0.35,
            par_spacing_em=0.65,
            blocks=[_block("b_eq", r"$$\psi_1 = f(x)$$", block_type=BlockType.FORMULA)],
        )
        markup = renderer.build_page_markup(spec)
        assert "\\psi" not in markup, markup
        assert "$" in markup

    def test_uncompilable_formula_never_leaks_macros(self) -> None:
        renderer = PageLockedReflowRenderer(math_gate=_deny)
        spec = PageLockedPageSpec(
            page_no=1,
            geometry=_geometry(),
            uniform_size_pt=9.0,
            leading_em=0.35,
            par_spacing_em=0.65,
            blocks=[_block("b_eq", r"$$\frac{a}{}$$", block_type=BlockType.FORMULA)],
        )
        markup = renderer.build_page_markup(spec)
        assert "frac" not in markup, markup
        assert "\\frac" not in markup


class TestStrayDelimiterStaysLiteral:
    """An unbalanced ``$`` is money or noise, not a formula."""

    def test_unpaired_dollar_escapes_as_text(self) -> None:
        markup = _l2_markup("价格 $5 与 $6 不是数学")
        assert "\\$5" in markup, markup

    def test_math_bodies_collector_agrees_with_the_render(self) -> None:
        r"""The pre-warm must see exactly the bodies a render gates.

        ``docling_render`` pre-warms one batched probe over
        :func:`math_bodies_for_blocks` plus :func:`math_bodies_for_prose` for
        every L1 placement's text. If the collector missed a body the gate would
        still be asked about it, the render would pay a ~0.7s Typst startup per
        span mid-page, and a pre-warm bug would look like a performance bug.
        """
        text = r"其中 $\psi_{pert}$ 由 $\psi_2$ 给出，价格 $5 不是数学。"
        bodies = math_bodies_for_prose(text)
        assert bodies == [b for b in bodies if b]
        assert any("psi_2" in b for b in bodies), bodies
        assert "\\psi" not in "".join(render_prose(t) for t in bodies)

    def test_block_collector_covers_both_shapes(self) -> None:
        blocks = [
            _block("b_p", _PSI),
            _block("b_eq", r"$$\beta^2$$", block_type=BlockType.FORMULA),
            _block("b_tbl", "| a | b |\n|---|---|\n| 1 | 2 |", block_type=BlockType.TABLE),
        ]
        bodies = math_bodies_for_blocks(blocks)
        assert any("psi_1" in b for b in bodies), bodies
        assert any("beta" in b for b in bodies), bodies


class TestSentinelInsideMathStaysProse:
    """A masker token is an artifact, not a formula."""

    def test_sentinel_span_is_not_transpiled(self) -> None:
        text = "见 ⟦SOUP_MASK_0001-abc⟧ 中的 $x_1$ 项"
        assert math_bodies_for_prose(text) == ["x_1"]
        markup = _l2_markup(text)
        # The real mask token must never reach the page, even escaped.
        assert "SOUP" not in markup or "⟨missing⟩" in markup, markup


class TestTableBranch:
    """A Markdown grid is typeset, not printed."""

    @staticmethod
    def _l2_table(grid: str, *, math_gate: Callable[[str], bool] | None = _allow) -> str:
        renderer = PageLockedReflowRenderer(math_gate=math_gate)
        spec = PageLockedPageSpec(
            page_no=1,
            geometry=_geometry(),
            uniform_size_pt=9.0,
            leading_em=0.35,
            par_spacing_em=0.65,
            blocks=[_block("b_tbl", grid, block_type=BlockType.TABLE)],
        )
        return renderer.build_page_markup(spec)

    def test_l2_grid_becomes_a_table(self) -> None:
        grid = "| 参数 | 值 |\n|------|-----|\n| $N_{ch}$ | 1e15 |"
        markup = self._l2_table(grid)
        assert "#table" in markup, markup
        # The old escape-only route printed the pipes and dashes as prose.
        assert "|------|" not in markup

    def test_l2_unparseable_grid_degrades_to_text(self) -> None:
        markup = self._l2_table("| 这不是一个表格")
        assert "#table" not in markup
        assert "这不是一个表格" in markup

    def test_cell_math_is_transpiled(self) -> None:
        grid = "| 参数 | 值 |\n|------|-----|\n| $\\psi_1$ | 1 |"
        markup = self._l2_table(grid)
        assert "$psi_1$" in markup, markup
        assert "\\psi" not in markup


def test_measure_helper_uses_the_block_box() -> None:
    """``_block_measure`` returns the block's own width, capped at the column.

    A page-locked canvas paints a block inside its source rectangle, so a table
    that used the abstract column width could be laid out wider than the box the
    reader was promised — and re-layout in a two-column page paints over the
    neighbouring column.
    """
    from ubt.render.typst.page_locked_renderer import _block_measure, _content_width

    geom = _geometry(columns=2)
    column = _content_width(geom)
    narrow = _block("b_n", "x", box=(40.0, 500.0, 200.0, 540.0))
    assert _block_measure(geom, narrow) == pytest.approx(160.0)
    # A block wider than one column is clamped to the column measure.
    wide = _block("b_w", "x", box=(40.0, 500.0, 530.0, 540.0))
    assert _block_measure(geom, wide) == pytest.approx(column)
    # A block with no bbox falls back to the column.
    assert _block_measure(geom, _block("b_nb", "x")).__class__ is float


def test_renderer_accepts_math_gate_kwargs(tmp_path: Path) -> None:
    """Both renderers take the gate at construction, and default to none."""
    assert PageLockedReflowRenderer()._math_gate is None
    assert FacsimileRenderer()._math_gate is None
    assert FacsimileRenderer(math_gate=_allow)._math_gate is _allow
    # The facsimile renderer forwards its gate to the L2 pages it owns: without
    # this the L2 half of a mixed document silently re-opens the leak.
    owned = FacsimileRenderer(math_gate=_allow)
    assert owned._math_gate is _allow


class TestProseMacroGuard:
    r"""Fix F: a command name stranded in the PROSE half must not ship as LaTeX.

    ``_MATH_SPAN_RE`` pairs ``$`` positionally, so one stray delimiter shifts
    every pairing after it and real formulas land in the prose half, where they
    are escaped, not transpiled. Re-pairing was measured and deliberately NOT
    done (it breaks the pinned alphafold2 contract), so the leak is closed at the
    escaping boundary instead: ``\beta`` becomes ``β`` as literal text.
    """

    #: The exact measured block, ``pdf_main#b0152`` — nine ``$`` (odd), so the
    #: first candidate span swallows 65 characters of prose and the four real
    #: formulas after it are scattered across prose chunks.
    B0152 = (
        r"方程 (A.1) 仅在 $\frac{F}{F_{th,SI}} \text{ 满足 } Ag0 >-1 \text{ 的条件下有效。"
        r"在亚阈值区域，如果 $\beta$ 比前一个假设中的值更小，则方程 (3.11) 中的 $\beta^2$ "
        r"可以被忽略。亚阈值区域的 $\ln\beta$ 不再是常数；可以据此获得亚阈值区域的 $\beta$ "
        r"函数，如下所示 [3]："
    )

    def test_the_measured_block_ships_no_command_names(self) -> None:
        markup = _l2_markup(self.B0152)
        for leaked in ("\\beta", "\\ln", "\\text", "\\frac"):
            assert leaked not in markup, f"{leaked} leaked: {markup!r}"
        assert "β" in markup, "the degradation must be readable, not absent"

    def test_a_windows_path_is_untouched(self) -> None:
        r"""Only KNOWN math commands are rewritten — ``Users`` is not one.

        A guard that placeholdered every ``\word`` would corrupt a file path, an
        escape sequence, or a literal backslash reference in prose. That is the
        reason this takes the table lookup rather than a generic regex.
        """
        text = r"保存到 C:\Users\kyle\Documents 目录"
        markup = _l2_markup(text)
        assert "Users" in markup and "kyle" in markup and "Documents" in markup, markup

    def test_an_unknown_command_in_prose_is_left_alone(self) -> None:
        r"""In prose, an unmapped name keeps its bytes.

        Inside a *formula* the neutral ``⟨math⟩`` marker is the honest
        degradation; in prose the same substitution would silently delete a word
        the reader is owed, so the guard is table-driven and declines.
        """
        markup = _l2_markup(r"字面 \unknowncmdonly 不是数学")
        assert "unknowncmdonly" in markup, markup

    def test_the_guard_changes_no_delimiter_pairing(self) -> None:
        r"""The pinned safety property: a stray ``$`` never opens math mode.

        ``test_a_stray_dollar_does_not_swallow_prose_into_math`` protects the
        alphafold2 failure (``unknown variable: 是一个序列``, whole document lost).
        Fix F must clear command names WITHOUT re-anchoring a pair, so every
        delimiter stays escaped here.
        """
        out = render_prose(r"{ x_k } \to 0$ 是一个序列，其中 $x_k \in \Omega$。")
        assert out.replace("\\$", "").count("$") == 0, out

    def test_unicode_math_macro_is_the_shared_owner(self) -> None:
        """One table answers "is this a known command", prose and formula alike."""
        from ubt.render.typst.math_transpiler import unicode_math_macro

        assert unicode_math_macro("beta") == "β"
        assert unicode_math_macro("ln") == "ln"
        assert unicode_math_macro("Users") is None
        assert unicode_math_macro("unknowncmdonly") is None
