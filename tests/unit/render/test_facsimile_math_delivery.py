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

_PSI = r"其中 $\psi_1$ 为表面势。"
_SUBSCRIPT = r"$N_{ch}$ 是沟道浓度"


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
    @pytest.mark.parametrize("channel", [_l1_markup, _l2_markup], ids=["L1", "L2"])
    def test_a_greek_macro_becomes_math(self, channel: Callable[[str], str]) -> None:
        markup = channel(_PSI)
        assert "$psi_1$" in markup, markup
        assert "\\$" not in markup
        assert "\\psi" not in markup

    @pytest.mark.parametrize("channel", [_l1_markup, _l2_markup], ids=["L1", "L2"])
    def test_subscript_braces_are_not_escaped(self, channel: Callable[[str], str]) -> None:
        markup = channel(_SUBSCRIPT)
        assert "\\_" not in markup, markup
        assert "\\{" not in markup
        assert "N_" in markup

    def test_measured_multiline_case(self) -> None:
        text = (
            r"其中 $\psi_{pert}$ 由在 $x = T_{fin}/2$ 处评估的 $\psi_2$ 给出，如图 3.5 所示 [3]。"
        )
        markup = _l2_markup(text)
        assert "\\$" not in markup
        for leaked in ("\\psi", "\\text", "\\frac", "\\_"):
            assert leaked not in markup, leaked
        assert "\\[3\\]" in markup


class TestRejectedBodyDegradesToUnicode:
    def test_l2_denied_span_degrades(self) -> None:
        markup = _l2_markup(_PSI, math_gate=_deny)
        assert "\\psi" not in markup, markup
        assert "ψ" in markup

    def test_l1_denied_span_degrades(self) -> None:
        markup = _l1_markup(_PSI, math_gate=_deny)
        assert "\\psi" not in markup, markup
        assert "ψ" in markup

    def test_no_gate_still_transpiles(self) -> None:
        assert "$psi_1$" in _l2_markup(_PSI, math_gate=None)


class TestDisplayFormulaBranch:
    def test_reproduced_compile_error_is_gone(self) -> None:
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
    def test_unpaired_dollar_escapes_as_text(self) -> None:
        markup = _l2_markup("价格 $5 与 $6 不是数学")
        assert "\\$5" in markup, markup

    def test_math_bodies_collector_agrees_with_the_render(self) -> None:
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
    def test_sentinel_span_is_not_transpiled(self) -> None:
        text = "见 ⟦SOUP_MASK_0001-abc⟧ 中的 $x_1$ 项"
        assert math_bodies_for_prose(text) == ["x_1"]
        markup = _l2_markup(text)
        assert "SOUP" not in markup or "⟨missing⟩" in markup, markup


class TestTableBranch:
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
    from ubt.render.typst.page_locked_renderer import _block_measure, _content_width

    geom = _geometry(columns=2)
    column = _content_width(geom)
    narrow = _block("b_n", "x", box=(40.0, 500.0, 200.0, 540.0))
    assert _block_measure(geom, narrow) == pytest.approx(160.0)
    wide = _block("b_w", "x", box=(40.0, 500.0, 530.0, 540.0))
    assert _block_measure(geom, wide) == pytest.approx(column)
    assert _block_measure(geom, _block("b_nb", "x")).__class__ is float


def test_renderer_accepts_math_gate_kwargs(tmp_path: Path) -> None:
    assert PageLockedReflowRenderer()._math_gate is None
    assert FacsimileRenderer()._math_gate is None
    assert FacsimileRenderer(math_gate=_allow)._math_gate is _allow
    owned = FacsimileRenderer(math_gate=_allow)
    assert owned._math_gate is _allow


class TestProseMacroGuard:
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
        text = r"保存到 C:\Users\kyle\Documents 目录"
        markup = _l2_markup(text)
        assert "Users" in markup and "kyle" in markup and "Documents" in markup, markup

    def test_an_unknown_command_in_prose_is_left_alone(self) -> None:
        markup = _l2_markup(r"字面 \unknowncmdonly 不是数学")
        assert "unknowncmdonly" in markup, markup

    def test_the_guard_changes_no_delimiter_pairing(self) -> None:
        out = render_prose(r"{ x_k } \to 0$ 是一个序列，其中 $x_k \in \Omega$。")
        assert out.replace("\\$", "").count("$") == 0, out

    def test_unicode_math_macro_is_the_shared_owner(self) -> None:
        from ubt.render.typst.math_transpiler import unicode_math_macro

        assert unicode_math_macro("beta") == "β"
        assert unicode_math_macro("ln") == "ln"
        assert unicode_math_macro("Users") is None
        assert unicode_math_macro("unknowncmdonly") is None
