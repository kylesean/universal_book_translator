"""Unit tests for B-track: math-aware table cells + CJK column widths."""

from ubt.adapters.pdf.typst_reconstructor import (
    _cell_to_typst,
    _display_width,
    _markdown_table_to_typst,
    _prose_to_typst,
)


def test_cell_math_not_escaped() -> None:
    out = _cell_to_typst("Attention with $K_t$ cache")
    assert "$K_t$" in out
    assert "\\$" not in out


def test_cell_prose_still_escaped() -> None:
    out = _cell_to_typst("a_b and 100%")
    assert "a\\_b" in out
    assert "$" not in out


def test_table_math_cells_survive() -> None:
    md = (
        "| Method | Cost |\n"
        "|---|---|\n"
        "| Attention with $K_t$ cache | $O(n^2)$ |\n"
        "| 译码阶段 | $O(n)$ |"
    )
    out = _markdown_table_to_typst(md)
    assert "\\$" not in out
    assert "$K_t$" in out
    # Pandoc escapes parens as \( \) — valid Typst math, verified compiling.
    assert "n^2" in out and "$O" in out
    # Prose cells still escaped/rendered.
    assert "译码阶段" in out


def test_cjk_column_gets_display_width_share() -> None:
    assert _display_width("译码阶段") == 8
    assert _display_width("Method") == 6
    md = "| Method | 译码阶段 |\n|---|---|\n| x | y |"
    out = _markdown_table_to_typst(md)
    columns_line = next(line for line in out.splitlines() if line.strip().startswith("columns:"))
    fractions = [
        float(part.strip().rstrip("%"))
        for part in columns_line.split("(")[1].split(")")[0].split(",")
    ]
    assert len(fractions) == 2
    # 译码阶段 (width 8) outweighs Method (width 6).
    assert fractions[1] > fractions[0]


def test_prose_math_preserved_and_compiled() -> None:
    prose = "其中 $V_{tm} = k_B T / q$ 是热电压，由边界条件 $V_{ch}(0) = V_s$ 导出"
    out = _prose_to_typst(prose)
    # Math mode preserved with Typst syntax, not escaped with literal \$
    assert "\\$" not in out
    assert "$V_" in out
    assert "热电压" in out
    assert "边界条件" in out


def test_prose_without_math_escaped_normally() -> None:
    prose = "Normal prose with special chars: #hashtag, *bold*, _underscore_"
    out = _prose_to_typst(prose)
    assert "\\#hashtag" in out
    assert "\\*bold\\*" in out
    assert "\\_underscore\\_" in out


def test_caption_and_inline_quantities_formatted() -> None:
    s1 = "图 3.1：依赖关系，N_ch 为沟道掺杂浓度，ψ_B = V_tm ln(N_ch = n_i)。需注意的是，式(3.1)为一维泊松方程"
    out1 = _prose_to_typst(s1)
    # Check ln(A / B) fixed from ln(A = B)
    assert 'ln(N_"ch" / n_"i")' in out1 or "ln(" in out1
    assert "ln(N_ch = n_i)" not in out1
    assert "$N_" in out1

    s2 = "使用了 N_ch = 1 × 10^15 cm^-3、TFIN = 20 nm、t_ox = 1 nm 和 V_ch = 0 V 进行仿真。"
    out2 = _prose_to_typst(s2)
    # Check scientific notation formatted into math
    assert "times 10^(15)" in out2
    assert 'T_"FIN"' in out2 or "T_FIN" in out2
    assert 't_"ox"' in out2
