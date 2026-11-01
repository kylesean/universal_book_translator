from __future__ import annotations

import pytest

from ubt.adapters.pdf.textgeom import CharStyle, styled_runs_in_box
from ubt.core.ir.models import BlockType, InlineRun, IRBlock, StyleMeta, make_element
from ubt.render.inline_spans import block_runs
from ubt.render.typst.document_renderer import render_prose

pytestmark = pytest.mark.fast


def _char(
    x0: float, text: str, size: float = 11.0, *, bold: bool = False, color: str | None = None
) -> CharStyle:
    return CharStyle((x0, 0.0, x0 + 5.0, 10.0), text, size, bold, color)


def _runs(*specs: tuple[str, bool, bool, bool, str | None]) -> tuple[InlineRun, ...]:
    return tuple(
        InlineRun(text=text, bold=bold, italic=italic, superscript=sup, color_hex=color)
        for text, bold, italic, sup, color in specs
    )


def test_styled_runs_group_bold_superscript_and_colour_only() -> None:
    chars = [
        _char(0, "A"),
        _char(5, "B"),
        _char(10, "C", bold=True),
        CharStyle((15.0, 2.5, 20.0, 10.0), "1", 7.0, False, None),
        _char(20, "D", color="#0000ff"),
    ]
    runs = styled_runs_in_box(chars, (0.0, 0.0, 25.0, 12.0))
    assert runs == [
        ("C", True, False, False, None),
        ("1", False, False, True, None),
        ("D", False, False, False, "#0000ff"),
    ]


def test_small_print_on_the_baseline_is_not_a_superscript() -> None:
    chars = [
        _body(0, "O", height=8.5),
        _body(5, "S", height=6.8),
        _body(10, "T", height=6.5),
        _body(15, "D", height=6.5),
        _body(20, "s", height=5.8),
    ]
    runs = styled_runs_in_box(chars, (0.0, -2.0, 25.0, 12.0))
    assert runs == []


def test_a_space_rides_the_surrounding_run() -> None:
    chars = [
        _char(0, "G", color="#0000ff"),
        CharStyle((5.0, 0.0, 5.0, 0.0), " ", 1.0, False, None),
        _char(5, "o", color="#0000ff"),
    ]
    runs = styled_runs_in_box(chars, (0.0, 0.0, 10.0, 12.0))
    assert runs == [("G o", False, False, False, "#0000ff")]


def _body(x0: float, text: str, *, bottom: float = 0.0, height: float = 10.0) -> CharStyle:
    return CharStyle((x0, bottom, x0 + 5.0, bottom + height), text, 1.0, False, None)


def test_a_matrix_scaled_marker_is_raised_by_geometry() -> None:
    chars = [
        _body(0, "Y"),
        _body(5, "u"),
        _body(10, "k"),
        _body(15, "e"),
        _body(20, " "),
        _body(25, "1", bottom=4.0, height=7.7),
    ]
    runs = styled_runs_in_box(chars, (0.0, -2.0, 30.0, 12.0))
    assert runs == [("1", False, False, True, None)]


def test_a_hanging_hyphen_is_not_a_marker() -> None:
    chars = [
        _body(0, "R"),
        _body(5, "u"),
        _body(10, "s"),
        _body(15, "t"),
        _body(20, "-", bottom=4.0, height=2.5),
        _body(25, "b"),
        _body(30, "a"),
    ]
    runs = styled_runs_in_box(chars, (0.0, -2.0, 35.0, 12.0))
    assert runs == []


def test_a_marker_comma_rides_the_raised_run() -> None:
    chars = [
        _body(0, "Z"),
        _body(5, "h"),
        _body(10, "a"),
        _body(15, "n"),
        _body(20, "g"),
        _body(25, "3", bottom=4.0, height=7.7),
        _body(30, ",", bottom=3.4, height=7.7),
        _body(35, "4", bottom=4.0, height=7.7),
    ]
    runs = styled_runs_in_box(chars, (0.0, -2.0, 40.0, 12.0))
    assert runs == [("3,4", False, False, True, None)]


def test_a_trailing_space_is_not_part_of_the_run() -> None:
    chars = [
        _char(0, "5", bold=True),
        _char(5, "0", bold=True),
        _char(10, ".", bold=True),
        _char(15, "0", bold=True),
        _char(20, "%", bold=True),
        CharStyle((25.0, 0.0, 25.0, 0.0), " ", 1.0, False, None),
        _char(25, "A"),
    ]
    runs = styled_runs_in_box(chars, (0.0, 0.0, 30.0, 12.0))
    assert runs == [("50.0%", True, False, False, None)]


def test_a_run_is_located_in_order_and_a_missing_one_is_skipped() -> None:
    text = "a Guo et al. b Guo et al. c missing"
    out = render_prose(
        text, _runs(("Guo et al.", True, False, False, None), ("nope", True, False, False, None))
    )
    assert "#strong[Guo et al.]" in out
    assert out.count("#strong[Guo et al.]") == 1


def test_fullwidth_punctuation_in_a_cjk_target_still_matches() -> None:
    text = "成本降低了 50.0%；随后"
    out = render_prose(text, _runs(("50.0%", True, False, False, None)))
    assert "#strong[50.0%]" in out


def test_fullwidth_digits_and_whitespace_still_match() -> None:
    text = "准确率为 ５０．０％ ，提升明显"
    out = render_prose(text, _runs(("50.0%", True, False, False, None)))
    assert "#strong[５０．０％]" in out


def test_a_run_with_a_trailing_space_still_matches() -> None:
    text = "降低了50.0%；随后"
    out = render_prose(text, _runs(("50.0% ", True, False, False, None)))
    assert "#strong[50.0%]" in out


def test_dash_variants_still_match() -> None:
    text = "成本降低了 44.7-49.0%；随后"
    out = render_prose(text, _runs(("44.7\u201349.0%", True, False, False, None)))
    assert "#strong[44.7-49.0%]" in out


def test_a_pangu_space_inserted_in_the_target_still_matches() -> None:
    text = "图 1 | SoL-Pi 通过自动化研究发现了更高效的架构。"
    run = "SoL-Pi通过自动化研究发现了更高效的架构。"
    out = render_prose(text, _runs((run, True, False, False, None)))
    assert "#strong[SoL-Pi 通过自动化研究发现了更高效的架构。]" in out


def test_a_colour_run_is_wrapped_in_a_fill() -> None:
    out = render_prose(
        "x (Guo et al., 2025) y", _runs(("Guo et al.", False, False, False, "#0000ff"))
    )
    assert '#text(fill: rgb("#0000ff"))[Guo et al.]' in out


def test_a_superscript_run_is_wrapped() -> None:
    out = render_prose("a ‡ b", _runs(("‡", False, False, True, None)))
    assert "#super(typographic: false, size: 0.62em)[‡]" in out


def test_a_run_inside_math_defers_to_the_math_render() -> None:
    out = render_prose("$x_i$", _runs(("x_i", True, False, False, None)))
    assert out == "$x_i$"
    assert "#strong" not in out


def test_a_bold_run_around_a_formula_keeps_the_formula_math() -> None:
    out = render_prose(
        "The $x_i$ term matters.", _runs(("The $x_i$ term", True, False, False, None))
    )
    assert "#strong[The $x_i$ term]" in out


def test_a_run_the_translation_rewrote_is_drawn_plain() -> None:
    out = render_prose("完全不同的译文。", _runs(("missing", True, False, False, None)))
    assert out == "完全不同的译文。"


def test_no_runs_leaves_prose_unchanged() -> None:
    assert render_prose("Plain $x_i$ text.") == "Plain $x_i$ text."


def test_source_and_target_runs_are_merged_source_first() -> None:
    style = StyleMeta(
        inline_runs=(InlineRun(text="2020", color_hex="#0000ff"),),
        target_runs=(InlineRun(text="重点", bold=True),),
    )
    element = make_element(
        id="b1", spine_index=1, block_type=BlockType.NARRATIVE, source_text="source"
    )
    runs = block_runs(IRBlock(element=element, style=style))
    assert [(r.text, r.bold, r.color_hex) for r in runs] == [
        ("2020", False, "#0000ff"),
        ("重点", True, None),
    ]


def test_a_bold_target_run_renders_as_strong() -> None:
    text = "成本降低了 50.0% 随后"
    out = render_prose(text, _runs(("50.0%", True, False, False, None)))
    assert "#strong[50.0%]" in out


def test_superscript_run_does_not_split_decimal_or_year() -> None:
    text = "模型参数达到 7.39 亿，引用自 7。"
    out = render_prose(text, _runs(("7", False, False, True, None)))
    assert "7.39" in out
    assert "[7].39" not in out
    assert "[7]" in out

    text_year = "如 Fedus 等（2021）所述"
    out_year = render_prose(text_year, _runs(("20", False, False, True, None)))
    assert "（2021）" in out_year
    assert "[20]" not in out_year
