"""Inline style runs: the pdfium char probe and the run-markup rendering."""

from __future__ import annotations

import pytest

from ubt.adapters.pdf.overlay_text import render_overlay_line
from ubt.adapters.pdf.textgeom import CharStyle, styled_runs_in_box
from ubt.core.ir.models import BlockType, InlineRun, IRBlock, StyleMeta, make_element
from ubt.render.outputs import StyledRun, _locate_runs, _styled_runs

pytestmark = pytest.mark.fast


def _char(
    x0: float, text: str, size: float = 11.0, *, bold: bool = False, color: str | None = None
) -> CharStyle:
    return CharStyle((x0, 0.0, x0 + 5.0, 10.0), text, size, bold, color)


def test_styled_runs_group_bold_superscript_and_colour_only() -> None:
    chars = [
        _char(0, "A"),
        _char(5, "B"),
        _char(10, "C", bold=True),
        _char(15, "1", size=7.0),  # < 0.8 * 11 -> superscript
        _char(20, "D", color="#0000ff"),
    ]
    runs = styled_runs_in_box(chars, (0.0, 0.0, 25.0, 12.0))
    assert runs == [
        ("C", True, False, False, None),
        ("1", False, False, True, None),
        ("D", False, False, False, "#0000ff"),
    ]


def test_a_space_rides_the_surrounding_run() -> None:
    # pdfium reports synthetic spaces with a degenerate 1pt box; they must not
    # start a spurious superscript run of their own.
    chars = [
        _char(0, "G", color="#0000ff"),
        CharStyle((5.0, 0.0, 5.0, 0.0), " ", 1.0, False, None),
        _char(5, "o", color="#0000ff"),
    ]
    runs = styled_runs_in_box(chars, (0.0, 0.0, 10.0, 12.0))
    assert runs == [("G o", False, False, False, "#0000ff")]


def test_a_trailing_space_is_not_part_of_the_run() -> None:
    # A run that ends at a synthetic space must not carry it: a CJK translation
    # drops the space (or turns it into full-width punctuation), so a run text
    # like "50.0% " would never be located in the target and the bold would be
    # silently dropped. Internal spaces still ride the run (previous test).
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


def test_locate_runs_matches_in_order_and_skips_missing() -> None:
    text = "a Guo et al. b Guo et al. c missing"
    spans = _locate_runs(
        text, (StyledRun("Guo et al."), StyledRun("Guo et al."), StyledRun("nope"))
    )
    assert [span[:2] for span in spans] == [(2, 12), (15, 25)]


def test_locate_runs_folds_fullwidth_punctuation_in_a_cjk_target() -> None:
    # The target's publishing pass replaces the source's space with full-width
    # punctuation ("50.0% " -> "50.0%；"), so an exact find fails and the run
    # would be dropped; the folded fallback still locates it.
    text = "成本降低了 50.0%；随后"
    spans = _locate_runs(text, (StyledRun("50.0%"),))
    start = text.index("50.0%")
    assert [span[:2] for span in spans] == [(start, start + len("50.0%"))]


def test_locate_runs_folds_fullwidth_digits_and_whitespace() -> None:
    text = "准确率为 ５０．０％ ，提升明显"
    spans = _locate_runs(text, (StyledRun("50.0%"),))
    assert len(spans) == 1
    assert text[spans[0][0] : spans[0][1]] == "５０．０％"


def test_locate_runs_ignores_run_edge_whitespace() -> None:
    # A run that still carries a trailing space (an older cache) must not lose
    # the span when the target dropped the space.
    text = "降低了50.0%；随后"
    spans = _locate_runs(text, (StyledRun("50.0% "),))
    assert len(spans) == 1
    assert text[spans[0][0] : spans[0][1]] == "50.0%"


def test_locate_runs_folds_dash_variants() -> None:
    # The source probe can carry an en dash the target renders as a hyphen.
    text = "成本降低了 44.7-49.0%；随后"
    spans = _locate_runs(text, (StyledRun("44.7\u201349.0%"),))
    assert len(spans) == 1
    assert text[spans[0][0] : spans[0][1]] == "44.7-49.0%"


def test_locate_runs_ignores_a_pangu_space_inserted_in_the_target() -> None:
    # The CJK pass inserts a space between Latin and CJK after the draft recorded
    # the run ("SoL-Pi通过" -> "SoL-Pi 通过"); the span must still be located.
    text = "图 1 | SoL-Pi 通过自动化研究发现了更高效的架构。"
    run = "SoL-Pi通过自动化研究发现了更高效的架构。"
    spans = _locate_runs(text, (StyledRun(run),))
    assert len(spans) == 1
    assert text[spans[0][0] : spans[0][1]] == "SoL-Pi 通过自动化研究发现了更高效的架构。"


def test_render_overlay_line_wraps_a_colour_run() -> None:
    out = render_overlay_line(
        "x (Guo et al., 2025) y", None, "zh", run_spans=[(3, 13, False, False, False, "#0000ff")]
    )
    assert '#text(fill: rgb("#0000ff"))[Guo et al.]' in out


def test_render_overlay_line_wraps_a_superscript_run() -> None:
    out = render_overlay_line("a ‡ b", None, "zh", run_spans=[(2, 3, False, False, True, None)])
    assert "#super(typographic: false, size: 0.62em)[‡]" in out


def test_bold_run_before_a_citation_paren_does_not_chain_a_call() -> None:
    # A styled run ends in ``]``; Typst reads a following ``(`` as another call
    # on the run's result (``#strong[x](y)`` -> "expected function, found
    # content") and the whole fragment fails to compile. The literal paren must
    # be emitted as an explicit ``#text("(")`` so the run's expression ends.
    line = "Firecracker microVMs (Agache et al., 2020) provide"
    bold_end = line.index(" (Agache")
    cite = line.index("Agache et al.")
    year = line.index("2020")
    out = render_overlay_line(
        line,
        None,
        "zh",
        run_spans=[
            (0, bold_end + 1, True, False, False, None),  # bold "Firecracker microVMs "
            (cite, cite + len("Agache et al."), False, False, False, "#0000ff"),
            (year, year + 4, False, False, False, "#0000ff"),
        ],
    )
    assert '#strong[Firecracker microVMs ]#text("(")' in out
    assert "](#text" not in out
    assert "](Agache" not in out


def test_a_bracket_after_a_run_is_already_escaped() -> None:
    # ``typst_escape`` backslash-escapes ``[``, so it can never chain a call on
    # the preceding run the way ``(`` does; no guard is needed.
    out = render_overlay_line("see[note]", None, "zh", run_spans=[(0, 3, True, False, False, None)])
    assert out == "#strong[see]\\[note\\]"


def test_a_run_overlapping_math_defers_to_the_math_render() -> None:
    out = render_overlay_line(
        "$x^2$", None, "zh", run_spans=[(0, 5, False, False, False, "#0000ff")]
    )
    assert "#text(fill" not in out


def test_styled_runs_merges_source_and_target_runs() -> None:
    style = StyleMeta(
        inline_runs=(InlineRun(text="2020", color_hex="#0000ff"),),
        target_runs=(InlineRun(text="重点", bold=True),),
    )
    element = make_element(
        id="b1", spine_index=1, block_type=BlockType.NARRATIVE, source_text="source"
    )
    runs = _styled_runs(IRBlock(element=element, style=style))
    assert [(r.text, r.bold, r.color_hex) for r in runs] == [
        ("2020", False, "#0000ff"),
        ("重点", True, None),
    ]


def test_a_bold_target_run_is_located_and_rendered_as_strong() -> None:
    text = "成本降低了 50.0%；随后"
    spans = _locate_runs(text, (StyledRun("50.0%", bold=True),))
    out = render_overlay_line(text, None, "zh", run_spans=spans)
    assert "#strong[50.0%]" in out
