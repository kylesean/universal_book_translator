"""Inline style runs: the pdfium char probe and the run-markup rendering."""

from __future__ import annotations

import pytest

from ubt.adapters.pdf.overlay_text import render_overlay_line
from ubt.adapters.pdf.textgeom import CharStyle, styled_runs_in_box
from ubt.render.outputs import StyledRun, _locate_runs

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


def test_locate_runs_matches_in_order_and_skips_missing() -> None:
    text = "a Guo et al. b Guo et al. c missing"
    spans = _locate_runs(
        text, (StyledRun("Guo et al."), StyledRun("Guo et al."), StyledRun("nope"))
    )
    assert [span[:2] for span in spans] == [(2, 12), (15, 25)]


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
