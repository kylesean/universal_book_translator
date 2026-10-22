"""Inline style runs: the pdfium char probe and the reflow renderer's run markup.

Two halves, tested separately because they live in different modules:

* `styled_runs_in_box` (``ubt/adapters/pdf/textgeom.py``) reads the pdfium char
  probe and decides which source spans carry a style;
* `render_prose` (``ubt/render/typst/document_renderer.py``) locates a run's
  literal text in the *target* and wraps it in Typst markup.

The locating half is the fragile one: a run's text is its literal characters, so
it is found by substring search, and the target's publishing pass (CJK spacing,
full-width punctuation, dash folding) routinely rewrites the characters a run
was recorded with. The folded fallback is what keeps bold from being silently
dropped on a Chinese page.
"""

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


# --- the pdfium char probe -------------------------------------------------


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


# --- locating a run in the target ------------------------------------------


def test_a_run_is_located_in_order_and_a_missing_one_is_skipped() -> None:
    """Repeated tokens align to their own occurrence, not the first one."""
    text = "a Guo et al. b Guo et al. c missing"
    out = render_prose(
        text, _runs(("Guo et al.", True, False, False, None), ("nope", True, False, False, None))
    )
    assert "#strong[Guo et al.]" in out
    assert out.count("#strong[Guo et al.]") == 1


def test_fullwidth_punctuation_in_a_cjk_target_still_matches() -> None:
    """The target's publishing pass replaces the space with full-width punctuation.

    ``"50.0% "`` becomes ``"50.0%；"``, so an exact find fails and the run would
    be dropped; the folded fallback still locates it.
    """
    text = "成本降低了 50.0%；随后"
    out = render_prose(text, _runs(("50.0%", True, False, False, None)))
    assert "#strong[50.0%]" in out


def test_fullwidth_digits_and_whitespace_still_match() -> None:
    text = "准确率为 ５０．０％ ，提升明显"
    out = render_prose(text, _runs(("50.0%", True, False, False, None)))
    assert "#strong[５０．０％]" in out


def test_a_run_with_a_trailing_space_still_matches() -> None:
    """A run that still carries a trailing space must not lose its span."""
    text = "降低了50.0%；随后"
    out = render_prose(text, _runs(("50.0% ", True, False, False, None)))
    assert "#strong[50.0%]" in out


def test_dash_variants_still_match() -> None:
    """The source probe can carry an en dash the target renders as a hyphen."""
    text = "成本降低了 44.7-49.0%；随后"
    out = render_prose(text, _runs(("44.7\u201349.0%", True, False, False, None)))
    assert "#strong[44.7-49.0%]" in out


def test_a_pangu_space_inserted_in_the_target_still_matches() -> None:
    """The CJK pass inserts a space between Latin and CJK after the run was recorded."""
    text = "图 1 | SoL-Pi 通过自动化研究发现了更高效的架构。"
    run = "SoL-Pi通过自动化研究发现了更高效的架构。"
    out = render_prose(text, _runs((run, True, False, False, None)))
    assert "#strong[SoL-Pi 通过自动化研究发现了更高效的架构。]" in out


# --- emitting the markup ---------------------------------------------------


def test_a_colour_run_is_wrapped_in_a_fill() -> None:
    out = render_prose(
        "x (Guo et al., 2025) y", _runs(("Guo et al.", False, False, False, "#0000ff"))
    )
    assert '#text(fill: rgb("#0000ff"))[Guo et al.]' in out


def test_a_superscript_run_is_wrapped() -> None:
    out = render_prose("a ‡ b", _runs(("‡", False, False, True, None)))
    assert "#super(typographic: false, size: 0.62em)[‡]" in out


def test_a_run_inside_math_defers_to_the_math_render() -> None:
    """A run inside a formula would split it and escape its halves into text.

    ``$x_i$`` with a run on ``x_i`` must stay a formula; the run is dropped.
    """
    out = render_prose("$x_i$", _runs(("x_i", True, False, False, None)))
    assert out == "$x_i$"
    assert "#strong" not in out


def test_a_bold_run_around_a_formula_keeps_the_formula_math() -> None:
    out = render_prose(
        "The $x_i$ term matters.", _runs(("The $x_i$ term", True, False, False, None))
    )
    assert "#strong[The $x_i$ term]" in out


def test_a_run_the_translation_rewrote_is_drawn_plain() -> None:
    """The documented contract: a span that did not survive verbatim is skipped."""
    out = render_prose("完全不同的译文。", _runs(("missing", True, False, False, None)))
    assert out == "完全不同的译文。"


def test_no_runs_leaves_prose_unchanged() -> None:
    assert render_prose("Plain $x_i$ text.") == "Plain $x_i$ text."


# --- combining the two run sources -----------------------------------------


def test_source_and_target_runs_are_merged_source_first() -> None:
    """Source spans come first: they are usually the more specific one.

    The renderer's ``next(...)`` lookup lets the first enclosing span win, so a
    blue citation inside a bold sentence stays blue rather than becoming bold.
    """
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
    """Plain bold rendering, with no full-width punctuation near the run.

    The sibling test above exercises the *folded* fallback (``50.0%`` next to
    ``；``); this one must take the exact-match branch, or neither test covers
    the ordinary path.
    """
    text = "成本降低了 50.0% 随后"
    out = render_prose(text, _runs(("50.0%", True, False, False, None)))
    assert "#strong[50.0%]" in out
