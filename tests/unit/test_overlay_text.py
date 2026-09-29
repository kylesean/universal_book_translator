"""``overlay_text`` math-span repair: fragment merging without duplicating gaps."""

from ubt.adapters.pdf.overlay_text import repair_malformed_math_spans


def test_merged_fragment_does_not_duplicate_the_gap_text() -> None:
    """A merge across two gaps must insert each gap once.

    The forward/backward merge loops kept the accumulated ``gap`` after folding
    it into ``body``, so a formula split into three delimited fragments
    ("$x_($ mid $+$ mid2 $n_i)$") re-inserted the first gap before every later
    fragment and silently corrupted the rendered formula.
    """
    out = repair_malformed_math_spans("$x_($ mid $+$ mid2 $n_i)$")
    assert out == "$x_( mid + mid2 n_i)$", out
    assert out.count(" mid ") == 1


def test_repair_is_idempotent() -> None:
    once = repair_malformed_math_spans("$x_($ mid $+$ mid2 $n_i)$")
    assert repair_malformed_math_spans(once) == once


def test_line_start_slash_is_escaped_after_a_newline() -> None:
    """A translated line beginning with ``/`` must not re-open Typst structure.

    ``/`` at line start is Typst's term-list marker; the line-start escape set
    covered ``=``/``+``/``-``/``N.`` but not ``/``, so a translation carrying
    ``first\\n/ and more`` failed the whole overlay compile (``error: expected
    colon``) and the page's translation silently demoted. The shared guard
    (``(?=\\s|$)``) keeps mid-line paths untouched.
    """
    from ubt.adapters.pdf.overlay_text import escape_line_start_markup, typst_escape

    assert escape_line_start_markup("first\n/ and more") == "first\n\\/ and more"
    assert escape_line_start_markup("first\n/ ") == "first\n\\/ "
    # A line-start path is not a term marker: the shared guard leaves it alone.
    assert escape_line_start_markup("/usr/bin") == "/usr/bin"
    # End-to-end: typst_escape must not emit a raw term marker either.
    assert "\\/ " in typst_escape("first\n/ and more")


def test_fragments_escape_covers_line_start_slash_after_newline() -> None:
    """The fragments path must escape ``/`` after an embedded newline, not only
    at position 0 (its old guard checked only ``result.startswith("/")``)."""
    from ubt.adapters.pdf.typst_fragments import _escape_typst_markup

    assert _escape_typst_markup("first\n/ term: x") == "first\n\\/ term: x"
    assert _escape_typst_markup("/ term: x") == "\\/ term: x"
