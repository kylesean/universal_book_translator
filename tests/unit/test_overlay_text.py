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
