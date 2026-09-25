"""Tests for the content-preserving Typst fallback ladder.

``_degrade_failing_line`` is the last resort when Typst refuses a line. The
contract: inline math is demoted to a visible code span (content survives as
text), and the whole line is commented out — recorded as content loss — only
when there is no inline math left to keep.
"""

from __future__ import annotations

from ubt.adapters.pdf.typst_reconstructor import _degrade_failing_line


def test_inline_math_is_demoted_not_removed() -> None:
    new_line, removed = _degrade_failing_line("其中 $V_{th}$ 是关键电压。")
    assert not removed
    assert new_line == "其中 `$V_{th}$` 是关键电压。"
    assert "// [UBT_SYNTAX_FALLBACK]" not in new_line


def test_multiple_spans_are_all_demoted() -> None:
    new_line, removed = _degrade_failing_line("$a$ 与 $b$ 的关系")
    assert not removed
    assert new_line == "`$a$` 与 `$b$` 的关系"


def test_line_without_math_is_commented_and_counted_as_removed() -> None:
    new_line, removed = _degrade_failing_line('broken " prose line')
    assert removed
    assert new_line.startswith("// [UBT_SYNTAX_FALLBACK] ")
    assert '"' not in new_line.split("] ", 1)[1]  # sanitized for the comment body


def test_unbalanced_dollar_falls_back_to_removal() -> None:
    # An odd `$` has no matching span to demote, so the line is commented.
    new_line, removed = _degrade_failing_line("$V_th = 1")
    assert removed
    assert new_line.startswith("// [UBT_SYNTAX_FALLBACK] ")


def test_backticks_in_span_are_neutralized() -> None:
    new_line, removed = _degrade_failing_line("x $a`b$ y")
    assert not removed
    assert new_line == "x `$a'b$` y"


def test_already_degraded_line_does_not_multiply_backticks() -> None:
    degraded1, removed1 = _degrade_failing_line("应用 $E_x$ = 0")
    assert not removed1
    assert degraded1 == "应用 `$E_x$` = 0"

    # Running degradation again on already degraded line should not produce quadruple backticks
    degraded2, removed2 = _degrade_failing_line(degraded1)
    assert not removed2 or degraded2.startswith("// [UBT_SYNTAX_FALLBACK]")
    assert "````" not in degraded2
