"""The 0-token fast-pass gate: deterministic structural checks, no model.

The gate decides whether a draft passes without spending a token. Passing means
no empty target, no prompt-template leak, no untranslated echo, no repetition
loop, no lossless-HTML/ numeric / script anomaly. It is deliberately conservative
in both directions, and the helpers below carry contracts worth pinning:

- an *exact* echo is exempt when the block is short or wordless (page numbers,
  ``Fig. 3``, DOIs legitimately survive) -- so the echo check needs a real word;
- a *near* echo exists because the exact check is blind to a one-character
  difference, but a target written in a CJK script (for a different-script
  source) is translated by definition, and a ``--dry-run`` rehearsal is not a
  defect;
- a repetition loop is a hallucination only when it is not aligned to a source
  repetition (a target refrain no longer than the source's is faithful, but a
  runaway loop that repeats far more lines than the source is still caught).
"""

from __future__ import annotations

import pytest

from ubt.core.qe.defect_taxonomy import ECHO_MARKER
from ubt.core.qe.fast_pass import (
    REHEARSAL_MARKER,
    FastPassFilter,
    _detect_line_repetition_loop,
    _has_repeated_line_run,
    _math_spans_equivalent,
    _max_repeated_line_run,
    _normalize_math_body,
    _strip_verbatim_term,
    grid_columns,
    is_near_verbatim_echo,
    is_verbatim_echo,
    markdown_grid_shape,
)

pytestmark = pytest.mark.fast

_ECHO = "The quick brown fox jumps over the lazy dog."
_CJK_TRANSLATION = "\u5feb\u901f\u7684\u68d5\u8272\u72d0\u72f8\u8df3\u8fc7\u4e86\u61d2\u72d7\u3002"
_CJK_SOURCE = (
    "\u8fd9\u662f\u4e00\u4e2a\u7528\u4e8e\u6d4b\u8bd5\u7684\u4e2d\u6587\u53e5\u5b50\uff0c"
    "\u5305\u542b\u8db3\u591f\u7684\u5b57\u7b26\u3002"
)
_CJK_OTHER = (
    "\u8fd9\u662f\u4e00\u4e2a\u5b8c\u5168\u4e0d\u540c\u7684\u4e2d\u6587\u53e5\u5b50\uff0c"
    "\u5185\u5bb9\u5df2\u7ecf\u6539\u53d8\u4e86\u3002"
)


# --------------------------------------------------------------------------- #
# Exact echo.
# --------------------------------------------------------------------------- #


def test_a_long_prose_echo_is_detected() -> None:
    assert is_verbatim_echo(_ECHO, _ECHO) is True


def test_the_echo_check_ignores_surrounding_whitespace() -> None:
    assert is_verbatim_echo(_ECHO, f"  {_ECHO}  ") is True


def test_a_different_target_is_not_an_echo() -> None:
    assert is_verbatim_echo(_ECHO, _CJK_TRANSLATION) is False


def test_a_short_block_is_exempt_from_the_echo_check() -> None:
    # "Fig. 3" and page numbers legitimately survive the trip unchanged.
    assert is_verbatim_echo("Fig. 3", "Fig. 3") is False


def test_a_formatting_only_block_is_exempt() -> None:
    assert is_verbatim_echo("| --- | --- | --- |", "| --- | --- | --- |") is False


def test_a_wordless_block_is_exempt_even_when_long() -> None:
    assert is_verbatim_echo("1234567890 12345 6789", "1234567890 12345 6789") is False


def test_an_empty_source_is_not_an_echo() -> None:
    assert is_verbatim_echo("", "") is False


# --------------------------------------------------------------------------- #
# Near echo.
# --------------------------------------------------------------------------- #


def test_an_identical_latin_pair_is_a_near_echo() -> None:
    assert is_near_verbatim_echo(_ECHO, _ECHO, target_is_cjk=False) is True


def test_a_rehearsal_marker_is_not_a_near_echo() -> None:
    # A --dry-run echo is an intentional rehearsal artifact, not a defect.
    target = f"{REHEARSAL_MARKER} {_ECHO}"
    assert is_near_verbatim_echo(_ECHO, target, target_is_cjk=False) is False


def test_a_cjk_target_for_a_different_script_source_is_exempt() -> None:
    # A target written in a CJK script is translated by definition. The exemption
    # matters for a technical paper: the correct translation still carries eight
    # Latin proper nouns, so without it the Latin retention test reads ~1.0 and
    # quarantines a translated paragraph.
    source = "FlashAttention Transformer BERT RoBERTa GPT LLaMA ViT ResNet models improve accuracy."
    target = (
        "FlashAttention Transformer BERT RoBERTa GPT LLaMA ViT ResNet "
        "\u6a21\u578b\u63d0\u5347\u4e86\u51c6\u786e\u7387\u3002"
    )
    assert is_near_verbatim_echo(source, target, target_is_cjk=True) is False


def test_a_same_script_cjk_copy_is_a_near_echo() -> None:
    # zh<->ja/ko: a CJK target is NOT translated by definition, so bigram
    # retention catches a copied sentence.
    assert is_near_verbatim_echo(_CJK_SOURCE, _CJK_SOURCE, target_is_cjk=True, source_is_cjk=True)


def test_a_same_script_cjk_translation_is_not_a_near_echo() -> None:
    assert (
        is_near_verbatim_echo(_CJK_SOURCE, _CJK_OTHER, target_is_cjk=True, source_is_cjk=True)
        is False
    )


# --------------------------------------------------------------------------- #
# Markdown grid shape.
# --------------------------------------------------------------------------- #


def test_grid_columns_counts_cells_and_ignores_edge_pipes() -> None:
    assert grid_columns("| a | b |") == 2
    assert grid_columns("| a | b") == 2


def test_a_line_with_fewer_than_two_pipes_is_not_a_grid() -> None:
    assert grid_columns("no pipe here") is None
    assert grid_columns("| only one") is None


def test_markdown_grid_shape_needs_at_least_two_rows() -> None:
    assert markdown_grid_shape("| a | b |\n| c | d |") == [2, 2]
    assert markdown_grid_shape("| a | b |") == []
    assert markdown_grid_shape("plain text") == []


# --------------------------------------------------------------------------- #
# Math span equivalence.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("span", "body"),
    [
        ("$$x^2$$", "x^2"),
        ("$x$", "x"),
        (r"\(x\)", "x"),
        (r"\[x\]", "x"),
        ("plain", "plain"),
    ],
)
def test_math_delimiters_are_stripped_for_comparison(span: str, body: str) -> None:
    assert _normalize_math_body(span) == body


def test_identical_math_spans_are_equivalent() -> None:
    assert _math_spans_equivalent(["$x$"], ["$x$"]) is True


def test_different_delimiters_around_the_same_body_are_equivalent() -> None:
    assert _math_spans_equivalent(["$x^2$"], ["$$x^2$$"]) is True


def test_a_different_span_count_is_not_equivalent() -> None:
    assert _math_spans_equivalent(["$x$"], ["$x$", "$y$"]) is False


def test_different_math_content_is_not_equivalent() -> None:
    assert _math_spans_equivalent([r"x^{2}+y"], [r"y + x^{2}"]) is False


# --------------------------------------------------------------------------- #
# Line-repetition loops.
# --------------------------------------------------------------------------- #


def test_a_run_of_identical_lines_is_a_loop() -> None:
    loop = "This is a repeated hallucination line.\n" * 4
    assert _detect_line_repetition_loop(loop) == "This is a repeated hallucination line."


def test_non_contiguous_repeats_are_not_a_loop() -> None:
    assert _detect_line_repetition_loop("alpha\nbeta\ngamma") is None


def test_a_blank_line_resets_the_run() -> None:
    # Legitimate stanzas separated by whitespace stay exempt.
    assert _detect_line_repetition_loop("same\nsame\n\nsame\nsame") is None


def test_a_run_without_word_content_is_not_a_loop() -> None:
    # Sparse numeric table rows like "| 0 | 0 |" must not trip the detector.
    assert _detect_line_repetition_loop("| 0 | 0 |\n| 0 | 0 |\n| 0 | 0 |\n| 0 | 0 |") is None


def test_has_repeated_line_run_matches_the_loop_detector() -> None:
    loop = "This is a repeated hallucination line.\n" * 4
    assert _has_repeated_line_run(loop) is True
    assert _has_repeated_line_run("alpha\nbeta") is False


# --------------------------------------------------------------------------- #
# FastPassFilter.evaluate: the gate end to end.
# --------------------------------------------------------------------------- #


def _filter() -> FastPassFilter:
    return FastPassFilter(source_lang="en", target_lang="zh")


def test_a_clean_translation_passes() -> None:
    decision = _filter().evaluate(
        "The quick brown fox jumps over the lazy dog and runs.",
        "\u5feb\u901f\u7684\u68d5\u8272\u72d0\u72f8\u8df3\u8fc7\u4e86\u61d2\u72d7\u5e76\u4e14\u8dd1\u8d70\u4e86\u3002",
    )
    assert decision.passed
    assert decision.reason.startswith("Flawless")


def test_an_empty_target_fails() -> None:
    decision = _filter().evaluate("Source text.", "")
    assert not decision.passed
    assert decision.reason == "Empty target text"


def test_prompt_template_leakage_fails() -> None:
    decision = _filter().evaluate("Source text.", "<issues>leaked</issues>")
    assert not decision.passed
    assert "XML artifacts leaked" in decision.reason


def test_a_verbatim_echo_is_rejected_with_the_echo_marker() -> None:
    decision = _filter().evaluate(_ECHO, _ECHO)
    assert not decision.passed
    assert decision.reason.startswith(ECHO_MARKER)


def test_a_repetition_loop_is_rejected() -> None:
    loop = "This is a repeated hallucination line.\n" * 4
    decision = _filter().evaluate("A completely different source sentence.", loop)
    assert not decision.passed
    assert "loop hallucination" in decision.reason


def test_a_runaway_loop_is_rejected_even_when_the_source_repeats_elsewhere() -> None:
    # The source carries an unrelated 4-line refrain; the target repeats its own
    # line 8 times. The old blanket "source repeats anything -> skip every
    # repetition check" let this hallucination loop through.
    src = "refrain here\n" * 4 + "A completely different closing sentence about the topic."
    tgt = "This is a repeated hallucination line.\n" * 8
    assert _max_repeated_line_run(src) == 4
    assert _max_repeated_line_run(tgt) == 8
    decision = _filter().evaluate(src, tgt)
    assert not decision.passed
    assert "loop hallucination" in decision.reason


def test_a_faithful_refrain_is_not_a_line_loop() -> None:
    # Source and target repeat the same number of lines: a preserved refrain.
    # The line-run gate must not fire (the run is not longer than the source's).
    src = "refrain here\n" * 4 + "closing line"
    tgt = "副歌在此\n" * 4 + "结尾行"
    assert _max_repeated_line_run(src) == 4
    assert _max_repeated_line_run(tgt) == 4
    decision = _filter().evaluate(src, tgt)
    assert "loop hallucination" not in decision.reason


def test_emails_and_author_lines_not_rejected_as_near_verbatim_echo() -> None:
    src = "Youngmok Jung: yjung24@apple.com, Manjot Bilkhu: mbilkhu@apple.com"
    tgt = "Youngmok Jung：yjung24@apple.com，Manjot Bilkhu：mbilkhu@apple.com"
    assert is_near_verbatim_echo(src, tgt, target_is_cjk=True) is False


# --------------------------------------------------------------------------- #
# Verbatim-term residue stripping.
# --------------------------------------------------------------------------- #


def test_verbatim_term_strip_keeps_letter_boundaries() -> None:
    # A bare substring strip would eat the PRI|OR|ITY of an untranslated word.
    residue = _strip_verbatim_term("PRIORITY data", "OR")
    assert "PRIORITY" in residue
    assert _strip_verbatim_term("low OR high", "OR") == "low  high"
    # CJK-adjacent carry-overs still strip: the guard is letter-only, and a
    # CJK character is not a letter boundary blocker.
    assert _strip_verbatim_term("内存PagedAttention池", "PagedAttention") == "内存池"
    # Identifier-shaped terms carrying digits keep plain replacement.
    assert _strip_verbatim_term("GQA-8 blocks", "GQA-8") == " blocks"


def test_repetition_detectors_agree() -> None:
    # _has_repeated_line_run delegates to the loop detector; pin the contract
    # so the two entry points cannot drift apart again.
    loop = "This is a repeated hallucination line.\n" * 4
    assert _has_repeated_line_run(loop) is True
    assert _detect_line_repetition_loop(loop) is not None
    plain = "alpha\nbeta\ngamma\n"
    assert _has_repeated_line_run(plain) is False
    assert _detect_line_repetition_loop(plain) is None


def test_a_blank_separated_repetition_loop_is_rejected() -> None:
    # A model loop often separates its repeats with a blank line, which the
    # contiguous detector resets on (to spare stanzas). A run this long is a
    # runaway loop, not a refrain.
    src = "Agentic sandbox workloads have several properties that shape the platform design:"
    tgt = "代理型沙箱工作负载具备若干特性，这些特性直接影响了平台的设计：\n\n" * 8
    decision = _filter().evaluate(src, tgt)
    assert not decision.passed
    assert "loop hallucination" in decision.reason


def test_a_short_blank_separated_refrain_is_not_a_loop() -> None:
    # Below the runaway threshold the stanza exemption still holds.
    src = "A source line."
    tgt = "同一行\n\n同一行\n\n同一行\n\n同一行"
    decision = _filter().evaluate(src, tgt)
    assert "loop hallucination" not in decision.reason
