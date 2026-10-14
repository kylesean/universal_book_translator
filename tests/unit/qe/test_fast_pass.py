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
    _missing_heading_label,
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


@pytest.mark.parametrize(
    "target",
    [
        '"The machine relies on attention and runs a forward pass."',
        "“The machine relies on attention and runs a forward pass.”",
        "'The machine relies on attention and runs a forward pass.'",
        "«The machine relies on attention and runs a forward pass.»",
        "「The machine relies on attention and runs a forward pass.」",
        "„The machine relies on attention and runs a forward pass.“",
    ],
)
def test_a_source_wrapped_in_quotes_is_still_a_near_echo(target: str) -> None:
    # The quoted-span mask exists so a preserved citation does not dominate a
    # real translation's tokens. When the quote *is* the whole target, masking
    # left nothing to judge and the retention test abstained: the same sentence
    # bare was caught but quote-wrapped ("<source>") shipped as Flawless. The
    # exact-echo check cannot see it either -- the quote characters make the
    # strings unequal -- so the near-echo check must unwrap a fully-quoted
    # target. Every delimiter style the mask knows is an evasion vector.
    src = "The machine relies on attention and runs a forward pass."
    assert is_near_verbatim_echo(src, target, target_is_cjk=False) is True


def test_a_quote_inside_real_prose_keeps_the_mask() -> None:
    # The mirror of the rule above: a target with translation words outside the
    # quote is judged on those words, not on the quotation it faithfully kept.
    src = "The machine relies on attention and runs a forward pass."
    target = 'Cette machine repose sur "the machine relies on attention".'
    assert is_near_verbatim_echo(src, target, target_is_cjk=False) is False


def test_a_preserved_quotation_does_not_read_as_a_near_echo() -> None:
    # A translation legitimately keeps a quoted string verbatim (the echo repair
    # hint tells the model to), so the quote's words must be masked before the
    # retention test: otherwise the quote dominates the target's Latin tokens and
    # a faithful translation reads as an untranslated echo.
    quote = (
        '"Please keep your hands and arms inside the vehicle '
        'at all times while the ride is in motion"'
    )
    assert (
        is_near_verbatim_echo(f"The sign reads {quote}", f"Es steht {quote}", target_is_cjk=False)
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


def test_a_line_without_a_leading_pipe_is_not_a_grid_row() -> None:
    # Inline pipes in prose are not cell separators.
    assert grid_columns("The x | y | z") is None


def test_markdown_grid_shape_needs_at_least_two_rows() -> None:
    assert markdown_grid_shape("| a | b |\n| c | d |") == [2, 2]
    assert markdown_grid_shape("| a | b |") == []
    assert markdown_grid_shape("plain text") == []


def test_prose_with_inline_pipes_is_not_a_grid() -> None:
    # Two prose lines that each happen to carry two pipes are not a table; else a
    # faithful translation without pipes is rejected as a dropped table.
    assert markdown_grid_shape("The options are A | B | C\nand D | E | F are also valid") == []


def test_a_ragged_run_of_pipe_lines_is_not_a_grid() -> None:
    # A real grid pads every row to the same column count; a ragged run does not.
    assert markdown_grid_shape("| a | b |\n| c |") == []


def test_a_grid_without_a_trailing_pipe_is_still_a_grid() -> None:
    # Only the opening pipe anchors a row; the trailing one is optional.
    assert markdown_grid_shape("| a | b\n| c | d") == [2, 2]


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


def test_half_translated_all_caps_heading_is_rejected_for_insufficient_script_density() -> None:
    src = "THE SUMMARY AND OVERVIEW"
    # THE and SUMMARY are left untranslated; only OVERVIEW is translated
    tgt = "THE SUMMARY AND 概述"
    decision = _filter().evaluate(src, tgt)
    assert not decision.passed
    assert "Insufficient zh script density" in decision.reason


# --------------------------------------------------------------------------- #
# Reasoning / chat-template scaffolding is the same defect as the XML pair
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "leaked",
    [
        "<think>reasoning</think>这是模型。",
        "这是模型。</think>",
        "<Think>x</Think>这是模型。",
        "<|im_start|>system 这是模型。",
        "<|im_end|>",
        "<|system|>这是模型。",
        "<|user|>这是模型。",
        "<issues>x</issues>",
        "<translation>y</translation>",
    ],
)
def test_model_scaffolding_is_a_fatal_leak(leaked: str) -> None:
    """A model echoing its envelope must not ship, whatever the envelope was.

    The check named two XML tags and nothing else, so a model that leaked its
    reasoning or a chat-template token passed every gate and reached the reader.
    """
    decision = FastPassFilter().validate_structural_invariants(
        source_text="The model is trained end to end.",
        target_text=leaked,
        block_type="narrative",
    )
    assert not decision.passed, leaked
    assert "Prompt template XML artifacts" in decision.reason


def test_ordinary_prose_is_not_a_leak() -> None:
    decision = FastPassFilter().validate_structural_invariants(
        source_text="The model is trained end to end.",
        target_text="该模型经过端到端训练。",
        block_type="narrative",
    )
    assert decision.passed


# ---------------------------------------------------------------------------
# arXiv 2101.03961 regression batch (real-run quarantine autopsy, 2026-10-14):
# four gate families quarantined correct work or missed real defects on that
# paper. Each test below pins one block from the run.
# ---------------------------------------------------------------------------


def test_a_fully_translated_benchmark_table_passes_the_density_gate() -> None:
    # b0142: the body is numerals and model names; only the header cells are
    # translatable, so the prose floor (0.25) read the correct table as
    # untranslated residue. Table blocks hold a near-zero identity floor —
    # an untouched table still fails (residue pure Latin, ratio 0.0).
    src = (
        "| Model | GLUE | SQuAD | SuperGLUE | Winogrande (XL) |\n"
        "|---|---|---|---|---|\n"
        "| T5-Base | 84.3 | 85.5 | 75.1 | 66.6 |\n"
        "| Switch-Base | 86.7 | 87.2 | 79.5 | 73.3 |"
    )
    tgt = src.replace("| Model |", "| 模型 |")
    decision = FastPassFilter().evaluate(src, tgt, block_type="table")
    assert decision.passed


def test_an_untouched_table_still_fails_the_density_gate() -> None:
    src = "| Model | GLUE | SQuAD |\n|---|---|---|\n| T5-Base | 84.3 | 85.5 |"
    decision = FastPassFilter().evaluate(src, src, block_type="table")
    assert not decision.passed
    assert "script density" in decision.reason


def test_a_heading_that_drops_its_appendix_letter_fails_with_a_label_reason() -> None:
    # b0028: "C Encouraging Exploration Across Experts" -> "鼓励跨专家探索".
    # The letter orders the document; no other gate saw the loss.
    decision = FastPassFilter().validate_structural_invariants(
        source_text="C Encouraging Exploration Across Experts",
        target_text="鼓励跨专家探索",
        block_type="heading",
    )
    assert not decision.passed
    assert "Heading label dropped" in decision.reason
    assert "'C'" in decision.reason


def test_a_heading_that_keeps_its_label_passes() -> None:
    for src, tgt in (
        (
            "B Preventing Token Dropping with No-Token-Left-Behind",
            "B 通过 “无令牌遗留” 机制防止令牌丢失",
        ),
        ("9 Conclusion", "9 结论"),
        ("4.1 Fine-Tuning", "4.1 微调"),
        ("2 Switch Transformer", "2 切换 Transformer"),
    ):
        decision = FastPassFilter().validate_structural_invariants(
            source_text=src, target_text=tgt, block_type="heading"
        )
        assert decision.passed, (src, tgt, decision.reason)


def test_a_dropped_section_number_fails_even_when_the_title_is_translated() -> None:
    decision = FastPassFilter().validate_structural_invariants(
        source_text="4 Downstream Results",
        target_text="下游任务结果",
        block_type="heading",
    )
    assert not decision.passed
    assert "'4'" in decision.reason


def test_an_unlabelled_heading_is_not_label_gated() -> None:
    decision = FastPassFilter().validate_structural_invariants(
        source_text="Introduction",
        target_text="引言",
        block_type="heading",
    )
    assert decision.passed


def test_a_partial_label_match_does_not_satisfy_a_deeper_label() -> None:
    # "4.1" must not be satisfied by "4" or by a fragment of "14.1".
    assert _missing_heading_label("4.1 Fine-Tuning", "4 微调") == "4.1"
    assert _missing_heading_label("4.1 Fine-Tuning", "14.1 微调") == "4.1"
    assert _missing_heading_label("4.1 Fine-Tuning", "4.1 微调") is None


def test_a_flattened_math_source_gets_the_reconstruction_hint() -> None:
    # b0068: the extractor flattened "N_{i=1}" into "N_i =1" and "W_r" into
    # "W r". The old instruction demanded full LaTeX re-emission the text
    # cannot support; three repair rounds shipped the block unchanged. The
    # flattened-source hint caps the ambition at what the fragments imply.
    src = (
        "Mixture of Expert Routing. Shazeer et al. (2017) proposed a natural "
        "language Mixtureof-Experts (MoE) layer which takes as an input a token "
        "representation x and then routes this to the best determined topk "
        "experts, selected from a set { E i ( x ) } N_i =1 of N experts. The "
        "router variable W r produces logits h ( x ) = W r · x which are "
        "normalized via a softmax distribution over the available N experts."
    )
    tgt = (
        "专家混合路由。Shazeer 等人（2017）提出了自然语言专家混合（MoE）层，该层以词元"
        "表示 x 为输入。专家 i 的门值由下式给出："
    )
    decision = FastPassFilter().evaluate(src, tgt, block_type="narrative")
    assert not decision.passed
    assert "flattened math fragments" in decision.reason
    assert "Do not invent new notation" in decision.reason


def test_a_real_math_source_keeps_the_full_redelimit_hint() -> None:
    # A source with genuine LaTeX evidence ($..$) still asks for full
    # re-delimitation. (Delimited sources route through the 3b span gate, so
    # the full hint here is exercised via a source whose markers are braces
    # and escapes without $..$ — the shape extraction leaves when it eats the
    # dollars but keeps the braces.)
    src = "The threshold F_{th,SI} governs the transition and \\tau bounds the device."
    tgt = "阈值 F_th,SI 控制转变过程，tau 限制器件。"
    decision = FastPassFilter().evaluate(src, tgt, block_type="narrative")
    assert not decision.passed
    assert "re-emit each inline physical variable" in decision.reason


def test_length_ratio_is_cjk_weighted_not_raw() -> None:
    # b0028 pre-fix: a correct 41-char heading translated to 7 Han glyphs read
    # ratio 0.17 (< floor 0.2) on raw characters. The CJK-weighted ratio puts
    # correct short translations above the floor while a genuinely truncated
    # prose block still fails.
    decision = FastPassFilter().evaluate(
        "C Encouraging Exploration Across Experts",
        "C 跨专家探索的激励机制",
        block_type="heading",
    )
    assert decision.passed
    assert decision.length_ratio > 0.2

    truncated = FastPassFilter().evaluate(
        "The model achieves state-of-the-art results on every benchmark we evaluated, "
        "spanning translation, summarization, and question answering across more than "
        "one hundred languages.",
        "该模型取得了最佳结果。",
        block_type="narrative",
    )
    assert not truncated.passed
    assert "truncated" in truncated.reason


def test_fast_pass_escaped_pipe_handling() -> None:
    r"""Verify table checker does not treat escaped pipes \| as cell separators."""
    # Row with 3 columns, middle column contains escaped pipe
    row = r"| Col 1 | Col \| 2 | Col 3 |"
    assert grid_columns(row) == 3

    # Multi-row markdown grid shape
    text = (
        r"| Col 1 | Col \| 2 | Col 3 |" + "\n"
        r"|---|---|---|" + "\n"
        r"| A | B \| C | D |"
    )
    shape = markdown_grid_shape(text)
    assert shape == [3, 3, 3]
