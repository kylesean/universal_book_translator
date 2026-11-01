from __future__ import annotations

import pytest

from ubt.core.qe.defect_taxonomy import ECHO_MARKER
from ubt.core.qe.fast_pass import (
    REHEARSAL_MARKER,
    FastPassFilter,
    _math_spans_equivalent,
    _max_repeated_line_run,
    _missing_heading_label,
    _normalize_math_body,
    _repeated_line_runs,
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


def test_a_long_prose_echo_is_detected() -> None:
    assert is_verbatim_echo(_ECHO, _ECHO) is True


def test_the_echo_check_ignores_surrounding_whitespace() -> None:
    assert is_verbatim_echo(_ECHO, f"  {_ECHO}  ") is True


def test_a_different_target_is_not_an_echo() -> None:
    assert is_verbatim_echo(_ECHO, _CJK_TRANSLATION) is False


def test_a_short_block_is_exempt_from_the_echo_check() -> None:
    assert is_verbatim_echo("Fig. 3", "Fig. 3") is False


def test_a_formatting_only_block_is_exempt() -> None:
    assert is_verbatim_echo("| --- | --- | --- |", "| --- | --- | --- |") is False


def test_a_wordless_block_is_exempt_even_when_long() -> None:
    assert is_verbatim_echo("1234567890 12345 6789", "1234567890 12345 6789") is False


def test_an_empty_source_is_not_an_echo() -> None:
    assert is_verbatim_echo("", "") is False


def test_an_identical_latin_pair_is_a_near_echo() -> None:
    assert is_near_verbatim_echo(_ECHO, _ECHO, target_is_cjk=False) is True


def test_a_rehearsal_marker_alone_does_not_exempt_a_near_echo() -> None:
    target = f"{REHEARSAL_MARKER} {_ECHO}"
    assert is_near_verbatim_echo(_ECHO, target, target_is_cjk=False) is True


def test_the_rehearsal_exemption_is_an_explicit_flag_not_a_substring() -> None:
    gate = FastPassFilter()
    echo_text = f"{REHEARSAL_MARKER} {_ECHO}"
    assert gate.evaluate(_ECHO, echo_text, block_type=None).passed is False
    assert gate.evaluate(_ECHO, echo_text, block_type=None, rehearsal=True).passed is True


def test_a_cjk_target_for_a_different_script_source_is_exempt() -> None:
    source = "FlashAttention Transformer BERT RoBERTa GPT LLaMA ViT ResNet models improve accuracy."
    target = (
        "FlashAttention Transformer BERT RoBERTa GPT LLaMA ViT ResNet "
        "\u6a21\u578b\u63d0\u5347\u4e86\u51c6\u786e\u7387\u3002"
    )
    assert is_near_verbatim_echo(source, target, target_is_cjk=True) is False


def test_a_same_script_cjk_copy_is_a_near_echo() -> None:
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
    src = "The machine relies on attention and runs a forward pass."
    assert is_near_verbatim_echo(src, target, target_is_cjk=False) is True


def test_a_quote_inside_real_prose_keeps_the_mask() -> None:
    src = "The machine relies on attention and runs a forward pass."
    target = 'Cette machine repose sur "the machine relies on attention".'
    assert is_near_verbatim_echo(src, target, target_is_cjk=False) is False


def test_a_preserved_quotation_does_not_read_as_a_near_echo() -> None:
    quote = (
        '"Please keep your hands and arms inside the vehicle '
        'at all times while the ride is in motion"'
    )
    assert (
        is_near_verbatim_echo(f"The sign reads {quote}", f"Es steht {quote}", target_is_cjk=False)
        is False
    )


def test_grid_columns_counts_cells_and_ignores_edge_pipes() -> None:
    assert grid_columns("| a | b |") == 2
    assert grid_columns("| a | b") == 2


def test_a_line_with_fewer_than_two_pipes_is_not_a_grid() -> None:
    assert grid_columns("no pipe here") is None
    assert grid_columns("| only one") is None


def test_a_line_without_a_leading_pipe_is_not_a_grid_row() -> None:
    assert grid_columns("The x | y | z") is None


def test_markdown_grid_shape_needs_at_least_two_rows() -> None:
    assert markdown_grid_shape("| a | b |\n| c | d |") == [2, 2]
    assert markdown_grid_shape("| a | b |") == []
    assert markdown_grid_shape("plain text") == []


def test_a_single_row_table_is_a_grid_when_it_has_a_delimiter_row() -> None:
    assert markdown_grid_shape("| Model | Acc |\n|---|---|") == [2, 2]


def test_prose_with_inline_pipes_is_not_a_grid() -> None:
    assert markdown_grid_shape("The options are A | B | C\nand D | E | F are also valid") == []


def test_a_ragged_run_of_pipe_lines_is_a_grid_with_reported_raggedness() -> None:
    assert markdown_grid_shape("| a | b |\n| c |") == [2, 1]


def test_a_ragged_target_is_a_grid_mismatch_not_a_dropped_table() -> None:
    fp = FastPassFilter(source_lang="en", target_lang="zh")
    decision = fp.evaluate(
        "| a | b |\n| 1 | 2 |",
        "| 甲 | 乙 |\n| 1 | 2 | 3 |",
        block_type="table",
    )
    assert not decision.passed
    assert "Table grid mismatch" in decision.reason
    assert "ragged" in decision.reason


def test_a_grid_without_a_trailing_pipe_is_still_a_grid() -> None:
    assert markdown_grid_shape("| a | b\n| c | d") == [2, 2]


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


def test_a_run_of_identical_lines_is_a_loop() -> None:
    loop = "This is a repeated hallucination line.\n" * 4
    assert _repeated_line_runs(loop)[0][0] == "This is a repeated hallucination line."


def test_non_contiguous_repeats_are_not_a_loop() -> None:
    assert _repeated_line_runs("alpha\nbeta\ngamma") == []


def test_a_blank_line_resets_the_run() -> None:
    assert _repeated_line_runs("same\nsame\n\nsame\nsame") == []


def test_a_run_without_word_content_is_not_a_loop() -> None:
    assert _repeated_line_runs("| 0 | 0 |\n| 0 | 0 |\n| 0 | 0 |\n| 0 | 0 |") == []


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
    src = "refrain here\n" * 4 + "A completely different closing sentence about the topic."
    tgt = "This is a repeated hallucination line.\n" * 8
    assert _max_repeated_line_run(src) == 4
    assert _max_repeated_line_run(tgt) == 8
    decision = _filter().evaluate(src, tgt)
    assert not decision.passed
    assert "loop hallucination" in decision.reason


def test_a_faithful_refrain_is_not_a_line_loop() -> None:
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


def test_verbatim_term_strip_keeps_letter_boundaries() -> None:
    residue = _strip_verbatim_term("PRIORITY data", "OR")
    assert "PRIORITY" in residue
    assert _strip_verbatim_term("low OR high", "OR") == "low  high"
    assert _strip_verbatim_term("内存PagedAttention池", "PagedAttention") == "内存池"
    assert _strip_verbatim_term("GQA-8 blocks", "GQA-8") == " blocks"


def test_repetition_detectors_agree() -> None:
    loop = "This is a repeated hallucination line.\n" * 4
    assert _max_repeated_line_run(loop) >= 4
    assert _repeated_line_runs(loop) != []
    plain = "alpha\nbeta\ngamma\n"
    assert _max_repeated_line_run(plain) < 2
    assert _repeated_line_runs(plain) == []


def test_a_blank_separated_repetition_loop_is_rejected() -> None:
    src = "Agentic sandbox workloads have several properties that shape the platform design:"
    tgt = "代理型沙箱工作负载具备若干特性，这些特性直接影响了平台的设计：\n\n" * 8
    decision = _filter().evaluate(src, tgt)
    assert not decision.passed
    assert "loop hallucination" in decision.reason


def test_a_short_blank_separated_refrain_is_not_a_loop() -> None:
    src = "A source line."
    tgt = "同一行\n\n同一行\n\n同一行\n\n同一行"
    decision = _filter().evaluate(src, tgt)
    assert "loop hallucination" not in decision.reason


def test_half_translated_all_caps_heading_is_rejected_for_insufficient_script_density() -> None:
    src = "THE SUMMARY AND OVERVIEW"
    tgt = "THE SUMMARY AND 概述"
    decision = _filter().evaluate(src, tgt)
    assert not decision.passed
    assert "Insufficient zh script density" in decision.reason


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
    decision = FastPassFilter().validate_structural_invariants(
        source_text="The model is trained end to end.",
        target_text=leaked,
        block_type="narrative",
    )
    assert not decision.passed, leaked
    assert "Prompt template XML artifacts" in decision.reason


@pytest.mark.parametrize(
    "token",
    [
        "⟦SOUP_MASK_0001-f2d⟧",
        "⟦MATH_MASK_0003-46⟧",
        "⟦CITE_MASK_0010-abc⟧",
    ],
)
def test_a_residual_mask_token_is_a_structural_failure(token: str) -> None:
    decision = FastPassFilter().validate_structural_invariants(
        source_text="The model is trained end to end.",
        target_text=f"该模型{token}经过端到端训练。",
        block_type="narrative",
    )
    assert not decision.passed
    assert "Residual mask token" in decision.reason


def test_emphasis_markers_are_not_mask_tokens() -> None:
    decision = FastPassFilter().validate_structural_invariants(
        source_text="The model is trained end to end.",
        target_text="该模型⟦B⟧经过⟦/B⟧端到端训练。",
        block_type="narrative",
    )
    assert decision.passed, decision.reason


def test_ordinary_prose_is_not_a_leak() -> None:
    decision = FastPassFilter().validate_structural_invariants(
        source_text="The model is trained end to end.",
        target_text="该模型经过端到端训练。",
        block_type="narrative",
    )
    assert decision.passed


def test_a_fully_translated_benchmark_table_passes_the_density_gate() -> None:
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
    assert _missing_heading_label("4.1 Fine-Tuning", "4 微调") == "4.1"
    assert _missing_heading_label("4.1 Fine-Tuning", "14.1 微调") == "4.1"
    assert _missing_heading_label("4.1 Fine-Tuning", "4.1 微调") is None


def test_a_flattened_math_source_gets_the_reconstruction_hint() -> None:
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
    src = "The threshold F_{th,SI} governs the transition and \\tau bounds the device."
    tgt = "阈值 F_th,SI 控制转变过程，tau 限制器件。"
    decision = FastPassFilter().evaluate(src, tgt, block_type="narrative")
    assert not decision.passed
    assert "re-emit each inline physical variable" in decision.reason


def test_length_ratio_is_cjk_weighted_not_raw() -> None:
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
    row = r"| Col 1 | Col \| 2 | Col 3 |"
    assert grid_columns(row) == 3

    text = (
        r"| Col 1 | Col \| 2 | Col 3 |" + "\n"
        r"|---|---|---|" + "\n"
        r"| A | B \| C | D |"
    )
    shape = markdown_grid_shape(text)
    assert shape == [3, 3, 3]


def test_the_grid_label_counts_rows_and_columns_separately() -> None:
    from ubt.core.qe.fast_pass import _grid_label

    assert _grid_label([5, 5, 5]) == "a 3-row x 5-column grid"
    assert _grid_label([2]) == "a 1-row x 2-column grid"
    assert _grid_label([]) == "an empty grid"
    assert _grid_label([3, 4]) == "a ragged 2-row grid (3, 4 columns per row)"


@pytest.mark.asyncio
async def test_a_dropped_table_row_is_reported_as_a_row_difference() -> None:
    from ubt.core.qe.fast_pass import FastPassFilter

    header = "| a | b | c | d | e |"
    sep = "| --- | --- | --- | --- | --- |"
    row = "| 1 | 2 | 3 | 4 | 5 |"
    source = "\n".join([header, sep, *[row] * 9])
    target = "\n".join([header, sep, *[row] * 8])

    decision = FastPassFilter().evaluate(source, target, block_type="table")
    assert decision.passed is False
    assert "Table grid mismatch" in decision.reason
    assert "11-row x 5-column" in decision.reason
    assert "10-row x 5-column" in decision.reason


def test_an_article_led_title_is_not_read_as_a_dropped_appendix_letter() -> None:
    for src, tgt in (
        ("A Survey of Transformer Models", "Transformer 模型综述"),
        ("A Survey of Large Language Models", "大语言模型综述"),
        ("I Introduction", "引言"),
        ("A New Approach to Retrieval", "一种新的检索方法"),
    ):
        decision = FastPassFilter().validate_structural_invariants(
            source_text=src, target_text=tgt, block_type="heading"
        )
        assert decision.passed, (src, decision.reason)
        assert "Heading label" not in decision.reason


def test_a_real_bare_appendix_label_is_still_gated() -> None:
    decision = FastPassFilter().validate_structural_invariants(
        source_text="C Encouraging Exploration Across Experts",
        target_text="鼓励跨专家探索",
        block_type="heading",
    )
    assert not decision.passed
    assert "Heading label dropped" in decision.reason
    assert "'C'" in decision.reason


def test_a_dropped_heading_label_is_classified_as_a_deterministic_defect() -> None:
    from ubt.core.qe.defect_taxonomy import classify_defect, has_structural_defect

    flag = (
        "Heading label dropped: the source heading is labelled 'B.' but the target "
        "does not carry it"
    )
    assert has_structural_defect([flag])
    assert classify_defect([flag], score=None, threshold=0.75) == "DETERMINISTIC_DEFECT"


def test_dialogue_is_prose_not_a_table() -> None:
    fp = FastPassFilter(source_lang="en", target_lang="zh")
    source = "Transformers rely on attention mechanisms to model long range dependencies."
    half = "Transformers rely on 注意力 mechanisms to model long range dependencies."
    for block_type in ("narrative", "dialogue"):
        decision = fp.evaluate(source, half, block_type=block_type)
        assert not decision.passed, block_type
        assert "script density" in decision.reason


def test_a_target_with_no_letters_is_not_a_translation() -> None:
    fp = FastPassFilter(source_lang="en", target_lang="zh")
    decision = fp.evaluate(
        "Attention is all you need in this paper about transformers", "!!!@@@###%%%"
    )
    assert not decision.passed
    assert "no letters" in decision.reason


def test_a_numeric_only_table_body_still_passes() -> None:
    fp = FastPassFilter(source_lang="en", target_lang="zh")
    decision = fp.evaluate(
        "| Model | Acc |\n|---|---|\n| T5 | 84.3 |",
        "| 模型 | 准确率 |\n|---|---|\n| T5 | 84.3 |",
        block_type="table",
    )
    assert decision.passed


@pytest.mark.parametrize("block_type", ["formula", "code"])
def test_a_verbatim_block_is_exempt_from_language_identity(block_type: str) -> None:
    fp = FastPassFilter(source_lang="en", target_lang="zh")
    decision = fp.evaluate("E = mc^2", "E = mc^2", block_type=block_type, skip_translate=True)
    assert decision.passed
    assert decision.target_ratio == 1.0
