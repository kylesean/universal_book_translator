from __future__ import annotations

import pytest

from ubt.core.ir.emphasis import (
    BOLD_CLOSE,
    BOLD_OPEN,
    ITALIC_CLOSE,
    ITALIC_OPEN,
    mark_bold_spans,
    mark_emphasis_spans,
    parse_bold,
    parse_emphasis,
    strip_emphasis_markers,
)

pytestmark = pytest.mark.fast


def test_no_markers_leaves_text_unchanged() -> None:
    clean, runs = parse_bold("图 1 | SoL-Pi 降低了 50.0%。")
    assert clean == "图 1 | SoL-Pi 降低了 50.0%。"
    assert runs == ()


def test_a_pair_becomes_one_bold_run() -> None:
    clean, runs = parse_bold(f"成本降低了 {BOLD_OPEN}50.0%{BOLD_CLOSE}；随后")
    assert clean == "成本降低了 50.0%；随后"
    assert [(r.text, r.bold) for r in runs] == [("50.0%", True)]


def test_two_pairs_become_two_runs() -> None:
    clean, runs = parse_bold(f"{BOLD_OPEN}甲{BOLD_CLOSE}和{BOLD_OPEN}乙{BOLD_CLOSE}")
    assert clean == "甲和乙"
    assert [r.text for r in runs] == ["甲", "乙"]


def test_nested_markers_are_one_run() -> None:
    clean, runs = parse_bold(f"{BOLD_OPEN}外{BOLD_OPEN}内{BOLD_CLOSE}外{BOLD_CLOSE}")
    assert clean == "外内外"
    assert [r.text for r in runs] == ["外内外"]


def test_unterminated_open_emphasizes_to_the_end() -> None:
    clean, runs = parse_bold(f"前{BOLD_OPEN}重点")
    assert clean == "前重点"
    assert [r.text for r in runs] == ["重点"]


def test_a_stray_close_is_dropped() -> None:
    clean, runs = parse_bold(f"前{BOLD_CLOSE}后")
    assert clean == "前后"
    assert runs == ()


def test_markers_with_inner_spacing_or_case_are_recognized() -> None:
    clean, runs = parse_bold("⟦ b ⟧重点⟦ / B ⟧")
    assert clean == "重点"
    assert [r.text for r in runs] == ["重点"]


def test_a_run_never_carries_edge_whitespace() -> None:
    clean, runs = parse_bold(f"{BOLD_OPEN} 重点 {BOLD_CLOSE}")
    assert clean == " 重点 "
    assert [r.text for r in runs] == ["重点"]


def test_mark_then_parse_round_trips() -> None:
    source = "SoL-Pi discovers a more token-efficient harness."
    marked = mark_bold_spans(source, ["SoL-Pi discovers"])
    assert marked == f"{BOLD_OPEN}SoL-Pi discovers{BOLD_CLOSE} a more token-efficient harness."
    clean, runs = parse_bold(marked)
    assert clean == source
    assert [r.text for r in runs] == ["SoL-Pi discovers"]


def test_mark_skips_missing_and_empty_spans() -> None:
    assert mark_bold_spans("abc", ["nope", "", "  "]) == "abc"


def test_mark_folds_a_dash_mismatch_between_extractor_and_reader() -> None:
    marked = mark_bold_spans("agent-environment loop", ["agent\u2013environment"])
    assert marked == f"{BOLD_OPEN}agent-environment{BOLD_CLOSE} loop"
    clean, runs = parse_bold(marked)
    assert clean == "agent-environment loop"
    assert [r.text for r in runs] == ["agent-environment"]


def test_mark_uses_each_occurrence_in_order() -> None:
    marked = mark_bold_spans("a X b X", ["X", "X"])
    assert marked == f"a {BOLD_OPEN}X{BOLD_CLOSE} b {BOLD_OPEN}X{BOLD_CLOSE}"


def test_strip_emphasis_markers_removes_every_marker() -> None:
    assert strip_emphasis_markers(f"a{BOLD_OPEN}b{BOLD_CLOSE}c") == "abc"


def test_parse_never_leaves_a_marker_in_the_clean_text() -> None:
    for raw in (
        f"{BOLD_OPEN}{BOLD_CLOSE}",
        f"{BOLD_OPEN}未闭合",
        f"孤立{BOLD_CLOSE}收尾",
        f"{BOLD_OPEN}a{BOLD_OPEN}b{BOLD_CLOSE}",
        "⟦ B ⟧x⟦ / B ⟧",
    ):
        clean, _ = parse_bold(raw)
        assert "⟦" not in clean
        assert "⟧" not in clean


@pytest.mark.parametrize(
    "marker_open",
    ["⟦B⟧", "⟦ B ⟧", "\\llbracket B \\rrbracket", "\\llbracket b \\rrbracket"],
)
@pytest.mark.parametrize(
    "marker_close",
    ["⟦/B⟧", "⟦ /B ⟧", "\\llbracket /B \\rrbracket", "\\llbracket / B \\rrbracket"],
)
def test_a_latex_transcribed_marker_is_still_recognised(
    marker_open: str, marker_close: str
) -> None:
    clean, runs = parse_bold(f"{marker_open}粗体{marker_close}")

    assert clean == "粗体", f"the marker must be stripped: {clean!r}"
    assert [r.text for r in runs] == ["粗体"]
    assert [r.bold for r in runs] == [True]


def test_a_marker_inside_a_formula_leaves_no_math_residue() -> None:
    text = r"$(\Delta, \llbracket B \rrbracket A \llbracket /B \rrbracket)$"
    clean, runs = parse_bold(text)

    assert "llbracket" not in clean and "rrbracket" not in clean, clean
    assert "⟦" not in clean, clean
    assert [r.text for r in runs] == ["A"]


def test_strip_emphasis_markers_handles_the_latex_form() -> None:
    assert strip_emphasis_markers(r"a \llbracket B \rrbracket b") == "a  b"


def test_ordinary_brackets_are_not_markers() -> None:
    for text in (r"$[a, b]$", r"see [21] and [32]", r"$\{x\}$"):
        clean, runs = parse_bold(text)
        assert clean == text, f"false positive on {text!r}"
        assert runs == ()


def test_word_boundary_prevents_substring_matching_in_longer_words() -> None:
    text = "Looking in the window."
    marked = mark_bold_spans(text, ["in"])
    assert marked == f"Looking {BOLD_OPEN}in{BOLD_CLOSE} the window."

    text_no_standalone = "Looking out the window."
    marked_none = mark_bold_spans(text_no_standalone, ["in"])
    assert marked_none == text_no_standalone


def test_mark_and_parse_emphasis_italic() -> None:
    text = "This is important and novel."
    marked = mark_emphasis_spans(text, bold_spans=["important"], italic_spans=["novel"])
    assert (
        marked == f"This is {BOLD_OPEN}important{BOLD_CLOSE} and {ITALIC_OPEN}novel{ITALIC_CLOSE}."
    )

    clean, runs = parse_emphasis(marked)
    assert clean == text
    assert len(runs) == 2
    assert runs[0].text == "important" and runs[0].bold is True and runs[0].italic is False
    assert runs[1].text == "novel" and runs[1].bold is False and runs[1].italic is True
    assert strip_emphasis_markers(marked) == text


def test_unclosed_markers_are_parsed_and_stripped() -> None:
    clean, runs = parse_emphasis("CPU ⟦/B 资源涉及控制")
    assert clean == "CPU 资源涉及控制"
    assert runs == ()
    assert "⟦" not in clean

    clean, runs = parse_emphasis("敏感的（sensitive）⟦/B，因为")
    assert clean == "敏感的（sensitive），因为"
    assert runs == ()

    clean, runs = parse_emphasis("这是 ⟦B 关键技术 ⟦/B 的实现")
    assert clean == "这是 关键技术 的实现"
    assert [(r.text, r.bold) for r in runs] == [("关键技术", True)]

    assert strip_emphasis_markers("CPU ⟦/B 资源") == "CPU 资源"
    assert strip_emphasis_markers(r"CPU \llbracket /B 资源") == "CPU 资源"


def test_symmetrical_and_malformed_closing_brackets_are_supported() -> None:
    # LLM closes bold with bare ⟧
    clean, runs = parse_emphasis("⟦B⟧循环 Transformer 与循环计算。⟧ 复用 Transformer 层最初被提出")
    assert clean == "循环 Transformer 与循环计算。 复用 Transformer 层最初被提出"
    assert [(r.text, r.bold) for r in runs] == [("循环 Transformer 与循环计算。", True)]
    assert "⟦" not in clean and "⟧" not in clean

    # LLM closes bold with /B⟧ without leading ⟦
    clean, runs = parse_emphasis("这是 ⟦B 重要参数/B⟧ 的说明")
    assert clean == "这是 重要参数 的说明"
    assert [(r.text, r.bold) for r in runs] == [("重要参数", True)]
    assert "⟦" not in clean and "⟧" not in clean

    # Stray bare ⟧ in text is safely stripped without poisoning clean text
    clean, runs = parse_emphasis("弱到强对比解码。⟧ 对比解码（CD）将专家模型的预测")
    assert clean == "弱到强对比解码。 对比解码（CD）将专家模型的预测"
    assert runs == ()
    assert "⟦" not in clean and "⟧" not in clean

    # Formula with bare ⟧ in text like b0361
    clean, runs = parse_emphasis(
        "说明了随着固定强度 ⟦B⟧$\\omega$ 从 0.25 增至 1.0⟧，在将作用范围扩展"
    )
    assert clean == "说明了随着固定强度 $\\omega$ 从 0.25 增至 1.0，在将作用范围扩展"
    assert [(r.text, r.bold) for r in runs] == [("$\\omega$ 从 0.25 增至 1.0", True)]
    assert "⟦" not in clean and "⟧" not in clean
