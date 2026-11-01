from __future__ import annotations

from typing import Any

import pytest

from ubt.core.validators.glossary_enforcer import (
    CJK_RANGES,
    DeterministicGlossaryEnforcer,
    EnforcementRecord,
    _is_latin_word_char,
    extract_protected_spans,
    find_term_occurrences,
    is_cjk_char,
)

pytestmark = pytest.mark.fast


@pytest.mark.parametrize("ch", ["\u4e2d", "\u6f22", "\u3042", "\u30a2", "\ud55c"])
def test_cjk_scripts_are_cjk(ch: str) -> None:
    assert is_cjk_char(ch) is True


@pytest.mark.parametrize("ch", ["", "a", "1", "\uff0c", "\u3002", " "])
def test_non_cjk_characters_are_not_cjk(ch: str) -> None:
    assert is_cjk_char(ch) is False


def test_every_declared_range_is_covered_at_both_ends() -> None:
    for low, high in CJK_RANGES:
        assert low <= high
        assert is_cjk_char(chr(low)) is True
        assert is_cjk_char(chr(high)) is True


@pytest.mark.parametrize("ch", ["a", "Z", "1", "_", "-", "\u00e9", "\u042f"])
def test_word_characters(ch: str) -> None:
    assert _is_latin_word_char(ch) is True


@pytest.mark.parametrize("ch", ["\u4e2d", " ", ".", "$", "\u2014"])
def test_non_word_characters(ch: str) -> None:
    assert _is_latin_word_char(ch) is False


@pytest.mark.parametrize(
    ("text", "span"),
    [
        ("a <b>x</b> c", (2, 5)),
        ("use `code` here", (4, 10)),
        ("see https://example.com/x end", (4, 25)),
        ("math $$a+b$$ tail", (5, 12)),
        ("[link](http://a.b) rest", (7, 17)),
        ("token \u27e6abc-def\u27e7 tail", (6, 15)),
    ],
)
def test_protected_regions_are_found(text: str, span: tuple[int, int]) -> None:
    assert span in extract_protected_spans(text)


def test_currency_dollars_are_not_treated_as_math() -> None:
    assert extract_protected_spans("price $5 and $10 total") == []


def test_occurrences_respect_word_boundaries() -> None:
    assert find_term_occurrences("cat category cat", "cat") == [(0, 3), (13, 16)]


def test_occurrences_are_case_sensitive_by_default() -> None:
    assert find_term_occurrences("Cat cat", "cat") == [(4, 7)]


def test_occurrences_can_fold_case_without_moving_offsets() -> None:
    assert find_term_occurrences("Cat cat", "cat", case_insensitive=True) == [(0, 3), (4, 7)]


def test_occurrences_exclude_protected_spans() -> None:
    assert find_term_occurrences("cat `cat` cat", "cat") == [(0, 3), (10, 13)]


def test_cjk_occurrences_use_substring_matching() -> None:
    assert find_term_occurrences("\u8fd9\u662f\u6a21\u578b\u6d4b\u8bd5", "\u6a21\u578b") == [(2, 4)]


@pytest.mark.parametrize(("text", "term"), [("", "x"), ("x", "")])
def test_occurrences_of_a_blank_side_are_empty(text: str, term: str) -> None:
    assert find_term_occurrences(text, term) == []


def _enforce(
    glossary: list[dict[str, Any]], text: str, **kwargs: Any
) -> tuple[str, list[EnforcementRecord]]:
    return DeterministicGlossaryEnforcer(glossary, **kwargs).enforce(text)


def test_an_alias_is_canonicalized_with_a_full_record() -> None:
    glossary = [{"source": "attention", "translation": "\u6ce8\u610f\u529b", "aliases": ["attn"]}]
    out, records = _enforce(glossary, "the attn mechanism")
    assert out == "the \u6ce8\u610f\u529b mechanism"
    assert records == [
        EnforcementRecord("attn", "\u6ce8\u610f\u529b", 4, 8, "alias:attention->\u6ce8\u610f\u529b")
    ]


def test_a_long_lowercase_source_leak_is_replaced() -> None:
    out, _ = _enforce(
        [{"source": "attention", "translation": "\u6ce8\u610f\u529b"}], "use attention"
    )
    assert out == "use \u6ce8\u610f\u529b"


def test_a_short_lowercase_source_is_not_treated_as_a_leak() -> None:
    out, records = _enforce([{"source": "ab", "translation": "\u7532\u4e59"}], "ab")
    assert (out, records) == ("ab", [])


def test_a_short_uppercase_source_leaks() -> None:
    out, _ = _enforce([{"source": "AB", "translation": "\u7532\u4e59"}], "AB")
    assert out == "\u7532\u4e59"


def test_source_leak_replacement_can_be_disabled_without_disabling_aliases() -> None:
    glossary = [{"source": "attention", "translation": "\u6ce8\u610f\u529b", "aliases": ["attn"]}]
    leaked, _ = _enforce(glossary, "attention", enforce_source_leak_replacement=False)
    aliased, _ = _enforce(glossary, "attn", enforce_source_leak_replacement=False)
    assert leaked == "attention"
    assert aliased == "\u6ce8\u610f\u529b"


def test_an_alias_colliding_with_another_entrys_target_is_left_alone() -> None:
    glossary: list[dict[str, Any]] = [
        {"source": "A", "translation": "\u7532", "aliases": ["Beta"]},
        {"source": "B", "translation": "Beta"},
    ]
    out, records = _enforce(glossary, "Beta \u7532")
    assert (out, records) == ("Beta \u7532", [])


def test_an_approved_inflected_variant_is_left_alone() -> None:
    glossary = [
        {
            "source": "run",
            "translation": "\u8fd0\u884c",
            "inflected_variants": ["\u8fd0\u884c\u4e2d"],
        },
        {"source": "x", "translation": "y", "aliases": ["\u8fd0\u884c\u4e2d"]},
    ]
    out, records = _enforce(glossary, "\u8fd0\u884c\u4e2d")
    assert (out, records) == ("\u8fd0\u884c\u4e2d", [])


def test_a_longer_cjk_replacement_inside_a_cjk_run_is_skipped() -> None:
    glossary = [{"source": "src", "translation": "\u6ce8\u610f\u529b", "aliases": ["\u5173\u6ce8"]}]
    out, records = _enforce(glossary, "\u5173\u6ce8\u5ea6\u5f88\u9ad8")
    assert (out, records) == ("\u5173\u6ce8\u5ea6\u5f88\u9ad8", [])


def test_a_longer_cjk_replacement_standalone_is_applied() -> None:
    glossary = [{"source": "src", "translation": "\u6ce8\u610f\u529b", "aliases": ["\u5173\u6ce8"]}]
    out, records = _enforce(glossary, "\u5173\u6ce8")
    assert out == "\u6ce8\u610f\u529b"
    assert len(records) == 1


def test_an_equal_length_cjk_swap_is_applied_even_when_flanked() -> None:
    glossary = [{"source": "src", "translation": "\u4e59\u4e19", "aliases": ["\u7532\u4e59"]}]
    out, records = _enforce(glossary, "\u7532\u4e59\u4e19")
    assert out == "\u4e59\u4e19\u4e19"
    assert len(records) == 1


@pytest.mark.parametrize(
    "text",
    ["https://x.com/Attention", "`Attention`", "\u27e6abc-0f\u27e7"],
)
def test_protected_regions_are_never_rewritten(text: str) -> None:
    out, records = _enforce([{"source": "Attention", "translation": "\u6ce8\u610f\u529b"}], text)
    assert (out, records) == (text, [])


def test_already_canonical_text_is_idempotent() -> None:
    out, records = _enforce(
        [{"source": "Attention", "translation": "\u6ce8\u610f\u529b"}], "\u6ce8\u610f\u529b"
    )
    assert (out, records) == ("\u6ce8\u610f\u529b", [])


def test_a_pattern_inside_its_own_replacement_is_not_re_substituted() -> None:
    glossary = [{"source": "src", "translation": "foo bar baz", "aliases": ["foo bar"]}]
    out, records = _enforce(glossary, "foo bar baz")
    assert (out, records) == ("foo bar baz", [])
    assert _enforce(glossary, "foo bar")[0] == "foo bar baz"


def test_the_longest_pattern_wins_over_a_shorter_prefix() -> None:
    glossary: list[dict[str, Any]] = [
        {"source": "ab.", "translation": "X"},
        {"source": "ab.cd", "translation": "Y"},
    ]
    out, records = _enforce(glossary, "ab.cd")
    assert out == "Y"
    assert records[0].original_span == "ab.cd"


def test_no_rules_is_unchanged() -> None:
    out, records = _enforce([], "Attention")
    assert (out, records) == ("Attention", [])


def test_empty_text_is_unchanged() -> None:
    out, records = _enforce([{"source": "A", "translation": "\u7532"}], "")
    assert (out, records) == ("", [])


def test_cjk_latin_spacing_is_tolerated_when_judging_a_term() -> None:
    assert find_term_occurrences(
        "基于 QoS 的 CPU 调度可在", "CPU调度", case_insensitive=True, allow_cjk_latin_space=True
    )
    assert find_term_occurrences(
        "基于 QoS 的 CPU调度可在", "CPU调度", case_insensitive=True, allow_cjk_latin_space=True
    )
    assert find_term_occurrences("CPU 调度", "CPU调度") == []
    assert find_term_occurrences("CP U调度", "CPU调度", allow_cjk_latin_space=True) == []


def test_single_character_cjk_embedded_in_compound_is_quarantined_not_corrupted() -> None:
    glossary = [{"source": "cloud", "translation": "端", "aliases": ["云"]}]
    enforcer = DeterministicGlossaryEnforcer(glossary)
    out, applied, quarantined = enforcer.enforce_audited("我们正在使用云计算平台")
    assert out == "我们正在使用云计算平台"
    assert applied == []
    assert len(quarantined) == 1
    assert quarantined[0].original_span == "云"


def test_cjk_term_inside_larger_compound_token_is_quarantined() -> None:
    glossary = [{"source": "status", "translation": "态势", "aliases": ["状态"]}]
    enforcer = DeterministicGlossaryEnforcer(glossary)
    out, applied, quarantined = enforcer.enforce_audited("这是系统的初始状态")
    assert out == "这是系统的初始状态"
    assert applied == []
    assert len(quarantined) >= 1
    assert quarantined[0].original_span == "状态"


def test_standalone_cjk_term_is_safely_replaced() -> None:
    glossary = [{"source": "status", "translation": "态势", "aliases": ["状态"]}]
    enforcer = DeterministicGlossaryEnforcer(glossary)
    out, applied, quarantined = enforcer.enforce_audited("状态：正常。")
    assert out == "态势：正常。"
    assert len(applied) == 1
    assert quarantined == []


@pytest.mark.parametrize(
    "text",
    [
        'the "Data Loader" step',
        "the “Data Loader” step",
        "the ‘Data Loader’ step",
        "the 「Data Loader」 step",
        "the 『Data Loader』 step",
        "the «Data Loader» step",
    ],
)
def test_a_quoted_span_is_protected(text: str) -> None:
    assert extract_protected_spans(text) != []


def test_an_inch_mark_does_not_open_a_span() -> None:
    spans = extract_protected_spans('a 10" ruler and the 5" one')
    assert spans == []


def test_an_apostrophe_does_not_open_a_span() -> None:
    assert extract_protected_spans("it's the user's Data Loader here") == []


def test_an_inch_mark_does_not_shadow_the_next_real_quote() -> None:
    text = 'a 10" ruler, the "Data Loader" step'
    assert extract_protected_spans(text) == [(17, 30)]
    assert text[18:29] == "Data Loader"


def test_an_unbalanced_quote_protects_nothing() -> None:
    assert extract_protected_spans('he said "Data Loader yesterday') == []


def test_a_quoted_source_term_is_a_citation_not_a_leak() -> None:
    glossary = [{"source": "Data Loader", "translation": "数据加载器"}]
    out, records = _enforce(glossary, 'the "Data Loader" step uses Data Loader')
    assert out == 'the "Data Loader" step uses 数据加载器'
    assert [record.original_span for record in records] == ["Data Loader"]


def test_a_quoted_alias_is_left_alone() -> None:
    glossary = [{"source": "attention", "translation": "注意力", "aliases": ["attn"]}]
    out, records = _enforce(glossary, 'the "attn" block and the attn block')
    assert out == 'the "attn" block and the 注意力 block'
    assert len(records) == 1


def test_an_alias_inside_an_approved_cjk_compound_is_not_rewritten() -> None:
    glossary: list[dict[str, Any]] = [
        {"source": "deep learning", "translation": "深度学习", "aliases": ["学习"]},
        {"source": "learning", "translation": "训练"},
    ]
    out, records = _enforce(glossary, "本文探讨深度学习的最新进展。")
    assert out == "本文探讨深度学习的最新进展。"
    assert records == []


def test_an_alias_inside_an_approved_latin_compound_is_not_rewritten() -> None:
    glossary: list[dict[str, Any]] = [
        {"source": "ML", "translation": "机器学习", "aliases": ["learning"]},
        {"source": "machine learning", "translation": "machine learning"},
    ]
    out, records = _enforce(glossary, "We study machine learning and learning theory.")
    assert out == "We study machine learning and 机器学习 theory."
    assert [record.original_span for record in records] == ["learning"]


def test_a_drifted_alias_outside_any_compound_is_still_enforced() -> None:
    glossary = [{"source": "deep learning", "translation": "深度学习", "aliases": ["深度学习法"]}]
    out, records = _enforce(glossary, "我们使用深度学习法来建模。")
    assert out == "我们使用深度学习来建模。"
    assert [record.original_span for record in records] == ["深度学习法"]


def test_a_spaced_canonical_term_matches_an_unspaced_target() -> None:
    hits = find_term_occurrences(
        "切换Transformer 是一个稀疏模型", "切换 Transformer", allow_cjk_latin_space=True
    )
    assert hits


def test_an_unspaced_canonical_term_matches_a_spaced_target() -> None:
    hits = find_term_occurrences(
        "切换 Transformer 是一个稀疏模型", "切换Transformer", allow_cjk_latin_space=True
    )
    assert hits


def test_a_latin_run_interior_space_is_never_optional() -> None:
    assert find_term_occurrences("New York city", "New York")
    assert not find_term_occurrences("NewYork city", "New York")
    assert find_term_occurrences("NewYork city", "NewYork")


def test_uppercase_latin_acronym_alias_is_not_forced_to_cjk() -> None:
    glossary = [
        {
            "source": "large language model",
            "translation": "大语言模型",
            "aliases": ["LLM", "llm_model"],
        }
    ]
    out, records = _enforce(glossary, "这个 LLM 的表现很棒")
    assert out == "这个 LLM 的表现很棒"
    assert not any(r.original_span == "LLM" for r in records)

    out2, records2 = _enforce(glossary, "本项目使用大语言模型（LLM）开发")
    assert out2 == "本项目使用大语言模型（LLM）开发"
    assert not any(r.original_span == "LLM" for r in records2)
