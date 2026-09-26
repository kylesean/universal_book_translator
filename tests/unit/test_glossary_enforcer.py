"""DeterministicGlossaryEnforcer: term replacement that cannot corrupt text.

Landed here from an earlier review-round module — twelve tests of one
class filed under the review round that produced them, with no canonical file
for the class at all. Enforced term substitution is the one place where a
subtle bug silently rewrites a book, so it belongs with the module.
"""

import pytest

from ubt.core.qe.term_drift import detect_term_drift
from ubt.core.validators.consistency import GlossaryConsistencyValidator
from ubt.core.validators.glossary_enforcer import DeterministicGlossaryEnforcer

pytestmark = pytest.mark.fast

# =============================================================================
# 1. Deterministic Glossary Enforcer Tests
# =============================================================================


def test_glossary_drift_detects_a_wrong_rendering() -> None:
    """The drift scan must separate canonical from wrong target renderings.

    Guards the terminology channel the offline baselines cannot exercise: their
    mock echoes the fixture's own strings, so a literal assertion there proves
    nothing about UBT's terminology handling.
    """
    from ubt.core.qe.term_drift import detect_term_drift

    glossary = [{"source": "working memory", "translation": "工作记忆", "aliases": []}]
    ok = detect_term_drift("Working memory is central.", "工作记忆是核心。", glossary)
    wrong = detect_term_drift("Working memory is central.", "工作内存是核心。", glossary)
    assert len(ok) == 1 and not ok[0].drifted
    assert len(wrong) == 1 and wrong[0].drifted


def test_alias_equal_to_another_entry_canonical_is_not_a_violation() -> None:
    """A target surface that is another entry's canonical rendering is approved.

    The rewriter's own global guard leaves it alone, so the span detector must
    not disagree and ask repair to "fix" an accepted term.
    """
    from ubt.core.qe.term_drift import detect_target_term_violations

    glossary = [
        {"source": "alpha", "translation": "甲", "aliases": ["Beta"]},
        {"source": "beta", "translation": "Beta", "aliases": []},
    ]
    assert detect_target_term_violations("Beta", glossary) == ()


def test_glossary_enforcer_alias_canonicalization() -> None:
    """Verify that non-preferred aliases and synonyms are replaced by canonical translations."""
    glossary = [
        {
            "source": "working memory",
            "translation": "工作记忆",
            "aliases": ["操作记忆", "短时工作记忆"],
            "inflected_variants": [],
            "kind": "term",
        },
        {
            "source": "distributed ledger",
            "translation": "分布式账本",
            "aliases": ["分散记账簿", "分布式记账册"],
            "inflected_variants": [],
            "kind": "term",
        },
    ]

    enforcer = DeterministicGlossaryEnforcer(glossary=glossary, target_lang="zh", source_lang="en")

    input_text = "在神经认知中，操作记忆是核心，而在区块链中则使用分散记账簿进行存证。"
    corrected, records = enforcer.enforce(input_text)

    assert corrected == "在神经认知中，工作记忆是核心，而在区块链中则使用分布式账本进行存证。"
    assert len(records) == 2
    # Longest-match-first processing order means records are not guaranteed to
    # be position-ordered; assert on the span->correction mapping instead.
    by_span = {r.original_span: r.corrected_span for r in records}
    assert by_span["操作记忆"] == "工作记忆"
    assert by_span["分散记账簿"] == "分布式账本"


def test_glossary_enforcer_source_leak_replacement() -> None:
    """Verify that untranslated source terms in target translation are replaced with approved target terms."""
    glossary = [
        {
            "source": "attention mechanism",
            "translation": "注意力机制",
            "aliases": [],
            "inflected_variants": [],
            "kind": "term",
        },
        {
            "source": "Transformer",
            "translation": "变换器架构",
            "aliases": [],
            "inflected_variants": [],
            "kind": "term",
        },
    ]

    enforcer = DeterministicGlossaryEnforcer(glossary=glossary, target_lang="zh", source_lang="en")

    # LLM left "attention mechanism" and "Transformer" untranslated
    input_text = "该模型的核心是attention mechanism，这是现代Transformer的基础。"
    corrected, records = enforcer.enforce(input_text)

    assert corrected == "该模型的核心是注意力机制，这是现代变换器架构的基础。"
    assert len(records) == 2


def test_glossary_enforcer_word_boundary_safety() -> None:
    """Verify Latin word boundary checks prevent sub-token false positive replacements."""
    glossary = [
        {
            "source": "chat",
            "translation": "猫",
            "aliases": ["cat"],
            "inflected_variants": [],
            "kind": "term",
        }
    ]

    enforcer = DeterministicGlossaryEnforcer(glossary=glossary, target_lang="en", source_lang="fr")

    # Should replace standalone "cat", but MUST NOT touch "concatenate", "category", or "scatter"
    input_text = "The cat ran to concatenate the data in each category."
    corrected, records = enforcer.enforce(input_text)

    assert corrected == "The 猫 ran to concatenate the data in each category."
    assert len(records) == 1
    assert records[0].start_pos == 4


def test_glossary_enforcer_longest_match_disambiguation() -> None:
    """Verify longest match priority when term patterns overlap."""
    glossary = [
        {
            "source": "memory",
            "translation": "记忆",
            "aliases": [],
            "inflected_variants": [],
            "kind": "term",
        },
        {
            "source": "working memory",
            "translation": "工作记忆",
            "aliases": ["短时工作记忆"],
            "inflected_variants": [],
            "kind": "term",
        },
    ]

    enforcer = DeterministicGlossaryEnforcer(glossary=glossary, target_lang="zh", source_lang="en")

    # "短时工作记忆" contains "工作记忆" and "记忆" as substrings; must replace the longest "短时工作记忆"
    input_text = "研究表明短时工作记忆容量有限。"
    corrected, records = enforcer.enforce(input_text)

    assert corrected == "研究表明工作记忆容量有限。"
    assert len(records) == 1
    assert records[0].original_span == "短时工作记忆"
    assert records[0].corrected_span == "工作记忆"


def test_glossary_enforcer_respects_inflected_variants() -> None:
    """Verify approved inflected variants in target language are not mistakenly overwritten."""
    glossary = [
        {
            "source": "distributed ledger",
            "translation": "verteiltes Hauptbuch",  # nominative
            "aliases": [],
            "inflected_variants": [
                "verteilten Hauptbüchern",
                "verteilten Hauptbuchs",
            ],  # dative/genitive
            "kind": "term",
        }
    ]

    enforcer = DeterministicGlossaryEnforcer(glossary=glossary, target_lang="de", source_lang="en")

    # Text already contains the grammatically correct dative variant
    input_text = "Wir arbeiten mit den verteilten Hauptbüchern des Systems."
    corrected, records = enforcer.enforce(input_text)

    # Must preserve the correct inflection and not force the nominative "verteiltes Hauptbuch"
    assert corrected == input_text
    assert len(records) == 0


def test_glossary_enforcer_idempotent_no_cyclic_resubstitution() -> None:
    """Regression test for P0 #3: a source term that is a prefix of its canonical target must
    not be re-substituted, otherwise re-running enforce on already-correct text explodes
    (e.g. "AI技术" -> "AI技术技术" -> "AI技术技术技术").

    """
    glossary = [
        {
            "source": "AI",
            "translation": "AI技术",
            "aliases": [],
            "inflected_variants": [],
            "kind": "term",
        }
    ]
    enforcer = DeterministicGlossaryEnforcer(glossary=glossary, target_lang="zh", source_lang="en")

    # First pass: untranslated source is corrected to the canonical target.
    once, rec1 = enforcer.enforce("This uses AI for inference.")
    assert once == "This uses AI技术 for inference."
    assert len(rec1) == 1

    # Second pass on the already-corrected text must be a fixed point (idempotent).
    twice, rec2 = enforcer.enforce(once)
    assert twice == once
    assert len(rec2) == 0

    # And a third pass must still be stable.
    thrice, rec3 = enforcer.enforce(twice)
    assert thrice == once
    assert len(rec3) == 0


def test_glossary_enforcer_alias_prefix_idempotent() -> None:
    """An alias that is a prefix of the canonical target must also be skipped to stay idempotent."""
    glossary = [
        {
            "source": "neural network",
            "translation": "神经网络",
            "aliases": ["网络"],
            "inflected_variants": [],
            "kind": "term",
        }
    ]
    enforcer = DeterministicGlossaryEnforcer(glossary=glossary, target_lang="zh", source_lang="en")

    once, _ = enforcer.enforce("这是一个网络模型。")
    twice, rec = enforcer.enforce(once)
    assert twice == once
    assert len(rec) == 0


def test_glossary_enforcer_alias_expansion_with_unrelated_flank() -> None:
    """A deprecated alias whose canonical replacement shares no text with the
    alias must still be corrected inside a CJK run (overlap-aware guard)."""
    glossary = [
        {
            "source": "attention",
            "translation": "注意力",
            "aliases": ["关注"],
            "inflected_variants": [],
            "kind": "term",
        }
    ]
    enforcer = DeterministicGlossaryEnforcer(glossary=glossary, target_lang="zh", source_lang="en")

    corrected, records = enforcer.enforce("请关注这个attention模块")

    assert corrected == "请注意力这个注意力模块"
    assert "关注" not in corrected
    assert len(records) == 2


def test_glossary_enforcer_protected_structures() -> None:
    """HTML tags, LaTeX formulas, and Markdown links must not be corrupted by glossary replacements."""
    glossary = [
        {
            "source": "table",
            "translation": "表格",
            "aliases": [],
            "inflected_variants": [],
            "kind": "term",
        },
        {
            "source": "class",
            "translation": "类别",
            "aliases": [],
            "inflected_variants": [],
            "kind": "term",
        },
    ]
    enforcer = DeterministicGlossaryEnforcer(glossary=glossary, target_lang="zh", source_lang="en")

    # HTML tag protection
    html_input = '<table class="sample">A table with class.</table>'
    html_out, _ = enforcer.enforce(html_input)
    assert '<table class="sample">' in html_out
    assert "</table>" in html_out
    assert "A 表格 with 类别." in html_out

    # LaTeX math protection
    math_input = "Formula $y = class + table$ and prose table."
    math_out, _ = enforcer.enforce(math_input)
    assert "$y = class + table$" in math_out
    assert "prose 表格." in math_out

    # Markdown link URL protection
    md_input = "[see table](https://example.com/table) here."
    md_out, _ = enforcer.enforce(md_input)
    assert "[see 表格](https://example.com/table) here." in md_out

    # Fenced code block protection
    fenced_code = "```python\ndef class_table():\n    return table\n```\nprose table."
    fenced_out, _ = enforcer.enforce(fenced_code)
    assert "```python\ndef class_table():\n    return table\n```" in fenced_out
    assert "prose 表格." in fenced_out

    # Inline code protection
    inline_code = "Use `class Table:` for table data."
    inline_out, _ = enforcer.enforce(inline_code)
    assert "`class Table:`" in inline_out
    assert "for 表格 data." in inline_out

    # Bare URL protection
    bare_url = "Visit https://api.example.com/class/table for table details."
    bare_out, _ = enforcer.enforce(bare_url)
    assert "https://api.example.com/class/table" in bare_out
    assert "for 表格 details." in bare_out


def test_cjk_char_definition_is_shared_with_the_memory_matcher() -> None:
    """One range table: the enforcer and cjk_matcher can no longer drift apart.

    The two copies this replaces had already diverged — cjk_matcher counted
    kana/hangul, the enforcer only ideographs. The wider (conservative) set
    now lives in the enforcer and is imported by the memory matcher.
    """
    from ubt.core.memory import cjk_matcher
    from ubt.core.validators.glossary_enforcer import CJK_RANGES, is_cjk_char

    assert cjk_matcher.is_cjk_char is is_cjk_char
    # Ideographs (both copies always agreed) plus kana/hangul (the addition):
    assert is_cjk_char("语") and is_cjk_char("カ") and is_cjk_char("한")
    assert not is_cjk_char("a") and not is_cjk_char("")
    assert (0x4E00, 0x9FFF) in CJK_RANGES
    assert (0x30A0, 0x30FF) in CJK_RANGES
    assert cjk_matcher.contains_cjk("カタカナ") is True


def test_find_term_occurrences_case_insensitive_keeps_original_offsets() -> None:
    """The judging side folds case WITHOUT rewriting either string."""
    from ubt.core.validators.glossary_enforcer import find_term_occurrences

    text = "A FinFET scales down. Again FinFET scales."
    # Case-sensitive default (what the rewriter uses): no lowercase hits.
    assert find_term_occurrences(text, "finfet") == []
    case_folded = find_term_occurrences(text, "finfet", case_insensitive=True)
    assert len(case_folded) == 2
    # Offsets index into the ORIGINAL text — offsets a splicer can trust.
    assert text[case_folded[0][0] : case_folded[0][1]] == "FinFET"
    assert text[case_folded[1][0] : case_folded[1][1]] == "FinFET"
    # Latin word boundaries still apply when folding case.
    assert find_term_occurrences("confinfet", "finfet", case_insensitive=True) == []


@pytest.mark.fast
def test_glossary_enforcer_multiple_occurrences_in_window() -> None:
    """Enforcer checks all occurrences in search window for outside overlap."""
    glossary = [{"source": "net", "translation": "neural net", "aliases": ["net"]}]
    enforcer = DeterministicGlossaryEnforcer(glossary=glossary)
    text = "neural net neural net"
    res, records = enforcer.enforce(text)
    assert res == "neural net neural net"
    assert records == []


@pytest.mark.fast
def test_term_drift_and_validator_inflected_variants() -> None:
    """detect_term_drift and GlossaryConsistencyValidator honor inflected_variants."""
    glossary = [
        {
            "source": "memory",
            "translation": "内存",
            "aliases": [],
            "inflected_variants": ["存储器", "记忆"],
        }
    ]
    # Source has 'memory', target uses inflected variant '存储器'
    src = "The system allocates memory dynamically."
    tgt = "系统动态分配存储器。"

    findings = detect_term_drift(src, tgt, glossary)
    assert len(findings) == 1
    assert not findings[0].drifted, "Inflected variant should be recognized as valid target form"

    validator = GlossaryConsistencyValidator(glossary)
    result = validator.validate(src, tgt)
    assert result.is_valid, f"Validator failed: {result.message}"
