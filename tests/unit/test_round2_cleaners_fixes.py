"""RED tests for Round 2 Batch 4: Cleaners, Validators, QE & Memory fixes."""

from __future__ import annotations

import pytest

from ubt.core.cleaners.html_sanitizer import scrub_source_document
from ubt.core.cleaners.math_masker import MathMasker
from ubt.core.cleaners.math_text import extract_text_spans, skeleton
from ubt.core.ir.models import BlockType, IRBlock
from ubt.core.memory.hierarchical_memory import HierarchicalMemoryManager, StepSnapshot
from ubt.core.qe.added_content import AddedContentGate
from ubt.core.qe.term_drift import detect_term_drift
from ubt.core.validators.consistency import GlossaryConsistencyValidator
from ubt.core.validators.glossary_enforcer import DeterministicGlossaryEnforcer


@pytest.mark.fast
def test_math_text_nested_spans() -> None:
    """extract_text_spans skips inner matches to avoid overlapping spans and corrupted skeletons."""
    latex = r"\text{outer \text{inner} text}"
    spans = extract_text_spans(latex)
    # Should only return the outermost span, not overlapping inner spans
    assert len(spans) == 1
    assert spans[0].inner == "outer \\text{inner} text"
    skel = skeleton(latex)
    assert "\x00SPAN0\x00" in skel
    assert skel == "\x00SPAN0\x00"


@pytest.mark.fast
def test_math_masker_currency_range_not_masked() -> None:
    """Currency chains ($10-$20, $5–$10, $10-20$) must not be treated as math formulas."""
    masker = MathMasker()
    # 1. $10-$20
    text1 = "The subscription fee is $10-$20 per month."
    masked1, mapping1 = masker.mask(text1)
    assert mapping1 == {}, f"Unexpected math masking: {mapping1}"
    assert masked1 == text1

    # 2. $5–$10 (en-dash)
    text2 = "Entry costs $5–$10 depending on age."
    masked2, mapping2 = masker.mask(text2)
    assert mapping2 == {}, f"Unexpected math masking: {mapping2}"
    assert masked2 == text2

    # 3. $10-20$ (price range in dollars)
    text3 = "Tickets are $10-20$ each."
    masked3, mapping3 = masker.mask(text3)
    assert mapping3 == {}, f"Unexpected math masking: {mapping3}"
    assert masked3 == text3


@pytest.mark.fast
def test_html_sanitizer_cdata_and_processing_instructions() -> None:
    """_SourceTagScrubber.scan must support CDATA ending at ]]> and PI ending at ?>."""
    # 1. CDATA containing internal '>' and tags that shouldn't be parsed as HTML elements
    cdata_doc = (
        '<root><![CDATA[ <div title="a">x > y</div> <script>test</script> ]]><p>Hello</p></root>'
    )
    scrubbed = scrub_source_document(cdata_doc)
    assert '<![CDATA[ <div title="a">x > y</div> <script>test</script> ]]>' in scrubbed
    assert "<p>Hello</p>" in scrubbed

    # 2. Processing instruction containing '>'
    pi_doc = '<root><?custom-pi note="a > b" ?><p>Body</p></root>'
    scrubbed_pi = scrub_source_document(pi_doc)
    assert '<?custom-pi note="a > b" ?>' in scrubbed_pi
    assert "<p>Body</p>" in scrubbed_pi


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
def test_added_content_fullwidth_parentheses() -> None:
    """Equation references with full-width parentheses （3.11） are recognized."""
    src = "As shown in Eq. (3.11), the rate is constant."
    tgt = "如式（3.11）所示，速率是恒定的。"

    gate = AddedContentGate()
    decision = gate.evaluate(src, tgt)
    assert decision.passed, f"Gate failed unexpectedly: {decision.reason}"
    assert "3.11" in decision.source_refs
    assert "3.11" in decision.target_refs

    # Conversely, a fabricated fullwidth reference should be caught
    tgt_bad = "如式（2.2）所示，速率是恒定的。"
    decision_bad = gate.evaluate(src, tgt_bad)
    assert not decision_bad.passed
    assert "2.2" in decision_bad.fabricated_refs


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


@pytest.mark.fast
def test_hierarchical_memory_macro_context_spine_boundary() -> None:
    """get_macro_context_for_block enforces snap.end_spine <= block.spine_index."""
    mgr = HierarchicalMemoryManager()

    # Pre-populate two snapshots
    snap1 = StepSnapshot(
        step_index=0,
        start_spine=0,
        end_spine=5,
        char_count=100,
        summary_text="Chapter 1 part 1 summary",
        chapter_id="ch01",
    )
    snap2 = StepSnapshot(
        step_index=1,
        start_spine=6,
        end_spine=10,
        char_count=100,
        summary_text="Chapter 1 part 2 summary",
        chapter_id="ch01",
    )
    mgr._snapshots.extend([snap1, snap2])

    # Block at spine_index 3 (in chapter 1) should NOT see snap2 (end_spine=10)
    # It shouldn't even see snap1 if snap1.end_spine > 3
    early_block = IRBlock(
        id="ch01#b002",
        spine_index=3,
        block_type=BlockType.NARRATIVE,
        source_text="Early paragraph",
    )
    ctx_early = mgr.get_macro_context_for_block(early_block)
    assert ctx_early == "", f"Early block leaked future snapshot: {ctx_early}"

    # Block at spine_index 7 should see snap1 (end_spine=5 <= 7), but NOT snap2 (end_spine=10 > 7)
    mid_block = IRBlock(
        id="ch01#b007",
        spine_index=7,
        block_type=BlockType.NARRATIVE,
        source_text="Mid paragraph",
    )
    ctx_mid = mgr.get_macro_context_for_block(mid_block)
    assert ctx_mid == "Chapter 1 part 1 summary"
