"""Unit tests for deterministic terminology metrics (TP/TF)."""

from __future__ import annotations

import pytest

from ubt.core.ir.models import FlowID, IRBlock
from ubt.core.qe.term_metrics import evaluate_terms
from ubt.core.validators.glossary_enforcer import find_term_occurrences

pytestmark = pytest.mark.fast


def _block(idx: int, source: str, target: str) -> IRBlock:
    return IRBlock(
        id=f"b{idx}",
        flow_id=FlowID.MAIN_STORY,
        spine_index=idx,
        source_text=source,
        target_text=target,
    )


def test_exact_cjk_term_is_precise_and_recalled() -> None:
    glossary = [{"source": "Darcy", "translation": "达西"}]
    metrics = evaluate_terms([_block(1, "Mr. Darcy entered.", "达西先生到了。")], glossary)
    assert metrics.terms_expected == 1
    assert metrics.terms_rendered == 1
    assert metrics.term_precision == 1.0
    assert metrics.fuzzy_term_precision == 1.0
    assert metrics.term_recall == 1.0


def test_short_cjk_drift_is_not_fuzzy_forgiven() -> None:
    """A 2-char CJK rendering is exact-only: '德西' must not pass as '达西'."""
    glossary = [{"source": "Darcy", "translation": "达西"}]
    metrics = evaluate_terms([_block(1, "Mr. Darcy entered.", "德西先生到了。")], glossary)
    assert metrics.terms_expected == 1
    assert metrics.term_precision == 0.0
    assert metrics.fuzzy_term_precision == 0.0
    assert metrics.term_recall == 0.0


def test_fuzzy_counts_inflected_latin_rendering() -> None:
    glossary = [{"source": "transformer", "translation": "bidirectional encoder"}]
    block = _block(1, "A transformer model.", "uses bidirectional encoders here")
    metrics = evaluate_terms([block], glossary)
    # Exact fails on the word boundary ('encoders'), fuzzy (>=80%) accepts it.
    assert metrics.term_precision == 0.0
    assert metrics.fuzzy_term_precision == 1.0


def test_latin_boundary_prevents_subterm_false_positive() -> None:
    glossary = [{"source": "cat", "translation": "cat"}]
    block = _block(1, "the cat sat", "the category list")
    metrics = evaluate_terms([block], glossary)
    assert metrics.term_precision == 0.0  # 'cat' inside 'category' must not count
    assert metrics.fuzzy_term_precision == 1.0  # but 3-char fuzzy does match


def test_empty_glossary_is_all_zero() -> None:
    metrics = evaluate_terms([_block(1, "src", "tgt")], [])
    assert metrics.terms_expected == 0
    assert metrics.term_precision == 0.0
    assert metrics.fuzzy_term_precision == 0.0
    assert metrics.term_recall == 0.0


def test_term_inside_math_span_is_not_evaluated() -> None:
    glossary = [{"source": "E", "translation": "E"}]
    metrics = evaluate_terms([_block(1, "$E = mc^2$", "$E = mc^2$")], glossary)
    # The only 'E' occurrence sits inside protected math, so nothing is expected.
    assert metrics.terms_expected == 0


def test_find_term_occurrences_shields_math() -> None:
    assert find_term_occurrences("$E = mc^2$", "E") == []
    assert find_term_occurrences("energy E here", "E") == [(7, 8)]


def test_summarize_drift_ranks_worst_first() -> None:
    from ubt.core.qe.term_metrics import summarize_drift

    glossary = [
        {"source": "Darcy", "translation": "达西"},
        {"source": "Bingley", "translation": "彬格莱"},
    ]
    blocks = [
        _block(1, "Darcy and Bingley met.", "达西和彬格莱见面了。"),
        _block(2, "Darcy left.", "德西离开了。"),
        _block(3, "Bingley stayed.", "宾利留了下来。"),
    ]
    drifts = summarize_drift(evaluate_terms(blocks, glossary))
    by_source = {d.source: d for d in drifts}
    assert by_source["Darcy"].occurrences == 2
    assert by_source["Darcy"].exact_renderings == 1
    assert by_source["Darcy"].drift_rate == 0.5
    assert by_source["Darcy"].drifted_block_ids == ("b2",)
    assert by_source["Bingley"].drifted_block_ids == ("b3",)


def test_summarize_drift_empty_when_no_drift() -> None:
    from ubt.core.qe.term_metrics import summarize_drift

    glossary = [{"source": "Darcy", "translation": "达西"}]
    metrics = evaluate_terms([_block(1, "Darcy entered.", "达西进来了。")], glossary)
    assert summarize_drift(metrics) == ()


def test_alias_only_source_block_agrees_with_consistency_validator() -> None:
    """Pin the alias-drift fix: source carries the term ONLY as an alias.

    Before consolidation ``evaluate_terms`` ignored aliases on the source
    side, so a block whose source mentioned a glossary entry only through an
    alias produced ZERO hits here — invisible to the consistency stage's
    planner — while ``GlossaryConsistencyValidator`` (the same scan the
    quality gate wraps) already reported that exact block as GLOSSARY_DRIFT.
    Both detectors must now return the same verdict for the same block.
    """
    from ubt.core.qe.consistency_enforce import ConsistencyTask, plan_consistency_tasks
    from ubt.core.validators.consistency import GlossaryConsistencyValidator

    glossary = [{"source": "Mr. Bingley", "translation": "彬格莱先生", "aliases": ["Bingley"]}]
    validator = GlossaryConsistencyValidator(glossary)

    drifted = _block(1, "Bingley arrived first.", "宾利先到了。")  # no canonical rendering
    metrics = evaluate_terms([drifted], glossary)
    # Validator flags drift (it always saw the alias)...
    assert validator.validate(drifted.source_text, drifted.target_text).is_valid is False
    # ...and the metrics/planner now see the very same block: the term is
    # expected here (1 expected, 0 rendered) and plans exactly one repair.
    assert metrics.terms_expected == 1
    assert metrics.terms_rendered == 0
    assert metrics.per_hit[0].source == "Mr. Bingley"
    assert metrics.per_hit[0].exact is False
    assert plan_consistency_tasks(metrics) == (
        ConsistencyTask(block_id="b1", source="Mr. Bingley", expected="彬格莱先生"),
    )

    # Rendered canonically: both detectors agree it is clean.
    rendered = _block(1, "Bingley arrived first.", "彬格莱先生先到了。")
    metrics = evaluate_terms([rendered], glossary)
    assert validator.validate(rendered.source_text, rendered.target_text).is_valid is True
    assert metrics.terms_rendered == 1
    assert plan_consistency_tasks(metrics) == ()


def test_drift_detection_matches_between_gate_and_metrics_matrix() -> None:
    """Same block pair, same verdict from both consumers — source or alias.

    A tiny truth table over the three source cases (canonical surface, alias
    surface, term absent) × two target cases (rendered, not rendered).
    """
    from ubt.core.validators.consistency import GlossaryConsistencyValidator

    glossary = [{"source": "Working Memory", "translation": "工作记忆", "aliases": ["WM"]}]
    validator = GlossaryConsistencyValidator(glossary)
    cases = [
        # (source, target, validator-valid, metrics-expected, metrics-exact)
        ("WM is central.", "工作记忆是核心。", True, True, True),
        ("WM is central.", "短期记忆是核心。", False, True, False),
        ("Working Memory is central.", "工作记忆是核心。", True, True, True),
        ("Working Memory is central.", "短期记忆是核心。", False, True, False),
        ("Nothing relevant here.", "无关内容。", True, False, False),
    ]
    for source, target, expect_valid, expect_expected, expect_exact in cases:
        block = _block(1, source, target)
        metrics = evaluate_terms([block], glossary)
        valid = validator.validate(source, target).is_valid
        assert valid is expect_valid, (source, target)
        assert (metrics.terms_expected == 1) is expect_expected, (source, target)
        if expect_expected:
            assert metrics.per_hit[0].exact is expect_exact, (source, target)
            # For a term the source carries, the gate's verdict IS the
            # metrics' verdict: valid == exactly rendered, drift == not exact.
            assert valid is expect_exact, (source, target)
