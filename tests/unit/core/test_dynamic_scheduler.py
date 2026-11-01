from __future__ import annotations

import pytest

from ubt.core.ir.models import BlockType, IRBlock, make_element
from ubt.core.router.dynamic_scheduler import (
    BlockComplexityClassifier,
    ComplexityTier,
    DynamicParetoScheduler,
    DynamicRoutingPolicy,
)

pytestmark = pytest.mark.fast


def _make_block(
    b_id: str,
    source: str,
    b_type: BlockType = BlockType.NARRATIVE,
) -> IRBlock:
    el = make_element(
        id=b_id,
        spine_index=1,
        block_type=b_type,
        source_text=source,
    )
    return IRBlock(element=el)


def test_classify_heading_as_trivial() -> None:
    classifier = BlockComplexityClassifier()
    block = _make_block("blk-h1", "# Chapter 1: Introduction", BlockType.HEADING)
    metrics = classifier.classify_block(block)

    assert metrics.tier == ComplexityTier.TRIVIAL
    assert metrics.score < 0.25
    assert "Heading structure" in metrics.reasons


def test_classify_standard_narrative() -> None:
    classifier = BlockComplexityClassifier()
    text = (
        "This chapter introduces fundamental architectural concepts. "
        "We discuss how systems process data streams and manage memory "
        "across distributed compute environments."
    )
    block = _make_block("blk-p1", text, BlockType.NARRATIVE)
    metrics = classifier.classify_block(block)

    assert metrics.tier == ComplexityTier.STANDARD
    assert 0.20 <= metrics.score < 0.60


def test_classify_math_complexity_and_extreme() -> None:
    classifier = BlockComplexityClassifier()

    text_inline = (
        "The loss function is defined as $L(\\theta) = \\frac{1}{N} \\sum_{i=1}^N (y_i - \\hat{y}_i)^2$ "
        "where $\\theta$ denotes the parameter vector."
    )
    block_inline = _make_block("blk-m1", text_inline, BlockType.NARRATIVE)
    metrics_inline = classifier.classify_block(block_inline)
    assert metrics_inline.tier in (ComplexityTier.COMPLEX, ComplexityTier.EXTREME)
    assert metrics_inline.math_inline_count >= 1

    text_display = "$$\\int_{-\\infty}^{+\\infty} e^{-x^2} dx = \\sqrt{\\pi}$$"
    block_display = _make_block("blk-m2", text_display, BlockType.FORMULA)
    metrics_display = classifier.classify_block(block_display)
    assert metrics_display.tier == ComplexityTier.EXTREME
    assert metrics_display.score >= 0.82


def test_classify_code_block() -> None:
    classifier = BlockComplexityClassifier()
    code_text = (
        "```python\n"
        "async def process_batch(items: list[str]) -> dict[str, int]:\n"
        "    results = {}\n"
        "    for item in items:\n"
        "        if len(item) > 10:\n"
        "            results[item] = len(item)\n"
        "    return results\n"
        "```"
    )
    block_code = _make_block("blk-c1", code_text, BlockType.CODE)
    metrics_code = classifier.classify_block(block_code)
    assert metrics_code.tier in (ComplexityTier.COMPLEX, ComplexityTier.EXTREME)
    assert metrics_code.code_syntax_score > 0.0


def test_classify_with_glossary_density() -> None:
    classifier = BlockComplexityClassifier()
    text = "The kernel driver configures the memory buffer via the bus controller."
    glossary = ["kernel", "driver", "memory buffer", "bus controller"]

    block = _make_block("blk-t1", text, BlockType.NARRATIVE)
    metrics_no_glossary = classifier.classify_block(block)
    metrics_with_glossary = classifier.classify_block(block, glossary_terms=glossary)

    assert metrics_with_glossary.score > metrics_no_glossary.score
    assert metrics_with_glossary.term_matches_count >= 3


def test_pareto_scheduler_model_assignment() -> None:
    scheduler = DynamicParetoScheduler(
        policy=DynamicRoutingPolicy.PARETO,
        budget_model="deepseek-chat",
        standard_model="deepseek-chat",
        flagship_model="claude-3-7-sonnet",
    )

    block_h = _make_block("h1", "## References", BlockType.HEADING)
    assign_h = scheduler.resolve_model_for_block(block_h)
    assert assign_h.assigned_model == "deepseek-chat"
    assert assign_h.tier == ComplexityTier.TRIVIAL

    block_m = _make_block(
        "m1",
        "$$\\nabla \\times \\mathbf{B} = \\mu_0 \\mathbf{J} + \\mu_0 \\epsilon_0 \\frac{\\partial \\mathbf{E}}{\\partial t}$$",
        BlockType.FORMULA,
    )
    assign_m = scheduler.resolve_model_for_block(block_m)
    assert assign_m.assigned_model == "claude-3-7-sonnet"
    assert assign_m.tier == ComplexityTier.EXTREME


def test_pareto_scheduler_retry_escalation() -> None:
    scheduler = DynamicParetoScheduler(
        policy=DynamicRoutingPolicy.PARETO,
        budget_model="deepseek-chat",
        standard_model="gpt-4o-mini",
        flagship_model="claude-3-7-sonnet",
    )

    block = _make_block(
        "p1",
        "This is an ordinary paragraph describing network protocols.",
        BlockType.NARRATIVE,
    )
    initial_assign = scheduler.resolve_model_for_block(block, is_retry_or_repair=False)
    assert initial_assign.assigned_model == "deepseek-chat"
    assert not initial_assign.is_escalated

    retry_assign = scheduler.resolve_model_for_block(block, is_retry_or_repair=True)
    assert retry_assign.is_escalated
    assert retry_assign.tier == ComplexityTier.COMPLEX
    assert retry_assign.assigned_model == "gpt-4o-mini"


def test_quality_first_and_cost_first_policies() -> None:
    scheduler_quality = DynamicParetoScheduler(
        policy=DynamicRoutingPolicy.QUALITY_FIRST,
        budget_model="deepseek-chat",
        standard_model="gpt-4o-mini",
        flagship_model="claude-3-7-sonnet",
    )
    block_code = _make_block(
        "c1",
        "```rust\nfn compute(x: f64) -> f64 { x * 2.0 }\n```",
        BlockType.CODE,
    )
    assign_q = scheduler_quality.resolve_model_for_block(block_code)
    assert assign_q.assigned_model == "claude-3-7-sonnet"

    scheduler_cost = DynamicParetoScheduler(
        policy=DynamicRoutingPolicy.COST_FIRST,
        budget_model="deepseek-chat",
        standard_model="gpt-4o-mini",
        flagship_model="claude-3-7-sonnet",
    )
    assign_c = scheduler_cost.resolve_model_for_block(block_code)
    assert assign_c.assigned_model == "deepseek-chat"


def test_simulate_document_routing() -> None:
    scheduler = DynamicParetoScheduler(
        policy=DynamicRoutingPolicy.PARETO,
        budget_model="deepseek-chat",
        standard_model="deepseek-chat",
        flagship_model="claude-3-7-sonnet",
    )

    blocks = [
        _make_block("b1", "# Title", BlockType.HEADING),
        _make_block("b2", "Introduction to algebra.", BlockType.NARRATIVE),
        _make_block("b3", "$$\\sum_{i=1}^n i = \\frac{n(n+1)}{2}$$", BlockType.FORMULA),
    ]

    report = scheduler.simulate_document_routing(blocks)
    assert report.total_blocks == 3
    assert report.total_estimated_cost_usd > 0.0
    assert report.total_monolithic_flagship_cost_usd >= report.total_estimated_cost_usd
    assert report.savings_percentage >= 0.0
    assert "deepseek-chat" in report.model_counts

    md = scheduler.render_simulation_markdown(report)
    assert "# Dynamic Pareto Routing Simulation Report" in md
    assert "Total Blocks" in md
