"""Unit tests for cross-chapter entity coreference resolution and pronoun disambiguation."""

from __future__ import annotations

import pytest

from ubt.core.memory.coreference import (
    CoreferencePronounType,
    CoreferenceResolver,
)
from ubt.core.memory.hierarchical_memory import HierarchicalMemoryManager, StepSnapshot

pytestmark = pytest.mark.fast


def test_extract_entities_from_text() -> None:
    """Verify extraction of technical acronyms, multi-word terms, and glossary concepts."""
    resolver = CoreferenceResolver()
    text = (
        "We implement the Asynchronous Replicated State Machine (ARSM) "
        "and compare it against Byzantine Fault Tolerant Protocol. "
        "The experiments evaluate throughput on Transformer models."
    )
    glossary = ["Transformer", "throughput"]

    entities = resolver.extract_entities_from_text(text, glossary_terms=glossary)
    names = [e.entity_text for e in entities]

    assert "ARSM" in names
    assert "Asynchronous Replicated State Machine" in names
    assert "Byzantine Fault Tolerant Protocol" in names
    assert "Transformer" in names


def test_find_pronoun_mentions() -> None:
    """Verify identification of personal, demonstrative, and relational pronouns."""
    resolver = CoreferenceResolver()
    text = (
        "The system loads the configuration. It initializes the cache. "
        "This approach guarantees convergence. We evaluate both algorithms; "
        "the former achieves higher precision while the latter is faster."
    )
    mentions = resolver.find_pronoun_mentions(text)

    types = [m.pronoun_type for m in mentions]
    spans = [m.span.lower() for m in mentions]

    assert CoreferencePronounType.PERSONAL in types
    assert "it" in spans
    assert CoreferencePronounType.DEMONSTRATIVE in types
    assert "this approach" in spans
    assert CoreferencePronounType.RELATIONAL in types
    assert "the former" in spans
    assert "the latter" in spans


def test_resolve_inter_block_neighbor_antecedent() -> None:
    """Verify resolving pronoun in current block using neighbor context."""
    resolver = CoreferenceResolver()
    neighbor = "In the previous section, we designed the Byzantine Fault Tolerant Protocol."
    current = "In this section, we analyze its resilience under asymmetric network partitions."

    mentions = resolver.resolve_mentions(text=current, neighbor_text=neighbor)
    its_mention = next((m for m in mentions if m.span.lower() == "its"), None)

    assert its_mention is not None
    assert its_mention.inferred_antecedent == "Byzantine Fault Tolerant Protocol"
    assert its_mention.confidence >= 0.70
    assert its_mention.recommended_chinese_rendering == "该Byzantine Fault Tolerant Protocol的"


def test_resolve_relational_former_and_latter() -> None:
    """Verify 'the former' and 'the latter' map to first and second candidate entities."""
    resolver = CoreferenceResolver()
    neighbor = "We compare Convolutional Neural Network against Vision Transformer."
    current = (
        "The former is parameter efficient, whereas the latter yields superior attention maps."
    )

    mentions = resolver.resolve_mentions(text=current, neighbor_text=neighbor)
    former_mention = next((m for m in mentions if "former" in m.span.lower()), None)
    latter_mention = next((m for m in mentions if "latter" in m.span.lower()), None)

    assert former_mention is not None
    assert former_mention.inferred_antecedent == "Convolutional Neural Network"
    assert "前者" in (former_mention.recommended_chinese_rendering or "")

    assert latter_mention is not None
    assert latter_mention.inferred_antecedent == "Vision Transformer"
    assert "后者" in (latter_mention.recommended_chinese_rendering or "")


def test_resolve_demonstrative_phrase() -> None:
    """Verify 'this approach' resolves to preceding algorithm or concept."""
    resolver = CoreferenceResolver()
    neighbor = "We propose Layer Normalization to stabilize recurrent dynamics."
    current = "This approach eliminates internal covariate shift without micro-batch dependencies."

    mentions = resolver.resolve_mentions(text=current, neighbor_text=neighbor)
    demo_mention = next((m for m in mentions if "approach" in m.span.lower()), None)

    assert demo_mention is not None
    assert demo_mention.inferred_antecedent == "Layer Normalization"
    assert "该" in (demo_mention.recommended_chinese_rendering or "")


def test_dummy_it_is_excluded() -> None:
    """Verify dummy expletive 'it is important to ...' is not resolved as an entity."""
    resolver = CoreferenceResolver()
    text = "It is important to observe that gradients vanish in deep networks."

    mentions = resolver.resolve_mentions(text=text)
    # The dummy 'it' should remain unresolved
    it_mention = next((m for m in mentions if m.span.lower() == "it"), None)
    if it_mention:
        assert it_mention.inferred_antecedent is None


def test_format_coreference_prompt_hint() -> None:
    """Verify formatting of resolved mentions into an LLM prompt hint block."""
    resolver = CoreferenceResolver()
    neighbor = "The team implemented the Raft Consensus Protocol."
    current = "It guarantees state machine safety across partitions."

    mentions = resolver.resolve_mentions(text=current, neighbor_text=neighbor)
    hint = resolver.format_coreference_prompt_hint(mentions)

    assert "### Antecedent & Coreference Disambiguation Guidance" in hint
    assert "Raft Consensus Protocol" in hint
    assert "建议译为" in hint


def test_audit_pronoun_usage_clean_vs_overuse() -> None:
    """Verify pronoun audit detects natural prose vs '它-ism' repetition."""
    resolver = CoreferenceResolver()

    # Clean natural Chinese translation
    clean_text = (
        "深度学习模型具有极强的表征能力。该架构通过多层非线性变换提取特征，"
        "并在计算机视觉与自然语言处理领域取得了突破性进展。"
    )
    clean_report = resolver.audit_pronoun_usage(clean_text)
    assert clean_report.is_acceptable
    assert clean_report.pronoun_count == 0
    assert len(clean_report.issues) == 0

    # Overused '它-ism' Chinese translation
    overuse_text = "它是很好的模型。它的结构很复杂，它的参数很多，它能够处理图像，它的速度很快。"
    overuse_report = resolver.audit_pronoun_usage(overuse_text)
    assert not overuse_report.is_acceptable
    assert overuse_report.pronoun_count >= 4
    assert len(overuse_report.issues) > 0


def test_hierarchical_memory_salient_entities_integration() -> None:
    """Verify HierarchicalMemoryManager extracts salient entities for coreference."""
    mem_mgr = HierarchicalMemoryManager()
    mem_mgr._snapshots.append(
        StepSnapshot(
            step_index=1,
            start_spine=1,
            end_spine=10,
            chapter_id="ch01",
            summary_text="Chapter 1 introduces Convolutional Neural Network and Gradient Descent.",
            char_count=500,
        )
    )

    entities = mem_mgr.get_salient_entities()
    entity_names = [e.entity_text for e in entities]

    assert "Convolutional Neural Network" in entity_names
    assert "Gradient Descent" in entity_names
