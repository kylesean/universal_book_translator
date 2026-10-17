"""Contract tests for the Translation Bible entry cleaning and merging."""

from __future__ import annotations

import pytest

from ubt.core.memory.bible import BibleEntry, BookBible, clean_bible_entry, merge_bible_entries

pytestmark = pytest.mark.fast


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #


def test_bible_entry_defaults() -> None:
    entry = BibleEntry(source="Foo", translation="福")
    assert entry.aliases == []
    assert entry.kind == "term"
    assert entry.frequency == 0


def test_bible_entry_ignores_extra_fields() -> None:
    entry = BibleEntry.model_validate({"source": "Foo", "translation": "福", "junk": 1})
    assert entry.source == "Foo"


def test_book_bible_defaults() -> None:
    bible = BookBible(doc_id="d")
    assert bible.language == "zh"
    assert bible.tone == ""
    assert bible.glossary == []
    assert bible.system_instruction == ""


# --------------------------------------------------------------------------- #
# clean_bible_entry
# --------------------------------------------------------------------------- #


def test_clean_strips_source_and_translation() -> None:
    entry = clean_bible_entry("  Foo  ", "  福  ")
    assert entry is not None
    assert entry.source == "Foo"
    assert entry.translation == "福"


@pytest.mark.parametrize(
    "wrapped", ["《福》", "〈福〉", "「福」", "『福』", '"福"', "'福'", "“福”", "”福”"]
)
def test_clean_strips_wrapping_quotes_and_brackets(wrapped: str) -> None:
    entry = clean_bible_entry("Foo", wrapped)
    assert entry is not None
    assert entry.translation == "福"


@pytest.mark.parametrize(("source", "translation"), [("", "福"), ("Foo", ""), ("  ", "  ")])
def test_clean_drops_empty_sides(source: str, translation: str) -> None:
    assert clean_bible_entry(source, translation) is None


def test_clean_drops_long_cjk_source() -> None:
    assert clean_bible_entry("中" * 13, "福") is None
    assert clean_bible_entry("中" * 12, "福") is not None


def test_clean_drops_alphabetic_source_over_four_words() -> None:
    assert clean_bible_entry("a b c d e", "福") is None
    assert clean_bible_entry("a b c d", "福") is not None


def test_clean_aliases_are_stripped_deduped_and_self_filtered() -> None:
    entry = clean_bible_entry("Foo", "福", aliases=["  Foo ", "F", "F", "", "  "])
    assert entry is not None
    assert entry.aliases == ["F"]


def test_clean_self_alias_match_is_case_insensitive() -> None:
    entry = clean_bible_entry("Foo", "福", aliases=["foo", "FOO", "Bar"])
    assert entry is not None
    assert entry.aliases == ["Bar"]


def test_clean_aliases_none_yields_empty_list() -> None:
    entry = clean_bible_entry("Foo", "福", aliases=None)
    assert entry is not None
    assert entry.aliases == []


def test_clean_passes_kind_through() -> None:
    entry = clean_bible_entry("Foo", "福", kind="person")
    assert entry is not None
    assert entry.kind == "person"


# --------------------------------------------------------------------------- #
# merge_bible_entries
# --------------------------------------------------------------------------- #


def test_merge_empty_list() -> None:
    assert merge_bible_entries([]) == []


def test_merge_is_case_insensitive_on_source() -> None:
    merged = merge_bible_entries(
        [
            BibleEntry(source="Foo", translation="甲", aliases=["F"], frequency=1),
            BibleEntry(source="foo", translation="乙", aliases=["G"], frequency=5),
        ]
    )
    assert len(merged) == 1
    assert merged[0].source == "Foo"  # first-seen casing
    assert merged[0].translation == "甲"  # first translation wins
    assert merged[0].aliases == ["F", "G"]
    assert merged[0].frequency == 5


def test_merge_fills_empty_translation_from_later_entry() -> None:
    merged = merge_bible_entries(
        [
            BibleEntry(source="Foo", translation="", aliases=[]),
            BibleEntry(source="FOO", translation="乙", aliases=[]),
        ]
    )
    assert merged[0].translation == "乙"


def test_merge_sorts_results_by_source_lower() -> None:
    merged = merge_bible_entries(
        [
            BibleEntry(source="zeta", translation="Z"),
            BibleEntry(source="Alpha", translation="A"),
        ]
    )
    assert [e.source for e in merged] == ["Alpha", "zeta"]


def test_merge_unions_and_sorts_aliases() -> None:
    merged = merge_bible_entries(
        [
            BibleEntry(source="Foo", translation="甲", aliases=["z", "a"]),
            BibleEntry(source="Foo", translation="甲", aliases=["m"]),
        ]
    )
    assert merged[0].aliases == ["a", "m", "z"]


def test_merge_keeps_first_kind() -> None:
    merged = merge_bible_entries(
        [
            BibleEntry(source="Foo", translation="甲", kind="person"),
            BibleEntry(source="foo", translation="甲", kind="term"),
        ]
    )
    assert merged[0].kind == "person"


def test_merge_keeps_distinct_sources_separate() -> None:
    merged = merge_bible_entries(
        [
            BibleEntry(source="Foo", translation="甲"),
            BibleEntry(source="Bar", translation="乙"),
        ]
    )
    assert [e.source for e in merged] == ["Bar", "Foo"]


# ---------------------------------------------------------------------------
# Canonical glossary typography (arXiv 2101.03961 autopsy): a rendering mined
# without the CJK<->Latin space ("切换Transformer") fought the document's own
# Pangu-spaced convention ("切换 Transformer") for the whole book — three
# blocks burned repair rounds for "drifting" away from a form the exporter
# itself would rewrite.
# ---------------------------------------------------------------------------


def test_clean_bible_entry_normalizes_cjk_latin_boundary_spacing() -> None:
    entry = clean_bible_entry(source="Switch Transformer", translation="切换Transformer")
    assert entry is not None
    assert entry.translation == "切换 Transformer"


def test_clean_bible_entry_normalizes_both_boundaries_of_a_mixed_rendering() -> None:
    entry = clean_bible_entry(source="PagedAttention", translation="分页注意力PagedAttention内核")
    assert entry is not None
    assert entry.translation == "分页注意力 PagedAttention 内核"


def test_clean_bible_entry_never_spaces_full_width_punctuation() -> None:
    # 多头注意力（MHA） keeps its full-width bracket glued, matching the
    # Pangu rules the export stage applies (half-width brackets DO space).
    glued = clean_bible_entry(source="MHA", translation="多头注意力（MHA）")
    assert glued is not None
    assert glued.translation == "多头注意力（MHA）"
    spaced = clean_bible_entry(source="MHA", translation="多头注意力(MHA)层")
    assert spaced is not None
    assert spaced.translation == "多头注意力 (MHA) 层"


def test_clean_bible_entry_leaves_latin_latin_spacing_alone() -> None:
    entry = clean_bible_entry(source="New York", translation="New York")
    assert entry is not None
    assert entry.translation == "New York"


def test_clean_bible_entry_preserves_an_already_spaced_boundary() -> None:
    entry = clean_bible_entry(source="Switch Transformer", translation="切换 Transformer")
    assert entry is not None
    assert entry.translation == "切换 Transformer"  # not doubled


def test_the_space_normalized_entry_still_catches_a_real_drift() -> None:
    # End to end: the canonical spaced form accepts both spellings on the
    # target side (detection is space-tolerant) but a target that keeps the
    # English leak is still drift.
    from ubt.core.validators.consistency import GlossaryConsistencyValidator

    entry = clean_bible_entry(source="Switch Transformer", translation="切换Transformer")
    assert entry is not None
    validator = GlossaryConsistencyValidator(glossary=[entry.model_dump()])
    assert validator.validate(
        "Is Switch Transformer better?", "切换 Transformer 的性能更好"
    ).is_valid
    assert validator.validate(
        "Is Switch Transformer better?", "切换Transformer 的性能更好"
    ).is_valid
    assert not validator.validate(
        "Is Switch Transformer better?", "Switch Transformer 性能更好"
    ).is_valid


# --------------------------------------------------------------------------- #
# Concept-level merging: one concept, one entry, every approved rendering kept
#
# Measured on arXiv 2606.23344: the bible carried BOTH
#   state-of-the-art -> 当前最先进的          (aliases: SOTA)
#   state-of-the-art (SOTA) -> 当前最优（SOTA）
# The first came from the abbreviation miner + backfill, the second from the
# LLM skeleton extractor. Each judged the other's correct rendering as drift, so
# no translation could satisfy both: b0002 wrote 当前最优（SOTA） and scored 0.25,
# b0006 wrote 当前最先进的 and scored 0.35. Both were correct.
# --------------------------------------------------------------------------- #


def test_an_attested_acronym_folds_onto_its_expansion() -> None:
    from ubt.core.memory.bible import concept_key

    assert concept_key("state-of-the-art (SOTA)") == concept_key("state-of-the-art")
    assert concept_key("Long Short-Term Memory (LSTM)") == concept_key("Long Short-Term Memory")
    assert concept_key("Parameter-Efficient Fine-Tuning (PEFT)") == concept_key(
        "Parameter-Efficient Fine-Tuning"
    )


def test_a_parenthetical_that_does_not_spell_its_base_stays_a_separate_concept() -> None:
    """The fold must be earned by the initials, not by the shape.

    ``Transformer (XL)`` and ``GPT (4)`` are not acronym attestations; folding
    them would silently merge unrelated terms.
    """
    from ubt.core.memory.bible import concept_key

    assert concept_key("Transformer (XL)") != concept_key("Transformer")
    assert concept_key("GPT (4)") != concept_key("GPT")


def test_two_renderings_of_one_concept_become_preferred_plus_admitted() -> None:
    merged = merge_bible_entries(
        [
            BibleEntry(source="state-of-the-art", translation="当前最先进的", aliases=["SOTA"]),
            BibleEntry(
                source="state-of-the-art (SOTA)", translation="当前最优（SOTA）", frequency=1
            ),
        ]
    )
    assert len(merged) == 1
    entry = merged[0]
    # The bare expansion is the preferred term even though the attestation was
    # seen more often: steering every sentence to embed "（SOTA）" is wrong.
    assert entry.source == "state-of-the-art"
    assert entry.translation == "当前最先进的"
    assert "当前最优（SOTA）" in entry.inflected_variants
    assert "state-of-the-art (SOTA)" in entry.aliases


def test_a_merged_concept_accepts_both_rendering_in_the_gate() -> None:
    """The end the merge exists for: neither rendering may read as drift."""
    from ubt.core.qe.term_drift import detect_term_drift

    glossary = [
        e.model_dump()
        for e in merge_bible_entries(
            [
                BibleEntry(source="state-of-the-art", translation="当前最先进的", aliases=["SOTA"]),
                BibleEntry(
                    source="state-of-the-art (SOTA)", translation="当前最优（SOTA）", frequency=1
                ),
            ]
        )
    ]
    source = "RT-DocLayout achieves state-of-the-art (SOTA) performance across benchmarks."
    for target in (
        "RT-DocLayout 在多个基准上达到了当前最优（SOTA）性能。",
        "RT-DocLayout 在多个基准上实现了当前最先进的性能。",
    ):
        (finding,) = detect_term_drift(source, target, glossary)
        assert finding.drifted is False, target


def test_an_admitted_rendering_is_never_rewritten_by_the_enforcer() -> None:
    """Admitted means accepted in place, not accepted-then-replaced."""
    from ubt.core.validators.glossary_enforcer import DeterministicGlossaryEnforcer

    glossary = [
        e.model_dump()
        for e in merge_bible_entries(
            [
                BibleEntry(source="state-of-the-art", translation="当前最先进的", aliases=["SOTA"]),
                BibleEntry(source="state-of-the-art (SOTA)", translation="当前最优（SOTA）"),
            ]
        )
    ]
    enforcer = DeterministicGlossaryEnforcer(glossary)
    target = "该方法达到了当前最优（SOTA）性能。"
    corrected, applied, _ = enforcer.enforce_audited(target)
    assert corrected == target
    assert applied == []


def test_distinct_concepts_are_not_merged_by_the_fold() -> None:
    merged = merge_bible_entries(
        [
            BibleEntry(source="attention", translation="注意力"),
            BibleEntry(source="Transformer (XL)", translation="变换器（XL）"),
        ]
    )
    assert len(merged) == 2


def test_merge_does_not_invent_admitted_variants_for_a_single_entry() -> None:
    merged = merge_bible_entries([BibleEntry(source="Foo", translation="甲")])
    assert merged[0].inflected_variants == []
