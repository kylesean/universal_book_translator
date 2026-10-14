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
