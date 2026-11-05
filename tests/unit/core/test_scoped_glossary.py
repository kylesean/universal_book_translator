from __future__ import annotations

import pytest

from ubt.core.memory.bible import BibleEntry
from ubt.core.memory.glossary_table import (
    build_chunk_glossary_table,
    build_global_glossary_table,
)
from ubt.core.memory.scoped_glossary import (
    GlossaryConflictVerdict,
    ScopeContext,
    ScopeLevel,
    arbitrate_competing_entries,
    audit_glossary_conflicts,
    chapter_scope_from_manifest,
    glossary_has_block_scope,
    matches_scope,
    parse_scope_spec,
    resolve_effective_glossary,
    scope_context_for_block,
    score_context_affinity,
)
from ubt.core.validators.consistency import GlossaryConsistencyValidator
from ubt.core.validators.glossary_enforcer import DeterministicGlossaryEnforcer

pytestmark = pytest.mark.fast


def test_parse_scope_spec() -> None:
    assert parse_scope_spec(None) == (ScopeLevel.GLOBAL, "*")
    assert parse_scope_spec("") == (ScopeLevel.GLOBAL, "*")
    assert parse_scope_spec("*") == (ScopeLevel.GLOBAL, "*")
    assert parse_scope_spec("global") == (ScopeLevel.GLOBAL, "*")

    assert parse_scope_spec("chapter:2") == (ScopeLevel.CHAPTER, "2")
    assert parse_scope_spec("ch:5") == (ScopeLevel.CHAPTER, "5")
    assert parse_scope_spec("ch3") == (ScopeLevel.CHAPTER, "3")
    assert parse_scope_spec("4") == (ScopeLevel.CHAPTER, "4")

    assert parse_scope_spec("section:3.1") == (ScopeLevel.SECTION, "3.1")
    assert parse_scope_spec("sec:Introduction") == (ScopeLevel.SECTION, "Introduction")

    assert parse_scope_spec("block:blk-99") == (ScopeLevel.BLOCK, "blk-99")
    assert parse_scope_spec("blk:12") == (ScopeLevel.BLOCK, "12")

    assert parse_scope_spec("domain:hardware") == (ScopeLevel.DOMAIN, "hardware")
    assert parse_scope_spec("dom:psychology") == (ScopeLevel.DOMAIN, "psychology")


def test_glossary_has_block_scope_accepts_both_parser_aliases() -> None:
    assert glossary_has_block_scope([{"source": "x", "scope": "block:12"}])
    assert glossary_has_block_scope([{"source": "x", "scope": "blk:12"}])
    assert not glossary_has_block_scope([{"source": "x", "scope": "chapter:2"}])


def test_scope_context_does_not_guess_a_chapter_from_arbitrary_digits() -> None:
    from types import SimpleNamespace

    pdf = scope_context_for_block(SimpleNamespace(id="pdf_main#b0308"))
    assert pdf.chapter_index is None
    assert pdf.chapter_title is None

    namespace = scope_context_for_block(SimpleNamespace(id="pdf_main2#b0001"))
    assert namespace.chapter_index is None
    assert namespace.chapter_title is None

    bare = scope_context_for_block(SimpleNamespace(id="b0308"))
    assert bare.chapter_index is None

    chapter = scope_context_for_block(SimpleNamespace(id="ch_002#b0007"))
    assert chapter.chapter_index == 2

    from ubt.core.memory.scoped_glossary import chapter_index_from_id

    assert chapter_index_from_id("ch_002") == 2
    assert chapter_index_from_id("chapter-7") == 7
    assert chapter_index_from_id("c0003") == 3
    assert chapter_index_from_id("pdf_main") is None
    assert chapter_index_from_id("pdf_main2") is None
    assert chapter_index_from_id("b0308") is None

    explicit = scope_context_for_block(
        SimpleNamespace(id="pdf_main#b0308"),
        chapter_index=4,
        chapter_title="Operating Systems",
    )
    assert explicit.chapter_index == 4
    assert explicit.chapter_title == "Operating Systems"


def test_manifest_chapter_records_resolve_the_real_adapter_ids() -> None:
    from types import SimpleNamespace

    from ubt.core.ir.models import ChapterMeta

    chapters = (
        ChapterMeta(chapter_id="ch_001_foo.xhtml", title="Foundations", spine_index=1),
        ChapterMeta(chapter_id="pdf_main", title="Monograph", spine_index=7),
        ChapterMeta(chapter_id="docx_main", title="Report", spine_index=9),
        ChapterMeta(chapter_id="html_main", title="Web Export", spine_index=11),
    )

    for block_id, expected_index, expected_title in (
        ("ch_001_foo.xhtml#b0001", 1, "Foundations"),
        ("pdf_main#b0308", 7, "Monograph"),
        ("docx_main#b0002", 9, "Report"),
        ("html_main#b0001", 11, "Web Export"),
    ):
        ctx = scope_context_for_block(SimpleNamespace(id=block_id), chapters=chapters)
        assert (ctx.chapter_index, ctx.chapter_title) == (expected_index, expected_title)

    assert chapter_scope_from_manifest("ch_001_foo.xhtml", chapters) == (1, "Foundations")
    assert chapter_scope_from_manifest("pdf_main", chapters) == (7, "Monograph")
    assert chapter_scope_from_manifest("missing_main", chapters) == (None, None)
    assert chapter_scope_from_manifest(None, chapters) == (None, None)


def test_a_manifest_chapter_applies_its_scoped_term_and_unknowns_fall_back() -> None:
    from types import SimpleNamespace

    from ubt.core.ir.models import ChapterMeta

    entries = [
        BibleEntry(source="driver", translation="驾驶员", scope="*"),
        BibleEntry(source="driver", translation="驱动程序", scope="chapter:1"),
    ]
    chapters = [ChapterMeta(chapter_id="ch_001_foo.xhtml", title="Foundations", spine_index=1)]

    scoped = scope_context_for_block(
        SimpleNamespace(id="ch_001_foo.xhtml#b0001"), chapters=chapters
    )
    assert scoped.chapter_index == 1
    assert scoped.chapter_title == "Foundations"
    lookup = {e["source"]: e["translation"] for e in resolve_effective_glossary(entries, scoped)}
    assert lookup["driver"] == "驱动程序"

    unknown = scope_context_for_block(
        SimpleNamespace(id="ch_002_bar.xhtml#b0001"), chapters=chapters
    )
    assert (unknown.chapter_index, unknown.chapter_title) == (None, None)
    fallback = {e["source"]: e["translation"] for e in resolve_effective_glossary(entries, unknown)}
    assert fallback["driver"] == "驾驶员"


def test_prompt_global_table_excludes_scoped_entries() -> None:
    entries = [
        {"source": "Transformer", "translation": "变换器", "scope": "chapter:2"},
        {"source": "Transformer", "translation": "转换器", "scope": "chapter:5"},
    ]

    assert build_global_glossary_table(entries, 80, include_scoped=False) == ""
    # The TM fingerprint keeps seeing both, so a scoped-entry change still
    # invalidates cached translations.
    assert build_global_glossary_table(entries, 80).count("Transformer") == 2


def test_prompt_global_table_prefers_the_global_rendering() -> None:
    entries = [
        {"source": "Transformer", "translation": "变换器", "scope": "*"},
        {"source": "Transformer", "translation": "转换器", "scope": "chapter:5"},
    ]

    table = build_global_glossary_table(entries, 80, include_scoped=False)

    assert "变换器" in table
    assert "转换器" not in table


def test_matches_scope() -> None:
    ctx = ScopeContext(
        chapter_index=2,
        chapter_title="Operating Systems and Kernels",
        section_index="2.1",
        section_title="Device Management",
        block_id="blk-os-01",
        domain="computer_science",
        surrounding_text="The kernel loads the peripheral driver into memory.",
    )

    assert matches_scope("*", ctx) == (True, ScopeLevel.GLOBAL)
    assert matches_scope(None, ctx) == (True, ScopeLevel.GLOBAL)

    assert matches_scope("chapter:2", ctx) == (True, ScopeLevel.CHAPTER)
    assert matches_scope("ch:2", ctx) == (True, ScopeLevel.CHAPTER)
    assert matches_scope("ch2", ctx) == (True, ScopeLevel.CHAPTER)
    assert matches_scope("ch:3", ctx) == (False, ScopeLevel.CHAPTER)
    assert matches_scope("chapter:Kernels", ctx) == (True, ScopeLevel.CHAPTER)

    assert matches_scope("section:2.1", ctx) == (True, ScopeLevel.SECTION)
    assert matches_scope("sec:Device", ctx) == (True, ScopeLevel.SECTION)
    assert matches_scope("section:2.2", ctx) == (False, ScopeLevel.SECTION)

    assert matches_scope("block:blk-os-01", ctx) == (True, ScopeLevel.BLOCK)
    assert matches_scope("block:blk-os-02", ctx) == (False, ScopeLevel.BLOCK)

    assert matches_scope("domain:computer_science", ctx) == (True, ScopeLevel.DOMAIN)
    assert matches_scope("domain:golf", ctx) == (False, ScopeLevel.DOMAIN)


def test_score_context_affinity() -> None:
    entry_hardware = {
        "source": "driver",
        "translation": "驱动程序",
        "context_hints": ["hardware", "kernel", "device", "usb"],
        "negative_hints": ["golf", "psychology", "chauffeur"],
    }

    score1 = score_context_affinity(
        entry_hardware, "The USB device hardware driver failed to enumerate."
    )
    assert score1 >= 2.0

    score2 = score_context_affinity(entry_hardware, "The weather is pleasant today.")
    assert score2 == 0.0

    score3 = score_context_affinity(
        entry_hardware, "He took out his golf driver and struck the ball."
    )
    assert score3 <= -100.0


def test_arbitrate_competing_entries_scope_override() -> None:
    entry_global = {
        "source": "driver",
        "translation": "驾驶员",
        "scope": "*",
    }
    entry_chapter = {
        "source": "driver",
        "translation": "驱动程序",
        "scope": "chapter:2",
    }
    entry_block = {
        "source": "driver",
        "translation": "特定块驱动",
        "scope": "block:blk-spec",
    }

    ctx_ch2 = ScopeContext(chapter_index=2, block_id="blk-normal")
    winner, verdict, reason = arbitrate_competing_entries(
        [entry_global, entry_chapter, entry_block], ctx_ch2
    )
    assert winner["translation"] == "驱动程序"
    assert verdict == GlossaryConflictVerdict.RESOLVED_BY_SCOPE
    assert "chapter:2" in reason

    ctx_blk = ScopeContext(chapter_index=2, block_id="blk-spec")
    winner_blk, verdict_blk, _ = arbitrate_competing_entries(
        [entry_global, entry_chapter, entry_block], ctx_blk
    )
    assert winner_blk["translation"] == "特定块驱动"
    assert verdict_blk == GlossaryConflictVerdict.RESOLVED_BY_SCOPE

    ctx_ch5 = ScopeContext(chapter_index=5)
    winner_ch5, _, _ = arbitrate_competing_entries(
        [entry_global, entry_chapter, entry_block], ctx_ch5
    )
    assert winner_ch5["translation"] == "驾驶员"


def test_arbitrate_competing_entries_context_hints() -> None:
    entry_cs = {
        "source": "driver",
        "translation": "驱动程序",
        "scope": "global",
        "context_hints": ["hardware", "software", "kernel", "usb"],
    }
    entry_psy = {
        "source": "driver",
        "translation": "内在驱力",
        "scope": "global",
        "context_hints": ["psychology", "motivation", "emotion", "ego"],
    }

    ctx = ScopeContext(
        surrounding_text="Freud discussed the subconscious motivation and psychological driver."
    )
    winner, verdict, _ = arbitrate_competing_entries([entry_cs, entry_psy], ctx)
    assert winner["translation"] == "内在驱力"
    assert verdict == GlossaryConflictVerdict.RESOLVED_BY_CONTEXT


def test_arbitrate_competing_entries_priority_and_frequency() -> None:
    entry_low_prio = {
        "source": "term",
        "translation": "低优",
        "priority": 1,
    }
    entry_high_prio = {
        "source": "term",
        "translation": "高优",
        "priority": 10,
    }

    ctx = ScopeContext()
    winner_prio, verdict_prio, _ = arbitrate_competing_entries(
        [entry_low_prio, entry_high_prio], ctx
    )
    assert winner_prio["translation"] == "高优"
    assert verdict_prio == GlossaryConflictVerdict.RESOLVED_BY_PRIORITY

    entry_low_freq = {
        "source": "term",
        "translation": "低频",
        "frequency": 2,
    }
    entry_high_freq = {
        "source": "term",
        "translation": "高频",
        "frequency": 50,
    }
    winner_freq, verdict_freq, _ = arbitrate_competing_entries(
        [entry_low_freq, entry_high_freq], ctx
    )
    assert winner_freq["translation"] == "高频"
    assert verdict_freq == GlossaryConflictVerdict.RESOLVED_BY_FREQUENCY


def test_resolve_effective_glossary() -> None:
    entries = [
        BibleEntry(
            source="driver",
            translation="驾驶员",
            scope="*",
        ),
        BibleEntry(
            source="driver",
            translation="驱动程序",
            scope="chapter:2",
        ),
        BibleEntry(
            source="driver",
            translation="心理驱力",
            scope="chapter:7",
        ),
        BibleEntry(
            source="driver",
            translation="一号木杆",
            scope="chapter:10",
        ),
        BibleEntry(
            source="compiler",
            translation="编译器",
            scope="*",
        ),
    ]

    eff_ch2 = resolve_effective_glossary(entries, ScopeContext(chapter_index=2))
    lookup_ch2 = {e["source"]: e["translation"] for e in eff_ch2}
    assert lookup_ch2["driver"] == "驱动程序"
    assert lookup_ch2["compiler"] == "编译器"

    eff_ch7 = resolve_effective_glossary(entries, ScopeContext(chapter_index=7))
    lookup_ch7 = {e["source"]: e["translation"] for e in eff_ch7}
    assert lookup_ch7["driver"] == "心理驱力"

    eff_ch10 = resolve_effective_glossary(entries, ScopeContext(chapter_index=10))
    lookup_ch10 = {e["source"]: e["translation"] for e in eff_ch10}
    assert lookup_ch10["driver"] == "一号木杆"

    eff_ch99 = resolve_effective_glossary(entries, ScopeContext(chapter_index=99))
    lookup_ch99 = {e["source"]: e["translation"] for e in eff_ch99}
    assert lookup_ch99["driver"] == "驾驶员"


def test_audit_glossary_conflicts() -> None:
    clean_entries = [
        {"source": "driver", "translation": "驱动程序", "scope": "chapter:2"},
        {"source": "driver", "translation": "心理驱力", "scope": "chapter:7"},
        {"source": "server", "translation": "服务器", "scope": "*"},
    ]
    summary_clean = audit_glossary_conflicts(
        clean_entries,
        [ScopeContext(chapter_index=2), ScopeContext(chapter_index=7)],
    )
    assert summary_clean.is_clean
    assert summary_clean.conflicting_sources == 1
    assert summary_clean.resolved_conflicts == 1
    assert summary_clean.unresolved_conflicts == 0

    ambiguous_entries = [
        {"source": "driver", "translation": "驱动程序", "scope": "*"},
        {"source": "driver", "translation": "驾驶员", "scope": "*"},
    ]
    summary_ambig = audit_glossary_conflicts(ambiguous_entries)
    assert not summary_ambig.is_clean
    assert summary_ambig.unresolved_conflicts == 1


def test_glossary_table_and_enforcer_integration() -> None:
    entries = [
        BibleEntry(source="kernel", translation="操作系统内核", scope="chapter:1"),
        BibleEntry(source="kernel", translation="果核", scope="chapter:5"),
    ]
    raw_dicts = [e.model_dump() for e in entries]

    table_ch1 = build_global_glossary_table(
        raw_dicts, max_entries=50, context=ScopeContext(chapter_index=1)
    )
    assert "操作系统内核" in table_ch1
    assert "果核" not in table_ch1

    table_ch5 = build_global_glossary_table(
        raw_dicts, max_entries=50, context=ScopeContext(chapter_index=5)
    )
    assert "果核" in table_ch5
    assert "操作系统内核" not in table_ch5

    chunk_table = build_chunk_glossary_table(
        raw_dicts,
        abbreviation_entries=[],
        source_text="The kernel boots device drivers.",
        context=ScopeContext(chapter_index=1),
    )
    assert "操作系统内核" in chunk_table

    enforcer = DeterministicGlossaryEnforcer(
        raw_dicts,
        context=ScopeContext(chapter_index=1),
    )
    enforced_text, records = enforcer.enforce("kernel")
    assert enforced_text == "操作系统内核"

    validator = GlossaryConsistencyValidator(
        raw_dicts,
        context=ScopeContext(chapter_index=5),
    )
    val_lookup = {g["source"]: g["translation"] for g in validator.glossary}
    assert val_lookup["kernel"] == "果核"
