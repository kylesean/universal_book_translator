"""Q-3: curated seed glossary + control-char debris strip."""

import asyncio
from pathlib import Path
from typing import Any

from tests.stage_ctx_factory import build_stage_ctx, drain, inert_event
from ubt.core.cleaners.lnds_pruner import strip_textbook_ocr_artifacts
from ubt.core.config import UBTConfig
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.engine.stages.bible import run_bible_stage
from ubt.core.ir.models import (
    BlockStatus,
    BlockType,
    BookManifest,
    ChapterIR,
    ChapterMeta,
    FlowID,
    IRBlock,
)
from ubt.core.ir.run_metadata import RunMetadata
from ubt.core.memory.bible import BibleEntry, merge_bible_entries
from ubt.core.memory.cjk_matcher import select_terms_for_chunk
from ubt.core.memory.seed_glossary import seed_entries_for_profile
from ubt.core.validators.glossary_enforcer import DeterministicGlossaryEnforcer


def _glossary_dicts(profile: str = "semiconductor") -> list[dict[str, Any]]:
    return [e.model_dump() for e in seed_entries_for_profile(profile)]


def test_all_seeds_survive_guardrails() -> None:
    entries = seed_entries_for_profile("semiconductor")
    assert len(entries) == 43
    assert all(e.translation for e in entries)


def test_unknown_profiles_get_no_seeds() -> None:
    assert seed_entries_for_profile("general") == []
    assert seed_entries_for_profile("fiction") == []
    assert seed_entries_for_profile("textbook") == []
    assert seed_entries_for_profile("paper") == []
    assert seed_entries_for_profile("") == []


def test_semiconductor_profiles_have_seeds() -> None:
    assert len(seed_entries_for_profile("semiconductor")) == 43
    assert len(seed_entries_for_profile("semiconductor_paper")) == 43
    assert len(seed_entries_for_profile("semiconductor_textbook")) == 43


def test_seed_translation_wins_merge() -> None:
    seeds = seed_entries_for_profile("semiconductor")
    mined_dup = BibleEntry(source="subthreshold swing", translation="摆动幅度", kind="term")
    merged = merge_bible_entries([*seeds, mined_dup])
    hit = next(e for e in merged if e.source == "subthreshold swing")
    assert hit.translation == "亚阈值摆幅"


def test_enforcer_replaces_source_leak_with_seed() -> None:
    enc = DeterministicGlossaryEnforcer(
        glossary=_glossary_dicts(), target_lang="zh", source_lang="en"
    )
    fixed, records = enc.enforce("器件表现出优异的 subthreshold swing 特性。")
    assert "亚阈值摆幅" in fixed
    assert "subthreshold swing" not in fixed
    assert records


def test_enforcer_keeps_acronyms_verbatim() -> None:
    enc = DeterministicGlossaryEnforcer(
        glossary=_glossary_dicts(), target_lang="zh", source_lang="en"
    )
    # No FET entry exists, and MOSFET must survive boundary checks intact.
    fixed, _ = enc.enforce("FinFET 和 MOSFET 都是多栅器件。")
    assert "FinFET" in fixed and "MOSFET" in fixed


def test_control_debris_stripped_sentinel_kept() -> None:
    assert strip_textbook_ocr_artifacts("T si \x07 L g") == "T si L g"
    assert "\x00SPAN0\x00" in strip_textbook_ocr_artifacts("\x00SPAN0\x00 内文")
    assert (
        strip_textbook_ocr_artifacts("a\tb\nc") == "a b\nc"
    )  # \t collapse is pre-existing; \n kept


def test_seed_term_selected_for_draft_chunk() -> None:
    dicts = _glossary_dicts()
    picked = select_terms_for_chunk(
        dicts, "The subthreshold swing degrades as Lg shrinks.", top_n=5
    )
    assert any(d.get("source") == "subthreshold swing" for d in picked)


async def test_bible_stage_emits_seeds_into_glossary_dicts(tmp_path: Path) -> None:
    ledger = SQLiteJobLedger(tmp_path / "t.sqlite")
    manifest = BookManifest(doc_id="d1", title="t", source_path="s.pdf", chapters=[], metadata={})
    ledger.init_job_from_manifest("j1", manifest)
    block = IRBlock(
        id="b1",
        flow_id=FlowID.MAIN_STORY,
        spine_index=1,
        block_type=BlockType.NARRATIVE,
        source_text="The subthreshold swing degrades as Lg shrinks.",
        target_text=None,
    )
    ledger.append_chapter(
        "j1",
        ChapterIR(
            doc_id="d1",
            chapter_id="c1",
            title="c1",
            spine_index=1,
            blocks=[block],
        ),
    )

    async def _empty_backfill(*args: object, **kwargs: object) -> str:
        return ""

    manifest2 = BookManifest(doc_id="d1", title="t", source_path="s.pdf", chapters=[], metadata={})
    ctx = build_stage_ctx(
        tmp_path,
        ledger=ledger,
        job_id="j1",
        manifest=manifest2,
        profile_name="semiconductor",
        target_lang="zh",
        source_lang="en",
    )
    _events = [e async for e in run_bible_stage(ctx)]
    glossary_dicts = ctx.glossary_dicts
    by_source = {d["source"]: d["translation"] for d in glossary_dicts}
    assert by_source.get("subthreshold swing") == "亚阈值摆幅"
    assert by_source.get("MOSFET") == "MOSFET"


async def test_bible_stage_generic_textbook_gets_no_semiconductor_seeds(tmp_path: Path) -> None:
    ledger = SQLiteJobLedger(tmp_path / "t2.sqlite")
    manifest = BookManifest(doc_id="d1", title="t", source_path="s.pdf", chapters=[], metadata={})
    ledger.init_job_from_manifest("j2", manifest)
    block = IRBlock(
        id="b1",
        flow_id=FlowID.MAIN_STORY,
        spine_index=1,
        block_type=BlockType.NARRATIVE,
        source_text="General biology discussion.",
        target_text=None,
    )
    ledger.append_chapter(
        "j2",
        ChapterIR(
            doc_id="d1",
            chapter_id="c1",
            title="c1",
            spine_index=1,
            blocks=[block],
        ),
    )

    async def _empty_backfill(*args: object, **kwargs: object) -> str:
        return ""

    manifest2 = BookManifest(doc_id="d1", title="t", source_path="s.pdf", chapters=[], metadata={})
    ctx = build_stage_ctx(
        tmp_path,
        ledger=ledger,
        job_id="j2",
        manifest=manifest2,
        profile_name="textbook",
        target_lang="zh",
        source_lang="en",
    )
    _events = [e async for e in run_bible_stage(ctx)]
    glossary_dicts = ctx.glossary_dicts
    by_source = {d["source"]: d["translation"] for d in glossary_dicts}
    assert "subthreshold swing" not in by_source
    assert "MOSFET" not in by_source


def test_load_external_glossary_json_and_csv(tmp_path: Path) -> None:
    import json

    from ubt.core.memory.seed_glossary import load_external_glossary

    json_file = tmp_path / "glossary.json"
    json_file.write_text(json.dumps({"Fin": "鳍片", "channel": "沟道"}), encoding="utf-8")
    entries = load_external_glossary(json_file)
    assert len(entries) == 2
    assert {e.source: e.translation for e in entries} == {"Fin": "鳍片", "channel": "沟道"}

    csv_file = tmp_path / "glossary.csv"
    csv_file.write_text("source,translation\ndrain,漏极\ngate,栅极\n", encoding="utf-8")
    csv_entries = load_external_glossary(csv_file)
    assert len(csv_entries) == 2
    assert {e.source: e.translation for e in csv_entries} == {"drain": "漏极", "gate": "栅极"}


async def test_bible_stage_with_custom_glossary(tmp_path: Path) -> None:
    ledger = SQLiteJobLedger(tmp_path / "t_custom.sqlite")
    manifest = BookManifest(doc_id="d1", title="t", source_path="s.pdf", chapters=[], metadata={})
    ledger.init_job_from_manifest("j_custom", manifest)
    block = IRBlock(
        id="b1",
        flow_id=FlowID.MAIN_STORY,
        spine_index=1,
        block_type=BlockType.NARRATIVE,
        source_text="The channel is doped with donors.",
        target_text=None,
    )
    ledger.append_chapter(
        "j_custom",
        ChapterIR(
            doc_id="d1",
            chapter_id="c1",
            title="c1",
            spine_index=1,
            blocks=[block],
        ),
    )

    csv_file = tmp_path / "my_terms.csv"
    csv_file.write_text("term,target\nchannel,沟道\ndonor,施主\n", encoding="utf-8")

    async def _empty_backfill(*args: object, **kwargs: object) -> str:
        return ""

    manifest2 = BookManifest(doc_id="d1", title="t", source_path="s.pdf", chapters=[], metadata={})
    ctx = build_stage_ctx(
        tmp_path,
        ledger=ledger,
        job_id="j_custom",
        manifest=manifest2,
        profile_name="general",
        target_lang="zh",
        source_lang="en",
        config=UBTConfig(glossary_path=csv_file),
    )
    _events = [e async for e in run_bible_stage(ctx)]
    glossary_dicts = ctx.glossary_dicts
    by_source = {d["source"]: d["translation"] for d in glossary_dicts}
    assert by_source.get("channel") == "沟道"
    assert by_source.get("donor") == "施主"


async def _r0917_empty_backfill(*args: object, **kwargs: object) -> str:
    return ""


def _r0917_manifest(**metadata: Any) -> BookManifest:
    """Run decisions go on the typed contract; source keys stay in the dict."""
    run_keys = {k: v for k, v in metadata.items() if k in RunMetadata.model_fields}
    artifact = {k: v for k, v in metadata.items() if k not in RunMetadata.model_fields}
    return BookManifest(
        doc_id="d1",
        title="Regression",
        source_path="/tmp/regression.md",
        chapters=[ChapterMeta(chapter_id="c1", title="Chapter One", spine_index=1)],
        metadata=artifact,
        run=RunMetadata(**run_keys),
    )


def _r0917_run_bible(
    ledger: SQLiteJobLedger, manifest: BookManifest, csv_file: Path
) -> tuple[list[dict[str, Any]], str]:
    ctx = build_stage_ctx(
        csv_file.parent,
        ledger=ledger,
        job_id="job_bible",
        manifest=manifest,
        profile_name="general",
        target_lang="zh",
        source_lang="en",
        create_event=inert_event,
        complete_raw_fn=_r0917_empty_backfill,
        config=UBTConfig(db_dir=csv_file.parent, glossary_path=csv_file),
    )

    async def _go() -> tuple[list[dict[str, Any]], str]:
        events = await drain(run_bible_stage(ctx))
        return ctx.glossary_dicts, events[0].message if events else ""

    return asyncio.run(_go())


def test_bible_cache_tracks_glossary_content(tmp_path: Path) -> None:
    """Editing the glossary CSV in place must invalidate the cached bible."""
    csv_file = tmp_path / "terms.csv"
    csv_file.write_text("term,target\nchannel,沟道\n", encoding="utf-8")
    ledger = SQLiteJobLedger(tmp_path / "bible.sqlite")
    manifest = _r0917_manifest()
    ledger.init_job_from_manifest("job_bible", manifest)
    ledger.append_chapter(
        "job_bible",
        ChapterIR(
            doc_id="d1",
            chapter_id="c1",
            title="Chapter One",
            spine_index=1,
            blocks=[
                IRBlock(
                    id="b1",
                    flow_id=FlowID.MAIN_STORY,
                    spine_index=1,
                    source_text="The channel is doped.",
                    status=BlockStatus.PENDING,
                )
            ],
        ),
    )

    first, msg_first = _r0917_run_bible(ledger, manifest, csv_file)
    assert {d["source"]: d["translation"] for d in first}["channel"] == "沟道"
    assert "established" in msg_first

    # Unchanged file: a real cache hit, same renderings.
    second, msg_second = _r0917_run_bible(ledger, manifest, csv_file)
    assert "reused" in msg_second
    assert {d["source"]: d["translation"] for d in second}["channel"] == "沟道"

    # Edited in place: the fingerprint changes, so the cache misses and the
    # new rendering is the one the pipeline enforces.
    csv_file.write_text("term,target\nchannel,通道\n", encoding="utf-8")
    third, msg_third = _r0917_run_bible(ledger, manifest, csv_file)
    assert "established" in msg_third
    assert {d["source"]: d["translation"] for d in third}["channel"] == "通道"

    # And a cache HIT must still return the freshly recomputed deterministic
    # entries (the old cached branch returned the payload and discarded them).
    manifest_with_title = _r0917_manifest(chapter_translations={"Chapter One": "第一章"})
    fourth, msg_fourth = _r0917_run_bible(ledger, manifest_with_title, csv_file)
    assert "reused" in msg_fourth
    assert {d["source"]: d["translation"] for d in fourth}["Chapter One"] == "第一章"


def test_global_terminology_sheet_is_capped_by_rank() -> None:
    from ubt.core.config import UBTConfig
    from ubt.core.memory.glossary_table import build_global_glossary_table

    terms = [
        {"source": "node", "translation": "节点", "kind": "term", "frequency": 5, "aliases": []},
        {
            "source": "Elizabeth",
            "translation": "伊丽莎白",
            "kind": "person",
            "frequency": 3,
            "aliases": ["Lizzy"],
        },
        # What load_external_glossary seeds for a user-supplied term.
        {
            "source": "gate",
            "translation": "栅极",
            "kind": "term",
            "frequency": 10_000,
            "aliases": [],
        },
    ]
    assert build_global_glossary_table(terms, 0) == ""
    two = build_global_glossary_table(terms, 2)
    # Names outrank frequency; a user's own decision outranks a mined common term.
    assert "| Elizabeth |" in two and "| gate |" in two
    assert "| node |" not in two
    # The documented default is the cap itself; the guide's table cites it.
    assert UBTConfig().glossary_max_global_entries == 100
