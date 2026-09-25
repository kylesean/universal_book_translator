"""Regression tests for pipeline quality and gate invariants.

Validates terminology enforcement, abbreviation prompts, quality gate routing,
and audit reporting consistency.
"""

import asyncio
from pathlib import Path
from typing import Any

from tests.stage_ctx_factory import build_stage_ctx, drain, inert_event
from ubt.core.config import UBTConfig
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.engine.reporter import build_quality_report, render_kdp_audit_markdown
from ubt.core.engine.stages.bible import run_bible_stage
from ubt.core.engine.stages.quality_gate import run_quality_gate_stage
from ubt.core.ir.models import (
    BlockStatus,
    BookManifest,
    ChapterIR,
    ChapterMeta,
    FlowID,
    IRBlock,
)
from ubt.core.ir.run_metadata import RunMetadata
from ubt.core.memory.abbreviation_backfill import build_backfill_prompt
from ubt.core.memory.character_miner import mine_characters
from ubt.core.qe.comet_runner import (
    GLOSSARY_VIOLATION_MARKER,
    QE_SCORE_GLOSSARY_VIOLATION,
    HeuristicQERunner,
)
from ubt.core.qe.fast_pass import FastPassFilter

# --- shared harness -------------------------------------------------------

_SRC = "The subthreshold swing degrades as the channel length shrinks to 20 nm."
_GOOD_TARGET = "当沟道长度缩短至 20 nm 时，亚阈值摆幅会退化。"
_BAD_TARGET = "当沟道长度缩短至 20 nm 时，短沟道效应会加剧。"
_GLOSSARY: list[dict[str, Any]] = [
    {"source": "subthreshold swing", "translation": "亚阈值摆幅", "aliases": []}
]


def _manifest(**metadata: Any) -> BookManifest:
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


# --- Terminology gate enforcement ---------------------------------------


def test_quality_gate_flags_glossary_violation(tmp_path: Path) -> None:
    """A fluent target that alters an enforced term cannot auto-pass.

    Ensures terminology violations are scored and routed to repair rather than auto-passing.
    """
    ledger = SQLiteJobLedger(tmp_path / "glossary_gate.sqlite")
    manifest = _manifest()
    ledger.init_job_from_manifest("job_gloss", manifest)
    blocks = [
        IRBlock(
            id="b1",
            flow_id=FlowID.MAIN_STORY,
            spine_index=1,
            source_text=_SRC,
            target_text=_GOOD_TARGET,
            status=BlockStatus.DRAFTED,
        ),
        IRBlock(
            id="b2",
            flow_id=FlowID.MAIN_STORY,
            spine_index=2,
            source_text=_SRC,
            target_text=_BAD_TARGET,
            status=BlockStatus.DRAFTED,
        ),
    ]
    ledger.append_chapter(
        "job_gloss",
        ChapterIR(doc_id="d1", chapter_id="c1", title="Chapter One", spine_index=1, blocks=blocks),
    )

    ctx = build_stage_ctx(
        tmp_path,
        ledger=ledger,
        job_id="job_gloss",
        fast_pass=FastPassFilter(source_lang="en", target_lang="zh"),
        qe_runner=HeuristicQERunner(),
        create_event=inert_event,
        glossary_dicts=_GLOSSARY,
    )
    asyncio.run(drain(run_quality_gate_stage(ctx)))

    by_id = {b.id: b for b in ledger.get_all_blocks("job_gloss")}
    assert by_id["b1"].status is BlockStatus.MTQE_PASSED
    assert by_id["b2"].status is BlockStatus.REPAIR_PENDING
    assert any(GLOSSARY_VIOLATION_MARKER in f for f in by_id["b2"].error_flags)
    assert by_id["b2"].mtqe_score == QE_SCORE_GLOSSARY_VIOLATION
    assert (by_id["b2"].mtqe_score or 0.0) < 0.75  # below the auto-pass band
    # The correct rendering is untouched by the new signal.
    assert by_id["b1"].mtqe_score is None


def test_structural_only_verdict_still_passes_without_glossary(tmp_path: Path) -> None:
    """With no glossary threaded in, clean blocks auto-pass without flags."""
    ledger = SQLiteJobLedger(tmp_path / "no_glossary.sqlite")
    ledger.init_job_from_manifest("job_plain", _manifest())
    blocks = [
        IRBlock(
            id="b1",
            flow_id=FlowID.MAIN_STORY,
            spine_index=1,
            source_text=_SRC,
            target_text=_BAD_TARGET,
            status=BlockStatus.DRAFTED,
        )
    ]
    ledger.append_chapter(
        "job_plain",
        ChapterIR(doc_id="d1", chapter_id="c1", title="Chapter One", spine_index=1, blocks=blocks),
    )
    ctx = build_stage_ctx(
        tmp_path,
        ledger=ledger,
        job_id="job_plain",
        fast_pass=FastPassFilter(source_lang="en", target_lang="zh"),
        qe_runner=HeuristicQERunner(),
        create_event=inert_event,
    )
    asyncio.run(drain(run_quality_gate_stage(ctx)))
    by_id = {b.id: b for b in ledger.get_all_blocks("job_plain")}
    assert by_id["b1"].status is BlockStatus.MTQE_PASSED
    assert by_id["b1"].error_flags == []


# --- Unbiased score metrics ---------------------------------------------


def test_reported_average_excludes_unscored_placeholders(tmp_path: Path) -> None:
    """A mostly-skip job must not report ~1.0 as if every block was scored.

    499 skips + one 0.30 defect used to average 0.9994 — an almost perfect book
    built from a single defective sample.
    """
    ledger = SQLiteJobLedger(tmp_path / "honest_avg.sqlite")
    manifest = _manifest()
    ledger.init_job_from_manifest("job_skip", manifest)
    blocks = [
        IRBlock(
            id=f"skip{i:02d}",
            flow_id=FlowID.MAIN_STORY,
            spine_index=i,
            source_text=f"Verbatim source line {i}.",
            target_text=f"Verbatim source line {i}.",
            status=BlockStatus.MTQE_PASSED,
            skip_translate=True,
            mtqe_score=1.0,  # placeholder, never met the QE gate
        )
        for i in range(1, 21)
    ]
    blocks.append(
        IRBlock(
            id="defect",
            flow_id=FlowID.MAIN_STORY,
            spine_index=99,
            source_text="The value is 42 units.",
            target_text="该值很大。",
            status=BlockStatus.REPAIR_PENDING,
            mtqe_score=0.30,
            error_flags=["Numeric fidelity failure: missing 42"],
        )
    )
    ledger.append_chapter(
        "job_skip",
        ChapterIR(doc_id="d1", chapter_id="c1", title="Chapter One", spine_index=1, blocks=blocks),
    )

    report = build_quality_report(
        ledger=ledger,
        job_id="job_skip",
        manifest=manifest,
        output_path=tmp_path / "out_bilingual.md",
    )
    assert report.score_metrics.scored_blocks == 1
    assert report.score_metrics.avg_qe == 0.30
    assert report.score_metrics.max_qe == 0.30
    # The number cannot be read as covering the whole book any more.
    markdown = render_kdp_audit_markdown(report)
    assert "Over 1 QE-scored block(s)" in markdown
    assert "Scored Population" in markdown


# --- Bible cache content invalidation -----------------------------------


async def _empty_backfill(*args: object, **kwargs: object) -> str:
    return ""


def _run_bible(
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
        complete_raw_fn=_empty_backfill,
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
    manifest = _manifest()
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

    first, msg_first = _run_bible(ledger, manifest, csv_file)
    assert {d["source"]: d["translation"] for d in first}["channel"] == "沟道"
    assert "established" in msg_first

    # Unchanged file: a real cache hit, same renderings.
    second, msg_second = _run_bible(ledger, manifest, csv_file)
    assert "reused" in msg_second
    assert {d["source"]: d["translation"] for d in second}["channel"] == "沟道"

    # Edited in place: the fingerprint changes, so the cache misses and the
    # new rendering is the one the pipeline enforces.
    csv_file.write_text("term,target\nchannel,通道\n", encoding="utf-8")
    third, msg_third = _run_bible(ledger, manifest, csv_file)
    assert "established" in msg_third
    assert {d["source"]: d["translation"] for d in third}["channel"] == "通道"

    # And a cache HIT must still return the freshly recomputed deterministic
    # entries (the old cached branch returned the payload and discarded them).
    manifest_with_title = _manifest(chapter_translations={"Chapter One": "第一章"})
    fourth, msg_fourth = _run_bible(ledger, manifest_with_title, csv_file)
    assert "reused" in msg_fourth
    assert {d["source"]: d["translation"] for d in fourth}["Chapter One"] == "第一章"


# --- Honorific parsing robustness ---------------------------------------


def test_general_relative_phrase_is_not_mined_as_person() -> None:
    """ "General Relativity" produced kind=person entry "Relativity"."""
    text = (
        "General Relativity is a theory of gravitation. "
        "General Relativity predicts the precession of Mercury. "
        "General Relativity has been tested repeatedly."
    )
    entries = mine_characters(text, min_freq=99)
    assert not [e for e in entries if e["source"].startswith("General")]
    _, prompt = build_backfill_prompt(entries, "zh")
    assert "Relativity" not in prompt


def test_honorific_plus_surname_still_mined_as_person() -> None:
    """The courtesy-title → surname path keeps extracting valid individuals."""
    entries = mine_characters("Dr. Smith arrived. Dr. Smith left again.", min_freq=99)
    smith = next(e for e in entries if e["source"] == "Dr. Smith")
    assert smith["kind"] == "person"
    assert smith["aliases"] == ["Smith"]
    _, prompt = build_backfill_prompt(entries, "zh")
    assert "Smith — Dr. Smith (person name)" in prompt


# --- Compliance statement consistency -----------------------------------


def _clean_report(tmp_path: Path, filename: str, route_mode: str | None) -> Any:
    """A passing report whose manifest records the given route mode."""
    metadata: dict[str, Any] = {}
    if route_mode is not None:
        metadata["route_decision"] = {"mode": route_mode, "pages": 5, "chars": 100}
    ledger = SQLiteJobLedger(tmp_path / f"{filename}.sqlite")
    manifest = _manifest(**metadata)
    ledger.init_job_from_manifest(filename, manifest)
    ledger.append_chapter(
        filename,
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
                    source_text="The gate oxide is 2 nm thick.",
                    target_text="栅氧化层厚度为 2 nm。",
                    status=BlockStatus.MTQE_PASSED,
                    mtqe_score=0.92,
                )
            ],
        ),
    )
    return build_quality_report(
        ledger=ledger,
        job_id=filename,
        manifest=manifest,
        output_path=tmp_path / f"{filename}_bilingual.md",
    )


def test_compliance_verdict_does_not_claim_enforcement_that_was_off(tmp_path: Path) -> None:
    """B4: the Aho-Corasick enforcer runs on the SHORT chain only.

    The pipeline sets ``deterministic_glossary_enforce=short_chain``, so a long
    chain job (the default for any book over the short-page cut-off) validates
    without enforcing. The KDP/compliance artifact nevertheless asserted
    "Enforced deterministically via Aho-Corasick glossary enforcer at export" for
    every clean job — claiming a control that was never switched on, in the one
    document a compliance reviewer would read.
    """
    long_md = render_kdp_audit_markdown(_clean_report(tmp_path, "longjob", "long"))
    assert "Enforced deterministically via Aho-Corasick" not in long_md
    assert "short chain only" in long_md
    assert "long" in long_md  # states the route this job actually took

    # The short chain genuinely does enforce deterministically.
    short_md = render_kdp_audit_markdown(_clean_report(tmp_path, "shortjob", "short"))
    assert "Enforced deterministically via Aho-Corasick" in short_md

    # An unknown route must not be upgraded into an enforcement claim either.
    unknown_md = render_kdp_audit_markdown(_clean_report(tmp_path, "noroute", None))
    assert "Enforced deterministically via Aho-Corasick" not in unknown_md
