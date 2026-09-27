"""Unit tests for QualityReport generator."""

from pathlib import Path
from typing import Any

from tests.unit.ir_seed import SeedDoc, seed_job
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.engine.reporter import (
    QualityReport,
    build_quality_report,
    render_kdp_audit_markdown,
    save_quality_report,
)
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


def test_build_and_save_quality_report(tmp_path: Path) -> None:
    db_path = tmp_path / "test_report.sqlite"
    ledger = SQLiteJobLedger(db_path)
    job_id = "job_report_001"

    manifest = BookManifest(
        doc_id="doc_hash_12345",
        title="Quality Test Book",
        source_path="source.md",
        chapters=[
            ChapterMeta(chapter_id="ch01", title="Chapter 1", spine_index=1),
        ],
        run=RunMetadata(
            config_snapshot={
                "draft_model": "test-draft-model",
                "repair_model": "test-repair-model",
            }
        ),
    )
    ledger.init_job_from_manifest(job_id, manifest)

    blocks = [
        IRBlock(
            id="b01",
            flow_id=FlowID.MAIN_STORY,
            spine_index=1,
            block_type=BlockType.NARRATIVE,
            source_text="First block",
            target_text="第一块",
            status=BlockStatus.MTQE_PASSED,
            mtqe_score=0.95,
            repair_rounds=0,
        ),
        IRBlock(
            id="b02",
            flow_id=FlowID.MAIN_STORY,
            spine_index=2,
            block_type=BlockType.NARRATIVE,
            source_text="Second block",
            target_text="第二块",
            status=BlockStatus.REPAIRED,
            mtqe_score=0.82,
            repair_rounds=1,
        ),
        IRBlock(
            id="b03",
            flow_id=FlowID.MAIN_STORY,
            spine_index=3,
            block_type=BlockType.NARRATIVE,
            source_text="Third block",
            target_text="第三块",
            status=BlockStatus.FAILED,
            mtqe_score=0.41,
            repair_rounds=2,
            error_flags=["html_tag_mismatch"],
        ),
    ]
    ledger.append_chapter(
        job_id,
        ChapterIR(
            doc_id=manifest.doc_id,
            chapter_id="ch01",
            title="Chapter 1",
            spine_index=1,
            blocks=blocks,
        ),
    )
    for b in blocks:
        ledger.save_checkpoint(
            block_id=b.id,
            target_text=b.target_text or "",
            status=b.status,
            mtqe_score=b.mtqe_score,
            repair_rounds=b.repair_rounds,
            error_flags=b.error_flags,
        )

    out_file = tmp_path / "out_bilingual.md"
    report = build_quality_report(
        ledger=ledger,
        job_id=job_id,
        manifest=manifest,
        output_path=out_file,
    )

    # 1. Check summary
    assert report.job_id == job_id
    assert report.book_title == "Quality Test Book"
    assert report.summary.total_blocks == 3
    assert report.summary.completed_blocks == 2  # MTQE_PASSED + REPAIRED
    assert report.summary.repaired_blocks == 1
    assert report.summary.failed_blocks == 1
    assert 0.65 <= report.summary.pass_rate <= 0.67
    # The summary's buckets partition the corpus only when repaired is read as a
    # subset of completed, not a sibling bucket: the ledger counts
    # ``status IN ('mtqe_passed', 'repaired')`` as completed. Adding repaired on
    # top of completed double-counts it — the exact bug the e2e smoke assertion
    # carried, which stayed green there only because the dry-run never repairs a
    # block. Pinning both halves here (subset + partition) keeps a re-added
    # ``repaired`` term from sneaking back into any accounting sum.
    assert report.summary.repaired_blocks <= report.summary.completed_blocks
    assert (
        report.summary.completed_blocks
        + report.summary.failed_blocks
        + report.summary.needs_human_blocks
        + report.summary.blocked_human_blocks
        == report.summary.total_blocks
    )

    # 2. Check score metrics
    assert report.score_metrics.min_qe == 0.41
    assert report.score_metrics.max_qe == 0.95
    assert report.score_metrics.avg_qe > 0.7
    assert report.score_metrics.bottom_15_avg_qe == 0.41

    # 3. Check repair breakdown
    assert report.repair_breakdown.direct_pass_count == 1
    assert report.repair_breakdown.round_1_repaired_count == 1
    assert report.repair_breakdown.exhausted_count == 1
    assert report.defect_flags.get("html_tag_mismatch") == 1

    # 4. Save and reload JSON + KDP Markdown companion
    json_path = tmp_path / "report.json"
    saved = save_quality_report(report, json_path, write_markdown=True)
    assert saved.exists()
    assert "Quality Test Book" in saved.read_text(encoding="utf-8")

    md_path = json_path.with_suffix(".md")
    assert md_path.exists()
    md_content = md_path.read_text(encoding="utf-8")
    assert "Amazon KDP AI Content Disclosure Statement" in md_content
    assert "Quality Test Book" in md_content
    assert "Configuration Provenance" in md_content
    assert "test-draft-model" in md_content
    assert "HITL Remediation Queue" in md_content
    assert "HITL Quarantined (Needs Human Review)" in md_content


def test_quality_report_markdown_companion_is_opt_in(tmp_path: Path) -> None:
    """The KDP Markdown audit must not be written unless explicitly asked for.

    Unconditional emission put a second artifact in every output dir that no
    stale-report sweep knows about; opt-in keeps the machine contract at one
    JSON file unless a human asked for the review copy.
    """
    report = _minimal_report(tmp_path, "optin", {})

    default_path = tmp_path / "default_report.json"
    save_quality_report(report, default_path)
    assert not default_path.with_suffix(".md").exists(), "Markdown audit must be opt-in"

    opted_path = tmp_path / "opted_report.json"
    save_quality_report(report, opted_path, write_markdown=True)
    assert opted_path.with_suffix(".md").exists()


def test_render_coverage_and_route_honesty(tmp_path: Path) -> None:
    """pass_rate 1.0 must not masquerade as full delivery (W1诚实化)."""
    db_path = tmp_path / "honest.sqlite"
    ledger = SQLiteJobLedger(db_path)
    job_id = "job_honest_001"
    manifest = BookManifest(
        doc_id="doc_honest",
        title="Honest Book",
        source_path="honest.pdf",
        chapters=[ChapterMeta(chapter_id="ch01", title="Chapter 1", spine_index=1)],
        run=RunMetadata(
            route_decision={
                "mode": "long",
                "pages": 100,
                "chars": 99999,
                "reason": "long book (100pp>30)",
            },
            formula_mode="strict",
        ),
    )
    ledger.init_job_from_manifest(job_id, manifest)
    blocks = [
        IRBlock(
            id="h01",
            flow_id=FlowID.MAIN_STORY,
            spine_index=1,
            block_type=BlockType.NARRATIVE,
            source_text="Hello",
            target_text="你好",
            status=BlockStatus.MTQE_PASSED,
            mtqe_score=0.92,
            repair_rounds=0,
        ),
        IRBlock(
            id="h02",
            flow_id=FlowID.MAIN_STORY,
            spine_index=2,
            block_type=BlockType.NARRATIVE,
            source_text="World",
            target_text="World",
            status=BlockStatus.MTQE_PASSED,
            mtqe_score=0.92,
            repair_rounds=0,
            error_flags=["inplace_skip:unmatched"],
        ),
    ]
    ledger.append_chapter(
        job_id,
        ChapterIR(
            doc_id=manifest.doc_id,
            chapter_id="ch01",
            title="Chapter 1",
            spine_index=1,
            blocks=blocks,
        ),
    )
    for b in blocks:
        ledger.save_checkpoint(
            block_id=b.id,
            target_text=b.target_text or "",
            status=b.status,
            mtqe_score=b.mtqe_score,
            repair_rounds=b.repair_rounds,
            error_flags=b.error_flags,
        )
    report = build_quality_report(
        ledger=ledger,
        job_id=job_id,
        manifest=manifest,
        output_path=tmp_path / "o.pdf",
    )
    # Pipeline completed everything; one block is fail-closed (source left
    # visible), which is now reported on its own dimension rather than by
    # shrinking render_coverage.
    assert report.summary.pass_rate == 1.0
    assert report.render_coverage.skipped_blocks == 1
    assert report.render_coverage.fail_closed_blocks == 1
    assert report.render_coverage.preserved_blocks == 0
    assert report.render_coverage.rendered_blocks == 2
    assert report.render_coverage.skip_families == {"unmatched": 1}
    assert report.route is not None
    assert report.route.mode == "long"
    assert report.route.pages == 100
    assert report.route.formula_mode == "strict"

    md_path = tmp_path / "honest_report.json"
    saved = save_quality_report(report, md_path, write_markdown=True)
    assert saved.exists()
    md_content = md_path.with_suffix(".md").read_text(encoding="utf-8")
    assert "Render Coverage" in md_content
    assert "left source-visible" in md_content


def test_reflow_engine_full_coverage(tmp_path: Path) -> None:
    """Publication reflow records no skips: coverage equals pass rate."""
    db_path = tmp_path / "full.sqlite"
    ledger = SQLiteJobLedger(db_path)
    job_id = "job_full_001"
    manifest = BookManifest(
        doc_id="doc_full",
        title="Full Book",
        source_path="full.pdf",
        chapters=[ChapterMeta(chapter_id="ch01", title="Chapter 1", spine_index=1)],
    )
    ledger.init_job_from_manifest(job_id, manifest)
    block = IRBlock(
        id="f01",
        flow_id=FlowID.MAIN_STORY,
        spine_index=1,
        block_type=BlockType.NARRATIVE,
        source_text="Hi",
        target_text="嗨",
        status=BlockStatus.MTQE_PASSED,
        mtqe_score=0.92,
        repair_rounds=0,
    )
    ledger.append_chapter(
        job_id,
        ChapterIR(
            doc_id=manifest.doc_id,
            chapter_id="ch01",
            title="Chapter 1",
            spine_index=1,
            blocks=[block],
        ),
    )
    ledger.save_checkpoint(
        block_id=block.id,
        target_text="嗨",
        status=block.status,
        mtqe_score=0.92,
        repair_rounds=0,
        error_flags=[],
    )
    report = build_quality_report(
        ledger=ledger,
        job_id=job_id,
        manifest=manifest,
        output_path=tmp_path / "o.pdf",
    )
    assert report.render_coverage.skipped_blocks == 0
    assert report.render_coverage.render_coverage == 1.0
    assert report.route is None


def _minimal_report(
    tmp_path: Path,
    name: str,
    metadata: dict[str, object],
    job_metadata: dict[str, object] | None = None,
    enforced_spans: int = 0,
) -> QualityReport:
    """Build a one-block quality report with the given manifest metadata.

    ``job_metadata`` is written to the ledger's job record (what the QE gate
    stamps, e.g. ``qe_score_source``), keeping it distinct from the manifest
    metadata the renderer carries.
    """
    from ubt.core.engine.reporter import build_quality_report

    db_path = tmp_path / f"{name}.sqlite"
    ledger = SQLiteJobLedger(db_path)
    job_id = f"job_{name}"
    manifest = BookManifest(
        doc_id=f"doc_{name}",
        title="Version Pin Book",
        source_path=f"{name}.pdf",
        chapters=[ChapterMeta(chapter_id="ch01", title="Chapter 1", spine_index=1)],
        metadata=metadata,
    )
    ledger.init_job_from_manifest(job_id, manifest)
    block = IRBlock(
        id="v01",
        flow_id=FlowID.MAIN_STORY,
        spine_index=1,
        block_type=BlockType.NARRATIVE,
        source_text="Hello",
        target_text="你好",
        status=BlockStatus.MTQE_PASSED,
        mtqe_score=0.92,
        repair_rounds=0,
    )
    ledger.append_chapter(
        job_id,
        ChapterIR(
            doc_id=manifest.doc_id,
            chapter_id="ch01",
            title="Chapter 1",
            spine_index=1,
            blocks=[block],
        ),
    )
    ledger.save_checkpoint(
        block_id=block.id,
        target_text=block.target_text or "",
        status=block.status,
        mtqe_score=block.mtqe_score,
        repair_rounds=block.repair_rounds,
        error_flags=block.error_flags,
    )
    for key, value in (job_metadata or {}).items():
        ledger.set_job_metadata_value(job_id, key, value)
    return build_quality_report(
        ledger=ledger,
        job_id=job_id,
        manifest=manifest,
        output_path=tmp_path / f"{name}.pdf",
        enforced_spans=enforced_spans,
    )


def test_quality_report_carries_schema_version_and_top_level_avg_qe(tmp_path: Path) -> None:
    """2026-09-22 review: the report had no schema_version and hid avg_qe
    under score_metrics, so consumers sniffed keys and had to know the nested
    path for the one number they always read."""
    import json

    from ubt.core.engine.reporter import QUALITY_REPORT_SCHEMA_VERSION

    report = _minimal_report(tmp_path, "versioned", {})

    # schema_version: field on the model, filled by build_quality_report,
    # and present in the serialized artifact.
    assert report.schema_version == QUALITY_REPORT_SCHEMA_VERSION
    dumped = report.model_dump()
    assert dumped["schema_version"] == QUALITY_REPORT_SCHEMA_VERSION
    saved = save_quality_report(report, tmp_path / "versioned_report.json")
    on_disk = json.loads(saved.read_text(encoding="utf-8"))
    assert on_disk["schema_version"] == QUALITY_REPORT_SCHEMA_VERSION

    # avg_qe: read-only alias of score_metrics.avg_qe — serialized at the
    # top level, never accepted as constructor input, never able to drift.
    assert report.avg_qe == report.score_metrics.avg_qe
    assert dumped["avg_qe"] == report.score_metrics.avg_qe
    assert on_disk["avg_qe"] == report.score_metrics.avg_qe
    assert "avg_qe" not in QualityReport.model_fields
    assert QualityReport.model_validate(dumped).avg_qe == report.avg_qe


def test_quality_report_carries_typst_version(tmp_path: Path) -> None:
    """Manifest `typst_version` surfaces on the report and in markdown."""
    from ubt.core.engine.reporter import render_kdp_audit_markdown

    report = _minimal_report(tmp_path, "pinned", {"typst_version": "0.15.1"})
    assert report.typst_version == "0.15.1"
    assert "`0.15.1`" in render_kdp_audit_markdown(report)


def test_quality_report_typst_version_none_when_unmeasured(tmp_path: Path) -> None:
    """Missing/empty pin reads as honest unknown, never fabricated."""
    from ubt.core.engine.reporter import render_kdp_audit_markdown

    assert _minimal_report(tmp_path, "unpinned", {}).typst_version is None
    assert _minimal_report(tmp_path, "empty", {"typst_version": ""}).typst_version is None
    md = render_kdp_audit_markdown(_minimal_report(tmp_path, "md", {}))
    assert "unknown" in md


def test_quality_report_surfaces_qe_judge_counters(tmp_path: Path) -> None:
    report = _minimal_report(
        tmp_path,
        "judge_counters",
        {},
        job_metadata={"qe_judge_calls": "11", "qe_judge_errors": "2"},
    )

    assert report.qe_judge_calls == 11
    assert report.qe_judge_errors == 2


def test_quality_report_surfaces_qe_score_source(tmp_path: Path) -> None:
    """The gate's engine label (COMET fallback) is part of the audit trail.

    A fallback batch scores in twelve discrete bands; the report must say so
    instead of presenting them as calibrated CometKiwi measurements.
    """
    report = _minimal_report(
        tmp_path, "qe_fallback", {}, job_metadata={"qe_score_source": "heuristic_fallback"}
    )
    assert report.qe_score_source == "heuristic_fallback"
    assert _minimal_report(tmp_path, "qe_none", {}).qe_score_source is None


def test_kdp_metric_table_survives_the_mtqe_note(tmp_path: Path) -> None:
    """A blockquote glued to the next table row deletes the rest of the table.

    CommonMark lazy continuation folded every row after the MTQE-semantics quote
    into the quote itself, so the GFM table extension never saw them: the
    delivered KDP audit rendered Bottom-15%, Failed Segments, the HITL queue,
    placeholder retention, terminology and the cost line as prose (2026-09
    review). The metrics table is the part a human reviewer reads.
    """
    from ubt.core.engine.reporter import render_kdp_audit_markdown

    lines = render_kdp_audit_markdown(_minimal_report(tmp_path, "table", {})).splitlines()
    assert [
        i for i in range(1, len(lines)) if lines[i].startswith("|") and lines[i - 1].startswith(">")
    ] == []
    for metric in (
        "Bottom 15% Score Average",
        "Failed Segments",
        "HITL Remediation Queue",
        "Estimated Token Cost",
    ):
        assert f"| **{metric}** |" in "\n".join(lines)


def _a0920_ledger_with_targets(tmp_path: Path, *, translated: int, total: int) -> SQLiteJobLedger:
    job_id = "job_coverage"
    ledger = SQLiteJobLedger(tmp_path / f"{job_id}.sqlite")
    blocks = [
        IRBlock(
            id=f"b{idx}",
            spine_index=idx,
            block_type=BlockType.NARRATIVE,
            flow_id=FlowID.MAIN_STORY,
            source_text=f"Source paragraph number {idx}.",
        )
        for idx in range(1, total + 1)
    ]
    seed_job(
        ledger,
        job_id,
        SeedDoc(
            doc_id=job_id,
            source_path=str(tmp_path / "book.md"),
            format_type="markdown",
            blocks=blocks,
        ),
        target_lang="zh",
    )
    for block in blocks[:translated]:
        ledger.save_checkpoint(
            block_id=block.id,
            status=BlockStatus.MTQE_PASSED,
            target_text=f"译文 {block.id}",
        )
    for block in blocks[translated:]:
        ledger.save_checkpoint(block_id=block.id, status=BlockStatus.FAILED)
    return ledger


def test_unpriced_model_reports_unknown_not_zero(tmp_path: Path) -> None:
    """A model missing from the price table must not read as a $0 delivery.

    The price table deliberately covers only the benchmarked models, so most
    operators' models are unpriced (the shipping default is explicitly priced at
    0.0). The pricing layer already returns ``None``; the report layer was
    the one collapsing it to 0.0, in a footnote that admitted "0 = unknown".
    """
    from ubt.core.engine.reporter import build_quality_report, render_kdp_audit_markdown

    ledger = _a0920_ledger_with_targets(tmp_path, translated=2, total=2)
    manifest = BookManifest(
        doc_id="job_coverage",
        title="Cost",
        source_path=str(tmp_path / "book.md"),
        target_lang="zh",
        source_lang="en",
    )

    unpriced = build_quality_report(ledger, "job_coverage", manifest, tmp_path / "out.md")
    assert unpriced.summary.estimated_cost_usd is None
    assert "unknown" in render_kdp_audit_markdown(unpriced)

    priced = build_quality_report(
        ledger, "job_coverage", manifest, tmp_path / "out.md", token_cost_usd=0.5
    )
    assert priced.summary.estimated_cost_usd == 0.5
    assert "$0.50000" in render_kdp_audit_markdown(priced)


def _r0917_clean_report(tmp_path: Path, filename: str, route_mode: str | None) -> Any:
    """A passing report whose manifest records the given route mode."""
    metadata: dict[str, Any] = {}
    if route_mode is not None:
        metadata["route_decision"] = {"mode": route_mode, "pages": 5, "chars": 100}
    ledger = SQLiteJobLedger(tmp_path / f"{filename}.sqlite")
    manifest = _r0917_manifest(**metadata)
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


def test_reported_average_excludes_unscored_placeholders(tmp_path: Path) -> None:
    """A mostly-skip job must not report ~1.0 as if every block was scored.

    499 skips + one 0.30 defect used to average 0.9994 — an almost perfect book
    built from a single defective sample.
    """
    ledger = SQLiteJobLedger(tmp_path / "honest_avg.sqlite")
    manifest = _r0917_manifest()
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


def test_compliance_verdict_does_not_claim_enforcement_that_was_off(tmp_path: Path) -> None:
    """B4: the Aho-Corasick enforcer runs on the SHORT chain only.

    The pipeline sets ``deterministic_glossary_enforce=short_chain``, so a long
    chain job (the default for any book over the short-page cut-off) validates
    without enforcing. The KDP/compliance artifact nevertheless asserted
    "Enforced deterministically via Aho-Corasick glossary enforcer at export" for
    every clean job — claiming a control that was never switched on, in the one
    document a compliance reviewer would read.
    """
    long_md = render_kdp_audit_markdown(_r0917_clean_report(tmp_path, "longjob", "long"))
    assert "Enforced deterministically via Aho-Corasick" not in long_md
    assert "short chain only" in long_md
    assert "long" in long_md  # states the route this job actually took

    # The short chain genuinely does enforce deterministically.
    short_md = render_kdp_audit_markdown(_r0917_clean_report(tmp_path, "shortjob", "short"))
    assert "Enforced deterministically via Aho-Corasick" in short_md

    # An unknown route must not be upgraded into an enforcement claim either.
    unknown_md = render_kdp_audit_markdown(_r0917_clean_report(tmp_path, "noroute", None))
    assert "Enforced deterministically via Aho-Corasick" not in unknown_md


def test_quality_report_surfaces_enforced_spans(tmp_path: Path) -> None:
    """The report measures the enforced text while TM keeps the draft.

    Surface the mechanically-corrected span count so a reader comparing the two
    can discount it (F4).
    """
    report = _minimal_report(tmp_path, "enforced", {}, enforced_spans=7)
    assert report.enforced_spans == 7
    assert report.model_dump()["enforced_spans"] == 7
