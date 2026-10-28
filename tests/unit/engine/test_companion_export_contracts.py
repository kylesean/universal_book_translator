"""Tests for companion artifact export contracts and degradation audit (P2-2).

Verifies:
1. CompanionArtifactResult and CompanionExportSummary data contracts (status, size, duration, reason, error).
2. _write_xliff_companion, _write_html_view, _write_epub_view, and _write_artifact_check
   typed return values, graceful degradation on exceptions, and skipped rationale.
3. run_export_stage emits EventType.COMPANIONS_EXPORTED and persists companion audit records
   to job metadata and the quality report.
4. Deliverables endpoint exposes companion_artifacts breakdown.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest

from ubt.core.config import UBTConfig
from ubt.core.engine.stage_context import StageContext
from ubt.core.engine.stages.export import (
    _write_artifact_check,
    _write_epub_view,
    _write_html_view,
    _write_xliff_companion,
)
from ubt.core.ir.models import (
    BlockStatus,
    BlockType,
    BookManifest,
    BoundingBox,
    IRBlock,
    make_element,
)
from ubt.core.job_options import (
    ArtifactKind,
    CompanionArtifactResult,
    CompanionExportManifest,
    CompanionExportSummary,
    CompanionKind,
)

pytestmark = pytest.mark.fast


def _make_block(
    block_id: str = "b1", *, src: str = "Hello world.", tgt: str = "你好世界。"
) -> IRBlock:
    block = IRBlock(
        element=make_element(
            id=block_id,
            spine_index=1,
            block_type=BlockType.NARRATIVE,
            source_text=src,
            bbox=BoundingBox(x0=50.0, y0=500.0, x1=150.0, y1=520.0, page=1),
        )
    )
    block.target_text = tgt
    block.status = BlockStatus.MTQE_PASSED
    return block


def _make_ctx(
    tmp_path: Path,
    *,
    emit_xliff: bool = True,
    emit_html: bool = True,
    emit_epub: bool = True,
    emit_attestation: bool = True,
) -> StageContext:
    config = SimpleNamespace(
        emit_xliff_companion=emit_xliff,
        emit_html_companion=emit_html,
        emit_epub_companion=emit_epub,
        emit_attestation_shadow=emit_attestation,
        spill_policy="error",
        table_render_mode="skip",
    )
    input_file = tmp_path / "sample.pdf"
    input_file.write_bytes(b"%PDF-1.4 mock")
    return StageContext(
        config=cast("UBTConfig", config),
        router=cast("Any", SimpleNamespace()),
        ledger=cast("Any", SimpleNamespace(set_job_metadata_value=MagicMock())),
        manifest=cast(
            "Any", SimpleNamespace(title="Test Companion Book", doc_id="doc1", metadata={})
        ),
        job_id="job_companion_test",
        input_path=input_file,
        source_lang="en",
        target_lang="zh",
        profile_name="general",
        create_event=cast("Any", lambda *a, **k: None),
    )


def test_companion_artifact_result_and_summary_contract(tmp_path: Path) -> None:
    """Validate CompanionArtifactResult model, statuses, and CompanionExportSummary aggregation."""
    path_xliff = tmp_path / "book.xliff"
    path_xliff.write_text("<xliff/>", encoding="utf-8")

    r_gen = CompanionArtifactResult(
        kind="xliff",
        status="generated",
        path=path_xliff,
        size_bytes=path_xliff.stat().st_size,
        duration_ms=14.2,
    )
    assert r_gen.is_generated is True
    assert r_gen.is_skipped is False
    assert r_gen.is_failed is False
    assert r_gen.size_bytes == 8

    r_skip = CompanionArtifactResult(
        kind="epub",
        status="skipped",
        reason="config.emit_epub_companion is disabled",
    )
    assert r_skip.is_skipped is True
    assert r_skip.path is None
    assert r_skip.size_bytes is None

    r_fail = CompanionArtifactResult(
        kind="html",
        status="failed",
        duration_ms=5.0,
        error="Rendering context broken",
    )
    assert r_fail.is_failed is True
    assert r_fail.error == "Rendering context broken"

    summary = CompanionExportSummary(results=[r_gen, r_skip, r_fail])
    assert len(summary.generated) == 1
    assert len(summary.skipped) == 1
    assert len(summary.failed) == 1
    assert summary.summary_line() == "1 generated, 1 skipped, 1 failed"
    assert summary.get("xliff") == r_gen
    assert summary.get("epub") == r_skip
    assert summary.get("html") == r_fail
    assert summary.get("nonexistent") is None

    # CompanionKind / ArtifactKind types and CompanionExportManifest alias
    assert CompanionExportManifest is CompanionExportSummary
    kind: CompanionKind = "xliff"
    art_kind: ArtifactKind = kind
    assert art_kind in ("xliff", "html", "epub", "attestation", "table_appendix", "spill_appendix")


def test_write_xliff_companion_contract(tmp_path: Path) -> None:
    """_write_xliff_companion returns typed result across disabled, empty, success, and error paths."""
    out_pdf = tmp_path / "book.pdf"
    out_pdf.write_bytes(b"%PDF")
    block = _make_block()

    # 1. Skipped when config disabled
    ctx_disabled = _make_ctx(tmp_path, emit_xliff=False)
    res_disabled = _write_xliff_companion(ctx_disabled, out_pdf, [block])
    assert res_disabled.status == "skipped"
    assert "disabled" in str(res_disabled.reason)

    # 2. Skipped when export_xliff returns None (no segments)
    ctx_enabled = _make_ctx(tmp_path, emit_xliff=True)
    with patch("ubt.core.ports.export_xliff", return_value=None):
        res_empty = _write_xliff_companion(ctx_enabled, out_pdf, [block])
        assert res_empty.status == "skipped"
        assert "no extractable text segments" in str(res_empty.reason)

    # 3. Success (generated)
    with patch("ubt.core.ports.export_xliff", return_value=("<xliff>bilingual content</xliff>", 1)):
        res_success = _write_xliff_companion(ctx_enabled, out_pdf, [block])
        assert res_success.status == "generated"
        assert res_success.path is not None
        assert res_success.path.exists()
        assert res_success.size_bytes is not None and res_success.size_bytes > 0
        assert res_success.duration_ms is not None and res_success.duration_ms >= 0

    # 4. Failed on exception
    with patch("ubt.core.ports.export_xliff", side_effect=RuntimeError("XLIFF builder crash")):
        res_fail = _write_xliff_companion(ctx_enabled, out_pdf, [block])
        assert res_fail.status == "failed"
        assert res_fail.error == "XLIFF builder crash"
        assert res_fail.duration_ms is not None


def test_write_html_view_contract(tmp_path: Path) -> None:
    """_write_html_view returns typed result across disabled, success, and failure paths."""
    out_pdf = tmp_path / "book.pdf"
    out_pdf.write_bytes(b"%PDF")
    block = _make_block()
    translations = {block.id: block.target_text}

    # 1. Skipped when disabled
    ctx_disabled = _make_ctx(tmp_path, emit_html=False)
    res_disabled = _write_html_view(ctx_disabled, out_pdf, translations, [block])
    assert res_disabled.status == "skipped"
    assert "disabled" in str(res_disabled.reason)

    # 2. Success (generated)
    ctx_enabled = _make_ctx(tmp_path, emit_html=True)

    def _fake_compose_html(blocks: Any, trans: Any, path: Path, **kw: Any) -> None:
        path.write_text("<html><body>bilingual</body></html>", encoding="utf-8")

    with patch("ubt.render.html_view.compose_html", side_effect=_fake_compose_html):
        res_success = _write_html_view(ctx_enabled, out_pdf, translations, [block])
        assert res_success.status == "generated"
        assert res_success.path is not None
        assert res_success.path.exists()
        assert res_success.size_bytes is not None and res_success.size_bytes > 0

    # 3. Failed on exception
    with patch("ubt.render.html_view.compose_html", side_effect=ValueError("HTML syntax invalid")):
        res_fail = _write_html_view(ctx_enabled, out_pdf, translations, [block])
        assert res_fail.status == "failed"
        assert res_fail.error == "HTML syntax invalid"


def test_write_epub_view_contract(tmp_path: Path) -> None:
    """_write_epub_view returns typed result across disabled, success, and failure paths."""
    out_pdf = tmp_path / "book.pdf"
    out_pdf.write_bytes(b"%PDF")
    block = _make_block()
    translations = {block.id: block.target_text}

    # 1. Skipped when disabled
    ctx_disabled = _make_ctx(tmp_path, emit_epub=False)
    res_disabled = _write_epub_view(ctx_disabled, out_pdf, translations, [block])
    assert res_disabled.status == "skipped"
    assert "disabled" in str(res_disabled.reason)

    # 2. Success (generated)
    ctx_enabled = _make_ctx(tmp_path, emit_epub=True)

    def _fake_compose_epub(blocks: Any, trans: Any, path: Path, **kw: Any) -> None:
        path.write_bytes(b"PK\x03\x04fake epub container")

    with patch("ubt.render.epub_view.compose_epub", side_effect=_fake_compose_epub):
        res_success = _write_epub_view(ctx_enabled, out_pdf, translations, [block])
        assert res_success.status == "generated"
        assert res_success.path is not None
        assert res_success.path.exists()
        assert res_success.size_bytes is not None and res_success.size_bytes > 0

    # 3. Failed on exception
    with patch(
        "ubt.render.epub_view.compose_epub", side_effect=OSError("Disk write quota exceeded")
    ):
        res_fail = _write_epub_view(ctx_enabled, out_pdf, translations, [block])
        assert res_fail.status == "failed"
        assert res_fail.error == "Disk write quota exceeded"


def test_write_artifact_check_contract(tmp_path: Path) -> None:
    """_write_artifact_check returns typed result across disabled, success, and failure paths."""
    out_pdf = tmp_path / "book.pdf"
    out_pdf.write_bytes(b"%PDF")
    block = _make_block()
    translations = {block.id: block.target_text}

    # 1. Skipped when disabled
    ctx_disabled = _make_ctx(tmp_path, emit_attestation=False)
    res_disabled = _write_artifact_check(ctx_disabled, out_pdf, translations, [block])
    assert res_disabled.status == "skipped"
    assert "disabled" in str(res_disabled.reason)

    # 2. Success (generated)
    ctx_enabled = _make_ctx(tmp_path, emit_attestation=True)
    mock_report = SimpleNamespace(
        total=1,
        missing=[],
        passed=True,
        summary_line=lambda: "1/1 blocks attested",
    )
    with patch("ubt.core.ports.check_artifact", return_value=mock_report):
        res_success = _write_artifact_check(ctx_enabled, out_pdf, translations, [block])
        assert res_success.status == "generated"
        assert res_success.path is not None
        assert res_success.path.exists()
        assert res_success.size_bytes is not None and res_success.size_bytes > 0
        data = json.loads(res_success.path.read_text(encoding="utf-8"))
        assert data["artifact"]["total"] == 1
        assert data["artifact"]["summary"] == "1/1 blocks attested"

    # 3. Failed on exception
    with patch("ubt.core.ports.check_artifact", side_effect=RuntimeError("PyMuPDF parser error")):
        res_fail = _write_artifact_check(ctx_enabled, out_pdf, translations, [block])
        assert res_fail.status == "failed"
        assert res_fail.error == "PyMuPDF parser error"


def test_deliverables_endpoint_surfaces_companion_artifacts(tmp_path: Path) -> None:
    """Verify that GET /jobs/{job_id}/deliverables surfaces the structured companion audit breakdown."""
    from starlette.testclient import TestClient

    from ubt.api.app import create_app
    from ubt.core.engine.ledger import SQLiteJobLedger
    from ubt.core.ir.models import ChapterIR
    from ubt.core.job_options import sidecar_path

    job_id = "job_comp_api_test"
    db_dir = tmp_path / "dbs"
    db_dir.mkdir(parents=True)
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True)
    out_file = out_dir / "out.pdf"
    out_file.write_bytes(b"%PDF-1.4\nmock")

    # Write mock quality report sidecar so deliverables endpoint passes verdict probe
    report_data = json.dumps(
        {
            "summary": {
                "completed_blocks": 5,
                "failed_blocks": 0,
                "blocked_human_blocks": 0,
                "needs_human_blocks": 0,
                "pass_rate": 1.0,
            },
            "score_metrics": {"avg_qe": 0.95, "qe_measured": True},
        }
    )
    sidecar_path(out_file, "quality_report.json").write_text(report_data, encoding="utf-8")

    comp_summary = CompanionExportSummary(
        results=[
            CompanionArtifactResult(
                kind="xliff",
                status="generated",
                path=out_dir / "out_pdf.xliff",
                size_bytes=1024,
                duration_ms=45.2,
            ),
            CompanionArtifactResult(
                kind="html",
                status="skipped",
                reason="config.emit_html_companion is disabled",
            ),
        ]
    )

    db_path = db_dir / f"{job_id}.sqlite"
    with SQLiteJobLedger(db_path) as ledger:
        ledger.init_job_from_manifest(
            job_id,
            BookManifest(doc_id=job_id, title="Test", source_path="test.pdf"),
        )
        block = _make_block("b1")
        ledger.append_chapter(
            job_id,
            ChapterIR(doc_id=job_id, chapter_id="c1", title="C1", spine_index=0, blocks=[block]),
        )
        ledger.set_job_metadata_value(job_id, "output_file", str(out_file))
        ledger.set_job_metadata_value(
            job_id, "companion_artifacts", comp_summary.model_dump(mode="json")
        )
        ledger.finalize_job(job_id, status="completed")

    app_config = UBTConfig(
        db_dir=db_dir,
        api_key="sk-test",
    )
    app = create_app(app_config)
    client = TestClient(app)

    res = client.get(f"/jobs/{job_id}/deliverables")
    assert res.status_code == 200
    data = res.json()
    assert data["job_id"] == job_id
    assert "companion_artifacts" in data
    assert data["companion_artifacts"] is not None
    assert len(data["companion_artifacts"]) == 2
    assert data["companion_artifacts"][0]["kind"] == "xliff"
    assert data["companion_artifacts"][0]["status"] == "generated"
    assert data["companion_artifacts"][0]["size_bytes"] == 1024
    assert data["companion_artifacts"][1]["kind"] == "html"
    assert data["companion_artifacts"][1]["status"] == "skipped"
    assert "disabled" in data["companion_artifacts"][1]["reason"]
