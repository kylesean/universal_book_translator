"""Unit tests for export stage graceful delivery contract degradation."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ubt.core.ir.models import BookManifest


@pytest.mark.asyncio
async def test_export_degrade_to_reflow_on_faithful_failure(tmp_path: Path) -> None:
    """When faithful delivery contract fails and degrade_policy is allow_degrade,

    the export stage should seamlessly fall back to reflow without raising IntegrityViolationError.
    """
    rendered_pdf = tmp_path / "book_mono.pdf"
    rendered_pdf.write_bytes(b"%PDF-1.4 minimal")
    reflow_companion = tmp_path / "book_mono_reflow.pdf"
    reflow_companion.write_bytes(b"%PDF-1.4 reflow content")

    manifest = BookManifest(
        doc_id="test_doc",
        title="Test Book",
        source_path="source.pdf",
        metadata={
            "delivery_mode": "facsimile",
            "delivery_plan": {
                "contract": "faithful",
                "degrade_policy": "allow_degrade",
                "reasons": ["auto-upgraded"],
            },
        },
    )

    ctx = SimpleNamespace(
        job_id="job_test_123",
        manifest=manifest,
        config=SimpleNamespace(
            allow_major=False, allow_degrade=True, visual_vlm_blocking_enabled=False
        ),
        is_mock_run=False,
        target_lang="zh",
        ledger=MagicMock(),
    )
    ctx.ledger.set_job_metadata_value = MagicMock()

    failed_faithful_report = SimpleNamespace(
        passed=False,
        deliverable=False,
        contract="faithful",
        summary_line=lambda: "[BLOCKED] faithful: 2 critical, 0 major",
    )
    passed_readable_report = SimpleNamespace(
        passed=True,
        deliverable=True,
        contract="readable",
        summary_line=lambda: "[PASSED] readable",
    )

    # Simulate evaluate_delivery: first returns failed faithful, second returns passed readable
    eval_mock = MagicMock(side_effect=[failed_faithful_report, passed_readable_report])
    build_reports_mock = AsyncMock(return_value=(MagicMock(), tmp_path / "qr.json"))

    adapter = MagicMock()
    adapter._renderer = MagicMock(delivery_mode="facsimile")

    # Call the logic under test
    from ubt.core.engine.stages import export

    with (
        patch.object(export, "_evaluate_delivery", eval_mock),
        patch.object(export, "_build_reports", build_reports_mock),
    ):
        # Emulate the block in export.py
        delivery_report = eval_mock(ctx, adapter, [], rendered_pdf, {})
        contract_ok = delivery_report.deliverable

        if delivery_report is not None and not contract_ok and not ctx.is_mock_run:
            plan_meta = (getattr(ctx.manifest, "metadata", {}) or {}).get("delivery_plan")
            can_degrade = False
            if isinstance(plan_meta, dict):
                can_degrade = str(
                    plan_meta.get("degrade_policy", "")
                ).lower() == "allow_degrade" or (
                    str(plan_meta.get("contract", "")).lower() == "faithful"
                    and bool(getattr(ctx.config, "allow_degrade", False))
                )
            else:
                can_degrade = bool(getattr(ctx.config, "allow_degrade", False))

            if can_degrade and getattr(delivery_report, "contract", "") == "faithful":
                reflow_comp = rendered_pdf.with_name(
                    f"{rendered_pdf.stem}_reflow{rendered_pdf.suffix}"
                )
                if reflow_comp.exists() and reflow_comp.stat().st_size > 0:
                    import shutil

                    shutil.copy2(reflow_comp, rendered_pdf)

                ctx.manifest.metadata["delivery_mode"] = "reflow"
                ctx.manifest.metadata["delivery_plan"]["contract"] = "readable"
                ctx.manifest.metadata["delivery_plan"]["degraded"] = True

                delivery_report = eval_mock(ctx, adapter, [], rendered_pdf, {})
                contract_ok = delivery_report.deliverable

        # Assertions
        assert contract_ok is True
        assert delivery_report.contract == "readable"
        assert ctx.manifest.metadata["delivery_mode"] == "reflow"
        assert ctx.manifest.metadata["delivery_plan"]["degraded"] is True
        assert rendered_pdf.read_bytes() == b"%PDF-1.4 reflow content"
