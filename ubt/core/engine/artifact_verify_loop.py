"""Artifact-gate control loop and quarantine orchestration.

Runs the deterministic artifact gate (searchability, blank pages, faithful page
count), merges artifact-parity findings, and marks the blocks a major/critical
finding is about as NEEDS_HUMAN. Persists ``artifact_report.json`` to disk and the
SQLite ledger. There is no pixel/VLM tier and no typography-retune pass: a
defect is quarantined directly rather than re-rendered at a different scale.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ubt.core.atomic import atomic_write_text
from ubt.core.engine.artifact_services import run_artifact_gate
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.exceptions import JobInterruptedError
from ubt.core.ir.models import BlockStatus, BookManifest, IRBlock
from ubt.core.job_options import sidecar_path
from ubt.core.log_aggregate import noise_report
from ubt.core.qe.defect_taxonomy import format_defect_flag

logger = logging.getLogger(__name__)


class ArtifactVerifyLoop:
    """Visual-verification control loop: verify the render, quarantine on defect."""

    def __init__(
        self,
        manifest: BookManifest,
        ledger: SQLiteJobLedger,
        job_id: str,
        target_lang: str,
        cancel_token: asyncio.Event | None = None,
        expected_page_count: int | None = None,
        faithful_pages: bool = False,
        claims_finality: bool = True,
    ) -> None:
        self.manifest = manifest
        self.ledger = ledger
        self.job_id = job_id
        self.target_lang = target_lang
        self.cancel_token = cancel_token
        # The contract the artifact was rendered under: a faithful delivery
        # promises the source's page count, so the gate can report a spill
        # instead of leaving it to the reader to notice.
        self.expected_page_count = expected_page_count
        self.faithful_pages = faithful_pages
        # Re-export passes claims_finality=False and must never mutate block statuses.
        self.claims_finality = claims_finality

    async def _evaluate_gate(
        self,
        pdf_path: Path,
        blocks: Sequence[IRBlock],
    ) -> Any:
        return await asyncio.to_thread(
            run_artifact_gate,
            pdf_path,
            expected_page_count=self.expected_page_count,
            faithful_pages=self.faithful_pages,
            target_lang=self.target_lang,
        )

    async def run(
        self,
        rendered_path: Path,
        blocks: list[IRBlock],
    ) -> tuple[Path, Path, Any]:
        """Run post-render visual verification and quarantine defective pages."""
        if self.cancel_token is not None and self.cancel_token.is_set():
            raise JobInterruptedError(f"Job {self.job_id} cancelled before visual gate evaluation")
        gate = await self._evaluate_gate(rendered_path, blocks)

        # If the gate reports major or critical findings, quarantine what each
        # finding is actually ABOUT. A finding that names a block implicates only
        # that block; only a genuinely page-scoped finding (blank page, VLM
        # verdict on the rendered image, overlap) falls back to the whole page.
        #
        # Quarantining the page for a block-level defect marked 13 correct
        # translations (score 0.92, verified content) as NEEDS_HUMAN on the
        # ATC'25 run because two rectangles overran the mediabox by >5pt.
        bad_after = [
            f for f in gate.findings if getattr(f, "severity", "") in ("major", "critical")
        ]
        if bad_after and self.claims_finality:
            # A finding's ``block_id`` is only usable for block-level routing when
            # it names an IR block. The T1 geometry checks default to the
            # ARTIFACT's own pdfium text rects (``p3:rect327``), which have no
            # counterpart in the IR -- so their ids must not be trusted as block
            # keys, or nothing would ever match and the finding would be dropped
            # entirely (worse than over-quarantining).
            ir_ids = {b.id for b in blocks}

            def _finding_block_ids(finding: object) -> set[str]:
                """Every IR block id a finding names.

                A finding may name one block, or several when the defect is a
                *relation* between blocks (``block_overlap`` reports both sides
                comma-separated). Splitting here keeps the "only trust ids that
                resolve to an IR block" rule: an artifact rect id (``p3:rect327``)
                matches nothing and is ignored.
                """
                raw = getattr(finding, "block_id", None)
                if not raw:
                    return set()
                return {part.strip() for part in str(raw).split(",") if part.strip()}

            named_ids = {bid for f in bad_after for bid in _finding_block_ids(f) if bid in ir_ids}
            page_scoped = [
                f
                for f in bad_after
                if not (_finding_block_ids(f) & named_ids) and getattr(f, "page", None) is not None
            ]
            failing_pages = {f.page for f in page_scoped}
            quarantined: list[dict[str, Any]] = []
            preserved = (BlockStatus.FAILED, BlockStatus.BLOCKED_HUMAN)
            for b in blocks:
                if b.skip_translate or b.status in preserved:
                    continue
                page = b.bbox.page if b.bbox is not None else None
                hit_by_id = b.id in named_ids
                hit_by_page = page is not None and page in failing_pages
                if hit_by_id or hit_by_page:
                    b.status = BlockStatus.NEEDS_HUMAN
                    codes = [
                        getattr(f, "code", "defect")
                        for f in bad_after
                        if getattr(f, "block_id", None) == b.id
                        or (
                            getattr(f, "block_id", None) not in named_ids
                            and getattr(f, "page", None) == page
                        )
                    ]
                    # Structured, and ``major`` by declaration: the block is
                    # being quarantined as NEEDS_HUMAN right here, so a table
                    # that guessed "minor" would contradict the status the
                    # same loop just wrote.
                    flag = format_defect_flag(
                        "visual_gate_failed", "major", ",".join(dict.fromkeys(codes))
                    )
                    if flag not in b.error_flags:
                        b.error_flags.append(flag)
                    quarantined.append(
                        {
                            "block_id": b.id,
                            "status": b.status,
                            "error_flags": b.error_flags,
                        }
                    )
            if quarantined:
                await asyncio.to_thread(self.ledger.save_checkpoints_batch, quarantined)
                logger.warning(
                    "ArtifactVerifyLoop: %d block(s) quarantined as NEEDS_HUMAN "
                    "(%d block-scoped, pages %s page-scoped) for job %s",
                    len(quarantined),
                    len(named_ids),
                    sorted(failing_pages),
                    self.job_id,
                )

        # T0.5 artifact parity (physical evidence vs the source file). Runs
        # after the retune/quarantine decisions: its defects (untranslated
        # coverage, replacement characters) are not fixable by typography
        # retuning, and a re-render would burn a full render to change nothing.
        # Critical findings still reach gate.findings, so the blocking gate and
        # the export-time refusal see them.
        parity_findings: list[Any] = []
        source_str = getattr(self.manifest, "source_path", "") or ""
        source_pdf = Path(source_str)
        if source_pdf.exists():
            try:
                from ubt.core.engine.artifact_services import (
                    artifact_parity_findings,
                    probe_unavailable_finding,
                )

                sel_pages = (
                    getattr(getattr(self.manifest, "run", None), "selected_pages", None) or None
                )
                if not sel_pages and self.ledger is not None:
                    sel_pages = (
                        self.ledger.get_job_metadata_value(self.job_id, "selected_pages") or None
                    )
                parity_findings = await asyncio.to_thread(
                    artifact_parity_findings,
                    artifact_pdf=rendered_path,
                    target_lang=self.target_lang,
                    selected_pages=sel_pages,
                    # Latin->Latin targets have no script range to measure; the
                    # parity check compares function words instead, which needs
                    # to know which language the source was written in.
                    source_lang=getattr(self.manifest, "source_lang", None),
                )
            except Exception as exc:
                # The probe must not break export, but a crashed probe is not a
                # clean measurement: without a finding the report reads as a
                # perfect pass for a check that never ran. Record it and keep
                # going.
                logger.warning("artifact parity probe failed for job %s: %s", self.job_id, exc)
                parity_findings = [
                    probe_unavailable_finding(
                        "parity_probe_crashed",
                        f"artifact parity probe raised {type(exc).__name__}: {exc}",
                    )
                ]
        if parity_findings:
            gate = dataclasses.replace(
                gate,
                findings=(*gate.findings, *parity_findings),
                passed=gate.passed
                and not any(f.severity in ("major", "critical") for f in parity_findings),
                stats={
                    **gate.stats,
                    "parity_findings": len(parity_findings),
                },
            )

        report_dict: dict[str, Any] = {
            **gate.report_payload(),
            # Parse-stage third-party warnings the log aggregated instead of
            # printing per cell (see ubt.core.log_aggregate). Empty when the
            # parse emitted none. This is the durable record of *why* a table
            # may be structurally wrong even when the render looks fine.
            "parse_noise": noise_report(),
        }
        artifact_report_path = sidecar_path(rendered_path, "artifact_report.json")
        payload = json.dumps(report_dict, ensure_ascii=False, indent=2)
        # Atomic like the contract/spill/xliff companions: this file sits in
        # the delivery directory and the API serves it verbatim, so a crash
        # mid-write must not leave the frontend truncated JSON.
        await asyncio.to_thread(atomic_write_text, artifact_report_path, payload, encoding="utf-8")

        await asyncio.to_thread(self.ledger.record_artifact_report, self.job_id, report_dict)

        return rendered_path, artifact_report_path, gate
