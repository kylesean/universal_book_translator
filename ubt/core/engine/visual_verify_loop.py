"""Visual gate control loop and quarantine orchestration.

Coordinates post-render visual verification (T0/T1/T2):
1. Runs post-render Visual Gate;
2. If major/critical layout defects are found, marks the affected page blocks
   as NEEDS_HUMAN with visual_gate_failed flags;
3. Persists visual_report.json to disk and SQLite ledger.

The publication renderer has no typography-retune self-healing pass: a
defect is quarantined directly rather than re-rendered at a different scale.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any

from ubt.core.atomic import atomic_write_text
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.exceptions import JobInterruptedError
from ubt.core.ir.models import BlockStatus, BookManifest, IRBlock
from ubt.core.ir.render_plan import RenderPlan
from ubt.core.job_options import sidecar_path
from ubt.core.log_aggregate import noise_report
from ubt.core.ports import DocumentAdapter, get_visual_gate_runner

logger = logging.getLogger(__name__)


class VisualVerifyLoop:
    """Visual-verification control loop: verify the render, quarantine on defect.

    See the module docstring for the coordination flow this class implements.
    """

    def __init__(
        self,
        adapter: DocumentAdapter,
        manifest: BookManifest,
        ledger: SQLiteJobLedger,
        job_id: str,
        target_lang: str,
        router: Any | None = None,
        render_plan: RenderPlan | None = None,
        sample_pages: int = 6,
        max_vlm_pages: int = 3,
        visual_judge_enabled: bool = False,
        visual_judge_model: str | None = None,
        cancel_token: asyncio.Event | None = None,
    ) -> None:
        self.adapter = adapter
        self.manifest = manifest
        self.ledger = ledger
        self.job_id = job_id
        self.target_lang = target_lang
        self.router = router
        # The render decision, so the retune re-render uses the same mode/engine
        # (compiler render plan protocol) and the gate reads the same facing/mode.
        self.render_plan = render_plan
        self.sample_pages = max(0, sample_pages)
        self.max_vlm_pages = max(0, max_vlm_pages)
        self.visual_judge_enabled = visual_judge_enabled
        self.visual_judge_model = visual_judge_model
        self.cancel_token = cancel_token

    def _build_vlm_judge(self) -> Any | None:
        if not self.visual_judge_enabled or self.router is None:
            return None
        router = self.router
        model = self.visual_judge_model

        async def _judge(images: list[str], prompt: str) -> str:
            res: str = await router.complete_with_images(prompt, images, model=model)
            return res

        return _judge

    async def _evaluate_gate(
        self,
        pdf_path: Path,
        blocks: Sequence[IRBlock],
    ) -> Any:
        runner = get_visual_gate_runner()
        typ_text: str | None = None
        typ_sidecar = pdf_path.with_suffix(".typ")
        if typ_sidecar.exists():
            with suppress(Exception):
                typ_text = (
                    await asyncio.to_thread(
                        typ_sidecar.read_text, encoding="utf-8", errors="replace"
                    )
                )[:200_000]

        # T1 (out of bounds) and the overlap check read the artifact's own text
        # boxes, not the IR's source bboxes: the renderer changes where
        # text lands, so the source rectangle says nothing about the delivered
        # page. ``run_visual_gate`` falls back to the IR and says so when pdfium
        # cannot read the artifact.
        gate_blocks = blocks
        vlm_judge = self._build_vlm_judge()
        return await runner(
            pdf_path,
            blocks=gate_blocks,
            typ_text=typ_text,
            sample_pages=self.sample_pages,
            max_vlm_pages=self.max_vlm_pages,
            vlm_judge=vlm_judge,
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
        if bad_after:
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
                    flag = f"visual_gate_failed:{','.join(dict.fromkeys(codes))}"
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
                    "VisualVerifyLoop: %d block(s) quarantined as NEEDS_HUMAN "
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
                from ubt.core.ports import artifact_parity_findings, probe_unavailable_finding

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
        visual_report_path = sidecar_path(rendered_path, "visual_report.json")
        payload = json.dumps(report_dict, ensure_ascii=False, indent=2)
        # Atomic like the contract/spill/xliff companions: this file sits in
        # the delivery directory and the API serves it verbatim, so a crash
        # mid-write must not leave the frontend truncated JSON.
        await asyncio.to_thread(atomic_write_text, visual_report_path, payload, encoding="utf-8")

        await asyncio.to_thread(self.ledger.record_visual_report, self.job_id, report_dict)

        return rendered_path, visual_report_path, gate
