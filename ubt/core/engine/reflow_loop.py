"""Visual gate control loop and quarantine orchestration.

Coordinates post-render visual verification (T0/T1/T2):
1. Runs post-render Visual Gate;
2. If major/critical layout defects are found, marks the affected page blocks
   as NEEDS_HUMAN with visual_gate_failed flags;
3. Persists visual_report.json to disk and SQLite ledger.

All PDF targets render through the unified LayerCompositor onto the source
canvas, so there is no typography-retune self-healing pass: a defect is
quarantined directly rather than re-rendered at a different scale.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
from collections.abc import Awaitable, Callable, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any

from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.exceptions import JobInterruptedError
from ubt.core.ir.models import BlockStatus, BookManifest, IRBlock
from ubt.core.ir.render_plan import RenderPlan
from ubt.core.job_options import sidecar_path
from ubt.core.log_aggregate import noise_report
from ubt.core.ports import DocumentAdapter, get_visual_gate_runner

logger = logging.getLogger(__name__)

RenderFn = Callable[..., Awaitable[Path]]


class ReflowControlLoop:
    """Visual gate reflow control loop and self-healing orchestration.

    See the module docstring for the coordination flow this class implements.
    """

    def __init__(
        self,
        adapter: DocumentAdapter,
        manifest: BookManifest,
        ledger: SQLiteJobLedger,
        job_id: str,
        target_lang: str,
        render_fn: RenderFn | None = None,
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
        self.render_fn = render_fn
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

        facing_spread = False
        if self.render_plan is not None:
            facing_spread = bool(
                self.render_plan.facing_spread
                or self.render_plan.bilingual_mode in ("facing", "facing_spread")
            )
        # Pages the alternator filled with intentional blanks. Without this the
        # gate flags the page-count padding as CRITICAL blank_page and flips a
        # correct facing artifact to passed=False.
        padding_pages: tuple[int, ...] = ()
        if isinstance(getattr(self.manifest, "metadata", None), dict):
            padding_pages = tuple(self.manifest.metadata.get("render_padding_pages") or ())
        # T1 (out of bounds) and the overlap check both read the SOURCE page's
        # bboxes, which describe the artifact: every PDF route composes onto the
        # source page canvas, so a block's source rectangle is where its text
        # lands.
        gate_blocks = blocks
        vlm_judge = self._build_vlm_judge()
        return await runner(
            pdf_path,
            blocks=gate_blocks,
            typ_text=typ_text,
            sample_pages=self.sample_pages,
            max_vlm_pages=self.max_vlm_pages,
            vlm_judge=vlm_judge,
            facing_spread=facing_spread,
            padding_pages=padding_pages,
        )

    async def run(
        self,
        rendered_path: Path,
        blocks: list[IRBlock],
    ) -> tuple[Path, Path, Any]:
        """Run post-render visual verification with automated self-healing."""
        if self.cancel_token is not None and self.cancel_token.is_set():
            raise JobInterruptedError(f"Job {self.job_id} cancelled before visual gate evaluation")
        gate = await self._evaluate_gate(rendered_path, blocks)
        self_healed = False
        healing_strategy: str | None = None
        healing_skipped_reason: str | None = None

        # All PDF targets render through LayerCompositor onto the source canvas,
        # so geometry is preserved. If gate has major or critical findings,
        # quarantine blocks on failing pages directly.
        bad_after = [
            f for f in gate.findings if getattr(f, "severity", "") in ("major", "critical")
        ]
        if bad_after:
            failing_pages = {f.page for f in bad_after if getattr(f, "page", None) is not None}
            quarantined: list[dict[str, Any]] = []
            preserved = (BlockStatus.FAILED, BlockStatus.BLOCKED_HUMAN)
            for b in blocks:
                if b.skip_translate or b.status in preserved:
                    continue
                page = b.bbox.page if b.bbox is not None else None
                if page is not None and page in failing_pages:
                    b.status = BlockStatus.NEEDS_HUMAN
                    codes = [
                        getattr(f, "code", "defect")
                        for f in bad_after
                        if getattr(f, "page", None) == page
                    ]
                    flag = f"visual_gate_failed:{','.join(codes)}"
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
                    "ReflowControlLoop: %d block(s) quarantined as NEEDS_HUMAN across pages %s for job %s",
                    len(quarantined),
                    sorted(failing_pages),
                    self.job_id,
                )

        # T0.5 artifact parity (physical evidence vs the source file). Runs
        # after the retune/quarantine decisions: its defects (lost coverage,
        # shifted geometry, changed asset counts) are not fixable by typography
        # retuning, and a re-render would burn a full render to change nothing.
        # Critical findings still reach gate.findings, so the blocking gate and
        # the export-time refusal see them.
        parity_findings: list[Any] = []
        source_str = getattr(self.manifest, "source_path", "") or ""
        source_pdf = Path(source_str)
        if source_pdf.exists():
            try:
                from ubt.core.ports import artifact_parity_findings

                sel_pages = (
                    getattr(getattr(self.manifest, "run", None), "selected_pages", None) or None
                )
                if not sel_pages and self.ledger is not None:
                    sel_pages = (
                        self.ledger.get_job_metadata_value(self.job_id, "selected_pages") or None
                    )
                parity_findings = await asyncio.to_thread(
                    artifact_parity_findings,
                    source_pdf=source_pdf,
                    artifact_pdf=rendered_path,
                    target_lang=self.target_lang,
                    keeps_source_geometry=True,
                    selected_pages=sel_pages,
                )
            except Exception as exc:  # pragma: no cover - probe must never break export
                logger.debug("artifact parity skipped for job %s: %s", self.job_id, exc)
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

        # T0.6 render fidelity (advisory ruler, never a gate). Every PDF route
        # composes onto the source canvas, so the residual and painted coverage
        # are measurable on every run; the flag is an explicit opt-in for any
        # non-source-canvas engine. ``gate.passed`` is untouched, so delivery is
        # never blocked by it.
        if source_pdf.exists():
            try:
                from ubt.core.ports import render_fidelity_findings, render_fidelity_stats

                fidelity_stats = await asyncio.to_thread(
                    render_fidelity_stats, source_pdf, rendered_path, blocks
                )
                if fidelity_stats.get("pages_measured", 0) > 0:
                    fid_findings = render_fidelity_findings(fidelity_stats)
                    gate = dataclasses.replace(
                        gate,
                        findings=(*gate.findings, *fid_findings),
                        stats={
                            **gate.stats,
                            "fidelity_non_text_residual": fidelity_stats.get(
                                "non_text_diff_ratio", 0.0
                            ),
                            "fidelity_painted_coverage": fidelity_stats.get(
                                "masked_coverage_ratio", 0.0
                            ),
                            "fidelity_pages": fidelity_stats.get("pages_measured", 0),
                        },
                    )
                    if isinstance(getattr(self.manifest, "metadata", None), dict):
                        self.manifest.metadata["fidelity"] = fidelity_stats
            except Exception as exc:  # pragma: no cover - probe must never break export
                logger.debug("render fidelity skipped for job %s: %s", self.job_id, exc)

        report_dict: dict[str, Any] = {
            **gate.report_payload(),
            "self_healed": self_healed,
            "healing_strategy": healing_strategy,
            "healing_skipped_reason": healing_skipped_reason,
            # Parse-stage third-party warnings the log aggregated instead of
            # printing per cell (see ubt.core.log_aggregate). Empty when the
            # parse emitted none. This is the durable record of *why* a table
            # may be structurally wrong even when the render looks fine.
            "parse_noise": noise_report(),
        }
        visual_report_path = sidecar_path(rendered_path, "visual_report.json")
        payload = json.dumps(report_dict, ensure_ascii=False, indent=2)
        await asyncio.to_thread(visual_report_path.write_text, payload, encoding="utf-8")

        await asyncio.to_thread(self.ledger.record_visual_report, self.job_id, report_dict)

        return rendered_path, visual_report_path, gate
