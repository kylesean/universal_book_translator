"""Visual gate reflow control loop and self-healing orchestration.

Coordinates post-render visual verification (T0/T1/T2) and single-pass
self-healing:
1. Runs post-render Visual Gate;
2. If major/critical layout defects are found:
   - Round 1: attempts typography tuning (scale: 92%, leading: 0.75em).
     The rigid engine is excluded: it typesets from source zone geometry
     and returns before the Typst reconstructor, so the retune would spend a
     full re-render without changing a glyph;
3. If defects persist after self-healing:
   - Marks affected page blocks as NEEDS_HUMAN with visual_gate_failed flags;
   - Persists visual_report.json to disk and SQLite ledger.
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


def effective_render_engine(manifest: Any) -> str:
    """Engine the renderer actually used, from the render telemetry.

    ``manifest.metadata["render_engine_effective"]`` is written by the PDF
    renderer for both of its tracks (it is render telemetry, so it lives in
    ``manifest.metadata``, not on ``manifest.run``). Without it the geometry
    predicates treat every render -- including a reflowed publication -- as
    geometry-preserving, which injects bogus page_count/image_count parity
    majors into the visual gate.
    """
    metadata = getattr(manifest, "metadata", None)
    if isinstance(metadata, dict):
        engine = metadata.get("render_engine_effective")
        if engine:
            return str(engine)
    return ""


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
        render_fidelity_enabled: bool = False,
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
        self.render_fidelity_enabled = render_fidelity_enabled
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

    def _typography_retune_possible(self) -> bool:
        """False for the rigid engine, which never reads the retuned knobs.

        ``docling_adapter.render_blocks`` returns to ``RigidTypesetter``
        before the Typst reconstructor whenever ``render_engine_effective`` is
        ``rigid``; rigid sizes come from the source zone's median line
        height, so mutating ``reconstructor.font_size_pt`` / ``leading_em``
        cannot change a single glyph. A retune would only burn a full
        re-render and report a heal that provably cannot happen.
        """
        return effective_render_engine(self.manifest) != "rigid"

    def _output_keeps_source_geometry(self) -> bool:
        """Whether the rendered page still is the source page.

        ``render_engine_effective`` is written by the PDF renderer for both of
        its tracks; anything else (a re-flowed publication render) places text
        by its own layout, so the IR bboxes no longer describe the artifact.
        """
        return effective_render_engine(self.manifest) != "publication"

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
        # bboxes. That is the artifact for the rigid overlay, whose canvas is
        # the original page, and for the alternating zipper, whose pages are the
        # source pages. The reflow engine rebuilds every page at A4 and re-flows
        # blocks across it, so a source rectangle says nothing about where its
        # text landed and must not be fed to the gate as bounds (a narrower A4
        # output would then flag every source-width block as out of bounds).
        gate_blocks = blocks if self._output_keeps_source_geometry() else ()
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

        bad = [f for f in gate.findings if getattr(f, "severity", "") in ("major", "critical")]

        # Attempt self-healing only if we have layout-related visual issues.
        # The rigid engine is never retunable (see
        # _typography_retune_possible), so its defects go straight to the
        # quarantine below instead of a re-render that cannot change them.
        render_fn = self.render_fn
        healing_relevant = bool(
            bad and self.adapter.engine_name is not None and render_fn is not None
        )
        retune_possible = healing_relevant and self._typography_retune_possible()
        if healing_relevant and not retune_possible:
            healing_skipped_reason = "rigid_engine_not_retunable"
            logger.info(
                "ReflowControlLoop: typography self-healing not applicable for job %s "
                "(rigid engine sizes come from source zone geometry); "
                "quarantining failing pages only",
                self.job_id,
            )
        if retune_possible and render_fn is not None:
            if self.cancel_token is not None and self.cancel_token.is_set():
                raise JobInterruptedError(f"Job {self.job_id} cancelled before typography retune")
            recon = getattr(self.adapter, "reconstructor", None)
            if recon is not None:
                orig_font_size = getattr(recon, "font_size_pt", 10.5)
                orig_leading = getattr(recon, "leading_em", 0.85)
                try:
                    # Strategy 1: typography tightening (scale font to 92%, leading to 0.75em)
                    tuned_font = round(orig_font_size * 0.92, 2)
                    tuned_leading = 0.75
                    recon.font_size_pt = tuned_font
                    recon.leading_em = tuned_leading
                    logger.info(
                        "ReflowControlLoop: attempting typography self-healing for job %s "
                        "(font: %.2fpt -> %.2fpt, leading: %.2fem -> %.2fem)",
                        self.job_id,
                        orig_font_size,
                        tuned_font,
                        orig_leading,
                        tuned_leading,
                    )
                    tuned_rendered = await render_fn(
                        adapter=self.adapter,
                        manifest=self.manifest,
                        ledger=self.ledger,
                        blocks=blocks,
                        target_lang=self.target_lang,
                        output_path=rendered_path,
                        job_id=self.job_id,
                        render_plan=self.render_plan,
                    )
                    gate2 = await self._evaluate_gate(tuned_rendered, blocks)
                    if gate2.passed:
                        gate = gate2
                        rendered_path = tuned_rendered
                        self_healed = True
                        healing_strategy = "typography_tuning"
                        logger.info(
                            "ReflowControlLoop: self-healing SUCCEEDED via typography tuning for job %s",
                            self.job_id,
                        )
                    else:
                        # Tuning did not clear the defects; keep the tuned
                        # artifact and let the page quarantine below own the
                        # failing pages.
                        gate = gate2
                        rendered_path = tuned_rendered
                finally:
                    recon.font_size_pt = orig_font_size
                    recon.leading_em = orig_leading

        # If gate still has unresolved major or critical findings, quarantine blocks on failing pages
        bad_after = [
            f for f in gate.findings if getattr(f, "severity", "") in ("major", "critical")
        ]
        if bad_after:
            failing_pages = {f.page for f in bad_after if getattr(f, "page", None) is not None}
            keeps_geometry = self._output_keeps_source_geometry()
            # Under a reflow render the source bbox says nothing about where the
            # text landed, so recover each block's output page from the
            # artifact's own text layer before matching it against the gate's
            # findings (which are always output pages). The geometry-preserving
            # route keeps bbox.page: its canvas *is* the source page.
            output_page_of: dict[str, int] = {}
            if not keeps_geometry:
                try:
                    from ubt.core.ports import extract_output_page_texts, map_blocks_to_output_pages

                    output_page_of = map_blocks_to_output_pages(
                        blocks, extract_output_page_texts(rendered_path)
                    )
                except Exception as exc:  # pragma: no cover - mapping must never break export
                    logger.debug("output-page mapping skipped for job %s: %s", self.job_id, exc)
            quarantined: list[dict[str, Any]] = []
            # Only ever escalate to NEEDS_HUMAN: FAILED and BLOCKED_HUMAN are
            # strictly stronger verdicts and must not be downgraded into a
            # re-review queue. skip_translate blocks are verbatim passthrough
            # (bibliography etc.) and carry no machine translation to quarantine.
            preserved = (BlockStatus.FAILED, BlockStatus.BLOCKED_HUMAN)
            for b in blocks:
                if b.skip_translate or b.status in preserved:
                    continue
                if keeps_geometry:
                    page = b.bbox.page if b.bbox is not None else None
                else:
                    page = output_page_of.get(b.id)
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
                    keeps_source_geometry=self._output_keeps_source_geometry(),
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

        # T0.55 asset-inclusion parity (reflow route only). The renderer records
        # every figure/asset it could not stage; a reflow that drops a content
        # figure still reports full render coverage, so this is the only gate
        # that sees it. Rigid is excluded on purpose: its skip reasons are
        # text-placement decisions (non_prose/spill), and its assets stay on the
        # source canvas — asset loss there is already caught by
        # ``image_count_changed`` above.
        if not self._output_keeps_source_geometry():
            try:
                from ubt.core.ports import asset_skip_findings, get_last_render_skips

                skip_findings = asset_skip_findings(blocks, get_last_render_skips(self.adapter))
            except Exception as exc:  # pragma: no cover - probe must never break export
                skip_findings = []
                logger.debug("asset-skip gate skipped for job %s: %s", self.job_id, exc)
            if skip_findings:
                gate = dataclasses.replace(
                    gate,
                    findings=(*gate.findings, *skip_findings),
                    passed=gate.passed
                    and not any(f.severity in ("major", "critical") for f in skip_findings),
                    stats={**gate.stats, "asset_skip_findings": len(skip_findings)},
                )

        # T0.6 render fidelity (advisory ruler, never a gate). The rigid route
        # promises every non-text region stays pixel-intact, so its residual and
        # painted coverage are measured on every run; the flag is an explicit
        # opt-in for any other engine (a reflow render moves text off its source
        # box, so the mask rectangles would be meaningless). ``gate.passed`` is
        # untouched, so delivery is never blocked by it.
        if source_pdf.exists() and (
            self._output_keeps_source_geometry() or self.render_fidelity_enabled
        ):
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
