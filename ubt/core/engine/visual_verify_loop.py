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
import os
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
from ubt.core.qe.defect_taxonomy import format_defect_flag

logger = logging.getLogger(__name__)


def _typst_source_path(pdf_path: Path) -> Path | None:
    """The Typst source the renderer wrote for ``pdf_path``, or ``None``.

    The publication renderer writes ``<output dir>/.<stem>_typst/main.typ``
    (``docling_render._render_publish_sync``), not a ``.typ`` sibling of the
    PDF. Looking for ``pdf.with_suffix(".typ")`` therefore never found
    anything: the source tier silently never ran, and nothing in the report
    said so. Both spellings are accepted so a caller holding a PDF from some
    other build dir still gets scanned.

    The facsimile renderer writes ``<output dir>/.build_<stem>_facsimile/
    facsimile.typ`` instead, and it was missing from this list for exactly as
    long as the canvas tier failed to run: the gate reads the canvas page map
    out of that file (``visual_gate.canvas_page_pairs``), so without it every
    facsimile delivery reported "no canvas pages recorded" and the ink-collision
    check never measured a single page.

    When several exist the **newest** wins. Order in the tuple is only a
    deterministic tie-break: a resumed job that changed delivery mode leaves the
    previous renderer's build directory behind, and the older source describes a
    document that was never delivered.
    """
    candidates = (
        pdf_path.parent / f".build_{pdf_path.stem}_facsimile" / "facsimile.typ",
        pdf_path.parent / f".{pdf_path.stem}_typst" / "main.typ",
        pdf_path.with_suffix(".typ"),
    )
    # Newest wins when more than one exists. A resumed job that changed delivery
    # mode leaves the previous renderer's build directory behind, and the older
    # source describes a document that was never delivered — reading it would
    # make the gate look for a canvas page map in markup for a different
    # artifact. Order is therefore a tie-break only, not the rule.
    found = [candidate for candidate in candidates if candidate.is_file()]
    if not found:
        return None
    if len(found) == 1:
        return found[0]
    try:
        return max(found, key=lambda candidate: candidate.stat().st_mtime)
    except OSError:
        return found[0]


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
        if not self.visual_judge_enabled:
            return None
        router = self.router
        model = self.visual_judge_model
        if router is not None:
            # A named judge model (or the repair model) on the router: the
            # strongest critic. When only the model name is unset the judge
            # falls through to the OCR vision endpoint below rather than
            # silently skipping the tier.
            async def _judge(images: list[str], prompt: str) -> str:
                res: str = await router.complete_with_images(prompt, images, model=model)
                return res

            return _judge

        # No router model for the judge: reuse the OCR channel's vision
        # endpoint (the same local server that transcribes scans — one vision
        # endpoint serves the whole pipeline). A doc-parsing specialist is a
        # weaker *layout* critic than a general VLM, but it catches the
        # text-level failures (missing/garbled delivered text) locally at
        # zero cost and zero egress; aesthetic judgement still wants a named
        # ``visual_judge_model`` on the router.
        endpoint = getattr(self.adapter, "ocr_endpoint", None) or os.environ.get(
            "UBT_OCR_ENDPOINT", ""
        )
        model_name = getattr(self.adapter, "ocr_model", None) or os.environ.get("UBT_OCR_MODEL", "")
        if not endpoint or not model_name:
            return None
        # Same choke point as the OCR drivers: without the opt-in, only a
        # local endpoint may receive page images — including from the judge.
        from ubt.core.endpoint_locality import endpoint_on_this_machine

        allow_upload = bool(getattr(self.adapter, "allow_page_upload", False))
        if not allow_upload and not endpoint_on_this_machine(endpoint):
            return None
        api_key = getattr(self.adapter, "ocr_api_key", None) or os.environ.get("UBT_OCR_API_KEY")
        from ubt.core.ports import vision_ocr_driver

        driver = vision_ocr_driver(endpoint, api_key, model_name)

        def _post_judge(images: list[str], prompt: str) -> str:
            """One bounded call per image; returns raw text for verdict parsing."""
            import httpx

            contents = []
            for img in images:
                # Images arrive as base64 data URLs or raw base64; pass through.
                if img.startswith("data:"):
                    contents.append({"type": "image_url", "image_url": {"url": img}})
                else:
                    contents.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img}"},
                        }
                    )
            payload = {
                "model": driver.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}, *contents],
                    }
                ],
                "temperature": 0.0,
            }
            headers = {"Content-Type": "application/json"}
            if driver.api_key:
                headers["Authorization"] = f"Bearer {driver.api_key}"
            url = (
                driver.endpoint
                if driver.endpoint.endswith("/chat/completions")
                else f"{driver.endpoint}/chat/completions"
            )
            with httpx.Client(timeout=driver.timeout) as client:
                resp = client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
            data = resp.json()
            # Core-internal usage sink: the judge's tokens must land in the
            # same ledger the OCR channel's do (unmeasured books as unknown,
            # never fake $0.00).
            from ubt.core.router.provider import _extract_cached_tokens, record_external_usage

            usage = data.get("usage")
            if isinstance(usage, dict) and usage:
                record_external_usage(
                    driver.model,
                    prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
                    completion_tokens=int(usage.get("completion_tokens", 0) or 0),
                    cached_tokens=_extract_cached_tokens(usage),
                )
            else:
                record_external_usage(driver.model, measured=False)
            return data["choices"][0]["message"]["content"] or ""

        async def _local_judge(images: list[str], prompt: str) -> str:
            # The gate awaits its judge; the blocking HTTP call goes to a
            # thread so the export event loop keeps pumping.
            return await asyncio.to_thread(_post_judge, images, prompt)

        return _local_judge

    async def _evaluate_gate(
        self,
        pdf_path: Path,
        blocks: Sequence[IRBlock],
    ) -> Any:
        runner = get_visual_gate_runner()
        typ_text: str | None = None
        typ_sidecar = _typst_source_path(pdf_path)
        if typ_sidecar is not None:
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
            canvas_background_pdf=self._canvas_background_path(),
        )

    def _canvas_background_path(self) -> Path | None:
        """The stripped background the facsimile renderer drew under the text.

        Only the facsimile path produces one, and it records the path on the
        manifest (``_render_facsimile``) because this loop runs a stage later and
        holds no reference to the render call. ``None`` is the ordinary answer —
        reflow and page_locked deliveries have no background — and the gate turns
        it into a printed "the canvas tier did not run" note rather than a
        silent omission.

        A recorded path that no longer exists is also ``None``: the gate must not
        crash on a stale metadata entry left by an earlier run of the same job,
        and reporting "measured clean" for a file it could not open would be
        worse than reporting nothing.
        """
        metadata = getattr(self.manifest, "metadata", None)
        if not isinstance(metadata, dict):
            return None
        recorded = metadata.get("background_pdf")
        if not recorded:
            return None
        candidate = Path(str(recorded))
        if candidate.is_file():
            return candidate
        logger.debug(
            "Visual gate: recorded background %s is gone; canvas tier will be skipped",
            candidate,
        )
        return None

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
