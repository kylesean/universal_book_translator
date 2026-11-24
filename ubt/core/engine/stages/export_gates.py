"""Export integrity gates, render skip/flag ledger passes, and delivery contracts."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ubt.core.atomic import atomic_write_text
from ubt.core.engine.adapter_runtime import (
    DocumentAdapter,
    get_last_render_flags,
    get_last_render_skips,
)
from ubt.core.engine.artifact_services import (
    blocking_gate_tripped,
    crashed_artifact_gate_result,
    read_delivery_evidence,
)
from ubt.core.engine.pdf_facts import pdf_page_count
from ubt.core.engine.stage_context import StageContext
from ubt.core.exceptions import (
    BudgetExceededError,
    IntegrityViolationError,
    JobInterruptedError,
    LedgerWriterLeaseLostError,
    UBTError,
)
from ubt.core.ir.models import BlockStatus, BlockType, IRBlock
from ubt.core.job_options import sidecar_path

if TYPE_CHECKING:
    from ubt.adapters.pdf.artifact_gate import ArtifactGateResult
    from ubt.core.content.contract import ReconciliationReport
    from ubt.core.engine.facts import RenderPlan

logger = logging.getLogger(__name__)


def apply_render_skip_flags(
    blocks: list[IRBlock],
    skips: list[tuple[str, str]],
) -> list[dict[str, Any]]:
    """Sync ``render_skip:{reason}`` error flags with this render's skips.

    Maps the adapter's ``(block_id, reason)``
    side channel onto the finalized blocks. Unknown block ids are ignored
    and flags already present are not duplicated. Stale flags from a
    previous render of the same job are dropped first: a render-only rerun
    (same ledger, different engine result) can render a block that an older
    artifact skipped, and a kept flag would make the quality report describe an
    artifact that does not exist. The quality report picks the flags up via
    ``defect_flags`` with no further wiring.
    """
    by_id = {b.id: b for b in blocks}
    wanted_by_block: dict[str, set[str]] = {}
    for block_id, reason in skips:
        wanted_by_block.setdefault(block_id, set()).add(f"render_skip:{reason}")
    for block in blocks:
        keep = wanted_by_block.get(block.id, set())
        block.error_flags = [
            flag
            for flag in block.error_flags
            if not flag.startswith("render_skip:") or flag in keep
        ]
    checkpoints: list[dict[str, Any]] = []
    for block_id, reason in skips:
        fb = by_id.get(block_id)
        if fb is None:
            continue
        flag = f"render_skip:{reason}"
        if flag in fb.error_flags:
            continue
        fb.error_flags.append(flag)
        checkpoints.append(
            {
                "block_id": fb.id,
                "status": fb.status,
                "error_flags": fb.error_flags,
            }
        )
    return checkpoints


#: Quality flags this stage owns: a *drawn* fragment whose fitted size fell
#: below the readable floor, a formula delivered as readable Unicode rather than
#: typeset math, or a block whose text still carried a pipeline sentinel (a
#: restore pass could not account for a span). Plain error flags, deliberately
#: never ``render_skip:`` -- a skip flag would make the contract read the block
#: as source-kept, and the whole point is that the content ships.
RENDER_QUALITY_FLAGS: tuple[str, ...] = (
    "low_legibility_font",
    "formula_degraded",
    "sentinel_leaked",
    "table_source_preserved",
)


def apply_render_quality_flags(
    blocks: list[IRBlock],
    flags: list[tuple[str, str]],
) -> set[str]:
    """Sync render-quality error flags with this render's flags.

    Stale flags from a previous render are dropped
    first, mirroring ``apply_render_skip_flags``: a render-only rerun at a larger
    size must clear a ``low_legibility_font`` that does not describe the current
    artifact. Unknown block ids are ignored and present flags are not
    duplicated.

    Returns the ids of the blocks whose flag set changed. The caller must
    persist them: unlike a skip, a quality flag never changes which blocks the
    reflow gate reports, so nothing else in this stage writes them -- and the
    quality report re-reads the ledger, not these in-memory blocks.
    """
    by_id = {b.id: b for b in blocks}
    wanted_by_block: dict[str, set[str]] = {}
    for block_id, flag in flags:
        if flag in RENDER_QUALITY_FLAGS:
            wanted_by_block.setdefault(block_id, set()).add(flag)
    changed: set[str] = set()
    for block in blocks:
        keep = wanted_by_block.get(block.id, set())
        before = list(block.error_flags)
        after = [flag for flag in before if flag not in RENDER_QUALITY_FLAGS or flag in keep]
        for flag in sorted(keep):
            if flag not in after:
                after.append(flag)
        if after != before:
            block.error_flags = after
            changed.add(block.id)
    for block_id in wanted_by_block:
        if block_id not in by_id:
            logger.debug("render quality flag for unknown block %s ignored", block_id)
    return changed


def _check_completion_ratio(
    actual_job_id: str, final_blocks: list[IRBlock], min_completion_ratio: float
) -> None:
    """Refuse to ship a mostly-untranslated book.

    The renderers fall back to ``source_text`` when a block carries no
    target, so a run in which most blocks failed would finalize as
    "completed" with an untranslated book. Refusing here keeps the ledger
    intact, so a resume retries exactly the blocks that have no target.

    ``BLOCKED_HUMAN`` blocks carry a non-empty ``<mark>`` placeholder wrapping
    the *source*, not a translation, so they must not count as delivered — an
    all-quarantined book would otherwise pass a completion floor of 1.0.

    The floor is measured over *translatable* blocks only. A ``skip_translate``
    block (formula, image, preserved marker) is rendered verbatim by design, so
    leaving it in the denominator let a mostly-skipped book clear the floor
    without a single translated sentence.
    """
    if min_completion_ratio <= 0 or not final_blocks:
        return

    translatable = [fb for fb in final_blocks if not fb.skip_translate]
    if not translatable:
        return

    untranslated = [
        fb.id
        for fb in translatable
        if not (fb.target_text or "")
        or fb.status in (BlockStatus.BLOCKED_HUMAN, BlockStatus.FAILED)
        or any(isinstance(f, str) and f.startswith("render_skip:") for f in fb.error_flags)
    ]
    completed_ratio = 1.0 - len(untranslated) / len(translatable)
    if completed_ratio < min_completion_ratio:
        raise IntegrityViolationError(
            f"Export blocked for job {actual_job_id}: only {completed_ratio:.0%} of "
            f"{len(translatable)} translatable block(s) carry a translation, below the "
            f"{min_completion_ratio:.0%} floor (UBT_EXPORT_MIN_COMPLETION_RATIO). "
            f"{len(untranslated)} block(s) have no target, e.g. "
            f"{', '.join(untranslated[:10])}. Resume the job to retry them, "
            "or lower the floor to ship the partial book knowingly."
        )


def _check_needs_human_gate(
    actual_job_id: str, final_blocks: list[IRBlock], *, allow: bool
) -> None:
    """Refuse to ship NEEDS_HUMAN blocks as finished output by default.

    A ``NEEDS_HUMAN`` block ships its *translation* — it cleared the draft gate
    but triage queued it for human post-editing, so the text in the book is
    unverified. Shipping it with nothing marking it presented machine output as
    finished: the review workbench could see the queue, the delivered PDF could
    not. Blocking here keeps the ledger intact (review, resume and re-export
    still work); an operator who accepts the risk sets
    ``UBT_EXPORT_ALLOW_NEEDS_HUMAN=1`` (CLI ``--allow-needs-human``) and the
    quality report's ``summary.needs_human_block_ids`` then names exactly what
    shipped unverified.
    """
    if allow:
        return
    held = [fb.id for fb in final_blocks if fb.status == BlockStatus.NEEDS_HUMAN]
    if not held:
        return
    raise IntegrityViolationError(
        f"Export blocked for job {actual_job_id}: {len(held)} block(s) are queued "
        f"for human post-editing (NEEDS_HUMAN) and would ship unverified, e.g. "
        f"{', '.join(held[:10])}. Review them in the workbench, then re-export; "
        "or ship knowingly with --allow-needs-human "
        "(UBT_EXPORT_ALLOW_NEEDS_HUMAN=1) — the quality report then lists every "
        "unverified block id."
    )


def _partition_render_skip_counts(
    skip_checkpoints: list[dict[str, Any]],
) -> tuple[int, int]:
    """Count the fail-closed ``render_skip:`` checkpoints.

    The second half of the returned pair is always 0: every ``render_skip:`` is
    fail-closed by definition now, because the intentional families
    (``policy``/``non_prose``/``chrome``/``footer``) never had a producer and
    were removed. The pair survives as the named seam so the call site keeps
    reading as a partition rather than a count with a zero bolted on.
    """
    fail_closed = sum(
        1
        for cp in skip_checkpoints
        if any(
            isinstance(f, str) and f.startswith("render_skip:")
            for f in (cp.get("error_flags") or [])
        )
    )
    return fail_closed, 0


async def _apply_render_skip_ledger_pass(
    ctx: StageContext,
    adapter: DocumentAdapter,
    final_blocks: list[IRBlock],
    *,
    persist: bool = True,
) -> None:
    """Land fail-closed render skips (and length-policy flips) on the ledger.

    Skips collected by the adapter's side channel go on the blocks'
    error_flags BEFORE the quality report is built, so the KDP audit lists
    every source-visible segment. Secondary (complementary) renders are
    excluded: their skips describe the same blocks under a different
    artifact. Flag removal must reach the ledger too: the report re-reads
    blocks from it, and a render-only rerun can render what an older
    artifact skipped — a kept stale flag would make the report describe a
    file that does not exist.

    ``persist=False`` (a re-export) still applies the skip flags **in memory**
    so this artifact's contract and report describe what was actually drawn,
    but writes nothing back and leaves the length-policy status flip alone: a
    re-render must not move a block into or out of NEEDS_HUMAN.
    """
    actual_job_id = ctx.job_id
    ledger = ctx.ledger
    stale_skips = {
        b.id: [f for f in b.error_flags if f.startswith("render_skip:")] for b in final_blocks
    }
    render_skip_checkpoints = apply_render_skip_flags(final_blocks, get_last_render_skips(adapter))
    # A drawn-but-below-floor fragment is a defect, not a skip: the translation
    # ships, so the block must stay deliverable while the shrink is recorded.
    quality_changed_ids = apply_render_quality_flags(final_blocks, get_last_render_flags(adapter))

    # Persist a block whose skip set *changed*, partial removals included: the
    # old filter only wrote when every skip flag was gone, so a block that kept
    # one current reason while losing a stale one stayed wrong in the ledger the
    # quality report re-reads. Snapshot AFTER the length policy pass so flipped
    # statuses are never overwritten by an un-flipped snapshot.
    cleared_skips = [
        {"block_id": b.id, "status": b.status, "error_flags": b.error_flags}
        for b in final_blocks
        if any(f not in b.error_flags for f in stale_skips[b.id])
    ]

    if persist and (render_skip_checkpoints or cleared_skips or quality_changed_ids):
        if render_skip_checkpoints:
            fail_closed_count, preserved_count = _partition_render_skip_counts(
                render_skip_checkpoints
            )
            if fail_closed_count > 0:
                logger.warning(
                    "Job %s rendered with %d fail-closed skipped block(s) left source-visible "
                    "(%d source element(s) intentionally preserved)",
                    actual_job_id,
                    fail_closed_count,
                    preserved_count,
                )
            elif preserved_count > 0:
                logger.info(
                    "Job %s preserved %d source element(s) intact (0 fail-closed skips)",
                    actual_job_id,
                    preserved_count,
                )
        by_block_id = {b.id: b for b in final_blocks}
        # One entry per block, constructed directly from authoritative final_blocks:
        # a block can be in a skip checkpoint, cleared_skips, and the quality set.
        # Building directly from final_blocks ensures no order-dependent overwriting.
        affected_block_ids = {
            c["block_id"] for c in (*render_skip_checkpoints, *cleared_skips)
        } | set(quality_changed_ids)
        batch: dict[str, dict[str, Any]] = {}
        for block_id in sorted(affected_block_ids):
            block = by_block_id.get(block_id)
            if block is not None:
                batch[block_id] = {
                    "block_id": block.id,
                    "status": block.status,
                    "error_flags": block.error_flags,
                }
        await asyncio.to_thread(ledger.save_checkpoints_batch, list(batch.values()))


def _is_faithful_delivery(ctx: StageContext) -> bool:
    """Whether this run's effective delivery mode is the faithful contract."""
    mode = str((getattr(ctx.manifest, "metadata", {}) or {}).get("delivery_mode", "") or "")
    return mode == "faithful"


def _faithful_expected_pages(ctx: StageContext, rendered_path: Path | None = None) -> int | None:
    """Source page count a faithful delivery promised, or None when unknown.

    Only read for faithful runs. A page-count probe failure returns None, which
    the gate treats as "no contract stated" rather than comparing against a
    guess.

    A ``--pages`` run promises the *slice*, not the whole document: a 5-page
    delivery of pages 1-5 is a complete 1:1 lock. Comparing it with the source's
    32 pages reported ``page_count_changed_faithful`` on every sliced run. The
    delivery contract already made this distinction; the artifact gate repeats it
    here, and only when the artifact really holds exactly the requested pages.
    """
    if not _is_faithful_delivery(ctx):
        return None
    try:
        count = pdf_page_count(ctx.input_path)
    except Exception as exc:  # noqa: BLE001 - an unmeasured contract is stated as unmeasured
        logger.debug("Artifact gate: source page count unavailable for %s: %s", ctx.input_path, exc)
        return None
    selected = ctx.config.get_selected_pages()
    if selected and rendered_path is not None and rendered_path.exists():
        try:
            rendered = pdf_page_count(rendered_path)
        except Exception:  # noqa: BLE001 - fall back to the full-source contract
            rendered = None
        if rendered == len(selected) and count != rendered:
            return len(selected)
    return count if count > 0 else None


async def _run_artifact_gate(
    ctx: StageContext,
    adapter: DocumentAdapter,
    final_blocks: list[IRBlock],
    rendered_path: Path,
    render: RenderPlan,
    *,
    claims_finality: bool = True,
) -> tuple[Path, Path | None, Any]:
    """Post-render artifact gate; never fatal except on cancel.

    Runs the deterministic artifact checks + ArtifactVerifyLoop, all driven by
    the config bound on ``ctx``. Returns the (possibly re-rendered) artifact,
    its artifact report path, and the gate object (None when the gate is off or
    not a PDF).
    """
    gate: Any = None
    artifact_report_path: Path | None = None
    if not ctx.config.artifact_gate_enabled or rendered_path.suffix.lower() != ".pdf":
        return rendered_path, artifact_report_path, gate
    try:
        from ubt.core.engine.artifact_verify_loop import ArtifactVerifyLoop

        verify_loop = ArtifactVerifyLoop(
            manifest=ctx.manifest,
            ledger=ctx.ledger,
            job_id=ctx.job_id,
            target_lang=ctx.target_lang,
            cancel_token=ctx.cancel_token,
            expected_page_count=_faithful_expected_pages(ctx, rendered_path),
            faithful_pages=_is_faithful_delivery(ctx),
            claims_finality=claims_finality,
        )
        rendered_path, artifact_report_path, gate = await verify_loop.run(
            rendered_path=rendered_path,
            blocks=final_blocks,
        )
        ctx.check_cancelled()
        bad = [f for f in gate.findings if getattr(f, "severity", "") in ("major", "critical")]
        if bad:
            logger.warning(
                "Artifact gate for job %s: %d warning(s), see %s",
                ctx.job_id,
                len(bad),
                artifact_report_path,
            )
        elif gate.skipped_reason:
            logger.info("Artifact gate for job %s skipped: %s", ctx.job_id, gate.skipped_reason)
    except (JobInterruptedError, LedgerWriterLeaseLostError, BudgetExceededError):
        # Cooperative cancellation, lost writer lease, or hard budget exhaustion
        # must propagate directly to the pipeline handler, not be downgraded into
        # a non-fatal gate crash.
        raise
    except Exception as exc:
        logger.warning("Artifact gate failed (non-fatal) for job %s: %s", ctx.job_id, exc)
        # A crashed gate must read as failed, not absent: blocking enforcement
        # and the KPI collector both key off this object, and None silently
        # skipped every rejection layer while metrics recorded a perfect pass.
        gate = crashed_artifact_gate_result(
            f"artifact gate did not complete: {type(exc).__name__}: {exc}"[:300]
        )
        artifact_report_path = await _persist_crashed_gate_report(ctx, rendered_path, gate)
    return rendered_path, artifact_report_path, gate


async def _persist_crashed_gate_report(
    ctx: StageContext,
    rendered_path: Path,
    gate: ArtifactGateResult,
) -> Path | None:
    """Write the crash gate where a live run would have written its report.

    The crash reason rides in the gate's ``artifact_gate_crashed`` finding, so the
    report needs no separate field for it. The sidecar, the ledger record and
    the KPI collector all read this state; best-effort only, the caller is
    already inside the failure path.
    """
    try:
        from ubt.core.log_aggregate import noise_report

        payload: dict[str, Any] = {
            **gate.report_payload(),
            "parse_noise": noise_report(),
        }
        artifact_report_path = sidecar_path(rendered_path, "artifact_report.json")
        await asyncio.to_thread(
            artifact_report_path.write_text,
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        await asyncio.to_thread(ctx.ledger.record_artifact_report, ctx.job_id, payload)
        return artifact_report_path
    except Exception as persist_exc:
        logger.debug(
            "crashed-gate report persistence failed for job %s: %s", ctx.job_id, persist_exc
        )
        return None


def _enforce_blocking_gate(
    actual_job_id: str,
    gate: Any,
    artifact_report_path: Path | None,
    blocking_enabled: bool,
    *,
    rehearsal: bool = False,
) -> None:
    """Refuse export on blocking visual findings, after persistence.

    Deterministic critical findings follow the existing opt-in/short-doc
    policy. There is no sampled VLM tier any more.

    Runs after the quality report is written so a refusal still leaves the
    ``*_quality_report.json`` / ``*_metrics.json`` audit trail beside the
    deliverable for diagnostic inspection.
    An unreadable rendered PDF refuses here too: the gate reports
    the pdf_page_count -1 sentinel and ``blocking_gate_tripped`` fails
    closed on it even while the opt-in blocking gate is off.

    ``rehearsal=True`` (mock/dry-run provider) downgrades the refusal to a
    warning: a rehearsal echoes source text, so ``target_language_sparse`` is
    guaranteed to fire and every ``--dry-run`` would exit 1 without ever
    proving the render path. Real runs never pass
    this flag.
    """
    if gate is None:
        return
    total_pages = 0
    stats = getattr(gate, "stats", None)
    if isinstance(stats, dict):
        try:
            total_pages = int(stats.get("total_pages", 0))
        except (TypeError, ValueError):
            total_pages = 0
    tripped = blocking_gate_tripped(
        getattr(gate, "findings", []),
        total_pages,
        blocking_enabled,
    )
    if not tripped:
        return
    codes = sorted({str(getattr(f, "code", "defect")) for f in tripped})
    pages = sorted(
        {page for page in (getattr(f, "page", None) for f in tripped) if isinstance(page, int)}
    )
    if rehearsal:
        logger.warning(
            "Rehearsal run: visual blocking gate would have refused job %s "
            "(%d finding(s), codes: %s); downgraded to warning because a mock "
            "draft echoes source text by design.",
            actual_job_id,
            len(tripped),
            ", ".join(codes),
        )
        return
    raise UBTError(
        f"Visual blocking gate refused export for job {actual_job_id}: "
        f"{len(tripped)} blocking finding(s) (severity: {', '.join(sorted({str(getattr(f, 'severity', 'critical')) for f in tripped}))}) "
        f"on page(s) {pages or 'document level'} "
        f"(codes: {', '.join(codes)}). See {artifact_report_path}."
    )


def _deliver_contract(
    ctx: StageContext,
    blocks: list[IRBlock],
    rendered_path: Path,
    reconstructions: list[Any] | None = None,
) -> ReconciliationReport:
    """Build the content graph, reconcile it, and persist the contract.

    Always writes the standalone ``*_contract.json`` and returns the report. The
    report is advisory unless ``config.strict_contract`` is set, in which case
    the caller aborts on an ERROR-severity violation; ``ubt verify`` applies the
    same reconciliation to a delivered artifact or a finished job's ledger.

    ``reconstructions`` is the renderer's per-asset rebuild ledger, so a table
    the renderer rebuilt from its translated grid is recorded RECONSTRUCTED
    (and verified or corrupt) rather than an opaque copy.
    """
    from ubt.core.content.adapt import graph_from_blocks
    from ubt.core.content.contract import reconcile

    manifest = ctx.manifest
    graph = graph_from_blocks(
        blocks,
        doc_id=str(getattr(manifest, "doc_id", "") or ""),
        title=str(getattr(manifest, "title", "") or ""),
        source_path=str(getattr(manifest, "source_path", "") or ""),
        reconstructions=reconstructions,
    )
    allow_spill_warning = getattr(ctx.config, "spill_policy", "error") in (
        "warn",
        "appendix",
        "source_kept_warn",
    )
    contract = reconcile(graph, allow_spill_warning=allow_spill_warning)
    payload = contract.model_dump(mode="json")
    contract_path = sidecar_path(rendered_path, "contract.json")
    try:
        atomic_write_text(
            contract_path, json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except OSError as exc:  # a missing audit file must not sink the artifact
        logger.warning("Could not write delivery contract for %s: %s", ctx.job_id, exc)
        return contract
    if contract.passed:
        logger.info("Delivery contract %s: %s", contract_path.name, contract.summary_line())
    else:
        logger.warning(
            "Delivery contract FAILED for job %s: %s (%d error(s)); see %s",
            ctx.job_id,
            contract.summary_line(),
            len(contract.errors),
            contract_path,
        )
    return contract


def _delivery_contract_evidence(
    ctx: StageContext,
    adapter: DocumentAdapter,
    final_blocks: list[IRBlock],
    rendered_path: Path,
    delivery_contract: dict[str, Any] | None,
    *,
    page_geometry: list[dict[str, Any]] | None = None,
) -> Any:
    """Build artifact evidence for the delivery contract.

    Reads the delivered PDF (text, boxes, page bounds) and the renderer's own
    side channels. The contract is selected from the delivery mode that
    actually produced this file: faithful promises a page-for-page artifact,
    readable does not.
    """
    from ubt.core.engine.delivery_contract import (
        ContractBlock,
        ContractTextBox,
        OverflowRecord,
        StyleSample,
    )
    from ubt.render.delivered_text import delivered_text, dropped_page_furniture, is_toc_entry

    # A snapshot taken before the complementary render: that render replaces
    # the adapter's side channel with the *secondary* artifact's page geometry,
    # and the primary contract must describe the primary file.
    geometry = (
        page_geometry
        if page_geometry is not None
        else (getattr(adapter, "last_page_geometry", None) or [])
    )

    metadata = getattr(ctx.manifest, "metadata", {}) or {}
    actual_mode = str(metadata.get("delivery_mode", "") or "")
    plan_meta = metadata.get("delivery_plan")
    plan_contract = (
        str(plan_meta.get("contract") or "")
        if isinstance(plan_meta, dict) and plan_meta.get("contract") in ("readable", "faithful")
        else ""
    )
    is_native = getattr(adapter, "engine_name", None) == "native"
    if is_native:
        # The native codec patches the source page in place. It can honestly
        # satisfy either promise, so audit the contract the plan asked for --
        # not an assumed faithful one.
        contract = plan_contract or "faithful"
    elif actual_mode == "readable":
        contract = "readable"
    elif plan_contract:
        contract = plan_contract
    else:
        contract = "readable"

    blocks: list[ContractBlock] = []
    # The renderer's flow-expansion declarations, read before the blocks: a
    # page that held the body group's size and flowed onto continuation pages
    # shifts every later block's output page, so faithful's per-page pinning
    # yields to the declared expansion (the page-count check carries it).
    declared_page_expansion = any(bool(pg.get("flow_expanded")) for pg in geometry)
    # Table grids are verified as assets, not as text blocks -- but the style
    # sampler needs their delivered text to recognize a grid's own characters
    # on the page and keep them out of the body-size vote (a grid is its own
    # style group; without the corpus the replay's lmbench page had
    # its 5.3pt grid characters win the body vote and the document read as a
    # 4.18pt body drift).
    selected_pages = ctx.config.get_selected_pages()
    native_units = getattr(adapter, "last_units", None) or ()
    native_unit_by_id = {str(u.unit_id): u for u in native_units} if is_native else {}
    exhibit_texts: list[str] = []
    for block in final_blocks:
        if dropped_page_furniture(block):
            continue
        if is_toc_entry(block) or (block.provenance and block.provenance.toc_entry):
            continue
        page = (
            int(block.bbox.page)
            if block.bbox is not None
            else int(getattr(block.provenance, "source_page", 0) or 0)
        )
        if page <= 0:
            continue
        if selected_pages and page not in selected_pages:
            continue
        if block.block_type is BlockType.TABLE:
            exhibit_text = delivered_text(block).strip()
            if exhibit_text:
                exhibit_texts.append(exhibit_text)
            continue
        if block.skip_translate:
            continue
        if block.block_type in (BlockType.FORMULA, BlockType.IMAGE):
            continue
        text = delivered_text(block).strip()
        segments: tuple[str, ...] = ()
        if is_native and text:
            from ubt.core.validators.inline_tags import strip_inline_tags

            unit_meta = native_unit_by_id.get(block.id)
            if unit_meta is not None:
                rigid_ids = {
                    str(getattr(tok, "tag_id", ""))
                    for tok in getattr(unit_meta, "tokens", ())
                    if getattr(tok, "kind", None) == "rigid" and getattr(tok, "tag_id", None)
                }
                if rigid_ids:
                    rigid_pattern = re.compile(
                        r"<\s*(?:"
                        + "|".join(re.escape(tag_id) for tag_id in rigid_ids)
                        + r")\s*/\s*>"
                    )
                    pieces = tuple(
                        strip_inline_tags(piece).strip()
                        for piece in rigid_pattern.split(text)
                        if strip_inline_tags(piece).strip()
                    )
                    if len(pieces) >= 2:
                        segments = pieces
                for tok in getattr(unit_meta, "tokens", ()):
                    if getattr(tok, "kind", None) == "rigid":
                        raw_glyph = str(tok.payload.get("raw_text", "") or "")
                        if raw_glyph:
                            text = text.replace(f"<{tok.tag_id}/>", raw_glyph)
            text = strip_inline_tags(text).strip()
        if not text:
            continue
        has_multi_page_boxes = False
        if (
            block.provenance
            and block.provenance.physical_boxes
            and len(block.provenance.physical_boxes) > 1
        ):
            box_pages = {
                int(pbox.get("page", 0) if isinstance(pbox, dict) else getattr(pbox, "page", 0))
                for pbox in block.provenance.physical_boxes
            }
            box_pages.discard(0)
            has_multi_page_boxes = len(box_pages) > 1
        blocks.append(
            ContractBlock(
                block_id=block.id,
                text=text,
                page=page,
                kind=str(block.block_type.value),
                # A reflow may move a block to another page; a faithful page
                # must keep it where the source put it, except when the source
                # block itself spans multiple physical pages -- or when a
                # declared flow expansion shifted later content onto
                # continuation pages.
                allow_cross_page=(
                    (contract == "readable")
                    or has_multi_page_boxes
                    or declared_page_expansion
                    # A repeated table header is a cell whose source page is the
                    # table's first page, while the header itself is drawn on
                    # every continuation page. Its text legitimately appears on
                    # another page.
                    or (
                        is_native
                        and getattr(
                            getattr(native_unit_by_id.get(block.id), "role", None),
                            "value",
                            None,
                        )
                        == "table_cell"
                    )
                ),
                segments=segments,
            )
        )

    expected_pages: int | None = None
    if contract == "faithful":
        source = str(getattr(ctx.manifest, "source_path", "") or "")
        if source:
            try:
                src_count = pdf_page_count(Path(source))
                rendered_count = None
                if rendered_path.exists():
                    rendered_count = pdf_page_count(rendered_path)
                selected_pages = ctx.config.get_selected_pages()
                render_meta = metadata.get("render_plan")
                render_bilingual = (
                    str(render_meta.get("bilingual_mode") or "").strip().lower()
                    if isinstance(render_meta, dict)
                    else ""
                )
                render_dual = (
                    str(render_meta.get("effective_dual_mode") or "").strip().lower()
                    if isinstance(render_meta, dict)
                    else ""
                )
                is_interleaved_render = (
                    actual_mode == "interleaved"
                    or render_bilingual in {"bilingual", "dual", "interleaved"}
                    or render_dual == "dual"
                )
                page_factor = 2 if (is_native and is_interleaved_render) else 1
                if (
                    selected_pages
                    and rendered_count == len(selected_pages) * page_factor
                    and src_count != len(selected_pages)
                ):
                    expected_pages = len(selected_pages) * page_factor
                else:
                    expected_pages = src_count * page_factor
            except Exception as exc:  # noqa: BLE001 - a missing count is reported, not fatal
                logger.debug("Delivery contract: source page count unavailable: %s", exc)

    missing_assets: list[str] = []
    if isinstance(delivery_contract, dict):
        for violation in delivery_contract.get("violations", []):
            if str(violation.get("kind", "")) == "asset_missing":
                missing_assets.append(str(violation.get("node_id", "")))

    overflow_records: list[OverflowRecord] = []
    for page_geom in geometry:
        if page_geom.get("fits") in (0, False):
            overflow_records.append(
                OverflowRecord(
                    block_id=f"page:{page_geom.get('page_no')}",
                    page=int(page_geom.get("page_no", 0) or 0),
                    deficit_pt=0.0,
                    reason="page_lock_overflow",
                )
            )
    if is_native:
        for ov in getattr(adapter, "last_overflows", ()) or ():
            overflow_records.append(
                OverflowRecord(
                    block_id=str(getattr(ov, "unit_id", "") or "native_unit"),
                    page=int(getattr(ov, "page_no", 0) or 0),
                    deficit_pt=max(0.0, float(getattr(ov, "overflow_ratio", 1.0)) - 1.0),
                    reason="native_clipped" if getattr(ov, "clipped", False) else "native_overflow",
                )
            )

    style_samples: list[StyleSample] = []
    for page_geom in geometry:
        # A page the solver flagged as a style exception (shrunk past the body
        # group to fit an exhibit) is not a body-group vote: its recorded size
        # is the *exception's* size, and letting it vote reads the document as
        # a body drift that is really one page's overflow -- which the
        # overflow records above already account for.
        if page_geom.get("style_exception"):
            continue
        size = page_geom.get("delivered_size_pt")
        if isinstance(size, (int, float)) and size > 0:
            style_samples.append(
                StyleSample(
                    page=int(page_geom.get("page_no", 0) or 0),
                    size_pt=float(size),
                    group_id=str(page_geom.get("style_group_id") or ""),
                )
            )

    evidence = read_delivery_evidence(
        rendered_path,
        contract=contract,
        blocks=blocks,
        exhibit_texts=exhibit_texts,
        expected_page_count=expected_pages,
        missing_assets=missing_assets,
        overflow_records=overflow_records,
        style_samples=style_samples,
        read_geometry=not is_native,
        declared_page_expansion=declared_page_expansion,
    )
    if is_native:
        from collections import Counter

        native_boxes: list[ContractTextBox] = []
        native_samples_by_page: dict[int, Counter[float]] = {}
        block_text_by_id = {cb.block_id: cb.text for cb in blocks}
        bounds_by_page = dict(getattr(evidence, "page_bounds", {}) or {})
        for unit in native_units:
            upage = int(getattr(unit, "page_no", 0) or 0)
            if upage <= 0 or (selected_pages and upage not in selected_pages):
                continue
            if not getattr(unit, "translate", True):
                continue
            meta = getattr(unit, "codec_meta", None) or {}
            hint = getattr(unit, "box_hint", None) or {}
            ubbox = meta.get("bbox")
            usize = float(hint.get("font_size_pt", 0.0) or 0.0)
            uid = str(getattr(unit, "unit_id", "") or "")
            utext = block_text_by_id.get(uid) or str(getattr(unit, "source_plain", "") or "")
            if isinstance(ubbox, (list, tuple)) and len(ubbox) == 4 and usize > 0.0 and utext:
                x0, y0, x1, y1 = float(ubbox[0]), float(ubbox[1]), float(ubbox[2]), float(ubbox[3])
                pbounds = bounds_by_page.get(upage)
                if pbounds is not None:
                    x0 = max(pbounds[0], x0)
                    y0 = max(pbounds[1], y0)
                    x1 = min(pbounds[2], x1)
                    y1 = min(pbounds[3], y1)
                if x1 > x0 and y1 > y0:
                    native_boxes.append(
                        ContractTextBox(
                            page=upage,
                            x0=x0,
                            y0=y0,
                            x1=x1,
                            y1=y1,
                            font_size_pt=round(usize, 2),
                            block_id=uid or None,
                            text=utext,
                        )
                    )
                role_val = getattr(unit, "role", None)
                if getattr(role_val, "value", role_val) == "paragraph":
                    native_samples_by_page.setdefault(upage, Counter())[round(usize, 2)] += max(
                        1, len(utext)
                    )
        native_samples = [
            StyleSample(page=pno, size_pt=weights.most_common(1)[0][0])
            for pno, weights in sorted(native_samples_by_page.items())
            if weights
        ]
        evidence = dataclasses.replace(
            evidence,
            text_boxes=tuple(native_boxes),
            style_samples=tuple(style_samples or native_samples),
            style_preserves_source=True,
        )

    # The renderer declares the document body group before it draws; the reader
    # above supplies the sizes. Keeping the two apart is what
    # stops a drifted page from excusing itself: membership is the plan's, the
    # number is the artifact's.
    return evidence


def _evaluate_delivery(
    ctx: StageContext,
    adapter: DocumentAdapter,
    final_blocks: list[IRBlock],
    rendered_path: Path,
    delivery_contract: dict[str, Any] | None,
    page_geometry: list[dict[str, Any]] | None = None,
) -> Any | None:
    """Evaluate the artifact contract and write its sidecar.

    The contract reads PDF text/geometry. A Markdown/EPUB/DOCX artifact has no
    artifact text boxes to measure; it is not a failed PDF, it is out of scope.
    Returning ``None`` keeps the caller from turning "not a PDF" into a
    ``page_count`` critical.
    """
    if rendered_path.suffix.lower() != ".pdf":
        return None
    from ubt.core.engine.delivery_contract import evaluate_delivery_contract

    export_mod = sys.modules.get("ubt.core.engine.stages.export")
    evidence_fn = getattr(export_mod, "_delivery_contract_evidence", _delivery_contract_evidence)
    evidence = evidence_fn(
        ctx,
        adapter,
        final_blocks,
        rendered_path,
        delivery_contract,
        page_geometry=page_geometry,
    )
    report = evaluate_delivery_contract(evidence)
    # ``sentinel_leaked`` is written by the renderers when an unrestorable
    # protected span was replaced with ``⟨missing⟩``. The contract deliberately
    # does not treat the marker itself as a leak (the pipeline wrote it on
    # purpose), but the block flag must not be a dead end: surface it as a
    # minor (informational) finding on the delivery record. Minor findings do
    # not change ``passed`` or ``deliverable``; the artifact still ships, and
    # the loss is now visible to every consumer of the contract sidecar.
    sentinel_ids = [
        block.id for block in final_blocks if "sentinel_leaked" in (block.error_flags or [])
    ]
    if sentinel_ids:
        from ubt.core.engine.delivery_contract import ContractFinding

        report = dataclasses.replace(
            report,
            findings=report.findings
            + (
                ContractFinding(
                    severity="minor",
                    code="sentinel_leaked",
                    message=(
                        f"{len(sentinel_ids)} block(s) shipped ⟨missing⟩ because a protected "
                        "span could not be restored; inspect the quality report for the "
                        f"affected block(s), starting with {sentinel_ids[0]}"
                    ),
                    block_id=sentinel_ids[0],
                ),
            ),
        )
    path = sidecar_path(rendered_path, "delivery_contract.json")
    try:
        payload = report.to_payload()
        plan_meta = (getattr(ctx.manifest, "metadata", {}) or {}).get("delivery_plan")
        if isinstance(plan_meta, dict):
            payload["delivery_plan"] = plan_meta
        atomic_write_text(
            path,
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        logger.warning("Could not write delivery contract sidecar: %s", exc)
    allow_major = bool(getattr(ctx.config, "allow_major", False))
    if not (report.passed if allow_major else report.deliverable):
        logger.warning(
            "Delivery contract BLOCKED for job %s: %s; see %s",
            ctx.job_id,
            report.summary_line(),
            path,
        )
    return report
