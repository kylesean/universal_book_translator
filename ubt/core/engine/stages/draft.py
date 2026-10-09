"""Streaming draft translation with keyset pagination."""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import random
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ubt.core import ports
from ubt.core.config import UBTConfig
from ubt.core.engine.events import EventType, TranslationProgressEvent
from ubt.core.engine.facts import Terminology
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.engine.ledger_flusher import CheckpointBatchFlusher
from ubt.core.engine.services import RunServices
from ubt.core.engine.stage_context import StageContext
from ubt.core.exceptions import BudgetExceededError, JobInterruptedError, UBTError
from ubt.core.ir.emphasis import mark_bold_spans, parse_bold, runs_from_json
from ubt.core.ir.models import BlockStatus, BlockType, BookManifest, IRBlock, StyleMeta
from ubt.core.memory.abbreviation_miner import format_abbreviations_markdown_table
from ubt.core.memory.glossary_table import (
    build_chunk_glossary_table,
    build_global_glossary_table,
)
from ubt.core.memory.hierarchical_memory import HierarchicalMemoryManager
from ubt.core.memory.rolling_summary import extract_chapter_id
from ubt.core.memory.tm import (
    PROMPT_VERSION,
    TranslationMemory,
    compute_tm_context,
    format_few_shot_reference,
)
from ubt.core.qe.comet_runner import glossary_violation_flag
from ubt.core.qe.defect_taxonomy import DRAFTING_ERROR_PREFIX, NON_RETRYABLE_DRAFT_PREFIX
from ubt.core.qe.fast_pass import FastPassFilter
from ubt.core.router.router import (
    BatchDraftRequest,
    BatchTranslationError,
    ModelRouter,
    classify_provider_error,
)
from ubt.core.validators.consistency import GlossaryConsistencyValidator

if TYPE_CHECKING:
    from ubt.translate.engine import TranslationEngine

logger = logging.getLogger(__name__)

ACADEMIC_PROFILES = frozenset(
    {"academic", "textbook", "paper", "technical", "nonfiction", "non-fiction"}
)
_SKIPPED_TYPES = frozenset({BlockType.IMAGE, BlockType.FORMULA})


def _source_with_bold_markers(block: IRBlock) -> str:
    """The source text with ``⟦B⟧…⟦/B⟧`` around each source-side bold span.

    Whole-block bold is already rendered from ``provenance["is_bold"]``; only a
    *partial* bold block needs the model to carry the emphasis into the target,
    because a source-derived inline run the translation rewrote has no target
    substring for the render to locate. The markers ride the masked source and
    are parsed back out in :meth:`_DraftProcessor.finalize_draft`.
    """
    style = block.style
    if style is None or block.provenance.get("is_bold"):
        return block.source_text
    spans = [run.text for run in style.inline_runs if run.bold and run.text.strip()]
    if not spans:
        return block.source_text
    return mark_bold_spans(block.source_text, spans)


@dataclass
class _DraftInputs:
    """Per-block prompt inputs shared by interactive and batch paths."""

    masked_source: str
    code_map: Any
    cite_map: Any
    math_map: Any
    glossary_table: str
    neighbor_ctx: str
    macro_ctx: str
    few_shot_reference: str
    epoch_ctx: str = ""
    soup_map: Any = None
    email_map: Any = None
    #: The block's ``BlockType`` value. Travels into the prompt builder so a
    #: heading gets the "do not expand" clause, and into the cache key so a
    #: heading and a prose block with identical source text do not collide.
    block_type: str = ""


def resolve_draft_policy(
    manifest: BookManifest,
    profile_name: str,
    config: UBTConfig,
    all_blocks_count: int,
) -> tuple[bool, bool, int]:
    """Public pure helper: (rolling_enabled, is_fast_path, batch_limit)."""
    is_fast_path = len(manifest.chapters) <= 1 and all_blocks_count <= 15
    is_page_slice = bool(manifest.metadata.get("is_page_slice_epub", False))
    is_academic = profile_name.lower() in ACADEMIC_PROFILES
    config_rolling = bool(config.enable_rolling_summary)
    rolling_enabled = not (
        len(manifest.chapters) <= 1
        or not config_rolling
        or is_page_slice
        or len(manifest.chapters) > 40
        or is_academic
    )
    batch_limit = max(all_blocks_count, 15) if is_fast_path else int(config.batch_limit)
    logger.info(
        "Context policy resolved: rolling_summary=%s (is_page_slice=%s, chapters=%d, profile=%s, fast_path=%s)",
        rolling_enabled,
        is_page_slice,
        len(manifest.chapters),
        profile_name,
        is_fast_path,
    )
    return rolling_enabled, is_fast_path, batch_limit


def segment_by_chapter(batch: list[IRBlock]) -> list[list[IRBlock]]:
    """Split a claimed batch into contiguous chapter segments (pure)."""
    segments: list[list[IRBlock]] = []
    for b in batch:
        if segments and extract_chapter_id(b.id) == extract_chapter_id(segments[-1][0].id):
            segments[-1].append(b)
        else:
            segments.append([b])
    return segments


def is_static_skip(block: IRBlock) -> bool:
    """Blocks that never reach the LLM (images / formulas / explicit skip)."""
    return bool(block.skip_translate) or block.block_type in _SKIPPED_TYPES


def is_already_final(block: IRBlock) -> bool:
    """Batch-path early exit for blocks finalized by a concurrent stage."""
    return block.status in (BlockStatus.MTQE_PASSED, BlockStatus.REPAIRED)


#: A blank-line paragraph break, the shape a heading's appended "explanation"
#: or abstract summary takes (``title\n\n<pages of prose>``).
_HEADING_PARAGRAPH_RE = re.compile(r"\n\s*\n")


def trim_heading_expansion(source_text: str, draft: str) -> str:
    """Drop the paragraphs a model appended after a heading's translation.

    A small model told to translate a title sometimes keeps writing -- an
    explanation of the title, or a summary of the abstract -- and the QE length
    gate then quarantines the block, which falls back to the source
    (``pdf_main#b0003``: a 96-char title drafted as a title plus an
    abstract-summary paragraph, ratio 3.35). A heading is one line, so when the
    source carries no line break and the draft does, the leading paragraph is
    the translation and the rest is fabrication. A source that already breaks
    (a genuine multi-line heading) is left alone, and a single-paragraph draft
    is returned untouched.
    """
    if not draft or "\n" in source_text.strip():
        return draft
    first = _HEADING_PARAGRAPH_RE.split(draft, maxsplit=1)[0].strip()
    return first or draft


def _decode_chunk(raw: str | None) -> dict[str, str]:
    """The block-id -> text map from a cached macro-chunk value (fail-closed)."""
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(key): str(value) for key, value in data.items()}


def _encode_chunk(extracted: dict[str, str]) -> str:
    """A macro-chunk result as cache text (JSON, so a file stays inspectable)."""
    return json.dumps(extracted, ensure_ascii=False)


@dataclass(slots=True)
class DraftRuntime:
    """Per-run mutable mechanisms: ledger/router/adapter handles, buffers, caches.

    Everything here is either rebound during the run or holds mutable state.
    """

    ledger: SQLiteJobLedger
    actual_job_id: str
    router: ModelRouter
    engine: TranslationEngine
    memory_mgr: HierarchicalMemoryManager
    concurrency_sem: asyncio.Semaphore
    tm: TranslationMemory | None
    active_fast_pass: FastPassFilter
    counters: dict[str, int]
    flusher: CheckpointBatchFlusher
    batch_active: bool
    # Job-level fail-fast circuit. A non-retryable provider error
    # (401/402/400 — bad credential, no credit, malformed) recurs on every
    # block, so without a breaker a whole book's worth of blocks each burns the
    # router's full retry chain before the job dies. Consecutive non-retryable
    # failures trip the breaker; any drafted block resets it.
    fail_fast_consecutive: int = 0
    last_fail_fast_reason: str = ""
    ctx: StageContext | None = None
    create_event_fn: Any = None


@dataclass(frozen=True, slots=True)
class DraftPolicy:
    """Assembly-time parameters fixed for the whole run (never reassigned)."""

    glossary_dicts: list[dict[str, Any]]
    abbreviation_entries: list[dict[str, Any]]
    global_glossary_table: str
    source_lang: str
    target_lang: str
    profile_name: str
    domain: str | None
    tm_fuzzy_threshold: float
    tm_context: str
    draft_max_retries: int
    draft_retry_base_delay: float
    batch_min_blocks: int
    batch_poll_interval: float
    batch_poll_timeout: float
    rolling_enabled: bool
    fail_fast_threshold: int = 6
    batch_delete_files: bool = True
    macro_chunk_size: int = 5


@dataclass
class _DraftProcessor:
    """Stateful per-run draft worker composed of runtime mechanisms + policy."""

    runtime: DraftRuntime
    policy: DraftPolicy

    @functools.cached_property
    def glossary_validator(self) -> GlossaryConsistencyValidator | None:
        """Built once per run: the constructor sorts the whole glossary."""
        if not self.policy.glossary_dicts:
            return None
        return GlossaryConsistencyValidator(glossary=self.policy.glossary_dicts)

    def chunk_context(self, chunk: list[tuple[IRBlock, _DraftInputs]]) -> str:
        """The content key of one macro-chunk draft prompt (content-addressed cache layer).

        A macro chunk is a *different* prompt from any single block's -- all the
        chunk's blocks ride one XML request -- so it keys separately. The merged
        glossary / first neighbour / few-shot / rolling-epoch inputs here are
        exactly what :meth:`draft_macro_chunk_group` sends, so the key and the
        call cannot drift.
        """
        from ubt.cache.store import step_key
        from ubt.core.router.prompts import build_macro_chunk_draft_prompt

        merged_glossary = "\n\n".join(
            filter(None, dict.fromkeys(inp.glossary_table.strip() for _, inp in chunk))
        )
        first_inp = chunk[0][1]
        last_inp = chunk[-1][1]
        system, user = build_macro_chunk_draft_prompt(
            blocks=[(b.id, inp.masked_source) for b, inp in chunk],
            glossary_table=merged_glossary,
            neighbor_context=first_inp.neighbor_ctx,
            target_lang=self.policy.target_lang,
            source_lang=self.policy.source_lang,
            genre_profile=self.policy.profile_name,
            rolling_summary=first_inp.macro_ctx,
            global_glossary=self.policy.global_glossary_table,
            few_shot_reference=first_inp.few_shot_reference or last_inp.few_shot_reference,
            epoch_summary=first_inp.epoch_ctx,
            domain=self.policy.domain,
        )
        return step_key(
            "macro_chunk_prompt",
            [
                system,
                user,
                self.policy.source_lang,
                self.policy.target_lang,
                self.policy.profile_name,
                self.policy.domain or "",
            ],
            {},
        )

    def prompt_context(self, inputs: _DraftInputs) -> str:
        """The content key of one draft prompt (content-addressed cache layer).

        The production generate step is *not* a pure function of the masked
        source: the same source drafted under a different glossary, neighbour
        window, rolling/epoch summary or few-shot reference is a different
        prompt, so the translate cache must key on the prompt itself. Building
        the exact prompt the router will send (the router's own builder, so the
        two cannot drift) and hashing it makes a changed prompt a miss rather
        than a stale draft.
        """
        from ubt.cache.store import step_key

        system, user = self.runtime.router.build_draft_prompt(
            source_text=inputs.masked_source,
            glossary_table=inputs.glossary_table,
            neighbor_context=inputs.neighbor_ctx,
            target_lang=self.policy.target_lang,
            source_lang=self.policy.source_lang,
            genre_profile=self.policy.profile_name,
            rolling_summary=inputs.macro_ctx,
            global_glossary=self.policy.global_glossary_table,
            few_shot_reference=inputs.few_shot_reference,
            epoch_summary=inputs.epoch_ctx,
            model=self.runtime.router.draft_model,
            domain=self.policy.domain,
            block_type=inputs.block_type or None,
        )
        # The prompt already carries the languages/profile/domain/block-type,
        # but fold them in explicitly too: the key must separate two runs that
        # differ on any output-bearing axis even if a future prompt builder
        # stopped embedding one of them.
        return step_key(
            "draft_prompt",
            [
                system,
                user,
                self.policy.source_lang,
                self.policy.target_lang,
                self.policy.profile_name,
                self.policy.domain or "",
                inputs.block_type,
            ],
            {},
        )

    async def prepare_draft_inputs(
        self,
        block: IRBlock,
        current_batch: list[IRBlock],
    ) -> _DraftInputs | None:
        """Shared preparation for one block's draft.

        Returns ``None`` when the block was already checkpointed from
        an exact TM hit (funnel tier 1) or an MT-tier draft.
        """
        # A partial-bold block needs the model's emphasis markers to reach the
        # target. A TM entry written before emphasis runs existed carries none,
        # so such a hit is refused (below) and the block is re-drafted; a hit
        # that *does* carry runs restores the bold without a model call.
        needs_emphasis = _source_with_bold_markers(block) != block.source_text
        if self.runtime.tm is not None:
            exact_hit = await asyncio.to_thread(
                self.runtime.tm.lookup_exact,
                self.policy.source_lang,
                self.policy.target_lang,
                block.source_text,
                self.policy.tm_context,
                # domain must stay None in production: passing profile_name
                # lets the same-domain pass override context_hash, so a
                # glossary/prompt change would not invalidate exact hits.
                None,
            )
            if exact_hit is not None and needs_emphasis and not exact_hit.runs_json:
                logger.info(
                    "TM exact hit for block %s predates emphasis runs; re-drafting "
                    "to preserve bold",
                    block.id,
                )
                exact_hit = None
            if exact_hit is not None:
                fp_decision = await asyncio.to_thread(
                    self.runtime.active_fast_pass.evaluate,
                    block.source_text,
                    exact_hit.target_text,
                    # Without the type every check is judged as prose: a cached
                    # table echo would be exempt from the grid checks that would
                    # have caught it, and a cached heading would face the prose
                    # script-density bar it was never meant to clear.
                    block_type=block.block_type,
                )
                if not fp_decision.passed:
                    logger.warning(
                        "TM exact hit for block %s rejected by FastPass "
                        "(%s); falling through to LLM draft",
                        block.id,
                        fp_decision.reason,
                    )
                else:
                    # The enforced-term check applies to TM-carried text too:
                    # an exact hit from an older glossary era (or a human_pe
                    # row) must meet the same check drafted text faces, not be
                    # stamped 1.0/MTQE_PASSED unseen. A violation falls through
                    # to the LLM instead.
                    violation = None
                    if self.glossary_validator is not None:
                        # Off the loop, like the FastPass check above: this walks
                        # the whole glossary against the block text.
                        violation = await asyncio.to_thread(
                            glossary_violation_flag,
                            block.source_text,
                            exact_hit.target_text,
                            self.glossary_validator,
                        )
                    if violation is not None:
                        logger.warning(
                            "TM exact hit for block %s violates the enforced glossary "
                            "(%s); falling through to LLM draft",
                            block.id,
                            violation,
                        )
                    else:
                        # Restore the emphasis the entry stored, so a TM hit keeps
                        # the bold without a model call.
                        tm_runs = runs_from_json(exact_hit.runs_json)
                        style = (
                            (block.style or StyleMeta()).model_copy(update={"target_runs": tm_runs})
                            if tm_runs or block.style is not None
                            else None
                        )
                        update = {
                            "block_id": block.id,
                            "target_text": exact_hit.target_text,
                            "draft_text": exact_hit.target_text,
                            "status": BlockStatus.MTQE_PASSED,
                            "mtqe_score": 1.0,
                            "tm_hit": True,
                            "style": style,
                        }
                        # Buffering accepts the write: a failing flush is
                        # retried and finally raised at the next enqueue /
                        # close, so the count may not precede the ledger.
                        await self.runtime.flusher.enqueue(update)
                        saved = True
                        if saved:
                            self.runtime.counters["tm_exact_hits"] += 1
                        return None

        few_shot_reference = ""
        if self.runtime.tm is not None:
            fuzzy_hit = await asyncio.to_thread(
                self.runtime.tm.lookup_fuzzy,
                self.policy.source_lang,
                self.policy.target_lang,
                block.source_text,
                self.policy.tm_fuzzy_threshold,
                self.policy.tm_context,
                None,
            )
            if fuzzy_hit is not None:
                # The formatter drops the reference when the pair diverges
                # on a polarity cue, so an inverted neighbour is never shown.
                few_shot_reference = format_few_shot_reference(fuzzy_hit, block.source_text)

        glossary_table = build_chunk_glossary_table(
            self.policy.glossary_dicts, self.policy.abbreviation_entries, block.source_text
        )

        fallback_prev: str | None = None
        if block.spine_index > 1 and (not current_batch or block.id == current_batch[0].id):
            fallback_prev = (
                await asyncio.to_thread(
                    self.runtime.ledger.get_preceding_text_tail,
                    self.runtime.actual_job_id,
                    block.flow_id,
                    block.spine_index,
                    max_chars=300,
                )
                or None
            )

        fallback_next: str | None = None
        current_ch = extract_chapter_id(block.id)
        if current_batch and block.id == current_batch[-1].id:
            fallback_next = (
                await asyncio.to_thread(
                    self.runtime.ledger.get_following_text_head,
                    self.runtime.actual_job_id,
                    block.flow_id,
                    block.spine_index,
                    chapter_id=current_ch,
                    max_chars=300,
                )
                or None
            )

        neighbor_ctx = self.runtime.memory_mgr.get_l1_context(
            target_block=block,
            surrounding_blocks=current_batch,
            fallback_prev_text=fallback_prev,
            fallback_next_text=fallback_next,
        )

        # Off-loop: regex/parse-heavy masking would otherwise serialize the
        # single event loop across concurrent draft tasks and SSE fan-out. The
        # mask order (code -> math -> soup -> citation) lives in the engine, not
        # here (translation unit segmentation layer).
        masked = await asyncio.to_thread(self.runtime.engine.mask, _source_with_bold_markers(block))

        macro_ctx = ""
        if self.policy.rolling_enabled:
            # One summary channel: the hierarchical L2 snapshot (see
            # HierarchicalMemoryManager). A chapter-boundary summarizer would be
            # a second code path writing the same ``macro_ctx`` and a second LLM
            # call per chapter.
            macro_ctx = self.runtime.memory_mgr.get_macro_context_for_block(block)
        # Compressed L3 epoch history rides the static prompt prefix.
        epoch_ctx = self.runtime.memory_mgr.get_l3_summary()

        inputs = _DraftInputs(
            masked_source=masked.text,
            code_map=masked.code_map,
            cite_map=masked.cite_map,
            math_map=masked.math_map,
            soup_map=masked.soup_map,
            email_map=masked.email_map,
            glossary_table=glossary_table,
            neighbor_ctx=neighbor_ctx,
            macro_ctx=macro_ctx,
            epoch_ctx=epoch_ctx,
            few_shot_reference=few_shot_reference,
            block_type=str(block.block_type),
        )

        return inputs

    async def finalize_draft(self, block: IRBlock, raw_text: str, inputs: _DraftInputs) -> bool:
        """Unmask, record in hierarchical memory, and checkpoint a draft.

        Returns whether the protective spans restored cleanly — the caller's
        signal for whether this raw draft is worth caching.
        """
        # A block that reached finalize_draft drafted successfully, so the
        # fail-fast breaker only counts *consecutive* non-retryable failures.
        self.runtime.fail_fast_consecutive = 0

        # Restore in reverse mask order (citation -> soup -> math -> code ->
        # email) and judge: the engine owns that order and verifies every
        # namespace with the same checksummed contract (translation unit
        # segmentation layer).
        masked = ports.masked_source(
            text=inputs.masked_source,
            code_map=inputs.code_map,
            math_map=inputs.math_map,
            soup_map=inputs.soup_map,
            cite_map=inputs.cite_map,
            email_map=inputs.email_map,
        )
        result = await asyncio.to_thread(self.runtime.engine.resolve, raw_text, masked)
        # The model bracketed emphasized target text with ⟦B⟧ … ⟦/B⟧; parse the
        # runs out (and strip every marker, balanced or not) so the target text
        # is always clean and the render can re-apply the weight the source-
        # derived inline run cannot locate.
        final_draft, target_runs = parse_bold(result.text)
        if block.block_type == BlockType.HEADING:
            trimmed = trim_heading_expansion(block.source_text, final_draft)
            if trimmed != final_draft:
                logger.info(
                    "Heading %s draft carried an appended paragraph; kept the leading one",
                    block.id,
                )
                final_draft = trimmed

        error_flags: list[str] = list(block.error_flags)
        # One rule for all four maskers: a restore that is not clean (missing /
        # mismatched / mutated / reordered / duplicated span) means the target
        # is not trustworthy. Record the evidence on the block instead of
        # shipping it.
        for label, report in result.dirty:
            detail = (
                f"{label} missing={report.missing} "
                f"mismatched={report.mismatched} "
                f"mutated={report.mutated} "
                f"reordered={report.reordered} "
                f"duplicated={report.duplicated}"
            )
            logger.warning("Block %s draft %s", block.id, detail)
            error_flags.append(detail)
            self.runtime.counters[label] = self.runtime.counters.get(label, 0) + 1

        restore_clean = result.clean

        # Overwrite (never merge) the target runs: an empty result must clear a
        # stale run from a previous draft of the same block.
        new_style = (
            (block.style or StyleMeta()).model_copy(update={"target_runs": target_runs})
            if target_runs or block.style is not None
            else None
        )
        drafted_block = block.model_copy(
            update={"target_text": final_draft, "draft_text": final_draft, "style": new_style}
        )
        self.runtime.memory_mgr.record_drafted_block(drafted_block)

        update = {
            "block_id": block.id,
            "target_text": final_draft,
            "status": BlockStatus.DRAFTED if restore_clean else BlockStatus.REPAIR_PENDING,
            "draft_text": final_draft,
            "error_flags": error_flags or None,
            "style": new_style,
        }
        await self.runtime.flusher.enqueue(update)
        return restore_clean

    async def draft_single_block(
        self,
        block: IRBlock,
        current_batch: list[IRBlock],
        inputs: _DraftInputs | None = None,
    ) -> None:
        if is_already_final(block):
            return

        if is_static_skip(block):
            skip_update = {
                "block_id": block.id,
                "target_text": block.source_text,
                "status": BlockStatus.MTQE_PASSED,
                "mtqe_score": 1.0,
            }
            await self.runtime.flusher.enqueue(skip_update)
            return

        async with self.runtime.concurrency_sem:
            if inputs is None:
                inputs = await self.prepare_draft_inputs(block, current_batch)
                if inputs is None:
                    return

            async def generate(masked_source: str) -> str:
                """The generate half: one provider draft, with this stage's retries.

                Captured here (not inlined) so the engine's cache-aware
                :meth:`~ubt.translate.engine.TranslationEngine.draft` owns the
                cache decision around it.
                """
                block_to_draft = block.with_source_text(masked_source)
                draft_raw: str | None = None
                last_exc: Exception | None = None
                for attempt in range(self.policy.draft_max_retries + 1):
                    try:
                        draft_raw = await self.runtime.router.draft(
                            block=block_to_draft,
                            glossary_table=inputs.glossary_table,
                            neighbor_context=inputs.neighbor_ctx,
                            target_lang=self.policy.target_lang,
                            source_lang=self.policy.source_lang,
                            genre_profile=self.policy.profile_name,
                            domain=self.policy.domain,
                            rolling_summary=inputs.macro_ctx,
                            epoch_summary=inputs.epoch_ctx,
                            global_glossary=self.policy.global_glossary_table,
                            few_shot_reference=inputs.few_shot_reference,
                        )
                        break
                    except Exception as exc:
                        last_exc = exc
                        # The router already walks its own retry + fallback
                        # chain and knows which failures are unrecoverable. A
                        # 401/402/400 will fail identically on every retry, so
                        # re-asking just multiplies quota burn on top of the
                        # router's internal retries — bail out immediately.
                        if not classify_provider_error(exc).retryable:
                            logger.warning(
                                "Non-retryable draft failure for block %s, giving up: %s",
                                block.id,
                                exc,
                            )
                            break
                        if attempt >= self.policy.draft_max_retries:
                            break
                        delay = min(8.0, self.policy.draft_retry_base_delay * (2**attempt))
                        delay += random.uniform(0, delay * 0.25)
                        await asyncio.sleep(delay)

                if draft_raw is None:
                    raise (
                        last_exc
                        if last_exc is not None
                        else RuntimeError("Draft returned no content")
                    )
                return draft_raw

            try:
                # The engine owns the cache-aware generate step (content-addressed
                # cache layer): a hit skips the provider entirely, and a changed
                # prompt (context) is a miss, not a stale draft. The prompt key
                # is built only when a cache is set, so a cache-less run pays
                # nothing for it.
                context = (
                    self.prompt_context(inputs) if self.runtime.engine.cache is not None else ""
                )
                draft_raw = await self.runtime.engine.draft(
                    inputs.masked_source, generate, context=context
                )
                clean = await self.finalize_draft(block, draft_raw, inputs)
                # Cache only a judged-clean draft: a cached defective draft
                # replays on every later run, and only repair could recover it.
                if clean and self.runtime.engine.cache is not None:
                    await asyncio.to_thread(
                        self.runtime.engine.remember_draft,
                        inputs.masked_source,
                        draft_raw,
                        context=context,
                    )
            except (asyncio.CancelledError, JobInterruptedError, BudgetExceededError):
                raise
            except Exception as exc:
                retryable = classify_provider_error(exc).retryable
                if retryable:
                    self.runtime.fail_fast_consecutive = 0
                else:
                    self.runtime.fail_fast_consecutive += 1
                    self.runtime.last_fail_fast_reason = str(exc)
                if isinstance(exc, (TypeError, AttributeError, NameError, UnboundLocalError)):
                    logger.error(
                        "Programming defect encountered during drafting for block %s: %s",
                        block.id,
                        exc,
                        exc_info=True,
                    )
                # A non-retryable failure is recorded under a prefix the resume
                # path does NOT treat as transient: the same 401/402/400 recurs
                # on every run, so re-queueing the block only re-bills it.
                prefix = DRAFTING_ERROR_PREFIX if retryable else NON_RETRYABLE_DRAFT_PREFIX
                fail_update = {
                    "block_id": block.id,
                    "status": BlockStatus.FAILED,
                    "error_flags": [f"{prefix} {exc}"],
                }
                await self.runtime.flusher.enqueue(fail_update)

    async def finalize_cache_hits(
        self, prepared: list[tuple[IRBlock, _DraftInputs]]
    ) -> list[tuple[IRBlock, _DraftInputs]]:
        """Finalize the translate-cache hits now; return the misses to submit.

        The cache is content-addressed on the exact prompt, so a hit is a draft
        this run would have generated identically. Finalizing it directly saves
        the provider request entirely (content-addressed cache layer) -- for the Batch API
        paths that is a request not submitted, not just a response reused.
        """
        if self.runtime.engine.cache is None:
            return prepared
        misses: list[tuple[IRBlock, _DraftInputs]] = []
        for b, inp in prepared:
            cached = await asyncio.to_thread(
                self.runtime.engine.cached_draft,
                inp.masked_source,
                context=self.prompt_context(inp),
            )
            if cached is None:
                misses.append((b, inp))
                continue
            await self.finalize_draft(b, cached, inp)
            self.runtime.counters["translate_cache_hits"] = (
                self.runtime.counters.get("translate_cache_hits", 0) + 1
            )
        return misses

    async def try_batch_draft(
        self,
        blocks: list[IRBlock],
        current_batch: list[IRBlock],
    ) -> bool:
        """Translate the whole claim batch through one Batch API job."""
        prepared: list[tuple[IRBlock, _DraftInputs]] = []
        for b in blocks:
            if is_already_final(b):
                continue
            if is_static_skip(b):
                await self.draft_single_block(b, current_batch)
                continue
            inputs = await self.prepare_draft_inputs(b, current_batch)
            if inputs is not None:
                prepared.append((b, inputs))
        if not prepared:
            return True
        # A cache hit needs no batch request: finalize it and submit the rest.
        prepared = await self.finalize_cache_hits(prepared)
        if not prepared:
            return True

        async def _batch_status_callback(status: str, job_dict: dict[str, Any]) -> None:
            if self.runtime.ctx is not None:
                self.runtime.ctx.check_cancelled()
            logger.info(
                "Batch draft job %s poll status: %s (counts=%s)",
                self.runtime.actual_job_id,
                status,
                job_dict.get("request_counts"),
            )
            if self.runtime.create_event_fn:
                await self.runtime.create_event_fn(
                    EventType.DRAFT_BATCH_COMPLETED,
                    self.runtime.actual_job_id,
                    self.runtime.ledger,
                    message=f"Batch draft job status: {status}",
                    active_block_id=prepared[0][0].id if prepared else None,
                )

        try:
            batch_results = await self.runtime.router.draft_batch(
                [
                    BatchDraftRequest(
                        custom_id=b.id,
                        source_text=inp.masked_source,
                        glossary_table=inp.glossary_table,
                        neighbor_context=inp.neighbor_ctx,
                        target_lang=self.policy.target_lang,
                        source_lang=self.policy.source_lang,
                        genre_profile=self.policy.profile_name,
                        domain=self.policy.domain,
                        rolling_summary=inp.macro_ctx,
                        epoch_summary=inp.epoch_ctx,
                        global_glossary=self.policy.global_glossary_table,
                        few_shot_reference=inp.few_shot_reference,
                        block_type=inp.block_type,
                    )
                    for b, inp in prepared
                ],
                poll_interval=self.policy.batch_poll_interval,
                poll_timeout=self.policy.batch_poll_timeout,
                ledger=self.runtime.ledger,
                job_id=self.runtime.actual_job_id,
                cleanup_files=self.policy.batch_delete_files,
                status_callback=_batch_status_callback,
            )
        except (BudgetExceededError, JobInterruptedError, asyncio.CancelledError):
            raise
        except BatchTranslationError as exc:
            logger.warning(
                "Batch draft unavailable (%s); falling back to interactive for %d blocks",
                exc,
                len(prepared),
            )
            # If a batch had already been submitted, cancel it now that we are
            # abandoning batch mode for these blocks — otherwise it keeps
            # billing while we re-draft them interactively (duplicate charge).
            if exc.batch_id:
                await self.runtime.router.abandon_batch(
                    exc.batch_id, ledger=self.runtime.ledger, job_id=self.runtime.actual_job_id
                )
            self.runtime.counters["batch_fallbacks"] += len(prepared)
            return False

        results_by_id = {r.custom_id: r for r in batch_results}
        retriable: list[tuple[IRBlock, _DraftInputs]] = []
        for b, inp in prepared:
            r = results_by_id.get(b.id)
            # A blank line is a missing line: the quality gate would otherwise
            # score an empty target, and the macro path already treats "" as
            # unusable, so the two batch shapes must not disagree here.
            if r is None or r.error or not r.text or not r.text.strip():
                retriable.append((b, inp))
                continue
            # Record the batch's own output under the same key the interactive
            # path reads, so a re-run reuses it (content-addressed cache layer)
            # — but only once the restore judged clean: a cached defective
            # draft replays on every later run, and only repair could recover it.
            clean = await self.finalize_draft(b, r.text, inp)
            if clean and self.runtime.engine.cache is not None:
                await asyncio.to_thread(
                    self.runtime.engine.remember_draft,
                    inp.masked_source,
                    r.text,
                    context=self.prompt_context(inp),
                )
            self.runtime.counters["batch_drafted"] += 1
        if retriable:
            self.runtime.counters["batch_fallbacks"] += len(retriable)
            logger.info(
                "Batch job returned %d unusable lines; redrafting interactively",
                len(retriable),
            )
            results = await asyncio.gather(
                *[self.draft_single_block(b, current_batch, inp) for b, inp in retriable],
                return_exceptions=True,
            )
            for res in results:
                if isinstance(
                    res, (BudgetExceededError, JobInterruptedError, asyncio.CancelledError)
                ):
                    raise res
                if isinstance(res, Exception):
                    logger.warning("Interactive redraft failed for block: %s", res)
        return True

    async def run_whole_book_batch(
        self,
        blocks: list[IRBlock],
        ctx: StageContext,
        create_event_fn: Any = None,
    ) -> bool:
        """Translate all pending blocks across the whole book through one unified Batch API job.

        Returns True if batch draft succeeded (with any per-line retries completed),
        or False if batch submission failed / is unsupported and interactive fallback is needed.
        """
        ctx.check_cancelled()
        segments = segment_by_chapter(blocks)
        prepared: list[tuple[IRBlock, _DraftInputs]] = []

        for segment in segments:
            for b in segment:
                if is_already_final(b):
                    continue
                if is_static_skip(b):
                    await self.draft_single_block(b, segment)
                    continue
                inputs = await self.prepare_draft_inputs(b, segment)
                if inputs is not None:
                    prepared.append((b, inputs))

        if not prepared:
            await self.runtime.flusher.flush_all()
            return True
        # A cache hit needs no batch request: finalize it and submit the rest.
        prepared = await self.finalize_cache_hits(prepared)
        if not prepared:
            await self.runtime.flusher.flush_all()
            return True

        async def _batch_status_callback(status: str, job_dict: dict[str, Any]) -> None:
            ctx.check_cancelled()
            logger.info(
                "Whole-book batch job %s poll status: %s (counts=%s)",
                self.runtime.actual_job_id,
                status,
                job_dict.get("request_counts"),
            )
            if create_event_fn:
                # create_event_fn is the budget enforcement point (it bills this
                # run's spend and raises BudgetExceededError); the event it
                # returns has no delivery channel of its own here.
                await create_event_fn(
                    EventType.DRAFT_BATCH_COMPLETED,
                    self.runtime.actual_job_id,
                    self.runtime.ledger,
                    message=f"Whole-book batch job status: {status}",
                    active_block_id=prepared[0][0].id if prepared else None,
                )

        try:
            batch_results = await self.runtime.router.draft_batch(
                [
                    BatchDraftRequest(
                        custom_id=b.id,
                        source_text=inp.masked_source,
                        glossary_table=inp.glossary_table,
                        neighbor_context=inp.neighbor_ctx,
                        target_lang=self.policy.target_lang,
                        source_lang=self.policy.source_lang,
                        genre_profile=self.policy.profile_name,
                        domain=self.policy.domain,
                        rolling_summary=inp.macro_ctx,
                        epoch_summary=inp.epoch_ctx,
                        global_glossary=self.policy.global_glossary_table,
                        few_shot_reference=inp.few_shot_reference,
                        block_type=inp.block_type,
                    )
                    for b, inp in prepared
                ],
                poll_interval=self.policy.batch_poll_interval,
                poll_timeout=self.policy.batch_poll_timeout,
                ledger=self.runtime.ledger,
                job_id=self.runtime.actual_job_id,
                cleanup_files=self.policy.batch_delete_files,
                status_callback=_batch_status_callback,
            )
        except BatchTranslationError as exc:
            logger.warning(
                "Whole-book batch draft unavailable (%s); falling back to interactive for %d blocks",
                exc,
                len(prepared),
            )
            if exc.batch_id:
                await self.runtime.router.abandon_batch(
                    exc.batch_id, ledger=self.runtime.ledger, job_id=self.runtime.actual_job_id
                )
            self.runtime.counters["batch_fallbacks"] += len(prepared)
            self.runtime.batch_active = False
            return False
        except (BudgetExceededError, JobInterruptedError):
            # Neither is a batch-infrastructure hiccup, and both mean "stop
            # spending". Falling through to the catch-all below would return
            # False and re-draft every block interactively — past the cap the
            # user just hit, or after the cancel that was supposed to end the
            # run — while the batch itself is still live at the provider.
            raise
        except Exception as exc:
            logger.warning(
                "Whole-book batch draft encountered exception (%s); interactive fallback", exc
            )
            self.runtime.counters["batch_fallbacks"] += len(prepared)
            self.runtime.batch_active = False
            return False

        results_by_id = {r.custom_id: r for r in batch_results}
        retriable: list[tuple[IRBlock, _DraftInputs]] = []
        for b, inp in prepared:
            r = results_by_id.get(b.id)
            # A blank line is a missing line: the quality gate would otherwise
            # score an empty target, and the macro path already treats "" as
            # unusable, so the two batch shapes must not disagree here.
            if r is None or r.error or not r.text or not r.text.strip():
                retriable.append((b, inp))
                continue
            # Record the batch's own output under the same key the interactive
            # path reads, so a re-run reuses it (content-addressed cache layer)
            # — but only once the restore judged clean: a cached defective
            # draft replays on every later run, and only repair could recover it.
            clean = await self.finalize_draft(b, r.text, inp)
            if clean and self.runtime.engine.cache is not None:
                await asyncio.to_thread(
                    self.runtime.engine.remember_draft,
                    inp.masked_source,
                    r.text,
                    context=self.prompt_context(inp),
                )
            self.runtime.counters["batch_drafted"] += 1

        if retriable:
            self.runtime.counters["batch_fallbacks"] += len(retriable)
            logger.info(
                "Whole-book batch job returned %d unusable lines; redrafting interactively",
                len(retriable),
            )
            if self.policy.macro_chunk_size <= 1:
                results = await asyncio.gather(
                    *[self.draft_single_block(b, [b], inp) for b, inp in retriable],
                    return_exceptions=True,
                )
            else:
                chunks = [
                    retriable[i : i + self.policy.macro_chunk_size]
                    for i in range(0, len(retriable), self.policy.macro_chunk_size)
                ]
                results = await asyncio.gather(
                    *[
                        self.draft_macro_chunk_group(chunk, [b for b, _ in chunk])
                        for chunk in chunks
                    ],
                    return_exceptions=True,
                )
            for res in results:
                if isinstance(
                    res, (BudgetExceededError, JobInterruptedError, asyncio.CancelledError)
                ):
                    raise res
                if isinstance(res, Exception):
                    logger.warning("Whole-book interactive fallback failed: %s", res)

        await self.runtime.flusher.flush_all()
        return True

    async def draft_macro_chunk_group(
        self,
        chunk: list[tuple[IRBlock, _DraftInputs]],
        current_batch: list[IRBlock],
    ) -> None:
        """Draft a group of consecutive blocks in a single structured XML LLM call."""
        if not chunk:
            return
        if len(chunk) == 1:
            block, inp = chunk[0]
            await self.draft_single_block(block, current_batch, inp)
            return

        # Chunk-level prompt inputs, computed once so the cache key and the call
        # cannot drift. Order-preserving dedup: a set of strings iterates in
        # hash-seed order, so a plain set would give the same book a differently
        # ordered term table (and therefore a different prompt) on every run.
        merged_glossary = "\n\n".join(
            filter(None, dict.fromkeys(inp.glossary_table.strip() for _, inp in chunk))
        )
        first_inp = chunk[0][1]
        last_inp = chunk[-1][1]
        macro_ctx = first_inp.macro_ctx
        epoch_ctx = first_inp.epoch_ctx
        few_shot = first_inp.few_shot_reference or last_inp.few_shot_reference

        extracted_by_id: dict[str, str] = {}
        cached_context: str | None = None
        if self.runtime.engine.cache is not None:
            # One provider call, many units: keyed on the chunk's own prompt
            # digest, so a re-run of the same chunk is a hit (content-addressed cache layer).
            cached_context = self.chunk_context(chunk)
            cached_raw = await asyncio.to_thread(
                self.runtime.engine.cached_value, cached_context, kind="translate_chunk"
            )
            extracted_by_id = _decode_chunk(cached_raw)

        if not extracted_by_id:
            async with self.runtime.concurrency_sem:
                blocks_to_draft = [b.with_source_text(inp.masked_source) for b, inp in chunk]
                try:
                    for attempt in range(self.policy.draft_max_retries + 1):
                        try:
                            extracted_by_id = await self.runtime.router.draft_macro_chunk(
                                blocks=blocks_to_draft,
                                glossary_table=merged_glossary,
                                neighbor_context=first_inp.neighbor_ctx,
                                target_lang=self.policy.target_lang,
                                source_lang=self.policy.source_lang,
                                genre_profile=self.policy.profile_name,
                                domain=self.policy.domain,
                                rolling_summary=macro_ctx,
                                epoch_summary=epoch_ctx,
                                global_glossary=self.policy.global_glossary_table,
                                few_shot_reference=few_shot,
                            )
                            break
                        except Exception as exc:
                            if (
                                not classify_provider_error(exc).retryable
                                or attempt >= self.policy.draft_max_retries
                            ):
                                raise
                            delay = min(8.0, self.policy.draft_retry_base_delay * (2**attempt))
                            delay += random.uniform(0, delay * 0.25)
                            await asyncio.sleep(delay)
                except (asyncio.CancelledError, JobInterruptedError, BudgetExceededError):
                    raise
                except Exception as exc:
                    logger.warning(
                        "Macro-chunk draft failed for %d blocks (%s); redrafting blocks individually",
                        len(chunk),
                        exc,
                    )
                    # Empty the extraction map and fall through to the missing-block
                    # loop below, which redrafts individually *outside* the
                    # semaphore: draft_single_block re-acquires it, so calling it
                    # here (while this group still holds a permit) self-deadlocks.
                    extracted_by_id = {}

        missing: list[tuple[IRBlock, _DraftInputs]] = []
        clean_by_id: dict[str, bool] = {}
        for b, inp in chunk:
            text = extracted_by_id.get(b.id)
            if text and text.strip():
                clean_by_id[b.id] = await self.finalize_draft(b, text.strip(), inp)
            else:
                missing.append((b, inp))

        # Cache the chunk's extraction map only when every block in it restored
        # cleanly: one dirty draft would otherwise be replayed from the map on
        # every resume of the run.
        if (
            extracted_by_id
            and cached_context is not None
            and clean_by_id
            and all(clean_by_id.get(b.id, False) for b, _ in chunk)
        ):
            await asyncio.to_thread(
                self.runtime.engine.remember_value,
                cached_context,
                _encode_chunk(extracted_by_id),
                kind="translate_chunk",
            )

        if missing:
            logger.info(
                "Macro-chunk missed %d/%d blocks; redrafting missing blocks individually",
                len(missing),
                len(chunk),
            )
            for b, inp in missing:
                await self.draft_single_block(b, current_batch, inp)

    async def run_draft_batch(
        self,
        blocks: list[IRBlock],
        current_batch: list[IRBlock],
    ) -> None:
        if self.runtime.batch_active and len(blocks) >= self.policy.batch_min_blocks:
            try:
                handled = await self.try_batch_draft(blocks, current_batch)
            except (BudgetExceededError, JobInterruptedError, asyncio.CancelledError):
                raise
            except Exception as exc:  # defensive: batch must never kill the stage
                logger.warning("Batch draft crashed (%s); interactive fallback", exc)
                handled = False
            if handled:
                return

        prepared: list[tuple[IRBlock, _DraftInputs]] = []
        for b in blocks:
            if is_already_final(b):
                continue
            if is_static_skip(b):
                await self.draft_single_block(b, current_batch)
                continue
            inputs = await self.prepare_draft_inputs(b, current_batch)
            if inputs is not None:
                prepared.append((b, inputs))

        if not prepared:
            return

        if self.policy.macro_chunk_size <= 1:
            groups: list[list[IRBlock]] = [[b] for b, _ in prepared]
            results = await asyncio.gather(
                *[self.draft_single_block(b, current_batch, inp) for b, inp in prepared],
                return_exceptions=True,
            )
        else:
            chunks = [
                prepared[i : i + self.policy.macro_chunk_size]
                for i in range(0, len(prepared), self.policy.macro_chunk_size)
            ]
            groups = [[b for b, _ in chunk] for chunk in chunks]
            results = await asyncio.gather(
                *[self.draft_macro_chunk_group(chunk, current_batch) for chunk in chunks],
                return_exceptions=True,
            )

        critical_exc: BaseException | None = None
        stranded: list[dict[str, Any]] = []
        for res, group in zip(results, groups, strict=True):
            if isinstance(res, (BudgetExceededError, JobInterruptedError, asyncio.CancelledError)):
                if critical_exc is None:
                    critical_exc = res
                continue
            if isinstance(res, Exception):
                logger.warning("Draft task failed with exception: %s", res)
                prefix = (
                    DRAFTING_ERROR_PREFIX
                    if classify_provider_error(res).retryable
                    else NON_RETRYABLE_DRAFT_PREFIX
                )
                for b in group:
                    stranded.append(
                        {
                            "block_id": b.id,
                            "status": BlockStatus.FAILED,
                            "error_flags": [f"{prefix} {res}"],
                        }
                    )
        if stranded:
            for update in stranded:
                await self.runtime.flusher.enqueue(update)
        if critical_exc is not None:
            raise critical_exc


def _persist_memory_state(
    ledger: SQLiteJobLedger,
    actual_job_id: str,
    memory_mgr: HierarchicalMemoryManager,
) -> None:
    """Persist hierarchical L2/L3 memory to job_meta so a resumed run keeps the context."""
    state = memory_mgr.export_state()
    try:
        ledger.set_job_metadata_value(actual_job_id, "memory_state", state)
    except Exception as exc:
        logger.warning("Failed to persist rolling memory state for %s: %s", actual_job_id, exc)


def _restore_memory_state(
    ledger: SQLiteJobLedger,
    actual_job_id: str,
    memory_mgr: HierarchicalMemoryManager,
) -> None:
    """Restore hierarchical L2/L3 memory persisted by a previous run of this job."""
    raw = ledger.get_job_metadata_value(actual_job_id, "memory_state")
    if not isinstance(raw, dict):
        return
    try:
        memory_mgr.restore_state(raw)
    except Exception as exc:
        logger.warning("Failed to restore rolling memory state for %s: %s", actual_job_id, exc)
        return
    logger.info(
        "Restored hierarchical memory state for %s (%d L2 snapshots, %d L3 epochs)",
        actual_job_id,
        len(memory_mgr.snapshots),
        len(memory_mgr.epochs),
    )


@dataclass(slots=True)
class _DraftAssembly:
    """Everything the draft streaming loop consumes, built once up front."""

    processor: _DraftProcessor
    flusher: CheckpointBatchFlusher
    memory_mgr: HierarchicalMemoryManager
    counters: dict[str, int]
    create_event_fn: Any
    router: ModelRouter
    target_lang: str
    batch_limit: int
    rolling_enabled: bool
    is_fast_path: bool


async def _build_draft_runtime(
    ctx: StageContext, services: RunServices, terminology: Terminology
) -> _DraftAssembly:
    """Assemble the draft stage's collaborators: cache, engine, memory, flusher, policy.

    Kept apart from :func:`run_draft_stage` so the streaming loop reads as the
    loop alone; this is the config-unpacking and object-assembly half.
    """
    ledger = ctx.ledger
    actual_job_id = ctx.job_id
    profile_name = ctx.profile_name
    target_lang = ctx.target_lang
    source_lang = ctx.source_lang
    router = ctx.router
    config = ctx.config
    from ubt.cache.store import DiskCacheStore

    # The translate step's content cache (content-addressed cache layer): a resumed or re-run
    # job reuses drafts whose exact prompt is unchanged. Fail-open, and off when
    # UBT_CACHE_ENABLED=0.
    draft_cache = DiskCacheStore(config.cache_dir) if config.cache_enabled else None
    # The per-unit transform (mask -> restore -> judge) has one owner now
    # (translation unit segmentation layer): the engine holds the fixed mask order, while the
    # router/batch/retry orchestration below stays in this stage. The cache is
    # keyed on the exact prompt, so the production path and the standalone path
    # share one content-addressed translate step (content-addressed cache layer).
    translation_engine = ports.translation_engine(
        placeholders=ports.placeholder_engine(),
        model=router.draft_model,
        prompt_version=PROMPT_VERSION,
        cache=draft_cache,
    )
    # How many blocks the book holds: derived here from the ledger (it is
    # observable state, not something a stage has to hand forward) and read by
    # resolve_draft_policy to decide fast-path vs rolling summaries.
    stats = await asyncio.to_thread(ledger.get_job_stats, actual_job_id)
    try:
        all_blocks_count = int(stats.get("total", 0)) if stats else 0
    except (TypeError, ValueError):
        all_blocks_count = 0
    glossary_dicts = terminology.glossary_dicts
    abbreviation_entries = terminology.abbreviation_entries
    concurrency_sem = services.concurrency_sem
    create_event_fn = ctx.create_event
    tm = services.tm
    fast_pass = services.fast_pass
    rolling_enabled, is_fast_path, batch_limit = resolve_draft_policy(
        ctx.manifest, profile_name, config, all_blocks_count
    )

    memory_mgr = HierarchicalMemoryManager(
        step_chars=int(config.step_chars),
        # Hierarchical L2/L3 is the multi-chapter long route only (see
        # resolve_draft_policy / README). Gating it here -- not just the macro_ctx
        # slot -- is what stops a single-chapter PDF from paying L2 llm_summary
        # calls for summaries the policy then refuses to inject.
        enabled=rolling_enabled,
    )
    global_glossary_table = build_global_glossary_table(
        glossary_dicts or [], config.glossary_max_global_entries
    )
    dropped_global = max(0, len(glossary_dicts or []) - config.glossary_max_global_entries)
    if dropped_global:
        logger.info(
            "Global terminology sheet capped at %d entries; %d lower-ranked entries now "
            "travel only via per-chunk retrieval (raise UBT_GLOSSARY_MAX_GLOBAL_ENTRIES, "
            "or set 0 to send no sheet at all)",
            config.glossary_max_global_entries,
            dropped_global,
        )
    draft_max_retries = max(0, int(config.draft_max_retries))
    draft_retry_base_delay = max(0.0, float(config.draft_retry_base_delay))

    tm_fuzzy_threshold = max(0.0, min(1.0, float(config.tm_fuzzy_threshold)))
    active_fast_pass = fast_pass or FastPassFilter(source_lang=source_lang, target_lang=target_lang)
    # The Bible exists now; teach the omission gate that a term the glossary
    # renders in the target language is kept, not omitted. Without this the two
    # gates contradict each other on entries like FnCall -> 函数调用接口 and the
    # correct translation is quarantined as an omission.
    active_fast_pass.set_glossary(glossary_dicts)
    batch_active = bool(config.batch_enabled) and (router.supports_batch_api)
    batch_min_blocks = max(2, int(config.batch_min_blocks))
    batch_poll_interval = max(0.5, float(config.batch_poll_interval))
    batch_poll_timeout = max(1.0, float(config.batch_poll_timeout))
    batch_delete_files = bool(config.batch_delete_files)
    counters = {
        "tm_exact_hits": 0,
        "batch_drafted": 0,
        "batch_fallbacks": 0,
        "translate_cache_hits": 0,
    }

    # Exact TM hits are only valid under this run's prompt/glossary context;
    # the same fingerprint gates writeback in pipeline.py. The abbreviation
    # channel is prompt-visible too, so it enters the hash.
    tm_context = compute_tm_context(
        PROMPT_VERSION,
        profile_name,
        global_glossary_table,
        source_lang,
        target_lang,
        format_abbreviations_markdown_table(abbreviation_entries),
        model=router.draft_model,
    )

    flusher = CheckpointBatchFlusher(
        ledger=ledger,
        flush_interval=float(config.ledger_flush_interval),
        max_batch_size=int(config.ledger_flush_batch_size),
    )

    processor = _DraftProcessor(
        runtime=DraftRuntime(
            ledger=ledger,
            actual_job_id=actual_job_id,
            router=router,
            engine=translation_engine,
            memory_mgr=memory_mgr,
            concurrency_sem=concurrency_sem,
            tm=tm,
            active_fast_pass=active_fast_pass,
            counters=counters,
            flusher=flusher,
            batch_active=batch_active,
            ctx=ctx,
            create_event_fn=create_event_fn,
        ),
        policy=DraftPolicy(
            glossary_dicts=glossary_dicts,
            abbreviation_entries=abbreviation_entries,
            global_glossary_table=global_glossary_table,
            source_lang=source_lang,
            target_lang=target_lang,
            profile_name=profile_name,
            domain=config.domain,
            tm_fuzzy_threshold=tm_fuzzy_threshold,
            tm_context=tm_context,
            draft_max_retries=draft_max_retries,
            draft_retry_base_delay=draft_retry_base_delay,
            batch_min_blocks=batch_min_blocks,
            batch_poll_interval=batch_poll_interval,
            batch_poll_timeout=batch_poll_timeout,
            rolling_enabled=rolling_enabled,
            batch_delete_files=batch_delete_files,
            macro_chunk_size=max(1, int(config.macro_chunk_size)),
        ),
    )
    return _DraftAssembly(
        processor=processor,
        flusher=flusher,
        memory_mgr=memory_mgr,
        counters=counters,
        create_event_fn=create_event_fn,
        router=router,
        target_lang=target_lang,
        batch_limit=batch_limit,
        rolling_enabled=rolling_enabled,
        is_fast_path=is_fast_path,
    )


async def run_draft_stage(
    ctx: StageContext,
    services: RunServices,
    terminology: Terminology,
    chapter_id: str | None = None,
) -> AsyncIterator[TranslationProgressEvent]:
    """Execute streaming drafted translation across keyset pagination batches."""
    assembly = await _build_draft_runtime(ctx, services, terminology)
    processor = assembly.processor
    flusher = assembly.flusher
    memory_mgr = assembly.memory_mgr
    counters = assembly.counters
    create_event_fn = assembly.create_event_fn
    router = assembly.router
    target_lang = assembly.target_lang
    batch_limit = assembly.batch_limit
    rolling_enabled = assembly.rolling_enabled
    ledger = ctx.ledger
    actual_job_id = ctx.job_id
    config = ctx.config

    # Resume recovery: re-queue blocks stranded in FAILED /
    # NEEDS_HUMAN by transient drafting/repair errors, so an API outage does
    # not permanently turn into a manual work order. Quality escalations
    # (MQM spans / needs_human_review without transient markers) stay
    # terminal. No-op on a fresh job (no blocks carry failure flags yet).
    # Chapter-scoped under streaming: this runs once per chapter, and a
    # job-wide reset would NULL the preserved paid drafts of earlier
    # chapters whose draft pass is over.
    await asyncio.to_thread(ledger.reset_transient_failures, actual_job_id, chapter_id)

    # Resume-stable rolling memory: continue with the same L2/L3
    # summaries the interrupted run used, keeping prompt context compatible.
    if rolling_enabled:
        await asyncio.to_thread(_restore_memory_state, ledger, actual_job_id, memory_mgr)

    try:
        # Whole-book offline Batch API mode.
        offline_batch_requested = getattr(config, "offline_batch_enabled", False)
        if offline_batch_requested and (router.supports_batch_api or bool(config.batch_enabled)):
            logger.info(
                "Whole-book offline batch mode active for job %s (chapter_id=%s)",
                actual_job_id,
                chapter_id,
            )
            all_pending_blocks: list[IRBlock] = []
            cursor_spine_init: int | None = None
            cursor_block_id_init: str | None = None
            while True:
                ctx.check_cancelled()
                chunk = await asyncio.to_thread(
                    ledger.fetch_pending_blocks,
                    actual_job_id,
                    limit=500,
                    after_spine_index=cursor_spine_init,
                    after_block_id=cursor_block_id_init,
                    chapter_id=chapter_id,
                )
                if not chunk:
                    break
                all_pending_blocks.extend(chunk)
                cursor_spine_init = chunk[-1].spine_index
                cursor_block_id_init = chunk[-1].id

            if all_pending_blocks:
                stats_before = await asyncio.to_thread(ledger.get_job_stats, actual_job_id)
                handled = await processor.run_whole_book_batch(
                    all_pending_blocks,
                    ctx,
                    create_event_fn=create_event_fn,
                )
                await flusher.flush_all()
                # The fail-fast circuit the interactive loop applies below: this
                # path returns before reaching it, so a dead provider would burn
                # every remaining block through the batch fallback instead of
                # aborting the job the way the interactive path does.
                if processor.runtime.fail_fast_consecutive >= processor.policy.fail_fast_threshold:
                    raise UBTError(
                        f"Whole-book offline batch aborted after "
                        f"{processor.runtime.fail_fast_consecutive} consecutive "
                        "non-retryable provider failures (fail-fast circuit). "
                        f"Last error: {processor.runtime.last_fail_fast_reason}"
                    )
                if handled:
                    remaining = await asyncio.to_thread(
                        ledger.fetch_pending_blocks,
                        actual_job_id,
                        limit=1,
                        chapter_id=chapter_id,
                    )
                    if not remaining:
                        stats_after = await asyncio.to_thread(ledger.get_job_stats, actual_job_id)
                        # FAILED is terminal, so "no pending blocks" alone read an
                        # all-failed batch as a clean success. Require delivered
                        # work (drafted / completed / flagged-for-review) before
                        # logging success; an empty delta means every line failed.
                        delivered = sum(
                            stats_after[key] - stats_before[key]
                            for key in (
                                "drafted",
                                "completed",
                                "needs_human",
                                "blocked_human",
                            )
                        )
                        if delivered <= 0:
                            raise UBTError(
                                f"Whole-book offline batch drafted 0 of "
                                f"{len(all_pending_blocks)} block(s) for {chapter_id}: "
                                "the provider rejected every line "
                                f"(failed={stats_after['failed']})"
                            )
                        logger.info(
                            "Whole-book offline batch draft succeeded: %d block(s) drafted",
                            delivered,
                        )
                        if memory_mgr.should_trigger_snapshot():
                            try:
                                await memory_mgr.generate_step_snapshot(
                                    router.complete_raw, target_lang=target_lang
                                )
                            except Exception as exc:
                                logger.warning(
                                    "Hierarchical step snapshot generation failed: %s", exc
                                )
                            if rolling_enabled:
                                await asyncio.to_thread(
                                    _persist_memory_state,
                                    ledger,
                                    actual_job_id,
                                    memory_mgr,
                                )
                        event = await create_event_fn(
                            EventType.DRAFT_BATCH_COMPLETED,
                            actual_job_id,
                            ledger,
                            message=f"Whole-book offline batch completed for {len(all_pending_blocks)} blocks",
                            active_block_id=all_pending_blocks[-1].id,
                        )
                        yield event
                        return

        cursor_spine: int | None = None
        cursor_block_id: str | None = None
        while True:
            ctx.check_cancelled()
            batch = await asyncio.to_thread(
                ledger.fetch_pending_blocks,
                actual_job_id,
                limit=batch_limit,
                after_spine_index=cursor_spine,
                after_block_id=cursor_block_id,
                chapter_id=chapter_id,
            )
            if not batch:
                break

            cursor_spine = batch[-1].spine_index
            cursor_block_id = batch[-1].id

            # No summarizer runs here: the hierarchical L2 manager snapshots on
            # step threshold and chapter transition itself, so the batch is
            # drafted as one claim.
            await processor.run_draft_batch(batch, batch)

            if processor.runtime.fail_fast_consecutive >= processor.policy.fail_fast_threshold:
                # The same unrecoverable provider error has now killed
                # `threshold` blocks back-to-back. Every remaining block would
                # repeat that burn for nothing, so abort the job with the
                # original reason (surfaces via the pipeline's failure handler)
                # instead of silently failing thousands of blocks.
                raise UBTError(
                    f"Draft aborted after {processor.runtime.fail_fast_consecutive} consecutive "
                    f"non-retryable provider failures (fail-fast circuit). "
                    f"Last error: {processor.runtime.last_fail_fast_reason}"
                )

            if memory_mgr.should_trigger_snapshot():
                try:
                    await memory_mgr.generate_step_snapshot(
                        router.complete_raw, target_lang=target_lang
                    )
                except Exception as exc:
                    logger.warning("Hierarchical step snapshot generation failed: %s", exc)
                if rolling_enabled:
                    await asyncio.to_thread(
                        _persist_memory_state, ledger, actual_job_id, memory_mgr
                    )

            await flusher.flush_all()
            event = await create_event_fn(
                EventType.DRAFT_BATCH_COMPLETED,
                actual_job_id,
                ledger,
                message=f"Drafted blocks through spine index {batch[-1].spine_index}",
                active_block_id=batch[-1].id,
            )
            yield event
    finally:
        await flusher.close()
        if rolling_enabled:
            await asyncio.to_thread(_persist_memory_state, ledger, actual_job_id, memory_mgr)
        memory_mgr.discard_pending_blocks()
        if (
            counters["tm_exact_hits"]
            or counters["batch_drafted"]
            or counters["batch_fallbacks"]
            or counters["translate_cache_hits"]
        ):
            logger.info(
                "Draft stage summary for %s: tm_exact_hits=%d batch_drafted=%d "
                "batch_fallbacks=%d translate_cache_hits=%d",
                actual_job_id,
                counters["tm_exact_hits"],
                counters["batch_drafted"],
                counters["batch_fallbacks"],
                counters["translate_cache_hits"],
            )
