"""Streaming draft translation with keyset pagination."""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import random
import re
from collections.abc import AsyncGenerator, Sequence
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
from ubt.core.exceptions import (
    BudgetExceededError,
    JobInterruptedError,
    LedgerWriterLeaseLostError,
    ProviderUnreachableError,
    UBTError,
)
from ubt.core.ir.document_tree import section_context_for
from ubt.core.ir.emphasis import (
    mark_emphasis_spans,
    parse_emphasis,
    runs_from_json,
)
from ubt.core.ir.models import BlockStatus, BlockType, BookManifest, IRBlock, StyleMeta
from ubt.core.memory.abbreviation_miner import format_abbreviations_markdown_table
from ubt.core.memory.glossary_table import (
    build_chunk_glossary_table,
    build_global_glossary_table,
)
from ubt.core.memory.hierarchical_memory import (
    DEFAULT_MAX_EPOCHS,
    HierarchicalMemoryManager,
)
from ubt.core.memory.rolling_summary import extract_chapter_id
from ubt.core.memory.tm import (
    PROMPT_VERSION,
    TranslationMemory,
    compute_tm_context,
    format_few_shot_reference,
)
from ubt.core.qe.defect_taxonomy import (
    DRAFTING_ERROR_PREFIX,
    NON_RETRYABLE_DRAFT_PREFIX,
    RETRYABLE_CONFIG_DRAFT_PREFIX,
)
from ubt.core.qe.fast_pass import FastPassFilter
from ubt.core.qe.heuristic import glossary_violation_flag
from ubt.core.router.router import (
    ModelRouter,
    classify_provider_error,
)
from ubt.core.validators.consistency import GlossaryConsistencyValidator

if TYPE_CHECKING:
    from ubt.translate.engine import TranslationEngine

logger = logging.getLogger(__name__)

#: Profiles that suppress the rolling chapter summary.
#:
#: These are the profiles whose documents are read section-by-section (a paper,
#: a textbook, a technical report), where a running prose summary of the
#: previous chapter buys nothing and costs a call per chapter. Narrative
#: profiles (``general``, ``fiction``) keep it.
#:
#: ``humanities`` is included even though ``--profile auto`` cannot currently
#: select it: ``infer_profile_from_archetype`` has a branch for it, but
#: ``detect_domain`` only ever returns a key of ``DOMAIN_PATTERNS``, and
#: ``humanities`` is not one — so that branch is unreachable today. An operator
#: can still pass ``--profile humanities`` by hand, and a humanities document is
#: argued section-by-section the way a paper is, so the classification must
#: match. Leaving it out meant the explicit and inferred paths would disagree
#: the moment the missing domain keyword is added. ``assess.py`` imports this
#: set to price the rollup, so a change here keeps the quote and the engine in
#: step by construction.
ACADEMIC_PROFILES = frozenset(
    {"academic", "textbook", "paper", "technical", "nonfiction", "non-fiction", "humanities"}
)
_SKIPPED_TYPES = frozenset({BlockType.IMAGE, BlockType.FORMULA})


def _source_with_bold_markers(block: IRBlock) -> str:
    """The source text with ``⟦B⟧…⟦/B⟧`` and ``⟦I⟧…⟦/I⟧`` around emphasized spans.

    Whole-block bold is already rendered from ``provenance["is_bold"]``; partial
    bold and italic spans need the model to carry the emphasis into the target,
    because a source-derived inline run the translation rewrote has no target
    substring for the render to locate. The markers ride the masked source and
    are parsed back out in :meth:`_DraftProcessor.finalize_draft`.
    """
    style = block.style
    if style is None:
        return block.source_text
    bold_spans = (
        []
        if block.provenance.is_bold
        else [run.text for run in style.inline_runs if run.bold and run.text.strip()]
    )
    italic_spans = (
        []
        if getattr(block.provenance, "is_italic", False)
        else [
            run.text
            for run in style.inline_runs
            if run.italic and not run.bold and run.text.strip()
        ]
    )
    if not bold_spans and not italic_spans:
        return block.source_text
    return mark_emphasis_spans(block.source_text, bold_spans=bold_spans, italic_spans=italic_spans)


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
        len(manifest.chapters) <= 1 or not config_rolling or is_page_slice or is_academic
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


def chunk_by_section(
    prepared: list[tuple[IRBlock, _DraftInputs]],
    chunk_size: int,
) -> list[list[tuple[IRBlock, _DraftInputs]]]:
    """Pack prepared blocks into LLM chunks that never cross a section boundary.

    A fixed ``chunks = prepared[i:i+size]`` split cuts sections in half. Measured
    on the arXiv 2101.03961 run with the default size 5: one chunk ended in the
    middle of the Abstract, so the model saw a heading whose prose lived in the
    *next* request. A heading and the paragraph it introduces are one translation
    unit (RFC-2026-02 §2.3); splitting them is how a draft loses the antecedent
    for a pronoun or mistranslates a title in isolation.

    Rules, in order:

    1. A heading starts a new chunk, even when the previous one has room —
       *unless* the current chunk holds nothing but headings yet. A run of
       consecutive headings is one unit (a title block: "arXiv:…", "Journal
       of…", "Submitted…", the title itself), and splitting it into four
       single-heading calls pays four times for one page.
    2. A chunk never exceeds ``chunk_size``; a section longer than that is split
       at the size limit (a long section still needs bounded prompts).
    3. A section shorter than ``chunk_size`` is *not* merged with its neighbour:
       merging would reintroduce exactly the cross-boundary packing this exists
       to prevent.

    Ordering is preserved, so the caller's index alignment with ``prepared``
    holds.
    """
    if chunk_size <= 1:
        return [[item] for item in prepared]

    chunks: list[list[tuple[IRBlock, _DraftInputs]]] = []
    current: list[tuple[IRBlock, _DraftInputs]] = []
    for item in prepared:
        block = item[0]
        starts_section = block.block_type == BlockType.HEADING
        has_prose = any(b.block_type != BlockType.HEADING for b, _ in current)
        if current and ((starts_section and has_prose) or len(current) >= chunk_size):
            chunks.append(current)
            current = []
        current.append(item)
    if current:
        chunks.append(current)
    return chunks


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
    rolling_enabled: bool
    fail_fast_threshold: int = 6
    #: Kept equal to ``UBTConfig.macro_chunk_size``'s default so a DraftPolicy
    #: built without the config kwarg does not silently disagree with production.
    macro_chunk_size: int = 8
    dynamic_routing_enabled: bool = False
    dynamic_scheduler: Any | None = None


def _draft_failure_prefix(exc: BaseException) -> str:
    """Choose the resume marker for a failed provider call.

    A missing route/model or rejected credential is not worth retrying against
    the same configuration, so the fail-fast breaker still trips. It becomes
    recoverable after the operator fixes the provider, though, so it must not
    share the permanent marker used for malformed source/content failures.
    """
    if classify_provider_error(exc).retryable:
        return DRAFTING_ERROR_PREFIX
    details = getattr(exc, "details", {})
    details = details if isinstance(details, dict) else {}
    # A missing local endpoint is a configuration problem the operator can fix
    # (start the server, correct the URL, switch provider) and then resume. It
    # must not become the permanent marker used for malformed source content.
    if details.get("reason") == "transport_unreachable":
        return RETRYABLE_CONFIG_DRAFT_PREFIX
    status = details.get("status_code")
    if status in (401, 403, 404):
        return RETRYABLE_CONFIG_DRAFT_PREFIX
    return NON_RETRYABLE_DRAFT_PREFIX


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

    def chunk_context(
        self, chunk: list[tuple[IRBlock, _DraftInputs]], section_context: str = ""
    ) -> str:
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
        from ubt.core.memory.neighbor_window import NeighborContextBuilder

        chunk_neighbor_ctx = NeighborContextBuilder.combine_chunk_neighbor_contexts(
            first_inp.neighbor_ctx, last_inp.neighbor_ctx
        )
        system, user = build_macro_chunk_draft_prompt(
            blocks=[(b.id, inp.masked_source) for b, inp in chunk],
            glossary_table=merged_glossary,
            neighbor_context=chunk_neighbor_ctx,
            target_lang=self.policy.target_lang,
            source_lang=self.policy.source_lang,
            genre_profile=self.policy.profile_name,
            rolling_summary=first_inp.macro_ctx,
            global_glossary=self.policy.global_glossary_table,
            few_shot_reference=first_inp.few_shot_reference or last_inp.few_shot_reference,
            epoch_summary=first_inp.epoch_ctx,
            domain=self.policy.domain,
            section_context=section_context,
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

        # Resolve cross-chapter and inter-block coreference disambiguation hints
        from ubt.core.memory.coreference import CoreferenceResolver

        coref_resolver = CoreferenceResolver()
        get_entities_fn = getattr(self.runtime.memory_mgr, "get_salient_entities", None)
        salient_entities = get_entities_fn() if callable(get_entities_fn) else []
        glossary_srcs = [
            str(g.get("source", ""))
            for g in getattr(self.policy, "glossary_dicts", [])
            if g.get("source")
        ]
        coref_mentions = coref_resolver.resolve_mentions(
            text=block.source_text,
            neighbor_text=neighbor_ctx,
            epoch_entities=salient_entities,
            chapter_title=current_ch or "",
            glossary_terms=glossary_srcs,
        )
        coref_hint = coref_resolver.format_coreference_prompt_hint(coref_mentions)
        if coref_hint:
            neighbor_ctx = f"{neighbor_ctx}\n\n{coref_hint}".strip() if neighbor_ctx else coref_hint

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

    async def prepare_blocks(
        self,
        blocks: Sequence[IRBlock],
        current_batch: list[IRBlock],
    ) -> list[tuple[IRBlock, _DraftInputs]]:
        """Prepare every translatable block of a batch, concurrently.

        Per-block preparation is a chain of thread off-loads (the TM exact and
        fuzzy lookups, and on a TM hit the FastPass screen and the whole-glossary
        walk, plus the neighbor tails and the mask). Run in a sequential loop it
        is pure serialized wall clock ahead of the batch's LLM gather, so it is
        fanned out here instead: results keep spine order (``gather`` preserves
        it) and the first failure in spine order is re-raised after the gather
        settles, so the batch still fails the way the sequential loop did.

        One deliberate ordering difference: a sibling prep that already reached
        its TM-exact checkpoint enqueue completes even when another prep raises,
        where the sequential loop never started the later blocks. Such a
        checkpoint is a terminal-correct state (``MTQE_PASSED`` carrying the TM
        target), so landing it early is progress a resume would have made anyway,
        never a state the failure path has to undo.
        """
        todo: list[IRBlock] = []
        for b in blocks:
            if is_already_final(b):
                continue
            if is_static_skip(b):
                await self.draft_single_block(b, current_batch)
                continue
            todo.append(b)
        if not todo:
            return []
        outcomes = await asyncio.gather(
            *(self.prepare_draft_inputs(b, current_batch) for b in todo),
            return_exceptions=True,
        )
        prepared: list[tuple[IRBlock, _DraftInputs]] = []
        first_error: BaseException | None = None
        for b, outcome in zip(todo, outcomes, strict=True):
            if isinstance(outcome, BaseException):
                if first_error is None:
                    first_error = outcome
                continue
            if outcome is not None:
                prepared.append((b, outcome))
        if first_error is not None:
            raise first_error
        return prepared

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
        # The model bracketed emphasized target text with ⟦B⟧/⟦I⟧; parse the
        # runs out (and strip every marker, balanced or not) so the target text
        # is always clean and the render can re-apply the weight the source-
        # derived inline run cannot locate.
        final_draft, target_runs = parse_emphasis(result.text)
        if block.block_type == BlockType.HEADING:
            trimmed = trim_heading_expansion(block.source_text, final_draft)
            if trimmed != final_draft:
                logger.info(
                    "Heading %s draft carried an appended paragraph; kept the leading one",
                    block.id,
                )
                final_draft = trimmed

        error_flags: list[str] = list(block.error_flags)
        # A restore that is not clean (missing / mismatched / mutated /
        # duplicated span) means the target is not trustworthy; record the
        # evidence on the block instead of shipping it. A *reordered* restore is
        # NOT dirty (natural cross-language reordering moves placeholders
        # legitimately — see ``UnmaskReport.clean``); it is carried in the report
        # for audit only.
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
                model_override: str | None = None
                if getattr(self.policy, "dynamic_routing_enabled", False) and getattr(
                    self.policy, "dynamic_scheduler", None
                ):
                    glossary_sources = [
                        str(g.get("source", ""))
                        for g in getattr(self.policy, "glossary_dicts", [])
                        if g.get("source")
                    ]
                    assignment = self.policy.dynamic_scheduler.resolve_model_for_block(
                        block,
                        glossary_terms=glossary_sources,
                        is_retry_or_repair=False,
                    )
                    model_override = assignment.assigned_model

                for attempt in range(self.policy.draft_max_retries + 1):
                    current_model = model_override
                    if (
                        attempt > 0
                        and getattr(self.policy, "dynamic_routing_enabled", False)
                        and getattr(self.policy, "dynamic_scheduler", None)
                    ):
                        glossary_sources = [
                            str(g.get("source", ""))
                            for g in getattr(self.policy, "glossary_dicts", [])
                            if g.get("source")
                        ]
                        escalated = self.policy.dynamic_scheduler.resolve_model_for_block(
                            block,
                            glossary_terms=glossary_sources,
                            is_retry_or_repair=True,
                        )
                        current_model = escalated.assigned_model

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
                            model=current_model,
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
            except (
                asyncio.CancelledError,
                JobInterruptedError,
                BudgetExceededError,
                LedgerWriterLeaseLostError,
            ):
                raise
            except (AssertionError, TypeError, AttributeError, NameError, UnboundLocalError) as exc:
                logger.critical(
                    "Programming defect encountered during drafting for block %s: %s",
                    block.id,
                    exc,
                    exc_info=True,
                )
                raise
            except Exception as exc:
                retryable = classify_provider_error(exc).retryable
                if retryable:
                    self.runtime.fail_fast_consecutive = 0
                else:
                    self.runtime.fail_fast_consecutive += 1
                    self.runtime.last_fail_fast_reason = str(exc)
                # A non-retryable failure is recorded under a prefix the resume
                # path does NOT treat as transient: the same 401/402/400 recurs
                # on every run, so re-queueing the block only re-bills it.
                prefix = _draft_failure_prefix(exc)
                fail_update = {
                    "block_id": block.id,
                    "status": BlockStatus.FAILED,
                    "error_flags": [f"{prefix} {exc}"],
                }
                await self.runtime.flusher.enqueue(fail_update)

    async def draft_macro_chunk_group(
        self,
        chunk: list[tuple[IRBlock, _DraftInputs]],
        current_batch: list[IRBlock],
        section_context: str = "",
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
            cached_context = self.chunk_context(chunk, section_context)
            cached_raw = await asyncio.to_thread(
                self.runtime.engine.cached_value, cached_context, kind="translate_chunk"
            )
            extracted_by_id = _decode_chunk(cached_raw)

        from ubt.core.memory.neighbor_window import NeighborContextBuilder

        chunk_neighbor_ctx = NeighborContextBuilder.combine_chunk_neighbor_contexts(
            first_inp.neighbor_ctx, last_inp.neighbor_ctx
        )
        if not extracted_by_id:
            async with self.runtime.concurrency_sem:
                blocks_to_draft = [b.with_source_text(inp.masked_source) for b, inp in chunk]
                chunk_model_override: str | None = None
                if getattr(self.policy, "dynamic_routing_enabled", False) and getattr(
                    self.policy, "dynamic_scheduler", None
                ):
                    glossary_sources = [
                        str(g.get("source", ""))
                        for g in getattr(self.policy, "glossary_dicts", [])
                        if g.get("source")
                    ]
                    assignments = [
                        self.policy.dynamic_scheduler.resolve_model_for_block(
                            b, glossary_terms=glossary_sources, is_retry_or_repair=False
                        )
                        for b in blocks_to_draft
                    ]
                    tier_ranks = {"trivial": 1, "standard": 2, "complex": 3, "extreme": 4}
                    chunk_model_override = max(
                        assignments, key=lambda a: tier_ranks.get(a.tier.value, 2)
                    ).assigned_model

                try:
                    for attempt in range(self.policy.draft_max_retries + 1):
                        current_chunk_model = chunk_model_override
                        if (
                            attempt > 0
                            and getattr(self.policy, "dynamic_routing_enabled", False)
                            and getattr(self.policy, "dynamic_scheduler", None)
                        ):
                            glossary_sources = [
                                str(g.get("source", ""))
                                for g in getattr(self.policy, "glossary_dicts", [])
                                if g.get("source")
                            ]
                            escalated_assignments = [
                                self.policy.dynamic_scheduler.resolve_model_for_block(
                                    b, glossary_terms=glossary_sources, is_retry_or_repair=True
                                )
                                for b in blocks_to_draft
                            ]
                            tier_ranks = {"trivial": 1, "standard": 2, "complex": 3, "extreme": 4}
                            current_chunk_model = max(
                                escalated_assignments,
                                key=lambda a: tier_ranks.get(a.tier.value, 2),
                            ).assigned_model

                        try:
                            extracted_by_id = await self.runtime.router.draft_macro_chunk(
                                blocks=blocks_to_draft,
                                glossary_table=merged_glossary,
                                neighbor_context=chunk_neighbor_ctx,
                                target_lang=self.policy.target_lang,
                                source_lang=self.policy.source_lang,
                                genre_profile=self.policy.profile_name,
                                domain=self.policy.domain,
                                rolling_summary=macro_ctx,
                                epoch_summary=epoch_ctx,
                                global_glossary=self.policy.global_glossary_table,
                                few_shot_reference=few_shot,
                                section_context=section_context,
                                model=current_chunk_model,
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
                except (
                    asyncio.CancelledError,
                    JobInterruptedError,
                    BudgetExceededError,
                    LedgerWriterLeaseLostError,
                ):
                    raise
                except (
                    AssertionError,
                    TypeError,
                    AttributeError,
                    NameError,
                    UnboundLocalError,
                ) as exc:
                    logger.critical(
                        "Programming defect encountered during macro-chunk drafting: %s",
                        exc,
                        exc_info=True,
                    )
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
                # Per-block fault isolation. ``finalize_draft`` checkpoints a
                # block the moment it succeeds, so a later block's finalize
                # raising must mark ONLY that block. Letting the exception escape
                # to ``run_draft_batch``'s gather marked the whole group FAILED —
                # including blocks already drafted and paid for — and the resume
                # path then cleared their target text and re-billed them.
                try:
                    clean_by_id[b.id] = await self.finalize_draft(b, text.strip(), inp)
                except (
                    asyncio.CancelledError,
                    JobInterruptedError,
                    BudgetExceededError,
                    LedgerWriterLeaseLostError,
                ):
                    raise
                except (
                    AssertionError,
                    TypeError,
                    AttributeError,
                    NameError,
                    UnboundLocalError,
                ) as exc:
                    logger.critical(
                        "Programming defect encountered finalizing block %s: %s",
                        b.id,
                        exc,
                        exc_info=True,
                    )
                    raise
                except Exception as exc:
                    logger.warning(
                        "Finalizing macro-chunk block %s failed (%s); marking only it",
                        b.id,
                        exc,
                        exc_info=True,
                    )
                    prefix = _draft_failure_prefix(exc)
                    await self.runtime.flusher.enqueue(
                        {
                            "block_id": b.id,
                            "status": BlockStatus.FAILED,
                            "error_flags": [f"{prefix} {exc}"],
                        }
                    )
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
            try:
                await asyncio.to_thread(
                    self.runtime.engine.remember_value,
                    cached_context,
                    _encode_chunk(extracted_by_id),
                    kind="translate_chunk",
                )
            except Exception as exc:
                # The drafts are already checkpointed; a cache-write failure must
                # not escape and mark the whole group FAILED.
                logger.warning("Caching macro-chunk draft failed for %s: %s", cached_context, exc)

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
        prepared = await self.prepare_blocks(blocks, current_batch)

        if not prepared:
            return

        if self.policy.macro_chunk_size <= 1:
            groups: list[list[IRBlock]] = [[b] for b, _ in prepared]
            results = await asyncio.gather(
                *[self.draft_single_block(b, current_batch, inp) for b, inp in prepared],
                return_exceptions=True,
            )
        else:
            chunks = chunk_by_section(prepared, self.policy.macro_chunk_size)
            groups = [[b for b, _ in chunk] for chunk in chunks]
            # Where each chunk sits in the document. A chunk that does not hold
            # its own heading otherwise reaches the model with no idea which
            # section it belongs to, and a back-reference loses its antecedent.
            section_ctx = {
                chunk[0][0].id: section_context_for(blocks, groups[i])
                for i, chunk in enumerate(chunks)
            }
            results = await asyncio.gather(
                *[
                    self.draft_macro_chunk_group(
                        chunk,
                        current_batch,
                        section_context=section_ctx.get(chunk[0][0].id, ""),
                    )
                    for chunk in chunks
                ],
                return_exceptions=True,
            )

        critical_exc: BaseException | None = None
        stranded: list[dict[str, Any]] = []
        for res, group in zip(results, groups, strict=True):
            if isinstance(
                res,
                (
                    BudgetExceededError,
                    JobInterruptedError,
                    asyncio.CancelledError,
                    LedgerWriterLeaseLostError,
                ),
            ):
                if critical_exc is None:
                    critical_exc = res
                continue
            if isinstance(res, Exception):
                logger.warning("Draft task failed with exception: %s", res)
                prefix = _draft_failure_prefix(res)
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


def _assert_no_stranded_pending(
    ledger: SQLiteJobLedger, actual_job_id: str, batch: Sequence[IRBlock]
) -> None:
    """Fail loud if a fetched block is still ``pending`` after the batch flush.

    The interactive loop claims blocks with a monotone keyset cursor
    (``after_spine_index``/``after_block_id``): once a batch is consumed, the
    cursor never returns. Every consumed block must therefore have left
    ``pending`` — drafted, passed, failed, or requeued — or it is stranded
    behind the cursor forever and only export's fail_non_terminal gate would
    surface it, after the money was spent. One COUNT over the batch's ids.
    """
    if not batch:
        return
    placeholders = ",".join("?" * len(batch))
    with ledger._get_conn() as conn:
        row = conn.execute(
            f"""
            SELECT block_id FROM blocks
            WHERE job_id = ? AND status = 'pending'
              AND block_id IN ({placeholders})
            LIMIT 3
            """,
            (actual_job_id, *[b.id for b in batch]),
        ).fetchone()
    if row is not None:
        raise UBTError(
            f"Draft batch left block {row['block_id']!r} pending after its writes "
            f"were flushed (job {actual_job_id}). The keyset cursor has already "
            "moved past it, so no later batch would revisit it. This is a "
            "draft-stage bug: every fetched block must be checkpointed "
            "(drafted / passed / failed / requeued) before the batch completes."
        )


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
        logger.warning(
            "Failed to persist rolling memory state for %s: %s", actual_job_id, exc, exc_info=True
        )


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
        logger.warning(
            "Failed to restore rolling memory state for %s: %s", actual_job_id, exc, exc_info=True
        )
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
        max_epochs=getattr(config, "memory_max_epochs", DEFAULT_MAX_EPOCHS),
        sliding_window=getattr(config, "memory_sliding_window", True),
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
    # Teach the omission gate the *shape* of the source document. A table sets an
    # ordinary word in capitals as a column label (``MASK | SAME | RND``), which
    # is indistinguishable from an acronym (``MHA``) from one block alone; with
    # the wider document the gate can see the word used in lower case elsewhere
    # and stop demanding the English surface in the translation. Measured on
    # ``bert`` ``pdf_main#b0256``: without this the untranslated source passed
    # every check while the correct Chinese draft was rejected as an omission,
    # and the repair shipped the English source instead.
    #
    # Read once per RUN, not once per chapter: ``source_text`` is immutable
    # after ingest, and the previous per-chapter ``get_all_blocks`` cost 346 ms
    # on a 20k-block book (329 ms of it ``IRBlock`` construction nobody used)
    # for a string that never changed. The index is also what the gate stage
    # and the omission gate share, so the book is read and indexed exactly once.
    source_index = getattr(services, "source_document", None)
    if source_index is not None:
        await asyncio.to_thread(source_index.load, ledger, actual_job_id)
        active_fast_pass.set_document_vocabulary(source_index.vocabulary())
    else:  # a caller built RunServices without the shared index
        document_source = "\n".join(
            block.source_text or ""
            for block in await asyncio.to_thread(ledger.get_all_blocks, actual_job_id)
        )
        active_fast_pass.set_document_context(document_source)
    counters = {
        "tm_exact_hits": 0,
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

    dynamic_scheduler = None
    if getattr(config, "enable_dynamic_routing", False):
        from ubt.core.router.dynamic_scheduler import (
            DynamicParetoScheduler,
            DynamicRoutingPolicy,
        )

        policy_raw = getattr(config, "dynamic_routing_policy", "pareto")
        try:
            policy_enum = DynamicRoutingPolicy(policy_raw)
        except Exception:
            policy_enum = DynamicRoutingPolicy.PARETO

        budget_m = getattr(config, "budget_model", None) or router.draft_model
        std_m = router.draft_model
        flagship_m = (
            getattr(config, "flagship_model", None) or router.repair_model or "claude-3-7-sonnet"
        )
        dynamic_scheduler = DynamicParetoScheduler(
            policy=policy_enum,
            budget_model=budget_m,
            standard_model=std_m,
            flagship_model=flagship_m,
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
            rolling_enabled=rolling_enabled,
            macro_chunk_size=max(1, int(config.macro_chunk_size)),
            dynamic_routing_enabled=bool(dynamic_scheduler is not None),
            dynamic_scheduler=dynamic_scheduler,
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
) -> AsyncGenerator[TranslationProgressEvent, None]:
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

            # Chapter transition, checked BEFORE the batch is drafted.
            #
            # ``should_trigger_snapshot`` has two triggers and the chapter one
            # needs the *incoming* block; the production loop called it with no
            # argument at every site, so that half was dead code and a chapter
            # boundary shorter than ``step_chars`` produced no summary until the
            # next threshold — or never. Summarizing here means the blocks about
            # to be drafted see the finished chapter's macro context, which is
            # the whole point of the transition trigger.
            if memory_mgr.should_trigger_snapshot(batch[0]):
                try:
                    await memory_mgr.generate_step_snapshot(
                        router.complete_raw, target_lang=target_lang
                    )
                except Exception as exc:
                    logger.warning(
                        "Hierarchical step snapshot generation failed: %s", exc, exc_info=True
                    )
                if rolling_enabled:
                    await asyncio.to_thread(
                        _persist_memory_state, ledger, actual_job_id, memory_mgr
                    )

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
                raise ProviderUnreachableError(
                    f"Draft aborted after {processor.runtime.fail_fast_consecutive} consecutive "
                    f"non-retryable provider failures (fail-fast circuit). "
                    f"Last error: {processor.runtime.last_fail_fast_reason}"
                )

            # Step-threshold trigger, checked after the batch: the buffer now
            # holds this batch's drafts, and the next batch has not been fetched.
            if memory_mgr.should_trigger_snapshot():
                try:
                    await memory_mgr.generate_step_snapshot(
                        router.complete_raw, target_lang=target_lang
                    )
                except Exception as exc:
                    logger.warning(
                        "Hierarchical step snapshot generation failed: %s", exc, exc_info=True
                    )
                if rolling_enabled:
                    await asyncio.to_thread(
                        _persist_memory_state, ledger, actual_job_id, memory_mgr
                    )

            await flusher.flush_all()
            # The stranded-block invariant: every block the loop FETCHED must
            # have left ``pending`` by the time the batch's writes are flushed.
            # Each early-exit path (TM hit, static skip, per-block failure)
            # enqueues a checkpoint or a FAILED record; if a future path
            # returns without one, the keyset cursor has already moved past
            # the block, so it would sit pending-unvisited until export
            # force-FAILED it — the cheapest place to catch that is here,
            # fail-loud, one COUNT per batch.
            #
            # A batch parked for retry is NOT stranded: ``flush_all`` retains it
            # on a transient SQLite failure by design, and the background
            # worker owns the retry budget. Asserting in that window aborted
            # the stage on exactly the condition the flusher exists to survive.
            if flusher.has_pending_retries:
                logger.warning(
                    "Draft batch for %s left %d checkpoint(s) queued for retry; "
                    "skipping the stranded-block assertion this batch",
                    actual_job_id,
                    flusher.pending_retry_count,
                )
            else:
                _assert_no_stranded_pending(ledger, actual_job_id, batch)
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
            # Chapter end: flush what the step threshold never reached.
            #
            # Under ``chapter_streaming`` each chapter gets its own
            # ``run_draft_stage`` call and therefore its own
            # ``HierarchicalMemoryManager``; the next chapter builds a fresh one
            # and restores only what was persisted. A chapter (or a chapter's
            # tail) shorter than ``step_chars`` never tripped
            # ``should_trigger_snapshot``, so its whole buffer was dropped here
            # and no summary for it ever existed — the L2 layer simply had
            # holes at every short chapter boundary. Summarize first, then
            # discard; a failure keeps the old behaviour rather than failing the
            # stage (the summaries are context, not delivery).
            if memory_mgr.pending_block_count:
                try:
                    await memory_mgr.generate_step_snapshot(
                        router.complete_raw, target_lang=target_lang
                    )
                except Exception as exc:
                    logger.warning(
                        "Chapter-end hierarchical snapshot failed for %s: %s",
                        actual_job_id,
                        exc,
                        exc_info=True,
                    )
            await asyncio.to_thread(_persist_memory_state, ledger, actual_job_id, memory_mgr)
        memory_mgr.discard_pending_blocks()
        if counters["tm_exact_hits"] or counters["translate_cache_hits"]:
            logger.info(
                "Draft stage summary for %s: tm_exact_hits=%d translate_cache_hits=%d",
                actual_job_id,
                counters["tm_exact_hits"],
                counters["translate_cache_hits"],
            )
