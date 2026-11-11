"""Three tiers of translation context, each paid for differently.

L1 is a local neighbour window around the block being drafted, L2 a rolling
macro snapshot taken every ``step_chars`` or at a chapter boundary, L3 a
deterministic compression of several L2 snapshots.

The split is a cacheability trade, not a granularity preference: L3 is written
once per closed epoch and rides the *static* prompt prefix right after the
global glossary, so chapter 30 can know what happened in chapters 1-10 without
paying per-step prefix-cache invalidation. L2 changes on every step, so it never
enters the prefix.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from ubt.core.ir.models import BlockType, IRBlock, make_element
from ubt.core.memory.neighbor_window import DEFAULT_NEIGHBOR_CHARS, NeighborContextBuilder
from ubt.core.memory.rolling_summary import (
    collect_chapter_text,
    deterministic_epoch_summary,
    deterministic_summary,
    extract_chapter_id,
    llm_summary,
)

logger = logging.getLogger(__name__)

DEFAULT_STEP_CHARS = 3500
DEFAULT_L3_EPOCH_STEPS = 4
DEFAULT_MAX_EPOCHS = 50
_MAX_L3_CHARS = 1200
# A chapter transition forces a macro snapshot only once the buffer holds real
# content; below this the transition folds into the next regular snapshot.
_SNAPSHOT_MIN_BUFFER_CHARS = 300
#: Upper bound on how many window snapshots the post-repair chapter refresh
#: may emit. A chapter much longer than this many ``step_chars`` windows gets
#: proportionally wider windows instead of unbounded LLM calls.
_MAX_REFRESH_SNAPSHOTS = 12

#: Residual placeholder tokens must never reach a summary: a draft that kept
#: its ``⟦FAM_MASK_0001-abc⟧`` wrappers is corrupt, and feeding the token into
#: the recap would teach later prompts the token is content. QE owns the
#: authoritative version (``fast_pass._RESIDUAL_MASK_TOKEN_RE``); this is the
#: cheap pre-summary screen, kept local so the memory layer does not import
#: the QE module graph.
_MEMORY_MASK_TOKEN_RE = re.compile(r"[⟦\[【]?[A-Z]+_MASK_\d{4}-[0-9a-z]{1,3}[⟧\]】]?")

#: Per-job serialization for the persisted ``memory_state`` read-modify-write.
#: Under chapter streaming the draft worker (chapter N+1) and the repair
#: worker's post-repair ``refresh_chapter_memory`` (chapter N) run concurrently
#: in one process; without this lock the repair refresh's read→replace→write
#: interleaves with the draft worker's absolute persist and last-writer-wins
#: discards one side's snapshots. Cross-process interference is already
#: excluded by the writer lease (one writer process per job).
_MEMORY_STATE_LOCKS: dict[str, asyncio.Lock] = {}


def memory_state_lock(job_id: str) -> asyncio.Lock:
    """The per-job lock guarding ``memory_state`` read-modify-write sequences."""
    lock = _MEMORY_STATE_LOCKS.get(job_id)
    if lock is None:
        lock = asyncio.Lock()
        _MEMORY_STATE_LOCKS[job_id] = lock
    return lock


def _summary_safe_text(block: IRBlock) -> IRBlock:
    """Return a buffer copy whose text is trustworthy for summarization.

    A draft that still carries residual mask tokens, or that is a verbatim
    echo of its source, is corrupt as *translation* — and the buffer feeds the
    L2/L3 recap that conditions every later prompt. Summarizing the corruption
    would propagate it (the finding that motivated this screen); dropping the
    block would punch a narrative hole in the recap. Falling back to the block's
    source text keeps the storyline intact while feeding only trustworthy text.
    The ledger block itself is untouched — QE gates the real draft.
    """
    text = block.target_text or block.draft_text or ""
    if not text:
        return block
    suspicious = bool(_MEMORY_MASK_TOKEN_RE.search(text)) or (
        block.source_text.strip() != "" and text.strip() == block.source_text.strip()
    )
    if not suspicious:
        return block
    safe = block.model_copy(deep=True)
    safe.target_text = None
    safe.draft_text = None
    return safe


def _buffer_block_to_dict(block: IRBlock) -> dict[str, Any]:
    """Compact serialization of one buffered block (see :meth:`export_state`)."""
    return {
        "id": block.id,
        "spine_index": block.spine_index,
        "block_type": block.block_type.value if block.block_type else None,
        "source_text": block.source_text,
        "draft_text": block.draft_text,
        "target_text": block.target_text,
    }


def _buffer_block_from_dict(data: Any) -> IRBlock | None:
    """Rebuild one buffered block from :meth:`export_state` output.

    Only the fields the summary path reads are reconstructed — the stand-in is
    never rendered or checkpointed, it exists solely so the interrupted step's
    drafted text reaches the next snapshot.
    """
    if not isinstance(data, dict):
        return None
    try:
        block_type = (
            BlockType(str(data["block_type"])) if data.get("block_type") else BlockType.NARRATIVE
        )
        element = make_element(
            id=str(data["id"]),
            spine_index=int(data["spine_index"]),
            block_type=block_type,
            source_text=str(data.get("source_text") or ""),
        )
    except (KeyError, TypeError, ValueError):
        return None
    block = IRBlock(element=element)
    block.draft_text = data.get("draft_text") or None
    block.target_text = data.get("target_text") or None
    return block


@dataclass(slots=True)
class StepSnapshot:
    """A point-in-time macro summary snapshot covering a sequence of blocks."""

    step_index: int
    start_spine: int
    end_spine: int
    chapter_id: str
    summary_text: str
    char_count: int


@dataclass(slots=True)
class EpochSnapshot:
    """A compressed L3 summary over ``l3_epoch_steps`` L2 macro snapshots."""

    epoch_index: int
    start_spine: int
    end_spine: int
    summary_text: str


class HierarchicalMemoryManager:
    """Three-tier memory: L1 neighbour context, L2 macro step snapshots, L3 epoch summaries.

    ``enabled`` is the draft policy's ``rolling_enabled`` decision (see
    :func:`ubt.core.engine.stages.draft.resolve_draft_policy`): a single-chapter
    document, an over-long book, an academic profile, or a page-slice ingest
    gets no hierarchical memory at all. Before this flag, those routes still
    buffered blocks and paid the L2 ``llm_summary`` calls while discarding the
    L2 ``macro_ctx`` slot -- L3 alone rode the prompt, at L2's price.
    """

    def __init__(
        self,
        step_chars: int = DEFAULT_STEP_CHARS,
        neighbor_chars: int = DEFAULT_NEIGHBOR_CHARS,
        l3_epoch_steps: int = DEFAULT_L3_EPOCH_STEPS,
        enabled: bool = True,
        max_epochs: int | None = DEFAULT_MAX_EPOCHS,
        max_l3_chars: int = _MAX_L3_CHARS,
        sliding_window: bool = False,
    ) -> None:
        self.step_chars = step_chars
        self.neighbor_builder = NeighborContextBuilder(neighbor_chars=neighbor_chars)
        self.l3_epoch_steps = max(1, l3_epoch_steps)
        self.enabled = enabled
        self.max_epochs = max_epochs
        self.max_l3_chars = max(1, max_l3_chars)
        self.sliding_window = sliding_window

        self._snapshots: list[StepSnapshot] = []
        self._epochs: list[EpochSnapshot] = []
        self._unsummarized_blocks: list[IRBlock] = []
        self._current_buffer_chars: int = 0
        self._step_counter: int = 0

    @property
    def snapshots(self) -> list[StepSnapshot]:
        """All recorded macro snapshots."""
        return list(self._snapshots)

    @property
    def epochs(self) -> list[EpochSnapshot]:
        """All recorded L3 epoch summaries."""
        return list(self._epochs)

    def get_l1_context(
        self,
        target_block: IRBlock,
        surrounding_blocks: list[IRBlock],
        fallback_prev_text: str | None = None,
        fallback_next_text: str | None = None,
    ) -> str:
        """Neighbour context for ``target_block``, isolated to its own semantic flow.

        The flow filter lives in
        :meth:`NeighborContextBuilder.extract_from_blocks`; the fallbacks cover
        the ends of a window the caller had to truncate (a chapter boundary, a
        page slice), where the true neighbour is not in ``surrounding_blocks``.
        """
        return self.neighbor_builder.extract_from_blocks(
            target_block=target_block,
            surrounding_blocks=surrounding_blocks,
            fallback_prev_text=fallback_prev_text,
            fallback_next_text=fallback_next_text,
        )

    def record_drafted_block(self, block: IRBlock) -> None:
        """Buffer a newly drafted or finalized block for macro snapshot tracking.

        Deduplicated by block id: a resumed run restores the interrupted step's
        buffer from persisted state (so already-drafted blocks are present) and
        a re-draft of the same block must update the entry, not duplicate it.
        The text is screened before entering the buffer — corrupt translations
        summarize into corruption that conditions every later prompt.
        """
        if not self.enabled:
            return
        safe = _summary_safe_text(block)
        for i, existing in enumerate(self._unsummarized_blocks):
            if existing.id == safe.id:
                self._unsummarized_blocks[i] = safe
                self._recount_buffer_chars()
                return
        self._unsummarized_blocks.append(safe)
        text = safe.target_text or safe.draft_text or safe.source_text or ""
        self._current_buffer_chars += len(text)

    def _recount_buffer_chars(self) -> None:
        """Recompute the buffer's char counter from its entries."""
        self._current_buffer_chars = sum(
            len(b.target_text or b.draft_text or b.source_text or "")
            for b in self._unsummarized_blocks
        )

    def discard_pending_blocks(self) -> None:
        """Drop buffered blocks without summarizing (shutdown/teardown path).

        Clears the char counter too — the same pairing the post-snapshot reset
        uses — so a reused manager never inherits stale buffer accounting.
        """
        self._unsummarized_blocks.clear()
        self._current_buffer_chars = 0

    @property
    def pending_block_count(self) -> int:
        """How many blocks are buffered but not yet summarized.

        Read by the draft stage's chapter-end path: under chapter streaming the
        manager is per-stage-call, so a chapter shorter than ``step_chars`` ends
        with its whole buffer pending. Discarding it there lost the summary
        permanently (nothing replays those blocks into a later manager).
        """
        return len(self._unsummarized_blocks)

    def should_trigger_snapshot(self, next_block: IRBlock | None = None) -> bool:
        """Whether the buffer must be recapped now.

        Either the step threshold is reached, or ``next_block`` starts a new
        chapter and the buffer already holds enough text to be worth cutting a
        snapshot here — below :data:`_SNAPSHOT_MIN_BUFFER_CHARS` a boundary
        simply folds into the next regular snapshot rather than producing a
        recap of two blocks.
        """
        if not self.enabled or not self._unsummarized_blocks:
            return False

        # 1. Step characters threshold reached
        if self._current_buffer_chars >= self.step_chars:
            return True

        # 2. Chapter transition check (if chapter namespace is present)
        if next_block is not None:
            next_ch = extract_chapter_id(next_block.id)
            curr_ch = extract_chapter_id(self._unsummarized_blocks[-1].id)
            if (
                curr_ch
                and next_ch
                and curr_ch != next_ch
                and (
                    self._current_buffer_chars >= _SNAPSHOT_MIN_BUFFER_CHARS
                    or len(self._unsummarized_blocks) >= 2
                )
            ):
                return True

        return False

    async def generate_step_snapshot(
        self,
        complete_fn: Callable[[str, str], Awaitable[str]],
        target_lang: str = "zh",
    ) -> str:
        """Generate a macro continuation snapshot from accumulated buffer and reset buffer."""
        if not self._unsummarized_blocks:
            return self.get_latest_macro_summary()

        # Sort by spine_index to ensure chronological narrative order
        # even when concurrent workers finished out of sequence
        blocks_to_summarize = sorted(self._unsummarized_blocks, key=lambda b: b.spine_index)
        first_block = blocks_to_summarize[0]
        last_block = blocks_to_summarize[-1]
        chapter_id = extract_chapter_id(first_block.id)

        # Same assembly rules as the per-chapter rolling summary — one owner
        # for the truncation policy so the two callers cannot drift.
        content = collect_chapter_text(blocks_to_summarize)

        summary = ""
        if content:
            summary = await llm_summary(
                content,
                complete_fn,
                target_lang,
                context="Hierarchical macro summary LLM call failed",
            ) or deterministic_summary(content)

        snapshot = StepSnapshot(
            step_index=self._step_counter,
            start_spine=first_block.spine_index,
            end_spine=last_block.spine_index,
            chapter_id=chapter_id,
            summary_text=summary,
            char_count=self._current_buffer_chars,
        )
        self._snapshots.append(snapshot)
        self._step_counter += 1

        # Compress the closed epoch of L2 snapshots into one L3 entry.
        if len(self._snapshots) % self.l3_epoch_steps == 0:
            self._generate_epoch_summary()

        # Clear buffer
        self._unsummarized_blocks.clear()
        self._current_buffer_chars = 0

        return summary

    def _generate_epoch_summary(self) -> None:
        """Deterministically compress the last ``l3_epoch_steps`` L2 snapshots.

        Deterministic on purpose: no extra LLM call per epoch, so the L3 text
        is stable once written and the prompt prefix it joins stays cacheable.
        """
        window = self._snapshots[-self.l3_epoch_steps :]
        combined = " ".join(s.summary_text for s in window if s.summary_text)
        if not combined:
            return
        # ``deterministic_summary`` defaults to a 400-char cap, so slicing its
        # result to ``_MAX_L3_CHARS`` (1200) was dead and the epoch text rode
        # the static prompt prefix at a third of the reserved budget. Pass the
        # L3 cap through instead.
        # Use the instance budget, not the module default: the rebuild path
        # (:meth:`rebuild_epochs`) already uses ``self.max_l3_chars``, and the two
        # only agreed because the default happens to be the same value. Once the
        # knob is configured, a mismatch would make a repaired chapter's epochs
        # differ from the ones built incrementally and drift the static prefix.
        compressed = deterministic_epoch_summary(combined, self.max_l3_chars).strip()
        if not compressed:
            compressed = combined[: self.max_l3_chars]
        epoch_idx = (self._epochs[-1].epoch_index + 1) if self._epochs else 0
        epoch = EpochSnapshot(
            epoch_index=epoch_idx,
            start_spine=window[0].start_spine,
            end_spine=window[-1].end_spine,
            summary_text=compressed,
        )
        self._epochs.append(epoch)
        self._trim_epochs()
        logger.info(
            "L3 epoch %d compressed (%d L2 steps, spine %d-%d)",
            epoch.epoch_index,
            len(window),
            epoch.start_spine,
            epoch.end_spine,
        )

    def _trim_epochs(self) -> None:
        """Enforce ``max_epochs`` while pinning epoch 0 as the long-range anchor.

        A plain FIFO slice evicted the earliest epoch once a long book passed
        ``max_epochs`` (50 epochs ≈ 700k target characters), which quietly turned
        this slot from "the anchor that remembers chapters 1-10" into one more
        sliding window — the opposite of its documented purpose — and rotated the
        static prompt prefix every few L2 steps.
        """
        if self.max_epochs is None or len(self._epochs) <= self.max_epochs:
            return
        if self.max_epochs <= 1:
            self._epochs = self._epochs[:1]
            return
        anchor = self._epochs[0]
        self._epochs = [anchor, *self._epochs[-(self.max_epochs - 1) :]]

    def get_l3_summary(self, *, sliding_window: bool | None = None) -> str:
        """Cumulative L3 epoch summary text across closed epochs (clamped to max_l3_chars).

        The clamp keeps the *earliest* whole epochs: this slot is the long-range
        anchor ("chapter 30 knows what happened in chapters 1-10"), while recent
        context is carried by the L2 snapshots and L1 neighbours.
        When ``sliding_window`` is True (or configured on the manager), keeps the
        most recent whole epochs within the budget instead. Epochs are kept whole;
        only a single over-budget epoch is truncated.
        """
        if not self.enabled:
            return ""
        use_sliding = self.sliding_window if sliding_window is None else sliding_window
        self._trim_epochs()
        epochs = self._epochs
        parts = [e.summary_text.strip() for e in epochs if e.summary_text.strip()]
        if not parts:
            return ""
        if use_sliding:
            parts = list(reversed(parts))
        kept: list[str] = []
        used = 0
        for part in parts:
            extra = len(part) + (1 if kept else 0)
            if kept and used + extra > self.max_l3_chars:
                break
            kept.append(part)
            used += extra
        if use_sliding:
            kept.reverse()
        return "\n".join(kept)[: self.max_l3_chars]

    def rebuild_epochs(self) -> None:
        """Recompute L3 epochs deterministically from the current L2 snapshots.

        Used after a repair refresh replaces a chapter's snapshots: the old
        epochs were built from the pre-repair text and would otherwise keep
        feeding the corrected chapter's stale renderings to later prompts.
        """
        self._epochs = []
        for start in range(0, len(self._snapshots), self.l3_epoch_steps):
            window = self._snapshots[start : start + self.l3_epoch_steps]
            combined = " ".join(s.summary_text for s in window if s.summary_text).strip()
            if not combined:
                continue
            epoch = EpochSnapshot(
                epoch_index=len(self._epochs),
                start_spine=window[0].start_spine,
                end_spine=window[-1].end_spine,
                summary_text=deterministic_epoch_summary(combined, self.max_l3_chars),
            )
            self._epochs.append(epoch)
        # Route through the same trim as the incremental path: a bare FIFO
        # slice evicted the epoch-0 long-range anchor, so a book that outgrew
        # ``max_epochs`` lost its "chapters 1-10" memory on the very repair
        # refresh that was supposed to renew it — and the rebuild path then
        # disagreed with the incremental path about what the anchor slot holds.
        self._trim_epochs()

    def replace_chapter_snapshot(
        self,
        chapter_id: str,
        *,
        start_spine: int,
        end_spine: int,
        summary_text: str,
        char_count: int,
    ) -> None:
        """Replace every L2 snapshot for one chapter with its post-repair recap."""
        self._snapshots = [s for s in self._snapshots if s.chapter_id != chapter_id]
        self._snapshots.append(
            StepSnapshot(
                step_index=self._step_counter,
                start_spine=start_spine,
                end_spine=end_spine,
                chapter_id=chapter_id,
                summary_text=summary_text,
                char_count=char_count,
            )
        )
        self._step_counter += 1
        self._snapshots.sort(key=lambda s: (s.start_spine, s.end_spine, s.step_index))
        self.rebuild_epochs()

    def replace_chapter_snapshots(
        self,
        chapter_id: str,
        windows: Sequence[tuple[int, int, str, int]],
    ) -> None:
        """Replace every L2 snapshot for one chapter with post-repair window recaps.

        Each window is ``(start_spine, end_spine, summary_text, char_count)``.
        The old single-recap variant collapsed a long chapter's N incremental
        snapshots into one tail-biased summary capped at ``step_chars * 2``, so
        everything earlier in a long chapter silently left L2/L3 after any
        repair. Window recaps preserve the chapter at ~``step_chars``
        granularity — the same shape the incremental path would have produced —
        and ``rebuild_epochs`` recomputes L3 from the restored history.
        """
        self._snapshots = [s for s in self._snapshots if s.chapter_id != chapter_id]
        for start_spine, end_spine, summary_text, char_count in windows:
            self._snapshots.append(
                StepSnapshot(
                    step_index=self._step_counter,
                    start_spine=start_spine,
                    end_spine=end_spine,
                    chapter_id=chapter_id,
                    summary_text=summary_text,
                    char_count=char_count,
                )
            )
            self._step_counter += 1
        self._snapshots.sort(key=lambda s: (s.start_spine, s.end_spine, s.step_index))
        self.rebuild_epochs()

    def merge_external_state(self, state: dict[str, Any]) -> None:
        """Import another writer's snapshots for chapters this manager is not drafting.

        Called under :func:`memory_state_lock` right before an absolute persist.
        Under chapter streaming the repair worker's post-repair refresh rewrites
        chapter N's snapshots while this manager (chapter N+1) holds its own
        copy of the persisted state; persisting blindly would discard the
        refresh. The persisted state is authoritative for every chapter the
        buffer is not currently drafting; this manager is authoritative for its
        buffer's chapters (it may hold unsaved work the external state lacks).
        In whole-book mode the stages are sequential, so the external state is
        this manager's own earlier persist and the merge is a content no-op.
        """
        external = [StepSnapshot(**s) for s in state.get("snapshots", []) if isinstance(s, dict)]
        if not external:
            return
        protected = {extract_chapter_id(b.id) for b in self._unsummarized_blocks}
        external_chapters = {s.chapter_id for s in external} - protected
        if not external_chapters:
            return
        kept = [s for s in self._snapshots if s.chapter_id not in external_chapters]
        imported = [s for s in external if s.chapter_id in external_chapters]
        self._snapshots = sorted(
            [*kept, *imported], key=lambda s: (s.start_spine, s.end_spine, s.step_index)
        )
        with contextlib.suppress(TypeError, ValueError):
            self._step_counter = max(self._step_counter, int(state.get("step_counter", 0)))
        self.rebuild_epochs()

    def get_latest_macro_summary(self) -> str:
        if not self._snapshots:
            return ""
        return self._snapshots[-1].summary_text

    def export_state(self) -> dict[str, Any]:
        """Serialize the L2/L3 snapshot history for resume persistence.

        The within-step buffer is serialized too. The old contract excluded it
        on the assumption that "a resumed run re-drafts those blocks" — false:
        ``fetch_pending_blocks`` selects ``pending`` only, so blocks already
        DRAFTED before a crash are never re-drafted and never re-entered the
        buffer, leaving a permanent hole in the rolling context exactly at
        resume boundaries. The buffer spans at most one step (~``step_chars``),
        so the payload is a few blocks.
        """
        return {
            "snapshots": [asdict(s) for s in self._snapshots],
            "epochs": [asdict(e) for e in self._epochs],
            "step_counter": self._step_counter,
            "buffer": [_buffer_block_to_dict(b) for b in self._unsummarized_blocks],
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        """Restore L2/L3 history produced by :meth:`export_state`.

        Keeps continued runs prompt-compatible with the interrupted one: the
        same macro/epoch summaries feed the same prompt slots, so translation
        context (and tm-adjacent prompt topology) survives a restart. The
        restored buffer carries the interrupted step's drafted blocks, so the
        next snapshot includes them.
        """
        snapshots = [StepSnapshot(**s) for s in state.get("snapshots", []) if isinstance(s, dict)]
        epochs = [EpochSnapshot(**e) for e in state.get("epochs", []) if isinstance(e, dict)]
        self._snapshots = snapshots
        self._epochs = epochs
        # Same trim contract as the incremental path (epoch 0 pinned as the
        # long-range anchor): a bare slice here made a resumed run forget the
        # book's opening exactly when ``max_epochs`` was first exceeded
        # across the restart.
        self._trim_epochs()
        self._unsummarized_blocks = [
            b
            for b in (_buffer_block_from_dict(d) for d in state.get("buffer", []))
            if b is not None
        ]
        self._recount_buffer_chars()
        try:
            self._step_counter = int(state.get("step_counter", len(snapshots)))
        except (TypeError, ValueError):
            self._step_counter = len(snapshots)

    def get_macro_context_for_block(self, block: IRBlock) -> str:
        """Retrieve appropriate macro context for a specific block."""
        # 1. Check if an exact chapter match exists among past snapshots
        ch_id = extract_chapter_id(block.id)
        if ch_id:
            for snap in reversed(self._snapshots):
                if (
                    snap.chapter_id == ch_id
                    and snap.end_spine <= block.spine_index
                    and snap.summary_text
                ):
                    return snap.summary_text

        # 2. Fall back to latest available step snapshot prior to this block's spine index
        for snap in reversed(self._snapshots):
            if snap.end_spine <= block.spine_index and snap.summary_text:
                return snap.summary_text

        # 3. No past snapshot prior to this block
        return ""

    def get_salient_entities(self, glossary_terms: Sequence[str] | None = None) -> list[Any]:
        """Extract salient entities from closed L3 epochs and L2 snapshots for coreference binding."""
        from ubt.core.memory.coreference import CoreferenceResolver

        resolver = CoreferenceResolver()
        combined_text_parts: list[str] = []
        for e in self._epochs:
            if e.summary_text:
                combined_text_parts.append(e.summary_text)
        for s in self._snapshots[-3:]:
            if s.summary_text:
                combined_text_parts.append(s.summary_text)

        if not combined_text_parts:
            return []

        return resolver.extract_entities_from_text(
            "\n".join(combined_text_parts),
            source_location="sliding_epoch_memory",
            glossary_terms=glossary_terms,
        )


def _refresh_windows(
    chapter_blocks: list[IRBlock], step_chars: int
) -> list[tuple[IRBlock, IRBlock, str]]:
    """Slice a chapter into ~``step_chars`` windows for the post-repair recap.

    The old refresh summarized the whole chapter through one tail-biased
    ``collect_chapter_text`` call capped at ``step_chars * 2``, so a chapter
    longer than that lost its earlier content from L2/L3 the moment any repair
    rewrote it. Window recaps keep the chapter's coverage at roughly the same
    granularity the incremental draft path would have produced. A chapter
    longer than ``_MAX_REFRESH_SNAPSHOTS`` windows gets proportionally wider
    windows (the per-window LLM input grows with them, still bounded), trading
    a little per-window detail for a hard call-count ceiling.
    """
    if not chapter_blocks:
        return []
    per_block: list[int] = []
    total = 0
    for b in chapter_blocks:
        text = b.target_text or b.draft_text or b.source_text or ""
        n = len(text.strip())
        per_block.append(n)
        total += n
    if total == 0:
        return []
    window_count = max(1, min(_MAX_REFRESH_SNAPSHOTS, math.ceil(total / max(1, step_chars))))
    target = max(1, math.ceil(total / window_count))

    groups: list[list[IRBlock]] = []
    chunk: list[IRBlock] = []
    chunk_chars = 0
    for block, n in zip(chapter_blocks, per_block, strict=True):
        chunk.append(block)
        chunk_chars += n
        if chunk_chars >= target and len(groups) < window_count - 1:
            groups.append(chunk)
            chunk = []
            chunk_chars = 0
    if chunk:
        groups.append(chunk)

    return [
        (blocks[0], blocks[-1], collect_chapter_text(blocks, max_chars=max(target, step_chars)))
        for blocks in groups
    ]


async def refresh_chapter_memory(
    ledger: Any,
    job_id: str,
    chapter_id: str,
    blocks: list[IRBlock],
    complete_fn: Callable[[str, str], Awaitable[str]],
    target_lang: str,
    *,
    step_chars: int = DEFAULT_STEP_CHARS,
    max_epochs: int | None = DEFAULT_MAX_EPOCHS,
    sliding_window: bool = False,
) -> bool:
    """Rebuild one chapter's L2 recaps from the *post-repair* target text.

    The draft stage summarizes the draft it just produced. Repair can rewrite
    that text afterwards, and the old summary then keeps feeding the superseded
    rendering to every later prompt. This replaces the chapter's snapshots with
    per-window recaps of the final text and recomputes L3 deterministically.

    The persisted-state read→replace→write runs under :func:`memory_state_lock`:
    under chapter streaming the draft worker (chapter N+1) persists its own copy
    of ``memory_state`` concurrently, and an unlocked last-writer-wins here
    would silently discard whichever side committed second. The LLM summary
    calls run *before* the lock is taken — they take seconds and never touch
    the persisted state.
    """
    if not chapter_id:
        return False
    chapter_blocks = sorted(
        (b for b in blocks if extract_chapter_id(b.id) == chapter_id),
        key=lambda b: b.spine_index,
    )
    if not chapter_blocks:
        return False

    windows = _refresh_windows(chapter_blocks, step_chars)
    built: list[tuple[int, int, str, int]] = []
    for start_block, end_block, content in windows:
        if not content.strip():
            continue
        summary = await llm_summary(
            content,
            complete_fn,
            target_lang,
            context="Post-repair chapter memory refresh failed",
        ) or deterministic_summary(content)
        if not summary.strip():
            continue
        built.append((start_block.spine_index, end_block.spine_index, summary, len(content)))
    if not built:
        return False

    async with memory_state_lock(job_id):
        manager = HierarchicalMemoryManager(
            step_chars=step_chars,
            enabled=True,
            max_epochs=max_epochs,
            sliding_window=sliding_window,
        )
        state = await asyncio.to_thread(ledger.get_job_metadata_value, job_id, "memory_state")
        if isinstance(state, dict):
            manager.restore_state(state)
        manager.replace_chapter_snapshots(chapter_id, built)
        await asyncio.to_thread(
            ledger.set_job_metadata_value, job_id, "memory_state", manager.export_state()
        )
    return True


__all__ = [
    "DEFAULT_L3_EPOCH_STEPS",
    "DEFAULT_MAX_EPOCHS",
    "DEFAULT_STEP_CHARS",
    "EpochSnapshot",
    "HierarchicalMemoryManager",
    "StepSnapshot",
    "memory_state_lock",
    "refresh_chapter_memory",
]
