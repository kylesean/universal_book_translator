"""Hierarchical memory management for translation pipelines.

Provides three-tier context memory:
1. L1 Micro-Context: Fast local neighbor sliding window (forward/backward within same semantic flow).
2. L2 Macro-Step Snapshot: Rolling cumulative summary triggered by character/token step threshold
   (default 3500 chars) or chapter transitions. Ensures continuity for both chaptered books and
   unsegmented long documents (papers, reports) while maximizing prompt prefix cacheability.
3. L3 Epoch Summary: every ``l3_epoch_steps`` L2 snapshots are compressed into one
   epoch entry that rides the static prompt prefix right after the global glossary — chapter 30
   finally knows what happened in chapters 1-10 without paying per-step cache invalidation.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from typing import Any

from ubt.core.ir.models import IRBlock
from ubt.core.memory.neighbor_window import DEFAULT_NEIGHBOR_CHARS, NeighborContextBuilder
from ubt.core.memory.rolling_summary import (
    build_summary_prompt,
    deterministic_summary,
    extract_chapter_id,
)

logger = logging.getLogger(__name__)

DEFAULT_STEP_CHARS = 3500
DEFAULT_L3_EPOCH_STEPS = 4
_MIN_SUMMARY_CHARS = 10
_MAX_SUMMARY_CHARS = 800
_MAX_L3_CHARS = 1200


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
    """Two-tier memory manager maintaining L1 neighbor context and L2 macro step snapshots."""

    def __init__(
        self,
        step_chars: int = DEFAULT_STEP_CHARS,
        neighbor_chars: int = DEFAULT_NEIGHBOR_CHARS,
        l3_epoch_steps: int = DEFAULT_L3_EPOCH_STEPS,
    ) -> None:
        self.step_chars = step_chars
        self.neighbor_builder = NeighborContextBuilder(neighbor_chars=neighbor_chars)
        self.l3_epoch_steps = max(1, l3_epoch_steps)

        self._snapshots: list[StepSnapshot] = []
        self._epochs: list[EpochSnapshot] = []
        self._unsummarized_blocks: list[IRBlock] = []
        self._current_buffer_chars: int = 0
        self._last_chapter: str = ""
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
        """Extract L1 micro neighbor context within the same semantic flow."""
        return self.neighbor_builder.extract_from_blocks(
            target_block=target_block,
            surrounding_blocks=surrounding_blocks,
            fallback_prev_text=fallback_prev_text,
            fallback_next_text=fallback_next_text,
        )

    def record_drafted_block(self, block: IRBlock) -> None:
        """Buffer a newly drafted or finalized block for macro snapshot tracking."""
        self._unsummarized_blocks.append(block)
        text = block.target_text or block.draft_text or block.source_text or ""
        self._current_buffer_chars += len(text)

    def should_trigger_snapshot(self, next_block: IRBlock | None = None) -> bool:
        """Check if buffer has accumulated enough content or crossed a chapter boundary."""
        if not self._unsummarized_blocks:
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
                and (self._current_buffer_chars >= 300 or len(self._unsummarized_blocks) >= 2)
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

        # Collect text
        parts: list[str] = []
        for b in blocks_to_summarize:
            t = b.target_text or b.draft_text or b.source_text or ""
            if t.strip():
                parts.append(t.strip())
        content = "\n".join(parts)[:3000].strip()

        summary = ""
        if content:
            try:
                system_prompt, user_prompt = build_summary_prompt(content, target_lang)
                raw = await complete_fn(system_prompt, user_prompt)
                cleaned = (raw or "").strip().strip('"“”').strip()
                normalized = " ".join(cleaned.split())
                if len(normalized) >= _MIN_SUMMARY_CHARS:
                    # Truncate an over-long summary at a sentence boundary
                    # instead of discarding it for a short head excerpt.
                    summary = deterministic_summary(normalized, _MAX_SUMMARY_CHARS)
            except Exception as exc:
                logger.warning("Hierarchical macro summary LLM call failed: %s", exc)

            if not summary:
                summary = deterministic_summary(content)

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
        if chapter_id:
            self._last_chapter = chapter_id

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
        compressed = deterministic_summary(combined, _MAX_L3_CHARS).strip()
        if not compressed:
            compressed = combined[:_MAX_L3_CHARS]
        epoch = EpochSnapshot(
            epoch_index=len(self._epochs),
            start_spine=window[0].start_spine,
            end_spine=window[-1].end_spine,
            summary_text=compressed,
        )
        self._epochs.append(epoch)
        logger.info(
            "L3 epoch %d compressed (%d L2 steps, spine %d-%d)",
            epoch.epoch_index,
            len(window),
            epoch.start_spine,
            epoch.end_spine,
        )

    def get_l3_summary(self) -> str:
        """Cumulative L3 epoch summary text across closed epochs (clamped to _MAX_L3_CHARS)."""
        if not self._epochs:
            return ""
        parts = [e.summary_text.strip() for e in self._epochs if e.summary_text.strip()]
        combined = "\n".join(parts)
        if len(combined) > _MAX_L3_CHARS:
            return combined[-_MAX_L3_CHARS:]
        return combined

    def format_l3_prompt_block(self, summary_text: str) -> str:
        """Render the L3 epoch block for prompt injection (static-prefix slot)."""
        clean = summary_text.strip()
        if not clean:
            return ""
        return f"### Book Continuity (Compressed History)\n{clean}"

    def get_latest_macro_summary(self) -> str:
        """Get the most recent macro snapshot summary text."""
        if not self._snapshots:
            return ""
        return self._snapshots[-1].summary_text

    def export_state(self) -> dict[str, Any]:
        """Serialize the L2/L3 snapshot history for resume persistence.

        The within-step buffer is intentionally excluded: it only spans the
        current step and is rebuilt from the blocks a resumed run drafts.
        """
        return {
            "snapshots": [asdict(s) for s in self._snapshots],
            "epochs": [asdict(e) for e in self._epochs],
            "step_counter": self._step_counter,
            "last_chapter": self._last_chapter,
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        """Restore L2/L3 history produced by :meth:`export_state`.

        Keeps continued runs prompt-compatible with the interrupted one: the
        same macro/epoch summaries feed the same prompt slots, so translation
        context (and tm-adjacent prompt topology) survives a restart.
        """
        snapshots = [StepSnapshot(**s) for s in state.get("snapshots", []) if isinstance(s, dict)]
        epochs = [EpochSnapshot(**e) for e in state.get("epochs", []) if isinstance(e, dict)]
        self._snapshots = snapshots
        self._epochs = epochs
        try:
            self._step_counter = int(state.get("step_counter", len(snapshots)))
        except (TypeError, ValueError):
            self._step_counter = len(snapshots)
        self._last_chapter = str(state.get("last_chapter", ""))

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
