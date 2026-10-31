"""Minimal AST Tagging & Self-Healing Decoder (Dominant Style Zero-Tag Protocol).

Implements the Stage 1-2 tag protection and self-healing decoder:
1. Dominant Style Zero-Tag:
   Computes the dominant style in a paragraph; characters sharing that dominant
   style are emitted bare without tags (90%+ zero-tag bare text).
   Deviating styles are tagged with minimal ``<sN>...</sN>`` wrappers.
2. Rigid Inline Slot Masking:
   Formulas and citations are represented as self-closing ``<sN/>`` slots to prevent
   LLM mutation, deletion, or hallucination.
3. Stack-based Self-Healing Decoder:
   Reconstructs styled InlineRuns from LLM-translated text while safely healing
   unclosed, malformed, or missing tags.
4. Slot Integrity Verification:
   Ensures slot IDs in the translation match the source text 1:1.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from ubt.core.ir.models import InlineRun

TAG_REGEX = re.compile(r"(<s\d+/>|</?s\d+>)")
SLOT_SELF_CLOSING_REGEX = re.compile(r"<s(\d+)/>")
TAG_OPEN_REGEX = re.compile(r"<s(\d+)>")
TAG_CLOSE_REGEX = re.compile(r"</s(\d+)>")


@dataclass(frozen=True, slots=True)
class TaggedPayload:
    """The result of encoding source text with dominant-style zero-tagging."""

    tagged_text: str
    span_style_map: dict[str, InlineRun]
    base_style: InlineRun
    slot_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DecodedPayload:
    """The result of self-healing decoding on translated text."""

    clean_text: str
    inline_runs: tuple[InlineRun, ...]
    slot_occurrences: dict[str, int]
    unclosed_tags: tuple[str, ...] = ()
    unmatched_closes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SlotReport:
    """Bidirectional slot integrity report."""

    missing_slots: tuple[str, ...]
    hallucinated_slots: tuple[str, ...]
    is_valid: bool


class DominantStyleTagger:
    """Encodes paragraph text into minimal tags using dominant style zero-tagging."""

    @classmethod
    def encode(
        cls,
        text: str,
        inline_runs: Sequence[InlineRun] = (),
        *,
        math_slots: Sequence[str] = (),
        citation_slots: Sequence[str] = (),
    ) -> TaggedPayload:
        if not text:
            base = InlineRun(text="")
            return TaggedPayload(tagged_text="", span_style_map={}, base_style=base, slot_ids=())

        # Build character-level style assignments
        char_styles: list[InlineRun | None] = [None] * len(text)
        span_style_map: dict[str, InlineRun] = {}
        slot_ids: list[str] = []
        counter = 1

        # 1. Map inline runs to character spans
        for run in inline_runs:
            needle = run.text.strip()
            if not needle:
                continue
            cursor = 0
            while cursor < len(text):
                idx = text.find(needle, cursor)
                if idx < 0:
                    break
                end = idx + len(needle)
                for i in range(idx, end):
                    char_styles[i] = run
                cursor = end

        # 2. Determine dominant style
        style_counts: Counter[tuple[bool, bool, bool, str | None]] = Counter()
        for s in char_styles:
            if s is not None:
                style_counts[(s.bold, s.italic, s.superscript, s.color_hex)] += 1
            else:
                style_counts[(False, False, False, None)] += 1

        most_common_key = style_counts.most_common(1)[0][0]
        base_style = InlineRun(
            text="",
            bold=most_common_key[0],
            italic=most_common_key[1],
            superscript=most_common_key[2],
            color_hex=most_common_key[3],
        )

        # 3. Replace math and citation slots with self-closing <sN/>
        # Sort slots by descending length to prevent partial substring collision
        all_rigid_slots = sorted(list(math_slots) + list(citation_slots), key=len, reverse=True)

        working_text = text
        for slot_content in all_rigid_slots:
            if not slot_content:
                continue
            while slot_content in working_text:
                sid = f"s{counter}"
                counter += 1
                slot_tag = f"<{sid}/>"
                working_text = working_text.replace(slot_content, slot_tag, 1)
                span_style_map[sid] = InlineRun(text=slot_content, bold=False, italic=False)
                slot_ids.append(sid)

        # 4. Wrap non-dominant styled runs with <sN>...</sN>
        if not inline_runs:
            return TaggedPayload(
                tagged_text=working_text,
                span_style_map=span_style_map,
                base_style=base_style,
                slot_ids=tuple(slot_ids),
            )

        # Apply styled spans on working text
        for run in inline_runs:
            needle = run.text.strip()
            if not needle or needle in span_style_map.values():
                continue
            # Check if run matches base style
            if (
                run.bold == base_style.bold
                and run.italic == base_style.italic
                and run.superscript == base_style.superscript
                and run.color_hex == base_style.color_hex
            ):
                continue

            # Need tagging
            if needle in working_text:
                sid = f"s{counter}"
                counter += 1
                span_style_map[sid] = run
                working_text = working_text.replace(needle, f"<{sid}>{needle}</{sid}>", 1)

        return TaggedPayload(
            tagged_text=working_text,
            span_style_map=span_style_map,
            base_style=base_style,
            slot_ids=tuple(slot_ids),
        )


class TagSelfHealingDecoder:
    """Decodes translated text with a style stack, healing broken or missing tags."""

    @classmethod
    def decode(
        cls,
        translated_text: str,
        span_style_map: dict[str, InlineRun],
        base_style: InlineRun | None = None,
    ) -> DecodedPayload:
        if base_style is None:
            base_style = InlineRun(text="")

        if not translated_text:
            return DecodedPayload(clean_text="", inline_runs=(), slot_occurrences={})

        tokens = TAG_REGEX.split(translated_text)
        clean_parts: list[str] = []
        runs_out: list[InlineRun] = []
        style_stack: list[InlineRun] = [base_style]
        slot_counts: Counter[str] = Counter()
        unclosed_tags: list[str] = []
        unmatched_closes: list[str] = []

        open_tags_stack: list[str] = []

        for token in tokens:
            if not token:
                continue

            # Case 1: Self-closing slot <sN/>
            m_self = SLOT_SELF_CLOSING_REGEX.fullmatch(token)
            if m_self:
                sid = f"s{m_self.group(1)}"
                slot_counts[sid] += 1
                original_run = span_style_map.get(sid)
                original_text = original_run.text if original_run else token
                clean_parts.append(original_text)
                runs_out.append(
                    InlineRun(
                        text=original_text,
                        bold=style_stack[-1].bold,
                        italic=style_stack[-1].italic,
                        superscript=style_stack[-1].superscript,
                        color_hex=style_stack[-1].color_hex,
                    )
                )
                continue

            # Case 2: Open tag <sN>
            m_open = TAG_OPEN_REGEX.fullmatch(token)
            if m_open:
                sid = f"s{m_open.group(1)}"
                open_tags_stack.append(sid)
                style = span_style_map.get(sid, style_stack[-1])
                style_stack.append(style)
                continue

            # Case 3: Close tag </sN>
            m_close = TAG_CLOSE_REGEX.fullmatch(token)
            if m_close:
                sid = f"s{m_close.group(1)}"
                if sid in open_tags_stack:
                    # Pop back to the matching open tag
                    while open_tags_stack and open_tags_stack[-1] != sid:
                        unclosed_tags.append(open_tags_stack.pop())
                        if len(style_stack) > 1:
                            style_stack.pop()
                    if open_tags_stack:
                        open_tags_stack.pop()
                    if len(style_stack) > 1:
                        style_stack.pop()
                else:
                    unmatched_closes.append(sid)
                continue

            # Case 4: Text content
            clean_parts.append(token)
            current_style = style_stack[-1]
            # Record run if it deviates from base style
            if (
                current_style.bold != base_style.bold
                or current_style.italic != base_style.italic
                or current_style.superscript != base_style.superscript
                or current_style.color_hex != base_style.color_hex
            ):
                runs_out.append(
                    InlineRun(
                        text=token,
                        bold=current_style.bold,
                        italic=current_style.italic,
                        superscript=current_style.superscript,
                        color_hex=current_style.color_hex,
                    )
                )

        # Merge adjacent runs with identical styling
        merged_runs: list[InlineRun] = []
        for run in runs_out:
            if not run.text:
                continue
            if (
                merged_runs
                and merged_runs[-1].bold == run.bold
                and merged_runs[-1].italic == run.italic
                and merged_runs[-1].superscript == run.superscript
                and merged_runs[-1].color_hex == run.color_hex
            ):
                merged_runs[-1] = InlineRun(
                    text=merged_runs[-1].text + run.text,
                    bold=run.bold,
                    italic=run.italic,
                    superscript=run.superscript,
                    color_hex=run.color_hex,
                )
            else:
                merged_runs.append(run)

        return DecodedPayload(
            clean_text="".join(clean_parts),
            inline_runs=tuple(merged_runs),
            slot_occurrences=dict(slot_counts),
            unclosed_tags=tuple(open_tags_stack + unclosed_tags),
            unmatched_closes=tuple(unmatched_closes),
        )


class SlotIntegrityValidator:
    """Validates that slot placeholders are preserved between source and draft."""

    @classmethod
    def validate(
        cls,
        expected_slot_ids: Sequence[str],
        observed_slot_counts: dict[str, int],
    ) -> SlotReport:
        missing = [sid for sid in expected_slot_ids if observed_slot_counts.get(sid, 0) == 0]
        expected_set = set(expected_slot_ids)
        hallucinated = [sid for sid in observed_slot_counts if sid not in expected_set]
        is_valid = len(missing) == 0 and len(hallucinated) == 0
        return SlotReport(
            missing_slots=tuple(missing),
            hallucinated_slots=tuple(hallucinated),
            is_valid=is_valid,
        )
