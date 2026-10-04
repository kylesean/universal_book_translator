"""AST and IRBlock bifurcation on layout collision / semantic break markers.

Defense 2 & 3 of the Unified Compiler Architecture:
When an upstream layout parser/OCR engine erroneously glues distinct passages
(such as cross-column prose, intervening running headers, or accidental paragraph
merges), the translation model detects the layout collision and inserts
`SEMANTIC_BREAK_TOKEN` (⟦SEMANTIC_BREAK⟧).

This module provides:
1. `SEMANTIC_BREAK_TOKEN`: Canonical delimiter injected in translation prompt;
2. `bifurcate_block`: Bifurcates a single IRBlock into sibling IRBlocks;
3. `bifurcate_blocks`: Maps over a sequence of IRBlocks, returning the bifurcated stream.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Sequence

from ubt.core.ir.models import IRBlock, _with_element_source
from ubt.model.span import CompositeSpan, Span

SEMANTIC_BREAK_TOKEN: str = "⟦SEMANTIC_BREAK⟧"
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？…])\s+")


def _split_prose_sentences(text: str) -> list[str]:
    """Split prose text into sentence chunks."""
    clean = text.strip()
    if not clean:
        return []
    chunks = _SENTENCE_SPLIT_RE.split(clean)
    return [c.strip() for c in chunks if c.strip()]


def _partition_source_by_parts(
    sentences: list[str], parts: list[str], full_source: str
) -> list[str]:
    """Partition source text sentences across parts proportional to target lengths."""
    if len(parts) <= 1 or not sentences:
        return [full_source] * len(parts)
    if len(sentences) == len(parts):
        return sentences

    total_target_chars = sum(len(p) for p in parts) or 1
    ratios = [len(p) / total_target_chars for p in parts]

    # Assign sentences to parts according to target weight
    assigned: list[list[str]] = [[] for _ in parts]
    part_idx = 0
    current_chars = 0
    total_src_chars = sum(len(s) for s in sentences) or 1

    for sent in sentences:
        assigned[part_idx].append(sent)
        current_chars += len(sent)
        # Advance part index if we crossed the proportional quota and not at last part
        target_quota = sum(ratios[: part_idx + 1]) * total_src_chars
        if current_chars >= target_quota and part_idx < len(parts) - 1:
            part_idx += 1

    return [" ".join(bucket) if bucket else full_source for bucket in assigned]


def bifurcate_block(block: IRBlock) -> list[IRBlock]:
    """Bifurcate an IRBlock into sibling blocks if a semantic break marker is present.

    When SEMANTIC_BREAK_TOKEN is detected in target_text or draft_text:
    - CompositeSpan: unbinds the physical box chain into individual single-box blocks,
      restoring individual box spans and source text from provenance if available.
    - Single-box: splits into sibling IRBlocks sharing the bounding box, tagged with
      'bifurcated:semantic_break' in error_flags, so the downstream typesetter formats
      them as distinct paragraphs within that box.
    """
    raw_text = block.target_text or block.draft_text or ""
    if SEMANTIC_BREAK_TOKEN not in raw_text:
        return [block]

    parts = [p.strip() for p in raw_text.split(SEMANTIC_BREAK_TOKEN) if p.strip()]
    if len(parts) <= 1:
        # Delimiter present but only one content chunk; clean the token and return
        clean_text = raw_text.replace(SEMANTIC_BREAK_TOKEN, "").strip()
        cleaned_block = block.model_copy(
            update={
                "target_text": clean_text if block.target_text else None,
                "draft_text": clean_text,
            }
        )
        return [cleaned_block]

    span = block.element.span
    if isinstance(span, CompositeSpan) and len(span.boxes) >= 2:
        boxes = span.boxes
        fused_sources: list[str] | None = block.provenance.get("fused_sources")
        fused_ids: list[str] | None = block.provenance.get("fused_block_ids")
        results: list[IRBlock] = []

        for i, part in enumerate(parts):
            box = boxes[min(i, len(boxes) - 1)]
            orig_id = fused_ids[i] if fused_ids and i < len(fused_ids) else f"{block.id}_s{i}"
            src = (
                fused_sources[i] if fused_sources and i < len(fused_sources) else block.source_text
            )
            new_span = Span(page=box.page, bbox=box.bbox)
            elem = _with_element_source(block.element, src)
            elem = dataclasses.replace(elem, id=orig_id, span=new_span)

            new_block = IRBlock(element=elem)
            new_block.draft_text = part
            new_block.target_text = part
            new_block.status = block.status
            new_block.error_flags = [*block.error_flags, "bifurcated:semantic_break"]
            new_block.provenance = {
                **block.provenance,
                "bifurcated_from": block.id,
                "bifurcation_index": i,
            }
            results.append(new_block)
        return results

    # Single box bifurcation
    sentences = _split_prose_sentences(block.source_text)
    src_parts = _partition_source_by_parts(sentences, parts, block.source_text)
    results = []
    for i, part in enumerate(parts):
        sub_id = f"{block.id}_s{i}"
        src = src_parts[i] if i < len(src_parts) else block.source_text
        elem = _with_element_source(block.element, src)
        elem = dataclasses.replace(elem, id=sub_id)

        new_block = IRBlock(element=elem)
        new_block.draft_text = part
        new_block.target_text = part
        new_block.status = block.status
        new_block.error_flags = [*block.error_flags, "bifurcated:semantic_break"]
        new_block.provenance = {
            **block.provenance,
            "bifurcated_from": block.id,
            "bifurcation_index": i,
        }
        results.append(new_block)
    return results


def bifurcate_blocks(blocks: Sequence[IRBlock]) -> list[IRBlock]:
    """Bifurcate all candidate blocks in a sequence."""
    output: list[IRBlock] = []
    for block in blocks:
        output.extend(bifurcate_block(block))
    return output


__all__ = [
    "SEMANTIC_BREAK_TOKEN",
    "bifurcate_block",
    "bifurcate_blocks",
]
