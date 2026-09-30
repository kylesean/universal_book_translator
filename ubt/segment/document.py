"""Build the delivered document's translation segments (the export join).

Translation interchange needs units, and by export time the units are the
pipeline's own blocks: the source text, its protected spans, and the target that
was actually delivered. This is that join -- one
:class:`~ubt.model.segment.Segment` per translatable block, its source's
protected spans masked into placeholders, its delivered target carried verbatim,
and its block id kept, so a reviewer's revision maps straight back to the block
the ledger, the quality report and the delivery contract already name.

The reader -> ``Document`` -> segment join is
:class:`~ubt.translate.engine.TranslationEngine`'s business; this module is the
block -> segment projection the export companion writes.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from ubt.model.segment import Segment, SegmentState
from ubt.segment.xliff import xml_safe

if TYPE_CHECKING:
    from ubt.core.ir.models import IRBlock
    from ubt.segment.placeholders import PlaceholderEngine

#: Pipeline block status -> the coarser XLIFF lifecycle a reviewer works in.
#: The precise status (and every QA flag) stays in the quality report; the
#: companion names the stage, not the whole audit trail. Two statuses share a
#: state on purpose: ``repair_pending`` and ``needs_human`` both describe a
#: draft that exists and is not released, which is exactly XLIFF ``translated``.
_BLOCK_STATE: dict[str, SegmentState] = {
    "pending": SegmentState.NEW,
    "drafted": SegmentState.TRANSLATED,
    "repair_pending": SegmentState.TRANSLATED,
    "mtqe_passed": SegmentState.VERIFIED,
    "repaired": SegmentState.FINAL,
    "failed": SegmentState.BLOCKED,
    "needs_human": SegmentState.TRANSLATED,
    "blocked_human": SegmentState.BLOCKED,
}


def segments_from_blocks(blocks: Sequence[IRBlock], *, engine: PlaceholderEngine) -> list[Segment]:
    """One segment per delivered translatable block, in reading order.

    ``skip_translate`` blocks (formulas, images, preserved markers) are carried
    as immutable assets by the delivery, not handed to a reviewer as prose, so
    they are omitted the same way an empty block is. A block with no delivered
    target carries no ``target``, which is the honest signal that its source,
    not a translation, is what shipped.
    """
    segments: list[Segment] = []
    for block in blocks:
        if block.skip_translate:
            continue
        source = block.source_text or ""
        if not source.strip():
            continue
        masked = engine.mask(xml_safe(source))
        target = block.target_text or ""
        segments.append(
            Segment(
                id=block.id,
                source=masked.text,
                placeholders=masked.placeholders,
                target=xml_safe(target) if target.strip() else None,
                state=_BLOCK_STATE.get(block.status.value, SegmentState.NEW),
            )
        )
    return segments


__all__ = ["segments_from_blocks"]
