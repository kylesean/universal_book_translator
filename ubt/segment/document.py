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

from ubt.model.segment import QA, Segment, SegmentState
from ubt.segment.xliff import xml_safe

if TYPE_CHECKING:
    from ubt.core.ir.models import BlockStatus, IRBlock
    from ubt.segment.placeholders import PlaceholderEngine

#: Pipeline block status -> the coarser XLIFF lifecycle a reviewer works in.
#: The precise status (and every QA flag) stays in the quality report; the
#: companion names the stage, not the whole audit trail. ``repair_pending``
#: still maps to ``translated`` because no verdict has been reached yet — there
#: is nothing for a reviewer to act on until triage decides.
#:
#: ``needs_human`` and ``blocked_human`` are deliberately NOT collapsed onto the
#: states they resemble. They are the two blocks a reviewer must actually find,
#: and the export is the only place that says which is which:
#:
#: - ``needs_human`` ships a machine draft and needs post-editing, so the segment
#:   is ``NEEDS_REVIEW`` (XLIFF ``needs-review-translation``).
#: - ``blocked_human`` ships the SOURCE under a quarantine marker (see
#:   ``triage._blocked_human_target``), so no translation exists and the segment
#:   is ``BLOCKED`` (XLIFF ``needs-translation``).
#:
#: Both previously mapped to a state that hid them: ``needs_human`` became
#: ``translated`` (indistinguishable from a clean delivery) and ``blocked_human``
#: became ``needs-review-translation`` (which claims a translation is there to
#: review).
_BLOCK_STATE: dict[str, SegmentState] = {
    "pending": SegmentState.NEW,
    "drafted": SegmentState.TRANSLATED,
    "repair_pending": SegmentState.TRANSLATED,
    "mtqe_passed": SegmentState.VERIFIED,
    "repaired": SegmentState.FINAL,
    "failed": SegmentState.BLOCKED,
    "needs_human": SegmentState.NEEDS_REVIEW,
    "blocked_human": SegmentState.BLOCKED,
}


def _block_state(status: BlockStatus) -> SegmentState:
    """Map a block status to its XLIFF lifecycle state.

    A status missing from ``_BLOCK_STATE`` is IR drift (a lifecycle stage this
    export does not understand) — fail loud instead of silently labelling the
    segment NEW, which would tell a reviewer to translate a block whose real
    state is unknown.
    """
    try:
        return _BLOCK_STATE[status.value]
    except KeyError as exc:
        raise ValueError(f"unmapped BlockStatus for export: {status.value!r}") from exc


def _block_qa(block: IRBlock) -> QA | None:
    """The evidence behind a block's verdict, or ``None`` when it is clean.

    A clean block carries no QA: an empty ``QA()`` would imply the block was
    assessed and found merely fine, which is not the same as never having been
    flagged. Returning ``None`` keeps "no evidence" distinguishable from
    "evidence of no problem" in anything that inspects the segments.
    """
    flags = tuple(block.error_flags)
    if not flags and block.mqm_severity is None and block.mtqe_score is None:
        return None
    return QA(score=block.mtqe_score, severity=block.mqm_severity, flags=flags)


def segments_from_blocks(blocks: Sequence[IRBlock], *, engine: PlaceholderEngine) -> list[Segment]:
    """One segment per delivered translatable block, in reading order.

    ``skip_translate`` blocks (formulas, images, preserved markers) are carried
    as immutable assets by the delivery, not handed to a reviewer as prose, so
    they are omitted the same way an empty block is. A block with no delivered
    target carries no ``target``, which is the honest signal that its source,
    not a translation, is what shipped.

    ``qa`` carries the reason a block was flagged (score, MQM severity, error
    flags), because a state alone only says *that* something needs attention.
    A reviewer opening 400 blocks is looking for "where", and the second
    question every time is "why" -- without it the queue is a list of ids.
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
                state=_block_state(block.status),
                qa=_block_qa(block),
            )
        )
    return segments


__all__ = ["segments_from_blocks"]
