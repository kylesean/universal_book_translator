"""One cross-format contract for blocks that left the pipeline unresolved.

A block whose draft never passed the quality gates must stay *visible and
labelled* in every deliverable. Markdown and HTML carry the export stage's
``<mark class="ubt-failed-draft">`` wrapper through; DOCX (which cannot store
the HTML mark) used to strip that wrapper and ship the raw machine draft as if
it were a finished translation. The status set and the human-readable notes
live here so the formats cannot drift apart again.
"""

from __future__ import annotations

from ubt.core.ir.models import BlockStatus

#: Statuses meaning "no approved translation shipped for this block".
UNRESOLVED_STATUSES = frozenset(
    {BlockStatus.FAILED, BlockStatus.NEEDS_HUMAN, BlockStatus.BLOCKED_HUMAN}
)

FAILURE_NOTES: dict[BlockStatus, str] = {
    BlockStatus.FAILED: "Translation FAILED quality gates — unresolved machine draft kept for review only:",
    BlockStatus.NEEDS_HUMAN: "Flagged NEEDS_HUMAN by quality triage — machine draft for human post-editing:",
    BlockStatus.BLOCKED_HUMAN: "Blocked from machine translation by quality gates — source kept, human translation required.",
}


def is_unresolved(status: BlockStatus) -> bool:
    """Whether ``status`` marks a block with no approved translation."""
    return status in UNRESOLVED_STATUSES


def failure_note(status: BlockStatus) -> str:
    """Plain-text (single-line) unresolved note for structured formats."""
    note = FAILURE_NOTES.get(status, "Translation unresolved — manual review required:")
    return f"[UBT] {note}"


def failure_note_markdown(status: BlockStatus) -> str:
    """Markdown-formatted unresolved note (kept identical to the old literal)."""
    note = FAILURE_NOTES.get(status, "Translation unresolved — manual review required:")
    return f"> ⚠️ **[UBT] {note}**"
