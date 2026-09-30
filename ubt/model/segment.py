"""Translation units: segments and protected placeholders (ADR-0001 model layer).

``IRBlock`` fused three concerns -- document structure, translation, and
execution state -- into one type. The structure half now lives in
:mod:`ubt.model.ast`; this module is the translation half.

A :class:`Segment` is the industry's unit of translation (the shape XLIFF
carries): source text with protected spans replaced by :class:`Placeholder`
tokens, plus the target, a lifecycle state, provenance, and QA. A
:class:`Placeholder` is one protected span -- inline math, code, a citation --
kept opaque so the model translates *around* it and the exact original is
restored afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class SegmentState(StrEnum):
    """Lifecycle of a translation unit (mirrors the block status vocabulary)."""

    NEW = "new"  # not translated yet
    TRANSLATED = "translated"  # a draft exists
    VERIFIED = "verified"  # draft passed quality estimation
    FINAL = "final"  # released (repaired or human-accepted)
    BLOCKED = "blocked"  # must not ship machine output (critical defect)


@dataclass(frozen=True, slots=True)
class Placeholder:
    """One protected span inside a segment.

    ``token`` is the exact opaque marker the model sees; ``original`` is the
    text it stands for. Restoring verifies the token's integrity checksum, so a
    model that rewrites or drops a token cannot silently corrupt the span.
    """

    token: str
    kind: str  # "code" | "math" | "soup" | "citation"
    original: str


@dataclass(frozen=True, slots=True)
class Provenance:
    """Where a translation came from -- for audit, TM gating and cost."""

    source: str = "mt"  # "mt" | "human" | "tm"
    model: str = ""
    prompt_version: str = ""


@dataclass(frozen=True, slots=True)
class QA:
    """Quality signals attached to a segment (MQM severity + score + flags)."""

    score: float | None = None
    severity: str | None = None  # "critical" | "major" | "minor" | None
    flags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Segment:
    """A translatable unit: source, protected spans, target, state, evidence."""

    id: str
    source: str  # protected spans replaced by their placeholder tokens
    placeholders: tuple[Placeholder, ...] = ()
    target: str | None = None
    state: SegmentState = SegmentState.NEW
    provenance: Provenance | None = None
    qa: QA | None = None


__all__ = ["Placeholder", "Provenance", "QA", "Segment", "SegmentState"]
