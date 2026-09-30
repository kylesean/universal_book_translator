"""``ubt.segment`` -- the translation-unit layer (ADR-0001 Phase 2).

- :class:`~ubt.segment.placeholders.PlaceholderEngine` -- the single owner of the
  placeholder mask order and its reverse.
- :class:`~ubt.segment.placeholders.MaskedSource` / ``RestoreOutcome`` -- the
  inputs and the verified output of that engine.

The translation-unit types themselves (:class:`~ubt.model.segment.Segment`,
:class:`~ubt.model.segment.Placeholder`) live in :mod:`ubt.model.segment`.
"""

from __future__ import annotations

from ubt.segment.placeholders import MaskedSource, PlaceholderEngine, RestoreOutcome

__all__ = ["MaskedSource", "PlaceholderEngine", "RestoreOutcome"]
