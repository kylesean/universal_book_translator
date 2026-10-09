"""``ubt.segment`` -- the translation-unit layer (translation unit segmentation layer).

- :class:`~ubt.segment.placeholders.PlaceholderEngine` -- the single owner of the
  placeholder mask order and its reverse.
- :class:`~ubt.segment.placeholders.MaskedSource` / ``RestoreOutcome`` -- the
  inputs and the verified output of that engine.

The translation-unit types themselves (:class:`~ubt.model.segment.Segment`,
:class:`~ubt.model.segment.Placeholder`) live in :mod:`ubt.model.segment`.
"""

from __future__ import annotations

from ubt.segment.document import segments_from_blocks
from ubt.segment.placeholders import MaskedSource, PlaceholderEngine, RestoreOutcome
from ubt.segment.spi import register_spi_providers
from ubt.segment.xliff import XliffDocument, from_xliff, to_xliff

__all__ = [
    "MaskedSource",
    "PlaceholderEngine",
    "RestoreOutcome",
    "XliffDocument",
    "from_xliff",
    "register_spi_providers",
    "segments_from_blocks",
    "to_xliff",
]
