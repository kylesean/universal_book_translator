"""Canonical markup hygiene for model repair responses.

:mod:`ubt.core.validators.span_repair` is the only consumer: model repair and
infilling output arrives wrapped in protocol scaffolding with entities still
escaped, and without one canonical pass each loop re-implemented the pattern and
they drifted.
"""

from __future__ import annotations

import html
import re

_PROTOCOL_TAGS_RE = re.compile(
    r"</?(?:error_span|correction|final_translation)(?:\s+[^>]*)?>",
    re.IGNORECASE,
)


def strip_protocol_tags(text: str) -> str:
    """Remove the ``error_span`` / ``correction`` / ``final_translation`` scaffolding."""
    return _PROTOCOL_TAGS_RE.sub("", text)


def clean_model_repair_text(text: str) -> str:
    """Strip protocol markers and decode HTML entities from LLM repair outputs.

    Models frequently echo escaped entities (e.g. ``AT&amp;T``) because the prompt
    context was sanitized/escaped. This function strips protocol scaffolding tags
    and restores natural text.
    """
    cleaned = strip_protocol_tags(text).strip()
    return html.unescape(cleaned)


__all__ = [
    "clean_model_repair_text",
    "strip_protocol_tags",
]
