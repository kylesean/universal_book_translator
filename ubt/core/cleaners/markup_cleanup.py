"""Canonical markup hygiene and repair protocol cleaning.

Shared by cleaners and validators to eliminate duplicated regex passes and
inconsistent entity unescaping when processing model repair responses.
"""

from __future__ import annotations

import html
import re

_PROTOCOL_TAGS_RE = re.compile(
    r"</?(?:error_span|correction|final_translation)(?:\s+[^>]*)?>",
    re.IGNORECASE,
)


def strip_protocol_tags(text: str) -> str:
    """Strip protocol XML/HTML tags emitted during repair and infilling loops."""
    return _PROTOCOL_TAGS_RE.sub("", text)


def clean_model_repair_text(text: str) -> str:
    """Strip protocol markers and decode HTML entities from LLM repair outputs.

    Models frequently echo escaped entities (e.g. ``AT&amp;T``) because the prompt
    context was sanitized/escaped. This function strips protocol scaffolding tags
    and restores natural text.
    """
    cleaned = strip_protocol_tags(text).strip()
    return html.unescape(cleaned)


def normalize_escaped_entities(text: str) -> str:
    """Normalize HTML entity sequences in translated text."""
    return html.unescape(text)


__all__ = [
    "clean_model_repair_text",
    "normalize_escaped_entities",
    "strip_protocol_tags",
]
