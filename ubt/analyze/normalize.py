"""Canonical text normalization (ADR-0001 §12 Q3).

Decision: the canonical stream is stable for *presentation* characters that
carry no translation meaning, and is otherwise left byte-faithful to the source
text. Concretely:

- Latin presentation ligatures (``ﬁ`` → ``fi``) are expanded, because a reader,
  a verifier or a translator comparing characters must see the letters, not one
  glyph.
- Invisible formatting characters — soft hyphen, zero-width space/joiners, the
  word-joiner, a stray BOM — are removed. They are invisible, so keeping them
  would let two "identical" strings differ.
- A non-breaking space becomes an ordinary space.

Deliberately *not* done: case folding, width folding (CJK compatibility), or
Unicode NFKC in general. Those would rewrite text a reader can see and change
what "source kept" means. Span offsets are assigned **after** normalization, so
they stay consistent with the normalized text.
"""

from __future__ import annotations

#: Presentation ligatures and the letters they stand for.
_LIGATURES: dict[str, str] = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\ufb05": "ft",
    "\ufb06": "st",
}

#: Invisible characters with no translation meaning.
_INVISIBLE: tuple[str, ...] = (
    "\u00ad",  # soft hyphen
    "\u200b",  # zero-width space
    "\u200c",  # zero-width non-joiner
    "\u200d",  # zero-width joiner
    "\u2060",  # word joiner
    "\ufeff",  # byte-order mark
)

_TRANSLATION: dict[int, str | None] = {ord(ch): text for ch, text in _LIGATURES.items()}
_TRANSLATION.update({ord(ch): None for ch in _INVISIBLE})


def normalize_text(text: str) -> str:
    """Expand presentation ligatures and drop invisible formatting characters."""
    if not text:
        return text
    return text.translate(_TRANSLATION).replace("\u00a0", " ")


__all__ = ["normalize_text"]
