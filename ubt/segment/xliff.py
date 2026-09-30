"""XLIFF 2.1 export / import for segments (ADR-0001 Phase 2).

XLIFF is the translation industry's interchange format (OASIS 2.1): the same
``Segment`` shape CAT tools -- Trados, memoQ, Phrase -- read and write. Emitting
it is what makes the kernel's units *interoperable* rather than a private model,
and importing it is what lets a human post-edit land back as a segment.

The two hard mappings:

- **Protected spans -> inline codes.** A placeholder token becomes a ``<ph>``
  element whose ``dataRef`` is the token (the round-trip key) and whose text is
  the original span (what a translator sees). Restoring on import puts the token
  back, so the bijection is exact.
- **State -> ``state``.** The five kernel states map onto five distinct XLIFF
  states, so a round-trip cannot collapse them.

The view models only what XLIFF can carry: identity, source, target, state, and
the visible placeholders. Nested placeholders internal to the restore pass are
not part of the interchange -- see :class:`~ubt.segment.placeholders.MaskedSource`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from xml.etree import ElementTree as ET

from ubt.model.segment import Placeholder, Segment, SegmentState

NS = "urn:oasis:names:tc:xliff:document:2.1"
_Q = f"{{{NS}}}"
ET.register_namespace("", NS)

#: KERNEL state -> XLIFF state (five distinct values, so the map is invertible).
_STATE_OUT: dict[SegmentState, str] = {
    SegmentState.NEW: "initial",
    SegmentState.TRANSLATED: "translated",
    SegmentState.VERIFIED: "reviewed",
    SegmentState.FINAL: "final",
    SegmentState.BLOCKED: "needs-review-translation",
}
_STATE_IN: dict[str, SegmentState] = {value: key for key, value in _STATE_OUT.items()}

#: Characters XML 1.0 forbids: C0 controls (minus tab/newline/carriage-return),
#: lone surrogates, and the non-characters. XLIFF cannot carry them, so they are
#: stripped rather than emitted into an unparsable file.
_XML_INVALID = re.compile("[^\u0009\u000a\u000d\u0020-\ud7ff\ue000-\ufffd\U00010000-\U0010ffff]")


def xml_safe(text: str) -> str:
    """Strip characters XML 1.0 forbids, so the output is always well-formed.

    Text pulled from PDFs can carry control bytes; ElementTree would write them
    literally and the reader would reject the file. Sanitizing is the only sound
    option -- an interchange format that cannot round-trip those bytes must not
    pretend to.
    """
    return _XML_INVALID.sub("", text)


@dataclass(frozen=True, slots=True)
class XliffDocument:
    """A parsed XLIFF 2.1 document: the languages and the segments it carried."""

    src_lang: str
    trg_lang: str
    original: str
    segments: tuple[Segment, ...]


def _emit_inline(element: ET.Element, text: str, placeholders: tuple[Placeholder, ...]) -> None:
    """Write ``text`` into ``element``, turning each placeholder token into a ``<ph>``."""
    by_token = {placeholder.token: placeholder for placeholder in placeholders if placeholder.token}
    visible = [token for token in by_token if token in text]
    if not visible:
        element.text = xml_safe(text)
        return
    # Longest token first so a token that is a prefix of another still matches whole.
    pattern = re.compile("|".join(re.escape(t) for t in sorted(visible, key=len, reverse=True)))
    anchor: ET.Element = element
    cursor = 0
    index = 0
    matched = False
    for index, match in enumerate(pattern.finditer(text), start=1):
        matched = True
        if cursor == 0:
            element.text = xml_safe(text[: match.start()])
        else:
            anchor.tail = xml_safe(text[cursor : match.start()])
        placeholder = by_token[match.group(0)]
        code = ET.SubElement(
            element,
            f"{_Q}ph",
            {"id": f"p{index}", "type": placeholder.kind, "dataRef": placeholder.token},
        )
        code.text = xml_safe(placeholder.original)
        anchor = code
        cursor = match.end()
    if matched:
        anchor.tail = xml_safe(text[cursor:])
    else:
        element.text = xml_safe(text)


def _read_inline(element: ET.Element, placeholders: dict[str, Placeholder]) -> str:
    """Rebuild the text, replacing every ``<ph>`` with its ``dataRef`` token."""
    parts: list[str] = [element.text or ""]
    for child in element:
        token = child.get("dataRef") or child.get("data-ref") or ""
        if token:
            placeholders[token] = Placeholder(
                token=token,
                kind=child.get("type") or "",
                original=child.text or "",
            )
        parts.append(token)
        parts.append(child.tail or "")
    return "".join(parts)


def to_xliff(
    segments: tuple[Segment, ...] | list[Segment],
    *,
    src_lang: str,
    trg_lang: str,
    original: str = "",
) -> str:
    """Serialize segments to an XLIFF 2.1 document (one ``unit`` per segment)."""
    root = ET.Element(f"{_Q}xliff", {"version": "2.1", "srcLang": src_lang, "trgLang": trg_lang})
    file_el = ET.SubElement(root, f"{_Q}file", {"id": "f1", "original": xml_safe(original)})
    for segment in segments:
        unit = ET.SubElement(file_el, f"{_Q}unit", {"id": xml_safe(segment.id)})
        seg_el = ET.SubElement(
            unit, f"{_Q}segment", {"id": "s1", "state": _STATE_OUT[segment.state]}
        )
        _emit_inline(ET.SubElement(seg_el, f"{_Q}source"), segment.source, segment.placeholders)
        if segment.target is not None:
            _emit_inline(ET.SubElement(seg_el, f"{_Q}target"), segment.target, segment.placeholders)
    # No pretty-printing: ET.indent() injects whitespace into text/tail, which
    # corrupts mixed content (prose interleaved with inline <ph> codes). The
    # source/target text must round-trip byte-for-byte.
    body = ET.tostring(root, encoding="unicode")
    # XML parsers normalize a literal carriage return to a newline; a character
    # reference survives intact, so a source that came in with CRLF round-trips
    # byte-for-byte instead of silently losing its CRs.
    body = body.replace("\r", "&#13;")
    return f'<?xml version="1.0" encoding="UTF-8"?>\n{body}\n'


def from_xliff(text: str) -> XliffDocument:
    """Parse an XLIFF 2.1 document back into segments."""
    root = ET.fromstring(text)
    src_lang = root.get("srcLang") or ""
    trg_lang = root.get("trgLang") or ""
    original = ""
    file_el = root.find(f"{_Q}file")
    if file_el is not None:
        original = file_el.get("original") or ""
    segments: list[Segment] = []
    for unit in root.iter(f"{_Q}unit"):
        seg_el = unit.find(f"{_Q}segment")
        if seg_el is None:
            continue
        source_el = seg_el.find(f"{_Q}source")
        target_el = seg_el.find(f"{_Q}target")
        placeholders: dict[str, Placeholder] = {}
        source = _read_inline(source_el, placeholders) if source_el is not None else ""
        target = _read_inline(target_el, placeholders) if target_el is not None else None
        segments.append(
            Segment(
                id=unit.get("id") or "",
                source=source,
                placeholders=tuple(placeholders.values()),
                target=target,
                state=_STATE_IN.get(seg_el.get("state") or "", SegmentState.NEW),
            )
        )
    return XliffDocument(
        src_lang=src_lang, trg_lang=trg_lang, original=original, segments=tuple(segments)
    )


__all__ = ["NS", "XliffDocument", "from_xliff", "to_xliff", "xml_safe"]
