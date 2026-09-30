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

The document is built with unqualified tags and a single ``xmlns`` on the root,
and parsed by *local name*. This is deliberate: ElementTree's default-namespace
registry is global mutable state that other modules also write (``svg_diagram``
registers one), so depending on it would let an unrelated import change the
XLIFF prefix. The serialized form is namespace-correct XML either way.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from xml.etree import ElementTree as ET

from ubt.model.segment import Placeholder, Segment, SegmentState

NS = "urn:oasis:names:tc:xliff:document:2.1"

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


def _local(tag: str) -> str:
    """Local name of an ElementTree tag, with any ``{namespace}`` stripped."""
    return tag.rsplit("}", 1)[-1]


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
            "ph",
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
    root = ET.Element(
        "xliff",
        {"xmlns": NS, "version": "2.1", "srcLang": src_lang, "trgLang": trg_lang},
    )
    file_el = ET.SubElement(root, "file", {"id": "f1", "original": xml_safe(original)})
    for segment in segments:
        unit = ET.SubElement(file_el, "unit", {"id": xml_safe(segment.id)})
        seg_el = ET.SubElement(unit, "segment", {"id": "s1", "state": _STATE_OUT[segment.state]})
        _emit_inline(ET.SubElement(seg_el, "source"), segment.source, segment.placeholders)
        if segment.target is not None:
            _emit_inline(ET.SubElement(seg_el, "target"), segment.target, segment.placeholders)
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
    """Parse an XLIFF 2.1 document back into segments (namespace-tolerant)."""
    root = ET.fromstring(text)
    src_lang = root.get("srcLang") or ""
    trg_lang = root.get("trgLang") or ""
    file_el = next((el for el in root.iter() if _local(el.tag) == "file"), None)
    original = (file_el.get("original") or "") if file_el is not None else ""
    segments: list[Segment] = []
    for unit in (el for el in root.iter() if _local(el.tag) == "unit"):
        seg_el = next((child for child in unit if _local(child.tag) == "segment"), None)
        if seg_el is None:
            continue
        source_el = next((child for child in seg_el if _local(child.tag) == "source"), None)
        target_el = next((child for child in seg_el if _local(child.tag) == "target"), None)
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
