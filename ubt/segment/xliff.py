"""XLIFF 2.1 export / import for segments (translation unit segmentation layer).

XLIFF is the translation industry's interchange format (OASIS 2.1): the same
``Segment`` shape CAT tools -- Trados, memoQ, Phrase -- read and write. Emitting
it is what makes the kernel's units *interoperable* rather than a private model,
and importing it is what lets a human post-edit land back as a segment.

The two hard mappings:

- **Protected spans -> inline codes.** A placeholder token becomes a ``<ph>``
  element whose ``dataRef`` is the token (the round-trip key) and whose text is
  the original span (what a translator sees). Restoring on import puts the token
  back, so the bijection is exact.
- **State -> ``state``.** Each kernel state maps to its own XLIFF state, so a
  round-trip cannot collapse them. The distinction that matters most is
  ``NEEDS_REVIEW`` (a draft shipped, post-edit it) versus ``BLOCKED`` (nothing
  shipped; the unit still needs translating) -- these were once both emitted as
  states that hid them, and they are the two blocks a reviewer is looking for.

A QA verdict travels as canonical ``<notes>`` (XLIFF 2.1's own extension point,
not a private element): ``ubt-mtqe-score``, ``ubt-mqm-severity`` and one
``ubt-error-flag`` per flag. Notes are advisory -- the import stays lenient
about them -- but they turn a queue of ids into a queue a reviewer can work.

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

from ubt.core.xml_safety import parse_xml
from ubt.model.segment import QA, Placeholder, Segment, SegmentState

NS = "urn:oasis:names:tc:xliff:document:2.1"

#: KERNEL state -> XLIFF state (distinct values, so the map is invertible).
#:
#: ``NEEDS_REVIEW`` and ``BLOCKED`` look similar and are opposites in practice,
#: which is why they need separate states rather than sharing ``translated``:
#:
#: - ``NEEDS_REVIEW`` -- a real draft shipped, and a human should post-edit it.
#: - ``BLOCKED`` -- no translation shipped; the target carries the source under
#:   a quarantine marker, so the unit still needs translating.
#:
#: Both used to collapse: ``needs_human`` was emitted as ``translated`` (so a
#: reviewer could not find it among a thousand clean blocks) while
#: ``blocked_human`` was emitted as ``needs-review-translation`` (implying a
#: translation existed to review when none had shipped). The routing was right
#: and the interchange format threw the routing away.
_STATE_OUT: dict[SegmentState, str] = {
    SegmentState.NEW: "initial",
    SegmentState.TRANSLATED: "translated",
    SegmentState.NEEDS_REVIEW: "needs-review-translation",
    SegmentState.VERIFIED: "reviewed",
    SegmentState.FINAL: "final",
    SegmentState.BLOCKED: "needs-translation",
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


def _qa_note_values(qa: QA) -> list[tuple[str, str]]:
    """The ``(category, text)`` notes that describe one QA verdict.

    Only fields that carry information are emitted. A note reading
    ``ubt-mtqe-score: 0.0`` for an unscored block would be indistinguishable
    from a real score of zero, so an absent value emits no note at all.

    One note per flag rather than a joined list: a CAT tool displays notes
    separately, and a reviewer scanning for "which defect" should not have to
    split on a delimiter whose format is ours alone.
    """
    notes: list[tuple[str, str]] = []
    if qa.score is not None:
        notes.append(("ubt-mtqe-score", f"{qa.score:.2f}"))
    if qa.severity:
        notes.append(("ubt-mqm-severity", qa.severity))
    notes.extend(("ubt-error-flag", flag) for flag in qa.flags)
    return notes


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


def _local_name(tag: object) -> str:
    """Tag name without the XLIFF namespace brace, for interchange robustness."""
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def _read_inline(element: ET.Element, placeholders: dict[str, Placeholder]) -> str:
    """Rebuild the text, replacing every ``<ph>`` with its ``dataRef`` token.

    Only ``<ph>`` maps to a placeholder. CAT tools also emit paired inline
    markup (``<pc>``); treating every child as a placeholder would append an
    empty token and silently drop the wrapped text, breaking the roundtrip.

    A ``<ph>`` without ``dataRef``/``data-ref`` carries no token to restore, so
    its visible text is kept verbatim (a third-party CAT tool may emit
    ``<ph id="1">Foo</ph>``); appending the empty token instead dropped it.
    """
    parts: list[str] = [element.text or ""]
    for child in element:
        if _local_name(child.tag) == "ph":
            token = child.get("dataRef") or child.get("data-ref") or ""
            if token:
                placeholders[token] = Placeholder(
                    token=token,
                    kind=child.get("type") or "",
                    original=child.text or "",
                )
                parts.append(token)
            else:
                parts.append("".join(child.itertext()))
        else:
            parts.append("".join(child.itertext()))
        parts.append(child.tail or "")
    return "".join(parts)


def _read_qa(unit: ET.Element) -> QA | None:
    """Rebuild the QA verdict from a unit's ``ubt-*`` notes.

    Returns ``None`` when the unit carries no QA note, matching
    :func:`ubt.segment.document._block_qa`: an absent verdict and an empty one
    are different facts and must not collapse into the same value.

    The parser is deliberately lenient about which notes exist (a hand-edited or
    third-party document may carry only a score) but strict about the payload:
    a score that is not a number is dropped rather than crashing the import. QA
    notes are advisory metadata; a malformed one must not make a translation
    file unreadable.
    """
    score: float | None = None
    severity: str | None = None
    flags: list[str] = []
    found = False
    for notes_el in (child for child in unit if _local_name(child.tag) == "notes"):
        for note in notes_el:
            if _local_name(note.tag) != "note":
                continue
            category = note.get("category") or ""
            text = (note.text or "").strip()
            if category == "ubt-mtqe-score" and text:
                found = True
                try:
                    score = float(text)
                except ValueError:
                    score = None
            elif category == "ubt-mqm-severity" and text:
                found = True
                severity = text
            elif category == "ubt-error-flag" and text:
                found = True
                flags.append(text)
    if not found:
        return None
    return QA(score=score, severity=severity, flags=tuple(flags))


def to_xliff(
    segments: tuple[Segment, ...] | list[Segment],
    *,
    src_lang: str,
    trg_lang: str,
    original: str = "",
    job_id: str = "",
) -> str:
    """Serialize segments to an XLIFF 2.1 document (one ``unit`` per segment)."""
    root = ET.Element(
        "xliff",
        {"xmlns": NS, "version": "2.1", "srcLang": src_lang, "trgLang": trg_lang},
    )
    file_el = ET.SubElement(root, "file", {"id": "f1", "original": xml_safe(original)})
    if job_id:
        file_notes = ET.SubElement(file_el, "notes")
        file_note = ET.SubElement(file_notes, "note", {"category": "ubt-job-id"})
        file_note.text = xml_safe(job_id)
    for segment in segments:
        unit = ET.SubElement(file_el, "unit", {"id": xml_safe(segment.id)})
        seg_job = getattr(segment, "job_id", "") or job_id
        if seg_job:
            notes_el = ET.SubElement(unit, "notes")
            note_el = ET.SubElement(notes_el, "note", {"category": "ubt-job-id"})
            note_el.text = xml_safe(seg_job)
        # What the QA verdict was, next to the text it applies to. The state
        # says a block needs attention; these say why, which is the difference
        # between a queue a reviewer can work and a list of opaque ids.
        if segment.qa is not None:
            qa_notes = ET.SubElement(unit, "notes")
            for category, value in _qa_note_values(segment.qa):
                qa_note = ET.SubElement(qa_notes, "note", {"category": category})
                qa_note.text = xml_safe(value)
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
    # Hardened parse: an XLIFF handed to a review import is untrusted input, so
    # a DTD, entity definition or external reference is refused rather than
    # expanded (billion-laughs / XXE).
    root = parse_xml(text)
    src_lang = root.get("srcLang") or ""
    trg_lang = root.get("trgLang") or ""
    file_el = next((el for el in root.iter() if _local(el.tag) == "file"), None)
    original = (file_el.get("original") or "") if file_el is not None else ""
    segments: list[Segment] = []
    for unit in (el for el in root.iter() if _local(el.tag) == "unit"):
        # XLIFF 2.1 lets one ``<unit>`` carry several ``<segment>`` children —
        # that is exactly how a CAT tool records its own sentence splits. Taking
        # only the first silently dropped every later sentence on import.
        unit_id = unit.get("id") or ""
        seg_elements = [child for child in unit if _local(child.tag) == "segment"]
        for position, seg_el in enumerate(seg_elements):
            source_el = next((child for child in seg_el if _local(child.tag) == "source"), None)
            target_el = next((child for child in seg_el if _local(child.tag) == "target"), None)
            placeholders: dict[str, Placeholder] = {}
            source = _read_inline(source_el, placeholders) if source_el is not None else ""
            target = _read_inline(target_el, placeholders) if target_el is not None else None
            # The first segment keeps the bare unit id (a single-segment unit is
            # the common case and its id is the one a caller already knows);
            # later ones get a stable position suffix so the ids stay unique.
            seg_id = unit_id if position == 0 else f"{unit_id}-{position}"
            segments.append(
                Segment(
                    id=seg_id,
                    source=source,
                    placeholders=tuple(placeholders.values()),
                    target=target,
                    state=_STATE_IN.get(seg_el.get("state") or "", SegmentState.NEW),
                    qa=_read_qa(unit),
                )
            )
    return XliffDocument(
        src_lang=src_lang, trg_lang=trg_lang, original=original, segments=tuple(segments)
    )


__all__ = ["NS", "XliffDocument", "from_xliff", "to_xliff", "xml_safe"]
