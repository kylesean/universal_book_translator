"""Hardened XML parsing for documents the operator did not author.

Every XML surface that reads untrusted input — a TMX import, an EPUB/DOCX
package, an XLIFF review export — must refuse the three XML attack classes
before the tree is built:

* **DTDs and entity definitions** — a billion-laughs document expands a few
  hundred bytes of entity declarations into gigabytes;
* **external references** — an external entity (``SYSTEM "file:///etc/passwd"``)
  or an external DTD is an XXE read of the host, or an SSRF probe;
* **amplification** — even without a DTD, nested entities can blow up.

CPython 3.12's ``xml.etree`` already refuses external entities and caps entity
amplification *by default*, but that is a default we do not control: a future
interpreter, a different build, or a caller that passes its own parser can
silently drop the protection. ``defusedxml`` makes the refusal explicit and
turns any of the three into a single, catchable :class:`UnsafeXMLError`.
"""

from __future__ import annotations

from pathlib import Path
from typing import IO, cast
from xml.etree.ElementTree import Element

from defusedxml.common import DefusedXmlException
from defusedxml.ElementTree import ParseError
from defusedxml.ElementTree import fromstring as _fromstring
from defusedxml.ElementTree import parse as _parse


class UnsafeXMLError(ValueError):
    """The XML is malformed, or uses a forbidden DTD / entity / external reference."""


def parse_xml(text: str | bytes) -> Element:
    """Parse XML from a string or bytes, refusing DTDs, entities and external refs."""
    try:
        return cast(Element, _fromstring(text))
    except (ParseError, DefusedXmlException) as exc:
        raise UnsafeXMLError(str(exc)) from exc


def parse_xml_file(source: str | Path | IO[bytes]) -> Element:
    """Parse XML from a path or file object, refusing DTDs, entities and external refs."""
    try:
        return cast(Element, _parse(source).getroot())
    except (ParseError, DefusedXmlException, OSError) as exc:
        raise UnsafeXMLError(str(exc)) from exc


__all__ = ["UnsafeXMLError", "parse_xml", "parse_xml_file"]
