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

Third, this module also carries the raw-bytes scanner the *non-defused*
parsers need. ``parse_xml``/``parse_xml_file`` only protect the ElementTree
surfaces; the EPUB chapters and package documents are parsed with
BeautifulSoup's lxml backend, and DOCX parts by python-docx's lxml — neither
of which goes through defusedxml. Their callers scan the raw bytes with
:func:`strip_doctype_declarations` (XHTML-ish content, where a DOCTYPE is a
parsing nicety and can be dropped) or :func:`assert_no_doctype` (OOXML parts,
where a DOCTYPE is never legitimate and the document fails closed) *before*
the parser ever sees the bytes.
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


#: The DOCTYPE keyword is case-sensitive per the XML grammar (``doctypedecl``
#: begins with exactly ``<!DOCTYPE``), so no ``re.IGNORECASE`` here.
_DOCTYPE_TOKEN = "<!DOCTYPE"

#: An XML stylesheet processing instruction: ``<?xml-stylesheet ... ?>``. Like
#: the DOCTYPE it is instructions to the *parser/renderer*, not content, and a
#: crafted one has no business in an untrusted document part.
_STYLESHEET_PI_TOKEN = "<?xml-stylesheet"


def _find_doctype_start(text: str) -> int:
    """Index of the first DOCTYPE declaration in ``text``, or ``-1``."""
    return text.find(_DOCTYPE_TOKEN)


def _find_stylesheet_pi_start(text: str) -> int:
    """Index of the first xml-stylesheet processing instruction, or ``-1``."""
    return text.find(_STYLESHEET_PI_TOKEN)


def strip_doctype_declarations(text: str) -> str:
    """Remove DOCTYPE declarations and xml-stylesheet PIs from raw markup.

    For XHTML-ish content — EPUB chapters, OPF packages, NCX maps — where the
    DOCTYPE is a parsing nicety and no entity declared inside it may be
    trusted: stripping the whole declaration (internal subset included)
    neutralizes billion-laughs expansions and external-entity reads before the
    lxml-based parser runs, instead of relying on the parser build's defaults.
    The scanner walks quotes, so a ``">"`` inside an entity's default value
    does not end the declaration early. An unterminated declaration is
    stripped to the end of the string: the parser would reject the document
    anyway, and there is nothing before the declaration worth keeping less
    than what follows it.
    """
    out: list[str] = []
    pos = 0
    while True:
        doctype = text.find(_DOCTYPE_TOKEN, pos)
        stylesheet = text.find(_STYLESHEET_PI_TOKEN, pos)
        if doctype == -1 and stylesheet == -1:
            out.append(text[pos:])
            break
        next_token = min(
            (idx for idx in (doctype, stylesheet) if idx != -1),
        )
        out.append(text[pos:next_token])
        if next_token == stylesheet:
            # A PI ends at the first "?>" and cannot contain it.
            end = text.find("?>", next_token)
            pos = len(text) if end == -1 else end + 2
            continue
        # Walk the DOCTYPE: the declaration ends at the first ">" that is
        # neither inside a quoted value ("..." or '...') nor inside the
        # internal subset bracket pair. Without the bracket depth, the ">" of
        # each ``<!ENTITY ...>`` inside ``[ ... ]`` ended the walk early and
        # left a stray ``]>`` behind — the document then failed to parse
        # instead of being neutralized.
        pos = next_token + len(_DOCTYPE_TOKEN)
        quote: str | None = None
        subset_depth = 0
        while pos < len(text):
            ch = text[pos]
            if quote is not None:
                if ch == quote:
                    quote = None
            elif ch in ('"', "'"):
                quote = ch
            elif ch == "[":
                subset_depth += 1
            elif ch == "]":
                subset_depth = max(0, subset_depth - 1)
            elif ch == ">" and subset_depth == 0:
                pos += 1
                break
            pos += 1
        else:
            # Unterminated: drop everything to the end of the string.
            pos = len(text)
    return "".join(out)


def assert_no_doctype(data: str | bytes, *, context: str) -> None:
    """Fail closed when raw markup carries a DOCTYPE or an xml-stylesheet PI.

    For document parts where a DOCTYPE is never legitimate — OOXML package
    parts (DOCX/XLSX/PPTX ``word/document.xml`` and friends) — so the caller
    refuses the *document*, not just the declaration: there is no harmless
    subset of a DTD inside an OOXML part.
    """
    text = data.decode("utf-8", errors="replace") if isinstance(data, bytes) else data
    for finder, what in (
        (_find_doctype_start, "a DOCTYPE declaration"),
        (_find_stylesheet_pi_start, "an xml-stylesheet processing instruction"),
    ):
        if finder(text) != -1:
            raise UnsafeXMLError(
                f"{context}: refusing untrusted XML that carries {what} — "
                "OOXML parts have no legitimate DTD, and entity declarations "
                "are an XXE/amplification vector. The document is refused."
            )


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


__all__ = [
    "UnsafeXMLError",
    "assert_no_doctype",
    "parse_xml",
    "parse_xml_file",
    "strip_doctype_declarations",
]
