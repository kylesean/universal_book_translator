"""Byte-stream text decoding for documents the operator did not author.

A leaf module: it depends on nothing in :mod:`ubt` so both the core-side
readers (:mod:`ubt.analyze.reader_md`, ``reader_html``) and the adapter layer
can share one decoder without the readers importing ``ubt.adapters`` — a
layering inversion, since the readers sit below the adapters. The BOM / UTF-8 /
declared / detected decode order has exactly one owner here.
"""

from __future__ import annotations

import re

import charset_normalizer

# Matches both XML ``encoding="..."`` and HTML ``charset="..."`` declarations
# (``<meta charset="gbk">`` and ``content="text/html; charset=gbk"``), so the
# same sniffer serves XHTML members, standalone HTML and any other markup.
_ENCODING_DECL_RE = re.compile(
    rb"""(?:encoding|charset)\s*=\s*["']?([A-Za-z0-9._-]+)["']?""", re.IGNORECASE
)

#: Candidate encodings charset-normalizer may choose between when a source has
#: neither a BOM nor a declared encoding. Restricting the set keeps it from
#: guessing UTF-16 or Korean for a short GBK sample; on whole-file input the
#: guess is reliable, but a narrow plausible set makes short files reliable too.
_FALLBACK_ENCODINGS = ("utf_8", "gb18030", "big5", "cp932", "cp1252")


def decode_markup(raw: bytes) -> str:
    """Decode an XML/HTML/Markdown byte stream, in this order and for these reasons.

    A BOM first: it is the source *asserting* its encoding, so it is honoured
    before anything is guessed (UTF-8 BOM decodes as ``utf-8-sig``, a UTF-16 BOM
    as UTF-16). Then strict UTF-8, ahead of the declaration, so a *stale*
    declaration on already-UTF-8 bytes (every member the EPUB adapter
    re-serialises) can never mis-decode. Then the declared
    ``encoding=``/``charset=`` read out of the first 1024 bytes, then charset
    detection over :data:`_FALLBACK_ENCODINGS`, and only then a replacement
    decode.

    The last step is ``errors="replace"``, not ``errors="ignore"``: ignoring
    silently dropped every byte a non-UTF-8 template could not represent, while
    a replacement marker makes the loss visible.
    """
    if raw.startswith(b"\xef\xbb\xbf"):
        try:
            return raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            # A truncated UTF-8-BOM file: fall through to the ordinary ladder
            # instead of raising out of an adapter's ``DocumentParseError``
            # contract.
            pass
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        try:
            return raw.decode("utf-16")
        except UnicodeDecodeError:
            # An odd byte count (truncated file) or an unpaired surrogate used to
            # raise a bare UnicodeDecodeError straight through the Markdown/EPUB
            # readers. Decode leniently: the BOM still tells us the family, and a
            # visible replacement is better than a job that dies with an
            # unclassified error.
            return raw.decode("utf-16", errors="replace")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        pass
    match = _ENCODING_DECL_RE.search(raw[:1024])
    if match:
        try:
            return raw.decode(match.group(1).decode("ascii", "ignore"))
        except (LookupError, UnicodeDecodeError):
            pass
    detected = _detect_and_decode(raw)
    if detected is not None:
        return detected
    return raw.decode("utf-8", errors="replace")


def _detect_and_decode(raw: bytes) -> str | None:
    """Decode bytes with no BOM and no declared encoding via charset detection.

    A GBK/Shift-JIS ``.md``/``.txt`` carries neither, so UTF-8 + BOM +
    declaration all fail; ``charset-normalizer`` is the last resort before the
    lossy ``errors="replace"`` decode. Returns ``None`` when it cannot decide.
    """
    best = charset_normalizer.from_bytes(raw, cp_isolation=list(_FALLBACK_ENCODINGS)).best()
    return str(best) if best is not None else None


__all__ = ["decode_markup"]
