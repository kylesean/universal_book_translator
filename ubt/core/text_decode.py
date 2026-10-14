"""Byte-stream text decoding for documents the operator did not author.

A leaf module: it depends on nothing in :mod:`ubt` so both the core-side
readers (:mod:`ubt.analyze.reader_md`, ``reader_html``) and the adapter layer
can share one decoder without the readers importing ``ubt.adapters`` — a
layering inversion, since the readers sit below the adapters. The declared /
BOM / charset-detected decode order has exactly one owner here.
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
    """Decode an XML/HTML/Markdown byte stream, preferring UTF-8 then its declared encoding.

    ``errors="ignore"`` silently dropped every byte a non-UTF-8 template could
    not represent. UTF-8 is tried first so a *stale* declaration on already
    UTF-8 bytes (every member the EPUB adapter re-serialises) can never
    mis-decode; only genuinely non-UTF-8 bytes fall back to the BOM / XML
    declaration / ``<meta charset>``, then to a replacement decode so nothing
    disappears without a visible marker.
    """
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig")
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
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
