"""Codec registry mapping document extensions to UBT v2 surgical codecs."""

from __future__ import annotations

from pathlib import Path

from ubt.codecs.docx.codec import DOCXCodec
from ubt.codecs.dom.codec import DOMCodec
from ubt.codecs.markdown.codec import MarkdownCodec
from ubt.codecs.pdf.codec import NativePDFCodec
from ubt.codecs.protocol import DocumentCodec
from ubt.core.exceptions import UnsupportedDocumentFormatError

#: Every suffix a v2 ``DocumentCodec`` owns. Kept next to ``resolve_codec`` so a
#: format added here is also visible to the CLI/API suffix surface without a
#: second hardcoded list drifting.
CODEC_SUFFIXES: frozenset[str] = frozenset(
    {".pdf", ".epub", ".html", ".htm", ".xhtml", ".docx", ".md", ".markdown", ".txt"}
)


def resolve_codec(source_path: Path) -> DocumentCodec:
    """Instantiate the native surgical :class:`DocumentCodec` for ``source_path``."""
    suffix = source_path.suffix.lower()
    if suffix == ".pdf":
        return NativePDFCodec()
    if suffix in {".epub", ".html", ".htm", ".xhtml"}:
        return DOMCodec()
    if suffix == ".docx":
        return DOCXCodec()
    if suffix in {".md", ".markdown", ".txt"}:
        return MarkdownCodec()
    raise UnsupportedDocumentFormatError(
        f"Unsupported document format: {suffix!r} ({source_path.name})"
    )
