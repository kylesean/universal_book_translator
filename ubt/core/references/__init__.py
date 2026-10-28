"""Cross-chapter citation linking, internal anchor topology, and hyperlink preservation."""

from __future__ import annotations

from ubt.core.references.linker import (
    CitationTopology,
    build_citation_topology,
    format_mention_text,
    inject_hyperlinks,
)
from ubt.core.references.models import (
    AnchorTarget,
    CitationLink,
    CrossRefConfig,
    CrossRefPrefixScheme,
    CrossRefStyle,
    ReferenceKind,
)

__all__ = [
    "AnchorTarget",
    "CitationLink",
    "CitationTopology",
    "CrossRefConfig",
    "CrossRefPrefixScheme",
    "CrossRefStyle",
    "ReferenceKind",
    "build_citation_topology",
    "format_mention_text",
    "inject_hyperlinks",
]
