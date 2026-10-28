"""Cross-chapter citation linking, internal anchor topology, and hyperlink preservation."""

from __future__ import annotations

from ubt.core.references.linker import (
    CitationTopology,
    build_citation_topology,
    inject_hyperlinks,
)
from ubt.core.references.models import (
    AnchorTarget,
    CitationLink,
    ReferenceKind,
)

__all__ = [
    "AnchorTarget",
    "CitationLink",
    "CitationTopology",
    "ReferenceKind",
    "build_citation_topology",
    "inject_hyperlinks",
]
