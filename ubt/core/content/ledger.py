"""The two delivery ledgers: content and assets.

A renderer does not get to declare the delivery valid. It returns, for every
node it saw, what it did with it; the ledgers record that, and
:func:`ubt.core.content.contract.reconcile` checks the books balance. Because
the ledgers are built from the *graph* (not the renderer), a node the renderer
forgot to mention shows up as a missing book entry -- which is exactly the
failure mode that would otherwise slip through as "no error reported".
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from ubt.core.content.graph import ContentGraph
from ubt.core.content.nodes import (
    AssetIntegrity,
    AssetRepresentation,
    TextDisposition,
)


class TextLedgerEntry(BaseModel):
    """One row of the content ledger: what happened to a text node."""

    model_config = ConfigDict(frozen=True)

    node_id: str
    disposition: TextDisposition
    reason: str = ""
    source_chars: int = 0
    target_chars: int = 0

    @property
    def delivered(self) -> bool:
        return self.disposition in (TextDisposition.TRANSLATED, TextDisposition.VERBATIM)

    @property
    def accounted(self) -> bool:
        return self.disposition is not TextDisposition.PENDING


class AssetLedgerEntry(BaseModel):
    """One row of the asset ledger: what happened to a non-text node."""

    model_config = ConfigDict(frozen=True)

    node_id: str
    integrity: AssetIntegrity
    representation: AssetRepresentation
    verified: bool = False
    corrupt: bool = False
    digest: str = ""
    reason: str = ""

    @property
    def accounted(self) -> bool:
        return self.integrity is not AssetIntegrity.MISSING


class ContentLedger(BaseModel):
    """Every translatable text node and its disposition."""

    model_config = ConfigDict(frozen=True)

    entries: dict[str, TextLedgerEntry] = Field(default_factory=dict)

    @property
    def total(self) -> int:
        return len(self.entries)

    @property
    def delivered(self) -> int:
        return sum(1 for e in self.entries.values() if e.delivered)

    @property
    def source_kept(self) -> int:
        return sum(1 for e in self.entries.values() if e.disposition is TextDisposition.SOURCE_KEPT)


class AssetLedger(BaseModel):
    """Every non-text node and its integrity."""

    model_config = ConfigDict(frozen=True)

    entries: dict[str, AssetLedgerEntry] = Field(default_factory=dict)

    @property
    def total(self) -> int:
        return len(self.entries)

    @property
    def preserved(self) -> int:
        return sum(
            1 for e in self.entries.values() if e.integrity is AssetIntegrity.PRESERVED_OPAQUE
        )


def build_ledgers(graph: ContentGraph) -> tuple[ContentLedger, AssetLedger]:
    """Derive both ledgers from the graph's per-node dispositions."""
    content = ContentLedger(
        entries={
            node.id: TextLedgerEntry(
                node_id=node.id,
                disposition=node.disposition,
                reason=node.reason,
                source_chars=len(node.source_text),
                target_chars=len(node.target_text or ""),
            )
            for node in graph.text_nodes
        }
    )
    assets = AssetLedger(
        entries={
            node.id: AssetLedgerEntry(
                node_id=node.id,
                integrity=node.descriptor.integrity,
                representation=node.descriptor.representation,
                verified=node.descriptor.verified,
                corrupt=node.descriptor.corrupt,
                digest=node.descriptor.digest,
                reason=node.descriptor.detail,
            )
            for node in graph.asset_nodes
        }
    )
    return content, assets
