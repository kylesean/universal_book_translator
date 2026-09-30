"""The ordered content graph a delivery must account for."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from ubt.core.content.nodes import AssetNode, ContentNode, TextNode


class ContentGraph(BaseModel):
    """Every content node of one document, in reading order.

    The graph is the *contract*: a delivery is judged by how it accounts for
    these nodes, never by the renderer that produced it. Two renderers (rigid,
    reflow) over the same graph must each return a disposition covering every
    node; a node missing from that disposition is a violation.
    """

    model_config = ConfigDict(frozen=True)

    doc_id: str = ""
    title: str = ""
    source_path: str = ""
    nodes: tuple[ContentNode, ...] = Field(default_factory=tuple)

    @property
    def text_nodes(self) -> tuple[TextNode, ...]:
        return tuple(n for n in self.nodes if isinstance(n, TextNode))

    @property
    def asset_nodes(self) -> tuple[AssetNode, ...]:
        return tuple(n for n in self.nodes if isinstance(n, AssetNode))

    def by_id(self, node_id: str) -> ContentNode | None:
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None
