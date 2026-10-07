"""Project a run's attestations onto the delivery contract (one build path).

The delivery contract has a single primary source: the per-element attestations
``realize()`` produced at construction time. The content graph is *not* a second
verdict -- it only classifies the delivered blocks into the contract's vocabulary
and supplies the detail the AST does not model (why a text node was kept, the
named violations). :func:`contract_from_attestations` owns that projection so the
export stage and ``ubt verify`` obtain the contract the same way and cannot
drift apart.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from ubt.core.content.adapt import graph_from_blocks
from ubt.core.content.contract import ReconciliationReport
from ubt.core.ir.models import IRBlock

if TYPE_CHECKING:
    from ubt.pipeline.attest import AttestationReport


def contract_from_attestations(
    report: AttestationReport,
    blocks: Sequence[IRBlock],
    *,
    doc_id: str = "",
    title: str = "",
    source_path: str = "",
) -> ReconciliationReport:
    """The delivery contract: the attestation projection over the block graph."""
    from ubt.pipeline.attest import project_contract

    graph = graph_from_blocks(
        blocks,
        doc_id=doc_id,
        title=title,
        source_path=source_path,
    )
    return project_contract(report, graph)


__all__ = ["contract_from_attestations"]
