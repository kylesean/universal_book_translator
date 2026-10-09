"""Reconciliation: does a delivery account for every content node?

This is the gate the whole architecture exists to provide. It is engine-,
language- and layout-agnostic: it reads the content graph and the ledgers, not
the render path. A regression on any path therefore surfaces the same way -- as
an unbalanced book -- instead of as a document-type-specific bug.

Severity tracks what a delivery can prove. *Unaccounted* and *dropped*
content are hard errors -- absence is detectable without a verifier. An
unverified reconstruction is a *warning*: the asset was accounted for, but no
round-trip check has proven the reconstruction faithful.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from ubt.core.content.graph import ContentGraph
from ubt.core.content.ledger import AssetLedger, ContentLedger, build_ledgers
from ubt.core.content.nodes import AssetIntegrity, TextDisposition


class ViolationKind(StrEnum):
    TEXT_UNACCOUNTED = "text_unaccounted"  # decided neither translated nor kept
    TEXT_UNDELIVERED = "text_undelivered"  # dropped without a kept-verbatim reason
    TEXT_SOURCE_KEPT = "text_source_kept"  # shipped source (translation not placed)
    ASSET_MISSING = "asset_missing"  # non-text content lost
    ASSET_CORRUPT_RECONSTRUCTION = "asset_corrupt_reconstruction"  # shattered/erroring rebuild
    ASSET_UNVERIFIED_RECONSTRUCTION = "asset_unverified_reconstruction"


class Severity(StrEnum):
    ERROR = "error"  # fails the contract
    WARNING = "warning"  # reported, does not fail (yet)


#: Source-kept reasons that are a *space* failure (Axiom B): the translation
#: existed but could not be placed, so the reader loses it. These are errors.
#: Other source-kept reasons (unrenderable math, a quarantined block) are honest,
#: explicit keeps -- permitted by Axiom B -- and stay warnings.
_SPACE_FAILURE_MARKERS = ("spill", "overflow", "no_fit", "does not fit")


def _is_space_failure(reason: str) -> bool:
    lowered = reason.lower()
    return any(marker in lowered for marker in _SPACE_FAILURE_MARKERS)


class Violation(BaseModel):
    """One way the delivery failed to account for the graph."""

    model_config = ConfigDict(frozen=True)

    kind: ViolationKind
    severity: Severity
    node_id: str
    detail: str = ""


class ReconciliationReport(BaseModel):
    """The delivery contract, serialized as ``*.contract.json``.

    ``passed`` is the single machine-checkable verdict: no ERROR-severity
    violation. Warnings are surfaced for audit but do not block.
    """

    model_config = ConfigDict(frozen=True)

    doc_id: str = ""
    title: str = ""
    schema_version: int = 1

    total_text: int = 0
    delivered_text: int = 0
    verbatim_text: int = 0
    source_kept_text: int = 0
    skipped_text: int = 0
    pending_text: int = 0

    total_assets: int = 0
    reconstructed_assets: int = 0
    preserved_assets: int = 0
    dropped_assets: int = 0
    missing_assets: int = 0

    violations: tuple[Violation, ...] = Field(default_factory=tuple)

    @property
    def errors(self) -> tuple[Violation, ...]:
        return tuple(v for v in self.violations if v.severity is Severity.ERROR)

    @property
    def warnings(self) -> tuple[Violation, ...]:
        return tuple(v for v in self.violations if v.severity is Severity.WARNING)

    @property
    def passed(self) -> bool:
        return not self.errors

    def summary_line(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        return (
            f"[{verdict}] text {self.delivered_text}/{self.total_text} delivered "
            f"({self.verbatim_text} verbatim, {self.source_kept_text} source-kept, "
            f"{self.skipped_text} skipped, {self.pending_text} pending) | assets "
            f"{self.total_assets} ({self.reconstructed_assets} reconstructed, "
            f"{self.preserved_assets} preserved, {self.missing_assets} missing) | "
            f"{len(self.errors)} error(s), {len(self.warnings)} warning(s)"
        )


def reconcile(
    graph: ContentGraph,
    ledgers: tuple[ContentLedger, AssetLedger] | None = None,
    *,
    allow_spill_warning: bool = False,
) -> ReconciliationReport:
    """The delivery contract: reconcile the content graph's two ledgers.

    A text node is accounted for only when it is TRANSLATED or VERBATIM; a
    PENDING node was never decided and a SKIPPED node was dropped. An asset is
    accounted for when it is PRESERVED_OPAQUE or a *verified* RECONSTRUCTED; an
    unverified reconstruction is a warning, and MISSING is an error.

    When ``allow_spill_warning`` is True (under graceful degradation policies
    such as 'warn' or 'appendix'), space-overflow failures that keep the source
    are downgraded from ERROR to WARNING so delivery can proceed with full audit trail.
    """
    content, assets = ledgers if ledgers is not None else build_ledgers(graph)
    violations: list[Violation] = []

    delivered = verbatim = source_kept = skipped = pending = 0
    for entry in content.entries.values():
        if entry.disposition is TextDisposition.TRANSLATED:
            delivered += 1
        elif entry.disposition is TextDisposition.VERBATIM:
            verbatim += 1
        elif entry.disposition is TextDisposition.SOURCE_KEPT:
            source_kept += 1
            is_space = _is_space_failure(entry.reason)
            sev = Severity.WARNING if (allow_spill_warning or not is_space) else Severity.ERROR
            reason_str = entry.reason or "shipped source; translation not placed"
            if is_space and allow_spill_warning:
                reason_str = f"{reason_str} [spill_degraded_to_warning]"
            violations.append(
                Violation(
                    kind=ViolationKind.TEXT_SOURCE_KEPT,
                    severity=sev,
                    node_id=entry.node_id,
                    detail=reason_str,
                )
            )
        elif entry.disposition is TextDisposition.SKIPPED:
            skipped += 1
            violations.append(
                Violation(
                    kind=ViolationKind.TEXT_UNDELIVERED,
                    severity=Severity.ERROR,
                    node_id=entry.node_id,
                    detail=entry.reason or "dropped without a kept-verbatim reason",
                )
            )
        else:  # PENDING
            pending += 1
            violations.append(
                Violation(
                    kind=ViolationKind.TEXT_UNACCOUNTED,
                    severity=Severity.ERROR,
                    node_id=entry.node_id,
                    detail="no disposition recorded at delivery",
                )
            )

    reconstructed = preserved = dropped = missing = 0
    for asset_entry in assets.entries.values():
        if asset_entry.integrity is AssetIntegrity.MISSING:
            missing += 1
            violations.append(
                Violation(
                    kind=ViolationKind.ASSET_MISSING,
                    severity=Severity.ERROR,
                    node_id=asset_entry.node_id,
                    detail=asset_entry.reason or "asset not carried into the delivery",
                )
            )
        elif asset_entry.integrity is AssetIntegrity.DROPPED:
            dropped += 1
        elif asset_entry.integrity is AssetIntegrity.PRESERVED_OPAQUE:
            preserved += 1
        else:  # RECONSTRUCTED
            reconstructed += 1
            if asset_entry.corrupt:
                violations.append(
                    Violation(
                        kind=ViolationKind.ASSET_CORRUPT_RECONSTRUCTION,
                        severity=Severity.ERROR,
                        node_id=asset_entry.node_id,
                        detail=asset_entry.reason
                        or f"corrupt reconstruction as {asset_entry.representation.value}",
                    )
                )
            elif not asset_entry.verified:
                violations.append(
                    Violation(
                        kind=ViolationKind.ASSET_UNVERIFIED_RECONSTRUCTION,
                        severity=Severity.WARNING,
                        node_id=asset_entry.node_id,
                        detail=(
                            f"reconstructed as {asset_entry.representation.value} without "
                            "structural verification"
                        ),
                    )
                )

    return ReconciliationReport(
        doc_id=graph.doc_id,
        title=graph.title,
        total_text=content.total,
        delivered_text=delivered,
        verbatim_text=verbatim,
        source_kept_text=source_kept,
        skipped_text=skipped,
        pending_text=pending,
        total_assets=assets.total,
        reconstructed_assets=reconstructed,
        preserved_assets=preserved,
        dropped_assets=dropped,
        missing_assets=missing,
        violations=tuple(violations),
    )


__all__ = [
    "ReconciliationReport",
    "Severity",
    "Violation",
    "ViolationKind",
    "reconcile",
]
