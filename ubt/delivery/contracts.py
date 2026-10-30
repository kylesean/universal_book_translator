"""Delivery-contract vocabulary for the adaptive delivery program.

Target design: ``docs/design/adaptive_delivery_prd.md`` §5 and §7.7.
This module is the typed boundary only: it does not render, measure, or route.
Keeping the vocabulary here lets the CLI, API, router, render strategy and QA
share one spelling of ``readable`` / ``faithful`` / ``auto`` instead of
re-inventing the mapping at each call site.

The artifact-side verdict already exists in
``ubt.core.engine.delivery_contract`` (``ContractName``). That module owns the
eight invariants; this module owns the routing/request vocabulary. A parity
test pins the two value sets together so a rename cannot split them.

Compatibility note: the current public config still speaks
``reflow`` / ``facsimile`` / ``page_locked`` (``ubt.core.config.DeliveryMode``).
:func:`legacy_delivery_contract` is the one-way migration map the Phase 0
adapter will call; it does not change stored jobs or the existing default.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class DeliveryMode(StrEnum):
    """What the user asked for, before feasibility is evaluated."""

    AUTO = "auto"
    READABLE = "readable"
    FAITHFUL = "faithful"


class DeliveryContract(StrEnum):
    """What the delivered artifact promises."""

    READABLE = "readable"
    FAITHFUL = "faithful"


class PageStrategy(StrEnum):
    """Internal per-page strategy; not a user-visible contract."""

    CANVAS = "canvas"
    PAGE_REFLOW = "page_reflow"
    DOCUMENT_REFLOW = "document_reflow"
    OCR_REFLOW = "ocr_reflow"


class PageKind(StrEnum):
    NATIVE = "native"
    SCANNED = "scanned"
    MIXED = "mixed"


class LayoutComplexity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ExpansionRisk(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class DocumentClass(StrEnum):
    """The routing matrix's document category, not a format."""

    ACADEMIC = "academic"
    TECHNICAL = "technical"
    CONTRACT = "contract"
    FORM = "form"
    DESIGN = "design"
    LONG_BOOK = "long_book"
    MOBILE = "mobile"
    OTHER = "other"


class DegradePolicy(StrEnum):
    FAIL = "fail"
    ALLOW_DEGRADE = "allow_degrade"


class DeliverySignals(BaseModel):
    """The measurable inputs the router is allowed to read.

    Deliberately small: every field must have an evidence source in the
    ingestion/analysis stages. A signal that cannot be measured is a reason to
    fail ``faithful`` eligibility, not to guess.
    """

    model_config = ConfigDict(frozen=True)

    text_trust: float = Field(ge=0.0, le=1.0)
    page_kind: PageKind = PageKind.NATIVE
    layout_complexity: LayoutComplexity = LayoutComplexity.MEDIUM
    source_overlap: bool = False
    expansion_risk: ExpansionRisk = ExpansionRisk.LOW
    page_lock_feasible: bool = True
    document_class: DocumentClass = DocumentClass.OTHER


class FaithfulEligibility(BaseModel):
    """Whether the source is allowed to make a page-lock promise."""

    model_config = ConfigDict(frozen=True)

    eligible: bool
    #: True when the decision came from a prefilter (structure census/legacy
    #: compatibility), not from the G2 per-page MeasurementService solve.
    provisional: bool = False
    #: Machine-readable enough for the report and the CLI error: each reason
    #: names the signal that failed (``text_trust``, ``page_kind``, ...).
    reasons: tuple[str, ...] = ()


class DeliveryPlan(BaseModel):
    """The router's single output, written to the manifest and read by QA.

    ``feasible=False`` means the requested contract must not be attempted. The
    caller either fails the export (default) or re-routes with
    ``allow_degrade``; the plan itself never silently changes the contract.
    """

    model_config = ConfigDict(frozen=True)

    schema_version: int = 1
    requested_mode: DeliveryMode
    contract: DeliveryContract | None
    feasible: bool
    eligibility: FaithfulEligibility
    default_page_strategy: PageStrategy | None = None
    #: Populated when a page-level strategy has been chosen. Empty on the
    #: document-level routing pass; the render strategy fills it in.
    page_strategies: dict[int, PageStrategy] = Field(default_factory=dict)
    degrade_policy: DegradePolicy = DegradePolicy.FAIL
    #: Whether a ``readable`` reflow companion copy should be emitted alongside
    #: a ``faithful`` artifact (PRD §5.2).
    companion_reflow: bool = False
    reasons: tuple[str, ...] = ()
    signals: DeliverySignals

    def to_manifest(self) -> dict[str, object]:
        """Serialize the same shape the quality report will carry."""
        return self.model_dump(mode="json")


_LEGACY_CONTRACTS: dict[str, DeliveryContract] = {
    "reflow": DeliveryContract.READABLE,
    "facsimile": DeliveryContract.FAITHFUL,
    "page_locked": DeliveryContract.FAITHFUL,
}


def legacy_delivery_contract(value: str) -> DeliveryContract:
    """Map a current public delivery mode to the target contract.

    Unknown values are refused rather than defaulted: a stored job resuming
    under a name the new router does not understand must fail loudly, not
    silently become a different contract.
    """
    try:
        return _LEGACY_CONTRACTS[value]
    except KeyError as exc:
        allowed = ", ".join(sorted(_LEGACY_CONTRACTS))
        raise ValueError(
            f"unknown legacy delivery mode {value!r}; expected one of: {allowed}"
        ) from exc
