"""Document facts -> DeliverySignals -> DeliveryPlan (Phase 0 wiring).

The router is deliberately pure; this module is the narrow adapter between the
PDF structure census and the router:

- ``auto`` / ``readable`` / ``faithful`` use the router;
- non-PDF targets can never be ``faithful`` (there is no source page to lock).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ubt.delivery.contracts import (
    DeliveryContract,
    DeliveryMode,
    DeliveryPlan,
    DeliverySignals,
    DocumentClass,
    ExpansionRisk,
    LayoutComplexity,
    PageKind,
)
from ubt.delivery.router import route_delivery

logger = logging.getLogger(__name__)

_TARGET_MODES: frozenset[str] = frozenset(mode.value for mode in DeliveryMode)

#: Provisional text trust for a clean native PDF with no confirmed font damage:
#: the extraction witness has not produced a calibrated score yet, but the page
#: is not known to be damaged. The measurement service replaces this with the
#: measured value.
_CLEAN_NATIVE_TEXT_TRUST = 0.95


@dataclass(frozen=True)
class DeliveryDecision:
    """The resolved plan plus the renderer mode it maps to.

    ``effective_mode=None`` means no contract is feasible; the caller must fail
    before rendering.
    """

    plan: DeliveryPlan
    effective_mode: str | None


def document_class_from_mode(document_mode: str) -> DocumentClass:
    """Map the existing paper/book/auto vocabulary to the routing matrix.

    Contract/form/design have no automatic signal in the current config; a user
    who needs those must request ``faithful`` explicitly. Guessing them from
    file content would be worse than the safe ``other -> readable`` default.
    """
    if document_mode == "paper":
        return DocumentClass.ACADEMIC
    if document_mode == "book":
        return DocumentClass.LONG_BOOK
    return DocumentClass.OTHER


def build_delivery_signals(
    *,
    document_mode: str,
    is_pdf: bool,
    has_scan: bool = False,
    font_damage_pages: int = 0,
    page_count: int = 0,
    multicolumn_page_share: float = 0.0,
    structural_page_share: float = 0.0,
    formula_heavy: bool = False,
    page_lock_feasible: bool | None = None,
    source_overlap: bool | None = None,
    expansion_risk: str | None = None,
) -> DeliverySignals:
    """Derive the router's inputs from the Phase 0 structure census.

    The census is a *prefilter*; the artifact-level probe
    (:func:`ubt.adapters.pdf.page_lock_measure.measure_page_lock`) supplies the
    measured facts through ``page_lock_feasible`` / ``source_overlap`` /
    ``expansion_risk``. When those are absent the clean-native defaults stand
    and ``page_lock_measured`` stays False, so the plan is honestly labelled
    provisional. A scanned or font-damaged PDF is vetoed here, because no
    later stage can make its text layer trustworthy without OCR/VLM.
    """
    document_class = document_class_from_mode(document_mode)
    if not is_pdf:
        return DeliverySignals(
            text_trust=1.0,
            page_kind=PageKind.NATIVE,
            layout_complexity=LayoutComplexity.LOW,
            source_overlap=False,
            expansion_risk=ExpansionRisk.LOW,
            page_lock_feasible=False,
            document_class=document_class,
        )

    if has_scan:
        text_trust = 0.0
    elif font_damage_pages > 0:
        text_trust = (
            max(0.0, 1.0 - float(font_damage_pages) / float(page_count)) if page_count > 0 else 0.0
        )
    else:
        text_trust = _CLEAN_NATIVE_TEXT_TRUST

    high_complexity = formula_heavy or structural_page_share >= 0.5 or multicolumn_page_share >= 0.5
    medium_complexity = formula_heavy or structural_page_share > 0.0 or multicolumn_page_share > 0.0
    measured = page_lock_feasible is not None
    clean_native = not has_scan and font_damage_pages == 0
    return DeliverySignals(
        text_trust=text_trust,
        page_kind=PageKind.SCANNED if has_scan else PageKind.NATIVE,
        layout_complexity=(
            LayoutComplexity.HIGH
            if high_complexity
            else (LayoutComplexity.MEDIUM if medium_complexity else LayoutComplexity.LOW)
        ),
        source_overlap=bool(source_overlap) if measured else False,
        expansion_risk=(
            ExpansionRisk(expansion_risk) if measured and expansion_risk else ExpansionRisk.LOW
        ),
        # A measurement replaces the prefilter's default; a scan/font-damaged
        # page is still vetoed outright, because the probe cannot make its text
        # layer trustworthy.
        page_lock_feasible=(bool(page_lock_feasible) and clean_native)
        if measured
        else clean_native,
        page_lock_measured=measured,
        document_class=document_class,
    )


def plan_for_document(
    requested: str,
    *,
    document_mode: str,
    is_pdf: bool,
    has_scan: bool = False,
    font_damage_pages: int = 0,
    page_count: int = 0,
    multicolumn_page_share: float = 0.0,
    structural_page_share: float = 0.0,
    formula_heavy: bool = False,
    allow_degrade: bool = False,
    page_lock_feasible: bool | None = None,
    source_overlap: bool | None = None,
    expansion_risk: str | None = None,
) -> DeliveryDecision:
    """Resolve the requested mode into a plan and an effective renderer mode.

    Raises ``ValueError`` for a mode outside the vocabulary. An infeasible
    ``faithful`` request returns ``effective_mode=None``; the pipeline raises
    before spending a model call.
    """
    normalized = requested.strip().lower()
    if normalized not in _TARGET_MODES:
        allowed = ", ".join(sorted(_TARGET_MODES))
        raise ValueError(f"unknown delivery mode {requested!r}; expected one of: {allowed}")

    signals = build_delivery_signals(
        document_mode=document_mode,
        is_pdf=is_pdf,
        has_scan=has_scan,
        font_damage_pages=font_damage_pages,
        page_count=page_count,
        multicolumn_page_share=multicolumn_page_share,
        structural_page_share=structural_page_share,
        formula_heavy=formula_heavy,
        page_lock_feasible=page_lock_feasible,
        source_overlap=source_overlap,
        expansion_risk=expansion_risk,
    )
    requested_mode = DeliveryMode(normalized)
    plan = route_delivery(signals, requested=requested_mode, allow_degrade=allow_degrade)
    if plan.contract is None:
        return DeliveryDecision(plan=plan, effective_mode=None)
    if plan.contract is DeliveryContract.FAITHFUL:
        return DeliveryDecision(
            plan=_mark_faithful_provisional(plan) if not plan.signals.page_lock_measured else plan,
            effective_mode=DeliveryContract.FAITHFUL.value,
        )
    return DeliveryDecision(plan=plan, effective_mode=DeliveryContract.READABLE.value)


def _mark_faithful_provisional(plan: DeliveryPlan) -> DeliveryPlan:
    """Record that the feasibility signal is a census, not an artifact probe.

    Applied only when the native preflight did not run (or could not measure):
    ``page_lock_measured`` is the difference between "we measured the source
    pages" and "the structure census found nothing wrong", and the plan must
    not blur the two.
    """
    return plan.model_copy(
        update={
            "eligibility": plan.eligibility.model_copy(update={"provisional": True}),
            "reasons": (
                *plan.reasons,
                "page_lock_feasible is provisional (structure census only; "
                "no source-page feasibility probe measured this document)",
            ),
        }
    )


#: Artifact contracts each PDF engine can honour. ``native`` patches the source
#: content stream in place, so it satisfies both promises: ``readable`` only
#: requires delivered, selectable text, while ``faithful`` additionally holds
#: the page grid. The Typst reflow engines rebuild the document and therefore
#: cannot promise page identity.
_ENGINE_CONTRACTS: dict[str, frozenset[DeliveryContract]] = {
    "native": frozenset({DeliveryContract.READABLE, DeliveryContract.FAITHFUL}),
    "docling": frozenset({DeliveryContract.READABLE}),
    "pdfium": frozenset({DeliveryContract.READABLE}),
}


def resolve_pdf_engine(
    configured: str,
    *,
    is_pdf: bool,
    contract: DeliveryContract | None,
) -> str:
    """The engine that can honour ``contract`` for an ``auto`` request.

    The delivery plan owns the promise; the engine follows it. ``auto`` used to
    resolve to ``native`` for every born-digital PDF in the factory, which made
    the plan's ``readable`` contract and the artifact's in-place delivery
    disagree (and the artifact audit then had to guess). Resolving here keeps
    one source of truth:

    - ``faithful`` -> ``native`` (1:1 in-place content-stream patch);
    - ``readable`` (or no plan yet) -> ``docling`` (Typst reflow / OCR path);
    - non-PDF or an explicit engine -> unchanged.
    """
    engine = (configured or "auto").strip().lower()
    if not is_pdf or engine != "auto":
        return engine
    if contract is DeliveryContract.READABLE:
        return "docling"
    return "native"


def engine_supports_contract(engine: str, contract: DeliveryContract | None) -> bool:
    """Whether ``engine`` can keep the artifact promise in ``contract``.

    An unknown engine is not vetoed here: ``custom_adapter`` and downstream
    extensions own their own capabilities. The three built-in PDF engines are
    pinned so an explicit ``--pdf-engine docling --delivery faithful`` fails
    before a single model call instead of silently rendering a different
    contract.
    """
    if contract is None:
        return True
    supported = _ENGINE_CONTRACTS.get((engine or "").strip().lower())
    if supported is None:
        return True
    return contract in supported
