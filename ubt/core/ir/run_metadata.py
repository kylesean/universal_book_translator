"""Typed contract for what a run decides about itself.

``BookManifest.metadata`` is a ``dict[str, Any]`` that 36 different string keys
flowed through: the pipeline wrote run decisions into it, the PDF renderer wrote
artifact telemetry into it, the export stage wrote both kinds back, and the
quality report read 10 of them out by name. Nothing declared which key held
what type, who owned it, or whether an absent key meant "not decided" or
"decided to nothing" — so a renamed key silently degraded into a report default
rather than failing.

This module is the decided half of that split: every key that records a *choice
the pipeline made during this run* lives here as a typed field. Keys that record
what an adapter observed about the source document or the rendered artifact
(``author``, ``opf_path``, ``page_kinds``, ``typst_version``, …) stay in
``manifest.metadata`` and are listed in :data:`ARTIFACT_METADATA_KEYS`. One key,
one home: a run decision never goes back to the dict, and the dict never carries
a key that is not declared there.

Field defaults are ``None`` and serialization goes through
:meth:`RunMetadata.to_metadata_dict`, which drops unset fields. That keeps the
persisted job JSON byte-identical to what the old dict wrote: a key the run
never set stays absent instead of appearing as an explicit ``null``.
"""

from __future__ import annotations

from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

#: Keys that stay in ``manifest.metadata`` because they describe the source
#: document or the rendered artifact rather than a decision made this run.
ARTIFACT_METADATA_KEYS: Final[frozenset[str]] = frozenset(
    {
        # Source document, read by the adapters and the bible stage.
        "author",
        "chapter_translations",
        "opf_path",
        "pdf_parser_engine",
        # Extraction telemetry, written by the PDF/EPUB adapters.
        "boilerplate_footers",
        "is_page_slice_epub",
        "page_kinds",
        # Render telemetry, consumed by the quality report.
        "formula_witness_findings",
        "typst_syntax_fallbacks",
        "typst_version",
    }
)


class RunMetadata(BaseModel):
    """What this run decided. Unset means the run never reached the decision."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    # --- Routing and cost policy ------------------------------------------
    route_decision: dict[str, Any] | None = None
    route_mode: str | None = None
    fast_lane: bool | None = None
    adaptive_policy: dict[str, Any] | None = Field(
        default=None, description="resolve_adaptive_policy().to_dict()"
    )
    config_snapshot: dict[str, Any] | None = None

    # --- Bilingual / render mode ------------------------------------------
    bilingual_mode: str | None = Field(default=None, description="RENDER_MODE_VALUE key")
    effective_dual_mode: str | None = None
    dual_mode_downgraded: str | None = Field(
        default=None, description="the mode before the rigid engine replaced it"
    )
    dual_enforcement: Literal["advise", "auto"] | None = None
    bilingual_advisory: dict[str, Any] | None = None
    facing_spread: bool | None = None
    render_padding_pages: list[int] | None = Field(
        default=None,
        description=(
            "1-based pages the bilingual alternator filled with intentional blanks "
            "(facing flyleaf + page-count padding); the visual gate exempts them "
            "from blank-page findings"
        ),
    )
    emit_secondary_mode: str | None = Field(
        default=None, description="--emit-both second mode, '' when not requested"
    )
    emit_secondary_engine: str | None = Field(
        default=None,
        description="secondary PDF render engine (e.g. 'rigid') when reflow is forced on a dense layout",
    )
    companion_output_path: str | None = Field(
        default=None,
        description="path to the rendered secondary/companion PDF artifact when dual delivery triggers",
    )
    render_engine: str | None = None
    render_engine_effective: str | None = Field(
        default=None, description="engine the PDF renderer actually used"
    )
    render_engine_advisory: str | None = None
    translate_chrome: bool | None = None
    cover_mode: str | None = None

    # --- Extraction / formula policy --------------------------------------
    formula_mode: str | None = None
    selected_pages: list[int] | None = None
    extraction_witness: dict[str, Any] | None = Field(
        default=None, description="font-encoding damage summary, zero tokens"
    )

    # --- OCR egress disclosure --------------------------------------------
    ocr_mode: str | None = None
    ocr_endpoint: str | None = None

    # --- Terminology and export-time policy -------------------------------
    terminology_table: dict[str, str] | None = Field(
        default=None, description="the capped sheet the draft prompts carried"
    )
    length_policy: dict[str, Any] | None = None

    # --- Delivery verdict -------------------------------------------------
    delivery_status: str | None = None
    delivery_warning: str | None = None

    def to_metadata_dict(self) -> dict[str, Any]:
        """Only the fields this run actually set, as JSON-native values.

        ``exclude_none`` (not ``exclude_unset``): ``validate_assignment`` marks a
        field "set" the moment anything assigns it — including assigning ``None``
        back — so ``exclude_unset`` leaked explicit ``null``s the contract says
        must stay absent. ``None`` is the "never decided" sentinel here; the one
        intentional empty value (``emit_secondary_mode``'s ``''``) is kept.
        """
        return self.model_dump(exclude_none=True, mode="json")
