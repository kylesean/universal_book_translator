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
:meth:`RunMetadata.to_metadata_dict`, which drops unset fields: a key the run
never set stays absent instead of appearing as an explicit ``null``, so the
persisted job JSON carries only what the run decided.
"""

from __future__ import annotations

from typing import Any, Final

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
        # Render telemetry, consumed by the quality report and the visual gate.
        "render_engine_effective",
        "render_padding_pages",
        "typst_version",
    }
)


class RunMetadata(BaseModel):
    """What this run decided. Unset means the run never reached the decision."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    # --- Routing and cost policy ------------------------------------------
    route_decision: dict[str, Any] | None = None
    route_mode: str | None = None
    adaptive_policy: dict[str, Any] | None = Field(
        default=None, description="resolve_adaptive_policy().to_dict()"
    )
    config_snapshot: dict[str, Any] | None = None

    # --- Bilingual / render mode ------------------------------------------
    # The whole render decision (mode, engine, chrome/cover, secondary request)
    # lives in ``RunFacts.RenderPlan`` and the renderer's result in
    # ``RenderOutcome`` (compiler render plan protocol). Nothing is posted here:
    # this was the last inter-stage bus on the manifest.

    # --- Extraction / formula policy --------------------------------------
    formula_mode: str | None = None
    selected_pages: list[int] | None = None

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
