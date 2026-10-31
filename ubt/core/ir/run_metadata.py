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
one home: a run decision never goes back to the dict. The dict is not closed to
other keys — an adapter may still record its own observation there (``page_count``
and ``page_range`` are written by the PDF path) — but every key this module knows
about has exactly one home.

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
#: Its reason for existing is the parity check in
#: ``tests/unit/test_option_surface_parity.py``: a key listed here whose writer
#: was deleted is an option that silently stopped being written, and this
#: frozenset is the only place that claims each key has a live writer. Removing
#: it removes the test's oracle, so it is read by a test on purpose.
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
        # Render telemetry, consumed by the quality report and run metrics.
        "typst_version",
    }
)


class RunMetadata(BaseModel):
    """What this run decided. Unset means the run never reached the decision."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    # Routing and cost policy.
    route_decision: dict[str, Any] | None = None
    route_mode: str | None = None
    adaptive_policy: dict[str, Any] | None = Field(
        default=None, description="resolve_adaptive_policy().to_dict()"
    )
    config_snapshot: dict[str, Any] | None = None

    # The render decision is not carried here: it lives in ``RunFacts.RenderPlan``
    # and the renderer's result in ``RenderOutcome``, both passed as arguments, so
    # nothing is posted onto the manifest between stages.

    # Extraction and formula policy.
    selected_pages: list[int] | None = None

    #: Explicit column override from ``--columns``/``UBT_LAYOUT_COLUMNS`` (1 or 2).
    #: ``None`` means "align to the source"; the renderer then reads the physical
    #: multi-column evidence in :attr:`route_decision`. A user override is a run
    #: decision, so it lives here — not in ``manifest.metadata`` (see that field's
    #: docstring).
    layout_columns: int | None = None

    # The delivery verdict is computed, never stored:
    # ``job_options.delivery_verdict_from_blocks`` derives it from the block
    # counters, and the CLI, REST and web console all call that one rule.

    def to_metadata_dict(self) -> dict[str, Any]:
        """Only the fields this run actually set, as JSON-native values.

        ``exclude_none`` (not ``exclude_unset``): ``validate_assignment`` marks a
        field "set" the moment anything assigns it — including assigning ``None``
        back — so ``exclude_unset`` leaked explicit ``null``s the contract says
        must stay absent. ``None`` is the "never decided" sentinel here; the one
        intentional empty value (``emit_secondary_mode``'s ``''``) is kept.
        """
        return self.model_dump(exclude_none=True, mode="json")
