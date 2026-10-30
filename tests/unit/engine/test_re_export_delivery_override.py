"""Re-export delivery override: one ledger, two contracts, zero tokens.

G1 needs to compare the readable path against the faithful one on the *same*
translations. Re-running the whole job in readable mode would pay for a second
translation just to change the renderer, so the override re-targets the export
stage only. These tests pin the metadata it writes, because the renderer and the
artifact contract both read those keys and drifting one of them would silently
render one contract while auditing the other.
"""

from __future__ import annotations

import pytest

from ubt.core.engine.re_export import ReExportError, _apply_delivery_override
from ubt.core.ir.models import BookManifest

pytestmark = pytest.mark.fast


def _manifest(source: str = "/tmp/book.pdf") -> BookManifest:
    return BookManifest(
        doc_id="job-1",
        title="t",
        source_path=source,
        metadata={
            "document_mode": "paper",
            "page_count": 26,
            "delivery_mode": "page_locked",
            "delivery_plan": {"contract": "faithful"},
        },
    )


def test_no_override_leaves_the_recorded_plan_untouched() -> None:
    manifest = _manifest()
    before = dict(manifest.metadata)

    assert _apply_delivery_override(manifest, None) is None
    assert manifest.metadata == before


def test_readable_override_rewrites_the_mode_and_the_plan() -> None:
    manifest = _manifest()

    effective = _apply_delivery_override(manifest, "readable")

    assert effective == "reflow"
    assert manifest.metadata["delivery_mode"] == "reflow"
    assert manifest.metadata["delivery_plan"]["contract"] == "readable"
    assert manifest.metadata["delivery_plan"]["requested_mode"] == "readable"
    assert manifest.metadata["delivery_mode_requested"] == "readable"
    assert manifest.metadata["delivery_mode_effective"] == "reflow"


def test_faithful_override_on_a_pdf_keeps_the_page_lock_contract() -> None:
    manifest = _manifest()

    effective = _apply_delivery_override(manifest, "faithful")

    assert effective == "facsimile", "the renderer's faithful mode for a paper PDF"
    assert manifest.metadata["delivery_plan"]["contract"] == "faithful"
    assert manifest.metadata["delivery_mode_effective"] == "facsimile"


def test_an_infeasible_request_fails_instead_of_rendering_the_other_contract() -> None:
    manifest = _manifest(source="/tmp/book.md")

    with pytest.raises(ReExportError, match="not feasible"):
        _apply_delivery_override(manifest, "faithful")

    # A refusal must not leave a half-written plan behind.
    assert manifest.metadata["delivery_plan"] == {"contract": "faithful"}


def test_faithful_override_refuses_font_damaged_pdf_unless_allow_degrade_is_set() -> None:
    manifest = _manifest()
    manifest.metadata["confirmed_font_damage_pages"] = [2, 8]

    with pytest.raises(ReExportError, match="not feasible"):
        _apply_delivery_override(manifest, "faithful", allow_degrade=False)

    effective = _apply_delivery_override(manifest, "faithful", allow_degrade=True)
    assert effective == "reflow"
    assert manifest.metadata["delivery_plan"]["contract"] == "readable"
    assert manifest.metadata["delivery_plan"]["eligibility"]["eligible"] is False
