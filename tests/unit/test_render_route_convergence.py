"""One render-route decision: the quoted route must be the route that runs.

The rigid/reflow choice used to be re-derived in three places — the runtime
dispatcher, the assessor and the advisor — with different criteria. The
assessor's recommendation becomes an explicit ``--render-engine`` flag
(``cli/commands/assess.py``), so a divergence meant the user was quoted one
route and the pipeline ran another. These tests pin the convergence onto
``adaptive_policy.resolve_render_engine_from_signals`` and the advisory that
depends on the *active* engine.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ubt.adapters.base import BaseDocumentAdapter
from ubt.core.archetype import DocCategory, MathDensity
from ubt.core.assess import _recommend_route
from ubt.core.config import RenderEngine, UBTConfig
from ubt.core.engine.pipeline import PipelineOrchestrator
from ubt.core.ir.models import BookManifest
from ubt.core.policy.adaptive_policy import resolve_render_engine_from_signals
from ubt.core.router.provider import MockModelProvider
from ubt.core.router.router import ModelRouter
from ubt.core.router_mode import RouteDecision

pytestmark = pytest.mark.fast


def _arch(**overrides: Any) -> Any:
    base: dict[str, Any] = {
        "format_ext": "pdf",
        "is_scanned": False,
        "math_density": MathDensity.NONE,
        "category": DocCategory.LITERATURE,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


# ---------------------------------------------------------------------------
# Canonical signal resolver
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("requested", "has_math", "struct_share", "has_geometry", "expected"),
    [
        ("auto", False, 0.0, True, "publication"),
        ("auto", True, 0.0, True, "rigid"),
        ("auto", False, 0.25, True, "rigid"),
        ("auto", False, 0.0, False, "publication"),  # geometry guard
        ("rigid", False, 0.0, False, "rigid"),  # explicit intent honored
        ("reflow", True, 0.5, True, "publication"),
        ("publication", False, 0.0, True, "publication"),
        ("quantum-typesetting", False, 0.0, True, "publication"),  # unknown -> fallback
        ("inplace", False, 0.0, True, "rigid"),  # retired alias folds to rigid
        ("hybrid", False, 0.0, True, "publication"),  # retired alias folds to auto
        ("hybrid", True, 0.0, True, "rigid"),  # ...auto + math -> rigid
    ],
)
def test_signal_resolver_semantics(
    requested: str, has_math: bool, struct_share: float, has_geometry: bool, expected: str
) -> None:
    assert (
        resolve_render_engine_from_signals(
            requested, has_math=has_math, struct_share=struct_share, has_geometry=has_geometry
        )
        == expected
    )


def test_canonical_render_engine_folds_retired_aliases() -> None:
    """The fold lives in the function, not only in the field validator.

    A legacy ``inplace`` reaching the resolver directly used to be treated as
    unknown and routed to publication — a silent rigid->reflow downgrade.
    """
    from ubt.core.config import canonical_render_engine

    assert canonical_render_engine("inplace") == "rigid"
    assert canonical_render_engine("hybrid") == "auto"
    assert canonical_render_engine("reflow") == "publication"


# ---------------------------------------------------------------------------
# Assessor recommendation follows the runtime dispatch
# ---------------------------------------------------------------------------


def test_academic_pdf_without_math_or_figures_reflows() -> None:
    """An academic classification is not a routing signal on its own.

    The runtime dispatcher sends prose-with-no-math to reflow; the assessor used
    to override that to rigid purely because the document was academic, so the
    quoted ``--render-engine rigid`` contradicted what ``auto`` would run.
    """
    rec = _recommend_route(
        _arch(category=DocCategory.ACADEMIC_PAPER),
        None,
        {"has_formulas": False, "has_vector_diagrams": False, "scan_page_share": 0.0},
        UBTConfig(),
    )
    assert rec.recommended_render_engine == "reflow"


def test_formula_dense_pdf_still_recommends_rigid() -> None:
    rec = _recommend_route(
        _arch(category=DocCategory.ACADEMIC_PAPER),
        None,
        {"has_formulas": True, "has_vector_diagrams": False, "scan_page_share": 0.0},
        UBTConfig(),
    )
    assert rec.recommended_render_engine == "rigid"


def test_scanned_pdf_still_recommends_rigid() -> None:
    rec = _recommend_route(
        _arch(is_scanned=True),
        None,
        {"has_formulas": False, "has_vector_diagrams": False, "scan_page_share": 0.8},
        UBTConfig(),
    )
    assert rec.recommended_render_engine == "rigid"


# ---------------------------------------------------------------------------
# The advisor follows the same resolver
# ---------------------------------------------------------------------------


def test_advisor_academic_pdf_without_math_reflows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ubt.core.advisor import DocumentAdvisor

    doc = tmp_path / "paper.pdf"
    doc.write_bytes(b"%PDF-1.4 fake")
    monkeypatch.setattr(
        "ubt.core.advisor.analyze_archetype",
        lambda _p: _arch(
            page_or_ch_count=30,
            detected_domain="general",
            domain_confidence=0.1,
            category=DocCategory.ACADEMIC_PAPER,
        ),
    )
    monkeypatch.setattr(
        "ubt.core.advisor.decide_route",
        lambda *a, **k: RouteDecision(
            mode="long", pages=30, chars=90_000, has_scan=False, formula_heavy=False, reason="long"
        ),
    )
    report = DocumentAdvisor.analyze(doc)
    assert report.recommended_render_engine == "reflow"


# ---------------------------------------------------------------------------
# Pipeline advisory uses the ACTIVE engine, not the config value
# ---------------------------------------------------------------------------


async def _run_pipeline_advisory(tmp_path: Path, *, render_engine: RenderEngine) -> BookManifest:
    input_pdf = tmp_path / "paper.pdf"
    input_pdf.write_bytes(b"%PDF-1.4 mock paper")
    output_pdf = tmp_path / "out.pdf"

    config = UBTConfig(
        render_engine=render_engine, exec_mode="short", db_dir=tmp_path / "db", tm_enabled=False
    )
    orchestrator = PipelineOrchestrator(
        config=config, router=ModelRouter(provider=MockModelProvider(), draft_model="mock")
    )
    manifest = BookManifest(
        doc_id="doc_paper",
        title="学术论文",
        source_path=str(input_pdf),
        metadata={"total_pages": 5},
    )

    async def _empty(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        if False:
            yield None

    with (
        patch("ubt.core.engine.pipeline.resolve_adapter") as mock_resolve,
        patch("ubt.core.engine.pipeline.decide") as mock_decide,
        patch("ubt.core.engine.pipeline.run_ingest_stage", side_effect=_empty),
        patch("ubt.core.engine.pipeline.run_bible_stage", side_effect=_empty),
        patch("ubt.core.engine.pipeline.run_draft_stage", side_effect=_empty),
        patch("ubt.core.engine.pipeline.run_quality_gate_stage", side_effect=_empty),
        patch("ubt.core.engine.pipeline.run_repair_stage", side_effect=_empty),
        patch("ubt.core.engine.pipeline.run_triage_stage", side_effect=_empty),
        patch("ubt.core.engine.pipeline.run_export_stage", side_effect=_empty),
    ):
        mock_adapter = MagicMock(spec=BaseDocumentAdapter)
        mock_adapter.extract_manifest = AsyncMock(return_value=manifest)
        mock_resolve.return_value = mock_adapter
        mock_decide.return_value = RouteDecision(
            mode="short",
            pages=5,
            chars=3000,
            has_scan=False,
            formula_heavy=True,
            reason="short paper with heavy math",
        )
        async for _ in orchestrator.run(input_path=input_pdf, output_path=output_pdf):
            pass
    return manifest


@pytest.mark.asyncio
async def test_auto_advisory_surfaces_on_auto_routed_formula_heavy(tmp_path: Path) -> None:
    """``auto`` sends a formula-dense document to rigid — the tradeoff must show.

    The advisory was gated on ``adaptive_policy.render_engine == 'rigid'``, which
    is ``'auto'`` at that point, so an auto-routed formula-dense run silently
    omitted the layout-tradeoff advisory from its report.
    """
    manifest = await _run_pipeline_advisory(tmp_path, render_engine="auto")
    assert manifest.run.delivery_status is not None
    assert manifest.run.delivery_status.startswith("LAYOUT_TRADEOFF_ADVISORY")


@pytest.mark.asyncio
async def test_explicit_reflow_advisory_not_surfaced(tmp_path: Path) -> None:
    """A forced reflow is not the rigid tradeoff, so no advisory is emitted."""
    manifest = await _run_pipeline_advisory(tmp_path, render_engine="reflow")
    assert manifest.run.delivery_status != "LAYOUT_TRADEOFF_ADVISORY"
    assert not (manifest.run.delivery_status or "").startswith("LAYOUT_TRADEOFF_ADVISORY")
