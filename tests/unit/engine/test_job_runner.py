"""Both job executors build their orchestrator through one factory (W3-10).

The embedded API manager and the queue worker are two deployment modes, each
driving exactly one executor — not two implementations of one job. They did
share a twelve-line orchestrator-construction block, and that block has drifted
before (the manager's hand-written copy silently dropped the quality preset).
``build_job_orchestrator`` is that block once; the *decision* to rehearse stays
with each executor, because the API honours its own process's ``dry_run`` while
the worker recomputes from its own key.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ubt.core.config import UBTConfig
from ubt.core.engine.dry_run import DryRunModelProvider
from ubt.core.engine.events import TranslationProgressEvent
from ubt.core.engine.job_runner import build_job_orchestrator, resolve_job_paths
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.qe.heuristic import MockQERunner

pytestmark = pytest.mark.fast


def test_a_real_run_builds_the_shared_orchestrator() -> None:
    config = UBTConfig(draft_model="mock-draft")
    orchestrator = build_job_orchestrator(
        config,
        router=None,
        qe_runner=None,
        rate_limiter=None,
        finalize_job=None,
        rehearse=False,
    )
    assert orchestrator.config is config
    # Not a rehearsal: the real provider path (mock-draft has no key, so the
    # orchestrator builds its own mock provider rather than the echo one).
    assert not isinstance(orchestrator.router.provider, DryRunModelProvider)


def test_a_rehearsal_builds_the_zero_spend_wiring() -> None:
    config = UBTConfig(draft_model="mock-draft")
    orchestrator = build_job_orchestrator(
        config,
        router=None,
        qe_runner=None,
        rate_limiter=None,
        finalize_job=None,
        rehearse=True,
    )
    assert isinstance(orchestrator.router.provider, DryRunModelProvider)
    assert isinstance(orchestrator.qe_runner, MockQERunner)
    # Every page-image egress path is closed on the rehearsal's config copy.
    assert orchestrator.config.ocr_mode == "off"
    assert orchestrator.config.allow_page_upload is False
    assert orchestrator.config.visual_judge_enabled is False
    # The caller's config is untouched (the rehearsal works on a copy).
    assert config.ocr_mode != "off"


def test_the_finalize_hook_reaches_both_wirings() -> None:
    """The completion hook is the persistence seam; it must ride both paths."""
    seen: list[str] = []

    def hook(event: TranslationProgressEvent, ledger: SQLiteJobLedger) -> None:
        seen.append("called")

    config = UBTConfig(draft_model="mock-draft")
    for rehearse in (False, True):
        orchestrator = build_job_orchestrator(
            config,
            router=None,
            qe_runner=None,
            rate_limiter=None,
            finalize_job=hook,
            rehearse=rehearse,
        )
        assert orchestrator.finalize_job is hook


def test_resolve_job_paths_reads_both_shapes(tmp_path: Path) -> None:
    input_file = tmp_path / "in.pdf"
    out_file = tmp_path / "out.pdf"
    assert resolve_job_paths({"input_path": str(input_file)}) == (input_file, None)
    assert resolve_job_paths({"input_path": str(input_file), "output_path": str(out_file)}) == (
        input_file,
        out_file,
    )
