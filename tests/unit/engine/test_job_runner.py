from __future__ import annotations

import pytest

from ubt.core.config import UBTConfig
from ubt.core.engine.dry_run import DryRunModelProvider
from ubt.core.engine.events import TranslationProgressEvent
from ubt.core.engine.job_runner import build_job_orchestrator
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
    assert orchestrator.config.ocr_mode == "off"
    assert orchestrator.config.allow_page_upload is False
    assert orchestrator.config.visual_judge_enabled is False
    assert config.ocr_mode != "off"


def test_the_finalize_hook_reaches_both_wirings() -> None:
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
