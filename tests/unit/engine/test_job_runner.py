from __future__ import annotations

import pytest

from ubt.app.job_service import JobService
from ubt.core.config import UBTConfig
from ubt.core.engine.dry_run import DryRunModelProvider
from ubt.core.engine.events import TranslationProgressEvent
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.qe.heuristic import MockQERunner

pytestmark = pytest.mark.fast


def test_a_real_run_builds_the_shared_orchestrator() -> None:
    config = UBTConfig(draft_model="mock-draft")
    orchestrator = JobService(config=config).orchestrator(rehearse=False)
    assert orchestrator.config is config
    assert not isinstance(orchestrator.router.provider, DryRunModelProvider)


def test_a_rehearsal_builds_the_zero_spend_wiring() -> None:
    config = UBTConfig(draft_model="mock-draft")
    orchestrator = JobService(config=config).orchestrator(rehearse=True)
    assert isinstance(orchestrator.router.provider, DryRunModelProvider)
    assert isinstance(orchestrator.qe_runner, MockQERunner)
    assert orchestrator.config.ocr_mode == "off"
    assert orchestrator.config.allow_page_upload is False
    assert config.ocr_mode != "off"


def test_the_finalize_hook_reaches_both_wirings() -> None:
    seen: list[str] = []

    def hook(event: TranslationProgressEvent, ledger: SQLiteJobLedger) -> None:
        seen.append("called")

    config = UBTConfig(draft_model="mock-draft")
    for rehearse in (False, True):
        orchestrator = JobService(config=config, finalize_job=hook).orchestrator(rehearse=rehearse)
        assert orchestrator.finalize_job is hook
