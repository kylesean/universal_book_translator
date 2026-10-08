from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ubt.core.config import UBTConfig
from ubt.core.engine.pipeline import PipelineOrchestrator, _RunBillingSession

pytestmark = pytest.mark.fast


def test_billing_session_isolation() -> None:
    """Verify that multiple jobs on the same orchestrator maintain isolated billing state."""
    config = UBTConfig(draft_model="mock-draft")
    mock_router = MagicMock()
    mock_router.usage_totals_by_model.return_value = {
        "model-a": {"prompt_tokens": 100, "completion_tokens": 50}
    }
    mock_router.billing_endpoint_map.return_value = {}

    orchestrator = PipelineOrchestrator(config=config, router=mock_router)

    # Session for job 1
    session1 = _RunBillingSession(
        sink={"model-a": {"prompt_tokens": 10, "completion_tokens": 5}},
        baseline={},
        billed_usage={},
    )
    # Session for job 2
    session2 = _RunBillingSession(
        sink={"model-a": {"prompt_tokens": 40, "completion_tokens": 20}},
        baseline={},
        billed_usage={},
    )

    orchestrator._billing_sessions["job-1"] = session1
    orchestrator._billing_sessions["job-2"] = session2

    usage1 = orchestrator._session_usage(orchestrator._get_billing_session("job-1"))
    usage2 = orchestrator._session_usage(orchestrator._get_billing_session("job-2"))

    assert usage1["model-a"]["prompt_tokens"] == 10
    assert usage2["model-a"]["prompt_tokens"] == 40

    # Ensure modifying session 1 doesn't affect session 2
    session1.billed_usage = usage1
    assert orchestrator._get_billing_session("job-1").billed_usage["model-a"]["prompt_tokens"] == 10
    assert orchestrator._get_billing_session("job-2").billed_usage == {}
