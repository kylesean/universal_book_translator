from __future__ import annotations

import pytest

from ubt.core.engine.usage import JobBill, budget_violation
from ubt.core.router.pricing import register_custom_model_pricing

pytestmark = pytest.mark.fast


def test_an_unmeasured_bill_hard_stops_a_capped_job() -> None:
    usage = {"gpt-5": {"prompt_tokens": 100, "completion_tokens": 50, "unmeasured_calls": 1}}
    bill = JobBill(run_usage=usage, lifetime_usage=usage)
    assert bill.cost_usd is None
    assert bill.unmeasurable is True

    violation = budget_violation(bill, budget_usd=1.0, job_id="j1")
    assert violation is not None
    assert "unenforceable" in violation.lower()
    assert "j1" in violation


def test_an_unmeasured_bill_without_a_cap_is_left_alone() -> None:
    usage = {"gpt-5": {"prompt_tokens": 100, "unmeasured_calls": 1}}
    bill = JobBill(run_usage=usage, lifetime_usage=usage)
    assert budget_violation(bill, budget_usd=None, job_id="j1") is None


def test_a_measured_over_budget_bill_still_reports_the_cap() -> None:
    register_custom_model_pricing("priced-model", (1_000_000.0, 0.0))
    usage = {"priced-model": {"prompt_tokens": 10, "completion_tokens": 0}}
    bill = JobBill(run_usage=usage, lifetime_usage=usage)
    violation = budget_violation(bill, budget_usd=1.0, job_id="j1")
    assert violation is not None
    assert "Budget exceeded" in violation


def test_a_measured_under_budget_bill_passes() -> None:
    register_custom_model_pricing("cheap-model", (1.0, 1.0))
    usage = {"cheap-model": {"prompt_tokens": 100, "completion_tokens": 100}}
    bill = JobBill(run_usage=usage, lifetime_usage=usage)
    assert budget_violation(bill, budget_usd=1.0, job_id="j1") is None


def test_an_unpriced_model_hard_stops_a_capped_job() -> None:
    """A model with no price entry must not leave ``UBT_BUDGET_USD`` inert.

    ``cost_usd`` is None for an unpriced model (correctly: unknown beats wrong),
    and the cap used to read that None as "nothing to enforce" and return
    silently. A dot-version name such as ``gpt-5.1`` has no table entry, so an
    operator who set a budget got no bound at all while spend continued.
    """
    usage = {"gpt-5.1": {"prompt_tokens": 100, "completion_tokens": 50}}
    bill = JobBill(run_usage=usage, lifetime_usage=usage)
    assert bill.cost_usd is None
    assert bill.unmeasurable is False
    assert bill.unpriced == ("gpt-5.1",)

    violation = budget_violation(bill, budget_usd=1.0, job_id="j1")
    assert violation is not None
    assert "unenforceable" in violation.lower()
    assert "gpt-5.1" in violation
    assert "j1" in violation


def test_an_unpriced_model_without_a_cap_is_left_alone() -> None:
    usage = {"gpt-5.1": {"prompt_tokens": 100, "completion_tokens": 50}}
    bill = JobBill(run_usage=usage, lifetime_usage=usage)
    assert budget_violation(bill, budget_usd=None, job_id="j1") is None


def test_a_locally_served_model_is_not_unpriced() -> None:
    """Self-hosted models bill $0 by construction — not "unknown"."""
    usage = {"qwen3:8b": {"prompt_tokens": 100, "completion_tokens": 50}}
    bill = JobBill(run_usage=usage, lifetime_usage=usage, base_url="http://127.0.0.1:11434/v1")
    assert bill.unpriced == ()
    assert bill.cost_usd is not None
