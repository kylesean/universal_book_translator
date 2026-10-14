"""P2-7: a budget cap must not go inert when a provider omits usage.

``estimate_cost_usd`` returns None when a billed call carried no ``usage``
block — correctly, since a fake total would be wrong. But ``budget_violation``
treated None as "nothing to check" and returned None, so the cap silently
stopped firing while spend continued. Under a set budget an unmeasurable job is
now a hard stop: the operator asked for a bound and there is none.

Pinned here: an unmeasured bill hard-stops a capped job; the same bill with no
cap is left alone; and an ordinary measured over-budget bill still reports the
familiar message.
"""

from __future__ import annotations

import pytest

from ubt.core.engine.usage import JobBill, budget_violation
from ubt.core.router.pricing import register_custom_model_pricing

pytestmark = pytest.mark.fast


def test_an_unmeasured_bill_hard_stops_a_capped_job() -> None:
    """A job whose spend cannot be measured must stop under a cap."""
    usage = {"gpt-5": {"prompt_tokens": 100, "completion_tokens": 50, "unmeasured_calls": 1}}
    bill = JobBill(run_usage=usage, lifetime_usage=usage)
    assert bill.cost_usd is None
    assert bill.unmeasurable is True

    violation = budget_violation(bill, budget_usd=1.0, job_id="j1")
    assert violation is not None
    assert "unenforceable" in violation.lower()
    assert "j1" in violation


def test_an_unmeasured_bill_without_a_cap_is_left_alone() -> None:
    """No cap: an unmeasurable bill is not a violation (it is a warning elsewhere)."""
    usage = {"gpt-5": {"prompt_tokens": 100, "unmeasured_calls": 1}}
    bill = JobBill(run_usage=usage, lifetime_usage=usage)
    assert budget_violation(bill, budget_usd=None, job_id="j1") is None


def test_a_measured_over_budget_bill_still_reports_the_cap() -> None:
    """The ordinary case is unchanged."""
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
