"""What a job has spent, and the cap that stops it.

Every progress event folds the run's token usage into the job's bill before it
reports, so a run killed mid-book still leaves its spend on disk. The arithmetic
lives beside the bill, decoupled from event generation.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass

from ubt.core.engine.ledger import SQLiteJobLedger, merge_usage_totals
from ubt.core.router.pricing import cache_hit_rate_from_usage, estimate_cost_usd

UsageTotals = dict[str, dict[str, int]]


@dataclass(frozen=True)
class JobBill:
    """This run's spend, and the job's total across every resume."""

    run_usage: UsageTotals
    lifetime_usage: UsageTotals
    #: Endpoint the run's models were reached through, plus per-model overrides
    #: (cloud OCR bills its own). Both empty for callers that do not know, in
    #: which case cost falls back to price-table-only accounting.
    base_url: str = ""
    endpoint_map: Mapping[str, str] | None = None

    @property
    def cost_usd(self) -> float | None:
        """Job total, or None while nothing is measured — never a fake $0.00."""
        if not self.lifetime_usage:
            return None
        return estimate_cost_usd(
            self.lifetime_usage, base_url=self.base_url, endpoint_map=self.endpoint_map
        )

    @property
    def run_cost_usd(self) -> float | None:
        if not self.run_usage:
            return None
        return estimate_cost_usd(
            self.run_usage, base_url=self.base_url, endpoint_map=self.endpoint_map
        )

    @property
    def cache_hit_rate(self) -> float:
        return cache_hit_rate_from_usage(self.lifetime_usage)


async def bill_job_run(
    ledger: SQLiteJobLedger,
    job_id: str,
    run_usage: UsageTotals,
    *,
    newly_spent: UsageTotals | None = None,
    base_url: str = "",
    endpoint_map: Mapping[str, str] | None = None,
) -> JobBill:
    """Fold this run's spend into the job's recorded bill and persist it.

    The prior bill is read from the ledger per event instead of snapshotted at
    run() entry, because ``--fresh`` zeroes the job's recorded usage during
    ingest: a start-of-run read would bill the run the bill it just discarded.

    ``run_usage`` is this run's *cumulative* total and is what the returned
    :class:`JobBill` reports. ``newly_spent`` is the part of it not yet in the
    ledger. They differ because the caller re-bills on every progress event:
    ``record_job_usage`` writes an absolute figure, so folding the whole
    cumulative total into an already-updated ``prior`` re-adds every earlier
    event's tokens and inflates the stored bill super-linearly — enough to trip
    ``UBT_BUDGET_USD`` on spend the job never made. Callers that cannot track
    the delta may omit it, which is correct for a one-shot write.
    """
    if not run_usage:
        prior = await asyncio.to_thread(ledger.get_job_usage, job_id)
        return JobBill(
            run_usage={}, lifetime_usage=prior, base_url=base_url, endpoint_map=endpoint_map
        )
    increment = run_usage if newly_spent is None else newly_spent
    atomic_fn = getattr(ledger, "atomic_increment_job_usage", None)
    if callable(atomic_fn):
        lifetime = await asyncio.to_thread(atomic_fn, job_id, increment)
    else:
        prior = await asyncio.to_thread(ledger.get_job_usage, job_id)
        lifetime = merge_usage_totals(prior, increment)
        await asyncio.to_thread(ledger.record_job_usage, job_id, lifetime)
    return JobBill(
        run_usage=run_usage,
        lifetime_usage=lifetime,
        base_url=base_url,
        endpoint_map=endpoint_map,
    )


def budget_violation(bill: JobBill, budget_usd: float | None, job_id: str) -> str | None:
    """Return the stop message when the job has outgrown its cap, else None.

    A hard stop by design: the ledger keeps everything drafted so far, so
    raising UBT_BUDGET_USD and rerunning the same job id resumes from here.
    """
    cost = bill.cost_usd
    if budget_usd is None or cost is None or cost <= budget_usd:
        return None
    run_cost = bill.run_cost_usd
    this_run = f" (this run ${run_cost:.4f})" if run_cost is not None else ""
    return (
        f"Budget exceeded: job cost ${cost:.4f}{this_run} > "
        f"--budget-usd/UBT_BUDGET_USD ${budget_usd:.4f}. "
        f"Job {job_id} stopped; blocks and spend so far are kept in the "
        "ledger — raise the budget and rerun the same job id to continue."
    )
