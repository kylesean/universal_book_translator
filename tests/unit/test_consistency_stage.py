def test_consistency_results_reraise_cooperative_interrupts() -> None:
    """Budget/cancel outcomes must reach the pipeline, not die as warnings.

    repair.py re-raises hard stops from its gather; consistency downgraded
    every non-dict result to a warning, so a cancellation mid-repair surfaced
    as a log line and the stage kept finalizing drifted blocks.
    """
    import asyncio

    import pytest

    from ubt.core.engine.stages.consistency import _split_repair_results
    from ubt.core.exceptions import BudgetExceededError, JobInterruptedError

    with pytest.raises(BudgetExceededError):
        _split_repair_results([{"block_id": "b"}, BudgetExceededError("cap")])
    with pytest.raises(JobInterruptedError):
        _split_repair_results([JobInterruptedError("cancel")])
    with pytest.raises(asyncio.CancelledError):
        _split_repair_results([asyncio.CancelledError()])
    updates = _split_repair_results([{"block_id": "b"}, RuntimeError("ledger hiccup")])
    assert updates == [{"block_id": "b"}]
