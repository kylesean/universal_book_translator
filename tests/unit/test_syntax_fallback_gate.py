"""RED: Typst syntax fallbacks must gate delivery, not just warn."""

import pytest

from ubt.core.config import UBTConfig
from ubt.core.engine.stages.export import _enforce_syntax_fallback_gate
from ubt.core.exceptions import UBTError

pytestmark = pytest.mark.fast


def test_syntax_fallback_gate_blocks_over_threshold():
    cfg = UBTConfig(export_max_syntax_fallbacks=2)
    fallbacks = ["line 1: foo", "line 2: bar", "line 3: baz"]
    with pytest.raises(UBTError):
        _enforce_syntax_fallback_gate("job1", fallbacks, cfg, rehearsal=False)


def test_syntax_fallback_gate_allows_under_threshold():
    cfg = UBTConfig(export_max_syntax_fallbacks=5)
    _enforce_syntax_fallback_gate("job1", ["line 1: foo"], cfg, rehearsal=False)


def test_syntax_fallback_gate_rehearsal_downgrades_to_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    cfg = UBTConfig(export_max_syntax_fallbacks=1)
    with caplog.at_level(logging.WARNING):
        # mock/dry-run must not raise
        _enforce_syntax_fallback_gate("job1", ["a", "b", "c"], cfg, rehearsal=True)
    # The behaviour under test is the downgrade *to a warning*; deleting the
    # warning branch must redden this, not silently pass.
    assert any("syntax-fallback gate would have refused" in rec.message for rec in caplog.records)


def test_syntax_fallback_gate_zero_means_fail_on_any():
    cfg = UBTConfig(export_max_syntax_fallbacks=0)
    with pytest.raises(UBTError):
        _enforce_syntax_fallback_gate("job1", ["line 1: foo"], cfg, rehearsal=False)
