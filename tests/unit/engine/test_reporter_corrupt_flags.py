"""Placeholder-corruption counting for the KDP audit report.

The audit recomputes retention from the persisted ``*_token_corrupt`` flags. All
four maskers (soup, math, cite, code) can corrupt a span, and a family omitted
from the flag regex reads as zero corruption -- the report then prints "Full
retention" while the ledger flagged corruption. These tests pin every family.
"""

from __future__ import annotations

import pytest

from ubt.core.engine.reporter import _parse_corrupt_count

pytestmark = pytest.mark.fast


@pytest.mark.parametrize("label", ["soup", "math", "cite", "code"])
def test_parse_corrupt_count_counts_every_masked_family(label: str) -> None:
    flag = (
        f"{label}_token_corrupt missing=[1] mismatched=[2] mutated=[3] reordered=[4] duplicated=[5]"
    )
    assert _parse_corrupt_count(flag) == 5


def test_parse_corrupt_count_reads_soup_corruption() -> None:
    # Regression: the regex listed only math/cite/code, so a corrupt soup span
    # was invisible and retention stayed at 1.0.
    flag = "soup_token_corrupt missing=[] mismatched=[] mutated=[7] reordered=[] duplicated=[]"
    assert _parse_corrupt_count(flag) == 1


def test_parse_corrupt_count_ignores_unrelated_flags() -> None:
    assert _parse_corrupt_count("html_attr_mismatch missing=[1] mismatched=[2]") == 0


def test_every_render_skip_is_fail_closed() -> None:
    """The report splits skips into fail-closed vs preserved; only the first exists.

    ``preserved`` used to be read from an "intentional skip" whitelist that no
    producer ever wrote. Its real meaning is ``skip_translate`` (a block shipped
    in the source by design), which is counted from the blocks themselves — see
    ``build_quality_report``. A ``render_skip:`` is always a translation the
    renderer could not place.
    """
    from ubt.core.engine.reporter import summarize_render_skips

    assert summarize_render_skips({}) == (0, 0, "No source-visible skips recorded")

    fail_closed, preserved, verdict = summarize_render_skips(
        {"render_skip:table_unparseable": 2, "render_skip:asset_not_found": 1}
    )
    assert (fail_closed, preserved) == (3, 0)
    assert "review required" in verdict
    # Families strip fit parameters so per-size variants do not fragment the audit.
    assert "asset_not_found×1" in verdict
