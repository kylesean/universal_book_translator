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
