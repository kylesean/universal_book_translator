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


def test_render_skips_split_into_fail_closed_and_preserved() -> None:
    """``preserved`` used to be read from a whitelist no producer wrote.

    The truth is the opposite: the renderer emits a ``render_skip:`` flag on
    *every* skip it makes, including the by-design keeps (``skip_translate`` on
    bibliography entries and page chrome, ``facsimile_preserved_opaque`` on the
    background-layer formula/figure ink). Booking those as fail-closed made
    ``render_coverage`` read 0.0 on the 26-page replay while all 105 translatable
    blocks had shipped.
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

    # A by-design keep did not lose a translation: it must not inflate the
    # fail-closed count. On the 26-page replay every non-translatable block
    # carried one of these, and counting them made render_coverage read 0.0
    # while all 105 translatable blocks had shipped.
    fail_closed, preserved, verdict = summarize_render_skips(
        {"render_skip:skip_translate": 61, "render_skip:facsimile_preserved_opaque": 85}
    )
    assert (fail_closed, preserved) == (0, 146)
    assert "No lost translations" in verdict
    assert "keeps" in verdict
