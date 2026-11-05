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
    flag = "soup_token_corrupt missing=[] mismatched=[] mutated=[7] reordered=[] duplicated=[]"
    assert _parse_corrupt_count(flag) == 1


def test_parse_corrupt_count_ignores_unrelated_flags() -> None:
    assert _parse_corrupt_count("html_attr_mismatch missing=[1] mismatched=[2]") == 0


def test_render_skips_split_into_fail_closed_and_preserved() -> None:
    from ubt.core.engine.reporter import summarize_render_skips

    assert summarize_render_skips({}) == (0, 0, "No source-visible skips recorded")

    fail_closed, preserved, verdict = summarize_render_skips(
        {"render_skip:table_unparseable": 2, "render_skip:asset_not_found": 1}
    )
    assert (fail_closed, preserved) == (3, 0)
    assert "review required" in verdict
    assert "asset_not_found×1" in verdict

    fail_closed, preserved, verdict = summarize_render_skips(
        {"render_skip:skip_translate": 61, "render_skip:facsimile_preserved_opaque": 85}
    )
    assert (fail_closed, preserved) == (0, 146)
    assert "No lost translations" in verdict
    assert "keeps" in verdict


def test_a_preserved_source_table_is_not_a_lost_translation() -> None:
    from ubt.core.engine.reporter import summarize_render_skips

    fail_closed, preserved, verdict = summarize_render_skips(
        {"render_skip:table_preserved_source": 3}
    )
    assert (fail_closed, preserved) == (0, 3)
    assert "No lost translations" in verdict
