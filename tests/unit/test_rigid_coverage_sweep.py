"""The rigid coverage sweep's page selector.

``_parse_pages`` is the only branching logic in
``scripts/rigid_coverage_sweep.py``: the sweep itself is measurement (exit 0),
but a wrong page subset silently measures the wrong geometry or nothing at all.
The two-corpus calibration that justified ``RIGID_MIN_FONT_PT`` 7.5 -> 7.0 needed
a page subset to make the 26-page two-column corpus tractable, so the selector's
edges are pinned here.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "rigid_coverage_sweep.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("ubt_rigid_coverage_sweep", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sweep = _load()

pytestmark = pytest.mark.fast


def test_none_selects_every_page() -> None:
    assert sweep._parse_pages(None, 4) == [1, 2, 3, 4]


def test_range_and_list_are_parsed_and_sorted_deduped() -> None:
    assert sweep._parse_pages("3-5,1", 10) == [1, 3, 4, 5]
    assert sweep._parse_pages("1,3,5", 10) == [1, 3, 5]


def test_out_of_range_pages_are_clamped_not_wrapped() -> None:
    # 12 does not exist in a 10-page book: drop it rather than index-error later.
    assert sweep._parse_pages("9-12", 10) == [9, 10]


def test_a_selection_with_no_valid_page_is_an_error() -> None:
    with pytest.raises(ValueError, match="selects no page"):
        sweep._parse_pages("11-12", 10)


def test_reversed_range_is_an_error() -> None:
    with pytest.raises(ValueError, match="invalid page range"):
        sweep._parse_pages("5-2", 10)
