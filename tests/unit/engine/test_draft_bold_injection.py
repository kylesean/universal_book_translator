"""Unit tests for injecting bold markers into the draft's masked source."""

from __future__ import annotations

from typing import Any

import pytest

from ubt.core.engine.stages.draft import _source_with_bold_markers
from ubt.core.ir.emphasis import BOLD_CLOSE, BOLD_OPEN
from ubt.core.ir.models import (
    BlockProvenance,
    BlockType,
    InlineRun,
    IRBlock,
    StyleMeta,
    make_element,
)

pytestmark = pytest.mark.fast


def _block(
    source: str, style: StyleMeta | None = None, provenance: dict[str, Any] | None = None
) -> IRBlock:
    element = make_element(
        id="b1", spine_index=1, block_type=BlockType.NARRATIVE, source_text=source
    )
    return IRBlock(
        element=element,
        style=style,
        provenance=BlockProvenance(**(provenance or {})),
    )


def test_injects_markers_around_bold_spans() -> None:
    block = _block(
        "图注 SoL-Pi 降低成本 50.0%。",
        StyleMeta(inline_runs=(InlineRun(text="50.0%", bold=True),)),
    )
    assert (
        _source_with_bold_markers(block) == f"图注 SoL-Pi 降低成本 {BOLD_OPEN}50.0%{BOLD_CLOSE}。"
    )


def test_whole_block_bold_is_left_to_is_bold() -> None:
    block = _block(
        "全部加粗",
        StyleMeta(inline_runs=(InlineRun(text="全部加粗", bold=True),)),
        {"is_bold": True},
    )
    assert _source_with_bold_markers(block) == "全部加粗"


def test_non_bold_runs_do_not_inject() -> None:
    block = _block("普通文本", StyleMeta(inline_runs=(InlineRun(text="普通", bold=False),)))
    assert _source_with_bold_markers(block) == "普通文本"


def test_no_style_leaves_source_unchanged() -> None:
    assert _source_with_bold_markers(_block("普通文本")) == "普通文本"
