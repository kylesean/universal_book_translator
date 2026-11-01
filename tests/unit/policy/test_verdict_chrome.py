from __future__ import annotations

import pytest

from ubt.core.ir.models import BlockType, BoundingBox, FlowID, IRBlock, make_element
from ubt.core.policy.verdict import judge_block
from ubt.model.ast import RegionKind

pytestmark = pytest.mark.fast


def _footer_block(text: str, *, region: RegionKind = RegionKind.FOOTER) -> IRBlock:
    return IRBlock(
        element=make_element(
            id="pdf_main#footer",
            spine_index=1,
            block_type=BlockType.NARRATIVE,
            flow_id=FlowID.FOOTNOTE,
            region=region,
            source_text=text,
            bbox=BoundingBox(page=1, x0=70.0, y0=20.0, x1=520.0, y1=40.0),
        )
    )


def test_a_prose_footer_is_chrome_and_is_kept_by_default() -> None:
    footer = _footer_block(
        "Jialiang Huang is a Ph.D. student advised by Mingxing Zhang. "
        "He contributed to this work during an internship at DeepSeek-AI."
    )
    verdict = judge_block(footer)

    assert verdict.translate is False
    assert verdict.reason == "verdict:chrome"


def test_translate_chrome_opts_a_prose_footer_into_translation() -> None:
    footer = _footer_block(
        "Jialiang Huang is a Ph.D. student advised by Mingxing Zhang. "
        "He contributed to this work during an internship at DeepSeek-AI."
    )
    verdict = judge_block(footer, translate_chrome=True)

    assert verdict.translate is True
    assert verdict.reason == "verdict:translate"


def test_a_page_number_is_never_translated_even_with_chrome_opt_in() -> None:
    footer = _footer_block("Page 17 of 50", region=RegionKind.PAGE_NUMBER)
    verdict = judge_block(footer, translate_chrome=True)

    assert verdict.translate is False
    assert verdict.reason == "verdict:chrome"
