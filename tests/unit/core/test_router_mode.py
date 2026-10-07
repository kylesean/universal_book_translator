"""PDF routing must not price CJK text with the ASCII ``chars // 4`` rule."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from ubt.core import router_mode
from ubt.core.ports import PdfStructureFacts

pytestmark = pytest.mark.fast


def test_pdf_token_estimate_uses_the_sample_script_ratio() -> None:
    cjk_preview = "中" * 500
    with patch.object(router_mode, "sample_pdf_pages", return_value=(10, False, cjk_preview)):
        est = router_mode._pdf_script_aware_tokens(Path("book.pdf"), 6000)
    # CJK ~0.85 tok/char, far above the flat 6000 // 4 = 1500.
    assert est > 6000 // 4


def test_pdf_token_estimate_falls_back_without_a_sample() -> None:
    with patch.object(router_mode, "sample_pdf_pages", return_value=(1, True, "")):
        est = router_mode._pdf_script_aware_tokens(Path("scan.pdf"), 400)
    assert est == 400 // 4


def test_a_cjk_pdf_is_not_misrouted_to_the_short_chain() -> None:
    facts = PdfStructureFacts(
        has_scan=False,
        formula_heavy=False,
        multicolumn_page_share=0.0,
        structural_page_share=0.0,
    )
    with (
        patch.object(router_mode, "probe_pdf_pages", return_value=(5, 6000)),
        patch.object(router_mode, "sample_pdf_pages", return_value=(5, False, "中" * 500)),
        patch.object(router_mode, "classify_pdf_structure", return_value=facts),
    ):
        decision = router_mode.decide(Path("book.pdf"), short_max_pages=5)

    # 6000 CJK chars ~= 5100 tokens, over the 5 * 800 budget; chars // 4 (1500)
    # would have wrongly kept it on the short single-shot chain.
    assert decision.estimated_tokens > 5 * 800
    assert decision.mode == "long"
