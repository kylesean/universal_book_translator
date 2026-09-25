"""Anchored placement gate: chrome opt-in, policy keeps and band heuristics."""

from ubt.adapters.pdf.rigid.gate import skip_reason
from ubt.core.ir.models import BlockType, BoundingBox, FlowID, IRBlock, LayoutRole


def _block(
    source: str = "Ordinary prose that deserves a translation.",
    target: str = "值得翻译的普通正文。",
    role: LayoutRole = LayoutRole.BODY,
    y0: float = 300.0,
    y1: float = 320.0,
    *,
    skip: bool = False,
    policy: bool | None = True,
) -> IRBlock:
    return IRBlock(
        id="g1",
        flow_id=FlowID.MAIN_STORY,
        spine_index=0,
        block_type=BlockType.NARRATIVE,
        source_text=source,
        target_text=target,
        layout_role=role,
        skip_translate=skip,
        policy_translate=policy,
        bbox=BoundingBox(page=1, x0=100.0, y0=y0, x1=400.0, y1=y1),
    )


def test_chrome_stays_source_visible_by_default() -> None:
    for role in (LayoutRole.HEADER, LayoutRole.FOOTER, LayoutRole.PAGE_NUMBER):
        assert skip_reason(_block(role=role, y0=630.0, y1=640.0), 660.0) == "chrome"


def test_chrome_opt_in_paints_a_translated_running_head() -> None:
    head = _block(
        source="3.2 Unified FinFET compact model",
        target="3.2 统一 FinFET 紧凑模型",
        role=LayoutRole.HEADER,
        y0=630.0,
        y1=640.0,
    )
    assert skip_reason(head, 660.0, translate_chrome=True) is None
    # Geometry alone no longer vetoes the band once the flag is on.
    foot = _block(
        source="Chapter 3 Compact modelling",
        target="第 3 章 紧凑建模",
        role=LayoutRole.FOOTER,
        y0=40.0,
        y1=50.0,
    )
    assert skip_reason(foot, 660.0, translate_chrome=True) is None


def test_chrome_opt_in_never_paints_numbers_or_untranslated_heads() -> None:
    number = _block(source="77", target="77", role=LayoutRole.PAGE_NUMBER, y0=630.0, y1=640.0)
    assert skip_reason(number, 660.0, translate_chrome=True) == "chrome"
    # A numeric band block roled HEADER is the same page number in all but name.
    mislabeled = _block(source="78", target="78", role=LayoutRole.HEADER, y0=630.0, y1=640.0)
    assert skip_reason(mislabeled, 660.0, translate_chrome=True) == "chrome"
    # Frozen ledgers carry skip=True on chrome: the flag must not repaint them.
    frozen = _block(
        source="FinFET/GAA Modeling for IC Simulation and Design.",
        target="FinFET/GAA Modeling for IC Simulation an",
        role=LayoutRole.FOOTER,
        y0=40.0,
        y1=50.0,
        skip=True,
    )
    assert skip_reason(frozen, 660.0, translate_chrome=True) == "policy"


def test_verbatim_target_is_never_repainted() -> None:
    same = _block(
        source="Same as source",
        target="Same as source",
        role=LayoutRole.HEADER,
        y0=630.0,
        y1=640.0,
    )
    assert skip_reason(same, 660.0, translate_chrome=True) == "verbatim"


def test_body_band_heuristics_still_apply() -> None:
    short = _block(source="Short tail", target="短尾", y0=40.0, y1=50.0)
    assert skip_reason(short, 660.0) == "footer_band"
