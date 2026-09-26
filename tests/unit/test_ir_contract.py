"""Document-v1 seven-layer contract on IRBlock (backward compatible)."""

import pytest

from ubt.core.ir.models import (
    BlockType,
    BoundingBox,
    FlowID,
    IRBlock,
    LayoutRole,
    SemanticRole,
    StructureRole,
)

pytestmark = pytest.mark.fast


def _block(**kwargs: object) -> IRBlock:
    base: dict[str, object] = {
        "id": "b1",
        "spine_index": 0,
        "source_text": "Hello world.",
    }
    base.update(kwargs)
    return IRBlock(**base)  # type: ignore[arg-type]


def test_legacy_blocks_default_to_old_behavior() -> None:
    b = _block(skip_translate=True)
    assert b.layout_role is None
    assert b.policy_translate is None
    assert b.provenance == {}
    assert b.effective_should_translate() is False
    assert b.validate_contract() == []


def test_explicit_policy_wins_over_skip_translate() -> None:
    b = _block(skip_translate=True, policy_translate=True)
    assert b.effective_should_translate() is True
    b2 = _block(skip_translate=False, policy_translate=False, policy_reason="header")
    assert b2.effective_should_translate() is False


def test_derive_roles_respects_explicit() -> None:
    b = _block(flow_id=FlowID.CAPTION, block_type=BlockType.NARRATIVE)
    b.derive_roles()
    assert b.layout_role == LayoutRole.CAPTION
    assert b.structure_role == StructureRole.PARAGRAPH
    assert b.semantic_role == SemanticRole.MAIN_TEXT
    # Explicit role is never overwritten by derivation.
    b.layout_role = LayoutRole.BODY
    b.derive_roles()
    assert b.layout_role == LayoutRole.BODY


def test_derive_roles_flow_mapping() -> None:
    foot = _block(flow_id=FlowID.FOOTNOTE)
    foot.derive_roles()
    assert foot.layout_role == LayoutRole.FOOTNOTE
    formula = _block(block_type=BlockType.FORMULA)
    formula.derive_roles()
    assert formula.structure_role == StructureRole.FORMULA


def test_validate_contract_bbox() -> None:
    bad = _block(bbox=BoundingBox(page=1, x0=10.0, y0=10.0, x1=5.0, y1=20.0))
    assert any("non-positive area" in v for v in bad.validate_contract())
    nan = _block(bbox=BoundingBox(page=1, x0=float("nan"), y0=0.0, x1=5.0, y1=5.0))
    assert any("non-finite" in v for v in nan.validate_contract())
    good = _block(bbox=BoundingBox(page=1, x0=0.0, y0=0.0, x1=5.0, y1=5.0))
    assert good.validate_contract() == []


def test_validate_contract_policy_reason() -> None:
    b = _block(policy_translate=False)
    assert any("policy_reason" in v for v in b.validate_contract())
    b.policy_reason = "header band"
    assert b.validate_contract() == []
