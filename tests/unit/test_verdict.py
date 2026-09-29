"""Verdict rules order, thresholds, and stamping."""

import pytest

from ubt.core.ir.models import BlockType, FlowID, IRBlock, LayoutRole, SemanticRole
from ubt.core.policy.verdict import (
    Verdict,
    apply_verdict,
    judge_block,
)


def _block(**kwargs: object) -> IRBlock:
    base: dict[str, object] = {
        "id": "v1",
        "spine_index": 0,
        "source_text": "Ordinary prose that deserves translation.",
    }
    base.update(kwargs)
    return IRBlock(**base)  # type: ignore[arg-type]


def test_prose_translates() -> None:
    v = judge_block(_block())
    assert v == Verdict(True, "verdict:translate", should_call_model=True)


def test_empty_and_placeholder_keep() -> None:
    assert judge_block(_block(source_text="   ")).reason == "verdict:empty"
    assert judge_block(_block(source_text="*** --- @@@")).reason == "verdict:placeholder"


def test_explicit_keep_is_respected() -> None:
    b = _block(policy_translate=False, policy_reason="verdict:chrome")
    assert judge_block(b) == Verdict(False, "verdict:chrome")
    # Explicit keep wins over content rules too: the audit reason survives
    # even for empty text (previously overwritten by verdict:empty).
    empty = _block(source_text="   ", policy_translate=False, policy_reason="verdict:chrome")
    assert judge_block(empty) == Verdict(False, "verdict:chrome")


def test_non_text_types_keep() -> None:
    for btype in (BlockType.FORMULA, BlockType.CODE, BlockType.IMAGE):
        assert judge_block(_block(block_type=btype)).reason == "verdict:non_text"
    assert judge_block(_block(skip_translate=True)).reason == "verdict:non_text"
    # A table's cells are book content, not verbatim residue: the parser marks
    # them skip=False and the QE layer requires an untranslated table to fail,
    # so keeping one here meant every table shipped in the source language
    # stamped MTQE_PASSED with a perfect pass rate.
    assert judge_block(_block(block_type=BlockType.TABLE)).translate is True


def test_chrome_and_role_keep() -> None:
    assert judge_block(_block(layout_role=LayoutRole.FOOTER)).reason == "verdict:chrome"
    assert judge_block(_block(layout_role=LayoutRole.PAGE_NUMBER)).reason == "verdict:chrome"
    assert (
        judge_block(_block(semantic_role=SemanticRole.REFERENCE)).reason == "verdict:non_prose_role"
    )


def test_chrome_opt_in_translates_heads_but_never_page_numbers() -> None:
    header = _block(layout_role=LayoutRole.HEADER)
    assert judge_block(header, translate_chrome=True).translate is True
    assert (
        judge_block(_block(layout_role=LayoutRole.FOOTER), translate_chrome=True).translate is True
    )
    # Page numbers stay chrome under the opt-in.
    assert (
        judge_block(_block(layout_role=LayoutRole.PAGE_NUMBER), translate_chrome=True).reason
        == "verdict:chrome"
    )
    # The adapter's parse-time skip still holds even with the flag on.
    assert (
        judge_block(
            _block(layout_role=LayoutRole.HEADER, skip_translate=True), translate_chrome=True
        ).reason
        == "verdict:non_text"
    )


def test_identifier_only_keep() -> None:
    assert judge_block(_block(source_text="https://doi.org/10.1000/xyz")).reason == (
        "verdict:identifier_only"
    )
    assert judge_block(_block(source_text="978-3-16-148410-0")).reason == "verdict:identifier_only"
    # Identifiers embedded in prose do NOT trigger the rule.
    assert judge_block(_block(source_text="See https://example.com for details.")).translate is True


def test_hexdump_keep() -> None:
    blob = " ".join(f"{i:02x}" for i in range(48))
    assert judge_block(_block(source_text=blob)).reason == "verdict:hexdump"
    # Short hex fragments in prose stay translatable.
    assert judge_block(_block(source_text="The bytes dead beef encode it.")).translate is True


def test_index_list_keep() -> None:
    entries = "\n".join(f"term-{i}" for i in range(14))
    assert judge_block(_block(source_text=entries)).reason == "verdict:index_list"
    # Few lines never trigger.
    assert judge_block(_block(source_text="alpha\nbeta\ngamma")).translate is True


def test_short_label_in_caption_keep() -> None:
    b = _block(source_text="B1", flow_id=FlowID.CAPTION)
    assert judge_block(b).reason == "verdict:short_label"
    # Same text in the main story translates (e.g. a heading).
    assert judge_block(_block(source_text="B1")).translate is True


def test_short_label_does_not_keep_prose_table_cells() -> None:
    """A multi-word prose label must be translated, not kept as a label.

    ``SHORT_LABEL_RE`` matched any alnum run <=24 chars, so a TABLE_GRID cell
    like "effective mobility" was kept in the source language.
    """
    prose = _block(source_text="effective mobility", flow_id=FlowID.TABLE_GRID)
    assert judge_block(prose).translate is True
    # A genuine label (single token / mixed case / unit) is still kept.
    assert (
        judge_block(_block(source_text="B1", flow_id=FlowID.CAPTION)).reason
        == "verdict:short_label"
    )
    assert (
        judge_block(_block(source_text="Fig. 3", flow_id=FlowID.CAPTION)).reason
        == "verdict:short_label"
    )


def test_rule_order_chrome_before_hexdump() -> None:
    blob = " ".join(f"{i:02x}" for i in range(48))
    v = judge_block(_block(source_text=blob, layout_role=LayoutRole.HEADER))
    assert v.reason == "verdict:chrome"


def test_apply_verdict_stamps_and_skips() -> None:
    b = _block(source_text="B1", flow_id=FlowID.CAPTION)
    apply_verdict(b, judge_block(b))
    assert b.policy_translate is False
    assert b.policy_reason == "verdict:short_label"
    assert b.skip_translate is True
    assert b.validate_contract() == []


def test_apply_verdict_translate_path() -> None:
    b = _block()
    apply_verdict(b, judge_block(b))
    assert b.policy_translate is True
    assert b.skip_translate is False


def test_verdict_invariant() -> None:
    with pytest.raises(ValueError):
        Verdict(True, "x", should_call_model=False)
