from __future__ import annotations

import pytest

from ubt.core.qe.roundtrip import (
    RoundTripVerifier,
    check_polarity_conservation,
    compute_sentence_bleu,
    extract_key_entities,
    tokenize_words,
)
from ubt.core.qe.roundtrip_matrix import (
    MatrixAuditSummary,
    MultiLanguageMatrixAuditor,
    render_json_matrix,
    render_markdown_matrix,
    render_terminal_matrix,
)

pytestmark = pytest.mark.fast


def test_tokenize_words() -> None:
    en_tokens = tokenize_words("The gradient descent algorithm converges at t=10.")
    assert "gradient" in en_tokens
    assert "algorithm" in en_tokens
    assert "10" in en_tokens

    zh_tokens = tokenize_words("梯度下降算法收敛", lang="zh")
    assert len(zh_tokens) >= 2


def test_compute_sentence_bleu() -> None:
    ref = ["the", "model", "converges", "quickly"]
    hyp = ["the", "model", "converges", "quickly"]
    assert compute_sentence_bleu(ref, hyp) == 1.0

    hyp_mismatch = ["totally", "different", "words", "here"]
    assert compute_sentence_bleu(ref, hyp_mismatch, smooth=False) == 0.0

    # Short sentence with smoothing
    hyp_partial = ["the", "model", "runs"]
    bleu_partial = compute_sentence_bleu(ref, hyp_partial, smooth=True)
    assert 0.0 < bleu_partial < 1.0


def test_extract_key_entities() -> None:
    text = "We achieved 98.5% accuracy with 1024 batch size, learning rate 1e-4 and $E=mc^2$."
    entities = extract_key_entities(text)
    assert "98.5%" in entities
    assert "1024" in entities
    assert "1e-4" in entities
    assert "$E=mc^2$" in entities


def test_check_polarity_conservation() -> None:
    # Affirmative to affirmative
    ok, _ = check_polarity_conservation("The model converges.", "The model converges.")
    assert ok

    # Negative to negative
    ok, _ = check_polarity_conservation(
        "The model does not converge.", "The model fails to converge."
    )
    assert ok

    # Polarity flip (negative -> affirmative)
    ok, reason = check_polarity_conservation(
        "The model does not converge.", "The model converges quickly."
    )
    assert not ok
    assert "Polarity flip" in reason

    # Polarity flip (affirmative -> negative)
    ok, reason = check_polarity_conservation(
        "The loss decreases monotonically.", "The loss does not decrease."
    )
    assert not ok
    assert "Polarity flip" in reason


@pytest.mark.asyncio
async def test_roundtrip_verifier_perfect_translation() -> None:
    verifier = RoundTripVerifier(threshold=0.65)
    src = "In Section 2.1, we evaluate 500 benchmark cases with $O(N^2)$ complexity."
    mt = "在第 2.1 节中，我们评估了复杂度为 $O(N^2)$ 的 500 个基准用例。"
    back = "In Section 2.1, we evaluate 500 benchmark cases with $O(N^2)$ complexity."

    res = await verifier.verify_pair(src, mt, back_translated=back)
    assert res.passed
    assert res.composite_score >= 0.90
    assert res.polarity_conserved
    assert res.entity_conservation == 1.0
    assert len(res.defect_flags) == 0
    assert res.mqm_category == "clean"


@pytest.mark.asyncio
async def test_roundtrip_verifier_negation_flip() -> None:
    verifier = RoundTripVerifier(threshold=0.65)
    src = "The optimizer does not diverge even under high learning rates."
    mt = "优化器在较高学习率下会发散。"
    back = "The optimizer diverges under high learning rates."

    res = await verifier.verify_pair(src, mt, back_translated=back)
    assert not res.passed
    assert not res.polarity_conserved
    assert res.composite_score <= 0.40
    assert any("Polarity flip" in f for f in res.defect_flags)
    assert res.mqm_severity == "major"


@pytest.mark.asyncio
async def test_roundtrip_verifier_lost_numbers() -> None:
    verifier = RoundTripVerifier(threshold=0.65)
    src = "The baseline error rate is 14.8% at epoch 50."
    mt = "基准错误率较低。"
    back = "The baseline error rate is relatively low."

    res = await verifier.verify_pair(src, mt, back_translated=back)
    assert not res.passed
    assert res.entity_conservation == 0.0
    assert "14.8%" in res.missing_entities
    assert "50" in res.missing_entities
    assert res.mqm_severity == "critical"
    assert any("Lost key numbers/entities" in f for f in res.defect_flags)


@pytest.mark.asyncio
async def test_roundtrip_verifier_empty_text() -> None:
    verifier = RoundTripVerifier()
    res = await verifier.verify_pair("", "")
    assert not res.passed
    assert res.mqm_severity == "critical"


@pytest.mark.asyncio
async def test_matrix_auditor_multiple_pairs() -> None:
    auditor = MultiLanguageMatrixAuditor()

    pairs = [
        (
            "Matrix multiplication takes $O(N^3)$ operations.",
            "矩阵乘法需要 $O(N^3)$ 次运算。",
            "Matrix multiplication requires $O(N^3)$ operations.",
        ),
        (
            "This property is not invariant.",
            "该性质是不变的。",
            "This property is invariant.",
        ),
    ]

    summary = await auditor.audit_pairs(pairs, src_lang="en", tgt_lang="zh")
    assert summary.sample_count == 2
    assert summary.passed_count == 1
    assert summary.pass_rate == 50.0
    assert summary.polarity_accuracy_rate == 50.0
    assert summary.mqm_major_count == 1


def test_matrix_reporters() -> None:
    summary = MatrixAuditSummary(
        direction="en <-> zh",
        sample_count=10,
        passed_count=8,
        pass_rate=80.0,
        avg_composite_score=0.825,
        avg_bleu=0.78,
        avg_fuzzy=0.85,
        entity_conservation_rate=95.0,
        polarity_accuracy_rate=90.0,
        mqm_critical_count=0,
        mqm_major_count=1,
        mqm_minor_count=1,
    )

    # Markdown export
    md = render_markdown_matrix([summary])
    assert "en <-> zh" in md
    assert "80.0%" in md

    # JSON export
    data = render_json_matrix([summary])
    assert len(data["matrix_summaries"]) == 1
    assert data["matrix_summaries"][0]["pass_rate"] == 80.0

    # Terminal rendering smoke
    render_terminal_matrix([summary])
