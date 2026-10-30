"""Unit tests for ContainerFitRepair (P2-16)."""

from __future__ import annotations

from pathlib import Path

import pytest

from ubt.core.ir.models import BlockType, IRBlock, make_element
from ubt.render.typst.container_repair import (
    ContainerFitSpec,
    compute_container_char_budget,
    concise_rewrite_heuristic,
    effective_char_width,
    extract_locked_numbers,
    fit_code_block,
    inject_soft_breaks,
    measure_line_width,
    repair_bottleneck_containers,
    repair_code_block_markup,
    smart_wrap_code_line,
    verify_concise_candidate,
)
from ubt.render.typst.document_renderer import TypstDocumentRenderer

pytestmark = pytest.mark.fast


def test_effective_char_width() -> None:
    assert effective_char_width("a") == 1.0
    assert effective_char_width("1") == 1.0
    assert effective_char_width(" ") == 1.0
    assert effective_char_width("中") == 2.0
    assert effective_char_width("文") == 2.0
    assert effective_char_width("（") == 2.0
    assert effective_char_width("，") == 2.0


def test_measure_line_width() -> None:
    assert measure_line_width("hello world") == 11.0
    # "print('你好')" -> p,r,i,n,t,(,',',)=9 + 你,好=4 -> 13.0
    assert measure_line_width("print('你好')") == 13.0
    # Tab expansion
    assert measure_line_width("\tx") == 5.0  # 4 spaces + 'x'


def test_smart_wrap_code_line_short() -> None:
    line = "def foo(a, b):"
    wrapped = smart_wrap_code_line(line, max_units=40)
    assert wrapped == [line]


def test_smart_wrap_code_line_at_commas() -> None:
    line = "def process_data(user_id, account_balance, transaction_history, authorization_token):"
    wrapped = smart_wrap_code_line(line, max_units=45)
    assert len(wrapped) >= 2
    # First line should break cleanly after a parameter comma
    assert wrapped[0].endswith(",") or wrapped[0].endswith("(")
    # Continuation line should have hanging indent
    assert wrapped[1].startswith("    ")


def test_smart_wrap_code_line_method_chaining() -> None:
    line = "result = query.filter(User.active == True).order_by(User.created_at.desc()).limit(10)"
    wrapped = smart_wrap_code_line(line, max_units=40)
    assert len(wrapped) >= 2
    for sub in wrapped[1:]:
        assert sub.startswith("    ")


def test_fit_code_block_adaptive_scaling() -> None:
    # 50 chars in ~240pt column: can fit at ~7.8pt without line breaking
    code = (
        "def compute_scaled_dot_product_attention(q, k, v):\n"
        "    return softmax(q @ k.T / math.sqrt(d_k)) @ v"
    )
    fitted, size = fit_code_block(code, container_width_pt=240.0, base_font_size_pt=8.5)
    assert "\n" in fitted
    # Should scale down below 8.5pt to fit container
    assert 7.0 <= size < 8.5
    # No line wrapping was required because scaling absorbed the width
    assert len(fitted.splitlines()) == len(code.splitlines())


def test_fit_code_block_wrapping_when_too_long() -> None:
    # Extremely long line that exceeds container even at 7.0pt floor
    code = (
        "const response = await fetch('https://api.example.com/v1/projects/quantum-computing-research/experiments/run', "
        "{ method: 'POST', headers: { 'Authorization': `Bearer ${token}` }, body: JSON.stringify(payload) });"
    )
    fitted, size = fit_code_block(
        code, container_width_pt=200.0, base_font_size_pt=8.5, min_font_size_pt=7.0
    )
    assert size == 7.0
    # Must have wrapped into multiple lines
    assert len(fitted.splitlines()) > 1


def test_repair_code_block_markup_quotes_and_slashes() -> None:
    code = 'path = "C:\\\\Users\\\\Alice"; print(f"Processing \\"{path}\\"")'
    markup = repair_code_block_markup(code, container_width_pt=450.0)
    assert "#block(fill: luma(245)" in markup
    assert '#raw("' in markup
    assert '\\"' in markup


def test_inject_soft_breaks_url_and_paths() -> None:
    long_url = "https://github.com/universal-book-translator/blob/main/ubt/render/typst/document_renderer.py"
    prose = f"Please refer to the repository link: {long_url} for the implementation details."
    repaired = inject_soft_breaks(prose, max_token_len=30)
    assert "\u200b" in repaired
    # Short words should not be touched
    assert "refer" in repaired
    assert "repository" in repaired


def test_inject_soft_breaks_skips_math() -> None:
    math_prose = "The formula is $E = \\frac{m c^2}{\\sqrt{1 - v^2/c^2}}$ in relativity theory."
    repaired = inject_soft_breaks(math_prose, max_token_len=15)
    # Math delimiter $...$ should not be mangled
    assert "$E = \\frac{m c^2}{\\sqrt{1 - v^2/c^2}}$" in repaired


def test_document_renderer_code_block_integration(tmp_path: Path) -> None:
    code_text = (
        "def evaluate_transformer_layer(hidden_states, attention_mask, layer_past=None):\n"
        "    attn_output, past = attention(hidden_states, mask=attention_mask, past=layer_past)\n"
        "    return feed_forward(layer_norm(attn_output + hidden_states))"
    )
    block = IRBlock(
        element=make_element(
            id="c1",
            spine_index=1,
            block_type=BlockType.CODE,
            source_text=code_text,
            skip_translate=True,
        ),
        target_text=code_text,
    )
    renderer = TypstDocumentRenderer(columns=2)
    source, stats = renderer.build_source([block], tmp_path)

    assert stats.blocks_rendered == 1
    assert 'raw("' in source
    assert "#block(fill: luma(245)" in source


def test_compute_container_char_budget() -> None:
    # 100pt container, 1 line, floor 7.5pt:
    # ZH (alpha=1.0) -> floor(100 / 7.5) = 13 chars
    assert compute_container_char_budget(100.0, max_lines=1, floor_pt=7.5, target_lang="zh") == 13
    # 2 lines -> 26 chars
    assert compute_container_char_budget(100.0, max_lines=2, floor_pt=7.5, target_lang="zh") == 26
    # EN (alpha=0.48) -> floor(100 / (0.48 * 7.5)) = floor(100 / 3.6) = 27 chars
    assert compute_container_char_budget(100.0, max_lines=1, floor_pt=7.5, target_lang="en") == 27


def test_extract_locked_numbers() -> None:
    text = "在 2026 年，模型精度提高了 15.5%，达到 98.2% 的准确率（测试集共 1000 样本）。"
    nums = extract_locked_numbers(text)
    assert nums == ["2026", "15.5%", "98.2%", "1000"]


def test_verify_concise_candidate_wmt26_rules() -> None:
    spec = ContainerFitSpec(
        seg_id="b1",
        current_translation="在 2026 年，我们可以看到，注意力机制对于提高 98.5% 的精度至关重要。",
        max_target_chars=30,
        locked_glossary_terms=["注意力机制"],
        locked_numbers=["2026", "98.5%"],
    )

    # 1. Valid concise candidate within budget and keeping all locked terms
    good_cand = "2026年，注意力机制使精度达98.5%。"
    assert len(good_cand) <= 30
    assert verify_concise_candidate(spec, good_cand) is True

    # 2. Exceeds budget -> rejected
    long_cand = "在2026年这一时期内，注意力机制对全面提升系统98.5%的准确度具有关键意义。"
    assert len(long_cand) > 30
    assert verify_concise_candidate(spec, long_cand) is False

    # 3. Missing locked glossary term -> rejected
    missing_term = "2026年，该机制使精度达到98.5%。"
    assert verify_concise_candidate(spec, missing_term) is False

    # 4. Missing or corrupted locked number -> rejected
    corrupt_num = "2026年，注意力机制使精度达98%。"
    assert verify_concise_candidate(spec, corrupt_num) is False


def test_verify_concise_candidate_math_and_sentinels() -> None:
    spec = ContainerFitSpec(
        seg_id="b2",
        current_translation="根据公式 $E = m c^2$ 与标记 ⟦sentinel_001⟧，可以计算能量。",
        max_target_chars=40,
        locked_glossary_terms=[],
        locked_numbers=[],
    )

    # Valid candidate keeping inline math and sentinel
    good_cand = "据公式 $E = m c^2$ 与 ⟦sentinel_001⟧ 计算能量。"
    assert verify_concise_candidate(spec, good_cand) is True

    # Broken math span -> rejected
    bad_math = "据公式 $E = mc$ 与 ⟦sentinel_001⟧ 计算能量。"
    assert verify_concise_candidate(spec, bad_math) is False

    # Dropped sentinel -> rejected
    bad_sentinel = "据公式 $E = m c^2$ 可以计算能量。"
    assert verify_concise_candidate(spec, bad_sentinel) is False


def test_concise_rewrite_heuristic() -> None:
    text = "我们可以看到，由于这个原因，可以通过使用自注意力机制进行计算分析。"
    budget = 20
    condensed = concise_rewrite_heuristic(
        text,
        budget,
        locked_terms=["自注意力机制"],
        locked_numbers=[],
        target_lang="zh",
    )
    # Should replace "我们可以看到，", "由于这个原因，", "可以通过使用", "进行计算分析"
    assert "自注意力机制" in condensed
    assert len(condensed) < len(text)
    assert "可以通过使用" not in condensed


def test_repair_bottleneck_containers() -> None:
    specs = [
        ContainerFitSpec(
            seg_id="seg1",
            current_translation="我们可以看到，由于这个原因，系统提升了 42% 的吞吐量。",
            max_target_chars=25,
            locked_glossary_terms=["系统"],
            locked_numbers=["42%"],
        ),
    ]

    repaired = repair_bottleneck_containers(specs)
    assert "seg1" in repaired
    rep_text = repaired["seg1"]
    assert len(rep_text) <= 25
    assert "系统" in rep_text
    assert "42%" in rep_text
