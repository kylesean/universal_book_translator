"""Unit tests for ContainerFitRepair (P2-16)."""

from __future__ import annotations

from pathlib import Path

import pytest

from ubt.core.ir.models import BlockType, IRBlock, make_element
from ubt.render.typst.container_repair import (
    effective_char_width,
    fit_code_block,
    inject_soft_breaks,
    measure_line_width,
    repair_code_block_markup,
    smart_wrap_code_line,
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
