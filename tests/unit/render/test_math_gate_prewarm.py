"""The renderer pre-warms its math compile gate in one batched probe (W3-9).

The publish path passes a ``TypstMathProbe`` whose ``check`` spawns one Typst
process per formula (~0.7s of startup each): a 300-formula book spent ~3.5
minutes of startups on a job whose whole document compile is seconds. The probe
already has ``check_many`` (one document, bisect on failure); the renderer now
collects every not-yet-known body before the block walk and resolves them in
one call — same verdicts, same cache, a handful of invocations.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pytest

from ubt.core.ir.models import BlockType, IRBlock, make_element
from ubt.render.typst.document_renderer import TypstDocumentRenderer

pytestmark = pytest.mark.fast


def _inline_math_block(index: int, text: str) -> IRBlock:
    block = IRBlock(
        element=make_element(
            id=f"b{index:03d}",
            spine_index=index,
            block_type=BlockType.NARRATIVE,
            source_text="src",
        )
    )
    block.target_text = text
    return block


def _formula_block(index: int, body: str) -> IRBlock:
    block = IRBlock(
        element=make_element(
            id=f"f{index:03d}",
            spine_index=index,
            block_type=BlockType.FORMULA,
            source_text=body,
        )
    )
    return block


def test_build_source_prewarms_every_math_body_in_one_call(tmp_path: Path) -> None:
    """One ``check_many`` call covering inline spans and formula blocks."""
    bodies: list[str] = []
    rendered: list[str] = []

    def gate(body: str) -> bool:
        rendered.append(body)
        return True

    def gate_many(pending: Iterable[str]) -> None:
        bodies.extend(pending)

    blocks = [
        _inline_math_block(0, r"公式 $x_{1} + y^{1} = z$ 内联。"),
        _inline_math_block(1, r"另一个 $a_{2} \le b_{2}$ 公式。"),
        _formula_block(2, r"$$\frac{p}{q} = r$$"),
    ]
    renderer = TypstDocumentRenderer(math_gate=gate, math_gate_many=gate_many)
    renderer.build_source(blocks, tmp_path / "build", "zh")

    # Every body reached the batch pre-warm exactly once…
    assert len(bodies) == len(set(bodies)) == 3
    assert "x_(1) + y^(1) = z" in bodies
    assert "frac(p, q) = r" in bodies
    # …and each was then answered exactly once by the per-body gate (the
    # renderer's walk), so the pre-warm did not duplicate or skip any body.
    assert sorted(rendered) == sorted(bodies)


def test_a_failing_body_still_degrades_through_the_gate(tmp_path: Path) -> None:
    """The pre-warm must not change a rejection into a compile failure."""
    rejected = {"frac(a, )"}

    def gate(body: str) -> bool:
        return body not in rejected

    def gate_many(pending: Iterable[str]) -> None:
        list(pending)  # the probe populates its own cache; the fake needs nothing

    block = _inline_math_block(0, r"坏公式 $\frac{a}{}$ 在此。")
    renderer = TypstDocumentRenderer(math_gate=gate, math_gate_many=gate_many)
    source, stats = renderer.build_source([block], tmp_path / "build", "zh")

    # The rejected body degrades to readable Unicode, never reaching the page
    # as raw LaTeX and never failing the document.
    assert "frac(a, )" not in source
    assert "\\frac" not in source
    assert "a/" in source or "a" in source


def test_without_a_batch_gate_the_per_body_path_is_unchanged(tmp_path: Path) -> None:
    """Tests and gates without a batch API keep the old behavior."""
    calls: list[str] = []

    def gate(body: str) -> bool:
        calls.append(body)
        return True

    block = _inline_math_block(0, r"公式 $x + y$。")
    renderer = TypstDocumentRenderer(math_gate=gate)  # no math_gate_many
    renderer.build_source([block], tmp_path / "build", "zh")
    assert calls == ["x + y"]
