"""Unit tests for the engine backend (MathJax -> SVG).

Covers the renderer wrapper's degrade paths and the reconstructor's engine
pass: dispatch, tag/number emission, witness fallback, and the automatic
fallback to the legacy Typst backend when Node/MathJax is unavailable.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from ubt.adapters.pdf import math_renderer as mr
from ubt.adapters.pdf.math_renderer import MathjaxRenderer, MathRender
from ubt.adapters.pdf.typst_reconstructor import TypstReconstructor
from ubt.core.ir.models import BlockType, BoundingBox, IRBlock


def _block(bid: str = "pdf_main#b0032") -> IRBlock:
    return IRBlock(
        id=bid,
        spine_index=1,
        block_type=BlockType.FORMULA,
        source_text=(
            r"\psi _ { 1 } ( x , y ) = \psi _ { 0 } ( y ) - 2 V _ { m } \ln"
            r" \left [ \cos ( x ) \right ] \quad (3.5)"
        ),
        bbox=BoundingBox(page=1, x0=10, y0=10, x1=210, y1=40),
    )


def _png_bytes(draw_formula: bool) -> bytes:
    img = Image.new("L", (300, 60), 255)
    if draw_formula:
        d = ImageDraw.Draw(img)
        for i in range(10):
            x = 12 + i * 26
            d.line((x, 18, x, 42), fill=0, width=2)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class _FakeRenderer:
    def __init__(self, result: MathRender | None = None) -> None:
        self.result = result or MathRender(ok=True, svg="<svg/>", png=None)
        self.calls: list[dict[str, object]] = []

    def render(self, latex: str, **kwargs: object) -> MathRender:
        self.calls.append({"latex": latex, **kwargs})
        return self.result


class TestRendererDegrade:
    def test_missing_script_is_unavailable(self, tmp_path: Path) -> None:
        renderer = MathjaxRenderer(script=tmp_path / "missing.mjs")
        assert renderer.available() is False
        outcome = renderer.render("x = 1")
        assert outcome.ok is False
        assert outcome.error is not None

    def test_cache_roundtrip(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(mr, "MATH_CACHE_DIR", tmp_path / "cache")
        renderer = MathjaxRenderer(script=tmp_path / "missing.mjs")
        result = MathRender(ok=True, svg="<svg><!--x--></svg>", png=b"png", width=10.0, height=5.0)
        renderer._write_cache("x = 1", "3.5", 300, True, result)
        cached = renderer._read_cache("x = 1", "3.5", 300, True)
        assert cached is not None
        assert cached.svg == result.svg
        assert cached.png == b"png"
        assert renderer._read_cache("x = 2", "3.5", 300, True) is None


class TestEnginePass:
    def test_engine_emits_svg_image_with_number(self, monkeypatch: pytest.MonkeyPatch) -> None:
        rec = TypstReconstructor()
        rec.math_backend = "mathjax"
        rec.source_pdf = None
        renderer = _FakeRenderer(MathRender(ok=True, svg="<svg/>"))
        block = _block()
        lines = [f"$ T = 1 $  // [formula {block.id}]"]
        swapped = rec._substitute_formulas_with_engine(lines, [block], renderer)
        assert swapped == 1
        assert '.svg"' in lines[0]
        assert "(3.5)" in lines[0]
        assert "#counter(math.equation).step()" in lines[0]

    def test_engine_uses_chapter_sequence_for_untagged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rec = TypstReconstructor()
        rec.source_pdf = None
        renderer = _FakeRenderer(MathRender(ok=True, svg="<svg/>"))
        block = IRBlock(
            id="pdf_main#b9001",
            spine_index=2,
            block_type=BlockType.FORMULA,
            source_text=r"E = m c ^ { 2 }",
            bbox=BoundingBox(page=1, x0=10, y0=10, x1=110, y1=40),
        )
        lines = ["= 第三章 核心模型", f"$ E = m c^2 $  // [formula {block.id}]"]
        rec._substitute_formulas_with_engine(lines, [block], renderer)
        assert "(3.1)" in lines[1]

    def test_render_failure_falls_back_to_source_graphic(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rec = TypstReconstructor()
        rec.source_pdf = None
        monkeypatch.setattr(rec, "_crop_formula_fallback", lambda block: f"#image({block.id}.png)")
        renderer = _FakeRenderer(MathRender(ok=False, error="Missing close brace"))
        block = _block()
        lines = [f"$ x = y $  // [formula {block.id}]"]
        swapped = rec._substitute_formulas_with_engine(lines, [block], renderer)
        assert swapped == 0
        assert lines[0] == f"#image({block.id}.png)"
        assert any("engine render failed" in f for f in rec.last_witness_findings)

    def test_witness_mismatch_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        rec = TypstReconstructor()
        rec.source_pdf = None
        monkeypatch.setattr(rec, "_crop_formula_fallback", lambda block: f"#image({block.id}.png)")
        monkeypatch.setattr(
            rec,
            "_engine_findings",
            lambda result, source, compare: ["aspect 9.00x of source"],
        )
        renderer = _FakeRenderer(MathRender(ok=True, svg="<svg/>", png=b"png"))
        block = _block()
        lines = [f"$ x = y $  // [formula {block.id}]"]
        swapped = rec._substitute_formulas_with_engine(lines, [block], renderer)
        assert swapped == 0
        assert lines[0] == f"#image({block.id}.png)"
        assert any("engine witness" in f for f in rec.last_witness_findings)


class TestEngineFindings:
    def test_matching_structures_pass(self) -> None:
        from ubt.adapters.pdf.formula_witness import compare_structure

        img = Image.new("L", (80, 40), 255)
        assert TypstReconstructor._engine_findings(
            MathRender(ok=True, png=_png_bytes(True)), img, compare_structure
        )

    def test_identical_rasters_pass(self) -> None:
        from ubt.adapters.pdf.formula_witness import compare_structure

        img = Image.open(io.BytesIO(_png_bytes(True))).convert("L")
        findings = TypstReconstructor._engine_findings(
            MathRender(ok=True, png=_png_bytes(True)), img, compare_structure
        )
        assert findings == []

    def test_missing_raster_is_unverifiable_not_failing(self) -> None:
        from ubt.adapters.pdf.formula_witness import compare_structure

        img = Image.new("L", (80, 40), 255)
        assert (
            TypstReconstructor._engine_findings(
                MathRender(ok=True, png=None), img, compare_structure
            )
            == []
        )


@pytest.mark.skipif(
    not MathjaxRenderer().available(), reason="Node + scripts/mathjax deps not installed"
)
def test_real_mathjax_renderer_smoke() -> None:
    renderer = MathjaxRenderer()
    try:
        result = renderer.render(r"E = m c ^ { 2 }", tag="3.1", png_width=200)
    finally:
        renderer.close()
    assert result.ok
    assert result.svg and result.svg.startswith("<svg")
    assert result.png is not None
    assert renderer.version()


class TestBackendDispatch:
    def _blocks(self) -> list[IRBlock]:
        heading = IRBlock(
            id="h3",
            spine_index=0,
            block_type=BlockType.HEADING,
            source_text="第三章 核心模型",
            target_text="第三章 核心模型",
        )
        return [heading, _block()]

    def test_mathjax_backend_replaces_display_formula(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rec = TypstReconstructor()
        rec.math_backend = "mathjax"
        fake = _FakeRenderer(MathRender(ok=True, svg="<svg/>"))
        monkeypatch.setattr(rec, "_engine", lambda: fake)
        code = rec.generate_typst_source(self._blocks(), bilingual=False, page_strict=False)
        assert '.svg"' in code
        assert "(3.5)" in code

    def test_mathjax_backend_degrades_without_engine(self, monkeypatch: pytest.MonkeyPatch) -> None:
        rec = TypstReconstructor()
        rec.math_backend = "mathjax"
        rec.formula_render = "native"
        monkeypatch.setattr(rec, "_engine", lambda: None)
        code = rec.generate_typst_source(self._blocks(), bilingual=False, page_strict=False)
        assert "$" in code  # legacy Typst math kept
        assert 'math.equation(block: true, numbering: _ => "(3.5)")' in code

    def test_image_backend_overrides_formula_render(self, monkeypatch: pytest.MonkeyPatch) -> None:
        rec = TypstReconstructor()
        rec.math_backend = "image"
        rec.formula_render = "native"
        monkeypatch.setattr(
            rec, "_crop_formula_fallback", lambda block: f'#image("{block.id}.png")'
        )
        code = rec.generate_typst_source(self._blocks(), bilingual=False, page_strict=False)
        assert f'#image("{_block().id}.png")' in code


class TestConfig:
    def test_default_backend_is_mathjax(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from pydantic import SecretStr

        from ubt.core.config import UBTConfig

        monkeypatch.delenv("UBT_MATH_BACKEND", raising=False)
        assert UBTConfig.from_env(api_key=SecretStr("mock-key")).math_backend == "mathjax"

    def test_env_override_is_lowercased(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from pydantic import SecretStr

        from ubt.core.config import UBTConfig

        monkeypatch.setenv("UBT_MATH_BACKEND", "IMAGE")
        assert UBTConfig.from_env(api_key=SecretStr("mock-key")).math_backend == "image"

    def test_invalid_backend_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from pydantic import SecretStr, ValidationError

        from ubt.core.config import UBTConfig

        monkeypatch.setenv("UBT_MATH_BACKEND", "tex")
        with pytest.raises(ValidationError):
            UBTConfig.from_env(api_key=SecretStr("mock-key"))


class TestInlineEngine:
    def test_prose_callback_only_sees_source_math(self) -> None:
        from ubt.adapters.pdf.typst_fragments import _prose_to_typst

        seen: list[str] = []

        def _cb(latex: str) -> str | None:
            seen.append(latex)
            return '#image("inline.svg")' if "psi" in latex else None

        text = "势$\\psi_{pert}$与 TFIN = 20 nm 的关系"
        out = _prose_to_typst(text, inline_math=_cb)
        assert seen == [r"\psi_{pert}"]
        assert '#image("inline.svg")' in out
        # The polish-synthesized span (TFIN = 20 nm) keeps the legacy form.
        assert "$T" in out and "nm" in out

    def test_prose_without_callback_is_unchanged(self) -> None:
        from ubt.adapters.pdf.typst_fragments import _prose_to_typst

        text = "势$\\psi_{pert}$的关系"
        assert "$" in _prose_to_typst(text)

    def test_inline_svg_uses_baseline_box(self, monkeypatch: pytest.MonkeyPatch) -> None:
        rec = TypstReconstructor()
        svg = '<svg style="vertical-align: -0.650ex" viewBox="0 0 2000 1000"></svg>'
        fake = _FakeRenderer(MathRender(ok=True, svg=svg, width=2000.0, height=1000.0))
        out = rec._inline_svg(r"\psi", fake)
        assert out is not None
        assert "baseline: 0.325em" in out
        assert "ex" not in out  # Typst has no ex unit
        assert "height: 1.000em" in out

    def test_inline_svg_unitless_zero_still_gets_a_box(self) -> None:
        # MathJax writes "vertical-align: 0;" (no unit) for an exactly
        # baseline-aligned span, e.g. a lone capital letter. A bare #image
        # would be block-level and force an unwanted line break.
        rec = TypstReconstructor()
        svg = '<svg style="vertical-align: 0;" viewBox="0 -683 2845 683"></svg>'
        fake = _FakeRenderer(MathRender(ok=True, svg=svg, width=2845.0, height=1366.0))
        out = rec._inline_svg(r"T", fake)
        assert out is not None
        assert out.startswith("#box(baseline: 0.000em)[#image(")

    def test_inline_svg_without_vertical_align_is_still_inline(self) -> None:
        rec = TypstReconstructor()
        svg = '<svg viewBox="0 -683 2845 683"></svg>'
        fake = _FakeRenderer(MathRender(ok=True, svg=svg, width=2845.0, height=1366.0))
        out = rec._inline_svg(r"T", fake)
        assert out is not None
        assert out.startswith("#box[#image(")

    def test_inline_svg_returns_none_on_failure(self) -> None:
        rec = TypstReconstructor()
        fake = _FakeRenderer(MathRender(ok=False, error="boom"))
        assert rec._inline_svg(r"\psi", fake) is None

    def test_inline_renderer_is_none_for_other_backends(self) -> None:
        rec = TypstReconstructor()
        rec.math_backend = "typst"
        assert rec._inline_math_renderer() is None
