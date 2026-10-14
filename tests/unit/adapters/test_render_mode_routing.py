"""The PDF render path is publish-only, and stays that way.

`overlay` was removed because it cannot produce a uniform body size, translated
table headers or native math — the three things the semantic reflow renderer
exists to deliver. What remains worth pinning is not the switch (there is no
switch) but the invariants around the single renderer:

- the semantic reflow path is what a PDF target reaches, unconditionally;
- a non-PDF target still gets the Markdown writer, not the renderer;
- a compile failure raises rather than emitting a partial file;
- a missing asset is reported instead of silently dropped.

The last two are the failure modes a reader cannot see: an empty PDF reads as
success downstream, and a dropped figure looks like a shorter paper.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from ubt.adapters.pdf.docling_adapter import DoclingPDFAdapter
from ubt.adapters.pdf.docling_render import DoclingRenderStrategy
from ubt.core.ir.models import BlockType, BookManifest, IRBlock, make_element
from ubt.core.ports import AdapterRuntimeConfig

pytestmark = pytest.mark.fast


def _runtime_config(**overrides: object) -> AdapterRuntimeConfig:
    base: dict[str, object] = {
        "ocr_mode": "off",
        "ocr_endpoint": None,
        "ocr_api_key": "",
        "ocr_model": "",
        "formula_enrichment": "off",
        "docling_device": "cpu",
        "font_family": None,
        "allow_page_upload": False,
    }
    base.update(overrides)
    return AdapterRuntimeConfig(**base)  # type: ignore[arg-type]


def _block(block_id: str, block_type: BlockType, source: str, target: str) -> IRBlock:
    return IRBlock(
        element=make_element(id=block_id, spine_index=1, block_type=block_type, source_text=source),
        target_text=target,
    )


# --- the single render path ------------------------------------------------


def test_the_strategy_no_longer_carries_a_render_mode() -> None:
    """`overlay` is gone, not merely defaulted away.

    A leftover `render_mode` attribute would let a stale config or an old test
    keep selecting a renderer that no longer exists — the failure would be a
    silently different artifact, which is the worst kind.
    """
    strategy = DoclingRenderStrategy()
    assert not hasattr(strategy, "render_mode")
    assert not hasattr(strategy, "_publish_requested")


def test_a_pdf_target_always_reaches_the_reflow_renderer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A PDF target must route to `_render_publish` with no flag to set.

    This replaces the old "the switch actually switches" test: with one
    renderer there is no switch, so the invariant is that the PDF branch is
    unconditional.
    """
    strategy = DoclingRenderStrategy()
    seen: dict[str, object] = {}

    def fake_publish_sync(
        manifest: BookManifest,
        blocks: list[IRBlock],
        target_lang: str,
        output_path: Path,
        bilingual: bool = False,
    ) -> Path:
        seen["bilingual"] = bilingual
        seen["output"] = output_path
        return output_path

    monkeypatch.setattr(strategy, "_render_publish_sync", fake_publish_sync)
    manifest = BookManifest(doc_id="d", title="T", source_path=str(tmp_path / "x.pdf"), chapters=[])
    out = tmp_path / "out.pdf"
    asyncio.run(
        strategy.render_blocks(
            manifest=manifest,
            blocks=[_block("b1", BlockType.NARRATIVE, "Source.", "译文。")],
            target_lang="zh",
            output_path=out,
        )
    )
    assert seen["output"] == out


def test_the_adapter_exposes_no_render_mode_knob() -> None:
    """The adapter must not offer a knob the engine cannot honour."""
    adapter = DoclingPDFAdapter()
    assert not hasattr(adapter, "render_mode")
    adapter.apply_config(_runtime_config())
    assert not hasattr(adapter, "render_mode")


def test_bilingual_is_decided_by_the_dual_mode_not_a_render_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`dual` still works: pairing is driven by the bilingual mode alone.

    Removing `render_mode` must not remove bilingual support — that is the
    behaviour the overlay path used to own and the reflow renderer now has to
    provide.
    """
    from ubt.core.ir.render_plan import RenderPlan

    strategy = DoclingRenderStrategy()
    seen: dict[str, object] = {}
    monkeypatch.setattr(
        strategy,
        "_render_publish_sync",
        lambda m, b, t, o, bilingual=False: seen.update(bilingual=bilingual) or o,
    )
    manifest = BookManifest(doc_id="d", title="T", source_path=str(tmp_path / "x.pdf"), chapters=[])
    blocks = [_block("b1", BlockType.NARRATIVE, "Source.", "译文。")]

    asyncio.run(
        strategy.render_blocks(
            manifest=manifest,
            blocks=blocks,
            target_lang="zh",
            output_path=tmp_path / "mono.pdf",
            render_plan=RenderPlan(bilingual_mode="monolingual", effective_dual_mode="mono"),
        )
    )
    assert seen["bilingual"] is False

    asyncio.run(
        strategy.render_blocks(
            manifest=manifest,
            blocks=blocks,
            target_lang="zh",
            output_path=tmp_path / "dual.pdf",
            render_plan=RenderPlan(bilingual_mode="bilingual", effective_dual_mode="dual"),
        )
    )
    assert seen["bilingual"] is True


# --- the non-PDF path ------------------------------------------------------


def test_a_non_pdf_target_uses_the_markdown_writer(tmp_path: Path) -> None:
    """Markdown has its own writer; the PDF renderer must not intercept it."""
    strategy = DoclingRenderStrategy()
    blocks = [_block("b1", BlockType.NARRATIVE, "Source.", "译文。")]
    manifest = BookManifest(doc_id="d", title="T", source_path=str(tmp_path / "x.pdf"), chapters=[])
    out = tmp_path / "out.md"

    asyncio.run(
        strategy.render_blocks(manifest=manifest, blocks=blocks, target_lang="zh", output_path=out)
    )
    assert out.exists()
    assert "译文" in out.read_text(encoding="utf-8")


# --- failure modes a reader cannot see -------------------------------------


def test_a_compile_failure_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing PDF is a clearer outcome than an empty one.

    The failure path must not leave a zero-byte artifact behind: the caller
    asked for a rebuilt document, and an empty file would read as success to
    everything downstream.
    """
    strategy = DoclingRenderStrategy()

    import ubt.adapters.pdf.typst_compile as tc

    monkeypatch.setattr(tc, "typst_compile", lambda *_a, **_kw: (False, "syntax error"))

    manifest = BookManifest(doc_id="d", title="T", source_path=str(tmp_path / "x.pdf"), chapters=[])
    with pytest.raises(Exception, match="publish render failed to compile"):
        strategy._render_publish_sync(
            manifest,
            [_block("b1", BlockType.NARRATIVE, "Source.", "译文。")],
            "zh",
            tmp_path / "out.pdf",
        )
    assert not (tmp_path / "out.pdf").exists()
    # The generated source is kept so the failure is inspectable.
    assert (tmp_path / ".out_typst" / "main.typ").exists()


def test_the_missing_asset_channel_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dropped figure must be visible in the quality report, not just absent."""
    strategy = DoclingRenderStrategy()

    import ubt.adapters.pdf.typst_compile as tc

    monkeypatch.setattr(tc, "typst_compile", lambda *_a, **_kw: (True, ""))
    manifest = BookManifest(doc_id="d", title="T", source_path=str(tmp_path / "x.pdf"), chapters=[])
    strategy._render_publish_sync(
        manifest,
        [_block("i1", BlockType.IMAGE, "/nonexistent/fig.png", "")],
        "zh",
        tmp_path / "out.pdf",
    )
    assert strategy.last_render_skips
    assert any("asset_not_found" in reason for _asset, reason in strategy.last_render_skips)


def test_the_publish_path_applies_the_math_compile_gate(tmp_path: Path) -> None:
    """P1-C5: a math body the transpiler accepts but Typst rejects must not
    reach the compiler.

    ``\\frac{a}{}`` converts to ``frac(a, )``, which the transpiler emits
    happily and Typst then refuses — failing the *whole* document at the single
    compile. The publish render supplies ``TypstMathProbe().check`` as the
    renderer's ``math_gate`` so such a span degrades to readable Unicode
    instead. Without the gate the bad body stays in ``main.typ``, so this pins
    the wiring by inspecting the generated source. The real Typst binary is used
    (the probe compiles its witness), so the assertion is the compiler's own
    verdict, not a stub's.
    """
    pytest.importorskip("ubt.adapters.pdf.typst_compile")
    from ubt.adapters.pdf.typst_compile import typst_version

    if typst_version() is None:  # pragma: no cover - environment without typst
        pytest.skip("typst binary not available")

    strategy = DoclingRenderStrategy()
    manifest = BookManifest(doc_id="d", title="T", source_path=str(tmp_path / "x.pdf"), chapters=[])
    strategy._render_publish_sync(
        manifest,
        [_block("m1", BlockType.NARRATIVE, "s", r"Bad $\frac{a}{}$ here.")],
        "zh",
        tmp_path / "out.pdf",
    )
    generated = (tmp_path / ".out_typst" / "main.typ").read_text(encoding="utf-8")
    assert "frac(a, )" not in generated
    assert "Bad" in generated  # the prose is still there, just the body degraded
