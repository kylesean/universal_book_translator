"""Render strategy for the Docling adapter.

Turns translated ``IRBlock``s into a delivered artifact: Markdown export,
three-layer absolute composition via LayerCompositor, and alternating/facing
bilingual interleaving. Holds only delivery collaborators (alternator, diagram localizer,
font family), so the adapter can construct it once and forward ``render_blocks`` to it.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from ubt.adapters.pdf.alternator import BilingualAlternator
from ubt.adapters.pdf.diagram_localizer import DiagramLocalizer
from ubt.adapters.pdf.typst_compile import typst_version
from ubt.core.config import PAGE_BILINGUAL_MODES, canonical_render_engine
from ubt.core.exceptions import DocumentParseError
from ubt.core.ir.models import BlockType, BookManifest, IRBlock
from ubt.core.ir.render_plan import RenderOutcome, RenderPlan
from ubt.core.policy.adaptive_policy import resolve_pdf_engine
from ubt.model.fidelity import Fidelity
from ubt.model.span import PhysicalBox

if TYPE_CHECKING:
    from ubt.render.outputs import Overlay

logger = logging.getLogger(__name__)


def record_toolchain_versions(manifest: BookManifest) -> None:
    """Persist the Typst compiler pin into manifest metadata.

    The generated markup's meaning depends on compiler behavior, so every
    delivered PDF's quality report carries the exact compiler version
    (None = unmeasured — never fabricated).
    """
    try:
        version = typst_version()
    except Exception:
        version = None
    if isinstance(version, str) and version:
        manifest.metadata["typst_version"] = version


def _reflow_obstacles(blocks: Sequence[IRBlock]) -> list[PhysicalBox]:
    """Boxes the page reflow must not cross: everything it does not draw itself.

    Figures, tables, code, formulas and any block held byte-identical are fixed
    furniture on the source canvas; a reflowed paragraph that crossed one would
    overlap it. Their boxes bound the bands.
    """
    obstacles: list[PhysicalBox] = []
    for block in blocks:
        box = block.bbox
        if box is None or box.page <= 0:
            continue
        if block.skip_translate or block.block_type in (
            BlockType.TABLE,
            BlockType.IMAGE,
            BlockType.CODE,
            BlockType.FORMULA,
        ):
            obstacles.append(PhysicalBox.of(box.page, (box.x0, box.y0, box.x1, box.y1)))
    return obstacles


def _warn_forced_engine(
    requested_engine: str,
    active_engine: str,
    blocks: list[IRBlock],
    manifest: BookManifest | None = None,
) -> None:
    """Warn when a forced render engine disagrees with what auto dispatch
    would have chosen for this document's density profile.

    A forced choice is legitimate (the user knows their fallback), but it
    silently bypasses the structure-density routing that protects figure/
    table-heavy documents; the disagreement is worth one WARNING at the
    moment the choice is made rather than a post-mortem.
    """
    if canonical_render_engine(requested_engine) == "auto":
        return
    metadata = getattr(manifest, "metadata", None)
    if isinstance(metadata, dict) and metadata.get("suppress_render_engine_warning"):
        # Companion second pass: the primary already took the auto route, so a
        # forced-engine warning here would contradict the delivered artifact.
        return
    auto_choice = resolve_pdf_engine("auto", blocks, manifest=manifest)
    if auto_choice != active_engine:
        logger.warning(
            "render_engine=%r was forced, but auto dispatch would route this "
            "document to %r (formula/structure density). If the output loses "
            "figures/tables/equations, re-run with --render-engine %s.",
            requested_engine,
            auto_choice,
            auto_choice,
        )


class DoclingRenderStrategy:
    """Delivery-leg collaborators plus the last render's skip side channel."""

    def __init__(
        self,
        *,
        alternator: BilingualAlternator | None = None,
        diagram_localizer: DiagramLocalizer | None = None,
        font_family: str | None = None,
    ) -> None:
        self.alternator = alternator or BilingualAlternator()
        self.diagram_localizer = diagram_localizer or DiagramLocalizer()
        self.font_family = font_family
        # Render skip side channel: plain (block_id, reason) pairs
        # from the most recent render_blocks call. Reset every render.
        self.last_render_skips: list[tuple[str, str]] = []
        # Render outcome side channel (compiler render plan protocol): the mode the
        # renderer actually used.
        self.last_outcome: RenderOutcome | None = None

    async def _render_composite(
        self,
        manifest: BookManifest,
        blocks: list[IRBlock],
        target_lang: str,
        output_path: Path,
        realization_plan: Mapping[str, Fidelity] | None,
        bilingual: bool = False,
    ) -> Path:
        """Compose the source page with translated fragments (LayerCompositor).

        Text above the fidelity floor is overlaid; an element whose fragment
        cannot be typeset keeps its source (recorded as a render skip), so the
        mask is never painted over content it cannot replace. The source PDF is
        the canvas, so non-text elements stay pixel-intact. ``bilingual`` carries
        each overlay's source so the fragment renders target over source.
        """
        from ubt.render.outputs import (
            LayerCompositor,
            TypstFragmentTypesetter,
            overlays_from_blocks,
        )
        from ubt.render.reflow import reflow_overlays

        source_pdf = str(getattr(manifest, "source_path", "") or "")
        if not source_pdf or not Path(source_pdf).exists():
            # No source canvas to compose onto; refuse rather than emit an empty artifact.
            raise DocumentParseError(
                "PDF render needs the source PDF as its canvas "
                f"(manifest.source_path={source_pdf!r})"
            )

        overlays = overlays_from_blocks(list(blocks), realization_plan, bilingual=bilingual)

        font = getattr(self, "font_family", None)
        from ubt.layout.theme import resolve_theme

        theme = resolve_theme(target_lang=target_lang)
        if not font:
            font = theme.font_stack
        profile = getattr(manifest, "profile", "") or ""
        base_size = 9.96 if profile == "paper" else theme.base_size_pt
        typesetter = TypstFragmentTypesetter(font=font, size_pt=base_size, target_lang=target_lang)
        # Repack the page's prose into its bands so a shorter CJK target hugs the
        # source instead of leaving the source box's slack as inter-paragraph
        # gaps. The measure is batched (one Typst invocation), and a band that
        # does not fit keeps its source geometry.
        overlays = reflow_overlays(
            overlays,
            _reflow_obstacles(list(blocks)),
            measure_many=typesetter.measure_many_fixed,
            cap_size=typesetter.cap_size,
        )
        compositor = LayerCompositor(source_pdf, typesetter=typesetter)
        try:
            composition = await asyncio.to_thread(compositor.compose, overlays, output_path)
        finally:
            typesetter.close()
        self.last_render_skips = [
            (placement.element_id, placement.detail)
            for placement in composition.placements
            if placement.descended
        ]
        self._relocate_link_annotations(source_pdf, composition.output_path, overlays, target_lang)
        record_toolchain_versions(manifest)
        return composition.output_path

    def _relocate_link_annotations(
        self, source_pdf: str, output_path: Path, overlays: Sequence[Overlay], target_lang: str
    ) -> None:
        """Move link annotations under replaced regions onto the translated glyphs.

        Layer 0 keeps the source page whole, so a link's ``/Rect`` still points at
        the source glyph positions; after the text under it is replaced the link
        has to follow the translated glyphs (or be pruned) or it becomes a ghost
        click zone. Best-effort and non-fatal.
        """
        import pikepdf

        from ubt.adapters.pdf.link_annotations import relocate_page_annotations
        from ubt.core.language_profile import resolve_font_config

        font_config = resolve_font_config(target_lang)
        strip_by_page: dict[int, list[tuple[float, float, float, float]]] = {}
        for overlay in overlays:
            # A reflowed overlay draws elsewhere but the source link still sits
            # under the *original* box, so detection uses the mask geometry.
            for box in overlay.mask_flow_boxes:
                strip_by_page.setdefault(box.page, []).append(box.bbox)
        if not strip_by_page:
            return
        try:
            with pikepdf.open(output_path, allow_overwriting_input=True) as pdf:
                mutated = 0
                for page_no in sorted(strip_by_page):
                    if page_no > len(pdf.pages):
                        continue
                    mutated += relocate_page_annotations(
                        page=pdf.pages[page_no - 1],
                        page_no=page_no,
                        source_pdf=source_pdf,
                        overlay_path=str(output_path),
                        strip_rects=strip_by_page[page_no],
                        overlay_page_no=page_no - 1,
                        figure_prefix=font_config.figure_prefix,
                        table_prefix=font_config.table_prefix,
                    )
                if mutated:
                    pdf.save(str(output_path))
        except Exception as exc:  # a link pass must never break the render
            logger.warning("Link annotation relocation failed (non-fatal): %s", exc)

    async def _interleave_source_and_target(
        self,
        manifest: BookManifest,
        composed_path: Path,
        out_path: Path,
        mode: str,
        facing_spread: bool,
    ) -> Path:
        """Zip source pages with the composed (translated) pages, 1:1.

        A source-canvas composition carries every source page whole, so it is
        page-aligned with the source; the two interleave exactly as a page-strict
        output does. A page-count mismatch is refused rather than silently shipping
        a misaligned bilingual book.
        """
        from ubt.adapters.pdf import pdf_struct as _pdf_struct

        source = Path(str(getattr(manifest, "source_path", "") or ""))
        if not source.exists():
            return composed_path
        source_pages = len(_pdf_struct.page_sizes(source))
        composed_pages = len(_pdf_struct.page_sizes(composed_path))
        if composed_pages != source_pages:
            raise DocumentParseError(
                f"Page-paired {mode} render produced {composed_pages} page(s) for a "
                f"{source_pages}-page source; refusing to interleave a misaligned "
                "bilingual book."
            )
        use_facing = mode in ("facing", "facing_spread") or facing_spread
        interleaved = await self.alternator.interleave_pages_async(
            source, composed_path, out_path, facing_spread=use_facing
        )
        if isinstance(getattr(manifest, "metadata", None), dict):
            manifest.metadata["render_padding_pages"] = list(interleaved.padding_pages) or None
        return out_path

    async def render_blocks(
        self,
        manifest: BookManifest,
        blocks: list[IRBlock],
        target_lang: str,
        output_path: Path,
        bilingual_mode: str | None = None,
        render_engine: str | None = None,
        render_plan: RenderPlan | None = None,
        realization_plan: Mapping[str, Fidelity] | None = None,
    ) -> Path:
        """Render publication-grade translated output from pre-fetched blocks.

        All PDF targets are rendered through LayerCompositor (the unified
        composition engine). Supports automatic facing-page bilingual
        interleaving and in-place bilingual fragments. Non-PDF targets
        (.md / .txt) export clean Markdown.
        """
        self.last_render_skips = []
        self.last_outcome = None
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        plan_mode = render_plan.bilingual_mode if render_plan is not None else None
        plan_engine = render_plan.render_engine if render_plan is not None else None
        active_mode = bilingual_mode or str(plan_mode or "bilingual")
        requested_engine = render_engine or str(plan_engine or "publication")
        is_pdf_target = out_path.suffix.lower() == ".pdf"
        active_engine = (
            resolve_pdf_engine(requested_engine, blocks, manifest=manifest)
            if is_pdf_target
            else "publication"
        )

        if is_pdf_target and blocks:
            _warn_forced_engine(requested_engine, active_engine, blocks, manifest=manifest)

        prior_downgrade = render_plan.dual_mode_downgraded if render_plan is not None else None
        outcome = RenderOutcome(
            bilingual_mode=active_mode,
            effective_dual_mode=(
                render_plan.effective_dual_mode if render_plan is not None else None
            ),
            dual_mode_downgraded=prior_downgrade,
        )
        self.last_outcome = outcome
        if isinstance(getattr(manifest, "metadata", None), dict):
            manifest.metadata["render_engine_effective"] = active_engine

        facing_spread_plan = render_plan.facing_spread if render_plan is not None else False

        if is_pdf_target:
            if active_mode in PAGE_BILINGUAL_MODES:
                staged = out_path.with_name(f"{out_path.stem}_trans_stage{out_path.suffix}")
                try:
                    composed = await self._render_composite(
                        manifest,
                        list(blocks),
                        target_lang,
                        staged,
                        realization_plan,
                        bilingual=False,
                    )
                    return await self._interleave_source_and_target(
                        manifest, composed, out_path, active_mode, bool(facing_spread_plan)
                    )
                finally:
                    staged.unlink(missing_ok=True)
            return await self._render_composite(
                manifest,
                list(blocks),
                target_lang,
                out_path,
                realization_plan,
                bilingual=active_mode == "bilingual",
            )

        # Non-PDF export: Markdown (.md / .markdown) or flat text (.txt)
        if out_path.suffix.lower() in (".md", ".markdown", ".txt"):
            md_lines: list[str] = [f"# {manifest.title} ({target_lang})\n\n"]
            for block in blocks:
                source = block.source_text
                target = block.target_text or block.draft_text or source
                if block.block_type == BlockType.HEADING:
                    md_lines.append(f"## {target}\n\n")
                elif block.block_type == BlockType.CODE:
                    md_lines.append(f"```\n{target}\n```\n\n")
                elif block.block_type == BlockType.FORMULA:
                    md_lines.append(f"$$\n{target}\n$$\n\n")
                else:
                    if (
                        active_mode in ("alternating", "bilingual")
                        and source
                        and target
                        and source != target
                    ):
                        md_lines.append(f"> {source}\n\n{target}\n\n")
                    else:
                        md_lines.append(f"{target}\n\n")
            out_path.write_text("".join(md_lines), encoding="utf-8")
            return out_path

        raise DocumentParseError(f"Unsupported output format: {out_path.suffix}")
