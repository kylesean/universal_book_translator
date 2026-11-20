"""Render strategy for the Docling adapter (scanned PDFs & Typst reflow).

Turns translated ``IRBlock``s into a delivered artifact: a reflowable PDF rebuilt
from the IR and laid out by Typst, or a Markdown/flat-text export.

Born-digital 1:1 in-place PDF translation is owned by :mod:`ubt.codecs.pdf`
(``NativePDFAdapter``); the legacy Typst 1:1 facsimile / page-locked overlay
subsystem has been physically retired.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
from pathlib import Path
from typing import Any

from ubt.adapters.pdf.typst_compile import typst_version
from ubt.core.content.nodes import ReconstructionRecord
from ubt.core.exceptions import DocumentParseError
from ubt.core.ir.models import BlockType, BookManifest, IRBlock
from ubt.core.ir.render_plan import RenderOutcome, RenderPlan

logger = logging.getLogger(__name__)


def record_toolchain_versions(manifest: BookManifest) -> None:
    """Persist the Typst compiler pin into manifest metadata."""
    try:
        version = typst_version()
    except Exception:
        version = None
    if isinstance(version, str) and version:
        manifest.metadata["typst_version"] = version


_KEEP_BUILD_DIR_ENV = "UBT_KEEP_BUILD_DIR"
_TRUTHY = frozenset({"1", "true", "yes", "on"})


def _cleanup_build_dir(build_dir: Path) -> None:
    """Delete a render's scratch directory after the PDF has been written."""
    if os.environ.get(_KEEP_BUILD_DIR_ENV, "").strip().lower() in _TRUTHY:
        return
    if not build_dir.exists():
        return
    try:
        shutil.rmtree(build_dir)
    except OSError as exc:  # never fail a delivered book over scratch space
        logger.warning("could not remove render scratch dir %s: %s", build_dir, exc)


class DoclingRenderStrategy:
    """Delivery-leg collaborators plus the last render's skip/quality side channels."""

    def __init__(
        self,
        *,
        font_family: str | None = None,
        enable_layout_auto_heal: bool = True,
        layout_auto_heal_max_passes: int = 2,
        delivery_mode: str = "reflow",
        enable_pdf_optimization: bool = True,
        pdf_optimize_profile: str = "lossless",
    ) -> None:
        self.font_family = font_family
        self.enable_layout_auto_heal = enable_layout_auto_heal
        self.layout_auto_heal_max_passes = layout_auto_heal_max_passes
        self.delivery_mode = delivery_mode
        self.enable_pdf_optimization = enable_pdf_optimization
        self.pdf_optimize_profile = pdf_optimize_profile
        self.allow_landscape_tables: bool | None = None
        self.toc_mode = "auto"
        self.last_render_skips: list[tuple[str, str]] = []
        self.last_render_flags: list[tuple[str, str]] = []
        self.last_render_reconstructions: list[ReconstructionRecord] = []
        self.last_page_geometry: list[dict[str, Any]] = []
        self.last_outcome: RenderOutcome | None = None

    async def _render_publish(
        self,
        manifest: BookManifest,
        blocks: list[IRBlock],
        target_lang: str,
        output_path: Path,
        bilingual: bool = False,
    ) -> Path:
        """Rebuild the document from the IR and compile it with Typst on a worker thread."""
        return await asyncio.to_thread(
            self._render_publish_sync, manifest, blocks, target_lang, output_path, bilingual
        )

    def _render_publish_sync(
        self,
        manifest: BookManifest,
        blocks: list[IRBlock],
        target_lang: str,
        output_path: Path,
        bilingual: bool = False,
    ) -> Path:
        from ubt.adapters.pdf.typst_compile import typst_compile
        from ubt.adapters.pdf.typst_math_probe import TypstMathProbe
        from ubt.render.page_trim import measure_trim, resolved_page_setup
        from ubt.render.typst.academic import is_academic_document, resolve_document_columns
        from ubt.render.typst.document_renderer import TypstDocumentRenderer
        from ubt.render.typst.style_profile import extract_document_style_profile

        self.last_page_geometry = []
        output_path.parent.mkdir(parents=True, exist_ok=True)
        build_dir = output_path.parent / f".{output_path.stem}_typst"
        assets_root = self._assets_root(manifest)

        math_probe = TypstMathProbe()
        is_academic = is_academic_document(manifest)
        columns = resolve_document_columns(manifest)
        raw_src = str(
            getattr(manifest, "source_path", "")
            or manifest.metadata.get("source_path", "")
            or manifest.metadata.get("input_path", "")
            or ""
        )
        source_pdf_path = Path(raw_src) if raw_src else None
        page_setup = resolved_page_setup(
            measure_trim(raw_src or None),
            columns=columns,
        )
        style_profile = extract_document_style_profile(blocks, source_path=source_pdf_path)
        manifest.metadata["document_style_profile"] = style_profile.to_metadata_dict()
        renderer = TypstDocumentRenderer(
            assets_root=assets_root,
            bilingual=bilingual,
            math_gate=math_probe.check,
            math_gate_many=math_probe.check_many,
            font_stack=self._font_stack_for(target_lang),
            layout_template="academic" if is_academic else "book",
            columns=columns,
            paper_title=manifest.title if is_academic else "",
            abstract=manifest.metadata.get("abstract", "") if is_academic else "",
            toc_mode=getattr(self, "toc_mode", "auto"),
            allow_landscape_tables=getattr(self, "allow_landscape_tables", None),
            page_setup=page_setup,
            source_pdf=source_pdf_path,
            style_profile=style_profile,
        )
        source, stats = renderer.build_source(blocks, build_dir, target_lang)
        typ_path = build_dir / "main.typ"
        typ_path.write_text(source, encoding="utf-8")

        ok, stderr = typst_compile(str(typ_path), str(output_path), target_lang=target_lang)
        if not ok:
            raise DocumentParseError(
                f"publish render failed to compile: {stderr.strip() or 'unknown Typst error'} "
                f"(generated source kept at {typ_path})"
            )

        if getattr(self, "enable_layout_auto_heal", True) and output_path.exists():
            from ubt.render.typst.auto_healing import LayoutAutoHealer

            layout_spec = renderer.layout_spec_for_audit()

            def _renderer_builder(**overrides: Any) -> TypstDocumentRenderer:
                return TypstDocumentRenderer(
                    assets_root=assets_root,
                    bilingual=bilingual,
                    math_gate=math_probe.check,
                    math_gate_many=math_probe.check_many,
                    font_stack=self._font_stack_for(target_lang),
                    layout_template="academic" if is_academic else "book",
                    columns=columns,
                    paper_title=manifest.title if is_academic else "",
                    abstract=manifest.metadata.get("abstract", "") if is_academic else "",
                    toc_mode=getattr(self, "toc_mode", "auto"),
                    allow_landscape_tables=getattr(self, "allow_landscape_tables", None),
                    page_setup=page_setup,
                    source_pdf=source_pdf_path,
                    **overrides,
                )

            max_p = getattr(self, "layout_auto_heal_max_passes", 2)
            healer = LayoutAutoHealer(spec=layout_spec, max_passes=max_p)
            _, heal_report = healer.run_healing_loop(
                output_path=output_path,
                build_dir=build_dir,
                blocks=blocks,
                target_lang=target_lang,
                renderer_builder_fn=_renderer_builder,
                compile_fn=lambda t, o: typst_compile(str(t), str(o), target_lang=target_lang),
            )
            if heal_report.healed:
                manifest.metadata["auto_healing"] = heal_report.to_dict()

        record_toolchain_versions(manifest)
        self.last_render_skips = list(renderer.last_render_skips) + [
            (block_id, "asset_not_found")
            for block_id, asset in stats.assets_missing
            if block_id and asset
        ]
        self.last_render_flags = list(renderer.last_render_flags)
        self.last_render_reconstructions = list(renderer.table_reconstructions)
        _cleanup_build_dir(build_dir)
        return output_path

    def _font_stack_for(self, target_lang: str) -> tuple[str, ...]:
        """Body/heading font stack for the preamble, resolved against this machine."""
        from ubt.adapters.pdf.font_probe import resolve_for_render

        return resolve_for_render(target_lang, font_override=self.font_family).families

    def _assets_root(self, manifest: BookManifest) -> Path | None:
        """Directory holding the extracted figure assets for this job."""
        try:
            from ubt.adapters.pdf.docling_parser import docling_asset_dir

            source = Path(str(getattr(manifest, "source_path", "") or ""))
            if not source.exists():
                return None
            import hashlib

            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            root = docling_asset_dir(digest)
            return root if root.exists() else None
        except Exception as exc:  # pragma: no cover - best-effort asset lookup
            logger.debug("asset root unavailable: %s", exc)
            return None

    async def render_blocks(
        self,
        manifest: BookManifest,
        blocks: list[IRBlock],
        target_lang: str,
        output_path: Path,
        bilingual_mode: str | None = None,
        render_plan: RenderPlan | None = None,
    ) -> Path:
        """Render the translated output from pre-fetched blocks.

        A PDF target is rebuilt from the IR and laid out by Typst reflow. Non-PDF
        targets (.md / .txt) export clean Markdown.
        """
        self.last_render_skips = []
        self.last_render_flags = []
        self.last_render_reconstructions = []
        self.last_page_geometry = []

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        plan_mode = render_plan.bilingual_mode if render_plan is not None else None
        active_mode = bilingual_mode or str(plan_mode or "bilingual")
        is_pdf_target = out_path.suffix.lower() == ".pdf"

        self.last_outcome = RenderOutcome(
            bilingual_mode=active_mode,
            effective_dual_mode=(
                render_plan.effective_dual_mode if render_plan is not None else None
            ),
            dual_mode_downgraded=(render_plan.dual_mode_downgraded if render_plan else None),
        )

        if is_pdf_target:
            if self.delivery_mode in ("facsimile", "page_locked"):
                logger.info(
                    "Docling delivery_mode=%r uses Typst reflow (born-digital 1:1 in-place "
                    "PDF delivery is handled by NativePDFCodec).",
                    self.delivery_mode,
                )
            publish_bilingual = active_mode != "monolingual"
            return await self._render_publish(
                manifest, list(blocks), target_lang, out_path, publish_bilingual
            )

        if out_path.suffix.lower() in (".md", ".markdown", ".txt"):
            from ubt.render.delivered_text import delivered_text

            md_lines: list[str] = [f"# {manifest.title} ({target_lang})\n\n"]
            for block in blocks:
                source = block.source_text
                target = delivered_text(block)
                if block.block_type == BlockType.HEADING:
                    md_lines.append(f"## {target}\n\n")
                elif block.block_type == BlockType.CODE:
                    md_lines.append(f"```\n{target}\n```\n\n")
                elif block.block_type == BlockType.FORMULA:
                    md_lines.append(f"$$\n{target}\n$$\n\n")
                else:
                    if (
                        active_mode not in ("mono", "monolingual")
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
