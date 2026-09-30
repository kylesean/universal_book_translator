"""Render strategy for the Docling adapter.

Turns translated ``IRBlock``s into a delivered artifact: Markdown/`.typ`
export, Typst compilation, alternating/facing bilingual interleaving,
rigid typesetting dispatch, diagram vectorization and the manifest
recording of witness/syntax/toolchain findings. Holds only delivery
collaborators (reconstructor, alternator, diagram localizer, font family),
so the adapter can construct it once and forward ``render_blocks`` to it.
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import suppress
from pathlib import Path

from ubt.adapters.pdf.alternator import BilingualAlternator
from ubt.adapters.pdf.diagram_localizer import DiagramLocalizer
from ubt.adapters.pdf.rigid import RigidTypesetter
from ubt.adapters.pdf.typst_reconstructor import TypstReconstructor
from ubt.core.config import canonical_render_engine
from ubt.core.exceptions import DocumentParseError
from ubt.core.ir.models import BlockType, BookManifest, BoundingBox, FlowID, IRBlock
from ubt.core.policy.adaptive_policy import resolve_pdf_engine

logger = logging.getLogger(__name__)


def record_witness_findings(reconstructor: TypstReconstructor, manifest: BookManifest) -> None:
    """Persist formula-witness fallbacks into manifest metadata.

    A witness FAIL means the converted formula differs structurally from
    the source equation and is replaced by the source graphic. The
    delivered content is lossless, but the substitution must stay
    visible in the publication quality report, exactly like
    the syntax fallbacks above.
    """
    findings = list(getattr(reconstructor, "last_witness_findings", []) or [])
    if not findings:
        return
    recorded = manifest.metadata.setdefault("formula_witness_findings", [])
    if isinstance(recorded, list):
        recorded.extend(findings)
    else:
        manifest.metadata["formula_witness_findings"] = findings


def record_toolchain_versions(reconstructor: TypstReconstructor, manifest: BookManifest) -> None:
    """Persist the Typst compiler pin into manifest metadata.

    the generated markup's meaning depends on compiler
    behavior, so every delivered PDF's quality report carries the exact
    compiler version (None = unmeasured, e.g. compiler absent and the
    Markdown-companion path taken — never fabricated).
    """
    try:
        version = reconstructor.compiler_version()
    except Exception:
        version = None
    if isinstance(version, str) and version:
        manifest.metadata["typst_version"] = version


def record_syntax_fallbacks(reconstructor: TypstReconstructor, manifest: BookManifest) -> None:
    """Persist Typst syntax-fallback removals into manifest metadata.

    lines the self-healing loop commented out are translated
    content GONE from the delivered PDF. Surfacing them through
    ``manifest.metadata`` lets ``build_quality_report`` render them in the
    publication quality report instead of leaving them only in the .typ
    source and logs.
    """
    fallbacks = list(getattr(reconstructor, "last_syntax_fallbacks", []) or [])
    if not fallbacks:
        return
    recorded = manifest.metadata.setdefault("typst_syntax_fallbacks", [])
    if isinstance(recorded, list):
        recorded.extend(fallbacks)
    else:
        manifest.metadata["typst_syntax_fallbacks"] = fallbacks


def _x_overlap_fraction(a: BoundingBox, b: BoundingBox) -> float:
    overlap = min(a.x1, b.x1) - max(a.x0, b.x0)
    if overlap <= 0:
        return 0.0
    return overlap / max(1e-6, min(a.x1 - a.x0, b.x1 - b.x0))


def _weave_images_preserving_order(blocks: list[IRBlock], images: list[IRBlock]) -> list[IRBlock]:
    """Splice freshly extracted figure assets into the block list.

    The pre-existing blocks arrive in the reading order the parser paid to
    reconstruct (multi-column aware). Re-sorting a page by pure y-coordinate
    would interleave left/right columns and destroy that order, so only the
    new IMAGE blocks move: each lands after the last same-column (or under a
    full-width) block that sits physically above it.
    """
    out = list(blocks)
    page_widths: dict[int, float] = {}
    for cand in blocks:
        cb = cand.bbox
        if cb is not None and cb.page > 0:
            page_widths[cb.page] = max(page_widths.get(cb.page, 0.0), cb.x1)
    items: list[tuple[IRBlock, BoundingBox]] = [
        (img, img.bbox) for img in images if img.bbox is not None
    ]
    for img, ib in sorted(items, key=lambda t: (t[1].page, -t[1].y0)):
        width = page_widths.get(ib.page, ib.x1) or (ib.x1 - ib.x0)
        full_width = (ib.x1 - ib.x0) >= 0.55 * width
        last_above = -1
        first_lower: int | None = None
        for i, cand in enumerate(out):
            cb = cand.bbox
            if cb is None or cb.page != ib.page:
                continue
            if not (full_width or _x_overlap_fraction(cb, ib) > 0.35):
                continue
            if cb.y0 >= ib.y1:
                last_above = i
            elif first_lower is None:
                first_lower = i
        pos = (
            last_above + 1
            if last_above >= 0
            else (first_lower if first_lower is not None else len(out))
        )
        out.insert(pos, img)
    return out


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
        reconstructor: TypstReconstructor,
        alternator: BilingualAlternator,
        diagram_localizer: DiagramLocalizer,
        font_family: str | None = None,
    ) -> None:
        self.reconstructor = reconstructor
        self.alternator = alternator
        self.diagram_localizer = diagram_localizer
        self.font_family = font_family
        # Render skip side channel: plain (block_id, reason) pairs
        # from the most recent render_blocks call. Reset every render.
        self.last_render_skips: list[tuple[str, str]] = []

    async def render_blocks(
        self,
        manifest: BookManifest,
        blocks: list[IRBlock],
        target_lang: str,
        output_path: Path,
        bilingual_mode: str | None = None,
        render_engine: str | None = None,
    ) -> Path:
        """Render publication-grade translated output from pre-fetched blocks.

        Supports non-blocking compilation, automatic facing-page bilingual
        interleaving, in-diagram text localization, and two PDF engines:

        - ``publication`` (default): full Typst reflow with academic typography;
        - ``rigid``: region-rigid adaptive typesetting that keeps the
          source page as the canvas (figures, equations and vectors untouched);
        - ``auto``: density dispatch between the two (formula/struct-dense
          documents take the rigid engine).
        """
        self.last_render_skips = []
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        active_mode = bilingual_mode or str(manifest.run.bilingual_mode or "bilingual")
        requested_engine = render_engine or str(manifest.run.render_engine or "publication")
        is_pdf_target = out_path.suffix.lower() == ".pdf"
        active_engine = (
            resolve_pdf_engine(requested_engine, blocks, manifest=manifest)
            if is_pdf_target
            else "publication"
        )

        # A forced engine bypasses the density dispatch silently: a
        # formula/table-dense document forced to reflow can come back
        # "almost undeliverable" with no hint why -- auto routes such
        # documents to rigid. Surface the disagreement at the moment the
        # choice matters.
        if is_pdf_target and blocks:
            _warn_forced_engine(requested_engine, active_engine, blocks, manifest=manifest)

        # Rigid typesetting is monolingual: never let a requested bilingual
        # mode silently produce a mono artifact. The downgrade is recorded so
        # the quality report and the secondary-render guard can see it.
        if active_engine == "rigid" and active_mode != "monolingual":
            was_downgraded = manifest.run.dual_mode_downgraded == active_mode
            manifest.run.dual_mode_downgraded = active_mode
            active_mode = "monolingual"
            manifest.run.bilingual_mode = "monolingual"
            manifest.run.effective_dual_mode = "monolingual"
            if not was_downgraded:
                logger.warning(
                    "render_engine='rigid' is monolingual; requested mode %r downgraded",
                    manifest.run.dual_mode_downgraded,
                )
        manifest.run.render_engine_effective = active_engine
        # The metadata copy is a fallback for manifests that predate the typed
        # field: ``reflow_loop.effective_render_engine`` reads
        # ``manifest.run.render_engine_effective`` first and falls back to
        # ``manifest.metadata["render_engine_effective"]``. Without either, the
        # geometry predicates treat every render -- including a reflowed
        # publication -- as geometry-preserving, which injects bogus
        # page_count/image_count parity majors into the visual gate.
        if isinstance(getattr(manifest, "metadata", None), dict):
            manifest.metadata["render_engine_effective"] = active_engine

        # --- Rigid typesetting (region-rigid adaptive typesetting) ---
        # Source page stays the canvas; prose regions are re-typeset, figures
        # and equations untouched.
        if active_engine == "rigid" and is_pdf_target:
            # Written by ``DoclingPDFAdapter.font_family``'s setter, which the
            # pipeline drives; a plain attribute set at construction would
            # stay ``None`` here forever.
            font_family = getattr(self, "font_family", None)
            rigid = RigidTypesetter(
                translate_chrome=bool(manifest.run.translate_chrome),
                font_family=font_family,
                target_lang=target_lang or getattr(manifest, "target_lang", "zh") or "zh",
            )
            staged_path, rigid_report = await rigid.render(
                manifest=manifest,
                blocks=list(blocks),
                target_lang=target_lang,
                output_path=out_path,
            )
            # Every entry is block-scoped: the rigid engine demotes a failed
            # page's blocks to per-block skips, so the loss reaches the quality
            # report instead of being filtered away here.
            self.last_render_skips = list(rigid_report.skipped)
            return staged_path

        # Shared doc title (extension/underscore cleaned) for both backends.
        doc_title = manifest.title or ""
        for ext in (".pdf", ".md", ".txt"):
            if doc_title.lower().endswith(ext):
                doc_title = doc_title[: -len(ext)]
        doc_title = doc_title.replace("_", " ").strip()

        # --- Publication engine (Typst reflow) ---

        # 1. Direct Markdown export (.md / .markdown)
        if out_path.suffix.lower() in (".md", ".markdown"):
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

        # 2. Typst markup source export (.typ)
        from ubt.adapters.pdf.docling_blocks import resolve_overlapping_formula_blocks

        blocks = resolve_overlapping_formula_blocks(list(blocks))
        # ``active_mode`` is the adapter vocabulary (``RENDER_MODE_VALUE`` in
        # bilingual_advisor: "bilingual"/"alternating"/"monolingual"/
        # "facing_spread"), not the ``DualMode`` Literal. Only "bilingual"
        # interleaves the target into the source flow; the old "inline" /
        # "interlinear" alternatives were never produced by any caller, so this
        # is the same predicate with the dead names dropped.
        is_in_place = active_mode == "bilingual"
        # Vectorize qualifying diagram images before emission.
        src_pdf_path = Path(manifest.source_path)
        # Hand the source PDF to the reconstructor so a formula that fails
        # conversion can fall back to its original graphic instead of a wall
        # of raw text (see TypstReconstructor._crop_formula_fallback).
        self.reconstructor.source_pdf = (
            src_pdf_path
            if src_pdf_path.exists() and src_pdf_path.suffix.lower() == ".pdf"
            else None
        )
        if src_pdf_path.exists() and src_pdf_path.suffix.lower() == ".pdf":
            blocks, _ = await asyncio.to_thread(
                self._vectorize_diagrams_sync,
                src_pdf_path,
                blocks,
                out_path.parent / "assets",
                target_lang,
            )

        # Automatic in-diagram text localization (experimental opt-in via UBT_LOCALIZE_DIAGRAMS=1).
        # By default, preserve pristine original figure graphics without drawing white boxes/PIL text.
        if (
            os.environ.get("UBT_LOCALIZE_DIAGRAMS") == "1"
            and src_pdf_path.exists()
            and src_pdf_path.suffix.lower() == ".pdf"
        ):
            image_blocks = [
                b
                for b in blocks
                if b.block_type == BlockType.IMAGE
                and b.bbox
                and not str(b.target_text or "").lower().endswith(".svg")
            ]
            if image_blocks:
                try:
                    # Localization spawns subprocesses and does PIL work —
                    # run it in a thread so the event loop is not blocked.
                    await asyncio.to_thread(
                        self.diagram_localizer.localize_all_diagrams,
                        source_pdf=src_pdf_path,
                        image_blocks=image_blocks,
                        target_lang=target_lang,
                    )
                except Exception as exc:
                    logger.warning("Diagram localization failed (non-fatal): %s", exc)

        # Cover subtitle/description cuts scale with the source page height, so
        # read page 1's size to classify a non-A4 cover correctly. Best-effort:
        # a missing/unreadable PDF leaves the A4-absolute fallback in place.
        source_page_height: float | None = None
        source_page_count: int | None = None
        if src_pdf_path.exists() and src_pdf_path.suffix.lower() == ".pdf":
            try:
                from ubt.adapters.pdf import pdf_struct

                page_one = pdf_struct.page_sizes(src_pdf_path).get(1)
                if page_one is not None:
                    source_page_height = page_one[1]
                if active_mode in ("alternating", "facing", "facing_spread"):
                    # Page-pad to the full source length so the translated PDF
                    # has one page per source page (see _generate_page_strict).
                    source_page_count = len(pdf_struct.page_sizes(src_pdf_path))
            except Exception as exc:  # noqa: BLE001 — cover heuristic only
                logger.debug("Could not read source page height for cover: %s", exc)

        # generate_typst_source walks every block through math conversion,
        # image cropping and font measurement — pure CPU work that would stall
        # the event loop for the whole document.
        typ_source = await asyncio.to_thread(
            self.reconstructor.generate_typst_source,
            blocks=blocks,
            title=doc_title,
            bilingual=is_in_place,
            page_strict=not is_in_place,
            cover_mode=str(manifest.run.cover_mode or "auto"),
            # Hard 1:1 pagebreaks exist only for the alternating page-zipper
            # (source/translation interleave). Monolingual reflow flows
            # continuously — forced breaks strand figure-only pages on
            # overflow and shift every later page.
            pagebreaks=(active_mode in ("alternating", "facing", "facing_spread")),
            target_lang=target_lang,
            source_page_height=source_page_height,
            source_page_count=source_page_count,
        )
        # The reflow path's fail-closed drops are staged image assets; without
        # recording them a rigid-only skip ledger leaves ``render_coverage`` at
        # 100% for a book whose figures never made it into the file.
        self.last_render_skips = list(getattr(self.reconstructor, "last_image_skips", []) or [])
        if out_path.suffix.lower() == ".typ":
            out_path.write_text(typ_source, encoding="utf-8")
            return out_path

        # 3. PDF compilation via Typst (non-blocking in executor)
        typ_file = out_path.with_suffix(".typ")
        typ_file.write_text(typ_source, encoding="utf-8")

        if self.reconstructor.is_compiler_available():
            src_path = Path(manifest.source_path)
            if (
                active_mode in ("alternating", "facing", "facing_spread")
                and src_path.exists()
                and src_path.suffix.lower() == ".pdf"
            ):
                use_facing = (
                    active_mode in ("facing", "facing_spread")
                    or getattr(self.reconstructor, "facing_spread", False)
                    or bool(manifest.run.facing_spread)
                )
                staging_trans_pdf = out_path.with_name(f"{out_path.stem}_trans_stage.pdf")
                # compile_pdf stages its own .typ next to the target; both the
                # staged PDF and the staged .typ must be cleaned up or the
                # output directory accumulates `<stem>_trans_stage.typ`.
                staging_trans_typ = staging_trans_pdf.with_suffix(".typ")
                try:
                    await self.reconstructor.compile_pdf_async(typ_source, staging_trans_pdf)
                    # Hard 1:1 guard: a page-count mismatch means the alternator
                    # would pair the wrong pages from that point on. Refuse to
                    # ship a silently misaligned bilingual book (the reconstructor
                    # pads empty source pages, so a mismatch is a real defect).
                    if source_page_count is not None:
                        from ubt.adapters.pdf import pdf_struct as _pdf_struct

                        trans_len = len(_pdf_struct.page_sizes(staging_trans_pdf))
                        if trans_len != source_page_count:
                            raise DocumentParseError(
                                f"Page-strict {active_mode} render produced {trans_len} "
                                f"page(s) for a {source_page_count}-page source; refusing "
                                "to interleave a misaligned bilingual book."
                            )
                    record_syntax_fallbacks(self.reconstructor, manifest)
                    record_witness_findings(self.reconstructor, manifest)
                    record_toolchain_versions(self.reconstructor, manifest)
                    interleaved = await self.alternator.interleave_pages_async(
                        src_path, staging_trans_pdf, out_path, facing_spread=use_facing
                    )
                    # The alternator pads the shorter side with intentional
                    # blanks; record exactly which pages so the visual gate
                    # does not report them as CRITICAL blank_page defects.
                    manifest.run.render_padding_pages = list(interleaved.padding_pages) or None
                finally:
                    staging_trans_pdf.unlink(missing_ok=True)
                    staging_trans_typ.unlink(missing_ok=True)
            else:
                await self.reconstructor.compile_pdf_async(typ_source, out_path)
                record_syntax_fallbacks(self.reconstructor, manifest)
                record_witness_findings(self.reconstructor, manifest)
                record_toolchain_versions(self.reconstructor, manifest)
        else:
            # Compiler not found: the .typ and a bilingual Markdown companion are
            # real artifacts the user can compile elsewhere, but `out_path` is
            # never written. Returning it anyway would report a successful
            # render of a PDF that does not exist.
            md_companion = out_path.with_suffix(".md")
            md_lines = [f"# {manifest.title} ({target_lang})\n\n"]
            for b in blocks:
                md_lines.append(f"{b.target_text or b.draft_text or b.source_text}\n\n")
            md_companion.write_text("".join(md_lines), encoding="utf-8")
            raise DocumentParseError(
                "Typst compiler not found — no PDF was produced. "
                f"Compile the source with `typst compile {typ_file}` or open "
                f"the Markdown fallback at {md_companion}."
            )

        return out_path

    def _vectorize_diagrams_sync(
        self,
        src_pdf_path: Path,
        blocks: list[IRBlock],
        assets_dir: Path,
        target_lang: str,
    ) -> tuple[list[IRBlock], set[int]]:
        """Replace qualifying diagram PNGs with vectorized SVG assets.

        Two sources, both with *real* bboxes only (never the weave fallback's
        fabricated rects):
        1. Docling-direct IMAGE blocks (``pdf_main#img_<digits>``) whose PNG
           target exists — re-rendered as cropped + label-backfilled SVG.
        2. Vector-diagram regions auto-detected from content-stream paths —
           covers books (like born-digital LaTeX handbooks) where Docling
           emits no PICTURE items at all. New ``pdf_main#svg_*`` IMAGE blocks
           are appended (``skip_translate=True``, render-only).

        Returns ``(blocks, covered_pages)`` so the caller can exclude those
        pages from the legacy PNG weave and avoid duplicate figures. Never
        raises: any per-diagram failure keeps the legacy path.
        """
        from ubt.adapters.pdf.svg_diagram import (
            detect_diagram_regions,
            is_docling_direct_image,
            is_svg_backend_available,
            is_svg_rendering_supported,
            render_diagram_png,
            render_diagram_svg,
        )

        # Work on a copy: export hands the same ``final_blocks`` list to every
        # render in the run (visual-gate reflow, --emit-both), and appending
        # ``pdf_main#fig_*`` to the caller's list would make the second render
        # ship each academic figure twice under a duplicated block id.
        blocks = list(blocks)

        covered: set[int] = set()
        svg_available = is_svg_backend_available()
        typst_svg_ok = is_svg_rendering_supported()
        vector_ok = svg_available and typst_svg_ok

        if vector_ok:
            logger.debug("PDF diagram extraction: vector SVG route active (pdftocairo + Typst SVG)")
        elif not svg_available:
            logger.info(
                "Poppler SVG tools (pdftocairo/pdftotext) unavailable; "
                "automatically falling back to raster diagram extraction"
            )
        elif not typst_svg_ok:
            logger.info(
                "Typst build does not support vector SVG rendering; "
                "automatically falling back to raster diagram extraction"
            )
        try:
            assets_dir.mkdir(parents=True, exist_ok=True)
            # Drop stale vector/raster assets from previous runs (region layout may
            # have changed); they are re-rendered below when still detected.
            for stale in (
                list(assets_dir.glob("vec_p*_*.svg"))
                + list(assets_dir.glob("ras_p*_*.png"))
                + list(assets_dir.glob("emb_p*_*.jpg"))
                + list(assets_dir.glob("emb_p*_*.png"))
            ):
                with suppress(OSError):
                    stale.unlink()
        except OSError:
            return blocks, covered

        converted_boxes: list[tuple[int, tuple[float, float, float, float]]] = []
        if vector_ok:
            for block in blocks:
                if block.block_type != BlockType.IMAGE or not block.bbox:
                    continue
                if not is_docling_direct_image(block.id):
                    continue
                png_path = Path(block.target_text or "")
                if png_path.suffix.lower() != ".png" or not png_path.exists():
                    continue
                bbox = block.bbox
                if bbox.page <= 0 or (bbox.x1 - bbox.x0) <= 0 or (bbox.y1 - bbox.y0) <= 0:
                    continue
                try:
                    page_height = self.diagram_localizer.get_page_height(src_pdf_path, bbox.page)
                    svg_path = render_diagram_svg(
                        src_pdf_path,
                        page_no=bbox.page,
                        bbox_bottomup=(bbox.x0, bbox.y0, bbox.x1, bbox.y1),
                        page_height=page_height,
                        localizer=self.diagram_localizer,
                        target_lang=target_lang,
                        work_dir=assets_dir / ".svg_work",
                        out_path=png_path.with_suffix(".svg"),
                    )
                except Exception as exc:
                    logger.debug("SVG vectorization skipped for %s: %s", block.id, exc)
                    continue
                if svg_path is not None:
                    block.source_text = str(svg_path.resolve())
                    block.target_text = str(svg_path.resolve())
                    covered.add(bbox.page)
                    converted_boxes.append((bbox.page, (bbox.x0, bbox.y0, bbox.x1, bbox.y1)))

        # Auto-detect vector-diagram regions Docling did not label PICTURE.
        try:
            from ubt.adapters.pdf import pdf_struct

            # One document open for every page's size, not one open per page:
            # ``get_page_height`` opened the whole PDF P times here.
            page_heights = {p: h for p, (_w, h) in pdf_struct.page_sizes(src_pdf_path).items()}
            page_count = len(page_heights)
        except Exception as exc:
            logger.debug("SVG detection: cannot read page count: %s", exc)
            return blocks, covered
        tables_by_page: dict[int, list[tuple[float, float, float, float]]] = {}
        for block in blocks:
            if block.block_type == BlockType.TABLE and block.bbox and block.bbox.page > 0:
                b = block.bbox
                if b.x1 > b.x0 and b.y1 > b.y0:
                    tables_by_page.setdefault(b.page, []).append((b.x0, b.y0, b.x1, b.y1))

        appended = 0
        # SOTA Academic Figure Extraction (MinerU / Marker model):
        # Scan page captions and extract complete figures with full axes, ticks, labels, and legends at 300 DPI.
        try:
            from ubt.adapters.pdf.asset_extractor import extract_pdf_figures

            extracted_figs = extract_pdf_figures(src_pdf_path, assets_dir, dpi=300)
        except Exception as exc:
            logger.debug("Academic figure extraction failed: %s", exc)
            extracted_figs = {}

        active_pages = {b.bbox.page for b in blocks if b.bbox and b.bbox.page > 0}

        # Largest crop first, and any crop that covers (or is covered by) an
        # already-placed image on the same page is dropped: caption-driven
        # extraction happily produced a FIG 3.3 rect that fully contained
        # FIG 3.2 plus its caption, so the reader got the same figure twice and
        # a caption printed inside the wrong figure.
        def _overlap_fraction(
            a: tuple[float, float, float, float], b: tuple[float, float, float, float]
        ) -> float:
            ix = min(a[2], b[2]) - max(a[0], b[0])
            iy = min(a[3], b[3]) - max(a[1], b[1])
            if ix <= 0 or iy <= 0:
                return 0.0
            inter = ix * iy
            smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
            return inter / smaller if smaller > 0 else 0.0

        placed_rects: dict[int, list[tuple[float, float, float, float]]] = {}
        for existing in blocks:
            if existing.block_type == BlockType.IMAGE and existing.bbox is not None:
                placed_rects.setdefault(existing.bbox.page, []).append(
                    (existing.bbox.x0, existing.bbox.y0, existing.bbox.x1, existing.bbox.y1)
                )

        def _fig_area(item: tuple[str, object]) -> float:
            box = getattr(item[1], "bbox", None)
            if not box:
                return 0.0
            return float((box[2] - box[0]) * (box[3] - box[1]))

        woven: list[IRBlock] = []
        for fig_id, fig in sorted(extracted_figs.items(), key=_fig_area, reverse=True):
            if active_pages and fig.page not in active_pages:
                continue
            if fig.bbox is not None:
                fig_rect = (
                    float(fig.bbox[0]),
                    float(fig.bbox[1]),
                    float(fig.bbox[2]),
                    float(fig.bbox[3]),
                )
                if any(
                    _overlap_fraction(fig_rect, rect) >= 0.5
                    for rect in placed_rects.get(fig.page, ())
                ):
                    logger.debug(
                        "Academic figure %s overlaps an already-placed image on page %d; skipped",
                        fig_id,
                        fig.page,
                    )
                    continue
            safe_id = fig_id.replace(".", "_")
            if not fig.image_path.exists():
                continue
            if not fig.bbox:
                # No fabricated rects — this function's contract
                # (and the weave guard downstream) requires REAL geometry.
                # A figure without a bbox is dropped, consistent with the
                # other unplaceable-asset paths above.
                logger.debug("Academic figure %s has no bbox; skipping", fig_id)
                continue
            bx0, by0, bx1, by1 = fig.bbox
            woven.append(
                IRBlock(
                    id=f"pdf_main#fig_{safe_id}",
                    spine_index=9999,
                    block_type=BlockType.IMAGE,
                    flow_id=FlowID.CAPTION,
                    source_text=str(fig.image_path.resolve()),
                    target_text=str(fig.image_path.resolve()),
                    skip_translate=True,
                    bbox=BoundingBox(page=fig.page, x0=bx0, y0=by0, x1=bx1, y1=by1),
                )
            )
            placed_rects.setdefault(fig.page, []).append((bx0, by0, bx1, by1))
            converted_boxes.append((fig.page, (bx0, by0, bx1, by1)))
            covered.add(fig.page)
            appended += 1

        scan_pages = sorted(active_pages) if active_pages else range(1, page_count + 1)
        for page_no in scan_pages:
            exclude = list(tables_by_page.get(page_no, ()))
            exclude.extend(rect for pg, rect in converted_boxes if pg == page_no)
            try:
                regions = detect_diagram_regions(src_pdf_path, page_no, exclude=exclude or None)
            except Exception as exc:
                logger.debug("SVG detection failed on p%d: %s", page_no, exc)
                continue
            for idx, (x0, y0, x1, y1) in enumerate(regions):
                if vector_ok:
                    svg_out = assets_dir / f"vec_p{page_no}_{idx}.svg"
                    try:
                        svg_path = render_diagram_svg(
                            src_pdf_path,
                            page_no=page_no,
                            bbox_bottomup=(x0, y0, x1, y1),
                            page_height=page_heights.get(page_no, 720.0),
                            localizer=self.diagram_localizer,
                            target_lang=target_lang,
                            work_dir=assets_dir / ".svg_work",
                            out_path=svg_out,
                        )
                    except Exception as exc:
                        logger.debug("SVG region render failed (p%d): %s", page_no, exc)
                        continue
                    if svg_path is None:
                        continue
                    asset_path = svg_path
                    block_id = f"pdf_main#svg_{page_no}_{idx}"
                else:
                    png_out = assets_dir / f"ras_p{page_no}_{idx}.png"
                    try:
                        ras_path = render_diagram_png(
                            src_pdf_path,
                            page_no=page_no,
                            bbox_bottomup=(x0, y0, x1, y1),
                            page_height=page_heights.get(page_no, 720.0),
                            out_path=png_out,
                        )
                    except Exception as exc:
                        logger.debug("PNG region render failed (p%d): %s", page_no, exc)
                        continue
                    if ras_path is None:
                        continue
                    asset_path = ras_path
                    block_id = f"pdf_main#ras_{page_no}_{idx}"
                woven.append(
                    IRBlock(
                        id=block_id,
                        spine_index=9999,
                        block_type=BlockType.IMAGE,
                        flow_id=FlowID.CAPTION,
                        source_text=str(asset_path.resolve()),
                        target_text=str(asset_path.resolve()),
                        skip_translate=True,
                        bbox=BoundingBox(page=page_no, x0=x0, y0=y0, x1=x1, y1=y1),
                    )
                )
                covered.add(page_no)
                appended += 1
        if appended:
            blocks = _weave_images_preserving_order(blocks, woven)
            logger.info("Figure assets woven: %d new IMAGE blocks", appended)
        return blocks, covered
