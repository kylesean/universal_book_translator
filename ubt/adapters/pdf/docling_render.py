"""Render strategy for the Docling adapter.

Turns translated ``IRBlock``s into a delivered artifact: a PDF rebuilt from the
IR and laid out by Typst, or a Markdown/flat-text export. Holds only delivery
collaborators, so the adapter can construct it once and forward ``render_blocks``
to it.
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from ubt.adapters.pdf.strip_pdf_text import BoxRect
from ubt.adapters.pdf.typst_compile import typst_version
from ubt.core.content.nodes import ReconstructionRecord
from ubt.core.exceptions import DocumentParseError
from ubt.core.ir.models import BlockType, BookManifest, IRBlock
from ubt.core.ir.render_plan import RenderOutcome, RenderPlan

if TYPE_CHECKING:
    from ubt.model.span import PhysicalBox

logger = logging.getLogger(__name__)


def _physical_box_region(pbox: PhysicalBox, page_width: float) -> tuple[int, int]:
    """The ``(page, column)`` region one physical box belongs to.

    ``page_width`` is a parameter, not a closed-over loop variable. The caller
    walks pages inside a loop, so a nested definition that captured ``pw`` would
    bind whatever width the loop last happened to hold -- and the column test
    (``w > 0.65 * page_width``) is a comparison against *this* page's measure.
    A box wider than 65% of the measure spans the gutter by construction, and so
    does one whose midpoint sits left of it; both are the left column.
    """
    bw = pbox.bbox[2] - pbox.bbox[0]
    bmid = (pbox.bbox[0] + pbox.bbox[2]) / 2.0
    bcol = 0 if bw > 0.65 * page_width or bmid < page_width * 0.50 else 1
    return (int(pbox.page), bcol)


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


#: A block whose own measured size is below this is not a real body size; fall
#: back to the page's. No delivered face sets text at 4.5pt (``textgeom`` drops
#: char styles under the same threshold).
_MIN_BLOCK_SIZE_PT: float = 4.5
#: A heading is set at least this many times the body size when its own size is
#: unmeasured. 1.6 is the smallest ratio that still reads as a heading against a
#: page of uniform body text.
_HEADING_SIZE_RATIO: float = 1.6


class _StripPlannable(Protocol):
    """The four attributes :func:`_facsimile_strip_plan` actually reads.

    Structural rather than ``IRBlock`` so a test (or any caller holding only a
    box and a kind) can plan strips without inventing an ``element``. The
    properties on ``IRBlock`` derive ``bbox``/``block_type``/``skip_translate``
    from the element, so a real block satisfies this unchanged.

    They are declared as read-only ``@property`` because that is what
    ``IRBlock`` exposes: a plain attribute annotation would demand a settable
    member and a read-only property does not satisfy it.
    """

    @property
    def bbox(self) -> Any: ...

    @property
    def skip_translate(self) -> bool: ...

    @property
    def block_type(self) -> BlockType: ...


class _SizedBlock(Protocol):
    """The only two attributes ``_block_font_size`` reads.

    A structural type rather than ``IRBlock``: the function needs a measured
    ``font_size`` on the style and on the provenance, and nothing else. Naming
    that lets a caller with just a measurement be checked, instead of being
    forced to build a whole block whose other fields are never touched.
    """

    style: Any
    provenance: Any


def _block_font_size(block: _SizedBlock, *, is_hdr: bool, body_size: float) -> float | None:
    """A block's own size in points, or ``None`` to inherit the page's body size.

    Body prose inherits the page size, which the solver already picked. A
    *heading* must not: the page solves one size for its body, so a 16.2pt
    chapter title was being drawn at ~10pt and every page read as a flat wall of
    uniform text. The size is taken from the block's own style/provenance
    measurement, and only when that measurement is plausible.

    Takes ``_SizedBlock`` rather than ``IRBlock`` because only two attributes
    are read. Requiring the full model forced callers with a measured size but
    no ``element`` to fabricate one.
    """
    candidates: list[float] = []
    if block.style is not None and block.style.font_size is not None:
        candidates.append(float(block.style.font_size))
    if block.provenance is not None and block.provenance.font_size is not None:
        candidates.append(float(block.provenance.font_size))
    measured = next((size for size in candidates if size >= _MIN_BLOCK_SIZE_PT), None)

    if not is_hdr:
        # Body prose: the page's uniform size is the whole point of the channel.
        return None
    if measured is not None:
        return round(measured, 2)
    return round(body_size * _HEADING_SIZE_RATIO, 2)


@dataclass(frozen=True, slots=True)
class _PageMetrics:
    """One page's measured body size and leading, in points."""

    size_pt: float
    leading_pt: float


def _page_metrics(source_path: Path, page_no: int) -> _PageMetrics | None:
    """Measure the source page's body size and leading, or ``None`` if unreadable.

    Measured from the page's own lines rather than assumed: the corpus spans
    9.0–11.3pt, and the previous hardcoded 10.0/13.0 forced every page to
    ``min(12.0, 10.0*1.10) = 11.0pt`` — smaller than the source on every 11.3pt
    page. ``textgeom.extract_lines`` is pdfium-backed and serialized, so this is
    one call per page and cheap relative to the compile that follows.
    """
    try:
        from ubt.adapters.pdf.source_metrics import (
            body_lines,
            source_body_size,
            source_leading,
        )
        from ubt.adapters.pdf.textgeom import extract_lines

        lines, (_page_w, page_h) = extract_lines(source_path, page_no)
        body = body_lines(lines, page_h)
        if not body:
            return None
        size = source_body_size(body)
        return _PageMetrics(size_pt=size, leading_pt=source_leading(body, size))
    except Exception as exc:
        logger.debug("Could not measure source metrics for page %d: %s", page_no, exc)
        return None


def _facsimile_strip_plan(
    blocks_by_page: Mapping[int, Sequence[_StripPlannable]],
    total_pages: int,
    source_path: Path,
) -> tuple[dict[int, list[BoxRect]], dict[int, list[BoxRect]]]:
    """Which text the background layer erases, and which it must protect.

    The background layer is the source page minus the text a Typst block will
    re-draw. Everything else must survive, and the distinction is not cosmetic:

    * **Erase** every block whose translation is placed by Typst — its box is
      exactly the region the new text will occupy.
    * **Keep** every block that is *not* re-drawn: formulas, figures, and any
      block the pipeline decided not to translate. Their source ink is the
      deliverable.
    * **Keep** detected vector-diagram regions, which carry axis labels the
      figure block's bbox does not always cover.

    Keep always wins over erase, so an overlap degrades to "kept".

    **There is no blanket furniture band, and that is deliberate.** The first
    implementation kept the full width of ``[0, 0.08·H]`` and ``[0.92·H, H]``
    on every page, on the reasoning that a folio or running head has no block in
    the IR (the parser discards repeating chrome) and so would be erased with
    nothing to redraw it. Measurement refuted that reasoning on the real IR:

    * The band protects nothing. Text outside every block box has no erase rect
      over it, so it survives on its own. On ``chapter-3.pdf`` p1–p3 the same
      furniture survived with the bands and without them, and the formula and
      running head were kept in both cases.
    * The band actively leaks English. Body prose routinely reaches below
      ``0.08·H``: p2's last paragraph line sits at ``y0 = 31.3`` (a 666pt page
      puts the band top at 53.3) and p1's at ``y0 = 69.4``. Those lines *do* have
      translated blocks, so they get an erase rect — which the band then vetoes.
      Result: the source line keeps its ink under the translation. Measured
      leak: 2 source lines on each of p1–p3, 0 with the bands removed.

    A running head that Docling keeps in the IR arrives as a ``skip_translate``
    block in a ``HEADER``/``FOOTER`` region and is protected as a keep rect like
    any other untranslated block. One it drops has no erase rect over it either.
    Neither case needs a geometric band.
    """
    erase: dict[int, list[BoxRect]] = {}
    keep: dict[int, list[BoxRect]] = {}
    seen_rects: set[tuple[bool, int, float, float, float, float]] = set()

    def _add_rect(is_keep: bool, page_no: int, x0: float, y0: float, x1: float, y1: float) -> None:
        if page_no < 1 or page_no > total_pages:
            return
        rect = BoxRect(page_index=page_no, x0=x0, y0=y0, x1=x1, y1=y1)
        if rect.width <= 0 or rect.height <= 0:
            return
        key = (is_keep, page_no, round(x0, 3), round(y0, 3), round(x1, 3), round(y1, 3))
        if key in seen_rects:
            return
        seen_rects.add(key)
        target = keep if is_keep else erase
        target.setdefault(page_no, []).append(rect)

    for pno, page_blocks in blocks_by_page.items():
        for b in page_blocks:
            if b.bbox is None:
                continue
            keeps_source = b.skip_translate or b.block_type in (
                BlockType.FORMULA,
                BlockType.IMAGE,
            )
            prov = getattr(b, "provenance", None)
            phys_boxes = getattr(prov, "physical_boxes", None) if prov is not None else None
            if phys_boxes and len(phys_boxes) > 1:
                from ubt.model.span import boxes_from_provenance

                for pbox in boxes_from_provenance(phys_boxes):
                    if pbox.page == pno or b.bbox.page == pno:
                        _add_rect(
                            keeps_source,
                            int(pbox.page),
                            float(pbox.bbox[0]),
                            float(pbox.bbox[1]),
                            float(pbox.bbox[2]),
                            float(pbox.bbox[3]),
                        )
                continue

            if b.bbox.page != pno:
                continue
            _add_rect(
                keeps_source,
                pno,
                float(b.bbox.x0),
                float(b.bbox.y0),
                float(b.bbox.x1),
                float(b.bbox.y1),
            )

    # No furniture bands are added here. See the docstring: a blanket
    # ``[0, 0.08·H]`` / ``[0.92·H, H]`` keep rect protects nothing (unowned text
    # has no erase rect over it anyway) while vetoing real body prose that
    # reaches into the band — which leaks the source line under its translation.
    # Page furniture that Docling keeps in the IR arrives as a ``skip_translate``
    # block and is protected by the loop above like any other untranslated block.

    # Vector-diagram regions: their labels are figure content, and a figure's IR
    # bbox does not always cover the axis text drawn outside it. Table bboxes are
    # excluded because a table has its own pipeline and its cell text must be
    # erased, not protected.
    try:
        from ubt.adapters.pdf.svg_diagram import detect_diagram_regions

        table_boxes: dict[int, list[tuple[float, float, float, float]]] = {}
        for pno, page_blocks in blocks_by_page.items():
            for b in page_blocks:
                if b.block_type is BlockType.TABLE and b.bbox is not None:
                    table_boxes.setdefault(pno, []).append(
                        (float(b.bbox.x0), float(b.bbox.y0), float(b.bbox.x1), float(b.bbox.y1))
                    )

        for pno in range(1, total_pages + 1):
            regions = detect_diagram_regions(source_path, pno, exclude=table_boxes.get(pno) or None)
            for x0, y0, x1, y1 in regions:
                keep.setdefault(pno, []).append(BoxRect(page_index=pno, x0=x0, y0=y0, x1=x1, y1=y1))
    except Exception as exc:
        logger.debug("Diagram-region detection unavailable: %s", exc)

    return erase, keep


class DoclingRenderStrategy:
    """Delivery-leg collaborators plus the last render's skip side channel."""

    def __init__(
        self,
        *,
        font_family: str | None = None,
        enable_layout_auto_heal: bool = True,
        layout_auto_heal_max_passes: int = 2,
        delivery_mode: str = "facsimile",
        enable_pdf_optimization: bool = True,
        pdf_optimize_profile: str = "lossless",
    ) -> None:
        self.font_family = font_family
        self.enable_layout_auto_heal = enable_layout_auto_heal
        self.layout_auto_heal_max_passes = layout_auto_heal_max_passes
        self.delivery_mode = delivery_mode
        self.enable_pdf_optimization = enable_pdf_optimization
        self.pdf_optimize_profile = pdf_optimize_profile
        #: ``auto`` | ``on`` | ``off`` for placing a recovered table of contents.
        #: Set from the resolved config by the adapter (``apply_config``).
        self.toc_mode = "auto"
        # Render skip side channel: plain (block_id, reason) pairs
        # from the most recent render_blocks call. Reset every render.
        self.last_render_skips: list[tuple[str, str]] = []
        # Render *quality* side channel: (block_id, flag) pairs for blocks that
        # were drawn but below the readable floor (``low_legibility_font``).
        # Kept apart from the skips: these blocks DID deliver a translation, so
        # the flag must not make the contract read them as source-kept.
        self.last_render_flags: list[tuple[str, str]] = []
        # Per-asset reconstruction verdicts (block_id, verified, corrupt, reason)
        # from the most recent render: the delivery contract records a rebuilt
        # table/formula as RECONSTRUCTED (verified or corrupt) from these.
        self.last_render_reconstructions: list[ReconstructionRecord] = []
        # Per-page typographic geometry from the most recent facsimile render:
        # the measured source size/leading and the size/tier actually delivered.
        # This fact used to exist only inside the build and died with the
        # process, so "why is page 12 at 7.5pt" was unanswerable after the run.
        self.last_page_geometry: list[dict[str, Any]] = []
        # Render outcome side channel (compiler render plan protocol): the mode
        # the renderer actually used. The renderer has no downgrade path
        # (bilingual pairing is per block and cannot fail independently), so this
        # only ever reports the mode it was asked for — but the export stage and
        # the complementary-artifact suffix read it, so it must be populated.
        self.last_outcome: RenderOutcome | None = None

    async def _render_publish(
        self,
        manifest: BookManifest,
        blocks: list[IRBlock],
        target_lang: str,
        output_path: Path,
        bilingual: bool = False,
    ) -> Path:
        """Rebuild the document from the IR and compile it with Typst.

        Runs on a worker thread: building the source walks every block and
        copies the figure assets, and the Typst compile is a subprocess. Neither
        belongs on the event loop.

        A compile failure raises ``DocumentParseError`` rather than silently
        producing a partial file: the caller asked for a rebuilt document, and a
        missing PDF is a clearer outcome than an empty one. The generated
        ``.typ`` is left beside the artifact so the failure is inspectable.

        ``bilingual`` pairs each block with its source above the translation.
        The pairing is per block, not per page: a rebuilt document is not
        page-aligned with its source, so a page zip cannot apply (it would need
        the page counts to match, and a rebuild changes them — 41 pages in, 29
        out on the reference paper).
        """
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
        from ubt.render.typst.document_renderer import TypstDocumentRenderer

        output_path.parent.mkdir(parents=True, exist_ok=True)
        build_dir = output_path.parent / f".{output_path.stem}_typst"
        assets_root = self._assets_root(manifest)

        # The transpiler can emit a math body that parses in its own grammar but
        # does not compile in Typst (``\frac{a}{}`` -> ``frac(a, )``), and one
        # such body fails the *whole* document at the compile below. The probe
        # catches those at the point of emission, where the span degrades to
        # readable Unicode instead, so a single malformed formula costs one
        # formula rather than every page. ``check_many`` pre-resolves the whole
        # document's bodies in one invocation (a 300-formula book paid ~3.5
        # minutes of per-body Typst startups before this); ``check`` still
        # answers any body the pre-warm did not see.
        math_probe = TypstMathProbe()
        from ubt.render.page_trim import measure_trim, resolved_page_setup
        from ubt.render.typst.academic import is_academic_document, resolve_document_columns

        is_academic = is_academic_document(manifest)
        columns = resolve_document_columns(manifest)
        # The source's own trim. ``None`` for a source with no pages, and the
        # caller-side fallback is A4 -- deliberately made *here*, where it can be
        # read, rather than inside the measurement function which would then be
        # reporting a measurement it never took.
        #
        # One object rather than four floats: the page that gets emitted and the
        # page the layout audit judges must be the same page, and passing the
        # size and the margins separately is what lets them drift.
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
        from ubt.render.typst.style_profile import extract_document_style_profile

        style_profile = extract_document_style_profile(blocks, source_path=source_pdf_path)
        manifest.metadata["document_style_profile"] = style_profile.to_metadata_dict()
        renderer = TypstDocumentRenderer(
            assets_root=assets_root,
            bilingual=bilingual,
            math_gate=math_probe.check,
            math_gate_many=math_probe.check_many,
            # Language-resolved font stack, user override first. Without this
            # the preamble hard-coded Latin+CJK-SC: a Japanese target
            # typeset kanji with Simplified-Chinese glyphs and the
            # config.font_family knob was stored and never read.
            font_stack=self._font_stack_for(target_lang),
            layout_template="academic" if is_academic else "book",
            columns=columns,
            paper_title=manifest.title if is_academic else "",
            abstract=manifest.metadata.get("abstract", "") if is_academic else "",
            toc_mode=getattr(self, "toc_mode", "auto"),
            page_setup=page_setup,
            source_pdf=source_pdf_path,
            style_profile=style_profile,
        )
        source, stats = renderer.build_source(blocks, build_dir, target_lang)
        typ_path = build_dir / "main.typ"
        typ_path.write_text(source, encoding="utf-8")

        # ``target_lang`` decides whether the compile seals system fonts off:
        # the bundled set covers simplified/traditional Chinese and Latin, so
        # those render reproducibly, while a Japanese or Arabic target keeps
        # system fonts (the set carries no kana or Arabic glyphs).
        ok, stderr = typst_compile(str(typ_path), str(output_path), target_lang=target_lang)
        if not ok:
            raise DocumentParseError(
                f"publish render failed to compile: {stderr.strip() or 'unknown Typst error'} "
                f"(generated source kept at {typ_path})"
            )

        # Adaptive layout auto-healing pass: checks for physical margin spills or
        # gutter overruns and iteratively remediates through smart formula breaking and table scaling.
        if getattr(self, "enable_layout_auto_heal", True) and output_path.exists():
            from ubt.render.typst.auto_healing import LayoutAutoHealer

            # The audit's spec must describe the page that was just emitted, not
            # A4 with 2.2cm margins. The renderer owns both the emitted geometry
            # and this spec, so asking it is the only way the two cannot drift.
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
                    page_setup=page_setup,
                    source_pdf=source_pdf_path,
                    **overrides,
                )

            max_p = getattr(self, "layout_auto_heal_max_passes", 2)
            healer = LayoutAutoHealer(spec=layout_spec, max_passes=max_p)
            healed_passed, heal_report = healer.run_healing_loop(
                output_path=output_path,
                build_dir=build_dir,
                blocks=blocks,
                target_lang=target_lang,
                renderer_builder_fn=_renderer_builder,
                compile_fn=lambda t, o: typst_compile(str(t), str(o), target_lang=target_lang),
            )
            if heal_report.healed:
                manifest.metadata["auto_healing"] = heal_report.to_dict()

        # The generated markup's meaning depends on compiler behavior, so every
        # delivered PDF's quality report carries the exact compiler version.
        record_toolchain_versions(manifest)
        # The render places every block it can; the ones it could not are
        # reported here so a dropped figure or an unparseable table stays
        # visible in the quality report instead of a silently shorter document.
        # ``renderer.last_render_skips`` carries the blocks the reflow could not
        # emit at all (a table whose target grid did not parse and whose source
        # grid did not either); the asset misses are added alongside.
        #
        # The misses are keyed by **block id**, which is the only key the export
        # stage's skip channel understands. Recording the asset *path* here (as
        # this did) matched no block, so a figure that fell back to its
        # "[missing figure: …]" placeholder was invisible to the delivery
        # contract: it reported ``missing_assets=0`` and no violation for a page
        # whose picture was gone.
        self.last_render_skips = list(renderer.last_render_skips) + [
            (block_id, "asset_not_found")
            for block_id, asset in stats.assets_missing
            if block_id and asset
        ]
        # The render-quality channel: blocks that were *drawn* but imperfectly
        # (a formula degraded to readable Unicode). Kept apart from the skips so
        # the contract does not read them as source-kept.
        self.last_render_flags = list(renderer.last_render_flags)
        # Per-table/formula reconstruction verdicts (verified / corrupt) that the
        # delivery contract reads to record a rebuilt asset as RECONSTRUCTED
        # rather than the existence-only PRESERVED_OPAQUE it used to assert for
        # every non-text node. Without this the contract's RECONSTRUCTED branch
        # and both ASSET_*_RECONSTRUCTION violation kinds were unreachable.
        self.last_render_reconstructions = list(renderer.table_reconstructions)

        # Miniaturization & lossless stream compression pass
        if getattr(self, "enable_pdf_optimization", True) and output_path.exists():
            from ubt.adapters.pdf.pdf_optimizer import optimize_pdf_in_place

            opt_prof = getattr(self, "pdf_optimize_profile", "lossless")
            opt_res = optimize_pdf_in_place(output_path, profile=opt_prof)
            if opt_res.status == "optimized":
                manifest.metadata["pdf_optimization"] = opt_res.model_dump()

        return output_path

    async def _render_facsimile(
        self,
        manifest: BookManifest,
        blocks: list[IRBlock],
        target_lang: str,
        output_path: Path,
        bilingual: bool = False,
    ) -> Path:
        """Rebuild the document using 1:1 in-place facsimile (Canvas) rendering."""
        import pikepdf

        from ubt.adapters.pdf.strip_pdf_text import strip_text_streams
        from ubt.render.delivered_text import delivered_text
        from ubt.render.flow import approximate_measure, solve_flow
        from ubt.render.layout_solver import solve_page_2d
        from ubt.render.typst.artifact_guard import has_sentinel
        from ubt.render.typst.facsimile_renderer import (
            FacsimileBlockPlacement,
            FacsimilePageSpec,
            FacsimileRenderer,
        )
        from ubt.render.typst.page_locked_renderer import (
            PageLockedPageSpec,
            PageLockedReflowRenderer,
            extract_page_geometry,
            make_asset_copier,
            make_formula_cropper,
            solve_page_lock,
        )

        source_path = Path(
            str(
                getattr(manifest, "source_path", "")
                or manifest.metadata.get("source_path", "")
                or manifest.metadata.get("input_path", "")
                or ""
            )
        )
        # Facsimile delivery *needs* the source page itself: the stripped vector
        # background it draws under the translation comes from source page N.
        # Without one there is no page to be isomorphic to, so degrade to reflow
        # and say why. Deliberately INFO, not a warning and not an error: the
        # pipeline passes the effective delivery decision, and a bare strategy
        # defaults to facsimile for tests/scripts, so "no source PDF" is the
        # ordinary case for a resumed job whose source moved, not a defect.
        # Raising here would make the legacy default break every run that used
        # to work.
        #
        # ``manifest.metadata["delivery_mode"]`` is rewritten to the mode that
        # actually ran: a report claiming ``facsimile`` over a reflow artefact is
        # worse than either, because it is unfalsifiable from the outside.
        if not source_path.exists() or source_path.suffix.lower() != ".pdf":
            logger.info(
                "Delivery mode 'facsimile' needs the source PDF, which is %s; "
                "delivering this book as 'reflow' instead (pass --delivery-mode reflow "
                "to silence this and pick the mode explicitly).",
                f"missing at {source_path}" if not source_path.exists() else "not a PDF",
            )
            manifest.metadata["delivery_mode"] = "reflow"
            return await self._render_publish(manifest, blocks, target_lang, output_path, bilingual)

        build_dir = output_path.parent / f".build_{output_path.stem}_facsimile"
        build_dir.mkdir(parents=True, exist_ok=True)

        # 1. Group blocks by page, and collect the page's physical size.
        with pikepdf.open(source_path) as src_doc:
            total_pages = len(src_doc.pages)
            page_dims: list[tuple[float, float]] = []
            for page in src_doc.pages:
                box = page.CropBox if page.get("/CropBox") is not None else page.MediaBox
                # ``box`` is a pikepdf Object whose items mypy types as opaque,
                # so the subtraction is done on explicit floats rather than on
                # the two Object operands.
                pw = float(box[2]) - float(box[0])
                ph = float(box[3]) - float(box[1])
                page_dims.append((pw, ph))

        blocks_by_page: dict[int, list[IRBlock]] = {}
        render_flags: list[tuple[str, str]] = []
        for b in blocks:
            if not b.skip_translate and has_sentinel(delivered_text(b, sanitize_sentinels=False)):
                render_flags.append((b.id, "sentinel_leaked"))

            pages_for_block: list[int] = []
            if (
                b.provenance
                and b.provenance.physical_boxes
                and len(b.provenance.physical_boxes) > 1
            ):
                from ubt.model.span import boxes_from_provenance

                for pbox in boxes_from_provenance(b.provenance.physical_boxes):
                    if pbox.page > 0 and pbox.page not in pages_for_block:
                        pages_for_block.append(pbox.page)
            if not pages_for_block:
                p = 0
                if b.bbox and b.bbox.page > 0:
                    p = b.bbox.page
                elif hasattr(b.element, "span") and b.element.span and b.element.span.page > 0:
                    p = b.element.span.page
                if p > 0:
                    pages_for_block.append(p)
            for p in pages_for_block:
                blocks_by_page.setdefault(p, []).append(b)

        # 2. Decide what the background layer keeps and what the translation
        #    replaces. This is the whole correctness of the channel: the
        #    background must lose exactly the text a Typst block will re-draw,
        #    and must keep everything else — formulas, figure labels, folios.
        erase_rects, keep_rects = _facsimile_strip_plan(blocks_by_page, total_pages, source_path)

        # 3. Background vector text stripping (for L1 Canvas pages)
        is_pure_page_locked = self.delivery_mode == "page_locked"
        bg_stripped = build_dir / "bg_stripped.pdf"
        if not is_pure_page_locked:
            try:
                strip_text_streams(
                    source_path,
                    bg_stripped,
                    per_page_erase_rects=erase_rects,
                    per_page_keep_rects=keep_rects,
                    descend_forms=True,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to strip background text streams: %s; falling back to reflow", exc
                )
                manifest.metadata["delivery_mode"] = "reflow"
                return await self._render_publish(
                    manifest, blocks, target_lang, output_path, bilingual
                )

        pages_spec: list[FacsimilePageSpec] = []
        skipped: list[tuple[str, str]] = []
        reconstructions: list[ReconstructionRecord] = []
        # Reset every render: geometry describes *this* artifact, and a stale
        # page count from a previous run would be reported as this page's.
        page_geometry: list[dict[str, Any]] = []
        preserved_by_page: dict[int, list[IRBlock]] = {}

        font_stack = self._font_stack_for(target_lang)
        # Both channels can hold L2 page-locked pages, and neither gives them a
        # background layer to inherit artwork from. The copier resolves the
        # extracted asset file the parser recorded as ``source_text`` into a
        # build-relative name so ``image()`` can address it.
        asset_copier = make_asset_copier(build_dir, self._assets_root(manifest))
        formula_cropper = make_formula_cropper(build_dir, source_path)
        from ubt.render.typst.style_profile import extract_document_style_profile

        style_profile = extract_document_style_profile(
            blocks, source_path=source_path, page_dims=page_dims
        )
        manifest.metadata["document_style_profile"] = style_profile.to_metadata_dict()

        for pno in range(1, total_pages + 1):
            pw, ph = page_dims[pno - 1] if pno <= len(page_dims) else (595.0, 842.0)
            page_blocks = blocks_by_page.get(pno, [])
            placements: list[FacsimileBlockPlacement] = []

            metrics = _page_metrics(source_path, pno)
            source_size_pt = metrics.size_pt if metrics else 9.0
            source_leading_pt = metrics.leading_pt if metrics else source_size_pt * 1.333

            records_by_column: dict[int, list[dict[str, Any]]] = {}
            obstacles_by_column: dict[int, list[float]] = {}
            page_preserved: list[IRBlock] = []

            for b in page_blocks:
                if not b.bbox:
                    continue
                w = max(0.0, b.bbox.x1 - b.bbox.x0)
                h = max(0.0, b.bbox.y1 - b.bbox.y0)
                if w <= 0 or h <= 0:
                    continue

                target_content = delivered_text(b)
                if not target_content:
                    target_content = b.source_text

                is_hdr = b.block_type == BlockType.HEADING
                style_color = (
                    getattr(b.style, "color_hex", None) or getattr(b.style, "text_color_hex", None)
                    if b.style
                    else None
                )
                color = (
                    style_color
                    if style_color
                    else (style_profile.primary_heading_hex if is_hdr else "#000000")
                )
                is_bold = getattr(b.style, "is_bold", False) if b.style else False
                weight = "bold" if is_hdr or is_bold else "regular"
                block_size = _block_font_size(b, is_hdr=is_hdr, body_size=source_size_pt)

                # A block that is not re-drawn keeps its own box protected in the
                # background layer: a formula or figure must survive untouched.
                skip_render = b.skip_translate or b.block_type in (
                    BlockType.FORMULA,
                    BlockType.IMAGE,
                )
                if skip_render:
                    # The background layer carries this block's ink; it draws no
                    # translated text and must not enter the 2D solver's column
                    # records as a text box. However, its top edge IS a vertical
                    # obstacle for any translated prose box above it in the same
                    # column (or full-width across both columns), so the solver's
                    # vertical glue must clamp at ``b.bbox.y1``.
                    page_preserved.append(b)
                    if b.bbox.page == pno:
                        obs_top = float(b.bbox.y1)
                        if w > 0.65 * pw:
                            obstacles_by_column.setdefault(0, []).append(obs_top)
                            obstacles_by_column.setdefault(1, []).append(obs_top)
                        else:
                            obs_mid_x = (b.bbox.x0 + b.bbox.x1) / 2.0
                            obs_col = 0 if obs_mid_x < pw * 0.50 else 1
                            obstacles_by_column.setdefault(obs_col, []).append(obs_top)
                    continue

                bx0 = float(b.bbox.x0)
                by0 = float(b.bbox.y0)
                bx1 = float(b.bbox.x1)
                by1 = float(b.bbox.y1)

                # Multi-region flow handling:
                # If a block spans multiple disjoint regions (different columns or different pages),
                # flow text across those region containers. If all physical boxes are in the same
                # column on the same page, they are consecutive lines within a single block and
                # must not be split into line-by-line fragments.
                if (
                    b.provenance
                    and b.provenance.physical_boxes
                    and len(b.provenance.physical_boxes) > 1
                ):
                    from ubt.model.span import PhysicalBox, boxes_from_provenance

                    raw_boxes = boxes_from_provenance(b.provenance.physical_boxes)

                    distinct_regions = {_physical_box_region(bx, pw) for bx in raw_boxes}
                    if len(distinct_regions) > 1:
                        grouped_boxes: list[PhysicalBox] = []
                        curr_reg = None
                        curr_page = 0
                        ux0, uy0, ux1, uy1 = 0.0, 0.0, 0.0, 0.0
                        for bx in raw_boxes:
                            reg = _physical_box_region(bx, pw)
                            if reg != curr_reg:
                                if curr_reg is not None:
                                    grouped_boxes.append(
                                        PhysicalBox.of(curr_page, (ux0, uy0, ux1, uy1))
                                    )
                                curr_reg = reg
                                curr_page = int(bx.page)
                                ux0, uy0, ux1, uy1 = (
                                    float(bx.bbox[0]),
                                    float(bx.bbox[1]),
                                    float(bx.bbox[2]),
                                    float(bx.bbox[3]),
                                )
                            else:
                                ux0 = min(ux0, float(bx.bbox[0]))
                                uy0 = min(uy0, float(bx.bbox[1]))
                                ux1 = max(ux1, float(bx.bbox[2]))
                                uy1 = max(uy1, float(bx.bbox[3]))
                        if curr_reg is not None:
                            grouped_boxes.append(PhysicalBox.of(curr_page, (ux0, uy0, ux1, uy1)))

                        if any(gbox.page == pno for gbox in grouped_boxes):
                            measure = approximate_measure(
                                font_size_pt=block_size or source_size_pt,
                                leading_pt=source_leading_pt,
                            )
                            flowed = solve_flow(target_content, grouped_boxes, measure)
                            for fl_idx, fl_pl in enumerate(flowed):
                                if fl_pl.box.page != pno or not fl_pl.text.strip():
                                    continue
                                fl_x0, fl_y0, fl_x1, fl_y1 = fl_pl.box.bbox
                                fl_w = max(0.0, fl_x1 - fl_x0)
                                fl_h = max(0.0, fl_y1 - fl_y0)
                                if fl_w <= 0 or fl_h <= 0:
                                    continue
                                fl_mid_x = (fl_x0 + fl_x1) / 2.0
                                fl_col = 0 if fl_w > 0.65 * pw or fl_mid_x < pw * 0.50 else 1
                                fl_ascii = sum(1 for c in fl_pl.text if ord(c) < 128)
                                fl_aspect = (len(fl_pl.text) - fl_ascii + fl_ascii * 0.50) / max(
                                    1, len(fl_pl.text)
                                )
                                records_by_column.setdefault(fl_col, []).append(
                                    {
                                        "id": f"{b.id}_flow_{fl_idx}",
                                        "top": fl_y1,
                                        "bot": fl_y0,
                                        "w": fl_w,
                                        "x0": fl_x0,
                                        "x1": fl_x1,
                                        "n": max(1, round(fl_h / source_leading_pt)),
                                        "glyphs": len(fl_pl.text.strip()),
                                        "glyph_aspect": fl_aspect,
                                    }
                                )
                                placements.append(
                                    FacsimileBlockPlacement(
                                        block_id=f"{b.id}_flow_{fl_idx}",
                                        page=pno,
                                        x0=fl_x0,
                                        y0=fl_y0,
                                        x1=fl_x1,
                                        y1=fl_y1,
                                        text=fl_pl.text,
                                        is_heading=is_hdr,
                                        heading_level=getattr(b.element, "level", 1)
                                        if is_hdr
                                        else 1,
                                        text_color_hex=color,
                                        font_weight=weight,
                                        font_size_pt=block_size,
                                    )
                                )
                            continue
                    else:
                        same_page_boxes = [bx for bx in raw_boxes if bx.page == pno]
                        if same_page_boxes:
                            bx0 = min(float(bx.bbox[0]) for bx in same_page_boxes)
                            by0 = min(float(bx.bbox[1]) for bx in same_page_boxes)
                            bx1 = max(float(bx.bbox[2]) for bx in same_page_boxes)
                            by1 = max(float(bx.bbox[3]) for bx in same_page_boxes)
                            w = max(0.0, bx1 - bx0)
                            h = max(0.0, by1 - by0)

                # Heading font size scaling: ensure multi-line heading fits in its allocated vertical box
                if is_hdr and block_size:
                    glyphs = len(target_content.strip())
                    needed_lines = math.ceil((glyphs * block_size) / max(10.0, w))
                    needed_h = needed_lines * block_size * 1.25
                    if needed_h > h + 2.0:
                        fitted_size = math.sqrt((w * h) / (max(1, glyphs) * 1.25))
                        block_size = max(
                            source_size_pt * 1.1, min(block_size, round(fitted_size, 2))
                        )

                # Column bucket for the 2D solver: a left-column paragraph must
                # not be bounded by where the right column happens to start.
                mid_x = (bx0 + bx1) / 2.0
                column = 0 if w > 0.65 * pw or mid_x < pw * 0.50 else 1
                ascii_count = sum(1 for c in target_content if ord(c) < 128)
                glyph_aspect = (len(target_content) - ascii_count + ascii_count * 0.50) / max(
                    1, len(target_content)
                )
                records_by_column.setdefault(column, []).append(
                    {
                        "id": b.id,
                        "top": by1,
                        "bot": by0,
                        "w": w,
                        "x0": bx0,
                        "x1": bx1,
                        "n": max(1, round(h / source_leading_pt)),
                        "glyphs": len(target_content.strip()),
                        "glyph_aspect": glyph_aspect,
                    }
                )

                is_table = b.block_type == BlockType.TABLE
                if is_table:
                    reconstructions.append(
                        ReconstructionRecord(
                            block_id=b.id, verified=False, corrupt=False, reason="facsimile_table"
                        )
                    )

                placements.append(
                    FacsimileBlockPlacement(
                        block_id=b.id,
                        page=pno,
                        x0=bx0,
                        y0=by0,
                        x1=bx1,
                        y1=by1,
                        text=target_content,
                        is_heading=is_hdr,
                        heading_level=getattr(b.element, "level", 1) if is_hdr else 1,
                        text_color_hex=color,
                        font_weight=weight,
                        font_size_pt=block_size,
                        is_table=is_table,
                        is_code=b.block_type == BlockType.CODE,
                    )
                )

            for col_idx, col_recs in records_by_column.items():
                col_obstacles = obstacles_by_column.get(col_idx, [])
                if not col_obstacles:
                    continue
                for rec in col_recs:
                    below_obs = [
                        obs_top for obs_top in col_obstacles if obs_top <= float(rec["bot"]) + 2.0
                    ]
                    if below_obs:
                        rec["obstacle_top"] = max(below_obs)

            columns = [
                sorted(records_by_column[k], key=lambda r: float(r["top"]), reverse=True)
                for k in sorted(records_by_column)
            ] or [[]]
            sol = solve_page_2d(
                columns,
                source_size_pt=source_size_pt,
                source_leading_pt=source_leading_pt,
            )

            # If layout failed to fit due to only 1-2 bottleneck containers (and no source overlap),
            # attempt controlled concise repair before falling back to L2
            # (docs/design/adaptive_delivery_prd.md §8.6; old §6.2).
            if (
                not is_pure_page_locked
                and not sol.fits
                and not sol.source_overlap
                and 1 <= len(sol.bottleneck_seg_ids) <= 2
            ):
                import dataclasses

                from ubt.render.typst.container_repair import (
                    ContainerFitSpec,
                    compute_container_char_budget,
                    extract_locked_numbers,
                    repair_bottleneck_containers,
                )

                rec_by_id = {str(r.get("id")): r for col in columns for r in col if r.get("id")}
                placement_by_id = {plc.block_id: plc for plc in placements}

                fit_specs: list[ContainerFitSpec] = []
                for b_id in sol.bottleneck_seg_ids:
                    r = rec_by_id.get(b_id)
                    plc = placement_by_id.get(b_id)
                    if not r or not plc:
                        continue
                    budget = compute_container_char_budget(
                        container_width_pt=float(r["w"]),
                        max_lines=max(1, int(r.get("n", 1))),
                        floor_pt=7.5,
                        target_lang=target_lang,
                    )
                    nums = extract_locked_numbers(plc.text)
                    fit_specs.append(
                        ContainerFitSpec(
                            seg_id=b_id,
                            current_translation=plc.text,
                            max_target_chars=budget,
                            locked_numbers=nums,
                        )
                    )

                if fit_specs:
                    repaired_texts = repair_bottleneck_containers(
                        fit_specs,
                        rewriter=getattr(self, "container_concise_rewriter", None),
                    )
                    if repaired_texts:
                        for rep_id, rep_text in repaired_texts.items():
                            if rep_id in rec_by_id:
                                rec_by_id[rep_id]["glyphs"] = len(rep_text.strip())
                        placements = [
                            dataclasses.replace(plc_item, text=repaired_texts[plc_item.block_id])
                            if plc_item.block_id in repaired_texts
                            else plc_item
                            for plc_item in placements
                        ]
                        sol = solve_page_2d(
                            columns,
                            source_size_pt=source_size_pt,
                            source_leading_pt=source_leading_pt,
                        )
                        for rep_id in repaired_texts:
                            render_flags.append((rep_id, "container_fit_repaired"))
                        repaired_meta = manifest.metadata.setdefault(
                            "container_fit_repaired_segments", []
                        )
                        if isinstance(repaired_meta, list):
                            repaired_meta.extend(repaired_texts.keys())

            use_l2 = (
                is_pure_page_locked
                or sol.delivery_tier == "L2"
                or sol.source_overlap
                or (not sol.fits and sol.uniform_size_pt <= 7.5)
            )

            preserved_by_page[pno] = list(page_preserved)
            if use_l2:
                for pb in page_preserved:
                    if pb.block_type == BlockType.IMAGE:
                        rel_asset = (
                            asset_copier(pb.source_text.strip())
                            if pb.source_text and pb.source_text.strip()
                            else None
                        )
                        if not rel_asset:
                            skipped.append((pb.id, "asset_not_found"))
                    elif pb.skip_translate:
                        skipped.append((pb.id, "skip_translate"))
                geom = extract_page_geometry(
                    pw, ph, page_blocks, default_theme_color=style_profile.primary_heading_hex
                )
                l2_size, l2_leading, l2_spacing, l2_tracking = solve_page_lock(
                    geom, page_blocks, target_lang=target_lang
                )
                l2_page = PageLockedPageSpec(
                    page_no=pno,
                    geometry=geom,
                    uniform_size_pt=l2_size,
                    leading_em=l2_leading,
                    par_spacing_em=l2_spacing,
                    tracking_em=l2_tracking,
                    blocks=page_blocks,
                )
                pages_spec.append(
                    FacsimilePageSpec(
                        page_no=pno,
                        page_w=round(pw, 2),
                        page_h=round(ph, 2),
                        uniform_size_pt=l2_size,
                        leading_em=l2_leading,
                        tracking_em=l2_tracking,
                        placements=placements,
                        tier="L2",
                        l2_spec=l2_page,
                    )
                )
                page_geometry.append(
                    {
                        "page_no": pno,
                        "page_w_pt": round(pw, 2),
                        "page_h_pt": round(ph, 2),
                        "measured_size_pt": round(source_size_pt, 3),
                        "measured_leading_pt": round(source_leading_pt, 3),
                        "delivered_size_pt": round(l2_size, 3),
                        "delivered_leading_em": round(l2_leading, 4),
                        "tracking_em": round(l2_tracking, 4),
                        "tier": "L2",
                        "fits": bool(sol.fits),
                    }
                )
            else:
                for pb in page_preserved:
                    skipped.append((pb.id, "facsimile_preserved_opaque"))
                pages_spec.append(
                    FacsimilePageSpec(
                        page_no=pno,
                        page_w=round(pw, 2),
                        page_h=round(ph, 2),
                        uniform_size_pt=sol.uniform_size_pt,
                        leading_em=sol.leading_em,
                        tracking_em=sol.tracking_em,
                        placements=placements,
                        tier="L1",
                    )
                )
                page_geometry.append(
                    {
                        "page_no": pno,
                        "page_w_pt": round(pw, 2),
                        "page_h_pt": round(ph, 2),
                        "measured_size_pt": round(source_size_pt, 3),
                        "measured_leading_pt": round(source_leading_pt, 3),
                        "delivered_size_pt": round(sol.uniform_size_pt, 3),
                        "delivered_leading_em": round(sol.leading_em, 4),
                        "tracking_em": round(sol.tracking_em, 4),
                        "tier": "L1",
                        "fits": bool(sol.fits),
                    }
                )

        # The transpile gate, same contract as ``_render_publish_sync`` above:
        # both facsimile channels now route their text through ``render_prose``,
        # which converts a model's LaTeX ``$…$`` span into Typst math. Typst
        # compiles this document in ONE invocation, so a body the transpiler
        # accepts but Typst rejects (``\frac{a}{}`` -> ``frac(a, )``) takes every
        # page with it. The probe answers at emission time so the span degrades to
        # readable Unicode instead.
        #
        # Built here rather than reused from ``_render_publish_sync`` because that
        # probe is local to a different channel's scope: the facade's two render
        # paths never see each other's cache.
        #
        # Pre-warmed in one batched call: an un-warmed body pays a ~0.7s Typst
        # startup per span, and this document shape is math-dense by nature (the
        # measured 26-page BSIM-CMG chapter carries 86 distinct inline bodies).
        # Bodies come from every block the L2 pages would gate AND every L1
        # placement's text, because a placement is a *flowed* fragment of a block
        # and a table's cells are only ever gated at the string level.
        from ubt.adapters.pdf.typst_math_probe import TypstMathProbe
        from ubt.render.typst.document_renderer import (
            math_bodies_for_blocks,
            math_bodies_for_prose,
        )

        math_probe = TypstMathProbe()
        pending_bodies = math_bodies_for_blocks(blocks)
        for spec in pages_spec:
            for pl in spec.placements:
                pending_bodies.extend(math_bodies_for_prose(pl.text))
        math_probe.check_many(pending_bodies)

        if is_pure_page_locked:
            l2_renderer = PageLockedReflowRenderer(
                font_stack=font_stack, math_gate=math_probe.check
            )
            l2_pages = [p.l2_spec for p in pages_spec if p.l2_spec is not None]
            l2_renderer.render_document(
                l2_pages,
                output_path,
                build_dir,
                target_lang=target_lang,
                asset_copier=asset_copier,
                formula_cropper=formula_cropper,
            )
            manifest.metadata["delivery_mode"] = "page_locked"
        else:
            renderer = FacsimileRenderer(font_stack=font_stack, math_gate=math_probe.check)
            renderer.render_document(
                pages_spec,
                bg_stripped,
                output_path,
                build_dir,
                target_lang=target_lang,
                asset_copier=asset_copier,
                formula_cropper=formula_cropper,
            )
            # §8.8 / §9.1: Single-page automatic L1 -> L2 reroute on ink collision.
            # When an L1 page's translated text lands on unstripped background
            # rules, vector diagrams, or residual ink above CANVAS_COLLISION_RATIO,
            # upgrade only the colliding page(s) to L2 page-locked reflow and
            # recompile once before the artifact ships.
            l1_candidate_pages = [
                spec.page_no for spec in pages_spec if spec.tier == "L1" and spec.placements
            ]
            if l1_candidate_pages and output_path.exists() and bg_stripped.exists():
                from ubt.adapters.pdf.visual_gate import detect_colliding_l1_pages

                l1_rois = {
                    spec.page_no: [
                        (
                            round(pl.x0, 2),
                            round(spec.page_h - pl.y1, 2),
                            round(max(10.0, pl.x1 - pl.x0), 2),
                            round(max(10.0, pl.y1 - pl.y0), 2),
                        )
                        for pl in spec.placements
                        if not pl.skip_render and pl.text.strip()
                    ]
                    for spec in pages_spec
                    if spec.tier == "L1"
                }

                colliding = await asyncio.to_thread(
                    detect_colliding_l1_pages,
                    bg_stripped,
                    output_path,
                    l1_candidate_pages,
                    page_rois=l1_rois,
                )
                if colliding:
                    logger.warning(
                        "L1 ink collision detected on pages %s; auto-rerouting to Tier L2 page-locked reflow",
                        {p: f"{r:.1%}" for p, r in sorted(colliding.items())},
                    )
                    for pno in sorted(colliding):
                        idx = pno - 1
                        if idx < 0 or idx >= len(pages_spec):
                            continue
                        old_spec = pages_spec[idx]
                        page_blocks = blocks_by_page.get(pno, [])
                        pw, ph = old_spec.page_w, old_spec.page_h
                        geom = extract_page_geometry(
                            pw,
                            ph,
                            page_blocks,
                            default_theme_color=style_profile.primary_heading_hex,
                        )
                        l2_size, l2_leading, l2_spacing, l2_tracking = solve_page_lock(
                            geom, page_blocks, target_lang=target_lang
                        )
                        l2_page = PageLockedPageSpec(
                            page_no=pno,
                            geometry=geom,
                            uniform_size_pt=l2_size,
                            leading_em=l2_leading,
                            par_spacing_em=l2_spacing,
                            tracking_em=l2_tracking,
                            blocks=page_blocks,
                        )
                        pages_spec[idx] = FacsimilePageSpec(
                            page_no=pno,
                            page_w=pw,
                            page_h=ph,
                            uniform_size_pt=l2_size,
                            leading_em=l2_leading,
                            tracking_em=l2_tracking,
                            placements=old_spec.placements,
                            tier="L2",
                            l2_spec=l2_page,
                        )
                        if idx < len(page_geometry):
                            page_geometry[idx].update(
                                {
                                    "delivered_size_pt": round(l2_size, 3),
                                    "delivered_leading_em": round(l2_leading, 4),
                                    "tracking_em": round(l2_tracking, 4),
                                    "tier": "L2",
                                }
                            )
                        for pb in preserved_by_page.get(pno, []):
                            pair_opaque = (pb.id, "facsimile_preserved_opaque")
                            if pair_opaque in skipped:
                                skipped.remove(pair_opaque)
                            if pb.skip_translate:
                                skipped.append((pb.id, "skip_translate"))
                            elif pb.block_type == BlockType.IMAGE:
                                rel_asset = (
                                    asset_copier(pb.source_text.strip())
                                    if pb.source_text and pb.source_text.strip()
                                    else None
                                )
                                if not rel_asset:
                                    skipped.append((pb.id, "asset_not_found"))
                    renderer.render_document(
                        pages_spec,
                        bg_stripped,
                        output_path,
                        build_dir,
                        target_lang=target_lang,
                        asset_copier=asset_copier,
                        formula_cropper=formula_cropper,
                    )
                    manifest.metadata["l1_ink_collision_rerouted_pages"] = sorted(colliding)
            manifest.metadata["delivery_mode"] = "facsimile"
            # Hand the stripped background to the visual gate. Without it the
            # ink-collision tier cannot run, and "the L1 channel was verified"
            # becomes an assumption nobody can falsify: every other tier reads
            # text rects and glyph heights, and none of them can see a translated
            # line landing on a table rule the stripper kept.
            #
            # Recorded as a path rather than passed through the adapter because
            # the gate runs in ``VisualVerifyLoop``, a later stage that holds the
            # manifest and nothing else from this call. It lives under the same
            # hidden ``.build_*`` directory as the renderer's Typst source, which
            # already survives the render, so this adds no new artifact to the
            # delivery listing. The gate checks the path exists before using it
            # and reports a skipped tier — never an error — when it does not.
            manifest.metadata["background_pdf"] = str(bg_stripped)

        record_toolchain_versions(manifest)
        # The delivery contract reads these to record preserved assets and to
        # raise ASSET_MISSING / ASSET_*_RECONSTRUCTION. The facsimile path used to
        # report all three empty, which made those violation kinds unreachable.
        self.last_render_skips = skipped
        self.last_render_flags = render_flags
        self.last_render_reconstructions = reconstructions
        self.last_page_geometry = page_geometry

        # Miniaturization & lossless stream compression pass
        if getattr(self, "enable_pdf_optimization", True) and output_path.exists():
            from ubt.adapters.pdf.pdf_optimizer import optimize_pdf_in_place

            opt_prof = getattr(self, "pdf_optimize_profile", "lossless")
            opt_res = optimize_pdf_in_place(output_path, profile=opt_prof)
            if opt_res.status == "optimized":
                manifest.metadata["pdf_optimization"] = opt_res.model_dump()

        # Chapter bookmarks, written *after* the optimization pass so the
        # outline belongs to the file that ships (an optimizer that rewrites the
        # object graph is exactly the step that could drop a tree written
        # before it). The source book had a working sidebar and the translation
        # had none — the reader's only way back to chapter 7 was scrolling.
        #
        # Only these two channels qualify. Both emit exactly one output page per
        # source page (``range(1, total_pages + 1)`` above), so a heading's
        # measured ``bbox.page`` names the delivered page. The reflow channel's
        # page count tracks the target language's length instead, so the same
        # page number would point at the wrong place; ``write_outline_from_blocks``
        # checks the invariant against ``expected_pages`` and writes nothing when
        # it does not hold, rather than mislabelling the navigation.
        outline_written = 0
        if output_path.exists():
            from ubt.adapters.pdf.outline_writer import write_outline_from_blocks

            outline_written = write_outline_from_blocks(
                output_path, blocks, expected_pages=total_pages
            )
        if outline_written:
            manifest.metadata["pdf_outline_entries"] = outline_written

        return output_path

    def _font_stack_for(self, target_lang: str) -> tuple[str, ...]:
        """Body/heading font stack for the preamble, resolved against this machine.

        ``language_profile.resolve_font_config`` supplies the per-language
        stack (ja -> Noto Serif CJK JP, ko -> KR, ar -> Naskh, zh -> SC, ...);
        ``font_probe`` then prunes it to the families Typst can actually
        resolve. Without the probe every absent spare face warned on every
        compile (arXiv 2608.25512 reported ``unknown font family: source han
        serif sc`` three times) and a stack with no face for the target script
        shipped tofu silently. ``config.font_family`` — the knob the preamble
        ignored for the whole life of the field — leads as a protected entry:
        an explicit choice is never silently replaced.
        """
        from ubt.adapters.pdf.font_probe import resolve_for_render

        return resolve_for_render(target_lang, font_override=self.font_family).families

    def _assets_root(self, manifest: BookManifest) -> Path | None:
        """Directory holding the extracted figure assets for this job.

        The PDF adapters cache assets under a content-addressed directory beside
        the source; the manifest does not carry it, so it is derived the same way
        the parser does (``docling_asset_dir``). A manifest whose source is gone
        yields ``None`` and the renderer falls back to absolute paths.
        """
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

        A PDF target is rebuilt from the IR and laid out by Typst. Non-PDF
        targets (.md / .txt) export clean Markdown.
        """
        self.last_render_skips = []
        self.last_render_flags = []
        # Reset alongside the other two channels: the per-render verdicts are
        # copied from the freshly built renderer below, but a second render on
        # the same strategy must not report the first render's verdicts if it
        # fails before that copy runs.
        self.last_render_reconstructions = []

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
            # Bilingual is a per-block pairing, so the mode only has to answer
            # "pair or not". ``inline`` and ``alternating`` both mean "pair";
            # ``facing`` used to mean a page-zip against the source, which the
            # renderer cannot do (it changes the page count), so it now
            # pairs per block like the others.
            publish_bilingual = active_mode != "monolingual"
            if self.delivery_mode in ("facsimile", "page_locked"):
                return await self._render_facsimile(
                    manifest, list(blocks), target_lang, out_path, publish_bilingual
                )
            return await self._render_publish(
                manifest, list(blocks), target_lang, out_path, publish_bilingual
            )

        # Non-PDF export: Markdown (.md / .markdown) or flat text (.txt)
        if out_path.suffix.lower() in (".md", ".markdown", ".txt"):
            # The single owner of "what a block delivers": target_text can carry
            # quarantine / failed-repair ``<mark>`` wrappers, and the PDF
            # renderer and companion views unwrap them — without this the
            # wrappers reach the .md deliverable as literal markup.
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
