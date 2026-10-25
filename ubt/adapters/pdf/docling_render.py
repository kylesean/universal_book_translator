"""Render strategy for the Docling adapter.

Turns translated ``IRBlock``s into a delivered artifact: a PDF rebuilt from the
IR and laid out by Typst, or a Markdown/flat-text export. Holds only delivery
collaborators, so the adapter can construct it once and forward ``render_blocks``
to it.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from ubt.adapters.pdf.typst_compile import typst_version
from ubt.core.content.nodes import ReconstructionRecord
from ubt.core.exceptions import DocumentParseError
from ubt.core.ir.models import BlockType, BookManifest, IRBlock
from ubt.core.ir.render_plan import RenderOutcome, RenderPlan

if TYPE_CHECKING:
    pass

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


class DoclingRenderStrategy:
    """Delivery-leg collaborators plus the last render's skip side channel."""

    def __init__(
        self,
        *,
        font_family: str | None = None,
    ) -> None:
        self.font_family = font_family
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
        from ubt.render.typst.academic import is_academic_document, resolve_document_columns

        is_academic = is_academic_document(manifest)
        columns = resolve_document_columns(manifest)
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
