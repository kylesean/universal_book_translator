"""Spend-the-tokens gate: compile a sample before the draft stage bills.

`README.md`'s dependency table warned that a broken Typst toolchain kills a PDF
job at Stage 6 — after every token of the whole book has been spent. That is
not hypothetical: ledger job `fc1d7bd7b799` shows 209 blocks translated in 3m03s
followed by a Typst compile failure and zero output.

This module runs the *real* PDF render path over a handful of source-text
blocks, before Stage 3, so toolchain-shaped failures (missing binary, version
drift, generated markup the compiler rejects) exit non-zero at zero token cost.
It is not a full rehearsal: defects that only appear once translated text
exists (e.g. the `#box[...](...)` code-mode crash fixed by a98a315) still
surface only at export — what changes is that a book cannot finish
translation only to discover the renderer itself is dead.
"""

from __future__ import annotations

import copy
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any

from ubt.core.exceptions import DocumentParseError
from ubt.core.ir.models import BlockType, IRBlock
from ubt.core.ir.render_plan import RenderPlan
from ubt.core.policy.adaptive_policy import resolve_pdf_engine
from ubt.core.ports import DocumentAdapter, get_last_render_skips

logger = logging.getLogger(__name__)

# Enough blocks to include an inline-math one and a structure block without
# turning the rehearsal into a second full render.
PREFLIGHT_MAX_BLOCKS = 5

_SAMPLE_PREFERENCE = (
    BlockType.FORMULA,
    BlockType.TABLE,
    BlockType.HEADING,
)


def _sample_key(block: IRBlock) -> tuple[int, int, int]:
    """Rank blocks so the sample exercises the most markup-sensitive paths.

    Blocks the compositor can never draw above the source (tables, figures,
    bbox-less blocks) rank last: ``overlays_from_blocks`` excludes them, so a
    sample made only of them renders Layer 0 alone and the pre-flight never
    reaches the typesetter — a dead Typst toolchain would pass silently. Within
    the drawable set, inline math and structure blocks come first.
    """
    drawable = (
        block.bbox is not None
        and block.bbox.page > 0
        and block.block_type not in (BlockType.TABLE, BlockType.IMAGE)
    )
    inline_math = "$" in (block.source_text or "")
    try:
        type_rank = _SAMPLE_PREFERENCE.index(block.block_type)
    except ValueError:
        type_rank = len(_SAMPLE_PREFERENCE)
    return (0 if drawable else 1, 0 if inline_math else 1, type_rank)


def select_preflight_sample(
    blocks: list[IRBlock], limit: int = PREFLIGHT_MAX_BLOCKS
) -> list[IRBlock]:
    """Pick a small, markup-diverse sample in reading order.

    Ranking picks the sensitive blocks first; re-sorting by spine index keeps
    the emitted document ordered so a failure reproduces the real layout path.
    """
    picked = sorted(blocks, key=_sample_key)[:limit]
    return sorted(picked, key=lambda b: b.spine_index)


def _placeholder_text(block: IRBlock) -> str:
    """The rehearsal body for one sample block: its source, or a stand-in.

    A formula's source is only typesettable when ``typeset_math`` can convert
    it; the extractor represents a formula it could not read with a bare ``$$``
    (see ``docling_parser``), and ``typstify_math`` rejects that empty body, so
    ``typeset_math`` returns ``None`` for reasons that have nothing to do with
    the toolchain. The real render never draws those placeholders either (they
    are static skips), so substituting a synthetic math body keeps the gate on
    the math path instead of failing the whole book on markup nobody renders.
    """
    text = (block.source_text or "").strip()
    if block.block_type == BlockType.FORMULA:
        from ubt.adapters.pdf.overlay_text import typstify_math  # noqa: PLC0415
        from ubt.render.outputs import _strip_math_delimiters  # noqa: PLC0415

        if typstify_math(_strip_math_delimiters(text)) is None:
            return "preflight"
    return text or "preflight"


def _placeholder_blocks(sample: list[IRBlock]) -> list[IRBlock]:
    """The sample carrying each source text as its placeholder translation.

    The compositor only overlays a block that already holds target text; a
    source-only sample renders Layer 0 alone and never reaches the typesetter,
    which would make this gate blind to a dead Typst toolchain. Substituting the
    source text keeps the rehearsal on the real render path and hands the
    compiler the markup (math, escapes) it has to accept.

    ``skip_translate`` is cleared too: a formula/table sample is a static skip
    by policy, and ``overlays_from_blocks`` never draws a skip block, so without
    this the substituted text would still be excluded and the gate would stay
    blind exactly on the formula/table-heavy books it most needs to check.
    """
    return [
        block.model_copy(
            update={
                "target_text": _placeholder_text(block),
                "skip_translate": False,
            }
        )
        for block in sample
    ]


def _isolated_manifest(manifest: Any) -> Any:
    """Return a throwaway copy so a scratch render cannot mutate the run.

    ``render_blocks`` records effective-mode facts (notably the rigid
    monolingual downgrade) on the manifest it is handed. The pre-flight sample
    can route differently from the full document — the 5-block sample prefers
    structure blocks, so it reaches the ``struct_share`` cutoff sooner — and
    rendering into the live manifest let a scratch compile permanently
    downgrade a requested bilingual run to monolingual.
    """
    model_copy = getattr(manifest, "model_copy", None)
    if callable(model_copy):
        return model_copy(deep=True)
    return copy.deepcopy(manifest)


async def run_render_preflight(
    *,
    adapter: DocumentAdapter,
    manifest: Any,
    blocks: list[IRBlock],
    target_lang: str,
    render_plan: RenderPlan | None = None,
) -> None:
    """Render a source-text sample through the adapter's real PDF path.

    Raises DocumentParseError with operator-facing guidance when the render
    fails; a non-PDF adapter or an empty sample skips, because text-only
    outputs never touch Typst. Callers run this before any billable stage.
    """
    if adapter.engine_name is None or not blocks:
        return

    sample = select_preflight_sample(blocks)
    scratch_blocks = _placeholder_blocks(sample)
    metadata = getattr(manifest, "metadata", None) or {}
    # The render decision is the plan's (compiler render plan protocol); the
    # manifest metadata copy is a fallback for callers that pass no plan.
    # Resolve the engine against ALL blocks, then force that engine onto the
    # sample. ``select_preflight_sample`` is deliberately structure-biased
    # (TABLE/FORMULA first), so resolving on the sample alone could route rigid
    # while the full document reflows: the scratch render would exercise a path
    # that never runs, and log a monolingual-downgrade warning for it.
    requested_engine = (
        (render_plan.render_engine if render_plan is not None else None)
        or metadata.get("render_engine")
        or "publication"
    )
    render_engine = resolve_pdf_engine(str(requested_engine), blocks, manifest=manifest)
    if render_plan is not None:
        bilingual_mode = (
            render_plan.effective_dual_mode or render_plan.bilingual_mode
        ) or metadata.get("bilingual_mode")
    else:
        bilingual_mode = metadata.get("bilingual_mode")
    # Render into an isolated copy: the scratch compile must not write its
    # route/downgrade facts back onto the live run manifest.
    scratch_manifest = _isolated_manifest(manifest)
    if hasattr(scratch_manifest, "metadata"):
        if isinstance(scratch_manifest.metadata, dict):
            scratch_manifest.metadata["suppress_render_engine_warning"] = True
        elif scratch_manifest.metadata is None:
            scratch_manifest.metadata = {"suppress_render_engine_warning": True}

    tmp_dir = Path(tempfile.mkdtemp(prefix="ubt-preflight-"))
    preflight_path = tmp_dir / "preflight.pdf"
    try:
        await adapter.render_blocks(
            manifest=scratch_manifest,
            blocks=scratch_blocks,
            target_lang=target_lang,
            output_path=preflight_path,
            bilingual_mode=bilingual_mode,
            render_engine=render_engine,
            render_plan=render_plan,
        )
        # LayerCompositor records a failed fragment as a skip and keeps the
        # source instead of raising, so a dead toolchain would still exit zero.
        # The preflight must fail when nothing at all was drawn, or a broken
        # Typst only surfaces after the whole book has been billed.
        from ubt.render.outputs import overlays_from_blocks

        expected = overlays_from_blocks(scratch_blocks, None)
        skips = get_last_render_skips(adapter)
        if expected and len(skips) >= len(expected):
            raise DocumentParseError(
                f"Render pre-flight could not typeset any of the {len(expected)} "
                f"sample fragment(s): {skips[:3]}"
            )
    except DocumentParseError as exc:
        raise DocumentParseError(
            "Render pre-flight failed before any translation was billed "
            f"({len(sample)} sample block(s) of {len(blocks)}): {exc}\n"
            "The configured PDF renderer could not compile a source-text "
            "sample. Check `ubt doctor` (Typst row) and the Typst binary "
            "setting, or switch with --render-engine; nothing has been sent "
            "to the model yet. The rejected scratch sample was removed."
        ) from exc
    finally:
        # A renderer may raise an unexpected exception (not only
        # DocumentParseError); scratch artifacts must not accumulate in either
        # path, especially on long-lived workers.
        shutil.rmtree(tmp_dir, ignore_errors=True)
    logger.info(
        "Render pre-flight passed on %d sample block(s); renderer is live "
        "before translation is billed",
        len(sample),
    )


__all__ = [
    "PREFLIGHT_MAX_BLOCKS",
    "run_render_preflight",
    "select_preflight_sample",
]
