"""Spend-the-tokens gate: compile a sample before the draft stage bills.

`README.md`'s dependency table warned that a broken Typst toolchain kills a PDF
job at Stage 6 — after every token of the whole book has been spent. That is
not hypothetical: ledger job `fc1d7bd7b799` shows 209 blocks translated in 3m03s
followed by a Typst compile failure and zero output.

This module runs the *real* PDF render path over a handful of source-text
blocks, before Stage 3, so toolchain-shaped failures (missing binary, version
drift, generated markup the compiler rejects) exit non-zero at zero token cost.
It is not a full rehearsal: defects that only appear once translated text
exists (e.g. the `#box[...](...)` code-mode crash fixed by a98a315) still fall
through to the Stage 6 healer — what changes is that a book cannot finish
translation only to discover the renderer itself is dead.
"""

from __future__ import annotations

import copy
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any, cast

from ubt.core.config import RIGID_ENGINES, canonical_render_engine
from ubt.core.exceptions import DocumentParseError
from ubt.core.ir.models import BlockType, IRBlock
from ubt.core.policy.adaptive_policy import resolve_pdf_engine
from ubt.core.ports import DocumentAdapter, is_pdf_engine_adapter

logger = logging.getLogger(__name__)

# Enough blocks to include an inline-math one and a structure block without
# turning the rehearsal into a second full render.
PREFLIGHT_MAX_BLOCKS = 5

_SAMPLE_PREFERENCE = (
    BlockType.FORMULA,
    BlockType.TABLE,
    BlockType.HEADING,
)


def _sample_key(block: IRBlock) -> tuple[int, int]:
    """Rank blocks so the sample exercises the most markup-sensitive paths."""
    inline_math = "$" in (block.source_text or "")
    try:
        type_rank = _SAMPLE_PREFERENCE.index(block.block_type)
    except ValueError:
        type_rank = len(_SAMPLE_PREFERENCE)
    return (0 if inline_math else 1, type_rank)


def select_preflight_sample(
    blocks: list[IRBlock], limit: int = PREFLIGHT_MAX_BLOCKS
) -> list[IRBlock]:
    """Pick a small, markup-diverse sample in reading order.

    Ranking picks the sensitive blocks first; re-sorting by spine index keeps
    the emitted document ordered so a failure reproduces the real layout path.
    """
    picked = sorted(blocks, key=_sample_key)[:limit]
    return sorted(picked, key=lambda b: b.spine_index)


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
) -> None:
    """Render a source-text sample through the adapter's real PDF path.

    Raises DocumentParseError with operator-facing guidance when the render
    fails; a non-PDF adapter or an empty sample skips, because text-only
    outputs never touch Typst. Callers run this before any billable stage.
    """
    if not is_pdf_engine_adapter(adapter) or not blocks:
        return

    sample = select_preflight_sample(blocks)
    metadata = getattr(manifest, "metadata", None) or {}
    run_meta = getattr(manifest, "run", None)
    # Resolve the engine against ALL blocks, then force that engine onto the
    # sample. ``select_preflight_sample`` is deliberately structure-biased
    # (TABLE/FORMULA first), so resolving on the sample alone could route rigid
    # while the full document reflows: the scratch render would exercise a path
    # that never runs, and log a monolingual-downgrade warning for it.
    effective_engine = getattr(run_meta, "render_engine_effective", None)
    if effective_engine:
        render_engine = str(effective_engine)
    else:
        requested_engine = (
            metadata.get("render_engine")
            or getattr(run_meta, "render_engine", None)
            or "publication"
        )
        render_engine = resolve_pdf_engine(str(requested_engine), blocks, manifest=manifest)
    bilingual_mode = (
        getattr(run_meta, "effective_dual_mode", None)
        or getattr(run_meta, "bilingual_mode", None)
        or metadata.get("bilingual_mode")
    )
    # Rigid is monolingual: match the scratch mode to the resolved engine so the
    # rehearsal is faithful and the scratch copy does not emit a downgrade
    # warning for a route the live manifest never took.
    if canonical_render_engine(render_engine) in RIGID_ENGINES:
        bilingual_mode = "monolingual"
    pdf_adapter = cast(Any, adapter)
    # Render into an isolated copy: the scratch compile must not write its
    # route/downgrade facts back onto the live run manifest.
    scratch_manifest = _isolated_manifest(manifest)

    tmp_dir = Path(tempfile.mkdtemp(prefix="ubt-preflight-"))
    preflight_path = tmp_dir / "preflight.pdf"
    try:
        await pdf_adapter.render_blocks(
            manifest=scratch_manifest,
            blocks=sample,
            target_lang=target_lang,
            output_path=preflight_path,
            bilingual_mode=bilingual_mode,
            render_engine=render_engine,
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
