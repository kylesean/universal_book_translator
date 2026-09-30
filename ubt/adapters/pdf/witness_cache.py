"""Content-addressed caching for the pixel witnesses (ADR-0001 Phase 4).

The formula and table witnesses are the render path's most expensive pure step:
each one compiles the emitted markup with Typst in a subprocess and rasterizes
the result for a pixel comparison. Their inputs fully determine their output, so
a resumed or re-run job can reuse the verdict instead of recompiling every
formula and table in the book.

The key covers every input the comparison reads: the emitted markup, the block's
identity and geometry, the source file's identity, the compiler binary and the
raster dpi. A witness is fail-open by contract, so a miss just computes, and an
undecodable entry is a miss too -- a cache must never change a verdict or break a
render.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path

from ubt.adapters.pdf.formula_witness import WITNESS_DPI, WitnessResult, witness_formula
from ubt.adapters.pdf.table_witness import witness_table
from ubt.cache.store import CacheStore, step_key
from ubt.core.ir.models import IRBlock

logger = logging.getLogger(__name__)

#: Bumped when the witness's comparison logic changes, so old verdicts are not
#: reused for a decision the current code would make differently.
_CACHE_VERSION = "1"


def _source_identity(source_pdf: Path | str) -> str:
    """A cheap, stable identity for the file the source crop is read from.

    Path plus size plus mtime: the crop is a function of the source bytes, and
    hashing a whole book PDF on every formula would cost more than it saves.
    """
    path = Path(source_pdf)
    try:
        stat = path.stat()
    except OSError:
        return str(path)
    return f"{path}:{stat.st_size}:{stat.st_mtime_ns}"


def _block_identity(block: IRBlock) -> str:
    """The block's id and geometry -- what the crop and the emitted line read."""
    bbox = block.bbox
    if bbox is None:
        return f"{block.id}:nobbox"
    return f"{block.id}:{bbox.page}:{bbox.x0:.3f}:{bbox.y0:.3f}:{bbox.x1:.3f}:{bbox.y1:.3f}"


def _encode(result: WitnessResult) -> str:
    return json.dumps(
        {"status": result.status, "findings": list(result.findings)}, ensure_ascii=False
    )


def _decode(text: str) -> WitnessResult:
    data = json.loads(text)
    return WitnessResult(
        str(data["status"]), [str(finding) for finding in data.get("findings", [])]
    )


def _cached(
    store: CacheStore | None,
    kind: str,
    inputs: list[str],
    params: dict[str, object],
    compute: Callable[[], WitnessResult],
) -> WitnessResult:
    """One cached witness call, fail-open on any cache problem."""
    if store is None:
        return compute()
    key = step_key(kind, inputs, params)
    try:
        return _decode(store.get_or_compute(key, lambda: _encode(compute())))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.debug("witness cache entry unusable for %s: %s", key, exc)
        return compute()


def cached_witness_formula(
    store: CacheStore | None,
    math_line: str,
    block: IRBlock,
    source_pdf: Path | str,
    typst_binary: str,
    dpi: int = WITNESS_DPI,
) -> WitnessResult:
    """``witness_formula`` through ``store`` (or straight through when None)."""
    return _cached(
        store,
        "formula_witness",
        [_CACHE_VERSION, math_line, _block_identity(block), _source_identity(source_pdf)],
        {"binary": str(typst_binary), "dpi": int(dpi)},
        lambda: witness_formula(math_line, block, source_pdf, typst_binary, dpi=dpi),
    )


def cached_witness_table(
    store: CacheStore | None,
    table_markup: str,
    block: IRBlock,
    source_pdf: Path | str,
    typst_binary: str,
    dpi: int = WITNESS_DPI,
) -> WitnessResult:
    """``witness_table`` through ``store`` (or straight through when None)."""
    return _cached(
        store,
        "table_witness",
        [_CACHE_VERSION, table_markup, _block_identity(block), _source_identity(source_pdf)],
        {"binary": str(typst_binary), "dpi": int(dpi)},
        lambda: witness_table(table_markup, block, source_pdf, typst_binary, dpi=dpi),
    )


__all__ = ["cached_witness_formula", "cached_witness_table"]
