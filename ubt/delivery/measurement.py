"""MeasurementService contract and cache scaffold.

The PRD's central rule is "measure the artifact, not a proxy" (§7.3, §8.1).
This module owns the typed request/result and a deterministic cache key so the
eventual Typst probe, HarfBuzz path, and tests all agree about what a
measurement *is*. It deliberately contains no estimator: a delegate must
return a real measurement, or this scaffold is not wired in.

The real Typst-backed service belongs to Phase 1 and is not part of this
scaffold; ``CachedMeasurementService`` is usable now with a fake delegate in
unit tests and with the Typst probe once it lands.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from hashlib import sha256
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class MeasurementStyle(BaseModel):
    """The style inputs that affect line breaking and height."""

    model_config = ConfigDict(frozen=True)

    font_family: str
    #: Typst/font version belongs in the cache key: a font upgrade changes
    #: shaping and line height without changing the request text or width.
    font_version: str = ""
    size_pt: float = Field(gt=0.0)
    #: Target baseline pitch in points (1.35 x size is a 1.35 line height).
    #: The Typst probe converts this to ``par(leading:)`` by subtracting the
    #: font's own line height; callers must not pass a Typst gap.
    leading_pt: float = Field(gt=0.0)
    #: The font's own line height in em; Typst adds the leading gap on top.
    font_line_height_em: float = Field(default=0.733, gt=0.0, le=2.0)
    tracking_em: float = 0.0
    #: Typst's static-font ``stretch`` does not change advance widths, so the
    #: probe refuses non-unity horizontal scale rather than reporting a fake
    #: measurement. CELS can add a HarfBuzz path when it needs γ_x.
    horizontal_scale: float = Field(default=1.0, ge=0.5, le=1.2)
    language: str = "zh"
    justify: bool = True


class LineBox(BaseModel):
    """One measured line rectangle in the block's local top-left space."""

    model_config = ConfigDict(frozen=True)

    x0: float
    y0: float
    x1: float
    y1: float


class MeasuredBlock(BaseModel):
    """A real layout measurement, not an estimate."""

    model_config = ConfigDict(frozen=True)

    height_pt: float = Field(ge=0.0)
    line_count: int = Field(ge=0)
    line_boxes: tuple[LineBox, ...] = ()
    overflow: bool = False
    #: Stable content hash of the request; the cache rewrites a delegate's
    #: provisional id to this value so a report can trace one measurement.
    measurement_id: str


class MeasurementError(RuntimeError):
    """A real measurement could not be produced (compiler/font/probe failure)."""


class MeasurementRequest(BaseModel):
    """One block measurement request for single or batched compilation."""

    model_config = ConfigDict(frozen=True)

    text: str
    style: MeasurementStyle
    width_pt: float = Field(gt=0.0)


@runtime_checkable
class MeasurementService(Protocol):
    """The only interface layout code may use for fit decisions."""

    def measure_block(
        self,
        text: str,
        style: MeasurementStyle,
        width_pt: float,
    ) -> MeasuredBlock: ...

    def measure_many(
        self,
        requests: Sequence[MeasurementRequest],
    ) -> list[MeasuredBlock]: ...


MeasureFn = Callable[[str, MeasurementStyle, float], MeasuredBlock]
BatchMeasureFn = Callable[[Sequence[MeasurementRequest]], Sequence[MeasuredBlock]]


def measurement_cache_key(text: str, style: MeasurementStyle, width_pt: float) -> str:
    """Stable hash of every input that can change a measurement."""
    payload = {
        "text": text,
        "style": style.model_dump(mode="json"),
        "width_pt": width_pt,
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(canonical.encode("utf-8")).hexdigest()


class CachedMeasurementService:
    """Cache real measurements behind the protocol.

    This class does not decide whether a block fits; it only makes repeated
    measurement cheap. A caller that receives ``overflow=True`` must still
    route to a fallback (page reflow, LLM budget, or failure) rather than
    shrink a font locally.
    """

    def __init__(
        self,
        delegate: MeasureFn,
        *,
        batch_delegate: BatchMeasureFn | None = None,
    ) -> None:
        self._delegate = delegate
        self._batch_delegate = batch_delegate
        self._cache: dict[str, MeasuredBlock] = {}

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    def clear(self) -> None:
        self._cache.clear()

    def measure_block(
        self,
        text: str,
        style: MeasurementStyle,
        width_pt: float,
    ) -> MeasuredBlock:
        if width_pt <= 0.0:
            raise ValueError(f"width_pt must be positive, got {width_pt!r}")
        key = measurement_cache_key(text, style, width_pt)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        measured = self._delegate(text, style, width_pt)
        if measured.measurement_id != key:
            measured = measured.model_copy(update={"measurement_id": key})
        self._cache[key] = measured
        return measured

    def measure_many(
        self,
        requests: Sequence[MeasurementRequest],
    ) -> list[MeasuredBlock]:
        """Measure a batch of requests, compiling uncached items in one delegate call."""
        if not requests:
            return []
        keys: list[str] = []
        uncached_by_key: dict[str, MeasurementRequest] = {}
        for req in requests:
            if req.width_pt <= 0.0:
                raise ValueError(f"width_pt must be positive, got {req.width_pt!r}")
            key = measurement_cache_key(req.text, req.style, req.width_pt)
            keys.append(key)
            if key not in self._cache and key not in uncached_by_key:
                uncached_by_key[key] = req

        if uncached_by_key:
            unique_keys = list(uncached_by_key.keys())
            unique_reqs = [uncached_by_key[k] for k in unique_keys]
            if self._batch_delegate is not None:
                batch_results = self._batch_delegate(unique_reqs)
                for key, measured in zip(unique_keys, batch_results, strict=True):
                    if measured.measurement_id != key:
                        measured = measured.model_copy(update={"measurement_id": key})
                    self._cache[key] = measured
            else:
                for key, req in zip(unique_keys, unique_reqs, strict=True):
                    measured = self._delegate(req.text, req.style, req.width_pt)
                    if measured.measurement_id != key:
                        measured = measured.model_copy(update={"measurement_id": key})
                    self._cache[key] = measured

        return [self._cache[k] for k in keys]
