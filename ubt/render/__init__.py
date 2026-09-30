"""``ubt.render`` -- lowering backends and their capability declarations (ADR-0001 L6).

A backend turns one element into a realization at a fidelity it declares it can
produce; :func:`ubt.pipeline.steps.realize` negotiates which. The concrete
backends (Typst, Overlay) land in later steps; this package starts with the
seam they plug into.
"""

from __future__ import annotations

from ubt.render.capability import Backend, Capabilities, Produced
from ubt.render.outputs import Composition, LoweringUnsupported, Placement, compose
from ubt.render.overlay_backend import OverlayBackend

__all__ = [
    "Backend",
    "Capabilities",
    "Composition",
    "LoweringUnsupported",
    "OverlayBackend",
    "Placement",
    "Produced",
    "compose",
]
