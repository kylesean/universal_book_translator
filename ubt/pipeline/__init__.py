"""``ubt.pipeline`` -- the pure pipeline steps and their sequencing (ADR-0001).

:func:`ubt.pipeline.steps.realize` is the single execution core: it lowers one
element through the fidelity descent under a backend's declared capabilities.
Sequencing, caching and event projection (``run.py``) arrive in later steps;
this package starts with the step itself.
"""

from __future__ import annotations

from ubt.pipeline.steps import IntegrityViolation, realize

__all__ = ["IntegrityViolation", "realize"]
