"""``ubt.pipeline`` -- the pure pipeline steps and their sequencing (document compiler architecture).

:func:`ubt.pipeline.steps.realize` is the single execution core: it lowers one
element through the fidelity descent under a backend's declared capabilities.
Sequencing, caching and event projection (``run.py``) arrive in later steps;
this package starts with the step itself.
"""

from __future__ import annotations

from ubt.pipeline.attest import AttestationReport, attest_document
from ubt.pipeline.steps import IntegrityViolation, realize

__all__ = ["AttestationReport", "IntegrityViolation", "attest_document", "realize"]
