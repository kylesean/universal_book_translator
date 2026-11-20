"""Document-level style group planning for PDF delivery contracts.

The legacy CELS layout solver (``cels.py``, ``coordinator.py``, ``distortion.py``,
``solver.py``) has been retired in favor of the v2 in-place PDF codec
(:mod:`ubt.codecs.pdf`). Only :mod:`ubt.core.layout.style_group` remains.
"""

from ubt.core.layout.style_group import (
    DRIFT_TOLERANCE_PT,
    StyleGroup,
    StyleGroupException,
    StyleGroupPlan,
    plan_body_group,
    plan_from_metadata,
)

__all__ = [
    "DRIFT_TOLERANCE_PT",
    "StyleGroup",
    "StyleGroupException",
    "StyleGroupPlan",
    "plan_body_group",
    "plan_from_metadata",
]
