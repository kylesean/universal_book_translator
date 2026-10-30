"""Delivery-domain errors that the pipeline can fail on before spending tokens."""

from __future__ import annotations

from ubt.core.exceptions import UBTError


class DeliveryInfeasibleError(UBTError):
    """The requested contract cannot be honestly executed for this document.

    Raised before the first model call. The message must carry the requested
    contract, the failed eligibility reasons, and the two allowed next steps:
    request ``readable`` or re-run with ``allow_degrade``.
    """


__all__ = ["DeliveryInfeasibleError"]
