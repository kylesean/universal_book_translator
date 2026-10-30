"""The one machine-boundary predicate behind the page-image egress gate.

A book page may never leave this machine while a local engine can read it
(``UBTConfig.allow_page_upload`` owns the exception). Every route that ships
page images — OCR drivers, a pinned ``ocr_mode='cloud'`` against a local
server, the visual judge's fallback — judges that rule from *where the
endpoint is*, never from what the driver is called.

Two locality questions exist and must not be conflated:

- **Privacy (this module)** — is the endpoint ON THIS MACHINE? Only loopback
  and the container-host aliases qualify. A box elsewhere on the LAN is
  another machine: a page sent there HAS left this one, whatever the billing
  table says. A private address therefore does *not* qualify.
- **Billing (``ubt.core.router.pricing.endpoint_is_local``)** — is the
  endpoint free of provider bills? That policy composes this primitive with
  the operator's declared-free boxes (``UBT_LOCAL_ENDPOINTS``); it may be
  wider than privacy because "free" and "on this machine" are different
  questions.

This module lives in ``core`` so the config layer (billing pre-flight) and
the adapter layer (driver probing) cannot disagree about the boundary.
"""

from __future__ import annotations

import ipaddress
import os
from urllib.parse import urlparse

__all__ = ["endpoint_on_this_machine", "resolved_ocr_endpoint"]

#: Hostnames that name this machine in the container era.
_LOCAL_HOSTNAMES = frozenset(
    {"localhost", "0.0.0.0", "host.docker.internal", "host.containers.internal"}
)


def endpoint_on_this_machine(endpoint: str | None) -> bool:
    """True when the endpoint stays on this machine (or is unset).

    Judged from the resolved host, not a string prefix: a DNS name like
    ``127.0.0.1.attacker.example`` is a *remote* host, and treating it as
    local would let ``allow_page_upload=false`` send page images off the
    machine. An unset endpoint is "on this machine" only in the sense that
    nothing is configured yet — callers that fall back to a remote default
    (the OpenAI endpoint behind ``ocr_mode='cloud'``) must resolve that
    default *before* asking, or they will read "no opinion" as "safe".
    """
    if not endpoint:
        return True
    # A schemeless host like "127.0.0.1:8765" parses with no hostname under
    # urlparse, which would misclassify a local endpoint as non-local and trip
    # the egress gate for an endpoint that never leaves the box.
    if "://" not in endpoint:
        endpoint = "http://" + endpoint
    host = (urlparse(endpoint).hostname or "").lower()
    if host in _LOCAL_HOSTNAMES or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def resolved_ocr_endpoint(endpoint: str | None, default: str) -> str:
    """The endpoint a driver will actually reach, mirroring the drivers' own
    fallback chain (``endpoint`` -> ``UBT_OCR_ENDPOINT`` -> ``default``).

    The egress gate must judge *this* value, not the raw argument:
    ``endpoint=None`` looks local but the driver would then reach the default
    — an unset argument must not read as "safe" when the default is remote.
    """
    return (endpoint or os.environ.get("UBT_OCR_ENDPOINT") or default).rstrip("/")
