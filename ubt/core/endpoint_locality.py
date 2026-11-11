"""The one machine-boundary predicate behind the page-image egress gate.

A book page may never leave this machine while a local engine can read it
(``UBTConfig.allow_page_upload`` owns the exception). Every route that ships
page images — OCR drivers, a pinned ``ocr_mode='cloud'`` against a local
server, the visual judge's fallback — judges that rule from *where the
endpoint is*, never from what the driver is called.

Two locality questions exist and must not be conflated:

- **Privacy (this module)** — is the endpoint ON THIS MACHINE? The loopback
  addresses and ``localhost`` qualify. A box elsewhere on the LAN is another
  machine: a page sent there HAS left this one, whatever the billing table
  says. A private address therefore does *not* qualify on its own — and the
  container-era aliases (``host.docker.internal``, ``0.0.0.0``) do not
  qualify by name either: from *inside* the shipped container those names
  reach the host, which IS another machine from the page's point of view. An
  operator who runs the server on the host itself and wants those names to
  count as local lists them in ``UBT_LOCAL_HOSTNAMES`` (a comma-separated
  list, default empty): an explicit acknowledgement of the boundary, checked
  by actually resolving the name and accepting only loopback / link-local /
  RFC1918-private answers. ``0.0.0.0`` as a destination is non-local, period
  — point a local endpoint at ``localhost`` or ``127.0.0.1`` instead.
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

__all__ = [
    "endpoint_on_this_machine",
    "operator_local_hostnames",
    "resolved_addresses_are_local",
    "resolved_ocr_endpoint",
]

#: Hostnames that name this machine without any declaration. ``localhost`` and
#: the ``*.localhost`` convention resolve to loopback everywhere. The
#: container-host aliases are deliberately NOT here: inside the shipped
#: container they name the *host*, which is another machine as far as a page
#: image is concerned — see ``UBT_LOCAL_HOSTNAMES`` for the explicit opt-in.
_LOCAL_HOSTNAMES = frozenset({"localhost"})

#: Comma-separated hostnames the operator explicitly acknowledges stay on this
#: machine (e.g. ``UBT_LOCAL_HOSTNAMES=host.docker.internal,ocr.lan.book``).
#: Default empty: a name counts as local only when the operator said so AND it
#: resolves to loopback / link-local / RFC1918-private addresses.
LOCAL_HOSTNAMES_ENV = "UBT_LOCAL_HOSTNAMES"

#: The RFC1918 private ranges an opted-in hostname's resolutions may land in.
_RFC1918_NETWORKS = (
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
)


def operator_local_hostnames() -> frozenset[str]:
    """The lowercased hostnames ``UBT_LOCAL_HOSTNAMES`` declares local."""
    raw = os.environ.get(LOCAL_HOSTNAMES_ENV, "")
    return frozenset(token.strip().lower() for token in raw.split(",") if token.strip())


def resolved_addresses_are_local(host: str) -> bool:
    """True when every address ``host`` resolves to stays on this machine's side.

    "Every", not "any": a name resolving to both a private and a public
    address can be connected to at its public answer, and the egress gate must
    not split the difference. An unresolvable name fails closed. Accepted
    ranges are loopback, link-local and RFC1918 private — the ranges a
    container-host alias or a LAN OCR box resolves to when it genuinely is
    this machine or an acknowledged neighbour the operator opted in.
    """
    import socket

    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return False
    addresses: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
    for _family, _type, _proto, _canonname, sockaddr in infos:
        try:
            addr: ipaddress.IPv4Address | ipaddress.IPv6Address = ipaddress.ip_address(sockaddr[0])
        except ValueError:
            return False
        addresses.append(addr)
    if not addresses:
        return False

    def _stays_local(addr: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
        if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
            addr = addr.ipv4_mapped
        if isinstance(addr, ipaddress.IPv6Address):
            return addr.is_loopback or addr.is_link_local
        return (
            addr.is_loopback or addr.is_link_local or any(addr in net for net in _RFC1918_NETWORKS)
        )

    return all(_stays_local(addr) for addr in addresses)


def endpoint_on_this_machine(endpoint: str | None) -> bool:
    """True when the endpoint stays on this machine (or is unset).

    Judged from the resolved host, not a string prefix: a DNS name like
    ``127.0.0.1.attacker.example`` is a *remote* host, and treating it as
    local would let ``allow_page_upload=false`` send page images off the
    machine. An unset endpoint is "on this machine" only in the sense that
    nothing is configured yet — callers that fall back to a remote default
    (the OpenAI endpoint behind ``ocr_mode='cloud'``) must resolve that
    default *before* asking, or they will read "no opinion" as "safe".

    A hostname can still count as local when the operator listed it in
    ``UBT_LOCAL_HOSTNAMES`` and it resolves to loopback / link-local /
    RFC1918-private addresses only — the explicit acknowledgement that the
    name really stays on this side of the boundary.
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
        pass
    # Operator-acknowledged hostnames only, and only when resolution agrees:
    # the opt-in widens the *name* set, never the address set.
    if host in operator_local_hostnames():
        return resolved_addresses_are_local(host)
    return False


def resolved_ocr_endpoint(endpoint: str | None, default: str) -> str:
    """The endpoint a driver will actually reach, mirroring the drivers' own
    fallback chain (``endpoint`` -> ``UBT_OCR_ENDPOINT`` -> ``default``).

    The egress gate must judge *this* value, not the raw argument:
    ``endpoint=None`` looks local but the driver would then reach the default
    — an unset argument must not read as "safe" when the default is remote.
    """
    return (endpoint or os.environ.get("UBT_OCR_ENDPOINT") or default).rstrip("/")
