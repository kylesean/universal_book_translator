"""Driver registry: name -> lazy factory. No hard deps at import.

The user-facing OCR vocabulary is the four modes on ``UBTConfig.ocr_mode`` —
``off`` / ``local`` / ``cloud`` / ``auto``. Driver names are internal wiring:
a user configures an endpoint and a model, and *locality* — not a driver's
name or a URL substring — decides what the egress gate allows. The one rule
every page-image route shares lives in :func:`assert_page_egress_allowed`;
custom drivers register with ``local=True`` when they never ship a page
off-machine, and the gate treats everything else as egress.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable

from ubt.adapters.pdf.vlm.types import VlmDriver
from ubt.core.endpoint_locality import endpoint_on_this_machine, resolved_ocr_endpoint

logger = logging.getLogger(__name__)


def _sidecar_endpoint(endpoint: str | None) -> str:
    """The endpoint ``SidecarOcrDriver`` will actually reach.

    Mirrors the driver's own fallback chain (``endpoint`` -> ``UBT_OCR_ENDPOINT``
    -> localhost default). The egress gate must judge *this* value, not the raw
    argument: ``endpoint=None`` looks local but the driver would then read a
    remote ``UBT_OCR_ENDPOINT`` and ship the pages there.
    """
    return endpoint or os.environ.get("UBT_OCR_ENDPOINT", "http://localhost:8765")


def assert_page_egress_allowed(endpoint: str | None, *, allow_page_upload: bool, mode: str) -> None:
    """The single choke point for page-image egress.

    Every route that ships rendered book pages — OCR drivers, a pinned
    ``ocr_mode='cloud'`` against a local server, non-local custom drivers —
    passes through here. ``allow_page_upload`` is the operator's explicit
    opt-in; endpoint locality is the technical exemption. A custom driver
    registered without ``local=True`` has no endpoint to judge, so it counts
    as egress and needs the opt-in.
    """
    if allow_page_upload or endpoint_on_this_machine(endpoint):
        return
    raise ValueError(
        f"ocr_mode='{mode}' targets the non-local endpoint {endpoint!r} and "
        "ships rendered book pages there, but allow_page_upload=false "
        "(UBT_ALLOW_PAGE_UPLOAD) forbids page-image egress. Point UBT_OCR_ENDPOINT "
        "at localhost or re-enable uploads."
    )


_FACTORIES: dict[str, tuple[Callable[[], VlmDriver], bool]] = {}


def register_driver(name: str, factory: Callable[[], VlmDriver], *, local: bool = False) -> None:
    """Register a driver factory.

    ``local=True`` declares the driver never ships a page image off this
    machine (in-process engines, localhost-only servers); only that declaration
    exempts it from the page-egress opt-in. Overrides are how tests inject
    fakes.
    """
    _FACTORIES[name] = (factory, local)


def list_drivers() -> list[str]:
    return sorted(_FACTORIES)


def default_driver_name() -> str:
    return "rapidocr"


def get_driver(name: str | None = None) -> VlmDriver:
    """Instantiate the named driver (KeyError on unknown — fail closed)."""
    key = (name or default_driver_name()).strip()
    try:
        factory, _ = _FACTORIES[key]
    except KeyError:
        raise KeyError(f"unknown VLM driver {key!r} (known: {list_drivers()})") from None
    return factory()


def _rapidocr_factory() -> VlmDriver:
    from ubt.adapters.pdf.vlm.drivers.rapidocr_driver import RapidOcrDriver

    return RapidOcrDriver()


def _sidecar_factory() -> VlmDriver:
    from ubt.adapters.pdf.vlm.drivers.sidecar_driver import SidecarOcrDriver

    return SidecarOcrDriver()


def _vision_factory() -> VlmDriver:
    from ubt.adapters.pdf.vlm.drivers.cloud_driver import VisionEndpointDriver

    return VisionEndpointDriver()


def _unlimited_ocr2_factory() -> VlmDriver:
    from ubt.adapters.pdf.vlm.drivers.unlimited_ocr2_driver import UnlimitedOcr2Driver

    return UnlimitedOcr2Driver()


register_driver("rapidocr", _rapidocr_factory, local=True)
register_driver("sidecar", _sidecar_factory, local=True)
register_driver("vision", _vision_factory, local=False)
register_driver("unlimited-ocr2", _unlimited_ocr2_factory, local=True)


def _probe_local(
    endpoint: str | None,
    api_key: str | None,
    model: str | None,
    allow_page_upload: bool,
) -> tuple[str, VlmDriver] | tuple[None, None]:
    """The best engine that never ships a page off this machine, fail-closed.

    Preference order: a *configured* local vision endpoint (an explicit
    ``ocr_endpoint`` + ``ocr_model`` is a deliberate pin — a model name is
    meaningless to the sidecar contract, so its presence selects the
    OpenAI-style route without URL sniffing) first, then the measured-box
    engines a scan transcription needs (sidecar, in-process rapidocr), then
    the in-process proofread VLM. ``allow_page_upload`` is forwarded so the
    egress choke point, not this function, owns the exception.
    """
    # 1. An explicitly configured local vision endpoint.
    vision_ep = endpoint or os.environ.get("UBT_OCR_ENDPOINT") or ""
    if vision_ep and model and endpoint_on_this_machine(vision_ep):
        from ubt.adapters.pdf.vlm.drivers.cloud_driver import VisionEndpointDriver

        return "vision", VisionEndpointDriver(endpoint=vision_ep, api_key=api_key, model=model)

    # 2. The sidecar (measured boxes): health-check the endpoint it will
    # actually reach, and let the choke point judge it.
    sidecar_ep = _sidecar_endpoint(endpoint)
    assert_page_egress_allowed(sidecar_ep, allow_page_upload=allow_page_upload, mode="local")
    from ubt.adapters.pdf.vlm.drivers.sidecar_driver import SidecarOcrDriver

    if SidecarOcrDriver.is_healthy(sidecar_ep, api_key=api_key, timeout=0.3):
        return "sidecar", SidecarOcrDriver(endpoint=sidecar_ep, api_key=api_key)

    # 3. In-process rapidocr (measured boxes) — no endpoint at all.
    from ubt.adapters.pdf.vlm.drivers.rapidocr_driver import RapidOcrDriver

    if RapidOcrDriver.is_available():
        return "rapidocr", get_driver("rapidocr")

    # 4. A local vision endpoint without a model name cannot route (the
    # driver fails closed on an empty model) — say so instead of silently
    # skipping past a configured server.
    if vision_ep and endpoint_on_this_machine(vision_ep) and not model:
        logger.warning(
            "UBT_OCR_ENDPOINT=%s is local but no model is set; set UBT_OCR_MODEL "
            "so the vision endpoint has something to serve.",
            vision_ep,
        )

    # 5. In-process Unlimited-OCR-2 (proofread only): configured via its model
    # path env, runs behind a resident worker on this machine.
    if os.environ.get("UBT_UNLIMITED_OCR2_MODEL", "").strip():
        return "unlimited-ocr2", get_driver("unlimited-ocr2")

    return None, None


def probe_effective_driver(
    mode: str = "auto",
    endpoint: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    allow_page_upload: bool = False,
) -> tuple[str, VlmDriver] | tuple[None, None]:
    """Probe and return the most appropriate OCR driver for the requested mode.

    The mode vocabulary is exactly ``UBTConfig.OcrMode``:

    - ``off``: explicitly disabled.
    - ``local``: the best engine that never ships a page off this machine
      (see :func:`_probe_local`). Can transcribe a true scan when a
      measured-box engine (sidecar/rapidocr) is available; a text-only
      local VLM proofreads pages that already have a text layer.
    - ``cloud``: a remote vision endpoint. Requires the page-egress opt-in
      — unless the configured endpoint is itself local, which is the
      "pin my localhost vision server" case and needs no opt-in.
    - ``auto``: ``local`` first, then ``cloud``. A book page must not leave
      the machine while a local engine can read it.

    ``allow_page_upload`` defaults to the banned answer on purpose: this is a
    probe of an egress route, and every caller that reached it without an
    opinion would otherwise get "yes, ship the pages" from a function whose
    own privacy policy says the opposite.
    """
    clean_mode = (mode or "auto").strip().lower()
    if clean_mode == "off":
        return None, None

    if clean_mode == "local":
        driver = _probe_local(endpoint, api_key, model, allow_page_upload)
        if driver[0] is None:
            logger.warning(
                "OCR mode 'local' found no usable engine: no local sidecar, no "
                "in-process rapidocr (install with: uv sync --all-extras), and "
                "no local vision endpoint (set UBT_OCR_ENDPOINT + UBT_OCR_MODEL "
                "to a localhost server)."
            )
        return driver

    if clean_mode == "cloud":
        from ubt.adapters.pdf.vlm.drivers.cloud_driver import DEFAULT_OPENAI_ENDPOINT

        target = resolved_ocr_endpoint(endpoint, DEFAULT_OPENAI_ENDPOINT)
        assert_page_egress_allowed(target, allow_page_upload=allow_page_upload, mode="cloud")
        from ubt.adapters.pdf.vlm.drivers.cloud_driver import VisionEndpointDriver

        # The driver fails closed on an empty model: a vision route must be
        # told which model to call, loudly, at build time.
        return "cloud", VisionEndpointDriver(endpoint=target, api_key=api_key, model=model)

    # Explicit custom registered driver (internal wiring; tests inject fakes).
    if clean_mode in _FACTORIES:
        _, is_local = _FACTORIES[clean_mode]
        if not is_local:
            assert_page_egress_allowed(None, allow_page_upload=allow_page_upload, mode=clean_mode)
        return clean_mode, get_driver(clean_mode)

    if clean_mode == "auto":
        local = _probe_local(endpoint, api_key, model, allow_page_upload)
        if local[0] is not None:
            return local
        from ubt.adapters.pdf.vlm.drivers.cloud_driver import DEFAULT_OPENAI_ENDPOINT

        target = resolved_ocr_endpoint(endpoint, DEFAULT_OPENAI_ENDPOINT)
        has_credentials = bool(api_key or os.environ.get("UBT_OCR_API_KEY"))
        if not allow_page_upload and not endpoint_on_this_machine(target):
            # No local engine and egress closed: say so once, plainly.
            logger.warning(
                "OCR auto mode found no local engine and page egress is "
                "disabled (UBT_ALLOW_PAGE_UPLOAD); scanned pages will be "
                "rejected rather than shipped out. Pin a local engine via "
                "UBT_OCR_MODE=local or re-enable uploads."
            )
            return None, None
        if has_credentials or not endpoint_on_this_machine(target):
            # A paid pick under auto must never be silent: every other paid
            # route in this project (batch, neural QE) requires an explicit
            # flag, and a surprise bill must not arrive through the back
            # door. The warning names the env knobs so the operator can pin
            # a local engine instead.
            logger.warning(
                "OCR auto mode selected the PAID cloud engine 'cloud' (no "
                "local engine available); every scanned page will be billed. "
                "Pin a free local engine via UBT_OCR_MODE or UBT_OCR_ENDPOINT "
                "to avoid this."
            )
        from ubt.adapters.pdf.vlm.drivers.cloud_driver import VisionEndpointDriver

        # A missing vision model must fail loudly, not degrade to a silent
        # skip: ``auto`` warns first, then the driver refuses to invent a
        # model, so the ValueError propagates.
        return "cloud", VisionEndpointDriver(endpoint=target, api_key=api_key, model=model)

    raise ValueError(f"unknown ocr_mode {mode!r}: expected one of off/local/cloud/auto")
