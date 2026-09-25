"""Driver registry: name -> lazy factory. No hard deps at import."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from urllib.parse import urlparse

from ubt.adapters.pdf.vlm.types import VlmDriver

logger = logging.getLogger(__name__)


def _endpoint_is_local(endpoint: str | None) -> bool:
    """True when the OCR endpoint stays on this machine (or is unset)."""
    if not endpoint:
        return True
    # A schemeless host like "127.0.0.1:8765" parses with no hostname under
    # urlparse, which would misclassify a local endpoint as non-local and trip
    # the egress gate for an endpoint that never leaves the box.
    if "://" not in endpoint:
        endpoint = "http://" + endpoint
    host = (urlparse(endpoint).hostname or "").lower()
    return host in ("localhost", "::1") or host.startswith("127.")


_FACTORIES: dict[str, Callable[[], VlmDriver]] = {}


def register_driver(name: str, factory: Callable[[], VlmDriver]) -> None:
    """Register (or override) a driver factory. Overrides are how tests inject fakes."""
    _FACTORIES[name] = factory


def list_drivers() -> list[str]:
    return sorted(_FACTORIES)


def default_driver_name() -> str:
    return os.environ.get("UBT_VLM_DRIVER", "rapidocr").strip() or "rapidocr"


def get_driver(name: str | None = None) -> VlmDriver:
    """Instantiate the named driver (KeyError on unknown — fail closed)."""
    key = (name or default_driver_name()).strip()
    try:
        factory = _FACTORIES[key]
    except KeyError:
        raise KeyError(f"unknown VLM driver {key!r} (known: {list_drivers()})") from None
    return factory()


def _rapidocr_factory() -> VlmDriver:
    from ubt.adapters.pdf.vlm.drivers.rapidocr_driver import RapidOcrDriver

    return RapidOcrDriver()


def _deepseek_ocr_factory() -> VlmDriver:
    from ubt.adapters.pdf.vlm.drivers.deepseek_driver import DeepSeekOcrDriver

    return DeepSeekOcrDriver()


def _sidecar_factory() -> VlmDriver:
    from ubt.adapters.pdf.vlm.drivers.sidecar_driver import SidecarOcrDriver

    return SidecarOcrDriver()


def _cloud_factory() -> VlmDriver:
    from ubt.adapters.pdf.vlm.drivers.cloud_driver import CloudOcrDriver

    return CloudOcrDriver(provider="cloud")


def _vlm_factory() -> VlmDriver:
    from ubt.adapters.pdf.vlm.drivers.cloud_driver import CloudOcrDriver

    return CloudOcrDriver(provider="vlm")


register_driver("rapidocr", _rapidocr_factory)
register_driver("deepseek-ocr", _deepseek_ocr_factory)
register_driver("sidecar", _sidecar_factory)
register_driver("http", _sidecar_factory)
register_driver("cloud", _cloud_factory)
register_driver("vlm", _vlm_factory)


def probe_effective_driver(
    mode: str = "auto",
    endpoint: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    allow_page_upload: bool = False,
) -> tuple[str, VlmDriver] | tuple[None, None]:
    """Probe and return the most appropriate OCR driver according to mode and availability.

    Modes:
    - 'off': Explicitly disabled.
    - 'sidecar': Explicitly requires local/remote Docker sidecar.
    - 'cloud': Explicitly requires cloud REST API.
    - 'vlm': Explicitly requires Vision LLM.
    - 'rapidocr': Explicitly requires local rapidocr package.
    - 'auto': Probes sidecar -> local rapidocr -> cloud/vlm (cloud/vlm last:
      page images must not leave the machine while a local engine can do the
      job; UBT_ALLOW_PAGE_UPLOAD=false skips them entirely).

    ``allow_page_upload`` defaults to the banned answer on purpose: this is a
    probe of an egress route, and every caller that reached it without an
    opinion would otherwise get "yes, ship the pages" from a function whose
    own privacy policy says the opposite.
    """
    clean_mode = (mode or "auto").strip().lower()
    if clean_mode == "off":
        return None, None

    if clean_mode in ("sidecar", "http"):
        # The sidecar is normally a local process, but UBT_OCR_ENDPOINT may point
        # it at a remote host — which would ship page images off-machine while
        # allow_page_upload=false. Gate only the non-local case so the local
        # sidecar keeps working under the default closed gate.
        if not allow_page_upload and not _endpoint_is_local(endpoint):
            raise ValueError(
                "ocr_mode='sidecar' targets a non-local endpoint and ships rendered "
                "book pages there, but allow_page_upload=false (UBT_ALLOW_PAGE_UPLOAD) "
                "forbids page-image egress. Point UBT_OCR_ENDPOINT at localhost or "
                "re-enable uploads."
            )
        from ubt.adapters.pdf.vlm.drivers.sidecar_driver import SidecarOcrDriver

        return "sidecar", SidecarOcrDriver(endpoint=endpoint, api_key=api_key)

    if clean_mode == "cloud":
        if not allow_page_upload:
            raise ValueError(
                "ocr_mode='cloud' ships rendered book pages to a remote endpoint, "
                "but allow_page_upload=false (UBT_ALLOW_PAGE_UPLOAD) forbids page-image "
                "egress. Choose a local engine (sidecar/rapidocr) or re-enable uploads."
            )
        from ubt.adapters.pdf.vlm.drivers.cloud_driver import CloudOcrDriver

        return "cloud", CloudOcrDriver(
            endpoint=endpoint, api_key=api_key, model=model, provider="cloud"
        )

    if clean_mode in ("vlm", "openai_vision"):
        if not allow_page_upload:
            raise ValueError(
                "ocr_mode='vlm' ships rendered book pages to a vision model, but "
                "allow_page_upload=false (UBT_ALLOW_PAGE_UPLOAD) forbids page-image "
                "egress. Choose a local engine (sidecar/rapidocr) or re-enable uploads."
            )
        from ubt.adapters.pdf.vlm.drivers.cloud_driver import CloudOcrDriver

        return "vlm", CloudOcrDriver(
            endpoint=endpoint, api_key=api_key, model=model, provider="vlm"
        )

    if clean_mode == "rapidocr":
        return "rapidocr", get_driver("rapidocr")

    # Explicit custom registered driver
    if clean_mode in _FACTORIES and clean_mode != "auto":
        return clean_mode, get_driver(clean_mode)

    # auto mode:
    # 0. If UBT_VLM_DRIVER environment variable is set to a registered driver, honor it
    vlm_env = os.environ.get("UBT_VLM_DRIVER", "").strip()
    if vlm_env and vlm_env in _FACTORIES:
        return vlm_env, get_driver(vlm_env)

    # 1. Probe sidecar health
    from ubt.adapters.pdf.vlm.drivers.sidecar_driver import SidecarOcrDriver

    target_ep = endpoint or os.environ.get("UBT_OCR_ENDPOINT", "http://localhost:8765")
    # The explicit-mode egress gate already blocks a non-local sidecar when
    # ``allow_page_upload=False``; auto must obey the same rule or a remote
    # endpoint silently ships pages off-machine. A local sidecar (or uploads
    # enabled) is fine and still preferred over the cloud route below.
    if (allow_page_upload or _endpoint_is_local(target_ep)) and SidecarOcrDriver.is_healthy(
        target_ep, api_key=api_key, timeout=0.3
    ):
        return "sidecar", SidecarOcrDriver(endpoint=target_ep, api_key=api_key)

    # 2. Local rapidocr before any cloud route: a book page must not leave
    # the machine while a local engine can read it.
    try:
        try:
            import rapidocr  # noqa: F401
        except ImportError:
            import rapidocr_onnxruntime  # noqa: F401

        return "rapidocr", get_driver("rapidocr")
    except ImportError:
        pass

    # 3. Cloud OCR / Vision only when page egress is allowed.
    if not allow_page_upload:
        return None, None
    has_cloud_key = bool(
        api_key or os.environ.get("UBT_OCR_API_KEY") or os.environ.get("OPENAI_API_KEY")
    )
    has_cloud_endpoint = bool(endpoint or os.environ.get("UBT_OCR_ENDPOINT"))
    if has_cloud_key or (has_cloud_endpoint and not target_ep.startswith("http://localhost")):
        from ubt.adapters.pdf.vlm.drivers.cloud_driver import CloudOcrDriver

        provider = "vlm" if bool(os.environ.get("OPENAI_API_KEY")) else "cloud"
        # A paid pick under auto must never be silent: every other paid route
        # in this project (batch, neural QE) requires an explicit flag, and a
        # surprise bill must not arrive through the back door. The warning
        # names the env knobs so the operator can pin a local engine instead.
        logger.warning(
            "OCR auto mode selected the PAID cloud engine '%s' (no local "
            "sidecar or rapidocr available); every scanned page will be "
            "billed. Pin a free local engine via UBT_VLM_DRIVER or UBT_OCR_MODE "
            "to avoid this.",
            provider,
        )
        return provider, CloudOcrDriver(
            endpoint=endpoint, api_key=api_key, model=model, provider=provider
        )

    return None, None
