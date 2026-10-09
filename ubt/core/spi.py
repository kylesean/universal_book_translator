"""Service Provider Interface (SPI) registry for Core.

Decouples ``ubt.core`` from concrete downstream packages (adapters, segment,
translate). Providers register callable implementations either via standard
Python entry points (group 'ubt.spi') or dynamic registration.

Core accesses downstream capabilities strictly through this registry without
naming any downstream package path at runtime.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class SPIRegistry:
    """Central registry and service locator for downstream capabilities."""

    _instance: SPIRegistry | None = None

    def __init__(self) -> None:
        self._providers: dict[str, Any] = {}
        self._discovered: bool = False

    @classmethod
    def instance(cls) -> SPIRegistry:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (primarily for tests)."""
        cls._instance = None

    def register(self, service_name: str, provider: Any) -> None:
        """Register a provider implementation for a named SPI service."""
        self._providers[service_name] = provider

    def register_many(self, mapping: dict[str, Any]) -> None:
        """Register multiple providers at once."""
        self._providers.update(mapping)

    def is_registered(self, service_name: str) -> bool:
        return service_name in self._providers

    def get(self, service_name: str) -> Any:
        """Resolve a service provider, triggering entry-point discovery if not yet present."""
        if service_name not in self._providers:
            self._ensure_discovered()
        provider = self._providers.get(service_name)
        if provider is None:
            raise KeyError(f"No SPI provider registered for service: {service_name!r}")
        return provider

    def _ensure_discovered(self) -> None:
        if self._discovered:
            return
        self._discovered = True
        self._load_entry_points()
        self._bootstrap_in_tree_providers()

    def _load_entry_points(self) -> None:
        try:
            import importlib.metadata

            eps = importlib.metadata.entry_points(group="ubt.spi")
            for ep in eps:
                try:
                    loader = ep.load()
                    if callable(loader):
                        loader(self)
                except Exception as exc:
                    logger.debug("Failed loading ubt.spi entry point %s: %s", ep.name, exc)
        except Exception as exc:
            logger.debug("Entry points scan encountered error: %s", exc)

    def _bootstrap_in_tree_providers(self) -> None:
        """Fallback discovery for in-tree development where editable entry points may be pending."""
        import importlib

        # Late dynamic resolution: module names are evaluated as strings, preventing
        # static module-level cycles in the core import graph.
        for mod_name in ("ubt.adapters.spi", "ubt.segment.spi", "ubt.translate.spi"):
            try:
                mod = importlib.import_module(mod_name)
                init_fn = getattr(mod, "register_spi_providers", None)
                if callable(init_fn):
                    init_fn(self)
            except (ImportError, AttributeError) as exc:
                logger.debug("Bootstrap fallback skipped %s: %s", mod_name, exc)


def get_spi_registry() -> SPIRegistry:
    """Return the global SPIRegistry instance."""
    return SPIRegistry.instance()
