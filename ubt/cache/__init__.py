"""``ubt.cache`` -- the content-addressed step cache (content-addressed cache layer).

:func:`ubt.cache.store.step_key` names a step invocation by its content, and a
:class:`~ubt.cache.store.CacheStore` returns the stored value or computes it
once. Only expensive pure steps are wrapped (see
:mod:`ubt.adapters.pdf.witness_cache` for the render path's pixel witnesses).
"""

from __future__ import annotations

from ubt.cache.store import CacheStore, DiskCacheStore, step_key

__all__ = ["CacheStore", "DiskCacheStore", "step_key"]
