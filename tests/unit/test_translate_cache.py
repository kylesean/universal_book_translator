"""Content-addressed translate cache: key stability and fail-open behaviour.

The cache wraps the pure translate step. Its whole safety argument is the key:
the masked source *plus* the model, prompt version and prompt context. A key
that missed any of those would serve a draft produced under different inputs --
a silent, stale mistranslation. The complementary rule is fail-open: an absent
store, a corrupt entry or an unreadable file must compute through the provider,
because a cache must never break a translation.

The provider is a fake coroutine, so the tests observe *whether* the cache hit
by counting calls -- no network, no model.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Awaitable, Callable, Iterator
from pathlib import Path

import pytest

from ubt.cache.store import DiskCacheStore, step_key
from ubt.segment.placeholders import default_placeholder_engine
from ubt.translate.engine import TranslationEngine

pytestmark = pytest.mark.fast


# --------------------------------------------------------------------------- #
# Doubles
# --------------------------------------------------------------------------- #


class _Provider:
    """A fake generate step: counts calls, returns a deterministic draft."""

    def __init__(self) -> None:
        self.calls = 0

    def translate(self, prefix: str = "") -> Callable[[str], Awaitable[str]]:
        async def _fn(masked: str) -> str:
            self.calls += 1
            return f"{prefix}[{masked}]"

        return _fn


class _BrokenStore:
    """A store whose every operation fails -- the cache must not break a run."""

    def get(self, key: str) -> str | None:
        raise OSError("disk gone")

    def put(self, key: str, value: str) -> None:
        raise OSError("disk gone")

    def get_or_compute(self, key: str, compute: Callable[[], str]) -> str:
        raise OSError("disk gone")


def _engine(
    store: DiskCacheStore | _BrokenStore | None, *, model: str = "m1", prompt: str = "v1"
) -> TranslationEngine:
    return TranslationEngine(
        placeholders=default_placeholder_engine(),
        model=model,
        prompt_version=prompt,
        cache=store,
    )


@pytest.fixture
def store() -> Iterator[DiskCacheStore]:
    with tempfile.TemporaryDirectory() as tmp:
        yield DiskCacheStore(tmp)


# --------------------------------------------------------------------------- #
# step_key: the identity of one step invocation.
# --------------------------------------------------------------------------- #


def test_step_key_is_stable_across_calls() -> None:
    assert step_key("translate", ["a", "b"], {"x": 1}) == step_key(
        "translate", ["a", "b"], {"x": 1}
    )


def test_step_key_is_a_sha256_hex_digest() -> None:
    key = step_key("k", ["v"], {})
    assert len(key) == 64
    int(key, 16)  # parses as hex, or raises


@pytest.mark.parametrize(
    "other",
    [
        step_key("other_kind", ["a", "b"], {"x": 1}),  # kind
        step_key("translate", ["b", "a"], {"x": 1}),  # input order
        step_key("translate", ["a", "b", "c"], {"x": 1}),  # extra input
        step_key("translate", ["a", "b"], {"x": 2}),  # param value
        step_key("translate", ["a", "b"], {"y": 1}),  # extra param
    ],
)
def test_step_key_changes_with_every_identity_bearing_part(other: str) -> None:
    assert step_key("translate", ["a", "b"], {"x": 1}) != other


def test_step_key_ignores_param_mapping_order() -> None:
    assert step_key("t", ["v"], {"a": 1, "b": 2}) == step_key("t", ["v"], {"b": 2, "a": 1})


# --------------------------------------------------------------------------- #
# DiskCacheStore: fail-open IO.
# --------------------------------------------------------------------------- #


def test_missing_key_is_a_miss(store: DiskCacheStore) -> None:
    assert store.get(step_key("k", ["never-written"], {})) is None


def test_put_get_roundtrip_under_the_sharded_path(store: DiskCacheStore) -> None:
    key = step_key("k", ["v"], {})
    store.put(key, "VALUE")
    assert store.get(key) == "VALUE"
    assert (Path(store.root) / key[:2] / f"{key}.json").is_file()


def test_get_or_compute_serves_an_existing_value_without_computing(
    store: DiskCacheStore,
) -> None:
    key = step_key("k", ["v"], {})
    store.put(key, "STORED")
    calls: list[int] = []

    def _compute() -> str:
        calls.append(1)
        return "COMPUTED"

    assert store.get_or_compute(key, _compute) == "STORED"
    assert calls == []


def test_get_or_compute_computes_once_then_serves_the_cache(store: DiskCacheStore) -> None:
    key = step_key("k", ["v"], {})
    calls: list[int] = []

    def _compute() -> str:
        calls.append(1)
        return "COMPUTED"

    assert store.get_or_compute(key, _compute) == "COMPUTED"
    assert store.get_or_compute(key, _compute) == "COMPUTED"
    assert calls == [1]


# --------------------------------------------------------------------------- #
# TranslationEngine: what the key covers.
# --------------------------------------------------------------------------- #


async def test_without_a_cache_the_provider_runs_every_time() -> None:
    engine = _engine(None)
    provider = _Provider()
    await engine.translate_text("e1", "hello world", provider.translate())
    await engine.translate_text("e1", "hello world", provider.translate())
    assert provider.calls == 2


async def test_identical_inputs_hit_the_cache(store: DiskCacheStore) -> None:
    engine = _engine(store)
    provider = _Provider()
    first = await engine.translate_text("e1", "hello world", provider.translate())
    second = await engine.translate_text("e1", "hello world", provider.translate())
    assert provider.calls == 1
    assert first.target == second.target


async def test_the_cache_never_changes_the_delivered_target(store: DiskCacheStore) -> None:
    provider = _Provider()
    uncached = await _engine(None).translate_text("e1", "hello world", provider.translate())
    cached = await _engine(store).translate_text("e1", "hello world", provider.translate())
    assert cached.target == uncached.target


async def test_a_changed_source_misses(store: DiskCacheStore) -> None:
    engine = _engine(store)
    provider = _Provider()
    await engine.translate_text("e1", "hello world", provider.translate())
    await engine.translate_text("e2", "another sentence", provider.translate())
    assert provider.calls == 2


async def test_a_changed_model_misses(store: DiskCacheStore) -> None:
    provider = _Provider()
    await _engine(store, model="m1").translate_text("e1", "hello world", provider.translate())
    await _engine(store, model="m2").translate_text("e1", "hello world", provider.translate())
    assert provider.calls == 2


async def test_a_changed_prompt_version_misses(store: DiskCacheStore) -> None:
    provider = _Provider()
    await _engine(store, prompt="v1").translate_text("e1", "hello world", provider.translate())
    await _engine(store, prompt="v2").translate_text("e1", "hello world", provider.translate())
    assert provider.calls == 2


async def test_the_prompt_context_is_part_of_the_key(store: DiskCacheStore) -> None:
    engine = _engine(store)
    provider = _Provider()
    await engine.translate_text("e1", "hello world", provider.translate(), context="ctx-A")
    await engine.translate_text("e1", "hello world", provider.translate(), context="ctx-A")
    assert provider.calls == 1  # same context: hit
    await engine.translate_text("e1", "hello world", provider.translate(), context="ctx-B")
    assert provider.calls == 2  # changed context: miss


# --------------------------------------------------------------------------- #
# Fail-open: corruption and an unavailable store compute through.
# --------------------------------------------------------------------------- #


async def test_a_corrupt_entry_is_recomputed(store: DiskCacheStore) -> None:
    engine = _engine(store)
    provider = _Provider()
    await engine.translate_text("e1", "hello world", provider.translate())
    for entry in Path(store.root).rglob("*.json"):
        entry.write_text("{not json", encoding="utf-8")
    out = await engine.translate_text("e1", "hello world", provider.translate())
    assert provider.calls == 2
    assert out.target is not None


async def test_an_envelope_with_a_mismatched_key_is_a_miss(store: DiskCacheStore) -> None:
    # The stored key must match the file it lives under, so a split-brain entry
    # cannot be served as if it were this unit's draft.
    key = step_key("translate", ["m1", "v1", "hello world"], {})
    path = Path(store.root) / key[:2] / f"{key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"key": "some-other-key", "text": "STALE"}), encoding="utf-8")

    provider = _Provider()
    out = await _engine(store).translate_text("e1", "hello world", provider.translate())
    assert provider.calls == 1
    assert out.target == "[hello world]"


async def test_an_unavailable_store_falls_back_to_the_provider() -> None:
    provider = _Provider()
    out = await _engine(_BrokenStore()).translate_text("e1", "hello world", provider.translate())
    assert provider.calls == 1
    assert out.target == "[hello world]"


# --------------------------------------------------------------------------- #
# The non-draft value cache (macro chunks) shares the store, keyed by kind.
# --------------------------------------------------------------------------- #


def test_unwritten_value_is_a_miss(store: DiskCacheStore) -> None:
    assert _engine(store).cached_value("chunk-A", kind="translate_chunk") is None


def test_value_roundtrips_under_its_kind(store: DiskCacheStore) -> None:
    engine = _engine(store)
    engine.remember_value("chunk-A", '{"b1": "x"}', kind="translate_chunk")
    assert engine.cached_value("chunk-A", kind="translate_chunk") == '{"b1": "x"}'


def test_value_kinds_are_isolated(store: DiskCacheStore) -> None:
    engine = _engine(store)
    engine.remember_value("chunk-A", "value", kind="translate_chunk")
    assert engine.cached_value("chunk-A", kind="other_kind") is None


def test_value_cache_is_absent_without_a_store() -> None:
    engine = _engine(None)
    engine.remember_value("chunk-A", "value", kind="translate_chunk")
    assert engine.cached_value("chunk-A", kind="translate_chunk") is None


# --------------------------------------------------------------------------- #
# Bounded store: the cache never deleted an entry, so it grew without bound.
# --------------------------------------------------------------------------- #


def test_disk_cache_prunes_oldest_entries_past_the_cap() -> None:
    import time

    with tempfile.TemporaryDirectory() as root:
        store = DiskCacheStore(root, max_entries=3)
        for i in range(3):
            store.put(step_key("k", [str(i)], {}), f"v{i}")
            time.sleep(0.01)
        assert len(list(Path(root).rglob("*.json"))) == 3

        # Writing past the cap prunes the oldest down to the hysteresis target.
        store.put(step_key("k", ["new"], {}), "vnew")
        store.prune()
        remaining = sorted(p.name for p in Path(root).rglob("*.json"))
        assert len(remaining) <= 3
        assert store._path(step_key("k", ["new"], {})).exists()


def test_disk_cache_prune_never_raises_on_a_missing_root() -> None:
    store = DiskCacheStore("/nonexistent/ubt-cache-does-not-exist", max_entries=10)
    assert store.prune() == 0
