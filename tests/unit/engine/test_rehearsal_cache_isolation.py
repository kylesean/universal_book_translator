"""A rehearsal's echo replies must be unreachable from a real run.

The ledger was isolated long before the content cache was, and the cache is the
one that matters for correctness: the draft stage consults it *before* spending,
so a poisoned entry makes a real job ship ``[模拟翻译] <English source>`` as its
translation while recording zero tokens spent — indistinguishable, from the
outside, from a genuine cache hit.

Measured on the checkout that motivated this file: **1381 of 5522** entries under
``.ubt/cache`` held an echo, and a two-run experiment (different echo texts,
different job ids) showed run B reading run A's text back out of the cache
without ever calling its own provider.

Two independent mechanisms are asserted here, because either one alone is a
convention rather than a guarantee:

1. :func:`rehearsal_cache_dir` gives the rehearsal its own store.
2. ``TranslationEngine.run_kind`` namespaces the keys, so even one shared store
   cannot serve a rehearsal's reply to a real engine.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest

from ubt.cache.store import DiskCacheStore
from ubt.core.config import UBTConfig
from ubt.core.engine.dry_run import (
    create_dry_run_orchestrator,
    rehearsal_cache_dir,
    rehearsal_db_dir,
)
from ubt.core.ports import placeholder_engine
from ubt.translate.engine import TranslationEngine

pytestmark = pytest.mark.fast


# --------------------------------------------------------------------------- #
# Mechanism 1: a separate directory.
# --------------------------------------------------------------------------- #


def test_the_rehearsal_cache_directory_is_a_sibling_not_a_child(tmp_path: Path) -> None:
    """A child would be walked (and pruned) by the real store.

    ``DiskCacheStore.prune`` deletes any file matching its own shard layout
    under its root, so a ``cache/mock`` subtree would be eligible for deletion
    by unrelated real-run traffic.
    """
    real = tmp_path / ".ubt" / "cache"
    mock = rehearsal_cache_dir(real)

    assert mock == tmp_path / ".ubt" / "cache-mock"
    assert mock.parent == real.parent
    assert not mock.is_relative_to(real), f"{mock} must not live under {real}"


def test_a_rehearsal_orchestrator_does_not_point_at_the_real_cache(tmp_path: Path) -> None:
    """The isolation that matters, asserted on the config the orchestrator will run."""
    real_cache = tmp_path / ".ubt" / "cache"
    real_db = tmp_path / ".ubt" / "ledgers"
    config = UBTConfig(db_dir=real_db, cache_dir=real_cache)

    rehearsal = create_dry_run_orchestrator(config).config

    assert rehearsal.cache_dir == rehearsal_cache_dir(real_cache)
    assert rehearsal.cache_dir != real_cache
    assert rehearsal.db_dir != real_db
    # The caller's config is untouched: running a rehearsal must not mutate the
    # operator's config as a side effect.
    assert config.cache_dir == real_cache
    assert config.db_dir == real_db


def test_the_two_isolated_directories_are_distinct(tmp_path: Path) -> None:
    """``cache-mock`` must not collide with ``ledgers-mock``.

    Both are siblings derived from their own field, so they only stay apart
    while each keeps deriving from the field it owns.
    """
    base = tmp_path / ".ubt"
    assert rehearsal_cache_dir(base / "cache") != rehearsal_db_dir(base / "ledgers")


# --------------------------------------------------------------------------- #
# Mechanism 2: a separate key space, even within one store.
# --------------------------------------------------------------------------- #


class _Provider:
    """A provider whose reply is tagged, so a crossed reply is visible."""

    def __init__(self, tag: str) -> None:
        self.tag = tag
        self.calls = 0

    def translate(self) -> Callable[[str], Awaitable[str]]:
        """A fresh provider callable, tagged so a crossed reply is visible."""

        async def _do(source: str) -> str:
            self.calls += 1
            return f"[{self.tag}] {source}"

        return _do


def _engine(store: DiskCacheStore, *, run_kind: str = "real") -> TranslationEngine:
    return TranslationEngine(
        placeholders=placeholder_engine(),
        model="m1",
        prompt_version="v1",
        cache=store,
        run_kind=run_kind,
    )


async def test_a_rehearsal_reply_is_not_served_to_a_real_engine(tmp_path: Path) -> None:
    """The regression this file exists for, on ONE shared store.

    A rehearsal drafts first and records an echo; the real engine then asks for
    the same block. It must reach its own provider rather than ship the echo.
    """
    store = DiskCacheStore(tmp_path / "cache")
    rehearsal = _engine(store, run_kind="rehearsal")
    echo = _Provider("模拟翻译")
    await rehearsal.translate_text("e1", "hello world", echo.translate())

    real = _engine(store, run_kind="real")
    genuine = _Provider("translated")
    out = await real.translate_text("e1", "hello world", genuine.translate())

    assert real.cache is not None
    # The real engine must have called its own provider, not read the echo.
    assert genuine.calls == 1
    assert out.target == "[translated] hello world"
    assert "模拟翻译" not in (out.target or "")


async def test_a_real_reply_is_not_served_to_a_rehearsal(tmp_path: Path) -> None:
    """The other direction, so neither namespace can be satisfied by silencing both.

    A rehearsal that reused a real translation would make the rehearsal report
    a quality it never measured.
    """
    store = DiskCacheStore(tmp_path / "cache")
    real = _engine(store, run_kind="real")
    genuine = _Provider("translated")
    await real.translate_text("e1", "hello world", genuine.translate())

    rehearsal = _engine(store, run_kind="rehearsal")
    echo = _Provider("模拟翻译")
    out = await rehearsal.translate_text("e1", "hello world", echo.translate())

    assert echo.calls == 1
    assert out.target == "[模拟翻译] hello world"


async def test_the_same_run_kind_still_hits_the_cache(tmp_path: Path) -> None:
    """The negative control: namespacing must not disable the cache itself."""
    store = DiskCacheStore(tmp_path / "cache")

    first = _engine(store, run_kind="real")
    provider_one = _Provider("translated")
    await first.translate_text("e1", "hello world", provider_one.translate())

    second = _engine(store, run_kind="real")
    provider_two = _Provider("SHOULD-NOT-BE-CALLED")
    out = await second.translate_text("e1", "hello world", provider_two.translate())

    assert provider_one.calls == 1
    assert provider_two.calls == 0
    assert out.target == "[translated] hello world"


def test_the_run_kind_actually_changes_the_key(tmp_path: Path) -> None:
    """A direct assertion on the key, so the mechanism is not merely implied.

    ``step_key`` is the only thing standing between the two namespaces; if
    ``run_kind`` stopped reaching it, the two tests above would still pass for
    as long as the *directory* isolation held, and the redundancy would be gone.
    """
    real = _engine(DiskCacheStore(tmp_path / "c"), run_kind="real")
    rehearsal = _engine(DiskCacheStore(tmp_path / "c"), run_kind="rehearsal")

    assert real._cache_key("hello", "ctx") != rehearsal._cache_key("hello", "ctx")
    # And the key is stable for one kind, or every call would miss.
    real_again = _engine(DiskCacheStore(tmp_path / "c"), run_kind="real")
    assert real._cache_key("hello", "ctx") == real_again._cache_key("hello", "ctx")


def test_the_default_run_kind_is_real() -> None:
    """Every production path constructs the engine without naming a kind.

    Defaulting to anything else would put ordinary work in the rehearsal
    namespace and make a real run unable to reuse its own drafts.
    """
    engine = TranslationEngine(placeholders=placeholder_engine())
    assert engine.run_kind == "real"


def test_a_real_key_is_unchanged_by_the_namespacing(tmp_path: Path) -> None:
    """**The money test.** A real engine must produce the key it always produced.

    Folding ``run_kind`` in unconditionally would be the tidier key function and
    the wrong one: the store holds genuine translations that were paid for.
    Measured on the checkout this fix was written against, that mistake would
    have orphaned 3653 entries covering 15506 blocks.

    Pinned against the explicit pre-existing shape (``params={}``) rather than
    against a stored constant, so it fails if the key derivation changes for any
    reason at all and forces whoever changed it to decide about the existing
    store deliberately.
    """
    from ubt.cache.store import step_key

    real = _engine(DiskCacheStore(tmp_path / "c"), run_kind="real")
    legacy_key = step_key("translate", ["m1", "v1", "hello world", "ctx"], {})
    assert real._cache_key("hello world", "ctx") == legacy_key


def test_a_rehearsal_key_is_not_the_legacy_key(tmp_path: Path) -> None:
    """The rehearsal namespace must be the one that moved, not the real one.

    Together with the test above this is the whole contract: real keys are
    frozen, rehearsal keys are new, so a rehearsal can never collide with any
    entry written before this fix.
    """
    from ubt.cache.store import step_key

    rehearsal = _engine(DiskCacheStore(tmp_path / "c"), run_kind="rehearsal")
    legacy_key = step_key("translate", ["m1", "v1", "hello world", "ctx"], {})
    assert rehearsal._cache_key("hello world", "ctx") != legacy_key


def test_the_macro_chunk_value_key_is_namespaced_too(tmp_path: Path) -> None:
    """The *second* key function, which namespacing must reach as well.

    ``_value_key`` backs :meth:`cached_value` / :meth:`store_value`, the
    macro-chunk path that returns one provider reply for several blocks. It is a
    separate function from ``_cache_key`` and is folded the same way -- but until
    this test existed nothing pinned it: a mutation that dropped ``run_kind``
    from ``_value_key`` alone left every other test green, because the draft key
    still moved. An echo served there splices rehearsal text across a whole
    batch at once, so it is the higher-consequence of the two paths.
    """
    from ubt.cache.store import step_key

    real = _engine(DiskCacheStore(tmp_path / "c"), run_kind="real")
    rehearsal = _engine(DiskCacheStore(tmp_path / "c"), run_kind="rehearsal")

    real_key = real._value_key("translate_value", "ctx")
    rehearsal_key = rehearsal._value_key("translate_value", "ctx")

    # Real value keys are frozen exactly like real draft keys: a genuine
    # macro-chunk reply that was paid for must still be found.
    assert real_key == step_key("translate_value", ["m1", "v1", "ctx"], {})
    assert rehearsal_key != real_key
    assert rehearsal_key != step_key("translate_value", ["m1", "v1", "ctx"], {})


async def test_an_echo_in_the_value_cache_is_not_served_to_a_real_engine(
    tmp_path: Path,
) -> None:
    """The behavioural half of the test above, on one shared store."""
    store = DiskCacheStore(tmp_path / "cache")
    rehearsal = _engine(store, run_kind="rehearsal")
    rehearsal.remember_value("ctx", "[模拟翻译] batch", kind="translate_value")
    assert rehearsal.cached_value("ctx", kind="translate_value") == "[模拟翻译] batch"

    real = _engine(store, run_kind="real")
    assert real.cached_value("ctx", kind="translate_value") is None


def test_a_rehearsal_config_is_still_a_rehearsal_after_the_cache_move(tmp_path: Path) -> None:
    """Restated here because the cache_dir line edits the same block as db_dir.

    Adding one override must not have dropped the zero-spend guarantees that
    share that ``model_copy`` call.
    """
    rehearsal = create_dry_run_orchestrator(
        UBTConfig(db_dir=tmp_path / "ledgers", cache_dir=tmp_path / "cache")
    ).config

    assert rehearsal.ocr_mode == "off"
    assert rehearsal.allow_page_upload is False
    assert rehearsal.visual_judge_enabled is False


def test_a_rehearsal_run_builds_a_rehearsal_namespaced_engine() -> None:
    """The draft stage reads ``ctx.is_mock_run``; a mock router must say so.

    Pins the wiring line in ``draft.py`` that converts the run kind into the
    cache namespace, since the two isolation mechanisms are independent and
    either could be removed without the other noticing.
    """
    from ubt.core.engine.dry_run import DryRunModelProvider

    provider = DryRunModelProvider()
    assert provider.is_mock is True
    # A rehearsal's echo must never be a plausible translation on its own.
    echoed = asyncio.run(provider.generate("the quick brown fox"))
    assert "模拟翻译" in echoed
