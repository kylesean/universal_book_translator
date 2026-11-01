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


def test_the_rehearsal_cache_directory_is_a_sibling_not_a_child(tmp_path: Path) -> None:
    real = tmp_path / ".ubt" / "cache"
    mock = rehearsal_cache_dir(real)

    assert mock == tmp_path / ".ubt" / "cache-mock"
    assert mock.parent == real.parent
    assert not mock.is_relative_to(real), f"{mock} must not live under {real}"


def test_a_rehearsal_orchestrator_does_not_point_at_the_real_cache(tmp_path: Path) -> None:
    real_cache = tmp_path / ".ubt" / "cache"
    real_db = tmp_path / ".ubt" / "ledgers"
    config = UBTConfig(db_dir=real_db, cache_dir=real_cache)

    rehearsal = create_dry_run_orchestrator(config).config

    assert rehearsal.cache_dir == rehearsal_cache_dir(real_cache)
    assert rehearsal.cache_dir != real_cache
    assert rehearsal.db_dir != real_db
    assert config.cache_dir == real_cache
    assert config.db_dir == real_db


def test_the_two_isolated_directories_are_distinct(tmp_path: Path) -> None:
    base = tmp_path / ".ubt"
    assert rehearsal_cache_dir(base / "cache") != rehearsal_db_dir(base / "ledgers")


class _Provider:
    def __init__(self, tag: str) -> None:
        self.tag = tag
        self.calls = 0

    def translate(self) -> Callable[[str], Awaitable[str]]:

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
    store = DiskCacheStore(tmp_path / "cache")
    rehearsal = _engine(store, run_kind="rehearsal")
    echo = _Provider("模拟翻译")
    await rehearsal.translate_text("e1", "hello world", echo.translate())

    real = _engine(store, run_kind="real")
    genuine = _Provider("translated")
    out = await real.translate_text("e1", "hello world", genuine.translate())

    assert real.cache is not None
    assert genuine.calls == 1
    assert out.target == "[translated] hello world"
    assert "模拟翻译" not in (out.target or "")


async def test_a_real_reply_is_not_served_to_a_rehearsal(tmp_path: Path) -> None:
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
    real = _engine(DiskCacheStore(tmp_path / "c"), run_kind="real")
    rehearsal = _engine(DiskCacheStore(tmp_path / "c"), run_kind="rehearsal")

    assert real._cache_key("hello", "ctx") != rehearsal._cache_key("hello", "ctx")
    real_again = _engine(DiskCacheStore(tmp_path / "c"), run_kind="real")
    assert real._cache_key("hello", "ctx") == real_again._cache_key("hello", "ctx")


def test_the_default_run_kind_is_real() -> None:
    engine = TranslationEngine(placeholders=placeholder_engine())
    assert engine.run_kind == "real"


def test_a_real_key_is_unchanged_by_the_namespacing(tmp_path: Path) -> None:
    from ubt.cache.store import step_key

    real = _engine(DiskCacheStore(tmp_path / "c"), run_kind="real")
    legacy_key = step_key("translate", ["m1", "v1", "hello world", "ctx"], {})
    assert real._cache_key("hello world", "ctx") == legacy_key


def test_a_rehearsal_key_is_not_the_legacy_key(tmp_path: Path) -> None:
    from ubt.cache.store import step_key

    rehearsal = _engine(DiskCacheStore(tmp_path / "c"), run_kind="rehearsal")
    legacy_key = step_key("translate", ["m1", "v1", "hello world", "ctx"], {})
    assert rehearsal._cache_key("hello world", "ctx") != legacy_key


def test_the_macro_chunk_value_key_is_namespaced_too(tmp_path: Path) -> None:
    from ubt.cache.store import step_key

    real = _engine(DiskCacheStore(tmp_path / "c"), run_kind="real")
    rehearsal = _engine(DiskCacheStore(tmp_path / "c"), run_kind="rehearsal")

    real_key = real._value_key("translate_value", "ctx")
    rehearsal_key = rehearsal._value_key("translate_value", "ctx")

    assert real_key == step_key("translate_value", ["m1", "v1", "ctx"], {})
    assert rehearsal_key != real_key
    assert rehearsal_key != step_key("translate_value", ["m1", "v1", "ctx"], {})


async def test_an_echo_in_the_value_cache_is_not_served_to_a_real_engine(
    tmp_path: Path,
) -> None:
    store = DiskCacheStore(tmp_path / "cache")
    rehearsal = _engine(store, run_kind="rehearsal")
    rehearsal.remember_value("ctx", "[模拟翻译] batch", kind="translate_value")
    assert rehearsal.cached_value("ctx", kind="translate_value") == "[模拟翻译] batch"

    real = _engine(store, run_kind="real")
    assert real.cached_value("ctx", kind="translate_value") is None


def test_a_rehearsal_config_is_still_a_rehearsal_after_the_cache_move(tmp_path: Path) -> None:
    rehearsal = create_dry_run_orchestrator(
        UBTConfig(db_dir=tmp_path / "ledgers", cache_dir=tmp_path / "cache")
    ).config

    assert rehearsal.ocr_mode == "off"
    assert rehearsal.allow_page_upload is False
    assert rehearsal.visual_judge_enabled is False


def test_a_rehearsal_run_builds_a_rehearsal_namespaced_engine() -> None:
    from ubt.core.engine.dry_run import DryRunModelProvider

    provider = DryRunModelProvider()
    assert provider.is_mock is True
    echoed = asyncio.run(provider.generate("the quick brown fox"))
    assert "模拟翻译" in echoed
