#!/usr/bin/env python
"""Phase-4 acceptance: content-addressed caching for the translate step.

``TranslationEngine.translate_text`` is the pure translate step; wrapping it is
the ADR's "translate" cache. The key covers the masked source *and* the model
and prompt version, so a changed prompt or model cannot reuse an old draft. The
cache is fail-open: a miss, an unavailable store, or a corrupt entry computes
through the provider.

Checks: identical inputs hit the cache (provider called once), a changed source /
model / prompt version misses, a corrupt entry is a miss (fail-open), and the
cache never changes the delivered target (equals the uncached translation).
"""

from __future__ import annotations

import asyncio
import tempfile
from collections.abc import Awaitable, Callable
from pathlib import Path

from ubt.cache.store import DiskCacheStore
from ubt.segment.placeholders import default_placeholder_engine
from ubt.translate.engine import TranslationEngine


def _engine(
    store: DiskCacheStore | None, *, model: str = "m1", prompt: str = "v1"
) -> TranslationEngine:
    return TranslationEngine(
        placeholders=default_placeholder_engine(), model=model, prompt_version=prompt, cache=store
    )


class _Counter:
    def __init__(self) -> None:
        self.calls = 0

    def translate(self, prefix: str = "") -> Callable[[str], Awaitable[str]]:
        async def _fn(masked: str) -> str:
            self.calls += 1
            return f"{prefix}[{masked}]"

        return _fn


def _run() -> list[str]:
    problems: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        store = DiskCacheStore(tmp)

        counter = _Counter()
        engine = _engine(store)
        first = asyncio.run(engine.translate_text("e1", "hello world", counter.translate()))
        second = asyncio.run(engine.translate_text("e1", "hello world", counter.translate()))
        if counter.calls != 1:
            problems.append(
                f"identical inputs called the provider {counter.calls} times, expected 1"
            )
        if first.target != second.target:
            problems.append("cached target differs from the first target")

        # A different source is a different unit: provider called again.
        asyncio.run(engine.translate_text("e2", "another sentence", counter.translate()))
        if counter.calls != 2:
            problems.append(f"a new source did not miss (calls={counter.calls})")

        # A different prompt version must not reuse the old draft.
        engine_v2 = _engine(store, prompt="v2")
        asyncio.run(engine_v2.translate_text("e1", "hello world", counter.translate()))
        if counter.calls != 3:
            problems.append(f"a new prompt_version reused the cache (calls={counter.calls})")

        # A different model must not reuse either.
        engine_m2 = _engine(store, model="m2")
        asyncio.run(engine_m2.translate_text("e1", "hello world", counter.translate()))
        if counter.calls != 4:
            problems.append(f"a new model reused the cache (calls={counter.calls})")

        # Corrupt entry -> miss -> recompute (fail-open), and target still correct.
        key_files = list(Path(tmp).rglob("*.json"))
        if not key_files:
            problems.append("no cache entries were written")
        else:
            for key_file in key_files:
                key_file.write_text("{not json", encoding="utf-8")
            before = counter.calls
            out = asyncio.run(engine_v2.translate_text("e1", "hello world", counter.translate()))
            if counter.calls != before + 1:
                problems.append("a corrupt entry was served instead of recomputed")
            if out.target is None:
                problems.append("corrupt-entry fallback produced no target")
    return problems


def main() -> int:
    problems = _run()
    print("\nPhase-4 translate cache acceptance")
    print(f"  problems={len(problems)} -> {'PASS' if not problems else 'FAIL'}")
    for problem in problems:
        print(f"    {problem}")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
