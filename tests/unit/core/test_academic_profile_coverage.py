from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from ubt.core import archetype
from ubt.core.engine.stages.draft import ACADEMIC_PROFILES

pytestmark = pytest.mark.fast


def _inferred_profile_names() -> set[str]:
    tree = ast.parse(inspect.getsource(archetype.infer_profile_from_archetype))
    names: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Return)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            names.add(node.value.value)
    names |= _detected_domain_names()
    return names


def _detected_domain_names() -> set[str]:
    from ubt.core.archetype import DOMAIN_PATTERNS

    return {*DOMAIN_PATTERNS, "general"}


_PROFILE_VOCAB = frozenset(
    {"paper", "fiction", "general", "humanities", "semiconductor", "textbook", "technical"}
)


def test_humanities_is_not_reachable_but_still_classified() -> None:
    assert "humanities" not in _detected_domain_names(), (
        "humanities became reachable — add it to DOMAIN_PATTERNS intentionally"
    )
    assert "humanities" in ACADEMIC_PROFILES


@pytest.mark.parametrize("profile", sorted(_inferred_profile_names()))
def test_every_inferable_profile_is_a_known_name(profile: str) -> None:
    from ubt.core.job_options import profile_name_is_valid

    assert profile_name_is_valid(profile), (
        f"{profile!r} is inferable but is not a legal resource-directory component"
    )


def test_humanities_suppresses_the_rolling_summary() -> None:
    from ubt.core.config import UBTConfig

    manifest = _manifest_with_chapters(12)
    rolling, _, _ = _policy(manifest, "humanities", UBTConfig(enable_rolling_summary=True))
    assert rolling is False


def test_paper_and_humanities_agree_on_the_rollup() -> None:
    from ubt.core.config import UBTConfig

    manifest = _manifest_with_chapters(12)
    paper_rolling, _, _ = _policy(manifest, "paper", UBTConfig(enable_rolling_summary=True))
    humanities_rolling, _, _ = _policy(
        manifest, "humanities", UBTConfig(enable_rolling_summary=True)
    )
    assert paper_rolling == humanities_rolling


def test_narrative_profiles_keep_the_rolling_summary() -> None:
    from ubt.core.config import UBTConfig

    manifest = _manifest_with_chapters(12)
    for profile in ("general", "fiction"):
        rolling, _, _ = _policy(manifest, profile, UBTConfig(enable_rolling_summary=True))
        assert rolling is True, f"{profile} should keep rolling summaries"


def test_long_form_books_over_40_chapters_keep_rolling_summary() -> None:
    from ubt.core.config import UBTConfig

    manifest = _manifest_with_chapters(100)
    rolling, _, _ = _policy(manifest, "fiction", UBTConfig(enable_rolling_summary=True))
    assert rolling is True, "books over 40 chapters should retain rolling summaries"


def test_assess_pricing_uses_the_same_classification() -> None:
    source = Path("ubt/core/assess.py").read_text(encoding="utf-8")
    assert "from ubt.core.engine.stages.draft import ACADEMIC_PROFILES" in source
    assert "not in ACADEMIC_PROFILES" in source
    assert "humanities" in ACADEMIC_PROFILES


def _manifest_with_chapters(count: int) -> object:
    from ubt.core.ir.models import BookManifest, ChapterMeta

    return BookManifest(
        doc_id="test-doc",
        title="Test",
        source_path="test.pdf",
        chapters=[
            ChapterMeta(chapter_id=f"ch{i}", title=f"Chapter {i}", spine_index=i)
            for i in range(count)
        ],
    )


def _policy(manifest: object, profile: str, config: object) -> tuple[bool, bool, int]:
    from ubt.core.engine.stages.draft import resolve_draft_policy

    return resolve_draft_policy(manifest, profile, config, all_blocks_count=200)  # type: ignore[arg-type]
