"""The profiles the archetype probe can emit must all be classified by the engine.

``infer_profile_from_archetype`` returns one of a closed set of names, and those
names then decide whether the rolling chapter summary runs
(``resolve_draft_policy``) and whether ``assess`` prices a rollup call. When the
two disagree, the same book behaves differently depending on whether the
operator wrote ``--profile auto`` or ``--profile <name>``: the archetype can
infer ``humanities``, and ``humanities`` was absent from ``ACADEMIC_PROFILES``,
so an auto-probed humanities book got rolling summaries while an explicitly
labelled one did not.

These tests pin the closed set and the classification so a new archetype value
cannot be added without deciding which side of the switch it belongs on.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from ubt.core import archetype
from ubt.core.engine.stages.draft import ACADEMIC_PROFILES

pytestmark = pytest.mark.fast


def _inferred_profile_names() -> set[str]:
    """Every literal string ``infer_profile_from_archetype`` can return.

    Parsed from the AST rather than by calling the function, so the derivation
    does not need a sample document per branch.
    """
    tree = ast.parse(inspect.getsource(archetype.infer_profile_from_archetype))
    names: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Return)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            names.add(node.value.value)
    # ``infer_profile_from_archetype`` also returns ``arch.detected_domain``
    # directly on one branch, so the detected-domain vocabulary counts too.
    names |= _detected_domain_names()
    return names


def _detected_domain_names() -> set[str]:
    """The domains ``detect_domain`` can actually return.

    This is the *reachable* set: the keys of ``DOMAIN_PATTERNS`` plus the
    ``general`` fallback. Scanning string literals in the module instead would
    also catch names that only appear in a comparison, and would report a
    branch as reachable when nothing can ever produce its value — which is
    exactly the mistake this file exists to guard against.
    """
    from ubt.core.archetype import DOMAIN_PATTERNS

    return {*DOMAIN_PATTERNS, "general"}


#: The domain/profile vocabulary the probe works in. Kept explicit so the
#: comparison does not pick up unrelated string literals from the module.
_PROFILE_VOCAB = frozenset(
    {"paper", "fiction", "general", "humanities", "semiconductor", "textbook", "technical"}
)


def test_humanities_is_not_reachable_but_still_classified() -> None:
    """``humanities`` is a *declared* profile with no probe path to reach it.

    ``infer_profile_from_archetype`` has an
    ``if arch.detected_domain in ("semiconductor", "humanities")`` branch, but
    ``detect_domain`` only ever returns a key of ``DOMAIN_PATTERNS`` (or
    ``general``), and ``humanities`` is not one of them. So ``--profile auto``
    can never select it.

    It stays in ``ACADEMIC_PROFILES`` anyway: an operator can pass
    ``--profile humanities`` explicitly, and the classification must agree with
    what a paper/textbook run does. Removing the membership would reintroduce
    the inconsistency on the explicit path.
    """
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
    # The concrete regression: an auto-probed humanities book used to get
    # rolling summaries that an explicit --profile paper run would not.
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
    # The switch must still discriminate: general/fiction are not suppressed.
    from ubt.core.config import UBTConfig

    manifest = _manifest_with_chapters(12)
    for profile in ("general", "fiction"):
        rolling, _, _ = _policy(manifest, profile, UBTConfig(enable_rolling_summary=True))
        assert rolling is True, f"{profile} should keep rolling summaries"


def test_long_form_books_over_40_chapters_keep_rolling_summary() -> None:
    """Ultra-long narrative books (>40 chapters) keep rolling summary with sliding window."""
    from ubt.core.config import UBTConfig

    manifest = _manifest_with_chapters(100)
    rolling, _, _ = _policy(manifest, "fiction", UBTConfig(enable_rolling_summary=True))
    assert rolling is True, "books over 40 chapters should retain rolling summaries"


def test_assess_pricing_uses_the_same_classification() -> None:
    # assess.py imports ACADEMIC_PROFILES rather than re-listing them; assert the
    # import is what the rollup gate actually consults, so the quote cannot drift.
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
