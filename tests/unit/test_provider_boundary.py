"""Boundary guard for the mock provider, and the tests it exists for (P1-7).

``MockModelProvider`` lives in the production package because production code
ships it: ``--dry-run`` rehearsal, the API's mock mode, the TUI rehearsal and
the credential-free cost assessment all assemble a router around it. Relocating
it to ``tests/`` would therefore break real features.

What the review is right about is the *boundary*: the engine used to answer "is
this a rehearsal run?" with ``isinstance(provider, MockModelProvider)``, which
made every production hot path import the test-double class and left the
distinction implicit. That is now ``BaseModelProvider.is_mock`` — a declared,
overridable contract — and these tests hold the line:

1. the flag is declared on the base class and honoured by the mock;
2. the engine no longer reaches for the concrete class at all;
3. only the four modules that define, subclass, construct or re-export the mock
   may import it, enforced by AST rather than by convention.
"""

import ast
from pathlib import Path

import pytest

from ubt.core.router.provider import BaseModelProvider, MockModelProvider

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
UBT_DIR = REPO_ROOT / "ubt"

#: Modules allowed to reference the concrete mock class, and why:
#: - provider.py defines it;
#: - dry_run.py subclasses it (the rehearsal provider);
#: - assess.py constructs it for a credential-free prompt-assembly router;
#: - router/__init__.py re-exports it as part of the router package's API.
MOCK_REFERENCE_ALLOWLIST = frozenset(
    {
        "ubt/core/router/provider.py",
        "ubt/core/router/__init__.py",
        "ubt/core/engine/dry_run.py",
        "ubt/core/assess.py",
    }
)


def test_is_mock_is_declared_on_the_base_provider_and_set_by_the_mock() -> None:
    """The rehearsal flag is a contract, not a private isinstance coincidence."""
    assert BaseModelProvider.is_mock is False
    assert MockModelProvider().is_mock is True


def test_dry_run_provider_inherits_the_rehearsal_flag() -> None:
    """--dry-run still reads as a mock run after the isinstance removal."""
    from ubt.core.engine.dry_run import DryRunModelProvider

    assert DryRunModelProvider().is_mock is True


def test_a_test_double_can_declare_is_mock_without_the_concrete_class() -> None:
    """A provider defined outside the package can take part in the contract.

    This is the decoupling the flag buys: a stub in ``tests/`` no longer has to
    subclass a production class to be recognized as simulated.
    """

    class _StubProvider(BaseModelProvider):
        is_mock = True

        @property
        def provider_name(self) -> str:
            return "stub"

        async def generate(
            self,
            prompt: str,
            system_prompt: str | None = None,
            model: str | None = None,
            temperature: float | None = 0.3,
            max_tokens: int | None = None,
            reasoning_effort: str | None = None,
        ) -> str:
            return prompt

    assert _StubProvider().is_mock is True


def _module_level_references(path: Path) -> bool:
    """Whether the module mentions ``MockModelProvider`` anywhere in its AST."""
    tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == "MockModelProvider":
            return True
        if isinstance(node, ast.Attribute) and node.attr == "MockModelProvider":
            return True
    return False


def test_only_the_declared_modules_reference_the_mock_provider() -> None:
    """The engine asks the contract, never the class (P1-7).

    Editing the allowlist is the conscious act: adding an entry means a
    production module has taken a hard dependency on simulated text, which is
    exactly the coupling this guard is here to make visible.
    """
    offenders: list[str] = []
    for py_file in sorted(UBT_DIR.rglob("*.py")):
        relative = py_file.relative_to(REPO_ROOT).as_posix()
        if relative in MOCK_REFERENCE_ALLOWLIST:
            continue
        if _module_level_references(py_file):
            offenders.append(relative)

    assert not offenders, (
        "These modules reference MockModelProvider directly; production code must "
        "read BaseModelProvider.is_mock instead, or the boundary regresses: "
        f"{offenders}"
    )


@pytest.mark.parametrize("module_path", sorted(MOCK_REFERENCE_ALLOWLIST))
def test_allowlisted_modules_still_exist(module_path: str) -> None:
    """A moved/renamed module must not leave a silently-unused allowlist entry."""
    assert (REPO_ROOT / module_path).is_file(), f"{module_path} is gone; update the allowlist"
