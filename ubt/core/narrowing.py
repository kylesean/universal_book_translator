"""Invariant checks that survive ``python -O``.

``assert`` is the natural way to state "this cannot happen", but ``python -O``
strips every assert from the bytecode, so an invariant stated that way becomes
silence in an optimized run: a ``None`` that should have been caught flows on
and surfaces later as an ``AttributeError`` on ``None``, with no trace of which
invariant was violated.

The project relies on asserts for two different jobs, and they need different
treatment:

* **Type narrowing.** ``assert x is not None`` after a guard that guarantees it
  is how mypy learns the type; the runtime check is incidental. Under ``-O``
  the narrowing is gone and the next attribute access is what fails.
* **Genuine invariants.** A condition that would be a bug if violated. These
  should be explicit raises, as :mod:`ubt.adapters.epub.adapter` already does.

:func:`narrow` serves the first case: it narrows for the type checker *and*
raises an actionable error at runtime, and it is not stripped by ``-O``.
"""

from __future__ import annotations


def narrow[T](value: T | None, *, what: str) -> T:
    """Return ``value``, or raise if it is ``None`` (survives ``python -O``).

    Use where a preceding guard already guarantees non-``None`` and the check
    exists to tell the type checker so. ``what`` names the value for the error
    message, so an optimized run reports *which* invariant failed instead of
    an ``AttributeError`` somewhere downstream.
    """
    if value is None:
        raise AssertionError(f"invariant violated: {what} is None")
    return value
