"""Backend capability declaration (render backend lowering layer).

``rigid`` vs ``reflow`` was a whole-document choice forced by two implementations;
the real question is per element -- can *this* backend reproduce *this* element
losslessly at *this* fidelity? A backend answers with a :class:`Capabilities`
value and :func:`ubt.pipeline.steps.realize` walks the fidelity descent asking
that question one element at a time, so the two engines stop being a dichotomy
and become two backends.

Capabilities are *data* (declarative capability invariant): a backend declares what it can
produce, and an unsupported rung is simply skipped. There is no plugin registry
-- backends are constructed where a run is assembled (renderer capability declaration).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Protocol

from ubt.model.ast import Element
from ubt.model.fidelity import Fidelity
from ubt.model.span import CanonicalSource


@dataclass(frozen=True, slots=True)
class Produced:
    """One backend's realization of an element, before verification.

    ``payload`` is the delivered text/markup the element's verifier checks -- a
    reconstructed formula or table, a translated target, or the opaque source
    slice. ``fragment`` is the backend's layout artifact (Typst source for a
    reflowed element), empty when the realization is placed rather than drawn.
    ``note`` is free-form provenance carried into the attestation.
    """

    payload: str
    note: str = ""
    fragment: str = ""


@dataclass(frozen=True, slots=True)
class Capabilities:
    """What one backend can realize, declared as data.

    ``supported`` is the set of ``(element class, fidelity)`` pairs the backend
    can produce; ``reflows`` says whether it re-typesets text rather than only
    placing opaque slices (it is reporting, not a switch -- the supported set
    already decides every assignment).
    """

    supported: frozenset[tuple[type[Element], Fidelity]] = frozenset()
    reflows: bool = False

    def supports(self, element_type: type[Element], fidelity: Fidelity) -> bool:
        """Whether this backend declares it can produce that class at that rung."""
        return (element_type, fidelity) in self.supported


class Backend(Protocol):
    """A lowering backend: declares capabilities, produces realizations."""

    name: ClassVar[str]

    def capabilities(self) -> Capabilities: ...

    def produce(
        self, element: Element, fidelity: Fidelity, source: CanonicalSource
    ) -> Produced | None:
        """Realize one element at one fidelity, or ``None`` if this one fails.

        ``None`` is a *skip*, not a loss: ``realize`` then tries the next rung
        down the descent.
        """
        ...


__all__ = ["Backend", "Capabilities", "Produced"]
