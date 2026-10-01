"""``ubt.segment`` -- the placeholder engine: protect spans, then restore them.

Phase 2 collapses the four maskers into one owner. The four *detectors* are
unchanged (they know LaTeX vs fenced code vs citations), but the order they run
in, the reverse order they restore in, and the integrity reporting are now
stated once, here, instead of being restated -- and kept in sync by hand -- at
the draft call site.

The order is load-bearing: **code -> math -> soup -> citation**. Earlier
families must already be opaque to later ones, so a citation-like bracket inside
a math span is caught as math, not stolen by the citation pass. Restore runs the
exact reverse, threading the text so every namespace is verified with the same
checksummed contract.
"""

from __future__ import annotations

from dataclasses import dataclass

from ubt.core.cleaners.citation_masker import CitationMasker
from ubt.core.cleaners.code_masker import CodeMasker
from ubt.core.cleaners.mask_tokens import UnmaskReport
from ubt.core.cleaners.math_masker import MathMasker
from ubt.core.cleaners.soup_math import SoupMathMasker
from ubt.model.segment import Placeholder

_KINDS = ("code", "math", "soup", "citation")


@dataclass(frozen=True, slots=True)
class MaskedSource:
    """A source string with its protected spans replaced, plus the per-family maps."""

    text: str
    code_map: dict[str, str]
    math_map: dict[str, str]
    soup_map: dict[str, str]
    cite_map: dict[str, str]

    @property
    def placeholders(self) -> tuple[Placeholder, ...]:
        """Every *visible* protected span, tagged by family, in mask order.

        A token hidden inside another token's original (a math environment
        nested inside inline math) is not in the masked text; it is an
        implementation detail of the restore pass, not part of the segment a
        translator or an XLIFF reader sees.
        """
        maps = {
            "code": self.code_map,
            "math": self.math_map,
            "soup": self.soup_map,
            "citation": self.cite_map,
        }
        return tuple(
            Placeholder(token=token, kind=kind, original=original)
            for kind in _KINDS
            for token, original in maps[kind].items()
            if token in self.text
        )


@dataclass(frozen=True, slots=True)
class RestoreOutcome:
    """The restored text plus one integrity report per family."""

    text: str
    code: UnmaskReport
    math: UnmaskReport
    soup: UnmaskReport
    citation: UnmaskReport

    @property
    def reports(self) -> tuple[tuple[str, UnmaskReport], ...]:
        """(flag-label, report) in the order the draft stage records them."""
        return (
            ("soup_token_corrupt", self.soup),
            ("math_token_corrupt", self.math),
            ("cite_token_corrupt", self.citation),
            ("code_token_corrupt", self.code),
        )

    @property
    def clean(self) -> bool:
        return all(report.clean for _, report in self.reports)


class PlaceholderEngine:
    """Owns the placeholder mask order and its exact reverse."""

    def __init__(
        self,
        *,
        code: CodeMasker,
        math: MathMasker,
        soup: SoupMathMasker,
        citation: CitationMasker,
    ) -> None:
        self._code = code
        self._math = math
        self._soup = soup
        self._citation = citation

    def mask(self, text: str) -> MaskedSource:
        masked, code_map = self._code.mask(text)
        # Gate 2: inline math is masked before citations so mathematical
        # intervals ($x \in [0, 1]$) and matrix brackets are protected as math
        # atoms and never intercepted by the citation pass.
        masked, math_map = self._math.mask(masked)
        # Gate 2b: delimiter-free unicode math merged into narrative is masked
        # behind its own namespace; restored verbatim below.
        masked, soup_map = self._soup.mask(masked)
        masked, cite_map = self._citation.mask(masked)
        return MaskedSource(
            text=masked,
            code_map=code_map,
            math_map=math_map,
            soup_map=soup_map,
            cite_map=cite_map,
        )

    def restore(self, text: str, masked: MaskedSource) -> RestoreOutcome:
        citation = self._citation.unmask_checked(text, masked.cite_map)
        soup = self._soup.unmask_checked(citation.text, masked.soup_map or {})
        math = self._math.unmask_checked(soup.text, masked.math_map)
        code = self._code.unmask_checked(math.text, masked.code_map)
        return RestoreOutcome(text=code.text, code=code, math=math, soup=soup, citation=citation)


def default_placeholder_engine() -> PlaceholderEngine:
    """The standard engine: the four detectors, wired in the one owned order.

    Callers that want the pipeline's placeholder behaviour construct this rather
    than restating the order (draft, export's XLIFF companion, the quality
    report), so the order has a single owner.
    """
    return PlaceholderEngine(
        code=CodeMasker(),
        math=MathMasker(),
        soup=SoupMathMasker(),
        citation=CitationMasker(),
    )


__all__ = [
    "MaskedSource",
    "PlaceholderEngine",
    "RestoreOutcome",
    "default_placeholder_engine",
]
