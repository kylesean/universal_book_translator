"""``ubt.segment`` -- the placeholder engine: protect spans, then restore them.

The maskers share one owner. The *detectors* are
unchanged (they know LaTeX vs fenced code vs citations), but the order they run
in, the reverse order they restore in, and the integrity reporting are stated
once, here, instead of being restated -- and kept in sync by hand -- at the
draft call site.

The order is load-bearing: **email -> code -> math -> soup -> citation**. Earlier
families must already be opaque to later ones, so a citation-like bracket inside
a math span is caught as math, not stolen by the citation pass, and an address
inside a URL is caught by the URL pass, not the bare-email pass. Restore runs the
exact reverse, threading the text so every namespace is verified with the same
checksummed contract.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from ubt.core.cleaners.citation_masker import CitationMasker
from ubt.core.cleaners.code_masker import CodeMasker
from ubt.core.cleaners.email_masker import EmailMasker
from ubt.core.cleaners.mask_tokens import UnmaskReport
from ubt.core.cleaners.math_masker import MathMasker
from ubt.core.cleaners.soup_math import SoupMathMasker
from ubt.model.segment import Placeholder

_KINDS = ("email", "code", "math", "soup", "citation")


@dataclass(frozen=True, slots=True)
class MaskedSource:
    """A source string with its protected spans replaced, plus the per-family maps."""

    text: str
    email_map: dict[str, str]
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
            "email": self.email_map,
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
    email: UnmaskReport
    code: UnmaskReport
    math: UnmaskReport
    soup: UnmaskReport
    citation: UnmaskReport

    @property
    def reports(self) -> tuple[tuple[str, UnmaskReport], ...]:
        """(flag-label, report) in the order the draft stage records them."""
        return (
            ("email_token_corrupt", self.email),
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
        email: EmailMasker,
        code: CodeMasker,
        math: MathMasker,
        soup: SoupMathMasker,
        citation: CitationMasker,
    ) -> None:
        self._email = email
        self._code = code
        self._math = math
        self._soup = soup
        self._citation = citation

    def mask(self, text: str) -> MaskedSource:
        # Gate 1: addresses and URLs are the most literal spans; masking them
        # first keeps a URL's ``user@host`` from being split by the email pass
        # and a query string's brackets from being read as a citation.
        masked, email_map = self._email.mask(text)
        masked, code_map = self._code.mask(masked)
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
            email_map=email_map,
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
        email = self._email.unmask_checked(code.text, masked.email_map)
        return RestoreOutcome(
            text=email.text, email=email, code=code, math=math, soup=soup, citation=citation
        )

    def missing_literal_spans(
        self, source_text: str, target_text: str
    ) -> tuple[tuple[str, str], ...]:
        """Literal protected spans in ``source_text`` that are absent from ``target_text``.

        The draft stage restores emails/URLs and code *before* repair sees the
        block, so a repair candidate that drops one leaves no mask token behind
        for FastPass's residual-token check to find. Compare the maskers'
        own inventories instead; order is irrelevant (translations reorder
        spans legitimately), only multiplicity counts.

        Math is deliberately excluded: FastPass already compares math spans.
        Citations are counted, not value-matched — renumbering is legitimate,
        a net deletion is not. Soup spans (delimiter-free unicode math mixed
        into prose) are literal atoms and are compared exactly, like code.
        """
        source = self.mask(source_text)
        target = self.mask(target_text)
        missing: list[tuple[str, str]] = []
        for label, source_map, target_map in (
            ("email", source.email_map, target.email_map),
            ("code", source.code_map, target.code_map),
            ("soup", source.soup_map, target.soup_map),
        ):
            remainder = Counter(source_map.values()) - Counter(target_map.values())
            for original, count in remainder.items():
                missing.extend((label, original) for _ in range(count))
        # Citations may be renumbered, so values are not compared literally; a
        # net loss is what matters. A repair that drops one marker leaves the
        # reader an uncited claim, which the exact-value families above cannot
        # see because every citation span is different by design.
        lost_citations = len(source.cite_map) - len(target.cite_map)
        if lost_citations > 0:
            missing.extend(
                ("citation", f"{lost_citations} marker(s) missing") for _ in range(lost_citations)
            )
        return tuple(missing)


def default_placeholder_engine() -> PlaceholderEngine:
    """The standard engine: the detectors, wired in the one owned order.

    Callers that want the pipeline's placeholder behaviour construct this rather
    than restating the order (draft, export's XLIFF companion, the quality
    report), so the order has a single owner.
    """
    return PlaceholderEngine(
        email=EmailMasker(),
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
