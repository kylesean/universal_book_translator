"""Dynamic footer-boilerplate harvester using consensus alignment.

Autonomously learns recurring publisher legal notices and copyright footers
directly from book samples without hardcoding publisher names. Running headers
and page markers are handled by the fixed structural patterns in
:meth:`BoilerplateFingerprint.clean_head`; consensus header harvesting is not
implemented, so ``header_patterns`` is reserved and always empty.
"""

from __future__ import annotations

import difflib
import re
from collections import Counter
from dataclasses import dataclass

_PHOTO_CREDIT_PATTERN = re.compile(
    r"\s*(?:Bettmann/Corbis|Fancy\s+Photography/Veer\s+Images|Getty\s+Images|Shutterstock|[A-Z][a-z]+\s*©\s*\d{4})\s*$",
    re.IGNORECASE,
)

# Anchored to a whole line on purpose. This cleaner is applied per *block* by
# the EPUB adapter, and a paragraph that merely starts with a number is content,
# not a page marker: the unanchored form ate the "3" out of "3. Preheat the
# oven…", the year out of "1812 was…" and the numeral off "IV. On the Origin of
# Species" — and the damaged text was written back into the delivered file.
#
# The roman-numeral alternative is a *validated, lowercase* roman numeral.
# The previous IGNORECASE ``[ivxlcdm]+`` matched any word spelled from those
# letters ("Mild", "Civil", "Dim", "Mix") and deleted the first word of a prose
# block. Front-matter page markers are lowercase roman, so this keeps them.
_ROMAN_NUMERAL = r"(?=[ivxlcdm])m{0,4}(?:cm|cd|d?c{0,3})(?:xc|xl|l?x{0,3})(?:ix|iv|v?i{0,3})"
_LEADING_PAGE_MARKER = re.compile(
    r"^\s*(?:[Pp][Aa][Gg][Ee]\s+\d+|\d+|" + _ROMAN_NUMERAL + r")[ \t]*(?:\n|$)",
)

_RUNNING_HEADER_PATTERN = re.compile(
    r"^\s*(?:[A-Za-z0-9\s,.'-]{2,50}?\s+[^\w\s]?\s*CHAPTER\s+\d+\s+\d+|\d+\s+CHAPTER\s+\d+\s+[^\w\s]?(?:\s+[A-Za-z0-9\s,.'-]{2,50})?)\s*",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class BoilerplateFingerprint:
    """Consensus boilerplate fingerprint learned dynamically for a specific book.

    Only footer disclaimers are harvested; ``header_patterns`` is reserved for
    a future consensus header learner and is currently always empty, while
    :meth:`clean_head` strips headers via fixed structural patterns.
    """

    footer_disclaimers: tuple[str, ...] = ()
    header_patterns: tuple[str, ...] = ()

    def clean_tail(self, text: str) -> tuple[str, str]:
        """Strip dynamically discovered footer disclaimer and photo credits from text tail.

        Returns (cleaned_text, stripped_boilerplate).
        """
        if not text:
            return "", ""

        cleaned = text.strip()
        stripped_parts: list[str] = []

        # 1. Strip trailing photo credit if present at very end
        photo_m = _PHOTO_CREDIT_PATTERN.search(cleaned)
        if photo_m:
            stripped_parts.append(photo_m.group(0).strip())
            cleaned = cleaned[: photo_m.start()].strip()

        # 2. Strip dynamic footer disclaimer if present
        for disclaimer in self.footer_disclaimers:
            disc = disclaimer.lstrip(" .:,;-\t\r\n").rstrip(" \t\r\n")
            if not disc or len(disc) < 20:
                continue

            # Exact rfind near tail
            idx = cleaned.rfind(disc)
            if idx != -1 and idx >= len(cleaned) - len(disc) - 200:
                stripped_parts.append(cleaned[idx:])
                cleaned = cleaned[:idx].rstrip(" .:,;-\t\r\n")
                break

            # Anchor on leading 30 chars
            lead = disc[: min(30, len(disc))]
            lead_idx = cleaned.rfind(lead)
            if lead_idx != -1 and lead_idx >= len(cleaned) - len(disc) - 200:
                stripped_parts.append(cleaned[lead_idx:])
                cleaned = cleaned[:lead_idx].rstrip(" .:,;-\t\r\n")
                break

        # 3. Strip photo credit again if it was positioned immediately before disclaimer
        photo_m2 = _PHOTO_CREDIT_PATTERN.search(cleaned)
        if photo_m2:
            stripped_parts.append(photo_m2.group(0).strip())
            cleaned = cleaned[: photo_m2.start()].strip()

        return cleaned, " ".join(stripped_parts).strip()

    def clean_head(self, text: str) -> tuple[str, str]:
        """Strip running headers and page markers from text head.

        Returns (cleaned_text, stripped_header).
        """
        if not text:
            return "", ""

        cleaned = text.strip()
        stripped_parts: list[str] = []

        # 1. Strip leading page markers like 'Page 42' or standalone page number
        page_m = _LEADING_PAGE_MARKER.match(cleaned)
        if page_m:
            matched = page_m.group(0)
            # Avoid stripping if the whole text is just a number
            if len(matched) < len(cleaned):
                stripped_parts.append(matched.strip())
                cleaned = cleaned[page_m.end() :].strip()

        # 2. Strip running chapter headers
        hdr_m = _RUNNING_HEADER_PATTERN.match(cleaned)
        if hdr_m:
            matched = hdr_m.group(0)
            if len(matched) < len(cleaned):
                stripped_parts.append(matched.strip())
                cleaned = cleaned[hdr_m.end() :].strip()

        return cleaned, " ".join(stripped_parts).strip()

    def clean(self, text: str) -> str:
        """Strip both head and tail noise, returning clean core text."""
        after_tail, _ = self.clean_tail(text)
        after_head, _ = self.clean_head(after_tail)
        return after_head


class DynamicBoilerplateHarvester:
    """Harvests recurring footer boilerplate by consensus across page samples."""

    def __init__(
        self,
        min_sample_length: int = 150,
        tail_scan_length: int = 600,
        min_match_size: int = 50,
        min_prevalence_ratio: float = 0.40,
    ) -> None:
        self.min_sample_length = min_sample_length
        self.tail_scan_length = tail_scan_length
        self.min_match_size = min_match_size
        self.min_prevalence_ratio = min_prevalence_ratio

    def harvest(self, sample_texts: list[str]) -> BoilerplateFingerprint:
        """Analyze page samples and extract consensus boilerplate fingerprints."""
        valid_samples = [
            t.strip() for t in sample_texts if len(t.strip()) >= self.min_sample_length
        ]
        if len(valid_samples) < 3:
            return BoilerplateFingerprint()

        tails = [t[-self.tail_scan_length :] for t in valid_samples]
        candidates = Counter[str]()

        # Sample pairwise comparisons across a sliding window
        window = min(8, len(tails))
        for i in range(len(tails)):
            for j in range(i + 1, min(i + window, len(tails))):
                matcher = difflib.SequenceMatcher(None, tails[i], tails[j])
                match = matcher.find_longest_match(0, len(tails[i]), 0, len(tails[j]))
                if match.size >= self.min_match_size:
                    anchor = tails[i][match.a : match.a + 40].strip()
                    if len(anchor) >= 30:
                        candidates[anchor] += 1

        footers: list[str] = []
        threshold = int(len(tails) * self.min_prevalence_ratio)

        for anchor, _ in candidates.most_common(3):
            matching_tails = [t for t in tails if anchor in t]
            if len(matching_tails) >= max(3, threshold):
                # Expand to maximal consensus suffix
                first_tail = matching_tails[0]
                idx = first_tail.find(anchor)
                common = first_tail[idx:].strip()

                for t in matching_tails[1:]:
                    while common and common not in t:
                        common = common[:-1].strip()

                if len(common) >= self.min_match_size and common not in footers:
                    footers.append(common)

        return BoilerplateFingerprint(footer_disclaimers=tuple(footers))
