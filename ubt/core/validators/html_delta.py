"""HTML and Markdown structural delta validator, tiered by confidence.

Structural findings (dropped/duplicated/fabricated images, malformed
attributes) are hard errors: they only ever appear when markup was lost or
broken, so a mismatch means RETRY. Formatting findings (emphasis-tag counts,
dropped anchor hrefs) are advisory warnings — fluent translation legitimately
adds and removes ``<b>``/``<em>`` at will, and a hard gate there would reject
good drafts by the thousands. Consumers surface warnings as flags, never as
re-runs.
"""

import re
from collections import Counter
from html.parser import HTMLParser
from typing import Any

from ubt.core.validators.base import ContentValidator, ValidationResult

_MD_IMG_RE = re.compile(r"(?<!\\)!\[[^\]]*\]\(\s*([^)\s]+)[^)]*\)")
_VALID_ATTR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_:.\-]*$")
_FORMAT_TAG_RE = re.compile(r"<(b|strong|i|em)\b", re.IGNORECASE)
_HREF_RE = re.compile(r"<a\b[^>]*?href\s*=\s*[\"']([^\"']+)", re.IGNORECASE)


class _ImgTagCollector(HTMLParser):
    """Collects every <img> tag and attributes using stdlib HTMLParser."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.records: list[tuple[str, list[tuple[str, str | None]]]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "img":
            self.records.append((self.get_starttag_text() or "", list(attrs)))

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)


class HTMLDeltaValidator(ContentValidator):
    """Verifies that translated text preserves structural image tags and attributes."""

    def validate(self, original: str, translated: str) -> ValidationResult:
        src_html, src_md, src_bad = self._scan_image_refs(original)
        tgt_html, tgt_md, tgt_bad = self._scan_image_refs(translated)
        warnings = self._formatting_warnings(original, translated)

        errors: list[str] = []
        details: dict[str, Any] = {}

        # 1. Delta check on malformed <img> tags (unescaped quotes inside
        # alt/title break attribute parsing). Counted per tag occurrence, not
        # per recovered attribute name: the bogus names are derived from the
        # attribute *content*, so a faithful translation of a dirty source
        # ("say "hi"" → "hello "world"") produces different names and a
        # name-keyed diff can never cancel — it RETRIED every block whose
        # source was already malformed.
        new_bad_count = len(tgt_bad) - len(src_bad)

        if new_bad_count > 0:
            new_bad_examples = tgt_bad[:new_bad_count]
            for raw_tag, attr_name in new_bad_examples:
                errors.append(
                    f"Malformed <img> tag introduced: attribute '{attr_name}' in tag '{raw_tag}' "
                    f"is invalid (unescaped quote inside alt/title attribute)."
                )
            details["malformed_tags"] = new_bad_examples

        # 2. Match counts of image sources
        missing_html = sorted((src_html - tgt_html).items())
        extra_html = sorted((tgt_html - src_html).items())
        missing_md = sorted((src_md - tgt_md).items())
        extra_md = sorted((tgt_md - src_md).items())

        if missing_html or extra_html or missing_md or extra_md:
            diff_msgs: list[str] = []
            if missing_html:
                diff_msgs.append(f"Missing HTML <img src>: {missing_html}")
            if extra_html:
                diff_msgs.append(f"Extra HTML <img src>: {extra_html}")
            if missing_md:
                diff_msgs.append(f"Missing Markdown ![](url): {missing_md}")
            if extra_md:
                diff_msgs.append(f"Extra Markdown ![](url): {extra_md}")
            errors.append("; ".join(diff_msgs))
            details["image_diff"] = {
                "missing_html": missing_html,
                "extra_html": extra_html,
                "missing_md": missing_md,
                "extra_md": extra_md,
            }

        if errors:
            if warnings:
                details["formatting_warnings"] = warnings
            return ValidationResult.failure(
                error_code="HTML_DELTA_MISMATCH",
                message=" | ".join(errors),
                suggested_action="RETRY",
                details=details,
            )

        if warnings:
            return ValidationResult(is_valid=True, details={"formatting_warnings": warnings})
        return ValidationResult.success()

    def _formatting_warnings(self, original: str, translated: str) -> list[str]:
        """Advisory markup-drift findings that must never fail the gate.

        Emphasis-tag deltas and dropped anchor hrefs are counted loosely on
        purpose: the consumer records them as ``error_flags`` for the human
        review queue, and a false positive costs one extra look while a false
        negative would have to be caught by the deterministic structural tier.
        """
        warnings: list[str] = []
        src_fmt = Counter(m.group(1).lower() for m in _FORMAT_TAG_RE.finditer(original))
        tgt_fmt = Counter(m.group(1).lower() for m in _FORMAT_TAG_RE.finditer(translated))
        for tag in sorted(set(src_fmt) | set(tgt_fmt)):
            if src_fmt[tag] != tgt_fmt[tag]:
                warnings.append(
                    f"formatting tag drift: <{tag}> x{src_fmt[tag]} in source, "
                    f"x{tgt_fmt[tag]} in target"
                )
        src_hrefs = Counter(m.group(1) for m in _HREF_RE.finditer(original))
        tgt_hrefs = Counter(m.group(1) for m in _HREF_RE.finditer(translated))
        for href in sorted((src_hrefs - tgt_hrefs).elements()):
            warnings.append(f"anchor href dropped: {href}")
        return warnings

    def _scan_img_tags(self, text: str) -> tuple[Counter[str], list[tuple[str, str]]]:
        src_counts: Counter[str] = Counter()
        bad_attrs: list[tuple[str, str]] = []
        if "<img" not in text.lower():
            return src_counts, bad_attrs

        parser = _ImgTagCollector()
        try:
            parser.feed(text)
            parser.close()
        except Exception as err:
            bad_attrs.append(("<unparseable input>", f"<parser error: {err}>"))
            return src_counts, bad_attrs

        for raw_tag, attrs in parser.records:
            for name, _ in attrs:
                if not _VALID_ATTR_NAME_RE.match(name):
                    # One entry per malformed TAG (first offending attribute
                    # names the shape); the delta check counts occurrences.
                    bad_attrs.append((raw_tag, name))
                    break
            for name, val in attrs:
                if name.lower() == "src" and val:
                    src_counts[val] += 1
        return src_counts, bad_attrs

    def _scan_image_refs(
        self, text: str
    ) -> tuple[Counter[str], Counter[str], list[tuple[str, str]]]:
        html_srcs, bad_attrs = self._scan_img_tags(text)
        md_srcs = Counter(_MD_IMG_RE.findall(text))
        return html_srcs, md_srcs, bad_attrs
