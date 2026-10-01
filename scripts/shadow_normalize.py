#!/usr/bin/env python
"""§12 Q3 acceptance: canonical text normalization (ADR-0001).

Checks the normalization contract directly and confirms both native readers emit
no presentation ligature or invisible formatting character into their canonical
text. Idempotence matters: the normalized stream is what ``Span.chars`` indexes,
so normalizing twice must not shift an offset.
"""

from __future__ import annotations

from pathlib import Path

from ubt.analyze.normalize import normalize_text

_CASES: tuple[tuple[str, str], ...] = (
    ("\ufb01n", "fin"),  # ﬁ -> fi
    ("\ufb02ow", "flow"),  # ﬂ -> fl
    ("\ufb00", "ff"),
    ("\ufb03", "ffi"),
    ("\ufb04", "ffl"),
    ("co\u00adoperate", "cooperate"),  # soft hyphen removed
    ("a\u200bb", "ab"),  # zero-width space removed
    ("a\u00a0b", "a b"),  # NBSP -> space
    ("plain text", "plain text"),
    ("", ""),
)

_FORBIDDEN = (
    "\u00ad",
    "\u200b",
    "\u200c",
    "\u200d",
    "\u2060",
    "\ufeff",
    "\ufb00",
    "\ufb01",
    "\ufb02",
    "\ufb03",
    "\ufb04",
    "\ufb05",
    "\ufb06",
)


def main() -> int:
    problems: list[str] = []
    for raw, expected in _CASES:
        got = normalize_text(raw)
        if got != expected:
            problems.append(f"normalize_text({raw!r}) = {got!r} != {expected!r}")
        if normalize_text(got) != got:
            problems.append(f"normalize_text is not idempotent on {raw!r}")

    # Both native readers must emit a normalized canonical stream.
    from ubt.analyze.reader_md import read_md
    from ubt.analyze.reader_pdf import read_pdf

    documents = [("md:README", read_md(Path("README.md")))]
    for pdf in sorted(Path("corpus/documents").glob("*.pdf")):
        documents.append((f"pdf:{pdf.name}", read_pdf(pdf)))
    for label, document in documents:
        for bad in _FORBIDDEN:
            if bad in document.source.text:
                problems.append(f"{label}: canonical text contains U+{ord(bad):04X}")
                break

    print(f"\nNormalization acceptance — {len(_CASES)} cases, {len(documents)} document(s)")
    print(f"  problems={len(problems)} -> {'PASS' if not problems else 'FAIL'}")
    for problem in problems:
        print(f"    {problem}")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
