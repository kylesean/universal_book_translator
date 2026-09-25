"""Shared constants of the Typst generation / self-healing pipeline.

Centralized definitions for compiler settings, retry limits, and regex patterns
shared between ``typst_reconstructor.py`` and ``typst_healer.py`` to ensure
consistent behavior across compilation, verification, and diagnostics.

Only ``re``, plain data, and frozen dataclasses live here: importing this module
remains free of side effects and heavyweight dependencies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Trailing tracking tag appended to every emitted formula line:
#: ``// [formula <block.id>]``. The reconstructor tags the line, the
#: compile-probe / substitution passes read the id back to map a failed line
#: onto its source block (verbatim degradation, source-graphic swap).
_FORMULA_ID_RE = re.compile(r"// \[formula (\S+)\]\s*$")

#: ``typst --version`` prints e.g. ``typst 0.15.1 (unknown commit)``. The
#: generated markup's meaning (and the stderr dialect the self-healing loop
#: parses) depends on the compiler version, so quality reports pin it.
_TYPST_VERSION_RE = re.compile(r"\btypst\s+(\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.\-+]+)?)")

#: Batch compile-probe rounds per ``verify_math_lines`` pass before the
#: per-line tails take over (bounded further by the wall-clock budget).
_MAX_MATH_VERIFY_ROUNDS = 3

#: Wall-clock cap for the Typst self-healing ladder: 3 attempts x the
#: 120s per-call Typst timeout. Each attempt can take up to 120s, so on a
#: wedged compiler the attempt count alone is no bound — this caps one
#: document's healing. Kept in lockstep with ``max_attempts``' default of 3.
_TYPST_HEAL_BUDGET_S = 360.0

# ---------------------------------------------------------------------------
# Emitter patterns + style profiles
# ---------------------------------------------------------------------------

#: Heading blocks that are really figure/table captions misclassified by the
#: parser: demoted to caption styling instead of a heading. Shared by both layouts.
_MISCLASSIFIED_CAPTION_RE = re.compile(
    r"^(?:fig(?:ure)?|table|tab|图|表)\.?\s*[A-Za-z0-9]", re.IGNORECASE
)

#: Bullet-ish markers that make a narrative block list content (shared by both
#: layouts — was spelled out inline in each emitter).
_LIST_MARKERS = ("•", "-", "*", "·", "–")


@dataclass(frozen=True)
class StyleProfile:
    """Style constants for Typst block emission — the single definition site.

    Two instances exist: ``_FLOWING_STYLE`` for the flowing/``==``-heading
    main path (all defaults) and ``_PAGE_STYLE`` for page-strict interior
    pages (overridden fields only). ``TypstReconstructor._emit_block`` is the
    ONE emission core; the two layouts pass their profile, so every emitter
    style value is declared exactly once, here.

    Style unification record (two historical divergences are pinned by
    ``tests/unit/test_cover_formula_prompt_gates.py``): the two emitters had
    silently diverged. Where the same markup existed in both paths, the
    flowing ``_emit_block`` values are canonical and the interior values they
    replaced are recorded here:

    - bilingual narrative block: interior ``#block(spacing: 1.1em)`` /
      ``rgb("#64748b") 9pt`` / ``rgb("#0f172a") 10pt`` → flowing ``1.2em`` /
      ``luma(90) 9.5pt`` / ``black 10.5pt`` (the defaults below).
    - bilingual heading source line: interior ``rgb("#64748b") 10pt`` (9pt for
      section subheads) → flowing ``luma(110) 10pt``.

    Kept as *page* differences because they are deliberate page-strict design,
    not drift: the publication chapter opener (eyebrow + 17pt title + rule,
    12.5pt section subheads instead of Typst ``== `` markup), the ``#v(0.5em)``
    rhythm around figures, the ``;``/``；`` list heuristic, contiguous list
    items (no blank line between them), and interior's guard that a
    verbatim-skipped (``skip_translate``) footnote-flow block stays prose
    instead of becoming a gray ``#footnote`` callout.
    """

    # -- bilingual narrative block (drift resolved toward the flowing path) --
    block_spacing: str = "1.2em"
    source_fill: str = "luma(90)"
    source_size: str = "9.5pt"
    target_fill: str = "black"
    target_size: str = "10.5pt"

    # -- bilingual heading source line (drift resolved toward the flowing path)
    heading_source_fill: str = "luma(110)"
    heading_source_size: str = "10pt"

    # -- shared: both layouts always agreed on these --
    caption_size: str = "8.5pt"
    caption_fill: str = 'rgb("#475569")'
    list_source_fill: str = 'rgb("#64748b")'
    list_source_size: str = "9pt"
    list_target_fill: str = 'rgb("#0f172a")'
    list_target_size: str = "10pt"

    # -- page-strict design (only overridden by _PAGE_STYLE) --
    custom_headings: bool = False
    chapter_title_size: str = "17pt"
    chapter_title_fill: str = 'rgb("#0f172a")'
    section_title_size: str = "12.5pt"
    section_title_fill: str = 'rgb("#1e293b")'
    eyebrow_fill: str = 'rgb("#94a3b8")'
    eyebrow_size: str = "8pt"
    eyebrow_label_fill: str = 'rgb("#64748b")'
    eyebrow_label_size: str = "8.5pt"
    chapter_rule_stroke: str = '0.6pt + rgb("#cbd5e1")'
    image_spacer: str = ""
    semicolon_list_heuristics: bool = False
    list_item_blank: bool = True
    anchor_verbatim_footnotes: bool = True


#: Flowing / ``_emit_block`` main path: headings delegate to Typst ``== ``.
_FLOWING_STYLE = StyleProfile()

#: Page-strict interior pages: publication chapter openers, figure rhythm,
#: semicolon list heuristic, contiguous list items, verbatim footnotes stay
#: prose (see the StyleProfile drift record).
_PAGE_STYLE = StyleProfile(
    custom_headings=True,
    image_spacer="#v(0.5em)",
    semicolon_list_heuristics=True,
    list_item_blank=False,
    anchor_verbatim_footnotes=False,
)
