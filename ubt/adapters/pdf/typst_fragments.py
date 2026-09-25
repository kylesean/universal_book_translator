"""Typst emit fragments: tables, text, images, references.

Cell/table emission, markup escaping, image staging/sizing, and heading
references for :mod:`ubt.adapters.pdf.typst_reconstructor`, which re-exports
these helpers. Math conversion itself lives in :mod:`ubt.adapters.pdf.typst_math`.
"""

from __future__ import annotations

import hashlib
import logging
import re
import shutil
import tempfile
import unicodedata
from collections.abc import Callable, Sequence
from pathlib import Path

from ubt.adapters.pdf.typst_math import _latex_math_to_typst, _strip_math_delimiters
from ubt.core.cleaners.math_masker import MathMasker
from ubt.core.ir.models import BlockType, IRBlock

logger = logging.getLogger(__name__)

_MARKUP_SPECIALS = ("\\", "#", "$", "*", "_", "`", "<", ">", "@", "[", "]", "~")

# Single source of truth for the Typst keywords/stopword list the healer uses
# to decide whether an "unknown variable" is a real identifier or markup it
# must never quote. Do not duplicate it here or in typst_healer /
# typst_reconstructor: drift between copies of this list is a known failure
# class (the `&\quad` gate bug), so the list lives here alone.
_TYPST_KEYWORDS_AND_STOPWORDS = frozenset(
    {
        "set",
        "show",
        "let",
        "import",
        "include",
        "image",
        "figure",
        "text",
        "table",
        "align",
        "page",
        "par",
        "where",
        "rect",
        "circle",
        "line",
        "move",
        "scale",
        "rotate",
        "block",
        "box",
        "grid",
        "stack",
        "heading",
        "counter",
        "locate",
        "context",
        "rgb",
        "none",
        "auto",
        "true",
        "false",
        "if",
        "else",
        "for",
        "while",
        "break",
        "continue",
        "return",
        "in",
        "not",
        "and",
        "or",
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "to",
        "of",
        "on",
        "with",
        "at",
        "by",
        "from",
        "up",
        "about",
        "into",
        "over",
        "after",
    }
)


def _display_width(text: str) -> int:
    """Terminal-style display width: CJK wide/fullwidth chars count double.

    Column fractions from raw ``len()`` starve CJK-heavy columns (one hanzi
    renders ~2x a latin letter), so width-proportional layout must use
    display width instead.
    """
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text)


_CELL_MATH_MASKER = MathMasker()


_PHYSICS_PREFIXES = r"(?:[NVTCtEI][a-zA-Z]?|psi|phi|theta|lambda|epsilon|eta|mu|sigma|omega|alpha|beta|gamma|[ψφεηθλμσωαβγ])"
_PHYSICS_SUB_RE = re.compile(rf"\b({_PHYSICS_PREFIXES})_([a-zA-Z0-9]+)\b")
# Units accepted by the parameter-assignment polish (rule 4). Exact case:
# "mM" is millimolar, "MM" is not a unit.
_POLISH_SI_UNITS = frozenset(
    {
        "nm",
        "um",
        "mm",
        "cm",
        "dm",
        "m",
        "km",
        "ps",
        "ns",
        "us",
        "ms",
        "s",
        "min",
        "hr",
        "eV",
        "meV",
        "keV",
        "MeV",
        "GeV",
        "J",
        "kJ",
        "V",
        "mV",
        "kV",
        "μV",
        "nV",
        "A",
        "mA",
        "uA",
        "nA",
        "K",
        "mK",
        "nK",
        "Hz",
        "kHz",
        "MHz",
        "GHz",
        "THz",
        "W",
        "mW",
        "kW",
        "F",
        "pF",
        "nF",
        "fF",
        "C",
        "S",
        "N",
        "g",
        "kg",
        "mg",
        "M",
        "mM",
        "uM",
        "nM",
        "pM",
        "mol",
        "mmol",
        "μm",
        "Ω",
    }
)


def _polish_inline_quantities_and_math(text: str) -> str:
    """Detect and format inline scientific quantities, units, and subscripts into Typst math ($...$).

    Handles:
    - Logarithmic ratio corruption: ln(A = B) -> ln(A / B)
    - Scientific notation and compound units: 1 × 10^15 cm^-3 -> $1 times 10^(15) "cm"^(-3)$
    - Parameter assignments with units: TFIN = 20 nm, t_ox = 1 nm, V_ch = 0 V
    - Inline relations: ψ_B = V_tm ln(N_ch / n_i)
    - Subscript variables: N_ch -> $N_"ch"$, V_tm -> $V_"tm"$
    """
    if not text:
        return text

    # 1. Fix ln(A = B) -> ln(A / B)
    text = re.sub(r"\bln\s*\(\s*([A-Za-z0-9_]+)\s*=\s*([A-Za-z0-9_]+)\s*\)", r"ln(\1 / \2)", text)

    # 2. Scientific notation with optional units (e.g. 1 × 10^15 cm^-3 or 10^15 cm^-3)
    def _sci_repl(m: re.Match[str]) -> str:
        pre = m.group(1) or ""
        exp = m.group(2)
        unit = m.group(3) or ""
        unit_str = ""
        if unit:
            unit_fmt = re.sub(r"([a-zA-Z]+)\^([+-]?\d+)", r'"\1"^(\2)', unit.strip())
            if not unit_fmt.startswith('"'):
                unit_fmt = f'"{unit_fmt}"'
            unit_str = f" {unit_fmt}"
        if pre:
            val = re.sub(r"\s*[×x*]\s*$", "", pre).strip()
            return f"${val} times 10^({exp}){unit_str}$"
        return f"$10^({exp}){unit_str}$"

    text = re.sub(
        r"(\d+(?:\.\d+)?\s*[×x*]\s*)?10\^([+-]?\d+)(?:\s*([a-zA-Z]+(?:\^[-+]?\d+)?))?",
        _sci_repl,
        text,
    )

    # 3. Complex formulas like ψ_B = V_tm ln(N_ch / n_i)
    def _formula_repl(m: re.Match[str]) -> str:
        expr = m.group(0)
        expr = expr.replace("ψ", "psi").replace("ε", "epsilon").replace("η", "eta")
        expr = re.sub(r"([A-Za-z]+)_([A-Za-z0-9]+)", r'\1_"\2"', expr)
        return f"${expr}$"

    text = re.sub(
        r"(?:[ψ\w]+_[A-Za-z0-9]+)\s*=\s*(?:[Vv]_?[A-Za-z0-9]+)?\s*ln\([^)]+\)",
        _formula_repl,
        text,
    )

    # 4. Physics parameter assignments with SI units: e.g. TFIN = 20 nm,
    # t_ox = 1 nm, V_ch = 0 V. Deliberately narrow: the
    # variable must start with a physics-quantity prefix and the unit must be
    # a known SI token, or general prose like "Price = 30 dollars" would be
    # rewritten into math mode in every book, not just semiconductor papers.
    def _param_repl(m: re.Match[str]) -> str:
        var = m.group(1)
        val = m.group(2)
        unit = m.group(3)
        if unit not in _POLISH_SI_UNITS:
            return m.group(0)
        if "_" in var:
            base, sub = var.split("_", 1)
            var_fmt = f'{base}_"{sub}"'
        elif var == "TFIN":
            var_fmt = 'T_"FIN"'
        else:
            var_fmt = var
        return f'${var_fmt} = {val} "{unit}"$'

    text = re.sub(
        rf"(?<!\$)\b({_PHYSICS_PREFIXES}(?:_[A-Za-z0-9]+)?|TFIN)\s*="
        r"\s*(\d+(?:\.\d+)?)\s*([a-zA-Z]+)\b(?!\$)",
        _param_repl,
        text,
    )

    # 5. Standalone subscript variables for physics quantities (N_ch, ψ_B, V_tm, etc.)
    def _sub_repl(m: re.Match[str]) -> str:
        base = m.group(1)
        sub = m.group(2)
        base = base.replace("ψ", "psi").replace("ε", "epsilon")
        return f'${base}_"{sub}"$'

    text = _PHYSICS_SUB_RE.sub(_sub_repl, text)
    return text


_SPACED_LATEX_DOLLAR_RE = re.compile(
    r"\$\s+([^$\n]*?(?:\\[A-Za-z]+|[_^{}]|[→←↔↦∘×÷±≤≥≠≈∈∉⊂⊆∪∩∀∃∂∇ΓΔΘΛΞΠΣΦΨΩα-ω])[^$\n]*?)\s*\$"
    r"|\$\s*([^$\n]*?(?:\\[A-Za-z]+|[_^{}]|[→←↔↦∘×÷±≤≥≠≈∈∉⊂⊆∪∩∀∃∂∇ΓΔΘΛΞΠΣΦΨΩα-ω])[^$\n]*?)\s+\$"
)
_BARE_DIGIT_MATH_RE = re.compile(r"(?<![\d$])\$(\d{1,2})\$(?!\s*(?:[-–—]|to|and)\s*\$\d)")
_BODY_SECTION_HEADING_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*(?:\s|[.、])|Abstract\b|摘要|Introduction\b|引言|[Cc]hapter\s+\d+|第\s*[0-9一二三四五六七八九十百]+\s*[章节篇])"
)


_SOFT_HYPHEN_SPLIT_RE = re.compile(r"([A-Za-z])\xad[ \t]*([a-z])")


def _normalize_inline_math_delimiters(text: str) -> str:
    """Tighten spaced '$ \\Gamma $' delimiters, unwrap bare '$0$' math tokens, and heal soft-hyphens."""
    if "\xad" in text:
        text = _SOFT_HYPHEN_SPLIT_RE.sub(r"\1\2", text).replace("\xad", "")
    if "$" not in text:
        return text
    out = _SPACED_LATEX_DOLLAR_RE.sub(lambda m: f"${(m.group(1) or m.group(2) or '').strip()}$", text)
    out = _BARE_DIGIT_MATH_RE.sub(r"\1", out)
    return out


def _prose_to_typst(
    text: str,
    *,
    inline_math: Callable[[str], str | None] | None = None,
) -> str:
    """Render narrative prose or cell content: prose escaped, math spans as Typst math.

    Preserves inline LaTeX math ($...$, \\(...\\)) by converting it to Typst math
    mode ($...$) instead of escaping dollar signs into literal plain text. All
    non-math prose has its Typst special characters safely escaped.

    ``inline_math`` optionally takes over *source* math spans (the ones the
    document actually carried): it receives the bare LaTeX and returns a Typst
    replacement, or None to keep the legacy conversion for that span. Spans
    synthesized by :func:`_polish_inline_quantities_and_math` are deterministic
    and never routed to the engine.
    """
    text = _normalize_inline_math_delimiters(text)
    # 1. Mask existing math first so it is never corrupted by prose polishing
    masked, math_map = _CELL_MATH_MASKER.mask(text)
    source_tokens = frozenset(math_map)
    # 2. Polish prose spans (quantities, units, physics subscripts)
    masked = _polish_inline_quantities_and_math(masked)
    # 3. Mask any new math spans created by polish
    masked, new_math_map = _CELL_MATH_MASKER.mask(masked)
    math_map.update(new_math_map)

    if not math_map:
        return _escape_typst_markup(masked)
    sentinels: dict[str, tuple[str, bool]] = {}
    for i, (token, original) in enumerate(math_map.items()):
        sentinel = f"\x00MATH{i}\x00"
        sentinels[sentinel] = (original, token in source_tokens)
        masked = masked.replace(token, sentinel)
    esc = _escape_typst_markup(masked)
    for sentinel, (original, from_source) in sentinels.items():
        inner = _strip_math_delimiters(original)
        replacement = inline_math(inner) if inline_math is not None and from_source else None
        if replacement is None:
            replacement = "$" + _latex_math_to_typst(inner) + "$"
        if replacement.startswith("#"):
            # When replacement is a Typst hash call (e.g. #box(...)[...]),
            # if immediately followed by '(', Typst parses '(' as a chained function call argument,
            # switching into code mode and failing on '#' or expected function.
            # Escaping '(' -> '\(' decouples the syntax into literal text while preserving identical PDF layout.
            esc = esc.replace(f"{sentinel}(", f"{replacement}\\(")
        esc = esc.replace(sentinel, replacement)
    return esc


_cell_to_typst = _prose_to_typst


def _escape_typst_markup(text: str) -> str:
    """Escape Typst markup special characters in narrative/table content."""
    # Strip HTML <mark> tags to prevent raw HTML leaking into Typst documents
    if "<mark" in text or "</mark>" in text:
        text = re.sub(r"</?mark\b[^>]*>", "", text)
    result = text
    for ch in _MARKUP_SPECIALS:
        result = result.replace(ch, "\\" + ch)
    # In Typst content blocks, a leading "/" is parsed as a line-break marker.
    # Escape it only at the start so URLs mid-text stay intact.
    if result.startswith("/"):
        result = "\\/" + result[1:]
    # A mid-text "//" opens a Typst line comment and silently drops the rest of
    # the paragraph -- compile still exits 0 with no diagnostic, so this is
    # content loss, not a formatting glitch. The zero-width space breaks the
    # comment token while rendering invisibly (same guard as
    # overlay_text.typst_escape).
    return result.replace("//", "/\u200b/")


_LANG_TAG_RE = re.compile(r"[A-Za-z]{2,3}(?:[_-][A-Za-z0-9]{2,4})?")


def sanitize_lang_tag(value: str) -> str:
    """Return a Typst-safe ``lang:`` tag, or "" when the value is not a language tag.

    The tag is interpolated into ``#set text(...)`` inside a string literal, and
    quoting does not contain a quote or a ``)`` there: a caller-supplied
    ``target_lang`` of ``zh") #import "x`` would compile as code. Anything that
    is not an ISO-ish tag is dropped rather than escaped, because Typst rejects
    unknown languages anyway and an omitted tag just uses its default.
    """
    clean = (value or "").strip()
    return clean if _LANG_TAG_RE.fullmatch(clean) else ""


# Font families reach Typst inside ``#set text(font: "...")``. Spaces and dots
# are part of real family names, the rest of Typst's markup vocabulary is not.
_FONT_FAMILY_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9 ._-]*")


def sanitize_font_family(value: str | None) -> str | None:
    """Return a font family usable in ``#set text(font: ...)`` or ``None``."""
    clean = (value or "").strip()
    return clean if _FONT_FAMILY_RE.fullmatch(clean) else None


def _sanitize_image_ref(path: str) -> str:
    """Make a filesystem path safe to embed in a Typst string literal.

    An embedded quote would terminate the string literal early and let
    adversarial content escape into code position; backslashes form escape
    sequences; newlines would break the source line structure. A ``..`` path
    component is rejected outright: it would let untrusted block text point
    ``#image`` at a file outside the compile/output tree. Absolute paths stay
    allowed because pipeline assets legitimately live in the system temp dir.
    """
    clean = re.sub(r'[\\"\r\n]', "", path)
    if any(part == ".." for part in clean.split("/")):
        return ""
    return clean


def _within(path: Path, base: Path) -> bool:
    """True when ``path`` is inside ``base`` (both resolved by the caller)."""
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


_IMAGE_REF_PATTERN = re.compile(r'#image\("([^"\n]+)"')
# Headings that open a bibliography section: list items inside are numbered
# [n] in reading order so in-text citations ([1], [2]) stay resolvable.
# Without this, docling's list markers degrade to uniform bullets and the
# numbers readers need for lookup are lost.
_REFERENCES_HEADING_RE = re.compile(r"references|bibliography|参考文献|引用文献", re.IGNORECASE)


def _reference_numbers(blocks: Sequence[IRBlock]) -> dict[str, str]:
    """Map block id -> [n] label for list items inside bibliography sections.

    Once a terminal references/bibliography heading opens, numbering stays on
    for its sub-headings (e.g. "Primary papers and current implementation
    sources"). However:
    1. Table-of-contents entries (``provenance["toc_entry"]``) must NEVER open
       reference numbering;
    2. If a numbered main-body chapter/section heading (e.g. ``1. Introduction``)
       appears after an earlier false trigger, ``in_refs`` resets and any
       premature mapping is discarded.
    """
    mapping: dict[str, str] = {}
    in_refs = False
    counter = 0
    for block in blocks:
        if (getattr(block, "provenance", None) or {}).get("toc_entry"):
            continue
        if block.block_type == BlockType.HEADING:
            src_txt = (block.source_text or "").strip()
            tgt_txt = (block.target_text or "").strip()
            text = f"{src_txt} {tgt_txt}"
            if _REFERENCES_HEADING_RE.search(text):
                in_refs = True
                counter = 0
            elif in_refs and (
                _BODY_SECTION_HEADING_RE.match(src_txt)
                or _BODY_SECTION_HEADING_RE.match(tgt_txt)
            ):
                in_refs = False
                counter = 0
                mapping.clear()
            continue
        if in_refs and block.block_type == BlockType.LIST_ITEM:
            counter += 1
            mapping[block.id] = f"[{counter}]"
    return mapping


def _footer_display_title(blocks: Sequence[IRBlock], title: str, bilingual: bool) -> str:
    """Running-footer title: translated cover title when available.

    A monolingual target-language book with a source-language running head
    on every page reads as untranslated chrome. The cover
    title is almost always the first HEADING block; prefer its target text
    and fall back to the source title. Bilingual outputs keep both.
    """
    source = " ".join(str(title).split())
    translated = ""
    for block in blocks:
        target = (block.target_text or "").strip()
        if block.block_type == BlockType.HEADING and target:
            translated = " ".join(target.split())
            break
    if bilingual and translated and translated != source:
        return _escape_typst_markup(f"{translated} ({source})")
    return _escape_typst_markup(translated or source)


# Vector diagrams render at native size: the source boxes are already laid
# out in PDF points, and reflow text width (~470pt on A4) matches the source.
# Anything wider is capped to avoid page overflow.

_SVG_NATIVE_MAX_WIDTH_PT = 440.0


def _svg_native_size(asset_path: str) -> tuple[float, float] | None:
    """Read an SVG's native size in points from its viewBox (stdlib only).

    Returns ``(width, height)`` or None when the file is not a parseable
    SVG. ``PIL`` cannot open SVGs, so without this every vector diagram
    falls through to the raster default (width:70%) and small chips blow up
    ~5x (the "giant Decode boxes" failure mode).
    """
    try:
        import xml.etree.ElementTree as ET

        root = ET.parse(asset_path).getroot()
        parts = [float(v) for v in root.get("viewBox", "").split()]
        if len(parts) != 4 or parts[2] <= 0 or parts[3] <= 0:
            return None
        return (parts[2], parts[3])
    except Exception:
        return None


def _image_size_spec(asset_path: str) -> str:
    """Typst size spec for an image asset, preserving native proportions.

    Vector SVGs render at native point size (capped to the text width):
    they are resolution-independent, so matching the source box size is
    always correct. Raster images keep the legacy aspect heuristics.
    """
    if asset_path.lower().endswith(".svg"):
        native = _svg_native_size(asset_path)
        if native is not None:
            return f"width: {min(native[0], _SVG_NATIVE_MAX_WIDTH_PT):g}pt"
        return "width: 70%"
    if Path(asset_path).exists():
        try:
            from PIL import Image

            with Image.open(asset_path) as im:
                aspect = im.width / max(im.height, 1)
                if aspect > 4.5:
                    return "width: 85%"
                if aspect < 1.2:
                    return "height: 4.5cm"
                return "width: 65%"
        except Exception as exc:
            # Falls back to the default width; report why so a systematically
            # unreadable asset isn't silently mis-sized for the whole book.
            logger.warning("typst_fragments: image size probe failed for %s (%s)", asset_path, exc)
    return "width: 70%"


def _stage_image_assets(typ_source: str, typ_file: Path) -> str:
    """Copy referenced image assets next to the .typ file and rewrite refs.

    Narrowing the Typst compile root to the output directory makes
    assets stored elsewhere (e.g. ``/tmp/ubt_assets``) unreachable, so they are
    staged into ``<output>/_assets/`` with collision-proof hashed names and the
    refs are rewritten to root-relative paths. Returns the original source
    unchanged on any failure — compilation then reports missing images
    itself instead of hiding the cause.
    """
    try:
        root = typ_file.parent
        refs = set(_IMAGE_REF_PATTERN.findall(typ_source))
        if not refs:
            return typ_source
        staged_dir = root / "_assets"
        staged_dir.mkdir(parents=True, exist_ok=True)
        # Only stage assets that resolve inside the compile root, the system
        # temp dir (where the pipeline deposits extracted assets), the
        # math SVG cache (the MathJax backend writes #image refs straight
        # there; excluding it made every engine formula fail to compile), or
        # the Docling conversion cache (where extracted document pictures live).
        # Without this allowlist a crafted ``#image("../../x")`` copies any
        # readable file into the deliverable.
        from ubt.adapters.pdf import docling_parser, math_renderer

        # ``_within`` requires both paths canonical, and ``src`` below is
        # resolved: a relative ``UBT_CACHE_DIR`` or a cache root behind a
        # symlink ($HOME on NFS, /tmp -> /private/tmp) would otherwise make
        # ``relative_to`` fail and every engine formula's #image look "outside
        # allowed roots", so staging was skipped and the compile failed.
        allowed_roots = (
            root.resolve(),
            Path(tempfile.gettempdir()).resolve(),
            Path(math_renderer.MATH_CACHE_DIR).resolve(),
            Path(docling_parser.DOCLING_CACHE_DIR).resolve(),
        )
        rewritten = typ_source
        for ref in sorted(refs):
            if ref.startswith("/_assets/"):
                continue
            src = Path(ref)
            if src.is_absolute():
                src = src.resolve()
            elif (root / src).is_file():
                src = (root / src).resolve()
            elif src.is_file():
                src = src.resolve()
            else:
                src = (root / src).resolve()
            if not any(_within(src, base) for base in allowed_roots):
                logger.warning("Image asset outside allowed roots, skipped staging: %s", ref)
                continue
            if not src.is_file():
                logger.warning("Image asset not found, skipped staging: %s", ref)
                continue
            digest = hashlib.md5(ref.encode("utf-8")).hexdigest()[:8]
            staged = staged_dir / f"{src.stem}_{digest}{src.suffix}"
            if src != staged:
                shutil.copyfile(src, staged)
            rel = staged.relative_to(root).as_posix()
            rewritten = rewritten.replace(f'#image("{ref}"', f'#image("/{rel}"')
        return rewritten
    except Exception as exc:
        logger.warning("Image asset staging failed, keeping original refs: %s", exc)
        return typ_source


_INLINE_CALL_COLLISION_RE = re.compile(r"(#box(?:\([^)\n]*\))?\[[^\]\n]*\]|#image\([^)\n]*\))\(")


def _decouple_inline_box_calls(typ_source: str) -> str:
    """Decouple #box[...] or #image(...) calls immediately followed by '(' in Typst markup.

    In Typst markup mode, `#box[...](...)` or `#image(...)(...)` is parsed as
    a function call passing arguments to content, which switches into code mode
    and errors with 'expected function, found content' or 'the character # is not valid in code'.
    Escaping the following '(' as '\\(' decouples the syntax into literal text.
    """
    return _INLINE_CALL_COLLISION_RE.sub(r"\1\\(", typ_source)


_PIPE_ESCAPED = "\ue012"
_PIPE_MATH = "\ue013"
_PIPE_CODE = "\ue014"


def _split_markdown_table_row(line: str) -> list[str]:
    # Protect escaped \|
    s = line.replace(r"\|", _PIPE_ESCAPED)
    # Protect pipes inside inline code `...`
    s = re.sub(r"`[^`\n]+`", lambda m: m.group(0).replace("|", _PIPE_CODE), s)
    # Protect pipes inside math $...$
    s = re.sub(r"\$[^\$\n]+\$", lambda m: m.group(0).replace("|", _PIPE_MATH), s)
    s = s.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    raw_cells = s.split("|")
    cells = []
    for c in raw_cells:
        c = c.replace(_PIPE_MATH, "|").replace(_PIPE_CODE, "|").replace(_PIPE_ESCAPED, r"\|")
        cells.append(c.strip())
    return cells


def _markdown_table_to_typst(md_table: str) -> str:
    """Convert a Markdown pipe table into a publication-grade Typst Booktabs three-line table."""
    raw_lines = [line.strip() for line in md_table.strip().splitlines() if line.strip()]
    if not raw_lines:
        return ""

    alignments: list[str] = []
    pipe_rows: list[list[str]] = []
    for line in raw_lines:
        if "|" in line:
            cells = _split_markdown_table_row(line)
            # ``any(cells)``: with every cell empty the ``if c`` filter leaves an
            # empty generator, so ``all([])`` is True and a blank row was eaten
            # as the alignment separator.
            if cells and any(cells) and all(re.match(r"^:?-+:?$", c) for c in cells if c):
                if not alignments:
                    for c in cells:
                        if c.startswith(":") and c.endswith(":"):
                            alignments.append("center")
                        elif c.endswith(":"):
                            alignments.append("right")
                        else:
                            alignments.append("left")
                continue
            pipe_rows.append(cells)

    if not pipe_rows:
        escaped_text = _escape_typst_markup(md_table)
        return f"#block(stroke: 0.5pt + luma(180), inset: 8pt, radius: 2pt)[\n{escaped_text}\n]"

    num_cols = max(len(row) for row in pipe_rows)
    if num_cols == 0:
        escaped_text = _escape_typst_markup(md_table)
        return f"#block(stroke: 0.5pt + luma(180), inset: 8pt, radius: 2pt)[\n{escaped_text}\n]"

    col_max_lens = [
        max(_display_width(row[i]) if i < len(row) else 0 for row in pipe_rows)
        for i in range(num_cols)
    ]
    total_len = sum(max(w, 6) for w in col_max_lens)
    col_fractions = [f"{(max(col_max_lens[i], 6) / total_len) * 100:.1f}%" for i in range(num_cols)]
    col_specs = ", ".join(col_fractions)

    if alignments and any(a != "left" for a in alignments):
        while len(alignments) < num_cols:
            alignments.append("left")
        align_tuple = ", ".join(alignments[:num_cols])
        align_str = f"align: (col, row) => ({align_tuple}).at(calc.min(col, {num_cols - 1})),"
    else:
        align_str = "align: (col, row) => left,"

    out = [
        "#v(0.6em)",
        f"#table(\n  columns: ({col_specs}),\n  stroke: none,\n  inset: (x: 8pt, y: 7pt),\n  {align_str}",
        '  table.hline(stroke: 1.1pt + rgb("#0f172a")),',
    ]

    header_row = pipe_rows[0]
    out.append("  table.header(")
    for cell in header_row + [""] * (num_cols - len(header_row)):
        esc = _cell_to_typst(cell)
        if esc:
            out.append(f'    [#text(weight: "bold", fill: rgb("#0f172a"))[{esc}]],')
        else:
            out.append("    [],")
    out.append("  ),")
    out.append('  table.hline(stroke: 0.6pt + rgb("#0f172a")),')

    for row in pipe_rows[1:]:
        row_cells = row + [""] * (num_cols - len(row))
        for col_idx, cell in enumerate(row_cells):
            esc = _cell_to_typst(cell)
            if col_idx == 0 and esc:
                out.append(f'  [#text(weight: "bold", fill: rgb("#1e293b"))[{esc}]],')
            else:
                out.append(f"  [{esc}],")

    out.append('  table.hline(stroke: 1.1pt + rgb("#0f172a")),')
    out.append(")")
    out.append("#v(0.6em)")
    return "\n".join(out)
