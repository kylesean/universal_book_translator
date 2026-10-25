"""Route each character range to the face designed for it (Typst ``covers``).

**The defect this fixes.** A plain priority list — ``("Liberation Serif",
"Noto Serif CJK SC")`` — resolves *per character*, first family that has the
glyph wins. That is right for ideographs and wrong for punctuation: in a
Chinese book the curly quotes, em dash and ellipsis are shared by both fonts,
so the Latin face claims them and the page carries Latin-width quotation marks
in Chinese prose. The visual result is subtly but pervasively wrong, and no
warning ever fires because every glyph *was* found.

Typst 0.13+ answers this directly: a list entry may carry ``covers``, which
scopes that family to a character range. The standard pattern for mixed
Chinese/Latin text is::

    #set text(font: (
      (name: "Liberation Serif", covers: "latin-in-cjk"),
      "Noto Serif CJK SC",
    ))

``latin-in-cjk`` means "this Latin face handles Latin *inside* CJK text, and
shared CJK/Latin punctuation belongs to the CJK font" — which is the
typographic convention, and the reason this module exists rather than a bare
tuple.

**Why a typed model rather than strings.** ``covers`` is structural: a regex,
a script keyword, or nothing. Building it by string concatenation is how the
quotes end up in the wrong place and the document fails to compile. The
renderer asks :func:`route_stack` for ready-to-emit entries and only formats
them.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Target scripts whose text wants the ``latin-in-cjk`` split. A Latin target
#: has no mixed-script problem: one face serves the page.
_CJK_SCRIPT_LANGS: frozenset[str] = frozenset(
    {"zh", "zh-cn", "zh-hans", "zh-tw", "zh-hant", "zh-hk", "ja", "ko"}
)
#: RTL targets: the Latin face still handles embedded Latin/digits, and the
#: script face owns the punctuation. Typst's ``covers`` accepts a regex, which
#: is how Arabic/Hebrew ranges are expressed today.
_RTL_SCRIPT_LANGS: frozenset[str] = frozenset(
    {"ar", "he", "fa", "ur", "ps", "sd", "ug", "yi", "dv"}
)

#: Arabic block + presentation forms + Hebrew block, as one regex. Digits and
#: Latin stay with the Latin face, which is the correct convention for
#: technical documents in RTL languages.
_RTL_COVERS = (
    r'regex("[\u0590-\u05FF\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]")'
)


@dataclass(frozen=True, slots=True)
class FontFace:
    """One entry of a Typst font list, with its character scope."""

    family: str
    #: ``None`` = plain entry (no ``covers``), the historical behaviour.
    #: ``"latin-in-cjk"`` = Latin face scoped inside CJK text.
    covers: str | None = None

    def to_typst(self) -> str:
        """The Typst source for this entry, ready to join into ``font: (…)``.

        A bare ``latin-in-cjk`` is a *variable* to Typst (``error: unknown
        variable``), so the keyword form is always quoted; a ``regex(...)``
        expression is emitted verbatim because it is a function call, not a
        string.
        """
        if self.covers is None:
            return f'"{self.family}"'
        value = self.covers if self.covers.startswith("regex(") else f'"{self.covers}"'
        return f'(name: "{self.family}", covers: {value})'


def _base_lang(target_lang: str | None) -> str:
    return (target_lang or "zh").strip().lower().replace("_", "-")


def script_class(target_lang: str | None) -> str:
    """``"cjk"``, ``"rtl"`` or ``"latin"`` — which routing the target needs."""
    tag = _base_lang(target_lang)
    base = tag.split("-")[0]
    if tag in _CJK_SCRIPT_LANGS or base in {"zh", "ja", "ko"}:
        return "cjk"
    if tag in _RTL_SCRIPT_LANGS or base in _RTL_SCRIPT_LANGS:
        return "rtl"
    return "latin"


def _is_cjk_family(family: str) -> bool:
    """Whether a family name denotes a CJK face (same markers as the probe)."""
    from ubt.adapters.pdf.font_probe import is_cjk_capable

    return is_cjk_capable(family)


def _is_rtl_family(family: str) -> bool:
    from ubt.adapters.pdf.font_probe import is_rtl_capable

    return is_rtl_capable(family)


def route_stack(families: tuple[str, ...], target_lang: str | None) -> tuple[FontFace, ...]:
    """Turn a resolved family stack into ``covers``-scoped entries.

    The routing rule, per class:

    * **cjk** — Latin-serif faces (the ones that can steal punctuation) get
      ``covers: "latin-in-cjk"``; CJK faces stay plain so they own everything
      else including shared punctuation. Order is preserved: the stack's
      preference still decides which *CJK* face wins.
    * **rtl** — the script face (Arabic/Hebrew) is scoped to its block with a
      regex; Latin faces stay plain so embedded Latin stays Latin.
    * **latin** — unchanged. One face serves the page and a ``covers`` entry
      would only add a condition that is always true.

    Mixed stacks keep their order because a reader of the emitted Typst
    should see the same preference the probe resolved.
    """
    cls = script_class(target_lang)
    if cls == "latin":
        return tuple(FontFace(family=f) for f in families)

    if cls == "cjk":
        routed: list[FontFace] = []
        for family in families:
            if _is_cjk_family(family):
                routed.append(FontFace(family=family))
            else:
                # Every non-CJK face in a CJK document is a Latin fallback:
                # scope it so it cannot claim CJK punctuation.
                routed.append(FontFace(family=family, covers="latin-in-cjk"))
        return tuple(routed)

    # rtl
    routed = []
    for family in families:
        if _is_rtl_family(family):
            routed.append(FontFace(family=family, covers=_RTL_COVERS))
        else:
            routed.append(FontFace(family=family))
    return tuple(routed)


def typst_font_list(families: tuple[str, ...], target_lang: str | None) -> str:
    """The ``font: (…)`` argument value for a preamble, routing included.

    Returns the *inside* of the tuple, e.g.
    ``'"Liberation Serif", "Noto Serif CJK SC"'`` or the routed form with
    ``covers``. The caller wraps it in ``font: (...)``.
    """
    return ", ".join(face.to_typst() for face in route_stack(families, target_lang))


__all__ = [
    "FontFace",
    "route_stack",
    "script_class",
    "typst_font_list",
]
