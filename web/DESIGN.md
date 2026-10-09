# Universal Book Translator (UBT) Console Design System & Taste Guidelines

> **Mandate**: "UBT is a precision compiler and publication studio, not a consumer reader or a generic AI dashboard."
> **Inspiration**: Linear, Geist (Vercel), Typst App, Raycast, Overleaf.

---

## 1. Anti-"AI Slop" Manifesto (What is Strictly Forbidden)

AI-generated interfaces suffer from predictable statistical defaults that destroy user trust. In this codebase, the following patterns are **permanently banned**:

1. **No Border-itis (边框综合征)**:
   - FORBIDDEN: Wrapping every tiny metric or paragraph in a separate card with a hairline border.
   - MANDATE: Use whitespace, typographic scale, and subtle surface elevation to structure information. Hairline dividers only where a real boundary exists (panel edges, table rows).
2. **No Icon Soup (图标大乱炖)**:
   - FORBIDDEN: Prepending an icon to every title, badge, card, and metric (e.g. a coin next to money, an activity pulse next to status, a layers glyph next to settings).
   - MANDATE: Let words and numbers do the work. Icons are strictly reserved for actionable toolbars (download, cancel, search, sign out) and compact navigation items.
3. **No Micro-Text & Low-Contrast Fog**:
   - FORBIDDEN: Sprinkling `text-[10px]` / `text-[11px]` in washed-out ink on any background.
   - MANDATE: Font floor is 12px for metadata, 14px for body and form controls, 18-24px for major section anchors. Every ink token must clear 4.5:1 WCAG AA against the *darkest* surface it is drawn on — this is why `--ink-muted` is `#68635b` and not a lighter grey.
4. **No Toy Workflows**:
   - FORBIDDEN: A bare text input asking users to manually type `/path/to/book.pdf` as the only path.
   - MANDATE: A drag-and-drop zone with system file dialog integration, visual drop feedback, and automatic document inspection. The manual path field is a secondary affordance for server-local files, never the primary one.
5. **No Gratuitous Neon & Gradients**:
   - FORBIDDEN: Glowing purple gradients, rainbow pill badges, neon borders.
   - MANDATE: A restrained paper palette with semantic accents only (forest green for a passed gate, cinnabar red for blocked, amber wax for caution, deep indigo for running).

---

## 2. Visual Palette & Typography

The console ships a **paper-light** theme, not a dark one. Every colour is a CSS custom property in `src/index.css` so the two paper tones below swap without touching component code.

### Surface Hierarchy

Two tones are available; the operator toggles them from the sidebar header and the choice persists in `localStorage` (`ubt_paper_tone`).

| Token | Cotton (default, `:root`) | Dowling (`[data-paper-tone="dowling"]`) | Used for |
| --- | --- | --- | --- |
| `--paper-bg` | `#faf8f5` | `#f5f2e9` | Base canvas |
| `--paper-surface` | `#ffffff` | `#fcfbf7` | Panels, cards, log output |
| `--paper-subsurface` | `#f4f1ea` | `#ebe6d8` | Inset blocks, hover rows |
| `--paper-border` | `#e6e2d8` | `#ded7c6` | Hairline boundaries |
| `--paper-border-hover` | `#d5cfc2` | `#cfc7b3` | Focus rings, selected states |

### Text Tokens

| Token | Cotton | Dowling | Notes |
| --- | --- | --- | --- |
| `--ink-primary` | `#18181b` | `#1c1917` | Carbon ink — headings, body |
| `--ink-secondary` | `#52525b` | `#57534e` | Descriptions, metadata |
| `--ink-muted` | `#68635b` | `#68635b` | Timestamps, line numbers, disabled. **Do not lighten without re-checking AA** |

### Semantic Status

| Token | Cotton | Dowling | Meaning |
| --- | --- | --- | --- |
| `--ink-highlight` | `#15803d` | `#15803d` | Verified gate / passed |
| `--ink-rose` | `#b91c1c` | `#b91c1c` | Blocked / error |
| `--ink-amber` | `#b45309` | `#c2410c` | Caution / advisory |

### Buttons

`--btn-bg` / `--btn-fg` / `--btn-hover` carry the primary button's ink-on-paper inversion. They are *not* the same as `--ink-primary`: a hardcoded `bg-[#18181b]` would ignore the Dowling tone.

### Typography

- UI Sans: `-apple-system, BlinkMacSystemFont, "Inter", "Geist", "Segoe UI", Roboto, sans-serif`
- Compiler Code/Math: `"JetBrains Mono", "SF Mono", ui-monospace, Menlo, Monaco, Consolas, monospace`
- Serif (editorial accents only): `"Source Han Serif SC", "Noto Serif SC", "Songti SC", Georgia, serif`

---

## 3. Colour Discipline (the rule that keeps the theme honest)

- **Never hardcode a hex** where a token exists. A literal `bg-[#141517]` or `text-[#a1a1aa]` is a dark-theme remnant that will be unreadable on paper and will not follow the tone toggle. Use `var(--paper-*)` / `var(--ink-*)`.
- **`/opacity` modifiers are fine** on a token (e.g. `bg-[#b45309]/10` for an advisory wash) as long as the base colour is a semantic status hue, not a surface.
- **SVG/`<img>` canvases are the one exception** — a PDF render is a bitmap, so it keeps a white matte behind it.

---

## 4. Bilingual Support (i18n) Mandate

- **Zero-Friction Detection**: automatically detect the user's OS/browser language:
  - If `navigator.language` starts with `zh` → Simplified Chinese (`zh`).
  - Otherwise → English (`en`).
- **Persistent Manual Toggle**: an instant, unobtrusive switch in the sidebar header.
- **100% Typed Translations**: every string lives in `src/i18n/translations/{en,zh}.ts`, typed by `TranslationDictionary` in `src/i18n/types.ts`. All three files move together — a key added to `en` must be added to `zh` and to the type, or `tsc -b` fails. No raw English fallbacks, no literal UI strings in JSX.
- **Copy must describe the engine, not a wish.** If the copy claims a capability, the copy is a bug until the engine actually does it.

---

## 5. Bounded DOM (no unbounded lists)

Any list whose length is driven by document size (segments, pages, log lines, visual findings) must be virtualized (`@tanstack/react-virtual`) or capped with an explicit, visible "showing N of M" affordance. Silent `slice(0, N)` truncation is banned: it hides faults past the cut from the operator with no signal.
