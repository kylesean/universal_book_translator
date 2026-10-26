# Universal Book Translator (UBT) Console Design System & Taste Guidelines

> **Mandate**: "UBT is a precision book-translation and document-reflow studio, not a consumer reader or a generic AI dashboard."
> **Inspiration**: shadcn/ui admin patterns, Linear, Geist (Vercel), Typst App, Raycast, Overleaf.

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

The console has exactly two themes: **light** and **dark**. Light is the
editorial paper workspace; dark is its deliberate low-light counterpart, not a
separate design language. Every colour is a CSS custom property in
`src/index.css`, so switching themes never requires component-level colour
branches. The shadcn semantic tokens (`--background`, `--card`, `--muted`,
`--sidebar-*`, and so on) are the implementation layer; the `--paper-*` and
`--ink-*` aliases remain for editorial proofreading typography and status
surfaces.

### Theme Selection

The operator chooses Light or Dark from the sidebar footer preferences menu or
the command palette. The control names both options explicitly; Sun/Moon icons
are optional affordances, never a replacement for the labels or a
three-state/cyclic control. The choice persists in `localStorage`
(`ubt_theme`). On first visit, the console follows
`prefers-color-scheme`.

### Surface Hierarchy

| Token | Light (`:root`) | Dark (`.dark`) | Used for |
| --- | --- | --- | --- |
| `--paper-bg` | `#faf8f5` | `#09090b` | Base canvas |
| `--paper-surface` | `#ffffff` | `#121215` | Panels, cards, log output |
| `--paper-subsurface` | `#f4f1ea` | `#18181b` | Inset blocks, hover rows |
| `--paper-border` | `#e6e2d8` | `#27272a` | Hairline boundaries |
| `--paper-border-hover` | `#d5cfc2` | `#3f3f46` | Focus rings, selected states |

### Text Tokens

| Token | Light | Dark | Notes |
| --- | --- | --- | --- |
| `--ink-primary` | `#18181b` | `#f4f4f5` | Carbon ink — headings, body |
| `--ink-secondary` | `#52525b` | `#a1a1aa` | Descriptions, metadata |
| `--ink-muted` | `#68635b` | `#71717a` | Timestamps, line numbers, disabled. Re-check contrast whenever either value changes |

### Semantic Status

| Token | Light | Dark | Meaning |
| --- | --- | --- | --- |
| `--ink-highlight` | `#16a34a` | `#22c55e` | Verified gate / passed |
| `--ink-rose` | `#dc2626` | `#ef4444` | Blocked / error |
| `--ink-amber` | `#d97706` | `#f59e0b` | Caution / advisory |
| `--ink-info` | `#0284c7` | `#38bdf8` | Informational state |

### Buttons

`--btn-bg` / `--btn-fg` / `--btn-hover` carry the primary button's ink-on-paper
inversion and swap with the active theme. Prefer the semantic shadcn tokens
where they express the same intent; a hardcoded `bg-[#18181b]` would ignore
the dark theme.

### Typography

- UI Sans: self-hosted **Geist Variable**, then Noto Sans CJK / Source Han Sans.
- Compiler code, IDs, measurements: self-hosted **Geist Mono Variable**, then
  JetBrains Mono / platform monospace.
- Serif (proofreading body and editorial accents only): Noto Serif / Source Han
  Serif / Georgia.
- The metadata floor is **12px** (`text-xs`). Do not use 10px or 11px for
  operator-facing metadata.

### Application shell

The console uses the shadcn **inset sidebar** shell:

- Desktop: a persistent, collapsible left rail with grouped Control and
  Workbenches navigation; the active job scopes Review and Quality Gate.
- Mobile: the same rail in a Radix Sheet drawer, opened from the compact top bar.
- Search: a command palette (`⌘K`, `Ctrl+K`, or `/`) for route jumps and
  preferences.
- Page content: one inset surface with a consistent 12–16px panel radius and a
  single elevation system. Avoid nested cards and decorative gradient chrome.

Reusable primitives live in `src/components/ui`. Components added through the
shadcn CLI keep their registry names and implementation; product-specific
composition belongs in views rather than in custom replacements for registry
primitives.

### Product boundary

UBT is not a consumer e-reader and has no bookshelf, reading progress, or
top-level Reader route. `/wizard` is the only translation intake. Continuous
bilingual proofing is a view inside the Review Workbench, where reading and
correction belong together.

The Quality Gate owns exactly one decision surface: one delivery verdict, five
equally weighted evidence checks, and one primary product. Preview belongs here
because the decision is "can this specific book ship?" and the operator must be
able to inspect that book before approving it. Job details owns the complete
file inventory, including secondary editions and technical reports. It is the
file manager, not a second delivery decision surface.

---

## 3. Colour Discipline (the rule that keeps the theme honest)

- **Never hardcode a hex** where a token exists. A literal `bg-[#141517]` or `text-[#a1a1aa]` is a dark-theme remnant that will be unreadable in the light theme and will not follow the theme switch. Use the semantic shadcn tokens, or `var(--paper-*)` / `var(--ink-*)` where an editorial alias is intentional.
- **`/opacity` modifiers are fine** on a token (e.g. `bg-[#b45309]/10` for an advisory wash) as long as the base colour is a semantic status hue, not a surface.
- **SVG and image canvases are the one exception** — a PDF render is a bitmap, so it keeps a white matte behind it.

---

## 4. Multilingual Support (i18n) Mandate

- **Zero-Friction Detection**: detect the user's OS/browser language through
  the prefix rules in `LANGUAGE_OPTIONS`, rather than a hard-coded language
  pair. Fall back to the canonical default (`en`) when no prefix matches.
- **Persistent Manual Choice**: an unobtrusive language menu in the sidebar
  footer preferences menu and the command palette. It lists the language's own
  name with a radio selection; the choice persists in `localStorage`
  (`ubt_lang`).
- **One Extensible Registry**: `src/i18n/languages.ts` is the single registry
  for language codes, display names, locale tags, browser-match prefixes,
  writing direction, and list separators. Adding a language means adding one
  registry entry, one typed dictionary, and one translator lookup. No component
  may introduce a new hard-coded ZH/EN toggle.
- **100% Typed Translations**: every string lives in `src/i18n/translations/{en,zh}.ts`, typed by `TranslationDictionary` in `src/i18n/types.ts`. All three files move together — a key added to `en` must be added to `zh` and to the type, or `tsc -b` fails. No raw English fallbacks, no literal UI strings in JSX.
- **Copy must describe the engine, not a wish.** If the copy claims a
  capability, the copy is a bug until the engine actually does it.

---

## 5. Bounded DOM (no unbounded lists)

Any list whose length is driven by document size (segments, pages, log lines, visual findings) must be virtualized (`@tanstack/react-virtual`) or capped with an explicit, visible "showing N of M" affordance. Silent `slice(0, N)` truncation is banned: it hides faults past the cut from the operator with no signal.
