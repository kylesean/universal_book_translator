# Universal Book Translator (UBT) Console Design System & Taste Guidelines

> **Mandate**: "UBT is a precision compiler and publication studio, not a consumer reader or a generic AI dashboard."
> **Inspiration**: Linear, Geist (Vercel), Typst App, Raycast, Overleaf.

---

## 1. Anti-"AI Slop" Manifesto (What is Strictly Forbidden)

AI-generated interfaces suffer from predictable statistical defaults that destroy user trust. In this codebase, the following patterns are **permanently banned**:

1. **No Border-itis (边框综合征)**:
   - FORBIDDEN: Wrapping every tiny metric or paragraph in a separate card with `#27272a` borders.
   - MANDATE: Use whitespace, typographic scale, and subtle surface elevation (`#0c0c0e` vs `#09090b`) to structure information. Hairline dividers (`border-t border-[#1f1f23]`) only where necessary.
2. **No Icon Soup (图标大乱炖)**:
   - FORBIDDEN: Prepending an icon to every title, badge, card, and metric (e.g. `Coins` next to money, `Activity` next to status, `Layers` next to settings).
   - MANDATE: Let words and numbers do the work. Icons are strictly reserved for actionable toolbars (e.g., download, cancel, search) and compact navigation items.
3. **No Micro-Text & Low-Contrast Fog**:
   - FORBIDDEN: Randomly sprinkling `text-[10px]` and `text-[11px]` in washed-out `#71717a` on black backgrounds.
   - MANDATE: Font floor is 12px for metadata, 14px for body and form controls, 18-24px for major section anchors. Maintain a minimum 4.5:1 WCAG AA contrast ratio.
4. **No Toy Workflows**:
   - FORBIDDEN: A bare text input asking users to manually type `/path/to/book.pdf`.
   - MANDATE: Seamless drag-and-drop zone with system file dialog integration, visual file drop feedback, and automatic document inspection.
5. **No Gratuitous Neon & Gradients**:
   - FORBIDDEN: Glowing purple gradients, rainbow pill badges, neon borders.
   - MANDATE: Monochromatic, high-contrast palette with subtle, restrained semantic accents (emerald for verified gate, rose for blocked, amber for caution).

---

## 2. Visual Palette & Typography

### Surface Hierarchy
- `Base Canvas`: `#09090b` (Deepest dark)
- `Surface Raised`: `#111114` (Panels, sidebars, dropzones)
- `Surface Overlay`: `#18181c` (Hover states, menus, dialogs)
- `Hairline Border`: `#222227` (Subtle 1px boundaries)
- `Active Border`: `#363640` (Focus rings, selected states)

### Text Tokens
- `Text Primary`: `#f4f4f6` (High contrast, crisp)
- `Text Secondary`: `#a1a1aa` (Readable metadata, descriptions)
- `Text Muted`: `#71717a` (Timestamps, keyboard shortcuts, disabled)

### Semantic Status
- `Status Verified`: `#10b981` (Emerald)
- `Status Blocked`: `#f43f5e` (Rose)
- `Status Warning`: `#f59e0b` (Amber)
- `Status Compiling`: `#38bdf8` (Sky)

### Typography
- UI Sans: `-apple-system, BlinkMacSystemFont, "Inter", "Geist", "Segoe UI", Roboto, sans-serif`
- Compiler Code/Math: `"JetBrains Mono", "SF Mono", "Fira Code", Menlo, monospace`

---

## 3. Bilingual Support (i18n) Mandate

- **Zero-Friction Detection**: Must automatically detect the user's OS/browser language:
  - If `navigator.language` starts with `zh` $\rightarrow$ Simplified Chinese (`zh`).
  - Otherwise $\rightarrow$ English (`en`).
- **Persistent Manual Toggle**: Provide an instant, unobtrusive switch in the UI header.
- **100% Typed Translations**: No missing keys, no untranslated fallback English strings in Chinese mode.
