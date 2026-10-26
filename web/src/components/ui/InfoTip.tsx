import React, { useCallback, useEffect, useId, useLayoutEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { Info } from 'lucide-react'
import { clsx } from 'clsx'

interface InfoTipProps {
  /** The explanation. Plain text; kept short enough to read at a glance. */
  children: React.ReactNode
  /** Accessible name for the trigger, e.g. "About columns". */
  label: string
  className?: string
}

/** Panel width in px; also used to clamp the horizontal position. */
const PANEL_WIDTH = 256
/** Gap between the trigger and the panel. */
const GAP = 6
/** Keep this much clear of the viewport edge. */
const MARGIN = 8

/**
 * A small "?" affordance that reveals static help on hover, focus or tap.
 *
 * The wizard used to render one explanatory paragraph under every control —
 * twelve of them, all 12px, most of them static ("leave blank to mirror the
 * source layout"). Read once and never again, they set the page's information
 * density and buried the handful of help lines that *do* change at runtime
 * (an OCR egress warning, a budget below the quote). Moving the static ones
 * behind this trigger halves the visual weight without losing any text.
 *
 * The panel is portalled to `document.body` with `position: fixed`, and its
 * coordinates are computed from the trigger's rect. That is not incidental:
 * an `absolute` panel anchored inside the form was measured covering a
 * neighbouring `<select>` in 4 of 7 cases, because form rows sit ~68px apart
 * while a panel is 54-90px tall. A portal escapes the parent's stacking and
 * overflow context, and the explicit placement picks whichever side has room.
 *
 * Interaction:
 * - hover, click, focus and Enter/Space all open it; Escape, mouseleave, an
 *   outside pointerdown and scroll/resize close it.
 * - `aria-describedby` points at the panel so a screen reader reads the help
 *   as the control's description rather than as an anonymous element.
 */
export function InfoTip({ children, label, className }: InfoTipProps) {
  const [open, setOpen] = useState(false)
  const [pos, setPos] = useState<{ top: number; left: number } | null>(null)
  const panelId = useId()
  const triggerRef = useRef<HTMLButtonElement>(null)
  const panelRef = useRef<HTMLSpanElement>(null)

  const close = useCallback(() => {
    setOpen(false)
    setPos(null)
  }, [])

  /**
   * Place the panel on whichever side of the trigger has room.
   *
   * Above is preferred: the rows below a label are other controls, while the
   * space above it is its own row's whitespace. Only when the trigger is near
   * the top of the viewport does it flip below.
   */
  const place = useCallback(() => {
    const el = triggerRef.current
    if (!el) return
    const rect = el.getBoundingClientRect()
    const panelH = panelRef.current?.offsetHeight ?? 80
    const above = rect.top - GAP - panelH
    const below = rect.bottom + GAP
    const fitsAbove = above >= MARGIN
    const top = fitsAbove ? above : below
    const left = Math.min(
      Math.max(MARGIN, rect.left),
      window.innerWidth - PANEL_WIDTH - MARGIN,
    )
    setPos({ top, left })
  }, [])

  const openPanel = useCallback(() => {
    setOpen(true)
  }, [])

  // Measure after the panel mounts (its height depends on the text), and keep
  // it anchored while the page moves under it.
  useLayoutEffect(() => {
    if (!open) return
    place()
    const onScroll = () => close()
    window.addEventListener('scroll', onScroll, true)
    window.addEventListener('resize', onScroll)
    return () => {
      window.removeEventListener('scroll', onScroll, true)
      window.removeEventListener('resize', onScroll)
    }
  }, [open, place, close])

  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') close()
    }
    const onPointerDown = (e: PointerEvent) => {
      // A tap elsewhere dismisses it; without this the panel stays open on
      // touch devices, where there is no mouseleave.
      const target = e.target as Node
      if (!panelRef.current?.contains(target) && !triggerRef.current?.contains(target)) {
        close()
      }
    }
    document.addEventListener('keydown', onKey)
    document.addEventListener('pointerdown', onPointerDown)
    return () => {
      document.removeEventListener('keydown', onKey)
      document.removeEventListener('pointerdown', onPointerDown)
    }
  }, [open, close])

  return (
    <>
      <button
        ref={triggerRef}
        type="button"
        aria-label={label}
        aria-describedby={open ? panelId : undefined}
        aria-expanded={open}
        // Open, never toggle. A pointer click is always preceded by
        // `mouseenter` on a device that has a mouse, so a toggle here would
        // open the panel on hover and immediately close it on the click that
        // follows — the panel would never stay open under a real cursor.
        onClick={openPanel}
        onFocus={openPanel}
        // Escape closes the panel without moving focus, so `onFocus` cannot
        // fire again for a keyboard user who wants to re-read it: pressing
        // Enter or Space on the still-focused trigger re-opens it.
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault()
            openPanel()
          }
        }}
        onMouseEnter={openPanel}
        onMouseLeave={close}
        className={clsx(
          'inline-flex h-3.5 w-3.5 items-center justify-center rounded-full text-muted-foreground hover:text-foreground focus:text-foreground focus:outline-none focus-visible:ring-1 focus-visible:ring-ring transition-colors cursor-help',
          className,
        )}
      >
        <Info className="h-3 w-3" aria-hidden="true" />
      </button>
      {open &&
        createPortal(
          <span
            ref={panelRef}
            id={panelId}
            role="tooltip"
            style={{
              position: 'fixed',
              top: pos?.top ?? -9999,
              left: pos?.left ?? -9999,
              width: PANEL_WIDTH,
            }}
            // ``pointer-events-none`` is load-bearing. The form's rows sit
            // 35-71px apart while a panel is 54-90px tall, so no placement can
            // avoid overlapping *something*; measured, an upward panel covered
            // a neighbouring <select> for 4 of 7 triggers. Making the panel
            // click-through means the control underneath stays operable and the
            // panel dismisses on the next pointerdown (the outside-click
            // handler ignores it by design), so the overlap costs nothing.
            className="pointer-events-none z-50 rounded-md border border-border bg-card p-2 text-xs font-normal normal-case tracking-normal leading-relaxed text-muted-foreground shadow-[0_2px_10px_rgba(0,0,0,0.10)]"
          >
            {children}
          </span>,
          document.body,
        )}
    </>
  )
}
