import React, { useCallback, useMemo, useRef, useState } from 'react'
import { AlertTriangle, CheckCircle2, Info, X } from 'lucide-react'
import { useI18n } from '@/i18n/useI18n'
import { ToastContext, type Toast, type ToastVariant } from './useToast'

/**
 * Console-wide transient notices.
 *
 * The console previously reported every failure through ``window.alert``: a
 * blocking, unstyled native dialog that cannot be read by a screen reader as
 * part of the page and does not match the console's active theme. Toasts render
 * in the app, carry ``role="status"`` / ``role="alert"`` so assistive tech
 * announces them, and never block the pipeline view.
 */
export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([])
  const nextIdRef = useRef(1)
  const { t } = useI18n()

  const dismiss = useCallback((id: number) => {
    setToasts((prev) => prev.filter((toast) => toast.id !== id))
  }, [])

  const push = useCallback(
    (message: string, variant: ToastVariant = 'info') => {
      // A monotonic id keeps React keys stable even when two notices of the
      // same text land in the same tick (``Date.now()`` alone would collide).
      const id = nextIdRef.current++
      setToasts((prev) => [...prev.slice(-4), { id, message, variant }])
      window.setTimeout(() => dismiss(id), variant === 'error' ? 8000 : 4500)
    },
    [dismiss]
  )

  const value = useMemo(() => ({ push }), [push])

  const icons: Record<ToastVariant, React.ReactNode> = {
    info: <Info className="h-3.5 w-3.5 shrink-0" />,
    success: <CheckCircle2 className="h-3.5 w-3.5 shrink-0" />,
    error: <AlertTriangle className="h-3.5 w-3.5 shrink-0" />,
  }
  const tones: Record<ToastVariant, string> = {
    info: 'text-muted-foreground border-border/80',
    success: 'text-emerald-700 dark:text-emerald-400 border-emerald-600/25',
    error: 'text-destructive border-destructive/25',
  }

  return (
    <ToastContext.Provider value={value}>
      {children}
      <div className="pointer-events-none fixed bottom-4 right-4 z-[60] flex w-[min(22rem,calc(100vw-2rem))] flex-col gap-2">
        {toasts.map((toast) => (
          <div
            key={toast.id}
            role={toast.variant === 'error' ? 'alert' : 'status'}
            className={`pointer-events-auto flex items-start gap-2.5 rounded-xl border bg-card px-3.5 py-3 text-sm shadow-floating ${tones[toast.variant]}`}
          >
            <span className="mt-0.5">{icons[toast.variant]}</span>
            <span className="flex-1 leading-relaxed text-foreground break-words">
              {toast.message}
            </span>
            <button
              type="button"
              onClick={() => dismiss(toast.id)}
              aria-label={t.common.dismiss}
              className="text-muted-foreground hover:text-foreground transition-colors"
            >
              <X className="size-3.5" />
            </button>
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  )
}
