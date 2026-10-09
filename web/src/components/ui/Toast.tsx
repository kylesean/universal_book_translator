import React, { useCallback, useMemo, useRef, useState } from 'react'
import { AlertTriangle, CheckCircle2, Info, X } from 'lucide-react'
import { useI18n } from '@/i18n/useI18n'
import { ToastContext, type Toast, type ToastVariant } from './useToast'

/**
 * Console-wide transient notices.
 *
 * The console previously reported every failure through ``window.alert``: a
 * blocking, unstyled native dialog that cannot be read by a screen reader as
 * part of the page and does not match the console's paper theme. Toasts render
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
    info: 'text-[var(--ink-secondary)] border-[var(--paper-border)]',
    success: 'text-[#15803d] border-[#15803d]/30',
    error: 'text-[#b91c1c] border-[#b91c1c]/30',
  }

  return (
    <ToastContext.Provider value={value}>
      {children}
      <div className="pointer-events-none fixed bottom-4 right-4 z-50 flex w-80 flex-col gap-2">
        {toasts.map((toast) => (
          <div
            key={toast.id}
            role={toast.variant === 'error' ? 'alert' : 'status'}
            className={`pointer-events-auto flex items-start gap-2 rounded-[5px] border bg-[var(--paper-surface)] px-3 py-2.5 text-xs shadow-[0_4px_16px_rgba(0,0,0,0.08)] ${tones[toast.variant]}`}
          >
            <span className="mt-0.5">{icons[toast.variant]}</span>
            <span className="flex-1 leading-relaxed text-[var(--ink-primary)] break-words">
              {toast.message}
            </span>
            <button
              type="button"
              onClick={() => dismiss(toast.id)}
              aria-label={t.common.dismiss}
              className="text-[var(--ink-muted)] hover:text-[var(--ink-primary)] transition-colors"
            >
              <X className="h-3 w-3" />
            </button>
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  )
}
