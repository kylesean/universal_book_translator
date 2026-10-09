import React, { useEffect, useRef } from 'react'
import { AlertTriangle } from 'lucide-react'
import { Button } from '@/components/ui/Button'

interface ConfirmDialogProps {
  open: boolean
  title: string
  body: string
  confirmLabel: string
  cancelLabel: string
  danger?: boolean
  onConfirm: () => void
  onCancel: () => void
}

/**
 * In-app replacement for ``window.confirm``.
 *
 * The native dialog is unstyled, untranslatable beyond its buttons, blocks the
 * event loop, and cannot carry the console's tone. This one traps focus on the
 * confirm button, closes on Escape and on backdrop click, and is announced as a
 * modal dialog.
 */
export function ConfirmDialog({
  open,
  title,
  body,
  confirmLabel,
  cancelLabel,
  danger = false,
  onConfirm,
  onCancel,
}: ConfirmDialogProps) {
  const confirmRef = useRef<HTMLButtonElement>(null)

  useEffect(() => {
    if (!open) return
    confirmRef.current?.focus()
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onCancel()
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [open, onCancel])

  if (!open) return null

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/25 p-6"
      onClick={onCancel}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-label={title}
        className="w-full max-w-sm rounded-lg border border-[var(--paper-border)] bg-[var(--paper-surface)] p-5 shadow-[0_8px_32px_rgba(0,0,0,0.14)]"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="mb-3 flex items-start gap-2.5">
          <AlertTriangle
            className={`h-4 w-4 mt-0.5 shrink-0 ${danger ? 'text-[#b91c1c]' : 'text-[#b45309]'}`}
          />
          <h2 className="text-sm font-semibold text-[var(--ink-primary)]">{title}</h2>
        </div>
        <p className="mb-5 text-xs leading-relaxed text-[var(--ink-secondary)]">{body}</p>
        <div className="flex justify-end gap-2">
          <Button onClick={onCancel} variant="secondary" size="sm">
            {cancelLabel}
          </Button>
          <Button
            ref={confirmRef}
            onClick={onConfirm}
            variant={danger ? 'danger' : 'primary'}
            size="sm"
          >
            {confirmLabel}
          </Button>
        </div>
      </div>
    </div>
  )
}
