import React, { useEffect, useRef, useState } from 'react'
import { AlertTriangle } from 'lucide-react'
import { Button } from '@/components/ui/Button'

interface ConfirmDialogProps {
  open: boolean
  title: string
  body: React.ReactNode
  confirmLabel: string
  cancelLabel: string
  danger?: boolean
  onConfirm: () => void
  onCancel: () => void
  /**
   * Require the operator to type this text before the confirm button enables.
   *
   * For an operation whose cost the button cannot convey — discarding a paid
   * run's ledger, say — a single click is too cheap. Typing something only this
   * dialog names forces the operator to read what is about to happen, and makes
   * an accidental Enter harmless.
   */
  typeToConfirm?: string
  /** Label above the text field; `{phrase}` is replaced with the phrase. */
  typeToConfirmLabel?: string
  /** Shown under the field while the text does not match. */
  typeToConfirmMismatch?: string
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
  typeToConfirm,
  typeToConfirmLabel,
  typeToConfirmMismatch,
}: ConfirmDialogProps) {
  const confirmRef = useRef<HTMLButtonElement>(null)
  const [typed, setTyped] = useState('')
  const needsPhrase = Boolean(typeToConfirm)
  const phraseOk = !needsPhrase || typed.trim() === typeToConfirm

  useEffect(() => {
    if (!open) return
    // With a phrase required, focus the field (where the operator must act)
    // rather than the button, which would make Enter confirm immediately.
    if (!needsPhrase) confirmRef.current?.focus()
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onCancel()
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [open, onCancel, needsPhrase])

  if (!open) return null

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/25 p-4 sm:p-6"
      onClick={onCancel}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-label={title}
        className="w-full max-w-md sm:max-w-lg rounded-lg border border-[var(--paper-border)] bg-[var(--paper-surface)] p-5 shadow-[0_8px_32px_rgba(0,0,0,0.14)]"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="mb-3 flex items-start gap-2.5">
          <AlertTriangle
            className={`h-4 w-4 mt-0.5 shrink-0 ${danger ? 'text-[#b91c1c]' : 'text-[#b45309]'}`}
          />
          <h2 className="text-sm font-semibold text-[var(--ink-primary)]">{title}</h2>
        </div>
        <div className="mb-5 text-xs leading-relaxed text-[var(--ink-secondary)] break-words">
          {body}
        </div>
        {needsPhrase && (
          <div className="mb-4 space-y-2">
            <label
              htmlFor="confirm-phrase"
              className="block text-xs leading-relaxed text-[var(--ink-secondary)]"
            >
              {typeToConfirmLabel ? (
                (() => {
                  const parts = typeToConfirmLabel.split('{phrase}')
                  if (parts.length === 2) {
                    return (
                      <span className="block space-y-1.5">
                        <span>{parts[0].trim()}</span>
                        <code className="block break-all rounded border border-[var(--paper-border)] bg-[var(--paper-subsurface)] px-2 py-1 font-mono text-xs font-semibold text-[var(--ink-primary)] select-all">
                          {typeToConfirm}
                        </code>
                        {parts[1].trim() && <span>{parts[1].trim()}</span>}
                      </span>
                    )
                  }
                  return (
                    <span className="break-all">
                      {typeToConfirmLabel.replace('{phrase}', typeToConfirm ?? '')}
                    </span>
                  )
                })()
              ) : (
                <span className="block space-y-1.5">
                  <span>Type to confirm:</span>
                  <code className="block break-all rounded border border-[var(--paper-border)] bg-[var(--paper-subsurface)] px-2 py-1 font-mono text-xs font-semibold text-[var(--ink-primary)] select-all">
                    {typeToConfirm}
                  </code>
                </span>
              )}
            </label>
            <input
              id="confirm-phrase"
              autoFocus
              value={typed}
              onChange={(event) => setTyped(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Enter' && phraseOk) onConfirm()
              }}
              spellCheck={false}
              autoComplete="off"
              className="w-full rounded border border-[var(--paper-border)] bg-[var(--paper-subsurface)] px-2.5 py-1.5 font-mono text-xs text-[var(--ink-primary)] focus:outline-none focus:ring-1 focus:ring-[var(--ink-primary)]"
            />
            {!phraseOk && typed.length > 0 && (
              <p className="mt-1 text-xs text-[#b91c1c] break-words">
                {typeToConfirmMismatch ?? 'That does not match.'}
              </p>
            )}
          </div>
        )}
        <div className="flex justify-end gap-2">
          <Button onClick={onCancel} variant="secondary" size="sm">
            {cancelLabel}
          </Button>
          <Button
            ref={confirmRef}
            onClick={onConfirm}
            disabled={!phraseOk}
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
