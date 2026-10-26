import React, { useState } from 'react'
import { AlertTriangle } from 'lucide-react'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog'
import { cn } from '@/lib/utils'
import { useI18n } from '@/i18n/useI18n'

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
  const { t } = useI18n()
  const [typed, setTyped] = useState('')
  const needsPhrase = Boolean(typeToConfirm)
  const phraseOk = !needsPhrase || typed.trim() === typeToConfirm

  return (
    <AlertDialog
      open={open}
      onOpenChange={(nextOpen) => {
        if (!nextOpen) {
          setTyped('')
          onCancel()
        }
      }}
    >
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle className="flex items-start gap-2.5">
            <AlertTriangle
              className={cn(
                'mt-0.5 size-4 shrink-0',
                danger ? 'text-destructive' : 'text-amber-700 dark:text-amber-400'
              )}
            />
            <span>{title}</span>
          </AlertDialogTitle>
          <AlertDialogDescription asChild>
            <div className="break-words text-sm">{body}</div>
          </AlertDialogDescription>
        </AlertDialogHeader>
        {needsPhrase && (
          <div className="space-y-2">
            <Label htmlFor="confirm-phrase" className="block text-xs leading-relaxed text-muted-foreground">
              {typeToConfirmLabel ? (
                (() => {
                  const parts = typeToConfirmLabel.split('{phrase}')
                  if (parts.length === 2) {
                    return (
                      <span className="block space-y-1.5">
                        <span>{parts[0].trim()}</span>
                        <code className="block break-all rounded border border-border bg-muted/40 px-2 py-1 font-mono text-xs font-semibold text-foreground select-all">
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
                  <span>{t.common.typeToConfirm}</span>
                  <code className="block break-all rounded border border-border bg-muted/40 px-2 py-1 font-mono text-xs font-semibold text-foreground select-all">
                    {typeToConfirm}
                  </code>
                </span>
              )}
            </Label>
            <Input
              id="confirm-phrase"
              autoFocus
              value={typed}
              onChange={(event) => setTyped(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Enter' && phraseOk) onConfirm()
              }}
              spellCheck={false}
              autoComplete="off"
              className="font-mono text-xs"
            />
            {!phraseOk && typed.length > 0 && (
              <p className="text-xs text-destructive break-words">
                {typeToConfirmMismatch ?? t.common.doesNotMatch}
              </p>
            )}
          </div>
        )}
        <AlertDialogFooter>
          <AlertDialogCancel className="h-8 px-3 text-xs">
            {cancelLabel}
          </AlertDialogCancel>
          <AlertDialogAction
            onClick={onConfirm}
            disabled={!phraseOk}
            className={cn(
              'h-8 px-3 text-xs shadow-xs',
              danger
                ? 'border border-destructive/80 bg-destructive text-destructive-foreground hover:bg-destructive/90'
                : 'border border-primary/90 bg-primary text-primary-foreground hover:bg-primary/90'
            )}
          >
            {confirmLabel}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  )
}
