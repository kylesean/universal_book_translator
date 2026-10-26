import { useCallback, useEffect, useState } from 'react'
import { Loader2, ShieldCheck, ShieldOff } from 'lucide-react'

import { getEgressConfig, saveEgressConfig } from '@/api/client'
import { Badge } from '@/components/ui/badge'
import { ConfirmDialog } from '@/components/ui/ConfirmDialog'
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import { useI18n } from '@/i18n/useI18n'
import { cn } from '@/lib/utils'

/**
 * The page-egress gate, with a runtime toggle.
 *
 * This switch sends rendered page images off the machine, so enabling it is
 * confirmed first. The runtime-only scope and its fallback to the server
 * setting are stated beside the switch rather than in a second status badge.
 */
export function EgressSettingsCard() {
  const { t } = useI18n()
  const [current, setCurrent] = useState<boolean | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [confirmOpen, setConfirmOpen] = useState(false)

  const load = useCallback(async () => {
    try {
      setCurrent((await getEgressConfig()).allow_page_upload)
    } catch {
      // Unknown must not render as either privacy verdict.
      setCurrent(null)
    }
  }, [])

  useEffect(() => {
    // oxlint-disable-next-line react/set-state-in-effect
    void load()
  }, [load])

  const setEgress = async (allow: boolean) => {
    setBusy(true)
    setError(null)
    try {
      setCurrent((await saveEgressConfig({ allow_page_upload: allow })).allow_page_upload)
      setConfirmOpen(false)
    } catch (err) {
      setError(err instanceof Error ? err.message : t.doctor.egressSaveFailed)
    } finally {
      setBusy(false)
    }
  }

  return (
    <section className="min-w-0 overflow-hidden rounded-xl border border-border/80 bg-card">
      <header className="border-b border-border/70 px-4 py-3">
        <h2 className="text-sm font-semibold">{t.doctor.egressTitle}</h2>
        <p className="mt-0.5 text-xs leading-relaxed text-muted-foreground">
          {t.doctor.egressCardHelp}
        </p>
      </header>

      {current === null ? (
        <div className="flex items-center justify-center px-4 py-10 text-muted-foreground">
          <Loader2 className="size-4 animate-spin" />
        </div>
      ) : (
        <div className="space-y-3 p-4">
          <div className="flex items-start justify-between gap-4">
            <div className="min-w-0">
              <div className="flex flex-wrap items-center gap-2">
                <Label htmlFor="allow-page-upload" className="text-sm font-medium">
                  {t.doctor.egressLabel}
                </Label>
                <Badge
                  variant="outline"
                  className={cn(
                    current
                      ? 'border-amber-600/25 bg-amber-500/10 text-amber-700 dark:text-amber-400'
                      : 'border-emerald-600/25 bg-emerald-500/10 text-emerald-700 dark:text-emerald-400'
                  )}
                >
                  {current ? (
                    <ShieldOff className="mr-1 size-3.5" />
                  ) : (
                    <ShieldCheck className="mr-1 size-3.5" />
                  )}
                  {current ? t.doctor.egressStateOn : t.doctor.egressStateOff}
                </Badge>
              </div>
              <p className="mt-2 text-xs leading-relaxed text-muted-foreground">
                {current ? t.doctor.egressDescOn : t.doctor.egressDescOff}
              </p>
            </div>
            <Switch
              id="allow-page-upload"
              checked={current}
              disabled={busy}
              onCheckedChange={(checked) => {
                if (checked) setConfirmOpen(true)
                else void setEgress(false)
              }}
            />
          </div>
          <p className="border-t border-border/60 pt-3 text-[11px] leading-relaxed text-muted-foreground">
            {t.doctor.egressRuntimeNote}
          </p>
          {error ? <p className="text-xs text-destructive">{error}</p> : null}
        </div>
      )}

      <ConfirmDialog
        open={confirmOpen}
        title={t.doctor.egressConfirmTitle}
        body={<span className="block">{t.doctor.egressConfirmBody}</span>}
        confirmLabel={t.doctor.egressConfirmOn}
        cancelLabel={t.doctor.egressConfirmCancel}
        onConfirm={() => void setEgress(true)}
        onCancel={() => setConfirmOpen(false)}
      />
    </section>
  )
}
