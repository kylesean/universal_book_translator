import { useCallback, useEffect, useState } from 'react'
import { Loader2 } from 'lucide-react'
import { ConfirmDialog } from '@/components/ui/ConfirmDialog'
import { getEgressConfig, saveEgressConfig } from '@/api/client'
import { useI18n } from '@/i18n/useI18n'

/**
 * The page-egress gate (``UBTConfig.allow_page_upload``), with a console toggle.
 *
 * The gate is fail-closed by design: the rendered pages of an unpublished
 * manuscript are the sensitive half of the payload, so shipping them must be
 * an explicit act. Until now that act lived only in a server environment
 * variable no console reader could perform, which made every egress-dependent
 * control name a fix the reader could not reach — the exact dead end the QE
 * card existed to remove.
 *
 * **The toggle is a runtime override, and says so.** ``ubt.toml`` has no key
 * for this field, so the choice holds for this service run and reverts at
 * restart; the permanent path stays the documented ``UBT_ALLOW_PAGE_UPLOAD``
 * environment variable. Turning it ON confirms first — it is the one click on
 * this card that sends data off the machine.
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
      // Unknown is rendered as "unknown", never as either verdict — a failed
      // read must not display a privacy state the server did not report.
      setCurrent(null)
    }
  }, [])

  useEffect(() => {
    // Fetch-on-mount: the gate's state lives on the server, so syncing with it
    // is what this effect is for.
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
    <div className="rounded-lg border border-[var(--paper-border)] bg-[var(--paper-surface)] overflow-hidden shadow-2xs">
      <div className="px-4 py-2.5 border-b border-[var(--paper-border)] bg-[var(--paper-subsurface)] flex items-center justify-between gap-3">
        <span className="type-group">{t.doctor.egressTitle}</span>
        {current !== null && (
          <span
            className={`type-help font-mono ${current ? 'text-[#15803d]' : 'text-[var(--ink-muted)]'}`}
          >
            {current ? t.doctor.egressStateOn : t.doctor.egressStateOff}
          </span>
        )}
      </div>

      {current === null ? (
        <div className="p-6 text-center text-[var(--ink-muted)]">
          <Loader2 className="h-4 w-4 animate-spin mx-auto" />
        </div>
      ) : (
        <div className="p-3.5 space-y-2">
          <div className="flex items-center justify-between gap-3">
            <span className="type-value font-semibold text-[var(--ink-primary)]">
              {t.doctor.egressLabel}
            </span>
            <label className="inline-flex items-center gap-2 cursor-pointer select-none">
              <input
                type="checkbox"
                checked={current}
                disabled={busy}
                onChange={(e) => {
                  if (e.target.checked) {
                    setConfirmOpen(true)
                  } else {
                    void setEgress(false)
                  }
                }}
                className="h-3.5 w-3.5 rounded border-[var(--paper-border)] accent-[var(--ink-primary)] cursor-pointer disabled:cursor-wait"
              />
              <span className="type-value">
                {current ? t.doctor.egressToggleOn : t.doctor.egressToggleOff}
              </span>
            </label>
          </div>
          <p className="type-help text-[var(--ink-secondary)] leading-relaxed">
            {current ? t.doctor.egressDescOn : t.doctor.egressDescOff}
          </p>
          <p className="type-help text-[var(--ink-muted)] leading-relaxed">
            {t.doctor.egressRuntimeNote}
          </p>
          {error && <p className="type-help text-[#b91c1c]">{error}</p>}
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
    </div>
  )
}
