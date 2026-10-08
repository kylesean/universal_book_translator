import React, { useCallback, useEffect, useState } from 'react'
import { CheckCircle2, RefreshCw, AlertTriangle, XCircle, MinusCircle, Loader2 } from 'lucide-react'
import { Button } from '@/components/ui/Button'
import { Badge } from '@/components/ui/Badge'
import {
  getDoctor,
  getSystemInfo,
  listModelProfiles,
  type DoctorCheck,
  type ModelProfile,
  type SystemInfo,
} from '@/api/client'
import { useI18n } from '@/i18n/useI18n'

function CheckIcon({ status }: { status: string }) {
  if (status === 'FAIL') return <XCircle className="h-4 w-4 text-[#b91c1c] shrink-0 mt-0.5" />
  if (status === 'WARN') return <AlertTriangle className="h-4 w-4 text-[#b45309] shrink-0 mt-0.5" />
  if (status === 'SKIP') return <MinusCircle className="h-4 w-4 text-[var(--ink-muted)] shrink-0 mt-0.5" />
  return <CheckCircle2 className="h-4 w-4 text-[#15803d] shrink-0 mt-0.5" />
}

function statusBadgeVariant(status: string): 'success' | 'warning' | 'destructive' | 'outline' {
  if (status === 'FAIL') return 'destructive'
  if (status === 'WARN') return 'warning'
  if (status === 'SKIP') return 'outline'
  return 'success'
}

export function SystemDoctorView() {
  const { t } = useI18n()
  const [checks, setChecks] = useState<DoctorCheck[]>([])
  const [profiles, setProfiles] = useState<ModelProfile[]>([])
  const [info, setInfo] = useState<SystemInfo | null>(null)
  const [isPinging, setIsPinging] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  const load = useCallback(async (probe: boolean) => {
    setLoading(true)
    try {
      const [doctor, modelProfiles, systemInfo] = await Promise.all([
        getDoctor(probe),
        listModelProfiles().catch(() => [] as ModelProfile[]),
        getSystemInfo().catch(() => null),
      ])
      setChecks(doctor.checks)
      setProfiles(modelProfiles)
      setInfo(systemInfo)
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Doctor check failed')
    } finally {
      setLoading(false)
      setIsPinging(false)
    }
  }, [])

  useEffect(() => {
    // Mount-time load; `load`'s setStates are the loading flag and the results,
    // and every one of them runs after the awaited fetches.
    // oxlint-disable-next-line react/set-state-in-effect
    void load(false)
  }, [load])

  const handleTestPings = () => {
    setIsPinging(true)
    void load(true)
  }

  const groups = Array.from(new Set(checks.map((check) => check.group)))

  return (
    <div className="flex-1 overflow-y-auto px-8 py-7 space-y-7 max-w-5xl mx-auto w-full">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-[var(--paper-border)] pb-4">
        <div>
          <h1 className="text-xl font-bold tracking-tight text-[var(--ink-primary)]">
            {t.doctor.title}
          </h1>
          <p className="text-xs text-[var(--ink-secondary)] mt-1">{t.doctor.subtitle}</p>
        </div>

        <Button onClick={handleTestPings} variant="secondary" size="sm" disabled={isPinging || loading}>
          <RefreshCw className={`h-3 w-3 mr-1.5 ${isPinging || loading ? 'animate-spin' : ''}`} />
          {t.doctor.pingAll}
        </Button>
      </div>

      {error && (
        <div className="p-3 rounded-lg border border-[var(--paper-border)] bg-[var(--paper-surface)] text-xs text-[var(--ink-secondary)] flex items-center gap-2">
          <AlertTriangle className="h-4 w-4 text-[#b45309] shrink-0" />
          {error}
        </div>
      )}

      {/* Diagnostics & Providers 2-Col Split */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Left: Diagnostics */}
        <div className="rounded-lg border border-[var(--paper-border)] bg-[var(--paper-surface)] overflow-hidden shadow-2xs">
          <div className="px-4 py-2.5 border-b border-[var(--paper-border)] bg-[var(--paper-subsurface)] text-[10px] font-mono uppercase tracking-wider text-[var(--ink-muted)] font-semibold">
            {t.doctor.engineDiagnostics}
          </div>
          <div className="divide-y divide-[var(--paper-border)]">
            {loading && checks.length === 0 ? (
              <div className="p-6 text-center text-[var(--ink-muted)]">
                <Loader2 className="h-4 w-4 animate-spin mx-auto" />
              </div>
            ) : (
              groups.map((group) => (
                <div key={group}>
                  <div className="px-3.5 pt-3 pb-1 text-[10px] font-mono uppercase tracking-wider text-[var(--ink-muted)]">
                    {group}
                  </div>
                  {checks
                    .filter((check) => check.group === group)
                    .map((check, idx) => (
                      <div
                        key={`${group}-${idx}`}
                        className="px-3.5 py-2.5 flex items-start justify-between gap-3 text-xs"
                      >
                        <div className="flex items-start gap-2.5">
                          <CheckIcon status={check.status} />
                          <div>
                            <div className="font-semibold text-[var(--ink-primary)]">
                              {check.name}
                            </div>
                            <div className="text-[11px] text-[var(--ink-secondary)] font-mono mt-0.5">
                              {check.detail}
                            </div>
                            {check.fix && (
                              <div className="text-[11px] text-[#b45309] mt-0.5">fix: {check.fix}</div>
                            )}
                          </div>
                        </div>
                        <Badge
                          variant={statusBadgeVariant(check.status)}
                          dot
                          className="text-[10px] shrink-0"
                        >
                          {check.status}
                        </Badge>
                      </div>
                    ))}
                </div>
              ))
            )}
          </div>
        </div>

        {/* Right: Model Provider Matrix */}
        <div className="rounded-lg border border-[var(--paper-border)] bg-[var(--paper-surface)] overflow-hidden shadow-2xs">
          <div className="px-4 py-2.5 border-b border-[var(--paper-border)] bg-[var(--paper-subsurface)] text-[10px] font-mono uppercase tracking-wider text-[var(--ink-muted)] font-semibold">
            {t.doctor.providerMatrix}
          </div>
          <div className="divide-y divide-[var(--paper-border)]">
            {profiles.length === 0 ? (
              <div className="p-6 text-center text-xs text-[var(--ink-muted)]">
                No model profiles registered.
              </div>
            ) : (
              profiles.map((profile) => (
                <div key={profile.model_pattern} className="p-3.5 text-xs">
                  <div className="font-semibold text-[var(--ink-primary)] flex items-center gap-2">
                    <span className="font-mono truncate">{profile.model_pattern}</span>
                    <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-[var(--paper-subsurface)] text-[var(--ink-muted)] border border-[var(--paper-border)]">
                      {profile.prompt_strategy}
                    </span>
                  </div>
                  <div className="text-[11px] text-[var(--ink-secondary)] mt-1 flex flex-wrap gap-1.5 font-mono">
                    {profile.supports_vision && <span>vision</span>}
                    {profile.supports_reasoning_effort && <span>reasoning</span>}
                    {profile.supports_system_prompt ? <span>system</span> : <span>no-system</span>}
                    <span>extract:{profile.extraction_strategy}</span>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      {/* Security Boundary Panel */}
      <div className="p-4 rounded-lg border border-[var(--paper-border)] bg-[var(--paper-surface)] space-y-1.5 text-xs font-mono text-[var(--ink-secondary)] shadow-2xs">
        <div className="text-[10px] font-mono uppercase tracking-wider text-[var(--ink-muted)] mb-2 font-semibold">
          {t.doctor.securityBoundary}
        </div>
        {info ? (
          <>
            <div className="flex items-center gap-2">
              <span className={info.is_loopback ? 'text-[#15803d]' : 'text-[#b45309]'}>
                • {t.doctor.listenInterface}: {info.host}
              </span>
              <Badge variant={info.is_loopback ? 'success' : 'warning'} className="text-[10px]">
                {info.is_loopback ? t.doctor.loopbackOnly : 'EXPOSED'}
              </Badge>
            </div>
            <div>• {t.doctor.authGate}: {info.auth_enabled ? t.doctor.enabled : t.doctor.disabled}</div>
            <div>• {t.doctor.jobMode}: {info.job_mode}</div>
            <div>• {t.doctor.dbDir}: {info.db_dir}</div>
            {info.disk_free_gb != null && (
              <div>• {t.doctor.diskFree}: {info.disk_free_gb} GB</div>
            )}
            {info.wal_status && (
              <div>• {t.doctor.walStatus}: {info.wal_status}</div>
            )}
            <div>
              • {t.doctor.allowedRoots}:{' '}
              {info.allowed_bases.length === 0 ? '—' : info.allowed_bases.join(', ')}
            </div>
            <div>• {t.doctor.forbiddenPaths}: /etc, /root, /var/run, ~/.ssh (Hard refusal)</div>
            {!info.is_loopback && (
              <div className="text-[#b45309] pt-1">⚠ {t.doctor.exposedWarning}</div>
            )}
          </>
        ) : (
          <div className="text-[var(--ink-muted)]">—</div>
        )}
      </div>
    </div>
  )
}
