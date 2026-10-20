import React, { useCallback, useEffect, useState } from 'react'
import { CheckCircle2, RefreshCw, AlertTriangle, XCircle, MinusCircle, Loader2 } from 'lucide-react'
import { Button } from '@/components/ui/Button'
import { Badge } from '@/components/ui/Badge'
import {
  getDoctor,
  getRuntimeConfig,
  getSystemInfo,
  listModelProfiles,
  type DoctorCheck,
  type ModelProfile,
  type RuntimeConfig,
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
  const [runtime, setRuntime] = useState<RuntimeConfig | null>(null)
  const [info, setInfo] = useState<SystemInfo | null>(null)
  const [isPinging, setIsPinging] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  const load = useCallback(async (probe: boolean) => {
    setLoading(true)
    try {
      const [doctor, modelProfiles, systemInfo, runtimeCfg] = await Promise.all([
        getDoctor(probe),
        listModelProfiles().catch(() => [] as ModelProfile[]),
        getSystemInfo().catch(() => null),
        getRuntimeConfig().catch(() => null),
      ])
      setChecks(doctor.checks)
      setProfiles(modelProfiles)
      setInfo(systemInfo)
      setRuntime(runtimeCfg)
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
          <div className="px-4 py-2.5 border-b border-[var(--paper-border)] bg-[var(--paper-subsurface)] text-xs font-mono uppercase tracking-wider text-[var(--ink-muted)] font-semibold">
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
                  <div className="px-3.5 pt-3 pb-1 text-xs font-mono uppercase tracking-wider text-[var(--ink-muted)]">
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
                            <div className="text-xs text-[var(--ink-secondary)] font-mono mt-0.5">
                              {check.detail}
                            </div>
                            {check.fix && (
                              <div className="text-xs text-[#b45309] mt-0.5">fix: {check.fix}</div>
                            )}
                          </div>
                        </div>
                        <Badge
                          variant={statusBadgeVariant(check.status)}
                          dot
                          className="text-xs shrink-0"
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

        {/* Right: what the engine will actually call.

            This panel used to render `/model-profiles` under the heading
            "model routing and latency matrix". That endpoint returns the
            engine's hard-coded capability *presets* — fourteen name patterns
            like `o1`, `o3`, `claude` — none of which is a configured model, and
            the payload contains no latency at all. On this machine the operator
            runs `local-gemma4` / `gemma4-e4b`, which matches none of the
            fourteen, so the screen showed fourteen irrelevant rows and hid the
            one configuration that was live. */}
        <div className="rounded-lg border border-[var(--paper-border)] bg-[var(--paper-surface)] overflow-hidden shadow-2xs">
          <div className="px-4 py-2.5 border-b border-[var(--paper-border)] bg-[var(--paper-subsurface)] text-xs font-mono uppercase tracking-wider text-[var(--ink-muted)] font-semibold">
            {t.doctor.runtimeConfig}
          </div>

          {/* Active configuration: the part the old screen never showed. */}
          {runtime ? (
            <div className="p-3.5 border-b border-[var(--paper-border)] bg-[var(--paper-subsurface)]">
              <div className="text-xs font-mono uppercase tracking-wider text-[var(--ink-muted)] mb-2 font-semibold">
                {t.doctor.activeNow}
              </div>
              <dl className="space-y-1 text-xs font-mono">
                <div className="flex justify-between gap-3">
                  <dt className="text-[var(--ink-secondary)]">{t.doctor.providerLabel}</dt>
                  <dd className="text-[var(--ink-primary)] font-semibold truncate">
                    {runtime.active.provider || '—'}
                  </dd>
                </div>
                <div className="flex justify-between gap-3">
                  <dt className="text-[var(--ink-secondary)]">{t.doctor.draftModel}</dt>
                  <dd className="text-[var(--ink-primary)] font-semibold truncate">
                    {runtime.active.draft_model || t.doctor.notConfigured}
                  </dd>
                </div>
                <div className="flex justify-between gap-3">
                  <dt className="text-[var(--ink-secondary)]">{t.doctor.repairModel}</dt>
                  <dd className="text-[var(--ink-primary)] truncate">
                    {runtime.active.repair_model || t.doctor.notConfigured}
                  </dd>
                </div>
                <div className="flex justify-between gap-3">
                  <dt className="text-[var(--ink-secondary)]">{t.doctor.endpoint}</dt>
                  <dd className="text-[var(--ink-primary)] truncate" title={runtime.active.base_url}>
                    {runtime.active.base_url || '—'}
                  </dd>
                </div>
                <div className="flex justify-between gap-3">
                  <dt className="text-[var(--ink-secondary)]">{t.doctor.capabilityProfile}</dt>
                  <dd
                    className={
                      runtime.active.matched_pattern
                        ? 'text-[var(--ink-primary)]'
                        : 'text-[#b45309] font-semibold'
                    }
                  >
                    {runtime.active.matched_pattern ?? t.doctor.noProfileMatch}
                  </dd>
                </div>
              </dl>
              {/* A model with no matching preset is not broken — it runs on the
                  generic default — but the operator should know the prompt
                  strategy was not tuned for it. */}
              {!runtime.active.matched_pattern && runtime.active.draft_model && (
                <p className="mt-2 text-xs text-[var(--ink-muted)] leading-relaxed">
                  {t.doctor.noProfileMatchHelp.replace(
                    '{strategy}',
                    runtime.active.prompt_strategy || 'hybrid',
                  )}
                </p>
              )}
            </div>
          ) : (
            <div className="p-4 text-xs text-[var(--ink-muted)]">
              {t.doctor.runtimeConfigUnavailable}
            </div>
          )}

          {/* Declared providers, credentials filtered server-side. */}
          <div className="px-3.5 py-2.5 border-b border-[var(--paper-border)]">
            <div className="text-xs font-mono uppercase tracking-wider text-[var(--ink-muted)] mb-2 font-semibold">
              {t.doctor.declaredProviders}
            </div>
            {/* Only what the operator declared. The API also returns shipped
                wire protocols and bare aliases (`chat`, `claude`, `responses`);
                rendering all three put seven aliases above the six real
                entries. Aliases are folded away below. */}
            <div className="space-y-1">
              {(runtime?.providers ?? [])
                .filter((p) => p.declared)
                .map((p) => (
                  <div
                    key={p.name}
                    className={`flex items-center justify-between gap-2 rounded-[4px] px-2 py-1 text-xs font-mono ${
                      p.active
                        ? 'bg-[#15803d]/10 border border-[#15803d]/25'
                        : 'border border-transparent'
                    }`}
                  >
                    <span
                      className={`truncate ${p.active ? 'font-semibold text-[#15803d]' : 'text-[var(--ink-secondary)]'}`}
                    >
                      {p.name}
                    </span>
                    <span className="flex items-center gap-1.5 shrink-0">
                      {p.is_free && (
                        <span className="text-[10px] text-[var(--ink-muted)]">
                          {t.doctor.freeLabel}
                        </span>
                      )}
                      <span className="text-[10px] text-[var(--ink-muted)] truncate max-w-[9rem]">
                        {p.draft_model || p.api_mode || '—'}
                      </span>
                    </span>
                  </div>
                ))}
            </div>

            {/* Built-in protocols and aliases, collapsed: they are usable names,
                not configurations. */}
            <details className="mt-2">
              <summary className="cursor-pointer text-[11px] font-mono text-[var(--ink-muted)] hover:text-[var(--ink-primary)] select-none">
                {t.doctor.builtinProviders} (
                {(runtime?.providers ?? []).filter((p) => !p.declared).length})
              </summary>
              <div className="mt-1 space-y-0.5">
                {(runtime?.providers ?? [])
                  .filter((p) => !p.declared)
                  .map((p) => (
                    <div
                      key={p.name}
                      className="flex items-center justify-between gap-2 px-2 text-[11px] font-mono text-[var(--ink-muted)]"
                    >
                      <span className="truncate">{p.name}</span>
                      <span className="text-[10px] shrink-0">
                        {p.is_alias ? t.doctor.aliasLabel : p.api_mode || '—'}
                      </span>
                    </div>
                  ))}
              </div>
            </details>
            <p className="mt-2 text-xs text-[var(--ink-muted)] leading-relaxed">
              {t.doctor.providerConfigHint}
            </p>
          </div>

          {/* Capability presets: explicitly labelled as matching rules, not as
              configured models, so the list is read for what it is. */}
          <details className="group">
            <summary className="px-3.5 py-2.5 cursor-pointer text-xs font-mono uppercase tracking-wider text-[var(--ink-muted)] font-semibold hover:text-[var(--ink-primary)] select-none">
              {t.doctor.capabilityPresets} ({profiles.length})
            </summary>
            <p className="px-3.5 pb-2 text-xs text-[var(--ink-muted)] leading-relaxed">
              {t.doctor.capabilityPresetsHelp}
            </p>
            <div className="divide-y divide-[var(--paper-border)]">
              {profiles.map((profile) => (
                <div key={profile.model_pattern} className="px-3.5 py-2 text-xs">
                  <div className="font-semibold text-[var(--ink-primary)] flex items-center gap-2">
                    <span className="font-mono truncate">{profile.model_pattern}</span>
                    <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-[var(--paper-subsurface)] text-[var(--ink-muted)] border border-[var(--paper-border)]">
                      {profile.prompt_strategy}
                    </span>
                    {runtime?.active.matched_pattern === profile.model_pattern && (
                      <span className="text-[10px] font-mono text-[#15803d]">
                        {t.doctor.inUse}
                      </span>
                    )}
                  </div>
                  <div className="text-[11px] text-[var(--ink-secondary)] mt-0.5 flex flex-wrap gap-1.5 font-mono">
                    {profile.supports_vision && <span>{t.doctor.capabilityVision}</span>}
                    {profile.supports_reasoning_effort && <span>{t.doctor.capabilityReasoning}</span>}
                    <span>
                      {profile.supports_system_prompt
                        ? t.doctor.capabilitySystem
                        : t.doctor.capabilityNoSystem}
                    </span>
                    <span>
                      {t.doctor.capabilityExtract}:{profile.extraction_strategy}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </details>
        </div>
      </div>

      {/* Security Boundary Panel */}
      <div className="p-4 rounded-lg border border-[var(--paper-border)] bg-[var(--paper-surface)] space-y-1.5 text-xs font-mono text-[var(--ink-secondary)] shadow-2xs">
        <div className="text-xs font-mono uppercase tracking-wider text-[var(--ink-muted)] mb-2 font-semibold">
          {t.doctor.securityBoundary}
        </div>
        {info ? (
          <>
            <div className="flex items-center gap-2">
              <span className={info.is_loopback ? 'text-[#15803d]' : 'text-[#b45309]'}>
                • {t.doctor.listenInterface}: {info.host}
              </span>
              <Badge variant={info.is_loopback ? 'success' : 'warning'} className="text-xs">
                {info.is_loopback ? t.doctor.loopbackOnly : t.doctor.exposed}
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
