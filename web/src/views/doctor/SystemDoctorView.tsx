import React, { useCallback, useEffect, useRef, useState } from 'react'
import {
  CheckCircle2,
  RefreshCw,
  AlertTriangle,
  XCircle,
  MinusCircle,
  Loader2,
  Pencil,
  Plus,
} from 'lucide-react'
import { Button } from '@/components/ui/Button'
import { Badge } from '@/components/ui/Badge'
import { InfoTip } from '@/components/ui/InfoTip'
import {
  getDoctor,
  getRuntimeConfig,
  getSystemInfo,
  setDefaultProvider,
  type DoctorCheck,
  type RuntimeConfig,
  type SystemInfo,
} from '@/api/client'
import { useI18n } from '@/i18n/useI18n'
import { localizeDoctorCheck } from '@/i18n/doctorText'
import { doctorCodesEn, doctorCodesZh } from '@/i18n/translations/doctorCodes'
import { ProviderEditor } from './ProviderEditor'
import { QESettingsCard } from './QESettingsCard'

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
  const { t, language } = useI18n()
  const [checks, setChecks] = useState<DoctorCheck[]>([])
  const [runtime, setRuntime] = useState<RuntimeConfig | null>(null)
  const [info, setInfo] = useState<SystemInfo | null>(null)
  // ``undefined`` = closed, ``null`` = creating a new block, a string = editing
  // that one. A single ``string | null`` cannot express all three, and the
  // difference between "closed" and "create" is the difference between a dead
  // button and a working one.
  const [editingProvider, setEditingProvider] = useState<string | null | undefined>(undefined)
  const [isPinging, setIsPinging] = useState(false)
  // Which row is mid-switch, so the button disables instead of firing twice.
  const [defaultBusy, setDefaultBusy] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const editorRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (editingProvider !== undefined && editorRef.current) {
      editorRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
  }, [editingProvider])

  const load = useCallback(async (probe: boolean) => {
    setLoading(true)
    try {
      const [doctor, systemInfo, runtimeCfg] = await Promise.all([
        getDoctor(probe),
        getSystemInfo().catch(() => null),
        getRuntimeConfig().catch(() => null),
      ])
      setChecks(doctor.checks)
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

  // Declared *after* `load`: it calls that binding, and a callback defined
  // above it would close over the uninitialized const (a TDZ ReferenceError
  // at first click that neither tsc nor oxlint reports).
  const handleMakeDefault = useCallback(
    async (name: string) => {
      setDefaultBusy(name)
      setError(null)
      try {
        await setDefaultProvider(name)
        // Re-read so `is_default` and the active provider both reflect the file.
        await load(false)
      } catch (err) {
        setError(err instanceof Error ? err.message : `Failed to select ${name}`)
      } finally {
        setDefaultBusy(null)
      }
    },
    [load],
  )

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

  // The engine prints English; this panel is the same checklist. Each check
  // carries a stable code plus the values interpolated into its English sentence,
  // so the console renders its own copy. A code the catalogue does not know falls
  // back to the engine's English text — never a raw key.
  const codeTables = language === 'zh' ? doctorCodesZh : doctorCodesEn
  const localized = checks.map((check) => ({
    check,
    text: localizeDoctorCheck(check, codeTables),
  }))
  // Ordered by the engine's group codes, so the sections keep the CLI's order
  // rather than the order they happen to arrive in.
  const groups = Array.from(new Set(localized.map(({ check }) => check.group_code)))

  return (
    <div className="flex-1 overflow-y-auto px-4 sm:px-8 py-5 sm:py-7 space-y-5 sm:space-y-7 max-w-5xl mx-auto w-full">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-[var(--paper-border)] pb-4">
        <div>
          <h1 className="text-xl font-bold tracking-tight text-[var(--ink-primary)]">
            {t.doctor.title}
          </h1>
          <p className="text-xs text-[var(--ink-secondary)] mt-1">{t.doctor.subtitle}</p>
        </div>

        <Button
          onClick={handleTestPings}
          variant="secondary"
          size="sm"
          disabled={isPinging || loading}
          className="self-start sm:self-auto min-h-[38px] sm:min-h-0"
        >
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
              groups.map((groupCode) => (
                <div key={groupCode}>
                  <div className="px-3.5 pt-3 pb-1 text-xs font-mono uppercase tracking-wider text-[var(--ink-muted)]">
                    {localized.find(({ check }) => check.group_code === groupCode)?.text.group ??
                      groupCode}
                  </div>
                  {localized
                    .filter(({ check }) => check.group_code === groupCode)
                    .map(({ check, text }, idx) => (
                      <div
                        key={`${groupCode}-${idx}`}
                        className="px-3.5 py-2.5 flex items-start justify-between gap-3 text-xs"
                      >
                        <div className="flex items-start gap-2.5 min-w-0 flex-1">
                          <CheckIcon status={check.status} />
                          <div className="min-w-0 flex-1">
                            <div className="font-semibold text-[var(--ink-primary)] break-words">
                              {text.name}
                            </div>
                            <div className="text-xs text-[var(--ink-secondary)] font-mono mt-0.5 break-words">
                              {text.detail}
                            </div>
                            {text.fix && (
                              <div className="text-xs text-[#b45309] mt-0.5 break-words">
                                {t.doctor.fixLabel}: {text.fix}
                              </div>
                            )}
                          </div>
                        </div>
                        <Badge
                          variant={statusBadgeVariant(check.status)}
                          dot
                          className="text-xs shrink-0"
                        >
                          {t.doctor.checkStatuses[check.status] ?? check.status}
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
            "model routing and latency matrix". That endpoint returned the
            engine's hard-coded capability *presets* — fourteen name patterns
            like `o1`, `o3`, `claude` — none of which was a configured model, and
            the payload contained no latency at all. On this machine the operator
            runs `local-gemma4` / `gemma4-e4b`, which matched none of the
            fourteen, so the screen showed fourteen irrelevant rows and hid the
            one configuration that was live.

            That table is gone. Capability is now declared per provider block, so
            this panel shows the live configuration and lets the operator edit
            the block it comes from.

            The panel and the editor share ONE grid cell (``col-start-2``), and
            the editor replaces the panel while it is open. They used to be two
            separate grid children, which put three boxes into a two-column grid:
            the editor took the right cell of row 1 and the whole runtime panel
            wrapped into row 2 at x=368 — under the diagnostics, in the *left*
            column, showing "local-gemma4 is active" while the operator edited
            index-translate. Same data, twice on screen, one copy in the wrong
            place. */}
        <div ref={editorRef} className="md:col-start-2 md:row-start-1">
          {editingProvider !== undefined ? (
            <ProviderEditor
              name={editingProvider}
              // ``undefined`` → null: the configuration could not be read, which
              // the form reports as unknown rather than as a refusal.
              canWriteCredential={
                runtime ? (runtime.can_write_credential ?? false) : null
              }
              onCancel={() => setEditingProvider(undefined)}
              onSaved={() => void load(false)}
            />
          ) : null}

          <div
            className={`rounded-lg border border-[var(--paper-border)] bg-[var(--paper-surface)] overflow-hidden shadow-2xs ${
              // ``hidden``, not ``display:none`` via a conditional: the editor
              // sits above it in the same cell, and unmounting the panel would
              // throw away its scroll position and re-fetch nothing. Keeping it
              // mounted but hidden is also what makes the "restart required"
              // warning reappear unchanged when the form closes.
              editingProvider !== undefined ? 'hidden' : ''
            }`}
          >
            <div className="px-4 py-2.5 border-b border-[var(--paper-border)] bg-[var(--paper-subsurface)] type-group">
              {t.doctor.runtimeConfig}
            </div>

          {/* First run: no provider declared means a submit would silently run as
              a zero-token rehearsal. Saying so here — before the wizard is ever
              opened — is the difference between a working first session and a
              mock deliverable the operator did not ask for. */}
          {runtime && (runtime.providers ?? []).filter((p) => p.declared).length === 0 ? (
            <div className="p-3.5 border-b border-[var(--paper-border)] bg-[#b45309]/10">
              <div className="type-value font-semibold text-[#b45309] mb-1">
                {t.doctor.firstRunTitle}
              </div>
              <p className="type-help text-[var(--ink-secondary)] leading-relaxed mb-2">
                {t.doctor.firstRunBody}
              </p>
              <Button size="sm" onClick={() => setEditingProvider(null)}>
                {t.doctor.firstRunCta}
              </Button>
            </div>
          ) : null}

          {/* A deployment-bound field edited in the file is not in force. Reporting
              it is the difference between a control that works and one that
              silently does nothing until the next restart. */}
          {(runtime?.frozen_fields_changed ?? []).length > 0 ? (
            <div className="px-3.5 py-2 border-b border-[var(--paper-border)] bg-[#b45309]/10 type-help text-[#b45309]">
              {t.doctor.frozenFieldsWarning.replace(
                '{fields}',
                (runtime?.frozen_fields_changed ?? []).join(', '),
              )}
            </div>
          ) : null}

          {/* Active configuration: the part the old screen never showed. */}
          {runtime ? (
            <div className="p-3.5 border-b border-[var(--paper-border)] bg-[var(--paper-subsurface)]">
              <div className="type-group mb-2">{t.doctor.activeNow}</div>
              <dl className="space-y-1.5 type-value font-mono">
                <div className="flex flex-col sm:flex-row sm:justify-between sm:items-baseline gap-0.5 sm:gap-3">
                  <dt className="text-[var(--ink-secondary)] shrink-0">{t.doctor.providerLabel}</dt>
                  <dd className="text-[var(--ink-primary)] font-semibold truncate sm:text-right">
                    {runtime.active.provider || '—'}
                  </dd>
                </div>
                <div className="flex flex-col sm:flex-row sm:justify-between sm:items-baseline gap-0.5 sm:gap-3">
                  <dt className="text-[var(--ink-secondary)] shrink-0">{t.doctor.draftModel}</dt>
                  <dd className="text-[var(--ink-primary)] font-semibold truncate sm:text-right">
                    {runtime.active.draft_model || t.doctor.notConfigured}
                  </dd>
                </div>
                <div className="flex flex-col sm:flex-row sm:justify-between sm:items-baseline gap-0.5 sm:gap-3">
                  <dt className="text-[var(--ink-secondary)] shrink-0">{t.doctor.repairModel}</dt>
                  <dd className="text-[var(--ink-primary)] truncate sm:text-right">
                    {runtime.active.repair_model || t.doctor.notConfigured}
                  </dd>
                </div>
                <div className="flex flex-col sm:flex-row sm:justify-between sm:items-baseline gap-0.5 sm:gap-3">
                  <dt className="text-[var(--ink-secondary)] shrink-0">{t.doctor.endpoint}</dt>
                  <dd className="text-[var(--ink-primary)] truncate sm:text-right" title={runtime.active.base_url}>
                    {runtime.active.base_url || '—'}
                  </dd>
                </div>
                <div className="flex flex-col sm:flex-row sm:justify-between sm:items-baseline gap-0.5 sm:gap-3">
                  <dt className="text-[var(--ink-secondary)] shrink-0">{t.doctor.capabilityProfile}</dt>
                  <dd
                    className={`truncate sm:text-right ${
                      runtime.active.matched_pattern
                        ? 'text-[var(--ink-primary)]'
                        : 'text-[#b45309] font-semibold'
                    }`}
                  >
                    {runtime.active.matched_pattern ?? t.doctor.noProfileMatch}
                  </dd>
                </div>
              </dl>

              {/* Which flags the block states, and which are the safe default.
                  A screen that showed all four as plain booleans would present
                  an assumption as if it were a measurement of the endpoint. */}
              <div className="mt-2 pt-2 border-t border-[var(--paper-border)]">
                <div className="flex items-center gap-1.5 mb-1">
                  <span className="type-label text-[var(--ink-muted)]">
                    {t.doctor.capabilityTitle}
                  </span>
                  <InfoTip label={t.doctor.capabilityHelp}>{t.doctor.capabilityHelp}</InfoTip>
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {(
                    // Only the two the operator can actually declare. The
                      // temperature and reasoning flags are learned by the engine
                      // and have no form control, so showing them here would
                      // present an engine assumption as a configuration choice.
                    [
                      ['supports_system_prompt', t.doctor.capabilitySystem],
                      ['supports_vision', t.doctor.capabilityVision],
                    ] as const
                  ).map(([field, label]) => {
                    const declared = (runtime.active.declared_capability ?? []).includes(field)
                    const value = runtime.active[field]
                    return (
                      <span
                        key={field}
                        className={`type-help font-mono px-1.5 py-0.5 rounded border ${
                          declared
                            ? value
                              ? 'border-[#15803d]/25 bg-[#15803d]/10 text-[#15803d]'
                              : 'border-[var(--paper-border)] bg-[var(--paper-surface)] text-[var(--ink-secondary)] line-through'
                            : 'border-dashed border-[var(--paper-border)] text-[var(--ink-muted)]'
                        }`}
                        title={declared ? undefined : t.doctor.capabilityUndeclared}
                      >
                        {label}
                      </span>
                    )
                  })}
                </div>
                {(runtime.active.declared_capability ?? []).length === 0 &&
                  runtime.active.draft_model && (
                    <p className="mt-1.5 type-help text-[var(--ink-muted)] leading-relaxed">
                      {t.doctor.noProfileMatchHelp}
                    </p>
                  )}
              </div>
            </div>
          ) : (
            <div className="p-4 type-help text-[var(--ink-muted)]">
              {t.doctor.runtimeConfigUnavailable}
            </div>
          )}

          {/* Declared providers, credentials filtered server-side, each row
              editable through the form above. */}
          <div className="px-3.5 py-2.5 border-b border-[var(--paper-border)]">
            <div className="flex items-center justify-between gap-2 mb-2">
              <span className="type-group">{t.doctor.declaredProviders}</span>
              {/* The create affordance. Without it a fresh install has zero
                  declared providers, so the list renders zero rows and there is
                  no way to add the first one from the console at all. */}
              <Button
                size="sm"
                variant="ghost"
                onClick={() => setEditingProvider(null)}
                className="!h-7 sm:!h-6 px-2 sm:!px-1.5 type-help"
              >
                <Plus size={12} className="sm:w-[11px] sm:h-[11px]" /> {t.doctor.addProvider}
              </Button>
            </div>
            {(runtime?.providers ?? []).filter((p) => p.declared).length === 0 ? (
              <p className="type-help text-[var(--ink-muted)] leading-relaxed">
                {t.doctor.firstRunBody}
              </p>
            ) : null}
            {/* Only what the operator declared. The API also returns shipped
                wire protocols and bare aliases (`chat`, `claude`, `responses`);
                rendering all three put seven aliases above the six real
                entries. Aliases are folded away below. */}
            <div className="space-y-1.5 sm:space-y-1">
              {(runtime?.providers ?? [])
                .filter((p) => p.declared)
                .map((p) => (
                  <div
                    key={p.name}
                    className={`flex flex-col sm:flex-row sm:items-center justify-between gap-1.5 sm:gap-2 rounded-[4px] px-2.5 py-2 sm:px-2 sm:py-1 type-value font-mono ${
                      editingProvider === p.name
                        ? // The row the form above is editing. Without this the
                          // list highlighted only the *active* provider, so
                          // editing index-translate left local-gemma4 green and
                          // the operator had no confirmation of what was open.
                          'bg-[var(--paper-subsurface)] border border-dashed border-[var(--ink-muted)]'
                        : p.active
                          ? 'bg-[#15803d]/10 border border-[#15803d]/25'
                          : 'border border-[var(--paper-border)] sm:border-transparent'
                    }`}
                  >
                    <div className="flex items-center justify-between sm:justify-start gap-2 min-w-0">
                      <span
                        className={`truncate ${
                          editingProvider === p.name
                            ? 'font-semibold text-[var(--ink-primary)]'
                            : p.active
                              ? 'font-semibold text-[#15803d]'
                              : 'text-[var(--ink-secondary)]'
                        }`}
                      >
                        {p.name}
                      </span>
                      <span className="type-help text-[var(--ink-muted)] truncate sm:hidden">
                        {p.draft_model || p.api_mode || '—'}
                      </span>
                    </div>
                    <span className="flex items-center justify-between sm:justify-end gap-2 shrink-0">
                      <span className="flex items-center gap-1.5 flex-wrap">
                        {/* Why this row cannot make a call, when it cannot. The
                            block is still listed — it is declared, so the operator
                            should see it — but silently, as it used to be, is how
                            an unset ${VAR} stayed invisible. The variable name is
                            not in this payload: it is shown in the form, which is
                            where the fix happens. */}
                        {p.credential_issue === 'unresolved_variable' ? (
                          <span className="type-help text-[#b45309]">
                            {t.doctor.credentialUnresolved}
                          </span>
                        ) : p.credential_issue === 'missing' ? (
                          <span className="type-help text-[#b45309]">
                            {t.doctor.credentialMissing}
                          </span>
                        ) : null}
                        {p.is_free && (
                          <span className="type-help text-[var(--ink-muted)]">
                            {t.doctor.freeLabel}
                          </span>
                        )}
                        {/* Which block the *file* names as default. Distinct from
                            "active": UBT_PROVIDER or --provider outranks it, so an
                            operator whose shell exported a provider had no way to
                            see what the file would pick without it. */}
                        {p.is_default ? (
                          <span className="type-help font-semibold text-[var(--ink-primary)]">
                            {t.doctor.defaultProviderLabel}
                          </span>
                        ) : (
                          <button
                            type="button"
                            onClick={() => void handleMakeDefault(p.name)}
                            disabled={defaultBusy === p.name}
                            aria-label={`${t.doctor.activateProvider} ${p.name}`}
                            className="type-help text-[var(--ink-muted)] hover:text-[var(--ink-primary)] disabled:opacity-50 py-2 sm:py-0 px-1 min-h-[40px] sm:min-h-0 flex items-center cursor-pointer"
                          >
                            {t.doctor.activateProvider}
                          </button>
                        )}
                        <span className="hidden sm:inline type-help text-[var(--ink-muted)] truncate max-w-[9rem]">
                          {p.draft_model || p.api_mode || '—'}
                        </span>
                      </span>
                      <button
                        type="button"
                        onClick={() => setEditingProvider(p.name)}
                        aria-label={`${t.doctor.editProvider} ${p.name}`}
                        className="p-2 sm:p-0.5 rounded text-[var(--ink-muted)] hover:text-[var(--ink-primary)] hover:bg-[var(--paper-subsurface)] shrink-0 min-h-[40px] min-w-[40px] sm:min-h-0 sm:min-w-0 flex items-center justify-center cursor-pointer"
                      >
                        <Pencil size={13} className="sm:w-[11px] sm:h-[11px]" />
                      </button>
                    </span>
                  </div>
                ))}
            </div>

            {/* Built-in protocols and aliases, collapsed: they are usable names,
                not configurations. */}
            <details className="mt-2">
              <summary className="cursor-pointer type-help font-mono text-[var(--ink-muted)] hover:text-[var(--ink-primary)] select-none">
                {t.doctor.builtinProviders} (
                {/* ``runtime === null`` means the configuration could not be read,
                    which is NOT the same as "there are none". The old code
                    coalesced to ``[]`` and rendered "(0)" — the panel asserting a
                    fact it did not have, which is the same failure the "latency
                    matrix" heading had. */}
                {runtime
                  ? (runtime.providers ?? []).filter((p) => !p.declared).length
                  : t.doctor.loadFailed})
              </summary>
              {/* Two kinds of entry, rendered as two groups. They used to be one
                  flat list where both were labelled "alias": ``openai-chat`` is a
                  wire *protocol* — the name a block should carry — while ``chat``
                  and ``claude`` merely resolve to one. Labelling a protocol an
                  alias, and printing ``anthropic-messages`` next to itself, told
                  the operator nothing about which name to write in a block. */}
              <div className="mt-1 space-y-0.5">
                {(runtime?.providers ?? [])
                  .filter((p) => !p.declared && p.is_protocol)
                  .map((p) => (
                    <div
                      key={p.name}
                      className="flex items-center justify-between gap-2 px-2 type-help font-mono text-[var(--ink-muted)]"
                    >
                      <span className="truncate">{p.name}</span>
                      <span className="shrink-0">{t.doctor.protocolLabel}</span>
                    </div>
                  ))}
                {(runtime?.providers ?? []).filter((p) => !p.declared && p.is_alias).length >
                0 ? (
                  <div className="pt-1 px-2 type-help text-[var(--ink-muted)]">
                    {t.doctor.aliasGroupLabel}
                  </div>
                ) : null}
                {(runtime?.providers ?? [])
                  .filter((p) => !p.declared && p.is_alias)
                  .map((p) => (
                    <div
                      key={p.name}
                      className="flex items-center justify-between gap-2 px-2 type-help font-mono text-[var(--ink-muted)]"
                    >
                      <span className="truncate">{p.name}</span>
                      {/* The mapping itself: ``claude → anthropic-messages``. */}
                      <span className="shrink-0">
                        {p.alias_of ? `\u2192 ${p.alias_of}` : t.doctor.aliasLabel}
                      </span>
                    </div>
                  ))}
              </div>
              {/* A protocol name is usable on its own, but carries no model id —
                  which is the actual prerequisite a newcomer trips over. */}
              <p className="mt-1.5 px-2 type-help text-[var(--ink-muted)] leading-relaxed">
                {t.doctor.builtinProvidersHelp}
              </p>
            </details>
            <p className="mt-2 type-help text-[var(--ink-muted)] leading-relaxed">
              {t.doctor.providerConfigHint}
            </p>
          </div>
          </div>
        </div>
      </div>

      {/* Quality estimation, full width: the provider panel above is about what
          the engine calls, this one is about whether the result is measured.
          Kept out of the two-column grid because it is a form with a warning
          paragraph, and squeezing it into a half-width cell is what made the
          previous "QE not measured" label unreadable. */}
      <QESettingsCard
        providerNames={(runtime?.providers ?? []).filter((p) => p.declared).map((p) => p.name)}
      />

      {/* Security Boundary Panel */}
      <div className="p-4 rounded-lg border border-[var(--paper-border)] bg-[var(--paper-surface)] space-y-1.5 text-xs font-mono text-[var(--ink-secondary)] shadow-2xs break-words">
        <div className="text-xs font-mono uppercase tracking-wider text-[var(--ink-muted)] mb-2 font-semibold">
          {t.doctor.securityBoundary}
        </div>
        {info ? (
          <>
            <div className="flex flex-wrap items-center gap-2">
              <span className={info.is_loopback ? 'text-[#15803d]' : 'text-[#b45309]'}>
                • {t.doctor.listenInterface}: {info.host}
              </span>
              <Badge variant={info.is_loopback ? 'success' : 'warning'} className="text-xs">
                {info.is_loopback ? t.doctor.loopbackOnly : t.doctor.exposed}
              </Badge>
            </div>
            <div className="break-words">• {t.doctor.authGate}: {info.auth_enabled ? t.doctor.enabled : t.doctor.disabled}</div>
            <div className="break-words">• {t.doctor.jobMode}: {info.job_mode}</div>
            <div className="break-words">• {t.doctor.dbDir}: {info.db_dir}</div>
            {info.disk_free_gb != null && (
              <div>• {t.doctor.diskFree}: {info.disk_free_gb} GB</div>
            )}
            {info.wal_status && (
              <div className="break-words">• {t.doctor.walStatus}: {info.wal_status}</div>
            )}
            <div className="break-words">
              • {t.doctor.allowedRoots}:{' '}
              {info.allowed_bases.length === 0 ? '—' : info.allowed_bases.join(', ')}
            </div>
            <div className="break-words">• {t.doctor.forbiddenPaths}: /etc, /root, /var/run, ~/.ssh (Hard refusal)</div>
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
