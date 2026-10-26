import { useCallback, useEffect, useRef, useState } from 'react'
import {
  AlertTriangle,
  Check,
  Copy,
  Loader2,
  Pencil,
  Plus,
  RefreshCw,
  Trash2,
} from 'lucide-react'

import {
  deleteProvider as apiDeleteProvider,
  getDoctor,
  getRuntimeConfig,
  getSystemBoundary,
  getSystemInfo,
  setDefaultProvider,
  type DoctorCheck,
  type RuntimeConfig,
  type SystemInfo,
} from '@/api/client'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/Alert'
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '@/components/ui/accordion'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { ButtonGroup } from '@/components/ui/button-group'
import { Card, CardContent } from '@/components/ui/card'
import { ConfirmDialog } from '@/components/ui/ConfirmDialog'
import { Empty, EmptyDescription, EmptyHeader, EmptyMedia, EmptyTitle } from '@/components/ui/empty'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { useToast } from '@/components/ui/useToast'
import { localizeDoctorCheck } from '@/i18n/doctorText'
import { doctorCodesEn, doctorCodesZh } from '@/i18n/translations/doctorCodes'
import { useI18n } from '@/i18n/useI18n'
import { cn } from '@/lib/utils'
import { Donut } from '@/views/quality-gate/charts'
import { EgressSettingsCard } from './EgressSettingsCard'
import { ProviderEditor } from './ProviderEditor'
import { QESettingsCard } from './QESettingsCard'
import {
  DoctorMetric,
  DoctorPanel,
  DoctorStatusBadge,
  DoctorStatusIcon,
} from './components'
import { toneForStatus } from './status'

type CheckFilter = 'issues' | 'all'

function providerStatusTone(provider: RuntimeConfig['providers'][number]) {
  if (!provider.available || provider.credential_issue) return 'warn' as const
  return 'ok' as const
}

function localizedStatus(
  status: string,
  t: ReturnType<typeof useI18n>['t'],
): string {
  return t.doctor.checkStatuses[status] ?? status
}

export function SystemDoctorView() {
  const { t, language } = useI18n()
  const toast = useToast()
  const [checks, setChecks] = useState<DoctorCheck[]>([])
  const [runtime, setRuntime] = useState<RuntimeConfig | null>(null)
  const [info, setInfo] = useState<SystemInfo | null>(null)
  const [boundary, setBoundary] = useState<{ forbidden_prefixes: string[] } | null>(null)
  const [editingProvider, setEditingProvider] = useState<string | null | undefined>(undefined)
  const [deletingProvider, setDeletingProvider] = useState<string | null>(null)
  const [deleteBusy, setDeleteBusy] = useState(false)
  const [deleteError, setDeleteError] = useState<string | null>(null)
  const [isPinging, setIsPinging] = useState(false)
  const [defaultBusy, setDefaultBusy] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [checkFilter, setCheckFilter] = useState<CheckFilter>('issues')
  const [expandedDetails, setExpandedDetails] = useState<Set<string>>(new Set())
  const [highlightedRow, setHighlightedRow] = useState<string | null>(null)
  const [copiedFix, setCopiedFix] = useState<string | null>(null)
  const editorRef = useRef<HTMLDivElement>(null)

  const load = useCallback(async (probe: boolean) => {
    setLoading(true)
    try {
      const [doctor, systemInfo, runtimeCfg, boundaryInfo] = await Promise.all([
        getDoctor(probe),
        getSystemInfo().catch(() => null),
        getRuntimeConfig().catch(() => null),
        getSystemBoundary().catch(() => null),
      ])
      setChecks(doctor.checks)
      setInfo(systemInfo)
      setRuntime(runtimeCfg)
      setBoundary(boundaryInfo)
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : t.doctor.refreshFailed)
    } finally {
      setLoading(false)
      setIsPinging(false)
    }
  }, [t.doctor.refreshFailed])

  useEffect(() => {
    // oxlint-disable-next-line react/set-state-in-effect
    void load(false)
  }, [load])

  useEffect(() => {
    if (editingProvider !== undefined && editorRef.current) {
      editorRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
  }, [editingProvider])

  const handleMakeDefault = useCallback(
    async (name: string) => {
      setDefaultBusy(name)
      setError(null)
      try {
        await setDefaultProvider(name)
        await load(false)
      } catch (err) {
        setError(err instanceof Error ? err.message : t.doctor.refreshFailed)
      } finally {
        setDefaultBusy(null)
      }
    },
    [load, t.doctor.refreshFailed],
  )

  const handleDeleteProvider = useCallback(
    async (name: string) => {
      setDeleteBusy(true)
      setDeleteError(null)
      try {
        await apiDeleteProvider(name)
        setDeletingProvider(null)
        toast.push(t.doctor.providerDeleted, 'success')
        await load(false)
      } catch (err) {
        setDeleteError(err instanceof Error ? err.message : String(err))
      } finally {
        setDeleteBusy(false)
      }
    },
    [load, t.doctor.providerDeleted, toast],
  )

  const handleRefresh = () => {
    setIsPinging(true)
    void load(true)
  }

  const toggleDetail = (rowId: string) => {
    setExpandedDetails((prev) => {
      const next = new Set(prev)
      if (next.has(rowId)) next.delete(rowId)
      else next.add(rowId)
      return next
    })
  }

  const jumpToCheck = (rowId: string) => {
    setCheckFilter('all')
    setExpandedDetails((prev) => new Set(prev).add(rowId))
    window.setTimeout(() => {
      document.getElementById(rowId)?.scrollIntoView({ behavior: 'smooth', block: 'center' })
      setHighlightedRow(rowId)
    })
    window.setTimeout(() => {
      setHighlightedRow((cur) => (cur === rowId ? null : cur))
    }, 2000)
  }

  const copyFix = async (rowId: string, fix: string) => {
    try {
      await navigator.clipboard.writeText(fix)
      setCopiedFix(rowId)
      window.setTimeout(() => {
        setCopiedFix((cur) => (cur === rowId ? null : cur))
      }, 1500)
    } catch {
      // The command remains selectable when the clipboard API is unavailable.
    }
  }

  const codeTables = language === 'zh' ? doctorCodesZh : doctorCodesEn
  const localized = checks.map((check) => ({
    check,
    text: localizeDoctorCheck(check, codeTables),
  }))
  const indexed = localized.map((item, index) => ({ item, index, rowId: `dr-${index}` }))
  const problems = indexed.filter(({ item }) => item.check.status !== 'OK')
  const okCount = checks.length - problems.length
  const hasFail = checks.some((check) => check.status === 'FAIL')
  const visible = checkFilter === 'issues' ? problems : indexed
  const groups = Array.from(new Set(visible.map(({ item }) => item.check.group_code)))
  const declaredProviders = (runtime?.providers ?? []).filter((provider) => provider.declared)
  const active = runtime?.active ?? null
  const canTranslate = Boolean(active?.can_translate)
  const statusLabel = (status: string) => localizedStatus(status, t)

  return (
    <div className="mx-auto w-full max-w-6xl flex-1 space-y-5 px-4 py-6 sm:px-7">
      <header className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
        <div className="space-y-1.5">
          <h1 className="text-2xl font-semibold tracking-[-0.03em] text-foreground">
            {t.doctor.title}
          </h1>
          <p className="max-w-2xl text-sm leading-relaxed text-muted-foreground">
            {t.doctor.subtitle}
          </p>
        </div>
        <Button
          onClick={handleRefresh}
          variant="outline"
          size="sm"
          disabled={isPinging || loading}
          className="self-start bg-card sm:self-auto"
        >
          <RefreshCw className={cn('size-3.5', (isPinging || loading) && 'animate-spin')} />
          {t.doctor.pingAll}
        </Button>
      </header>

      {error ? (
        <Alert variant="warning">
          <AlertTriangle />
          <AlertTitle>{t.doctor.refreshFailed}</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      ) : null}

      <Card className="overflow-hidden">
        <CardContent className="p-0">
          <div className="flex flex-col gap-5 p-5 sm:flex-row sm:items-center sm:p-6">
            {checks.length > 0 ? (
              <Donut
                value={checks.length > 0 ? okCount / checks.length : 0}
                tone={hasFail ? 'fail' : problems.length > 0 ? 'warn' : 'pass'}
                center={`${okCount}/${checks.length}`}
                caption={t.doctor.verdictCaption}
              />
            ) : (
              <div className="grid size-[76px] shrink-0 place-items-center rounded-full border-[7px] border-border text-sm text-muted-foreground">
                {loading ? <Loader2 className="size-4 animate-spin" /> : '—'}
              </div>
            )}
            <div className="min-w-0 flex-1 space-y-2">
              <div className="flex flex-wrap items-center gap-2">
                <h2 className="text-base font-semibold tracking-[-0.02em]">
                  {checks.length === 0
                    ? t.doctor.emptyChecksTitle
                    : hasFail
                      ? t.doctor.overviewBlocked
                      : problems.length > 0
                        ? t.doctor.overviewNeedsAttention.replace('{n}', String(problems.length))
                        : t.doctor.overviewReady}
                </h2>
                {active ? (
                  <Badge
                    variant="outline"
                    className={cn(
                      canTranslate
                        ? 'border-emerald-600/25 bg-emerald-500/10 text-emerald-700 dark:text-emerald-400'
                        : 'border-amber-600/25 bg-amber-500/10 text-amber-700 dark:text-amber-400'
                    )}
                  >
                    {canTranslate ? t.doctor.runtimeRunnable : t.doctor.runtimeRehearsal}
                  </Badge>
                ) : null}
              </div>
              <p className="text-xs leading-relaxed text-muted-foreground">
                {active
                  ? t.doctor.overviewSummary
                      .replace('{model}', active.draft_model || t.doctor.notConfigured)
                      .replace('{ok}', String(okCount))
                      .replace('{total}', String(checks.length))
                  : checks.length === 0
                    ? t.doctor.emptyChecksBody
                    : t.doctor.runtimeConfigUnavailable}
              </p>
              {problems.length > 0 ? (
                <div className="flex flex-wrap gap-1.5 pt-0.5">
                  {problems.map(({ item, rowId }) => (
                    <button
                      key={rowId}
                      type="button"
                      onClick={() => jumpToCheck(rowId)}
                      className={cn(
                        'inline-flex min-h-7 items-center gap-1.5 rounded-md border px-2 py-0.5 text-left text-xs transition-colors hover:bg-muted/50',
                        item.check.status === 'FAIL'
                          ? 'border-destructive/25 bg-destructive/5 text-destructive'
                          : 'border-amber-600/25 bg-amber-500/5 text-amber-700 dark:text-amber-400'
                      )}
                    >
                      <span className="font-medium">{statusLabel(item.check.status)}</span>
                      <span className="truncate">{item.text.name}</span>
                    </button>
                  ))}
                </div>
              ) : null}
            </div>
          </div>
        </CardContent>
      </Card>

      <Tabs defaultValue="diagnostics" className="space-y-4">
        <TabsList className="h-auto w-full justify-start gap-1 bg-transparent p-0 sm:w-auto">
          <TabsTrigger
            value="diagnostics"
            className="h-8 rounded-md px-3 data-[state=active]:bg-card data-[state=active]:shadow-card"
          >
            {t.doctor.tabDiagnostics}
          </TabsTrigger>
          <TabsTrigger
            value="models"
            className="h-8 rounded-md px-3 data-[state=active]:bg-card data-[state=active]:shadow-card"
          >
            {t.doctor.tabModels}
          </TabsTrigger>
          <TabsTrigger
            value="security"
            className="h-8 rounded-md px-3 data-[state=active]:bg-card data-[state=active]:shadow-card"
          >
            {t.doctor.tabSecurity}
          </TabsTrigger>
        </TabsList>

        <TabsContent value="diagnostics" className="mt-0 space-y-4">
          <DoctorPanel
            title={t.doctor.diagnosticsTitle}
            description={t.doctor.diagnosticsHelp}
            action={
              <ButtonGroup className="w-full sm:w-auto">
                <Button
                  type="button"
                  variant={checkFilter === 'issues' ? 'secondary' : 'ghost'}
                  size="sm"
                  onClick={() => setCheckFilter('issues')}
                  className="flex-1 sm:flex-none"
                >
                  {t.doctor.issuesOnly}
                  <Badge variant="outline" className="h-4 min-w-4 px-1 text-[10px]">
                    {problems.length}
                  </Badge>
                </Button>
                <Button
                  type="button"
                  variant={checkFilter === 'all' ? 'secondary' : 'ghost'}
                  size="sm"
                  onClick={() => setCheckFilter('all')}
                  className="flex-1 sm:flex-none"
                >
                  {t.doctor.allChecks}
                  <Badge variant="outline" className="h-4 min-w-4 px-1 text-[10px]">
                    {checks.length}
                  </Badge>
                </Button>
              </ButtonGroup>
            }
          >
            {loading && checks.length === 0 ? (
              <div className="flex items-center justify-center gap-2 px-4 py-12 text-sm text-muted-foreground">
                <Loader2 className="size-4 animate-spin" />
                {t.doctor.loadingChecks}
              </div>
            ) : visible.length === 0 ? (
              <Empty className="border-0 py-10">
                <EmptyHeader>
                  <EmptyMedia variant="icon">
                    <Check />
                  </EmptyMedia>
                  <EmptyTitle>{t.doctor.overviewReady}</EmptyTitle>
                  <EmptyDescription>
                    {checks.length === 0 ? t.doctor.emptyChecksBody : t.doctor.diagnosticsHelp}
                  </EmptyDescription>
                </EmptyHeader>
              </Empty>
            ) : (
              groups.map((groupCode) => {
                const rows = visible.filter(({ item }) => item.check.group_code === groupCode)
                return (
                  <section key={groupCode} aria-labelledby={`group-${groupCode}`}>
                    <h3
                      id={`group-${groupCode}`}
                      className="border-b border-border/60 bg-muted/20 px-4 py-2 text-[11px] font-semibold uppercase tracking-[0.08em] text-muted-foreground"
                    >
                      {rows[0]?.item.text.group ?? groupCode}
                    </h3>
                    <div className="divide-y divide-border/70">
                      {rows.map(({ item, rowId }) => {
                        const detailLong = item.text.detail.length > 100
                        const expanded = expandedDetails.has(rowId)
                        const tone = toneForStatus(item.check.status)
                        return (
                          <div
                            key={rowId}
                            id={rowId}
                            className={cn(
                              'px-4 py-3 transition-colors',
                              highlightedRow === rowId && 'bg-amber-500/10'
                            )}
                          >
                            <div className="flex items-start justify-between gap-3">
                              <div className="flex min-w-0 flex-1 items-start gap-2.5">
                                <DoctorStatusIcon
                                  status={item.check.status}
                                  className="mt-0.5"
                                />
                                <div className="min-w-0 flex-1">
                                  <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
                                    <h4 className="text-sm font-medium text-foreground">
                                      {item.text.name}
                                    </h4>
                                    <DoctorStatusBadge
                                      status={item.check.status}
                                      label={statusLabel(item.check.status)}
                                      className="h-5 px-1.5 py-0 text-[10px]"
                                    />
                                  </div>
                                  <p
                                    className={cn(
                                      'mt-1 break-words font-mono text-[11px] leading-relaxed text-muted-foreground',
                                      detailLong && !expanded && 'line-clamp-2'
                                    )}
                                  >
                                    {item.text.detail}
                                  </p>
                                  {detailLong ? (
                                    <Button
                                      type="button"
                                      variant="link"
                                      size="sm"
                                      onClick={() => toggleDetail(rowId)}
                                      className="mt-0.5 h-auto px-0 text-[11px] text-muted-foreground"
                                    >
                                      {expanded ? t.doctor.collapseDetail : t.doctor.expandDetail}
                                    </Button>
                                  ) : null}
                                  {item.text.fix ? (
                                    <div className="mt-2 flex flex-wrap items-start gap-2 rounded-md border border-border/70 bg-muted/25 px-2.5 py-2">
                                      <code className="min-w-0 flex-1 break-all font-mono text-[11px] leading-relaxed text-foreground">
                                        {item.text.fix}
                                      </code>
                                      <Button
                                        type="button"
                                        variant="ghost"
                                        size="sm"
                                        onClick={() => void copyFix(rowId, item.text.fix ?? '')}
                                        className="h-7 shrink-0 px-2 text-[11px]"
                                      >
                                        {copiedFix === rowId ? (
                                          <Check className="text-emerald-700 dark:text-emerald-400" />
                                        ) : (
                                          <Copy />
                                        )}
                                        {copiedFix === rowId
                                          ? t.doctor.fixCopied
                                          : t.doctor.fixLabel}
                                      </Button>
                                    </div>
                                  ) : null}
                                </div>
                              </div>
                              {tone === 'ok' && item.check.status === 'OK' ? (
                                <span className="hidden text-[11px] text-muted-foreground sm:block">
                                  {statusLabel(item.check.status)}
                                </span>
                              ) : null}
                            </div>
                          </div>
                        )
                      })}
                    </div>
                  </section>
                )
              })
            )}
          </DoctorPanel>
        </TabsContent>

        <TabsContent value="models" className="mt-0">
          {editingProvider !== undefined ? (
            <div ref={editorRef}>
              <ProviderEditor
                name={editingProvider}
                canWriteCredential={runtime ? (runtime.can_write_credential ?? false) : null}
                onCancel={() => setEditingProvider(undefined)}
                onSaved={() => void load(false)}
              />
            </div>
          ) : null}

          <div className={cn('space-y-4', editingProvider !== undefined && 'hidden')}>
            <DoctorPanel
              title={t.doctor.runtimeCardTitle}
              description={t.doctor.runtimeCardHelp}
              action={
                active ? (
                  <Badge
                    variant="outline"
                    className={cn(
                      canTranslate
                        ? 'border-emerald-600/25 bg-emerald-500/10 text-emerald-700 dark:text-emerald-400'
                        : 'border-amber-600/25 bg-amber-500/10 text-amber-700 dark:text-amber-400'
                    )}
                  >
                    {canTranslate ? t.doctor.runtimeRunnable : t.doctor.runtimeRehearsal}
                  </Badge>
                ) : undefined
              }
            >
              {active ? (
                <div className="space-y-4 p-4">
                  <div className="grid grid-cols-2 gap-2 lg:grid-cols-4">
                    <DoctorMetric
                      label={t.doctor.providerLabel}
                      value={active.provider || '—'}
                      mono
                    />
                    <DoctorMetric
                      label={t.doctor.draftModel}
                      value={active.draft_model || t.doctor.notConfigured}
                      tone={active.draft_model ? undefined : 'warn'}
                      mono
                    />
                    <DoctorMetric
                      label={t.doctor.repairModel}
                      value={active.repair_model || t.doctor.notConfigured}
                      mono
                    />
                    <DoctorMetric
                      label={t.doctor.endpoint}
                      value={active.base_url || '—'}
                      mono
                    />
                  </div>
                  <div className="flex flex-wrap items-center gap-2 border-t border-border/70 pt-3">
                    <span className="text-[11px] font-medium text-muted-foreground">
                      {t.doctor.capabilityTitle}
                    </span>
                    {(
                      [
                        ['supports_system_prompt', t.doctor.capabilitySystem],
                        ['supports_vision', t.doctor.capabilityVision],
                      ] as const
                    ).map(([field, label]) => {
                      const declared = (active.declared_capability ?? []).includes(field)
                      const value = active[field]
                      return (
                        <Badge
                          key={field}
                          variant="outline"
                          title={declared ? undefined : t.doctor.capabilityUndeclared}
                          className={cn(
                            'font-normal',
                            declared
                              ? value
                                ? 'border-emerald-600/20 bg-emerald-500/10 text-emerald-700 dark:text-emerald-400'
                                : 'text-muted-foreground line-through'
                              : 'border-dashed text-muted-foreground'
                          )}
                        >
                          {label}
                        </Badge>
                      )
                    })}
                    {(active.declared_capability ?? []).length === 0 && active.draft_model ? (
                      <span className="text-[11px] text-muted-foreground">
                        {t.doctor.noProfileMatchHelp}
                      </span>
                    ) : null}
                  </div>
                </div>
              ) : (
                <div className="p-4 text-sm text-muted-foreground">
                  {t.doctor.runtimeConfigUnavailable}
                </div>
              )}
            </DoctorPanel>

            {runtime && declaredProviders.length === 0 ? (
              <Alert variant="warning">
                <AlertTriangle />
                <AlertTitle>{t.doctor.firstRunTitle}</AlertTitle>
                <AlertDescription className="flex flex-col items-start gap-3 pt-2">
                  <span>{t.doctor.firstRunBody}</span>
                  <Button size="sm" onClick={() => setEditingProvider(null)}>
                    <Plus />
                    {t.doctor.firstRunCta}
                  </Button>
                </AlertDescription>
              </Alert>
            ) : null}

            {(runtime?.frozen_fields_changed ?? []).length > 0 ? (
              <Alert variant="warning">
                <AlertTriangle />
                <AlertDescription>
                  {t.doctor.frozenFieldsWarning.replace(
                    '{fields}',
                    (runtime?.frozen_fields_changed ?? []).join(', '),
                  )}
                </AlertDescription>
              </Alert>
            ) : null}

            <DoctorPanel
              title={t.doctor.providerListTitle}
              description={t.doctor.providerListHelp}
              action={
                <Button size="sm" variant="outline" onClick={() => setEditingProvider(null)}>
                  <Plus />
                  {t.doctor.addProvider}
                </Button>
              }
            >
              {declaredProviders.length === 0 ? (
                <div className="px-4 py-8 text-center text-sm text-muted-foreground">
                  {t.doctor.firstRunBody}
                </div>
              ) : (
                <div className="divide-y divide-border/70">
                  {declaredProviders.map((provider) => {
                    const tone = providerStatusTone(provider)
                    const readiness = provider.credential_issue
                      ? provider.credential_issue === 'unresolved_variable'
                        ? t.doctor.providerUnresolved
                        : t.doctor.providerNoCredential
                      : provider.available
                        ? t.doctor.providerReady
                        : t.doctor.refreshFailed
                    return (
                      <div key={provider.name} className="space-y-3 px-4 py-3.5">
                        <div className="flex flex-wrap items-start justify-between gap-2">
                          <div className="min-w-0">
                            <div className="flex flex-wrap items-center gap-1.5">
                              <h3 className="font-mono text-sm font-semibold text-foreground">
                                {provider.name}
                              </h3>
                              {provider.active ? (
                                <Badge variant="outline" className="h-5 px-1.5 py-0 text-[10px]">
                                  {t.doctor.activeProviderLabel}
                                </Badge>
                              ) : provider.is_default ? (
                                <Badge variant="secondary" className="h-5 px-1.5 py-0 text-[10px]">
                                  {t.doctor.defaultProviderLabel}
                                </Badge>
                              ) : null}
                              {provider.is_free ? (
                                <Badge variant="outline" className="h-5 px-1.5 py-0 text-[10px]">
                                  {t.doctor.freeLabel}
                                </Badge>
                              ) : null}
                            </div>
                            <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] text-muted-foreground">
                              <span className="inline-flex items-center gap-1">
                                <DoctorStatusIcon
                                  status={tone === 'ok' ? 'OK' : 'WARN'}
                                  className="size-3.5"
                                />
                                {readiness}
                              </span>
                              <span className="truncate font-mono">
                                {provider.draft_model || t.doctor.notConfigured}
                              </span>
                              <span className="truncate font-mono">
                                {provider.api_mode || '—'}
                              </span>
                            </div>
                          </div>
                          <div className="flex flex-wrap items-center gap-1">
                            {!provider.active && !provider.is_default ? (
                              <Button
                                size="sm"
                                variant="ghost"
                                onClick={() => void handleMakeDefault(provider.name)}
                                disabled={defaultBusy === provider.name}
                                className="h-8 px-2 text-xs"
                              >
                                {t.doctor.activateProvider}
                              </Button>
                            ) : null}
                            <Button
                              size="sm"
                              variant="outline"
                              onClick={() => setEditingProvider(provider.name)}
                              className="h-8 px-2 text-xs"
                            >
                              <Pencil />
                              {t.doctor.editProvider}
                            </Button>
                            <Button
                              size="sm"
                              variant="ghost"
                              onClick={() => {
                                setDeleteError(null)
                                setDeletingProvider(provider.name)
                              }}
                              className="h-8 px-2 text-xs text-muted-foreground hover:text-destructive"
                            >
                              <Trash2 />
                              {t.doctor.deleteProvider}
                            </Button>
                          </div>
                        </div>
                        <div className="grid grid-cols-1 gap-2 rounded-md bg-muted/20 px-3 py-2 text-[11px] sm:grid-cols-2">
                          <div className="min-w-0">
                            <span className="text-muted-foreground">{t.doctor.providerModels}</span>
                            <span className="ml-2 truncate font-mono">
                              {provider.draft_model || '—'}
                              {provider.repair_model && provider.repair_model !== provider.draft_model
                                ? ` / ${provider.repair_model}`
                                : ''}
                            </span>
                          </div>
                          <div className="min-w-0 sm:text-right">
                            <span className="text-muted-foreground">{t.doctor.providerEndpoint}</span>
                            <span
                              className="ml-2 inline-block max-w-full truncate align-bottom font-mono"
                              title={provider.base_url ?? undefined}
                            >
                              {provider.base_url || '—'}
                            </span>
                          </div>
                        </div>
                      </div>
                    )
                  })}
                </div>
              )}
              <div className="border-t border-border/70 px-4 py-2.5 text-[11px] leading-relaxed text-muted-foreground">
                {t.doctor.providerConfigHint}
              </div>
            </DoctorPanel>

            <Accordion type="single" collapsible>
              <AccordionItem
                value="built-in-connections"
                className="rounded-xl border border-border/80 bg-card shadow-card"
              >
                <AccordionTrigger className="px-4 py-3 hover:no-underline">
                  <span className="flex min-w-0 flex-wrap items-center gap-2">
                    <span>
                      {t.doctor.builtinDetails}{' '}
                      <span className="font-normal tabular-nums text-muted-foreground">
                        (
                        {runtime
                          ? (runtime.providers ?? []).filter((provider) => !provider.declared)
                              .length
                          : t.doctor.loadFailed}
                        )
                      </span>
                    </span>
                    <Badge
                      variant="outline"
                      className="font-normal text-muted-foreground"
                    >
                      {t.doctor.builtinHelp}
                    </Badge>
                  </span>
                </AccordionTrigger>
                <AccordionContent className="space-y-3 border-t border-border/70 px-4 py-3">
                <div className="space-y-1">
                  {(runtime?.providers ?? [])
                    .filter((provider) => !provider.declared && provider.is_protocol)
                    .map((provider) => (
                      <div
                        key={provider.name}
                        className="flex items-center justify-between gap-3 text-xs"
                      >
                        <span className="font-mono">{provider.name}</span>
                        <span className="text-muted-foreground">{t.doctor.protocolName}</span>
                      </div>
                    ))}
                </div>
                {(runtime?.providers ?? []).some(
                  (provider) => !provider.declared && provider.is_alias,
                ) ? (
                  <div className="border-t border-border/60 pt-2">
                    <div className="mb-1 text-[11px] font-medium text-muted-foreground">
                      {t.doctor.aliasResolution}
                    </div>
                    {(runtime?.providers ?? [])
                      .filter((provider) => !provider.declared && provider.is_alias)
                      .map((provider) => (
                        <div
                          key={provider.name}
                          className="flex items-center justify-between gap-3 py-0.5 text-xs"
                        >
                          <span className="font-mono">{provider.name}</span>
                          <span className="font-mono text-muted-foreground">
                            → {provider.alias_of ?? t.doctor.aliasLabel}
                          </span>
                        </div>
                      ))}
                  </div>
                ) : null}
                <p className="text-[11px] leading-relaxed text-muted-foreground">
                  {t.doctor.builtinProvidersHelp}
                </p>
                </AccordionContent>
              </AccordionItem>
            </Accordion>
          </div>
        </TabsContent>

        <TabsContent value="security" className="mt-0 space-y-4">
          <DoctorPanel
            title={t.doctor.securityCardTitle}
            description={t.doctor.securityCardHelp}
          >
            <div className="grid grid-cols-1 gap-4 p-4 lg:grid-cols-2">
              <QESettingsCard providerNames={declaredProviders.map((provider) => provider.name)} />
              <EgressSettingsCard />
            </div>
          </DoctorPanel>

          <DoctorPanel
            title={t.doctor.securityBoundary}
            description={t.doctor.boundaryCardHelp}
          >
            {info ? (
              <>
                <div className="divide-y divide-border/70">
                  <div className="flex flex-col gap-1.5 px-4 py-3 text-xs sm:flex-row sm:items-center sm:justify-between">
                    <span className="text-muted-foreground">{t.doctor.listenInterface}</span>
                    <span className="flex min-w-0 items-center gap-2">
                      <span className="truncate font-mono" title={info.host}>
                        {info.host}
                        {info.port ? `:${info.port}` : ''}
                      </span>
                      <Badge
                        variant="outline"
                        className={cn(
                          'shrink-0',
                          info.is_loopback
                            ? 'border-emerald-600/25 bg-emerald-500/10 text-emerald-700 dark:text-emerald-400'
                            : 'border-amber-600/25 bg-amber-500/10 text-amber-700 dark:text-amber-400'
                        )}
                      >
                        {info.is_loopback ? t.doctor.loopbackOnly : t.doctor.exposed}
                      </Badge>
                    </span>
                  </div>
                  <div className="flex items-center justify-between gap-3 px-4 py-3 text-xs">
                    <span className="text-muted-foreground">{t.doctor.authGate}</span>
                    <Badge
                      variant="outline"
                      className={cn(
                        info.auth_enabled
                          ? 'border-emerald-600/25 bg-emerald-500/10 text-emerald-700 dark:text-emerald-400'
                          : 'border-amber-600/25 bg-amber-500/10 text-amber-700 dark:text-amber-400'
                      )}
                    >
                      {info.auth_enabled ? t.doctor.enabled : t.doctor.disabled}
                    </Badge>
                  </div>
                  <div className="flex items-center justify-between gap-3 px-4 py-3 text-xs">
                    <span className="text-muted-foreground">{t.doctor.jobMode}</span>
                    <span className="font-mono">{info.job_mode}</span>
                  </div>
                  <div className="flex flex-col gap-2 px-4 py-3 text-xs sm:flex-row sm:items-start sm:justify-between">
                    <span className="shrink-0 text-muted-foreground">
                      {t.doctor.allowedRoots}
                    </span>
                    <span className="flex min-w-0 flex-wrap gap-1 sm:justify-end">
                      {info.allowed_bases.length === 0 ? (
                        <span className="text-muted-foreground">—</span>
                      ) : (
                        info.allowed_bases.map((base) => (
                          <span
                            key={base}
                            title={base}
                            className="max-w-[18rem] truncate rounded border border-border bg-muted/30 px-1.5 py-0.5 font-mono text-[11px]"
                          >
                            {base}
                          </span>
                        ))
                      )}
                    </span>
                  </div>
                  <div className="flex flex-col gap-2 px-4 py-3 text-xs sm:flex-row sm:items-start sm:justify-between">
                    <span className="shrink-0 text-muted-foreground">
                      {t.doctor.forbiddenPaths}
                    </span>
                    <span className="flex min-w-0 items-center gap-2">
                      <span
                        className="truncate font-mono text-[11px]"
                        title={boundary?.forbidden_prefixes.join(', ')}
                      >
                        {boundary ? boundary.forbidden_prefixes.join(', ') : '—'}
                      </span>
                      <Badge variant="outline" className="shrink-0 text-[10px]">
                        {t.doctor.securityForbiddenBadge}
                      </Badge>
                    </span>
                  </div>
                </div>
                {!info.is_loopback ? (
                  <div className="border-t border-border/70 bg-amber-500/10 px-4 py-2.5 text-[11px] text-amber-700 dark:text-amber-400">
                    {t.doctor.exposedWarning}
                  </div>
                ) : null}
              </>
            ) : (
              <div className="px-4 py-8 text-sm text-muted-foreground">
                {t.doctor.runtimeConfigUnavailable}
              </div>
            )}
          </DoctorPanel>
        </TabsContent>
      </Tabs>

      <ConfirmDialog
        open={deletingProvider !== null}
        title={t.doctor.deleteProviderTitle}
        body={
          <>
            {t.doctor.deleteProviderBody.replace('{name}', deletingProvider ?? '')}
            {deleteError ? <span className="mt-2 block text-destructive">{deleteError}</span> : null}
          </>
        }
        confirmLabel={t.doctor.deleteProviderConfirm}
        cancelLabel={t.doctor.cancel}
        danger
        onConfirm={() => deletingProvider && void handleDeleteProvider(deletingProvider)}
        onCancel={() => {
          if (deleteBusy) return
          setDeletingProvider(null)
          setDeleteError(null)
        }}
      />
    </div>
  )
}
