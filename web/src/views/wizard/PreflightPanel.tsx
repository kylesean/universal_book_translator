import type { ReactNode } from 'react'
import { AlertTriangle, ArrowRight, Loader2 } from 'lucide-react'
import { Alert, AlertDescription } from '@/components/ui/Alert'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { InfoTip } from '@/components/ui/InfoTip'
import { cn } from '@/lib/utils'
import type { JobAssessResponse } from '@/api/client'
import { useI18n } from '@/i18n/useI18n'

function formatDuration(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`
  return `${(seconds / 3600).toFixed(1)}h`
}

function interpolate(
  template: string,
  params: Record<string, unknown>,
  listSeparator: string,
): string {
  return template.replace(/\{(\w+)\}/g, (whole, key: string) => {
    if (!(key in params)) return whole
    const value = params[key]
    return Array.isArray(value) ? value.join(listSeparator) : String(value)
  })
}

function warningText(
  warning: JobAssessResponse['warnings'][number],
  t: ReturnType<typeof useI18n>['t'],
): string {
  const template = t.wizard.warningCodes[warning.code]
  if (!template) return warning.detail_zh
  const params: Record<string, unknown> = { ...(warning.params ?? {}) }
  if (warning.code === 'FONT_RESIDUE_RISK') {
    params.remedy = params.is_scanned
      ? t.wizard.warningRemedyRescan
      : t.wizard.warningRemedyOcr
    const atRisk = Number(params.at_risk_pages ?? 0)
    params.at_risk_clause =
      atRisk > 0
        ? t.wizard.warningAtRiskClause.replace('{count}', String(atRisk))
        : ''
  }
  return interpolate(template, params, t.wizard.listSeparator)
}

function routeLabel(
  dual: string,
  preset: string,
  t: ReturnType<typeof useI18n>['t'],
): string {
  const duals: Record<string, string> = {
    mono: t.wizard.routeMono,
    dual: t.wizard.routeDual,
  }
  const presets: Record<string, string> = {
    high_fidelity: t.wizard.routeHighFidelity,
    standard: t.wizard.routeStandard,
  }
  return `${duals[dual] ?? dual} · ${presets[preset] ?? preset}`
}

function mathDensityLabel(
  density: string | null | undefined,
  t: ReturnType<typeof useI18n>['t'],
): string {
  switch (density?.toLowerCase()) {
    case 'high':
      return t.wizard.mathDensityHigh
    case 'medium':
      return t.wizard.mathDensityMedium
    case 'low':
      return t.wizard.mathDensityLow
    default:
      return density?.toUpperCase() ?? '—'
  }
}

function DetailRow({
  label,
  value,
  help,
  valueClassName,
}: {
  label: string
  value: ReactNode
  help?: string
  valueClassName?: string
}) {
  return (
    <div className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-4 border-b border-border/60 py-2.5 last:border-b-0">
      <dt className="flex min-w-0 items-center gap-1.5 text-xs text-muted-foreground">
        <span className="truncate">{label}</span>
        {help ? <InfoTip label={label}>{help}</InfoTip> : null}
      </dt>
      <dd className={cn('text-right font-mono text-xs font-semibold tabular-nums', valueClassName)}>
        {value}
      </dd>
    </div>
  )
}

export interface PreflightPanelProps {
  assessment: JobAssessResponse | null
  assessStale: boolean
  canSubmit: boolean
  deliveryMode: 'auto' | 'readable' | 'faithful'
  isReassessing: boolean
  isSubmitting: boolean
  onAcknowledgeRehearsal: () => void
  onOpenReusedJob: () => void
  onOpenSystem: () => void
  onRequestSubmit: () => void
  rehearsalAcknowledged: boolean
  rehearsalOnly: boolean
  reusedJobId: string | null
  submitBlockedReason: string | null
  submitError: string | null
}

export function PreflightPanel({
  assessment,
  assessStale,
  canSubmit,
  deliveryMode,
  isReassessing,
  isSubmitting,
  onAcknowledgeRehearsal,
  onOpenReusedJob,
  onOpenSystem,
  onRequestSubmit,
  rehearsalAcknowledged,
  rehearsalOnly,
  reusedJobId,
  submitBlockedReason,
  submitError,
}: PreflightPanelProps) {
  const { t } = useI18n()
  const cost = assessment?.cost.total_cost_usd
  const deliveryLabel =
    deliveryMode === 'faithful'
      ? t.wizard.deliveryModeFaithful
      : deliveryMode === 'readable'
        ? t.wizard.deliveryModeReadable
        : t.wizard.deliveryModeAuto

  return (
    <aside className="min-w-0 self-start overflow-hidden rounded-2xl border border-border bg-card xl:sticky xl:top-6 xl:max-h-[calc(100vh-3rem)] xl:overflow-y-auto">
      <header className="border-b border-border/80 px-5 py-4">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <h2 className="text-base font-semibold tracking-tight text-foreground">
              {t.wizard.preflightTitle}
            </h2>
            <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
              {t.wizard.preflightSubtitle}
            </p>
          </div>

          {isReassessing ? (
            <Badge variant="outline" className="shrink-0 gap-1.5 font-mono text-xs">
              <Loader2 className="size-3 animate-spin" />
              {t.wizard.reassessing}
            </Badge>
          ) : assessment ? (
            <Badge
              variant="outline"
              className={cn(
                'shrink-0 font-mono text-xs',
                assessment.cost.billable_blocks_is_exact
                  ? 'border-emerald-600/25 bg-emerald-500/10 text-emerald-700 dark:text-emerald-400'
                  : 'border-amber-600/25 bg-amber-500/10 text-amber-700 dark:text-amber-400',
              )}
            >
              {assessment.cost.billable_blocks_is_exact
                ? t.wizard.estimateExact
                : t.wizard.estimateApproximate}
            </Badge>
          ) : (
            <Badge variant="outline" className="shrink-0 text-xs">
              {t.common.idle}
            </Badge>
          )}
        </div>

        {assessment &&
        typeof assessment.meta?.provider === 'string' &&
        assessment.meta.provider !== '' ? (
          <p className="mt-2 font-mono text-xs text-muted-foreground">
            {t.wizard.quoteProviderNote.replace('{provider}', assessment.meta.provider)}
          </p>
        ) : null}

        {assessStale && assessment ? (
          <p className="mt-2 text-xs leading-relaxed text-amber-700 dark:text-amber-400">
            {t.wizard.assessStale}
          </p>
        ) : null}
      </header>

      <div className="px-5 py-5">
        {assessment ? (
          <div className="space-y-5">
            <dl className="grid grid-cols-2 divide-x divide-border/70">
              <div className="min-w-0 pr-4">
                <dt className="text-xs font-medium text-muted-foreground">
                  {t.wizard.sourceChars}
                </dt>
                <dd className="mt-1.5 font-mono text-2xl font-semibold tracking-tight text-foreground tabular-nums">
                  {assessment.document.source_chars.toLocaleString()}
                </dd>
                <dd className="mt-1 font-mono text-xs leading-relaxed text-muted-foreground">
                  {assessment.document.selected_pages != null
                    ? `${assessment.document.selected_pages} / ${assessment.document.pages} ${t.wizard.pagesUnit}`
                    : `${assessment.document.pages} ${t.wizard.pagesUnit}`}
                  {' · '}
                  {assessment.document.chapters} {t.wizard.chapters}
                </dd>
              </div>

              <div className="min-w-0 pl-4">
                <dt className="text-xs font-medium text-muted-foreground">{t.wizard.estCost}</dt>
                <dd className="mt-1.5 font-mono text-2xl font-semibold tracking-tight tabular-nums">
                  {cost === null || cost === undefined ? (
                    <span className="text-sm font-normal text-muted-foreground">
                      {t.wizard.notPriced}
                    </span>
                  ) : cost === 0 ? (
                    <span className="text-emerald-700 dark:text-emerald-400">
                      <span className="block">$0.00</span>
                      <span className="mt-0.5 block font-sans text-xs font-normal text-muted-foreground">
                        {t.wizard.localFreeLabel}
                      </span>
                    </span>
                  ) : (
                    <span className="text-amber-700 dark:text-amber-400">${cost.toFixed(2)}</span>
                  )}
                </dd>
                <dd className="mt-1 font-mono text-xs leading-relaxed text-muted-foreground">
                  {formatDuration(assessment.runtime.est_seconds_low)} –{' '}
                  {formatDuration(assessment.runtime.est_seconds_high)}
                </dd>
              </div>
            </dl>

            <section className="border-t border-border/70 pt-4">
              <div className="mb-1">
                <h3 className="text-xs font-semibold uppercase tracking-[0.08em] text-muted-foreground">
                  {t.wizard.scorecardSub}
                </h3>
                <p className="mt-1 font-mono text-xs font-medium text-primary">
                  {routeLabel(
                    assessment.route.recommended_dual_mode,
                    assessment.route.recommended_preset,
                    t,
                  )}
                </p>
              </div>

              <dl>
                <DetailRow
                  label={t.wizard.estTokens}
                  value={(
                    assessment.cost.prompt_tokens + assessment.cost.completion_tokens
                  ).toLocaleString()}
                />
                <DetailRow
                  label={t.wizard.mathDensity}
                  help={
                    assessment.document.has_formulas
                      ? t.wizard.mathDensityWithObjects
                      : t.wizard.mathDensityTextOnly
                  }
                  value={mathDensityLabel(assessment.document.math_density, t)}
                  valueClassName={
                    assessment.document.math_density === 'high'
                      ? 'text-amber-700 dark:text-amber-400'
                      : undefined
                  }
                />
                <DetailRow
                  label={t.wizard.scannedRatio}
                  value={
                    assessment.document.scan_page_share != null
                      ? `${(assessment.document.scan_page_share * 100).toFixed(0)}%`
                      : assessment.document.is_scanned
                        ? '100%'
                        : '0%'
                  }
                />
                <DetailRow
                  label={t.wizard.confidence}
                  value={`${(assessment.route.confidence * 100).toFixed(0)}%`}
                  valueClassName="text-primary"
                />
                <DetailRow label={t.wizard.deliveryMode} value={deliveryLabel} />
              </dl>
            </section>

            {assessment.warnings.length > 0 ? (
              <section className="rounded-lg bg-amber-500/10 p-3">
                <div className="flex items-center gap-1.5 text-xs font-semibold text-amber-800 dark:text-amber-300">
                  <AlertTriangle className="size-3.5 shrink-0" />
                  <span>
                    {t.wizard.probeWarnings} ({assessment.warnings.length})
                  </span>
                </div>
                <div className="mt-2 max-h-40 space-y-2 overflow-y-auto pr-1 text-xs leading-relaxed text-amber-900/90 dark:text-amber-100/80">
                  {assessment.warnings.map((warning) => (
                    <p key={warning.code}>{warningText(warning, t)}</p>
                  ))}
                </div>
              </section>
            ) : null}
          </div>
        ) : (
          <div className="py-2">
            <p className="text-sm font-medium text-foreground">{t.wizard.waitingFile}</p>
          </div>
        )}
      </div>

      {rehearsalOnly ? (
        <div className="mx-5 mb-5 rounded-lg bg-amber-500/10 p-3">
          <p className="text-xs font-semibold text-amber-800 dark:text-amber-300">
            {t.wizard.rehearsalBannerTitle}
          </p>
          <p className="mt-1 text-xs leading-relaxed text-amber-900/90 dark:text-amber-100/80">
            {t.wizard.rehearsalBannerBody}
          </p>
          <div className="mt-2 flex flex-wrap items-center gap-2">
            <Button type="button" size="sm" onClick={onOpenSystem}>
              {t.wizard.rehearsalBannerCta}
            </Button>
            {!rehearsalAcknowledged ? (
              <Button
                type="button"
                size="sm"
                variant="ghost"
                onClick={onAcknowledgeRehearsal}
              >
                {t.wizard.rehearsalBannerAck}
              </Button>
            ) : null}
          </div>
        </div>
      ) : null}

      <footer
        className={cn(
          'space-y-3 border-t border-border/80 bg-muted/25 p-5',
          submitError || reusedJobId || (!isSubmitting && submitBlockedReason)
            ? 'block'
            : 'hidden lg:block',
        )}
      >
        {submitError ? (
          <Alert variant="destructive">
            <AlertTriangle className="size-4" />
            <AlertDescription>{submitError}</AlertDescription>
          </Alert>
        ) : null}

        {reusedJobId ? (
          <Alert className="flex items-start justify-between gap-2">
            <AlertDescription>{t.wizard.submitReusedNote}</AlertDescription>
            <button
              type="button"
              onClick={onOpenReusedJob}
              className="shrink-0 cursor-pointer font-mono text-xs font-semibold text-primary underline underline-offset-2 hover:no-underline"
            >
              {reusedJobId}
            </button>
          </Alert>
        ) : null}

        {!isSubmitting && submitBlockedReason ? (
          <p className="text-xs leading-relaxed text-muted-foreground">{submitBlockedReason}</p>
        ) : null}

        <Button
          type="button"
          onClick={onRequestSubmit}
          disabled={!canSubmit}
          size="lg"
          className="hidden h-11 w-full text-sm font-semibold lg:flex"
        >
          {isSubmitting ? (
            <>
              <Loader2 className="mr-2 size-4 animate-spin" />
              {t.wizard.compiling}
            </>
          ) : (
            <>
              {t.wizard.startCompile}
              <ArrowRight className="ml-2 size-4" />
            </>
          )}
        </Button>
      </footer>
    </aside>
  )
}
