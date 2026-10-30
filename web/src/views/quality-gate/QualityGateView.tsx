import React, { useCallback, useEffect, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import {
  ShieldCheck,
  ArrowUpRight,
  ArrowLeft,
  CheckCircle2,
  AlertTriangle,
  ChevronDown,
  ChevronRight,
  Eye,
  RefreshCw,
  FileText,
  Copy,
  CheckCheck,
  Download,
  Edit3,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Card } from '@/components/ui/card'
import { useToast } from '@/components/ui/useToast'
import { InfoTip } from '@/components/ui/InfoTip'
import { cn } from '@/lib/utils'
import {
  getJobReport,
  getVisualReport,
  getDeliveryContract,
  listDeliverables,
  deliverableDownloadUrl,
  triggerBrowserDownload,
  reExportJob,
  pagePreviewUrl,
  ApiError,
  type CompanionArtifactInfo,
  type DeliverableItem,
} from '@/api/client'
import { useI18n } from '@/i18n/useI18n'
import { PreviewDrawer } from './PreviewDrawer'
import { PostEditDrawer } from './PostEditDrawer'
import { deliverableName, fmtSize, isPreviewableProduct } from './artifactMeta'

type CheckStatus = 'pass' | 'warn' | 'fail'

interface Check {
  title: string
  summary: string
  status: CheckStatus
  /** 0-1 ratio used by the compact progress rail. */
  value: number
  detail: React.ReactNode
}

const STATUS_TONE: Record<CheckStatus, string> = {
  pass: 'text-emerald-700 dark:text-emerald-400',
  warn: 'text-amber-700 dark:text-amber-400',
  fail: 'text-destructive',
}

const STATUS_BAR: Record<CheckStatus, string> = {
  pass: 'bg-emerald-600 dark:bg-emerald-500',
  warn: 'bg-amber-600 dark:bg-amber-500',
  fail: 'bg-destructive',
}

function qeEngineLabel(
  source: string | null | undefined,
  t: ReturnType<typeof useI18n>['t']
): string {
  switch (source) {
    case 'heuristic':
      return t.quality.auditQeEngineRuleChecks
    case 'llm_judge':
      return t.quality.auditQeEngineModelScoring
    case 'tiered':
      return `${t.quality.auditQeEngineRuleChecks} + ${t.quality.auditQeEngineModelScoring}`
    case 'mock':
      return t.quality.auditQeEngineRehearsal
    default:
      return t.quality.auditQeEngineUnknown
  }
}

function statusLabel(t: ReturnType<typeof useI18n>['t'], status: CheckStatus): string {
  if (status === 'pass') return t.quality.pillarStatusPass
  if (status === 'warn') return t.quality.pillarStatusWarn
  return t.quality.pillarStatusFail
}

export function QualityGateView() {
  const { t, language } = useI18n()
  const toast = useToast()
  const navigate = useNavigate()
  const { jobId: routeJobId } = useParams<{ jobId: string }>()
  const jobId = routeJobId ?? null
  const [copiedId, setCopiedId] = useState(false)
  const [report, setReport] = useState<Record<string, any> | null>(null)
  const [deliverables, setDeliverables] = useState<DeliverableItem[]>([])
  const [companionArtifacts, setCompanionArtifacts] = useState<CompanionArtifactInfo[]>([])
  const [deliveryStale, setDeliveryStale] = useState(false)
  const [downloadBusy, setDownloadBusy] = useState(false)
  const [downloadError, setDownloadError] = useState<string | null>(null)
  const [visual, setVisual] = useState<Record<string, any> | null>(null)
  const [visualMissing, setVisualMissing] = useState(false)
  const [deliveryContract, setDeliveryContract] = useState<Record<string, any> | null>(null)
  const [expanded, setExpanded] = useState<number | null>(null)
  const [previewItem, setPreviewItem] = useState<DeliverableItem | null>(null)
  const [postEditOpen, setPostEditOpen] = useState(false)
  const [jobMissing, setJobMissing] = useState(false)

  const handleCopyJobId = (idToCopy: string) => {
    void navigator.clipboard.writeText(idToCopy)
    setCopiedId(true)
    toast.push(t.quality.jobIdCopied, 'success')
    setTimeout(() => setCopiedId(false), 1500)
  }

  const loadAll = useCallback(async () => {
    if (!jobId) return
    try {
      const rep = await getJobReport(jobId)
      setReport(rep)
    } catch (err) {
      if (err instanceof ApiError && err.status === 404) setJobMissing(true)
      else setJobMissing(false)
      return
    }
    try {
      const listing = await listDeliverables(jobId)
      setDeliverables(listing.items)
      setCompanionArtifacts(listing.companionArtifacts ?? [])
      setDeliveryStale(listing.stale)
    } catch {
      setDeliverables([])
      setCompanionArtifacts([])
    }
    try {
      setVisual(await getVisualReport(jobId))
      setVisualMissing(false)
    } catch {
      setVisual(null)
      setVisualMissing(true)
    }
    try {
      setDeliveryContract(await getDeliveryContract(jobId))
    } catch {
      setDeliveryContract(null)
    }
  }, [jobId])

  useEffect(() => {
    // oxlint-disable-next-line react/set-state-in-effect
    void loadAll()
  }, [loadAll])

  useEffect(() => {
    if (!jobId || report !== null || jobMissing) return
    const timer = setInterval(() => void loadAll(), 5000)
    return () => clearInterval(timer)
  }, [jobId, report, jobMissing, loadAll])

  if (!jobId) {
    return (
      <div className="flex flex-1 items-center justify-center p-8 text-center">
        <Card className="max-w-sm p-6">
          <ShieldCheck className="mx-auto size-8 text-muted-foreground/40" />
          <h2 className="mt-3 text-sm font-semibold text-foreground">{t.quality.noJobTitle}</h2>
          <p className="mt-1 text-sm leading-relaxed text-muted-foreground">{t.quality.noJobDesc}</p>
          <Button onClick={() => navigate('/jobs')} variant="secondary" size="sm" className="mt-3">
            {t.mission.backToQueue}
          </Button>
        </Card>
      </div>
    )
  }

  const summary = report?.summary ?? {}
  const score = report?.score_metrics ?? {}
  const placeholder = report?.placeholder ?? {}
  const terminology = report?.terminology ?? {}
  const entity = report?.entity_consistency ?? {}
  const coverage = report?.render_coverage ?? {}
  const repair = report?.repair_breakdown ?? {}
  const config = report?.config_snapshot ?? {}
  const visualSkipped: string | null =
    typeof visual?.skipped_reason === 'string' && visual.skipped_reason
      ? visual.skipped_reason
      : null
  const tiersRan: string[] = Array.isArray(visual?.tiers_ran) ? visual.tiers_ran : []
  const tiersSkipped: string[] = Array.isArray(visual?.tiers_skipped) ? visual.tiers_skipped : []
  const skipCodes: string[] = Array.isArray(visual?.skip_codes) ? visual.skip_codes : []
  const tierLabel = (tier: string): string =>
    (t.quality.visualTierNames as Record<string, string>)[tier] ?? tier
  const skipLabel = (code: string): string =>
    (t.quality.visualSkipCodes as Record<string, string>)[code] ?? code

  const failedBlocks = summary.failed_blocks ?? 0
  const verdict: string | null = report?.delivery_verdict ?? null
  const rehearsal: boolean = report?.rehearsal === true
  const hasReport = report !== null
  const isPassed = hasReport && (verdict ? verdict === 'passed' : failedBlocks === 0)
  const isWarning = hasReport && verdict === 'warning'
  const costPriced = hasReport && typeof summary.estimated_cost_usd === 'number'
  const pct = (value: number | undefined, digits = 1) =>
    value === undefined ? '—' : `${(value * 100).toFixed(digits)}%`
  const clamp = (value: number | undefined): number =>
    value === undefined || Number.isNaN(value) ? 0 : Math.min(1, Math.max(0, value))

  const topDrifted: Array<Record<string, any>> = entity.top_drifted ?? []
  const visualFindings: Array<Record<string, any>> = visual?.findings ?? []
  const flaggedPages = Array.from(
    new Set(
      visualFindings
        .map((finding) => finding.page)
        .filter((page): page is number => typeof page === 'number')
    )
  ).sort((a, b) => a - b)
  const findingsByPage = new Map<number, number>()
  for (const finding of visualFindings) {
    if (typeof finding.page === 'number') {
      findingsByPage.set(finding.page, (findingsByPage.get(finding.page) ?? 0) + 1)
    }
  }
  const sampledPages: number[] = Array.isArray(visual?.sampled_pages)
    ? visual.sampled_pages.filter((p: unknown): p is number => typeof p === 'number')
    : []

  const visualSummary = visualMissing
    ? t.quality.visualNoReport
    : visualSkipped
      ? tiersRan.length > 0
        ? t.quality.visualPartialRan
            .replace('{ran}', String(tiersRan.length))
            .replace('{skipped}', tiersSkipped.map(tierLabel).join(language === 'zh' ? '、' : ', '))
        : t.quality.visualGateSkipped
      : `${flaggedPages.length} ${t.quality.flaggedPages}`

  const visualStatus: CheckStatus =
    visualMissing || visualSkipped
      ? 'warn'
      : flaggedPages.length > 0 || visual?.passed === false
        ? 'fail'
        : 'pass'

  const contractSummary = (deliveryContract?.summary as Record<string, any>) ?? {}
  const contractCritical = Number(contractSummary.critical ?? 0)
  const contractMajor = Number(contractSummary.major ?? 0)
  const contractChecks: Array<{ name: string; status: string }> = Array.isArray(deliveryContract?.checks)
    ? deliveryContract.checks
    : []
  const contractFindings: Array<Record<string, any>> = Array.isArray(deliveryContract?.findings)
    ? deliveryContract.findings
    : []
  const celsDistortions: Array<{
    page: number
    score: number
    verdict: string
    deficit_pt?: number
    track_count?: number
    is_multicolumn?: boolean
  }> = Array.isArray(deliveryContract?.cels_distortions)
    ? deliveryContract.cels_distortions
    : []

  const contractStatus: CheckStatus = !deliveryContract
    ? 'warn'
    : contractCritical > 0 || deliveryContract.passed === false
      ? 'fail'
      : contractMajor > 0
        ? 'warn'
        : 'pass'

  const avgDistortion =
    celsDistortions.length > 0
      ? celsDistortions.reduce((acc, cur) => acc + (cur.score || 0), 0) / celsDistortions.length
      : null

  const checks: Check[] = [
    {
      title: t.quality.chipDeliveryContract,
      summary: !deliveryContract
        ? (language === 'zh' ? '暂未生成交付契约报告' : 'Delivery contract report not available')
        : `${String(deliveryContract.contract ?? 'CONTRACT').toUpperCase()} · ${
            deliveryContract.deliverable
              ? (language === 'zh' ? '准予出厂交付' : 'Deliverable')
              : (language === 'zh' ? '阻断交付' : 'Blocked')
          } (${contractCritical} critical, ${contractMajor} major)`,
      status: contractStatus,
      value: !deliveryContract ? 0 : contractStatus === 'pass' ? 1 : contractStatus === 'warn' ? 0.7 : 0,
      detail: (
        <div className="space-y-4">
          <dl className="grid gap-x-8 gap-y-2 sm:grid-cols-2">
            {[
              [t.quality.contractModeLabel, String(deliveryContract?.contract ?? '—')],
              [
                t.quality.contractDeliverableLabel,
                deliveryContract ? (
                  <Badge
                    variant={deliveryContract.deliverable ? 'default' : 'destructive'}
                    className={cn(
                      'font-mono text-xs',
                      deliveryContract.deliverable
                        ? 'border border-emerald-600/30 bg-emerald-600/15 text-emerald-700 dark:text-emerald-400'
                        : ''
                    )}
                  >
                    {deliveryContract.deliverable ? 'PASS · DELIVERABLE' : 'FAIL · BLOCKED'}
                  </Badge>
                ) : '—',
              ],
              [language === 'zh' ? 'Critical 违规' : 'Critical Violations', contractCritical],
              [language === 'zh' ? 'Major 违规' : 'Major Violations', contractMajor],
              [
                language === 'zh' ? '平均版面畸变分' : 'Avg Distortion Score',
                avgDistortion !== null ? avgDistortion.toFixed(4) : '—',
              ],
            ].map(([label, value]) => (
              <div key={String(label)} className="flex items-baseline justify-between gap-4 border-b border-border/60 pb-1.5">
                <dt className="text-muted-foreground">{label}</dt>
                <dd className="font-mono font-medium tabular-nums text-foreground">{value}</dd>
              </div>
            ))}
          </dl>

          {contractChecks.length > 0 && (
            <div className="space-y-2">
              <div className="text-xs font-medium text-muted-foreground">{t.quality.contractChecksTitle}</div>
              <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                {contractChecks.map((chk) => (
                  <div
                    key={chk.name}
                    className={cn(
                      'flex items-center justify-between rounded border px-2.5 py-1.5 text-xs',
                      chk.status === 'pass'
                        ? 'border-emerald-600/20 bg-emerald-500/10 text-emerald-700 dark:text-emerald-400'
                        : chk.status === 'warn'
                          ? 'border-amber-600/20 bg-amber-500/10 text-amber-700 dark:text-amber-400'
                          : 'border-destructive/20 bg-destructive/10 text-destructive'
                    )}
                  >
                    <span className="font-mono">{chk.name}</span>
                    <span className="font-bold uppercase">{chk.status}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {celsDistortions.length > 0 && (
            <div className="space-y-2">
              <div className="text-xs font-medium text-muted-foreground">{t.quality.celsDistortionTitle}</div>
              <div className="max-h-40 overflow-y-auto rounded border border-border/60">
                <table className="w-full text-left text-xs">
                  <thead className="border-b border-border/60 bg-muted/40 font-medium text-muted-foreground">
                    <tr>
                      <th className="px-3 py-1.5">Page</th>
                      <th className="px-3 py-1.5">{t.quality.celsDistortionScore}</th>
                      <th className="px-3 py-1.5">Verdict</th>
                      <th className="px-3 py-1.5">{t.quality.celsTracks}</th>
                      <th className="px-3 py-1.5">{t.quality.celsDeficit}</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border/40 font-mono">
                    {celsDistortions.map((d) => (
                      <tr key={d.page} className="hover:bg-muted/30">
                        <td className="px-3 py-1">P{d.page}</td>
                        <td
                          className={cn(
                            'px-3 py-1 font-semibold tabular-nums',
                            d.score < 0.08
                              ? 'text-emerald-700 dark:text-emerald-400'
                              : d.score <= 0.15
                                ? 'text-amber-700 dark:text-amber-400'
                                : 'text-destructive'
                          )}
                        >
                          {d.score.toFixed(4)}
                        </td>
                        <td className="px-3 py-1 text-xs uppercase">{d.verdict}</td>
                        <td className="px-3 py-1 tabular-nums">{d.track_count ?? 1}</td>
                        <td className="px-3 py-1 tabular-nums">{d.deficit_pt != null ? `${d.deficit_pt}pt` : '0pt'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {contractFindings.length > 0 ? (
            <div className="space-y-2">
              <div className="text-xs font-medium text-destructive">
                {language === 'zh' ? '违规项明细' : 'Findings Detail'} ({contractFindings.length})
              </div>
              <ul className="max-h-48 space-y-1.5 overflow-y-auto">
                {contractFindings.map((finding, idx) => (
                  <li
                    key={`${finding.code}-${finding.page}-${idx}`}
                    className="flex gap-2 rounded border border-destructive/20 bg-destructive/5 p-2 text-xs"
                  >
                    <Badge variant="destructive" className="shrink-0 text-[10px] uppercase">
                      {String(finding.severity)}
                    </Badge>
                    <div className="flex-1 space-y-0.5">
                      <div className="flex items-center gap-2 font-mono text-muted-foreground">
                        {finding.page != null && <span>P{finding.page}</span>}
                        {finding.block_id && <span>[{finding.block_id}]</span>}
                        <span className="font-semibold text-foreground">{String(finding.code)}</span>
                      </div>
                      <p className="text-foreground">{String(finding.message)}</p>
                    </div>
                  </li>
                ))}
              </ul>
            </div>
          ) : (
            <p className="text-xs text-muted-foreground">
              {language === 'zh'
                ? '✓ 无契约违规项，版面几何与文本实现物理不变量完全闭环。'
                : '✓ Zero contract findings. Layout geometry and text realization invariants strictly held.'}
            </p>
          )}
        </div>
      ),
    },
    {
      title: t.quality.chipMath,
      summary: t.quality.mathSummary
        .replace('{count}', String(placeholder.corrupt_spans ?? 0))
        .replace('{rate}', pct(placeholder.retention_rate)),
      status: (placeholder.corrupt_spans ?? 0) > 0 ? 'fail' : 'pass',
      value: clamp(placeholder.retention_rate),
      detail: (
        <dl className="grid gap-x-8 gap-y-2 sm:grid-cols-2">
          {[
            [t.quality.metricMaskedSpans, placeholder.masked_spans ?? 0],
            [t.quality.metricCorruptSpans, placeholder.corrupt_spans ?? 0],
            [t.quality.metricMaskedBlocks, placeholder.masked_blocks ?? 0],
            [t.quality.metricCorruptBlocks, placeholder.corrupt_blocks ?? 0],
            [t.quality.metricFormulaBlocks, report?.formula_blocks ?? 0],
          ].map(([label, value]) => (
            <div key={String(label)} className="flex items-baseline justify-between gap-4 border-b border-border/60 pb-1.5">
              <dt className="text-muted-foreground">{label}</dt>
              <dd className="font-mono font-medium tabular-nums text-foreground">{value}</dd>
            </div>
          ))}
        </dl>
      ),
    },
    {
      title: t.quality.chipTerm,
      summary: t.quality.termSummary
        .replace('{rate}', pct(terminology.term_recall))
        .replace('{count}', String(entity.terms_with_drift ?? 0)),
      status: (entity.terms_with_drift ?? 0) > 0 ? 'warn' : 'pass',
      value: clamp(terminology.term_recall),
      detail:
        topDrifted.length === 0 ? (
          <p className="text-sm text-muted-foreground">{t.quality.noDrift}</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[420px] text-left text-sm">
              <thead className="border-b border-border text-xs text-muted-foreground">
                <tr>
                  <th className="py-2 pr-4 font-medium">{t.quality.colTerm}</th>
                  <th className="py-2 pr-4 font-medium">{t.quality.colExpected}</th>
                  <th className="py-2 pr-4 font-medium">{t.quality.colOccurrences}</th>
                  <th className="py-2 font-medium">{t.quality.colDriftRate}</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border/70">
                {topDrifted.map((row) => (
                  <tr key={`${row.source}-${row.expected}`}>
                    <td className="py-2 pr-4 font-medium text-foreground">{row.source}</td>
                    <td className="py-2 pr-4 text-emerald-700 dark:text-emerald-400">{row.expected}</td>
                    <td className="py-2 pr-4 font-mono tabular-nums">{row.occurrences ?? 0}</td>
                    <td className="py-2 font-mono tabular-nums text-amber-700 dark:text-amber-400">
                      {typeof row.drift_rate === 'number' ? pct(row.drift_rate, 0) : '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ),
    },
    {
      title: t.quality.chipLayout,
      summary: visualSummary,
      status: visualStatus,
      value: visualMissing || visualSkipped ? 0 : visualStatus === 'pass' ? 1 : 0,
      detail: (
        <div className="space-y-3">
          {visualMissing && <p className="text-sm text-muted-foreground">{t.quality.visualNoReportDesc}</p>}
          {(tiersRan.length > 0 || tiersSkipped.length > 0) && (
            <div className="grid gap-2 sm:grid-cols-2">
              {tiersRan.length > 0 && (
                <div>
                  <div className="text-xs font-medium text-muted-foreground">{t.quality.visualTiersRan}</div>
                  <div className="mt-1 text-sm text-foreground">{tiersRan.map(tierLabel).join(' · ')}</div>
                </div>
              )}
              {tiersSkipped.length > 0 && (
                <div>
                  <div className="text-xs font-medium text-muted-foreground">{t.quality.visualTiersSkipped}</div>
                  <div className="mt-1 text-sm text-amber-700 dark:text-amber-400">
                    {tiersSkipped.map(tierLabel).join(' · ')}
                    {skipCodes.length > 0 && ` (${skipCodes.map(skipLabel).join('、')})`}
                  </div>
                </div>
              )}
            </div>
          )}
          {visualFindings.length === 0 ? (
            <p className="text-sm text-muted-foreground">{t.quality.noVisualFindings}</p>
          ) : (
            <div className="space-y-2">
              <div className="text-xs font-medium text-muted-foreground">{t.quality.visualFindingsTitle}</div>
              <ul className="max-h-48 space-y-1.5 overflow-y-auto">
                {visualFindings.map((finding, index) => (
                  <li key={`${finding.page}-${finding.code}-${index}`} className="flex gap-3 text-sm">
                    <span className="w-10 shrink-0 font-mono text-muted-foreground">
                      {finding.page != null ? `P${finding.page}` : '—'}
                    </span>
                    <span className="text-muted-foreground">{finding.message}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
          {sampledPages.length > 0 && (
            <div>
              <div className="mb-2 text-xs font-medium text-muted-foreground">
                {t.quality.sampledPages} · {sampledPages.length}
              </div>
              <div className="flex flex-wrap gap-1">
                {sampledPages.slice(0, 60).map((page) => (
                  <span
                    key={page}
                    className={cn(
                      'inline-flex h-6 min-w-7 items-center justify-center rounded border px-1 font-mono text-xs',
                      findingsByPage.has(page)
                        ? 'border-destructive/25 bg-destructive/10 text-destructive'
                        : 'border-emerald-600/20 bg-emerald-500/10 text-emerald-700 dark:text-emerald-400'
                    )}
                  >
                    {page}
                  </span>
                ))}
                {sampledPages.length > 60 && (
                  <span className="px-1 text-xs text-muted-foreground">+{sampledPages.length - 60}</span>
                )}
              </div>
            </div>
          )}
          {flaggedPages.length > 0 && (
            <div className="grid max-h-64 grid-cols-2 gap-2 overflow-y-auto sm:grid-cols-4">
              {flaggedPages.map((page) => (
                <a
                  key={page}
                  href={pagePreviewUrl(jobId, page, { dpi: 140 })}
                  target="_blank"
                  rel="noreferrer"
                  className="overflow-hidden rounded-md border border-border bg-white transition-colors hover:border-border-hover"
                >
                  <img
                    src={pagePreviewUrl(jobId, page, { dpi: 70 })}
                    alt={t.quality.pageAlt.replace('{n}', String(page))}
                    className="h-24 w-full object-contain"
                  />
                  <span className="block border-t border-border py-1 text-center font-mono text-xs text-muted-foreground">
                    P{page}
                  </span>
                </a>
              ))}
            </div>
          )}
        </div>
      ),
    },
    {
      title: t.quality.chipComplete,
      summary: t.quality.coverageSummary
        .replace('{rate}', pct(coverage.render_coverage))
        .replace('{count}', String(coverage.preserved_blocks ?? 0)),
      status: (coverage.fail_closed_blocks ?? 0) > 0 ? 'fail' : 'pass',
      value: clamp(coverage.render_coverage),
      detail: (
        <dl className="grid gap-x-8 gap-y-2 sm:grid-cols-2">
          {[
            [t.quality.metricRenderedBlocks, coverage.rendered_blocks ?? 0],
            [t.quality.metricSkippedBlocks, coverage.skipped_blocks ?? 0],
            [t.quality.metricFailClosedBlocks, coverage.fail_closed_blocks ?? 0],
            [t.quality.metricPreservedBlocks, coverage.preserved_blocks ?? 0],
          ].map(([label, value]) => (
            <div key={String(label)} className="flex items-baseline justify-between gap-4 border-b border-border/60 pb-1.5">
              <dt className="text-muted-foreground">{label}</dt>
              <dd className="font-mono font-medium tabular-nums text-foreground">{value}</dd>
            </div>
          ))}
        </dl>
      ),
    },
    {
      title: t.quality.chipCost,
      summary: costPriced
        ? `$${Number(summary.estimated_cost_usd).toFixed(2)} · ${t.quality.auditCacheHit} ${pct(summary.cache_hit_rate)}`
        : t.quality.costNotPriced,
      status: costPriced ? 'pass' : 'warn',
      value: costPriced ? 1 : 0.5,
      detail: (
        <dl className="grid gap-x-8 gap-y-2 sm:grid-cols-2">
          {[
            [t.quality.auditDraftModel, config.draft_model ?? '—'],
            [t.quality.auditRepairModel, config.repair_model ?? '—'],
            [t.quality.auditQeEngine, qeEngineLabel(report?.qe_score_source, t)],
            [
              t.quality.auditRetries,
              `${repair.direct_pass_count ?? 0}/${repair.round_1_repaired_count ?? 0}/${repair.round_2_repaired_count ?? 0}/${repair.exhausted_count ?? 0}`,
            ],
            [t.quality.auditCacheHit, pct(summary.cache_hit_rate)],
            [t.quality.auditTypst, report?.typst_version ?? '—'],
          ].map(([label, value]) => (
            <div key={String(label)} className="flex items-baseline justify-between gap-4 border-b border-border/60 pb-1.5">
              <dt className="text-muted-foreground">{label}</dt>
              <dd className="truncate text-right font-mono font-medium text-foreground">{value}</dd>
            </div>
          ))}
        </dl>
      ),
    },
  ]

  const handleDownload = async (key: string) => {
    if (downloadBusy) return
    setDownloadBusy(true)
    setDownloadError(null)
    try {
      let currentItems = deliverables
      if (deliveryStale) {
        await reExportJob(jobId)
        const refreshed = await listDeliverables(jobId)
        setDeliverables(refreshed.items)
        setCompanionArtifacts(refreshed.companionArtifacts ?? [])
        setDeliveryStale(refreshed.stale)
        currentItems = refreshed.items
      }
      const target = currentItems.find((d) => d.key === key) ?? currentItems[0]
      triggerBrowserDownload(deliverableDownloadUrl(jobId, key), target?.filename)
    } catch (err) {
      setDownloadError(err instanceof Error ? err.message : t.quality.downloadFailed)
    } finally {
      setDownloadBusy(false)
    }
  }

  const primaryItem =
    deliverables.find((item) => item.key === 'primary') ??
    deliverables.find((item) => isPreviewableProduct(item)) ??
    null
  const primaryPreviewable = primaryItem !== null && isPreviewableProduct(primaryItem)
  const verdictTitle = rehearsal
    ? t.quality.rehearsalBadge
    : isPassed
      ? t.quality.decisionReady
      : isWarning
        ? t.quality.decisionWarning
        : t.quality.decisionBlocked
  const verdictBody = rehearsal
    ? t.quality.rehearsalBody
    : isPassed
      ? t.quality.allPassedDesc
      : isWarning
        ? t.quality.warningDesc
        : t.quality.blockedDesc

  return (
    <div className="mx-auto w-full max-w-5xl flex-1 space-y-6 overflow-y-auto px-4 py-6 sm:px-7">
      <header className="flex flex-col justify-between gap-4 border-b border-border/80 pb-5 sm:flex-row sm:items-end">
        <div className="min-w-0">
              <div className="flex items-center gap-1.5">
            <Button
              variant="ghost"
              size="icon"
              onClick={() => navigate(`/jobs/${jobId}`)}
              className="-ml-2 size-8"
              title={t.quality.backToJob}
              aria-label={t.quality.backToJob}
            >
              <ArrowLeft />
            </Button>
            <span className="text-muted-foreground/40">/</span>
            <div className="flex min-w-0 items-center gap-1.5 font-mono text-xs text-muted-foreground">
              <span className="truncate font-medium text-foreground select-all">{jobId}</span>
              <button
                type="button"
                onClick={() => handleCopyJobId(jobId)}
                className="rounded p-1 transition-colors hover:bg-muted hover:text-foreground"
                title={t.quality.copyJobId}
                aria-label={t.quality.copyJobId}
              >
                {copiedId ? <CheckCheck className="size-3 text-emerald-600" /> : <Copy className="size-3" />}
              </button>
            </div>
          </div>
          <h1 className="mt-1 text-2xl font-semibold tracking-[-0.025em] text-foreground">
            {t.quality.title}
          </h1>
          <p className="mt-1 max-w-2xl text-sm leading-relaxed text-muted-foreground">
            {t.quality.subtitle}
          </p>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          <Button onClick={() => setPostEditOpen(true)} variant="secondary" size="sm">
            <Edit3 />
            {t.quality.postEditAction}
          </Button>
          <Button onClick={() => navigate(`/jobs/${jobId}/review`)} variant="outline" size="sm">
            {t.mission.openWorkbench}
            <ArrowUpRight />
          </Button>
        </div>
      </header>

      {!hasReport && (
        <Card className="flex items-start gap-3.5 p-5">
          <AlertTriangle className="mt-0.5 size-5 shrink-0 text-amber-600 dark:text-amber-500" />
          <div className="space-y-2">
            <div className="text-sm font-semibold text-foreground">{t.quality.noVerdictTitle}</div>
            <p className="text-sm leading-relaxed text-muted-foreground">{t.quality.noVerdictBody}</p>
            {!jobMissing && (
              <Button variant="secondary" size="sm" onClick={() => void loadAll()}>
                <RefreshCw />
                {t.quality.refreshBtn}
              </Button>
            )}
          </div>
        </Card>
      )}

      {hasReport && (
        <>
          <Card className="overflow-hidden">
            <div className="flex flex-col gap-5 p-5 sm:p-6 lg:flex-row lg:items-center lg:justify-between">
              <div className="flex min-w-0 items-start gap-4">
                <div
                  className={cn(
                    'flex size-11 shrink-0 items-center justify-center rounded-full',
                    rehearsal
                      ? 'bg-blue-500/10 text-blue-700 dark:text-blue-400'
                      : isPassed
                        ? 'bg-emerald-500/10 text-emerald-700 dark:text-emerald-400'
                        : isWarning
                          ? 'bg-amber-500/10 text-amber-700 dark:text-amber-400'
                          : 'bg-destructive/10 text-destructive'
                  )}
                >
                  {isPassed && !rehearsal ? (
                    <CheckCircle2 className="size-5" />
                  ) : (
                    <AlertTriangle className="size-5" />
                  )}
                </div>
                <div className="min-w-0">
                  <div className="flex flex-wrap items-center gap-2">
                    <h2 className="text-lg font-semibold tracking-[-0.015em] text-foreground">
                      {verdictTitle}
                    </h2>
                    <Badge
                      variant="outline"
                      className={cn(
                        isPassed
                          ? 'border-emerald-600/25 bg-emerald-500/10 text-emerald-700 dark:text-emerald-400'
                          : isWarning
                            ? 'border-amber-600/25 bg-amber-500/10 text-amber-700 dark:text-amber-400'
                            : 'border-destructive/25 bg-destructive/10 text-destructive'
                      )}
                    >
                      <span
                        className={cn(
                          'mr-1.5 size-1.5 rounded-full',
                          isPassed
                            ? 'bg-emerald-500'
                            : isWarning
                              ? 'bg-amber-500'
                              : 'bg-destructive'
                        )}
                      />
                      {isPassed
                        ? t.quality.verdictPass
                        : isWarning
                          ? t.quality.verdictWarning
                          : t.quality.verdictBlocked}
                    </Badge>
                  </div>
                  <p className="mt-1 max-w-2xl text-sm leading-relaxed text-muted-foreground">
                    {verdictBody}
                  </p>
                  <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1.5 text-sm">
                    <span className="text-foreground">
                      {t.quality.passRate.replace('{value}', pct(summary.pass_rate))}
                    </span>
                    {score.qe_measured ? (
                      <span className="text-muted-foreground">
                        {t.quality.avgQeValue.replace('{score}', Number(score.avg_qe ?? 0).toFixed(3))}
                        {' · '}
                        {t.quality.qeMeasuredOn.replace('{count}', String(score.scored_blocks ?? 0))}
                      </span>
                    ) : (
                      <span className="inline-flex items-center gap-1 text-muted-foreground">
                        {t.quality.qeNotMeasured}
                        <InfoTip label={t.quality.qeNotMeasured}>
                          <span className="block space-y-1">
                            <span className="block">
                              {config.qe_judge_enabled
                                ? t.quality.qeNotMeasuredWhyJudge
                                : t.quality.qeNotMeasuredWhyHeuristic}
                            </span>
                            <span className="block text-muted-foreground">
                              {t.quality.qeNotMeasuredHow}
                            </span>
                          </span>
                        </InfoTip>
                      </span>
                    )}
                  </div>
                </div>
              </div>

            </div>

            {downloadError && (
              <div className="border-t border-destructive/20 bg-destructive/5 px-5 py-2.5 text-sm text-destructive sm:px-6">
                {downloadError}
              </div>
            )}
          </Card>

          <section className="space-y-3">
            <div className="flex items-end justify-between gap-4">
              <div>
                <h2 className="text-base font-semibold text-foreground">{t.quality.checksTitle}</h2>
                <p className="mt-0.5 text-sm text-muted-foreground">
                  {t.quality.deliverySummaryDesc}
                </p>
              </div>
            </div>

            <Card className="overflow-hidden">
              <div className="divide-y divide-border/70">
                {checks.map((check, index) => {
                  const open = expanded === index
                  return (
                    <div key={check.title}>
                      <button
                        type="button"
                        onClick={() => setExpanded(open ? null : index)}
                        aria-expanded={open}
                        className="flex w-full items-start gap-3 px-4 py-3.5 text-left transition-colors hover:bg-muted/25 sm:px-5"
                      >
                        {open ? (
                          <ChevronDown className="mt-1 size-4 shrink-0 text-muted-foreground" />
                        ) : (
                          <ChevronRight className="mt-1 size-4 shrink-0 text-muted-foreground" />
                        )}
                        <div className="min-w-0 flex-1">
                          <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
                            <span className="text-sm font-medium text-foreground">{check.title}</span>
                            <span className={cn('text-xs font-medium', STATUS_TONE[check.status])}>
                              {statusLabel(t, check.status)}
                            </span>
                          </div>
                          <p className="mt-0.5 text-sm leading-relaxed text-muted-foreground">
                            {check.summary}
                          </p>
                          <div className="mt-2 h-1 overflow-hidden rounded-full bg-muted">
                            <div
                              className={cn('h-full rounded-full', STATUS_BAR[check.status])}
                              style={{ width: `${Math.max(3, check.value * 100)}%` }}
                            />
                          </div>
                        </div>
                      </button>
                      {open && (
                        <div className="border-t border-border/50 bg-muted/15 px-4 py-4 text-sm sm:px-5 sm:pl-12">
                          {check.detail}
                        </div>
                      )}
                    </div>
                  )
                })}
              </div>
            </Card>
          </section>

          <section className="space-y-3">
            <div className="flex items-end justify-between gap-4">
              <div>
                <h2 className="text-base font-semibold text-foreground">
                  {t.quality.primaryArtifact}
                </h2>
                <p className="mt-0.5 text-sm text-muted-foreground">
                  {t.quality.primaryArtifactDesc}
                </p>
              </div>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => navigate(`/jobs/${jobId}?tab=deliverables`)}
              >
                {t.quality.viewAllFiles}
                <ArrowUpRight />
              </Button>
            </div>

            {deliveryStale && (
              <div className="rounded-lg border border-amber-500/25 bg-amber-500/10 px-3.5 py-2.5 text-sm text-amber-700 dark:text-amber-400">
                {t.quality.deliveryStaleNote}
              </div>
            )}

            <Card>
              {primaryItem ? (
                <div className="flex flex-col gap-4 p-4 sm:flex-row sm:items-center sm:justify-between sm:p-5">
                  <div className="flex min-w-0 items-start gap-3">
                    <div className="flex size-10 shrink-0 items-center justify-center rounded-lg bg-muted">
                      <FileText className="size-4 text-muted-foreground" />
                    </div>
                    <div className="min-w-0">
                      <div className="text-sm font-medium text-foreground">
                        {deliverableName(primaryItem.key, primaryItem.label, t)}
                      </div>
                      <div className="mt-0.5 truncate font-mono text-xs text-muted-foreground">
                        {primaryItem.filename} · {fmtSize(primaryItem.size_bytes)}
                      </div>
                    </div>
                  </div>
                  <div className="flex shrink-0 gap-2">
                    {primaryPreviewable && (
                      <Button variant="secondary" size="sm" onClick={() => setPreviewItem(primaryItem)}>
                        <Eye />
                        {t.quality.previewAction}
                      </Button>
                    )}
                    <Button
                      size="sm"
                      onClick={() => void handleDownload(primaryItem.key)}
                      disabled={downloadBusy}
                    >
                      <Download />
                      {t.quality.downloadAction}
                    </Button>
                  </div>
                </div>
              ) : (
                <p className="p-5 text-sm text-muted-foreground">{t.quality.primaryProductMissing}</p>
              )}
            </Card>

            {companionArtifacts.length > 0 && (
              <div className="flex flex-wrap items-center gap-2 pt-1">
                <span className="text-xs font-medium text-muted-foreground">
                  {t.mission.companionArtifactsTitle}:
                </span>
                {companionArtifacts.map((comp) => (
                  <span
                    key={comp.kind}
                    className={cn(
                      'inline-flex items-center gap-1 rounded px-2 py-0.5 font-mono text-xs',
                      comp.status === 'generated'
                        ? 'border border-emerald-500/20 bg-emerald-500/10 text-emerald-600 dark:text-emerald-400'
                        : comp.status === 'failed'
                        ? 'border border-destructive/20 bg-destructive/10 text-destructive'
                        : 'border border-border bg-muted text-muted-foreground'
                    )}
                    title={comp.reason || comp.error || `${comp.kind.toUpperCase()} ${comp.status}`}
                  >
                    <span className="font-bold">{comp.kind.toUpperCase()}</span>
                    <span className="opacity-80">
                      {comp.status === 'generated'
                        ? (comp.sizeBytes ? fmtSize(comp.sizeBytes) : t.mission.companionGeneratedBadge)
                        : comp.status === 'skipped'
                        ? t.mission.companionSkippedBadge
                        : t.mission.companionFailedBadge}
                    </span>
                  </span>
                ))}
              </div>
            )}

            <p className="text-xs leading-relaxed text-muted-foreground">
              {t.quality.technicalReportHint}
            </p>
          </section>
        </>
      )}

      {previewItem && (
        <PreviewDrawer
          key={`${previewItem.key}:${previewItem.filename}`}
          jobId={jobId}
          item={previewItem}
          onClose={() => setPreviewItem(null)}
        />
      )}

      {postEditOpen && (
        <PostEditDrawer
          jobId={jobId}
          onClose={() => setPostEditOpen(false)}
          onReExportComplete={() => void loadAll()}
        />
      )}
    </div>
  )
}
