import React, { useCallback, useEffect, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { clsx } from 'clsx'
import {
  ShieldCheck,
  FileCheck2,
  Download,
  ArrowUpRight,
  CheckCircle2,
  AlertTriangle,
  ChevronDown,
  ChevronRight,
  Info,
  Eye,
  RefreshCw,
  Sigma,
  BookOpen,
  LayoutTemplate,
  ListChecks,
  Wallet,
  FileText,
  FileStack,
  Globe,
  Languages,
  ClipboardCheck,
  ScanSearch,
  Activity,
} from 'lucide-react'
import { Button } from '@/components/ui/Button'
import { Badge } from '@/components/ui/Badge'
import { InfoTip } from '@/components/ui/InfoTip'
import {
  getJobReport,
  getVisualReport,
  listDeliverables,
  deliverableDownloadUrl,
  reExportJob,
  pagePreviewUrl,
  ApiError,
  type DeliverableItem,
} from '@/api/client'
import { useI18n } from '@/i18n/useI18n'
import { BarRow, Donut, PageGrid, RadarChart, type RadarAxis } from './charts'
import { PreviewDrawer } from './PreviewDrawer'
import {
  deliverableDesc,
  deliverableGroup,
  deliverableName,
  fmtSize,
  previewKind,
  type DeliverableGroup,
} from './artifactMeta'

type PillarStatus = 'pass' | 'warn' | 'fail'

/** Which density the page renders at. 默认简明：面向"等一本书的人"；专业面向操作员。 */
type GateMode = 'simple' | 'pro'
const MODE_STORAGE = 'ubt.quality.mode'

interface Pillar {
  title: string
  desc: string
  metric: string
  status: PillarStatus
  /** 0-100 axis value for the radar. */
  score: number
  detail: React.ReactNode
}

/** Plain-language name for the QE engine token the report records. */
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

/** The checklist's per-pillar verdict, translated. */
function pillarStatusLabel(
  t: ReturnType<typeof useI18n>['t'],
  status: 'pass' | 'warn' | 'fail'
): string {
  if (status === 'pass') return t.quality.pillarStatusPass
  if (status === 'warn') return t.quality.pillarStatusWarn
  return t.quality.pillarStatusFail
}

const PILLAR_TONE: Record<PillarStatus, string> = {
  pass: 'text-[#15803d] bg-[#15803d]/10 border-[#15803d]/25',
  warn: 'text-[#b45309] bg-[#b45309]/10 border-[#b45309]/25',
  fail: 'text-[#b91c1c] bg-[#b91c1c]/10 border-[#b91c1c]/25',
}

const DELIVERABLE_ICONS: Record<string, React.ComponentType<{ className?: string }>> = {
  primary: FileText,
  secondary: FileStack,
  epub: BookOpen,
  html: Globe,
  xliff: Languages,
  contract: FileCheck2,
  quality_report: ClipboardCheck,
  visual_report: ScanSearch,
  metrics: Activity,
}

const GROUP_ORDER: DeliverableGroup[] = ['reading', 'l10n', 'audit']

export function QualityGateView() {
  const { t } = useI18n()
  const navigate = useNavigate()
  const { jobId: routeJobId } = useParams<{ jobId: string }>()
  const jobId = routeJobId ?? null
  const onOpenReview = (id: string) => navigate(`/jobs/${id}/review`)
  const [report, setReport] = useState<Record<string, any> | null>(null)
  const [reportError, setReportError] = useState<string | null>(null)
  const [deliverables, setDeliverables] = useState<DeliverableItem[]>([])
  // True when the files predate the ledger text. The download endpoint
  // refuses such a job with 409, and `window.open` would show the operator
  // a page of raw JSON instead of a file.
  const [deliveryStale, setDeliveryStale] = useState(false)
  const [downloadBusy, setDownloadBusy] = useState(false)
  const [downloadError, setDownloadError] = useState<string | null>(null)
  const [visual, setVisual] = useState<Record<string, any> | null>(null)
  // Whether the visual report itself is absent (gate disabled, or the
  // deliverable is not a PDF — the endpoint then answers 404/400). Distinct
  // from a loaded report: an absent report is no attestation either way, and
  // must not read as a clean gate.
  const [visualMissing, setVisualMissing] = useState(false)
  const [expanded, setExpanded] = useState<number | null>(null)
  const [previewItem, setPreviewItem] = useState<DeliverableItem | null>(null)
  // 简明 by default: this screen serves "is this book ready?" first and
  // engine forensics second. The choice survives reloads in localStorage.
  const [mode, setMode] = useState<GateMode>(() => {
    try {
      return localStorage.getItem(MODE_STORAGE) === 'pro' ? 'pro' : 'simple'
    } catch {
      return 'simple'
    }
  })
  const switchMode = (next: GateMode) => {
    setMode(next)
    try {
      localStorage.setItem(MODE_STORAGE, next)
    } catch {
      // Storage can be refused (private mode); the toggle still works live.
    }
  }
  // Set when the report fetch answers 404: polling for a report that will
  // never exist must stop, while a 409 (job still running) keeps polling.
  const [jobMissing, setJobMissing] = useState(false)

  const loadAll = useCallback(
    async (isPoll: boolean = false) => {
      if (!jobId) return
      try {
        const rep = await getJobReport(jobId)
        setReport(rep)
        setReportError(null)
      } catch (err) {
        // 404 latches "the report will never exist"; any other outcome —
        // success, 409 while running, a network blip — clears the latch so
        // polling continues. Both setStates run after the await, so the
        // effect body itself triggers no synchronous cascade.
        if (err instanceof ApiError && err.status === 404) setJobMissing(true)
        else setJobMissing(false)
        if (!isPoll) {
          setReportError(err instanceof Error ? err.message : 'Report unavailable')
        }
        return
      }
      try {
        const listing = await listDeliverables(jobId)
        setDeliverables(listing.items)
        setDeliveryStale(listing.stale)
      } catch {
        setDeliverables([])
      }
      try {
        setVisual(await getVisualReport(jobId))
        setVisualMissing(false)
      } catch {
        setVisual(null)
        setVisualMissing(true)
      }
    },
    [jobId]
  )

  useEffect(() => {
    // Mount-time load: every setState inside `loadAll` runs after its first
    // `await`, so there is no synchronous cascade for the rule to be about.
    // oxlint-disable-next-line react/set-state-in-effect
    void loadAll(false)
  }, [loadAll])

  // A page that only ever fetched once would sit on "no verdict" forever if
  // opened while the job was still running: the report appears server-side,
  // and nothing told the reader it had. Poll until the report exists.
  useEffect(() => {
    if (!jobId || report !== null || jobMissing) return
    const timer = setInterval(() => void loadAll(true), 5000)
    return () => clearInterval(timer)
  }, [jobId, report, jobMissing, loadAll])

  if (!jobId) {
    return (
      <div className="flex-1 flex items-center justify-center p-8 text-center text-[var(--ink-muted)]">
        <div className="max-w-sm space-y-2">
          <ShieldCheck className="h-8 w-8 mx-auto text-[var(--paper-border-hover)]" />
          <h2 className="text-sm font-semibold text-[var(--ink-primary)]">{t.quality.noJobTitle}</h2>
          <p className="text-xs text-[var(--ink-secondary)] leading-relaxed">{t.quality.noJobDesc}</p>
        </div>
      </div>
    )
  }

  const summary = report?.summary ?? {}
  const score = report?.score_metrics ?? {}
  const placeholder = report?.placeholder ?? {}
  const terminology = report?.terminology ?? {}
  const entity = report?.entity_consistency ?? {}
  // `report.fidelity` no longer exists — the overlay engine it described was
  // deleted, and nothing in the report schema carries `pages_measured` or
  // `non_text_residual` any more. The real pixel verdict lives on the
  // *visual report* payload, which the route serves beside it.
  const visualSkipped: string | null =
    typeof visual?.skipped_reason === 'string' && visual.skipped_reason
      ? visual.skipped_reason
      : null
  const tiersRan: string[] = Array.isArray(visual?.tiers_ran) ? visual.tiers_ran : []
  const tiersSkipped: string[] = Array.isArray(visual?.tiers_skipped) ? visual.tiers_skipped : []
  const skipCodes: string[] = Array.isArray(visual?.skip_codes) ? visual.skip_codes : []
  /** One translated phrase per skipped tier, e.g. "未运行：语义视觉判定". */
  const tierLabel = (tier: string): string =>
    (t.quality.visualTierNames as Record<string, string>)[tier] ?? tier
  const skipLabel = (code: string): string =>
    (t.quality.visualSkipCodes as Record<string, string>)[code] ?? code
  const coverage = report?.render_coverage ?? {}
  const repair = report?.repair_breakdown ?? {}
  const config = report?.config_snapshot ?? {}

  const failedBlocks = summary.failed_blocks ?? 0
  // The gate decision is the shared three-state verdict the backend computes
  // (`job_options.delivery_verdict_from_summary` — the same rule the CLI banner
  // applies), falling back to failed-blocks-only for reports served before the
  // field existed.
  const verdict: string | null = report?.delivery_verdict ?? null
  // A rehearsal's artifact is echo text, not a translation. Every structural
  // gate passes it by construction, so the verdict must never read as
  // "PRODUCTION READY" over `[模拟翻译]` text.
  const rehearsal: boolean = report?.rehearsal === true
  // No report means NO VERDICT — never a pass. The report fetch legitimately
  // 409s while a job is still running and 404s before the reporter writes it,
  // so a bare `failed_blocks === 0` fallback attested nothing.
  const hasReport = report !== null
  const isPassed = hasReport && (verdict ? verdict === 'passed' : failedBlocks === 0)
  const isWarning = hasReport && verdict === 'warning'
  // ``null`` means "not priced", never "$0" (see ReportSummary). The ledger
  // pillar is the only place that distinction is visible.
  const costPriced = hasReport && typeof summary.estimated_cost_usd === 'number'

  const pct = (value: number | undefined, digits = 1) =>
    value === undefined ? '—' : `${(value * 100).toFixed(digits)}%`

  const clampScore = (value: number | undefined, fallback = 100) =>
    value === undefined || Number.isNaN(value) ? fallback : Math.min(100, Math.max(0, value * 100))

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

  const pillarStatus = (bad: boolean, warn = false): PillarStatus =>
    bad ? 'fail' : warn ? 'warn' : 'pass'

  // The visual report's honest one-line state: which tiers ran, which did not.
  // The old "像素校验已跳过" claimed the whole pixel gate was skipped whenever
  // *any* tier was — for a job that ran four tiers over every page, that
  // misdescribed a near-clean pass as "never ran" on the attestation screen.
  const visualMetric = visualMissing
    ? t.quality.visualNoReport
    : visualSkipped
      ? tiersRan.length > 0
        ? t.quality.visualPartialRan
            .replace('{ran}', String(tiersRan.length))
            .replace('{skipped}', tiersSkipped.map(tierLabel).join('、'))
        : t.quality.visualGateSkipped
      : `${flaggedPages.length} ${t.quality.flaggedPages}`

  const pillar3 = {
    metric: visualMetric,
    // A skipped gate is not a passing gate, and neither is an absent one.
    status:
      visualMissing || visualSkipped
        ? ('warn' as PillarStatus)
        : pillarStatus(
            flaggedPages.length > 0 ||
              (typeof visual?.passed === 'boolean' && visual.passed === false),
            false
          ),
  }

  const pillars: Pillar[] = [
    {
      title: t.quality.pillar1Title,
      desc: t.quality.pillar1Desc,
      metric: `${placeholder.corrupt_spans ?? 0} corrupt / ${placeholder.masked_spans ?? 0} masked (retention ${pct(placeholder.retention_rate)})`,
      status: (placeholder.corrupt_spans ?? 0) > 0 ? 'fail' : 'pass',
      score: clampScore(placeholder.retention_rate),
      detail: (
        <dl className="grid grid-cols-1 sm:grid-cols-2 gap-x-6 gap-y-1 text-xs font-mono">
          <div className="flex justify-between">
            <dt className="text-[var(--ink-muted)]">{t.quality.metricMaskedSpans}</dt>
            <dd>{placeholder.masked_spans ?? 0}</dd>
          </div>
          <div className="flex justify-between">
            <dt className="text-[var(--ink-muted)]">{t.quality.metricCorruptSpans}</dt>
            <dd>{placeholder.corrupt_spans ?? 0}</dd>
          </div>
          <div className="flex justify-between">
            <dt className="text-[var(--ink-muted)]">{t.quality.metricMaskedBlocks}</dt>
            <dd>{placeholder.masked_blocks ?? 0}</dd>
          </div>
          <div className="flex justify-between">
            <dt className="text-[var(--ink-muted)]">{t.quality.metricCorruptBlocks}</dt>
            <dd>{placeholder.corrupt_blocks ?? 0}</dd>
          </div>
          <div className="flex justify-between">
            <dt className="text-[var(--ink-muted)]">{t.quality.metricFormulaBlocks}</dt>
            <dd>{report?.formula_blocks ?? 0}</dd>
          </div>
        </dl>
      ),
    },
    {
      title: t.quality.pillar2Title,
      desc: t.quality.pillar2Desc,
      metric: `recall ${pct(terminology.term_recall)} · drift ${entity.terms_with_drift ?? 0}/${entity.terms_audited ?? 0}`,
      status: (entity.terms_with_drift ?? 0) > 0 ? 'warn' : 'pass',
      score: clampScore(terminology.term_recall),
      detail:
        topDrifted.length === 0 ? (
          <div className="text-xs text-[var(--ink-muted)]">{t.quality.noDrift}</div>
        ) : (
          <div className="space-y-1.5">
            <div className="text-xs font-mono uppercase tracking-wider text-[var(--ink-muted)] font-semibold">
              {t.quality.termDriftTitle}
            </div>
            <div className="overflow-x-auto min-w-0">
              <table className="w-full min-w-[340px] text-xs text-left font-mono">
                <thead className="text-[var(--ink-muted)]">
                  <tr>
                    <th className="py-1 pr-3 font-semibold">{t.quality.colTerm}</th>
                    <th className="py-1 pr-3 font-semibold">{t.quality.colExpected}</th>
                    <th className="py-1 pr-3 font-semibold">{t.quality.colOccurrences}</th>
                    <th className="py-1 font-semibold">{t.quality.colDriftRate}</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[var(--paper-border)]">
                  {topDrifted.map((row) => (
                    <tr key={`${row.source}-${row.expected}`}>
                      <td className="py-1 pr-3 text-[var(--ink-primary)]">{row.source}</td>
                      <td className="py-1 pr-3 text-[#15803d]">{row.expected}</td>
                      <td className="py-1 pr-3">{row.occurrences ?? 0}</td>
                      <td className="py-1 text-[#b45309]">
                        {typeof row.drift_rate === 'number'
                          ? `${(row.drift_rate * 100).toFixed(0)}%`
                          : '—'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        ),
    },
    {
      title: t.quality.pillar3Title,
      desc: t.quality.pillar3Desc,
      metric: pillar3.metric,
      status: pillar3.status,
      score: visualMissing
        ? 0
        : visualSkipped
          ? 0
          : typeof visual?.passed === 'boolean' && !visual.passed
            ? 55
            : flaggedPages.length > 0
              ? 60
              : 100,
      detail: (
        <div className="space-y-2">
          {visualMissing && (
            <div className="text-xs text-[var(--ink-secondary)]">
              {t.quality.visualNoReportDesc}
            </div>
          )}
          {tiersRan.length > 0 && (
            <div className="text-xs text-[var(--ink-secondary)]">
              <span className="text-[var(--ink-muted)]">{t.quality.visualTiersRan}: </span>
              {tiersRan.map(tierLabel).join(' · ')}
              {tiersSkipped.length > 0 && (
                <>
                  <span className="text-[var(--ink-muted)]">
                    {'  ·  '}
                    {t.quality.visualTiersSkipped}:{' '}
                  </span>
                  {tiersSkipped.map(tierLabel).join(' · ')}
                  {skipCodes.length > 0 && (
                    <span className="text-[var(--ink-muted)]">
                      {' '}
                      ({skipCodes.map(skipLabel).join('、')})
                    </span>
                  )}
                </>
              )}
            </div>
          )}
          {visualFindings.length === 0 ? (
            <div className="text-xs text-[var(--ink-muted)]">{t.quality.noVisualFindings}</div>
          ) : (
            <>
              <div className="text-xs font-mono uppercase tracking-wider text-[var(--ink-muted)] font-semibold">
                {t.quality.visualFindingsTitle}
              </div>
              <ul className="space-y-1 text-xs max-h-56 overflow-y-auto">
                {visualFindings.map((finding, index) => (
                  <li key={index} className="flex gap-2">
                    <span className="font-mono text-[var(--ink-muted)] shrink-0">
                      {finding.page != null ? `P${finding.page}` : '—'}
                    </span>
                    <span className="text-[var(--ink-secondary)]">
                      <span className="font-mono text-[#b45309]">{finding.code}</span>{' '}
                      {finding.message}
                    </span>
                  </li>
                ))}
              </ul>
            </>
          )}
          {flaggedPages.length > 0 && (
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-1 max-h-72 overflow-y-auto">
              {flaggedPages.map((page) => (
                <div
                  key={page}
                  className="rounded-[4px] border border-[var(--paper-border)] overflow-hidden bg-white"
                >
                  <img
                    src={pagePreviewUrl(jobId, page, { dpi: 70 })}
                    alt={t.quality.pageAlt.replace('{n}', String(page))}
                    className="w-full h-24 object-contain"
                    onError={(e) => {
                      e.currentTarget.style.display = 'none'
                    }}
                  />
                  <div className="text-center text-xs font-mono text-[var(--ink-muted)] py-0.5 border-t border-[var(--paper-border)]">
                    P{page}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      ),
    },
    {
      title: t.quality.pillar4Title,
      desc: t.quality.pillar4Desc,
      metric: `fail-closed ${coverage.fail_closed_blocks ?? 0} · coverage ${pct(coverage.render_coverage)}`,
      status: (coverage.fail_closed_blocks ?? 0) > 0 ? 'fail' : 'pass',
      score: clampScore(coverage.render_coverage),
      detail: (
        <dl className="grid grid-cols-1 sm:grid-cols-2 gap-x-6 gap-y-1 text-xs font-mono">
          {[
            ['rendered_blocks', coverage.rendered_blocks ?? 0],
            ['skipped_blocks', coverage.skipped_blocks ?? 0],
            ['fail_closed_blocks', coverage.fail_closed_blocks ?? 0],
            ['preserved_blocks', coverage.preserved_blocks ?? 0],
          ].map(([label, value]) => (
            <div key={String(label)} className="flex justify-between">
              <dt className="text-[var(--ink-muted)]">{label}</dt>
              <dd>{value}</dd>
            </div>
          ))}
        </dl>
      ),
    },
    {
      title: t.quality.pillar5Title,
      desc: t.quality.pillar5Desc,
      metric:
        summary.estimated_cost_usd === null || summary.estimated_cost_usd === undefined
          ? t.quality.costNotPriced
          : `$${Number(summary.estimated_cost_usd).toFixed(2)} · cache ${pct(summary.cache_hit_rate)}`,
      // This pillar never failed before: a constant `pass` / 100 renders a green
      // check over an unpriced run and over a ledger nobody reconciled. The two
      // states it can actually establish are "the run was priced" and "it was
      // not"; the reconciliation against the pre-flight quote is not
      // implemented anywhere in the engine, so it is reported as a warn rather
      // than dressed up as a passed check.
      status: costPriced ? 'pass' : 'warn',
      score: costPriced ? 100 : 50,
      detail: (
        <dl className="grid grid-cols-1 sm:grid-cols-2 gap-x-6 gap-y-1 text-xs font-mono">
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
            <div key={String(label)} className="flex justify-between gap-3">
              <dt className="text-[var(--ink-muted)] shrink-0">{label}</dt>
              <dd className="truncate text-right">{value}</dd>
            </div>
          ))}
        </dl>
      ),
    },
  ]

  const radarAxes: RadarAxis[] = [
    { label: t.quality.axisMath, value: pillars[0].score, measured: true },
    { label: t.quality.axisTerminology, value: pillars[1].score, measured: true },
    // A gate that never ran is not a zero: the radar chords across it instead
    // of drawing a failure spike, and the label drops its number.
    {
      label: t.quality.axisVisual,
      value: pillars[2].score,
      measured: !(visualMissing || visualSkipped),
    },
    { label: t.quality.axisCompleteness, value: pillars[3].score, measured: true },
    { label: t.quality.axisDelivered, value: clampScore(summary.pass_rate), measured: true },
  ]
  const radarHasUnmeasured = radarAxes.some((axis) => axis.measured === false)

  // A stale delivery is re-rendered before the download, the same way Mission
  // Control does it: the backend guard exists so a customer never receives the
  // pre-edit file, and the console's job is to make that guard invisible rather
  // than to hand the operator a 409 they cannot act on.
  const handleDownload = async (key: string) => {
    if (downloadBusy) return
    setDownloadBusy(true)
    // The tab is opened inside the click gesture, before any await: a stale
    // delivery re-renders first (whole-book, easily past the browser's ~5 s
    // transient-activation window), and a `window.open` after that wait is
    // silently swallowed by the popup blocker.
    const tab = window.open('', '_blank')
    try {
      if (deliveryStale) {
        await reExportJob(jobId)
        const refreshed = await listDeliverables(jobId)
        setDeliverables(refreshed.items)
        setDeliveryStale(refreshed.stale)
      }
      if (tab) tab.location.href = deliverableDownloadUrl(jobId, key)
      else window.open(deliverableDownloadUrl(jobId, key), '_blank')
    } catch (err) {
      tab?.close()
      setDownloadError(err instanceof Error ? err.message : t.quality.downloadFailed)
    } finally {
      setDownloadBusy(false)
    }
  }

  const statusBadge = (status: PillarStatus) =>
    status === 'pass' ? 'success' : status === 'warn' ? 'warning' : 'destructive'

  // ----------------------------------------------------------------------- //
  // 简明视图 pieces: five human checks, each with a chart and one sentence.
  // ----------------------------------------------------------------------- //
  const checkChips = [
    { Icon: Sigma, label: t.quality.chipMath, status: pillars[0].status },
    { Icon: BookOpen, label: t.quality.chipTerm, status: pillars[1].status },
    { Icon: LayoutTemplate, label: t.quality.chipLayout, status: pillars[2].status },
    { Icon: ListChecks, label: t.quality.chipComplete, status: pillars[3].status },
    { Icon: Wallet, label: t.quality.chipCost, status: pillars[4].status },
  ]
  const simpleTitle = rehearsal
    ? t.quality.rehearsalBadge
    : isPassed
      ? t.quality.simpleReadyTitle
      : isWarning
        ? t.quality.simpleWarningTitle
        : t.quality.simpleBlockedTitle
  const simpleBody = rehearsal
    ? t.quality.rehearsalBody
    : isPassed
      ? t.quality.simpleReadyBody
      : isWarning
        ? t.quality.simpleWarningBody
        : t.quality.simpleBlockedBody.replace('{failed}', String(failedBlocks))

  const visualStats = visual?.stats ?? {}
  const simpleCards = [
    {
      Icon: Sigma,
      title: t.quality.chipMath,
      status: pillars[0].status,
      chart: (
        <Donut
          value={(clampScore(placeholder.retention_rate)) / 100}
          tone={pillars[0].status === 'fail' ? 'fail' : pillars[0].status === 'warn' ? 'warn' : 'pass'}
          center={`${Math.round(clampScore(placeholder.retention_rate))}%`}
        />
      ),
      line: t.quality.simpleMathLine
        .replace('{masked}', String(placeholder.masked_spans ?? 0))
        .replace('{corrupt}', String(placeholder.corrupt_spans ?? 0)),
      extra: null,
    },
    {
      Icon: BookOpen,
      title: t.quality.chipTerm,
      status: pillars[1].status,
      chart: (
        <div className="flex-1 min-w-[120px] self-center">
          <BarRow
            label={t.quality.delivRecall}
            value={typeof terminology.term_recall === 'number' ? terminology.term_recall : 0}
            tone={pillars[1].status === 'warn' ? 'warn' : 'pass'}
            display={pct(terminology.term_recall)}
          />
        </div>
      ),
      line: t.quality.simpleTermLine
        .replace('{audited}', String(entity.terms_audited ?? 0))
        .replace('{drift}', String(entity.terms_with_drift ?? 0)),
      extra: null,
    },
    {
      Icon: LayoutTemplate,
      title: t.quality.chipLayout,
      status: pillars[2].status,
      chart: null,
      line: visualMissing
        ? t.quality.simpleVisualMissingLine
        : visualFindings.length > 0
          ? t.quality.simpleVisualFindingsLine.replace('{count}', String(flaggedPages.length))
          : visualSkipped
            ? t.quality.simpleVisualSkippedLine.replace('{ran}', String(tiersRan.length))
            : t.quality.simpleVisualCleanLine.replace(
                '{pages}',
                String(visualStats.sampled ?? sampledPages.length)
              ),
      extra: (
        <div className="space-y-2">
          {sampledPages.length > 0 && !visualMissing && (
            <PageGrid pages={sampledPages} findingsByPage={findingsByPage} />
          )}
          {flaggedPages.length > 0 && (
            <div className="grid grid-cols-2 gap-2 max-h-40 overflow-y-auto">
              {flaggedPages.map((page) => (
                <div
                  key={page}
                  className="rounded-[4px] border border-[var(--paper-border)] overflow-hidden bg-white"
                >
                  <img
                    src={pagePreviewUrl(jobId, page, { dpi: 70 })}
                    alt={t.quality.pageAlt.replace('{n}', String(page))}
                    className="w-full h-20 object-contain"
                    onError={(e) => {
                      e.currentTarget.style.display = 'none'
                    }}
                  />
                  <div className="text-center text-[10px] font-mono text-[var(--ink-muted)] py-0.5 border-t border-[var(--paper-border)]">
                    P{page}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      ),
    },
    {
      Icon: ListChecks,
      title: t.quality.chipComplete,
      status: pillars[3].status,
      chart: (
        <Donut
          value={(clampScore(coverage.render_coverage)) / 100}
          tone={pillars[3].status === 'fail' ? 'fail' : 'pass'}
          center={`${Math.round(clampScore(coverage.render_coverage))}%`}
        />
      ),
      line:
        (coverage.fail_closed_blocks ?? 0) > 0
          ? t.quality.simpleCompleteFailLine.replace(
              '{failClosed}',
              String(coverage.fail_closed_blocks ?? 0)
            )
          : t.quality.simpleCompleteLine.replace(
              '{skipped}',
              String(coverage.skipped_blocks ?? 0)
            ),
      extra: null,
    },
    {
      Icon: Wallet,
      title: t.quality.chipCost,
      status: pillars[4].status,
      chart: (
        <div className="flex-1 min-w-[120px] self-center">
          <BarRow
            label={t.quality.auditCacheHit}
            value={typeof summary.cache_hit_rate === 'number' ? summary.cache_hit_rate : 0}
            tone="muted"
            display={pct(summary.cache_hit_rate)}
          />
        </div>
      ),
      line: costPriced
        ? t.quality.simpleCostLine
            .replace('{cost}', `$${Number(summary.estimated_cost_usd).toFixed(2)}`)
            .replace('{cache}', pct(summary.cache_hit_rate))
        : t.quality.simpleCostUnpricedLine,
      extra: null,
    },
  ]

  const primaryItem = deliverables.find((item) => item.key === 'primary')
  const primaryPreviewable = primaryItem !== undefined && previewKind(primaryItem) === 'embed'

  const groupMeta: Record<DeliverableGroup, { title: string; desc: string }> = {
    reading: { title: t.quality.delivGroupReading, desc: t.quality.delivGroupReadingDesc },
    l10n: { title: t.quality.delivGroupL10n, desc: t.quality.delivGroupL10nDesc },
    audit: { title: t.quality.delivGroupAudit, desc: t.quality.delivGroupAuditDesc },
  }

  const modeToggle = (
    <div
      role="group"
      aria-label={t.quality.modeLabel}
      className="inline-flex rounded-[5px] border border-[var(--paper-border)] bg-[var(--paper-subsurface)] p-0.5"
    >
      {(['simple', 'pro'] as const).map((m) => (
        <button
          key={m}
          onClick={() => switchMode(m)}
          aria-pressed={mode === m}
          className={clsx(
            'h-7 px-2.5 text-xs rounded-[4px] transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-[var(--ink-primary)]',
            mode === m
              ? 'bg-[var(--paper-surface)] shadow-sm font-semibold text-[var(--ink-primary)]'
              : 'text-[var(--ink-secondary)] hover:text-[var(--ink-primary)]'
          )}
        >
          {m === 'simple' ? t.quality.modeSimple : t.quality.modePro}
        </button>
      ))}
    </div>
  )

  return (
    <div className="flex-1 overflow-y-auto px-4 sm:px-8 py-5 sm:py-7 space-y-5 sm:space-y-7 max-w-5xl mx-auto w-full">
      {/* View Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-[var(--paper-border)] pb-4">
        <div>
          <h1 className="text-xl font-bold tracking-tight text-[var(--ink-primary)]">
            {t.quality.title}
          </h1>
          <p className="text-xs text-[var(--ink-secondary)] mt-1">{t.quality.subtitle}</p>
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          {modeToggle}
          <Button onClick={() => onOpenReview(jobId)} variant="secondary" size="sm">
            {t.mission.openWorkbench}
            <ArrowUpRight className="h-3.5 w-3.5 ml-1 text-[var(--ink-secondary)]" />
          </Button>
        </div>
      </div>

      {reportError && !report && (
        <div className="p-3 rounded-lg border border-[var(--paper-border)] bg-[var(--paper-surface)] text-xs text-[var(--ink-secondary)] flex items-center gap-2">
          <AlertTriangle className="h-4 w-4 text-[#b45309] shrink-0" />
          {reportError}
        </div>
      )}

      {/* No report yet: say so, and do not render a verdict block at all.
          The pillars below read from the same absent payload, so they would
          otherwise all render as passing. */}
      {!hasReport && (
        <div className="p-5 rounded-lg border border-[var(--paper-border)] bg-[var(--paper-surface)] flex items-start gap-3 shadow-2xs">
          <AlertTriangle className="h-5 w-5 text-[#b45309] shrink-0 mt-0.5" />
          <div className="space-y-2">
            <div className="text-sm font-semibold text-[var(--ink-primary)]">
              {t.quality.noVerdictTitle}
            </div>
            <p className="text-xs text-[var(--ink-secondary)] leading-relaxed">
              {t.quality.noVerdictBody}
            </p>
            {!jobMissing && (
              <Button variant="secondary" size="sm" onClick={() => void loadAll(false)}>
                <RefreshCw className="h-3.5 w-3.5" />
                {t.quality.refreshBtn}
              </Button>
            )}
          </div>
        </div>
      )}

      {hasReport && (
        mode === 'simple' && (
          <>
          {/* 简明结论卡: one verdict, five check chips, one obvious next action. */}
          <div className="p-4 sm:p-5 rounded-lg border border-[var(--paper-border)] bg-[var(--paper-surface)] flex flex-col lg:flex-row gap-4 lg:gap-6 shadow-2xs">
            <div className="flex items-start gap-3.5 flex-1 min-w-0">
              <div
                className={clsx(
                  'h-10 w-10 rounded-[6px] border flex items-center justify-center shrink-0',
                  rehearsal
                    ? 'text-[#1d4ed8] bg-[#1d4ed8]/5 border-[#1d4ed8]/30'
                    : isPassed
                      ? 'text-[#15803d] bg-[#15803d]/10 border-[#15803d]/25'
                      : isWarning
                        ? 'text-[#b45309] bg-[#b45309]/10 border-[#b45309]/25'
                        : 'text-[#b91c1c] bg-[#b91c1c]/10 border-[#b91c1c]/25'
                )}
              >
                {isPassed && !rehearsal ? (
                  <CheckCircle2 className="h-6 w-6" />
                ) : (
                  <AlertTriangle className="h-6 w-6" />
                )}
              </div>
              <div className="min-w-0">
                <div className="text-base font-bold text-[var(--ink-primary)] leading-snug">
                  {simpleTitle}
                </div>
                <p className="text-xs text-[var(--ink-secondary)] mt-1 leading-relaxed max-w-xl">
                  {simpleBody}
                </p>
                <div className="flex flex-wrap gap-1.5 mt-3">
                  {checkChips.map(({ Icon, label, status }) => (
                    <span
                      key={label}
                      className={clsx(
                        'inline-flex items-center gap-1 rounded-[4px] border px-1.5 py-0.5 text-[11px] font-medium',
                        PILLAR_TONE[status]
                      )}
                    >
                      <Icon className="h-3 w-3" />
                      {label}
                    </span>
                  ))}
                </div>
                {!score.qe_measured && (
                  <p className="text-xs text-[var(--ink-muted)] mt-2.5 leading-relaxed flex items-start gap-1">
                    <span className="inline-flex items-center gap-1">
                      <Info className="h-3 w-3 shrink-0" />
                      {t.quality.qeSimpleNote}
                    </span>
                    <InfoTip label={t.quality.qeNotMeasured}>
                      <span className="block space-y-1">
                        <span className="block">
                          {config.qe_judge_enabled
                            ? t.quality.qeNotMeasuredWhyJudge
                            : t.quality.qeNotMeasuredWhyHeuristic}
                        </span>
                        <span className="block text-[var(--ink-muted)]">
                          {t.quality.qeNotMeasuredHow}
                        </span>
                      </span>
                    </InfoTip>
                  </p>
                )}
              </div>
            </div>
            {primaryPreviewable && (
              <div className="flex lg:flex-col gap-2 lg:justify-center shrink-0">
                <Button variant="primary" size="md" onClick={() => setPreviewItem(primaryItem!)}>
                  <Eye className="h-3.5 w-3.5" />
                  {t.quality.previewTranslation}
                </Button>
              </div>
            )}
          </div>

          {/* 五张检查卡：图标 + 图表 + 一句人话。 */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
            {simpleCards.map(({ Icon, title, status, chart, line, extra }) => (
              <div
                key={title}
                className="rounded-lg border border-[var(--paper-border)] bg-[var(--paper-surface)] p-3.5 flex flex-col gap-2.5 shadow-2xs"
              >
                <div className="flex items-center justify-between gap-2">
                  <div className="flex items-center gap-2 min-w-0">
                    <span
                      className={clsx(
                        'h-6 w-6 rounded-[4px] border flex items-center justify-center shrink-0',
                        PILLAR_TONE[status]
                      )}
                    >
                      <Icon className="h-3.5 w-3.5" />
                    </span>
                    <span className="text-xs font-semibold text-[var(--ink-primary)] truncate">
                      {title}
                    </span>
                  </div>
                  <Badge variant={statusBadge(status)} dot className="shrink-0">
                    {pillarStatusLabel(t, status)}
                  </Badge>
                </div>
                {chart}
                <p className="text-xs text-[var(--ink-secondary)] leading-relaxed">{line}</p>
                {extra}
              </div>
            ))}
            <button
              onClick={() => switchMode('pro')}
              className="rounded-lg border border-dashed border-[var(--paper-border)] hover:border-[var(--paper-border-hover)] p-3.5 flex items-center justify-center gap-2 text-xs text-[var(--ink-muted)] hover:text-[var(--ink-secondary)] transition-colors text-left"
            >
              {t.quality.switchToPro}
              <ArrowUpRight className="h-3.5 w-3.5 shrink-0" />
            </button>
          </div>
        </>
      ))}

      {hasReport && (
        mode === 'pro' && (
          <>
          {/* Decision Banner + Radar */}
          <div className="p-4 sm:p-5 rounded-lg border border-[var(--paper-border)] bg-[var(--paper-surface)] flex flex-col md:flex-row items-start md:items-center justify-between gap-5 md:gap-6 shadow-2xs">
            <div className="flex items-start sm:items-center gap-3.5">
              <div
                className={`h-10 w-10 rounded-[6px] bg-[var(--paper-subsurface)] border border-[var(--paper-border)] flex items-center justify-center shrink-0 ${
                  isPassed ? 'text-[#15803d]' : isWarning ? 'text-[#b45309]' : 'text-[#b91c1c]'
                }`}
              >
                {isPassed ? <CheckCircle2 className="h-6 w-6" /> : <AlertTriangle className="h-6 w-6" />}
              </div>
              <div>
                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-sm font-bold text-[var(--ink-primary)]">
                    {isPassed
                      ? t.quality.decisionReady
                      : isWarning
                        ? t.quality.decisionWarning
                        : t.quality.decisionBlocked}
                  </span>
                  <Badge
                    variant={isPassed ? 'success' : isWarning ? 'warning' : 'destructive'}
                    dot
                  >
                    {isPassed
                      ? t.quality.verdictPass
                      : isWarning
                        ? t.quality.verdictWarning
                        : t.quality.verdictBlocked}
                  </Badge>
                </div>
                <p className="text-xs text-[var(--ink-secondary)] mt-0.5">
                  {isPassed
                    ? t.quality.allPassedDesc
                    : isWarning
                      ? t.quality.warningDesc
                      : t.quality.blockedDesc}
                </p>
                <div className="text-xs font-mono text-[var(--ink-muted)] mt-1.5">
                  {t.quality.passRate.replace('{value}', pct(summary.pass_rate))} ·{' '}
                  {score.qe_measured ? (
                    <span>
                      {t.quality.avgQeValue.replace('{score}', Number(score.avg_qe ?? 0).toFixed(3))}
                      {' · '}
                      {t.quality.qeMeasuredOn.replace(
                        '{count}',
                        String(score.scored_blocks ?? 0)
                      )}
                    </span>
                  ) : (
                    <span className="uppercase" title={t.quality.qeNotMeasuredHint}>
                      {t.quality.qeNotMeasured}
                    </span>
                  )}
                </div>
                {!score.qe_measured && (
                  <p className="text-xs text-[var(--ink-secondary)] mt-1.5 leading-relaxed max-w-xl">
                    {config.qe_judge_enabled
                      ? t.quality.qeNotMeasuredWhyJudge
                      : t.quality.qeNotMeasuredWhyHeuristic}{' '}
                    <span className="text-[var(--ink-muted)]">{t.quality.qeNotMeasuredHow}</span>
                  </p>
                )}
              </div>
            </div>

            <div className="flex flex-col items-center justify-center md:justify-end gap-1 w-full md:w-auto pt-3 md:pt-0 border-t md:border-t-0 border-[var(--paper-border)]">
              <div className="text-xs font-mono uppercase tracking-wider text-[var(--ink-muted)] font-semibold text-center">
                {t.quality.fidelityScore}
              </div>
              <div className="text-2xl font-bold font-mono text-[var(--ink-primary)]">
                {Math.round((summary.pass_rate ?? 0) * 1000) / 10}{' '}
                <span className="text-xs text-[var(--ink-muted)] font-normal">/ 100</span>
              </div>
              <RadarChart axes={radarAxes} />
              {radarHasUnmeasured && (
                <div className="text-[11px] text-[var(--ink-muted)] text-center max-w-[220px] leading-snug">
                  {t.quality.radarUnmeasuredNote}
                </div>
              )}
            </div>
          </div>

          {/* Formal Verification Checklist */}
          <div className="rounded-lg border border-[var(--paper-border)] bg-[var(--paper-surface)] overflow-hidden shadow-2xs">
            <div className="px-4 py-2.5 border-b border-[var(--paper-border)] bg-[var(--paper-subsurface)] text-xs font-mono uppercase tracking-wider text-[var(--ink-muted)] font-semibold">
              {t.quality.pillarsTitle}
            </div>
            <div className="divide-y divide-[var(--paper-border)]">
              {pillars.map((item, idx) => (
                <div key={idx}>
                  <button
                    onClick={() => setExpanded(expanded === idx ? null : idx)}
                    className="w-full p-3.5 sm:p-4 flex flex-col sm:flex-row sm:items-center justify-between gap-2.5 sm:gap-4 hover:bg-[var(--paper-subsurface)] transition-colors text-left"
                  >
                    <div className="flex items-start gap-2.5 min-w-0 flex-1">
                      {expanded === idx ? (
                        <ChevronDown className="h-4 w-4 mt-0.5 text-[var(--ink-muted)] shrink-0" />
                      ) : (
                        <ChevronRight className="h-4 w-4 mt-0.5 text-[var(--ink-muted)] shrink-0" />
                      )}
                      <div className="min-w-0">
                        <div className="text-xs font-semibold text-[var(--ink-primary)]">
                          {item.title}
                        </div>
                        <div className="text-xs text-[var(--ink-secondary)] mt-0.5 leading-relaxed">
                          {item.desc}
                        </div>
                      </div>
                    </div>
                    {/* The metric is a *sentence* for some pillars, so it wraps
                        rather than squeezing the status badge. */}
                    <div className="flex items-center gap-3 shrink-0 w-full sm:w-auto sm:max-w-[45%] justify-between sm:justify-end pl-6 sm:pl-0">
                      <span className="font-mono text-xs text-[var(--ink-primary)] font-medium text-left sm:text-right leading-snug break-words">
                        {item.metric}
                      </span>
                      <Badge variant={statusBadge(item.status)} dot className="shrink-0 whitespace-nowrap">
                        {pillarStatusLabel(t, item.status)}
                      </Badge>
                    </div>
                  </button>
                  {expanded === idx && (
                    <div className="px-4 pb-4 pl-8 sm:pl-11 text-[var(--ink-primary)] overflow-x-auto">{item.detail}</div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </>
      ))}

      {/* Deliverables — grouped by who each file is for, with in-console
          previews instead of a flat wall of identical download buttons. */}
      <div className="rounded-lg border border-[var(--paper-border)] bg-[var(--paper-surface)] p-4 sm:p-5 space-y-4 shadow-2xs">
        <div>
          <h3 className="text-xs font-bold text-[var(--ink-primary)] tracking-tight">
            {t.quality.deliverablesTitle}
          </h3>
          <p className="text-xs text-[var(--ink-secondary)] mt-0.5">
            {t.quality.deliverablesSubtitle}
          </p>
        </div>

        {deliveryStale && (
          <div className="rounded border border-[#b45309]/30 bg-[#b45309]/5 px-2.5 py-1.5 text-xs text-[#b45309]">
            {t.quality.deliveryStaleNote}
          </div>
        )}
        {downloadError && (
          <div className="rounded border border-[#b91c1c]/30 bg-[#b91c1c]/5 px-2.5 py-1.5 text-xs text-[#b91c1c]">
            {downloadError}
          </div>
        )}
        {deliverables.length === 0 ? (
          <div className="text-xs text-[var(--ink-muted)] py-2">
            {t.quality.noDeliverables}
          </div>
        ) : (
          <div className="space-y-4">
            {GROUP_ORDER.map((group) => {
              const items = deliverables.filter((item) => deliverableGroup(item.key) === group)
              if (items.length === 0) return null
              return (
                <div key={group}>
                  <div className="flex items-baseline gap-2 mb-1">
                    <h4 className="text-xs font-semibold text-[var(--ink-primary)]">
                      {groupMeta[group].title}
                    </h4>
                    <span className="text-[11px] text-[var(--ink-muted)]">
                      {groupMeta[group].desc}
                    </span>
                  </div>
                  <div className="divide-y divide-[var(--paper-border)]">
                    {items.map((item) => {
                      const Icon = DELIVERABLE_ICONS[item.key] ?? FileText
                      const kind = previewKind(item)
                      const isPrimary = item.key === 'primary'
                      return (
                        <div
                          key={item.key}
                          className="flex flex-col sm:flex-row sm:items-center gap-2 py-2.5"
                        >
                          <div className="flex items-start gap-2.5 min-w-0 flex-1">
                            <Icon
                              className={clsx(
                                'h-4 w-4 mt-0.5 shrink-0',
                                item.key === 'contract' ? 'text-[#15803d]' : 'text-[var(--ink-secondary)]'
                              )}
                            />
                            <div className="min-w-0">
                              <div className="flex items-baseline gap-2 flex-wrap">
                                <span className="text-xs font-semibold text-[var(--ink-primary)]">
                                  {deliverableName(item.key, item.label, t)}
                                </span>
                                <span className="text-[11px] font-mono text-[var(--ink-muted)]">
                                  {item.filename} · {fmtSize(item.size_bytes)}
                                </span>
                              </div>
                              <div className="text-xs text-[var(--ink-secondary)] mt-0.5 leading-relaxed">
                                {deliverableDesc(item.key, t)}
                              </div>
                            </div>
                          </div>
                          <div className="flex items-center gap-2 shrink-0 pl-6 sm:pl-0">
                            {kind !== null && (
                              <Button
                                variant="secondary"
                                size="sm"
                                onClick={() => setPreviewItem(item)}
                              >
                                <Eye className="h-3.5 w-3.5" />
                                {t.quality.previewAction}
                              </Button>
                            )}
                            <Button
                              variant={isPrimary ? 'primary' : 'outline'}
                              size="sm"
                              disabled={downloadBusy}
                              onClick={() => void handleDownload(item.key)}
                            >
                              <Download className="h-3.5 w-3.5" />
                              {t.quality.downloadAction}
                            </Button>
                          </div>
                        </div>
                      )
                    })}
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </div>

      {previewItem !== null && (
        <PreviewDrawer
          key={`${previewItem.key}:${previewItem.filename}`}
          jobId={jobId}
          item={previewItem}
          onClose={() => setPreviewItem(null)}
        />
      )}
    </div>
  )
}
