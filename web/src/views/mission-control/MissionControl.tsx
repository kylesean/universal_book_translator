import React, { useCallback, useEffect, useMemo, useState, useRef } from 'react'
import { Link, useNavigate, useParams, useSearchParams } from 'react-router-dom'
import {
  Activity,
  X,
  Square,
  Trash2,
  ShieldCheck,
  SplitSquareVertical,
  RotateCcw,
  RefreshCw,
  Download,
  AlertTriangle,
  ChevronDown,
  ChevronRight,
  Check,
  Copy,
  CheckCheck,
  Search,
  ArrowLeft,
  ArrowUpRight,
  FileCheck,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs'
import { ConfirmDialog } from '@/components/ui/ConfirmDialog'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import { Progress } from '@/components/ui/progress'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { cn } from '@/lib/utils'
import {
  ApiError,
  getJobStatus,
  cancelJob,
  resumeJob,
  deleteJob,
  listJobs,
  listDeliverables,
  deliverableDownloadUrl,
  triggerBrowserDownload,
  reExportJob,
  reparseJob,
  subscribeJobProgress,
  type DeliverableItem,
  type DeliverableListing,
  type JobSummary,
  type PipelineStage,
  type ProgressStreamFrame,
} from '@/api/client'
import { useI18n } from '@/i18n/useI18n'
import { useToast } from '@/components/ui/useToast'
import { deliverableName } from '@/views/quality-gate/artifactMeta'
const getFormatBadge = (filename?: string): string => {
  if (!filename) return 'FILE'
  const ext = filename.split('.').pop()?.toLowerCase() ?? ''
  switch (ext) {
    case 'pdf': return 'PDF'
    case 'epub': return 'EPUB'
    case 'docx': return 'DOCX'
    case 'md':
    case 'markdown': return 'MD'
    case 'html':
    case 'htm': return 'HTML'
    case 'xliff': return 'XLIFF'
    case 'json': return 'JSON'
    default: return ext.toUpperCase() || 'FILE'
  }
}

const formatBytes = (bytes?: number): string => {
  if (!bytes || bytes <= 0 || isNaN(bytes)) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(1024))
  return `${(bytes / Math.pow(1024, i)).toFixed(i > 0 ? 1 : 0)} ${units[i]}`
}

function displayFileName(rawName?: string | null): string {
  if (!rawName) return '—'
  const match = rawName.match(/^(?:\d{8}-\d{6}-[a-f0-9]{8}-)?(.+)$/)
  return match ? match[1] : rawName
}

function compactJobId(jobId: string): string {
  const separator = jobId.indexOf('_')
  if (separator < 0) return jobId
  return `${jobId.slice(0, separator + 1)}…${jobId.slice(-6)}`
}

type LogLevel = 'ALL' | 'INFO' | 'WARN' | 'ERROR'

interface LogEntry {
  text: string
  level: 'INFO' | 'WARN' | 'ERROR'
  at?: number
  eventType?: string
}

function renderHighlightedText(text: string, query: string): React.ReactNode {
  if (!query.trim()) return text
  const escaped = query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  const parts = text.split(new RegExp(`(${escaped})`, 'gi'))
  if (parts.length === 1) return text
  return parts.map((part, i) =>
    part.toLowerCase() === query.toLowerCase() ? (
      <mark key={i} className="bg-amber-400/35 text-amber-100 rounded-xs px-0.5 font-semibold">
        {part}
      </mark>
    ) : (
      part
    )
  )
}

/**
 * Live progress from the shared counters.
 *
 * ``progress_percent`` counts successful terminal work plus in-flight work,
 * never failed blocks. A dead provider must not walk the bar to 100% while
 * translating nothing. ``drafted_blocks`` is work that has started; use it
 * until successful finalization overtakes it, and reserve 100% for a
 * genuinely finalized run.
 */
const liveProgressPct = (finalized: number, drafted: number, total: number): number => {
  if (total <= 0) return 0
  const finalizedWork = Math.max(0, Math.min(total, finalized))
  const draftedWork = Math.max(0, Math.min(total, drafted))
  if (finalizedWork >= total) return 100
  return Math.min(99.9, Math.round((Math.max(finalizedWork, draftedWork) / total) * 1000) / 10)
}

const jobStatusLabel = (t: ReturnType<typeof useI18n>['t'], status: string): string =>
  t.mission.jobStatuses[status] ?? status.toUpperCase()

const statusBadgeClass = (status: string): string =>
  status === 'completed'
    ? 'border-emerald-600/25 bg-emerald-500/10 text-emerald-700 dark:text-emerald-400'
    : status === 'failed' || status === 'cancelled'
      ? 'border-destructive/25 bg-destructive/10 text-destructive'
      : status === 'needs_human' || status === 'blocked_human'
        ? 'border-amber-600/25 bg-amber-500/10 text-amber-700 dark:text-amber-400'
        : 'border-sky-600/25 bg-sky-500/10 text-sky-700 dark:text-sky-400'

const stageLabel = (t: ReturnType<typeof useI18n>['t'], stage: string): string =>
  t.mission.stageNames[stage] ?? t.mission.jobStatuses[stage] ?? stage.toUpperCase()

/**
 * Turn the server's safe terminal error into one product-language sentence.
 *
 * The server already strips provider bodies and host paths; the remaining
 * exception class is still developer vocabulary, not something an operator
 * should have to parse. The raw message stays available under "Technical
 * detail" for support, collapsed by default.
 */
const failureSummary = (
  t: ReturnType<typeof useI18n>['t'],
  error: string,
): string => {
  if (/transport_unreachable|ConnectError|All connection attempts failed/i.test(error)) {
    return t.mission.failureProviderUnreachable
  }
  if (/ModelProviderError|ProviderConfigError|status_code|no router/i.test(error)) {
    return t.mission.failureConfig
  }
  if (/IntegrityViolationError|completion ratio|completion floor/i.test(error)) {
    return t.mission.failureIntegrity
  }
  if (/DocumentParseError|UnsupportedDocumentFormatError|Document .* produced 0/i.test(error)) {
    return t.mission.failureDocument
  }
  if (/timeout|timed out|connection|503|502|504/i.test(error)) {
    return t.mission.failureProvider
  }
  return t.mission.failureGeneric
}

const LOG_HISTORY = new Map<string, LogEntry[]>()
const LOG_HISTORY_MAX_JOBS = 20
const LOG_HISTORY_MAX_LINES = 400

function historyFor(jobId: string | null): LogEntry[] {
  if (!jobId) return []
  return LOG_HISTORY.get(jobId) ?? []
}

function rememberLogs(jobId: string | null, entries: LogEntry[]): void {
  if (!jobId) return
  LOG_HISTORY.delete(jobId)
  LOG_HISTORY.set(jobId, entries)
  while (LOG_HISTORY.size > LOG_HISTORY_MAX_JOBS) {
    const oldest = LOG_HISTORY.keys().next()
    if (oldest.done) break
    LOG_HISTORY.delete(oldest.value)
  }
}

type QueueFilter = 'ALL' | 'ACTIVE' | 'COMPLETED' | 'FAILED'

const READER_PRODUCT_KEYS = new Set(['primary', 'secondary', 'epub', 'html'])

function LogFilters({
  t,
  search,
  onSearchChange,
  level,
  onLevelChange,
  labels,
  counts,
}: {
  t: ReturnType<typeof useI18n>['t']
  search: string
  onSearchChange: (value: string) => void
  level: LogLevel
  onLevelChange: (level: LogLevel) => void
  labels: Record<LogLevel, string>
  counts: Record<LogLevel, number>
}) {
  return (
    <div className="flex min-w-0 flex-wrap items-center justify-end gap-2">
      <div className="relative min-w-32 flex-1 sm:max-w-48">
        <Search className="pointer-events-none absolute left-2.5 top-1/2 size-3.5 -translate-y-1/2 text-muted-foreground" />
        <Input
          type="text"
          value={search}
          onChange={(event) => onSearchChange(event.target.value)}
          placeholder={t.mission.logSearch}
          aria-label={t.mission.logSearch}
          className="h-8 bg-background pl-8 pr-2 font-mono text-xs"
        />
      </div>
      <div
        className="inline-flex h-8 items-center rounded-md border border-border/80 bg-muted/60 p-0.5"
        role="group"
        aria-label={t.mission.tabLogs}
      >
        {(['ALL', 'INFO', 'WARN', 'ERROR'] as const).map((item) => (
          <button
            key={item}
            type="button"
            onClick={() => onLevelChange(item)}
            className={cn(
              'flex h-7 cursor-pointer items-center gap-1 rounded px-2 font-mono text-xs transition-colors',
              level === item
                ? 'bg-background font-semibold text-foreground shadow-xs'
                : 'text-muted-foreground hover:text-foreground'
            )}
          >
            <span>{labels[item]}</span>
            <span
              className={cn(
                'tabular-nums opacity-70',
                item === 'ERROR' && counts.ERROR > 0 && 'font-semibold text-destructive opacity-100',
                item === 'WARN' && counts.WARN > 0 && 'font-semibold text-amber-600 opacity-100 dark:text-amber-400'
              )}
            >
              {counts[item]}
            </span>
          </button>
        ))}
      </div>
    </div>
  )
}

function JobQueueTable({
  jobs,
  currentJobId,
  t,
  onOpen,
}: {
  jobs: JobSummary[]
  currentJobId: string | null
  t: ReturnType<typeof useI18n>['t']
  onOpen: (jobId: string) => void
}) {
  return (
    <Table className="text-xs">
      <TableHeader className="sticky top-0 z-10 bg-muted/35">
        <TableRow className="border-border hover:bg-transparent">
          <TableHead className="w-[38%] px-4 py-2.5 font-medium text-muted-foreground">
            {t.mission.colFile}
          </TableHead>
          <TableHead className="hidden w-[20%] px-3 py-2.5 font-medium text-muted-foreground sm:table-cell">
            {t.mission.colJob}
          </TableHead>
          <TableHead className="hidden w-[12%] px-3 py-2.5 font-medium text-muted-foreground md:table-cell">
            {t.mission.colStatus}
          </TableHead>
          <TableHead className="hidden w-[14%] px-3 py-2.5 font-medium text-muted-foreground lg:table-cell">
            {t.mission.colProgress}
          </TableHead>
          <TableHead className="hidden w-[9%] px-3 py-2.5 text-right font-medium text-muted-foreground lg:table-cell">
            {t.mission.colCost}
          </TableHead>
          <TableHead className="hidden w-[10%] px-4 py-2.5 text-right font-medium text-muted-foreground xl:table-cell">
            {t.mission.colUpdated}
          </TableHead>
          <TableHead className="w-12 px-3 py-2.5 text-right">
            <span className="sr-only">{t.mission.jobActions}</span>
          </TableHead>
        </TableRow>
      </TableHeader>
      <TableBody className="divide-y divide-border">
        {jobs.map((job) => {
          const isActive = job.job_id === currentJobId
          const pct = liveProgressPct(
            job.completed_blocks + job.needs_human_blocks + job.blocked_human_blocks,
            job.drafted_blocks + (job.repair_pending_blocks ?? 0),
            job.total_blocks
          )

          return (
            <TableRow
              key={job.job_id}
              tabIndex={0}
              aria-label={`${t.mission.openJob}: ${displayFileName(job.file_name)}`}
              onClick={() => onOpen(job.job_id)}
              onKeyDown={(event) => {
                if (event.target !== event.currentTarget) return
                if (event.key === 'Enter' || event.key === ' ') {
                  event.preventDefault()
                  onOpen(job.job_id)
                }
              }}
              className={cn(
                'group cursor-pointer border-border transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-ring/35',
                isActive ? 'bg-muted/50' : 'hover:bg-muted/30'
              )}
            >
              <TableCell className="relative px-4 py-3.5 text-foreground">
                {isActive && (
                  <span className="absolute inset-y-2 left-0 w-[3px] rounded-r bg-primary" />
                )}
                <div className="flex min-w-0 items-center gap-2">
                  <span className="truncate text-[13px] font-medium text-foreground" title={job.file_name}>
                    {displayFileName(job.file_name)}
                  </span>
                  {job.rehearsal && (
                    <Badge variant="outline" className="shrink-0 px-1.5 py-0 text-xs">
                      {t.mission.rowRehearsal}
                    </Badge>
                  )}
                  {job.page_scope && (
                    <Badge variant="outline" className="shrink-0 px-1.5 py-0 font-mono text-xs">
                      p{job.page_scope}
                    </Badge>
                  )}
                </div>
              </TableCell>
              <TableCell
                className="hidden truncate px-3 py-3.5 font-mono text-xs text-muted-foreground sm:table-cell"
                title={job.job_id}
              >
                {job.job_id}
              </TableCell>
              <TableCell className="hidden px-3 py-3.5 md:table-cell">
                <Badge
                  variant="outline"
                  className={cn('font-normal', statusBadgeClass(job.status))}
                >
                  {jobStatusLabel(t, job.status)}
                </Badge>
              </TableCell>
              <TableCell className="hidden px-3 py-3.5 font-mono text-xs tabular-nums text-muted-foreground lg:table-cell">
                {job.status === 'completed' || pct >= 100 ? (
                  <span className="font-medium text-muted-foreground">100%</span>
                ) : (
                  <div className="flex items-center gap-2">
                    <Progress value={pct} className="h-1.5 w-14 bg-muted/60" />
                    <span className="font-medium text-foreground">{pct.toFixed(1)}%</span>
                  </div>
                )}
              </TableCell>
              <TableCell className="hidden px-3 py-3.5 text-right font-mono text-xs tabular-nums lg:table-cell">
                {job.estimated_cost_usd === null || job.estimated_cost_usd === undefined ? (
                  <span className="text-muted-foreground">—</span>
                ) : (
                  <span className={job.estimated_cost_usd === 0 ? 'text-muted-foreground' : 'font-medium text-foreground'}>
                    ${job.estimated_cost_usd.toFixed(3)}
                  </span>
                )}
              </TableCell>
              <TableCell className="hidden px-4 py-3.5 text-right font-mono text-xs tabular-nums text-muted-foreground xl:table-cell">
                {formatTimestamp(job.updated_at)}
              </TableCell>
              <TableCell className="px-3 py-3 text-right text-muted-foreground">
                <ChevronRight
                  className="ml-auto size-4 transition-transform group-hover:translate-x-0.5"
                  aria-hidden="true"
                />
              </TableCell>
            </TableRow>
          )
        })}
      </TableBody>
    </Table>
  )
}

function DeliverableRow({
  item,
  isPrimary,
  downloadLabel,
  onDownload,
}: {
  item: DeliverableItem
  isPrimary: boolean
  downloadLabel: string
  onDownload: () => void
}) {
  const { t } = useI18n()

  return (
    <div className="flex items-center justify-between gap-3 p-3 transition-colors hover:bg-muted/25">
      <div className="flex min-w-0 items-center gap-3">
        <span className="shrink-0 rounded bg-muted px-1.5 py-0.5 font-mono text-xs font-bold text-foreground">
          {getFormatBadge(item.filename)}
        </span>
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="truncate text-sm font-medium text-foreground">
              {deliverableName(item.key, item.label || item.filename, t)}
            </span>
            {isPrimary && (
              <span className="rounded bg-emerald-500/15 px-1.5 py-0.5 font-mono text-xs font-semibold text-emerald-600 dark:text-emerald-400">
                {t.mission.exportPrimaryBadge}
              </span>
            )}
          </div>
          <div className="mt-0.5 truncate font-mono text-xs text-muted-foreground">
            {item.filename} · {formatBytes(item.size_bytes)}
          </div>
        </div>
      </div>
      <Button
        onClick={onDownload}
        variant={isPrimary ? 'default' : 'outline'}
        size="sm"
        className="h-8 shrink-0 px-3 text-xs"
      >
        <Download className="mr-1.5 size-3.5" />
        <span>{downloadLabel}</span>
      </Button>
    </div>
  )
}

export function MissionControl() {
  const { t } = useI18n()
  const missionT = t.mission
  const toast = useToast()
  const navigate = useNavigate()
  const { jobId } = useParams<{ jobId: string }>()
  const [searchParams, setSearchParams] = useSearchParams()
  const currentJobId = jobId ?? null
  const onSelectJob = (id: string) => {
    navigate(`/jobs/${id}`)
  }

  const logLevelLabels: Record<LogLevel, string> = {
    ALL: t.mission.logLevelAll,
    INFO: t.mission.logLevelInfo,
    WARN: t.mission.logLevelWarn,
    ERROR: t.mission.logLevelError,
  }

  const [logs, setLogs] = useState<LogEntry[]>(() => historyFor(jobId ?? null))
  const tRef = useRef(t)
  useEffect(() => {
    tRef.current = t
  }, [t])
  const activityText = useCallback(
    (eventType: string | null | undefined, fallback: string): string =>
      (eventType && eventType in tRef.current.mission.activityEvents
        ? tRef.current.mission.activityEvents[
            eventType as keyof typeof tRef.current.mission.activityEvents
          ]
        : null) || fallback,
    []
  )

  const appendLog = useCallback(
    (entries: LogEntry[]) => {
      setLogs((prev) => {
        const next = [...prev]
        for (const entry of entries) {
          const last = next[next.length - 1]
          if (last && last.text === entry.text && last.level === entry.level) continue
          next.push(entry)
        }
        if (next.length === prev.length) return prev
        const bounded = next.slice(-LOG_HISTORY_MAX_LINES)
        rememberLogs(currentJobId, bounded)
        return bounded
      })
    },
    [currentJobId]
  )

  const [logSearch, setLogSearch] = useState('')
  const [logLevel, setLogLevel] = useState<LogLevel>('ALL')
  const [isLiveStreaming, setIsLiveStreaming] = useState(false)
  const [lastFrameAt, setLastFrameAt] = useState<number | null>(null)
  const [statusStr, setStatusStr] = useState<string>('running')
  const requestedTab = searchParams.get('tab')
  const currentTab: 'logs' | 'deliverables' =
    requestedTab === 'logs' || requestedTab === 'deliverables'
      ? requestedTab
      : statusStr === 'completed'
        ? 'deliverables'
        : 'logs'
  const setCurrentTab = (tab: 'logs' | 'deliverables') => {
    const next = new URLSearchParams(searchParams)
    if (tab === 'logs') next.set('tab', 'logs')
    else next.set('tab', 'deliverables')
    setSearchParams(next, { replace: true })
  }
  const [queuePosition, setQueuePosition] = useState<number | null>(null)
  const [stage, setStage] = useState<PipelineStage | null>(null)
  const [completedBlocks, setCompletedBlocks] = useState<number>(0)
  const [draftedBlocks, setDraftedBlocks] = useState<number>(0)
  const [repairPendingBlocks, setRepairPendingBlocks] = useState<number>(0)
  const [totalBlocks, setTotalBlocks] = useState<number>(0)
  const [costUsd, setCostUsd] = useState<number>(0)
  const [skippedBlocks, setSkippedBlocks] = useState<number>(0)
  const [failedBlocks, setFailedBlocks] = useState<number>(0)
  const [needsHumanBlocks, setNeedsHumanBlocks] = useState<number>(0)
  const [blockedHumanBlocks, setBlockedHumanBlocks] = useState<number>(0)
  const completedRef = useRef(0)
  const draftedRef = useRef(0)
  const repairPendingRef = useRef(0)
  const totalRef = useRef(0)
  const skippedRef = useRef(0)
  const failedRef = useRef(0)
  const [monotonicProgress, setMonotonicProgress] = useState(0)
  // Counters are the source of truth. Reconstructing a numerator from
  // ``progress_percent`` let a stale server value (failures counted as
  // progress) survive the fix and override the corrected counters.
  const finalizedBlocks = completedBlocks + needsHumanBlocks + blockedHumanBlocks
  const detailLivePct = liveProgressPct(
    finalizedBlocks,
    draftedBlocks + repairPendingBlocks,
    totalBlocks
  )
  const isTerminal =
    statusStr === 'completed' || statusStr === 'failed' || statusStr === 'cancelled'
  const displayedProgressPct = isTerminal
    ? detailLivePct
    : Math.max(detailLivePct, monotonicProgress)
  const [jobs, setJobs] = useState<JobSummary[]>([])
  const [queueFilter, setQueueFilter] = useState<QueueFilter>('ALL')
  const [queueSearch, setQueueSearch] = useState('')

  const activeJob = useMemo(() => jobs.find((j) => j.job_id === currentJobId), [jobs, currentJobId])

  // The queue snapshot already carries the ledger counters. Seeding from it
  // prevents the first paint of a terminal job from showing 0/0 until the
  // status request lands.
  useEffect(() => {
    if (!activeJob) return
    // oxlint-disable-next-line react/set-state-in-effect
    setStatusStr(activeJob.status)
    // oxlint-disable-next-line react/set-state-in-effect
    setCompletedBlocks(activeJob.completed_blocks)
    // oxlint-disable-next-line react/set-state-in-effect
    setNeedsHumanBlocks(activeJob.needs_human_blocks)
    // oxlint-disable-next-line react/set-state-in-effect
    setBlockedHumanBlocks(activeJob.blocked_human_blocks)
    // oxlint-disable-next-line react/set-state-in-effect
    setTotalBlocks(activeJob.total_blocks)
  }, [activeJob])

  const filteredJobs = useMemo(() => {
    return jobs.filter((job) => {
      if (queueFilter === 'ACTIVE') {
        if (job.status === 'completed' || job.status === 'failed' || job.status === 'cancelled') return false
      } else if (queueFilter === 'COMPLETED') {
        if (job.status !== 'completed') return false
      } else if (queueFilter === 'FAILED') {
        if (job.status !== 'failed' && job.status !== 'cancelled') return false
      }
      if (queueSearch.trim()) {
        const q = queueSearch.trim().toLowerCase()
        const idMatch = job.job_id.toLowerCase().includes(q)
        const nameMatch = (job.file_name || '').toLowerCase().includes(q)
        return idMatch || nameMatch
      }
      return true
    })
  }, [jobs, queueFilter, queueSearch])

  const [resuming, setResuming] = useState(false)
  const [downloading, setDownloading] = useState(false)
  const [deliverables, setDeliverables] = useState<DeliverableItem[]>([])

  useEffect(() => {
    let alive = true
    if (currentJobId && statusStr === 'completed') {
      listDeliverables(currentJobId)
        .then((listing) => {
          if (alive) setDeliverables(listing.items)
        })
        .catch(() => {
          if (alive) setDeliverables([])
        })
    } else {
      queueMicrotask(() => {
        if (alive) setDeliverables((prev) => (prev.length > 0 ? [] : prev))
      })
    }
    return () => {
      alive = false
    }
  }, [currentJobId, statusStr])

  const readingDeliverables = useMemo(() => {
    return deliverables.filter((d) => {
      const ext = d.filename.split('.').pop()?.toLowerCase() ?? ''
      return (
        d.key === 'primary' ||
        d.key === 'secondary' ||
        d.key === 'epub' ||
        d.key === 'html' ||
        d.key === 'xliff' ||
        ext === 'pdf' ||
        ext === 'epub' ||
        ext === 'docx' ||
        ext === 'md' ||
        ext === 'html' ||
        ext === 'xliff'
      )
    })
  }, [deliverables])

  const primaryDeliverable = useMemo(() => {
    return (
      readingDeliverables.find((d) => d.key === 'primary') ??
      readingDeliverables[0] ??
      deliverables[0] ??
      null
    )
  }, [readingDeliverables, deliverables])

  // Product vs. working-file split mirrors how the operator chooses: the first
  // group answers "which format do I publish?", the second answers "which tool
  // do I hand this to?".
  const readerKeys = useMemo(
    () => new Set(deliverables.filter((item) => READER_PRODUCT_KEYS.has(item.key)).map((item) => item.key)),
    [deliverables]
  )

  const [pendingDelete, setPendingDelete] = useState<string | null>(null)
  const [pendingReparse, setPendingReparse] = useState<string | null>(null)
  const [reparsing, setReparsing] = useState(false)
  const [streamGeneration, setStreamGeneration] = useState(0)
  const [statusMissing, setStatusMissing] = useState(false)
  const [pendingCancel, setPendingCancel] = useState(false)
  const [pendingBlockedDownload, setPendingBlockedDownload] = useState<{ jobId: string; key: string } | null>(null)
  const [cancelling, setCancelling] = useState(false)
  const [jobError, setJobError] = useState<string | null>(null)
  const [showFailureDetail, setShowFailureDetail] = useState(false)

  const refreshQueue = useCallback(async () => {
    try {
      setJobs(await listJobs(200))
    } catch {
      // A queue read failure must not blank the live dashboard.
    }
  }, [])

  useEffect(() => {
    // oxlint-disable-next-line react/set-state-in-effect
    void refreshQueue()
    const timer = setInterval(() => void refreshQueue(), 5000)
    return () => clearInterval(timer)
  }, [refreshQueue])

  useEffect(() => {
    if (!currentJobId) return

    // oxlint-disable-next-line react/set-state-in-effect
    setLogs(historyFor(currentJobId))
      setStatusStr('running')
    setStage(null)
    setCompletedBlocks(0)
    setDraftedBlocks(0)
    setTotalBlocks(0)
    setSkippedBlocks(0)
    setFailedBlocks(0)
    setNeedsHumanBlocks(0)
    setBlockedHumanBlocks(0)
    setCostUsd(0)
    setQueuePosition(null)
    setJobError(null)
    setStatusMissing(false)
    setLastFrameAt(null)

    completedRef.current = 0
    draftedRef.current = 0
    repairPendingRef.current = 0
    totalRef.current = 0
    skippedRef.current = 0
    failedRef.current = 0
    setMonotonicProgress(0)

    let isMounted = true
    let missing = false
    let timer: ReturnType<typeof setInterval> | undefined = undefined
    let unsubscribe = () => {}

    const applyStatus = (res: Record<string, unknown>) => {
      if (res.status) setStatusStr(String(res.status))
      if (typeof res.stage === 'string') setStage(res.stage as PipelineStage)
      if (typeof res.completed_blocks === 'number') {
        setCompletedBlocks(res.completed_blocks)
        completedRef.current = res.completed_blocks
      }
      if (typeof res.drafted_blocks === 'number') {
        setDraftedBlocks(res.drafted_blocks)
        draftedRef.current = res.drafted_blocks
      }
      if (typeof res.repair_pending_blocks === 'number') {
        setRepairPendingBlocks(res.repair_pending_blocks)
        repairPendingRef.current = res.repair_pending_blocks
      }
      if (typeof res.total_blocks === 'number') {
        setTotalBlocks(res.total_blocks)
        totalRef.current = res.total_blocks
      }
      if (typeof res.estimated_cost_usd === 'number') setCostUsd(res.estimated_cost_usd)
      if (typeof res.skipped_blocks === 'number') {
        setSkippedBlocks(res.skipped_blocks)
        skippedRef.current = res.skipped_blocks
      }
      if (typeof res.failed_blocks === 'number') {
        setFailedBlocks(res.failed_blocks)
        failedRef.current = res.failed_blocks
      }
      if (typeof res.needs_human_blocks === 'number') {
        setNeedsHumanBlocks(res.needs_human_blocks)
      }
      if (typeof res.blocked_human_blocks === 'number') {
        setBlockedHumanBlocks(res.blocked_human_blocks)
      }
      if (typeof res.error === 'string' && res.error.trim()) setJobError(res.error)
      else if (res.status && String(res.status) !== 'failed') setJobError(null)

      if (typeof res.queue_position === 'number' && res.queue_position > 0) {
        setQueuePosition(res.queue_position)
      } else {
        setQueuePosition(null)
      }
    }

    const stopForMissingJob = () => {
      if (missing) return
      missing = true
      clearInterval(timer)
      unsubscribe()
      setIsLiveStreaming(false)
      setStatusMissing(true)
    }

    const startStream = () => {
      if (!isMounted || missing) return
      unsubscribe = subscribeJobProgress(
        currentJobId,
        (frame: ProgressStreamFrame) => {
          if (!isMounted) return
          setLastFrameAt(Date.now())
          if (frame.status) setStatusStr(frame.status)
          if (frame.stage) setStage(frame.stage)
          if (typeof frame.completed_blocks === 'number') setCompletedBlocks(frame.completed_blocks)
          if (typeof frame.drafted_blocks === 'number') setDraftedBlocks(frame.drafted_blocks)
          if (typeof frame.repair_pending_blocks === 'number') {
            setRepairPendingBlocks(frame.repair_pending_blocks)
          }
          if (typeof frame.total_blocks === 'number') setTotalBlocks(frame.total_blocks)
          if (typeof frame.estimated_cost_usd === 'number') setCostUsd(frame.estimated_cost_usd)
          if (typeof frame.completed_blocks === 'number') completedRef.current = frame.completed_blocks
          if (typeof frame.drafted_blocks === 'number') draftedRef.current = frame.drafted_blocks
          if (typeof frame.repair_pending_blocks === 'number') {
            repairPendingRef.current = frame.repair_pending_blocks
          }
          if (typeof frame.total_blocks === 'number') totalRef.current = frame.total_blocks
          if (typeof frame.skipped_blocks === 'number') {
            setSkippedBlocks(frame.skipped_blocks)
            skippedRef.current = frame.skipped_blocks
          }
          if (typeof frame.failed_blocks === 'number') {
            setFailedBlocks(frame.failed_blocks)
            failedRef.current = frame.failed_blocks
          }
          if (typeof frame.needs_human_blocks === 'number') setNeedsHumanBlocks(frame.needs_human_blocks)
          if (typeof frame.blocked_human_blocks === 'number')
            setBlockedHumanBlocks(frame.blocked_human_blocks)

          const level: 'INFO' | 'WARN' | 'ERROR' = frame.error ? 'ERROR' : frame.status === 'failed' ? 'ERROR' : 'INFO'
          const qePart =
            (frame.current_avg_qe ?? 0) > 0
              ? `QE ${(frame.current_avg_qe ?? 0).toFixed(3)}`
              : ''

          const done = frame.completed_blocks ?? completedRef.current
          const total = frame.total_blocks ?? totalRef.current
          const drafted = (frame.drafted_blocks ?? draftedRef.current) || 0
          // Lifecycle events carry no block counters yet; don't show 0/total.
          const counters =
            done > 0 || drafted > 0
              ? tRef.current.mission.activityProgress
                  .replace('{done}', String(done))
                  .replace('{total}', String(total || '—'))
              : ''
          const preserved =
            skippedRef.current > 0
              ? ` · ${tRef.current.mission.activityPreserved.replace(
                  '{count}',
                  String(skippedRef.current)
                )}`
              : ''
          const text = frame.error
            ? `ERROR: ${frame.error}`
            : [
                activityText(
                  frame.event_type,
                  jobStatusLabel(tRef.current, frame.status ?? 'running')
                ),
                counters,
                preserved,
                qePart,
              ]
                .filter(Boolean)
                .join(' · ')
          appendLog([{ text, level, at: Date.now(), eventType: frame.event_type ?? undefined }])

          if (frame.status && ['completed', 'failed', 'cancelled'].includes(frame.status)) {
            clearInterval(timer)
            unsubscribe()
            setIsLiveStreaming(false)
          }
        },
        () => {
          if (missing) {
            unsubscribe()
            setIsLiveStreaming(false)
          }
        },
        (connected: boolean) => {
          if (isMounted && !missing) setIsLiveStreaming(connected)
        }
      )
    }

    const fetchStatus = async () => {
      if (missing) return
      try {
        const res = await getJobStatus(currentJobId)
        if (isMounted) {
          applyStatus(res as unknown as Record<string, unknown>)
          const isTerm =
            res.status && ['completed', 'failed', 'cancelled'].includes(String(res.status))
          if (isTerm) {
            clearInterval(timer)
            unsubscribe()
            setIsLiveStreaming(false)
          } else if (timer === undefined) {
            timer = setInterval(fetchStatus, 3000)
            startStream()
          }
        }
      } catch (err) {
        if (isMounted && err instanceof ApiError && err.status === 404) {
          stopForMissingJob()
        }
      }
    }

    void fetchStatus()

    return () => {
      isMounted = false
      clearInterval(timer)
      unsubscribe()
    }
  }, [currentJobId, appendLog, activityText, streamGeneration])

  useEffect(() => {
    // oxlint-disable-next-line react/set-state-in-effect
    setMonotonicProgress((current) => Math.max(current, detailLivePct))
  }, [detailLivePct])

  const handleCancel = async () => {
    setPendingCancel(false)
    if (!currentJobId) return
    setCancelling(true)
    try {
      await cancelJob(currentJobId)
      appendLog([{ text: t.mission.logAborted, level: 'WARN', at: Date.now() }])
    } catch (err) {
      toast.push(err instanceof Error ? err.message : t.mission.cancelFailed, 'error')
    } finally {
      setCancelling(false)
    }
  }

  const handleResume = async () => {
    if (!currentJobId) return
    setResuming(true)
    try {
      await resumeJob(currentJobId)
      appendLog([{ text: t.mission.logResumed, level: 'INFO', at: Date.now() }])
      setStatusStr('submitted')
      setMonotonicProgress(0)
      setStreamGeneration((n) => n + 1)
    } catch (err) {
      toast.push(err instanceof Error ? err.message : t.mission.resumeFailed, 'error')
    } finally {
      setResuming(false)
    }
  }

  const issueDownload = async (jobIdToDownload: string, key: string, force = false) => {
    setDownloading(true)
    try {
      let listing = await listDeliverables(jobIdToDownload)
      if (listing.stale) {
        toast.push(t.mission.reRenderingBeforeDownload, 'info')
        await reExportJob(jobIdToDownload)
        listing = await listDeliverables(jobIdToDownload)
        if (listing.deliveryVerdict === 'blocked' && !force) {
          setPendingBlockedDownload({ jobId: jobIdToDownload, key })
          return
        }
      }
      const target = listing.items.find((d) => d.key === key) ?? listing.items[0]
      if (target) {
        const url = deliverableDownloadUrl(jobIdToDownload, target.key, { force })
        triggerBrowserDownload(url, target.filename)
        toast.push(t.mission.exportSuccess, 'success')
      } else {
        toast.push(t.mission.noDeliverables, 'error')
      }
    } catch (err) {
      toast.push(err instanceof Error ? err.message : t.mission.downloadFailed, 'error')
    } finally {
      setDownloading(false)
    }
  }

  const handleDownload = async (key?: string) => {
    if (!currentJobId || downloading) return
    const targetKey = key ?? primaryDeliverable?.key ?? 'primary'
    let verdict: DeliverableListing['deliveryVerdict'] = null
    try {
      verdict = (await listDeliverables(currentJobId)).deliveryVerdict
    } catch {
      // Fall through to download
    }
    if (verdict === 'blocked') {
      setPendingBlockedDownload({ jobId: currentJobId, key: targetKey })
      return
    }
    await issueDownload(currentJobId, targetKey)
  }

  const confirmDeleteJob = async () => {
    const jobIdToDelete = pendingDelete
    setPendingDelete(null)
    if (!jobIdToDelete) return
    try {
      await deleteJob(jobIdToDelete)
      if (jobIdToDelete === currentJobId) {
        // The ledger is gone; the detail view cannot show anything useful, so
        // land back on the queue instead of a silently dead detail page.
        navigate('/jobs')
        appendLog([{ text: t.mission.logDeleted, level: 'WARN', at: Date.now() }])
        toast.push(t.mission.jobDeleted, 'success')
      }
      await refreshQueue()
    } catch (err) {
      toast.push(err instanceof Error ? err.message : t.mission.deleteFailed, 'error')
    }
  }

  const confirmReparse = async () => {
    const jobIdToReparse = pendingReparse
    setPendingReparse(null)
    if (!jobIdToReparse) return
    setReparsing(true)
    try {
      await reparseJob(jobIdToReparse)
      setStreamGeneration((n) => n + 1)
      toast.push(t.mission.reparseStarted, 'info')
      await refreshQueue()
    } catch (err) {
      toast.push(err instanceof Error ? err.message : t.mission.reparseFailed, 'error')
    } finally {
      setReparsing(false)
    }
  }

  const pipelineStages: { key: PipelineStage; label: string }[] = useMemo(
    () => [
      { key: 'extract', label: t.mission.stageExtract },
      { key: 'segment', label: t.mission.stageSegment },
      { key: 'tm', label: t.mission.stageTm },
      { key: 'translate', label: t.mission.stageTranslate },
      { key: 'qe', label: t.mission.stageQe },
      { key: 'repair', label: t.mission.stageRepair },
      { key: 'render', label: t.mission.stageRender },
      { key: 'verify', label: t.mission.stageVerify },
      { key: 'package', label: t.mission.stagePackage },
    ],
    [t]
  )

  const pipelineGroups = useMemo(
    () => [
      { label: t.mission.phaseParse, stages: pipelineStages.slice(0, 3) },
      { label: t.mission.phaseTranslate, stages: pipelineStages.slice(3, 6) },
      { label: t.mission.phaseDeliver, stages: pipelineStages.slice(6, 9) },
    ],
    [pipelineStages, t]
  )

  const filterOptions = useMemo<{ key: QueueFilter; label: string }[]>(
    () => [
      { key: 'ALL', label: t.mission.filterAll },
      { key: 'ACTIVE', label: t.mission.filterActive },
      { key: 'COMPLETED', label: t.mission.filterCompleted },
      { key: 'FAILED', label: t.mission.filterFailed },
    ],
    [t]
  )

  const isCompleted = statusStr === 'completed'
  const isCancelled = statusStr === 'cancelled'
  const isFailed = statusStr === 'failed' || isCancelled
  // Progress is not a pipeline cursor: a run can report 100% after its last
  // block failed before export. Keep the live stage on failure and only fall
  // back to the last stage when the terminal record no longer carries one.
  const terminalStage: PipelineStage | '' = isFailed ? (stage ?? 'package') : ''
  const gateBlocked = blockedHumanBlocks > 0 || failedBlocks > 0
  const gateNeedsReview = !gateBlocked && needsHumanBlocks > 0
  const failureText = isCancelled
    ? t.mission.cancelledDetail
    : jobError
      ? failureSummary(t, jobError)
      : t.mission.failureShort

  const getExecutionSubtext = (): string => {
    if (isCompleted) {
      if (gateBlocked) return t.mission.stateBlocked
      if (gateNeedsReview) return t.mission.stateNeedsReview
      return t.mission.stateCertified
    }
    if (isCancelled) return t.mission.cancelledShort
    if (isFailed) return t.mission.failureShort
    switch (derivedStage) {
      case 'extract':
        return t.mission.stageStateExtract
      case 'segment':
        return t.mission.stageStateSegment
      case 'tm':
        return t.mission.stageStateTm
      case 'translate':
        return t.mission.stageStateTranslate
      case 'qe':
        return t.mission.stageStateQe
      case 'repair':
        return t.mission.stageStateRepair
      case 'render':
        return t.mission.stageStateRender
      case 'verify':
        return t.mission.stageStateVerify
      case 'package':
        return t.mission.stageStatePackage
      default:
        return t.mission.stateDrafting
    }
  }

  const derivedStage: PipelineStage | '' = isFailed
    ? terminalStage
    : statusStr === 'completed'
      ? 'package'
      : statusStr === 'queued' || statusStr === 'submitted'
        ? 'extract'
        : (stage ?? '')

  const activeStageRef = useRef<HTMLLIElement>(null)
  useEffect(() => {
    if (activeStageRef.current) {
      activeStageRef.current.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' })
    }
  }, [derivedStage])

  const [copiedId, setCopiedId] = useState(false)
  const handleCopyJobId = (idToCopy: string) => {
    void navigator.clipboard.writeText(idToCopy)
    setCopiedId(true)
    toast.push(t.mission.jobIdCopied, 'success')
    setTimeout(() => setCopiedId(false), 1500)
  }

  // The activity list is a task timeline, not a raw server log. A finished job
  // whose stream is no longer retained gets an honest empty state.
  const activeLogs = logs

  const filteredLogs = activeLogs.filter((entry) => {
    if (logLevel !== 'ALL' && entry.level !== logLevel) return false
    if (logSearch && !entry.text.toLowerCase().includes(logSearch.toLowerCase())) return false
    return true
  })

  const logCounts = useMemo(() => {
    let info = 0
    let warn = 0
    let error = 0
    for (const item of activeLogs) {
      if (item.level === 'INFO') info++
      else if (item.level === 'WARN') warn++
      else if (item.level === 'ERROR') error++
    }
    return {
      ALL: activeLogs.length,
      INFO: info,
      WARN: warn,
      ERROR: error,
    }
  }, [activeLogs])

  const currentJobFileName = activeJob?.file_name || primaryDeliverable?.filename

  return (
    <div className="flex-1 flex flex-col min-h-0 overflow-y-auto px-4 md:px-8 py-4 md:py-6 space-y-4 md:space-y-5">
      <ConfirmDialog
        open={pendingCancel}
        title={t.mission.cancelTitle}
        body={t.mission.cancelConfirm}
        confirmLabel={t.mission.cancelCompile}
        cancelLabel={t.common.cancel}
        danger
        onConfirm={() => void handleCancel()}
        onCancel={() => setPendingCancel(false)}
      />
      <ConfirmDialog
        open={pendingBlockedDownload !== null}
        title={t.mission.blockedDownloadTitle}
        body={t.mission.blockedDownloadBody}
        confirmLabel={t.mission.blockedDownloadConfirm}
        cancelLabel={t.common.cancel}
        danger
        onConfirm={() => {
          const pending = pendingBlockedDownload
          setPendingBlockedDownload(null)
          if (pending) void issueDownload(pending.jobId, pending.key, true)
        }}
        onCancel={() => setPendingBlockedDownload(null)}
      />
      <ConfirmDialog
        open={pendingDelete !== null}
        title={t.mission.deleteJobTitle}
        body={
          <div className="space-y-3">
            <p>{t.mission.deleteJobBody}</p>
            <div className="rounded-lg border border-border bg-muted/35 px-3 py-2.5">
              <div className="truncate text-sm font-medium text-foreground">
                {displayFileName(jobs.find((job) => job.job_id === pendingDelete)?.file_name)}
              </div>
              <div className="mt-0.5 truncate font-mono text-xs text-muted-foreground">
                {pendingDelete}
              </div>
            </div>
          </div>
        }
        confirmLabel={t.mission.deleteJob}
        cancelLabel={t.common.cancel}
        danger
        onConfirm={() => void confirmDeleteJob()}
        onCancel={() => setPendingDelete(null)}
      />
      <ConfirmDialog
        open={pendingReparse !== null}
        title={t.mission.reparseTitle}
        body={
          <div className="space-y-2.5">
            <p>{t.mission.reparseConfirm}</p>
            {pendingReparse && (
              <div className="break-all rounded-lg border border-border bg-muted/35 px-3 py-2 font-mono text-xs text-foreground select-all">
                {pendingReparse}
              </div>
            )}
          </div>
        }
        confirmLabel={t.mission.reparseAction}
        cancelLabel={t.common.cancel}
        danger
        onConfirm={() => void confirmReparse()}
        onCancel={() => setPendingReparse(null)}
        typeToConfirm={pendingReparse ?? undefined}
        typeToConfirmLabel={t.mission.reparseTypePrompt.replace('{phrase}', pendingReparse ?? '')}
        typeToConfirmMismatch={t.mission.reparseTypeMismatch}
      />

      {/* Header bar: Clean, precise, and un-bloated */}
      <div className="flex flex-col justify-between gap-3 border-b border-border/80 pb-5 sm:flex-row sm:items-end">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2 sm:gap-3">
            {currentJobId && (
              <Button
                variant="ghost"
                size="icon"
                onClick={() => navigate('/jobs')}
                className="-ml-1 size-8 shrink-0"
                title={t.mission.backToJobList}
                aria-label={t.mission.backToJobList}
              >
                <ArrowLeft />
              </Button>
            )}
            <h1 className="truncate text-2xl font-semibold tracking-[-0.025em] text-foreground">
              {currentJobId ? displayFileName(currentJobFileName) : t.mission.queueTitle}
            </h1>
            {currentJobId && (
              <Badge
                variant="outline"
                className={cn(
                  'font-normal',
                  isCompleted
                    ? 'border-emerald-600/25 bg-emerald-500/10 text-emerald-700 dark:text-emerald-400'
                    : isFailed
                      ? 'border-destructive/25 bg-destructive/10 text-destructive'
                      : 'border-sky-600/25 bg-sky-500/10 text-sky-700 dark:text-sky-400'
                )}
              >
                <span
                  className={cn(
                    'mr-1.5 size-1.5 rounded-full',
                    isCompleted
                      ? 'bg-emerald-500'
                      : isFailed
                        ? 'bg-destructive'
                        : 'bg-sky-500'
                  )}
                />
                {jobStatusLabel(t, statusStr)}
              </Badge>
            )}
          </div>
          {currentJobId && (
            <div className="mt-1.5 flex min-w-0 items-center gap-1.5 font-mono text-xs text-muted-foreground">
              <span className="shrink-0">{t.mission.jobIdLabel}</span>
              <span
                className="min-w-0 truncate font-medium text-foreground select-all"
                title={currentJobId}
              >
                {compactJobId(currentJobId)}
              </span>
              <button
                type="button"
                onClick={() => handleCopyJobId(currentJobId)}
                className="p-1 rounded text-muted-foreground hover:text-foreground hover:bg-muted transition-colors cursor-pointer"
                title={t.mission.copyJobId}
                aria-label={t.mission.copyJobId}
              >
                {copiedId ? <CheckCheck className="size-3 text-emerald-600" /> : <Copy className="size-3" />}
              </button>
            </div>
          )}
        </div>

        {/* Compact action buttons */}
        <div className="flex flex-wrap items-center gap-1.5 sm:gap-2">
          {currentJobId && isCompleted && (
            <div className="inline-flex items-center rounded-md shadow-xs">
              <Button
                onClick={() => void handleDownload()}
                variant="default"
                size="sm"
                disabled={downloading}
                className={cn(
                  "h-8 px-3 text-xs font-semibold",
                  readingDeliverables.length > 1 && "rounded-r-none border-r border-primary-foreground/20"
                )}
                title={t.mission.exportPrimaryHint}
              >
                {downloading ? (
                  <RefreshCw className="size-3.5 mr-1.5 animate-spin" />
                ) : (
                  <Download className="size-3.5 mr-1.5" />
                )}
                {t.mission.oneClickExport}
              </Button>

              {readingDeliverables.length > 1 && (
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button
                      variant="default"
                      size="sm"
                      disabled={downloading}
                      className="h-8 px-2 rounded-l-none"
                      title={t.mission.exportFormats}
                      aria-label={t.mission.exportFormats}
                    >
                      <ChevronDown className="size-3.5" />
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end" className="w-64 max-h-72 overflow-y-auto">
                    <DropdownMenuLabel className="text-xs uppercase tracking-wider text-muted-foreground">
                      {t.mission.exportFormats}
                    </DropdownMenuLabel>
                    <DropdownMenuSeparator />
                    {readingDeliverables.map((item) => {
                      const isPrimary = item.key === 'primary' || item === primaryDeliverable
                      const badge = getFormatBadge(item.filename)
                      return (
                        <DropdownMenuItem
                          key={item.key}
                          onSelect={() => void handleDownload(item.key)}
                          className="flex items-start justify-between gap-2 cursor-pointer py-2"
                        >
                          <div className="min-w-0 flex-1">
                            <div className="flex items-center gap-1.5">
                              <span className="font-mono text-xs font-bold px-1.5 py-0.5 rounded bg-muted text-foreground">
                                {badge}
                              </span>
                              <span className="font-medium text-xs text-foreground truncate">
                                {deliverableName(item.key, item.label || item.filename, t)}
                              </span>
                            </div>
                            <div className="text-xs text-muted-foreground font-mono mt-0.5 truncate">
                              {item.filename} · {formatBytes(item.size_bytes)}
                            </div>
                          </div>
                          <div className="flex shrink-0 items-center gap-1">
                            {isPrimary && (
                              <span className="text-xs px-1 py-0.5 rounded bg-emerald-500/15 text-emerald-600 dark:text-emerald-400 font-semibold font-mono">
                                {t.mission.exportPrimaryBadge}
                              </span>
                            )}
                          </div>
                        </DropdownMenuItem>
                      )
                    })}
                  </DropdownMenuContent>
                </DropdownMenu>
              )}
            </div>
          )}

          {currentJobId && (
            <Button
              asChild
              variant="secondary"
              size="sm"
              className="h-8 px-2.5 text-xs font-medium"
            >
              <Link to={`/jobs/${currentJobId}/review`}>
                <SplitSquareVertical className="mr-1.5 size-3.5 text-muted-foreground" />
                {t.nav.reviewWorkbench}
              </Link>
            </Button>
          )}

          {currentJobId && (
            <Button
              asChild
              variant="secondary"
              size="sm"
              className="h-8 px-2.5 text-xs font-medium"
            >
              <Link to={`/jobs/${currentJobId}/quality`}>
                <ShieldCheck className="mr-1.5 size-3.5 text-emerald-600 dark:text-emerald-400" />
                {t.quality.title}
              </Link>
            </Button>
          )}

          {currentJobId && isFailed && (
            <Button onClick={handleResume} variant="secondary" size="sm" disabled={resuming} className="h-8 px-2.5 text-xs font-medium">
              <RotateCcw className={cn("size-3.5 mr-1.5", resuming && "animate-spin")} />
              {resuming ? t.mission.resuming : t.mission.resumeCompile}
            </Button>
          )}

          {currentJobId && isCompleted && (
            <Button
              onClick={() => setPendingReparse(currentJobId)}
              variant="secondary"
              size="sm"
              disabled={reparsing}
              title={t.mission.reparseHint}
              className="h-8 px-2.5 text-xs font-medium"
            >
              <RefreshCw className={cn("size-3.5 mr-1.5", reparsing && "animate-spin")} />
              {reparsing ? t.mission.reparsing : t.mission.reparseAction}
            </Button>
          )}

          {currentJobId && (
            <Button
              onClick={() => setPendingDelete(currentJobId)}
              variant="ghost"
              size="icon"
              className="size-8 text-muted-foreground hover:text-destructive"
              title={t.mission.deleteJob}
              aria-label={t.mission.deleteJob}
            >
              <Trash2 className="size-3.5" />
            </Button>
          )}

          {currentJobId && !isCompleted && !isFailed && (
            <Button
              onClick={() => setPendingCancel(true)}
              variant="outline"
              size="sm"
              disabled={cancelling}
              className="h-8 px-2.5 text-xs font-medium text-destructive hover:bg-destructive/10 hover:text-destructive"
            >
              <Square className="size-3 mr-1.5 fill-current" />
              {cancelling ? t.mission.cancelling : t.mission.cancelCompile}
            </Button>
          )}
        </div>
      </div>

      {/* Terminal failure reason. A user cancellation is not a fault, so it
          gets its own plain explanation instead of a red failure banner. */}
      {isFailed && !isCancelled && (
        <div className="rounded-lg border border-destructive/25 bg-destructive/[0.04] px-4 py-3 text-sm text-muted-foreground flex items-start gap-3 shrink-0">
          <AlertTriangle className="size-4 text-destructive shrink-0 mt-0.5" />
          <div className="min-w-0 flex-1 space-y-1">
            <div className="font-semibold text-foreground">{t.mission.failureReason}</div>
            <p className="leading-relaxed">{failureText}</p>
            {jobError && (
              <>
                <button
                  type="button"
                  onClick={() => setShowFailureDetail((open) => !open)}
                  className="inline-flex items-center gap-1 text-xs font-medium text-muted-foreground underline-offset-4 hover:text-foreground hover:underline"
                  aria-expanded={showFailureDetail}
                >
                  <ChevronDown className={cn('size-3 transition-transform', showFailureDetail && 'rotate-180')} />
                  {t.mission.technicalDetail}
                </button>
                {showFailureDetail && (
                  <pre className="max-h-28 overflow-auto whitespace-pre-wrap break-words rounded-md bg-background/70 p-2.5 font-mono text-xs leading-relaxed text-muted-foreground">
                    {jobError}
                  </pre>
                )}
              </>
            )}
          </div>
        </div>
      )}

      {statusMissing ? (
        <div className="flex-1 flex items-center justify-center text-center text-muted-foreground">
          <div className="max-w-sm space-y-2">
            <AlertTriangle className="size-8 mx-auto text-amber-600 dark:text-amber-500" />
            <h2 className="text-sm font-semibold text-foreground">
              {t.mission.jobMissingTitle}
            </h2>
            <p className="text-xs text-muted-foreground leading-relaxed">
              {t.mission.jobMissingDesc}
            </p>
            <Button onClick={() => navigate('/jobs')} variant="secondary" size="sm">
              {t.mission.backToQueue}
            </Button>
          </div>
        </div>
      ) : !currentJobId ? (
        /* Global Job Queue (When no jobId is selected) */
        <Card className="overflow-hidden border-border shadow-card">
          <div className="flex flex-col items-stretch justify-between gap-3 border-b border-border/70 bg-card px-4 py-3 sm:flex-row sm:items-center">
            <div className="flex items-center gap-2.5">
              <span className="text-sm font-semibold tracking-[-0.01em] text-foreground">
                {t.mission.queueTitle}
              </span>
              <span className="text-xs font-mono font-medium text-muted-foreground px-2 py-0.5 rounded-full bg-border/60 tabular-nums">
                {t.mission.runsCount.replace('{filtered}', String(filteredJobs.length)).replace('{total}', String(jobs.length))}
              </span>
            </div>

            <div className="flex w-full flex-col items-stretch gap-2 sm:w-auto sm:flex-row sm:items-center">
              <div className="relative min-w-0 sm:w-64">
                <Input
                  type="text"
                  placeholder={t.mission.searchJobsPlaceholder}
                  value={queueSearch}
                  onChange={(e) => setQueueSearch(e.target.value)}
                  aria-label={t.mission.searchJobsPlaceholder}
                  className="h-8 bg-background pl-8 text-xs"
                />
                <Search className="pointer-events-none absolute left-2.5 top-2.5 size-3.5 text-muted-foreground" />
              </div>

              <Tabs
                value={queueFilter}
                onValueChange={(val) => setQueueFilter(val as QueueFilter)}
                  className="inline-flex w-full sm:w-auto sm:shrink-0"
              >
                <TabsList className="h-8 w-full max-w-full justify-start overflow-x-auto p-0.5 sm:w-auto">
                  {filterOptions.map((opt) => (
                    <TabsTrigger
                      key={opt.key}
                      value={opt.key}
                      className="text-xs h-6 px-2.5 font-medium"
                    >
                      {opt.label}
                    </TabsTrigger>
                  ))}
                </TabsList>
              </Tabs>
            </div>
          </div>

          <div className="max-h-[70vh] overflow-auto">
            {filteredJobs.length === 0 ? (
              <div className="py-12 text-center text-sm text-muted-foreground">
                {jobs.length === 0 ? t.mission.queueEmpty : t.mission.queueNoMatch}
              </div>
      ) : null}

      {statusMissing ? null : (
              <JobQueueTable
                jobs={filteredJobs}
                currentJobId={currentJobId}
                t={t}
                onOpen={onSelectJob}
              />
            )}
          </div>
        </Card>
      ) : (
        /* Selected Job Overview */
        <>
          {/* Execution Pipeline Stepper */}
          <Card className="p-3.5 sm:p-4 shrink-0 shadow-sm space-y-3 border-border">
            <div className="grid gap-5 md:grid-cols-3 md:gap-0 md:divide-x md:divide-border/70">
              {pipelineGroups.map((group) => (
                <section
                  key={group.label}
                  className="min-w-0 md:px-4 md:first:pl-0 md:last:pr-0"
                >
                  <h3 className="mb-3 text-xs font-medium text-muted-foreground">
                    {group.label}
                  </h3>
                  <ol className="grid grid-cols-3">
                    {group.stages.map((stg) => {
                      const idx = pipelineStages.findIndex((item) => item.key === stg.key)
                      const currentIndex = pipelineStages.findIndex(
                        (item) => item.key === derivedStage
                      )
                      const isCurrent = derivedStage !== '' && derivedStage === stg.key
                      const isDone = isCompleted || (currentIndex >= 0 && idx < currentIndex)
                      const isCurrentFailed = statusStr === 'failed' && isCurrent
                      const isCurrentCancelled = isCancelled && isCurrent

                      return (
                        <li
                          key={stg.key}
                          ref={isCurrent ? activeStageRef : undefined}
                          aria-current={isCurrent ? 'step' : undefined}
                          className="relative flex min-w-0 flex-col items-center text-center"
                        >
                          <span
                            className={cn(
                              'relative z-10 flex size-8 items-center justify-center rounded-full border text-xs font-semibold tabular-nums transition-colors',
                              isCurrentCancelled
                                ? 'border-muted-foreground/30 bg-muted text-muted-foreground'
                                : isCurrentFailed
                                ? 'border-destructive/30 bg-destructive text-destructive-foreground'
                                : isCurrent
                                  ? 'border-primary bg-primary text-primary-foreground'
                                  : isDone
                                    ? 'border-emerald-600/20 bg-emerald-500/10 text-emerald-700 dark:text-emerald-400'
                                    : 'border-border bg-background text-muted-foreground'
                            )}
                          >
                            {isCurrentCancelled ? (
                              <X className="size-3.5" />
                            ) : isCurrentFailed ? (
                              <AlertTriangle className="size-3.5" />
                            ) : isDone ? (
                              <Check className="size-3.5 stroke-[2.5]" />
                            ) : (
                              String(idx + 1).padStart(2, '0')
                            )}
                          </span>
                          <span
                            className={cn(
                              'mt-2 max-w-full truncate text-xs leading-tight',
                              isCurrent
                                ? 'font-semibold text-foreground'
                                : isDone
                                  ? 'font-medium text-foreground'
                                  : 'text-muted-foreground'
                            )}
                          >
                            {stg.label}
                          </span>
                        </li>
                      )
                    })}
                  </ol>
                </section>
              ))}
            </div>
          </Card>

          {/* Four compact facts. Each tile has one label, one value and one
              supporting line; terminal states do not repeat the header badge. */}
          <Card className="grid grid-cols-2 md:grid-cols-4 shrink-0 shadow-sm overflow-hidden p-0 border-border">
            <div className="flex min-w-0 flex-col justify-between border-b border-border p-3.5 sm:p-4 md:border-b-0 md:border-r">
              <div className="flex items-center justify-between type-group">
                <span>{t.mission.executionState}</span>
              </div>
              <div className="h-8 flex items-center gap-2 mt-1">
                <span
                  className={cn(
                    "size-2.5 rounded-full shrink-0",
                    isCompleted
                      ? "bg-emerald-500"
                      : isCancelled
                        ? "bg-muted-foreground"
                        : isFailed
                          ? "bg-destructive"
                          : "bg-primary animate-pulse"
                  )}
                />
                <span
                  className={cn(
                    "truncate text-xl font-semibold tracking-tight sm:text-2xl",
                    isFailed && !isCancelled ? "text-destructive" : "text-foreground"
                  )}
                >
                  {jobStatusLabel(t, statusStr)}
                </span>
              </div>
              <div
                className="mt-2 flex h-5 items-center truncate text-xs text-muted-foreground"
                title={getExecutionSubtext()}
              >
                {getExecutionSubtext() || (isCompleted ? t.mission.deliverablesReady : t.mission.backgroundProcessing)}
              </div>
            </div>

            <div className="flex min-w-0 flex-col justify-between border-b border-border p-3.5 sm:p-4 md:border-b-0 md:border-r">
              <div className="flex items-center justify-between type-group">
                <span>{t.mission.progress}</span>
                {queuePosition !== null && (
                  <span className="text-xs font-mono text-muted-foreground">
                    {t.mission.queuePosition}: {queuePosition}
                  </span>
                )}
              </div>
              <div
                className={cn(
                  'mt-1 flex h-8 items-baseline gap-1 font-semibold tabular-nums',
                  isFailed ? 'text-base text-muted-foreground' : 'text-2xl text-foreground'
                )}
              >
                {isCancelled ? (
                  t.mission.jobStatuses.cancelled
                ) : isFailed ? (
                  t.mission.pipelineHalted
                ) : (
                  <>
                    {displayedProgressPct.toFixed(1)}
                    <span className="text-base font-normal text-muted-foreground">%</span>
                  </>
                )}
              </div>
              <div className="mt-2 flex h-5 w-full items-center gap-2">
                <Progress
                  value={
                    isTerminal
                      ? isCompleted
                        ? 100
                        : 0
                      : Math.min(100, Math.max(0, displayedProgressPct))
                  }
                  className="h-1.5 flex-1 bg-muted/60"
                />
                <span className="max-w-[45%] shrink-0 truncate text-xs text-muted-foreground">
                  {isTerminal
                    ? jobStatusLabel(t, statusStr)
                    : stageLabel(t, derivedStage || statusStr)}
                </span>
              </div>
            </div>

            <div className="flex min-w-0 flex-col justify-between p-3.5 sm:p-4">
              <div className="flex items-center type-group">
                <span>{t.mission.completedPages}</span>
              </div>
              <div className="mt-1 flex h-8 items-baseline gap-1 text-2xl font-semibold tabular-nums text-foreground">
                {isCompleted ? totalBlocks : completedBlocks}
                <span className="text-xs font-normal text-muted-foreground">
                  / {totalBlocks || '—'}
                </span>
              </div>
              <div className="mt-2 flex h-5 items-center truncate text-xs">
                {failedBlocks > 0 || needsHumanBlocks > 0 || skippedBlocks > 0 ? (
                  <div className="flex items-center gap-2 truncate text-xs">
                    {failedBlocks > 0 && (
                      <span className="text-destructive">
                        {failedBlocks} {t.mission.failedBlocks}
                      </span>
                    )}
                    {needsHumanBlocks > 0 && (
                      <span className="text-amber-700 dark:text-amber-400">
                        {needsHumanBlocks} {t.mission.needsHumanBlocks}
                      </span>
                    )}
                    {skippedBlocks > 0 && (
                      <span className="text-muted-foreground">
                        {skippedBlocks} {t.mission.preservedBlocks}
                      </span>
                    )}
                  </div>
                ) : (
                  <span className="truncate text-xs text-muted-foreground">
                    {totalBlocks > 0 ? t.mission.allBlocksVerified.replace('{total}', String(totalBlocks)) : t.mission.waitingForExtraction}
                  </span>
                )}
              </div>
            </div>

            <div className="flex min-w-0 flex-col justify-between border-r border-border p-3.5 sm:p-4">
              <div className="flex items-center justify-between type-group">
                <span>{t.mission.totalSpend}</span>
              </div>
              <div className="mt-1 flex h-8 items-baseline gap-1 text-2xl font-semibold tabular-nums text-foreground">
                ${costUsd.toFixed(3)}
              </div>
              <div className="mt-2 flex h-5 items-center truncate text-xs text-muted-foreground">
                {isCompleted ? (
                  <span>{t.mission.deliverablesReady}</span>
                ) : totalBlocks > completedBlocks ? (
                  <span>{totalBlocks - completedBlocks} {t.mission.blocksRemaining}</span>
                ) : completedBlocks > 0 && costUsd > 0 ? (
                  <span>{t.mission.avgCostPerBlock.replace('{cost}', (costUsd / completedBlocks).toFixed(3))}</span>
                ) : (
                  <span>{t.mission.costFreeLocal}</span>
                )}
              </div>
            </div>
          </Card>

          {/* A job page has two answers: what was produced, and what happened. */}
          <Tabs
            value={currentTab}
            onValueChange={(val) => setCurrentTab(val as 'logs' | 'deliverables')}
            className="flex flex-col space-y-2"
          >
            <div className="flex flex-wrap items-center justify-between gap-2">
              <TabsList className="h-8 p-0.5">
                <TabsTrigger value="logs" className="text-xs h-7 px-3 gap-1.5">
                  <Activity className="size-3.5" />
                  <span>{t.mission.tabLogs}</span>
                  {activeLogs.length > 0 && (
                    <span className="font-mono text-xs opacity-70">({activeLogs.length})</span>
                  )}
                </TabsTrigger>
                <TabsTrigger value="deliverables" className="text-xs h-7 px-3 gap-1.5">
                  <FileCheck className="size-3.5" />
                  <span>{t.mission.tabDeliverables}</span>
                  {deliverables.length > 0 && (
                    <span className="font-mono text-xs opacity-70">({deliverables.length})</span>
                  )}
                </TabsTrigger>
              </TabsList>
              {currentTab === 'logs' && activeLogs.length > 0 && (
                <LogFilters
                  t={t}
                  search={logSearch}
                  onSearchChange={setLogSearch}
                  level={logLevel}
                  onLevelChange={setLogLevel}
                  labels={logLevelLabels}
                  counts={logCounts}
                />
              )}
            </div>

            {/* The activity timeline. A finished job whose event history is no
                longer retained gets an honest empty state, not a fake terminal. */}
            <TabsContent value="logs" className="mt-0 flex flex-col">
              {isTerminal && activeLogs.length === 0 ? (
                <Card className="flex min-h-52 flex-1 items-center justify-center p-6 text-center">
                  <div className="max-w-md space-y-2">
                    {isCompleted ? (
                      <Check className="mx-auto size-5 text-emerald-600 dark:text-emerald-400" />
                    ) : isCancelled ? (
                      <X className="mx-auto size-5 text-muted-foreground" />
                    ) : (
                      <AlertTriangle className="mx-auto size-5 text-destructive" />
                    )}
                    <div className="text-sm font-medium text-foreground">
                      {t.mission.logUnavailableTitle}
                    </div>
                    <p className="text-sm leading-relaxed text-muted-foreground">
                      {t.mission.logUnavailableTerminal}
                    </p>
                  </div>
                </Card>
              ) : (
              <Card className="flex w-full flex-col overflow-hidden border-border shadow-sm">
                {/* Activity Timeline */}
                <div className="max-h-[500px] min-h-24 flex-1 overflow-y-auto bg-card px-4 py-3 select-text">
                  {activeLogs.length === 0 ? (
                    <div className="py-8 text-center space-y-2">
                      <Activity className="size-7 mx-auto text-muted-foreground/50" />
                      <p className="mx-auto max-w-md text-sm leading-relaxed text-muted-foreground">
                        {isTerminal ? t.mission.logUnavailableTerminal : t.mission.waitingStream}
                      </p>
                    </div>
                  ) : filteredLogs.length === 0 ? (
                    <div className="py-8 text-center space-y-2">
                      <Search className="size-7 mx-auto text-muted-foreground/50" />
                      <p className="text-sm text-muted-foreground">{t.mission.noMatch}</p>
                    </div>
                  ) : (
                    <ol className="relative ml-2 border-l border-border pl-5">
                      {filteredLogs.map((log, idx) => {
                        const isSuccess =
                          log.text.includes('[SUCCESS]') ||
                          log.text.includes('[成功]') ||
                          log.text.includes('100.0%')
                        return (
                          <li key={idx} className="relative py-2">
                            <span
                              className={cn(
                                'absolute -left-[25px] top-[15px] size-2 rounded-full ring-4 ring-card',
                                log.level === 'ERROR'
                                  ? 'bg-destructive'
                                  : log.level === 'WARN'
                                    ? 'bg-amber-500'
                                    : isSuccess
                                      ? 'bg-emerald-500'
                                      : 'bg-muted-foreground/45'
                              )}
                              aria-hidden="true"
                            />
                            <div className="flex flex-wrap items-baseline gap-x-2 gap-y-0.5">
                              {log.at !== undefined && (
                                <time
                                  className="font-mono text-xs tabular-nums text-muted-foreground"
                                  dateTime={new Date(log.at).toISOString()}
                                >
                                  {formatClock(log.at)}
                                </time>
                              )}
                              {log.level !== 'INFO' && (
                                <span
                                  className={cn(
                                    'text-xs font-semibold',
                                    log.level === 'ERROR'
                                      ? 'text-destructive'
                                      : 'text-amber-700 dark:text-amber-400'
                                  )}
                                >
                                  {logLevelLabels[log.level]}
                                </span>
                              )}
                            </div>
                            <p
                              className={cn(
                                'mt-1 break-words text-sm leading-relaxed',
                                log.level === 'ERROR'
                                  ? 'text-destructive'
                                  : log.level === 'WARN'
                                    ? 'text-amber-800 dark:text-amber-300'
                                    : 'text-foreground'
                              )}
                            >
                              {renderHighlightedText(log.text, logSearch)}
                            </p>
                          </li>
                        )
                      })}
                    </ol>
                  )}
                </div>

                {/* Activity Footer */}
                <div className="flex shrink-0 flex-col gap-2 border-t border-border/70 bg-muted/25 px-3.5 py-2.5 text-xs text-muted-foreground sm:flex-row sm:items-center sm:justify-between">
                  <div className="flex min-w-0 items-center gap-2.5">
                    {isCompleted ? (
                      <div className="flex items-center gap-1.5 text-xs font-medium text-emerald-600 dark:text-emerald-400">
                        <Check className="size-3.5 shrink-0" />
                        <span className="truncate">{t.mission.terminalReadyForDownload}</span>
                      </div>
                    ) : !isFailed ? (
                      <>
                        <span
                          className={cn(
                            'size-1.5 shrink-0 rounded-full',
                            isLiveStreaming ? 'bg-emerald-500' : 'bg-amber-500'
                          )}
                          aria-hidden="true"
                        />
                        <span className="shrink-0 text-xs font-medium text-foreground">
                          {isLiveStreaming
                            ? t.mission.logStreamConnected
                            : lastFrameAt === null
                              ? t.mission.logStreamConnecting
                              : t.mission.logStreamReconnecting}
                        </span>
                        <span className="text-muted-foreground/50" aria-hidden="true">
                          ·
                        </span>
                        <span className="min-w-0 truncate font-mono text-xs tabular-nums text-muted-foreground">
                          {lastFrameAt === null
                            ? t.mission.logAwaitingFirstEvent
                            : `${t.mission.logLastEventAt} ${formatClock(lastFrameAt)}`}
                        </span>
                      </>
                    ) : (
                      <div className="flex items-center gap-1.5 text-xs text-destructive">
                        <AlertTriangle className="size-3.5 shrink-0" />
                        <span className="truncate">{failureText}</span>
                      </div>
                    )}
                  </div>

                  <div className="ml-5 shrink-0 font-mono text-xs tabular-nums text-muted-foreground">
                    {filteredLogs.length}/{activeLogs.length} {t.mission.bufferedLines}
                  </div>
                </div>
              </Card>
              )}
            </TabsContent>

            {/* Tab 2: Deliverables List */}
            <TabsContent value="deliverables" className="mt-0">
              <Card className="overflow-hidden shadow-sm border-border">
                <CardContent className="p-0">
                  {deliverables.length === 0 ? (
                    <div className="py-12 text-center text-xs text-muted-foreground font-mono">
                      {isCompleted ? t.mission.noDeliverables : t.mission.backgroundProcessing}
                    </div>
                  ) : (
                    <div className="space-y-3 p-3.5">
                      <section className="overflow-hidden rounded-lg border border-border/70">
                        <div className="border-b border-border/70 bg-emerald-500/5 px-3 py-2">
                          <h3 className="text-xs font-semibold tracking-wide text-foreground">
                            {missionT.readerProductsTitle}
                          </h3>
                        </div>
                        <div className="divide-y divide-border/60">
                          {deliverables
                            .filter((item) => readerKeys.has(item.key))
                            .map((item) => (
                              <DeliverableRow
                                key={item.key}
                                item={item}
                                isPrimary={item.key === primaryDeliverable?.key}
                                downloadLabel={missionT.downloadPrimary}
                                onDownload={() => void handleDownload(item.key)}
                              />
                            ))}
                        </div>
                      </section>

                      <section className="overflow-hidden rounded-lg border border-border/70">
                        <div className="border-b border-border/70 bg-muted/30 px-3 py-2">
                          <h3 className="text-xs font-semibold tracking-wide text-foreground">
                            {missionT.reviewFilesTitle}
                          </h3>
                          <Button
                            asChild
                            variant="ghost"
                            size="sm"
                            className="h-6 px-2 text-xs"
                          >
                            <Link to={`/jobs/${currentJobId}/review`}>
                              {missionT.openReview}
                              <ArrowUpRight className="ml-1 size-3" />
                            </Link>
                          </Button>
                        </div>
                        <div className="divide-y divide-border/60">
                          {deliverables
                            .filter((item) => !readerKeys.has(item.key))
                            .map((item) => (
                              <DeliverableRow
                                key={item.key}
                                item={item}
                                isPrimary={false}
                                downloadLabel={missionT.downloadPrimary}
                                onDownload={() => void handleDownload(item.key)}
                              />
                            ))}
                        </div>
                      </section>
                    </div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>

            {/* Tab 3: Global Queue Table */}
            <TabsContent value="queue" className="mt-0">
              <Card className="overflow-hidden shadow-sm border-border">
                <div className="h-11 px-4 border-b border-border/70 bg-muted/25 flex flex-wrap items-center justify-between gap-3">
                  <div className="flex items-center gap-2.5">
                    <span className="text-xs font-semibold tracking-wide text-foreground">
                      {t.mission.queueTitle}
                    </span>
                    <span className="text-xs font-mono font-medium text-muted-foreground px-2 py-0.5 rounded-full bg-border/60 tabular-nums">
                      {t.mission.runsCount.replace('{filtered}', String(filteredJobs.length)).replace('{total}', String(jobs.length))}
                    </span>
                  </div>

                  <div className="flex items-center gap-2">
                    <div className="relative">
                      <input
                        type="text"
                        placeholder={t.mission.searchJobsPlaceholder}
                        value={queueSearch}
                        onChange={(e) => setQueueSearch(e.target.value)}
                        className="h-7 text-xs pl-7 pr-2.5 rounded-md bg-card border border-border text-foreground placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring w-36 sm:w-48 transition-all"
                      />
                      <Search className="size-3.5 text-muted-foreground absolute left-2 top-2 pointer-events-none" />
                    </div>

                    <Tabs
                      value={queueFilter}
                      onValueChange={(val) => setQueueFilter(val as QueueFilter)}
                      className="hidden sm:inline-flex w-auto"
                    >
                      <TabsList className="h-7 p-0.5">
                        {filterOptions.map((opt) => (
                          <TabsTrigger
                            key={opt.key}
                            value={opt.key}
                            className="text-xs h-6 px-2.5 font-medium"
                          >
                            {opt.label}
                          </TabsTrigger>
                        ))}
                      </TabsList>
                    </Tabs>
                  </div>
                </div>

                <div className="max-h-[460px] overflow-y-auto">
                  {filteredJobs.length === 0 ? (
                    <div className="py-8 text-center text-sm text-muted-foreground">
                      {jobs.length === 0 ? t.mission.queueEmpty : t.mission.queueNoMatch}
                    </div>
                  ) : (
                    <JobQueueTable
                      jobs={filteredJobs}
                      currentJobId={currentJobId}
                      t={t}
                      onOpen={onSelectJob}
                    />
                  )}
                </div>
              </Card>
            </TabsContent>
          </Tabs>
        </>
      )}
    </div>
  )
}

function formatTimestamp(value: string | null | undefined): string {
  if (!value) return '—'
  const normalized = value.includes('T') ? value : value.replace(' ', 'T')
  const hasZone = /(?:Z|[+-]\d{2}:?\d{2})$/.test(normalized)
  const parsed = new Date(hasZone ? normalized : `${normalized}Z`)
  if (Number.isNaN(parsed.getTime())) return value
  const pad = (n: number) => String(n).padStart(2, '0')
  return `${pad(parsed.getMonth() + 1)}-${pad(parsed.getDate())} ${pad(parsed.getHours())}:${pad(parsed.getMinutes())}`
}

function formatClock(value: number): string {
  const parsed = new Date(value)
  const pad = (n: number) => String(n).padStart(2, '0')
  return `${pad(parsed.getHours())}:${pad(parsed.getMinutes())}:${pad(parsed.getSeconds())}`
}
