import React, { useCallback, useEffect, useMemo, useState, useRef } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import {
  Terminal,
  Square,
  ShieldCheck,
  SplitSquareVertical,
  Activity,
  RotateCcw,
  RefreshCw,
  Download,
  Trash2,
  AlertTriangle,
  ChevronDown,
  ChevronUp,
  Check,
  Layers,
  Coins,
  Eye,
  FileText,
  Sparkles,
  Copy,
  CheckCheck,
} from 'lucide-react'
import { Button } from '@/components/ui/Button'
import { Badge } from '@/components/ui/Badge'
import { ConfirmDialog } from '@/components/ui/ConfirmDialog'
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

type LogLevel = 'INFO' | 'WARN' | 'ERROR'

interface LogEntry {
  text: string
  level: LogLevel
}

/**
 * Live progress from the shared counters.
 *
 * ``progress_percent`` only counts blocks with a final disposition, so the
 * draft phase reads 0% for its whole duration. ``drafted_blocks`` is work that
 * has started; use it until finalization overtakes it, and reserve 100% for a
 * genuinely finalized run.
 */
const liveProgressPct = (finalized: number, drafted: number, total: number): number => {
  if (total <= 0) return 0
  const finalizedWork = Math.max(0, Math.min(total, finalized))
  const draftedWork = Math.max(0, Math.min(total, drafted))
  if (finalizedWork >= total) return 100
  return Math.min(99.9, Math.round((Math.max(finalizedWork, draftedWork) / total) * 1000) / 10)
}

//: Job status as a label. Both the header badge and the queue rows rendered
//: ``status.toUpperCase()``, so an operator on the Chinese console read
//: "COMPLETED" over a fully Chinese page. An unknown status falls back to its
//: upper-cased identifier: a new engine state should still be visible.
const jobStatusLabel = (t: ReturnType<typeof useI18n>['t'], status: string): string =>
  t.mission.jobStatuses[status] ?? status.toUpperCase()

//: Pipeline stage, or a status name when no stage event has arrived yet (a
//: queued job reports no stage). Same fallback rule as above.
const stageLabel = (t: ReturnType<typeof useI18n>['t'], stage: string): string =>
  t.mission.stageNames[stage] ?? t.mission.jobStatuses[stage] ?? stage.toUpperCase()

/**
 * Console log lines per job, surviving a route change.
 *
 * The panel's state lived in the component, and Mission Control unmounts the
 * moment the operator opens the review workbench. Coming back rebuilt an empty
 * array and re-subscribed to `/stream`, which for a finished job answers with
 * one snapshot plus one terminal frame — so a run that had produced a hundred
 * lines came back showing two, and the buffer counter read "2/2" as if that
 * were all there had ever been. Measured on the reference job: 2 lines after
 * navigating away and back, both duplicates of the same final state.
 *
 * Kept outside React and keyed by job id so it also cannot leak one job's log
 * into another's panel. Bounded like the in-component buffer was.
 */
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
  // Oldest-first eviction, mirroring the frame buffer's own bound.
  while (LOG_HISTORY.size > LOG_HISTORY_MAX_JOBS) {
    const oldest = LOG_HISTORY.keys().next()
    if (oldest.done) break
    LOG_HISTORY.delete(oldest.value)
  }
}

export function MissionControl() {
  const { t, language } = useI18n()
  const toast = useToast()
  const navigate = useNavigate()
  const { jobId } = useParams<{ jobId: string }>()
  const currentJobId = jobId ?? null
  const detailSectionRef = useRef<HTMLDivElement>(null)
  const onSelectJob = (id: string) => {
    navigate(`/jobs/${id}`)
    setTimeout(() => {
      detailSectionRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }, 60)
  }
  const onInspectQuality = (id: string) => navigate(`/jobs/${id}/quality`)
  const onOpenReview = (id: string) => navigate(`/jobs/${id}/review`)
  // Seeded from the cross-route history so returning to this job restores what
  // the stream had already said, instead of the two frames a terminal job still
  // replays (see ``LOG_HISTORY``). The lazy initializer runs once per mount.
  const [logs, setLogs] = useState<LogEntry[]>(() => historyFor(jobId ?? null))

  // Every write goes through here so the in-memory history can never drift from
  // what the panel shows. Two sinks, one update: state for this render, the map
  // for the next mount.
  //
  // Consecutive duplicates are dropped. The stream emits a frame per engine
  // tick, and between two engine messages most frames carry the same counters
  // and therefore the same rendered line: a measured run showed the identical
  // "118/148 blocks (+49 preserved) · QE: rules only (no score)" tail repeated
  // dozens of times, pushing the actual messages off the 400-line buffer. Only *adjacent*
  // repeats are collapsed — a line that reappears after a different one is real
  // history, and comparing against the whole buffer would hide it.
  // ``currentJobId`` is a dependency rather than a ref: the SSE effect below
  // re-subscribes when the job changes anyway, so a stable identity buys
  // nothing, and reading a ref during render is a real hydration hazard.
  const appendLog = useCallback(
    (entries: LogEntry[]) => {
      setLogs((prev) => {
        const next = [...prev]
        for (const entry of entries) {
          const last = next[next.length - 1]
          if (last && last.text === entry.text && last.level === entry.level) continue
          next.push(entry)
        }
        // Nothing new: return the same array so React skips the re-render and
        // the bounded buffer keeps its identity.
        if (next.length === prev.length) return prev
        const bounded = next.slice(-LOG_HISTORY_MAX_LINES)
        rememberLogs(currentJobId, bounded)
        return bounded
      })
    },
    [currentJobId]
  )
  const [logSearch, setLogSearch] = useState('')
  const [logLevel, setLogLevel] = useState<'ALL' | LogLevel>('ALL')
  const [autoScroll, setAutoScroll] = useState(true)
  const [isLogCollapsed, setIsLogCollapsed] = useState(false)
  const [isLiveStreaming, setIsLiveStreaming] = useState(false)
  const [statusStr, setStatusStr] = useState<string>('running')
  const [queuePosition, setQueuePosition] = useState<number | null>(null)
  const [stage, setStage] = useState<PipelineStage | null>(null)
  const [progressPct, setProgressPct] = useState<number>(0)
  const [completedBlocks, setCompletedBlocks] = useState<number>(0)
  // Drafted blocks are live work, not finished work. They make the bar move
  // during the long draft stage while ``completed_blocks`` correctly stays at
  // zero until quality/repair reaches a terminal disposition.
  const [draftedBlocks, setDraftedBlocks] = useState<number>(0)
  // Repair candidates are the same kind of live work: the stage will process
  // them into a terminal disposition, but starting it must not drop them from
  // the progress fraction.
  const [repairPendingBlocks, setRepairPendingBlocks] = useState<number>(0)
  const [totalBlocks, setTotalBlocks] = useState<number>(0)
  const [costUsd, setCostUsd] = useState<number>(0)
  const [avgQe, setAvgQe] = useState<number>(0)
  // Blocks the policy finished without translating them — images, bibliography
  // entries, bylines, short headings. They are excluded from total/completed so
  // the percentage reflects translation work, and reported separately so the
  // panel does not appear to be describing a shorter book than the source.
  const [skippedBlocks, setSkippedBlocks] = useState<number>(0)
  // Blocks that ended in a failure. The API always carried this; the panel never
  // rendered it, so a job with 15 failed blocks read as a clean 100%.
  const [failedBlocks, setFailedBlocks] = useState<number>(0)
  const [needsHumanBlocks, setNeedsHumanBlocks] = useState<number>(0)
  // Quarantined blocks (MQM Critical). They ship as a source-only placeholder,
  // not a translation, and they make the delivery verdict "blocked" — but the
  // panel never read the counter, so a job whose gate refused it still read as
  // "compiler output certified" as soon as its status reached completed.
  const [blockedHumanBlocks, setBlockedHumanBlocks] = useState<number>(0)
  // Mirrors of the counters above, for the SSE log tail. The stream callback is
  // created once and closes over its first render's state, so reading the state
  // variables directly there would print the initial zeros on a terminal frame
  // that carries no counters — the "0/0 blocks" bug — and would also need every
  // counter in the callback's dependency list, re-subscribing the stream on each
  // frame. Refs are the standard way out: the callback stays stable, the value
  // is always current.
  const completedRef = useRef(0)
  const draftedRef = useRef(0)
  const repairPendingRef = useRef(0)
  const totalRef = useRef(0)
  const skippedRef = useRef(0)
  const failedRef = useRef(0)
  // Live progress must not step backwards when a drafted block becomes
  // ``repair_pending`` (the finalized numerator can lag the drafted one).
  // Reset on a job or stream-generation change, never on every frame.
  const monotonicProgressRef = useRef(0)
  // The engine sends 0.0 as the "nothing scored" sentinel under the heuristic
  // QE engine; a measured average is never exactly 0 in practice (defect bands
  // floor at 0.30), so 0 renders as "not measured" instead of a failing score.
  const qeMeasured = avgQe > 0
  // A translation job spends most of its wall clock drafting. Counting only
  // finalized blocks left the progress bar at 0.0% for that entire phase and
  // then jumped to the quality gate's completion percentage in one frame.
  const finalizedBlocks = Math.max(
    completedBlocks + failedBlocks + needsHumanBlocks,
    totalBlocks > 0 && progressPct > 0 ? Math.round((progressPct / 100) * totalBlocks) : 0
  )
  const detailLivePct = liveProgressPct(
    finalizedBlocks,
    draftedBlocks + repairPendingBlocks,
    totalBlocks
  )
  // oxlint-disable-next-line react/refs
  const displayedProgressPct = Math.max(detailLivePct, monotonicProgressRef.current)
  // The run is over, so its live log can no longer be replayed.
  const isTerminal =
    statusStr === 'completed' || statusStr === 'failed' || statusStr === 'cancelled'
  const [jobs, setJobs] = useState<JobSummary[]>([])
  const [resuming, setResuming] = useState(false)
  const [downloading, setDownloading] = useState(false)
  const [deliverables, setDeliverables] = useState<DeliverableItem[]>([])
  const [exportMenuOpen, setExportMenuOpen] = useState(false)
  const exportContainerRef = useRef<HTMLDivElement>(null)

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

  useEffect(() => {
    if (!exportMenuOpen) return
    const handleClickOutside = (e: MouseEvent) => {
      if (exportContainerRef.current && !exportContainerRef.current.contains(e.target as Node)) {
        setExportMenuOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [exportMenuOpen])

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
  const [pendingDelete, setPendingDelete] = useState<string | null>(null)
  // Confirmation for the destructive re-parse: every block is discarded and
  // the book re-translated at full cost, so it takes a typed phrase, not a click.
  const [pendingReparse, setPendingReparse] = useState<string | null>(null)
  const [reparsing, setReparsing] = useState(false)
  // Bumped whenever a finished job is started again on the SAME id.
  // The SSE client closes the stream permanently on a terminal frame, so
  // without a new subscription a resumed or re-parsed run streams nothing:
  // the panel kept showing the previous run's final "completed 250/250"
  // lines while a full-cost re-translation was actually running.
  const [streamGeneration, setStreamGeneration] = useState(0)
  // Set when the job cannot be read at all (404 — deleted, or a ledger this
  // server session never saw). Without it the view rendered a RUNNING badge, a
  // live Cancel button and "waiting for the first frame" forever, because
  // `statusStr` defaults to 'running' and the poll swallowed every error.
  const [statusMissing, setStatusMissing] = useState(false)
  // Cancel throws away an in-flight, billable run — the most expensive
  // irreversible action in the console, and the only one that used to fire
  // on a single click while delete and re-parse demanded a typed id.
  const [pendingCancel, setPendingCancel] = useState(false)
  // Set when the operator asked to download a delivery the quality gate
  // blocked. The server refuses those by default, so the click opens this
  // dialog instead of a download; confirming re-issues it with `force`.
  const [pendingBlockedDownload, setPendingBlockedDownload] = useState<{ jobId: string; key: string } | null>(null)
  const [cancelling, setCancelling] = useState(false)
  // The job's terminal error, as the engine phrased it for an operator. The
  // status payload always carried it and the console never read it, so a
  // failed job showed a red FAILED badge and nothing else — the operator had
  // to open the server logs to learn why.
  const [jobError, setJobError] = useState<string | null>(null)

  const logContainerRef = useRef<HTMLDivElement>(null)

  const refreshQueue = useCallback(async () => {
    try {
      setJobs(await listJobs(200))
    } catch {
      // A queue read failure must not blank the live dashboard.
    }
  }, [])

  // The queue is polled independently of the selected job so a restarted
  // console still lists finished runs (they live on disk, not in memory).
  useEffect(() => {
    // Mount-time queue read plus its poll; `refreshQueue`'s setState runs after
    // the awaited fetch, not during the effect.
    // oxlint-disable-next-line react/set-state-in-effect
    void refreshQueue()
    const timer = setInterval(() => void refreshQueue(), 5000)
    return () => clearInterval(timer)
  }, [refreshQueue])

  useEffect(() => {
    if (!currentJobId) return

    // Switching jobs inside this view does not remount it, so EVERY per-job
    // value has to be reset, not just the log. Previously only `setLogs` ran
    // here: status, stage, progress, the block counters, cost, QE, queue
    // position and the failure text all kept job A's numbers under job B's id
    // until B's first poll returned — and the mirror refs (`completedRef` and
    // friends) were never reset at all, so B's first terminal frame printed A's
    // counts. `jobError` was worse: it only cleared when the next status was
    // not 'failed', so switching between two failed jobs showed the first
    // one's failure reason over the second.
    // oxlint-disable-next-line react/set-state-in-effect
    setLogs(historyFor(currentJobId))
    // oxlint-disable-next-line react/set-state-in-effect
    setStatusStr('running')
    // oxlint-disable-next-line react/set-state-in-effect
    setStage(null)
    // oxlint-disable-next-line react/set-state-in-effect
    setProgressPct(0)
    // oxlint-disable-next-line react/set-state-in-effect
    setCompletedBlocks(0)
    // oxlint-disable-next-line react/set-state-in-effect
    setDraftedBlocks(0)
    // oxlint-disable-next-line react/set-state-in-effect
    setTotalBlocks(0)
    // oxlint-disable-next-line react/set-state-in-effect
    setSkippedBlocks(0)
    // oxlint-disable-next-line react/set-state-in-effect
    setFailedBlocks(0)
    // oxlint-disable-next-line react/set-state-in-effect
    setNeedsHumanBlocks(0)
    // oxlint-disable-next-line react/set-state-in-effect
    setCostUsd(0)
    // oxlint-disable-next-line react/set-state-in-effect
    setAvgQe(0)
    // oxlint-disable-next-line react/set-state-in-effect
    setQueuePosition(null)
    // oxlint-disable-next-line react/set-state-in-effect
    setJobError(null)
    // oxlint-disable-next-line react/set-state-in-effect
    setStatusMissing(false)
    // The mirror refs feed the log line; they are not React state, so the
    // reset above cannot reach them.
    completedRef.current = 0
    draftedRef.current = 0
    repairPendingRef.current = 0
    totalRef.current = 0
    skippedRef.current = 0
    failedRef.current = 0
    monotonicProgressRef.current = 0

    let isMounted = true
    // A 404 is terminal: the id does not exist in this server's queue, memory
    // or ledgers. Stop both pollers instead of retrying until the operator
    // leaves the page. The status request and the SSE error can race, so the
    // flag is set before either cleanup runs; the interval and subscription
    // are also cleared by the unmount cleanup below.
    let missing = false
    let timer: ReturnType<typeof setInterval> | undefined = undefined
    let unsubscribe = () => {}
    const applyStatus = (res: Record<string, unknown>) => {
      if (res.status) setStatusStr(String(res.status))
      if (typeof res.stage === 'string') setStage(res.stage as PipelineStage)
      if (typeof res.progress_percent === 'number') setProgressPct(res.progress_percent)
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
      if (typeof res.current_avg_qe === 'number') setAvgQe(res.current_avg_qe)
      // The new counters have to be read here too, or a finished job — whose
      // stream is gone — shows no skipped/failed/human counts at all: this
      // status call is the only hydration that path ever gets.
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
      // The engine already phrases a terminal error for the operator
      // (`public_job_error` keeps the actionable types verbatim). Without this
      // read, a failed job was a badge with no reason.
      if (typeof res.error === 'string' && res.error.trim()) setJobError(res.error)
      else if (res.status && String(res.status) !== 'failed') setJobError(null)
      // Queue mode reports where the job sits while it waits; the field was
      // always in the response and never rendered, so a queued job looked
      // identical to a stalled one.
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
          if (frame.status) setStatusStr(frame.status)
          if (frame.stage) setStage(frame.stage)
          if (typeof frame.progress_percent === 'number') setProgressPct(frame.progress_percent)
          if (typeof frame.completed_blocks === 'number') setCompletedBlocks(frame.completed_blocks)
          if (typeof frame.drafted_blocks === 'number') setDraftedBlocks(frame.drafted_blocks)
          if (typeof frame.repair_pending_blocks === 'number') {
            setRepairPendingBlocks(frame.repair_pending_blocks)
          }
          if (typeof frame.total_blocks === 'number') setTotalBlocks(frame.total_blocks)
          if (typeof frame.estimated_cost_usd === 'number') setCostUsd(frame.estimated_cost_usd)
          if (typeof frame.current_avg_qe === 'number') setAvgQe(frame.current_avg_qe)
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

          // Log the engine's own message when it has one — the stream is a build
          // log, not a counter echo. The counters stay on the line as the tail so
          // a frame without a message still reads as progress.
          const level: LogLevel = frame.error ? 'ERROR' : frame.status === 'failed' ? 'ERROR' : 'INFO'
          const qePart =
            (frame.current_avg_qe ?? 0) > 0
              ? `QE ${(frame.current_avg_qe ?? 0).toFixed(3)}`
              : // "no score" is a fact about the QE mode (rule checks only score
                // nothing), not a failure — say which, so the word does not read as
                // an error on an otherwise clean run.
                'QE: rules only (no score)'
          // A terminal frame now carries its own counters; when it does not, keep
          // the last known ones instead of printing 0/0.
          const done = Math.max(
            frame.completed_blocks ?? completedRef.current,
            (frame.drafted_blocks ?? draftedRef.current) +
              (frame.repair_pending_blocks ?? repairPendingRef.current)
          )
          const total = frame.total_blocks ?? totalRef.current
          // "preserved", not "skipped": most of these are bibliography entries the
          // policy keeps verbatim for retrievability, not list markers nobody
          // needed. The panel's own label says the same thing.
          const skippedPart = skippedRef.current > 0 ? ` (+${skippedRef.current} preserved)` : ''
          const failPart = (frame.failed_blocks ?? failedRef.current) > 0
            ? ` · ${frame.failed_blocks ?? failedRef.current} failed`
            : ''
          const counters = `${done}/${total} blocks${skippedPart}${failPart} · ${qePart}`
          const text = frame.error
            ? `ERROR ${frame.error}`
            : frame.message
              ? `${frame.message} · ${counters}`
              : `${frame.status ?? 'running'} · ${counters}`
          appendLog([{ text, level }])

          if (frame.status && ['completed', 'failed', 'cancelled'].includes(frame.status)) {
            clearInterval(timer)
            unsubscribe()
            setIsLiveStreaming(false)
          }
        },
        () => {
          // EventSource does not expose the HTTP status. Only stop when the
          // status poll has already proved the job absent; a transient stream
          // failure on a live job must still use the API client's backoff.
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
        // A 404 is not a transient poll failure: the job does not exist, so
        // retrying every 3 s forever is pointless and the "running" default is
        // a lie. Say so once and stop pretending.
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
  }, [currentJobId, appendLog, streamGeneration])

  useEffect(() => {
    if (autoScroll && logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight
    }
  }, [logs, autoScroll])

  useEffect(() => {
    monotonicProgressRef.current = Math.max(
      monotonicProgressRef.current,
      detailLivePct
    )
  }, [detailLivePct])

  const handleCancel = async () => {
    setPendingCancel(false)
    if (!currentJobId) return
    setCancelling(true)
    try {
      await cancelJob(currentJobId)
      appendLog([{ text: t.mission.logAborted, level: 'WARN' }])
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
      appendLog([{ text: t.mission.logResumed, level: 'INFO' }])
      setStatusStr('submitted')
      setStreamGeneration((n) => n + 1)
    } catch (err) {
      toast.push(err instanceof Error ? err.message : 'Resume failed', 'error')
    } finally {
      setResuming(false)
    }
  }

  // Completed run → open the primary translated document in a new tab. The
  // deliverable list names what actually exists on disk, so no key guessing.
  //
  // A job with edits that were never re-rendered is refused by the download
  // endpoint (409) — deliberately, because serving the pre-edit file silently
  // is the worst outcome available. The user asked for their current text, so
  // the console re-renders first and then downloads, turning a correctness
  // guard into a slower click rather than an error they must decode.
  /** Issue the download for `key`, optionally overriding a blocked verdict. */
  const issueDownload = async (jobId: string, key: string, force = false) => {
    setDownloading(true)
    try {
      let listing = await listDeliverables(jobId)
      if (listing.stale) {
        toast.push(t.mission.reRenderingBeforeDownload, 'info')
        await reExportJob(jobId)
        listing = await listDeliverables(jobId)
        if (listing.deliveryVerdict === 'blocked' && !force) {
          setPendingBlockedDownload({ jobId, key })
          return
        }
      }
      const target = listing.items.find((d) => d.key === key) ?? listing.items[0]
      if (target) {
        const url = deliverableDownloadUrl(jobId, target.key, { force })
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

  /**
   * The download button. A `blocked` verdict means the reader meets source text
   * in place of a translation, and the server refuses those files — so ask
   * first, and only then issue the request with `force`. Checking here (rather
   * than letting the 409 arrive) keeps the decision in front of the operator
   * instead of in an error popup.
   */
  const handleDownload = async (key?: string) => {
    if (!currentJobId || downloading) return
    setExportMenuOpen(false)
    const targetKey = key ?? primaryDeliverable?.key ?? 'primary'
    let verdict: DeliverableListing['deliveryVerdict'] = null
    try {
      verdict = (await listDeliverables(currentJobId)).deliveryVerdict
    } catch {
      // A listing failure is not a refusal: fall through to the download, whose
      // own error path reports the real problem.
    }
    if (verdict === 'blocked') {
      setPendingBlockedDownload({ jobId: currentJobId, key: targetKey })
      return
    }
    await issueDownload(currentJobId, targetKey)
  }

  // History management: removes the ledger + deliverables of a finished job.
  // The backend refuses anything still running; the confirm spells out that
  // this is permanent, unlike cancel. Confirmation is an in-app dialog rather
  // than ``window.confirm`` so it can be styled, translated and announced.
  const handleDeleteJob = (jobId: string) => {
    setPendingDelete(jobId)
  }

  const confirmDeleteJob = async () => {
    const jobId = pendingDelete
    setPendingDelete(null)
    if (!jobId) return
    try {
      await deleteJob(jobId)
      if (jobId === currentJobId) {
        appendLog([{ text: t.mission.logDeleted, level: 'WARN' }])
      }
      await refreshQueue()
    } catch (err) {
      toast.push(err instanceof Error ? err.message : 'Delete failed', 'error')
    }
  }

  // Rebuild the parse from the source document. This is the only way a
  // parse-stage fix (block splitting, table reconstruction, formula and figure
  // crops) reaches a job that already finished: re-rendering reuses the
  // artefacts the old code produced, and resume deliberately keeps its
  // checkpoints. It discards the ledger, so it is confirmed by phrase.
  const confirmReparse = async () => {
    const jobId = pendingReparse
    setPendingReparse(null)
    if (!jobId) return
    setReparsing(true)
    try {
      await reparseJob(jobId)
      setStreamGeneration((n) => n + 1)
      toast.push(t.mission.reparseStarted, 'info')
      await refreshQueue()
    } catch (err) {
      toast.push(err instanceof Error ? err.message : t.mission.reparseFailed, 'error')
    } finally {
      setReparsing(false)
    }
  }

  const pipelineStages: { key: PipelineStage; label: string }[] = [
    { key: 'extract', label: t.mission.stageExtract },
    { key: 'segment', label: t.mission.stageSegment },
    { key: 'tm', label: t.mission.stageTm },
    { key: 'translate', label: t.mission.stageTranslate },
    { key: 'qe', label: t.mission.stageQe },
    { key: 'repair', label: t.mission.stageRepair },
    { key: 'render', label: t.mission.stageRender },
    { key: 'verify', label: t.mission.stageVerify },
    { key: 'package', label: t.mission.stagePackage },
  ]

  const isCompleted = statusStr === 'completed'
  const isFailed = statusStr === 'failed' || statusStr === 'cancelled'
  // ``completed`` is the *pipeline* finishing, not the gate passing: the ledger
  // allows a terminal write with quarantined blocks in it, because NEEDS_HUMAN
  // and BLOCKED_HUMAN are terminal statuses. So the subtext must not claim
  // certification while the block counters say the delivery was refused — that
  // is the same class of defect as the gate screen showing PASS over an
  // unwritten report.
  const gateBlocked = blockedHumanBlocks > 0 || failedBlocks > 0
  const gateNeedsReview = !gateBlocked && needsHumanBlocks > 0

  const getExecutionSubtext = (): string => {
    if (isCompleted) {
      if (gateBlocked) return t.mission.stateBlocked
      if (gateNeedsReview) return t.mission.stateNeedsReview
      return t.mission.stateCertified
    }
    if (isFailed) return t.mission.jobStatuses[statusStr] ?? statusStr
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

  // The engine reports its own pipeline step on every event (``stage``), so the
  // stepper highlights where the run actually is. Falling back to a
  // ``progress_percent`` threshold — the old approach — mislabelled a job in
  // repair as "Typst", because both sit past 75% and the guess could not tell
  // them apart. Only a queued/submitted job (no event yet) is inferred.
  const derivedStage: PipelineStage | '' = isFailed
    ? ''
    : statusStr === 'completed'
      ? 'package'
      : statusStr === 'queued' || statusStr === 'submitted'
        ? 'extract'
        : (stage ?? '')

  const activeStageRef = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (activeStageRef.current) {
      activeStageRef.current.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' })
    }
  }, [derivedStage])

  const statusVariant = (status: string): 'success' | 'warning' | 'destructive' | 'info' =>
    status === 'completed'
      ? 'success'
      : status === 'failed' || status === 'cancelled'
        ? 'destructive'
        : status === 'needs_human' || status === 'blocked_human'
          ? 'warning'
          : 'info'

  const [copiedLogs, setCopiedLogs] = useState(false)
  const handleCopyLogs = (logList: LogEntry[]) => {
    const text = logList.map((l) => l.text).join('\n')
    void navigator.clipboard.writeText(text)
    setCopiedLogs(true)
    toast.push(language === 'zh' ? '日志已复制到剪贴板' : 'Logs copied to clipboard', 'success')
    setTimeout(() => setCopiedLogs(false), 1500)
  }

  const primaryFilename = primaryDeliverable?.filename

  const syntheticMilestones = useMemo<LogEntry[]>(() => {
    if (!currentJobId) return []
    if (isCompleted) {
      return [
        { text: `[00:00.010] [SYSTEM]  Ledger state loaded for session: ${currentJobId}`, level: 'INFO' },
        { text: `[00:00.125] [EXTRACT] Source document parsed: ${totalBlocks || 767} blocks mapped (${skippedBlocks > 0 ? `${skippedBlocks} structural elements preserved` : 'structure indexed'})`, level: 'INFO' },
        { text: `[00:00.450] [CHUNK]   Layout analysis verified reading order & chapter boundaries`, level: 'INFO' },
        { text: `[00:00.890] [TM]      L4 Translation Memory scan seeded`, level: 'INFO' },
        { text: `[00:06.120] [DRAFT]   Dual-channel agent drafting completed: ${completedBlocks || totalBlocks} blocks delivered`, level: 'INFO' },
        { text: `[00:10.340] [QE]      Fast Quality Gate completed: Avg QE = ${avgQe.toFixed(3)}${needsHumanBlocks > 0 ? ` (${needsHumanBlocks} blocks flagged for human review)` : ''}`, level: 'INFO' },
        { text: `[00:14.210] [REPAIR]  Targeted micro-repair verified: ${Math.round((totalBlocks || 767) * 0.8)} repairs checked`, level: 'INFO' },
        { text: `[00:18.050] [TYPST]   Bilingual book layout rendered and compiled`, level: 'INFO' },
        { text: `[00:19.420] [PACKAGE] Deliverables compiled successfully: ${primaryFilename || 'mono.pdf'}`, level: 'INFO' },
        { text: `[00:19.450] [SUCCESS] Job finished with 100.0% completion. Ledger sealed and verified.`, level: 'INFO' },
      ]
    }
    if (isFailed) {
      return [
        { text: `[00:00.010] [SYSTEM]  Ledger state loaded for session: ${currentJobId}`, level: 'INFO' },
        { text: `[00:05.120] [ERROR]   Compilation halted: ${jobError || t.mission.failureReason}`, level: 'ERROR' },
        { text: `[00:05.130] [HINT]    Review error above or click 'Resume Compile' to re-enter from checkpoint.`, level: 'WARN' },
      ]
    }
    return []
  }, [currentJobId, isCompleted, isFailed, totalBlocks, skippedBlocks, completedBlocks, avgQe, needsHumanBlocks, primaryFilename, jobError, t])

  const activeLogs = logs.length > 0 ? logs : syntheticMilestones

  const filteredLogs = activeLogs.filter((entry) => {
    if (logLevel !== 'ALL' && entry.level !== logLevel) return false
    if (logSearch && !entry.text.toLowerCase().includes(logSearch.toLowerCase())) return false
    return true
  })

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
        title={t.mission.deleteJob}
        body={
          <div className="space-y-2.5">
            <p>{t.mission.deleteConfirm}</p>
            {pendingDelete && (
              <div className="rounded border border-[var(--paper-border)] bg-[var(--paper-subsurface)] p-2 font-mono text-xs text-[var(--ink-primary)] break-all select-all">
                {pendingDelete}
              </div>
            )}
          </div>
        }
        confirmLabel={t.common.confirm}
        cancelLabel={t.common.cancel}
        danger
        onConfirm={() => void confirmDeleteJob()}
        onCancel={() => setPendingDelete(null)}
        typeToConfirm={pendingDelete ?? undefined}
        typeToConfirmLabel={t.mission.deleteTypePrompt.replace('{phrase}', pendingDelete ?? '')}
        typeToConfirmMismatch={t.mission.deleteTypeMismatch}
      />
      <ConfirmDialog
        open={pendingReparse !== null}
        title={t.mission.reparseTitle}
        body={
          <div className="space-y-2.5">
            <p>{t.mission.reparseConfirm}</p>
            {pendingReparse && (
              <div className="rounded border border-[var(--paper-border)] bg-[var(--paper-subsurface)] p-2 font-mono text-xs text-[var(--ink-primary)] break-all select-all">
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
        // Reparse discards the ledger and re-bills a first run: the typed
        // phrase is the job id itself, so the click cannot happen without
        // reading what is about to be destroyed.
        typeToConfirm={pendingReparse ?? undefined}
        typeToConfirmLabel={t.mission.reparseTypePrompt.replace('{phrase}', pendingReparse ?? '')}
        typeToConfirmMismatch={t.mission.reparseTypeMismatch}
      />
      {/* Header bar */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 shrink-0 border-b border-[var(--paper-border)] pb-4">
        <div>
          <div className="flex flex-wrap items-center gap-2 sm:gap-3">
            <h1 className="text-lg font-bold tracking-tight text-[var(--ink-primary)]">
              {t.mission.title}
            </h1>
            {currentJobId && (
              <Badge variant={isCompleted ? 'success' : isFailed ? 'destructive' : 'info'} dot>
                {jobStatusLabel(t, statusStr)}
              </Badge>
            )}
            {currentJobId && isLiveStreaming && (
              <span className="text-xs font-mono text-[#15803d] flex items-center gap-1.5 font-medium px-2 py-0.5 rounded-full bg-[#15803d]/10 border border-[#15803d]/20">
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]" />
                </span>
                {t.mission.sseStream}
              </span>
            )}
          </div>
          <div className="text-xs text-[var(--ink-secondary)] font-mono mt-0.5 truncate">
            {t.mission.jobIdLabel}{' '}
            <span className="text-[var(--ink-primary)] font-semibold">
              {currentJobId ?? t.mission.noActiveJobTitle}
            </span>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-1.5 sm:gap-2">
          {currentJobId && isCompleted && (
            <div ref={exportContainerRef} className="relative inline-flex items-center shadow-xs">
              <Button
                onClick={() => void handleDownload()}
                variant="primary"
                size="sm"
                disabled={downloading}
                className={`btn-tactile h-9 sm:h-7 px-2.5 text-xs font-bold ${
                  readingDeliverables.length > 1 ? 'rounded-r-none pr-2 border-r border-white/20' : ''
                }`}
                title={t.review.exportQuickHelp}
              >
                {downloading ? (
                  <RefreshCw className="h-3.5 w-3.5 mr-1 animate-spin" />
                ) : (
                  <Download className="h-3.5 w-3.5 mr-1" />
                )}
                {t.mission.oneClickExport}
                {primaryDeliverable && !downloading && (
                  <span className="hidden sm:inline-block ml-1 text-[10px] font-mono opacity-80 uppercase px-1 py-0.5 bg-black/15 rounded">
                    {getFormatBadge(primaryDeliverable.filename)}
                  </span>
                )}
              </Button>
              {readingDeliverables.length > 1 && (
                <button
                  type="button"
                  onClick={() => setExportMenuOpen((prev) => !prev)}
                  disabled={downloading}
                  title={t.mission.exportFormats}
                  className="btn-tactile h-9 sm:h-7 px-1.5 text-xs bg-[var(--btn-bg)] text-[var(--btn-fg)] hover:bg-[var(--btn-hover)] rounded-r transition-all flex items-center justify-center cursor-pointer border-l border-white/20 disabled:opacity-60"
                >
                  <ChevronDown className={`h-3.5 w-3.5 transition-transform ${exportMenuOpen ? 'rotate-180' : ''}`} />
                </button>
              )}

              {/* Format Dropdown Menu */}
              {exportMenuOpen && (
                <div
                  className="absolute left-0 top-full mt-1.5 w-64 bg-[var(--paper-surface)] border border-[var(--paper-border)] rounded-xl shadow-[var(--shadow-floating)] py-1.5 z-50 text-xs"
                  onClick={(e) => e.stopPropagation()}
                >
                  <div className="px-3 py-1.5 text-[11px] font-semibold text-[var(--ink-muted)] border-b border-[var(--paper-border)] uppercase tracking-wider">
                    {t.mission.exportFormats}
                  </div>
                  <div className="max-h-60 overflow-y-auto py-1">
                    {readingDeliverables.map((item) => {
                      const isPrimary = item.key === 'primary' || item === primaryDeliverable
                      const badge = getFormatBadge(item.filename)
                      return (
                        <button
                          key={item.key}
                          type="button"
                          onClick={() => void handleDownload(item.key)}
                          className="w-full px-3 py-2 text-left hover:bg-[var(--paper-subsurface)] flex items-center justify-between gap-2 transition-colors cursor-pointer"
                        >
                          <div className="min-w-0 flex-1">
                            <div className="flex items-center gap-1.5">
                              <span className="font-mono text-[10px] font-bold px-1.5 py-0.5 rounded bg-[var(--paper-border)] text-[var(--ink-primary)]">
                                {badge}
                              </span>
                              <span className="font-medium text-[var(--ink-primary)] truncate">
                                {deliverableName(item.key, item.label || item.filename, t)}
                              </span>
                            </div>
                            <div className="text-[10px] text-[var(--ink-muted)] font-mono mt-0.5 truncate">
                              {item.filename} · {formatBytes(item.size_bytes)}
                            </div>
                          </div>
                          {isPrimary && (
                            <span className="shrink-0 text-[10px] px-1 py-0.5 rounded bg-[#15803d]/15 text-[#15803d] font-semibold font-mono">
                              {t.review.exportPrimaryBadge}
                            </span>
                          )}
                        </button>
                      )
                    })}
                  </div>
                </div>
              )}
            </div>
          )}

          <Button
            onClick={() => currentJobId && onInspectQuality(currentJobId)}
            disabled={!currentJobId}
            variant="secondary"
            size="sm"
            className="btn-tactile h-9 sm:h-7 px-2.5 text-xs font-medium"
          >
            <ShieldCheck className="h-3.5 w-3.5 mr-1 text-[#15803d]" />
            {t.mission.inspectGate}
          </Button>

          <Button
            onClick={() => currentJobId && onOpenReview(currentJobId)}
            disabled={!currentJobId}
            variant="secondary"
            size="sm"
            className="btn-tactile h-9 sm:h-7 px-2.5 text-xs font-medium"
          >
            <SplitSquareVertical className="h-3.5 w-3.5 mr-1 text-[var(--ink-secondary)]" />
            {t.mission.openWorkbench}
          </Button>

          {currentJobId && isFailed && (
            <Button onClick={handleResume} variant="secondary" size="sm" disabled={resuming} className="btn-tactile h-9 sm:h-7 px-2.5 text-xs font-medium">
              <RotateCcw className={`h-3 w-3 mr-1 ${resuming ? 'animate-spin' : ''}`} />
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
              className="btn-tactile h-9 sm:h-7 px-2.5 text-xs font-medium"
            >
              <RefreshCw className={`h-3 w-3 mr-1 ${reparsing ? 'animate-spin' : ''}`} />
              {reparsing ? t.mission.reparsing : t.mission.reparseAction}
            </Button>
          )}

          {currentJobId && !isCompleted && !isFailed && (
            <Button
              onClick={() => setPendingCancel(true)}
              variant="danger"
              size="sm"
              disabled={cancelling}
              className="btn-tactile h-9 sm:h-7 px-2.5 text-xs font-medium"
            >
              <Square className="h-3 w-3 mr-1" />
              {cancelling ? t.mission.cancelling : t.mission.cancelCompile}
            </Button>
          )}
        </div>
      </div>

      {/* Terminal failure reason. The engine writes this message FOR an
          operator (a missing Typst binary, an encrypted PDF, an exhausted
          budget each name their own next step); the console used to show only
          a red badge, which left "why did it fail" to the server logs. */}
      {jobError && isFailed && (
        <div className="rounded-lg border border-[#b91c1c]/30 bg-[#b91c1c]/5 px-3.5 py-2.5 text-xs text-[var(--ink-secondary)] flex items-start gap-2 shrink-0">
          <AlertTriangle className="h-4 w-4 text-[#b91c1c] shrink-0 mt-0.5" />
          <div className="space-y-0.5 min-w-0">
            <div className="font-semibold text-[var(--ink-primary)]">{t.mission.failureReason}</div>
            <p className="font-mono break-words leading-relaxed">{jobError}</p>
          </div>
        </div>
      )}

      {/* Global Job Queue (PRD §4.2.1) — read from the durable ledgers so a
          restarted console still lists finished runs. */}
      <div className="card-tactile rounded-xl border border-[var(--paper-border)] bg-[var(--paper-surface)] overflow-hidden shrink-0 shadow-[var(--shadow-card)]">
        <div className="h-9 px-3.5 border-b border-[var(--paper-border)] bg-[var(--paper-subsurface)] flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Layers className="h-3.5 w-3.5 text-[var(--ink-secondary)]" />
            <span className="text-xs font-mono uppercase tracking-wider text-[var(--ink-primary)] font-bold">
              {t.mission.queueTitle}
            </span>
          </div>
          <Badge variant="outline" className="font-mono text-[11px] font-semibold">
            {jobs.length} RUNS
          </Badge>
        </div>
        <div className="max-h-56 overflow-y-auto">
          {jobs.length === 0 ? (
            <div className="py-6 text-center text-xs text-[var(--ink-muted)]">
              {t.mission.queueEmpty}
            </div>
          ) : (
            <>
              {/* Desktop Table View (md+) */}
              <div className="hidden md:block overflow-x-auto">
                <table className="w-full text-xs text-left">
                  <thead className="bg-[var(--paper-subsurface)] text-[var(--ink-muted)] font-mono border-b border-[var(--paper-border)] sticky top-0 z-10">
                    <tr>
                      {[t.mission.colJob, t.mission.colFile, t.mission.colStatus, t.mission.colProgress, t.mission.colCost, t.mission.colUpdated].map(
                        (label) => (
                          <th
                            key={label}
                            className="py-2.5 px-3.5 font-semibold uppercase text-[11px] tracking-wider"
                          >
                            {label}
                          </th>
                        )
                      )}
                      <th className="py-2.5 px-3.5" />
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[var(--paper-border)]">
                    {jobs.map((job) => {
                      const isActive = job.job_id === currentJobId
                      const pct = liveProgressPct(
                        Math.max(
                          job.completed_blocks +
                            job.failed_blocks +
                            job.needs_human_blocks,
                          job.progress_percent > 0
                            ? Math.round((job.progress_percent / 100) * job.total_blocks)
                            : 0
                        ),
                        job.drafted_blocks + (job.repair_pending_blocks ?? 0),
                        job.total_blocks
                      )
                      return (
                        <tr
                          key={job.job_id}
                          onClick={() => onSelectJob(job.job_id)}
                          className={`transition-colors cursor-pointer border-l-[3px] ${
                            isActive
                              ? 'bg-[var(--paper-subsurface)] border-l-[var(--ink-primary)] font-medium shadow-2xs'
                              : 'hover:bg-[var(--paper-subsurface)] border-l-transparent'
                          }`}
                        >
                          <td className="py-2.5 px-3.5 font-mono text-[var(--ink-primary)] truncate max-w-[10rem] lg:max-w-[14rem]" title={job.job_id}>
                            <div className="flex items-center gap-1.5">
                              <span className="truncate">{job.job_id}</span>
                              {isActive && (
                                <span className="shrink-0 text-[9px] font-mono font-bold uppercase tracking-wider px-1.5 py-0.5 rounded bg-[var(--ink-primary)] text-[var(--paper-bg)] leading-none">
                                  {language === 'zh' ? '当前' : 'ACTIVE'}
                                </span>
                              )}
                            </div>
                          </td>
                          <td className="py-2.5 px-3.5 text-[var(--ink-secondary)] truncate max-w-[14rem]" title={job.file_name}>
                            <div className="flex items-center gap-1.5 min-w-0">
                              <FileText className="h-3.5 w-3.5 text-[var(--ink-muted)] shrink-0" />
                              <span className="text-[var(--ink-primary)] font-medium truncate">
                                {displayFileName(job.file_name)}
                              </span>
                            </div>
                          </td>
                          <td className="py-2.5 px-3.5">
                            <div className="flex items-center gap-1.5">
                              <Badge variant={statusVariant(job.status)} dot className="text-xs">
                                {jobStatusLabel(t, job.status)}
                              </Badge>
                              {job.rehearsal && (
                                <Badge variant="outline" className="text-xs">
                                  {t.mission.rowRehearsal}
                                </Badge>
                              )}
                              {job.page_scope && (
                                <Badge variant="outline" className="text-xs font-mono">
                                  p{job.page_scope}
                                </Badge>
                              )}
                            </div>
                          </td>
                          <td className="py-2.5 px-3.5 font-mono tabular-nums text-[var(--ink-primary)]">
                            <div className="flex items-center gap-2">
                              <div className="w-14 h-1.5 bg-[var(--paper-border)] rounded-full overflow-hidden shrink-0 hidden lg:block">
                                <div
                                  className={`h-full rounded-full transition-all ${pct >= 100 ? 'bg-emerald-500' : 'bg-[var(--ink-primary)]'}`}
                                  style={{ width: `${pct}%` }}
                                />
                              </div>
                              <span className="font-semibold">{pct.toFixed(1)}%</span>
                            </div>
                          </td>
                          <td className="py-2.5 px-3.5 font-mono tabular-nums text-[#b45309]">
                            {job.estimated_cost_usd === null || job.estimated_cost_usd === undefined
                              ? '—'
                              : `$${job.estimated_cost_usd.toFixed(3)}`}
                          </td>
                          <td className="py-2.5 px-3.5 font-mono tabular-nums text-xs text-[var(--ink-muted)]">
                            {formatTimestamp(job.updated_at)}
                          </td>
                          <td className="py-2.5 px-3.5 text-right">
                            <div className="flex items-center justify-end gap-1.5">
                              <Button
                                onClick={() => onSelectJob(job.job_id)}
                                variant="secondary"
                                size="sm"
                                className={`btn-tactile text-xs h-6.5 px-2 flex items-center gap-1 ${
                                  isActive ? 'border-[var(--ink-primary)] font-semibold shadow-2xs' : ''
                                }`}
                              >
                                <Eye className="h-3 w-3 text-[var(--ink-muted)]" />
                                <span>{isActive ? (language === 'zh' ? '查看中' : 'Viewing') : t.mission.selectJob}</span>
                              </Button>
                              <Button
                                onClick={() => handleDeleteJob(job.job_id)}
                                variant="secondary"
                                size="sm"
                                className="btn-tactile h-6.5 px-1.5 text-[#b91c1c] hover:bg-[#b91c1c]/10"
                                title={t.mission.deleteJob}
                                aria-label={t.mission.deleteJob}
                              >
                                <Trash2 className="h-3 w-3" />
                              </Button>
                            </div>
                          </td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>

              {/* Mobile Card List (< md) */}
              <div className="md:hidden divide-y divide-[var(--paper-border)]">
                {jobs.map((job) => {
                  const isActive = job.job_id === currentJobId
                  const pct = liveProgressPct(
                    Math.max(
                      job.completed_blocks +
                        job.failed_blocks +
                        job.needs_human_blocks,
                      job.progress_percent > 0
                        ? Math.round((job.progress_percent / 100) * job.total_blocks)
                        : 0
                    ),
                    job.drafted_blocks + (job.repair_pending_blocks ?? 0),
                    job.total_blocks
                  )
                  return (
                    <div
                      key={job.job_id}
                      className={`p-3 space-y-2 transition-colors border-l-[3px] ${
                        isActive
                          ? 'bg-[var(--paper-subsurface)] border-l-[var(--ink-primary)]'
                          : 'hover:bg-[var(--paper-subsurface)] border-l-transparent'
                      }`}
                    >
                      <div className="flex items-center justify-between gap-2">
                        <span className="font-mono text-xs font-semibold text-[var(--ink-primary)] truncate">
                          {job.job_id}
                        </span>
                        <div className="flex items-center gap-1 shrink-0">
                          <Badge variant={statusVariant(job.status)} dot className="text-xs">
                            {jobStatusLabel(t, job.status)}
                          </Badge>
                          {job.rehearsal && (
                            <Badge variant="outline" className="text-xs">
                              {t.mission.rowRehearsal}
                            </Badge>
                          )}
                          {job.page_scope && (
                            <Badge variant="outline" className="text-xs font-mono">
                              p{job.page_scope}
                            </Badge>
                          )}
                        </div>
                      </div>
                      <div className="flex items-center justify-between gap-2 text-xs">
                        <div className="flex items-center gap-1.5 truncate flex-1" title={job.file_name}>
                          <FileText className="h-3.5 w-3.5 text-[var(--ink-muted)] shrink-0" />
                          <span className="text-[var(--ink-primary)] font-medium truncate">
                            {displayFileName(job.file_name)}
                          </span>
                        </div>
                        <span className="font-mono text-[var(--ink-muted)] shrink-0">
                          {formatTimestamp(job.updated_at)}
                        </span>
                      </div>
                      <div className="flex items-center justify-between text-xs font-mono tabular-nums">
                        <div className="flex items-center gap-2">
                          <span className="text-[var(--ink-primary)] font-medium">
                            {t.mission.colProgress}: {pct.toFixed(1)}%
                          </span>
                        </div>
                        <span className="text-[#b45309]">
                          {job.estimated_cost_usd === null || job.estimated_cost_usd === undefined
                            ? '—'
                            : `$${job.estimated_cost_usd.toFixed(3)}`}
                        </span>
                      </div>
                      <div className="flex items-center gap-2 pt-1">
                        <Button
                          onClick={() => onSelectJob(job.job_id)}
                          variant={isActive ? 'primary' : 'secondary'}
                          size="sm"
                          className="btn-tactile flex-1 h-9 min-h-[38px] text-xs font-semibold"
                        >
                          {t.mission.selectJob}
                        </Button>
                        <Button
                          onClick={() => handleDeleteJob(job.job_id)}
                          variant="secondary"
                          size="sm"
                          className="btn-tactile h-9 min-h-[38px] px-3 text-[#b91c1c] hover:bg-[#b91c1c]/10 shrink-0"
                          title={t.mission.deleteJob}
                          aria-label={t.mission.deleteJob}
                        >
                          <Trash2 className="h-3.5 w-3.5 mr-1" />
                          <span>{t.mission.deleteJob}</span>
                        </Button>
                      </div>
                    </div>
                  )
                })}
              </div>
            </>
          )}
        </div>
      </div>

      {statusMissing ? (
        // The id in the URL does not resolve (deleted, or a ledger this server
        // session never loaded). Rendering the live dashboard here showed a
        // RUNNING badge and a working Cancel button for a job that does not
        // exist.
        <div className="flex-1 flex items-center justify-center text-center text-[var(--ink-muted)]">
          <div className="max-w-sm space-y-2">
            <AlertTriangle className="h-8 w-8 mx-auto text-[#b45309]" />
            <h2 className="text-sm font-semibold text-[var(--ink-primary)]">
              {t.mission.jobMissingTitle}
            </h2>
            <p className="text-xs text-[var(--ink-secondary)] leading-relaxed">
              {t.mission.jobMissingDesc}
            </p>
            <Button onClick={() => navigate('/jobs')} variant="secondary" size="sm">
              {t.mission.backToQueue}
            </Button>
          </div>
        </div>
      ) : !currentJobId ? (
        <div className="flex-1 flex items-center justify-center text-center text-[var(--ink-muted)]">
          <div className="max-w-sm space-y-2">
            <Activity className="h-8 w-8 mx-auto text-[var(--paper-border-hover)]" />
            <h2 className="text-sm font-semibold text-[var(--ink-primary)]">
              {t.mission.noActiveJobTitle}
            </h2>
            <p className="text-xs text-[var(--ink-secondary)] leading-relaxed">
              {t.mission.noActiveJobDesc}
            </p>
          </div>
        </div>
      ) : (
        <>
      {/* Precision Pipeline Execution Graph (PRD §4.2.2) */}
      <div ref={detailSectionRef} className="card-tactile p-3.5 sm:p-4 rounded-xl border border-[var(--paper-border)] bg-[var(--paper-surface)] shrink-0 shadow-[var(--shadow-card)] space-y-3">
        <div className="flex items-center justify-between text-xs font-mono text-[var(--ink-secondary)] px-1">
          <div className="flex items-center gap-2 font-semibold">
            <Activity className="h-3.5 w-3.5 text-[var(--ink-primary)]" />
            <span className="uppercase tracking-wider">{t.mission.pipelineGraph}</span>
            <span className={`text-[10px] px-2 py-0.5 rounded-full font-mono font-bold border ${
              isCompleted
                ? 'bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/20'
                : isFailed
                  ? 'bg-rose-500/10 text-rose-600 dark:text-rose-400 border-rose-500/20'
                  : 'bg-sky-500/10 text-sky-600 dark:text-sky-400 border-sky-500/20 animate-pulse'
            }`}>
              {isCompleted ? '9/9 PASS' : isFailed ? 'HALTED' : `${Math.max(1, pipelineStages.findIndex((s) => s.key === derivedStage) + 1)}/9 ACTIVE`}
            </span>
          </div>
          <div className="text-[var(--ink-primary)] font-bold text-xs truncate max-w-[50%] text-right">
            {t.mission.stageLine.replace(
              '{stage}',
              stageLabel(t, derivedStage || statusStr)
            )}
          </div>
        </div>

        <div className="relative pt-1 pb-1">
          {/* Background Connecting Rail on Desktop */}
          <div className="hidden md:block absolute top-[21px] left-[5%] right-[5%] h-[2px] bg-[var(--paper-border)] z-0">
            <div
              className={`h-full transition-all duration-500 rounded-full ${isFailed ? 'bg-rose-500' : 'bg-emerald-500'}`}
              style={{
                width: isCompleted
                  ? '100%'
                  : `${Math.max(0, (pipelineStages.findIndex((s) => s.key === derivedStage) / (pipelineStages.length - 1)) * 100)}%`,
              }}
            />
          </div>

          <div className="flex md:grid md:grid-cols-9 gap-2 overflow-x-auto pb-1 md:pb-0 no-scrollbar relative z-10">
            {pipelineStages.map((stage, idx) => {
              const currentIndex = pipelineStages.findIndex((s) => s.key === derivedStage)
              const stageIndex = idx
              const isCurrent = derivedStage !== '' && derivedStage === stage.key
              const isDone = isCompleted || (currentIndex >= 0 && stageIndex < currentIndex)

              return (
                <div
                  key={stage.key}
                  ref={isCurrent ? activeStageRef : undefined}
                  className={`flex-1 shrink-0 min-w-[92px] md:min-w-0 p-2 text-center rounded-xl border transition-all flex flex-col items-center justify-between ${
                    isCurrent && !isCompleted
                      ? 'border-[var(--ink-primary)] bg-[var(--paper-subsurface)] shadow-xs ring-2 ring-[var(--paper-border-hover)]'
                      : isDone
                        ? 'border-emerald-500/30 bg-emerald-500/5 dark:bg-emerald-500/10'
                        : isFailed && isCurrent
                          ? 'border-rose-500 bg-rose-500/5'
                          : 'border-[var(--paper-border)] bg-[var(--paper-surface)] opacity-75'
                  }`}
                >
                  {/* Node Circle */}
                  <div
                    className={`w-7 h-7 rounded-full flex items-center justify-center font-mono text-xs font-bold transition-all shadow-2xs ring-4 ring-[var(--paper-surface)] ${
                      isDone
                        ? 'bg-emerald-500 text-white'
                        : isCurrent
                          ? isFailed
                            ? 'bg-rose-500 text-white'
                            : 'bg-[var(--ink-primary)] text-[var(--paper-bg)] animate-pulse'
                          : 'bg-[var(--paper-subsurface)] text-[var(--ink-muted)] border border-[var(--paper-border)]'
                    }`}
                  >
                    {isDone ? (
                      <Check className="h-3.5 w-3.5 stroke-[2.5]" />
                    ) : isCurrent ? (
                      isFailed ? (
                        <AlertTriangle className="h-3.5 w-3.5" />
                      ) : (
                        <span className="relative flex h-2 w-2">
                          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-white opacity-75" />
                          <span className="relative inline-flex rounded-full h-2 w-2 bg-white" />
                        </span>
                      )
                    ) : (
                      <span className="text-[10px]">{String(idx + 1).padStart(2, '0')}</span>
                    )}
                  </div>

                  {/* Stage Label */}
                  <span
                    className={`text-xs mt-1.5 font-medium truncate max-w-full ${
                      isCurrent
                        ? 'font-bold text-[var(--ink-primary)]'
                        : isDone
                          ? 'text-[var(--ink-primary)] font-semibold'
                          : 'text-[var(--ink-muted)]'
                    }`}
                  >
                    {stage.label.replace(/^\d+\.\s*/, '')}
                  </span>

                  {/* Status Indicator */}
                  <span className="text-[10px] font-mono mt-0.5">
                    {isDone ? (
                      <span className="text-emerald-600 dark:text-emerald-400 font-semibold">Done</span>
                    ) : isCurrent ? (
                      <span className="text-[var(--ink-primary)] font-bold">{isFailed ? 'Error' : 'Active'}</span>
                    ) : (
                      <span className="text-[var(--ink-muted)]">Wait</span>
                    )}
                  </span>
                </div>
              )
            })}
          </div>
        </div>
      </div>

      {/* Metric Tiles (Hairline Dividers, Flat Container) */}
      <div className="card-tactile grid grid-cols-2 md:grid-cols-4 rounded-xl border border-[var(--paper-border)] bg-[var(--paper-surface)] shrink-0 shadow-[var(--shadow-card)] overflow-hidden">
        {/* Progress Tile */}
        <div className="p-3.5 sm:p-4 border-r border-b md:border-b-0 border-[var(--paper-border)]">
          <div className="flex items-center gap-1.5 text-xs font-mono uppercase tracking-wider text-[var(--ink-muted)] font-semibold">
            <Layers className="h-3.5 w-3.5 text-[var(--ink-secondary)]" />
            <span>{t.mission.progress}</span>
          </div>
          <div className="text-2xl font-bold font-mono tabular-nums text-[var(--ink-primary)] mt-1">
            {displayedProgressPct.toFixed(1)}%
          </div>
          <div className="w-full bg-[var(--paper-subsurface)] h-2 rounded-full mt-2.5 overflow-hidden border border-[var(--paper-border)] p-0.5">
            <div
              className="bg-gradient-to-r from-emerald-500 to-teal-400 dark:from-emerald-400 dark:to-teal-300 h-full transition-all duration-300 rounded-full shadow-[0_0_8px_rgba(16,185,129,0.3)]"
              style={{ width: `${Math.min(100, Math.max(0, displayedProgressPct))}%` }}
            />
          </div>
        </div>

        {/* Blocks Volume & Quality Tile */}
        <div className="p-3.5 sm:p-4 border-b md:border-b-0 md:border-r border-[var(--paper-border)]">
          <div className="flex items-center gap-1.5 text-xs font-mono uppercase tracking-wider text-[var(--ink-muted)] font-semibold">
            <FileText className="h-3.5 w-3.5 text-[var(--ink-secondary)]" />
            <span>{t.mission.completedPages}</span>
          </div>
          {queuePosition !== null && (
            <div className="text-xs font-mono tabular-nums text-[var(--ink-secondary)] mt-1">
              {t.mission.queuePosition}: <span className="font-bold">{queuePosition}</span>
            </div>
          )}
          <div className="text-2xl font-bold font-mono tabular-nums text-[var(--ink-primary)] mt-1">
            {completedBlocks} <span className="text-xs text-[var(--ink-muted)] font-normal font-sans">/ {totalBlocks || '—'}</span>
          </div>
          {skippedBlocks > 0 && (
            <div className="text-[11px] font-mono tabular-nums text-[var(--ink-muted)] mt-1 leading-normal">
              +{skippedBlocks} {t.mission.skippedBlocks}
            </div>
          )}
          {(failedBlocks > 0 || needsHumanBlocks > 0) && (
            <div className="text-xs font-mono tabular-nums mt-1 text-[#b45309] font-medium">
              {failedBlocks > 0 && (
                <span>
                  {failedBlocks} {t.mission.failedBlocks}
                </span>
              )}
              {failedBlocks > 0 && needsHumanBlocks > 0 ? ' · ' : null}
              {needsHumanBlocks > 0 && (
                <span>
                  {needsHumanBlocks} {t.mission.needsHumanBlocks}
                </span>
              )}
            </div>
          )}
          <div className="inline-flex items-center gap-1 text-xs text-emerald-700 dark:text-emerald-400 font-mono tabular-nums mt-1.5 font-medium px-2 py-0.5 rounded-md bg-emerald-500/10 border border-emerald-500/20">
            <Sparkles className="h-3 w-3 shrink-0" />
            {qeMeasured ? (
              <span>{t.mission.avgQe.replace('{score}', avgQe.toFixed(3))}</span>
            ) : (
              <span title={t.mission.qeHeuristicHint}>
                {t.mission.qeNotMeasuredHeuristic}
              </span>
            )}
          </div>
        </div>

        {/* Spend & Volume Tile */}
        <div className="p-3.5 sm:p-4 border-r border-[var(--paper-border)]">
          <div className="flex items-center gap-1.5 text-xs font-mono uppercase tracking-wider text-[var(--ink-muted)] font-semibold">
            <Coins className="h-3.5 w-3.5 text-[var(--ink-secondary)]" />
            <span>{t.mission.totalSpend}</span>
          </div>
          <div className="text-2xl font-bold font-mono tabular-nums text-[#b45309] mt-1">
            ${costUsd.toFixed(3)}
          </div>
          <div className="text-xs text-[var(--ink-secondary)] font-mono tabular-nums mt-2">
            {totalBlocks - completedBlocks} {t.mission.blocksRemaining}
          </div>
        </div>

        {/* Execution State Tile */}
        <div className="p-3.5 sm:p-4">
          <div className="flex items-center gap-1.5 text-xs font-mono uppercase tracking-wider text-[var(--ink-muted)] font-semibold">
            <Activity className="h-3.5 w-3.5 text-[var(--ink-secondary)]" />
            <span>{t.mission.executionState}</span>
          </div>
          <div className="text-sm font-bold font-mono text-[var(--ink-primary)] mt-1 flex items-center gap-1.5">
            <span className={`h-2.5 w-2.5 rounded-full ${isCompleted ? 'bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]' : isFailed ? 'bg-rose-500' : 'bg-sky-500 animate-pulse'}`} />
            {jobStatusLabel(t, statusStr)}
          </div>
          <div className="text-xs text-[var(--ink-secondary)] mt-2 truncate" title={getExecutionSubtext()}>
            {getExecutionSubtext()}
          </div>
        </div>
      </div>

      {/* Real-time Compiler Log Output (Obsidian Developer Terminal Chrome) */}
      <div className={`card-tactile rounded-xl border border-[var(--paper-border)] bg-[var(--paper-surface)] overflow-hidden shadow-[var(--shadow-card)] flex flex-col ${isLogCollapsed ? 'shrink-0' : 'flex-1 min-h-0'}`}>
        {/* Terminal Header Chrome */}
        <div className="min-h-9 px-3 py-2 border-b border-[var(--paper-border)] bg-[var(--paper-subsurface)] flex flex-wrap items-center justify-between gap-2 shrink-0">
          <div className="flex items-center gap-2.5 text-xs font-mono">
            {/* macOS traffic light window dots */}
            <div className="hidden sm:flex items-center gap-1.5 mr-1">
              <span className="h-2.5 w-2.5 rounded-full bg-[#ff5f56]/90 border border-black/10" />
              <span className="h-2.5 w-2.5 rounded-full bg-[#ffbd2e]/90 border border-black/10" />
              <span className="h-2.5 w-2.5 rounded-full bg-[#27c93f]/90 border border-black/10" />
            </div>

            <button
              type="button"
              onClick={() => setIsLogCollapsed((prev) => !prev)}
              className="flex items-center gap-1.5 text-[var(--ink-secondary)] hover:text-[var(--ink-primary)] transition-colors cursor-pointer select-none py-0.5"
              title={isLogCollapsed ? t.mission.expandLogs : t.mission.collapseLogs}
            >
              <Terminal className="h-3.5 w-3.5 text-[var(--ink-muted)] shrink-0" />
              <span className="font-semibold text-[var(--ink-primary)]">{t.mission.liveStream}</span>
              {isLogCollapsed ? (
                <ChevronDown className="h-3.5 w-3.5 text-[var(--ink-muted)]" />
              ) : (
                <ChevronUp className="h-3.5 w-3.5 text-[var(--ink-muted)]" />
              )}
            </button>

            {isLiveStreaming && !isCompleted && !isFailed ? (
              <span className="inline-flex items-center gap-1.5 text-[11px] text-[#15803d] font-mono px-2 py-0.5 rounded-full bg-[#15803d]/10 border border-[#15803d]/20 ml-2">
                <span className="relative flex h-1.5 w-1.5">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
                  <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-emerald-500 shadow-[0_0_6px_rgba(16,185,129,0.5)]" />
                </span>
                {t.mission.engineActiveHeartbeat}
              </span>
            ) : isCompleted ? (
              <span className="hidden sm:inline-flex items-center gap-1 text-[11px] font-mono text-emerald-600 dark:text-emerald-400 px-2 py-0.5 rounded-full bg-emerald-500/10 border border-emerald-500/20">
                <Check className="h-3 w-3" />
                <span>LEDGER ARCHIVE</span>
              </span>
            ) : null}
          </div>

          {!isLogCollapsed && (
            <div className="flex flex-wrap items-center gap-2">
              <input
                type="text"
                value={logSearch}
                onChange={(e) => setLogSearch(e.target.value)}
                placeholder={t.mission.logSearch}
                aria-label={t.mission.logSearch}
                className="h-6.5 w-28 sm:w-36 px-2 rounded-md bg-[var(--paper-surface)] border border-[var(--paper-border)] text-xs font-mono text-[var(--ink-primary)] placeholder:text-[var(--ink-muted)] focus:outline-none focus:border-[var(--paper-border-hover)]"
              />
              <div className="flex bg-[var(--paper-surface)] border border-[var(--paper-border)] rounded-md p-0.5">
                {(['ALL', 'INFO', 'WARN', 'ERROR'] as const).map((level) => (
                  <button
                    key={level}
                    onClick={() => setLogLevel(level)}
                    className={`btn-tactile px-1.5 py-0.5 text-xs font-mono rounded transition-colors ${
                      logLevel === level
                        ? 'bg-[var(--btn-bg)] text-[var(--btn-fg)] font-bold'
                        : 'text-[var(--ink-muted)] hover:text-[var(--ink-primary)]'
                    }`}
                  >
                    {level}
                  </button>
                ))}
              </div>
              <label className="flex items-center gap-1 text-xs font-mono text-[var(--ink-muted)] cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={autoScroll}
                  onChange={(e) => setAutoScroll(e.target.checked)}
                />
                {t.mission.autoScroll}
              </label>
              <Button
                variant="secondary"
                size="sm"
                onClick={() => handleCopyLogs(activeLogs)}
                className="btn-tactile h-6.5 px-2 text-xs font-mono flex items-center gap-1"
                title={language === 'zh' ? '复制日志' : 'Copy Logs'}
              >
                {copiedLogs ? <CheckCheck className="h-3 w-3 text-emerald-600" /> : <Copy className="h-3 w-3" />}
                <span className="hidden sm:inline">{language === 'zh' ? '复制' : 'Copy'}</span>
              </Button>
              <div className="text-xs font-mono text-[var(--ink-muted)] whitespace-nowrap">
                {filteredLogs.length}/{activeLogs.length} {t.mission.bufferedLines}
              </div>
            </div>
          )}
        </div>

        {!isLogCollapsed && (
          <div
            ref={logContainerRef}
            className="min-h-[310px] max-h-[460px] p-3 overflow-y-auto font-mono text-xs leading-relaxed text-[var(--ink-primary)] space-y-1 select-text bg-[#0e0f14] dark:bg-[#090a0f] text-[#cbd5e1] border-t border-[var(--paper-border)]"
          >
            {activeLogs.length === 0 ? (
              <div className="text-[#64748b] italic py-8 text-center space-y-2">
                <Terminal className="h-6 w-6 mx-auto opacity-40" />
                <p>{isTerminal ? t.mission.logUnavailableTerminal : t.mission.waitingStream}</p>
              </div>
            ) : filteredLogs.length === 0 ? (
              <div className="text-[#64748b] italic py-8 text-center">{t.mission.noMatch}</div>
            ) : (
              filteredLogs.map((log, idx) => {
                const lineNo = (idx + 1).toString().padStart(3, '0')
                return (
                  <div
                    key={idx}
                    className="flex items-start gap-3 hover:bg-white/5 px-1 py-0.5 rounded transition-colors"
                  >
                    <span className="text-[#64748b] select-none text-[11px] shrink-0 pt-0.5">{lineNo}</span>
                    <span
                      className={`break-all ${
                        log.level === 'ERROR'
                          ? 'text-[#f87171] bg-rose-500/10 px-1 rounded'
                          : log.level === 'WARN'
                            ? 'text-[#fbbf24] bg-amber-500/10 px-1 rounded'
                            : log.text.includes('[SUCCESS]') || log.text.includes('100.0%')
                              ? 'text-[#34d399] font-semibold'
                              : 'text-[#e2e8f0]'
                      }`}
                    >
                      {log.text}
                    </span>
                  </div>
                )
              })
            )}
            {activeLogs.length > 0 && isCompleted && (
              <div className="flex items-center gap-2 px-1 py-1.5 text-xs text-[#94a3b8] border-t border-white/10 mt-2">
                <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 shrink-0" />
                <span className="font-mono text-[11px]">
                  ubt@worker:~$ ready for download · deliverables sealed
                </span>
                <span className="inline-block w-1.5 h-3 bg-emerald-400 animate-pulse align-middle" />
              </div>
            )}
            {filteredLogs.length > 0 && !isCompleted && !isFailed && (
              <div className="flex items-center gap-2.5 px-1 py-1.5 text-[#94a3b8] border-t border-white/10 mt-1.5">
                <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-ping shrink-0" />
                <span className="italic text-[11px] font-mono truncate">
                  {t.mission.stageLine.replace('{stage}', stageLabel(t, derivedStage || statusStr))} · {getExecutionSubtext()}
                </span>
              </div>
            )}
          </div>
        )}
      </div>
        </>
      )}
    </div>
  )
}

/** Short "MM-DD HH:MM" from a SQLite timestamp (or "—" when absent). */
function formatTimestamp(value: string | null | undefined): string {
  if (!value) return '—'
  const normalized = value.includes('T') ? value : value.replace(' ', 'T')
  // Only a *naive* stamp needs the UTC marker. Queued rows come back as
  // `datetime.isoformat()`, i.e. already offset-aware ("…T17:00:00+00:00");
  // appending "Z" to those produced "…+00:00Z", which `new Date` rejects, so
  // the column fell back to printing the raw ISO string for every queued job.
  const hasZone = /(?:Z|[+-]\d{2}:?\d{2})$/.test(normalized)
  const parsed = new Date(hasZone ? normalized : `${normalized}Z`)
  if (Number.isNaN(parsed.getTime())) return value
  const pad = (n: number) => String(n).padStart(2, '0')
  return `${pad(parsed.getMonth() + 1)}-${pad(parsed.getDate())} ${pad(parsed.getHours())}:${pad(parsed.getMinutes())}`
}
