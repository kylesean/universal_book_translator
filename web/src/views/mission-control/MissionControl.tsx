import React, { useCallback, useEffect, useState, useRef } from 'react'
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
} from 'lucide-react'
import { Button } from '@/components/ui/Button'
import { Badge } from '@/components/ui/Badge'
import { ConfirmDialog } from '@/components/ui/ConfirmDialog'
import {
  getJobStatus,
  cancelJob,
  resumeJob,
  deleteJob,
  listJobs,
  listDeliverables,
  deliverableDownloadUrl,
  reExportJob,
  reparseJob,
  subscribeJobProgress,
  type JobSummary,
  type PipelineStage,
  type ProgressStreamFrame,
} from '@/api/client'
import { useI18n } from '@/i18n/useI18n'
import { useToast } from '@/components/ui/useToast'

type LogLevel = 'INFO' | 'WARN' | 'ERROR'

interface LogEntry {
  text: string
  level: LogLevel
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
  const { t } = useI18n()
  const toast = useToast()
  const navigate = useNavigate()
  const { jobId } = useParams<{ jobId: string }>()
  const currentJobId = jobId ?? null
  const onSelectJob = (id: string) => navigate(`/jobs/${id}`)
  const onInspectQuality = (id: string) => navigate(`/jobs/${id}/quality`)
  const onOpenReview = (id: string) => navigate(`/jobs/${id}/review`)
  // Seeded from the cross-route history so returning to this job restores what
  // the stream had already said, instead of the two frames a terminal job still
  // replays (see ``LOG_HISTORY``). The lazy initializer runs once per mount.
  const [logs, setLogs] = useState<LogEntry[]>(() => historyFor(jobId ?? null))

  // Every write goes through here so the in-memory history can never drift from
  // what the panel shows. Two sinks, one update: state for this render, the map
  // for the next mount.
  // ``currentJobId`` is a dependency rather than a ref: the SSE effect below
  // re-subscribes when the job changes anyway, so a stable identity buys
  // nothing, and reading a ref during render is a real hydration hazard.
  const appendLog = useCallback(
    (entries: LogEntry[]) => {
      setLogs((prev) => {
        const next = [...prev, ...entries].slice(-LOG_HISTORY_MAX_LINES)
        rememberLogs(currentJobId, next)
        return next
      })
    },
    [currentJobId]
  )
  const [logSearch, setLogSearch] = useState('')
  const [logLevel, setLogLevel] = useState<'ALL' | LogLevel>('ALL')
  const [autoScroll, setAutoScroll] = useState(true)
  const [isLiveStreaming, setIsLiveStreaming] = useState(false)
  const [statusStr, setStatusStr] = useState<string>('running')
  const [queuePosition, setQueuePosition] = useState<number | null>(null)
  const [stage, setStage] = useState<PipelineStage | null>(null)
  const [progressPct, setProgressPct] = useState<number>(0)
  const [completedBlocks, setCompletedBlocks] = useState<number>(0)
  const [totalBlocks, setTotalBlocks] = useState<number>(0)
  const [costUsd, setCostUsd] = useState<number>(0)
  const [avgQe, setAvgQe] = useState<number>(0)
  // Blocks the skip rule finished without translating them (images, list
  // markers, short headings). They are excluded from total/completed so the
  // percentage reflects translation work, and reported separately so the panel
  // does not appear to be describing a shorter book than the source.
  const [skippedBlocks, setSkippedBlocks] = useState<number>(0)
  // Blocks that ended in a failure. The API always carried this; the panel never
  // rendered it, so a job with 15 failed blocks read as a clean 100%.
  const [failedBlocks, setFailedBlocks] = useState<number>(0)
  const [needsHumanBlocks, setNeedsHumanBlocks] = useState<number>(0)
  // Mirrors of the counters above, for the SSE log tail. The stream callback is
  // created once and closes over its first render's state, so reading the state
  // variables directly there would print the initial zeros on a terminal frame
  // that carries no counters — the "0/0 blocks" bug — and would also need every
  // counter in the callback's dependency list, re-subscribing the stream on each
  // frame. Refs are the standard way out: the callback stays stable, the value
  // is always current.
  const completedRef = useRef(0)
  const totalRef = useRef(0)
  const skippedRef = useRef(0)
  const failedRef = useRef(0)
  // The engine sends 0.0 as the "nothing scored" sentinel under the heuristic
  // QE engine; a measured average is never exactly 0 in practice (defect bands
  // floor at 0.30), so 0 renders as "not measured" instead of a failing score.
  const qeMeasured = avgQe > 0
  // The run is over, so its live log can no longer be replayed.
  const isTerminal =
    statusStr === 'completed' || statusStr === 'failed' || statusStr === 'cancelled'
  const [jobs, setJobs] = useState<JobSummary[]>([])
  const [resuming, setResuming] = useState(false)
  const [downloading, setDownloading] = useState(false)
  const [pendingDelete, setPendingDelete] = useState<string | null>(null)
  // Confirmation for the destructive re-parse: every block is discarded and
  // the book re-translated at full cost, so it takes a typed phrase, not a click.
  const [pendingReparse, setPendingReparse] = useState<string | null>(null)
  const [reparsing, setReparsing] = useState(false)
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

    // Switching jobs inside this view does not remount it, so the previous
    // job's lines have to be swapped out explicitly — otherwise job B's panel
    // opens showing job A's log above job B's first frame.
    // oxlint-disable-next-line react/set-state-in-effect
    setLogs(historyFor(currentJobId))

    let isMounted = true
    const applyStatus = (res: Record<string, unknown>) => {
      if (res.status) setStatusStr(String(res.status))
      if (typeof res.stage === 'string') setStage(res.stage as PipelineStage)
      if (typeof res.progress_percent === 'number') setProgressPct(res.progress_percent)
      if (typeof res.completed_blocks === 'number') {
        setCompletedBlocks(res.completed_blocks)
        completedRef.current = res.completed_blocks
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

    const fetchStatus = async () => {
      try {
        const res = await getJobStatus(currentJobId)
        if (isMounted) {
          applyStatus(res as unknown as Record<string, unknown>)
        }
      } catch {
        // Polling catch
      }
    }

    fetchStatus()
    const timer = setInterval(fetchStatus, 3000)

    const unsubscribe = subscribeJobProgress(
      currentJobId,
      (frame: ProgressStreamFrame) => {
        if (!isMounted) return
        if (frame.status) setStatusStr(frame.status)
        if (frame.stage) setStage(frame.stage)
        if (typeof frame.progress_percent === 'number') setProgressPct(frame.progress_percent)
        if (typeof frame.completed_blocks === 'number') setCompletedBlocks(frame.completed_blocks)
        if (typeof frame.total_blocks === 'number') setTotalBlocks(frame.total_blocks)
        if (typeof frame.estimated_cost_usd === 'number') setCostUsd(frame.estimated_cost_usd)
        if (typeof frame.current_avg_qe === 'number') setAvgQe(frame.current_avg_qe)
        if (typeof frame.completed_blocks === 'number') completedRef.current = frame.completed_blocks
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

        // Log the engine's own message when it has one — the stream is a build
        // log, not a counter echo. The counters stay on the line as the tail so
        // a frame without a message still reads as progress.
        const level: LogLevel = frame.error ? 'ERROR' : frame.status === 'failed' ? 'ERROR' : 'INFO'
        const qePart =
          (frame.current_avg_qe ?? 0) > 0
            ? `QE ${(frame.current_avg_qe ?? 0).toFixed(3)}`
            : // "not measured" is a fact about the QE engine (heuristic runs score
              // nothing), not a failure — say which, so the word does not read as
              // an error on an otherwise clean run.
              'QE n/a (heuristic)'
        // A terminal frame now carries its own counters; when it does not, keep
        // the last known ones instead of printing 0/0.
        const done = frame.completed_blocks ?? completedRef.current
        const total = frame.total_blocks ?? totalRef.current
        const skippedPart = skippedRef.current > 0 ? ` (+${skippedRef.current} skipped)` : ''
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
      },
      undefined,
      (connected: boolean) => {
        if (isMounted) setIsLiveStreaming(connected)
      }
    )

    return () => {
      isMounted = false
      clearInterval(timer)
      unsubscribe()
    }
  }, [currentJobId, appendLog])

  useEffect(() => {
    if (autoScroll && logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight
    }
  }, [logs, autoScroll])

  const handleCancel = async () => {
    if (!currentJobId) return
    try {
      await cancelJob(currentJobId)
      appendLog([{ text: t.mission.logAborted, level: 'WARN' }])
    } catch (err) {
      toast.push(err instanceof Error ? err.message : 'Cancel failed', 'error')
    }
  }

  const handleResume = async () => {
    if (!currentJobId) return
    setResuming(true)
    try {
      await resumeJob(currentJobId)
      appendLog([{ text: t.mission.logResumed, level: 'INFO' }])
      setStatusStr('submitted')
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
  const handleDownload = async () => {
    if (!currentJobId || downloading) return
    setDownloading(true)
    try {
      let listing = await listDeliverables(currentJobId)
      if (listing.stale) {
        toast.push(t.mission.reRenderingBeforeDownload, 'info')
        await reExportJob(currentJobId)
        listing = await listDeliverables(currentJobId)
      }
      const primary = listing.items.find((d) => d.key === 'primary') ?? listing.items[0]
      if (primary) {
        window.open(deliverableDownloadUrl(currentJobId, primary.key), '_blank')
      } else {
        toast.push(t.mission.noDeliverables, 'error')
      }
    } catch (err) {
      toast.push(err instanceof Error ? err.message : t.mission.downloadFailed, 'error')
    } finally {
      setDownloading(false)
    }
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

  const statusVariant = (status: string): 'success' | 'warning' | 'destructive' | 'info' =>
    status === 'completed'
      ? 'success'
      : status === 'failed' || status === 'cancelled'
        ? 'destructive'
        : status === 'needs_human' || status === 'blocked_human'
          ? 'warning'
          : 'info'

  const filteredLogs = logs.filter((entry) => {
    if (logLevel !== 'ALL' && entry.level !== logLevel) return false
    if (logSearch && !entry.text.toLowerCase().includes(logSearch.toLowerCase())) return false
    return true
  })

  return (
    <div className="flex-1 flex flex-col min-h-0 overflow-hidden px-8 py-6 space-y-5">
      <ConfirmDialog
        open={pendingDelete !== null}
        title={t.mission.deleteJob}
        body={t.mission.deleteConfirm}
        confirmLabel={t.common.confirm}
        cancelLabel={t.common.cancel}
        danger
        onConfirm={() => void confirmDeleteJob()}
        onCancel={() => setPendingDelete(null)}
      />
      <ConfirmDialog
        // ``key`` remounts the dialog per job, so the typed phrase cannot
        // survive into a second confirmation and let one click through.
        key={pendingReparse ?? 'none'}
        open={pendingReparse !== null}
        title={t.mission.reparseTitle}
        body={t.mission.reparseConfirm}
        confirmLabel={t.mission.reparseAction}
        cancelLabel={t.common.cancel}
        danger
        typeToConfirm={pendingReparse ?? ''}
        typeToConfirmLabel={t.mission.reparseTypePrompt}
        typeToConfirmMismatch={t.mission.reparseTypeMismatch}
        onConfirm={() => void confirmReparse()}
        onCancel={() => setPendingReparse(null)}
      />
      {/* Header bar */}
      <div className="flex items-center justify-between shrink-0 border-b border-[var(--paper-border)] pb-4">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-lg font-bold tracking-tight text-[var(--ink-primary)]">
              {t.mission.title}
            </h1>
            {currentJobId && (
              <Badge variant={isCompleted ? 'success' : isFailed ? 'destructive' : 'info'} dot>
                {jobStatusLabel(t, statusStr)}
              </Badge>
            )}
            {currentJobId && isLiveStreaming && (
              <span className="text-xs font-mono text-[#15803d] flex items-center gap-1.5 font-medium">
                <span className="h-1.5 w-1.5 rounded-full bg-[#15803d] animate-pulse" />
                {t.mission.sseStream}
              </span>
            )}
          </div>
          <div className="text-xs text-[var(--ink-secondary)] font-mono mt-0.5">
            JOB:{' '}
            <span className="text-[var(--ink-primary)] font-semibold">
              {currentJobId ?? t.mission.noActiveJobTitle}
            </span>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {currentJobId && isCompleted && (
            <Button onClick={handleDownload} variant="primary" size="sm" disabled={downloading}>
              <Download className="h-3.5 w-3.5 mr-1" />
              {t.mission.downloadPrimary}
            </Button>
          )}

          <Button
            onClick={() => currentJobId && onInspectQuality(currentJobId)}
            disabled={!currentJobId}
            variant="secondary"
            size="sm"
          >
            <ShieldCheck className="h-3.5 w-3.5 mr-1 text-[#15803d]" />
            {t.mission.inspectGate}
          </Button>

          <Button
            onClick={() => currentJobId && onOpenReview(currentJobId)}
            disabled={!currentJobId}
            variant="secondary"
            size="sm"
          >
            <SplitSquareVertical className="h-3.5 w-3.5 mr-1 text-[var(--ink-secondary)]" />
            {t.mission.openWorkbench}
          </Button>

          {currentJobId && isFailed && (
            <Button onClick={handleResume} variant="secondary" size="sm" disabled={resuming}>
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
            >
              <RefreshCw className={`h-3 w-3 mr-1 ${reparsing ? 'animate-spin' : ''}`} />
              {reparsing ? t.mission.reparsing : t.mission.reparseAction}
            </Button>
          )}

          {currentJobId && !isCompleted && !isFailed && (
            <Button onClick={handleCancel} variant="danger" size="sm">
              <Square className="h-3 w-3 mr-1" />
              {t.mission.cancelCompile}
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
      <div className="rounded-lg border border-[var(--paper-border)] bg-[var(--paper-surface)] overflow-hidden shrink-0 shadow-2xs">
        <div className="h-8 px-3.5 border-b border-[var(--paper-border)] bg-[var(--paper-subsurface)] flex items-center justify-between">
          <span className="text-xs font-mono uppercase tracking-wider text-[var(--ink-muted)] font-semibold">
            {t.mission.queueTitle}
          </span>
          <Badge variant="outline">{jobs.length}</Badge>
        </div>
        <div className="max-h-56 overflow-y-auto">
          {jobs.length === 0 ? (
            <div className="py-6 text-center text-xs text-[var(--ink-muted)]">
              {t.mission.queueEmpty}
            </div>
          ) : (
            <table className="w-full text-xs text-left">
              <thead className="bg-[var(--paper-subsurface)] text-[var(--ink-muted)] font-mono border-b border-[var(--paper-border)] sticky top-0">
                <tr>
                  {[t.mission.colJob, t.mission.colFile, t.mission.colStatus, t.mission.colProgress, t.mission.colCost, t.mission.colUpdated].map(
                    (label) => (
                      <th
                        key={label}
                        className="py-2 px-3 font-semibold uppercase text-xs tracking-wider"
                      >
                        {label}
                      </th>
                    )
                  )}
                  <th className="py-2 px-3" />
                </tr>
              </thead>
              <tbody className="divide-y divide-[var(--paper-border)]">
                {jobs.map((job) => (
                  <tr
                    key={job.job_id}
                    className={`transition-colors ${
                      job.job_id === currentJobId
                        ? 'bg-[var(--paper-subsurface)]'
                        : 'hover:bg-[var(--paper-subsurface)]'
                    }`}
                  >
                    <td className="py-2 px-3 font-mono text-[var(--ink-primary)] truncate max-w-[9rem]">
                      {job.job_id}
                    </td>
                    <td className="py-2 px-3 text-[var(--ink-secondary)] truncate max-w-[12rem]">
                      {job.file_name}
                    </td>
                    <td className="py-2 px-3">
                      <div className="flex items-center gap-1.5">
                        <Badge variant={statusVariant(job.status)} dot className="text-xs">
                          {jobStatusLabel(t, job.status)}
                        </Badge>
                        {/* A rehearsal ships echo text and a --pages run ships a
                            fragment; both used to look like a full real run in
                            this list. */}
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
                    <td className="py-2 px-3 font-mono text-[var(--ink-primary)]">
                      {job.progress_percent.toFixed(1)}%
                    </td>
                    <td className="py-2 px-3 font-mono text-[#b45309]">
                      {job.estimated_cost_usd === null || job.estimated_cost_usd === undefined
                        ? '—'
                        : `$${job.estimated_cost_usd.toFixed(3)}`}
                    </td>
                    <td className="py-2 px-3 font-mono text-xs text-[var(--ink-muted)]">
                      {formatTimestamp(job.updated_at)}
                    </td>
                    <td className="py-2 px-3 text-right">
                      <div className="flex items-center justify-end gap-1.5">
                        <Button
                          onClick={() => onSelectJob(job.job_id)}
                          variant="secondary"
                          size="sm"
                          className="text-xs h-6 px-2"
                        >
                          {t.mission.selectJob}
                        </Button>
                        <Button
                          onClick={() => handleDeleteJob(job.job_id)}
                          variant="secondary"
                          size="sm"
                          className="h-6 px-1.5 text-[#b91c1c] hover:bg-[#b91c1c]/10"
                          title={t.mission.deleteJob}
                        >
                          <Trash2 className="h-3 w-3" />
                        </Button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>

      {!currentJobId ? (
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
      {/* Slim Pipeline Stepper */}
      <div className="p-3.5 rounded-lg border border-[var(--paper-border)] bg-[var(--paper-surface)] shrink-0 shadow-2xs">
        <div className="flex items-center justify-between text-xs font-mono uppercase text-[var(--ink-muted)] mb-2 px-1 font-semibold">
          <span>{t.mission.pipelineGraph}</span>
          <span className="text-[var(--ink-primary)] font-bold">
            {t.mission.stageLine.replace(
              '{stage}',
              stageLabel(t, derivedStage || statusStr)
            )}
          </span>
        </div>
        <div className="grid grid-cols-3 sm:grid-cols-5 lg:grid-cols-9 gap-1.5">
          {pipelineStages.map((stage) => {
            const currentIndex = pipelineStages.findIndex((s) => s.key === derivedStage)
            const stageIndex = pipelineStages.findIndex((s) => s.key === stage.key)
            const isCurrent = derivedStage !== '' && derivedStage === stage.key
            // A stage the run has already passed stays marked, so the stepper
            // reads as progress rather than a single moving highlight.
            const isDone = currentIndex >= 0 && stageIndex < currentIndex
            return (
              <div
                key={stage.key}
                className={`py-1.5 text-center text-xs font-mono font-medium rounded-[4px] border transition-colors ${
                  isCurrent
                    ? 'border-[var(--ink-primary)] bg-[var(--paper-subsurface)] text-[var(--ink-primary)] font-bold shadow-2xs'
                    : isDone
                      ? 'border-[#15803d]/30 bg-[#15803d]/5 text-[#15803d]'
                      : 'border-[var(--paper-border)] bg-[var(--paper-surface)] text-[var(--ink-muted)]'
                }`}
              >
                {stage.label}
              </div>
            )
          })}
        </div>
      </div>

      {/* Metric Tiles (Hairline Dividers, Flat Container) */}
      <div className="grid grid-cols-4 rounded-lg border border-[var(--paper-border)] bg-[var(--paper-surface)] divide-x divide-[var(--paper-border)] shrink-0 shadow-2xs">
        <div className="p-4">
          <div className="text-xs font-mono uppercase tracking-wider text-[var(--ink-muted)] font-semibold">{t.mission.progress}</div>
          <div className="text-xl font-bold font-mono text-[var(--ink-primary)] mt-0.5">
            {progressPct.toFixed(1)}%
          </div>
          <div className="w-full bg-[var(--paper-subsurface)] h-1.5 rounded-full mt-2 overflow-hidden border border-[var(--paper-border)]">
            <div
              className="bg-[var(--ink-primary)] h-full transition-all duration-200"
              style={{ width: `${Math.min(100, Math.max(0, progressPct))}%` }}
            />
          </div>
        </div>

        <div className="p-4">
          <div className="text-xs font-mono uppercase tracking-wider text-[var(--ink-muted)] font-semibold">{t.mission.completedPages}</div>
          {queuePosition !== null && (
            <div className="text-xs font-mono text-[var(--ink-secondary)] mt-1">
              {t.mission.queuePosition}: <span className="font-bold">{queuePosition}</span>
            </div>
          )}
          <div className="text-xl font-bold font-mono text-[var(--ink-primary)] mt-0.5">
            {completedBlocks} <span className="text-xs text-[var(--ink-muted)] font-normal">/ {totalBlocks || '—'}</span>
          </div>
          {/* The denominator counts translatable blocks only; skipped ones are
              reported beside it so the panel does not look like it is describing
              a shorter book than the source. */}
          {skippedBlocks > 0 && (
            <div className="text-xs font-mono text-[var(--ink-muted)] mt-0.5">
              +{skippedBlocks} {t.mission.skippedBlocks}
            </div>
          )}
          {/* Failures and the human queue were in the API and never rendered: a
              job that ended with 15 blocks needing a person read as a clean
              100%. Shown next to the completed count, and red when non-zero. */}
          {(failedBlocks > 0 || needsHumanBlocks > 0) && (
            <div className="text-xs font-mono mt-0.5 text-[#b45309]">
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
          <div className="text-xs text-[#15803d] font-mono mt-1.5 font-medium">
            {qeMeasured ? (
              t.mission.avgQe.replace('{score}', avgQe.toFixed(3))
            ) : (
              <span title={t.mission.qeHeuristicHint}>
                {t.mission.qeNotMeasuredHeuristic}
              </span>
            )}
          </div>
        </div>

        <div className="p-4">
          <div className="text-xs font-mono uppercase tracking-wider text-[var(--ink-muted)] font-semibold">{t.mission.totalSpend}</div>
          <div className="text-xl font-bold font-mono text-[#b45309] mt-0.5">
            ${costUsd.toFixed(3)}
          </div>
          <div className="text-xs text-[var(--ink-secondary)] font-mono mt-1.5">
            {totalBlocks - completedBlocks} {t.mission.blocksRemaining}
          </div>
        </div>

        <div className="p-4">
          <div className="text-xs font-mono uppercase tracking-wider text-[var(--ink-muted)] font-semibold">{t.mission.executionState}</div>
          <div className="text-sm font-bold font-mono text-[var(--ink-primary)] mt-1">
            {jobStatusLabel(t, statusStr)}
          </div>
          <div className="text-xs text-[var(--ink-secondary)] mt-1.5 truncate">
            {isCompleted ? t.mission.stateCertified : t.mission.stateDrafting}
          </div>
        </div>
      </div>

      {/* Real-time Compiler Log Output */}
      <div className="flex-1 min-h-0 flex flex-col rounded-lg border border-[var(--paper-border)] bg-[var(--paper-surface)] overflow-hidden shadow-2xs">
        <div className="h-9 px-3 border-b border-[var(--paper-border)] bg-[var(--paper-subsurface)] flex items-center justify-between gap-3 shrink-0">
          <div className="flex items-center gap-2 text-xs font-mono text-[var(--ink-secondary)]">
            <Terminal className="h-3.5 w-3.5 text-[var(--ink-muted)]" />
            <span>{t.mission.liveStream}</span>
          </div>
          <div className="flex items-center gap-2">
            <input
              type="text"
              value={logSearch}
              onChange={(e) => setLogSearch(e.target.value)}
              placeholder={t.mission.logSearch}
              className="h-6 w-40 px-2 rounded-[4px] bg-[var(--paper-surface)] border border-[var(--paper-border)] text-xs font-mono text-[var(--ink-primary)] placeholder:text-[var(--ink-muted)] focus:outline-none focus:border-[var(--paper-border-hover)]"
            />
            <div className="flex bg-[var(--paper-surface)] border border-[var(--paper-border)] rounded-[4px] p-0.5">
              {(['ALL', 'INFO', 'WARN', 'ERROR'] as const).map((level) => (
                <button
                  key={level}
                  onClick={() => setLogLevel(level)}
                  className={`px-1.5 py-0.5 text-xs font-mono rounded-[3px] transition-colors ${
                    logLevel === level
                      ? 'bg-[var(--btn-bg)] text-[var(--btn-fg)]'
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
            <div className="text-xs font-mono text-[var(--ink-muted)]">
              {filteredLogs.length}/{logs.length} {t.mission.bufferedLines}
            </div>
          </div>
        </div>

        <div
          ref={logContainerRef}
          className="flex-1 p-3 overflow-y-auto font-mono text-xs leading-relaxed text-[var(--ink-primary)] space-y-1 select-text bg-[var(--paper-surface)]"
        >
          {logs.length === 0 ? (
            // A finished job has no live stream to wait for: the server drops
            // terminal jobs from memory, so ``/stream`` answers 404 and this
            // placeholder would sit on "waiting for the first frame" forever —
            // on a job that already delivered its files. Say which case it is.
            <div className="text-[var(--ink-muted)] italic">
              {isTerminal ? t.mission.logUnavailableTerminal : t.mission.waitingStream}
            </div>
          ) : filteredLogs.length === 0 ? (
            <div className="text-[var(--ink-muted)] italic">{t.mission.noMatch}</div>
          ) : (
            filteredLogs.map((log, idx) => {
              const lineNo = (idx + 1).toString().padStart(3, '0')
              return (
                <div
                  key={idx}
                  className="flex gap-3 hover:bg-[var(--paper-subsurface)] px-1 py-0.5 rounded"
                >
                  <span className="text-[var(--ink-muted)] select-none text-xs">{lineNo}</span>
                  <span
                    className={
                      log.level === 'ERROR'
                        ? 'text-[var(--ink-rose)]'
                        : log.level === 'WARN'
                          ? 'text-[var(--ink-amber)]'
                          : 'text-[var(--ink-primary)]'
                    }
                  >
                    {log.text}
                  </span>
                </div>
              )
            })
          )}
        </div>
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
  const parsed = new Date(normalized.endsWith('Z') ? normalized : `${normalized}Z`)
  if (Number.isNaN(parsed.getTime())) return value
  const pad = (n: number) => String(n).padStart(2, '0')
  return `${pad(parsed.getMonth() + 1)}-${pad(parsed.getDate())} ${pad(parsed.getHours())}:${pad(parsed.getMinutes())}`
}
