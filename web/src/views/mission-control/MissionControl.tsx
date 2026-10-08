import React, { useCallback, useEffect, useState, useRef } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import {
  Terminal,
  Square,
  ShieldCheck,
  SplitSquareVertical,
  Activity,
  RotateCcw,
  Download,
  Trash2,
} from 'lucide-react'
import { Button } from '@/components/ui/Button'
import { Badge } from '@/components/ui/Badge'
import {
  getJobStatus,
  cancelJob,
  resumeJob,
  deleteJob,
  listJobs,
  listDeliverables,
  deliverableDownloadUrl,
  subscribeJobProgress,
  type JobSummary,
  type ProgressStreamFrame,
} from '@/api/client'
import { useI18n } from '@/i18n/useI18n'

type LogLevel = 'INFO' | 'WARN' | 'ERROR'

interface LogEntry {
  text: string
  level: LogLevel
}

export function MissionControl() {
  const { t } = useI18n()
  const navigate = useNavigate()
  const { jobId } = useParams<{ jobId: string }>()
  const currentJobId = jobId ?? null
  const onSelectJob = (id: string) => navigate(`/jobs/${id}`)
  const onInspectQuality = (id: string) => navigate(`/jobs/${id}/quality`)
  const onOpenReview = (id: string) => navigate(`/jobs/${id}/review`)
  const [logs, setLogs] = useState<LogEntry[]>([])
  const [logSearch, setLogSearch] = useState('')
  const [logLevel, setLogLevel] = useState<'ALL' | LogLevel>('ALL')
  const [autoScroll, setAutoScroll] = useState(true)
  const [isLiveStreaming, setIsLiveStreaming] = useState(false)
  const [statusStr, setStatusStr] = useState<string>('running')
  const [progressPct, setProgressPct] = useState<number>(0)
  const [completedBlocks, setCompletedBlocks] = useState<number>(0)
  const [totalBlocks, setTotalBlocks] = useState<number>(0)
  const [costUsd, setCostUsd] = useState<number>(0)
  const [avgQe, setAvgQe] = useState<number>(0)
  const [jobs, setJobs] = useState<JobSummary[]>([])
  const [resuming, setResuming] = useState(false)
  const [downloading, setDownloading] = useState(false)

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

    let isMounted = true
    const applyStatus = (res: Record<string, unknown>) => {
      if (res.status) setStatusStr(String(res.status))
      if (typeof res.progress_percent === 'number') setProgressPct(res.progress_percent)
      if (typeof res.completed_blocks === 'number') setCompletedBlocks(res.completed_blocks)
      if (typeof res.total_blocks === 'number') setTotalBlocks(res.total_blocks)
      if (typeof res.estimated_cost_usd === 'number') setCostUsd(res.estimated_cost_usd)
      if (typeof res.current_avg_qe === 'number') setAvgQe(res.current_avg_qe)
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
        if (typeof frame.progress_percent === 'number') setProgressPct(frame.progress_percent)
        if (typeof frame.completed_blocks === 'number') setCompletedBlocks(frame.completed_blocks)
        if (typeof frame.total_blocks === 'number') setTotalBlocks(frame.total_blocks)
        if (typeof frame.estimated_cost_usd === 'number') setCostUsd(frame.estimated_cost_usd)
        if (typeof frame.current_avg_qe === 'number') setAvgQe(frame.current_avg_qe)

        const level: LogLevel = frame.error ? 'ERROR' : frame.status === 'failed' ? 'ERROR' : 'INFO'
        const text = frame.error
          ? `ERROR ${frame.error}`
          : `${frame.status ?? 'running'} · ${frame.completed_blocks ?? 0}/${frame.total_blocks ?? 0} blocks · QE ${(frame.current_avg_qe ?? 0).toFixed(3)}`
        setLogs((prev) => [...prev.slice(-400), { text, level }])
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
  }, [currentJobId])

  useEffect(() => {
    if (autoScroll && logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight
    }
  }, [logs, autoScroll])

  const handleCancel = async () => {
    if (!currentJobId) return
    try {
      await cancelJob(currentJobId)
      setLogs((prev) => [
        ...prev,
        { text: '[USER_SIGNAL] Compilation aborted by operator.', level: 'WARN' },
      ])
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Cancel failed')
    }
  }

  const handleResume = async () => {
    if (!currentJobId) return
    setResuming(true)
    try {
      await resumeJob(currentJobId)
      setLogs((prev) => [
        ...prev,
        { text: '[USER_SIGNAL] Resuming from the last ledger checkpoint.', level: 'INFO' },
      ])
      setStatusStr('submitted')
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Resume failed')
    } finally {
      setResuming(false)
    }
  }

  // Completed run → open the primary translated document in a new tab. The
  // deliverable list names what actually exists on disk, so no key guessing.
  const handleDownload = async () => {
    if (!currentJobId || downloading) return
    setDownloading(true)
    try {
      const items = await listDeliverables(currentJobId)
      const primary = items.find((d) => d.key === 'primary') ?? items[0]
      if (primary) {
        window.open(deliverableDownloadUrl(currentJobId, primary.key), '_blank')
      } else {
        alert(t.mission.noDeliverables)
      }
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Download failed')
    } finally {
      setDownloading(false)
    }
  }

  // History management: removes the ledger + deliverables of a finished job.
  // The backend refuses anything still running; the confirm spells out that
  // this is permanent, unlike cancel.
  const handleDeleteJob = async (jobId: string) => {
    if (!window.confirm(t.mission.deleteConfirm)) return
    try {
      await deleteJob(jobId)
      if (jobId === currentJobId) {
        setLogs((prev) => [
          ...prev,
          { text: '[USER_SIGNAL] Job history deleted from the console.', level: 'WARN' },
        ])
      }
      await refreshQueue()
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Delete failed')
    }
  }

  const pipelineStages = [
    { key: 'extract', label: '1. Extract' },
    { key: 'segment', label: '2. Chunk' },
    { key: 'tm', label: '3. TM' },
    { key: 'translate', label: '4. Drafting' },
    { key: 'qe', label: '5. Fast QE' },
    { key: 'render', label: '6. Typst' },
    { key: 'verify', label: '7. Gate' },
    { key: 'package', label: '8. Package' },
  ]

  const isCompleted = statusStr === 'completed'
  const isFailed = statusStr === 'failed' || statusStr === 'cancelled'
  // The backend stream carries no `stage` field, so the stepper highlight is
  // derived from the real status + progress_percent rather than a fabricated
  // stage the server never sends.
  const derivedStage = (() => {
    if (statusStr === 'completed') return 'package'
    if (statusStr === 'queued' || statusStr === 'submitted') return 'extract'
    if (statusStr === 'failed' || statusStr === 'cancelled') return ''
    if (progressPct >= 100) return 'package'
    if (progressPct >= 90) return 'verify'
    if (progressPct >= 75) return 'render'
    if (progressPct >= 40) return 'translate'
    if (progressPct >= 15) return 'segment'
    return 'extract'
  })()

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
      {/* Header bar */}
      <div className="flex items-center justify-between shrink-0 border-b border-[var(--paper-border)] pb-4">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-lg font-bold tracking-tight text-[var(--ink-primary)]">
              {t.mission.title}
            </h1>
            {currentJobId && (
              <Badge variant={isCompleted ? 'success' : isFailed ? 'destructive' : 'info'} dot>
                {statusStr.toUpperCase()}
              </Badge>
            )}
            {currentJobId && isLiveStreaming && (
              <span className="text-[11px] font-mono text-[#15803d] flex items-center gap-1.5 font-medium">
                <span className="h-1.5 w-1.5 rounded-full bg-[#15803d] animate-pulse" />
                SSE STREAM
              </span>
            )}
          </div>
          <div className="text-[11px] text-[var(--ink-secondary)] font-mono mt-0.5">
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

          {currentJobId && !isCompleted && !isFailed && (
            <Button onClick={handleCancel} variant="danger" size="sm">
              <Square className="h-3 w-3 mr-1" />
              {t.mission.cancelCompile}
            </Button>
          )}
        </div>
      </div>

      {/* Global Job Queue (PRD §4.2.1) — read from the durable ledgers so a
          restarted console still lists finished runs. */}
      <div className="rounded-lg border border-[var(--paper-border)] bg-[var(--paper-surface)] overflow-hidden shrink-0 shadow-2xs">
        <div className="h-8 px-3.5 border-b border-[var(--paper-border)] bg-[var(--paper-subsurface)] flex items-center justify-between">
          <span className="text-[10px] font-mono uppercase tracking-wider text-[var(--ink-muted)] font-semibold">
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
                        className="py-2 px-3 font-semibold uppercase text-[10px] tracking-wider"
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
                        : 'hover:bg-[var(--paper-subsurface)]/60'
                    }`}
                  >
                    <td className="py-2 px-3 font-mono text-[var(--ink-primary)] truncate max-w-[9rem]">
                      {job.job_id}
                    </td>
                    <td className="py-2 px-3 text-[var(--ink-secondary)] truncate max-w-[12rem]">
                      {job.file_name}
                    </td>
                    <td className="py-2 px-3">
                      <Badge variant={statusVariant(job.status)} dot className="text-[10px]">
                        {job.status.toUpperCase()}
                      </Badge>
                    </td>
                    <td className="py-2 px-3 font-mono text-[var(--ink-primary)]">
                      {job.progress_percent.toFixed(1)}%
                    </td>
                    <td className="py-2 px-3 font-mono text-[#b45309]">
                      {job.estimated_cost_usd === null || job.estimated_cost_usd === undefined
                        ? '—'
                        : `$${job.estimated_cost_usd.toFixed(3)}`}
                    </td>
                    <td className="py-2 px-3 font-mono text-[11px] text-[var(--ink-muted)]">
                      {formatTimestamp(job.updated_at)}
                    </td>
                    <td className="py-2 px-3 text-right">
                      <div className="flex items-center justify-end gap-1.5">
                        <Button
                          onClick={() => onSelectJob(job.job_id)}
                          variant="secondary"
                          size="sm"
                          className="text-[11px] h-6 px-2"
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
        <div className="flex items-center justify-between text-[10px] font-mono uppercase text-[var(--ink-muted)] mb-2 px-1 font-semibold">
          <span>{t.mission.pipelineGraph}</span>
          <span className="text-[var(--ink-primary)] font-bold">
            STAGE: {(derivedStage || statusStr).toUpperCase()}
          </span>
        </div>
        <div className="grid grid-cols-8 gap-1.5">
          {pipelineStages.map((stage) => {
            const isCurrent = derivedStage !== '' && derivedStage === stage.key
            return (
              <div
                key={stage.key}
                className={`py-1.5 text-center text-xs font-mono font-medium rounded-[4px] border transition-colors ${
                  isCurrent
                    ? 'border-[var(--ink-primary)] bg-[var(--paper-subsurface)] text-[var(--ink-primary)] font-bold shadow-2xs'
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
          <div className="text-[10px] font-mono uppercase tracking-wider text-[var(--ink-muted)] font-semibold">{t.mission.progress}</div>
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
          <div className="text-[10px] font-mono uppercase tracking-wider text-[var(--ink-muted)] font-semibold">{t.mission.completedPages}</div>
          <div className="text-xl font-bold font-mono text-[var(--ink-primary)] mt-0.5">
            {completedBlocks} <span className="text-xs text-[var(--ink-muted)] font-normal">/ {totalBlocks || '—'}</span>
          </div>
          <div className="text-[11px] text-[#15803d] font-mono mt-1.5 font-medium">
            Avg QE {avgQe.toFixed(3)}
          </div>
        </div>

        <div className="p-4">
          <div className="text-[10px] font-mono uppercase tracking-wider text-[var(--ink-muted)] font-semibold">{t.mission.totalSpend}</div>
          <div className="text-xl font-bold font-mono text-[#b45309] mt-0.5">
            ${costUsd.toFixed(3)}
          </div>
          <div className="text-[11px] text-[var(--ink-secondary)] font-mono mt-1.5">
            {totalBlocks - completedBlocks} blocks remaining
          </div>
        </div>

        <div className="p-4">
          <div className="text-[10px] font-mono uppercase tracking-wider text-[var(--ink-muted)] font-semibold">{t.mission.executionState}</div>
          <div className="text-sm font-bold font-mono text-[var(--ink-primary)] mt-1 capitalize">
            {statusStr}
          </div>
          <div className="text-[11px] text-[var(--ink-secondary)] mt-1.5 truncate">
            {isCompleted ? 'Compiler output certified' : 'Drafting and typesetting blocks'}
          </div>
        </div>
      </div>

      {/* Real-time Compiler Log Output */}
      <div className="flex-1 min-h-0 flex flex-col rounded-lg border border-[var(--paper-border)] bg-[#141517] overflow-hidden shadow-2xs">
        <div className="h-9 px-3 border-b border-[#23252a] bg-[#1a1b1f] flex items-center justify-between gap-3 shrink-0">
          <div className="flex items-center gap-2 text-xs font-mono text-[#a1a1aa]">
            <Terminal className="h-3.5 w-3.5 text-[#a1a1aa]" />
            <span>{t.mission.liveStream}</span>
          </div>
          <div className="flex items-center gap-2">
            <input
              type="text"
              value={logSearch}
              onChange={(e) => setLogSearch(e.target.value)}
              placeholder={t.mission.logSearch}
              className="h-6 w-40 px-2 rounded-[4px] bg-[#141517] border border-[#2a2c31] text-[11px] font-mono text-[#d4d4d8] placeholder:text-[#52525b] focus:outline-none focus:border-[#3f4147]"
            />
            <div className="flex bg-[#141517] border border-[#2a2c31] rounded-[4px] p-0.5">
              {(['ALL', 'INFO', 'WARN', 'ERROR'] as const).map((level) => (
                <button
                  key={level}
                  onClick={() => setLogLevel(level)}
                  className={`px-1.5 py-0.5 text-[10px] font-mono rounded-[3px] transition-colors ${
                    logLevel === level
                      ? 'bg-[#2a2c31] text-[#e4e4e7]'
                      : 'text-[#71717a] hover:text-[#a1a1aa]'
                  }`}
                >
                  {level}
                </button>
              ))}
            </div>
            <label className="flex items-center gap-1 text-[10px] font-mono text-[#71717a] cursor-pointer select-none">
              <input
                type="checkbox"
                checked={autoScroll}
                onChange={(e) => setAutoScroll(e.target.checked)}
              />
              {t.mission.autoScroll}
            </label>
            <div className="text-[11px] font-mono text-[#71717a]">
              {filteredLogs.length}/{logs.length} {t.mission.bufferedLines}
            </div>
          </div>
        </div>

        <div
          ref={logContainerRef}
          className="flex-1 p-3 overflow-y-auto font-mono text-xs leading-relaxed text-[#d4d4d8] space-y-1 select-text bg-[#141517]"
        >
          {logs.length === 0 ? (
            <div className="text-[#71717a] italic">{t.mission.waitingStream}</div>
          ) : filteredLogs.length === 0 ? (
            <div className="text-[#71717a] italic">{t.mission.noMatch}</div>
          ) : (
            filteredLogs.map((log, idx) => {
              const lineNo = (idx + 1).toString().padStart(3, '0')
              return (
                <div key={idx} className="flex gap-3 hover:bg-[#1c1d22] px-1 py-0.5 rounded">
                  <span className="text-[#52525b] select-none text-[11px]">{lineNo}</span>
                  <span
                    className={
                      log.level === 'ERROR'
                        ? 'text-[#f87171]'
                        : log.level === 'WARN'
                          ? 'text-[#fbbf24]'
                          : 'text-[#e4e4e7]'
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
