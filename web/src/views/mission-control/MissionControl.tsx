import React, { useEffect, useState, useRef } from 'react'
import {
  Terminal,
  Square,
  ShieldCheck,
  SplitSquareVertical,
  Activity,
} from 'lucide-react'
import { Button } from '@/components/ui/Button'
import { Badge } from '@/components/ui/Badge'
import {
  getJobStatus,
  cancelJob,
  subscribeJobProgress,
  type ProgressStreamFrame,
} from '@/api/client'
import { useI18n } from '@/i18n/I18nContext'

interface MissionControlProps {
  currentJobId: string | null
  onSelectJob: (jobId: string) => void
  onInspectQuality: (jobId: string) => void
  onOpenReview: (jobId: string) => void
}

export function MissionControl({
  currentJobId,
  onInspectQuality,
  onOpenReview,
}: MissionControlProps) {
  const { t } = useI18n()
  const [logs, setLogs] = useState<string[]>([])
  const [isLiveStreaming, setIsLiveStreaming] = useState(false)
  const [statusStr, setStatusStr] = useState<string>('running')
  const [progressPct, setProgressPct] = useState<number>(0)
  const [completedBlocks, setCompletedBlocks] = useState<number>(0)
  const [totalBlocks, setTotalBlocks] = useState<number>(0)
  const [costUsd, setCostUsd] = useState<number>(0)
  const [avgQe, setAvgQe] = useState<number>(0)

  const logContainerRef = useRef<HTMLDivElement>(null)

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

    setIsLiveStreaming(true)
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

        const line =
          frame.error
            ? `ERROR ${frame.error}`
            : `${frame.status ?? 'running'} · ${frame.completed_blocks ?? 0}/${frame.total_blocks ?? 0} blocks · QE ${(frame.current_avg_qe ?? 0).toFixed(3)}`
        setLogs((prev) => [...prev.slice(-400), line])
      },
      () => {
        if (isMounted) setIsLiveStreaming(false)
      }
    )

    return () => {
      isMounted = false
      clearInterval(timer)
      unsubscribe()
    }
  }, [currentJobId])

  useEffect(() => {
    if (logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight
    }
  }, [logs])

  const handleCancel = async () => {
    if (!currentJobId) return
    try {
      await cancelJob(currentJobId)
      setLogs((prev) => [...prev, `[USER_SIGNAL] Compilation aborted by operator.`])
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Cancel failed')
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

  if (!currentJobId) {
    return (
      <div className="flex-1 flex items-center justify-center p-8 text-center text-[var(--ink-muted)]">
        <div className="max-w-sm space-y-2">
          <Activity className="h-8 w-8 mx-auto text-[var(--paper-border-hover)]" />
          <h2 className="text-sm font-semibold text-[var(--ink-primary)]">{t.mission.noActiveJobTitle}</h2>
          <p className="text-xs text-[var(--ink-secondary)] leading-relaxed">{t.mission.noActiveJobDesc}</p>
        </div>
      </div>
    )
  }

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

  return (
    <div className="flex-1 flex flex-col min-h-0 overflow-hidden px-8 py-6 space-y-5">
      {/* Header bar */}
      <div className="flex items-center justify-between shrink-0 border-b border-[var(--paper-border)] pb-4">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-lg font-bold tracking-tight text-[var(--ink-primary)]">
              {t.mission.title}
            </h1>
            <Badge
              variant={isCompleted ? 'success' : isFailed ? 'destructive' : 'info'}
              dot
            >
              {statusStr.toUpperCase()}
            </Badge>
            {isLiveStreaming && (
              <span className="text-[11px] font-mono text-[#15803d] flex items-center gap-1.5 font-medium">
                <span className="h-1.5 w-1.5 rounded-full bg-[#15803d] animate-pulse" />
                SSE STREAM
              </span>
            )}
          </div>
          <div className="text-[11px] text-[var(--ink-secondary)] font-mono mt-0.5">
            JOB: <span className="text-[var(--ink-primary)] font-semibold">{currentJobId}</span>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <Button
            onClick={() => onInspectQuality(currentJobId)}
            variant="secondary"
            size="sm"
          >
            <ShieldCheck className="h-3.5 w-3.5 mr-1 text-[#15803d]" />
            {t.mission.inspectGate}
          </Button>

          <Button
            onClick={() => onOpenReview(currentJobId)}
            variant="secondary"
            size="sm"
          >
            <SplitSquareVertical className="h-3.5 w-3.5 mr-1 text-[var(--ink-secondary)]" />
            {t.mission.openWorkbench}
          </Button>

          {!isCompleted && !isFailed && (
            <Button onClick={handleCancel} variant="danger" size="sm">
              <Square className="h-3 w-3 mr-1" />
              {t.mission.cancelCompile}
            </Button>
          )}
        </div>
      </div>

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
        <div className="h-8 px-3 border-b border-[#23252a] bg-[#1a1b1f] flex items-center justify-between shrink-0">
          <div className="flex items-center gap-2 text-xs font-mono text-[#a1a1aa]">
            <Terminal className="h-3.5 w-3.5 text-[#a1a1aa]" />
            <span>{t.mission.liveStream}</span>
          </div>
          <div className="text-[11px] font-mono text-[#71717a]">
            {logs.length} {t.mission.bufferedLines}
          </div>
        </div>

        <div
          ref={logContainerRef}
          className="flex-1 p-3 overflow-y-auto font-mono text-xs leading-relaxed text-[#d4d4d8] space-y-1 select-text bg-[#141517]"
        >
          {logs.length === 0 ? (
            <div className="text-[#71717a] italic">{t.mission.waitingStream}</div>
          ) : (
            logs.map((log, idx) => {
              const lineNo = (idx + 1).toString().padStart(3, '0')
              const isErr = log.includes('ERROR') || log.includes('fail')
              const isWarn = log.includes('WARNING')
              const isOk = log.includes('completed') || log.includes('success')
              return (
                <div key={idx} className="flex gap-3 hover:bg-[#1c1d22] px-1 py-0.5 rounded">
                  <span className="text-[#52525b] select-none text-[11px]">{lineNo}</span>
                  <span
                    className={
                      isErr
                        ? 'text-[#f87171]'
                        : isWarn
                        ? 'text-[#fbbf24]'
                        : isOk
                        ? 'text-[#4ade80]'
                        : 'text-[#e4e4e7]'
                    }
                  >
                    {log}
                  </span>
                </div>
              )
            })
          )}
        </div>
      </div>
    </div>
  )
}
