import { ChevronRight, Download, Eye, Search } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Progress } from '@/components/ui/progress'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import {
  type CompanionArtifactInfo,
  type DeliverableItem,
  type JobSummary,
} from '@/api/client'
import { useI18n } from '@/i18n/useI18n'
import { cn } from '@/lib/utils'
import { deliverableName } from '@/views/quality-gate/artifactMeta'
import {
  type LogLevel,
  displayFileName,
  formatBytes,
  formatTimestamp,
  getFormatBadge,
  jobStatusLabel,
  liveProgressPct,
  statusBadgeClass,
} from './missionHelpers'

export function LogFilters({
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

export function JobQueueTable({
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

export function DeliverableCard({
  item,
  isPrimary,
  deliveryMode,
  downloadLabel,
  previewLabel,
  previewUrl,
  onDownload,
}: {
  item: DeliverableItem
  isPrimary: boolean
  deliveryMode?: string | null
  downloadLabel: string
  previewLabel: string
  previewUrl?: string
  onDownload: () => void
}) {
  const { t } = useI18n()
  const badge = getFormatBadge(item.filename)
  const isHtml = badge === 'HTML'
  const isPdf = badge === 'PDF'

  return (
    <Card
      className={cn(
        'group relative flex flex-col justify-between overflow-hidden border transition-all duration-200 hover:shadow-md',
        isPrimary
          ? 'border-primary/40 bg-card shadow-xs ring-1 ring-primary/20'
          : 'border-border/80 bg-card/60'
      )}
    >
      {isPrimary && (
        <div className="absolute right-0 top-0 rounded-bl-md bg-emerald-500/15 px-2 py-0.5 font-mono text-[10px] font-semibold text-emerald-600 dark:text-emerald-400">
          {t.mission.exportPrimaryBadge}
        </div>
      )}
      <CardContent className="space-y-3 p-3.5">
        <div className="flex items-start gap-3">
          <span
            className={cn(
              'flex size-9 shrink-0 items-center justify-center rounded-lg font-mono text-xs font-bold shadow-2xs',
              isPdf
                ? 'border border-rose-500/25 bg-rose-500/10 text-rose-600 dark:text-rose-400'
                : badge === 'EPUB'
                  ? 'border border-purple-500/25 bg-purple-500/10 text-purple-600 dark:text-purple-400'
                  : isHtml
                    ? 'border border-sky-500/25 bg-sky-500/10 text-sky-600 dark:text-sky-400'
                    : badge === 'XLIFF'
                      ? 'border border-amber-500/25 bg-amber-500/10 text-amber-600 dark:text-amber-400'
                      : 'border border-border bg-muted text-foreground'
            )}
          >
            {badge}
          </span>
          <div className="min-w-0 flex-1 pr-6">
            <h4
              className="truncate text-sm font-semibold text-foreground"
              title={item.label || item.filename}
            >
              {deliverableName(item.key, item.label || item.filename, t, {
                deliveryMode,
                filename: item.filename,
              })}
            </h4>
            <p className="mt-0.5 truncate font-mono text-xs text-muted-foreground">
              {formatBytes(item.size_bytes)}
            </p>
          </div>
        </div>
        <p
          className="line-clamp-1 font-mono text-[11px] text-muted-foreground/80 truncate"
          title={item.filename}
        >
          {item.filename}
        </p>
      </CardContent>
      <div className="flex items-center gap-1.5 border-t border-border/60 bg-muted/20 px-3 py-2">
        <Button
          onClick={onDownload}
          variant={isPrimary ? 'default' : 'secondary'}
          size="sm"
          className="h-7 flex-1 text-xs font-medium"
        >
          <Download className="mr-1.5 size-3" />
          <span>{downloadLabel}</span>
        </Button>
        {previewUrl && (isHtml || isPdf) && (
          <Button
            asChild
            variant="ghost"
            size="sm"
            className="h-7 px-2 text-xs text-muted-foreground hover:text-foreground"
            title={previewLabel}
          >
            <a href={previewUrl} target="_blank" rel="noreferrer">
              <Eye className="size-3.5" />
            </a>
          </Button>
        )}
      </div>
    </Card>
  )
}

export function DeliverableRow({
  item,
  isPrimary,
  deliveryMode,
  downloadLabel,
  onDownload,
}: {
  item: DeliverableItem
  isPrimary: boolean
  deliveryMode?: string | null
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
              {deliverableName(item.key, item.label || item.filename, t, {
                deliveryMode,
                filename: item.filename,
              })}
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

export function CompanionArtifactRow({
  comp,
  onDownload,
}: {
  comp: CompanionArtifactInfo
  onDownload: () => void
}) {
  const { t } = useI18n()

  const kindLabel = () => {
    switch (comp.kind) {
      case 'xliff':
        return t.quality.xliffDoc
      case 'html':
        return t.quality.htmlView
      case 'epub':
        return t.quality.epub
      case 'attestation':
        return t.quality.attestationJson
      default:
        return comp.kind.toUpperCase()
    }
  }

  const kindBadge = () => {
    switch (comp.kind) {
      case 'xliff':
        return 'XLIFF'
      case 'html':
        return 'HTML'
      case 'epub':
        return 'EPUB'
      case 'attestation':
        return 'ATTEST'
      default:
        return comp.kind.toUpperCase()
    }
  }

  const isGenerated = comp.status === 'generated'
  const isSkipped = comp.status === 'skipped'
  const isFailed = comp.status === 'failed'

  return (
    <div className="flex items-center justify-between gap-3 p-3 transition-colors hover:bg-muted/25">
      <div className="flex min-w-0 items-center gap-3">
        <span className="shrink-0 rounded bg-muted px-1.5 py-0.5 font-mono text-xs font-bold text-foreground">
          {kindBadge()}
        </span>
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="truncate text-sm font-medium text-foreground">
              {kindLabel()}
            </span>
            {isGenerated && (
              <span className="rounded bg-emerald-500/15 px-1.5 py-0.5 font-mono text-xs font-semibold text-emerald-600 dark:text-emerald-400">
                {t.mission.companionGeneratedBadge}
              </span>
            )}
            {isSkipped && (
              <span className="rounded bg-muted px-1.5 py-0.5 font-mono text-xs font-semibold text-muted-foreground">
                {t.mission.companionSkippedBadge}
              </span>
            )}
            {isFailed && (
              <span className="rounded bg-destructive/15 px-1.5 py-0.5 font-mono text-xs font-semibold text-destructive">
                {t.mission.companionFailedBadge}
              </span>
            )}
          </div>
          <div className="mt-0.5 truncate font-mono text-xs text-muted-foreground">
            {isGenerated && (
              <>
                {comp.path ? comp.path.split('/').pop() : ''}
                {comp.sizeBytes != null ? ` · ${formatBytes(comp.sizeBytes)}` : ''}
                {comp.durationMs != null ? ` · ${comp.durationMs.toFixed(1)} ms` : ''}
              </>
            )}
            {isSkipped && (
              <span>{comp.reason || t.mission.companionSkippedBadge}</span>
            )}
            {isFailed && (
              <span className="text-destructive/90">{comp.error || t.mission.companionFailedBadge}</span>
            )}
          </div>
        </div>
      </div>
      {isGenerated && (
        <Button
          onClick={onDownload}
          variant="outline"
          size="sm"
          className="h-8 shrink-0 px-3 text-xs"
        >
          <Download className="mr-1.5 size-3.5" />
          <span>{t.mission.downloadPrimary}</span>
        </Button>
      )}
    </div>
  )
}
