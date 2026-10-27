import React from 'react'
import { AlertTriangle, CheckCircle2, MinusCircle, XCircle } from 'lucide-react'

import { Badge } from '@/components/ui/badge'
import { cn } from '@/lib/utils'
import { toneForStatus, type DoctorTone } from './status'

const toneClasses: Record<DoctorTone, string> = {
  ok: 'border-emerald-600/25 bg-emerald-500/10 text-emerald-700 dark:text-emerald-400',
  warn: 'border-amber-600/25 bg-amber-500/10 text-amber-700 dark:text-amber-400',
  fail: 'border-destructive/25 bg-destructive/10 text-destructive',
  skip: 'border-border bg-muted/30 text-muted-foreground',
}

const toneIconClasses: Record<DoctorTone, string> = {
  ok: 'text-emerald-700 dark:text-emerald-400',
  warn: 'text-amber-700 dark:text-amber-400',
  fail: 'text-destructive',
  skip: 'text-muted-foreground',
}

export function DoctorStatusIcon({
  status,
  className,
}: {
  status: string
  className?: string
}) {
  const tone = toneForStatus(status)
  const Icon =
    tone === 'fail'
      ? XCircle
      : tone === 'warn'
        ? AlertTriangle
        : tone === 'skip'
          ? MinusCircle
          : CheckCircle2
  return <Icon className={cn('size-4 shrink-0', toneIconClasses[tone], className)} />
}

export function DoctorStatusBadge({
  status,
  label,
  className,
}: {
  status: string
  label: string
  className?: string
}) {
  const tone = toneForStatus(status)
  return (
    <Badge
      variant="outline"
      className={cn('shrink-0 font-medium', toneClasses[tone], className)}
    >
      <span
        className={cn(
          'mr-1.5 size-1.5 rounded-full',
          tone === 'fail'
            ? 'bg-destructive'
            : tone === 'warn'
              ? 'bg-amber-500'
              : tone === 'skip'
                ? 'bg-muted-foreground'
                : 'bg-emerald-500'
        )}
      />
      {label}
    </Badge>
  )
}

export function DoctorPanel({
  title,
  description,
  action,
  children,
  className,
}: {
  title: React.ReactNode
  description?: React.ReactNode
  action?: React.ReactNode
  children: React.ReactNode
  className?: string
}) {
  return (
    <section
      className={cn(
        'min-w-0 overflow-hidden rounded-xl border border-border/80 bg-card shadow-card',
        className
      )}
    >
      <header className="flex flex-col gap-2 border-b border-border/70 px-4 py-3 sm:flex-row sm:items-start sm:justify-between">
        <div className="min-w-0 space-y-0.5">
          <h2 className="text-sm font-semibold tracking-[-0.01em] text-foreground">
            {title}
          </h2>
          {description ? (
            <p className="max-w-2xl text-xs leading-relaxed text-muted-foreground">
              {description}
            </p>
          ) : null}
        </div>
        {action ? <div className="shrink-0">{action}</div> : null}
      </header>
      {children}
    </section>
  )
}

export function DoctorMetric({
  label,
  value,
  tone,
  mono = false,
  className,
}: {
  label: string
  value: React.ReactNode
  tone?: DoctorTone
  mono?: boolean
  className?: string
}) {
  return (
    <div
      className={cn(
        'min-w-0 rounded-lg border border-border/70 bg-background/55 px-3 py-2.5',
        className
      )}
    >
      <div className="text-xs font-medium text-muted-foreground">{label}</div>
      <div
        title={typeof value === 'string' ? value : undefined}
        className={cn(
          'mt-0.5 truncate text-sm font-semibold',
          mono && 'font-mono text-[13px]',
          tone === 'fail'
            ? 'text-destructive'
            : tone === 'warn'
              ? 'text-amber-700 dark:text-amber-400'
              : tone === 'skip'
                ? 'text-muted-foreground'
                : tone === 'ok'
                  ? 'text-emerald-700 dark:text-emerald-400'
                  : 'text-foreground'
        )}
      >
        {value}
      </div>
    </div>
  )
}

export function DoctorField({
  label,
  hint,
  children,
  className,
}: {
  label: React.ReactNode
  hint?: React.ReactNode
  children: React.ReactNode
  className?: string
}) {
  return (
    <div className={cn('space-y-1.5', className)}>
      <div className="text-xs font-medium text-foreground">{label}</div>
      {children}
      {hint ? (
        <p className="text-xs leading-relaxed text-muted-foreground">{hint}</p>
      ) : null}
    </div>
  )
}
