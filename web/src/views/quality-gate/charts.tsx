import React from 'react'

/**
 * Inline-SVG ratio ring used by the System Doctor.
 *
 * The quality gate no longer renders radar/bar/page-grid charts: its five
 * checks use compact progress rails so the page stays a decision surface, not
 * a dashboard. Keeping only the primitive still in use removes dead chart code.
 */
export type ChartTone = 'pass' | 'warn' | 'fail' | 'muted'

const TONE_STROKE: Record<ChartTone, string> = {
  pass: 'var(--ink-highlight)',
  warn: 'var(--ink-amber)',
  fail: 'var(--ink-rose)',
  muted: 'hsl(var(--muted-foreground))',
}

/** A ratio donut with the number in the middle. */
export function Donut({
  value,
  tone,
  center,
  caption,
}: {
  /** 0-1. Clamped; NaN renders as empty. */
  value: number
  tone: ChartTone
  /** Text inside the ring, e.g. "100%". */
  center: string
  /** One-line label under the ring. */
  caption?: string
}) {
  const size = 76
  const stroke = 7
  const r = (size - stroke) / 2
  const c = 2 * Math.PI * r
  const clamped = Number.isFinite(value) ? Math.min(1, Math.max(0, value)) : 0
  return (
    <div className="flex flex-col items-center gap-1 shrink-0">
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} role="img" aria-label={caption ?? center}>
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="var(--border)" strokeWidth={stroke} />
        {clamped > 0 && (
          <circle
            cx={size / 2}
            cy={size / 2}
            r={r}
            fill="none"
            stroke={TONE_STROKE[tone]}
            strokeWidth={stroke}
            strokeLinecap="round"
            strokeDasharray={`${(c * clamped).toFixed(1)} ${c.toFixed(1)}`}
            transform={`rotate(-90 ${size / 2} ${size / 2})`}
          />
        )}
        <text
          x={size / 2}
          y={size / 2}
          textAnchor="middle"
          dominantBaseline="central"
          fontSize={13}
          fontFamily="monospace"
          fontWeight={600}
          fill="currentColor"
        >
          {center}
        </text>
      </svg>
      {caption && <div className="text-xs text-muted-foreground text-center leading-tight">{caption}</div>}
    </div>
  )
}
