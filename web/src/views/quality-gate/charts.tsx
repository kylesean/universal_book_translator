import React from 'react'

/**
 * Inline-SVG charts for the quality gate. No chart dependency: every figure
 * here is a few SVG primitives, the same discipline as the radar below — the
 * values they render are a handful of 0-1 ratios, and a chart library would
 * outweigh the whole view.
 */

export type ChartTone = 'pass' | 'warn' | 'fail' | 'muted'

const TONE_STROKE: Record<ChartTone, string> = {
  pass: '#15803d',
  warn: '#b45309',
  fail: '#b91c1c',
  muted: 'var(--ink-muted)',
}

/** A ratio donut with the number in the middle — for retention/coverage. */
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
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="var(--paper-border)" strokeWidth={stroke} />
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
          fill="var(--ink-primary)"
        >
          {center}
        </text>
      </svg>
      {caption && <div className="text-[11px] text-[var(--ink-muted)] text-center leading-tight">{caption}</div>}
    </div>
  )
}

/** A labelled horizontal ratio bar — for recall, cache hit, anything linear. */
export function BarRow({
  label,
  value,
  tone,
  display,
}: {
  label: string
  /** 0-1 fill ratio. */
  value: number
  tone: ChartTone
  /** Right-aligned value text, e.g. "100.0%". */
  display: string
}) {
  const clamped = Number.isFinite(value) ? Math.min(1, Math.max(0, value)) : 0
  return (
    <div className="w-full min-w-0">
      <div className="flex items-baseline justify-between gap-2 text-[11px]">
        <span className="text-[var(--ink-muted)] truncate">{label}</span>
        <span className="font-mono font-medium text-[var(--ink-primary)]">{display}</span>
      </div>
      <div className="mt-1 h-1.5 rounded-full bg-[var(--paper-border)] overflow-hidden">
        <div
          className="h-full rounded-full transition-all"
          style={{ width: `${(clamped * 100).toFixed(1)}%`, backgroundColor: TONE_STROKE[tone] }}
        />
      </div>
    </div>
  )
}

/**
 * Per-page sampling grid: one chip per scanned page, green when the scan
 * found nothing there and red when it flagged the page. This is the visual
 * gate made legible — "we really looked at every page" becomes something the
 * reader can see rather than a claim they must take on faith.
 */
export function PageGrid({
  pages,
  findingsByPage,
  maxChips = 60,
}: {
  /** Every sampled page number, ascending. */
  pages: number[]
  /** Page number → finding count on that page. */
  findingsByPage: Map<number, number>
  /** Books run long: past this many pages, show the first chunk + a count. */
  maxChips?: number
}) {
  if (pages.length === 0) return null
  const shown = pages.slice(0, maxChips)
  const overflow = pages.length - shown.length
  return (
    <div>
      <div className="flex flex-wrap gap-1">
        {shown.map((page) => {
          const flagged = (findingsByPage.get(page) ?? 0) > 0
          return (
            <span
              key={page}
              title={
                flagged
                  ? `P${page}: ${findingsByPage.get(page)}`
                  : `P${page}`
              }
              className={
                'inline-flex items-center justify-center h-5 min-w-[22px] px-1 rounded-[3px] border text-[10px] font-mono ' +
                (flagged
                  ? 'bg-[#b91c1c]/10 border-[#b91c1c]/30 text-[#b91c1c]'
                  : 'bg-[#15803d]/8 border-[#15803d]/25 text-[#15803d]')
              }
            >
              {page}
            </span>
          )
        })}
        {overflow > 0 && (
          <span className="inline-flex items-center text-[10px] font-mono text-[var(--ink-muted)] px-1">
            +{overflow}
          </span>
        )}
      </div>
    </div>
  )
}

export interface RadarAxis {
  label: string
  /** 0-100. */
  value: number
  /**
   * False for a check that never ran. An unmeasured axis used to be drawn as
   * a 0 — on the one screen whose job is attestation, a collapsed notch read
   * as "scored zero" for a measurement that was never taken. The polygon now
   * chords across it (dashed) and the axis label drops its number.
   */
  measured?: boolean
}

/** A five-axis radar drawn as inline SVG (no chart dependency). */
export function RadarChart({ axes }: { axes: RadarAxis[] }) {
  const cx = 110
  const cy = 100
  const radius = 60
  const point = (index: number, ratio: number): [number, number] => {
    const angle = ((-90 + index * (360 / axes.length)) * Math.PI) / 180
    return [cx + radius * ratio * Math.cos(angle), cy + radius * ratio * Math.sin(angle)]
  }
  const measuredPoints = axes
    .map((axis, index) =>
      axis.measured === false
        ? null
        : point(index, Math.min(1, Math.max(0, axis.value / 100)))
    )
  const polygon = measuredPoints
    .filter((p): p is [number, number] => p !== null)
    .map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`)
    .join(' ')

  // Dashed chords across each unmeasured axis: the shape visibly "skips"
  // that axis instead of spiking to the centre. Collected as endpoint pairs.
  const chords: Array<[number, number]> = []
  for (let i = 0; i < axes.length; i++) {
    if (axes[i].measured !== false) continue
    const prev = axes[(i - 1 + axes.length) % axes.length].measured === false ? null : measuredPoints[(i - 1 + axes.length) % axes.length]
    const next = axes[(i + 1) % axes.length].measured === false ? null : measuredPoints[(i + 1) % axes.length]
    if (prev && next) chords.push(prev, next)
  }

  return (
    <svg viewBox="0 0 220 200" className="w-48 h-44 sm:w-56 sm:h-52 shrink-0 max-w-full" role="img">
      {[0.25, 0.5, 0.75, 1].map((ratio) => (
        <polygon
          key={ratio}
          points={axes
            .map((_, index) => point(index, ratio))
            .map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`)
            .join(' ')}
          fill="none"
          stroke="var(--paper-border)"
          strokeWidth={0.8}
        />
      ))}
      {axes.map((_, index) => {
        const [x, y] = point(index, 1)
        return (
          <line
            key={index}
            x1={cx}
            y1={cy}
            x2={x}
            y2={y}
            stroke="var(--paper-border)"
            strokeWidth={0.8}
          />
        )
      })}
      <polygon
        points={polygon}
        fill="rgba(21,128,61,0.18)"
        stroke="#15803d"
        strokeWidth={1.4}
      />
      {chords.length > 0 &&
        // Pair up the collected endpoints: one dashed line per unmeasured axis.
        Array.from({ length: chords.length / 2 }, (_, i) => {
          const [x1, y1] = chords[i * 2]
          const [x2, y2] = chords[i * 2 + 1]
          return (
            <line
              key={`dash-${i}`}
              x1={x1}
              y1={y1}
              x2={x2}
              y2={y2}
              stroke="var(--ink-muted)"
              strokeWidth={1}
              strokeDasharray="3 3"
            />
          )
        })}
      {axes.map((axis, index) => {
        const [x, y] = point(index, 1.25)
        return (
          <text
            key={axis.label}
            x={x}
            y={y}
            textAnchor="middle"
            dominantBaseline="middle"
            fontSize={9}
            fontFamily="monospace"
            fill="var(--ink-muted)"
          >
            {axis.measured === false ? axis.label : `${axis.label} ${Math.round(axis.value)}`}
          </text>
        )
      })}
    </svg>
  )
}
