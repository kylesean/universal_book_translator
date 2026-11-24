import { type Segment } from '@/api/client'
import { useI18n } from '@/i18n/useI18n'

export type Scope = 'issues' | 'all'

export function scopeLabel(t: ReturnType<typeof useI18n>['t'], scope: Scope): string {
  return scope === 'issues' ? t.review.scopeToReview : t.review.scopeEverything
}

export const issueLabel = (t: ReturnType<typeof useI18n>['t'], kind: string): string =>
  t.review.issueKinds[kind] ?? kind

export const blockTypeLabel = (t: ReturnType<typeof useI18n>['t'], kind: string): string =>
  t.review.blockTypes[kind] ?? kind

export const blockStatusLabel = (t: ReturnType<typeof useI18n>['t'], status: string): string =>
  t.review.blockStatuses[status] ?? status.toUpperCase()

export const mqmSeverityLabel = (t: ReturnType<typeof useI18n>['t'], severity: string | null): string =>
  severity ? (t.review.mqmSeverities[severity] ?? severity) : '—'

export function segmentLocation(t: ReturnType<typeof useI18n>['t'], segment: Segment): string {
  const parts: string[] = []
  if (segment.page !== null) {
    parts.push(t.review.pageLabel.replace('{n}', String(segment.page)))
  }
  parts.push(blockTypeLabel(t, segment.block_type))
  parts.push(t.review.blockPosition.replace('{n}', String(segment.spine_index + 1)))
  return parts.join(' · ')
}

export function findingLabel(t: ReturnType<typeof useI18n>['t'], kind: string): string {
  const labels: Record<string, string> = {
    review: t.review.flagHumanReview,
    critical: t.review.flagCritical,
    omission: t.review.flagOmission,
    numeric: t.review.flagNumeric,
    terminology: t.review.flagTerminology,
    formula: t.review.flagFormula,
    structure: t.review.flagStructure,
    repetition: t.review.flagRepetition,
    fabrication: t.review.flagFabrication,
    echo: t.review.flagEcho,
    render: t.review.flagRender,
  }
  return labels[kind] ?? t.review.flagGeneric
}

export function findingKindForFlag(flag: string): string {
  const value = flag.toLowerCase()
  if (value.includes('needs_human_review')) return 'review'
  if (value.includes('mqm_critical_blocked')) return 'critical'
  if (value.includes('omission') || value.includes('empty target') || value.includes('table dropped')) {
    return 'omission'
  }
  if (value.includes('numeric')) return 'numeric'
  if (value.includes('glossary') || value.includes('consistency:')) return 'terminology'
  if (value.includes('formula') || value.includes('math') || value.includes('latex')) return 'formula'
  if (value.includes('repetition') || value.includes('loop hallucination')) return 'repetition'
  if (value.includes('fabrication') || value.includes('added reference') || value.includes('prompt')) {
    return 'fabrication'
  }
  if (value.includes('render_skip')) return 'render'
  if (value.includes('echo')) return 'echo'
  return 'structure'
}
