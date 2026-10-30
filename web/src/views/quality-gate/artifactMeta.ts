import type { DeliverableItem } from '@/api/client'
import type { useI18n } from '@/i18n/useI18n'

/** The `t` object's type, shared by the gate's helper modules. */
export type Translation = ReturnType<typeof useI18n>['t']

/**
 * Whether a file is a finished, reader-facing product the browser can render.
 *
 * The quality gate answers "can this be delivered?"; it is not a second file
 * manager. Technical sidecars (XLIFF, contract, reports, metrics) belong in the
 * job's Deliverables tab and never masquerade as a previewable product here.
 */
export function isPreviewableProduct(item: DeliverableItem): boolean {
  if (!['primary', 'secondary', 'html'].includes(item.key)) return false
  const ext = item.filename.split('.').pop()?.toLowerCase() ?? ''
  return ext === 'pdf' || ext === 'html' || ext === 'htm'
}

/** The friendly display name; falls back to the backend label, then the key. */
export function deliverableName(key: string, fallback: string, t: Translation): string {
  switch (key) {
    case 'primary':
      return t.quality.targetPdf
    case 'secondary':
      return t.quality.secondaryDoc
    case 'epub':
      return t.quality.epub
    case 'contract':
      return t.quality.contractJson
    case 'delivery_contract':
      return t.quality.deliveryContractJson
    case 'quality_report':
      return t.quality.qualityReportJson
    case 'visual_report':
      return t.quality.visualReportJson
    case 'metrics':
      return t.quality.metricsJson
    case 'html':
      return t.quality.htmlView
    case 'xliff':
      return t.quality.xliffDoc
    case 'attestation':
      return t.quality.attestationJson
    default:
      return fallback
  }
}

/** "1.2 MB" / "843 KB" — the size tells the reader what they are about to get. */
export function fmtSize(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes < 0) return '—'
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}
