/**
 * Typed API Client for Universal Book Translator backend.
 */

import type { paths } from './generated-types'

export type HealthResponse = paths['/health']['get']['responses']['200']['content']['application/json']
export type JobAssessRequest = paths['/jobs/assess']['post']['requestBody']['content']['application/json']
export type JobAssessResponse = paths['/jobs/assess']['post']['responses']['200']['content']['application/json']
export type JobSubmitRequest = paths['/jobs/submit']['post']['requestBody']['content']['application/json']
export type JobSubmitResponse = paths['/jobs/submit']['post']['responses']['202']['content']['application/json']
export type JobStatusResponse = paths['/jobs/{job_id}/status']['get']['responses']['200']['content']['application/json']
export type ModelProfile = paths['/api/v1/model-profiles']['get']['responses']['200']['content']['application/json'][number]

export interface DeliverableItem {
  key: string
  label: string
  filename: string
  size_bytes: number
  media_type: string
}

export interface GlossaryTerm {
  source: string
  target: string
}

export interface TmEntry {
  id: number
  src_lang: string
  tgt_lang: string
  source_text: string
  target_text: string
  provenance: string
  domain: string | null
  use_count: number
}

export interface DoctorCheck {
  group: string
  name: string
  status: string
  detail: string
  fix: string | null
}

export interface DoctorReport {
  status: string
  summary: Record<string, number>
  checks: DoctorCheck[]
}

/**
 * One frame of `/jobs/{id}/stream`.
 *
 * The backend emits the shared `ProgressSnapshot` (block counters, live QE,
 * priced cost, artifact pointers) plus a `status`; there is no `stage` or
 * `pages` field on the wire, so the UI derives stage from `status` +
 * `progress_percent` rather than reading fields the server never sends.
 */
export interface ProgressStreamFrame {
  job_id?: string
  status?: string
  progress_percent?: number
  total_blocks?: number
  completed_blocks?: number
  processed_blocks?: number
  repaired_blocks?: number
  failed_blocks?: number
  needs_human_blocks?: number
  blocked_human_blocks?: number
  current_avg_qe?: number
  bottom_15_avg_qe?: number
  estimated_cost_usd?: number | null
  output_file?: string | null
  report_file?: string | null
  visual_report_file?: string | null
  error?: string | null
}

const BASE_URL = '' // Empty means same-origin or proxied via Vite dev server

export async function checkHealth(): Promise<HealthResponse> {
  const res = await fetch(`${BASE_URL}/health`)
  if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`)
  return res.json()
}

export async function assessJob(req: JobAssessRequest): Promise<JobAssessResponse> {
  const res = await fetch(`${BASE_URL}/jobs/assess`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  })
  if (!res.ok) {
    const errorBody = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(errorBody.detail?.message || errorBody.detail || 'Assessment failed')
  }
  return res.json()
}

export async function submitJob(req: JobSubmitRequest): Promise<JobSubmitResponse> {
  const res = await fetch(`${BASE_URL}/jobs/submit`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  })
  if (!res.ok) {
    const errorBody = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(errorBody.detail || 'Job submission failed')
  }
  return res.json()
}

export async function getJobStatus(jobId: string): Promise<JobStatusResponse> {
  const res = await fetch(`${BASE_URL}/jobs/${encodeURIComponent(jobId)}/status`)
  if (!res.ok) throw new Error(`Failed to fetch job status: ${res.statusText}`)
  return res.json()
}

export async function cancelJob(jobId: string): Promise<void> {
  const res = await fetch(`${BASE_URL}/jobs/${encodeURIComponent(jobId)}/cancel`, {
    method: 'POST',
  })
  if (!res.ok) throw new Error(`Failed to cancel job: ${res.statusText}`)
}

export async function getJobReport(jobId: string): Promise<Record<string, unknown>> {
  const res = await fetch(`${BASE_URL}/jobs/${encodeURIComponent(jobId)}/report`)
  if (!res.ok) throw new Error(`Failed to fetch report: ${res.statusText}`)
  return res.json()
}

export async function getVisualReport(jobId: string): Promise<Record<string, unknown>> {
  const res = await fetch(`${BASE_URL}/jobs/${encodeURIComponent(jobId)}/visual-report`)
  if (!res.ok) throw new Error(`Failed to fetch visual report: ${res.statusText}`)
  return res.json()
}

//: Terminal SSE event names the backend emits (``event: <status>``).
const TERMINAL_EVENTS = ['completed', 'failed', 'cancelled'] as const

export function subscribeJobProgress(
  jobId: string,
  onFrame: (frame: ProgressStreamFrame) => void,
  onError?: (err: Event) => void
): () => void {
  // The backend serves the stream at /jobs/{id}/stream and frames it as named
  // SSE events ("event: progress" then one terminal "event: <status>"). An
  // EventSource `onmessage` only fires for the unnamed default event, so the
  // stream must be subscribed per event name.
  const url = `${BASE_URL}/jobs/${encodeURIComponent(jobId)}/stream`
  const es = new EventSource(url)
  let done = false

  const dispatch = (event: MessageEvent) => {
    try {
      onFrame({ job_id: jobId, ...JSON.parse(event.data) })
    } catch {
      onFrame({ job_id: jobId })
    }
  }

  const onTerminal = (event: MessageEvent) => {
    done = true
    dispatch(event)
    es.close()
  }

  es.addEventListener('progress', dispatch as EventListener)
  for (const name of TERMINAL_EVENTS) {
    es.addEventListener(name, onTerminal as EventListener)
  }

  es.onerror = (err) => {
    // A close after a terminal frame also fires onerror; do not report that as
    // a stream failure.
    if (!done && onError) onError(err)
  }

  return () => {
    done = true
    es.close()
  }
}

// --------------------------------------------------------------------------- //
// Deliverables
// --------------------------------------------------------------------------- //

export async function listDeliverables(jobId: string): Promise<DeliverableItem[]> {
  const res = await fetch(`${BASE_URL}/jobs/${encodeURIComponent(jobId)}/deliverables`)
  if (!res.ok) throw new Error(`Failed to list deliverables: ${res.statusText}`)
  const data = await res.json()
  return data.deliverables ?? []
}

export function deliverableDownloadUrl(jobId: string, key: string): string {
  return `${BASE_URL}/jobs/${encodeURIComponent(jobId)}/download/${encodeURIComponent(key)}`
}

// --------------------------------------------------------------------------- //
// Language assets
// --------------------------------------------------------------------------- //

export async function getGlossary(): Promise<GlossaryTerm[]> {
  const res = await fetch(`${BASE_URL}/assets/glossary`)
  if (res.status === 409) return []
  if (!res.ok) throw new Error(`Failed to load glossary: ${res.statusText}`)
  const data = await res.json()
  return data.terms ?? []
}

export async function upsertGlossaryTerm(source: string, target: string): Promise<GlossaryTerm[]> {
  const res = await fetch(`${BASE_URL}/assets/glossary`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ source, target }),
  })
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(body.detail || 'Failed to save term')
  }
  const data = await res.json()
  return data.terms ?? []
}

export async function deleteGlossaryTerm(source: string): Promise<GlossaryTerm[]> {
  const res = await fetch(`${BASE_URL}/assets/glossary?source=${encodeURIComponent(source)}`, {
    method: 'DELETE',
  })
  if (!res.ok) throw new Error(`Failed to delete term: ${res.statusText}`)
  const data = await res.json()
  return data.terms ?? []
}

export async function listTm(params: {
  limit?: number
  offset?: number
  src_lang?: string
  tgt_lang?: string
} = {}): Promise<{ total: number; entries: TmEntry[] }> {
  const query = new URLSearchParams()
  if (params.limit !== undefined) query.set('limit', String(params.limit))
  if (params.offset !== undefined) query.set('offset', String(params.offset))
  if (params.src_lang) query.set('src_lang', params.src_lang)
  if (params.tgt_lang) query.set('tgt_lang', params.tgt_lang)
  const res = await fetch(`${BASE_URL}/assets/tm?${query.toString()}`)
  if (!res.ok) throw new Error(`Failed to load translation memory: ${res.statusText}`)
  const data = await res.json()
  return { total: data.total ?? 0, entries: data.entries ?? [] }
}

export async function evictTm(ids: number[]): Promise<number> {
  const res = await fetch(`${BASE_URL}/assets/tm/evict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ids }),
  })
  if (!res.ok) throw new Error(`Failed to evict TM entries: ${res.statusText}`)
  const data = await res.json()
  return data.removed ?? 0
}

// --------------------------------------------------------------------------- //
// L3 review workbench
// --------------------------------------------------------------------------- //

export interface Segment {
  block_id: string
  spine_index: number
  page: number | null
  block_type: string
  status: string
  source_text: string
  target_text: string
  mtqe_score: number | null
  error_flags: string[]
  issues: string[]
  human_verified: boolean
}

export interface IssuesReport {
  job_id: string
  counts: Record<string, number>
  status: { needs_human: number; blocked_human: number; failed: number }
  total_issues: number
  term_drift: Array<Record<string, unknown>>
}

export async function listSegments(
  jobId: string,
  params: { status?: string; block_type?: string; limit?: number; offset?: number } = {}
): Promise<{ total: number; segments: Segment[] }> {
  const query = new URLSearchParams()
  if (params.status) query.set('status', params.status)
  if (params.block_type) query.set('block_type', params.block_type)
  if (params.limit !== undefined) query.set('limit', String(params.limit))
  if (params.offset !== undefined) query.set('offset', String(params.offset))
  const res = await fetch(`${BASE_URL}/jobs/${encodeURIComponent(jobId)}/segments?${query.toString()}`)
  if (!res.ok) throw new Error(`Failed to load segments: ${res.statusText}`)
  const data = await res.json()
  return { total: data.total ?? 0, segments: data.segments ?? [] }
}

export async function listIssues(jobId: string): Promise<IssuesReport> {
  const res = await fetch(`${BASE_URL}/jobs/${encodeURIComponent(jobId)}/issues`)
  if (!res.ok) throw new Error(`Failed to load issues: ${res.statusText}`)
  return res.json()
}

/**
 * URL of the single-page re-render preview (PNG). Composed server-side from the
 * current ledger text via the delivery compositor; `cacheBust` forces a refetch
 * after an edit.
 */
export function pagePreviewUrl(jobId: string, page: number, opts: { dpi?: number; cacheBust?: number } = {}): string {
  const query = new URLSearchParams()
  if (opts.dpi !== undefined) query.set('dpi', String(opts.dpi))
  if (opts.cacheBust !== undefined) query.set('t', String(opts.cacheBust))
  const suffix = query.toString()
  const base = `${BASE_URL}/jobs/${encodeURIComponent(jobId)}/pages/${page}/preview`
  return suffix ? `${base}?${suffix}` : base
}

export async function editSegment(
  jobId: string,
  blockId: string,
  targetText: string
): Promise<{ changed: boolean; tm_written: number; segment: Segment | null }> {
  const res = await fetch(
    `${BASE_URL}/jobs/${encodeURIComponent(jobId)}/segments/${encodeURIComponent(blockId)}`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ target_text: targetText }),
    }
  )
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(body.detail || 'Failed to save revision')
  }
  return res.json()
}

// --------------------------------------------------------------------------- //
// System
// --------------------------------------------------------------------------- //

export async function getDoctor(probe = false): Promise<DoctorReport> {
  const url = `${BASE_URL}/system/doctor` + (probe ? '?probe=true' : '')
  const res = await fetch(url)
  if (!res.ok) throw new Error(`Doctor check failed: ${res.statusText}`)
  return res.json()
}

export async function listModelProfiles(): Promise<ModelProfile[]> {
  const res = await fetch(`${BASE_URL}/api/v1/model-profiles`)
  if (!res.ok) throw new Error(`Failed to load model profiles: ${res.statusText}`)
  return res.json()
}
