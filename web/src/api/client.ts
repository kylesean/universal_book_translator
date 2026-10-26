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
export type JobUploadResponse = paths['/jobs/upload']['post']['responses']['201']['content']['application/json']
export type JobSummary = paths['/jobs']['get']['responses']['200']['content']['application/json']['jobs'][number]
export type ModelProfile = paths['/api/v1/model-profiles']['get']['responses']['200']['content']['application/json'][number]
export type RuntimeConfig = paths['/system/runtime-config']['get']['responses']['200']['content']['application/json']
/** The quality-estimation gate actually in force, and which fields an env var overrides. */
export type QESettings = paths['/system/qe-config']['get']['responses']['200']['content']['application/json']
/** An edit of the `[defaults]` QE keys; omitted fields are left untouched. */
export type QESettingsRequest = paths['/system/qe-config']['put']['requestBody']['content']['application/json']
export type ProviderDetail = paths['/system/providers/{name}']['get']['responses']['200']['content']['application/json']
export type ProviderEditRequest = paths['/system/providers/{name}']['put']['requestBody']['content']['application/json']
export type SystemInfo = paths['/system/info']['get']['responses']['200']['content']['application/json']

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
  created_at: string
  updated_at: string
}

export interface TmPage {
  total: number
  offset: number
  limit: number
  machine_count: number
  human_count: number
  never_used_count: number
  entries: TmEntry[]
}

export interface DoctorCheck {
  /** English group label, as the CLI prints it. */
  group: string
  /** Stable group identifier a localized console switches on. */
  group_code: string
  /** English check name, as the CLI prints it. */
  name: string
  /** Stable check identifier (`api_key`, `disk_space`, …). */
  code: string
  status: string
  /** English diagnosis, printed verbatim by the CLI. */
  detail: string
  /** Stable variant identifier for `detail` (`disk_space.low`, …). */
  detail_code: string
  /** English remediation, or null when the check needs no action. */
  fix: string | null
  /** Stable variant identifier for `fix`. */
  fix_code: string
  /** Values interpolated into the English copy, for a localized renderer. */
  params: Record<string, unknown>
}

export interface DoctorReport {
  status: string
  summary: Record<string, number>
  checks: DoctorCheck[]
}

/** The console's pipeline stepper position, as reported by the engine. */
export type PipelineStage =
  | 'extract'
  | 'segment'
  | 'tm'
  | 'translate'
  | 'qe'
  | 'repair'
  | 'render'
  | 'verify'
  | 'package'

/**
 * One frame of `/jobs/{id}/stream`.
 *
 * The backend emits the shared `ProgressSnapshot` (block counters, live QE,
 * priced cost, artifact pointers, the engine's own `stage`, and its last
 * human-readable `message`) plus a `status`.
 */
export interface ProgressStreamFrame {
  job_id?: string
  status?: string
  stage?: PipelineStage | null
  message?: string | null
  /** Machine-readable engine event; the console localizes this, never `message`. */
  event_type?: string | null
  progress_percent?: number
  total_blocks?: number
  /** Blocks drafted but not yet finalized. Shown as live translation work,
   * distinct from ``completed_blocks`` so a draft is never read as delivered. */
  drafted_blocks?: number
  /** Drafted blocks now queued for another repair pass. Live work, not done. */
  repair_pending_blocks?: number
  /** Blocks the skip rule finished without translating them (images, list
   * markers, short headings). Excluded from total/completed. */
  skipped_blocks?: number
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

/**
 * The API key the operator entered, kept in sessionStorage.
 *
 * ``sessionStorage``, not ``localStorage``: the key dies with the tab rather
 * than persisting on a shared machine. The server also sets an HttpOnly
 * session cookie at sign-in, which is what actually authenticates the SSE
 * stream, page previews and downloads — ``EventSource`` and ``<img>`` cannot
 * send a header, so the header path alone would leave those 401ing.
 */
const _KEY_STORAGE = 'ubt_api_key'

export function getStoredApiKey(): string {
  try {
    return sessionStorage.getItem(_KEY_STORAGE) ?? ''
  } catch {
    return ''
  }
}

export function setStoredApiKey(key: string): void {
  try {
    if (key) sessionStorage.setItem(_KEY_STORAGE, key)
    else sessionStorage.removeItem(_KEY_STORAGE)
  } catch {
    // Private-mode browsers can refuse storage; the cookie still authenticates.
  }
}

/** ``fetch`` with the stored API key attached (when one is set). */
function apiFetch(input: string, init: RequestInit = {}): Promise<Response> {
  const key = getStoredApiKey()
  if (!key) return fetch(input, init)
  const headers = new Headers(init.headers)
  headers.set('X-API-Key', key)
  return fetch(input, { ...init, headers })
}

/**
 * A failed API call, carrying the HTTP status alongside the server's message.
 *
 * Callers need the status, not just the text: a 409 from the review surface
 * means "another writer holds this job" or "the job is running", which the
 * operator can act on (retry, stop the job) — while a 422 means the request
 * itself was wrong and retrying is pointless. Flattening both into a bare
 * `Error` forced every call site to pattern-match on prose, so none of them
 * distinguished the cases at all.
 */
export class ApiError extends Error {
  readonly status: number

  constructor(message: string, status: number) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

/**
 * The server's own explanation of a failed request, or a labelled fallback.
 *
 * ``res.statusText`` — what these call sites used to interpolate — is EMPTY
 * under HTTP/2 and HTTP/3 (the protocols do not carry a reason phrase), so
 * every error surfaced to the operator as "Failed to cancel job: " with no
 * reason at all. The backend writes an actionable message into the body's
 * ``detail`` field in three shapes: a plain string (most routes), an object
 * ``{code, message}`` (structured domain errors such as unsupported format),
 * and a list (FastAPI's 422 validation errors, one entry per bad field).
 * This helper reads all three; anything unparseable degrades to
 * ``<label> (HTTP <status>)`` rather than an empty suffix.
 */
export async function apiError(res: Response, label: string): Promise<ApiError> {
  const fallback = `${label} (HTTP ${res.status})`
  let message = fallback
  try {
    const body: unknown = await res.json()
    const detail =
      body !== null && typeof body === 'object' && 'detail' in body
        ? (body as { detail: unknown }).detail
        : null
    if (typeof detail === 'string' && detail.trim()) {
      message = detail
    } else if (detail !== null && typeof detail === 'object' && 'message' in detail) {
      const text = (detail as { message: unknown }).message
      if (typeof text === 'string' && text.trim()) message = text
    } else if (Array.isArray(detail)) {
      // FastAPI 422: [{loc: [...], msg: "...", type: "..."}, ...]
      const messages = detail
        .map((item) =>
          item !== null && typeof item === 'object' && 'msg' in item
            ? String((item as { msg: unknown }).msg)
            : null,
        )
        .filter((msg): msg is string => Boolean(msg))
      if (messages.length) message = messages.join('; ')
    }
  } catch {
    // Not JSON (a proxy error page, a truncated body): fall through.
  }
  return new ApiError(message, res.status)
}

/**
 * Exchange the entered key for the session cookie (and remember the key for
 * header-authenticated fetches). Returns false when the key is rejected.
 */
export async function signIn(apiKey: string): Promise<boolean> {
  setStoredApiKey(apiKey)
  const res = await fetch(`${BASE_URL}/system/session`, {
    method: 'POST',
    headers: { 'X-API-Key': apiKey },
  })
  if (!res.ok) {
    setStoredApiKey('')
    return false
  }
  return true
}

export async function signOut(): Promise<void> {
  setStoredApiKey('')
  await fetch(`${BASE_URL}/system/session`, { method: 'DELETE' }).catch(() => undefined)
}

export async function checkHealth(): Promise<HealthResponse> {
  const res = await fetch(`${BASE_URL}/health`)
  if (!res.ok) throw await apiError(res, "Health check failed")
  return res.json()
}

export async function assessJob(req: JobAssessRequest): Promise<JobAssessResponse> {
  const res = await apiFetch(`${BASE_URL}/jobs/assess`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  })
  if (!res.ok) {
    throw await apiError(res, "Assessment failed")
  }
  return res.json()
}

export async function submitJob(req: JobSubmitRequest): Promise<JobSubmitResponse> {
  const res = await apiFetch(`${BASE_URL}/jobs/submit`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  })
  if (!res.ok) {
    throw await apiError(res, "Job submission failed")
  }
  return res.json()
}

/**
 * Upload a source document to the server and return the staged path.
 *
 * Browsers cannot expose a real filesystem path (`File.path` is an
 * Electron-only property), so the wizard stages the bytes here and submits
 * the returned `file_path` as `input_path` instead of a client-side guess.
 */
export async function uploadSourceDocument(file: File): Promise<JobUploadResponse> {
  const form = new FormData()
  form.append('file', file)
  const res = await apiFetch(`${BASE_URL}/jobs/upload`, { method: 'POST', body: form })
  if (!res.ok) {
    throw await apiError(res, "Upload failed")
  }
  return res.json()
}

export async function getJobStatus(jobId: string): Promise<JobStatusResponse> {
  const res = await apiFetch(`${BASE_URL}/jobs/${encodeURIComponent(jobId)}/status`)
  if (!res.ok) throw await apiError(res, "Failed to fetch job status")
  return res.json()
}

export async function cancelJob(jobId: string): Promise<void> {
  const res = await apiFetch(`${BASE_URL}/jobs/${encodeURIComponent(jobId)}/cancel`, {
    method: 'POST',
  })
  if (!res.ok) throw await apiError(res, "Failed to cancel job")
}

/** Re-run a failed/cancelled job from its ledger checkpoints (embedded mode). */
export async function resumeJob(jobId: string): Promise<{ job_id: string; status: string }> {
  const res = await apiFetch(`${BASE_URL}/jobs/${encodeURIComponent(jobId)}/resume`, {
    method: 'POST',
  })
  if (!res.ok) {
    throw await apiError(res, "Failed to resume job")
  }
  return res.json()
}

/**
 * Discard a finished job's parse and rebuild it from the source document.
 *
 * The one operation neither resume nor re-export can perform: a fix to how the
 * source is *parsed* (block splitting, table reconstruction, formula and figure
 * crops) never reaches a job that has already finished, because the ledger
 * still names the artefacts the old code produced. Re-rendering reuses them.
 *
 * Destructive and expensive — every block is discarded and the book is
 * re-translated, incurring the provider cost of a first run, and human
 * revisions in the ledger go with it. Callers must confirm before invoking.
 */
export async function reparseJob(jobId: string): Promise<{ job_id: string; status: string }> {
  const res = await apiFetch(`${BASE_URL}/jobs/${encodeURIComponent(jobId)}/reparse`, {
    method: 'POST',
  })
  if (!res.ok) {
    throw await apiError(res, "Failed to re-parse job")
  }
  return res.json()
}

/** Remove a finished job's ledger and deliverables from the console. */
export async function deleteJob(jobId: string): Promise<{ job_id: string }> {
  const res = await apiFetch(`${BASE_URL}/jobs/${encodeURIComponent(jobId)}`, { method: 'DELETE' })
  if (!res.ok) {
    throw await apiError(res, "Failed to delete job")
  }
  return res.json()
}

/** The job queue: every ledger under db_dir, newest first. */
export async function listJobs(limit = 200): Promise<JobSummary[]> {
  const res = await apiFetch(`${BASE_URL}/jobs?limit=${limit}`)
  if (!res.ok) throw await apiError(res, "Failed to list jobs")
  const data = await res.json()
  return data.jobs ?? []
}

export async function getJobReport(jobId: string): Promise<Record<string, unknown>> {
  const res = await apiFetch(`${BASE_URL}/jobs/${encodeURIComponent(jobId)}/report`)
  if (!res.ok) throw await apiError(res, "Failed to fetch report")
  return res.json()
}

export async function getVisualReport(jobId: string): Promise<Record<string, unknown>> {
  const res = await apiFetch(`${BASE_URL}/jobs/${encodeURIComponent(jobId)}/visual-report`)
  if (!res.ok) throw await apiError(res, "Failed to fetch visual report")
  return res.json()
}

//: Terminal SSE event names the backend emits (``event: <status>``).
const TERMINAL_EVENTS = ['completed', 'failed', 'cancelled'] as const

//: Reconnect backoff bounds (PRD §9.2): 1s, 2s, 4s … capped at 30s.
const _RECONNECT_BASE_MS = 1000
const _RECONNECT_MAX_MS = 30000

export function subscribeJobProgress(
  jobId: string,
  onFrame: (frame: ProgressStreamFrame) => void,
  onError?: (err: Event) => void,
  onConnectionChange?: (connected: boolean) => void
): () => void {
  // The backend serves the stream at /jobs/{id}/stream and frames it as named
  // SSE events ("event: progress" then one terminal "event: <status>"). An
  // EventSource `onmessage` only fires for the unnamed default event, so the
  // stream must be subscribed per event name.
  //
  // Reconnection is manual so it can back off exponentially and carry the
  // cursor: a native EventSource retries on a fixed cadence and cannot set
  // Last-Event-ID on a fresh connection, so the missed frames are lost. We
  // remember the last `id:` and pass it back as `?last_event_id=`.
  let done = false
  let es: EventSource | null = null
  let lastEventId = ''
  let attempt = 0
  let everOpened = false
  let timer: ReturnType<typeof setTimeout> | null = null

  const open = () => {
    if (done) return
    const base = `${BASE_URL}/jobs/${encodeURIComponent(jobId)}/stream`
    const url = lastEventId ? `${base}?last_event_id=${encodeURIComponent(lastEventId)}` : base
    const source = new EventSource(url)
    es = source

    const remember = (event: MessageEvent) => {
      if (event.lastEventId) lastEventId = event.lastEventId
    }

    const dispatch = (event: MessageEvent) => {
      remember(event)
      try {
        onFrame({ job_id: jobId, ...JSON.parse(event.data) })
      } catch {
        onFrame({ job_id: jobId })
      }
    }

    const onTerminal = (event: MessageEvent) => {
      done = true
      remember(event)
      dispatch(event)
      source.close()
    }

    source.addEventListener('progress', dispatch as EventListener)
    for (const name of TERMINAL_EVENTS) {
      source.addEventListener(name, onTerminal as EventListener)
    }

    source.onopen = () => {
      attempt = 0
      everOpened = true
      onConnectionChange?.(true)
    }

    source.onerror = (err) => {
      // A close after a terminal frame also fires onerror; do not report that as
      // a stream failure.
      if (done) return
      onConnectionChange?.(false)
      if (onError) onError(err)
      source.close()
      attempt += 1
      // Give up on a stream that never connected: an EventSource does not
      // expose the HTTP status, but a job whose stream answers 404 on every
      // attempt (embedded mode drops terminal jobs from memory — anything a
      // CLI run completed, or after a restart) never opens at all, while the
      // status poll keeps the panel working. Retrying forever costs a request
      // per backoff window per open tab and nothing else. A stream that
      // connected once (a live job through an outage) keeps retrying as
      // before.
      if (!everOpened && attempt >= 6) {
        done = true
        return
      }
      const delay = Math.min(_RECONNECT_MAX_MS, _RECONNECT_BASE_MS * 2 ** (attempt - 1))
      timer = setTimeout(open, delay)
    }
  }

  open()

  return () => {
    done = true
    if (timer) clearTimeout(timer)
    es?.close()
  }
}

// --------------------------------------------------------------------------- //
// Deliverables
// --------------------------------------------------------------------------- //

export interface DeliverableListing {
  items: DeliverableItem[]
  /** True when the files on disk predate the job's ledger text. The download
   *  endpoint refuses such a job with 409, so callers must re-render first. */
  stale: boolean
  /** The shared three-state delivery verdict, or null when the job has no
   *  quality report yet. `blocked` means the download endpoint refuses the
   *  reader-facing files unless `force` is set — callers should warn before
   *  opening the download rather than surface the 409 afterwards. */
  deliveryVerdict: 'passed' | 'warning' | 'blocked' | null
}

export async function listDeliverables(jobId: string): Promise<DeliverableListing> {
  const res = await apiFetch(`${BASE_URL}/jobs/${encodeURIComponent(jobId)}/deliverables`)
  if (!res.ok) throw await apiError(res, "Failed to list deliverables")
  const data = await res.json()
  return {
    items: data.deliverables ?? [],
    stale: Boolean(data.stale),
    deliveryVerdict: data.delivery_verdict ?? null,
  }
}

export function deliverableDownloadUrl(
  jobId: string,
  key: string,
  opts: { inline?: boolean; force?: boolean } = {}
): string {
  const base = `${BASE_URL}/jobs/${encodeURIComponent(jobId)}/download/${encodeURIComponent(key)}`
  // `inline` asks the server to drop Content-Disposition: attachment and serve
  // a renderable media type, so the console's preview pane can embed the file
  // instead of triggering a download. The server CSP-sandboxes the response.
  //
  // `force` takes a delivery the quality gate blocked. The server refuses those
  // by default; the console only sets it after the operator confirms, having
  // read that some blocks ship the source text instead of a translation.
  if (opts.inline) return `${base}?inline=1`
  if (opts.force) return `${base}?force=1`
  return base
}

/**
 * Trigger browser native download of a file from a URL.
 * Uses an invisible link element with download attribute to avoid creating
 * blank about:blank tabs or running into browser popup blocker heuristics.
 */
export function triggerBrowserDownload(url: string, filename?: string): void {
  const link = document.createElement('a')
  link.href = url
  if (filename) link.download = filename
  link.style.display = 'none'
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
}


/**
 * The raw bytes of one deliverable as text, for in-console previews that need
 * to parse content (XLIFF segment tables, JSON digests). Binary deliverables
 * (PDF, EPUB) are previewed via the `?inline=1` URL instead — the browser
 * streams those itself.
 */
export async function fetchDeliverableText(jobId: string, key: string): Promise<string> {
  const res = await apiFetch(deliverableDownloadUrl(jobId, key))
  if (!res.ok) throw await apiError(res, "Failed to load preview")
  return res.text()
}

// --------------------------------------------------------------------------- //
// Language assets
// --------------------------------------------------------------------------- //

export interface GlossaryListing {
  terms: GlossaryTerm[]
  /**
   * False when no glossary file is configured at all (the server answers 409).
   *
   * Distinguishing this from "configured but empty" matters: the panel used to
   * render "0 active terms" plus a green "No glossary conflicts detected" for a
   * missing `glossary_path`, which asserts two facts it does not have — and the
   * operator never learns what to set.
   */
  configured: boolean
  /** The server's own explanation, when `configured` is false. */
  reason: string | null
}

export async function getGlossary(): Promise<GlossaryListing> {
  const res = await apiFetch(`${BASE_URL}/assets/glossary`)
  if (res.status === 409) {
    const err = await apiError(res, "Glossary is not configured")
    return { terms: [], configured: false, reason: err.message }
  }
  if (!res.ok) throw await apiError(res, "Failed to load glossary")
  const data = await res.json()
  return { terms: data.terms ?? [], configured: true, reason: null }
}

/** Create and activate the managed glossary when none is configured yet. */
export async function enableGlossary(): Promise<GlossaryTerm[]> {
  const res = await apiFetch(`${BASE_URL}/assets/glossary/enable`, { method: 'POST' })
  if (!res.ok) throw await apiError(res, 'Failed to enable glossary')
  const data = await res.json()
  return data.terms ?? []
}

export async function upsertGlossaryTerm(source: string, target: string): Promise<GlossaryTerm[]> {
  const res = await apiFetch(`${BASE_URL}/assets/glossary`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ source, target }),
  })
  if (!res.ok) {
    throw await apiError(res, "Failed to save term")
  }
  const data = await res.json()
  return data.terms ?? []
}

/** Bulk-import CSV/TSV/JSON pairs into the configured glossary file. */
export async function importGlossary(payload: {
  format: 'csv' | 'tsv' | 'json'
  content: string
}): Promise<{ parsed: number; imported: number }> {
  const res = await apiFetch(`${BASE_URL}/assets/glossary/import`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!res.ok) {
    throw await apiError(res, "Failed to import glossary")
  }
  return res.json()
}

/** Stage a per-book glossary server-side; returns the path a job submit accepts. */
export async function uploadGlossaryFile(file: File): Promise<{ path: string; name: string }> {
  const form = new FormData()
  form.append('file', file)
  const res = await apiFetch(`${BASE_URL}/assets/glossary/upload`, {
    method: 'POST',
    body: form,
  })
  if (!res.ok) {
    throw await apiError(res, "Failed to upload glossary")
  }
  return res.json()
}

export async function deleteGlossaryTerm(source: string): Promise<GlossaryTerm[]> {
  const res = await apiFetch(`${BASE_URL}/assets/glossary?source=${encodeURIComponent(source)}`, {
    method: 'DELETE',
  })
  if (!res.ok) throw await apiError(res, "Failed to delete term")
  const data = await res.json()
  return data.terms ?? []
}

export async function listTm(params: {
  limit?: number
  offset?: number
  src_lang?: string
  tgt_lang?: string
  query?: string
  provenance?: 'machine' | 'human_pe'
  usage?: 'never' | 'used'
  sort?: 'newest' | 'oldest' | 'most_used'
} = {}): Promise<TmPage> {
  const query = new URLSearchParams()
  if (params.limit !== undefined) query.set('limit', String(params.limit))
  if (params.offset !== undefined) query.set('offset', String(params.offset))
  if (params.src_lang) query.set('src_lang', params.src_lang)
  if (params.tgt_lang) query.set('tgt_lang', params.tgt_lang)
  if (params.query) query.set('query', params.query)
  if (params.provenance) query.set('provenance', params.provenance)
  if (params.usage) query.set('usage', params.usage)
  if (params.sort) query.set('sort', params.sort)
  const res = await apiFetch(`${BASE_URL}/assets/tm?${query.toString()}`)
  if (!res.ok) throw await apiError(res, "Failed to load translation memory")
  const data = await res.json()
  return {
    total: data.total ?? 0,
    offset: data.offset ?? 0,
    limit: data.limit ?? 50,
    machine_count: data.machine_count ?? 0,
    human_count: data.human_count ?? 0,
    never_used_count: data.never_used_count ?? 0,
    entries: data.entries ?? [],
  }
}

export async function evictTm(ids: number[]): Promise<number> {
  const res = await apiFetch(`${BASE_URL}/assets/tm/evict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ids }),
  })
  if (!res.ok) throw await apiError(res, "Failed to evict TM entries")
  const data = await res.json()
  return data.removed ?? 0
}

/** Delete the exact filtered scope that was shown in the confirmation dialog. */
export async function evictMatchingTm(params: {
  expected_total: number
  query?: string
  provenance?: 'machine' | 'human_pe'
  usage?: 'never' | 'used'
  src_lang?: string
  tgt_lang?: string
}): Promise<number> {
  const res = await apiFetch(`${BASE_URL}/assets/tm/evict-matching`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  })
  if (!res.ok) throw await apiError(res, 'Failed to evict filtered TM entries')
  const data = await res.json()
  return data.removed ?? 0
}

export interface GlossaryConflict {
  source: string
  targets: string[]
}

/** Sources configured with more than one target rendering. */
export async function getGlossaryConflicts(): Promise<GlossaryConflict[]> {
  const res = await apiFetch(`${BASE_URL}/assets/glossary/conflicts`)
  if (!res.ok) throw await apiError(res, "Failed to load glossary conflicts")
  const data = await res.json()
  return data.conflicts ?? []
}

/** Import TMX/JSON pairs into the shared translation memory. */
export async function importTm(payload: {
  format: 'tmx' | 'json'
  content: string
  src_lang: string
  tgt_lang: string
  provenance?: 'machine' | 'human_pe'
}): Promise<{ parsed: number; imported: number }> {
  const res = await apiFetch(`${BASE_URL}/assets/tm/import`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!res.ok) {
    throw await apiError(res, "Failed to import translation memory")
  }
  return res.json()
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
  /** The text a human reads and edits: the engine's mark wrappers already
   * unwrapped server-side (one owner — the engine's own delivered-text rule).
   * `target_text` stays raw for callers that want the storage truth. */
  delivered_text: string
  repair_failed: boolean
  repair_error: string | null
  quarantined: boolean
  mtqe_score: number | null
  error_flags: string[]
  issues: string[]
  human_verified: boolean
  tm_hit: boolean
  repair_rounds: number
  glossary_hits: string[]
  mqm_severity: string | null
  mqm_spans: Array<Record<string, unknown>>
  provenance: Record<string, unknown>
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
  const res = await apiFetch(`${BASE_URL}/jobs/${encodeURIComponent(jobId)}/segments?${query.toString()}`)
  if (!res.ok) throw await apiError(res, "Failed to load segments")
  const data = await res.json()
  return { total: data.total ?? 0, segments: data.segments ?? [] }
}

export async function listIssues(jobId: string): Promise<IssuesReport> {
  const res = await apiFetch(`${BASE_URL}/jobs/${encodeURIComponent(jobId)}/issues`)
  if (!res.ok) throw await apiError(res, "Failed to load issues")
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

/** URL of the source page PNG (the "before" half of the pixel-witness view). */
export function sourcePageUrl(jobId: string, page: number, opts: { dpi?: number } = {}): string {
  const query = new URLSearchParams()
  if (opts.dpi !== undefined) query.set('dpi', String(opts.dpi))
  const suffix = query.toString()
  const base = `${BASE_URL}/jobs/${encodeURIComponent(jobId)}/pages/${page}/source`
  return suffix ? `${base}?${suffix}` : base
}

export async function editSegment(
  jobId: string,
  blockId: string,
  targetText: string
): Promise<{ changed: boolean; tm_written: number; segment: Segment | null }> {
  const res = await apiFetch(
    `${BASE_URL}/jobs/${encodeURIComponent(jobId)}/segments/${encodeURIComponent(blockId)}`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ target_text: targetText }),
    }
  )
  if (!res.ok) {
    throw await apiError(res, "Failed to save revision")
  }
  return res.json()
}

/**
 * Re-render a finished job's deliverable from its current ledger text.
 *
 * A human edit lands in the ledger, but the delivered file was rendered before
 * it; without this the reviewer sees the fix in the preview while the customer
 * downloads the stale machine draft. Runs the export stage only — no token is
 * spent (every block is already terminal).
 */
export async function reExportJob(
  jobId: string
): Promise<{ artifact_path: string | null; message: string }> {
  const res = await apiFetch(`${BASE_URL}/jobs/${encodeURIComponent(jobId)}/re-export`, {
    method: 'POST',
  })
  if (!res.ok) {
    throw await apiError(res, "Failed to re-export")
  }
  return res.json()
}

export interface TermViolation {
  source: string
  expected: string
  surface: string
  kind: string
  occurrences: number
  cascade_all: number
  cascade_subsequent: number
}

export interface BlockTermsReport {
  job_id: string
  block_id: string
  glossary_size: number
  violations: TermViolation[]
}

/** Terminology findings for one block, with the cascade size each implies. */
export async function getBlockTerms(jobId: string, blockId: string): Promise<BlockTermsReport> {
  const res = await apiFetch(
    `${BASE_URL}/jobs/${encodeURIComponent(jobId)}/segments/${encodeURIComponent(blockId)}/terms`
  )
  if (!res.ok) throw await apiError(res, "Failed to load terminology")
  return res.json()
}

export interface TermPropagationResult {
  job_id: string
  block_id: string
  surface: string
  expected: string
  scope: string
  blocks_updated: number
  blocks_planned: number
  replacements: number
  capped: boolean
  block_ids: string[]
  tm_written: number
}

/** Replace one offending term surface with its canonical rendering (optionally book-wide). */
export async function propagateTerm(
  jobId: string,
  payload: { block_id: string; surface: string; expected: string; scope: 'block' | 'subsequent' | 'all' }
): Promise<TermPropagationResult> {
  const res = await apiFetch(`${BASE_URL}/jobs/${encodeURIComponent(jobId)}/term-propagation`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!res.ok) {
    throw await apiError(res, "Failed to propagate term")
  }
  return res.json()
}

// --------------------------------------------------------------------------- //
// System
// --------------------------------------------------------------------------- //

export async function getDoctor(probe = false): Promise<DoctorReport> {
  const url = `${BASE_URL}/system/doctor` + (probe ? '?probe=true' : '')
  const res = await apiFetch(url)
  if (!res.ok) throw await apiError(res, "Doctor check failed")
  return res.json()
}

export async function listModelProfiles(): Promise<ModelProfile[]> {
  const res = await apiFetch(`${BASE_URL}/api/v1/model-profiles`)
  if (!res.ok) throw await apiError(res, "Failed to load model profiles")
  return res.json()
}

/**
 * What the engine will actually call: provider, models, and the capability
 * profile the draft model resolves to.
 *
 * Capability is declared per provider block (`supports_temperature`,
 * `supports_vision`, …), not inferred from the model name. There is no preset
 * table any more: the engine used to ship fourteen name patterns (`o1`,
 * `claude`, …) and resolve by substring, which mis-answered for any model it had
 * not heard of — including every locally hosted one.
 */
export async function getRuntimeConfig(): Promise<RuntimeConfig> {
  const res = await apiFetch(`${BASE_URL}/system/runtime-config`)
  if (!res.ok) throw await apiError(res, "Failed to load runtime configuration")
  return res.json()
}

/** One provider block, with the credential reduced to a mask. */
export async function getProvider(name: string): Promise<ProviderDetail> {
  const res = await apiFetch(`${BASE_URL}/system/providers/${encodeURIComponent(name)}`)
  if (!res.ok) throw await apiError(res, `Failed to load provider ${name}`)
  return res.json()
}

/**
 * Create or update a `[providers.<name>]` block in `ubt.toml`.
 *
 * An `api_key` in the body is accepted only on a loopback, non-strict
 * deployment — the same server that lets the operator read `ubt.toml` already.
 * Anywhere else the field is refused with a message saying to use the file or
 * the environment, because a shared server must not let a bearer token mint
 * another one. The engine re-reads the file on each job start, so a save applies
 * to the next submitted job without a restart.
 *
 * A save is not necessarily *effective*: `UBT_LLM_API_KEY` outranks every
 * provider block, so the response carries `credential_shadowed_by_env` and the
 * form shows it rather than promising the next job.
 */
export async function saveProvider(
  name: string,
  changes: ProviderEditRequest,
): Promise<ProviderDetail> {
  const res = await apiFetch(`${BASE_URL}/system/providers/${encodeURIComponent(name)}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(changes),
  })
  if (!res.ok) throw await apiError(res, `Failed to save provider ${name}`)
  return res.json()
}

/**
 * Remove one declared `[providers.<name>]` block.
 *
 * A block still named by `[defaults]` (provider / repair_provider /
 * qe_judge_provider) is refused server-side with the referencing field, so the
 * UI can tell the operator what to clear first rather than orphaning a
 * reference that fails every future job.
 */
export async function deleteProvider(name: string): Promise<void> {
  const res = await apiFetch(`${BASE_URL}/system/providers/${encodeURIComponent(name)}`, {
    method: 'DELETE',
  })
  if (!res.ok) throw await apiError(res, `Failed to delete provider ${name}`)
}

/**
 * Make `name` the provider a run uses when nothing more explicit does.
 *
 * Writes `[defaults] provider` in `ubt.toml`. This is the step a fresh install
 * was missing: a declared block that nothing names leaves `provider` unset, so
 * the first job became a rehearsal even with a working endpoint and key.
 */
export async function setDefaultProvider(name: string): Promise<{ provider: string }> {
  const res = await apiFetch(`${BASE_URL}/system/providers/${encodeURIComponent(name)}/default`, {
    method: 'PUT',
  })
  if (!res.ok) throw await apiError(res, `Failed to make ${name} the default provider`)
  return res.json()
}

/**
 * The quality-estimation gate the engine will run.
 *
 * `engine` and `judgeEnabled` are two switches and only the second one scores
 * anything: `qe_engine = "tiered"` alone is a no-op the pipeline merely warns
 * about. `measurable` is the derived answer the card renders — no judge model
 * means no block is scored and the report's average QE stays at `0.0`.
 */
export async function getQeSettings(): Promise<QESettings> {
  const res = await apiFetch(`${BASE_URL}/system/qe-config`)
  if (!res.ok) throw await apiError(res, 'Failed to load QE configuration')
  return res.json()
}

/**
 * Write the QE gate into `ubt.toml [defaults]`.
 *
 * The console sends `qeEngine` and `judgeEnabled` together, because the two
 * documented-but-useless states (`tiered` with the judge off, and the judge on
 * with a `heuristic` label) are not worth offering as choices. The response
 * re-reads the effective configuration: `UBT_QE_*` outranks the file, so a save
 * can be written and still not be what the next job uses.
 */
export async function saveQeSettings(changes: QESettingsRequest): Promise<QESettings> {
  const res = await apiFetch(`${BASE_URL}/system/qe-config`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(changes),
  })
  if (!res.ok) throw await apiError(res, 'Failed to save QE configuration')
  return res.json()
}

/** The page-egress gate in force (runtime override; reverts at restart). */
export async function getEgressConfig(): Promise<{ allow_page_upload: boolean }> {
  const res = await apiFetch(`${BASE_URL}/system/egress-config`)
  if (!res.ok) throw await apiError(res, 'Failed to load egress configuration')
  return res.json()
}

export async function saveEgressConfig(changes: {
  allow_page_upload: boolean
}): Promise<{ allow_page_upload: boolean }> {
  const res = await apiFetch(`${BASE_URL}/system/egress-config`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(changes),
  })
  if (!res.ok) throw await apiError(res, 'Failed to save egress configuration')
  return res.json()
}

/** The live security boundary: reachable host, auth gate, and allowed roots. */
export async function getSystemInfo(): Promise<SystemInfo> {
  const res = await apiFetch(`${BASE_URL}/system/info`)
  if (!res.ok) throw await apiError(res, "Failed to load system info")
  return res.json()
}

/** The forbidden-path prefixes the server actually refuses (platform-specific). */
export async function getSystemBoundary(): Promise<{ forbidden_prefixes: string[] }> {
  const res = await apiFetch(`${BASE_URL}/system/boundary`)
  if (!res.ok) throw await apiError(res, "Failed to load the boundary list")
  return res.json()
}
