/**
 * Typed HTTP client.
 *
 * Two jobs: send cookies (the API authenticates with an HttpOnly session cookie, so
 * every request needs `credentials: 'include'`), and turn the server's error envelope
 * into one exception type the UI can branch on.
 *
 * The server guarantees one error shape for every failure:
 *
 *     { "error": { "code": "voice_too_short", "message": "...", "retryable": false } }
 *
 * so the client never has to guess whether a body is an error.
 */

import type {
  CreateJobRequest,
  CreateJobResponse,
  ErrorCode,
  Job,
  JobPage,
  Voice,
  VoicePage,
} from '@/api/types'

/** Same-origin in dev (Vite proxies `/v1`), and in prod when served behind one host. */
const API_BASE: string = import.meta.env.VITE_API_ORIGIN ?? ''

export class ApiError extends Error {
  readonly code: ErrorCode | 'network' | 'unknown'
  readonly status: number
  readonly retryable: boolean
  /** Field paths from a validation failure, when the server named them. */
  readonly fields: string[]

  constructor(init: {
    code: ErrorCode | 'network' | 'unknown'
    message: string
    status: number
    retryable: boolean
    fields?: string[] | undefined
  }) {
    super(init.message)
    this.name = 'ApiError'
    this.code = init.code
    this.status = init.status
    this.retryable = init.retryable
    this.fields = init.fields ?? []
  }
}

/** Normalise the several shapes `HeadersInit` can take into a plain record. */
function toRecord(headers: HeadersInit | undefined): Record<string, string> {
  if (!headers) return {}
  if (headers instanceof Headers) return Object.fromEntries(headers.entries())
  if (Array.isArray(headers)) return Object.fromEntries(headers)
  return headers
}

interface ErrorEnvelope {
  error?: { code?: ErrorCode; message?: string; retryable?: boolean; fields?: string[] }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${API_BASE}${path}`, {
      ...init,
      // Required: the session is an HttpOnly cookie, and without this the browser
      // silently omits it and every request looks like a brand new visitor.
      credentials: 'include',
      headers: new Headers({ Accept: 'application/json', ...toRecord(init.headers) }),
    })
  } catch {
    throw new ApiError({
      code: 'network',
      message: 'Could not reach the server. Check your connection.',
      status: 0,
      retryable: true,
    })
  }

  // 204 has no body, and calling `.json()` on it throws.
  if (response.status === 204) {
    return undefined as T
  }

  const body: unknown = await response.json().catch(() => null)

  if (!response.ok) {
    const envelope = (body ?? {}) as ErrorEnvelope
    throw new ApiError({
      code: envelope.error?.code ?? 'unknown',
      message: envelope.error?.message ?? 'Something went wrong.',
      status: response.status,
      retryable: envelope.error?.retryable ?? response.status >= 500,
      fields: envelope.error?.fields,
    })
  }

  return body as T
}

function json(method: string, payload: unknown, headers: Record<string, string> = {}): RequestInit {
  return {
    method,
    body: JSON.stringify(payload),
    headers: { 'Content-Type': 'application/json', ...headers },
  }
}

// --- Jobs ---------------------------------------------------------------------------

export const api = {
  createJob(body: CreateJobRequest, idempotencyKey?: string): Promise<CreateJobResponse> {
    // An idempotency key means a retried submission returns the original job rather
    // than paying for the same generation twice.
    const headers = idempotencyKey ? { 'Idempotency-Key': idempotencyKey } : {}
    return request<CreateJobResponse>('/v1/jobs', json('POST', body, headers))
  },

  getJob(id: string): Promise<Job> {
    return request<Job>(`/v1/jobs/${id}`)
  },

  listJobs(params: { limit?: number; cursor?: string } = {}): Promise<JobPage> {
    const query = new URLSearchParams()
    query.set('limit', String(params.limit ?? 20))
    if (params.cursor) query.set('cursor', params.cursor)
    return request<JobPage>(`/v1/jobs?${query.toString()}`)
  },

  cancelJob(id: string): Promise<Job> {
    return request<Job>(`/v1/jobs/${id}`, { method: 'DELETE' })
  },

  // --- Voices -------------------------------------------------------------------------

  listVoices(params: { limit?: number; cursor?: string } = {}): Promise<VoicePage> {
    const query = new URLSearchParams()
    query.set('limit', String(params.limit ?? 50))
    if (params.cursor) query.set('cursor', params.cursor)
    return request<VoicePage>(`/v1/voices?${query.toString()}`)
  },

  createVoice(name: string, file: Blob, filename = 'voice.wav'): Promise<Voice> {
    const form = new FormData()
    form.set('name', name)
    form.set('file', file, filename)
    // No Content-Type header: the browser must set the multipart boundary itself.
    return request<Voice>('/v1/voices', { method: 'POST', body: form })
  },

  async deleteVoice(id: string): Promise<void> {
    await request<undefined>(`/v1/voices/${id}`, { method: 'DELETE' })
  },
}

/** WebSocket URL for a job's event stream, on whichever origin serves the API. */
/**
 * Absolute URL for one rendered segment's audio.
 *
 * Built through the same base as every other call rather than as a bare path: with
 * `VITE_API_ORIGIN` set, a relative fetch would go to whatever host is serving the
 * frontend, which in a split deployment is not the API at all.
 */
export function segmentAudioUrl(jobId: string, index: number): string {
  return `${API_BASE}/v1/jobs/${jobId}/segments/${index}/audio`
}

export function jobEventsUrl(jobId: string): string {
  const base = API_BASE !== '' ? API_BASE : window.location.origin
  const url = new URL(`/v1/jobs/${jobId}/events`, base)
  url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:'
  return url.toString()
}
