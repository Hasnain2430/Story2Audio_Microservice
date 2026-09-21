/**
 * The WebSocket event contract (ADR-0003).
 *
 * Mirrors `story2audio_shared.events`. These are the one set of types not derived from
 * the OpenAPI schema, because WebSocket frames do not appear in it — so the discriminated
 * union is maintained by hand and `assertNever` in the consumer makes a missing variant a
 * compile error rather than a silent no-op.
 *
 * Two properties the client depends on:
 *
 *  - every frame carries a per-job monotonic `seq`, so a gap means frames were dropped;
 *  - the first frame of every connection is a `status` with `snapshot: true`, built from
 *    the database, so a late or reconnecting client starts from truth.
 *
 * On a gap the client refetches rather than replaying: Redis pub/sub keeps no history.
 */

import type { AudioAsset, ErrorCode, JobStatus } from '@/api/types'

/** Bumped only on a breaking change. A frame from an unknown version is ignored. */
export const EVENT_SCHEMA_VERSION = 1

interface BaseEvent {
  job_id: string
  seq: number
  at: string
  v: number
}

export interface StatusEvent extends BaseEvent {
  type: 'status'
  status: JobStatus
  /** True for the connect-time snapshot, which reuses the current sequence number. */
  snapshot: boolean
}

export interface TokenEvent extends BaseEvent {
  type: 'token'
  text: string
}

export interface StoryDoneEvent extends BaseEvent {
  type: 'story_done'
  text: string
  word_count: number
}

export interface ProgressEvent extends BaseEvent {
  type: 'progress'
  done: number
  total: number
}

export interface DoneEvent extends BaseEvent {
  type: 'done'
  audio: AudioAsset[]
  duration_seconds: number
}

export interface FailedEvent extends BaseEvent {
  type: 'failed'
  code: ErrorCode
  message: string
  retryable: boolean
}

export interface CancelledEvent extends BaseEvent {
  type: 'cancelled'
}

export type JobEvent =
  | StatusEvent
  | TokenEvent
  | StoryDoneEvent
  | ProgressEvent
  | DoneEvent
  | FailedEvent
  | CancelledEvent

const TERMINAL_EVENT_TYPES = new Set<JobEvent['type']>(['done', 'failed', 'cancelled'])

export function isTerminalEvent(event: JobEvent): boolean {
  return TERMINAL_EVENT_TYPES.has(event.type)
}

const KNOWN_TYPES = new Set<string>([
  'status',
  'token',
  'story_done',
  'progress',
  'done',
  'failed',
  'cancelled',
])

/**
 * Parse a frame, returning `null` for anything unrecognised.
 *
 * A frame from a future schema version, or of an unknown type, is dropped rather than
 * coerced. Mis-rendering an event we do not understand is worse than ignoring it — and
 * `GET /v1/jobs/{id}` remains authoritative either way.
 */
export function parseJobEvent(raw: string): JobEvent | null {
  let value: unknown
  try {
    value = JSON.parse(raw)
  } catch {
    return null
  }

  if (typeof value !== 'object' || value === null) return null
  const frame = value as Record<string, unknown>

  if (frame.v !== EVENT_SCHEMA_VERSION) return null
  if (typeof frame.type !== 'string' || !KNOWN_TYPES.has(frame.type)) return null
  if (typeof frame.seq !== 'number') return null

  return frame as unknown as JobEvent
}
