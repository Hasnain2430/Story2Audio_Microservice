/** Formatting helpers. Small, pure, and tested by eye in one place rather than inline. */

import type { JobStatus } from '@/api/types'

/** `3:07` — the form a player uses. Not `3m 7s`, which nobody scrubs by. */
export function formatDuration(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds) || seconds < 0) return '—'
  const total = Math.round(seconds)
  const minutes = Math.floor(total / 60)
  const rest = total % 60
  return `${minutes}:${String(rest).padStart(2, '0')}`
}

/** Elapsed time in the shortest honest unit: `840 ms`, `4.2 s`, `2 min 05 s`. */
export function formatElapsed(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds)) return '—'
  if (seconds < 1) return `${Math.round(seconds * 1000)} ms`
  if (seconds < 60) return `${seconds.toFixed(1)} s`
  const minutes = Math.floor(seconds / 60)
  return `${minutes} min ${String(Math.round(seconds % 60)).padStart(2, '0')} s`
}

export function formatBytes(bytes: number | null | undefined): string {
  if (!bytes) return '—'
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(0)} KB`
  return `${(bytes / 1024 ** 2).toFixed(1)} MB`
}

const RELATIVE = new Intl.RelativeTimeFormat(undefined, { numeric: 'auto' })

export function formatRelative(iso: string | null | undefined): string {
  if (!iso) return '—'
  const then = new Date(iso).getTime()
  if (Number.isNaN(then)) return '—'

  const deltaSeconds = (then - Date.now()) / 1000
  const abs = Math.abs(deltaSeconds)

  if (abs < 60) return RELATIVE.format(Math.round(deltaSeconds), 'second')
  if (abs < 3600) return RELATIVE.format(Math.round(deltaSeconds / 60), 'minute')
  if (abs < 86_400) return RELATIVE.format(Math.round(deltaSeconds / 3600), 'hour')
  return RELATIVE.format(Math.round(deltaSeconds / 86_400), 'day')
}

/** Seconds between two ISO timestamps, or `null` if either is missing. */
export function secondsBetween(from: string | null | undefined, to: string | null | undefined) {
  if (!from || !to) return null
  const delta = (new Date(to).getTime() - new Date(from).getTime()) / 1000
  return Number.isFinite(delta) ? delta : null
}

/** Wording the user sees. Deliberately not the raw enum. */
export const STATUS_LABEL: Record<JobStatus, string> = {
  queued: 'Queued',
  writing: 'Writing',
  written: 'Written',
  synthesizing: 'Recording',
  done: 'Ready',
  failed: 'Failed',
  cancelled: 'Cancelled',
}

/** Present tense, for the live view: what the machine is doing right now. */
export const STATUS_ACTIVITY: Record<JobStatus, string> = {
  queued: 'Waiting for a writer',
  writing: 'Writing the story',
  written: 'Story written — queued for recording',
  synthesizing: 'Recording the narration',
  done: 'Finished',
  failed: 'Stopped',
  cancelled: 'Cancelled',
}

export function truncate(text: string, max: number): string {
  if (text.length <= max) return text
  return `${text.slice(0, max - 1).trimEnd()}…`
}
