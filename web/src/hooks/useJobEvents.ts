/**
 * Live job progress, over a WebSocket when one is available and by polling when it is
 * not.
 *
 * The important property (ADR-0003): the WebSocket is an *optimisation* over polling
 * `GET /v1/jobs/{id}`, never a requirement. Both paths write into the same TanStack
 * Query cache, so nothing downstream can tell which one is running — that
 * indistinguishability is the contract, and it is what makes the fallback safe rather
 * than a degraded mode nobody tests.
 *
 * Streamed tokens are held in local state rather than the cache: they are a rendering
 * detail that exists only while the story is being written, and the authoritative text
 * arrives with `story_done` and on every subsequent fetch.
 */

import { useCallback, useEffect, useRef, useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'

import { api, jobEventsUrl } from '@/api/client'
import { isTerminalEvent, parseJobEvent, type JobEvent } from '@/api/events'
import type { Job } from '@/api/types'
import { isTerminal } from '@/api/types'
import { jobKeys } from '@/hooks/queryKeys'

/** How the client is currently learning about progress. Surfaced in the UI, honestly. */
export type Transport = 'connecting' | 'live' | 'polling' | 'closed'

interface JobEventsState {
  /** Text accumulated from `token` frames, shown while the story is being written. */
  streamedText: string
  segmentsDone: number
  segmentsTotal: number
  transport: Transport
  /** True once a `seq` gap is seen — the client refetches rather than replaying. */
  recoveredFromGap: boolean
}

const INITIAL: JobEventsState = {
  streamedText: '',
  segmentsDone: 0,
  segmentsTotal: 0,
  transport: 'connecting',
  recoveredFromGap: false,
}

/** Give up on the socket after this many consecutive failures and poll instead. */
const MAX_WS_ATTEMPTS = 3
const POLL_INTERVAL_MS = 1500

export function useJobEvents(jobId: string | undefined, enabled: boolean): JobEventsState {
  const queryClient = useQueryClient()
  const [state, setState] = useState<JobEventsState>(INITIAL)
  const [trackedJob, setTrackedJob] = useState(jobId)

  // Refs, not state: mutating these must not re-render, and the socket callbacks need
  // the current value rather than the one captured when the effect ran.
  const lastSeq = useRef(0)
  const attempts = useRef(0)

  // Resetting during render rather than in an effect. React documents this as the way
  // to adjust state when a prop changes; doing it in an effect would render the new
  // job's page with the previous job's streamed text for one frame first.
  if (jobId !== trackedJob) {
    setTrackedJob(jobId)
    setState(INITIAL)
  }

  const refetchJob = useCallback(
    (id: string) => {
      void queryClient.invalidateQueries({ queryKey: jobKeys.detail(id) })
    },
    [queryClient],
  )

  const applyEvent = useCallback(
    (id: string, event: JobEvent) => {
      // A gap means frames were dropped. There is no history in Redis to replay, so the
      // response is to refetch the authoritative record and carry on.
      const expected = lastSeq.current + 1
      if (lastSeq.current > 0 && event.seq > expected) {
        setState((s) => ({ ...s, recoveredFromGap: true }))
        refetchJob(id)
      }
      lastSeq.current = Math.max(lastSeq.current, event.seq)

      switch (event.type) {
        case 'status':
          queryClient.setQueryData<Job>(jobKeys.detail(id), (prev) =>
            prev ? { ...prev, status: event.status } : prev,
          )
          // A snapshot of an already-running job means we joined late and missed the
          // earlier frames; the fetched record fills them in.
          if (event.snapshot && event.status !== 'queued') refetchJob(id)
          break

        case 'token':
          setState((s) => ({ ...s, streamedText: s.streamedText + event.text }))
          break

        case 'story_done':
          setState((s) => ({ ...s, streamedText: event.text }))
          queryClient.setQueryData<Job>(jobKeys.detail(id), (prev) =>
            prev ? { ...prev, story_text: event.text } : prev,
          )
          break

        case 'progress':
          setState((s) => ({ ...s, segmentsDone: event.done, segmentsTotal: event.total }))
          break

        case 'done':
        case 'failed':
        case 'cancelled':
          // The terminal event carries no presigned URL — signing belongs to the read
          // path, where the TTL is meaningful — so the record is refetched to get it.
          refetchJob(id)
          break

        default:
          assertNever(event)
      }
    },
    [queryClient, refetchJob],
  )

  useEffect(() => {
    if (!jobId || !enabled) return

    // Counters belong to one subscription, and this effect is that subscription's
    // lifetime. Resetting them during render would touch a ref mid-render.
    lastSeq.current = 0
    attempts.current = 0

    let socket: WebSocket | null = null
    let pollTimer: number | null = null
    let reconnectTimer: number | null = null
    let disposed = false

    const stopPolling = () => {
      if (pollTimer !== null) {
        window.clearInterval(pollTimer)
        pollTimer = null
      }
    }

    const startPolling = () => {
      if (disposed || pollTimer !== null) return
      setState((s) => ({ ...s, transport: 'polling' }))

      pollTimer = window.setInterval(() => {
        void api
          .getJob(jobId)
          .then((job) => {
            queryClient.setQueryData<Job>(jobKeys.detail(jobId), job)
            if (job.segment_count) {
              setState((s) => ({ ...s, segmentsTotal: job.segment_count ?? s.segmentsTotal }))
            }
            if (isTerminal(job.status)) {
              stopPolling()
              setState((s) => ({ ...s, transport: 'closed' }))
            }
          })
          .catch(() => {
            // Transient failures are expected while polling; the next tick retries.
          })
      }, POLL_INTERVAL_MS)
    }

    const connect = () => {
      if (disposed) return

      try {
        socket = new WebSocket(jobEventsUrl(jobId))
      } catch {
        startPolling()
        return
      }

      socket.onopen = () => {
        attempts.current = 0
        stopPolling()
        setState((s) => ({ ...s, transport: 'live' }))
      }

      socket.onmessage = (message) => {
        if (typeof message.data !== 'string') return
        const event = parseJobEvent(message.data)
        if (!event) return

        applyEvent(jobId, event)
        if (isTerminalEvent(event)) {
          disposed = true
          socket?.close()
          setState((s) => ({ ...s, transport: 'closed' }))
        }
      }

      socket.onerror = () => {
        // `onclose` always follows, and that is where reconnection is decided.
      }

      socket.onclose = () => {
        if (disposed) return
        attempts.current += 1

        if (attempts.current >= MAX_WS_ATTEMPTS) {
          // Fall back for good. The UI keeps working identically; only the `transport`
          // label changes, which is the visible proof the fallback is real.
          startPolling()
          return
        }

        const backoff = Math.min(1000 * 2 ** (attempts.current - 1), 8000)
        reconnectTimer = window.setTimeout(connect, backoff)
      }
    }

    connect()

    return () => {
      disposed = true
      stopPolling()
      if (reconnectTimer !== null) window.clearTimeout(reconnectTimer)
      socket?.close()
    }
  }, [jobId, enabled, applyEvent, queryClient])

  // `transport` is derived rather than stored for the disabled case: a finished job
  // has no live connection, and reporting one would be a lie the UI acts on.
  return enabled ? state : { ...state, transport: 'closed' }
}

function assertNever(value: never): never {
  throw new Error(`unhandled job event: ${JSON.stringify(value)}`)
}
