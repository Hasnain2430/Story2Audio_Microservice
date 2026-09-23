/**
 * Playing a story while it is still being recorded.
 *
 * The worker publishes each segment as it finishes, so the first few seconds of audio
 * exist long before the last ones do — 8 seconds into a job that takes a minute. This
 * hook turns that into listening: it fetches each segment as it is announced, decodes
 * it, and schedules it on one clock so the result is the file the worker is still
 * building, playing as it is built.
 *
 * **Why the Web Audio API and not a chain of `<audio>` elements.** Swapping the `src` of
 * an element, or starting the next element when the previous one ends, both schedule on
 * the *event loop* — the gap between clips is whenever the browser gets round to it,
 * typically tens of milliseconds and never the same twice. `start(when)` schedules on the
 * audio hardware clock, so segments butt together exactly where the timeline says they
 * do. The silences here are meaningful: 80 ms where the text was split for length, 420 ms
 * where the speaker changes. Rendering them with event-loop jitter would undo the work
 * that put them there.
 *
 * **Underrun.** Synthesis normally runs about twice as fast as playback, so segments
 * arrive ahead of the playhead. When they do not — a slow GPU, a cold engine — the
 * schedule is rebased rather than dropped: playback pauses at the last segment it has and
 * resumes where it left off, keeping every gap intact. Dropping a segment would silently
 * desynchronise the read-along from the audio, and skipping ahead would lose a sentence.
 */

import { useCallback, useEffect, useRef, useState } from 'react'

import { segmentAudioUrl } from '@/api/client'
import type { SpokenSegment } from '@/api/types'

export interface LivePlayback {
  playing: boolean
  /** Position in the assembled story, in seconds. */
  currentTime: number
  /** How much of the story has been rendered and fetched, in seconds. */
  bufferedUntil: number
  /** True while waiting for a segment that has not been rendered yet. */
  starved: boolean
  supported: boolean
  play: () => void
  pause: () => void
  seek: (seconds: number) => void
}

interface Loaded {
  segment: SpokenSegment
  buffer: AudioBuffer
}

export function useLivePlayback(jobId: string | undefined, segments: SpokenSegment[]): LivePlayback {
  const [playing, setPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [bufferedUntil, setBufferedUntil] = useState(0)
  const [starved, setStarved] = useState(false)

  const context = useRef<AudioContext | null>(null)
  const loaded = useRef<Map<number, Loaded>>(new Map())
  const fetching = useRef<Set<number>>(new Set())
  const sources = useRef<AudioBufferSourceNode[]>([])
  /** Audio-clock time that corresponds to story time zero. */
  const origin = useRef(0)
  /** Story time the playhead should resume from when paused. */
  const offset = useRef(0)
  const scheduledUpTo = useRef(-1)
  const frame = useRef<number | null>(null)
  /** The job whose downloads are wanted; anything else that lands is discarded. */
  const loadingFor = useRef<string | undefined>(undefined)
  /** Audio-clock time at which playback ran out of rendered audio, if it has. */
  const stalledAt = useRef<number | null>(null)
  /** Where the contiguous run of loaded audio ends, read on the animation frame. */
  const playableUntil = useRef(0)
  /** Read inside callbacks that must not be re-created when playback starts or stops. */
  const playingRef = useRef(false)

  const supported = typeof window !== 'undefined' && 'AudioContext' in window

  // Mirrored in an effect rather than assigned during render: writing a ref while
  // rendering is what makes a component show one value and behave as another.
  useEffect(() => {
    playingRef.current = playing
  }, [playing])

  useEffect(() => {
    playableUntil.current = bufferedUntil
  }, [bufferedUntil])

  // --- Fetching -------------------------------------------------------------------------

  useEffect(() => {
    if (!jobId || !supported) return
    // Captured after the guard: the narrowing above does not reach into the nested
    // async function, and the job must not change under an in-flight download anyway.
    const id = jobId

    async function load(segment: SpokenSegment) {
      if (loaded.current.has(segment.index) || fetching.current.has(segment.index)) return
      fetching.current.add(segment.index)

      try {
        const response = await fetch(segmentAudioUrl(id, segment.index), {
          credentials: 'include',
        })
        if (!response.ok) return

        const bytes = await response.arrayBuffer()
        const ctx = context.current ?? new AudioContext()
        context.current = ctx
        const buffer = await ctx.decodeAudioData(bytes)
        // Guarded on the job, not on the effect run. This effect re-runs every time a
        // segment is announced, and an earlier version cancelled its in-flight fetches
        // when it did — so during a live render, where a segment lands every few
        // seconds, almost every download was thrown away just before being stored. The
        // player looked like it was playing while holding no audio at all.
        if (loadingFor.current !== id) return

        loaded.current.set(segment.index, { segment, buffer })
        setBufferedUntil(contiguousEnd(loaded.current, segments.length))
      } catch {
        // A segment that cannot be fetched or decoded is left out; the next poll of the
        // job, or the finished file, covers it. Playback is an enhancement here.
      } finally {
        fetching.current.delete(segment.index)
      }
    }

    for (const segment of segments) void load(segment)
  }, [jobId, segments, supported])

  // Discards anything still in flight for a job the page has moved away from.
  useEffect(() => {
    loadingFor.current = jobId
    // Captured now: the cleanup runs after the refs may have been reassigned, and these
    // are the maps this run populated.
    const decoded = loaded.current
    const inFlight = fetching.current

    return () => {
      loadingFor.current = undefined
      decoded.clear()
      inFlight.clear()
    }
  }, [jobId])

  // --- Scheduling -----------------------------------------------------------------------

  const schedule = useCallback(() => {
    const ctx = context.current
    if (!ctx) return

    // Only segments that are contiguous from the playhead can be scheduled: a gap means
    // the one before it has not arrived, and starting the later one would play the story
    // out of order.
    for (let index = scheduledUpTo.current + 1; ; index += 1) {
      const entry = loaded.current.get(index)
      if (!entry) {
        // Nothing more to schedule. While playing, that means synthesis has not caught
        // up yet, which the UI says out loud rather than looking frozen.
        setStarved(playingRef.current)
        break
      }

      // Playback ran out of audio earlier and has been holding. Move the whole schedule
      // forward by exactly how long it waited, so the story resumes where it stopped and
      // every gap between segments keeps its intended length.
      if (stalledAt.current !== null) {
        origin.current += ctx.currentTime - stalledAt.current
        stalledAt.current = null
      }

      const at = origin.current + entry.segment.start_seconds
      if (at < ctx.currentTime - 0.05) {
        // Late for another reason — a slow decode, a tab that was backgrounded. Same
        // remedy: shift the schedule rather than dropping or overlapping anything.
        origin.current += ctx.currentTime - at
      }

      const source = ctx.createBufferSource()
      source.buffer = entry.buffer
      source.connect(ctx.destination)
      source.start(Math.max(ctx.currentTime, origin.current + entry.segment.start_seconds))
      sources.current.push(source)
      scheduledUpTo.current = index
      setStarved(false)
    }
  }, [])

  // Re-run whenever a new segment lands, so a story that is still rendering keeps going.
  useEffect(() => {
    if (playing) schedule()
  }, [playing, bufferedUntil, schedule])

  // --- Clock ----------------------------------------------------------------------------

  useEffect(() => {
    if (!playing) return

    const tick = () => {
      const ctx = context.current
      if (ctx) {
        const position = Math.max(0, ctx.currentTime - origin.current)

        // The clock stops where the rendered audio stops. Letting it run on would carry
        // the read-along past audio nobody can hear yet — the highlight would sail ahead
        // of the narration and never come back, which is worse than pausing.
        if (position >= playableUntil.current && playableUntil.current > 0) {
          stalledAt.current ??= ctx.currentTime
          setCurrentTime(playableUntil.current)
          setStarved(true)
        } else {
          setCurrentTime(position)
        }
      }
      frame.current = window.requestAnimationFrame(tick)
    }

    frame.current = window.requestAnimationFrame(tick)
    return () => {
      if (frame.current !== null) window.cancelAnimationFrame(frame.current)
    }
  }, [playing])

  // --- Transport ------------------------------------------------------------------------

  const stopAll = useCallback(() => {
    for (const source of sources.current) {
      try {
        source.stop()
      } catch {
        // Already finished; stopping twice is not an error worth reporting.
      }
    }
    sources.current = []
    scheduledUpTo.current = -1
  }, [])

  const play = useCallback(() => {
    // Created on a user gesture: browsers refuse to start an AudioContext otherwise, and
    // one created earlier would sit suspended.
    const ctx = context.current ?? new AudioContext()
    context.current = ctx
    void ctx.resume()

    origin.current = ctx.currentTime - offset.current
    setPlaying(true)
  }, [])

  const pause = useCallback(() => {
    const ctx = context.current
    stalledAt.current = null
    if (ctx) offset.current = Math.min(playableUntil.current, Math.max(0, ctx.currentTime - origin.current))
    stopAll()
    setPlaying(false)
  }, [stopAll])

  const seek = useCallback(
    (seconds: number) => {
      const ctx = context.current
      offset.current = Math.max(0, seconds)
      setCurrentTime(offset.current)
      stopAll()

      if (ctx && playingRef.current) {
        origin.current = ctx.currentTime - offset.current
        // Anything before the new playhead counts as already played.
        scheduledUpTo.current = lastIndexBefore(segments, offset.current)
        schedule()
      }
    },
    [schedule, segments, stopAll],
  )

  useEffect(
    () => () => {
      stopAll()
      void context.current?.close()
      context.current = null
    },
    [stopAll],
  )

  return { playing, currentTime, bufferedUntil, starved, supported, play, pause, seek }
}

/**
 * How far the story can be played without a hole in it.
 *
 * Not simply the furthest segment received: segments can in principle arrive out of
 * order, and reporting a later one as buffered would promise audio that cannot be played
 * yet.
 */
function contiguousEnd(loaded: Map<number, Loaded>, total: number): number {
  let end = 0
  for (let index = 0; index < total; index += 1) {
    const entry = loaded.get(index)
    if (!entry) break
    end = entry.segment.end_seconds
  }
  return end
}

/** The last segment that finishes before `seconds`, or -1. */
function lastIndexBefore(segments: SpokenSegment[], seconds: number): number {
  let result = -1
  for (const segment of segments) {
    if (segment.end_seconds <= seconds) result = segment.index
    else break
  }
  return result
}
