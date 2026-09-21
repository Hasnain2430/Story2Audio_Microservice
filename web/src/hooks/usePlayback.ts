/**
 * One audio element, shared by everything that needs to know where playback is.
 *
 * The player draws a waveform and the transcript highlights the line being read, and
 * both have to agree to the frame. Keeping the element and its clock here — rather than
 * inside the player, with the transcript guessing — is what makes them the same truth.
 *
 * Time is sampled with `requestAnimationFrame`, not from `timeupdate`. The DOM event
 * fires roughly four times a second, which is fine for a clock readout and visibly wrong
 * for highlighting a word: at 4 Hz a short word is skipped entirely and every transition
 * lands up to 250 ms late. The rAF loop runs only while audio is playing, so a paused
 * page costs nothing.
 */

import { useCallback, useEffect, useRef, useState } from 'react'

export interface Playback {
  /** Attach to the single `<audio>` element. */
  ref: React.RefObject<HTMLAudioElement | null>
  playing: boolean
  /** Seconds, sampled per animation frame while playing. */
  currentTime: number
  duration: number
  rate: number
  toggle: () => void
  seek: (seconds: number) => void
  setRate: (rate: number) => void
  /** Wire these to the element; they keep the hook's state honest. */
  handlers: {
    onPlay: () => void
    onPause: () => void
    onEnded: () => void
    onTimeUpdate: (event: React.SyntheticEvent<HTMLAudioElement>) => void
    onLoadedMetadata: (event: React.SyntheticEvent<HTMLAudioElement>) => void
  }
}

export function usePlayback(fallbackDuration = 0): Playback {
  const ref = useRef<HTMLAudioElement | null>(null)
  const frame = useRef<number | null>(null)

  const [playing, setPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  // Zero until `loadedmetadata`; the job's own figure stands in until then, so the
  // scrubber has a scale from the first paint. Derived rather than seeded into state,
  // which would need an effect to keep it in step with a changing asset.
  const [measured, setMeasured] = useState(0)
  const [rate, setRateState] = useState(1)

  useEffect(() => {
    if (!playing) return

    const tick = () => {
      const element = ref.current
      if (element) setCurrentTime(element.currentTime)
      frame.current = window.requestAnimationFrame(tick)
    }

    frame.current = window.requestAnimationFrame(tick)
    return () => {
      if (frame.current !== null) window.cancelAnimationFrame(frame.current)
      frame.current = null
    }
  }, [playing])

  useEffect(() => {
    const element = ref.current
    if (element) element.playbackRate = rate
  }, [rate])

  const toggle = useCallback(() => {
    const element = ref.current
    if (!element) return
    if (element.paused) {
      // `play()` rejects when the browser blocks autoplay or the source is gone. There is
      // nothing useful to do about it here, and an unhandled rejection in the console is
      // noise that hides real errors.
      void element.play().catch(() => undefined)
    } else {
      element.pause()
    }
  }, [])

  const seek = useCallback((seconds: number) => {
    const element = ref.current
    if (!element) return
    const limit = Number.isFinite(element.duration) ? element.duration : seconds
    const target = Math.max(0, Math.min(limit, seconds))
    element.currentTime = target
    // Set immediately rather than waiting for the next frame, so a click on the
    // transcript moves the highlight even while paused.
    setCurrentTime(target)
  }, [])

  return {
    ref,
    playing,
    currentTime,
    duration: measured || fallbackDuration,
    rate,
    toggle,
    seek,
    setRate: setRateState,
    handlers: {
      onPlay: () => setPlaying(true),
      onPause: () => setPlaying(false),
      onEnded: () => setPlaying(false),
      // Still wired: it is the only signal while paused (after a seek, or a buffering
      // stall), and it costs nothing next to the rAF loop.
      onTimeUpdate: (event) => setCurrentTime(event.currentTarget.currentTime),
      onLoadedMetadata: (event) => {
        const value = event.currentTarget.duration
        if (Number.isFinite(value) && value > 0) setMeasured(value)
      },
    },
  }
}
