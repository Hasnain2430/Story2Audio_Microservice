/**
 * Audio player.
 *
 * A custom transport rather than `<audio controls>`, because the default control strip
 * is the one element that would make this page look like every other web app — and
 * because the waveform doubles as the scrubber, which is the natural gesture for
 * something you are listening to rather than watching.
 *
 * The waveform is drawn from the decoded MP3 once, then cached: it is a real picture of
 * the audio, not a decorative squiggle.
 */

import { useCallback, useEffect, useRef, useState } from 'react'

import type { AudioAsset } from '@/api/types'
import { formatBytes, formatDuration } from '@/lib/format'
import { Button } from '@/components/primitives'
import '@/components/AudioPlayer.css'

const BUCKETS = 220

interface AudioPlayerProps {
  assets: AudioAsset[]
  /** Shown while the waveform is still being computed. */
  fallbackDuration?: number | undefined
}

export function AudioPlayer({ assets, fallbackDuration }: AudioPlayerProps) {
  const playable = assets.find((a) => a.format === 'mp3') ?? assets[0]

  const audioRef = useRef<HTMLAudioElement | null>(null)
  const [playing, setPlaying] = useState(false)
  const [current, setCurrent] = useState(0)
  const [duration, setDuration] = useState(fallbackDuration ?? 0)
  const [peaks, setPeaks] = useState<number[] | null>(null)
  const [rate, setRate] = useState(1)

  // --- Waveform ------------------------------------------------------------------------

  useEffect(() => {
    if (!playable) return
    let cancelled = false

    async function analyse(url: string) {
      try {
        const response = await fetch(url)
        const buffer = await response.arrayBuffer()
        const ctx = new OfflineAudioContext(1, 1, 44_100)
        const decoded = await ctx.decodeAudioData(buffer)
        if (cancelled) return

        const data = decoded.getChannelData(0)
        const size = Math.floor(data.length / BUCKETS)
        const result: number[] = []

        for (let i = 0; i < BUCKETS; i += 1) {
          let peak = 0
          for (let j = 0; j < size; j += 1) {
            const value = Math.abs(data[i * size + j] ?? 0)
            if (value > peak) peak = value
          }
          result.push(peak)
        }

        const max = Math.max(...result, 0.01)
        setPeaks(result.map((p) => p / max))
      } catch {
        // Decoding is a nicety. If the browser cannot decode this codec, the transport
        // still works and a flat bar is drawn instead.
        if (!cancelled) setPeaks([])
      }
    }

    void analyse(playable.url)
    return () => {
      cancelled = true
    }
  }, [playable])

  // --- Transport ------------------------------------------------------------------------

  const toggle = useCallback(() => {
    const el = audioRef.current
    if (!el) return
    if (el.paused) void el.play()
    else el.pause()
  }, [])

  const seekTo = useCallback(
    (fraction: number) => {
      const el = audioRef.current
      if (!el || !duration) return
      el.currentTime = Math.max(0, Math.min(duration, fraction * duration))
    },
    [duration],
  )

  const onScrub = useCallback(
    (event: React.MouseEvent<HTMLDivElement>) => {
      const rect = event.currentTarget.getBoundingClientRect()
      seekTo((event.clientX - rect.left) / rect.width)
    },
    [seekTo],
  )

  useEffect(() => {
    const el = audioRef.current
    if (!el) return
    el.playbackRate = rate
  }, [rate])

  if (!playable) return null

  const download = assets.find((a) => a.format === 'wav') ?? playable
  const progress = duration > 0 ? current / duration : 0

  return (
    <div className="player">
      <audio
        ref={audioRef}
        src={playable.url}
        preload="metadata"
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
        onEnded={() => setPlaying(false)}
        onTimeUpdate={(e) => setCurrent(e.currentTarget.currentTime)}
        onLoadedMetadata={(e) => {
          const value = e.currentTarget.duration
          if (Number.isFinite(value)) setDuration(value)
        }}
      />

      <Button
        variant="primary"
        size="lg"
        className="player__toggle"
        onClick={toggle}
        aria-label={playing ? 'Pause' : 'Play'}
      >
        {playing ? <PauseIcon /> : <PlayIcon />}
      </Button>

      <div className="player__body">
        {/* Keyboard users get a real slider; the waveform is the pointer affordance. */}
        <div
          className="player__wave"
          onClick={onScrub}
          role="presentation"
          style={{ '--progress': progress } as React.CSSProperties}
        >
          {peaks === null ? (
            <div className="player__wave-loading" />
          ) : peaks.length === 0 ? (
            <div className="player__wave-flat" />
          ) : (
            peaks.map((peak, index) => (
              <span
                key={index}
                className={`player__bar ${index / peaks.length <= progress ? 'is-played' : ''}`}
                style={{ height: `${Math.max(6, peak * 100)}%` }}
              />
            ))
          )}
        </div>

        <input
          className="player__slider visually-hidden"
          type="range"
          min={0}
          max={1000}
          value={Math.round(progress * 1000)}
          onChange={(e) => seekTo(Number(e.currentTarget.value) / 1000)}
          aria-label="Seek"
        />

        <div className="player__meta">
          <span className="mono player__time">
            {formatDuration(current)} / {formatDuration(duration)}
          </span>

          <div className="player__actions">
            <div className="player__rates" role="group" aria-label="Playback speed">
              {[0.75, 1, 1.25, 1.5].map((value) => (
                <button
                  key={value}
                  type="button"
                  className={`player__rate mono ${rate === value ? 'is-active' : ''}`}
                  onClick={() => setRate(value)}
                  aria-pressed={rate === value}
                >
                  {value}×
                </button>
              ))}
            </div>

            <a
              className="player__download mono"
              href={download.url}
              download
              title={`${download.format.toUpperCase()} · ${formatBytes(download.size_bytes)}`}
            >
              ↓ {download.format.toUpperCase()}
            </a>
          </div>
        </div>
      </div>
    </div>
  )
}

function PlayIcon() {
  return (
    <svg width="16" height="18" viewBox="0 0 16 18" fill="currentColor" aria-hidden="true">
      <path d="M1 1.8v14.4a1 1 0 0 0 1.53.85l11.4-7.2a1 1 0 0 0 0-1.7L2.53.95A1 1 0 0 0 1 1.8Z" />
    </svg>
  )
}

function PauseIcon() {
  return (
    <svg width="14" height="18" viewBox="0 0 14 18" fill="currentColor" aria-hidden="true">
      <rect x="1" y="1" width="4" height="16" rx="1" />
      <rect x="9" y="1" width="4" height="16" rx="1" />
    </svg>
  )
}
