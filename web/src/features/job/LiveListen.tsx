/**
 * Listening to a story that is still being recorded.
 *
 * The control that makes the architecture audible. The worker publishes each segment as
 * it finishes, so the opening of a story exists about eight seconds into a job that runs
 * for a minute — this is the button that lets someone start there instead of waiting.
 *
 * It states what it knows rather than implying more: how much is ready, and when it is
 * waiting for the renderer to catch up. A player that silently stalls looks broken; one
 * that says "waiting for the next part" is obviously doing the thing it was asked to do.
 */

import type { LivePlayback } from '@/hooks/useLivePlayback'
import { formatDuration } from '@/lib/format'
import { Button } from '@/components/primitives'
import '@/features/job/LiveListen.css'

interface LiveListenProps {
  playback: LivePlayback
  /** Segments rendered so far; the control is pointless before the first one. */
  ready: number
  total: number
}

export function LiveListen({ playback, ready, total }: LiveListenProps) {
  if (!playback.supported || ready === 0) return null

  return (
    <div className="live">
      <Button
        variant="primary"
        size="lg"
        className="live__toggle"
        onClick={() => (playback.playing ? playback.pause() : playback.play())}
        aria-label={playback.playing ? 'Pause' : 'Listen while it records'}
      >
        {playback.playing ? <PauseIcon /> : <PlayIcon />}
      </Button>

      <div className="live__body">
        <p className="live__label">
          {playback.playing ? (
            playback.starved ? (
              <>
                <span className="live__pulse" aria-hidden="true" />
                Waiting for the next part…
              </>
            ) : (
              <>
                <span className="live__pulse is-on" aria-hidden="true" />
                Listening while it records
              </>
            )
          ) : (
            'Start listening — the rest arrives as it is made'
          )}
        </p>

        <p className="live__meta mono">
          {formatDuration(playback.currentTime)} · {ready} of {total} parts ready ·{' '}
          {formatDuration(playback.bufferedUntil)} buffered
        </p>
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
