/**
 * The tape: a story's segments laid out as they are recorded.
 *
 * One block per segment, in order, each as wide as it is long, coloured by who is
 * speaking. Blocks appear as the worker finishes them, so the strip visibly fills from
 * the left while the story is still being written — and because narration and each
 * character have their own colour, the shape of a scene is legible before a word is
 * heard: a run of long narration blocks, then a rally of short alternating ones where
 * the two characters argue.
 *
 * It is also the transport. The playhead rides across it and clicking anywhere seeks,
 * which is the natural gesture for something laid out in time. A progress bar could show
 * the same completion percentage and none of the same information.
 *
 * Widths come from the measured timeline, so a block's size on screen is the segment's
 * true share of the story.
 *
 * The strip's *total* is the one estimate here, and it has to be. Scaling to the segments
 * that exist would draw two finished parts of twenty-three as a full strip, which reads
 * as "done" — the opposite of what is happening. So while rendering is in progress the
 * span is projected from the average segment so far across the expected count, and the
 * strip fills gradually. The estimate is replaced by the real duration the moment the
 * job finishes, and no number derived from it is ever shown.
 */

import { useMemo } from 'react'

import type { SpokenSegment } from '@/api/types'
import { formatDuration } from '@/lib/format'
import { voiceIndexes } from '@/lib/transcript'
import '@/features/job/Tape.css'

interface TapeProps {
  segments: SpokenSegment[]
  currentTime: number
  /** Total length once known; until then the strip projects one. */
  duration?: number | undefined
  /** Segments the job expects in total, used to project the strip's width. */
  expected?: number | undefined
  /** How far the audio has actually been fetched, for the buffered shading. */
  bufferedUntil?: number | undefined
  onSeek?: ((seconds: number) => void) | undefined
}

export function Tape({
  segments,
  currentTime,
  duration,
  expected,
  bufferedUntil,
  onSeek,
}: TapeProps) {
  const span = useMemo(() => {
    const rendered = segments.at(-1)?.end_seconds ?? 0
    if (duration) return Math.max(duration, rendered, 0.001)

    // Still rendering: project the finished length from what has been made so far, so the
    // strip fills as the story is recorded rather than always looking complete.
    const projected =
      expected && segments.length > 0 ? (rendered / segments.length) * expected : rendered
    return Math.max(projected, rendered, 0.001)
  }, [segments, duration, expected])

  // Shared with the transcript, so a character is one colour everywhere on the page.
  const voices = useMemo(() => voiceIndexes(segments), [segments])

  if (segments.length === 0) return null

  const playhead = Math.min(1, currentTime / span)

  return (
    <div className="tape">
      <div
        className="tape__strip"
        role="presentation"
        onClick={(occurrence) => {
          if (!onSeek) return
          const box = occurrence.currentTarget.getBoundingClientRect()
          onSeek(((occurrence.clientX - box.left) / box.width) * span)
        }}
      >
        {bufferedUntil !== undefined && (
          <span
            className="tape__buffered"
            style={{ width: `${Math.min(100, (bufferedUntil / span) * 100)}%` }}
            aria-hidden="true"
          />
        )}

        {segments.map((segment) => {
          const voice = segment.speaker ? voices.get(segment.speaker) : undefined
          const active = currentTime >= segment.start_seconds && currentTime < segment.end_seconds

          return (
            <span
              key={segment.index}
              className={`tape__block ${active ? 'is-active' : ''}`}
              data-voice={voice === undefined ? 'narration' : voice % 4}
              style={{
                left: `${(segment.start_seconds / span) * 100}%`,
                width: `${((segment.end_seconds - segment.start_seconds) / span) * 100}%`,
              }}
              title={`${segment.speaker ?? 'Narration'} · ${formatDuration(
                segment.end_seconds - segment.start_seconds,
              )}`}
            />
          )
        })}

        <span className="tape__playhead" style={{ left: `${playhead * 100}%` }} aria-hidden="true" />
      </div>

      <div className="tape__legend">
        <span className="tape__key" data-voice="narration">
          Narration
        </span>
        {[...voices.entries()].map(([name, index]) => (
          <span key={name} className="tape__key" data-voice={index % 4}>
            {name}
          </span>
        ))}
        <span className="tape__count mono">
          {segments.length} {segments.length === 1 ? 'part' : 'parts'}
        </span>
      </div>
    </div>
  )
}
