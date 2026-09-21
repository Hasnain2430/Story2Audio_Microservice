/**
 * The story, read along with.
 *
 * The line being spoken lifts out of the page and the words inside it fill as they are
 * said. Click any word to jump the audio there — the transcript is a control surface,
 * not a caption track, which is the thing that makes a seven minute recording navigable.
 *
 * Three honesty constraints shape it:
 *
 *  - Without timings — an older job, or one still being written — this renders the plain
 *    story and nothing else changes. The highlight is an enhancement, never a
 *    requirement.
 *  - In the gaps between segments nothing is highlighted, because in those moments
 *    nothing is being said.
 *  - Word positions inside a segment are interpolated from character offsets (see
 *    `lib/transcript`), so the fill is smooth rather than word-perfect. It is presented
 *    as following along, and never as a claim of forced alignment.
 */

import { useEffect, useMemo, useRef } from 'react'

import type { SpokenSegment } from '@/api/types'
import {
  buildTranscript,
  progressWithin,
  segmentAt,
  timeOfWord,
  type Word,
} from '@/lib/transcript'
import '@/features/job/Transcript.css'

interface TranscriptProps {
  story: string
  segments: SpokenSegment[]
  currentTime: number
  playing: boolean
  onSeek: (seconds: number) => void
  /** True while tokens are still arriving, which is the one time a caret means something. */
  writing?: boolean
}

export function Transcript({
  story,
  segments,
  currentTime,
  playing,
  onSeek,
  writing = false,
}: TranscriptProps) {
  // Rebuilt only when the text or the timings change. This component re-renders on
  // every animation frame while audio plays, and the parse walks the whole story.
  const paragraphs = useMemo(() => buildTranscript(story, segments), [story, segments])

  const active = segments.length > 0 ? segmentAt(segments, currentTime) : -1
  const progress = active >= 0 ? progressWithin(segments, active, currentTime) : 0

  // Scroll to the *start* of the line being read. Anchoring every word in it would leave
  // the ref pointing at whichever rendered last, which is the end of the line — so the
  // page would scroll the line's final word to centre and push its opening off the top.
  const anchorOffset = useMemo(() => {
    if (active < 0) return -1
    for (const paragraph of paragraphs) {
      const first = paragraph.words.find((word) => word.segment === active)
      if (first) return first.offset
    }
    return -1
  }, [paragraphs, active])

  const activeRef = useRef<HTMLSpanElement | null>(null)
  const lastScrolled = useRef(-1)

  useEffect(() => {
    // Follow the narration, but only while it is actually playing: yanking the page
    // around while someone is reading ahead or scrubbing is hostile.
    if (!playing || active < 0 || active === lastScrolled.current) return
    lastScrolled.current = active

    const element = activeRef.current
    if (!element) return

    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    element.scrollIntoView({ behavior: reduced ? 'auto' : 'smooth', block: 'center' })
  }, [active, playing])

  const timed = segments.length > 0

  return (
    <div className={`transcript ${timed ? 'transcript--timed' : ''}`}>
      {paragraphs.map((paragraph) => (
        <p key={paragraph.offset} className="transcript__p">
          {paragraph.words.map((word) => (
            <WordSpan
              key={word.offset}
              word={word}
              active={active}
              progress={progress}
              segments={segments}
              onSeek={onSeek}
              anchorRef={word.offset === anchorOffset ? activeRef : undefined}
            />
          ))}
        </p>
      ))}

      {writing && <span className="transcript__caret" aria-hidden="true" />}
    </div>
  )
}

interface WordSpanProps {
  word: Word
  active: number
  progress: number
  segments: SpokenSegment[]
  onSeek: (seconds: number) => void
  anchorRef?: React.RefObject<HTMLSpanElement | null> | undefined
}

function WordSpan({ word, active, progress, segments, onSeek, anchorRef }: WordSpanProps) {
  const inActiveSegment = word.segment >= 0 && word.segment === active
  const spoken = word.segment >= 0 && (word.segment < active || (inActiveSegment && word.until <= progress))
  const speaking = inActiveSegment && word.at <= progress && progress < word.until

  const className = [
    'w',
    word.segment >= 0 ? 'w--timed' : '',
    spoken ? 'is-spoken' : '',
    speaking ? 'is-speaking' : '',
    inActiveSegment ? 'is-current-line' : '',
  ]
    .filter(Boolean)
    .join(' ')

  // Untimed text — quotation marks, the spacing around them — is not interactive: there
  // is no moment in the audio to send someone to.
  if (word.segment < 0) {
    return <span className="w">{word.text}</span>
  }

  const seconds = timeOfWord(segments, word)

  return (
    <span
      ref={anchorRef ?? undefined}
      className={className}
      role="button"
      tabIndex={0}
      onClick={() => {
        if (seconds !== null) onSeek(seconds)
      }}
      onKeyDown={(event) => {
        if (seconds === null) return
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault()
          onSeek(seconds)
        }
      }}
    >
      {word.text}
    </span>
  )
}
