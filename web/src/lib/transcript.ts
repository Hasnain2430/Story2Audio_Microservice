/**
 * Turning a story and its segment timings into something that can be read along with.
 *
 * The API gives two coordinate systems that have to be reconciled here. `story_text` is
 * the written story, paragraphs and punctuation intact. Each segment carries the span of
 * that text it covers (`start_char`/`end_char`) and when it is spoken
 * (`start_seconds`/`end_seconds`). Segments do not cover everything: the quotation marks
 * around dialogue sit between two spans and are never spoken, and a story finished before
 * timings existed has no spans at all.
 *
 * So the story is cut into pieces — timed and untimed — and then into paragraphs, with
 * every piece keeping its absolute offset. Nothing is reordered and no character is
 * invented: concatenating every piece reproduces `story_text` exactly, which is the
 * property that keeps a highlight from sliding off the word it belongs to.
 *
 * **Word timing inside a segment is interpolated, and that is worth being plain about.**
 * The segment boundaries are measured — the worker rendered each one and knows its exact
 * duration — but XTTS returns no per-word alignment, so a word's position within its
 * segment is estimated from where its characters fall. Even speech makes that nearly
 * right; a long pause mid-sentence will drift. It reads as "following along", not as a
 * forced alignment, and no part of the app depends on it being exact.
 */

import type { SpokenSegment } from '@/api/types'

export interface Word {
  text: string
  /** Absolute offset in the story, so React keys stay stable across re-renders. */
  offset: number
  /** Index into the segment list, or -1 for text no segment covers. */
  segment: number
  /** Where this word starts within its segment, 0–1. -1 when untimed. */
  at: number
  /** Where it ends within its segment, 0–1. -1 when untimed. */
  until: number
}

export interface Paragraph {
  offset: number
  words: Word[]
}

/** Runs of two or more newlines separate paragraphs, matching how the story is written. */
const PARAGRAPH_BREAK = /\n{2,}/g

interface Piece {
  start: number
  end: number
  segment: number
}

/**
 * Cut the story into paragraphs of timed words.
 *
 * Returns an empty list for an empty story. A story with no segments still comes back
 * fully rendered, every word untimed — the transcript is then just the story, which is
 * exactly what it should degrade to.
 */
export function buildTranscript(story: string, segments: SpokenSegment[]): Paragraph[] {
  if (!story) return []

  const pieces = cut(story, segments)
  const paragraphs: Paragraph[] = []
  let current: Word[] = []
  let currentOffset = 0

  const flush = () => {
    if (current.length > 0) {
      paragraphs.push({ offset: currentOffset, words: current })
      current = []
    }
  }

  for (const piece of pieces) {
    const text = story.slice(piece.start, piece.end)
    PARAGRAPH_BREAK.lastIndex = 0

    let consumed = 0
    let match: RegExpExecArray | null

    while ((match = PARAGRAPH_BREAK.exec(text)) !== null) {
      addWords(current, text.slice(consumed, match.index), piece, story, segments)
      if (current.length === 0 && paragraphs.length === 0) currentOffset = piece.start
      flush()
      currentOffset = piece.start + match.index + match[0].length
      consumed = match.index + match[0].length
    }

    if (current.length === 0) currentOffset = piece.start + consumed
    addWords(current, text.slice(consumed), piece, story, segments)
  }

  flush()
  return paragraphs
}

/**
 * Split the story at every segment boundary.
 *
 * Gaps between segments become untimed pieces rather than being dropped, so the quotation
 * marks and spacing around dialogue survive into the rendered story.
 */
function cut(story: string, segments: SpokenSegment[]): Piece[] {
  const ordered = [...segments].sort((a, b) => a.start_char - b.start_char)
  const pieces: Piece[] = []
  let cursor = 0

  ordered.forEach((segment, index) => {
    // Defensive: the spans come from another service. Overlapping or out-of-range spans
    // would otherwise duplicate or drop text, and a garbled story is a worse failure
    // than an unhighlighted one.
    const start = Math.max(cursor, Math.min(segment.start_char, story.length))
    const end = Math.max(start, Math.min(segment.end_char, story.length))
    if (start > cursor) pieces.push({ start: cursor, end: start, segment: -1 })
    if (end > start) pieces.push({ start, end, segment: index })
    cursor = end
  })

  if (cursor < story.length) pieces.push({ start: cursor, end: story.length, segment: -1 })
  return pieces
}

/** Whitespace is kept with the word that precedes it, so spacing survives rendering. */
const WORD = /\S+\s*/g

function addWords(
  into: Word[],
  text: string,
  piece: Piece,
  story: string,
  segments: SpokenSegment[],
): void {
  if (!text) return

  const segment = piece.segment >= 0 ? segments[piece.segment] : undefined
  const span = segment ? Math.max(1, segment.end_char - segment.start_char) : 1
  const base = story.slice(piece.start, piece.end).indexOf(text)
  const origin = piece.start + (base >= 0 ? base : 0)

  WORD.lastIndex = 0
  let match: RegExpExecArray | null

  while ((match = WORD.exec(text)) !== null) {
    const offset = origin + match.index
    const trimmed = match[0].trimEnd()

    into.push({
      text: match[0],
      offset,
      segment: piece.segment,
      at: segment ? clamp((offset - segment.start_char) / span) : -1,
      until: segment ? clamp((offset + trimmed.length - segment.start_char) / span) : -1,
    })
  }
}

function clamp(value: number): number {
  return Math.max(0, Math.min(1, value))
}

/**
 * Which segment is being spoken at `seconds`, or -1.
 *
 * Binary search because this runs on every animation frame; a linear scan over a few
 * hundred segments is work the browser does not need to repeat sixty times a second.
 * Returns -1 in the gaps — the lead-in silence and the pauses between segments are real
 * time in which nothing is being said, and highlighting through them would be a lie.
 */
export function segmentAt(segments: SpokenSegment[], seconds: number): number {
  let low = 0
  let high = segments.length - 1

  while (low <= high) {
    const middle = (low + high) >> 1
    const segment = segments[middle]
    if (!segment) break
    if (seconds < segment.start_seconds) high = middle - 1
    else if (seconds >= segment.end_seconds) low = middle + 1
    else return middle
  }

  return -1
}

/** How far through the segment at `index` playback is, 0–1. */
export function progressWithin(
  segments: SpokenSegment[],
  index: number,
  seconds: number,
): number {
  const segment = segments[index]
  if (!segment) return 0
  const span = segment.end_seconds - segment.start_seconds
  if (span <= 0) return 0
  return clamp((seconds - segment.start_seconds) / span)
}

/** The time a word begins, for seeking when one is clicked. */
export function timeOfWord(segments: SpokenSegment[], word: Word): number | null {
  const segment = segments[word.segment]
  if (!segment || word.at < 0) return null
  return segment.start_seconds + word.at * (segment.end_seconds - segment.start_seconds)
}
