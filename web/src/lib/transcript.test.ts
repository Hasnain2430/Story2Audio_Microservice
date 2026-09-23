/**
 * The read-along is only as trustworthy as this file.
 *
 * The property that matters most is not "does it highlight" but "does it render the
 * story unchanged". A transcript that drops the quotation marks around dialogue, or
 * duplicates a word at a segment boundary, is a corrupted document — and it would look
 * plausible on screen while being wrong. So the first test reassembles the story from
 * the pieces and demands every non-whitespace character back, in order.
 */

import { describe, expect, it } from 'vitest'

import type { SpokenSegment } from '@/api/types'
import {
  buildTranscript,
  progressWithin,
  segmentAt,
  timeOfWord,
  voiceIndexes,
} from '@/lib/transcript'

function segment(partial: Partial<SpokenSegment> & Pick<SpokenSegment, 'start_char' | 'end_char'>): SpokenSegment {
  return {
    index: 0,
    kind: 'narration',
    text: '',
    start_seconds: 0,
    end_seconds: 1,
    ...partial,
  }
}

const STORY = 'She stopped at the door.\n\n"I am not leaving," she said. He nodded once.'

// The spans a real split produces: narration, the quoted line without its quotes, then
// the rest. Dialogue is a separate segment with its own voice.
const SEGMENTS: SpokenSegment[] = [
  segment({ index: 0, start_char: 0, end_char: 24, start_seconds: 0.1, end_seconds: 2.1 }),
  segment({
    index: 1,
    kind: 'dialogue',
    start_char: 27,
    end_char: 43,
    start_seconds: 2.2,
    end_seconds: 4.2,
  }),
  segment({ index: 2, start_char: 45, end_char: 74, start_seconds: 4.3, end_seconds: 6.3 }),
]

/** The rendered text, with paragraphs rejoined the way the DOM separates them. */
function reassemble(story: string, segments: SpokenSegment[]): string {
  const paragraphs = buildTranscript(story, segments)
  return paragraphs.map((p) => p.words.map((w) => w.text).join('')).join('\n\n')
}

/**
 * The invariant that actually matters, stated without whitespace.
 *
 * Paragraph breaks become structure rather than text, and trailing spaces ride along
 * with the word before them, so an exact string match would be testing formatting.
 * Every non-whitespace character, in order, is the real contract: nothing dropped at a
 * segment boundary, nothing duplicated by an overlapping span.
 */
function visibleCharacters(text: string): string {
  return text.replace(/\s/g, '')
}

describe('buildTranscript', () => {
  it('reproduces every character of the story, in order', () => {
    expect(visibleCharacters(reassemble(STORY, SEGMENTS))).toBe(visibleCharacters(STORY))
  })

  it('keeps text no segment covers, so the quotation marks survive', () => {
    const rendered = reassemble(STORY, SEGMENTS)
    expect(rendered).toContain('"I am not leaving,"')
  })

  it('splits paragraphs on blank lines', () => {
    const paragraphs = buildTranscript(STORY, SEGMENTS)
    expect(paragraphs).toHaveLength(2)
    expect(paragraphs[0]?.words.map((w) => w.text).join('')).toBe('She stopped at the door.')
  })

  it('renders the whole story untimed when there are no segments', () => {
    const paragraphs = buildTranscript(STORY, [])
    expect(paragraphs.flatMap((p) => p.words).every((w) => w.segment === -1)).toBe(true)
    expect(reassemble(STORY, [])).toContain('She stopped at the door.')
  })

  it('returns nothing for an empty story', () => {
    expect(buildTranscript('', SEGMENTS)).toEqual([])
  })

  it('assigns each word a position inside its segment, in order', () => {
    const words = buildTranscript(STORY, SEGMENTS)
      .flatMap((p) => p.words)
      .filter((w) => w.segment === 0)

    expect(words.length).toBeGreaterThan(1)
    expect(words[0]?.at).toBe(0)
    for (const word of words) {
      expect(word.at).toBeLessThan(word.until)
      expect(word.until).toBeLessThanOrEqual(1)
    }
  })

  it('survives spans that overlap or run past the end', () => {
    // These come from another service over JSON. Garbled input must degrade to a
    // readable story, never to duplicated or missing text.
    const broken: SpokenSegment[] = [
      segment({ index: 0, start_char: 0, end_char: 40 }),
      segment({ index: 1, start_char: 10, end_char: 9_999 }),
    ]
    expect(visibleCharacters(reassemble(STORY, broken))).toBe(visibleCharacters(STORY))
  })
})

describe('segmentAt', () => {
  it('finds the segment being spoken', () => {
    expect(segmentAt(SEGMENTS, 0.5)).toBe(0)
    expect(segmentAt(SEGMENTS, 3.0)).toBe(1)
    expect(segmentAt(SEGMENTS, 5.0)).toBe(2)
  })

  it('returns nothing during the lead-in and the pauses between segments', () => {
    // Silence is real time in which nothing is said. Highlighting through it would be a
    // lie, and it is what makes the gaps feel like breaths rather than like a stall.
    expect(segmentAt(SEGMENTS, 0.05)).toBe(-1)
    expect(segmentAt(SEGMENTS, 2.15)).toBe(-1)
    expect(segmentAt(SEGMENTS, 99)).toBe(-1)
  })

  it('has no segment to find in an empty timeline', () => {
    expect(segmentAt([], 1)).toBe(-1)
  })
})

describe('progressWithin', () => {
  it('runs 0 to 1 across the segment', () => {
    expect(progressWithin(SEGMENTS, 0, 0.1)).toBeCloseTo(0)
    expect(progressWithin(SEGMENTS, 0, 1.1)).toBeCloseTo(0.5)
    expect(progressWithin(SEGMENTS, 0, 2.1)).toBeCloseTo(1)
  })

  it('clamps rather than extrapolating outside the segment', () => {
    expect(progressWithin(SEGMENTS, 0, -5)).toBe(0)
    expect(progressWithin(SEGMENTS, 0, 500)).toBe(1)
  })
})

describe('timeOfWord', () => {
  it('maps a word back to a moment inside its segment', () => {
    const words = buildTranscript(STORY, SEGMENTS)
      .flatMap((p) => p.words)
      .filter((w) => w.segment === 2)

    const first = words[0]
    if (!first) throw new Error('the third segment rendered no words')

    const at = timeOfWord(SEGMENTS, first)
    expect(at).not.toBeNull()
    expect(at).toBeGreaterThanOrEqual(4.3)
    expect(at).toBeLessThan(6.3)
  })

  it('has no time for text outside every segment', () => {
    const untimed = buildTranscript(STORY, SEGMENTS)
      .flatMap((p) => p.words)
      .find((w) => w.segment === -1)

    if (!untimed) throw new Error('expected the quotation marks to be untimed')
    expect(timeOfWord(SEGMENTS, untimed)).toBeNull()
  })
})

describe('voiceIndexes', () => {
  it('assigns a slot per character, in order of first appearance', () => {
    const voices = voiceIndexes([
      segment({ index: 0, start_char: 0, end_char: 5 }),
      segment({ index: 1, kind: 'dialogue', speaker: 'Mara', start_char: 6, end_char: 10 }),
      segment({ index: 2, kind: 'dialogue', speaker: 'Lila', start_char: 11, end_char: 15 }),
      segment({ index: 3, kind: 'dialogue', speaker: 'Mara', start_char: 16, end_char: 20 }),
    ])

    expect(voices.get('Mara')).toBe(0)
    expect(voices.get('Lila')).toBe(1)
    expect(voices.size).toBe(2)
  })

  it('gives narration no slot at all', () => {
    // Narration is the bed the story sits on, not a character competing for a colour.
    const voices = voiceIndexes([segment({ index: 0, start_char: 0, end_char: 5 })])

    expect(voices.size).toBe(0)
  })

  it('is stable, so a character keeps one colour across a re-render', () => {
    const segments = [
      segment({ index: 0, kind: 'dialogue', speaker: 'Tom', start_char: 0, end_char: 4 }),
      segment({ index: 1, kind: 'dialogue', speaker: 'Lily', start_char: 5, end_char: 9 }),
    ]

    expect([...voiceIndexes(segments)]).toEqual([...voiceIndexes(segments)])
  })
})
