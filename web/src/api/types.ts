/**
 * Domain types, taken from the generated OpenAPI schema.
 *
 * Nothing here is hand-written structure — these are aliases onto `schema.d.ts`, which
 * is generated from `openapi.json`, which is dumped from the FastAPI app itself. A field
 * that changes shape on the server becomes a type error here rather than a runtime
 * surprise. That chain is the whole point; do not hand-roll an interface to "fix" a
 * mismatch.
 */

import type { components } from '@/api/schema'

type Schemas = components['schemas']

export type JobStatus = Schemas['JobStatus']
export type StoryLength = Schemas['StoryLength']
export type VoiceMode = Schemas['VoiceMode']
export type Emotion = Schemas['Emotion']
export type Language = Schemas['Language']
export type AudioFormat = Schemas['AudioFormat']
export type ErrorCode = Schemas['ErrorCode']

export type Job = Schemas['JobResponse']
export type JobTimings = Schemas['JobTimings']
export type AudioAsset = Schemas['AudioAsset']
export type ErrorDetail = Schemas['ErrorDetail']
export type SpokenSegment = Schemas['SpokenSegment']
export type SegmentKind = Schemas['SegmentKind']
export type Voice = Schemas['VoiceResponse']
export type CreateJobRequest = Schemas['CreateJobRequest']
export type CreateJobResponse = Schemas['CreateJobResponse']

export type JobPage = Schemas['Page_JobResponse_']
export type VoicePage = Schemas['Page_VoiceResponse_']

/** Statuses from which no further transition is possible. Mirrors `enums.TERMINAL_STATUSES`. */
export const TERMINAL_STATUSES = ['done', 'failed', 'cancelled'] as const satisfies readonly JobStatus[]

export function isTerminal(status: JobStatus): boolean {
  return (TERMINAL_STATUSES as readonly JobStatus[]).includes(status)
}

/** Ordered lifecycle, used to render progress through the pipeline. */
export const PIPELINE_STAGES = ['queued', 'writing', 'written', 'synthesizing', 'done'] as const

export const STORY_LENGTHS: readonly { value: StoryLength; label: string; hint: string }[] = [
  { value: 'short', label: 'Short', hint: '300–400 words' },
  { value: 'medium', label: 'Medium', hint: '500–700 words' },
  { value: 'long', label: 'Long', hint: '800–1200 words' },
]

export const EMOTIONS: readonly { value: Emotion; label: string }[] = [
  { value: 'neutral', label: 'Even' },
  { value: 'happy', label: 'Warm' },
  { value: 'sad', label: 'Sombre' },
  { value: 'angry', label: 'Tense' },
]

export const LANGUAGES: readonly { value: Language; label: string }[] = [
  { value: 'en', label: 'English' },
  { value: 'es', label: 'Spanish' },
  { value: 'fr', label: 'French' },
  { value: 'de', label: 'German' },
  { value: 'it', label: 'Italian' },
  { value: 'ru', label: 'Russian' },
  { value: 'hi', label: 'Hindi' },
]

export const VOICE_MODES: readonly { value: VoiceMode; label: string; hint: string }[] = [
  { value: 'narration', label: 'Narration', hint: 'One voice throughout' },
  {
    value: 'narration_with_dialogue',
    label: 'Two voices',
    hint: 'A second voice reads spoken lines',
  },
]
