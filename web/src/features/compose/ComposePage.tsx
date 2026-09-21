/**
 * Compose: the page where a story is commissioned.
 *
 * Asymmetric on purpose. The prompt is the only thing that genuinely matters, so it
 * takes the full editorial column; everything else is a settings rail that never
 * competes with it. The submit button leads straight to `/jobs/:id` — the job has its
 * own URL from the instant it exists, which is the whole architectural point made
 * visible.
 */

import { useState, type SyntheticEvent } from 'react'
import { useNavigate } from 'react-router'

import { ApiError } from '@/api/client'
import {
  EMOTIONS,
  LANGUAGES,
  STORY_LENGTHS,
  VOICE_MODES,
  type Emotion,
  type Language,
  type StoryLength,
  type VoiceMode,
} from '@/api/types'
import { useCreateJob } from '@/hooks/useCreateJob'
import { useVoices } from '@/hooks/useVoices'
import { Banner, Button, Field, Segmented } from '@/components/primitives'
import { VoicePicker } from '@/features/voices/VoicePicker'
import '@/features/compose/ComposePage.css'

const MAX_PROMPT = 2000

const EXAMPLES = [
  'A lighthouse keeper finds the lamp cold on a night with no moon.',
  'Two sisters argue on a pier about whether to sell their father’s boat.',
  'A cartographer discovers a road that appears on no map.',
  'The night watchman hears the docks creak in a language he almost knows.',
]

export function ComposePage() {
  const navigate = useNavigate()
  const voices = useVoices()
  const createJob = useCreateJob()

  const [prompt, setPrompt] = useState('')
  const [length, setLength] = useState<StoryLength>('short')
  const [mode, setMode] = useState<VoiceMode>('narration')
  const [emotion, setEmotion] = useState<Emotion>('neutral')
  const [language, setLanguage] = useState<Language>('en')
  const [speed, setSpeed] = useState(1)
  const [voiceId, setVoiceId] = useState<string | null>(null)
  const [dialogueVoiceId, setDialogueVoiceId] = useState<string | null>(null)

  const available = voices.data ?? []
  const narrator = voiceId ?? available[0]?.id ?? null
  const needsDialogueVoice = mode === 'narration_with_dialogue'
  const dialogue =
    dialogueVoiceId ?? available.find((voice) => voice.id !== narrator)?.id ?? null

  const error = createJob.error instanceof ApiError ? createJob.error : null
  const ready = prompt.trim().length > 0 && narrator !== null && (!needsDialogueVoice || dialogue)

  async function onSubmit(event: SyntheticEvent) {
    event.preventDefault()
    if (!ready || !narrator) return

    const job = await createJob.mutateAsync({
      prompt: prompt.trim(),
      length,
      mode,
      emotion,
      language,
      speed,
      voice_id: narrator,
      // The API rejects a dialogue voice in narration mode outright, so it is only sent
      // when the mode actually calls for one.
      dialogue_voice_id: needsDialogueVoice ? dialogue : null,
    })

    void navigate(`/jobs/${job.id}`)
  }

  return (
    <form
      className="compose"
      onSubmit={(event) => {
        void onSubmit(event)
      }}
    >
      <header className="compose__head rise" style={{ '--i': 0 } as React.CSSProperties}>
        <p className="eyebrow">New recording</p>
        <h1 className="compose__title">
          Give it a premise.
          <br />
          <span className="compose__title-accent">It writes and reads the rest.</span>
        </h1>
      </header>

      <div className="compose__grid">
        <div className="compose__main rise" style={{ '--i': 1 } as React.CSSProperties}>
          <div className="prompt">
            <label className="visually-hidden" htmlFor="prompt">
              Storyline
            </label>
            <textarea
              id="prompt"
              className="prompt__input"
              value={prompt}
              onChange={(e) => setPrompt(e.currentTarget.value.slice(0, MAX_PROMPT))}
              placeholder="A lighthouse keeper finds the lamp cold on a night with no moon…"
              rows={5}
              spellCheck
            />
            <div className="prompt__footer">
              <span className="prompt__count mono">
                {prompt.length}/{MAX_PROMPT}
              </span>
            </div>
          </div>

          <div className="examples">
            <span className="eyebrow">Try</span>
            <div className="examples__list">
              {EXAMPLES.map((example) => (
                <button
                  key={example}
                  type="button"
                  className="examples__item"
                  onClick={() => setPrompt(example)}
                >
                  {example}
                </button>
              ))}
            </div>
          </div>

          <div className="compose__voices">
            <VoicePicker
              label={needsDialogueVoice ? 'Narrator' : 'Voice'}
              voices={available}
              loading={voices.isLoading}
              selectedId={narrator}
              onSelect={setVoiceId}
            />

            {needsDialogueVoice && (
              <VoicePicker
                label="Spoken lines"
                hint="Used for dialogue inside quotation marks"
                voices={available.filter((voice) => voice.id !== narrator)}
                loading={voices.isLoading}
                selectedId={dialogue}
                onSelect={setDialogueVoiceId}
              />
            )}
          </div>
        </div>

        <aside className="compose__rail rise" style={{ '--i': 2 } as React.CSSProperties}>
          <Field label="Length">
            <Segmented name="Length" value={length} options={STORY_LENGTHS} onChange={setLength} />
          </Field>

          <Field label="Voices">
            <Segmented name="Voices" value={mode} options={VOICE_MODES} onChange={setMode} />
          </Field>

          <Field label="Register" hint="Steers the writing, not a filter on the audio">
            <Segmented name="Register" value={emotion} options={EMOTIONS} onChange={setEmotion} />
          </Field>

          <Field label="Language" htmlFor="language">
            <select
              id="language"
              className="select"
              value={language}
              onChange={(e) => setLanguage(e.currentTarget.value as Language)}
            >
              {LANGUAGES.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </Field>

          <Field label="Pace" htmlFor="speed">
            <div className="speed">
              <input
                id="speed"
                className="speed__input"
                type="range"
                min={0.5}
                max={1.5}
                step={0.05}
                value={speed}
                onChange={(e) => setSpeed(Number(e.currentTarget.value))}
              />
              <output className="speed__value mono" htmlFor="speed">
                {speed.toFixed(2)}×
              </output>
            </div>
          </Field>

          {error && <Banner tone="error">{error.message}</Banner>}

          <Button
            type="submit"
            variant="primary"
            size="lg"
            className="compose__submit"
            busy={createJob.isPending}
            disabled={!ready}
          >
            {createJob.isPending ? 'Sending' : 'Start recording'}
          </Button>

          <p className="compose__note">
            Returns immediately with a link. Close the tab and come back — the recording
            keeps going.
          </p>
        </aside>
      </div>
    </form>
  )
}
