/**
 * Voice library: the built-in pack, plus whatever this session has added.
 *
 * Uploads are validated **before** they leave the browser. v1 enforced its "minimum 15
 * seconds" as `len(bytes) < 15000` — about a sixth of a second — and the user found out
 * server-side, if at all. Decoding here means the duration shown is the real one and a
 * too-short clip is rejected instantly.
 */

import { useRef, useState } from 'react'

import { ApiError } from '@/api/client'
import type { Voice } from '@/api/types'
import { useDeleteVoice, useUploadVoice, useVoices } from '@/hooks/useVoices'
import { decodeDuration } from '@/lib/audio'
import { formatDuration } from '@/lib/format'
import { Banner, Button, Card, Empty, Field } from '@/components/primitives'
import { VoiceRecorder } from '@/features/voices/VoiceRecorder'
import '@/features/voices/VoicesPage.css'

const MIN_SECONDS = 6
const MAX_SECONDS = 120

export function VoicesPage() {
  const voices = useVoices()
  const upload = useUploadVoice()
  const remove = useDeleteVoice()

  const fileRef = useRef<HTMLInputElement | null>(null)
  const [name, setName] = useState('')
  const [file, setFile] = useState<File | null>(null)
  const [duration, setDuration] = useState<number | null>(null)
  const [localError, setLocalError] = useState<string | null>(null)

  const all = voices.data ?? []
  const builtIn = all.filter((voice) => voice.is_builtin)
  const mine = all.filter((voice) => !voice.is_builtin)

  const serverError = upload.error instanceof ApiError ? upload.error.message : null

  async function onPick(picked: File | null) {
    setFile(picked)
    setDuration(null)
    setLocalError(null)
    if (!picked) return

    if (!name) setName(picked.name.replace(/\.[^.]+$/, ''))

    const seconds = await decodeDuration(picked)
    if (seconds === null) {
      setLocalError('That file could not be read as audio.')
      return
    }
    setDuration(seconds)
    if (seconds < MIN_SECONDS) {
      setLocalError(
        `That clip is ${formatDuration(seconds)}. Cloning needs at least ${MIN_SECONDS} seconds.`,
      )
    } else if (seconds > MAX_SECONDS) {
      setLocalError(`That clip is longer than ${MAX_SECONDS} seconds.`)
    }
  }

  async function submit() {
    if (!file || !name.trim() || localError) return
    await upload.mutateAsync({ name: name.trim(), file, filename: file.name })
    setFile(null)
    setName('')
    setDuration(null)
    if (fileRef.current) fileRef.current.value = ''
  }

  function onRecorded(blob: Blob, seconds: number) {
    const recorded = new File([blob], 'recording.webm', { type: blob.type })
    setFile(recorded)
    setDuration(seconds)
    setLocalError(
      seconds < MIN_SECONDS
        ? `That take is ${formatDuration(seconds)}. Cloning needs at least ${MIN_SECONDS} seconds.`
        : null,
    )
    if (!name) setName('My voice')
  }

  return (
    <div className="voices">
      <header className="voices__head">
        <p className="eyebrow">Library</p>
        <h1 className="voices__title">Voices</h1>
        <p className="voices__lede">
          Every recording is read in one of these. Add your own with a clip of at least{' '}
          {MIN_SECONDS} seconds — clear speech, little background noise.
        </p>
      </header>

      <div className="voices__grid">
        <Card className="add">
          <h2 className="add__title">Add a voice</h2>

          <Field label="Name" htmlFor="voice-name">
            <input
              id="voice-name"
              className="input"
              value={name}
              onChange={(e) => setName(e.currentTarget.value.slice(0, 60))}
              placeholder="Narrator"
            />
          </Field>

          <Field
            label="Sample"
            htmlFor="voice-file"
            hint={
              duration != null
                ? `${formatDuration(duration)} — stored as the first 20 seconds, mono`
                : 'WAV, MP3, FLAC or OGG'
            }
          >
            <input
              id="voice-file"
              ref={fileRef}
              className="input"
              type="file"
              accept="audio/*"
              onChange={(e) => void onPick(e.currentTarget.files?.[0] ?? null)}
            />
          </Field>

          <VoiceRecorder onRecorded={onRecorded} />

          {(localError ?? serverError) && (
            <Banner tone="error">{localError ?? serverError}</Banner>
          )}

          <Button
            variant="primary"
            onClick={() => void submit()}
            busy={upload.isPending}
            disabled={!file || !name.trim() || Boolean(localError)}
          >
            Add to library
          </Button>
        </Card>

        <div className="voices__lists">
          {mine.length > 0 && (
            <VoiceList
              title="Yours"
              voices={mine}
              onDelete={(id) => remove.mutate(id)}
              deleting={remove.isPending}
            />
          )}
          <VoiceList title="Built in" voices={builtIn} />
        </div>
      </div>
    </div>
  )
}

function VoiceList({
  title,
  voices,
  onDelete,
  deleting,
}: {
  title: string
  voices: Voice[]
  onDelete?: (id: string) => void
  deleting?: boolean
}) {
  if (voices.length === 0) {
    return <Empty title={`No ${title.toLowerCase()} voices`} />
  }

  return (
    <section className="voice-list">
      <h2 className="eyebrow">{title}</h2>
      <ul className="voice-list__items">
        {voices.map((voice) => (
          <li key={voice.id} className="voice-card">
            <div className="voice-card__body">
              <span className="voice-card__name">{voice.name}</span>
              <span className="voice-card__meta mono">
                {formatDuration(voice.duration_seconds)} · {(voice.sample_rate / 1000).toFixed(1)} kHz
              </span>
            </div>

            {voice.preview_url && (
              // The native element is right here: this is a bare preview, not a
              // transport, and `controls` gives keyboard and screen-reader support free.
              <audio className="voice-card__audio" src={voice.preview_url} controls preload="none" />
            )}

            {onDelete && (
              <Button
                variant="quiet"
                size="sm"
                onClick={() => onDelete(voice.id)}
                disabled={deleting}
                aria-label={`Delete ${voice.name}`}
              >
                Remove
              </Button>
            )}
          </li>
        ))}
      </ul>
    </section>
  )
}
