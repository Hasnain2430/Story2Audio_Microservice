/**
 * Voice picker — a station list rather than a dropdown.
 *
 * Choosing a voice is the one decision on the compose page that benefits from hearing
 * the options, so each row previews in place. A `<select>` would hide them behind a
 * click and make the preview impossible.
 */

import { useEffect, useRef, useState } from 'react'

import type { Voice } from '@/api/types'
import { formatDuration } from '@/lib/format'
import '@/features/voices/VoicePicker.css'

interface VoicePickerProps {
  label: string
  hint?: string
  voices: Voice[]
  loading: boolean
  selectedId: string | null
  onSelect: (id: string) => void
}

export function VoicePicker({
  label,
  hint,
  voices,
  loading,
  selectedId,
  onSelect,
}: VoicePickerProps) {
  const [previewing, setPreviewing] = useState<string | null>(null)
  const audioRef = useRef<HTMLAudioElement | null>(null)

  // One audio element for the whole list: starting a second preview must stop the
  // first, or the page ends up playing two voices at once.
  useEffect(() => {
    const el = (audioRef.current ??= new Audio())
    const stop = () => setPreviewing(null)
    el.addEventListener('ended', stop)
    el.addEventListener('pause', stop)
    return () => {
      el.removeEventListener('ended', stop)
      el.removeEventListener('pause', stop)
      el.pause()
    }
  }, [])

  function preview(voice: Voice) {
    const el = audioRef.current
    if (!el || !voice.preview_url) return

    if (previewing === voice.id) {
      el.pause()
      return
    }
    el.src = voice.preview_url
    void el.play().then(() => setPreviewing(voice.id))
  }

  return (
    <fieldset className="picker">
      <legend className="picker__legend eyebrow">{label}</legend>
      {hint && <p className="picker__hint">{hint}</p>}

      {loading ? (
        <div className="picker__loading mono">loading voices…</div>
      ) : voices.length === 0 ? (
        <div className="picker__loading mono">no voices available</div>
      ) : (
        <div className="picker__list" role="radiogroup" aria-label={label}>
          {voices.map((voice) => {
            const selected = voice.id === selectedId
            return (
              <div key={voice.id} className={`picker__row ${selected ? 'is-selected' : ''}`}>
                <label className="picker__choice">
                  <input
                    type="radio"
                    name={`voice-${label}`}
                    value={voice.id}
                    checked={selected}
                    onChange={() => onSelect(voice.id)}
                    className="visually-hidden"
                  />
                  <span className="picker__indicator" aria-hidden="true" />
                  <span className="picker__name">{voice.name}</span>
                  <span className="picker__meta mono">{formatDuration(voice.duration_seconds)}</span>
                </label>

                {voice.preview_url && (
                  <button
                    type="button"
                    className={`picker__preview ${previewing === voice.id ? 'is-playing' : ''}`}
                    onClick={() => preview(voice)}
                    aria-label={`${previewing === voice.id ? 'Stop' : 'Preview'} ${voice.name}`}
                    title="Preview"
                  >
                    {previewing === voice.id ? '■' : '▶'}
                  </button>
                )}
              </div>
            )
          })}
        </div>
      )}
    </fieldset>
  )
}
