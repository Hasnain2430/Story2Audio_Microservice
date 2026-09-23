/**
 * In-browser recorder.
 *
 * `MediaRecorder` plus a live level meter, so the user can see the microphone is
 * actually picking them up before committing thirty seconds to it. v1 offered a
 * recorder with no feedback at all and a length check that did not work.
 */

import { useCallback, useEffect, useRef, useState } from 'react'

import { Button } from '@/components/primitives'
import { formatDuration } from '@/lib/format'
import '@/features/voices/VoiceRecorder.css'

const METER_SEGMENTS = 28

interface VoiceRecorderProps {
  onRecorded: (blob: Blob, seconds: number) => void
}

export function VoiceRecorder({ onRecorded }: VoiceRecorderProps) {
  const [recording, setRecording] = useState(false)
  const [elapsed, setElapsed] = useState(0)
  const [level, setLevel] = useState(0)
  const [error, setError] = useState<string | null>(null)

  const recorderRef = useRef<MediaRecorder | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const rafRef = useRef<number | null>(null)
  const startedRef = useRef(0)

  const cleanup = useCallback(() => {
    if (rafRef.current !== null) cancelAnimationFrame(rafRef.current)
    rafRef.current = null
    streamRef.current?.getTracks().forEach((track) => track.stop())
    streamRef.current = null
    setLevel(0)
  }, [])

  // Releasing the microphone on unmount is not optional: leaving the track open keeps
  // the browser's recording indicator lit after the user has navigated away.
  useEffect(() => cleanup, [cleanup])

  async function start() {
    setError(null)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      streamRef.current = stream

      const context = new AudioContext()
      const source = context.createMediaStreamSource(stream)
      const analyser = context.createAnalyser()
      analyser.fftSize = 512
      source.connect(analyser)

      const data = new Uint8Array(analyser.frequencyBinCount)
      const tick = () => {
        analyser.getByteTimeDomainData(data)
        let peak = 0
        for (const sample of data) {
          const value = Math.abs(sample - 128) / 128
          if (value > peak) peak = value
        }
        setLevel(peak)
        setElapsed((Date.now() - startedRef.current) / 1000)
        rafRef.current = requestAnimationFrame(tick)
      }

      const chunks: Blob[] = []
      const recorder = new MediaRecorder(stream)
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) chunks.push(event.data)
      }
      recorder.onstop = () => {
        const seconds = (Date.now() - startedRef.current) / 1000
        void context.close()
        cleanup()
        onRecorded(new Blob(chunks, { type: recorder.mimeType }), seconds)
      }

      recorderRef.current = recorder
      startedRef.current = Date.now()
      recorder.start()
      setRecording(true)
      tick()
    } catch {
      setError('Microphone unavailable. Check the browser permission for this site.')
      cleanup()
    }
  }

  function stop() {
    recorderRef.current?.stop()
    setRecording(false)
  }

  return (
    <div className="recorder">
      <div className="recorder__controls">
        <Button
          variant={recording ? 'danger' : 'ghost'}
          size="sm"
          onClick={recording ? stop : () => void start()}
        >
          {recording ? 'Stop recording' : 'Record instead'}
        </Button>
        {recording && <span className="recorder__time mono">{formatDuration(elapsed)}</span>}
      </div>

      {recording && (
        <div className="recorder__meter" aria-hidden="true">
          {Array.from({ length: METER_SEGMENTS }, (_, index) => (
            <span
              key={index}
              className={`recorder__bar ${index / METER_SEGMENTS < level ? 'is-lit' : ''}`}
            />
          ))}
        </div>
      )}

      {error && <p className="recorder__error">{error}</p>}
    </div>
  )
}
