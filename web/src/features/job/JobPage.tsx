/**
 * The live job view.
 *
 * This page is the argument. v1 showed a spinner for up to ten minutes and lost
 * everything on refresh; here the story types itself in within about two seconds, the
 * segment meter counts real work, and the URL survives a reload, a reconnect, or coming
 * back tomorrow.
 *
 * The transport indicator is deliberately visible. When the WebSocket drops and polling
 * takes over, the page says so and keeps working — a fallback nobody can see is a
 * fallback nobody tests.
 *
 * Once audio exists, the story becomes a read-along: the line being spoken is lit and any
 * word can be clicked to seek. That is the point of the segment timeline the worker
 * records — it turns a seven minute file into something you can navigate by reading.
 *
 * There are two ways audio can exist here, and the page deliberately treats them the
 * same. While a job is running the segments arrive one at a time and are played through
 * the Web Audio API as they land; when it is finished there is a single assembled file
 * played through an `<audio>` element. Both expose a position in the story, and the tape
 * and transcript are given whichever one is live. Nothing downstream knows which it is —
 * that indistinguishability is what makes listening-while-recording a feature rather than
 * a second, lesser mode.
 */

import { useParams } from 'react-router'

import { ApiError } from '@/api/client'
import { isTerminal, type Job } from '@/api/types'
import { useCancelJob } from '@/hooks/useCancelJob'
import { useJob } from '@/hooks/useJob'
import { useJobEvents, type Transport } from '@/hooks/useJobEvents'
import { useLivePlayback } from '@/hooks/useLivePlayback'
import { usePlayback } from '@/hooks/usePlayback'
import { formatElapsed, secondsBetween, STATUS_ACTIVITY } from '@/lib/format'
import { AudioPlayer } from '@/components/AudioPlayer'
import { Banner, Button, Card, SegmentMeter, StatusLight } from '@/components/primitives'
import { LiveListen } from '@/features/job/LiveListen'
import { Pipeline } from '@/features/job/Pipeline'
import { Tape } from '@/features/job/Tape'
import { Transcript } from '@/features/job/Transcript'
import '@/features/job/JobPage.css'

export function JobPage() {
  const { jobId } = useParams<{ jobId: string }>()
  const job = useJob(jobId)
  const data = job.data
  const events = useJobEvents(jobId, data !== undefined && !isTerminal(data.status))
  const cancel = useCancelJob(jobId ?? '')

  // Declared before the early returns: hooks cannot be called conditionally, and both
  // players are inert until they are given something to play.
  const playback = usePlayback(data?.audio?.[0]?.duration_seconds ?? 0)
  const live = useLivePlayback(jobId, events.segments)

  if (job.isLoading) {
    return <div className="job__loading">Tuning in…</div>
  }

  if (job.error) {
    const message =
      job.error instanceof ApiError ? job.error.message : 'That story could not be loaded.'
    return <Banner tone="error">{message}</Banner>
  }

  if (!data) return null

  const running = !isTerminal(data.status)
  // While tokens are arriving the streamed text leads; once the story is persisted the
  // stored copy is authoritative, and a page reload shows that copy with no gap.
  const story = data.story_text ?? events.streamedText
  const segmentsTotal = events.segmentsTotal || (data.segment_count ?? 0)
  const audio = data.audio ?? []

  // Two sources for the same list. The record is written as each segment is rendered,
  // so a reload recovers what already exists; the WebSocket delivers the rest live.
  // Whichever is further ahead wins — the stored copy lags by one segment while a job is
  // running, and the streamed copy is empty for a page that was just opened.
  const streamed = events.segments
  const stored = data.segments ?? []
  const segments = stored.length >= streamed.length ? stored : streamed

  // Whichever player currently owns the story's clock. The finished file wins when it
  // exists, because it is the authoritative render.
  const position = audio.length > 0 ? playback.currentTime : live.currentTime
  const isPlaying = audio.length > 0 ? playback.playing : live.playing
  const seek = audio.length > 0 ? playback.seek : live.seek

  return (
    <article className="job">
      <header className="job__head rise" style={{ '--i': 0 } as React.CSSProperties}>
        <div className="job__head-top">
          <StatusLight status={data.status} />
          <TransportBadge transport={events.transport} running={running} />
        </div>

        <h1 className="job__prompt">{data.prompt}</h1>

        <p className="job__activity">{STATUS_ACTIVITY[data.status]}</p>
      </header>

      <Pipeline job={data} />

      {data.error && (
        <Banner tone="error">
          <div>
            <strong>{data.error.message}</strong>
            {data.error.retryable && (
              <p className="job__retry-note">This one is worth trying again.</p>
            )}
          </div>
        </Banner>
      )}

      {events.recoveredFromGap && (
        <Banner tone="warn">
          Some live updates were missed, so the page re-synced from the server. Nothing was
          lost.
        </Banner>
      )}

      {(running || segmentsTotal > 0) && data.status !== 'queued' && (
        <Card className="job__progress rise" style={{ '--i': 1 } as React.CSSProperties}>
          <div className="job__progress-head">
            <span className="eyebrow">Recording</span>
            {running && (
              <Button variant="danger" size="sm" busy={cancel.isPending} onClick={() => cancel.mutate()}>
                Stop
              </Button>
            )}
          </div>
          {segmentsTotal > 0 ? (
            <SegmentMeter done={events.segmentsDone} total={segmentsTotal} />
          ) : (
            <p className="job__progress-wait mono">waiting for the writer…</p>
          )}
        </Card>
      )}

      {audio.length === 0 && segments.length > 0 && (
        <div className="rise" style={{ '--i': 2 } as React.CSSProperties}>
          <LiveListen playback={live} ready={segments.length} total={segmentsTotal} />
        </div>
      )}

      {segments.length > 0 && (
        <div className="rise" style={{ '--i': 2 } as React.CSSProperties}>
          <Tape
            segments={segments}
            currentTime={position}
            duration={audio[0]?.duration_seconds}
            expected={segmentsTotal}
            bufferedUntil={audio.length > 0 ? undefined : live.bufferedUntil}
            onSeek={seek}
          />
        </div>
      )}

      {audio.length > 0 && (
        <div className="job__player rise" style={{ '--i': 2 } as React.CSSProperties}>
          <AudioPlayer assets={audio} playback={playback} />
        </div>
      )}

      {story && (
        <section className="story rise" style={{ '--i': 3 } as React.CSSProperties}>
          <div className="story__head">
            <h2 className="eyebrow">Story</h2>
            {segments.length > 0 && (
              <span className="story__hint">Click any word to jump there</span>
            )}
          </div>
          <Transcript
            story={story}
            segments={segments}
            currentTime={position}
            playing={isPlaying}
            onSeek={seek}
            writing={data.status === 'writing'}
            rendering={running}
          />
        </section>
      )}

      {!story && data.status === 'queued' && (
        <div className="job__queued">
          <span className="job__queued-dots" aria-hidden="true">
            <i />
            <i />
            <i />
          </span>
          <p>Waiting for a writer to pick this up.</p>
        </div>
      )}

      <Timings job={data} />
    </article>
  )
}

function TransportBadge({ transport, running }: { transport: Transport; running: boolean }) {
  if (!running) return null

  const label: Record<Transport, string> = {
    connecting: 'connecting',
    live: 'live',
    polling: 'polling',
    closed: '',
  }

  if (!label[transport]) return null

  return (
    <span className={`transport transport--${transport}`} title="How updates are arriving">
      <span className="transport__dot" aria-hidden="true" />
      <span className="mono">{label[transport]}</span>
    </span>
  )
}

/**
 * Per-stage timings.
 *
 * Shown because they are the evidence for the whole rebuild: the LLM and TTS stages are
 * separately measured, so "it used to block for ten minutes" becomes a number rather
 * than a claim.
 */
function Timings({ job }: { job: Job }) {
  const t = job.timings
  const rows = [
    { label: 'Accepted', value: secondsBetween(t.queued_at, t.writing_at) },
    { label: 'Writing', value: secondsBetween(t.writing_at, t.written_at) },
    { label: 'Recording', value: secondsBetween(t.synthesizing_at, t.finished_at) },
    { label: 'Total', value: secondsBetween(t.queued_at, t.finished_at) },
  ].filter((row) => row.value !== null)

  if (rows.length === 0) return null

  return (
    <section className="timings rise" style={{ '--i': 4 } as React.CSSProperties}>
      <h2 className="eyebrow">Stage timings</h2>
      <dl className="timings__grid">
        {rows.map((row) => (
          <div key={row.label} className="timings__row">
            <dt>{row.label}</dt>
            <dd className="mono">{formatElapsed(row.value)}</dd>
          </div>
        ))}
      </dl>
    </section>
  )
}
