/**
 * The pipeline, drawn as stations on a line.
 *
 * Exists to make the architecture legible: a reader can see that writing and recording
 * are separate stages on separate workers, which is precisely what v1 could not show
 * because it did all of it inside one blocking call.
 */

import { PIPELINE_STAGES, type Job, type JobStatus } from '@/api/types'
import { secondsBetween, formatElapsed } from '@/lib/format'

const STAGE_LABEL: Record<(typeof PIPELINE_STAGES)[number], string> = {
  queued: 'Queued',
  writing: 'Writing',
  written: 'Written',
  synthesizing: 'Recording',
  done: 'Ready',
}

/** How far along the line a status sits. Terminal failures stop wherever they stopped. */
function reachedIndex(status: JobStatus): number {
  const index = PIPELINE_STAGES.indexOf(status as (typeof PIPELINE_STAGES)[number])
  if (index >= 0) return index
  return -1
}

export function Pipeline({ job }: { job: Job }) {
  const current = reachedIndex(job.status)
  const stopped = job.status === 'failed' || job.status === 'cancelled'

  const durations: Partial<Record<(typeof PIPELINE_STAGES)[number], number | null>> = {
    writing: secondsBetween(job.timings.writing_at, job.timings.written_at),
    synthesizing: secondsBetween(job.timings.synthesizing_at, job.timings.finished_at),
  }

  return (
    <ol className={`pipeline ${stopped ? 'is-stopped' : ''}`} aria-label="Pipeline progress">
      {PIPELINE_STAGES.map((stage, index) => {
        const done = current > index || job.status === 'done'
        const active = current === index && !stopped
        const elapsed = durations[stage]

        return (
          <li
            key={stage}
            className={`pipeline__stage ${done ? 'is-done' : ''} ${active ? 'is-active' : ''}`}
          >
            <span className="pipeline__node" aria-hidden="true" />
            <span className="pipeline__label">{STAGE_LABEL[stage]}</span>
            {elapsed != null && <span className="pipeline__time mono">{formatElapsed(elapsed)}</span>}
          </li>
        )
      })}
    </ol>
  )
}
