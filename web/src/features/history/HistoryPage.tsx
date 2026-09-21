/**
 * History.
 *
 * Every past job, each one a link back to its own URL. This page exists because v1's
 * results were unreachable the moment the Streamlit session ended -- persistence is the
 * feature, and a list is the plainest way to show it.
 */

import { Link } from 'react-router'

import type { Job } from '@/api/types'
import { useJobHistory } from '@/hooks/useJobs'
import { formatDuration, formatRelative, truncate } from '@/lib/format'
import { Button, Empty, StatusLight } from '@/components/primitives'
import '@/features/history/HistoryPage.css'

export function HistoryPage() {
  const history = useJobHistory()
  const jobs: Job[] = history.data?.pages.flatMap((page) => page.items) ?? []

  return (
    <section className="history">
      <header className="history__head">
        <p className="eyebrow">Archive</p>
        <h1 className="history__title">Everything you have recorded</h1>
      </header>

      {history.isLoading ? (
        <p className="mono history__loading">loading…</p>
      ) : jobs.length === 0 ? (
        <Empty title="Nothing recorded yet">
          <Link to="/">Write the first one →</Link>
        </Empty>
      ) : (
        <ol className="history__list">
          {jobs.map((job, index) => (
            <li key={job.id} className="rise" style={{ '--i': Math.min(index, 8) } as React.CSSProperties}>
              <Link to={`/jobs/${job.id}`} className="entry">
                <span className="entry__status">
                  <StatusLight status={job.status} label={false} />
                </span>

                <span className="entry__body">
                  <span className="entry__prompt">{truncate(job.prompt, 90)}</span>
                  <span className="entry__meta mono">
                    {job.length} · {job.language} · {formatRelative(job.created_at)}
                  </span>
                </span>

                <span className="entry__duration mono">
                  {job.audio?.[0] ? formatDuration(job.audio[0].duration_seconds) : '—'}
                </span>
              </Link>
            </li>
          ))}
        </ol>
      )}

      {history.hasNextPage && (
        <div className="history__more">
          <Button
            onClick={() => void history.fetchNextPage()}
            busy={history.isFetchingNextPage}
          >
            Load more
          </Button>
        </div>
      )}
    </section>
  )
}
