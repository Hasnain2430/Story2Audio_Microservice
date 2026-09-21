/**
 * Job history, cursor-paginated.
 *
 * The cursor is the last job's UUIDv7, which is time-ordered, so paging cannot skip or
 * repeat a row as new jobs arrive mid-scroll (ADR-0002).
 */

import { useInfiniteQuery } from '@tanstack/react-query'

import { api } from '@/api/client'
import { jobKeys } from '@/hooks/queryKeys'

export function useJobHistory(pageSize = 20) {
  return useInfiniteQuery({
    queryKey: jobKeys.list(),
    queryFn: ({ pageParam }) =>
      api.listJobs({ limit: pageSize, ...(pageParam ? { cursor: pageParam } : {}) }),
    initialPageParam: undefined as string | undefined,
    getNextPageParam: (last) => (last.has_more ? (last.next_cursor ?? undefined) : undefined),
  })
}
